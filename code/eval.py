import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import nibabel as nib
import torchvision.transforms.functional as TF

from scipy.special import erf

import random
import copy
import re
import os
import torch.nn.functional as F

from scipy.ndimage import laplace
from scipy.spatial.distance import directed_hausdorff
from skimage.measure import label
from skimage.morphology import binary_erosion

def dice_coeff(input, target, epsilon=1e-6):
    input = input.contiguous().view(-1).float()
    target = target.contiguous().view(-1).float()
    intersection = (input * target).sum()
    dice = (2. * intersection + epsilon) / (input.sum() + target.sum() + epsilon)
    return dice

def lesionwise_dice(pred, target, num_classes=4, epsilon=1e-6):
    dices = []
    for cls in range(1, num_classes):
        pred_mask = (pred == cls).astype(np.uint8)
        target_mask = (target == cls).astype(np.uint8)
        pred_labels = label(pred_mask)
        target_labels = label(target_mask)
        pred_props = []
        target_props = []
        for region in range(1, pred_labels.max()+1):
            pred_props.append(pred_labels == region)
        for region in range(1, target_labels.max()+1):
            target_props.append(target_labels == region)
        for pred_region in pred_props:
            best_dice = 0
            for target_region in target_props:
                intersection = np.logical_and(pred_region, target_region).sum()
                dice = (2. * intersection + epsilon) / (pred_region.sum() + target_region.sum() + epsilon)
                if dice > best_dice:
                    best_dice = dice
            dices.append(best_dice)
    if len(dices) == 0:
        return 1.0
    return np.mean(dices)

def hausdorff95(pred, target):
    pred = pred.astype(np.bool)
    target = target.astype(np.bool)
    pred_border = pred ^ binary_erosion(pred)
    target_border = target ^ binary_erosion(target)
    pred_coords = np.column_stack(np.where(pred_border))
    target_coords = np.column_stack(np.where(target_border))
    if len(pred_coords) == 0 or len(target_coords) == 0:
        return 0.0
    distances = []
    for p in pred_coords:
        d = np.linalg.norm(target_coords - p, axis=1)
        distances.append(np.min(d))
    for t in target_coords:
        d = np.linalg.norm(pred_coords - t, axis=1)
        distances.append(np.min(d))
    return np.percentile(distances, 95)

# ===============================
# Model Definitions
# ===============================

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Handle padding if necessary
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = TF.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)      # (B, 64, H, W)
        x2 = self.down1(x1)   # (B, 128, H/2, W/2)
        x3 = self.down2(x2)   # (B, 256, H/4, W/4)
        x4 = self.down3(x3)   # (B, 512, H/8, W/8)
        x5 = self.down4(x4)   # (B, 1024, H/16, W/16)
        x = self.up1(x5, x4)  # (B, 512, H/8, W/8)
        x = self.up2(x, x3)   # (B, 256, H/4, W/4)
        x = self.up3(x, x2)   # (B, 128, H/2, W/2)
        x = self.up4(x, x1)   # (B, 64, H, W)
        logits = self.outc(x) # (B, n_classes, H, W)
        return logits

class BrainSegmentationDataset(Dataset):
    def __init__(self, csv_path, crop_coords=None):
        self.data = pd.read_csv(csv_path)
        self.crop_coords = crop_coords
        self.slice_info = self._create_slice_index()
    
    def _create_slice_index(self):
        slice_info = []
        subjects = self.data['Subject ID'].unique()
        for subject in subjects:
            subject_data = self.data[self.data['Subject ID'] == subject]
            flair_path = subject_data[subject_data['Scan Type'] == 'flair']['File Path'].values[0]
            nii = nib.load(flair_path)
            image = nii.get_fdata().astype(np.float32)
            depth = image.shape[2]
            for z in range(15, depth - 12):
                slice_img = image[:, :, z]
                if np.any(slice_img > 0):
                    slice_info.append((subject, z))
        return slice_info
    
    def __len__(self):
        return len(self.slice_info)
    
    @staticmethod
    def Img_proc(image, _lambda=-0.8, epsilon=1e-6):
        if np.isnan(image).any() or np.isinf(image).any():
            raise ValueError("Invalid image values.")
        I_img = image
        min_val = np.min(I_img)
        max_val = np.max(I_img)
        if max_val == min_val:
            return np.zeros_like(I_img)
        I_img_norm = (I_img - min_val) / (max_val - min_val + epsilon)
        max_I_img = np.max(I_img_norm)
        IMG1 = (max_I_img / np.log(max_I_img + 1 + epsilon)) * np.log(I_img_norm + 1)
        IMG2 = 1 - np.exp(-I_img_norm)
        IMG3 = (IMG1 + IMG2) / (_lambda + (IMG1 * IMG2))
        IMG4 = erf(_lambda * np.arctan(np.exp(IMG3)) - 0.5 * IMG3)
        min_IMG4 = np.min(IMG4)
        max_IMG4 = np.max(IMG4)
        if max_IMG4 == min_IMG4:
            return np.zeros_like(IMG4)
        IMG5 = (IMG4 - min_IMG4) / (max_IMG4 - min_IMG4 + epsilon)
        return IMG5
    
    def __getitem__(self, idx):
        subject, z = self.slice_info[idx]
        subject_data = self.data[self.data['Subject ID'] == subject]
        modalities = ['flair', 't1', 't1ce', 't2']
        slices = []
        for modality in modalities:
            file_path = subject_data[subject_data['Scan Type'] == modality]['File Path'].values[0]
            nii = nib.load(file_path)
            image = nii.get_fdata().astype(np.float32)
            image = (image - np.mean(image)) / (np.std(image) + 1e-6)
            if self.crop_coords:
                min_x, max_x, min_y, max_y = self.crop_coords
                image = image[min_x:max_x, min_y:max_y, :]
            slice_img = image[:, :, z]
            slice_img = self.Img_proc(slice_img)
            slices.append(slice_img)
        images = np.stack(slices, axis=0)
        seg_data = subject_data[subject_data['Scan Type'] == 'seg']
        if seg_data.empty:
            raise ValueError(f"Missing segmentation for subject {subject}")
        seg_path = seg_data['File Path'].values[0]
        seg_nii = nib.load(seg_path)
        seg_mask = seg_nii.get_fdata().astype(np.uint8)
        seg_mask[seg_mask == 4] = 3
        seg_slice = seg_mask[:, :, z]
        if self.crop_coords:
            seg_slice = seg_mask[min_x:max_x, min_y:max_y, z]
        return torch.tensor(images, dtype=torch.float32), torch.tensor(seg_slice, dtype=torch.long)


def evaluate(net, dataloader, device, crop_coords=None):
    net.eval()
    
    # 初始化存储每个标签的指标
    voxel_dice_scores = {1: [], 2: [], 3: []}
    lw_dice_scores    = {1: [], 2: [], 3: []}
    hd95_scores       = {1: [], 2: [], 3: []}

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device, dtype=torch.float32)
            masks = masks.numpy()  # Shape: (B, H, W)

            outputs = net(images)  # Shape: (B, 4, H, W)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()  # Shape: (B, H, W)

            for i in range(preds.shape[0]):
                pred = preds[i]    # Shape: (H, W)
                target = masks[i]  # Shape: (H, W)

                for label_id in [1, 2, 3]:
                    label_pred = (pred == label_id).astype(np.uint8)
                    label_target = (target == label_id).astype(np.uint8)

                    # 计算 Voxel-wise Dice
                    v_dice = dice_coeff(
                        torch.tensor(label_pred),
                        torch.tensor(label_target)
                    )
                    
                    # 计算 Lesion-wise Dice
                    l_dice = lesionwise_dice(
                        label_pred, label_target, 
                        num_classes=2  # 仅背景和当前标签
                    )
                    
                    # 计算 Hausdorff95
                    hd = hausdorff95(label_pred, label_target)
                    
                    # 存储指标
                    voxel_dice_scores[label_id].append(v_dice.item())
                    lw_dice_scores[label_id].append(l_dice)
                    hd95_scores[label_id].append(hd)

    # 计算每个标签的平均值
    voxel_dice_averages = {}
    lw_dice_averages = {}
    hd95_averages = {}
    
    for label_id in [1, 2, 3]:
        voxel_vals = voxel_dice_scores[label_id]
        lw_vals    = lw_dice_scores[label_id]
        hd95_vals  = hd95_scores[label_id]
        
        voxel_dice_averages[label_id] = np.mean(voxel_vals)  if len(voxel_vals) > 0 else 0.0
        lw_dice_averages[label_id]    = np.mean(lw_vals)     if len(lw_vals) > 0 else 0.0
        hd95_averages[label_id]       = np.mean(hd95_vals)   if len(hd95_vals) > 0 else 0.0
    
    # 计算所有标签的宏平均
    mean_voxel_dice = np.mean(list(voxel_dice_averages.values()))
    mean_lw_dice    = np.mean(list(lw_dice_averages.values()))
    mean_hd95       = np.mean(list(hd95_averages.values()))

    # 打印结果
    print("==== Per-Label Results ====")
    for label_id in [1, 2, 3]:
        print(f"Label {label_id}: "
              f"Voxel Dice = {voxel_dice_averages[label_id]:.4f}, "
              f"LW Dice = {lw_dice_averages[label_id]:.4f}, "
              f"HD95 = {hd95_averages[label_id]:.4f}")

    print("==== Averages Across All Labels (1,2,3) ====")
    print(f"Mean Voxel-wise Dice = {mean_voxel_dice:.4f}")
    print(f"Mean Lesion-wise Dice = {mean_lw_dice:.4f}")
    print(f"Mean Hausdorff95 = {mean_hd95:.4f}")

    return (
        voxel_dice_averages, 
        lw_dice_averages, 
        hd95_averages, 
        mean_voxel_dice, 
        mean_lw_dice, 
        mean_hd95
    )



def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v  # Remove 'module.' prefix
        else:
            new_state_dict[k] = v
    return new_state_dict

def sanity_check_label2_occurrence(dataloader):
    count_label2_non_empty = 0
    total_slices = 0

    for images, masks in dataloader:
        # masks has shape (B, H, W) or (B, D, H, W) depending on how you loaded it
        masks_np = masks.numpy()  # convert to NumPy if needed
        for i in range(masks_np.shape[0]):
            slice_mask = masks_np[i]
            total_slices += 1
            if (slice_mask == 2).any():
                count_label2_non_empty += 1

    print(f"Out of {total_slices} total slices, {count_label2_non_empty} contain label-2 voxels.")


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 检查是否是 DataParallel 模型
    model_keys = list(model.state_dict().keys())
    checkpoint_keys = list(checkpoint.keys())

    if any(key.startswith("module.") for key in checkpoint_keys) and not any(key.startswith("module.") for key in model_keys):
        # 去除 `module.` 前缀
        checkpoint = {key.replace("module.", ""): value for key, value in checkpoint.items()}
    elif not any(key.startswith("module.") for key in checkpoint_keys) and any(key.startswith("module.") for key in model_keys):
        # 添加 `module.` 前缀
        checkpoint = {"module." + key: value for key, value in checkpoint.items()}

    model.load_state_dict(checkpoint)
    logging.info(f"Loaded checkpoint from {checkpoint_path}")
    return model


def main():
    print("Start evaluation process")
    csv_path = '../data/selected_test_subject.csv'
    model_dir = "../model"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Reading dataset")
    dataset = BrainSegmentationDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=160, shuffle=False, num_workers=4, pin_memory=True)
        # Suppose you already created the DataLoader named `test_dataloader`
    #sanity_check_label2_occurrence(dataloader)

    all_results = []
    print("Start evaluating")
    
    for root, _, files in os.walk(model_dir):
        for file in files:
            if file.endswith(".pt") or file.endswith(".pth"):
                checkpoint_path = os.path.join(root, file)
                print(f"Read file from checkpoint {checkpoint_path}")
                
                # Initialize the model
                model = UNet(n_channels=4, n_classes=4, bilinear=True).to(device)
                
                try:
                    # Use `load_checkpoint` to load the weights
                    model = load_checkpoint(model, checkpoint_path, device)
                    print("Model Loaded Successfully")
                except Exception as e:
                    print(f"Fail to load the Model: {e}")
                    continue
                
                model.eval()
                print(f"Dataset size: {len(dataset)}, Dataloader batches: {len(dataloader)}")
                
                # --- Call our new evaluate function ---
                (
                    voxel_dice_averages, 
                    lw_dice_averages, 
                    hd95_averages, 
                    mean_voxel_dice, 
                    mean_lw_dice, 
                    mean_hd95
                ) = evaluate(model, dataloader, device)

                # Print or log the results
                print(f"\nEvaluation Results for {file}:")
                for label_id in [1, 2, 3]:
                    print(f"  Label {label_id}: "
                          f"VoxelDice = {voxel_dice_averages[label_id]:.4f}, "
                          f"LesionWiseDice = {lw_dice_averages[label_id]:.4f}, "
                          f"HD95 = {hd95_averages[label_id]:.4f}")
                print("  --- Averages Across Labels (1,2,3) ---")
                print(f"  Mean Voxel-wise Dice = {mean_voxel_dice:.4f}")
                print(f"  Mean Lesion-wise Dice = {mean_lw_dice:.4f}")
                print(f"  Mean Hausdorff95      = {mean_hd95:.4f}\n")

                # --- Prepare a row for CSV ---
                all_results.append({
                    "Checkpoint": file,
                    
                    # Per-label Voxel Dice
                    "VoxelDice_Label1": voxel_dice_averages[1],
                    "VoxelDice_Label2": voxel_dice_averages[2],
                    "VoxelDice_Label3": voxel_dice_averages[3],
                    
                    # Mean Voxel Dice
                    "Mean Voxel Dice": mean_voxel_dice,

                    # Per-label Lesion-wise Dice
                    "LWDice_Label1": lw_dice_averages[1],
                    "LWDice_Label2": lw_dice_averages[2],
                    "LWDice_Label3": lw_dice_averages[3],
                    
                    # Mean LW Dice
                    "Mean LW Dice": mean_lw_dice,

                    # Per-label HD95
                    "HD95_Label1": hd95_averages[1],
                    "HD95_Label2": hd95_averages[2],
                    "HD95_Label3": hd95_averages[3],
                    
                    # Mean HD95
                    "Mean HD95": mean_hd95
                })
    
    # Save the results to a CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("evaluation_results.csv", index=False)
    print("Evaluation results saved to 'evaluation_results.csv'")



if __name__ == "__main__":
    main()
