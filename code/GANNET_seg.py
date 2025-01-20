import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from tqdm import tqdm
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import nibabel as nib
import torchvision.transforms.functional as TF
from torchvision import transforms
from scipy.special import erf
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import random
import copy
import re
import os
import torch.nn.functional as F
from torch.nn import MaxPool2d
from scipy.ndimage import laplace

# ===============================
# Utility Functions
# ===============================

def dice_coeff(input: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6):
     # Convert target to one-hot: shape (B, H, W) -> (B, H, W, C) -> (B, C, H, W)
    num_classes = input.shape[1]  # e.g. 4
    target_onehot = F.one_hot(target.long(), num_classes=num_classes)  # (B,H,W,C)
    target_onehot = target_onehot.permute(0, 3, 1, 2).float()          # (B,C,H,W)

    # Apply softmax to get predicted probabilities per class
    probs = F.softmax(input, dim=1)  # (B,C,H,W)

    # Compute Dice for each class separately
    dims = (0, 2, 3)  # sum over batch, height, width
    intersection = (probs * target_onehot).sum(dim=dims)
    cardinality = probs.sum(dim=dims) + target_onehot.sum(dim=dims)
    dice_per_class = (2. * intersection + epsilon) / (cardinality + epsilon)
    
    dice_fg = dice_per_class[1:]  # classes = 1,2,3
    mean_fg_dice = dice_fg.mean() # average dice across labels {1,2,3}
    # Average (1 - dice) over classes => multi-class dice loss
    return 1 - mean_fg_dice

def dice_loss(input: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6):
    return 1 - dice_coeff(input, target, epsilon)

def cross_entropy_loss(input: torch.Tensor, target: torch.Tensor):
    return F.cross_entropy(input, target.long())

def unwrap_model(model):
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    return model

def get_latest_checkpoint(checkpoint_dir: str, script_name: str):
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)


    checkpoints = list(checkpoint_path.glob(f"checkpoint_epoch{script_name}_*.pth"))
    if not checkpoints:
        return None, 1  


    latest_epoch = 0
    latest_checkpoint = None
    for ckpt in checkpoints:
        match = re.search(rf'checkpoint_epoch{script_name}_(\d+)\.pth', ckpt.name)
        if match:
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_checkpoint = ckpt

    return latest_checkpoint, latest_epoch + 1 if latest_checkpoint else 1

def save_checkpoint(model, epoch, checkpoint_dir, script_name):
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_path / f"checkpoint_epoch{script_name}_{epoch}.pth"
    torch.save(model.state_dict(), checkpoint_file)
    logging.info(f"Checkpoint saved at epoch {epoch}: {checkpoint_file}")

def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ===============================
# Dataset Class
# ===============================

def Img_proc(image, _lambda=-0.8, epsilon=1e-6):
    """
    Image processing function with safeguards for invalid input values.
    """
    if np.isnan(image).any() or np.isinf(image).any():
        raise ValueError("Input image contains NaN or infinity values.")
    
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

class CropOptimizer:
    def __init__(self, csv_path):
        self.data_summary = pd.read_csv(csv_path)
        self.modalities = ['flair', 't1', 't1ce', 't2']
    
    def find_optimal_crop(self):
        """Finds the minimum bounding box for all slices across all modalities."""
        crop_coords = []
        for subject_id in self.data_summary['Subject ID'].unique():
            for modality in self.modalities:
                file_path = self.data_summary[
                    (self.data_summary['Subject ID'] == subject_id) &
                    (self.data_summary['Scan Type'] == modality)
                ]['File Path'].values[0]
                nii = nib.load(file_path)
                data = nii.get_fdata()
                non_zero_coords = np.argwhere(data > 0)
                crop_coords.append([
                    np.min(non_zero_coords[:, 0]), np.max(non_zero_coords[:, 0]),
                    np.min(non_zero_coords[:, 1]), np.max(non_zero_coords[:, 1])
                ])
        
        crop_coords = np.array(crop_coords)
        min_x, max_x = np.min(crop_coords[:, 0]), np.max(crop_coords[:, 1])
        min_y, max_y = np.min(crop_coords[:, 2]), np.max(crop_coords[:, 3])
        return min_x, max_x, min_y, max_y


class BrainDataset(Dataset):  
    def __init__(self, dataframe, crop_coords=None, transform=None):  
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing file paths and labels.
            crop_coords (tuple, optional): Coordinates for cropping (min_x, max_x, min_y, max_y).
            transform (callable, optional): Optional transforms to be applied on a sample.
        """  
        self.dataframe = dataframe  
        self.transform = transform  
        self.crop_coords = crop_coords
        self.subjects = self.dataframe['Subject ID'].unique()
        self.slice_info = self._create_slice_index()
    
    def _create_slice_index(self):
        """Create a list of (subject_id, slice_idx) pairs, ignoring blank and edge slices."""
        slice_info = []
        for subject_id in self.subjects:
            subject_data = self.dataframe[self.dataframe['Subject ID'] == subject_id]
            flair_data = subject_data[subject_data['Scan Type'] == 'flair']
            if flair_data.empty:
                logging.warning(f"Missing 'flair' scan for subject {subject_id}. Skipping.")
                continue  # Skip subjects without flair scans
            
            flair_path = flair_data['File Path'].values[0]
            try:
                nii = nib.load(flair_path)
                image = nii.get_fdata().astype(np.float32)
            except Exception as e:
                logging.error(f"Error loading 'flair' scan for subject {subject_id}: {e}")
                continue  # Skip subjects with corrupted flair scans
            
            depth = image.shape[2]
            for z in range(15, depth - 12):  # Exclude the first 15 and last 12 slices
                slice_image = image[:, :, z]
                if np.any(slice_image > 0):  # Check if the slice contains any non-zero values
                    slice_info.append((subject_id, z))
        logging.info(f"Total slices prepared: {len(slice_info)}")
        return slice_info
    
    def __len__(self):  
        return len(self.slice_info)  
    
    def __getitem__(self, idx):
        """
        Retrieves the image, segmentation mask, and T1 image for a given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (images, seg_slice, t1_image)
                - images (torch.Tensor): Stacked multi-modal MRI images (C x H x W).
                - seg_slice (torch.Tensor): Segmentation mask for the slice (H x W).
                - t1_image (torch.Tensor): T1-weighted MRI slice (H x W).
        """
        subject_id, slice_idx = self.slice_info[idx]
        subject_data = self.dataframe[self.dataframe['Subject ID'] == subject_id]

        modalities = ['flair', 't1', 't1ce', 't2']
        slices = []
        for modality in modalities:
            modality_data = subject_data[subject_data['Scan Type'] == modality]
            if modality_data.empty:
                raise ValueError(f"Missing {modality} scan for subject {subject_id}")
            file_path = modality_data['File Path'].values[0]
    

            try:
                nii = nib.load(file_path)
                image = nii.get_fdata().astype(np.float32)
            except Exception as e:
                raise ValueError(f"Error loading {modality} scan for subject {subject_id}: {e}")
    
            mean = np.mean(image)
            std = np.std(image)
            if std == 0:
                std = 1e-6  # Prevent division by zero
            image = (image - mean) / (std + 1e-6)
    
            if self.crop_coords:
                min_x, max_x, min_y, max_y = self.crop_coords
                min_x = max(min_x, 0)
                max_x = min(max_x, image.shape[0])
                min_y = max(min_y, 0)
                max_y = min(max_y, image.shape[1])
                image = image[min_x:max_x, min_y:max_y, :]
    
            try:
                slice_image = image[:, :, slice_idx]
            except IndexError:
                raise IndexError(f"Slice index {slice_idx} out of range for subject {subject_id}")
    
            slice_image = Img_proc(slice_image)
            slices.append(slice_image)
    

        images = np.stack(slices, axis=0)
        seg_data = subject_data[subject_data['Scan Type'] == 'seg']
        if seg_data.empty:
            raise ValueError(f"Missing segmentation mask for subject {subject_id}")
        seg_path = seg_data['File Path'].values[0]
        try:
            seg_nii = nib.load(seg_path)
            seg_mask = seg_nii.get_fdata().astype(np.uint8)
        except Exception as e:
            raise ValueError(f"Error loading segmentation mask for subject {subject_id}: {e}")
    

        seg_mask[seg_mask == 4] = 3
    

        try:
            seg_slice = seg_mask[:, :, slice_idx]
        except IndexError:
            raise IndexError(f"Slice index {slice_idx} out of range for segmentation mask of subject {subject_id}")
    

        if self.crop_coords:
            min_x, max_x, min_y, max_y = self.crop_coords
            min_x = max(min_x, 0)
            max_x = min(max_x, seg_mask.shape[0])
            min_y = max(min_y, 0)
            max_y = min(max_y, seg_mask.shape[1])
            seg_slice = seg_slice[min_x:max_x, min_y:max_y]
    

        if self.transform:
            images = torch.tensor(images, dtype=torch.float32)
            seg_slice = torch.tensor(seg_slice, dtype=torch.long)  
            if random.random() > 0.5:
                images = TF.hflip(images)
                seg_slice = TF.hflip(seg_slice)
        else:
            images = torch.tensor(images, dtype=torch.float32)
            seg_slice = torch.tensor(seg_slice, dtype=torch.long)
    
        t1_image = images[1, :, :]  # (H, W)
    
        return (
            images,       # (C, H, W)
            seg_slice,    # (H, W)
            t1_image      # (H, W)
        )



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

class Discriminator(nn.Module):  
    def __init__(self, in_channels):  
        super(Discriminator, self).__init__()  
        self.model = nn.Sequential(  
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),  
            nn.LeakyReLU(0.2, inplace=True),  
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  
            nn.BatchNorm2d(128),  
            nn.LeakyReLU(0.2, inplace=True),  
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  
            nn.BatchNorm2d(256),  
            nn.LeakyReLU(0.2, inplace=True),  
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  
            nn.BatchNorm2d(512),  
            nn.LeakyReLU(0.2, inplace=True),  
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),  
            nn.Sigmoid()  
        )  

    def forward(self, x):  
        return self.model(x)  

class SparsityLoss(nn.Module):
    def __init__(self, alpha=0.1):
        super(SparsityLoss, self).__init__()
        self.alpha = alpha

    def forward(self, mask):
        return self.alpha * torch.sum(torch.abs(mask))

class SizeConsistencyLoss(nn.Module):
    def __init__(self, gamma=0.1, epsilon=1e-6):
        super(SizeConsistencyLoss, self).__init__()
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, mask, S_label):
        predicted_size = torch.sum(mask)
        loss = torch.abs((predicted_size - S_label) / (S_label + self.epsilon))
        return self.gamma * loss

def max_pool_expansion(mask, kernel_size=5):
    pool = MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    return pool(mask)

def laplacian_filter(mask):
    mask_np = mask.cpu().numpy()
    laplacian = np.zeros_like(mask_np)
    for i in range(mask_np.shape[0]):
        laplacian[i] = laplace(mask_np[i])
    laplacian = torch.tensor(laplacian, dtype=mask.dtype, device=mask.device)
    return laplacian

def train_model(
    unet,
    generator,
    discriminator,
    train_loader,
    val_loader,
    optimizer,
    criterion_segmentation,
    criterion_adversarial,
    sparsity_loss_fn,
    size_consistency_loss_fn,
    device,
    epochs=25,
    adversarial_weight=0.1,
    checkpoint_dir="./checkpoints",
    scheduler=None,
    scaler=None,
    writer=None,
    early_stopping_patience=5
):
    unet.train()
    best_val_dice = 0.0
    trigger_times = 0
    best_model_wts = copy.deepcopy(unet.state_dict())

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        unet.train()

        phase = 1 if epoch <=10 else 2

        with tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch") as tepoch:
            for batch_idx, (images, masks, t1_images) in enumerate(tepoch):
                images = images.to(device)
                masks = masks.to(device)
                t1_images = t1_images.to(device)

                mask_expanded = masks.unsqueeze(1).float()
                t1_images_expanded = t1_images.unsqueeze(1).float()

                with torch.no_grad():
                    M_expanded = max_pool_expansion(mask_expanded)
                    M_edges = laplacian_filter(mask_expanded.squeeze(1))
                    M_edges = M_edges.unsqueeze(1)
                    edge_attention = M_expanded * M_edges
                gen_input = torch.cat([mask_expanded, t1_images_expanded, edge_attention], dim=1)

                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=(scaler is not None)):
                    outputs = unet(images)
                    outputs_sigmoid = torch.sigmoid(outputs).squeeze(1)

                    if phase ==1:
                        loss_ce = criterion_segmentation(outputs.squeeze(1), masks.float())
                        loss_dice = dice_loss(outputs_sigmoid, masks.float())
                        loss_seg = 0.5 * loss_ce + 2 * loss_dice  # This part we use the Unet++
                        loss = loss_seg
                    elif phase ==2:
                        loss_ce = criterion_segmentation(outputs.squeeze(1), masks.float())
                        loss_dice = dice_loss(outputs_sigmoid, masks.float())
                        loss_seg = 0.3 * loss_ce + 0.7 * loss_dice

                        loss_sparse = sparsity_loss_fn(outputs_sigmoid)

                        S_label = torch.sum(masks.view(masks.size(0), -1), dim=1)
                        loss_size = 0.0
                        for i in range(masks.size(0)):
                            loss_size += size_consistency_loss_fn(outputs_sigmoid[i], S_label[i])
                        loss_size = loss_size / masks.size(0)

                        generated_images = generator(gen_input)

                        preds_discriminator = discriminator(generated_images)

                        real_labels = torch.ones_like(preds_discriminator, device=device)

                        loss_adv = criterion_adversarial(preds_discriminator, real_labels)

                        epsilon = 1e-8
                        lambda_seg = 1.0 / (loss_seg.item() + epsilon)
                        lambda_sparse = 1.0 / (loss_sparse.item() + epsilon)
                        lambda_size = 1.0 / (loss_size.item() + epsilon)
                        lambda_adv = 1.0 / (loss_adv.item() + epsilon)

                        sum_weights = lambda_seg + lambda_sparse + lambda_size + lambda_adv
                        lambda_seg /= sum_weights
                        lambda_sparse /= sum_weights
                        lambda_size /= sum_weights
                        lambda_adv /= sum_weights

                        loss = (lambda_seg * loss_seg) + (lambda_sparse * loss_sparse) + (lambda_size * loss_size) + (lambda_adv * adversarial_weight * loss_adv)

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                epoch_loss += loss.item()
                tepoch.set_postfix({"Loss": loss.item()})

                if writer is not None:
                    global_step = (epoch - 1) * len(train_loader) + batch_idx
                    writer.add_scalar('Loss/Train_Total', loss.item(), global_step)
                    writer.add_scalar('Loss/Train_Segmentation', loss_seg.item(), global_step)
                    if phase ==2:
                        writer.add_scalar('Loss/Train_Sparsity', loss_sparse.item(), global_step)
                        writer.add_scalar('Loss/Train_SizeConsistency', loss_size.item(), global_step)
                        writer.add_scalar('Loss/Train_Adversarial', loss_adv.item(), global_step)

        avg_epoch_loss = epoch_loss / len(train_loader)
        logging.info(f"Epoch {epoch}/{epochs} - Training Loss: {avg_epoch_loss:.4f}")

        val_dice = validate_model(unet, val_loader, device)
        logging.info(f"Epoch {epoch}/{epochs} - Validation Dice Score: {val_dice:.4f}")

        if writer is not None:
            writer.add_scalar('Dice_Score/Validation', val_dice, epoch)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_model_wts = copy.deepcopy(unet.state_dict())
            trigger_times = 0
            logging.info(f"Epoch {epoch} - New best model found with Dice Score: {val_dice:.4f}")
        else:
            trigger_times += 1
            logging.info(f"Epoch {epoch} - No improvement in Dice Score.")
            if trigger_times >= early_stopping_patience:
                logging.info("Early stopping triggered.")
                unet.load_state_dict(best_model_wts)
                if writer is not None:
                    writer.close()
                return

        if scheduler is not None:
            scheduler.step()
            logging.info(f"Epoch {epoch} - Learning Rate: {scheduler.get_last_lr()[0]}")

        save_checkpoint(
            model=unwrap_model(unet),
            epoch=epoch,
            checkpoint_dir=checkpoint_dir,
            script_name="unet_training"
        )

    unet.load_state_dict(best_model_wts)
    logging.info(f"Training complete. Best Validation Dice Score: {best_val_dice:.4f}")
    if writer is not None:
         writer.close()

def validate_model(unet, val_loader, device):
    unet.eval()
    dice_scores = {
    1: [],
    2: [],
    3: []
    }

    with torch.no_grad():
        for images, masks, t1_images in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = unet(images)
            preds = torch.sigmoid(outputs).squeeze(1)
            preds = outputs.argmax(dim=1)

            # Loop through each pair in the batch
            for pred, gt_mask in zip(preds, masks):
                # Loop over each label of interest
                for label_id in [1, 2, 3]:
                    # Pick out voxels == label_id
                    label_pred = (pred == label_id).float()
                    label_gt   = (gt_mask == label_id).float()

                    intersection = (label_pred * label_gt).sum()
                    union        = label_pred.sum() + label_gt.sum()

                    # Handle union=0 (no voxels of this label in pred/gt)
                    if union == 0:
                        dice = torch.tensor(1.0, device=device)  # or 0.0, depending on your convention
                    else:
                        dice = (2.0 * intersection) / union

                    # Store the per-slice Dice
                    dice_scores[label_id].append(dice.item())

    # After the loop, you can compute average Dice per label:
    for label_id in [1, 2, 3]:
        scores_for_label = dice_scores[label_id]
        if len(scores_for_label) == 0:
            avg_dice = 1
        else:
            avg_dice = sum(scores_for_label) / len(scores_for_label)
            
    unet.train()
    return avg_dice

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

    set_seed(42)

    writer = SummaryWriter(log_dir="runs/unet_training")

    GENERATOR_PATH = "../GAN/generator.pth"
    DISCRIMINATOR_PATH = "../GAN/discriminator.pth"
    CHECKPOINT_DIR = "./checkpoints"

    generator = UNet(n_channels=2, n_classes=1, bilinear=True)
    discriminator = Discriminator(in_channels=1)

    if not os.path.exists(GENERATOR_PATH):
        logging.error(f"Generator checkpoint not found at {GENERATOR_PATH}")
        return
    if not os.path.exists(DISCRIMINATOR_PATH):
        logging.error(f"Discriminator checkpoint not found at {DISCRIMINATOR_PATH}")
        return

    generator.load_state_dict(torch.load(GENERATOR_PATH, map_location='cpu'))
    discriminator.load_state_dict(torch.load(DISCRIMINATOR_PATH, map_location='cpu'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator.to(device)
    discriminator.to(device)

    generator.eval()
    discriminator.eval()
    for param in generator.parameters():
        param.requires_grad = False
    for param in discriminator.parameters():
        param.requires_grad = False

    unet_model = UNet(n_channels=4, n_classes=4, bilinear=True)
    unet_model.to(device)

    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs for training.")
        unet_model = nn.DataParallel(unet_model)
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)

    data_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    ])

    csv_path = "../data/selected_train_subject.csv"
    if not os.path.exists(csv_path):
        logging.error(f"CSV file not found at {csv_path}")
        return
    dataframe = pd.read_csv(csv_path)
    
    crop_optimizer = CropOptimizer(csv_path)
    crop_coords = None
    #crop_coords = crop_optimizer.find_optimal_crop()


    dataset = BrainDataset(dataframe=dataframe, crop_coords=crop_coords, transform=data_transforms)

    val_percent = 0.1
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    batch_size = 80
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True) if torch.cuda.is_available() else dict(batch_size=batch_size)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    criterion_segmentation = cross_entropy_loss
    dice_loss_fn = dice_loss

    criterion_adversarial = nn.BCELoss()

    sparsity_loss_fn = SparsityLoss(alpha=0.1)
    size_consistency_loss_fn = SizeConsistencyLoss(gamma=0.1)

    learning_rate = 1e-4
    adversarial_weight = 0.01
    optimizer = optim.Adam(unet_model.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    scaler = GradScaler()

    train_model(
        unet=unet_model,
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion_segmentation=criterion_segmentation,
        criterion_adversarial=criterion_adversarial,
        sparsity_loss_fn=sparsity_loss_fn,
        size_consistency_loss_fn=size_consistency_loss_fn,
        device=device,
        epochs=25,
        adversarial_weight=adversarial_weight,
        checkpoint_dir=CHECKPOINT_DIR,
        scheduler=scheduler,
        scaler=scaler,
        writer=writer,
        early_stopping_patience=5
    )

    writer.close()
    logging.info("Training complete.")

if __name__ == "__main__":
        main()
