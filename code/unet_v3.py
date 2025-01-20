import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import logging
import wandb
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import os
import numpy as np
import nibabel as nib
import pandas as pd
from torch import Tensor
import logging
import wandb
import torchvision.transforms.functional as TF 
from scipy.special import erf
import re


class BrainSegmentationDataset(Dataset):
    def __init__(self, csv_path, crop_coords=None, transform=None):
        self.data_summary = pd.read_csv(csv_path)
        self.subjects = self.data_summary['Subject ID'].unique()
        self.transform = transform
        self.crop_coords = crop_coords
        self.slice_info = self._create_slice_index()

    def _create_slice_index(self):
        slice_info = []
        for subject_id in self.subjects:
            subject_data = self.data_summary[self.data_summary['Subject ID'] == subject_id]
            flair_path = subject_data[subject_data['Scan Type'] == 'flair']['File Path'].values[0]
            nii = nib.load(flair_path)
            image = nii.get_fdata().astype(np.float32)
            
            depth = image.shape[2] 
            for z in range(15, depth - 12):  
                slice_image = image[:, :, z]
                if np.any(slice_image > 0): 
                    slice_info.append((subject_id, z))
        return slice_info


    def __len__(self):
        return len(self.slice_info)
    
    @staticmethod
    def Img_proc(image, _lambda=-0.8, epsilon=1e-6):

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


    def __getitem__(self, idx):
        subject_id, slice_idx = self.slice_info[idx]
        subject_data = self.data_summary[self.data_summary['Subject ID'] == subject_id]

        # Load input modalities
        modalities = ['flair', 't1', 't1ce', 't2']
        slices = []
        for modality in modalities:
            file_path = subject_data[subject_data['Scan Type'] == modality]['File Path'].values[0]
            nii = nib.load(file_path)
            image = nii.get_fdata().astype(np.float32)
            image = (image - np.mean(image)) / np.std(image)  
            
            if self.crop_coords:
                min_x, max_x, min_y, max_y = self.crop_coords
                image = image[min_x:max_x, min_y:max_y, :] 
            
            slice_image = image[:, :, slice_idx]
            slice_image = self.Img_proc(slice_image)
                
            slices.append(slice_image) 
        images = np.stack(slices, axis=0)
        
        seg_data = subject_data[subject_data['Scan Type'] == 'seg']
        if seg_data.empty:
            raise ValueError(f"Missing segmentation mask for subject {subject_id}")
        seg_path = seg_data['File Path'].values[0]
        seg_nii = nib.load(seg_path)
        seg_mask = seg_nii.get_fdata().astype(np.uint8)
        
        seg_mask[seg_mask == 4] = 3
        seg_slice = seg_mask[:, :, slice_idx] 
        if self.crop_coords:
            seg_mask = seg_mask[min_x:max_x, min_y:max_y, :]  
        
        seg_slice = seg_mask[:, :, slice_idx]

        if self.transform:
            images, seg_slice = self.transform(images, seg_slice)

        return torch.tensor(images, dtype=torch.float32), torch.tensor(seg_slice, dtype=torch.long)


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

    def visualize_crop(self, crop_coords):
        """Visualizes the crop rectangle on the summed intensity plot."""
        min_x, max_x, min_y, max_y = crop_coords
        
        summed_image = None
        for subject_id in self.data_summary['Subject ID'].unique():
            subject_data = self.data_summary[self.data_summary['Subject ID'] == subject_id]
            for modality in self.modalities:
                file_path = subject_data[subject_data['Scan Type'] == modality]['File Path'].values[0]
                nii = nib.load(file_path)
                data = nii.get_fdata()
                summed_image = data.sum(axis=2) if summed_image is None else summed_image + data.sum(axis=2)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(summed_image, cmap="gray")
        plt.gca().add_patch(plt.Rectangle(
            (min_y, min_x),  # Rectangle origin (top-left corner)
            max_y - min_y,  # Width
            max_x - min_x,  # Height
            edgecolor='red', linewidth=2, fill=False
        ))
        plt.title("Summed Intensity with Crop Rectangle")
        plt.xlabel("Width (pixels)")
        plt.ylabel("Height (pixels)")
        plt.colorbar(label="Summed Intensity")
        plt.show()


import pandas as pd
import random

def split_subjects(csv_path, train_csv_path, test_csv_path, test_subject_ids=None, test_split=0.2, seed=42):
    """
    Splits the subjects into training and testing datasets and saves them as new CSVs.

    Args:
        csv_path (str): Path to the original CSV file.
        train_csv_path (str): Path to save the training CSV file.
        test_csv_path (str): Path to save the testing CSV file.
        test_subject_ids (list, optional): Specific subject IDs for the test set. Defaults to None.
        test_split (float): Proportion of subjects to use for the test set (if `test_subject_ids` is None).
        seed (int): Random seed for reproducibility.

    Returns:
        None
    """
    # Read the original CSV
    data = pd.read_csv(csv_path)
    all_subjects = data['Subject ID'].unique()

    # Select test subjects
    if test_subject_ids is None:
        random.seed(seed)
        test_subject_ids = random.sample(list(all_subjects), int(len(all_subjects) * test_split))
    print(f"Test subjects: {test_subject_ids}")

    # Split data
    test_data = data[data['Subject ID'].isin(test_subject_ids)]
    train_data = data[~data['Subject ID'].isin(test_subject_ids)]

    # Save to CSV
    train_data.to_csv(train_csv_path, index=False)
    test_data.to_csv(test_csv_path, index=False)
    print(f"Train subjects saved to {train_csv_path}")
    print(f"Test subjects saved to {test_csv_path}")

# Example usage
csv_path = "../data/training_detailed_summary_2020.csv"
train_csv_path = "../data/selected_train_subject.csv"
test_csv_path = "../data/selected_test_subject.csv"

test_subject_ids = None
split_subjects(csv_path, train_csv_path, test_csv_path, test_subject_ids=None, test_split=0.1)



csv_path = "../data/selected_train_subject.csv"

crop_optimizer = CropOptimizer(csv_path)
crop_coords = crop_optimizer.find_optimal_crop()
print(f"Optimal crop coordinates: {crop_coords}")



def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

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

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
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

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


def unwrap_model(model):
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    return model



def get_latest_checkpoint(checkpoint_dir: str):
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Extract script name dynamically
    script_name = Path(__file__).stem  # e.g., "unet_v3"
    start_epoch = 1
    latest_checkpoint = None

    if checkpoint_path.exists():
        # Use glob to filter checkpoints matching the script name
        checkpoints = list(checkpoint_path.glob(f"checkpoint_epoch{script_name}*.pth"))
        if checkpoints:
            # Extract epoch numbers dynamically using regex
            checkpoint_epochs = []
            for checkpoint in checkpoints:
                match = re.search(rf'checkpoint_epoch{script_name}(\d+)\.pth$', checkpoint.name)
                if match:
                    epoch = int(match.group(1))
                    checkpoint_epochs.append((epoch, checkpoint))

            if checkpoint_epochs:
                # Find the checkpoint with the maximum epoch
                latest_epoch, latest_checkpoint = max(checkpoint_epochs, key=lambda x: x[0])
                start_epoch = latest_epoch + 1

    return latest_checkpoint, start_epoch, script_name

def train_model(
    model,
    dataset,
    device,
    epochs: int = 5,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    val_percent: float = 0.1,
    save_checkpoint: bool = True,
    amp: bool = False,
    weight_decay: float = 1e-8,
    momentum: float = 0.999,
    gradient_clipping: float = 1.0,
    pin_memory=True,
    checkpoint_dir: str = "./checkpoints"
):
    
    latest_checkpoint, start_epoch, script_name = get_latest_checkpoint(checkpoint_dir)
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    print("Checkpoint keys:", checkpoint.keys())




    # Load model and optimizer state if a checkpoint exists
    if latest_checkpoint:
        logging.info(f"Resuming training from {latest_checkpoint}, starting at epoch {start_epoch}.")
        checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=True)

        try:
            model.load_state_dict(checkpoint)
        except RuntimeError as e:
            logging.warning(f"Mismatch during state_dict loading: {e}")
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k.replace("module.", "")  
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        logging.info(f"No checkpoints found for {script_name}. Starting training from scratch.")
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    grad_scaler = torch.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if unwrap_model(model).n_classes > 1 else nn.BCEWithLogitsLoss()

    logging.info(f"Starting training for {epochs} epochs...")
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{epochs}", unit="batch") as pbar:
            for images, masks in train_loader:
                images = images.to(device, dtype=torch.float32)
                masks = masks.to(device, dtype=torch.long)

                # Forward pass
                with torch.amp.autocast(device_type=device.type if device.type != 'cuda' else 'cpu', enabled=amp):
                    predictions = model(images)
                    if unwrap_model(model).n_classes == 1:
                        loss = criterion(predictions.squeeze(1), masks.float())
                        loss += dice_loss(torch.sigmoid(predictions.squeeze(1)), masks.float(), multiclass=False)
                    else:
                        loss = criterion(predictions, masks)
                        loss += dice_loss(
                            torch.softmax(predictions, dim=1),
                            torch.nn.functional.one_hot(masks, num_classes=unwrap_model(model).n_classes)
                                .permute(0, 3, 1, 2)
                                .float(),
                            multiclass=True
                        )

                # Backward pass
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(1)
                epoch_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

        logging.info(f"Epoch {epoch} - Training loss: {epoch_loss:.4f}")

        # Validation loop
        val_score = evaluate(model, val_loader, device, amp)
        logging.info(f"Epoch {epoch} - Validation Dice Score: {val_score:.4f}")
        
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        

        # Save checkpoint
        if save_checkpoint:
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path / f"checkpoint_epoch{script_name}{epoch}.pth")
            logging.info(f"Checkpoint saved at epoch {epoch}")

def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'cuda' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if unwrap_model(net).n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < unwrap_model(net).n_classes, 'True mask indices should be in [0, n_classes['
                mask_true = F.one_hot(mask_true, unwrap_model(net).n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), unwrap_model(net).n_classes).permute(0, 3, 1, 2).float()
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()
    return dice_score / max(num_val_batches, 1)


device = torch.device('cuda')
dataset = BrainSegmentationDataset(csv_path, crop_coords=crop_coords)


sample_image, sample_mask = dataset[0]
print(f"Sample image shape: {sample_image.shape}, Sample mask shape: {sample_mask.shape}")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")

# Initialize the model
model = UNet(n_channels=4, n_classes=4)  
if num_gpus > 1:
    model = torch.nn.DataParallel(model)  

model.to(device)

train_model(
    model=model,
    dataset=dataset,
    device=device,
    epochs=20,
    batch_size=80,
    learning_rate=1e-5,
    checkpoint_dir="./checkpoints"
)
