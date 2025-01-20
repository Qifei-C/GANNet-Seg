from scipy.special import erf
from scipy.ndimage import laplace
print("Scipy imports successful")

import nibabel as nib
print("Nibabel imports successful")

from tqdm import tqdm
import logging
from pathlib import Path
import pandas as pd
import numpy as np
print("Training Phase Visualization and data writers imports successful")

import random
import copy
import re
import os
print("Other supplies successful")

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.nn import MaxPool2d
import torch.nn.functional as F
print("Torch supplies successful")
import torchvision
print("Line 1 supplies successful")
from torchvision import transforms
print("Line 2 supplies successful")
import torchvision.transforms.functional as TF
print("Line 3 supplies successful")


print("Torchvision imports successful")

def dice_coeff(input: torch.Tensor, target: torch.Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: torch.Tensor, target: torch.Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: torch.Tensor, target: torch.Tensor, multiclass: bool = False):
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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
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

class BrainSegmentationDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data_summary = pd.read_csv(csv_path)
        self.subjects = self.data_summary['Subject ID'].unique()
        self.transform = transform
        self.slice_info = self._create_slice_index()

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

    
    def _create_slice_index(self):
        """Create a list of (subject_id, slice_idx) pairs."""
        slice_info = []
        for subject_id in self.subjects:
            subject_data = self.data_summary[self.data_summary['Subject ID'] == subject_id]
            flair_path = subject_data[subject_data['Scan Type'] == 'flair']['File Path'].values[0]
            nii = nib.load(flair_path)
            depth = nii.shape[2]  # Assume all modalities have the same depth
            slice_info.extend([(subject_id, z) for z in range(depth)])
        return slice_info

    def __len__(self):
        return len(self.slice_info)
        
    
        
    def __getitem__(self, idx):
        subject_id, slice_idx = self.slice_info[idx]
        subject_data = self.data_summary[self.data_summary['Subject ID'] == subject_id]
        modalities = ['flair', 't1', 't1ce', 't2']
        slices = []
        for modality in modalities:
            file_path = subject_data[subject_data['Scan Type'] == modality]['File Path'].values[0]
            nii = nib.load(file_path)
            image = nii.get_fdata().astype(np.float32)
            image = (image - np.mean(image)) / np.std(image) 
            image = Img_proc(image)
            slices.append(image[:, :, slice_idx])  

        images = np.stack(slices, axis=0)
        seg_data = subject_data[subject_data['Scan Type'] == 'seg']
        if seg_data.empty:
            raise ValueError(f"Missing segmentation mask for subject {subject_id}")
        seg_path = seg_data['File Path'].values[0]
        seg_nii = nib.load(seg_path)
        seg_mask = seg_nii.get_fdata().astype(np.uint8)
        seg_mask[seg_mask == 4] = 3
        seg_slice = seg_mask[:, :, slice_idx]  

        if self.transform:
            images, seg_slice = self.transform(images, seg_slice)

        return torch.tensor(images, dtype=torch.float32), torch.tensor(seg_slice, dtype=torch.long)



csv_path = '../data/selected_test_subject.csv'
dataset = BrainSegmentationDataset(csv_path)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)


for batch_idx, (images, masks) in enumerate(dataloader):
    print(f"Batch {batch_idx+1}")
    print(f"Images shape: {images.shape}")
    print(f"Masks shape: {masks.shape}")
    break

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
    pin_memory=False,
    checkpoint_dir: str = "./checkpoints"
):
    
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()

    logging.info(f"Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{epochs}", unit="batch") as pbar:
            for images, masks in train_loader:
                images = images.to(device, dtype=torch.float32)
                masks = masks.to(device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    predictions = model(images)
                    if model.n_classes == 1:
                        loss = criterion(predictions.squeeze(1), masks.float())
                        loss += dice_loss(torch.sigmoid(predictions.squeeze(1)), masks.float(), multiclass=False)
                    else:
                        loss = criterion(predictions, masks)
                        loss += dice_loss(
                            torch.softmax(predictions, dim=1),
                            torch.nn.functional.one_hot(masks, num_classes=model.n_classes)
                                .permute(0, 3, 1, 2)
                                .float(),
                            multiclass=True
                        )

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

        val_score = evaluate(model, val_loader, device, amp)
        logging.info(f"Epoch {epoch} - Validation Dice Score: {val_score:.4f}")

        if save_checkpoint:
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path / f"checkpoint_epoch{epoch}.pth")
            logging.info(f"Checkpoint saved at epoch {epoch}")

def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()
    return dice_score / max(num_val_batches, 1)
