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
import torchvision.transforms.functional as TF
from scipy.special import erf
import random

# Image Processing Function
def Img_proc(image, _lambda=-0.8, epsilon=1e-6):
    I_img = image
    I_img_norm = (I_img - np.min(I_img)) / (np.max(I_img) - np.min(I_img) + epsilon)

    # Step 1: Compute IMG1
    max_I_img = np.max(I_img_norm)
    IMG1 = (max_I_img / np.log(max_I_img + 1 + epsilon)) * np.log(I_img_norm + 1)

    # Step 2: Compute IMG2
    IMG2 = 1 - np.exp(-I_img_norm)

    # Step 3: Compute IMG3
    IMG3 = (IMG1 + IMG2) / (_lambda + (IMG1 * IMG2))

    # Step 4: Compute IMG4
    IMG4 = erf(_lambda * np.arctan(np.exp(IMG3)) - 0.5 * IMG3)

    # Step 5: Compute IMG5 (Normalization)
    min_IMG4 = np.min(IMG4)
    max_IMG4 = np.max(IMG4)
    IMG5 = (IMG4 - min_IMG4) / (max_IMG4 - min_IMG4 + epsilon)

    return IMG5

# Enhanced Mask Generation Function
def generate_mask(image_shape, **kwargs):
    """
    Generates a mask based on the specified type.

    Args:
        image_shape (tuple): Shape of the image tensor (C, H, W).
        **kwargs: Additional parameters for mask generation.

    Returns:
        torch.Tensor: Mask tensor with the same spatial dimensions as the image.
    """
    C, H, W = image_shape
    mask = torch.ones((1, H, W), dtype=torch.float32)
    
    # Randomly choose mask type
    mask_type = random.choice(['random_square', 'circle', 'center_square'])

    if mask_type == 'random_square':
        # Example: Random square mask
        mask_size = kwargs.get('mask_size', 64)
        if H - mask_size > 0 and W - mask_size > 0:
            x = random.randint(0, H - mask_size)
            y = random.randint(0, W - mask_size)
            mask[:, x:x + mask_size, y:y + mask_size] = 0
        else:
            print("Mask size too large for the image dimensions. Skipping mask.")

    elif mask_type == 'center_square':
        # Example: Center square mask
        mask_size = kwargs.get('mask_size', 64)
        if mask_size < H and mask_size < W:
            x = (H - mask_size) // 2
            y = (W - mask_size) // 2
            mask[:, x:x + mask_size, y:y + mask_size] = 0
        else:
            print("Mask size too large for the image dimensions. Skipping mask.")

    elif mask_type == 'circle':
        # Example: Circular mask
        radius = kwargs.get('radius', min(H, W) // 8)
        center_x = H // 2
        center_y = W // 2
        Y, X = torch.meshgrid(torch.arange(H), torch.arange(W))
        dist = (X - center_y) ** 2 + (Y - center_x) ** 2
        mask[0, :, :] = (dist > radius ** 2).float()

    # Add more mask types as needed

    return mask

# Define the U-Net architecture (Generator)
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
        # If you have padding issues, see the comments in your original code
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

# Define the Dataset Class
class BrainDataset(Dataset):
    def __init__(self, dataframe, transform=None, mask_type='random_square', mask_kwargs=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing file paths.
            transform (callable, optional): Optional transform to be applied on a sample.
            mask_type (str): Type of mask to generate.
            mask_kwargs (dict): Additional arguments for mask generation.
        """
        self.dataframe = dataframe
        self.transform = transform
        self.mask_type = mask_type
        self.mask_kwargs = mask_kwargs if mask_kwargs is not None else {}

    def __len__(self):
        return len(self.dataframe) * self.get_num_slices(self.dataframe.iloc[0]['File Path'])

    def get_num_slices(self, file_path):
        mri_image = nib.load(file_path)
        return mri_image.shape[2]

    def __getitem__(self, idx):
        file_index = idx // self.get_num_slices(self.dataframe.iloc[0]['File Path'])
        slice_index = idx % self.get_num_slices(self.dataframe.iloc[0]['File Path'])
        img_path = self.dataframe.iloc[file_index]['File Path']
        image = self.load_mri_slice(img_path, slice_index)

        if self.transform:
            image = self.transform(image)

        # Generate mask using the custom mask generation function
        mask = generate_mask(image.shape, **self.mask_kwargs)

        # Original image minus the mask
        original_minus_mask = image * mask

        # Mask area slice (where mask == 0)
        mask_area = (1 - mask) * image  # This will zero out the unmasked regions

        # Prepare the generator input by concatenating masked area and original minus mask
        generator_input = torch.cat([mask_area, original_minus_mask], dim=0)  # Shape: (2, H, W)

        return generator_input, image, mask

    def load_mri_slice(self, file_path, slice_index):
        mri_image = nib.load(file_path)
        image = mri_image.get_fdata()[:, slice_index, :]  # Assuming the slice is along the second axis
        image = np.expand_dims(image, axis=0)  # Add channel dimension
        image = torch.tensor(image, dtype=torch.float32)
        image = TF.rotate(image, 90)

        # Convert back to numpy array for image enhancement
        image_np = image.squeeze().numpy()

        # Apply image enhancement
        enhanced_image_np = Img_proc(image_np)

        # Convert back to tensor
        enhanced_image = torch.tensor(enhanced_image_np, dtype=torch.float32).unsqueeze(0)

        return enhanced_image

# Define the Discriminator
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

# Load the Dataset
summary_file = '../data/NFBS_detailed_summary.csv'
filtered_df = pd.read_csv(summary_file)

# Example usage of different mask types
mask_type = 'random_square'  # Change as needed: 'random_square', 'center_square', 'circle', etc.
mask_kwargs = {'mask_size': 64}  # Parameters for the mask

# Initialize the dataset
dataset = BrainDataset(filtered_df, mask_type=mask_type, mask_kwargs=mask_kwargs)

# Determine the device to use (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print(f"Number of GPUs available: {n_gpu}")

# Initialize DataLoader with multi-GPU considerations
dataloader = DataLoader(
    dataset,
    batch_size=80,
    shuffle=True,
    num_workers=4,       # Adjust based on your system
    pin_memory=True      # Improves data transfer speed to GPU
)

# Initialize models
generator = UNet(n_channels=2, n_classes=1, bilinear=True).to(device)  # Input: 2 channels
discriminator = Discriminator(in_channels=1).to(device)  # Discriminator expects 1-channel images

# Wrap models with DataParallel if multiple GPUs are available
if n_gpu > 1:
    generator = nn.DataParallel(generator)
    discriminator = nn.DataParallel(discriminator)
    print("Models are wrapped with DataParallel for multi-GPU support.")

# Loss functions
adversarial_loss = nn.BCELoss()  # Binary Cross Entropy Loss
l1_loss = nn.L1Loss()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

# Training parameters
num_epochs = 30
checkpoint_dir = "./checkpoints"

# Ensure checkpoint directory exists
Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

# Training Loop with Discriminator Update Scheduling
for epoch in range(num_epochs):
    generator.train()
    discriminator.train()
    
    epoch_d_loss = 0.0
    epoch_g_loss = 0.0

    # Initialize a counter for generator updates
    generator_update_counter = 0

    for i, (generator_inputs, real_images, masks) in enumerate(dataloader):
        generator_inputs = generator_inputs.to(device)  # Shape: (B, 2, H, W)
        real_images = real_images.to(device)            # Shape: (B, 1, H, W)
        masks = masks.to(device)                        # Shape: (1, H, W)

        # =====================
        # Train Generator 5 Times
        # =====================
        for _ in range(5):
            optimizer_G.zero_grad()

            # Forward pass
            fake_images = generator(generator_inputs)

            # Adversarial loss
            fake_preds = discriminator(fake_images)
            real_targets = torch.ones_like(fake_preds, device=device)  # Labels for generator to fool discriminator
            g_adv_loss = adversarial_loss(fake_preds, real_targets)

            # L1 loss
            g_l1_loss = l1_loss(fake_images, real_images) * 100  # Adjust the weight as needed

            # Total generator loss
            g_loss = g_adv_loss + g_l1_loss

            # Backward pass and optimization
            g_loss.backward()
            optimizer_G.step()

            # Accumulate generator loss
            epoch_g_loss += g_loss.item()
            generator_update_counter += 1

        # =====================
        # Train Discriminator Once
        # =====================
        optimizer_D.zero_grad()

        # Real images
        real_preds = discriminator(real_images)
        real_targets = torch.ones_like(real_preds, device=device)  # Real labels
        real_loss = adversarial_loss(real_preds, real_targets)

        # Fake images
        fake_images = generator(generator_inputs).detach()  # Detach to avoid backprop through generator
        fake_preds = discriminator(fake_images)
        fake_targets = torch.zeros_like(fake_preds, device=device)  # Fake labels
        fake_loss = adversarial_loss(fake_preds, fake_targets)

        # Total discriminator loss
        d_loss = (real_loss + fake_loss) / 2

        # Backward pass and optimization
        d_loss.backward()
        optimizer_D.step()

        # Accumulate discriminator loss
        epoch_d_loss += d_loss.item()

        # Optional: Print losses every certain number of iterations
        if (i + 1) % 50 == 0 or (i + 1) == len(dataloader):
            print(f"[Epoch {epoch+1}/{num_epochs}] [Batch {i+1}/{len(dataloader)}] "
                  f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

    # Calculate average losses for the epoch
    avg_d_loss = epoch_d_loss / len(dataloader)
    avg_g_loss = epoch_g_loss / (len(dataloader) * 5)  # Since generator was updated 5 times per batch
    print(f"==> Epoch {epoch+1}/{num_epochs} Summary: [D loss: {avg_d_loss:.4f}] [G loss: {avg_g_loss:.4f}]")

    # Optionally save the model checkpoints
    if (epoch + 1) % 1 == 0:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save generator
        if isinstance(generator, nn.DataParallel):
            torch.save(generator.module.state_dict(), checkpoint_path / f"generator_epoch_{epoch + 1}.pth")
        else:
            torch.save(generator.state_dict(), checkpoint_path / f"generator_epoch_{epoch + 1}.pth")
        
        # Save discriminator
        if isinstance(discriminator, nn.DataParallel):
            torch.save(discriminator.module.state_dict(), checkpoint_path / f"discriminator_epoch_{epoch + 1}.pth")
        else:
            torch.save(discriminator.state_dict(), checkpoint_path / f"discriminator_epoch_{epoch + 1}.pth")
        
        print(f"Saved checkpoints for epoch {epoch + 1} to {checkpoint_dir}")

print("Training finished.")
