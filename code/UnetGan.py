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

# ===============================
# Utility Functions
# ===============================

def dice_coeff(input: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6):
    """
    Compute Dice Coefficient.
    """
    assert input.size() == target.size(), "Input and target must have the same shape."
    input = input.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (input * target).sum()
    dice = (2. * intersection + epsilon) / (input.sum() + target.sum() + epsilon)
    return dice

def dice_loss(input: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6):
    """
    Compute Dice Loss.
    """
    return 1 - dice_coeff(input, target, epsilon)

def unwrap_model(model):
    """
    Unwrap the model if it's wrapped in DataParallel.
    """
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    return model

def get_latest_checkpoint(checkpoint_dir: str, script_name: str):
    """
    Automatically detects the latest checkpoint based on the script name and epoch number.
    
    Args:
        checkpoint_dir (str): Directory where checkpoints are saved.
        script_name (str): Name of the training script.
    
    Returns:
        Tuple[str, int]: Path to the latest checkpoint and the next epoch number.
    """
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # List all checkpoint files matching the pattern
    checkpoints = list(checkpoint_path.glob(f"checkpoint_epoch{script_name}_*.pth"))
    if not checkpoints:
        return None, 1  # No checkpoint found, start from epoch 1

    # Extract epoch numbers and find the latest
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
    """
    Saves the model's state_dict as a checkpoint.
    
    Args:
        model (nn.Module): The U-Net model.
        epoch (int): Current epoch number.
        checkpoint_dir (str): Directory to save checkpoints.
        script_name (str): Name of the training script.
    """
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
        return np.zeros_like(I_img)  # Avoid division by zero
    
    I_img_norm = (I_img - min_val) / (max_val - min_val + epsilon)
    
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
    if max_IMG4 == min_IMG4:
        return np.zeros_like(IMG4)  # Avoid division by zero
    
    IMG5 = (IMG4 - min_IMG4) / (max_IMG4 - min_IMG4 + epsilon)
    
    return IMG5

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
            flair_path = subject_data[subject_data['Scan Type'] == 'flair']['File Path'].values[0]
            nii = nib.load(flair_path)
            image = nii.get_fdata().astype(np.float32)
            depth = image.shape[2]
            for z in range(15, depth - 12):  # Exclude the first 15 and last 12 slices
                slice_image = image[:, :, z]
                if np.any(slice_image > 0):
                    slice_info.append((subject_id, z))
        return slice_info
    
    def __len__(self):  
        return len(self.slice_info)  
    
    def __getitem__(self, idx):  
        subject_id, slice_idx = self.slice_info[idx]
        subject_data = self.dataframe[self.dataframe['Subject ID'] == subject_id]
    
        # Load input modalities
        modalities = ['flair', 't1', 't1ce', 't2']
        slices = []
        for modality in modalities:
            file_path = subject_data[subject_data['Scan Type'] == modality]['File Path'].values[0]
            nii = nib.load(file_path)
            image = nii.get_fdata().astype(np.float32)
            image = (image - np.mean(image)) / (np.std(image) + 1e-6)  # Normalize to avoid division by zero
            
            # Crop if coordinates are provided
            if self.crop_coords:
                min_x, max_x, min_y, max_y = self.crop_coords
                image = image[min_x:max_x, min_y:max_y, :]  # Crop spatial dimensions
            
            slice_image = image[:, :, slice_idx]
            slice_image = Img_proc(slice_image)
            slices.append(slice_image)
        
        # Stack modalities into a single tensor (C x H x W)
        images = np.stack(slices, axis=0)
        
        # Load segmentation mask
        seg_data = subject_data[subject_data['Scan Type'] == 'seg']
        if seg_data.empty:
            raise ValueError(f"Missing segmentation mask for subject {subject_id}")
        seg_path = seg_data['File Path'].values[0]
        seg_nii = nib.load(seg_path)
        seg_mask = seg_nii.get_fdata().astype(np.uint8)
        
        # Remap labels if necessary (e.g., 4 -> 3)
        seg_mask[seg_mask == 4] = 3
        seg_slice = seg_mask[:, :, slice_idx]
        
        # Crop mask if coordinates are provided
        if self.crop_coords:
            seg_mask = seg_mask[min_x:max_x, min_y:max_y, :]  # Crop spatial dimensions
            seg_slice = seg_mask[:, :, slice_idx]
        
        if self.transform:
            # Apply transforms to images and masks if necessary
            # This requires custom transform logic compatible with both images and masks
            pass  # Implement custom transform logic as needed
        
        # Extract T1 image (assuming it's the second modality)
        t1_image = images[1, :, :]  # (H, W)
        
        return (
            torch.tensor(images, dtype=torch.float32),       # (C, H, W)
            torch.tensor(seg_slice, dtype=torch.long),       # (H, W)
            torch.tensor(t1_image, dtype=torch.float32)      # (H, W)
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

# ===============================
# Main Training Function
# ===============================

def train_model(
    unet,
    generator,
    discriminator,
    train_loader,
    val_loader,
    optimizer,
    criterion_segmentation,
    criterion_adversarial,
    device,
    epochs=25,
    adversarial_weight=0.1,
    checkpoint_dir="./checkpoints",
    scheduler=None,
    scaler=None,
    writer=None,
    early_stopping_patience=5
):
    """
    Trains the U-Net model with adversarial loss from a pre-trained GAN.

    Args:
        unet (nn.Module): The U-Net segmentation model.
        generator (nn.Module): The pre-trained Generator model.
        discriminator (nn.Module): The pre-trained Discriminator model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer for U-Net.
        criterion_segmentation (nn.Module): Segmentation loss function.
        criterion_adversarial (nn.Module): Adversarial loss function.
        device (torch.device): Device to run training on.
        epochs (int): Number of training epochs.
        adversarial_weight (float): Weight to balance segmentation and adversarial loss.
        checkpoint_dir (str): Directory to save checkpoints.
        scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler.
        scaler (torch.cuda.amp.GradScaler, optional): GradScaler for AMP.
        writer (SummaryWriter, optional): TensorBoard writer.
        early_stopping_patience (int): Number of epochs to wait for improvement before stopping.
    """
    unet.train()  # Set U-Net to training mode
    best_val_dice = 0.0
    trigger_times = 0
    best_model_wts = copy.deepcopy(unet.state_dict())

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        unet.train()  # Ensure the model is in training mode

        with tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch") as tepoch:
            for batch_idx, (images, masks, t1_images) in enumerate(tepoch):
                # Move data to device
                images = images.to(device)        # (B, C, H, W)
                masks = masks.to(device)          # (B, H, W)
                t1_images = t1_images.to(device)  # (B, H, W)
                # Prepare input for Generator: concatenate mask and T1 image
                
                mask_expanded = masks.unsqueeze(1).float()          # (B, 1, H, W)
                t1_images_expanded = t1_images.unsqueeze(1).float()  # (B, 1, H, W)
                gen_input = torch.cat([mask_expanded, t1_images_expanded], dim=1)  # (B, 2, H, W)

                # Forward pass through U-Net
                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=(scaler is not None)):
                    outputs = unet(images)            # (B, 1, H, W)

                # Compute segmentation loss
                    loss_seg = criterion_segmentation(outputs.squeeze(1), masks.float())
                    loss_dice = dice_loss(torch.sigmoid(outputs.squeeze(1)), masks.float())
                    loss_total_seg = loss_seg + loss_dice

                # Generate images using the Generator
                    with torch.no_grad():  # No gradients for Generator
                        generated_images = generator(gen_input)         # (B, 1, H, W)

                # Pass generated images to Discriminator
                    preds_discriminator = discriminator(generated_images)  # (B, 1, H', W') or (B, 1)

                # Create real labels (1) to encourage U-Net to produce realistic masks
                    real_labels = torch.ones_like(preds_discriminator, device=device)

                # Compute adversarial loss
                    loss_adv = criterion_adversarial(preds_discriminator, real_labels)

                # Total loss
                    loss = loss_total_seg + adversarial_weight * loss_adv

                # Backward pass and optimization
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                
                else:
                    loss.backward()
                    optimizer.step()

                # Update epoch loss
                epoch_loss += loss.item()
                    
                # Update progress bar
                tepoch.set_postfix({"Loss": loss.item()})
                
                # Log to TensorBoard
                if writer is not None:
                    global_step = (epoch - 1) * len(train_loader) + batch_idx
                    writer.add_scalar('Loss/Train_Segmentation', loss_total_seg.item(), global_step)
                    writer.add_scalar('Loss/Train_Adversarial', loss_adv.item(), global_step)
                    writer.add_scalar('Loss/Train_Total', loss.item(), global_step)

        avg_epoch_loss = epoch_loss / len(train_loader)
        logging.info(f"Epoch {epoch}/{epochs} - Training Loss: {avg_epoch_loss:.4f}")

        # Perform validation
        val_dice = validate_model(unet, val_loader, device)
        logging.info(f"Epoch {epoch}/{epochs} - Validation Dice Score: {val_dice:.4f}")

        if writer is not None:
            writer.add_scalar('Dice_Score/Validation', val_dice, epoch)

        # Check for improvement
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

            # Step the scheduler if provided
        if scheduler is not None:
            scheduler.step()
            logging.info(f"Epoch {epoch} - Learning Rate: {scheduler.get_last_lr()[0]}")

        # Save checkpoint
        save_checkpoint(
            model=unwrap_model(unet),
            epoch=epoch,
            checkpoint_dir=checkpoint_dir,
            script_name="unet_training"  # Adjust based on your script name
        )

        # Load best model weights after training
    unet.load_state_dict(best_model_wts)
    logging.info(f"Training complete. Best Validation Dice Score: {best_val_dice:.4f}")
    if writer is not None:
         writer.close()


def validate_model(unet, val_loader, device):
    """
    Validates the U-Net model on the validation dataset.

    Args:
        unet (nn.Module): The U-Net segmentation model.
        val_loader (DataLoader): DataLoader for validation data.
        device (torch.device): Device to run validation on.

    Returns:
        float: Average Dice Score over the validation set.
    """
    unet.eval()  # Set U-Net to evaluation mode
    dice_scores = []

    with torch.no_grad():
        for images, masks, t1_images in val_loader:
            # Move data to device
            images = images.to(device)        # (B, C, H, W)
            masks = masks.to(device)          # (B, H, W)
            t1_images = t1_images.to(device)  # (B, H, W)

            # Forward pass through U-Net
            outputs = unet(images)            # (B, 1, H, W)
            preds = torch.sigmoid(outputs).squeeze(1)  # (B, H, W)

            # Binarize predictions
            preds = (preds > 0.5).float()

            # Compute Dice Score for each image in the batch
            for pred, mask in zip(preds, masks):
                intersection = (pred * mask.float()).sum()
                union = pred.sum() + mask.float().sum()
                if union == 0:
                    dice = 1.0  # If both pred and mask are empty
                else:
                    dice = (2. * intersection) / union
                dice_scores.append(dice.item())

    # Calculate average Dice Score
    avg_dice = sum(dice_scores) / len(dice_scores)
    unet.train()  # Set U-Net back to training mode
    return avg_dice

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

    # Set random seed for reproducibility
    set_seed(42)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir="runs/unet_training")

    # Paths to the pre-trained Generator and Discriminator models
    GENERATOR_PATH = "../GAN/generator.pth"       # Update with your Generator path
    DISCRIMINATOR_PATH = "../GAN/discriminator.pth"  # Update with your Discriminator path
    CHECKPOINT_DIR = "./checkpoints"

    # Initialize Generator and Discriminator
    generator = UNet(n_channels=2, n_classes=1, bilinear=True)  # Mask + T1 as input
    discriminator = Discriminator(in_channels=1)                # Generated image has 1 channel

    # Load pre-trained weights
    if not os.path.exists(GENERATOR_PATH):
        logging.error(f"Generator checkpoint not found at {GENERATOR_PATH}")
        return
    if not os.path.exists(DISCRIMINATOR_PATH):
        logging.error(f"Discriminator checkpoint not found at {DISCRIMINATOR_PATH}")
        return

    generator.load_state_dict(torch.load(GENERATOR_PATH, map_location='cpu'))
    discriminator.load_state_dict(torch.load(DISCRIMINATOR_PATH, map_location='cpu'))

    # Move models to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator.to(device)
    discriminator.to(device)

    # Set Generator and Discriminator to evaluation mode and freeze parameters
    generator.eval()
    discriminator.eval()
    for param in generator.parameters():
        param.requires_grad = False
    for param in discriminator.parameters():
        param.requires_grad = False

    # Initialize U-Net model
    unet_model = UNet(n_channels=4, n_classes=1, bilinear=True)  # Adjust n_channels based on your data
    unet_model.to(device)

    # Enable Multi-GPU support if available
    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs for training.")
        unet_model = nn.DataParallel(unet_model)
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)

    # Define data augmentation transforms
    data_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        # Add more transforms as needed
    ])

    # Load dataset
    csv_path = "../data/selected_train_subject.csv"  # Update with your CSV path
    if not os.path.exists(csv_path):
        logging.error(f"CSV file not found at {csv_path}")
        return
    dataframe = pd.read_csv(csv_path)
    dataset = BrainDataset(dataframe=dataframe, crop_coords=None, transform=data_transforms)  # Provide crop_coords if necessary

    # Split dataset into training and validation sets
    val_percent = 0.1
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    # Create DataLoader objects
    batch_size = 16  # Adjust based on GPU memory
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True) if torch.cuda.is_available() else dict(batch_size=batch_size)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # Define loss functions
    criterion_segmentation = nn.BCEWithLogitsLoss()
    criterion_adversarial = nn.BCEWithLogitsLoss()

    # Define optimizer
    learning_rate = 1e-4
    adversarial_weight = 0.1  # Weight to balance segmentation and adversarial loss
    optimizer = optim.Adam(unet_model.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # Define Learning Rate Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Initialize GradScaler for AMP
    scaler = GradScaler()

    # Start training
    train_model(
        unet=unet_model,
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion_segmentation=criterion_segmentation,
        criterion_adversarial=criterion_adversarial,
        device=device,
        epochs=25,
        adversarial_weight=adversarial_weight,
        checkpoint_dir=CHECKPOINT_DIR,
        scheduler=scheduler,
        scaler=scaler,
        writer=writer,
        early_stopping_patience=5
    )

    # Close the TensorBoard writer
    writer.close()

    logging.info("Training complete.")


if __name__ == "__main__":
        main()