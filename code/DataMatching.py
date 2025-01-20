import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

##################################
# Example File Paths (Adjust These)
##################################
nfbs_file = "sub-A00033747_ses-NFB3_T1w_brain.nii"
brats_file = "BraTS20_Training_001_t1.nii"

##################################
# 1. Load NIfTI volumes
##################################
nfbs_img = nib.load(nfbs_file)
nfbs_data = nfbs_img.get_fdata()  # shape e.g. (X, Y, Z)
print(f"NFBS volume shape: {nfbs_data.shape}")

brats_img = nib.load(brats_file)
brats_data = brats_img.get_fdata()  # shape e.g. (X, Y, Z)
print(f"BraTS volume shape: {brats_data.shape}")

##################################
# 2. Slicing
# NFBS: horizontally along the dimension (Y), rotate counterclockwise
# BraTS: along the last dimension (Z)
##################################
nfbs_slices = [np.rot90(nfbs_data[:, i, :], k=1) for i in range(nfbs_data.shape[1])]
brats_slices = [brats_data[:, :, i] for i in range(brats_data.shape[2])]

print(f"NFBS: Extracted {len(nfbs_slices)} Y-axis slices (rotated conterclockwise).")
print(f"BraTS: Extracted {len(brats_slices)} Z-axis slices.")
'''
nfbs_vol = np.stack(nfbs_slices, axis=0)
brats_vol = np.stack(brats_slices, axis=0)

nfbs_shape = nfbs_vol.shape
brats_shape = brats_vol.shape

print(f"NFBS volume shape: {nfbs_shape}")
print(f"BraTS volume shape: {brats_shape}")

##################################
# 2.5. Symmetrical Cropping to Unified Size (192 x 240)
##################################
def crop_to_unified_size(vol, target_height, target_width):
    """
    Symmetrically crops the 2D slices of a 3D volume to the target size (height, width).
    """
    current_height, current_width = vol.shape[1:3]
    
    # Calculate cropping margins
    height_diff = current_height - target_height
    width_diff = current_width - target_width
    
    if height_diff < 0 or width_diff < 0:
        raise ValueError("Target size is larger than current size, cannot crop.")

    top_crop = height_diff // 2
    bottom_crop = height_diff - top_crop
    left_crop = width_diff // 2
    right_crop = width_diff - left_crop
    
    # Crop symmetrically
    cropped_vol = vol[:, top_crop:current_height - bottom_crop, left_crop:current_width - right_crop]
    return cropped_vol

# Target size
target_height = 192
target_width = 240

# Apply symmetrical cropping
nfbs_cropped_ori_vol = crop_to_unified_size(nfbs_vol, target_height, target_width)
brats_cropped_ori_vol = crop_to_unified_size(brats_vol, target_height, target_width)

print(f"NFBS cropped volume shape: {nfbs_cropped_ori_vol.shape}")
print(f"BraTS cropped volume shape: {brats_cropped_ori_vol.shape}")
'''

##################################
# 3. Identify the "Full Brain" Slice
# We'll find the slice with the fewest zero pixels for each dataset.
##################################
def find_full_brain_slice(slices):
    zero_counts = [np.sum(s == 0) for s in slices]
    full_slice = np.argmin(zero_counts)
    return full_slice

nfbs_full_slice = find_full_brain_slice(nfbs_slices)
brats_full_slice = find_full_brain_slice(brats_slices)

print(f"NFBS full brain slice index: {nfbs_full_slice}")
print(f"BraTS full brain slice index: {brats_full_slice}")

# Visualize these "full brain" slices before any cropping or alignment
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(nfbs_slices[nfbs_full_slice], cmap='gray')
ax1.set_title(f"NFBS Full Brain Slice (Index {nfbs_full_slice}) Before Cropping")
ax1.axis('off')

ax2.imshow(brats_slices[brats_full_slice], cmap='gray')
ax2.set_title(f"BraTS Full Brain Slice (Index {brats_full_slice}) Before Cropping")
ax2.axis('off')

plt.tight_layout()
plt.show()

##################################
# 4. Align slices around the "full brain" slice
#
# We want to make sure that when we proceed to bounding box and final visualization,
# we're looking at a comparable range of slices around these "full" slices.
#
# Let's assume we pick a window of slices around the full brain slice from each dataset,
# and then we will proceed. For example, if NFBS full slice is 90 and BraTS is 50,
# we can align them by shifting one dataset so that the "full" slices line up.
#
# We'll define a range around the full slice. For simplicity:
# Let's pick ±50 slices around the full brain slice for each dataset (adjust as needed).
##################################
window = 75
def extract_around_center(slices, center_index, window):
    start = max(center_index - window, 0)
    end = min(center_index + window + 1, len(slices))
    return slices[start:end], start, end

nfbs_aligned_slices, nfbs_start, nfbs_end = extract_around_center(nfbs_slices, nfbs_full_slice, window)
brats_aligned_slices, brats_start, brats_end = extract_around_center(brats_slices, brats_full_slice, window)

print(f"NFBS aligned slice range: [{nfbs_start}, {nfbs_end})")
print(f"BraTS aligned slice range: [{brats_start}, {brats_end})")
print(f"NFBS aligned slices count: {len(nfbs_aligned_slices)}")
print(f"BraTS aligned slices count: {len(brats_aligned_slices)}")

# If they differ in length, you can further trim to the minimum length
min_len = min(len(nfbs_aligned_slices), len(brats_aligned_slices))
nfbs_aligned_slices = nfbs_aligned_slices[:min_len]
brats_aligned_slices = brats_aligned_slices[:min_len]

print(f"After ensuring same count: {len(nfbs_aligned_slices)} slices each")

##################################
# 5. Find bounding box and crop using each dataset's own bounding box
##################################
def find_bounding_box_single_slice(slice_data):
    nonzero = np.argwhere(slice_data > 0)
    if len(nonzero) == 0:
        # If there's no nonzero pixel, return entire slice
        return (0, slice_data.shape[0] - 1, 0, slice_data.shape[1] - 1)
    ymin, xmin = nonzero.min(axis=0)
    ymax, xmax = nonzero.max(axis=0)
    return ymin, ymax, xmin, xmax

# Calculate bounding boxes for the "full brain" slices individually
nfbs_bbox_2d = find_bounding_box_single_slice(nfbs_slices[nfbs_full_slice])
brats_bbox_2d = find_bounding_box_single_slice(brats_slices[brats_full_slice])

print(f"NFBS 2D bounding box (y,x): {nfbs_bbox_2d}")
print(f"BraTS 2D bounding box (y,x): {brats_bbox_2d}")

def crop_slices_2d(slices, ymin, ymax, xmin, xmax):
    """
    Crops all slices to the given bounding box.
    """
    return [s[ymin:ymax + 1, xmin:xmax + 1] for s in slices]

# Crop slices using their own bounding boxes
nfbs_cropped_slices = crop_slices_2d(nfbs_aligned_slices, *nfbs_bbox_2d)
brats_cropped_slices = crop_slices_2d(brats_aligned_slices, *brats_bbox_2d)

nfbs_cropped_vol = np.stack(nfbs_cropped_slices, axis=0)
brats_cropped_vol = np.stack(brats_cropped_slices, axis=0)

print(f"NFBS cropped volume shape: {nfbs_cropped_vol.shape}")
print(f"BraTS cropped volume shape: {brats_cropped_vol.shape}")

##################################
# 6. If shapes differ, pad the smaller one with black pixels
##################################
nfbs_shape = nfbs_cropped_vol.shape
brats_shape = brats_cropped_vol.shape

def pad_to_match(shape_a, vol_b):
    """
    Pads vol_b with zeros on the right and bottom edges to match shape_a.
    shape_a and vol_b.shape are expected to be (Z, Y, X).
    """
    z_a, y_a, x_a = shape_a
    z_b, y_b, x_b = vol_b.shape
    
    # Pad along Z (depth) if needed (unlikely since we aligned slices count)
    z_pad = max(0, z_a - z_b)
    # Pad along Y
    y_pad = max(0, y_a - y_b)
    # Pad along X
    x_pad = max(0, x_a - x_b)
    
    # Padding format for np.pad: ((before, after), (before, after), ...)
    padding = ((0, z_pad), (0, y_pad), (0, x_pad))
    
    return np.pad(vol_b, padding, mode='constant', constant_values=0)

# Determine which volume needs padding
if nfbs_shape != brats_shape:
    # Compare each dimension
    final_z = max(nfbs_shape[0], brats_shape[0])
    final_y = max(nfbs_shape[1], brats_shape[1])
    final_x = max(nfbs_shape[2], brats_shape[2])
    final_shape = (final_z, final_y, final_x)
    
    # Pad NFBS if needed
    if nfbs_cropped_vol.shape != final_shape:
        nfbs_cropped_vol = pad_to_match(final_shape, nfbs_cropped_vol)
    
    # Pad BraTS if needed
    if brats_cropped_vol.shape != final_shape:
        brats_cropped_vol = pad_to_match(final_shape, brats_cropped_vol)

print(f"Final NFBS shape: {nfbs_cropped_vol.shape}")
print(f"Final BraTS shape: {brats_cropped_vol.shape}")

##################################
# 7. Visualization after cropping/padding
##################################
center_index = len(nfbs_cropped_vol)//2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(nfbs_cropped_vol[center_index], cmap='gray')
ax1.set_title("NFBS 'Full Brain' Slice After Cropping/Padding")
ax1.axis('off')

ax2.imshow(brats_cropped_vol[center_index], cmap='gray')
ax2.set_title("BraTS 'Full Brain' Slice After Cropping/Padding")
ax2.axis('off')

plt.tight_layout()
plt.show()
