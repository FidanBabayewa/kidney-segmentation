import os
import nibabel as nib
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from puzzle_mix import SegmentationPuzzleMix
from scipy.ndimage import zoom

input_img_dir = "/home/fidan/.local/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Dataset221_KiTS2023_test/imagesTr"
input_lbl_dir = "/home/fidan/.local/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Dataset221_KiTS2023_test/labelsTr"
output_dir = "/home/fidan/augmented_outputs5"
png_dir = os.path.join(output_dir, "previews")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(png_dir, exist_ok=True)

# List case IDs
case_ids = sorted([f.split("_")[1] for f in os.listdir(input_img_dir) if f.endswith("_0000.nii.gz")])

# Initialize PuzzleMix
augmentor = SegmentationPuzzleMix(alpha=0.5)

# Mix pairs (batch-wise)
for i in range(0, len(case_ids) - 1, 2):
    id1 = case_ids[i]
    id2 = case_ids[i + 1]

    # Load images and labels
    img1 = nib.load(os.path.join(input_img_dir, f"case_{id1}_0000.nii.gz")).get_fdata().astype(np.float32)
    img2 = nib.load(os.path.join(input_img_dir, f"case_{id2}_0000.nii.gz")).get_fdata().astype(np.float32)
    lbl1 = nib.load(os.path.join(input_lbl_dir, f"case_{id1}.nii.gz")).get_fdata().astype(np.float32)
    lbl2 = nib.load(os.path.join(input_lbl_dir, f"case_{id2}.nii.gz")).get_fdata().astype(np.float32)

    # Resize images and labels to the largest image size
    target_shape = np.maximum(img1.shape, img2.shape)
    
    # Resize function to scale images and labels to target shape
    def resize_to_target(arr, target_shape):
        zoom_factors = np.array(target_shape) / np.array(arr.shape)
        return zoom(arr, zoom_factors, order=1)

    img1_resized = resize_to_target(img1, target_shape)
    img2_resized = resize_to_target(img2, target_shape)
    lbl1_resized = resize_to_target(lbl1, target_shape)
    lbl2_resized = resize_to_target(lbl2, target_shape)

    # Convert to torch tensors
    img1_tensor = torch.from_numpy(img1_resized).unsqueeze(0)  # (1, H, W, D)
    img2_tensor = torch.from_numpy(img2_resized).unsqueeze(0)
    lbl1_tensor = torch.from_numpy(lbl1_resized)
    lbl2_tensor = torch.from_numpy(lbl2_resized)

    # Apply PuzzleMix
    aug_img, aug_lbl = augmentor(img1_tensor, lbl1_tensor, img2_tensor, lbl2_tensor)

    # Save augmented volume
    out_id = f"{id1}"
    affine = nib.load(os.path.join(input_img_dir, f"case_{id1}_0000.nii.gz")).affine
    nib.save(nib.Nifti1Image(aug_img.squeeze().numpy(), affine), os.path.join(output_dir, f"aug_case_{out_id}_img.nii.gz"))
    nib.save(nib.Nifti1Image(aug_lbl.numpy(), affine), os.path.join(output_dir, f"aug_case_{out_id}_label.nii.gz"))

    # Save visualization (middle slice)
    mid_slice = aug_img.shape[-1] // 2
    slice_img = aug_img.squeeze().numpy()[:, :, mid_slice]
    slice_mask = aug_lbl.numpy()[:, :, mid_slice]
    overlay = np.ma.masked_where(slice_mask == 0, slice_mask)

    plt.figure(figsize=(6, 6))
    plt.imshow(slice_img, cmap="gray")
    plt.imshow(overlay, cmap="jet", alpha=0.5)
    plt.axis("off")
    plt.title(f"Augmented Slice Case {out_id}")
    plt.savefig(os.path.join(png_dir, f"aug_case_{out_id}.png"), bbox_inches="tight")
    plt.close()

    print(f"✅ Saved augmentation for pair {id1} + {id2} → aug_case_{out_id}_img.nii.gz")

print("✨ All pairs augmented and saved.")
