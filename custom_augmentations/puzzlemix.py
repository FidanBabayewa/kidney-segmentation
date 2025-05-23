import os
import nibabel as nib
import numpy as np
import torch
from puzzlemix_augmentation import PuzzleMixAugmentation  # Make sure this matches your path

# Directories
input_dir = "/home/fidan/.local/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Dataset221_KiTS2023_test/imagesTr"
output_dir = "/home/fidan/augmented_outputs"
os.makedirs(output_dir, exist_ok=True)

# Initialize PuzzleMix
augmentor = PuzzleMixAugmentation(alpha=0.3, beta=0.3, patch_size=32)

# Process first 5 cases
for i in range(5):
    filename = f"case_0000{i}_0000.nii.gz"
    input_path = os.path.join(input_dir, filename)

    # Load image
    img_nii = nib.load(input_path)
    img_np = img_nii.get_fdata().astype(np.float32)

    # Fake label (for testing), same shape as image but without channel
    # In real training, use real segmentation masks!
    label_np = np.copy(img_np)
    label_np[label_np > 0] = 1  # simulate binary mask

    # Convert to torch tensors
    img_tensor = torch.from_numpy(img_np).unsqueeze(0)  # shape: (C, H, W, D)
    label_tensor = torch.from_numpy(label_np)

    # Apply PuzzleMix
    aug_img, aug_label = augmentor(img_tensor.clone(), label_tensor.clone())

    # Convert back to NumPy
    aug_img_np = aug_img.squeeze().numpy()
    aug_label_np = aug_label.numpy()

    # Save augmented image and mask
    nib.save(nib.Nifti1Image(aug_img_np, img_nii.affine), os.path.join(output_dir, f"aug_img_{i}.nii.gz"))
    nib.save(nib.Nifti1Image(aug_label_np, img_nii.affine), os.path.join(output_dir, f"aug_label_{i}.nii.gz"))

    print(f"✅ Augmented {filename} saved.")

print("✨ All 5 images processed with PuzzleMix!")
