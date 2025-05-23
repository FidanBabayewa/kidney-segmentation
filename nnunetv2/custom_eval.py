import os
import numpy as np
from medpy.metric.binary import hd95
import nibabel as nib
import os
import numpy as np



# Only evaluate cases that were predicted
pred_folder = "/home/fidan/.local/nnunet/nnUNet_results/Dataset220_KiTS2023/nnUNetTrainer_100epochs__nnUNetPlans__2d/fold_0/validation"
gt_folder = "/home/fidan/.local/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Dataset220_KiTS2023/labelsTr"


pred_files = sorted([f for f in os.listdir(pred_folder) if f.endswith(".nii.gz")])
gt_files = sorted([f for f in pred_files])  # Match GT files by name

assert len(gt_files) == len(pred_files), "Mismatch in number of GT and predicted files"

hd95_scores = []

for fname in pred_files:
    pred_path = os.path.join(pred_folder, fname)
    gt_path = os.path.join(gt_folder, fname)

    pred_nii = nib.load(pred_path)
    gt_nii = nib.load(gt_path)

    pred = pred_nii.get_fdata()
    gt = gt_nii.get_fdata()

    # Binary masks: assume label 1 is tumor/kidney, skip background
    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)

    if np.sum(pred_bin) == 0 or np.sum(gt_bin) == 0:
        print(f"Skipping {fname} (empty mask)")
        continue

    try:
        score = hd95(pred_bin, gt_bin)
        hd95_scores.append(score)
        print(f"{fname} - HD95: {score:.2f}")
    except Exception as e:
        print(f"Error computing HD95 for {fname}: {e}")


if hd95_scores:  
    mean_hd95 = np.mean(hd95_scores)
    median_hd95 = np.median(hd95_scores)
    std_hd95 = np.std(hd95_scores)

    print(f"✅ Mean HD95: {mean_hd95:.2f}")  
    print(f"✅ Median HD95: {median_hd95:.2f}")  
    print(f"✅ Standard Deviation of HD95: {std_hd95:.2f}")  
else:  
    print("⚠️ No HD95 scores computed.")
