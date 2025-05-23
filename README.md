# Kidney Tumor Segmentation with nnU-Net and PuzzleMix Augmentation

This repository contains code developed as part of a master's thesis titled:  
**"Segmentation of Kidney Tumors using nnU-Net with PuzzleMix Augmentation and HD95 Evaluation"**.

## 🔍 Overview

This project builds upon the [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) framework to perform medical image segmentation on the **KiTS23** dataset.  
Key contributions:
- **Integration of PuzzleMix**: a novel augmentation method for 3D medical images.
- **HD95 Evaluation Metric**: Added support for Hausdorff Distance 95 (HD95), which is not included by default in nnU-Net.

---

## ⚙️ Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.10
- nnU-Net v2 installed and configured
- nibabel, numpy, medpy, tqdm

Install dependencies:
```bash
pip install -r requirements.txt
📁 Project Structure
bash
Copy code
.
├── puzzlemix/
│   └── puzzlemix.py              # Custom PuzzleMix augmentation
├── metrics/
│   └── hd95_eval.py              # HD95 metric evaluation script
├── scripts/
│   └── run_training.sh           # nnU-Net training command
├── README.md
└── requirements.txt
🛠️ Instructions
1. Download KiTS23 Dataset
Download the official KiTS23 dataset and save it to your local machine.

2. Run Data Conversion
Use nnU-Net's data conversion tool to convert the raw KiTS23 files to the nnUNet format:

bash
Copy code
nnUNetv2_convert_KiTS23
Or manually structure the dataset into:

bash
Copy code
nnUNet_raw_data_base/nnUNet_raw_data/Dataset220_KiTS2023/
├── imagesTr/
├── labelsTr/
3. Define nnU-Net Paths
Set environment variables:

bash
Copy code
export nnUNet_raw_data_base="/your/path/to/raw"
export nnUNet_preprocessed="/your/path/to/preprocessed"
export nnUNet_results="/your/path/to/results"
4. Run PuzzleMix Augmentation
Run puzzlemix.py to apply PuzzleMix on a few selected KiTS23 cases:

bash
Copy code
python puzzlemix/puzzlemix.py
The output augmented images will be saved in the defined output folder.

5. Add Augmented Images to Validation Set
Take the augmented .nii.gz files and copy them to:

bash
Copy code
nnUNet_raw_data_base/nnUNet_raw_data/Dataset220_KiTS2023/imagesTr/
nnUNet_raw_data_base/nnUNet_raw_data/Dataset220_KiTS2023/labelsTr/
Ensure the filenames follow nnU-Net naming conventions.

6. Preprocess and Train
bash
Copy code
nnUNetv2_plan_and_preprocess -d 220 -np 1
nnUNetv2_train 220 2d 0 -tr nnUNetTrainer_100epochs -p nnUNetPlans
7. Run Validation Only (Optional)
After training is complete:

bash
Copy code
nnUNetv2_train 220 2d 0 -tr nnUNetTrainer_100epochs -p nnUNetPlans --val
8. HD95 Evaluation
bash
Copy code
python metrics/hd95_eval.py
This will compute and print Mean, Median, and Standard Deviation of HD95 for validation results.

📌 Notes
Ensure you’ve run all required nnU-Net preprocessing steps before training.

PuzzleMix and HD95 are not part of nnU-Net by default — this repository provides extensions for them.

This code was developed under GPU memory and storage constraints — you may need to adjust batch_size, patch_size, or augment fewer samples if using limited hardware.

📚 Citation
If you use this code or approach, please consider citing the original nnU-Net paper:

Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021).
nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods, 18(2), 203–211.
