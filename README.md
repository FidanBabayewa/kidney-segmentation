# Kidney Tumor Segmentation with nnU-Net and PuzzleMix Augmentation

This repository contains code developed as part of a master's thesis titled:

**"Segmentation of Kidney Tumors using nnU-Net with PuzzleMix Augmentation and HD95 Evaluation"**

---

## Overview

This project builds upon the nnU-Net framework to perform medical image segmentation on the **KiTS23 dataset**.

### Key Contributions:
-  **PuzzleMix Augmentation** for enhanced 3D medical image generalization
-  **HD95 Evaluation Metric** for better boundary-focused performance measurement

---

##  Requirements

- Python ≥ 3.8  
- PyTorch ≥ 1.10  
- nnU-Net v2 (installed and configured)  
- `nibabel`, `numpy`, `medpy`, `tqdm`  

Install all dependencies:
```bash
pip install -r requirements.txt
📁 Project Structure
graphql
Copy
Edit
├── puzzlemix/
│   ├── puzzlemix.py                  # Original PuzzleMix implementation
│   ├── puzzle_mix.py                 # Alternative version
│   ├── puzzlemix_augmentation.py    # Variant with integration hooks
│   └── test_puzzle_mix.py           # Test script to generate augmented data
├── metrics/
│   └── hd95_eval.py                  # HD95 metric evaluation script
├── scripts/
│   └── run_training.sh               # nnU-Net training shell script
├── README.md
├── requirements.txt
🛠️ Instructions
1. Download the KiTS23 Dataset
Download from the official source and place it on your machine.

2. Convert Data to nnU-Net Format
Use nnU-Net's conversion tool:


nnUNetv2_convert_KiTS23
Or manually structure the dataset:


nnUNet_raw_data_base/nnUNet_raw_data/Dataset220_KiTS2023/
├── imagesTr/
├── labelsTr/
3. Set Environment Variables
bash
Copy
Edit
export nnUNet_raw_data_base="/your/path/to/raw"
export nnUNet_preprocessed="/your/path/to/preprocessed"
export nnUNet_results="/your/path/to/results"
4. Run PuzzleMix Augmentation
 Test PuzzleMix:

python puzzlemix/test_puzzle_mix.py
This script loads sample data and applies one of the PuzzleMix algorithms. Augmented outputs are saved in your target folder.

You can experiment with puzzlemix.py, puzzle_mix.py, and puzzlemix_augmentation.py for different behaviors.

5. Add Augmented Images to Training Set
Move augmented .nii.gz files into:


Dataset220_KiTS2023/imagesTr/
Dataset220_KiTS2023/labelsTr/
Follow nnU-Net naming conventions for file names.

6. Preprocess and Train

nnUNetv2_plan_and_preprocess -d 220 -np 1
nnUNetv2_train 220 2d 0 -tr nnUNetTrainer_100epochs -p nnUNetPlans
7. Run Validation Only (Optional)

nnUNetv2_train 220 2d 0 -tr nnUNetTrainer_100epochs -p nnUNetPlans --val
8. Run HD95 Evaluation

python metrics/hd95_eval.py
This script computes:

Mean HD95

Median HD95

Standard Deviation

Notes
Make sure to preprocess with nnUNetv2_plan_and_preprocess before training.

PuzzleMix and HD95 are custom additions.


Author
Fidan Babayeva
MSc Computer Science
University of Szeged
