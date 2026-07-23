# Datasets

This directory contains all training and evaluation datasets. Its contents are **git-ignored** to avoid committing large data files.

## Expected Layout

```
datasets/
├── brats2020_dev/          # BraTS 2020 Development Subset (segmentation)
│   ├── BraTS20_Training_001/
│   ├── BraTS20_Training_002/
│   └── ...
├── brats2020_full/         # BraTS 2020 Full Dataset (optional)
│   └── ...
└── kaggle_brain_mri/       # Kaggle Brain Tumor MRI Dataset (classification)
    ├── yes/
    ├── no/
    └── ...
```

## Validation

Run `python manage_local.py validate` from the project root to verify datasets are correctly placed.
