# Local Setup Guide

This guide describes the prerequisites and dataset configurations needed to run the ARMT-GAN Platform locally.

## Prerequisites

To run the local application orchestrator (`manage_local.py`), ensure the following are installed:
- **Python 3.11+**
- **Node.js (v18+)**
- **Docker & Docker Compose**

## Environment Variables

The application relies on default environment variables in `.env` (backend) or `.env.local` (frontend). 
You may override standard configuration via these files. The orchestrator automatically picks up the configuration defaults:
- `BACKEND_PORT`: Default `8000`
- `FRONTEND_PORT`: Default `3000`

## Datasets

Datasets are stored inside the repository's `datasets/` directory. Place your data in the following locations:

```
datasets/
├── brats2020_dev/          # BraTS 2020 Development Subset (segmentation)
├── brats2020_full/         # BraTS 2020 Full Dataset (optional)
└── kaggle_brain_mri/       # Kaggle Brain Tumor MRI (classification)
```

The system validates these directories during the `validate` step.

**Environment variable overrides** (optional):
- `BRA_TS_DEV_PATH` — Override path for the BraTS dev dataset
- `BRA_TS_FULL_PATH` — Override path for the BraTS full dataset
- `KAGGLE_DATASET_PATH` — Override path for the Kaggle dataset

If these are not set, the canonical `datasets/` directory inside the repository is used automatically.

Run `python manage_local.py validate` or `python manage_local.py doctor` to verify that the environment and datasets are correctly configured before attempting to start the system.
