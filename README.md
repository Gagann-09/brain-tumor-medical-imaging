# ARMT-GAN
Robust Hybrid CNN + Adversarial Medical Training for Brain Tumor Segmentation and Classification

ARMT-GAN Platform is an end-to-end medical imaging application that combines deep learning–based brain tumor segmentation with explainable AI to assist radiologists and clinicians in diagnostic workflows. The system ingests MRI scans, runs multi-stage inference pipelines, and delivers classification, segmentation masks, and interpretability overlays.

Goals are to provide robust, reproducible, and verifiable deep learning pipelines for clinical decision support.

Supported capabilities include end-to-end training pipelines, interactive dataset exploration, medical image preprocessing, and a full-stack dashboard for clinical review.

---

## Features

- Production-ready architecture
- Hybrid CNN
- ARMT-GAN training
- Brain MRI segmentation
- Brain tumor classification
- Medical image preprocessing
- Experiment tracking
- Evaluation pipeline
- Model registry
- Dataset registry
- Local orchestration
- Docker infrastructure

---

## Repository Structure

```text
arm-gan-platform/
├── backend/                    # FastAPI backend service & AI models
├── frontend/                   # Next.js web application
├── datasets/                   # Canonical dataset locations (git-ignored)
├── docs/                       # Architecture, ADRs, RFCs, roadmap
├── diagnostics/                # System health & diagnostic logs
├── tests/                      # Unit, integration, API tests
├── .github/                    # CI/CD workflows, issue & PR templates
└── manage_local.py             # Local orchestrator script
```

---

## Dataset Layout

The datasets must be placed in the `datasets/` root folder. This is the canonical dataset location.

```text
datasets/
├── brats2020_dev/              # BraTS 2020 Development Subset
├── brats2020_full/             # BraTS 2020 Full Dataset (optional)
└── kaggle_brain_mri/           # Kaggle Brain Tumor MRI Dataset
```

These canonical locations are the single source of truth for all training, validation, and testing tasks. Environment variables (`BRA_TS_DEV_PATH`, `BRA_TS_FULL_PATH`, `KAGGLE_DATASET_PATH`) may override these defaults if intentionally configured, but it is highly recommended to use the standard repo-local paths.

---

## Installation

1. **Clone repository**
```bash
git clone https://github.com/Gagann-09/brain-tumor-medical-imaging.git
cd brain-tumor-medical-imaging
```

2. **Prerequisites**
- Python version: ≥ 3.11
- Node version: ≥ 18
- Docker requirement: Docker & Docker Compose must be installed and running.

3. **Install backend dependencies** (optional for local non-Docker development)
```bash
cd backend
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install uv
uv sync
```

4. **Install frontend dependencies** (optional for local non-Docker development)
```bash
cd frontend
npm install
```

---

## Local Development

The project includes a central orchestrator: `manage_local.py`.

- `python manage_local.py doctor`: Checks environment prerequisites, Python, Node, and Docker availability without starting services.
- `python manage_local.py validate`: Validates dataset placement, verifies fingerprint matching, counts cases, and generates a structured validation report.
- `python manage_local.py start`: Starts all infrastructure (PostgreSQL, Redis) and application services (Backend, Frontend, Celery Workers) via Docker Compose.
- `python manage_local.py status`: Checks and reports the health status of all currently running services.
- `python manage_local.py stop`: Cleanly shuts down all services and removes orphan processes.

---

## Dataset Validation

Running `python manage_local.py validate` checks the integrity of the configured datasets against the expected structure and profiles. 

The expected output is a tabular summary indicating:
- **Resolved Path:** Exact location where the dataset was found.
- **Exists:** Confirmation that the dataset directory is accessible.
- **Case counting:** Total number of patient cases detected.
- **Image counting:** Total number of individual medical images.
- **Class counting:** Distribution of labels detected.
- **Validation Status:** Pass or Fail for each registered dataset.

---

## Training

The platform supports multiple training regimes. **Training uses the `datasets/` repository root directory by default.**

- **Segmentation training**: Run `python -m ai.pipeline.train_segmentation` to train standalone segmentation models.
- **Classification training**: Run `python -m ai.pipeline.train_classification` to train the classification networks.
- **ARMT-GAN**: Run `python -m ai.pipeline.train_armt_gan` for robust adversarial training.
- **U-Net baseline**: Run `python -m scripts.train_unet_baseline` to train or reproduce the baseline model.
- **Experiment profiles**: Training behaviors, batch sizes, and epochs are managed via configurable dataset and execution profiles (e.g., `development`, `research`).

---

## Inference

The inference pipeline accepts uploaded medical imaging files. Supported uploads include:
- `.nii`
- `.nii.gz`
- `.png`
- `.jpg`
- `.jpeg`

*Note: Uploaded files used for inference are stored separately (or ephemerally) and are **NOT** added to the training datasets.*

---

## Configuration

Important environment variables configured in `.env`:
- `DATASET_ROOT`: Root folder for datasets (default is the repository-local `datasets/` folder).
- `BRA_TS_DEV_PATH`, `BRA_TS_FULL_PATH`, `KAGGLE_DATASET_PATH`: Overrides for specific dataset locations. Defaults point to the repository-local `datasets/` folder.
- `DATABASE_URL`: PostgreSQL connection string.
- `REDIS_URL`: Redis connection string.

The defaults enforce the repository-local dataset architecture, ensuring that all components agree on data placement without complex configuration.

---

## Docker Services

The `manage_local.py start` command orchestrates the following containers:
- **PostgreSQL**: Relational database for metadata and users.
- **Redis**: In-memory message broker and cache.
- **Backend**: FastAPI service exposing REST APIs and orchestrating AI inference.
- **Frontend**: Next.js clinical dashboard.
- **Celery Worker**: Background task executor for long-running inference and training jobs.
- **Celery Beat**: Scheduler for periodic cleanup and background tasks.

Health checks are built-in and actively monitored during startup.

---

## Verification Workflow

To verify a fresh clone of the repository:
1. `python manage_local.py doctor`
2. `python manage_local.py validate`
3. `python manage_local.py start`
4. `python manage_local.py status`
5. `python manage_local.py stop`

---

## Development Workflow

1. **Dataset validation**: Ensure new data passes strict validator requirements.
2. **Training**: Launch model runs via configured pipeline scripts.
3. **Evaluation**: Generate evaluation metrics across test splits.
4. **Benchmarking**: Compare results against existing baselines.
5. **Experiment tracking**: Track iterations, loss curves, and artifact logging.

---

## Project Documentation

For deeper dives into specific components, refer to:
- [LOCAL_SETUP.md](LOCAL_SETUP.md)
- [STARTUP_GUIDE.md](STARTUP_GUIDE.md)
- [datasets/README.md](datasets/README.md)
- [docs/](docs/)

---

## Troubleshooting

- **Datasets not found**: Ensure datasets are placed strictly in the `datasets/` folder and that you have not misconfigured override environment variables.
- **Docker not running**: Ensure Docker Desktop or the Docker daemon is active before running `manage_local.py`.
- **Redis / PostgreSQL**: Check `manage_local.py status` or Docker container logs if these fail to bind. Verify port conflicts.
- **Frontend / Backend**: If services show as unhealthy, verify `.env` configurations and ensure the correct Node/Python versions were used if running bare-metal.
- **Validation failures**: Ensure your dataset directories strictly match the expected structures documented in `datasets/README.md`.

---

## License

This project is licensed under the [MIT License](LICENSE).

Copyright © 2026 ARMT-GAN Platform Contributors.
