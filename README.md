# ARMT-GAN Platform

> Medical imaging platform for brain tumor detection, segmentation, and analysis using Attention-Residual Multi-scale Transformer GAN (ARMT-GAN) and Hybrid CNN architectures. Provides XAI-driven diagnostic insights for clinical decision support.

---

## Project Overview

ARMT-GAN Platform is an end-to-end medical imaging application that combines deep learning–based brain tumor segmentation with explainable AI to assist radiologists and clinicians in diagnostic workflows. The system ingests MRI scans (DICOM/NIfTI), runs multi-stage inference pipelines, and delivers classification, segmentation masks, and interpretability overlays through a modern web interface.

### Key Capabilities

- **Tumor Segmentation** — U-Net and ARMT-GAN models for pixel-level delineation of tumor regions.
- **Tumor Classification** — Hybrid CNN architectures for multi-class brain tumor grading.
- **Explainable AI (XAI)** — Grad-CAM, saliency maps, and SHAP-based explanations for model predictions.
- **Multi-stage Inference Pipeline** — Orchestrated pipeline with validation, preprocessing, segmentation, classification, explainability, and report generation stages.
- **Clinical Audit Trail** — Full provenance tracking from upload through diagnosis.

---

## Features

| Category | Capabilities |
|---|---|
| **Medical Imaging** | DICOM/NIfTI parsing, validation, format conversion, metadata extraction |
| **AI Models** | ARMT-GAN, U-Net segmentation, Hybrid CNN, Baseline CNN classification |
| **Explainability** | Grad-CAM, saliency maps, SHAP adapter, overlay generation, XAI benchmarks |
| **Inference** | Multi-stage pipeline with policy-based orchestration and manifest tracking |
| **Training** | Configurable training engine with callbacks, hardware abstraction, and experiment tracking |
| **Data Management** | Dataset registry, BraTS/Kaggle adapters, split management, data versioning |
| **API** | RESTful API with auth, RBAC, audit logging, rate limiting, idempotency |
| **Frontend** | Next.js dashboard with MRI viewer, upload flow, results display, monitoring |
| **Observability** | Prometheus metrics, OpenTelemetry tracing, structured logging |
| **Security** | JWT auth, Argon2 hashing, security headers, CORS, upload size limits |

---

## Architecture

The platform follows a **layered monolithic architecture** with dedicated AI processing components, enforced by `import-linter`:

```
┌──────────────────────────────────────────────────┐
│                  Frontend (Next.js)               │
│          Dashboard · MRI Viewer · Upload          │
├──────────────────────────────────────────────────┤
│                 API Gateway (FastAPI)              │
│     Auth · Upload · Prediction · Segmentation     │
│            XAI · Audit · Admin · Health           │
├──────────────────────────────────────────────────┤
│              Services & Business Logic            │
│    Inference Pipeline · Report Generation         │
├──────────────────────────────────────────────────┤
│                AI / ML Layer                      │
│  ARMT-GAN · U-Net · Hybrid CNN · XAI Engine      │
│  Training Engine · Model Registry · Evaluation    │
├──────────────────────────────────────────────────┤
│             Data & Storage Layer                  │
│   PostgreSQL · Redis · S3-compatible Storage      │
│        DICOM/NIfTI Processing · Alembic           │
└──────────────────────────────────────────────────┘
```

For detailed architecture documentation see [`docs/architecture/`](docs/architecture/).

---

## Repository Structure

```
arm-gan-platform/
├── .github/                    # CI/CD workflows, issue & PR templates
├── backend/                    # FastAPI backend service
│   ├── ai/                     # ML models, training, inference, XAI
│   │   ├── classification/     # CNN classifiers, metrics, transforms
│   │   ├── inference/          # Multi-stage inference pipeline
│   │   ├── segmentation/       # U-Net, ARMT-GAN segmentation
│   │   ├── training/           # Training engine & callbacks
│   │   └── xai/                # Explainability (Grad-CAM, SHAP)
│   ├── alembic/                # Database migrations
│   ├── api/                    # REST API (v1 routes)
│   ├── core/                   # Config, middleware, security
│   ├── medical/                # DICOM/NIfTI processing
│   ├── models/                 # SQLAlchemy ORM models
│   ├── repositories/           # Data access layer
│   ├── schemas/                # Pydantic schemas
│   ├── scripts/                # Developer utility scripts
│   ├── services/               # Business logic services
│   ├── storage/                # Storage abstraction (local/S3)
│   ├── tests/                  # Unit, integration, API tests
│   └── workers/                # Celery background workers
├── datasets/                   # Training data (not tracked — see README)
├── docs/                       # Architecture, ADRs, RFCs, roadmap
│   └── assets/                 # Diagrams, screenshots, illustrations
├── frontend/                   # Next.js web application
│   ├── app/                    # Pages (upload, results, monitor, etc.)
│   ├── components/             # UI components (MRI viewer, sidebar)
│   ├── services/               # API client
│   └── styles/                 # Global CSS & design tokens
└── models/                     # Trained models (not tracked — see README)
```

---

## Technology Stack

### Backend
| Component | Technology |
|---|---|
| Framework | FastAPI ≥ 0.115 |
| Language | Python ≥ 3.12 |
| ORM | SQLAlchemy ≥ 2.0 |
| Migrations | Alembic ≥ 1.14 |
| Task Queue | Celery ≥ 5.6 |
| ML Framework | PyTorch ≥ 2.12, MONAI ≥ 1.6 |
| Medical Imaging | pydicom ≥ 3.0, nibabel ≥ 5.3 |
| Observability | Prometheus, OpenTelemetry, structlog |

### Frontend
| Component | Technology |
|---|---|
| Framework | Next.js 14 |
| Language | TypeScript 5 |
| UI | React 18, Lucide Icons, Recharts |
| Styling | CSS Modules |

### Infrastructure
| Component | Technology |
|---|---|
| Containerization | Docker |
| Orchestration | Kubernetes |
| Database | PostgreSQL |
| Cache | Redis |
| Object Storage | S3-compatible |
| CI/CD | GitHub Actions |

---

## Quick Start

### Prerequisites

- Python ≥ 3.12
- Node.js ≥ 18
- PostgreSQL 15+
- Redis 7+

### Backend

```bash
cd backend
cp .env.example .env          # configure database URL, secrets, storage
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install uv
uv sync                       # install dependencies from uv.lock
alembic upgrade head          # run database migrations
uvicorn main:app --reload     # start dev server at http://localhost:8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev                   # start dev server at http://localhost:3000
```

---

## Dataset Setup

Training and evaluation datasets are **not tracked** in this repository. Place data in the `datasets/` directory following this structure:

```
datasets/
├── BraTS2021/                 # Brain Tumor Segmentation Challenge
│   ├── training/
│   └── validation/
└── kaggle-brain-tumor/        # Kaggle Brain Tumor MRI Dataset
    ├── Training/
    └── Testing/
```

The platform includes built-in dataset adapters for BraTS and Kaggle formats. See `backend/ai/datasets/adapters/` for supported formats. Configure the dataset path in your `.env` file.

---

## Training

```bash
cd backend
source .venv/bin/activate

# Train ARMT-GAN segmentation model
python -m ai.pipeline.train_armt_gan

# Train segmentation model
python -m ai.pipeline.train_segmentation

# Run classification validation
python -m ai.pipeline.run_classification_validation

# Evaluate models
python -m ai.pipeline.evaluate_models
```

Training outputs (checkpoints, metrics, logs) are written to runtime directories that are excluded from version control. Configure experiment tracking via MLflow or Weights & Biases in your `.env` file.

---

## Inference

The platform provides a multi-stage inference pipeline:

```bash
# Run inference pipeline script
python -m scripts.test_inference_pipeline
```

**Pipeline stages**: Validation → Loading → Preprocessing → Segmentation → ROI Extraction → Classification → Explainability → Report Generation → Response

Inference can also be triggered through the REST API (see below).

---

## API

The backend exposes a RESTful API at `http://localhost:8000/api/v1/`:

| Endpoint | Description |
|---|---|
| `GET /health` | System health and readiness probes |
| `POST /api/v1/auth/login` | JWT authentication |
| `POST /api/v1/upload/` | Upload medical images (DICOM/NIfTI) |
| `POST /api/v1/inference/run` | Run inference pipeline |
| `GET /api/v1/prediction/{id}` | Retrieve classification results |
| `GET /api/v1/segmentation/{id}` | Retrieve segmentation masks |
| `POST /api/v1/xai/explain` | Generate XAI visualizations |
| `GET /api/v1/audit/` | Query audit trail |

Interactive API documentation is available at:
- **Swagger UI**: `http://localhost:8000/api/v1/docs`
- **ReDoc**: `http://localhost:8000/api/v1/redoc`

---

## Frontend

The Next.js frontend provides a clinical dashboard interface:

| Page | Description |
|---|---|
| `/` | Dashboard overview with system status |
| `/upload` | Drag-and-drop MRI upload with format validation |
| `/results` | Prediction results with confidence scores |
| `/health` | System health monitoring |
| `/history` | Analysis history and audit trail |
| `/monitor` | Real-time inference monitoring |
| `/settings` | Application configuration |

---

## Deployment

### Docker

```bash
# Build backend image
docker build -t armt-gan-backend ./backend

# Run with environment variables
docker run -p 8000:8000 --env-file backend/.env armt-gan-backend
```

### Kubernetes

Deployment manifests and Helm charts are planned for the `infrastructure/` directory. The CI/CD pipeline (`.github/workflows/`) automates:

1. Linting and testing (backend + frontend)
2. Docker image builds
3. Deployment to the Kubernetes cluster

---

## Roadmap

| Phase | Focus | Status |
|---|---|---|
| **Phase 1** | Core web platform, auth, infrastructure scaffolding | ✅ Complete |
| **Phase 1.5** | AI pipeline integration, inference, XAI, model registry | ✅ Complete |
| **Phase 2** | Enhanced UI, real-time WebSocket updates, interactive MRI viewer | 🔄 In Progress |
| **Phase 3** | Auto-scaling, multi-region deployment, advanced caching | 📋 Planned |

See [`docs/phases/roadmap.md`](docs/phases/roadmap.md) for detailed milestones.

---

## Citation

If you use this platform in your research, please cite:

```bibtex
@software{armt_gan_platform,
  title   = {ARMT-GAN Platform: Medical Imaging for Brain Tumor Analysis},
  author  = {ARMT-GAN Platform Contributors},
  year    = {2026},
  url     = {https://github.com/Gagann-09/brain-tumor-medical-imaging}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).

Copyright © 2026 ARMT-GAN Platform Contributors.
