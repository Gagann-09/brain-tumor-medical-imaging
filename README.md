# ARMT-GAN Platform

An advanced AI platform leveraging generative adversarial networks for intelligent insights. The platform features a high-performance web-only architecture designed for scale, resilience, and rapid iteration.

## Project Overview

ARMT-GAN Platform is a modern Web application with a robust backend and an AI pipeline for serving models efficiently. It is built to operate securely in a containerized, orchestrated environment.

## High-Level Architecture

The architecture is entirely **Web-based**. It follows a microservices-inspired monolithic core augmented with dedicated AI processing nodes. 
- **Frontend**: Next.js application for an interactive and dynamic user experience.
- **Backend**: FastAPI providing high-performance RESTful APIs.
- **Data & Storage**: PostgreSQL for relational data, Redis for caching, and Object Storage for artifacts.
- **Infrastructure**: Containerized with Docker and orchestrated via Kubernetes.
- **AI Pipeline**: Dedicated inference services serving the GAN models.

## Technology Stack

- **Frontend**: Next.js, React
- **Backend**: FastAPI, Python
- **Database**: PostgreSQL
- **Cache**: Redis
- **Storage**: Object Storage (S3-compatible)
- **Deployment**: Kubernetes, Docker
- **CI/CD**: GitHub Actions
- **AI**: Specialized AI Pipeline

## Repository Structure

```
arm-gan-platform/
├── backend/            # FastAPI backend service
├── frontend/           # Next.js web application
├── infrastructure/     # Terraform, Kubernetes, Helm configs
├── models/             # AI models and pipelines
├── datasets/           # Training and testing datasets
├── docs/               # System architecture and verification docs
├── scripts/            # Build and utility scripts
├── security/           # Security configurations
└── monitoring/         # Observability setup
```

## Development Setup

1. **Prerequisites**: Ensure you have Docker, Docker Compose, Node.js, and Python 3.10+ installed.
2. **Backend Setup**:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   uvicorn main:app --reload
   ```
3. **Frontend Setup**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

## Deployment Overview

Deployment is fully automated using GitHub Actions. 
Code pushed to the `main` branch triggers:
1. Docker image builds for the Next.js frontend, FastAPI backend, and AI workers.
2. Deployment to the Kubernetes cluster via Helm charts.

## Current Implementation Status

- **Web Refactoring**: Complete. The architecture has been refactored to focus purely on the Web interface.
- **Scaffolding**: The project structure has been generated.
- **Documentation**: Baseline architecture and roadmap established.

## Planned Roadmap

See `docs/phases/roadmap.md` for a detailed breakdown of upcoming milestones.
