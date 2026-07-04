# Deployment Architecture

## Overview
The ARMT-GAN Platform relies on Kubernetes and Docker for reliable, scalable deployment. The focus is exclusively on the Web architecture; no mobile app distribution systems are utilized.

## Pipeline
1. **Source Control**: GitHub repository containing all source code and infrastructure configurations.
2. **CI/CD Integration**: GitHub Actions pipelines are triggered on push/merge to specific branches.
3. **Build Phase**: Docker images are built for the Next.js frontend, FastAPI backend, and AI workers.
4. **Registry**: Images are pushed to a container registry.
5. **Deployment Phase**: Helm charts are applied to the Kubernetes cluster, pulling the latest images and rolling out updates seamlessly.

## Environments
- **Development**: Local environment utilizing Docker Compose.
- **Staging**: A Kubernetes cluster mimicking production for final validation.
- **Production**: A highly available Kubernetes cluster handling live web traffic.
