# AI Architecture

## Overview
The AI Architecture is decoupled from the main web API to ensure intensive computational tasks (like GAN inference) do not block web requests.

## Workflow
1. The FastAPI backend enqueues inference requests to a Redis message broker.
2. AI worker nodes consume tasks from the queue.
3. Models perform data processing and generation.
4. Generated artifacts are uploaded to Object Storage.
5. Metadata and statuses are updated in PostgreSQL via the backend.
6. The Next.js frontend is notified (via polling or WebSockets) and renders the artifact from Object Storage.

## Principles
- **Asynchronous Execution**: All AI processing happens out-of-band.
- **Hardware Acceleration**: Worker nodes are scheduled on Kubernetes nodes with GPU support.
- **Scalability**: Workers autoscale based on Redis queue depth.
