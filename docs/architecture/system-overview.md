# System Overview

## Purpose
The ARMT-GAN Platform provides an advanced AI-driven ecosystem utilizing Generative Adversarial Networks (GANs). It is built as a **Web-only** platform prioritizing scale, performance, and accessibility through modern web browsers.

## Core Components
1. **Frontend (Web-Only)**
   - Framework: Next.js
   - Purpose: Provides the rich, interactive interface for end-users to interact with the platform and visualize AI outputs.

2. **Backend Services**
   - Framework: FastAPI
   - Purpose: Handles business logic, authentication, and orchestrates requests between the frontend and the AI pipeline.

3. **AI Pipeline**
   - Purpose: Performs heavy lifting for model inference, data processing, and asynchronous tasks.

4. **Data Layer**
   - Relational Data: PostgreSQL
   - Cache/Broker: Redis
   - Unstructured Data: Object Storage

5. **Infrastructure**
   - Orchestration: Kubernetes
   - Containerization: Docker
   - CI/CD: GitHub Actions

## Architecture Principles
- **Web-First**: No native mobile support is included; the platform targets modern web browsers on any device.
- **Scalability**: Components are containerized and deployed via Kubernetes for independent scaling.
- **Separation of Concerns**: Strict boundaries between the web frontend, REST APIs, and asynchronous AI tasks.
