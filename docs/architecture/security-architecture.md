# Security Architecture

## Overview
Security is paramount in the ARMT-GAN Platform. The architecture implements defense-in-depth strategies across the Web stack.

## Identity and Access Management (IAM)
- **Authentication**: JWT-based authentication via the FastAPI backend.
- **Authorization**: Role-Based Access Control (RBAC) enforced on API endpoints.

## Network Security
- **Ingress**: All web traffic must pass through a WAF and Ingress controller enforcing TLS 1.3.
- **Internal Communication**: Microservices communicate over isolated Kubernetes network policies.

## Data Security
- **At Rest**: PostgreSQL databases and Object Storage buckets are encrypted at rest.
- **In Transit**: All external and sensitive internal communications are encrypted.

## Application Security
- Regular scanning of Docker images and Python/Node.js dependencies in the CI/CD pipeline.
- Input validation utilizing Pydantic on the backend.
