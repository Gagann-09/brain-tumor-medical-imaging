# Backend Architecture

## Overview
The backend is the central nervous system of the ARMT-GAN Platform, built entirely in Python using FastAPI.

## Responsibilities
- Serve RESTful APIs to the Next.js Web frontend.
- Enforce authentication, authorization, and RBAC policies.
- Manage transactional data in PostgreSQL.
- Communicate with Redis for caching and asynchronous task queueing.

## Key Technologies
- **Framework**: FastAPI (async-first API development).
- **ORM**: SQLAlchemy.
- **Migrations**: Alembic.
- **Data Validation**: Pydantic.

## Data Flow
1. The Next.js frontend sends an HTTP request.
2. The API Gateway/Ingress routes it to the FastAPI pod in Kubernetes.
3. FastAPI validates the request using Pydantic.
4. Business logic executes, communicating with PostgreSQL or caching in Redis.
5. If a long-running AI task is requested, it is pushed to a Redis queue for the AI Pipeline to process.
