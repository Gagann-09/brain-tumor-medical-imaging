# Startup Guide

The ARMT-GAN Platform comes with a custom orchestrator to simplify the developer experience locally. The orchestrator spawns native processes for the backend and frontend while relying on Docker Compose for infrastructure.

## Usage

### 1. Pre-flight Validation
```bash
python manage_local.py validate
# or
python manage_local.py doctor
```
Checks for Python, Node, Docker, and runs the dataset validation script. It does not start any services.

> **Datasets:** Ensure your datasets are placed inside the `datasets/` directory at the project root before running validation. See [LOCAL_SETUP.md](LOCAL_SETUP.md) for the expected layout.

### 2. Start Services
```bash
python manage_local.py start
```
1. Performs environment and dataset validation.
2. Starts PostgreSQL and Redis via Docker Compose.
3. Spawns Backend (`uvicorn`), Frontend (`next dev`), Celery Worker, and Celery Beat as background processes.
4. Creates a `logs/` directory and streams individual component outputs to dedicated log files (`backend.log`, `frontend.log`, etc.).
5. Waits for health endpoints (`/health/live` and Next.js root) to return 200 OK.
6. Automatically opens the application in your default web browser.

### 3. Check Status
```bash
python manage_local.py status
```
Reads internal state and verifies process existence (`PID` tracking) as well as actual HTTP/Container health statuses. Reports `Healthy`, `Unhealthy`, or `Unknown` for all subsystems.

### 4. Stop Services
```bash
python manage_local.py stop
```
Gracefully terminates child processes via `SIGTERM` (or `taskkill` on Windows), ensuring no orphan processes remain. Additionally, stops the Docker Compose infrastructure services.
