import os
import sys
import json
import time
import signal
import logging
import argparse
import subprocess
import urllib.request
import urllib.error
import webbrowser
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("orchestrator")

# Constants
ROOT_DIR = Path(__file__).resolve().parent
STATE_FILE = ROOT_DIR / ".manage_local_state.json"
LOGS_DIR = ROOT_DIR / "logs"

# Config Defaults (Can be overridden by .env if we were fully parsing it, but we'll use standard defaults)
BACKEND_PORT = os.getenv("BACKEND_PORT", "8000")
FRONTEND_PORT = os.getenv("FRONTEND_PORT", "3000")

def print_step(step_name):
    print(f"\n--- {step_name} ---")

def run_command(cmd, cwd=None, capture_output=False, check=False):
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or ROOT_DIR,
            shell=True,
            capture_output=capture_output,
            text=True
        )
        elapsed = time.time() - t0
        if check and result.returncode != 0:
            logger.error(f"Failure running '{cmd}' (Elapsed: {elapsed:.2f}s)")
            if capture_output:
                logger.error(f"STDOUT:\n{result.stdout}")
                logger.error(f"STDERR:\n{result.stderr}")
            else:
                logger.error(f"Reason: Command exited with code {result.returncode}")
            sys.exit(1)
        return result, elapsed
    except Exception as e:
        logger.error(f"Exception running '{cmd}': {e}")
        sys.exit(1)

def is_process_alive(pid):
    if sys.platform == "win32":
        try:
            output = subprocess.check_output(f'tasklist /FI "PID eq {pid}" /NH', shell=True, text=True)
            return str(pid) in output
        except subprocess.CalledProcessError:
            return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

def kill_process(pid):
    if not is_process_alive(pid):
        return
    logger.info(f"Terminating process {pid}...")
    if sys.platform == "win32":
        subprocess.run(f"taskkill /F /T /PID {pid}", shell=True, capture_output=True)
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except Exception:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)

def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}

def check_http_health(url):
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=2) as response:
            return response.getcode() == 200
    except Exception:
        return False

def check_docker_health(service_name):
    # Check if a container with service_name in its name is running and healthy
    try:
        output = subprocess.check_output('docker ps --format "{{.Names}} {{.Status}}"', shell=True, text=True)
        for line in output.strip().split("\n"):
            if service_name in line:
                if "(healthy)" in line:
                    return "Healthy"
                elif "Up" in line:
                    return "Up (Health unknown/starting)"
        return "Unhealthy"
    except Exception:
        return "Unknown"

def validate():
    print_step("Validation Pre-flight Checks")
    
    # 1. Python version
    print("Checking Python version...")
    run_command("python --version", check=True)
    print("Success")

    # 2. Node version
    print("Checking Node version...")
    run_command("node --version", check=True)
    print("Success")

    # 3. Docker CLI
    print("Checking Docker availability...")
    run_command("docker --version", check=True)
    print("Success")

    # 4. Docker daemon
    print("Checking Docker daemon...")
    run_command("docker ps", capture_output=True, check=True)
    print("Success")

    # 5. Dataset validation
    print("Running Dataset Validation...")
    res, elapsed = run_command("python backend/scripts/validate_dataset.py", capture_output=True)
    print(res.stdout)
    if res.returncode != 0 or "Fail" in res.stdout or "Warning" in res.stdout:
        logger.error(f"Dataset validation failed or had warnings. (Elapsed: {elapsed:.2f}s)")
        logger.error(f"STDERR:\n{res.stderr}")
        logger.error("Suggested action: Check datasets directory and validate_dataset.py logic.")
        sys.exit(1)
    print("Success")

    # 6. Environment validation
    print("Checking required directories...")
    if not (ROOT_DIR / "backend").exists() or not (ROOT_DIR / "frontend").exists():
        logger.error("Missing backend or frontend directory.")
        sys.exit(1)
    
    print("Success: Pre-flight validation passed.")

def start():
    validate()
    
    print_step("Starting Infrastructure")
    LOGS_DIR.mkdir(exist_ok=True)
    state = load_state()
    state["start_time"] = time.time()
    
    # Docker Compose for db and redis
    res, elapsed = run_command("docker compose up -d db redis", capture_output=True, check=True)
    print(f"Infrastructure started in {elapsed:.2f}s")
    
    print("Waiting for PostgreSQL and Redis to be healthy...")
    db_healthy = False
    redis_healthy = False
    for _ in range(30):
        if not db_healthy and check_docker_health("db") == "Healthy":
            db_healthy = True
        if not redis_healthy and check_docker_health("redis") == "Healthy":
            redis_healthy = True
        if db_healthy and redis_healthy:
            break
        time.sleep(1)
        
    if not db_healthy or not redis_healthy:
        logger.error("Timeout waiting for db/redis to become healthy.")
        sys.exit(1)
    print("Infrastructure is Healthy.")
    
    print_step("Starting Application Services")
    # Backend
    backend_log = open(LOGS_DIR / "backend.log", "w")
    backend_proc = subprocess.Popen(
        f"{sys.executable} -m uvicorn main:app --reload --port {BACKEND_PORT} --host 0.0.0.0",
        cwd=ROOT_DIR / "backend", shell=True, stdout=backend_log, stderr=backend_log
    )
    state["backend_pid"] = backend_proc.pid
    
    # Frontend
    frontend_log = open(LOGS_DIR / "frontend.log", "w")
    frontend_proc = subprocess.Popen(
        "npm run dev",
        cwd=ROOT_DIR / "frontend", shell=True, stdout=frontend_log, stderr=frontend_log
    )
    state["frontend_pid"] = frontend_proc.pid
    
    # Celery Worker
    cworker_log = open(LOGS_DIR / "celery_worker.log", "w")
    cworker_proc = subprocess.Popen(
        f"{sys.executable} -m celery -A workers.celery_app worker --loglevel=info",
        cwd=ROOT_DIR / "backend", shell=True, stdout=cworker_log, stderr=cworker_log
    )
    state["celery_worker_pid"] = cworker_proc.pid
    
    # Celery Beat
    cbeat_log = open(LOGS_DIR / "celery_beat.log", "w")
    cbeat_proc = subprocess.Popen(
        f"{sys.executable} -m celery -A workers.celery_app beat --loglevel=info",
        cwd=ROOT_DIR / "backend", shell=True, stdout=cbeat_log, stderr=cbeat_log
    )
    state["celery_beat_pid"] = cbeat_proc.pid
    
    save_state(state)
    
    print("Waiting for Health Endpoints...")
    backend_url = f"http://localhost:{BACKEND_PORT}/health/live"
    frontend_url = f"http://localhost:{FRONTEND_PORT}/"
    
    b_healthy = False
    f_healthy = False
    
    for _ in range(60):
        if not b_healthy:
            b_healthy = check_http_health(backend_url)
        if not f_healthy:
            f_healthy = check_http_health(frontend_url)
            
        if b_healthy and f_healthy:
            break
        time.sleep(2)
        
    if not b_healthy or not f_healthy:
        logger.error("Timeout waiting for backend/frontend to become healthy.")
        logger.error(f"Backend: {b_healthy}, Frontend: {f_healthy}")
        stop()
        sys.exit(1)
        
    print("Backend and Frontend are Healthy.")
    
    # Open Browser
    print(f"Opening browser at {frontend_url}")
    webbrowser.open(frontend_url)
    print("Start complete.")

def status():
    print_step("System Status")
    state = load_state()
    start_time = state.get("start_time")
    if start_time:
        uptime = time.time() - start_time
        print(f"Uptime: {uptime:.2f} seconds\n")
    else:
        print("Uptime: Unknown (Not started via orchestrator or state missing)\n")
        
    # Infrastructure
    db_health = check_docker_health("db")
    redis_health = check_docker_health("redis")
    print(f"PostgreSQL:   {db_health}")
    print(f"Redis:        {redis_health}")
    
    # Backend
    b_pid = state.get("backend_pid")
    b_alive = is_process_alive(b_pid) if b_pid else False
    b_http = check_http_health(f"http://localhost:{BACKEND_PORT}/health/live")
    b_status = "Healthy" if b_alive and b_http else ("Unhealthy" if b_pid else "Unknown (Stopped)")
    print(f"Backend:      {b_status}")
    
    # Frontend
    f_pid = state.get("frontend_pid")
    f_alive = is_process_alive(f_pid) if f_pid else False
    f_http = check_http_health(f"http://localhost:{FRONTEND_PORT}/")
    f_status = "Healthy" if f_alive and f_http else ("Unhealthy" if f_pid else "Unknown (Stopped)")
    print(f"Frontend:     {f_status}")
    
    # Celery Worker
    cw_pid = state.get("celery_worker_pid")
    cw_alive = is_process_alive(cw_pid) if cw_pid else False
    cw_status = "Healthy" if cw_alive else ("Unhealthy" if cw_pid else "Unknown (Stopped)")
    print(f"Celery Worker:{cw_status}")

    # Celery Beat
    cb_pid = state.get("celery_beat_pid")
    cb_alive = is_process_alive(cb_pid) if cb_pid else False
    cb_status = "Healthy" if cb_alive else ("Unhealthy" if cb_pid else "Unknown (Stopped)")
    print(f"Celery Beat:  {cb_status}")
    
    # Overall
    components = [db_health, redis_health, b_status, f_status, cw_status, cb_status]
    if all(s == "Healthy" for s in components):
        print("\nOverall Status: ALL HEALTHY")
    elif all("Unknown" in s for s in components[2:]) and all(s != "Healthy" for s in components[:2]):
        print("\nOverall Status: STOPPED")
    else:
        print("\nOverall Status: DEGRADED / UNHEALTHY")

def stop():
    print_step("Stopping Application")
    state = load_state()
    
    for proc in ["backend_pid", "frontend_pid", "celery_worker_pid", "celery_beat_pid"]:
        pid = state.get(proc)
        if pid:
            kill_process(pid)
            
    print("Stopping infrastructure (docker compose)...")
    run_command("docker compose stop db redis", capture_output=True)
    
    if STATE_FILE.exists():
        STATE_FILE.unlink()
    print("Stop complete. No orphan processes remain.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local Application Orchestrator")
    parser.add_argument("command", choices=["start", "stop", "status", "validate", "doctor"])
    args = parser.parse_args()
    
    if args.command in ["validate", "doctor"]:
        validate()
    elif args.command == "start":
        start()
    elif args.command == "status":
        status()
    elif args.command == "stop":
        stop()
