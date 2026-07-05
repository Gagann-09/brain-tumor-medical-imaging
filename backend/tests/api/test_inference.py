
import os
import sys
import json
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from main import app
from schemas.inference import JobState

client = TestClient(app)

def test_inference_endpoints():
    # 1. Submit sync inference
    print("Testing /api/v1/inference submission (sync)...")
    study_metadata = {"patient_id": "PT123", "study_id": "ST123", "segmentation_model": "v1"}
    
    # We need to simulate a multipart file upload
    # For TestClient, we use files argument
    files = {
        'files': ('test_image.nii.gz', b'dummy content', 'application/gzip')
    }
    data = {
        'study_metadata': json.dumps(study_metadata),
        'is_async': 'false' # Run sync
    }
    
    response = client.post("/api/v1/inference", data=data, files=files)
    assert response.status_code == 200, response.text
    job = response.json()
    assert "job_id" in job
    assert job["status"] == "completed"
    assert job["study_metadata"]["patient_id"] == "PT123"
    job_id = job["job_id"]
    
    # 2. Get status
    print(f"Testing /api/v1/inference/{job_id} status...")
    response_status = client.get(f"/api/v1/inference/{job_id}")
    assert response_status.status_code == 200
    job_status = response_status.json()
    assert job_status["status"] == "completed"
    
    # 3. Get artifacts
    print(f"Testing /api/v1/inference/{job_id}/artifacts...")
    response_artifacts = client.get(f"/api/v1/inference/{job_id}/artifacts")
    assert response_artifacts.status_code == 200
    artifacts = response_artifacts.json()
    assert "explainability" in artifacts
    assert "report" in artifacts
    
    # 4. Cancel job
    print(f"Testing /api/v1/inference/{job_id} cancellation...")
    # Should fail or return 400 because job is already completed
    response_cancel = client.delete(f"/api/v1/inference/{job_id}")
    assert response_cancel.status_code == 400
    
    print("FastAPI Inference Integration Tests Passed!")

if __name__ == "__main__":
    test_inference_endpoints()
