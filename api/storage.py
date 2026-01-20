import os
from pathlib import Path

BASE_DIR = Path(os.environ.get("SAM3D_JOBS_DIR", "/tmp/sam3d-jobs")).resolve()

def job_dir(job_id: str) -> Path:
    d = BASE_DIR / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d

def job_input_dir(job_id: str) -> Path:
    d = job_dir(job_id) / "in"
    d.mkdir(parents=True, exist_ok=True)
    return d

def job_output_dir(job_id: str) -> Path:
    d = job_dir(job_id) / "out"
    d.mkdir(parents=True, exist_ok=True)
    return d

def job_input_path(job_id: str, filename: str) -> Path:
    return job_input_dir(job_id) / filename

def job_output_path(job_id: str, filename: str) -> Path:
    return job_output_dir(job_id) / filename