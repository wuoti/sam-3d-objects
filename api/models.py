from pydantic import BaseModel
from typing import Optional, Dict, Any, Literal

JobStatus = Literal["queued", "running", "succeeded", "failed"]

class CreateJobResponse(BaseModel):
    job_id: str

class JobInfo(BaseModel):
    job_id: str
    status: JobStatus
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    outputs: Dict[str, str] = {}  # artifact filename -> download URL
    meta: Dict[str, Any] = {}