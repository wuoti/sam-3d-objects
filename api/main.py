import time
import os
import hmac
import hashlib
import uuid
import threading
from typing import Dict

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, Response

from .models import CreateJobResponse, JobInfo
from .storage import job_input_path, job_output_path, job_dir
from .bg_remove import remove_background
from .worker import run_job

app = FastAPI(title="SAM-3D Objects API", version="0.1.0")

JOBS: Dict[str, JobInfo] = {}
LOCK = threading.Lock()

AUTH_HEADER = "x-sam-auth"
TIMESTAMP_HEADER = "x-sam-timestamp"
AUTH_WINDOW_SEC = int(os.environ.get("SAM_AUTH_WINDOW_SEC", "300"))


def _auth_disabled() -> bool:
    return os.environ.get("SAM_AUTH_DISABLED", "false").lower() in {"1", "true", "yes"}


def _auth_secret() -> str:
    return os.environ.get("SAM_AUTH_SECRET", "")


def _constant_time_eq(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def _sign(secret: str, ts: str, method: str, path: str, body: bytes) -> str:
    msg = b".".join([ts.encode("utf-8"), method.encode("utf-8"), path.encode("utf-8"), body])
    return hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()

def _download_url(job_id: str, artifact: str) -> str:
    return f"/v1/jobs/{job_id}/artifact/{artifact}"

def _set(job_id: str, **kwargs):
    with LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        for k, v in kwargs.items():
            setattr(job, k, v)
        JOBS[job_id] = job

def _run_in_background(job_id: str, image_fs_path: str, seed: int, export_usd: bool, usd_scale_factor: float):
    _set(job_id, status="running", started_at=time.time())
    try:
        result = run_job(
            job_id=job_id,
            image_path=image_fs_path,
            seed=seed,
            export_usd=export_usd,
            usd_scale_factor=usd_scale_factor,
        )

        outputs = {}
        for name in result["files"].keys():
            outputs[name] = _download_url(job_id, name)

        _set(
            job_id,
            status="succeeded",
            finished_at=time.time(),
            outputs=outputs,
            meta={"elapsed_sec": result.get("elapsed_sec")},
        )
    except Exception as e:
        _set(job_id, status="failed", finished_at=time.time(), error=str(e))

@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.middleware("http")
async def require_hmac(request: Request, call_next):
    if _auth_disabled() or request.url.path == "/healthz":
        return await call_next(request)

    secret = _auth_secret()
    if not secret:
        return Response(status_code=500, content="auth secret not configured")

    ts = request.headers.get(TIMESTAMP_HEADER, "")
    sig = request.headers.get(AUTH_HEADER, "")
    if not ts or not sig:
        return Response(status_code=401, content="missing auth headers")

    try:
        ts_int = int(ts)
    except ValueError:
        return Response(status_code=401, content="invalid timestamp")

    now = int(time.time())
    if abs(now - ts_int) > AUTH_WINDOW_SEC:
        return Response(status_code=401, content="timestamp outside allowed window")

    body = await request.body()
    expected = _sign(secret, ts, request.method.upper(), request.url.path, body)
    if not _constant_time_eq(expected, sig):
        return Response(status_code=401, content="invalid signature")

    return await call_next(request)

@app.post("/v1/jobs", response_model=CreateJobResponse)
async def create_job(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    seed: int = Form(42),
    export_usd: bool = Form(True),
    usd_scale_factor: float = Form(100.0),
):
    job_id = uuid.uuid4().hex
    job_dir(job_id)  # ensure dirs exist

    job = JobInfo(job_id=job_id, status="queued", created_at=time.time(), outputs={}, meta={})
    with LOCK:
        JOBS[job_id] = job

    # Save input image
    filename = image.filename or "input.png"
    img_path = job_input_path(job_id, filename)
    img_path.write_bytes(await image.read())

    background_tasks.add_task(_run_in_background, job_id, str(img_path), seed, export_usd, usd_scale_factor)
    return CreateJobResponse(job_id=job_id)

@app.get("/v1/jobs/{job_id}", response_model=JobInfo)
def get_job(job_id: str):
    with LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job

@app.get("/v1/jobs/{job_id}/artifact/{name}")
def get_artifact(job_id: str, name: str):
    # Only allow known names to avoid path traversal
    allowed = {"splat.ply", "reconstruction.usd", "reconstruction.usdz", "reconstruction.glb", "outputs.zip"}
    if name not in allowed:
        raise HTTPException(status_code=404, detail="unknown artifact")

    p = job_output_path(job_id, name)
    if not p.exists():
        raise HTTPException(status_code=404, detail="artifact not found")

    return FileResponse(str(p), filename=p.name)


@app.post("/v1/remove-bg")
async def remove_bg(
    image: UploadFile = File(...),
    invert: bool = False,
):
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="unsupported image type")

    data = await image.read()
    try:
        output = remove_background(data, invert=invert)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return Response(content=output, media_type="image/png")
