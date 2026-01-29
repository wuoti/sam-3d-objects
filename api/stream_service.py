import json
import logging
import os
import time
from typing import Dict, Optional, Tuple

import boto3
from botocore.config import Config
import redis

from .bg_remove import remove_background
from .storage import job_input_path
from .worker import run_job

STREAM_NAME = os.environ.get("REDIS_STREAM_NAME", "jobs:stream")
REDIS_URL = os.environ.get("REDIS_URL", "")

S3_BUCKET = os.environ.get("S3_JOB_BUCKET_NAME", "")
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT")
S3_REGION = os.environ.get("S3_REGION")
S3_ACCESS_KEY_ID = os.environ.get("S3_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.environ.get("S3_SECRET_ACCESS_KEY")
S3_FORCE_PATH_STYLE = os.environ.get("S3_FORCE_PATH_STYLE", "false").lower() in {
    "1",
    "true",
    "yes",
}

STREAM_BLOCK_MS = int(os.environ.get("REDIS_STREAM_BLOCK_MS", "5000"))
STREAM_COUNT = int(os.environ.get("REDIS_STREAM_COUNT", "1"))
STREAM_START = os.environ.get("REDIS_STREAM_START", "$")

STREAM_GROUP = os.environ.get("REDIS_STREAM_GROUP")
STREAM_CONSUMER = os.environ.get("REDIS_STREAM_CONSUMER", "worker-1")


def _require_env() -> None:
    missing = []
    if not REDIS_URL:
        missing.append("REDIS_URL")
    if not S3_BUCKET:
        missing.append("S3_JOB_BUCKET_NAME")
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")


def _redis_client() -> redis.Redis:
    return redis.from_url(REDIS_URL, decode_responses=False)


def _s3_client():
    config = Config(s3={"addressing_style": "path" if S3_FORCE_PATH_STYLE else "virtual"})
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        region_name=S3_REGION,
        aws_access_key_id=S3_ACCESS_KEY_ID,
        aws_secret_access_key=S3_SECRET_ACCESS_KEY,
        config=config,
    )


def _decode_fields(fields: Dict[bytes, bytes]) -> Dict[str, str]:
    decoded: Dict[str, str] = {}
    for k, v in fields.items():
        key = k.decode("utf-8") if isinstance(k, (bytes, bytearray)) else str(k)
        val = v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)
        decoded[key] = val
    return decoded


def _extract_job_id(fields: Dict[str, str]) -> Optional[str]:
    return fields.get("jobId") or fields.get("job_id") or fields.get("jobID")


def _set_status(rdb: redis.Redis, job_id: str, status: str) -> None:
    rdb.hset(f"jobs:{job_id}", mapping={"status": status})


def _fetch_metadata(s3, job_id: str) -> Dict[str, object]:
    key = f"{job_id}/metadata.json"
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    data = obj["Body"].read()
    return json.loads(data)


def _download_image(s3, job_id: str, stored_name: str) -> bytes:
    key = f"{job_id}/{stored_name}"
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return obj["Body"].read()


def _upload_glb(s3, job_id: str, glb_path: str) -> None:
    key = f"{job_id}/output.glb"
    s3.upload_file(glb_path, S3_BUCKET, key)


def _process_job(
    rdb: redis.Redis,
    s3,
    job_id: str,
) -> Tuple[bool, Optional[str]]:
    try:
        _set_status(rdb, job_id, "running")

        metadata = _fetch_metadata(s3, job_id)
        stored_name = metadata.get("storedName")
        if not stored_name:
            raise RuntimeError("metadata.json missing storedName")

        image_bytes = _download_image(s3, job_id, stored_name)
        output = remove_background(image_bytes, invert=False)

        processed_path = job_input_path(job_id, "input.png")
        processed_path.write_bytes(output)

        result = run_job(
            job_id=job_id,
            image_path=str(processed_path),
        )

        glb_path = result["files"].get("reconstruction.glb")
        if not glb_path:
            raise RuntimeError("GLB output was not produced")

        _upload_glb(s3, job_id, glb_path)
        _set_status(rdb, job_id, "succeeded")
        return True, None
    except Exception as exc:
        logging.exception("Failed processing job %s", job_id)
        _set_status(rdb, job_id, "failed")
        return False, str(exc)


def _read_stream(
    rdb: redis.Redis,
    last_id: str,
) -> Tuple[Optional[str], Optional[Dict[str, str]]]:
    if STREAM_GROUP:
        try:
            rdb.xgroup_create(STREAM_NAME, STREAM_GROUP, id="0-0", mkstream=True)
        except redis.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise
        response = rdb.xreadgroup(
            STREAM_GROUP,
            STREAM_CONSUMER,
            {STREAM_NAME: ">"},
            count=STREAM_COUNT,
            block=STREAM_BLOCK_MS,
        )
    else:
        response = rdb.xread({STREAM_NAME: last_id}, count=STREAM_COUNT, block=STREAM_BLOCK_MS)

    if not response:
        return last_id, None

    _, entries = response[0]
    if not entries:
        return last_id, None

    entry_id, fields = entries[0]
    decoded = _decode_fields(fields)
    return entry_id, decoded


def run_forever() -> None:
    _require_env()
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    rdb = _redis_client()
    s3 = _s3_client()

    last_id = STREAM_START
    logging.info("Listening on redis stream %s", STREAM_NAME)
    while True:
        entry_id, fields = _read_stream(rdb, last_id)
        if not fields:
            continue

        job_id = _extract_job_id(fields)
        if not job_id:
            logging.warning("Stream entry %s missing jobId field", entry_id)
            if STREAM_GROUP:
                rdb.xack(STREAM_NAME, STREAM_GROUP, entry_id)
            last_id = entry_id
            continue

        _process_job(rdb, s3, job_id)

        if STREAM_GROUP:
            rdb.xack(STREAM_NAME, STREAM_GROUP, entry_id)
        last_id = entry_id
        time.sleep(0.1)


if __name__ == "__main__":
    run_forever()
