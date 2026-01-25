#!/usr/bin/env bash
set -euo pipefail

/app/scripts/ensure_sam_checkpoint.sh

exec uvicorn api.main:app --host 0.0.0.0 --port 8000
