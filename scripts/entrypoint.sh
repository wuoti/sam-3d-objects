#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

/app/scripts/ensure_sam_checkpoint.sh
/app/scripts/ensure_sam3d_checkpoints.sh

exec uvicorn api.main:app --host 0.0.0.0 --port 8000
