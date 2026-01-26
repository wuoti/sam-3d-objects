#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_DIR="${SAM3D_CHECKPOINT_DIR:-/app/checkpoints/hf}"
DOWNLOAD_DIR="${SAM3D_CHECKPOINT_DOWNLOAD_DIR:-/app/checkpoints}"

if [[ -d "${CHECKPOINT_DIR}" ]] && [[ -n "$(ls -A "${CHECKPOINT_DIR}" 2>/dev/null)" ]]; then
  echo "SAM-3D-Objects checkpoints already present: ${CHECKPOINT_DIR}"
  exit 0
fi

if ! command -v git-xet >/dev/null 2>&1; then
  echo "git-xet is not installed" >&2
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set" >&2
  exit 1
fi

echo "Downloading SAM-3D-Objects checkpoints to ${DOWNLOAD_DIR}"
git clone "https://hf_user:${HF_TOKEN}@huggingface.co/facebook/sam-3d-objects" "${DOWNLOAD_DIR}/sam-3d-objects"

if [[ ! -d "${DOWNLOAD_DIR}/sam-3d-objects/checkpoints" ]]; then
  echo "Download did not produce ${DOWNLOAD_DIR}/sam-3d-objects/checkpoints" >&2
  exit 1
fi

mv "${DOWNLOAD_DIR}/sam-3d-objects/checkpoints" "${CHECKPOINT_DIR}"
rm -rf "${DOWNLOAD_DIR}/sam-3d-objects"
echo "Checkpoints ready at ${CHECKPOINT_DIR}"
