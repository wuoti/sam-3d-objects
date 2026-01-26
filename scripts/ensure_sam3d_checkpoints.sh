#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_DIR="${SAM3D_CHECKPOINT_DIR:-/app/checkpoints/hf}"
DOWNLOAD_DIR="${SAM3D_CHECKPOINT_DOWNLOAD_DIR:-/app/checkpoints}"
REPO_ID="${SAM3D_HF_REPO:-facebook/sam-3d-objects}"

if [[ -d "${CHECKPOINT_DIR}" ]] && [[ -n "$(ls -A "${CHECKPOINT_DIR}" 2>/dev/null)" ]]; then
  echo "SAM-3D-Objects checkpoints already present: ${CHECKPOINT_DIR}"
  exit 0
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "huggingface-cli (hf) is not installed" >&2
  exit 1
fi

if ! command -v git-xet >/dev/null 2>&1; then
  echo "git-xet is not installed" >&2
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set" >&2
  exit 1
fi

mkdir -p "${DOWNLOAD_DIR}"
rm -rf "${DOWNLOAD_DIR}/checkpoints"

echo "Authenticating to Hugging Face"
hf auth login --token "${HF_TOKEN}"

echo "Downloading SAM-3D-Objects checkpoints to ${DOWNLOAD_DIR}"
hf download "${REPO_ID}" --include checkpoints/* --local-dir "${DOWNLOAD_DIR}"

if [[ ! -d "${DOWNLOAD_DIR}/checkpoints" ]]; then
  echo "Download did not produce ${DOWNLOAD_DIR}/checkpoints" >&2
  exit 1
fi

mv "${DOWNLOAD_DIR}/checkpoints" "${CHECKPOINT_DIR}"
echo "Checkpoints ready at ${CHECKPOINT_DIR}"
