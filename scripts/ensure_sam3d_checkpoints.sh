#!/usr/bin/env bash
set -euo pipefail

echo "ensure_sam3d_checkpoints v2"

CHECKPOINT_DIR="${SAM3D_CHECKPOINT_DIR:-/app/checkpoints/hf}"
DOWNLOAD_DIR="${SAM3D_CHECKPOINT_DOWNLOAD_DIR:-/app/checkpoints}"
REPO_ID="${SAM3D_HF_REPO:-facebook/sam-3d-objects}"
RETRY_COUNT="${SAM3D_HF_RETRY_COUNT:-5}"
RETRY_DELAY_SEC="${SAM3D_HF_RETRY_DELAY_SEC:-5}"

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

for ((i=1; i<=RETRY_COUNT; i++)); do
  if touch "${DOWNLOAD_DIR}/.rw-test" 2>/dev/null; then
    rm -f "${DOWNLOAD_DIR}/.rw-test"
    break
  fi
  echo "Waiting for ${DOWNLOAD_DIR} to be writable... (${i}/${RETRY_COUNT})"
  sleep "${RETRY_DELAY_SEC}"
done

mkdir -p "${DOWNLOAD_DIR}"
rm -rf "${DOWNLOAD_DIR}/checkpoints"

echo "Authenticating to Hugging Face"
hf auth login --token "${HF_TOKEN}"

echo "Checking checkpoint access for ${REPO_ID}"
repo_files=""
for ((i=1; i<=RETRY_COUNT; i++)); do
  repo_files="$(hf repo-files --repo-id "${REPO_ID}" --repo-type model 2>/dev/null | grep -E '^checkpoints/' || true)"
  if [[ -z "${repo_files}" ]]; then
    repo_files="$(hf repo-files --repo-type model "${REPO_ID}" 2>/dev/null | grep -E '^checkpoints/' || true)"
  fi
  if [[ -n "${repo_files}" ]]; then
    break
  fi
  echo "No checkpoint files visible yet; retrying in ${RETRY_DELAY_SEC}s (${i}/${RETRY_COUNT})"
  sleep "${RETRY_DELAY_SEC}"
done
if [[ -z "${repo_files}" ]]; then
  echo "No checkpoint files visible for ${REPO_ID}. Check HF access approval." >&2
  exit 1
fi

echo "Downloading SAM-3D-Objects checkpoints to ${DOWNLOAD_DIR}"
hf download "${REPO_ID}" --repo-type model --include "checkpoints/**" --local-dir "${DOWNLOAD_DIR}" --max-workers 1

if [[ ! -d "${DOWNLOAD_DIR}/checkpoints" ]]; then
  echo "Download did not produce ${DOWNLOAD_DIR}/checkpoints" >&2
  exit 1
fi

mv "${DOWNLOAD_DIR}/checkpoints" "${CHECKPOINT_DIR}"
echo "Checkpoints ready at ${CHECKPOINT_DIR}"
