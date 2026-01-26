#!/usr/bin/env bash
set -euo pipefail

MODEL_TYPE="${SAM_MODEL_TYPE:-vit_h}"
CHECKPOINT_PATH="${SAM_CHECKPOINT:-/app/checkpoints/sam/${MODEL_TYPE}.pth}"

case "${MODEL_TYPE}" in
  vit_h)
    DEFAULT_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    ;;
  vit_l)
    DEFAULT_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
    ;;
  vit_b)
    DEFAULT_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    ;;
  *)
    echo "Unknown SAM_MODEL_TYPE: ${MODEL_TYPE}" >&2
    exit 1
    ;;
 esac

export SAM_MODEL_TYPE="${MODEL_TYPE}"
export SAM_CHECKPOINT="${CHECKPOINT_PATH}"

if [[ -f "${CHECKPOINT_PATH}" ]]; then
  echo "SAM checkpoint already present: ${CHECKPOINT_PATH}"
  exit 0
fi

mkdir -p "$(dirname "${CHECKPOINT_PATH}")"
URL="${SAM_CHECKPOINT_URL:-${DEFAULT_URL}}"

if [[ -z "${URL}" ]]; then
  echo "SAM checkpoint missing and no download URL provided." >&2
  exit 1
fi

echo "Downloading SAM checkpoint (${MODEL_TYPE}) to ${CHECKPOINT_PATH}"
curl -L --retry 3 --retry-delay 2 -o "${CHECKPOINT_PATH}" "${URL}"
