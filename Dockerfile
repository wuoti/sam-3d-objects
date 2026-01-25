# -----------------------------
# Stage 1: builder (has nvcc)
# -----------------------------
FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04 AS builder
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# ---- OS deps + deadsnakes PPA tools ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    git curl ca-certificates \
    build-essential ninja-build cmake \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    libstdc++6 libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

# ---- Install stable Python 3.11 ----
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# ---- Create venv early: everything goes into /app/.venv ----
RUN python3 -m venv /app/.venv
ENV PATH="/app/.venv/bin:/root/.local/bin:${PATH}"

RUN python -m pip install --upgrade pip setuptools wheel

# ---- Install uv (used for project deps) ----
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# ---- Torch CUDA 12.9 wheels (into venv) ----
RUN python -m pip install \
    torch==2.8.0+cu129 \
    torchvision==0.23.0+cu129 \
    torchaudio==2.8.0+cu129 \
    --index-url https://download.pytorch.org/whl/cu129

# ---- Build env for CUDA extensions ----
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="8.9+PTX"
ENV PYTORCH_JIT=0

# ---- PyTorch3D from source (needs nvcc + torch) ----
RUN python -m pip install --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"

# ---- App deps via uv (from pyproject/lock) ----
COPY . /app
ENV UV_CACHE_DIR=/tmp/uv-cache
RUN uv sync --no-dev --active

# ---- Sanity checks (this time we DO validate nvdiffrast presence) ----
RUN python - <<'PY'
import sys, torch, numpy as np
print("python:", sys.version)
print("torch:", torch.__version__, "cuda available at build:", torch.cuda.is_available())
print("numpy:", np.__version__)
import pytorch3d
import nvdiffrast.torch as dr
print("pytorch3d ok, nvdiffrast ok")
PY

# ---- Cleanup caches to keep the copied venv smaller ----
RUN rm -rf /root/.cache /tmp/uv-cache /var/tmp/*


# -----------------------------
# Stage 2: runtime (smaller)
# -----------------------------
FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu22.04 AS runtime
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Runtime packages (you asked to keep git/git-lfs/curl/unzip)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    ca-certificates \
    git git-lfs curl unzip \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    libgl1 libgomp1 \
    libstdc++6 libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

# Python 3.11 in runtime too
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Copy app + venv from builder
COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:${PATH}"
ENV PYTHONPATH=/app
ENV PYTORCH_JIT=0

EXPOSE 8000
CMD ["bash", "-lc", "/app/scripts/entrypoint.sh"]
