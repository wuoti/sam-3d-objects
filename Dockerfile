# CUDA devel so nvcc exists (needed for pytorch3d build)
FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# ---- OS deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    python3.11 python3.11-venv python3.11-dev \
    build-essential ninja-build cmake \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Make python3 -> python3.11
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# ---- pip + uv ----
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# ---- Torch CUDA 12.9 wheels ----
RUN python3 -m pip install --upgrade pip setuptools wheel \
 && pip install \
    torch==2.8.0+cu129 torchvision==0.23.0+cu129 torchaudio==2.8.0+cu129 \
    --index-url https://download.pytorch.org/whl/cu129

# Sanity check: torch imports (no GPU required at build time)
RUN python3 - <<'PY'
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
PY

# ---- PyTorch3D from source (needs nvcc, build-essential, etc.) ----
# NOTE: This can take a while.
ENV CUDA_HOME=/usr/local/cuda
RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# ---- Copy repo ----
COPY . /app

# ---- Install project deps using uv (uses your lockfile if present) ----
# If you have uv.lock / pyproject.toml, this will reproduce your env.
RUN uv sync --frozen --no-dev || uv sync --no-dev

# Default env
ENV PYTHONPATH=/app

# You can override in docker run
CMD ["bash"]