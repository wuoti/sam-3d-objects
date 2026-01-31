import os
import sys
import time
from typing import Optional, Dict, Any

import numpy as np
from PIL import Image

# Fix for the timm/torchscript segfault you already saw
os.environ.setdefault("PYTORCH_JIT", "0")

# mirror demo.py imports
sys.path.append("notebook")
from inference import Inference, load_image  # noqa

from .storage import job_output_path

_INFERENCE: Optional[Inference] = None
MAX_INPUT_SIZE = int(os.environ.get("SAM3D_MAX_INPUT_SIZE", "0"))

def get_inference() -> Inference:
    global _INFERENCE
    if _INFERENCE is None:
        tag = "hf"
        config_path = f"checkpoints/{tag}/pipeline.yaml"
        _INFERENCE = Inference(config_path, compile=False)
    return _INFERENCE


def _load_image_with_resize(image_path: str) -> np.ndarray:
    if MAX_INPUT_SIZE <= 0:
        return load_image(image_path)

    image = Image.open(image_path).convert("RGBA")
    w, h = image.size
    max_side = max(w, h)
    if max_side > MAX_INPUT_SIZE:
        scale = MAX_INPUT_SIZE / max_side
        new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        image = image.resize(new_size, resample=Image.LANCZOS)
    return np.array(image, dtype=np.uint8)

def run_job(
    job_id: str,
    image_path: str,
    seed: int = 42,
) -> Dict[str, Any]:
    t0 = time.time()
    inference = get_inference()

    # Your flow: RGBA with mask embedded in alpha
    image = _load_image_with_resize(image_path)
    mask = None

    output = inference(
        image,
        mask,
        seed=seed,
        export_usd_path=None,
        usd_scale_factor=100.0,
    )

    produced: Dict[str, str] = {}

    # GLB export: pipeline returns a trimesh object, not a path
    if output.get("glb") is not None:
        glb_path = job_output_path(job_id, "reconstruction.glb")
        output["glb"].export(str(glb_path))
        produced["reconstruction.glb"] = str(glb_path)
    elif output.get("glb_path") and os.path.exists(output["glb_path"]):
        produced["reconstruction.glb"] = output["glb_path"]
    else:
        raise RuntimeError("GLB output was not produced")

    return {"files": produced, "elapsed_sec": time.time() - t0}
