import os
import sys
import time
import zipfile
from typing import Optional, Dict, Any

# Fix for the timm/torchscript segfault you already saw
os.environ.setdefault("PYTORCH_JIT", "0")

# mirror demo.py imports
sys.path.append("notebook")
from inference import Inference, load_image  # noqa

from .storage import job_output_path

_INFERENCE: Optional[Inference] = None

def get_inference() -> Inference:
    global _INFERENCE
    if _INFERENCE is None:
        tag = "hf"
        config_path = f"checkpoints/{tag}/pipeline.yaml"
        _INFERENCE = Inference(config_path, compile=False)
    return _INFERENCE

def run_job(
    job_id: str,
    image_path: str,
    seed: int = 42,
    export_usd: bool = True,
    usd_scale_factor: float = 100.0,
) -> Dict[str, Any]:
    t0 = time.time()
    inference = get_inference()

    # Your flow: RGBA with mask embedded in alpha
    image = load_image(image_path)
    mask = None

    usd_path = str(job_output_path(job_id, "reconstruction.usd")) if export_usd else None

    output = inference(
        image,
        mask,
        seed=seed,
        export_usd_path=usd_path,
        usd_scale_factor=usd_scale_factor,
    )

    produced: Dict[str, str] = {}

    # Always export gaussian splat
    ply_path = job_output_path(job_id, "splat.ply")
    output["gs"].save_ply(str(ply_path))
    produced["splat.ply"] = str(ply_path)

    # USD export is optional and can fail
    if output.get("usd_path") and os.path.exists(output["usd_path"]):
        produced["reconstruction.usd"] = output["usd_path"]

    if output.get("usdz_path") and os.path.exists(output["usdz_path"]):
        produced["reconstruction.usdz"] = output["usdz_path"]

    # If your pipeline also returns glb_path, include it
    if output.get("glb_path") and os.path.exists(output["glb_path"]):
        produced["reconstruction.glb"] = output["glb_path"]

    # zip outputs for convenience
    zip_path = job_output_path(job_id, "outputs.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, p in produced.items():
            z.write(p, arcname=name)
    produced["outputs.zip"] = str(zip_path)

    return {"files": produced, "elapsed_sec": time.time() - t0}