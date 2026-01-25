import io
import os
import threading
from typing import Optional

import numpy as np
from PIL import Image
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

_LOCK = threading.Lock()
_GENERATOR: Optional[SamAutomaticMaskGenerator] = None


def _build_generator() -> SamAutomaticMaskGenerator:
    checkpoint = os.environ.get("SAM_CHECKPOINT")
    if not checkpoint:
        raise RuntimeError("SAM_CHECKPOINT is not set")

    model_type = os.environ.get("SAM_MODEL_TYPE", "vit_h")
    if model_type not in sam_model_registry:
        raise RuntimeError(f"Unknown SAM model type: {model_type}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; GPU is required for this endpoint")

    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device="cuda")

    points_per_side = int(os.environ.get("SAM_POINTS_PER_SIDE", "32"))
    pred_iou_thresh = float(os.environ.get("SAM_PRED_IOU_THRESH", "0.88"))
    stability_score_thresh = float(os.environ.get("SAM_STABILITY_SCORE_THRESH", "0.95"))
    min_mask_region_area = int(os.environ.get("SAM_MIN_MASK_REGION_AREA", "0"))

    return SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area,
    )


def get_sam_generator() -> SamAutomaticMaskGenerator:
    global _GENERATOR
    if _GENERATOR is None:
        with _LOCK:
            if _GENERATOR is None:
                _GENERATOR = _build_generator()
    return _GENERATOR


def remove_background(image_bytes: bytes) -> bytes:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    np_image = np.array(image)

    generator = get_sam_generator()
    with torch.inference_mode():
        masks = generator.generate(np_image)

    if not masks:
        raise RuntimeError("No masks generated")

    best = max(masks, key=lambda m: m.get("area", 0))
    mask = best["segmentation"].astype(np.uint8)
    alpha = (mask * 255).astype(np.uint8)

    rgba = np.dstack([np_image, alpha])
    out = Image.fromarray(rgba, mode="RGBA")

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()
