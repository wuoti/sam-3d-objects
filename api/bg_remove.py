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
    model_type = os.environ.get("SAM_MODEL_TYPE", "vit_h")
    if model_type not in sam_model_registry:
        raise RuntimeError(f"Unknown SAM model type: {model_type}")

    checkpoint = os.environ.get("SAM_CHECKPOINT", f"/app/checkpoints/sam/{model_type}.pth")
    if not checkpoint:
        raise RuntimeError("SAM_CHECKPOINT is not set")

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


def _score_mask_background(mask: np.ndarray, image_shape) -> float:
    h, w = image_shape[:2]
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return -1.0

    cy = ys.mean() / max(h - 1, 1)
    cx = xs.mean() / max(w - 1, 1)
    center_dist = ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5

    edge = np.zeros_like(mask, dtype=bool)
    edge[0, :] = True
    edge[-1, :] = True
    edge[:, 0] = True
    edge[:, -1] = True
    edge_touch = (mask.astype(bool) & edge).sum() / max(edge.sum(), 1)

    size_ratio = mask.sum() / (h * w)

    center_w = float(os.environ.get("SAM_CENTER_WEIGHT", "1.0"))
    edge_w = float(os.environ.get("SAM_EDGE_WEIGHT", "0.7"))
    size_w = float(os.environ.get("SAM_SIZE_WEIGHT", "0.3"))

    return (edge_w * edge_touch) + (size_w * size_ratio) - (center_w * (1.0 - center_dist))


def _pick_background_mask(masks, image_shape, image_area: int) -> np.ndarray:
    # Prefer masks not covering almost the whole image.
    max_bg_ratio = float(os.environ.get("SAM_BG_MAX_RATIO", "0.95"))
    allowed = [m for m in masks if (m.get("area", 0) / image_area) <= max_bg_ratio]
    if not allowed:
        allowed = masks
    best = max(allowed, key=lambda m: _score_mask_background(m["segmentation"], image_shape))
    return best["segmentation"].astype(np.uint8)


def remove_background(image_bytes: bytes, invert: bool = False) -> bytes:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    np_image = np.array(image)

    generator = get_sam_generator()
    with torch.inference_mode():
        masks = generator.generate(np_image)

    if not masks:
        raise RuntimeError("No masks generated")

    image_area = np_image.shape[0] * np_image.shape[1]
    mask = _pick_background_mask(masks, np_image.shape, image_area)
    invert = not invert
    if invert:
        mask = 1 - mask
    alpha = (mask * 255).astype(np.uint8)

    rgba = np.dstack([np_image, alpha])
    out = Image.fromarray(rgba, mode="RGBA")

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()
