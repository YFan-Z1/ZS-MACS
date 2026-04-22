from __future__ import annotations
from typing import Any, List, Optional, Sequence, Tuple
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms import InterpolationMode

BICUBIC = InterpolationMode.BICUBIC
NEAREST = InterpolationMode.NEAREST
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def normalize_term(term: str) -> str:
    return term.strip().lower().replace("_", " ")


def build_default_object_prompt(obj_name: str) -> str:
    return f"a photo of a {obj_name}"


def build_default_attribute_prompt(attr_name: str) -> str:
    return f"a photo of something that is {attr_name}"


def build_default_pair_prompt(attr_name: str, obj_name: str) -> str:
    return f"a photo of a {attr_name} {obj_name}"


# -----------------------------
# Geometry / mask helpers
# -----------------------------
def clamp_xyxy(box_xyxy: Sequence[float], width: int, height: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box_xyxy
    x1 = float(max(0.0, min(x1, width - 1)))
    y1 = float(max(0.0, min(y1, height - 1)))
    x2 = float(max(x1 + 1.0, min(x2, width)))
    y2 = float(max(y1 + 1.0, min(y2, height)))
    return x1, y1, x2, y2


def xywh_to_xyxy(box_xywh: Sequence[float]) -> Tuple[float, float, float, float]:
    x, y, w, h = box_xywh
    return float(x), float(y), float(x + w), float(y + h)


def polygon_to_mask(
    polygons: Optional[Sequence[Any]],
    width: int,
    height: int,
    *,
    fallback_box_xywh: Optional[Sequence[float]] = None,
) -> np.ndarray:
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    valid_polygon_drawn = False
    if polygons is not None:
        for poly in polygons:
            if poly is None:
                continue
            points: List[Tuple[float, float]] = []
            if len(poly) == 0:
                continue
            if isinstance(poly[0], (list, tuple)):
                points = [(float(p[0]), float(p[1])) for p in poly if len(p) >= 2]
            else:
                flat = [float(v) for v in poly]
                if len(flat) >= 6 and len(flat) % 2 == 0:
                    points = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]
            if len(points) >= 3:
                draw.polygon(points, outline=1, fill=1)
                valid_polygon_drawn = True

    if not valid_polygon_drawn and fallback_box_xywh is not None:
        x1, y1, x2, y2 = clamp_xyxy(xywh_to_xyxy(fallback_box_xywh), width, height)
        draw.rectangle([x1, y1, x2, y2], outline=1, fill=1)

    return np.array(mask_img, dtype=np.uint8)


def boxes_from_masks(masks: torch.Tensor) -> torch.Tensor:
    if masks.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32)

    boxes = []
    for mask in masks:
        ys, xs = torch.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            boxes.append(torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32))
            continue
        x1 = xs.min().float()
        y1 = ys.min().float()
        x2 = (xs.max() + 1).float()
        y2 = (ys.max() + 1).float()
        boxes.append(torch.tensor([x1, y1, x2, y2], dtype=torch.float32))
    return torch.stack(boxes, dim=0)
