from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageColor, ImageDraw
import torch


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

DEFAULT_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33",
    "#a65628", "#f781bf", "#17becf", "#bcbd22", "#1f77b4", "#d62728",
]


def patch_torchvision_register_fake() -> None:
    """
    Work around environments where torchvision import fails because optional C++ ops
    like torchvision::nms are unavailable during fake-op registration.
    """
    import torch

    original_register_fake = torch.library.register_fake

    def safe_register_fake(op_name, *args, **kwargs):
        decorator = original_register_fake(op_name, *args, **kwargs)

        def wrapped(fn):
            try:
                return decorator(fn)
            except RuntimeError as exc:
                if "torchvision::" in str(op_name) and "does not exist" in str(exc):
                    return fn
                raise

        return wrapped

    torch.library.register_fake = safe_register_fake


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    mean = torch.tensor(CLIP_MEAN, dtype=image_tensor.dtype, device=image_tensor.device).view(3, 1, 1)
    std = torch.tensor(CLIP_STD, dtype=image_tensor.dtype, device=image_tensor.device).view(3, 1, 1)
    image = image_tensor * std + mean
    image = image.clamp(0.0, 1.0)
    image = (image * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(image, mode="RGB")


def mask_to_rgba(mask: np.ndarray, color_rgb: Tuple[int, int, int], alpha: int) -> Image.Image:
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = color_rgb[0]
    rgba[..., 1] = color_rgb[1]
    rgba[..., 2] = color_rgb[2]
    rgba[..., 3] = (mask > 0).astype(np.uint8) * alpha
    return Image.fromarray(rgba, mode="RGBA")


def draw_mask_bbox(draw: ImageDraw.ImageDraw, mask: np.ndarray, color_rgb: Tuple[int, int, int], width: int = 2) -> None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=width)


def wrap_text(text: str, max_chars: int) -> List[str]:
    words = text.split()
    if not words:
        return [""]
    lines: List[str] = []
    cur = words[0]
    for word in words[1:]:
        candidate = f"{cur} {word}"
        if len(candidate) <= max_chars:
            cur = candidate
        else:
            lines.append(cur)
            cur = word
    lines.append(cur)
    return lines


def format_instance_lines(obj_name: str, attrs: Sequence[str], max_pairs_per_instance: int) -> List[str]:
    if not attrs:
        return [f"{obj_name} (no positive attrs)"]
    return [f"{attr} {obj_name}" for attr in attrs[:max_pairs_per_instance]]


def build_visualization(
    image: Image.Image,
    target: dict,
    *,
    alpha: int = 110,
    max_instances: int = 10,
    max_pairs_per_instance: int = 4,
    legend_width: int = 520,
) -> Image.Image:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)

    masks = target["masks"].cpu().numpy().astype(np.uint8)
    boxes = target["boxes"].cpu().numpy()
    object_names = target["object_names"]
    positive_attribute_names = target["positive_attribute_names"]

    num_instances = min(len(object_names), max_instances)
    legend_lines: List[Tuple[Tuple[int, int, int], List[str]]] = []

    for i in range(num_instances):
        color_rgb = ImageColor.getrgb(DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
        overlay.alpha_composite(mask_to_rgba(masks[i], color_rgb, alpha))
        draw_mask_bbox(draw_overlay, masks[i], color_rgb, width=2)

        x1, y1, x2, y2 = boxes[i].tolist()
        instance_lines = format_instance_lines(object_names[i], positive_attribute_names[i], max_pairs_per_instance)
        label = f"#{i}: {instance_lines[0]}"
        tx = int(max(2, x1))
        ty = int(max(2, y1 - 14))
        text_bbox = draw_overlay.textbbox((tx, ty), label)
        pad = 2
        draw_overlay.rectangle(
            [text_bbox[0] - pad, text_bbox[1] - pad, text_bbox[2] + pad, text_bbox[3] + pad],
            fill=(0, 0, 0, 160),
            outline=color_rgb,
            width=1,
        )
        draw_overlay.text((tx, ty), label, fill=(255, 255, 255, 255))
        legend_lines.append((color_rgb, [f"Instance #{i}: {object_names[i]}"] + [f"  - {p}" for p in instance_lines]))

    composed = Image.alpha_composite(base, overlay).convert("RGB")

    legend_panel = Image.new("RGB", (legend_width, composed.height), (250, 250, 250))
    draw_legend = ImageDraw.Draw(legend_panel)
    draw_legend.rectangle([0, 0, legend_panel.width - 1, legend_panel.height - 1], outline=(180, 180, 180), width=1)

    draw_legend.text((16, 12), f"image_id: {target['image_id']} | instances shown: {num_instances}", fill=(0, 0, 0))
    draw_legend.text((16, 34), "Positive pair names", fill=(60, 60, 60))

    y = 62
    line_h = 16
    for color_rgb, lines in legend_lines:
        if y > legend_panel.height - 24:
            break
        draw_legend.rounded_rectangle([16, y + 2, 30, y + 14], radius=3, fill=color_rgb)
        x_text = 40
        for j, line in enumerate(lines):
            for subline in wrap_text(line, max_chars=55):
                if y > legend_panel.height - 18:
                    break
                draw_legend.text((x_text, y), subline, fill=(0, 0, 0) if j == 0 else (60, 60, 60))
                y += line_h
        y += 8

    canvas = Image.new("RGB", (composed.width + legend_panel.width, composed.height), (255, 255, 255))
    canvas.paste(composed, (0, 0))
    canvas.paste(legend_panel, (composed.width, 0))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize VAW instance masks with pair names overlaid on the image.")
    parser.add_argument("--ann", type=str, required=True, help="Path to VAW annotation json.")
    parser.add_argument("--image_root", type=str, required=True, help="Path to image directory root.")
    parser.add_argument(
        "--dataset_module_path",
        type=str,
        default=str(Path(__file__).with_name("vaw_clip_baseline_dataset.py")),
        help="Path to vaw_clip_baseline_dataset.py",
    )
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--input_size", type=int, default=512)
    parser.add_argument("--index", type=int, default=0, help="Dataset index to visualize.")
    parser.add_argument("--image_id", type=str, default=None, help="Optional image_id override. If set, it takes precedence over --index.")
    parser.add_argument("--save_path", type=str, default="/tmp/vaw_demo_overlay.png")
    parser.add_argument("--max_instances", type=int, default=10)
    parser.add_argument("--max_pairs_per_instance", type=int, default=4)
    parser.add_argument("--alpha", type=int, default=110)
    args = parser.parse_args()

    patch_torchvision_register_fake()
    sys.path.insert(0, str(Path(args.dataset_module_path).resolve().parent))
    from dataset import VAWMaskDataset

    dataset = VAWMaskDataset(
        annotation_json=args.ann,
        image_root=args.image_root,
        split=args.split,
        input_size=args.input_size,
        return_region_crops=False,
    )

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty after filtering.")

    if args.image_id is not None:
        image_id = str(args.image_id)
        index_lookup = {record["image_id"]: i for i, record in enumerate(dataset.records)}
        if image_id not in index_lookup:
            raise KeyError(f"image_id={image_id} not found in dataset records.")
        sample_index = index_lookup[image_id]
    else:
        sample_index = max(0, min(args.index, len(dataset) - 1))

    image_tensor, target = dataset[sample_index]
    image = tensor_to_pil(image_tensor)
    vis = build_visualization(
        image,
        target,
        alpha=args.alpha,
        max_instances=args.max_instances,
        max_pairs_per_instance=args.max_pairs_per_instance,
    )

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    vis.save(save_path)

    print(f"Loaded dataset size: {len(dataset)}")
    print(f"Visualized sample index: {sample_index}")
    print(f"image_id: {target['image_id']}")
    print(f"num instances in target: {len(target['object_names'])}")
    print(f"Saved overlay to: {save_path}")


if __name__ == "__main__":
    main()