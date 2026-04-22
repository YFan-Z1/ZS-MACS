from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import torch.nn.functional as F
from PIL import ImageEnhance
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from .utils import *

# -----------------------------
# Image path resolver
# -----------------------------
class DefaultVAWImageResolver:
    def __init__(self, image_root: str | Path, extra_subdirs: Optional[Sequence[str]] = None) -> None:
        self.image_root = Path(image_root)
        default_subdirs = ["", "_images", "images", "VG_100K", "VG_100K_2"]
        if extra_subdirs:
            default_subdirs.extend(list(extra_subdirs))
        seen = set()
        self.subdirs = []
        for item in default_subdirs:
            if item not in seen:
                seen.add(item)
                self.subdirs.append(item)

    def __call__(self, image_id: str | int) -> Path:
        image_stem = str(image_id)
        candidates: List[Path] = []
        for sub in self.subdirs:
            base = self.image_root / sub if sub else self.image_root
            candidates.append(base / f"{image_stem}.jpg")
            candidates.append(base / f"{image_stem}.png")
            candidates.append(base / image_stem)
        for path in candidates:
            if path.exists():
                return path
        tried = "\n".join(str(p) for p in candidates)
        raise FileNotFoundError(f"Could not resolve image for image_id={image_id}. Tried:\n{tried}")


# -----------------------------
# Official-style augmentation configs
# -----------------------------
@dataclass
class SegAugConfig:
    input_size: int = 512
    pipeline: str = "semseg"  # one of {"semseg", "lsj", "simple"}

    # semantic-seg style resize (torchvision/mmseg-like)
    base_size: Optional[int] = None
    min_scale: float = 0.5
    max_scale: float = 2.0

    # detectron2-style ResizeScale / LSJ
    lsj_min_scale: float = 0.1
    lsj_max_scale: float = 2.0

    # common
    crop_size: Optional[Tuple[int, int]] = None
    flip_prob: float = 0.5
    pad_value: Tuple[float, float, float] = tuple(v * 255.0 for v in CLIP_MEAN)
    photometric_distort: bool = True
    photo_prob: float = 0.8
    brightness_delta: float = 32.0 / 255.0
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    saturation_range: Tuple[float, float] = (0.8, 1.2)
    hue_delta: float = 0.05
    min_kept_masks_after_crop: int = 1
    max_crop_tries: int = 10

    def __post_init__(self) -> None:
        if self.base_size is None:
            self.base_size = self.input_size
        if self.crop_size is None:
            self.crop_size = (self.input_size, self.input_size)


class PhotoMetricDistortion:
    """
    Mild PhotoMetricDistortion adapted for PIL images.
    We keep it lighter than classic semantic-seg defaults because open-vocabulary CLIP alignment is usually more fragile to heavy color perturbation.
    """

    def __init__(
        self,
        prob: float = 0.8,
        brightness_delta: float = 32.0 / 255.0,
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        saturation_range: Tuple[float, float] = (0.8, 1.2),
        hue_delta: float = 0.05,
    ) -> None:
        self.prob = prob
        self.brightness_delta = brightness_delta
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.prob:
            return image

        # brightness
        if random.random() < 0.5:
            factor = 1.0 + random.uniform(-self.brightness_delta, self.brightness_delta)
            image = ImageEnhance.Brightness(image).enhance(max(0.0, factor))

        contrast_first = random.random() < 0.5
        if contrast_first and random.random() < 0.5:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(*self.contrast_range))

        if random.random() < 0.5:
            image = ImageEnhance.Color(image).enhance(random.uniform(*self.saturation_range))

        # hue in HSV space
        if random.random() < 0.5:
            hsv = np.array(image.convert("HSV"), dtype=np.uint8)
            delta = int(round(random.uniform(-self.hue_delta, self.hue_delta) * 255.0))
            hsv[..., 0] = (hsv[..., 0].astype(np.int16) + delta) % 256
            image = Image.fromarray(hsv, mode="HSV").convert("RGB")

        if (not contrast_first) and random.random() < 0.5:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(*self.contrast_range))

        return image


class OfficialJointSegTransform:
    """
    Joint image/mask transform inspired by three common families:
      - torchvision semantic segmentation reference presets
      - mmsegmentation semantic segmentation train pipelines
      - detectron2 ResizeScale / LSJ-style train augmentation

    Output image is always normalized for CLIP and padded/cropped to fixed square size.
    """

    def __init__(self, split: str, cfg: Optional[SegAugConfig] = None) -> None:
        self.split = split
        self.cfg = cfg or SegAugConfig()
        self.photo = PhotoMetricDistortion(
            prob=self.cfg.photo_prob,
            brightness_delta=self.cfg.brightness_delta,
            contrast_range=self.cfg.contrast_range,
            saturation_range=self.cfg.saturation_range,
            hue_delta=self.cfg.hue_delta,
        )

    def _resize_keep_ar_to_long(self, image: Image.Image, masks: torch.Tensor, target_long: int) -> Tuple[Image.Image, torch.Tensor]:
        w, h = image.size
        scale = float(target_long) / max(h, w)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        image = image.resize((new_w, new_h), resample=Image.BICUBIC)
        if masks.numel() > 0:
            masks = F.interpolate(masks.unsqueeze(1).float(), size=(new_h, new_w), mode="nearest").squeeze(1).to(torch.uint8)
        return image, masks

    def _resize_scale_like_detectron2(self, image: Image.Image, masks: torch.Tensor) -> Tuple[Image.Image, torch.Tensor, float]:
        target_h, target_w = self.cfg.crop_size
        sampled_scale = random.uniform(self.cfg.lsj_min_scale, self.cfg.lsj_max_scale)
        scaled_target_h = max(1, int(round(target_h * sampled_scale)))
        scaled_target_w = max(1, int(round(target_w * sampled_scale)))
        w, h = image.size
        scale = min(float(scaled_target_h) / h, float(scaled_target_w) / w)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        image = image.resize((new_w, new_h), resample=Image.BICUBIC)
        if masks.numel() > 0:
            masks = F.interpolate(masks.unsqueeze(1).float(), size=(new_h, new_w), mode="nearest").squeeze(1).to(torch.uint8)
        return image, masks, scale

    def _pad_if_smaller(self, image: Image.Image, masks: torch.Tensor, target_hw: Tuple[int, int]) -> Tuple[Image.Image, torch.Tensor]:
        target_h, target_w = target_hw
        w, h = image.size
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)
        if pad_h == 0 and pad_w == 0:
            return image, masks
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top
        image = TF.pad(image, [left, top, right, bottom], fill=tuple(int(v) for v in self.cfg.pad_value))
        if masks.numel() > 0:
            masks = F.pad(masks, (left, right, top, bottom), value=0)
        return image, masks

    def _random_crop_with_instance_retry(
        self, image: Image.Image, masks: torch.Tensor, crop_hw: Tuple[int, int]
    ) -> Tuple[Image.Image, torch.Tensor, Tuple[int, int]]:
        crop_h, crop_w = crop_hw
        image, masks = self._pad_if_smaller(image, masks, crop_hw)
        w, h = image.size

        if h == crop_h and w == crop_w:
            return image, masks, (0, 0)

        best = None
        best_score = (-1, -1)
        max_y = h - crop_h
        max_x = w - crop_w
        tries = max(1, self.cfg.max_crop_tries)
        for _ in range(tries):
            top = random.randint(0, max_y)
            left = random.randint(0, max_x)
            cropped_masks = masks[:, top : top + crop_h, left : left + crop_w] if masks.numel() > 0 else masks
            if cropped_masks.numel() == 0:
                score = (0, 0)
            else:
                areas = cropped_masks.flatten(1).sum(-1)
                kept = int((areas >= 1).sum().item())
                score = (kept, int(areas.sum().item()))
            if score > best_score:
                best_score = score
                best = (left, top)
            if score[0] >= self.cfg.min_kept_masks_after_crop:
                break

        assert best is not None
        left, top = best
        image = TF.crop(image, top, left, crop_h, crop_w)
        if masks.numel() > 0:
            masks = masks[:, top : top + crop_h, left : left + crop_w]
        return image, masks, (left, top)

    def _filter_small_masks(self, masks: torch.Tensor) -> torch.Tensor:
        if masks.numel() == 0:
            return torch.zeros((0,), dtype=torch.bool)
        areas = masks.flatten(1).sum(-1)
        return areas >= 1

    def _train_semseg(self, image: Image.Image, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        meta: Dict[str, Any] = {"pipeline": "semseg", "flipped": False, "crop_xy": (0, 0), "scale": 1.0}
        target_long = int(round(random.uniform(self.cfg.min_scale, self.cfg.max_scale) * self.cfg.base_size))
        image, masks = self._resize_keep_ar_to_long(image, masks, target_long)
        meta["target_long"] = target_long

        if self.cfg.photometric_distort:
            image = self.photo(image)

        if random.random() < self.cfg.flip_prob:
            image = TF.hflip(image)
            if masks.numel() > 0:
                masks = torch.flip(masks, dims=[2])
            meta["flipped"] = True

        image, masks, crop_xy = self._random_crop_with_instance_retry(image, masks, self.cfg.crop_size)
        meta["crop_xy"] = crop_xy
        return self._finalize(image, masks, meta)

    def _train_lsj(self, image: Image.Image, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        meta: Dict[str, Any] = {"pipeline": "lsj", "flipped": False, "crop_xy": (0, 0), "scale": 1.0}
        image, masks, scale = self._resize_scale_like_detectron2(image, masks)
        meta["scale"] = scale

        if self.cfg.photometric_distort:
            image = self.photo(image)

        if random.random() < self.cfg.flip_prob:
            image = TF.hflip(image)
            if masks.numel() > 0:
                masks = torch.flip(masks, dims=[2])
            meta["flipped"] = True

        image, masks, crop_xy = self._random_crop_with_instance_retry(image, masks, self.cfg.crop_size)
        meta["crop_xy"] = crop_xy
        return self._finalize(image, masks, meta)

    def _eval(self, image: Image.Image, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        meta: Dict[str, Any] = {"pipeline": "eval", "flipped": False, "crop_xy": (0, 0), "scale": 1.0}
        image, masks = self._resize_keep_ar_to_long(image, masks, self.cfg.input_size)
        image, masks = self._pad_if_smaller(image, masks, self.cfg.crop_size)
        # center crop to fixed window if needed
        crop_h, crop_w = self.cfg.crop_size
        w, h = image.size
        top = max(0, (h - crop_h) // 2)
        left = max(0, (w - crop_w) // 2)
        image = TF.crop(image, top, left, crop_h, crop_w)
        if masks.numel() > 0:
            masks = masks[:, top : top + crop_h, left : left + crop_w]
        meta["crop_xy"] = (left, top)
        return self._finalize(image, masks, meta)

    def _finalize(self, image: Image.Image, masks: torch.Tensor, meta: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, CLIP_MEAN, CLIP_STD)
        masks = (masks > 0).to(torch.uint8)
        meta["out_size"] = (image_tensor.shape[-2], image_tensor.shape[-1])
        return image_tensor, masks, meta

    def __call__(self, image: Image.Image, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if self.split != "train":
            return self._eval(image, masks)
        if self.cfg.pipeline == "lsj":
            return self._train_lsj(image, masks)
        if self.cfg.pipeline == "simple":
            # legacy-ish simple CLIP-friendly variant
            meta: Dict[str, Any] = {"pipeline": "simple", "flipped": False, "crop_xy": (0, 0), "scale": 1.0}
            image, masks = self._resize_keep_ar_to_long(image, masks, self.cfg.input_size)
            if random.random() < self.cfg.flip_prob:
                image = TF.hflip(image)
                if masks.numel() > 0:
                    masks = torch.flip(masks, dims=[2])
                meta["flipped"] = True
            image, masks = self._pad_if_smaller(image, masks, self.cfg.crop_size)
            return self._finalize(image, masks, meta)
        return self._train_semseg(image, masks)


# -----------------------------
# Dataset
# -----------------------------
class VAWMaskDataset(Dataset):
    """
    Image-level VAW dataset for CLIP + MaskFormer / Mask2Former style baselines.

    Compared to a barebones implementation, this version exposes an official-style
    augmentation pipeline that can mimic semantic-segmentation references (resize + crop + flip +
    photometric distortion) or LSJ-style resize-scale + crop used in modern detection/segmentation codebases.
    """

    def __init__(
        self,
        annotation_json: str | Path,
        image_root: str | Path,
        *,
        split: str = "train",
        input_size: int = 512,
        min_mask_area: int = 16,
        min_mask_area_after_aug: int = 4,
        drop_empty_images: bool = True,
        object_vocab: Optional[Sequence[str]] = None,
        attribute_vocab: Optional[Sequence[str]] = None,
        resolver: Optional[Any] = None,
        return_region_crops: bool = False,
        region_crop_size: int = 224,
        masked_crop_fill: str = "mean",
        keep_images_with_zero_pos_attr: bool = True,
        seed: int = 0,
        aug_cfg: Optional[SegAugConfig] = None,
    ) -> None:
        super().__init__()
        random.seed(seed)
        np.random.seed(seed)

        self.annotation_json = Path(annotation_json)
        self.image_root = Path(image_root)
        self.split = split
        self.min_mask_area = min_mask_area
        self.min_mask_area_after_aug = min_mask_area_after_aug
        self.drop_empty_images = drop_empty_images
        self.return_region_crops = return_region_crops
        self.region_crop_size = region_crop_size
        self.masked_crop_fill = masked_crop_fill
        self.keep_images_with_zero_pos_attr = keep_images_with_zero_pos_attr
        self.resolver = resolver or DefaultVAWImageResolver(image_root)
        self.aug_cfg = aug_cfg or SegAugConfig(input_size=input_size)
        self.transform = OfficialJointSegTransform(split=split, cfg=self.aug_cfg)

        with open(self.annotation_json, "r", encoding="utf-8") as f:
            raw_ann = json.load(f)

        normalized_ann = []
        inferred_objects = set()
        inferred_attrs = set()
        for ann in raw_ann:
            obj = normalize_term(str(ann["object_name"]))
            pos_attrs = [normalize_term(a) for a in ann.get("positive_attributes", [])]
            neg_attrs = [normalize_term(a) for a in ann.get("negative_attributes", [])]
            inferred_objects.add(obj)
            inferred_attrs.update(pos_attrs)
            inferred_attrs.update(neg_attrs)
            normalized_ann.append(
                {
                    "image_id": str(ann["image_id"]),
                    "instance_id": str(ann["instance_id"]),
                    "instance_bbox": ann["instance_bbox"],
                    "instance_polygon": ann.get("instance_polygon", None),
                    "object_name": obj,
                    "positive_attributes": pos_attrs,
                    "negative_attributes": neg_attrs,
                }
            )

        self.object_vocab = sorted(object_vocab) if object_vocab is not None else sorted(inferred_objects)
        self.attribute_vocab = sorted(attribute_vocab) if attribute_vocab is not None else sorted(inferred_attrs)
        self.obj2idx = {name: idx for idx, name in enumerate(self.object_vocab)}
        self.attr2idx = {name: idx for idx, name in enumerate(self.attribute_vocab)}

        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for ann in normalized_ann:
            if ann["object_name"] not in self.obj2idx:
                continue
            ann["positive_attributes"] = [a for a in ann["positive_attributes"] if a in self.attr2idx]
            ann["negative_attributes"] = [a for a in ann["negative_attributes"] if a in self.attr2idx]
            grouped[ann["image_id"]].append(ann)

        self.records: List[Dict[str, Any]] = []
        for image_id, anns in grouped.items():
            kept = anns
            if not keep_images_with_zero_pos_attr:
                kept = [a for a in kept if len(a["positive_attributes"]) > 0]
            if len(kept) == 0 and drop_empty_images:
                continue
            self.records.append({"image_id": image_id, "annotations": kept})
        self.records.sort(key=lambda x: x["image_id"])

    def __len__(self) -> int:
        return len(self.records)

    def _make_attr_targets(self, anns: Sequence[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_inst = len(anns)
        num_attr = len(self.attribute_vocab)
        pos = torch.zeros((num_inst, num_attr), dtype=torch.float32)
        neg = torch.zeros((num_inst, num_attr), dtype=torch.float32)
        labeled = torch.zeros((num_inst, num_attr), dtype=torch.float32)
        for i, ann in enumerate(anns):
            for attr in ann["positive_attributes"]:
                idx = self.attr2idx[attr]
                pos[i, idx] = 1.0
                labeled[i, idx] = 1.0
            for attr in ann["negative_attributes"]:
                idx = self.attr2idx[attr]
                neg[i, idx] = 1.0
                labeled[i, idx] = 1.0
        return pos, neg, labeled

    def _make_region_crop(self, pil_image: Image.Image, mask: torch.Tensor, box_xyxy: torch.Tensor) -> torch.Tensor:
        x1, y1, x2, y2 = [int(v) for v in box_xyxy.tolist()]
        x2 = max(x1 + 1, x2)
        y2 = max(y1 + 1, y2)
        crop = pil_image.crop((x1, y1, x2, y2)).convert("RGB")
        crop_mask = mask[y1:y2, x1:x2].cpu().numpy().astype(np.uint8)
        crop_np = np.array(crop, dtype=np.uint8)
        if self.masked_crop_fill == "mean":
            fill_rgb = np.array([int(round(v * 255.0)) for v in CLIP_MEAN], dtype=np.uint8)
        elif self.masked_crop_fill == "black":
            fill_rgb = np.zeros((3,), dtype=np.uint8)
        else:
            fill_rgb = np.array([int(round(v * 255.0)) for v in CLIP_MEAN], dtype=np.uint8)
        crop_np[crop_mask == 0] = fill_rgb
        crop = Image.fromarray(crop_np)
        crop = crop.resize((self.region_crop_size, self.region_crop_size), resample=Image.BICUBIC)
        crop = TF.to_tensor(crop)
        crop = TF.normalize(crop, CLIP_MEAN, CLIP_STD)
        return crop

    def _filter_instances_after_aug(self, masks: torch.Tensor, anns: List[Dict[str, Any]]) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        if masks.numel() == 0:
            return masks, []
        areas = masks.flatten(1).sum(-1)
        valid = areas >= self.min_mask_area_after_aug
        if valid.sum() == 0:
            return torch.zeros((0, masks.shape[-2], masks.shape[-1]), dtype=masks.dtype), []
        kept_masks = masks[valid]
        kept_anns = [ann for ann, keep in zip(anns, valid.tolist()) if keep]
        return kept_masks, kept_anns

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        record = self.records[index]
        image_id = record["image_id"]
        anns = record["annotations"]

        image_path = self.resolver(image_id)
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        masks_np = []
        kept_anns = []
        for ann in anns:
            mask = polygon_to_mask(
                ann["instance_polygon"],
                width=width,
                height=height,
                fallback_box_xywh=ann["instance_bbox"],
            )
            if int(mask.sum()) < self.min_mask_area:
                continue
            masks_np.append(mask)
            kept_anns.append(ann)

        if len(masks_np) == 0:
            if not self.drop_empty_images:
                masks = torch.zeros((0, height, width), dtype=torch.uint8)
            else:
                return self.__getitem__((index + 1) % len(self.records))
        else:
            masks = torch.from_numpy(np.stack(masks_np, axis=0)).to(torch.uint8)

        image_tensor, masks, aug_meta = self.transform(image, masks)
        masks, kept_anns = self._filter_instances_after_aug(masks, kept_anns)
        if masks.numel() == 0 and self.drop_empty_images and self.split == "train":
            return self.__getitem__((index + 1) % len(self.records))

        boxes = boxes_from_masks(masks)
        obj_labels = torch.tensor([self.obj2idx[ann["object_name"]] for ann in kept_anns], dtype=torch.long)
        attr_pos, attr_neg, attr_labeled = self._make_attr_targets(kept_anns)

        region_crops: Optional[List[torch.Tensor]] = None
        if self.return_region_crops and masks.numel() > 0:
            denorm = torch.clamp(
                image_tensor * torch.tensor(CLIP_STD).view(3, 1, 1) + torch.tensor(CLIP_MEAN).view(3, 1, 1),
                0.0,
                1.0,
            )
            padded_image = TF.to_pil_image(denorm)
            region_crops = [self._make_region_crop(padded_image, m, b) for m, b in zip(masks, boxes)]

        target: Dict[str, Any] = {
            "image_id": image_id,
            "image_path": str(image_path),
            "size": torch.tensor([image_tensor.shape[1], image_tensor.shape[2]], dtype=torch.long),
            "orig_size": torch.tensor([height, width], dtype=torch.long),
            "masks": masks,
            "boxes": boxes,
            "labels_obj": obj_labels,
            "labels_attr_pos": attr_pos,
            "labels_attr_neg": attr_neg,
            "attr_is_labeled": attr_labeled,
            "instance_ids": [ann["instance_id"] for ann in kept_anns],
            "object_names": [ann["object_name"] for ann in kept_anns],
            "positive_attribute_names": [ann["positive_attributes"] for ann in kept_anns],
            "negative_attribute_names": [ann["negative_attributes"] for ann in kept_anns],
            "aug_meta": aug_meta,
        }
        if region_crops is not None:
            target["region_crops"] = region_crops
        return image_tensor, target

    def build_object_prompts(self) -> List[str]:
        return [build_default_object_prompt(o) for o in self.object_vocab]

    def build_attribute_prompts(self) -> List[str]:
        return [build_default_attribute_prompt(a) for a in self.attribute_vocab]

    def build_pair_prompts_for_instance(self, target: Dict[str, Any], instance_idx: int) -> List[str]:
        obj_name = target["object_names"][instance_idx]
        attrs = target["positive_attribute_names"][instance_idx]
        return [build_default_pair_prompt(attr, obj_name) for attr in attrs]


class VAWMaskedRegionDataset(Dataset):
    def __init__(
        self,
        annotation_json: str | Path,
        image_root: str | Path,
        *,
        crop_size: int = 224,
        min_mask_area: int = 16,
        object_vocab: Optional[Sequence[str]] = None,
        attribute_vocab: Optional[Sequence[str]] = None,
        resolver: Optional[Any] = None,
        masked_crop_fill: str = "mean",
        aug_cfg: Optional[SegAugConfig] = None,
    ) -> None:
        super().__init__()
        self.base = VAWMaskDataset(
            annotation_json=annotation_json,
            image_root=image_root,
            split="val",
            input_size=aug_cfg.input_size if aug_cfg is not None else 512,
            min_mask_area=min_mask_area,
            object_vocab=object_vocab,
            attribute_vocab=attribute_vocab,
            resolver=resolver,
            return_region_crops=True,
            region_crop_size=crop_size,
            masked_crop_fill=masked_crop_fill,
            aug_cfg=aug_cfg,
        )
        self.index_map: List[Tuple[int, int]] = []
        for img_idx in range(len(self.base)):
            _, tgt = self.base[img_idx]
            self.index_map.extend((img_idx, inst_idx) for inst_idx in range(len(tgt["instance_ids"])))

    @property
    def object_vocab(self) -> List[str]:
        return self.base.object_vocab

    @property
    def attribute_vocab(self) -> List[str]:
        return self.base.attribute_vocab

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        img_idx, inst_idx = self.index_map[index]
        _, target = self.base[img_idx]
        region_crops = target.get("region_crops", None)
        if region_crops is None:
            raise RuntimeError("VAWMaskedRegionDataset expects return_region_crops=True in the base dataset.")
        if inst_idx >= len(region_crops):
            return self.__getitem__((index + 1) % len(self.index_map))
        return {
            "crop": region_crops[inst_idx],
            "object_label": target["labels_obj"][inst_idx],
            "labels_attr_pos": target["labels_attr_pos"][inst_idx],
            "labels_attr_neg": target["labels_attr_neg"][inst_idx],
            "attr_is_labeled": target["attr_is_labeled"][inst_idx],
            "object_name": target["object_names"][inst_idx],
            "positive_attribute_names": target["positive_attribute_names"][inst_idx],
            "negative_attribute_names": target["negative_attribute_names"][inst_idx],
            "image_id": target["image_id"],
            "instance_id": target["instance_ids"][inst_idx],
        }


# -----------------------------
# Collate functions
# -----------------------------
def collate_vaw_mask_batch(batch: Sequence[Tuple[torch.Tensor, Dict[str, Any]]]) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets)


def collate_vaw_region_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    crops = torch.stack([item["crop"] for item in batch], dim=0)
    object_labels = torch.stack([item["object_label"] for item in batch], dim=0)
    attr_pos = torch.stack([item["labels_attr_pos"] for item in batch], dim=0)
    attr_neg = torch.stack([item["labels_attr_neg"] for item in batch], dim=0)
    attr_labeled = torch.stack([item["attr_is_labeled"] for item in batch], dim=0)
    return {
        "crops": crops,
        "object_labels": object_labels,
        "labels_attr_pos": attr_pos,
        "labels_attr_neg": attr_neg,
        "attr_is_labeled": attr_labeled,
        "meta": batch,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--input_size", type=int, default=512)
    parser.add_argument("--pipeline", type=str, default="semseg", choices=["semseg", "lsj", "simple"])
    args = parser.parse_args()

    ds = VAWMaskDataset(
        annotation_json=args.ann,
        image_root=args.image_root,
        split=args.split,
        input_size=args.input_size,
        return_region_crops=True,
        aug_cfg=SegAugConfig(input_size=args.input_size, pipeline=args.pipeline),
    )
    print(f"Dataset size: {len(ds)}")
    image, target = ds[0]
    print("image:", tuple(image.shape))
    print("masks:", tuple(target["masks"].shape))
    print("boxes:", tuple(target["boxes"].shape))
    print("labels_obj:", tuple(target["labels_obj"].shape))
    print("labels_attr_pos:", tuple(target["labels_attr_pos"].shape))
    print("num region crops:", len(target.get("region_crops", [])))
    print("aug_meta:", target["aug_meta"])