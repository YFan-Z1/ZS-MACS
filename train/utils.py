from __future__ import annotations

import importlib
import json
import logging
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import FullConfig


def setup_logging(output_dir: str | Path, rank: int = 0) -> logging.Logger:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f'vaw_open_vocab_seg.rank{rank}')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    if rank == 0:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    fh = logging.FileHandler(output_dir / f'train_rank{rank}.log', encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_targets_to_device(targets: List[Dict[str, Any]], device: torch.device) -> List[Dict[str, Any]]:
    moved = []
    for t in targets:
        out = {}
        for k, v in t.items():
            if torch.is_tensor(v):
                out[k] = v.to(device, non_blocking=True)
            else:
                out[k] = v
        moved.append(out)
    return moved


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return {'total': int(total), 'trainable': int(trainable), 'frozen': int(total - trainable)}


def import_dataset_components(module_name: str, dataset_class: str, collate_fn_name: str):
    mod = importlib.import_module(module_name)
    ds_cls = getattr(mod, dataset_class)
    collate_fn = getattr(mod, collate_fn_name)
    seg_aug_cls = getattr(mod, 'SegAugConfig', None)
    return ds_cls, collate_fn, seg_aug_cls


def build_datasets_and_loaders(cfg, accelerator=None):
    ds_cls, collate_fn, seg_aug_cls = import_dataset_components(
        cfg.dataset_module, cfg.dataset_class, cfg.collate_fn_name
    )

    aug_cfg = None
    if seg_aug_cls is not None:
        aug_cfg = seg_aug_cls(input_size=cfg.input_size, pipeline=cfg.dataset_pipeline)

    common_kwargs = dict(
        image_root=cfg.image_root,
        input_size=cfg.input_size,
        return_region_crops=cfg.return_region_crops,
        min_mask_area=cfg.min_mask_area,
    )
    if aug_cfg is not None:
        common_kwargs['aug_cfg'] = aug_cfg

    train_dataset = ds_cls(
        annotation_json=cfg.train_annotation_json,
        split='train',
        **common_kwargs,
    )

    val_ann = cfg.val_annotation_json or cfg.train_annotation_json
    val_dataset = ds_cls(
        annotation_json=val_ann,
        split='val',
        **common_kwargs,
    )

    # Avoid persistent_workers when num_workers == 0.
    persistent_workers = bool(cfg.persistent_workers and cfg.num_workers > 0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.val_batch_size,
        shuffle=False,
        num_workers=max(1, cfg.num_workers // 2) if cfg.num_workers > 0 else 0,
        pin_memory=cfg.pin_memory,
        persistent_workers=bool(persistent_workers and cfg.num_workers // 2 > 0),
        collate_fn=collate_fn,
        drop_last=False,
    )
    return train_dataset, val_dataset, train_loader, val_loader


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)