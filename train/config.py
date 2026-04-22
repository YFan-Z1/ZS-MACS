from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import copy
import yaml

from models.config import ModelConfig


@dataclass
class TrainConfig:
    exp_name: str = "vaw_open_vocab_seg"
    output_dir: str = "work_dirs/vaw_open_vocab_seg"
    seed: int = 42

    train_annotation_json: str = "data/train.json"
    val_annotation_json: Optional[str] = None
    image_root: str = "full-images"
    dataset_module: str = "vaw_seg_dataset_official"
    dataset_class: str = "VAWMaskDataset"
    collate_fn_name: str = "collate_vaw_mask_batch"
    dataset_pipeline: str = "semseg"
    input_size: int = 224
    return_region_crops: bool = False
    min_mask_area: int = 16
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    train_batch_size: int = 2
    val_batch_size: int = 2

    epochs: int = 20
    max_steps: int = -1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    mixed_precision: str = "no"  # no|fp16|bf16
    find_unused_parameters: bool = False

    optimizer: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.05
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    momentum: float = 0.9
    nesterov: bool = False
    separate_no_decay: bool = True
    no_decay_keywords: List[str] = field(default_factory=lambda: [
        'bias','bn','ln','norm','layernorm','layer_norm','batchnorm','batch_norm'
    ])
    no_decay_weight_decay: float = 0.0
    optimizer_param_groups: Optional[List[Dict[str, Any]]] = None
    trainable_keywords: Optional[List[str]] = None
    frozen_keywords: Optional[List[str]] = None
    reset_trainable: bool = False
    use_group_keywords_as_trainable: bool = True

    lr_scheduler: str = "linear_warmup_cosine"
    warmup_ratio: float = 0.05
    warmup_steps: int = 0
    min_lr_ratio: float = 0.0
    step_size: int = 0
    gamma: float = 0.5
    milestones: Optional[List[int]] = None
    eta_min_ratio: float = 0.0

    log_interval: int = 20
    val_interval: int = 1000
    checkpoint_interval: int = 1000
    save_every_epoch: bool = True
    keep_last_k: int = 3
    monitor: str = "val/total_loss"
    monitor_mode: str = "min"

    use_wandb: bool = True
    wandb_project: str = "vaw-open-vocab-seg"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    wandb_watch: bool = False

    resume_from: Optional[str] = None
    save_best: bool = True
    debug: bool = False


@dataclass
class FullConfig:
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


def _update_dataclass(instance: Any, update_dict: Dict[str, Any]) -> Any:
    if not is_dataclass(instance):
        raise TypeError(f"Expected dataclass instance, got {type(instance)}")
    for f in fields(instance):
        if f.name not in update_dict:
            continue
        value = update_dict[f.name]
        current = getattr(instance, f.name)
        if is_dataclass(current) and isinstance(value, dict):
            _update_dataclass(current, value)
        else:
            setattr(instance, f.name, value)
    return instance


def load_config(config_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> FullConfig:
    cfg = FullConfig()
    if config_path is not None:
        with open(config_path, 'r', encoding='utf-8') as f:
            loaded = yaml.safe_load(f) or {}
        _update_dataclass(cfg, loaded)
    if overrides:
        _update_dataclass(cfg, overrides)
    return cfg


def dump_config(cfg: FullConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)