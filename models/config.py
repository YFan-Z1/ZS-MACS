from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple


@dataclass
class CLIPBackboneConfig:
    pretrained_name: str = "openai/clip-vit-base-patch16"
    selected_hidden_states: Tuple[int, ...] = (3, 6, 9, 12)
    freeze_vision: bool = False
    freeze_text: bool = False
    train_projection_only: bool = False
    gradient_checkpointing: bool = False
    patch_dropout: float = 0.0


@dataclass
class PixelDecoderConfig:
    hidden_dim: int = 256
    mask_dim: int = 256
    fpn_dim: int = 256
    conv_norm: str = "gn"  # gn | bn | none


@dataclass
class QueryDecoderConfig:
    hidden_dim: int = 256
    num_queries: int = 100
    num_feature_levels: int = 4
    num_layers: int = 6
    nheads: int = 8
    dim_feedforward: int = 1024
    dropout: float = 0.0
    pre_norm: bool = False
    mask_threshold: float = 0.5
    mask_attention_stride: int = 4
    deep_supervision: bool = True


@dataclass
class TextHeadConfig:
    hidden_dim: int = 256
    logit_scale_init: float = 10.0
    background_init_std: float = 0.02
    attribute_mlp_ratio: int = 2
    normalize_queries: bool = True
    normalize_text: bool = True


@dataclass
class LossConfig:
    no_object_weight: float = 0.1
    matcher_cost_class: float = 2.0
    matcher_cost_mask: float = 5.0
    matcher_cost_dice: float = 5.0
    loss_object_weight: float = 2.0
    loss_mask_weight: float = 5.0
    loss_dice_weight: float = 5.0
    loss_attr_weight: float = 2.0
    loss_attr_pos_weight: float = 1.0
    aux_loss: bool = True


@dataclass
class ModelConfig:
    backbone: CLIPBackboneConfig = field(default_factory=CLIPBackboneConfig)
    pixel_decoder: PixelDecoderConfig = field(default_factory=PixelDecoderConfig)
    query_decoder: QueryDecoderConfig = field(default_factory=QueryDecoderConfig)
    text_head: TextHeadConfig = field(default_factory=TextHeadConfig)
    loss: LossConfig = field(default_factory=LossConfig)


DEFAULT_OBJECT_PROMPT = "a photo of a {name}"
DEFAULT_ATTRIBUTE_PROMPT = "a photo of something that is {name}"
DEFAULT_PAIR_PROMPT = "a photo of a {attr} {obj}"