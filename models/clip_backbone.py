from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CLIPBackboneConfig

_TRANSFORMERS_IMPORT_ERROR: Optional[Exception] = None
from transformers import AutoProcessor, AutoTokenizer, CLIPModel

@dataclass
class CLIPImageFeatures:
    multiscale_features: List[torch.Tensor]
    pooled_output: torch.Tensor
    projected_pooled: torch.Tensor
    patch_tokens: torch.Tensor
    projected_patch_tokens: torch.Tensor
    patch_grid_hw: tuple[int, int]


class CLIPOpenVocabBackbone(nn.Module):

    def __init__(self, cfg: Optional[CLIPBackboneConfig] = None) -> None:
        super().__init__()
        if CLIPModel is None:
            raise ImportError(
                "transformers is required for CLIPOpenVocabBackbone. "
                f"Original import error: {repr(_TRANSFORMERS_IMPORT_ERROR)}"
            )

        self.cfg = cfg or CLIPBackboneConfig()
        self.clip = CLIPModel.from_pretrained(self.cfg.pretrained_name, use_safetensors=True,)
        self.processor = AutoProcessor.from_pretrained(self.cfg.pretrained_name, use_fast=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.pretrained_name)

        if self.cfg.gradient_checkpointing:
            self.clip.vision_model.gradient_checkpointing_enable()
            self.clip.text_model.gradient_checkpointing_enable()

        self.vision_hidden_size = self.clip.vision_model.config.hidden_size
        self.text_hidden_size = self.clip.text_model.config.hidden_size
        self.projection_dim = self.clip.projection_dim
        self.patch_size = self.clip.vision_model.config.patch_size

        self.selected_hidden_states = tuple(self.cfg.selected_hidden_states)
        self._freeze_modules()

    def _freeze_modules(self) -> None:
        if self.cfg.train_projection_only:
            for p in self.clip.parameters():
                p.requires_grad = False
            for p in self.clip.visual_projection.parameters():
                p.requires_grad = True
            for p in self.clip.text_projection.parameters():
                p.requires_grad = True
            self.clip.logit_scale.requires_grad = True
            return

        if self.cfg.freeze_vision:
            for p in self.clip.vision_model.parameters():
                p.requires_grad = False
            for p in self.clip.visual_projection.parameters():
                p.requires_grad = False
        if self.cfg.freeze_text:
            for p in self.clip.text_model.parameters():
                p.requires_grad = False
            for p in self.clip.text_projection.parameters():
                p.requires_grad = False

    def encode_image(self, pixel_values: torch.Tensor) -> CLIPImageFeatures:
        outputs = self.clip.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            # return_dict=True,
        )

        hidden_states = outputs.hidden_states
        selected: List[torch.Tensor] = []
        for layer_idx in self.selected_hidden_states:
            state = hidden_states[layer_idx]  # [B, 1+HW, C]
            patch_tokens = state[:, 1:, :]
            selected.append(self._tokens_to_map(patch_tokens, pixel_values.shape[-2], pixel_values.shape[-1]))

        last_hidden = outputs.last_hidden_state
        patch_tokens = last_hidden[:, 1:, :]
        projected_patch_tokens = self.clip.visual_projection(patch_tokens)
        projected_patch_tokens = F.normalize(projected_patch_tokens, dim=-1)

        pooled = outputs.pooler_output
        projected_pooled = self.clip.visual_projection(pooled)
        projected_pooled = F.normalize(projected_pooled, dim=-1)

        h = pixel_values.shape[-2] // self.patch_size
        w = pixel_values.shape[-1] // self.patch_size
        return CLIPImageFeatures(
            multiscale_features=selected,
            pooled_output=pooled,
            projected_pooled=projected_pooled,
            patch_tokens=patch_tokens,
            projected_patch_tokens=projected_patch_tokens,
            patch_grid_hw=(h, w),
        )

    def encode_text(self, texts: Sequence[str], device: Optional[torch.device] = None) -> torch.Tensor:
        inputs = self.tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt",)
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        # Use the public CLIP API for compatibility across transformers versions.
        text_embeds = self.clip.get_text_features(**inputs)
        text_embeds = F.normalize(text_embeds, dim=-1)
        return text_embeds

    def _tokens_to_map(self, patch_tokens: torch.Tensor, image_h: int, image_w: int) -> torch.Tensor:
        b, hw, c = patch_tokens.shape
        grid_h = image_h // self.patch_size
        grid_w = image_w // self.patch_size
        if grid_h * grid_w != hw:
            grid_h = grid_w = int(hw ** 0.5)
            if grid_h * grid_w != hw:
                raise ValueError(f"Cannot infer patch grid from hw={hw}, image=({image_h}, {image_w}), patch={self.patch_size}")
        return patch_tokens.transpose(1, 2).reshape(b, c, grid_h, grid_w)

    def forward(self, pixel_values: torch.Tensor) -> CLIPImageFeatures:
        return self.encode_image(pixel_values)