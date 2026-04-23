from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CLIPBackboneConfig

_TRANSFORMERS_IMPORT_ERROR: Optional[Exception] = None
try:
    from transformers import AutoTokenizer, CLIPModel
except Exception as e:
    CLIPModel = None
    AutoTokenizer = None
    _TRANSFORMERS_IMPORT_ERROR = e


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
        self.clip = CLIPModel.from_pretrained(
            self.cfg.pretrained_name,
            use_safetensors=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.pretrained_name)

        if self.cfg.gradient_checkpointing:
            try:
                self.clip.vision_model.gradient_checkpointing_enable()
                self.clip.text_model.gradient_checkpointing_enable()
            except Exception:
                pass

        self.vision_hidden_size = self.clip.vision_model.config.hidden_size
        self.text_hidden_size = self.clip.text_model.config.hidden_size
        self.projection_dim = self.clip.projection_dim
        self.patch_size = self.clip.vision_model.config.patch_size

        num_vision_layers = int(self.clip.vision_model.config.num_hidden_layers)
        validated = []
        for idx in self.cfg.selected_hidden_states:
            if idx < 0:
                idx = num_vision_layers + 1 + idx
            idx = max(1, min(int(idx), num_vision_layers))
            validated.append(idx)
        self.selected_hidden_states = tuple(dict.fromkeys(validated))

        self._configure_finetuning()

    def _set_requires_grad(self, module: nn.Module, flag: bool) -> None:
        for p in module.parameters():
            p.requires_grad = flag

    def _set_ln_only_trainable(self, module: nn.Module) -> None:
        for name, p in module.named_parameters():
            lname = name.lower()
            p.requires_grad = ("layer_norm" in lname or "layernorm" in lname or "ln" in lname)

    def _unfreeze_last_n_blocks(self, blocks: nn.ModuleList, n: int) -> None:
        n = max(0, int(n))
        if n <= 0:
            return
        for block in list(blocks)[-n:]:
            self._set_requires_grad(block, True)

    def _freeze_embeddings(self) -> None:
        if self.cfg.freeze_vision_embeddings:
            emb = getattr(self.clip.vision_model, "embeddings", None)
            if emb is not None:
                self._set_requires_grad(emb, False)
        if self.cfg.freeze_text_embeddings:
            emb = getattr(self.clip.text_model, "embeddings", None)
            if emb is not None:
                self._set_requires_grad(emb, False)

    def _configure_finetuning(self) -> None:
        mode = str(self.cfg.finetune_mode).lower()

        if self.cfg.train_projection_only:
            mode = "projection_only"
        elif self.cfg.freeze_vision and self.cfg.freeze_text:
            mode = "frozen"

        for p in self.clip.parameters():
            p.requires_grad = False

        if mode == "full":
            for p in self.clip.parameters():
                p.requires_grad = True

        elif mode == "frozen":
            pass

        elif mode == "projection_only":
            if self.cfg.unfreeze_visual_projection:
                self._set_requires_grad(self.clip.visual_projection, True)
            if self.cfg.unfreeze_text_projection:
                self._set_requires_grad(self.clip.text_projection, True)
            if self.cfg.unfreeze_logit_scale:
                self.clip.logit_scale.requires_grad = True

        elif mode == "last_n":
            vision_encoder = getattr(self.clip.vision_model, "encoder", None)
            text_encoder = getattr(self.clip.text_model, "encoder", None)

            if vision_encoder is not None and hasattr(vision_encoder, "layers"):
                self._unfreeze_last_n_blocks(vision_encoder.layers, self.cfg.vision_unfreeze_last_n)
            if text_encoder is not None and hasattr(text_encoder, "layers"):
                self._unfreeze_last_n_blocks(text_encoder.layers, self.cfg.text_unfreeze_last_n)

            post_ln = getattr(self.clip.vision_model, "post_layernorm", None)
            if post_ln is not None:
                self._set_requires_grad(post_ln, True)

            final_ln = getattr(self.clip.text_model, "final_layer_norm", None)
            if final_ln is not None:
                self._set_requires_grad(final_ln, True)

            if self.cfg.unfreeze_visual_projection:
                self._set_requires_grad(self.clip.visual_projection, True)
            if self.cfg.unfreeze_text_projection:
                self._set_requires_grad(self.clip.text_projection, True)
            if self.cfg.unfreeze_logit_scale:
                self.clip.logit_scale.requires_grad = True

            self._freeze_embeddings()

        elif mode == "ln_only":
            self._set_ln_only_trainable(self.clip.vision_model)
            self._set_ln_only_trainable(self.clip.text_model)
            if self.cfg.unfreeze_visual_projection:
                self._set_requires_grad(self.clip.visual_projection, True)
            if self.cfg.unfreeze_text_projection:
                self._set_requires_grad(self.clip.text_projection, True)
            if self.cfg.unfreeze_logit_scale:
                self.clip.logit_scale.requires_grad = True

        else:
            raise ValueError(f"Unsupported finetune_mode: {self.cfg.finetune_mode}")

        if self.cfg.freeze_vision:
            self._set_requires_grad(self.clip.vision_model, False)
            self._set_requires_grad(self.clip.visual_projection, False)
        if self.cfg.freeze_text:
            self._set_requires_grad(self.clip.text_model, False)
            self._set_requires_grad(self.clip.text_projection, False)

    def _tokens_to_map(self, patch_tokens: torch.Tensor, image_h: int, image_w: int) -> torch.Tensor:
        b, hw, c = patch_tokens.shape
        h = image_h // self.patch_size
        w = image_w // self.patch_size
        if h * w != hw:
            side = int(hw ** 0.5)
            h = side
            w = side
        return patch_tokens.transpose(1, 2).reshape(b, c, h, w)

    def encode_image(self, pixel_values: torch.Tensor) -> CLIPImageFeatures:
        outputs = self.clip.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            interpolate_pos_encoding=True,
        )
        hidden_states = outputs.hidden_states

        selected: List[torch.Tensor] = []
        for layer_idx in self.selected_hidden_states:
            state = hidden_states[layer_idx]
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
        inputs = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        text_features = self.clip.get_text_features(**inputs)
        return F.normalize(text_features, dim=-1)