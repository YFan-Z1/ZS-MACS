from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TextHeadConfig
from .modules import MLP


class OpenVocabObjectHead(nn.Module):
    def __init__(self, cfg: TextHeadConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.query_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.background_embed = nn.Parameter(torch.empty(1, cfg.hidden_dim))
        nn.init.normal_(self.background_embed, std=cfg.background_init_std)
        self.logit_scale = nn.Parameter(torch.tensor(float(cfg.logit_scale_init)).log())

    def forward(self, query_states: torch.Tensor, object_text_embeds: torch.Tensor) -> torch.Tensor:
        q = self.query_proj(query_states)
        if self.cfg.normalize_queries:
            q = F.normalize(q, dim=-1)
        text_bank = object_text_embeds
        if self.cfg.normalize_text:
            text_bank = F.normalize(text_bank, dim=-1)
        text_bank = torch.cat([text_bank, F.normalize(self.background_embed, dim=-1)], dim=0)
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = scale * torch.einsum("bqd,kd->bqk", q, text_bank)
        return logits


class ObjectConditionedAttributeHead(nn.Module):
    """
    Attribute prediction is conditioned on the query feature and a soft object context.
    This keeps the baseline factorized while still acknowledging that attribute semantics
    shift with the object category.
    """

    def __init__(self, cfg: TextHeadConfig, object_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.fuse = MLP(cfg.hidden_dim + object_dim, cfg.hidden_dim * cfg.attribute_mlp_ratio, cfg.hidden_dim, 2)
        self.logit_scale = nn.Parameter(torch.tensor(float(cfg.logit_scale_init)).log())

    def forward(
        self,
        query_states: torch.Tensor,
        object_logits: torch.Tensor,
        object_text_embeds: torch.Tensor,
        attribute_text_embeds: torch.Tensor,
        matched_object_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # object_logits includes the appended background class; exclude it when forming the context.
        object_probs = object_logits[..., :-1].softmax(dim=-1)
        soft_context = torch.einsum("bqo,od->bqd", object_probs, object_text_embeds)

        if matched_object_labels is not None:
            teacher = matched_object_labels.clamp(min=0)
            teacher_context = object_text_embeds[teacher]
            use_teacher = (matched_object_labels >= 0).unsqueeze(-1)
            soft_context = torch.where(use_teacher, teacher_context, soft_context)

        fused = self.fuse(torch.cat([query_states, soft_context], dim=-1))
        if self.cfg.normalize_queries:
            fused = F.normalize(fused, dim=-1)
        text_bank = attribute_text_embeds
        if self.cfg.normalize_text:
            text_bank = F.normalize(text_bank, dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = scale * torch.einsum("bqd,ad->bqa", fused, text_bank)
        return logits