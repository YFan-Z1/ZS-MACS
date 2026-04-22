from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import QueryDecoderConfig
from .modules import MLP, SinePositionEmbedding2D


class MaskedCrossAttention(nn.Module):
    def __init__(self, hidden_dim: int, nheads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, nheads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        query_pos: torch.Tensor,
        pos: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = query + query_pos
        k = memory + pos
        out, _ = self.attn(q, k, memory, attn_mask=attn_mask)
        query = self.norm(query + self.dropout(out))
        return query


class SelfAttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int, nheads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, nheads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, query_pos: torch.Tensor) -> torch.Tensor:
        q = k = query + query_pos
        out, _ = self.attn(q, k, query)
        query = self.norm(query + self.dropout(out))
        return query


class FFNLayer(nn.Module):
    def __init__(self, hidden_dim: int, dim_feedforward: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear2(self.dropout(F.relu(self.linear1(x), inplace=True)))
        x = self.norm(x + self.dropout(out))
        return x


class Mask2FormerStyleQueryDecoder(nn.Module):
    """
    Lightweight Mask2Former-style query decoder.

    It keeps the core design motifs that matter for a clean baseline:
      - learnable instance queries
      - iterative self-attn / masked cross-attn / FFN updates
      - per-layer mask prediction for deep supervision
      - attention masks derived from current query masks

    This is intentionally simpler than the full Detectron2/MMDetection implementation,
    but the interface is close enough for later upgrades.
    """

    def __init__(self, cfg: QueryDecoderConfig, mask_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = cfg.hidden_dim
        self.num_queries = cfg.num_queries
        self.num_layers = cfg.num_layers
        self.num_feature_levels = cfg.num_feature_levels
        self.mask_threshold = cfg.mask_threshold
        self.nheads = cfg.nheads

        self.query_feat = nn.Embedding(cfg.num_queries, cfg.hidden_dim)
        self.query_pos = nn.Embedding(cfg.num_queries, cfg.hidden_dim)
        self.level_embed = nn.Embedding(cfg.num_feature_levels, cfg.hidden_dim)
        self.input_proj = nn.ModuleList([nn.Conv2d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=1) for _ in range(cfg.num_feature_levels)])

        self.position_embedding = SinePositionEmbedding2D(cfg.hidden_dim // 2)

        self.self_attn_layers = nn.ModuleList([
            SelfAttentionLayer(cfg.hidden_dim, cfg.nheads, cfg.dropout) for _ in range(cfg.num_layers)
        ])
        self.cross_attn_layers = nn.ModuleList([
            MaskedCrossAttention(cfg.hidden_dim, cfg.nheads, cfg.dropout) for _ in range(cfg.num_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            FFNLayer(cfg.hidden_dim, cfg.dim_feedforward, cfg.dropout) for _ in range(cfg.num_layers)
        ])
        self.mask_embed_head = MLP(cfg.hidden_dim, cfg.hidden_dim, mask_dim, 3)

    def _prepare_level(self, feat: torch.Tensor, level: int) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        feat = self.input_proj[level](feat) + self.level_embed.weight[level][None, :, None, None]
        pos = self.position_embedding(feat)
        b, c, h, w = feat.shape
        memory = feat.flatten(2).transpose(1, 2)
        pos_flat = pos.flatten(2).transpose(1, 2)
        return memory, pos_flat, (h, w)

    def _predict_masks(self, query: torch.Tensor, mask_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_embed = self.mask_embed_head(query)
        pred_masks = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        return pred_masks, mask_embed

    def _make_attn_mask(self, pred_masks: torch.Tensor, spatial_shape: Tuple[int, int]) -> torch.Tensor:
        b, q, _, _ = pred_masks.shape
        resized = F.interpolate(pred_masks, size=spatial_shape, mode="bilinear", align_corners=False)
        # [B, Q, H, W] -> [B, Q, HW]
        mask = (resized.sigmoid() < self.mask_threshold).flatten(2)  # [B, 1, Q, HW]
        # MultiheadAttention 的 3D attn_mask 要求 [B * num_heads, Q, HW]
        mask = (
            mask.unsqueeze(1).expand(b, self.nheads, q, mask.shape[-1])  # [B, heads, Q, HW]
            .reshape(b * self.nheads, q, mask.shape[-1])  # [B*heads, Q, HW]
        )
        all_true = mask.all(dim=-1, keepdim=True)
        if all_true.any():
            mask = mask.masked_fill(all_true, False)

        return mask.contiguous()

    def forward(self, multi_scale_memory: List[torch.Tensor], mask_features: torch.Tensor) -> Dict[str, torch.Tensor | List[Dict[str, torch.Tensor]]]:
        assert len(multi_scale_memory) == self.num_feature_levels, (
            f"Expected {self.num_feature_levels} feature levels, got {len(multi_scale_memory)}"
        )

        b = multi_scale_memory[0].shape[0]
        query = self.query_feat.weight.unsqueeze(0).repeat(b, 1, 1)
        query_pos = self.query_pos.weight.unsqueeze(0).repeat(b, 1, 1)

        aux_outputs: List[Dict[str, torch.Tensor]] = []

        pred_masks, _ = self._predict_masks(query, mask_features)
        for layer_idx in range(self.num_layers):
            level_idx = layer_idx % self.num_feature_levels
            memory, pos, spatial_shape = self._prepare_level(multi_scale_memory[level_idx], level_idx)
            attn_mask = self._make_attn_mask(pred_masks.detach(), spatial_shape)

            query = self.self_attn_layers[layer_idx](query, query_pos)
            query = self.cross_attn_layers[layer_idx](query, memory, query_pos, pos, attn_mask=attn_mask)
            query = self.ffn_layers[layer_idx](query)
            pred_masks, _ = self._predict_masks(query, mask_features)

            if self.cfg.deep_supervision and layer_idx < self.num_layers - 1:
                aux_outputs.append({"pred_masks": pred_masks, "query_states": query})

        return {
            "query_states": query,
            "pred_masks": pred_masks,
            "aux_outputs": aux_outputs,
        }