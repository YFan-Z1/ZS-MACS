from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import PixelDecoderConfig
from .modules import ConvNormAct


class CLIPFeaturePyramidPixelDecoder(nn.Module):
    """
    Simple FPN-style pixel decoder for CLIP ViT hidden states.

    Input feature order is assumed to be low->high level (early -> late hidden states).
    The decoder projects all maps to a common hidden dimension, fuses them top-down,
    and outputs:
      - multi_scale_memory: feature maps consumed by the query decoder
      - mask_features: high-resolution feature map used for dynamic mask prediction
    """

    def __init__(self, in_channels: List[int], cfg: PixelDecoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.lateral_convs = nn.ModuleList([ConvNormAct(c, cfg.hidden_dim, 1, norm=cfg.conv_norm, act=False) for c in in_channels])
        self.output_convs = nn.ModuleList([ConvNormAct(cfg.hidden_dim, cfg.hidden_dim, 3, norm=cfg.conv_norm, act=True) for _ in in_channels])
        self.mask_proj = ConvNormAct(cfg.hidden_dim, cfg.mask_dim, 3, norm=cfg.conv_norm, act=True)

    def forward(self, features: List[torch.Tensor]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        assert len(features) == len(self.lateral_convs)
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]

        # top-down fusion
        results: List[torch.Tensor] = [None] * len(laterals)  # type: ignore
        prev = None
        for idx in reversed(range(len(laterals))):
            cur = laterals[idx]
            if prev is not None:
                prev_up = F.interpolate(prev, size=cur.shape[-2:], mode="bilinear", align_corners=False)
                cur = cur + prev_up
            cur = self.output_convs[idx](cur)
            results[idx] = cur
            prev = cur

        mask_features = self.mask_proj(results[0])
        return results, mask_features