from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Simple MLP used for mask embeddings / lightweight fusion heads."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x, inplace=True)
        return x


class ConvNormAct(nn.Module):
    """Conv -> optional norm -> optional GELU/ReLU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Optional[int] = None,
        norm: str = "gn",
        act: bool = True,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        bias = norm == "none"
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

        if norm == "gn":
            num_groups = 32 if out_channels % 32 == 0 else 16 if out_channels % 16 == 0 else 8 if out_channels % 8 == 0 else 1
            self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        elif norm == "bn":
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == "none":
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unsupported norm type: {norm}")

        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class SinePositionEmbedding2D(nn.Module):
    """
    DETR/Mask2Former-style sine-cosine 2D positional embedding.
    Input: [B, C, H, W]
    Output: [B, 2*num_pos_feats*2, H, W] if num_pos_feats is per-axis.
    With temperature=10000 and normalize=True by default.
    """

    def __init__(self, num_pos_feats: int = 128, temperature: int = 10000, normalize: bool = True, scale: Optional[float] = None) -> None:
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale if scale is not None else 2 * math.pi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        device = x.device
        not_mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        pos = torch.cat((pos_y, pos_x), dim=-1).permute(0, 3, 1, 2)
        return pos


# -----------------------------
# Mask losses / matcher costs
# -----------------------------

def sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float) -> torch.Tensor:
    """Per-element BCE averaged over masks in DETR-style normalization."""
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean(1).sum() / max(num_masks, 1.0)



def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float) -> torch.Tensor:
    """Soft Dice loss on flattened masks."""
    probs = inputs.sigmoid()
    numerator = 2 * (probs * targets).sum(dim=1)
    denominator = probs.sum(dim=1) + targets.sum(dim=1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / max(num_masks, 1.0)


@torch.no_grad()
def batch_sigmoid_ce_cost(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Pairwise BCE cost between predicted masks and target masks.
    inputs: [Q, HW] logits
    targets: [T, HW] binary masks
    returns: [Q, T]
    """
    pos = F.binary_cross_entropy_with_logits(inputs[:, None, :].expand(-1, targets.shape[0], -1), targets[None].expand(inputs.shape[0], -1, -1), reduction="none")
    return pos.mean(dim=-1)


@torch.no_grad()
def batch_dice_cost(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Pairwise soft dice cost.
    inputs: [Q, HW] logits
    targets: [T, HW] binary masks
    returns: [Q, T]
    """
    probs = inputs.sigmoid()
    numerator = 2 * torch.einsum("qh,th->qt", probs, targets)
    denominator = probs.sum(dim=-1, keepdim=True) + targets.sum(dim=-1)[None, :]
    return 1 - (numerator + 1) / (denominator + 1)