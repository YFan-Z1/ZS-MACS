from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .modules import batch_dice_cost, batch_sigmoid_ce_cost


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 2.0, cost_mask: float = 5.0, cost_dice: float = 5.0) -> None:
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

    @staticmethod
    def _resize_target_masks(tgt_masks: torch.Tensor, spatial_shape: Tuple[int, int]) -> torch.Tensor:
        """Resize GT masks to the prediction resolution for matching."""
        if tgt_masks.numel() == 0:
            return tgt_masks.float()
        tgt_masks = F.interpolate(
            tgt_masks.unsqueeze(1).float(),
            size=spatial_shape,
            mode="nearest",
        ).squeeze(1)
        return tgt_masks

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        pred_logits = outputs["pred_object_logits"]  # [B, Q, O+1]
        pred_masks = outputs["pred_masks"]          # [B, Q, H, W]

        bs, num_queries = pred_logits.shape[:2]
        indices: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for b in range(bs):
            tgt_ids = targets[b]["labels_obj"]
            tgt_masks = targets[b]["masks"].float()
            if tgt_ids.numel() == 0:
                indices.append((torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)))
                continue

            out_prob = pred_logits[b].softmax(-1)
            cost_class = -out_prob[:, tgt_ids]  # [Q, T]

            spatial_shape = tuple(pred_masks[b].shape[-2:])
            tgt_masks = self._resize_target_masks(tgt_masks, spatial_shape)

            out_mask = pred_masks[b].flatten(1)  # [Q, HW]
            tgt_mask = tgt_masks.flatten(1)      # [T, HW]
            cost_mask = batch_sigmoid_ce_cost(out_mask, tgt_mask)
            cost_dice = batch_dice_cost(out_mask, tgt_mask)

            cost = self.cost_class * cost_class + self.cost_mask * cost_mask + self.cost_dice * cost_dice
            src_idx, tgt_idx = linear_sum_assignment(cost.cpu())
            indices.append((
                torch.as_tensor(src_idx, dtype=torch.int64),
                torch.as_tensor(tgt_idx, dtype=torch.int64),
            ))
        return indices