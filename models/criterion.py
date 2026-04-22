from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LossConfig
from .matcher import HungarianMatcher
from .modules import dice_loss, sigmoid_ce_loss


class VAWSegCriterion(nn.Module):
    def __init__(self, num_object_classes: int, matcher: HungarianMatcher, cfg: LossConfig) -> None:
        super().__init__()
        self.num_object_classes = num_object_classes
        self.matcher = matcher
        self.cfg = cfg
        empty_weight = torch.ones(num_object_classes + 1)
        empty_weight[-1] = cfg.no_object_weight
        self.register_buffer("empty_weight", empty_weight)

    def _get_num_masks(self, targets: List[Dict[str, torch.Tensor]]) -> float:
        num_masks = sum(len(t["labels_obj"]) for t in targets)
        return float(max(num_masks, 1))

    def _get_src_permutation_idx(self, indices: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @staticmethod
    def _resize_target_masks(tgt_masks: torch.Tensor, spatial_shape: Tuple[int, int]) -> torch.Tensor:
        if tgt_masks.numel() == 0:
            return tgt_masks.float()
        return F.interpolate(
            tgt_masks.unsqueeze(1).float(),
            size=spatial_shape,
            mode="nearest",
        ).squeeze(1)

    def _loss_objects(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]], indices):
        src_logits = outputs["pred_object_logits"]
        bs, num_queries, _ = src_logits.shape
        target_classes = torch.full(
            (bs, num_queries),
            fill_value=self.num_object_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                target_classes[b, src_idx] = targets[b]["labels_obj"][tgt_idx].to(src_logits.device)

        loss = F.cross_entropy(src_logits.transpose(1, 2), target_classes, weight=self.empty_weight)
        return {"loss_object": loss * self.cfg.loss_object_weight}

    def _loss_masks(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]], indices, num_masks: float):
        src_idx = self._get_src_permutation_idx(indices)
        pred_masks = outputs["pred_masks"][src_idx]
        if pred_masks.numel() == 0:
            zero = outputs["pred_masks"].sum() * 0.0
            return {"loss_mask": zero, "loss_dice": zero}

        spatial_shape = tuple(pred_masks.shape[-2:])
        tgt_masks = []
        for b, (_, tgt_idx) in enumerate(indices):
            if len(tgt_idx) > 0:
                tgt = targets[b]["masks"][tgt_idx].to(pred_masks.device)
                tgt = self._resize_target_masks(tgt, spatial_shape)
                tgt_masks.append(tgt)
        tgt_masks = torch.cat(tgt_masks, dim=0).float()

        pred_masks = pred_masks.flatten(1)
        tgt_masks = tgt_masks.flatten(1)

        return {
            "loss_mask": sigmoid_ce_loss(pred_masks, tgt_masks, num_masks) * self.cfg.loss_mask_weight,
            "loss_dice": dice_loss(pred_masks, tgt_masks, num_masks) * self.cfg.loss_dice_weight,
        }

    def _loss_attributes(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]], indices):
        pred_attrs = outputs["pred_attr_logits"]
        collected_logits = []
        collected_targets = []
        collected_known = []
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
            collected_logits.append(pred_attrs[b, src_idx])
            collected_targets.append(targets[b]["labels_attr_pos"][tgt_idx].to(pred_attrs.device))
            collected_known.append(targets[b]["attr_is_labeled"][tgt_idx].to(pred_attrs.device))

        if len(collected_logits) == 0:
            zero = outputs["pred_attr_logits"].sum() * 0.0
            return {"loss_attr": zero}

        logits = torch.cat(collected_logits, dim=0)
        pos_targets = torch.cat(collected_targets, dim=0)
        known = torch.cat(collected_known, dim=0)

        bce = F.binary_cross_entropy_with_logits(logits, pos_targets, reduction="none")
        loss = (bce * known).sum() / known.sum().clamp(min=1.0)
        return {"loss_attr": loss * self.cfg.loss_attr_weight}

    def _compute_single(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]], prefix: str = "") -> Dict[str, torch.Tensor]:
        indices = self.matcher(outputs, targets)
        num_masks = self._get_num_masks(targets)
        losses = {}
        losses.update({prefix + k: v for k, v in self._loss_objects(outputs, targets, indices).items()})
        losses.update({prefix + k: v for k, v in self._loss_masks(outputs, targets, indices, num_masks).items()})
        losses.update({prefix + k: v for k, v in self._loss_attributes(outputs, targets, indices).items()})
        return losses

    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        losses = self._compute_single(outputs, targets)
        if self.cfg.aux_loss and "aux_outputs" in outputs:
            for i, aux in enumerate(outputs["aux_outputs"]):
                aux_out = {
                    "pred_object_logits": aux["pred_object_logits"],
                    "pred_attr_logits": aux["pred_attr_logits"],
                    "pred_masks": aux["pred_masks"],
                }
                losses.update(self._compute_single(aux_out, targets, prefix=f"aux_{i}_"))
        return losses