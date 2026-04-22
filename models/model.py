from __future__ import annotations

from dataclasses import asdict, replace
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .clip_backbone import CLIPOpenVocabBackbone
from .config import ModelConfig
from .criterion import VAWSegCriterion
from .matcher import HungarianMatcher
from .pixel_decoder import CLIPFeaturePyramidPixelDecoder
from .query_decoder import Mask2FormerStyleQueryDecoder
from .text_heads import ObjectConditionedAttributeHead, OpenVocabObjectHead


class VAWOpenVocabSegBaseline(nn.Module):
    """
    CLIP + Mask2Former-style baseline for zero-shot multi-attribute compositional segmentation.

    Design choices:
      - CLIP from `transformers` provides both the vision and text towers.
      - A lightweight FPN-style pixel decoder turns CLIP hidden states into multi-scale features.
      - A query decoder predicts instance masks with iterative masked cross-attention.
      - Object logits are open-vocabulary via dot-product against object text embeddings.
      - Attribute logits are multi-label and object-conditioned.

    This is intentionally leaner than MMDetection/Detectron2 stacks while preserving the main
    architectural ideas that matter for a faithful baseline.
    """

    def __init__(self, cfg: Optional[ModelConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.backbone = CLIPOpenVocabBackbone(self.cfg.backbone)

        in_channels = [self.backbone.vision_hidden_size] * len(self.cfg.backbone.selected_hidden_states)
        self.feature_adapter = nn.ModuleList([
            nn.Conv2d(ch, self.cfg.pixel_decoder.hidden_dim, kernel_size=1) for ch in in_channels
        ])
        self.pixel_decoder = CLIPFeaturePyramidPixelDecoder(
            in_channels=[self.cfg.pixel_decoder.hidden_dim] * len(in_channels),
            cfg=self.cfg.pixel_decoder,
        )
        self.query_decoder = Mask2FormerStyleQueryDecoder(self.cfg.query_decoder, mask_dim=self.cfg.pixel_decoder.mask_dim)

        text_hidden_dim = self.backbone.projection_dim
        self.query_to_text_dim = nn.Linear(self.cfg.query_decoder.hidden_dim, text_hidden_dim)
        self.text_head_cfg = replace(self.cfg.text_head, hidden_dim=text_hidden_dim)
        self.object_head = OpenVocabObjectHead(self.text_head_cfg)
        self.attr_head = ObjectConditionedAttributeHead(self.text_head_cfg, object_dim=text_hidden_dim)

        self.register_buffer("object_text_features", torch.empty(0, text_hidden_dim), persistent=False)
        self.register_buffer("attribute_text_features", torch.empty(0, text_hidden_dim), persistent=False)
        self.object_prompts: List[str] = []
        self.attribute_prompts: List[str] = []

    @property
    def num_objects(self) -> int:
        return int(self.object_text_features.shape[0])

    @property
    def num_attributes(self) -> int:
        return int(self.attribute_text_features.shape[0])

    def set_text_prompts(self, object_prompts: Sequence[str], attribute_prompts: Sequence[str], device: Optional[torch.device] = None) -> None:
        self.object_prompts = list(object_prompts)
        self.attribute_prompts = list(attribute_prompts)
        dev = device or next(self.parameters()).device
        with torch.no_grad():
            obj_text = self.backbone.encode_text(self.object_prompts, device=dev)
            attr_text = self.backbone.encode_text(self.attribute_prompts, device=dev)
        self.object_text_features = obj_text.detach()
        self.attribute_text_features = attr_text.detach()

    def build_criterion(self) -> VAWSegCriterion:
        matcher = HungarianMatcher(
            cost_class=self.cfg.loss.matcher_cost_class,
            cost_mask=self.cfg.loss.matcher_cost_mask,
            cost_dice=self.cfg.loss.matcher_cost_dice,
        )
        return VAWSegCriterion(num_object_classes=self.num_objects, matcher=matcher, cfg=self.cfg.loss)

    def _adapt_features(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        return [adapter(feat) for adapter, feat in zip(self.feature_adapter, features)]

    def _decode_core(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor | List[Dict[str, torch.Tensor]]]:
        image_features = self.backbone.encode_image(pixel_values)
        feats = self._adapt_features(image_features.multiscale_features)
        multi_scale_memory, mask_features = self.pixel_decoder(feats)
        dec = self.query_decoder(multi_scale_memory, mask_features)
        query_states = dec["query_states"]
        query_states_text = self.query_to_text_dim(query_states)
        query_states_text = F.normalize(query_states_text, dim=-1)

        object_logits = self.object_head(query_states_text, self.object_text_features)
        attr_logits = self.attr_head(
            query_states=query_states_text,
            object_logits=object_logits,
            object_text_embeds=self.object_text_features,
            attribute_text_embeds=self.attribute_text_features,
        )

        aux_outputs = []
        for aux in dec.get("aux_outputs", []):
            aux_query = self.query_to_text_dim(aux["query_states"])
            aux_query = F.normalize(aux_query, dim=-1)
            aux_obj = self.object_head(aux_query, self.object_text_features)
            aux_attr = self.attr_head(
                query_states=aux_query,
                object_logits=aux_obj,
                object_text_embeds=self.object_text_features,
                attribute_text_embeds=self.attribute_text_features,
            )
            aux_outputs.append({
                "pred_masks": aux["pred_masks"],
                "pred_object_logits": aux_obj,
                "pred_attr_logits": aux_attr,
            })

        return {
            "pred_masks": dec["pred_masks"],
            "pred_object_logits": object_logits,
            "pred_attr_logits": attr_logits,
            "aux_outputs": aux_outputs,
        }

    def forward(self, pixel_values: torch.Tensor, targets: Optional[List[Dict[str, torch.Tensor]]] = None):
        if self.num_objects == 0 or self.num_attributes == 0:
            raise RuntimeError("Text prompts are not initialized. Call `set_text_prompts(...)` before forward().")

        outputs = self._decode_core(pixel_values)
        if targets is None:
            return outputs

        criterion = self.build_criterion().to(pixel_values.device)
        losses = criterion(outputs, targets)
        return outputs, losses

    @torch.no_grad()
    def predict_instances(
        self,
        pixel_values: torch.Tensor,
        mask_threshold: float = 0.5,
        object_threshold: float = 0.2,
        topk_attributes: int = 5,
        attr_threshold: float = 0.5,
    ) -> List[Dict[str, torch.Tensor | List[int]]]:
        self.eval()
        outputs = self._decode_core(pixel_values)
        pred_masks = outputs["pred_masks"].sigmoid()
        pred_obj = outputs["pred_object_logits"].softmax(-1)
        pred_attr = outputs["pred_attr_logits"].sigmoid()

        results = []
        for b in range(pred_masks.shape[0]):
            object_probs, object_labels = pred_obj[b, :, :-1].max(dim=-1)
            keep = object_probs >= object_threshold
            masks = pred_masks[b, keep]
            masks_bin = masks >= mask_threshold
            attr_scores = pred_attr[b, keep]
            attr_topk = attr_scores.topk(k=min(topk_attributes, attr_scores.shape[-1]), dim=-1)
            attr_keep = attr_scores >= attr_threshold
            results.append({
                "scores": object_probs[keep],
                "labels_obj": object_labels[keep],
                "masks": masks_bin,
                "attr_scores": attr_scores,
                "attr_topk_indices": attr_topk.indices,
                "attr_topk_scores": attr_topk.values,
                "attr_binary": attr_keep,
            })
        return results


def build_model_from_dataset(dataset, cfg: Optional[ModelConfig] = None) -> VAWOpenVocabSegBaseline:
    model = VAWOpenVocabSegBaseline(cfg=cfg)
    object_prompts = dataset.build_object_prompts()
    attribute_prompts = dataset.build_attribute_prompts()
    model.set_text_prompts(object_prompts, attribute_prompts)
    return model