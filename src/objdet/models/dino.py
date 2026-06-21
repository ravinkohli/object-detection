"""DINO — DETR with Improved deNoising anchOr boxes (Zhang et al., 2022).

A much stronger DETR. Builds on Deformable DETR and adds three ideas that make
DETR-style training converge fast and accurately:

  1) Contrastive DeNoising (CDN) training: feed noised versions of GT boxes/labels
     as extra decoder queries and ask the model to reconstruct them. Positive
     queries get small noise, negative queries get larger noise -> the model learns
     to reject duplicates. (Only active in training; removed at inference.)
  2) Mixed query selection: initialize decoder *positional* queries from the
     top-scoring encoder tokens (anchors), while keeping content queries learnable.
  3) Look-forward-twice: each decoder layer refines the box prediction from the
     previous layer (iterative box refinement), with a gradient detail that lets
     later layers correct earlier ones.

This is a big stub — implement incrementally: start as Deformable DETR
(multi-scale features + ops.deformable_attn), then add query selection, then CDN.
Reference: DINO paper §3 (the three contributions map to the three blocks above).
"""

from __future__ import annotations

import torch.nn as nn


class DINO(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, num_queries: int = 300, d_model: int = 256, num_feature_levels: int = 4, cfg: dict | None = None) -> None:
        super().__init__()
        # TODO (build in stages):
        #   STAGE A — Deformable-DETR core:
        #     - multi-scale backbone (ResNetBackbone returning C3,C4,C5) + input_proj
        #       per level (+ one extra stride-2 level), each to d_model.
        #     - deformable encoder/decoder using ops.deformable_attn.MSDeformAttn.
        #     - class_embed + bbox_embed (MLP) heads (per decoder layer for refinement).
        #   STAGE B — Mixed query selection:
        #     - an encoder output head scores tokens; take top-k as initial reference
        #       points / positional queries; content queries stay learnable.
        #   STAGE C — Contrastive denoising (train only):
        #     - build noised GT query groups (prepare_cdn): positive (small noise) +
        #       negative (large noise) labels & boxes; attention mask so denoising and
        #       matching groups don't see each other; add a denoising loss branch in
        #       losses.set_criterion.
        raise NotImplementedError("Implement DINO.__init__ (start from Deformable DETR)")

    def forward(self, images, targets=None):
        """
        Train: returns/loss includes both the matching loss AND the CDN denoising loss.
        Eval:  CDN disabled; standard set-prediction output (top-k, no NMS).

        TODO: see the staged plan in __init__. In train mode, when targets are given,
        construct CDN groups and concatenate them to the object queries with the right
        attention mask; split outputs back into (matching, denoising) before the loss.
        """
        raise NotImplementedError("Implement DINO.forward")

    def postprocess(self, outputs, target_sizes):
        """Same as DETR (top-k over set predictions, no NMS). TODO."""
        raise NotImplementedError("Implement DINO.postprocess")


def prepare_cdn(targets, num_classes, num_denoising_queries, label_noise_ratio, box_noise_scale):
    """Build contrastive denoising query groups from GT (train only).

    TODO: for each image, replicate GT into positive/negative groups; randomly flip
    some labels (label_noise_ratio); jitter boxes (box_noise_scale, positives small /
    negatives large); return the dn queries, their targets, and the attention mask.
    """
    raise NotImplementedError("Implement prepare_cdn (DINO contrastive denoising)")
