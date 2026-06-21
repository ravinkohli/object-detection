"""RT-DETR — Real-Time DEtection TRansformer (Zhao et al., 2023).

The first real-time end-to-end detector that beats YOLO at comparable speed, by
fixing what made DETR-family models slow:

  1) Efficient Hybrid Encoder: instead of running expensive multi-scale attention
     over all levels, it splits the work into
        - AIFI (Attention-based Intra-scale Feature Interaction): self-attention ONLY
          on the highest-level (smallest) feature map S5, where it's cheap and most
          semantic;
        - CCFM (CNN-based Cross-scale Feature Fusion): a lightweight FPN/PAN-style conv
          module to fuse scales — no attention across scales.
  2) IoU-aware Query Selection: pick the initial object queries from encoder tokens
     using a score that's supervised to correlate with IoU, so the decoder starts
     from high-quality priors.
  3) Decoder with deformable attention + iterative box refinement (like Deformable
     DETR / DINO). No NMS at inference.

Reference: RT-DETR paper §4 (Efficient Hybrid Encoder) + §5 (query selection).
Build order: reuse the deformable decoder you wrote for DINO, then add AIFI + CCFM.
"""

from __future__ import annotations

import torch.nn as nn


class AIFI(nn.Module):
    """Attention-based Intra-scale Feature Interaction (self-attn on the top level only).

    TODO: a single transformer encoder layer applied to the flattened S5 feature map
    (+ 2D pos enc), then reshape back to [B,C,H,W].
    """

    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024) -> None:
        super().__init__()
        raise NotImplementedError("Implement AIFI")

    def forward(self, x):
        raise NotImplementedError


class CCFM(nn.Module):
    """CNN-based Cross-scale Feature Fusion (lightweight PAN/FPN, conv-only).

    TODO: top-down + bottom-up conv fusion across {S3,S4,S5} producing fused
    multi-scale features for the decoder. Plain Conv/upsample blocks — no attention.
    """

    def __init__(self, in_channels: list[int], d_model: int = 256) -> None:
        super().__init__()
        raise NotImplementedError("Implement CCFM")

    def forward(self, feats):
        raise NotImplementedError


class RTDETR(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, num_queries: int = 300, d_model: int = 256, cfg: dict | None = None) -> None:
        super().__init__()
        # TODO:
        #   - multi-scale backbone (ResNetBackbone -> C3,C4,C5) + per-level proj to d_model.
        #   - hybrid encoder = AIFI(on S5) then CCFM(fuse C3..C5).
        #   - IoU-aware query selection head over encoder tokens (top-k -> queries +
        #     reference boxes).
        #   - deformable decoder (reuse DINO's) + class/box heads with refinement.
        raise NotImplementedError("Implement RTDETR.__init__")

    def forward(self, images, targets=None):
        """
        Train: Hungarian matching + set loss (losses.set_criterion); optionally CDN
               (RT-DETR also benefits from denoising). Eval: top-k, NO NMS.

        TODO:
            1. feats = backbone(images) -> {C3,C4,C5}; project each to d_model.
            2. S5' = AIFI(C5); fused = CCFM({C3,C4,S5'}).
            3. queries, ref_boxes = iou_aware_query_selection(fused).
            4. hs = deformable_decoder(queries, fused, ref_boxes) with box refinement.
            5. logits/boxes heads -> outputs; (train) criterion(outputs, targets).
        """
        raise NotImplementedError("Implement RTDETR.forward")

    def postprocess(self, outputs, target_sizes):
        """Top-k over set predictions, no NMS (same family as DETR/DINO). TODO."""
        raise NotImplementedError("Implement RTDETR.postprocess")
