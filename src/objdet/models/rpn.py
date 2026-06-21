"""Region Proposal Network (Faster R-CNN, Ren et al. 2015, §3.1).

The RPN is Faster R-CNN's key innovation: a small conv network that slides over the
backbone feature map and, at each of the A anchors per location, predicts
  - an objectness score (object vs background), and
  - a box delta refining the anchor.
It replaces Selective Search with a learned, GPU-friendly, end-to-end module.

At train time the RPN has its own loss (objectness BCE + box smooth-L1 over a
sampled anchor minibatch). At inference it emits the top-scoring, NMS'd proposals
that feed the shared ROIHead.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RPNHead(nn.Module):
    """3x3 conv -> two sibling 1x1 convs (objectness, box deltas)."""

    def __init__(self, in_channels: int, num_anchors: int) -> None:
        super().__init__()
        # TODO:
        #   - self.conv = Conv2d(in_channels, in_channels, 3, padding=1) + ReLU.
        #   - self.cls_logits = Conv2d(in_channels, num_anchors, 1)        # objectness.
        #   - self.bbox_pred  = Conv2d(in_channels, num_anchors * 4, 1).
        raise NotImplementedError("Implement RPNHead.__init__")

    def forward(self, feature: torch.Tensor):
        """[B,C,H',W'] -> (objectness [B,A,H',W'], deltas [B,A*4,H',W']). TODO."""
        raise NotImplementedError("Implement RPNHead.forward")


class RPN(nn.Module):
    """Anchor generation + head + proposal filtering + training loss."""

    def __init__(self, in_channels: int, cfg: dict | None = None) -> None:
        super().__init__()
        # TODO: store anchor settings (scales/ratios/stride), build RPNHead, stash
        #       pre/post-NMS top-k, NMS thresh, matcher thresholds, sampler sizes.
        raise NotImplementedError("Implement RPN.__init__")

    def forward(self, images, features: torch.Tensor, targets=None):
        """
        Returns:
            proposals: list of B tensors [post_nms_top_n, 4] xyxy.
            losses:    {"loss_rpn_obj","loss_rpn_box"} in train mode, else {}.

        TODO:
            1. objectness, deltas = head(features).
            2. anchors = ops.anchors.generate_anchors_single_level(feat_size, stride, ...).
            3. Decode deltas -> proposal boxes (ops.boxes.decode_boxes); clip to image;
               remove tiny boxes; take pre-NMS top-k by score; NMS; post-NMS top-k.
            4. (train) match anchors to GT (ops.matcher), sample minibatch, encode
               targets, compute objectness BCE + box smooth-L1. Return proposals+losses.
        """
        raise NotImplementedError("Implement RPN.forward")
