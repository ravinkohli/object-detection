"""Faster R-CNN (Ren et al., 2015) = backbone + RPN + the Fast R-CNN ROIHead.

The punchline of the whole two-stage story: Faster R-CNN is *literally* Fast R-CNN
with the Selective Search proposals replaced by the learned RPN. We reuse
``models.fast_rcnn.ROIHead`` unchanged — the only new piece is models.rpn.RPN.

Total loss = RPN loss (obj + box) + ROI head loss (cls + box).
"""

from __future__ import annotations

import torch.nn as nn

from .fast_rcnn import ROIHead
from .rpn import RPN


class FasterRCNN(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, cfg: dict | None = None) -> None:
        super().__init__()
        # TODO:
        #   - self.backbone = backbone.
        #   - self.rpn = RPN(backbone.out_channels, cfg).
        #   - self.roi_head = ROIHead(backbone.out_channels, num_classes, ...).
        #   - stash matcher/sampler/box-coder settings for the 2nd-stage head.
        raise NotImplementedError("Implement FasterRCNN.__init__")

    def forward(self, images, targets=None):
        """
        Train: return {"loss_rpn_obj","loss_rpn_box","loss_cls","loss_box"}.
        Eval:  return list of per-image detections.

        TODO:
            1. features = self.backbone(images).
            2. proposals, rpn_losses = self.rpn(images, features, targets).
            3. (train) match proposals to GT, sample, encode targets — exactly the
               2nd-stage logic from FastRCNN.forward, but proposals come from the RPN.
            4. class_logits, box_deltas = self.roi_head(features, proposals, image_shapes).
            5. (train) head_losses = detection_loss.fast_rcnn_loss(...).
               return {**rpn_losses, **head_losses}.
            6. (eval) decode + postprocess (score thresh + per-class NMS) -> detections.

        Note: detaching proposals before the head (so RPN grads don't flow through the
        2nd stage) is the simple "approximate joint training" — fine for learning.
        """
        raise NotImplementedError("Implement FasterRCNN.forward")
