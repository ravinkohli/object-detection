"""Fast R-CNN (Girshick, 2015).

The middle step of the R-CNN family. Key ideas vs the original R-CNN:
  - run the CNN over the WHOLE image once (not once per proposal),
  - project each external proposal onto the feature map and ROI-pool it to a fixed
    size, then classify + regress with shared FC layers,
  - a single multi-task loss (softmax classification + smooth-L1 box regression).
Proposals are EXTERNAL (Selective Search, see data/proposals.py) — Fast R-CNN does
not generate them. That's the limitation Faster R-CNN fixes.

The ROI head here is reused by Faster R-CNN (which only swaps the proposal source).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ROIHead(nn.Module):
    """ROI pooling + FC trunk + (classification head, box-regression head).

    Reused by both Fast R-CNN and Faster R-CNN.
    """

    def __init__(self, in_channels: int, num_classes: int, roi_output_size: int = 7, representation_dim: int = 4096) -> None:
        super().__init__()
        # TODO:
        #   - self.roi_output_size = roi_output_size (e.g. 7x7).
        #   - FC trunk: Linear(in_channels*7*7, 4096) -> ReLU -> Linear(4096,4096) -> ReLU.
        #   - cls head:  Linear(4096, num_classes + 1)        # +1 background class.
        #   - box head:  Linear(4096, num_classes * 4)        # per-class box deltas.
        raise NotImplementedError("Implement ROIHead.__init__")

    def forward(self, features: torch.Tensor, proposals: list[torch.Tensor], image_shapes):
        """
        Args:
            features:  [B, C, H', W'] backbone feature map.
            proposals: list of B tensors [P_b, 4] xyxy (in input-image coords).
        Returns:
            class_logits: [sum(P_b), num_classes+1],
            box_deltas:   [sum(P_b), num_classes*4].

        TODO:
            1. ROI-pool: torchvision.ops.roi_align(features, proposals, output_size,
               spatial_scale=1/stride, sampling_ratio=2)  -> [sumP, C, 7, 7].
               (roi_align wants a list of [P,4] boxes per image — that's ``proposals``.)
            2. Flatten -> FC trunk -> cls head + box head. Return both.
        """
        raise NotImplementedError("Implement ROIHead.forward")


class FastRCNN(nn.Module):
    """Backbone + ROIHead consuming precomputed Selective Search proposals."""

    def __init__(self, backbone: nn.Module, num_classes: int, cfg: dict | None = None) -> None:
        super().__init__()
        # TODO: store backbone + ROIHead(backbone.out_channels, num_classes); stash
        #       the matcher/sampler thresholds and box-coder weights from cfg.
        raise NotImplementedError("Implement FastRCNN.__init__")

    def forward(self, images, targets=None, proposals=None):
        """
        Train  (targets given): return loss_dict {"loss_cls","loss_box"}.
        Eval   (no targets):    return list of detections per image.

        ``proposals`` are the external Selective Search boxes (required).

        TODO (train):
            1. features = backbone(images).
            2. For each image: match proposals to GT (ops.matcher.match using IoU),
               sample a balanced minibatch (ops.matcher.sample_minibatch),
               compute regression targets via ops.boxes.encode_boxes.
            3. class_logits, box_deltas = roi_head(features, sampled_proposals).
            4. loss via losses.detection_loss.fast_rcnn_loss(...). Return loss_dict.

        TODO (eval):
            1. features = backbone(images); class_logits, box_deltas = roi_head(...).
            2. Decode per-class deltas (ops.boxes.decode_boxes), softmax scores,
               then engine.inference.postprocess_detections (score thresh + NMS).
        """
        raise NotImplementedError("Implement FastRCNN.forward")
