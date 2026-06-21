"""Set-prediction criterion for the DETR family (DETR §3.1, DINO adds the dn branch).

Given the Hungarian matching (ops.hungarian_matcher) between the N predictions and
the GT objects, the loss is the sum over matched pairs of:
    - classification : cross-entropy (DETR) or focal loss (Deformable DETR / DINO / RT-DETR),
                       with unmatched queries supervised toward the "no object" class.
    - box L1         : L1 on normalized cxcywh coordinates.
    - box GIoU       : 1 - generalized_IoU.
Plus "auxiliary losses": the same loss applied to EVERY decoder layer's output, not
just the last (DETR appendix) — helps the decoder learn faster.

DINO/RT-DETR additionally compute a denoising (dn) loss over the CDN query groups,
using the *known* GT assignment (no Hungarian needed there).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SetCriterion(nn.Module):
    def __init__(
        self,
        num_classes: int,
        matcher,                       # ops.hungarian_matcher.HungarianMatcher
        weight_dict: dict[str, float], # e.g. {"loss_ce":1, "loss_bbox":5, "loss_giou":2}
        eos_coef: float = 0.1,         # down-weight the "no object" class (DETR)
        use_focal: bool = False,       # CE (DETR) vs focal (DINO/RT-DETR)
    ) -> None:
        super().__init__()
        # TODO: store fields; if using CE build the empty-weight vector (eos_coef for
        #       the no-object class, 1 for the rest) and register_buffer it.
        raise NotImplementedError("Implement SetCriterion.__init__")

    def loss_labels(self, outputs, targets, indices) -> dict:
        """Classification loss over matched pairs (+ no-object for the rest).

        TODO: build a [B,N] target-class tensor defaulting to no-object, scatter the
        matched GT classes into the matched query slots, then CE (with empty-weight)
        or sigmoid-focal loss.
        """
        raise NotImplementedError("Implement loss_labels")

    def loss_boxes(self, outputs, targets, indices, num_boxes) -> dict:
        """L1 + GIoU over matched boxes, normalized by num_boxes.

        TODO: gather matched pred/GT boxes; F.l1_loss(sum)/num_boxes; and
        (1 - generalized_box_iou(xyxy(pred), xyxy(gt))).diag().sum()/num_boxes.
        """
        raise NotImplementedError("Implement loss_boxes")

    def forward(self, outputs: dict, targets: list[dict]) -> dict[str, torch.Tensor]:
        """Run matcher, compute label+box losses on the final layer and all aux layers.

        TODO:
            1. indices = self.matcher(outputs_without_aux, targets).
            2. num_boxes = total #GT across batch (clamp >=1); for multi-device you'd
               all-reduce — single device just sum.
            3. losses = loss_labels(...) | loss_boxes(...).
            4. For each entry in outputs["aux_outputs"]: re-match + re-loss, suffix keys
               with f"_{i}". (DINO: also add the denoising-group losses here.)
            5. Apply weight_dict and return the dict (engine sums the values).
        """
        raise NotImplementedError("Implement SetCriterion.forward")
