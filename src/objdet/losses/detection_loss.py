"""Multi-task losses for the R-CNN family (Fast R-CNN §2.3, RPN in Faster R-CNN §3.1.2).

Both the RPN and the ROI head use the same recipe:
    L = L_classification + lambda * L_box_regression
  - classification: cross-entropy (RPN: object-vs-bg binary; head: C+1-way softmax).
  - box regression: smooth-L1 (a.k.a. Huber), applied ONLY to positive samples, and
    for the head, only to the columns of the predicted box for the GT class.
"""

from __future__ import annotations

import torch


def smooth_l1_loss(pred: torch.Tensor, target: torch.Tensor, beta: float = 1.0, reduction: str = "sum") -> torch.Tensor:
    """Smooth-L1 / Huber loss.

    TODO: you can call torch.nn.functional.smooth_l1_loss(pred, target, beta=beta,
    reduction=reduction). (Writing it by hand once is instructive: 0.5*x^2/beta for
    |x|<beta else |x|-0.5*beta.)
    """
    raise NotImplementedError("Implement smooth_l1_loss")


def fast_rcnn_loss(
    class_logits: torch.Tensor,   # [N, C+1]
    box_deltas: torch.Tensor,     # [N, C*4]  (per-class boxes)
    labels: torch.Tensor,         # [N] in [0..C], 0 = background
    regression_targets: torch.Tensor,  # [N, 4] encoded deltas for the matched GT
) -> dict[str, torch.Tensor]:
    """ROI-head loss (Fast R-CNN). Returns {"loss_cls","loss_box"}.

    TODO:
        1. loss_cls = F.cross_entropy(class_logits, labels).
        2. Positive mask = labels > 0. For each positive, select the 4 columns of
           box_deltas corresponding to its GT class (labels[i]).
        3. loss_box = smooth_l1_loss(pos_pred_deltas, regression_targets[pos]) / N
           (normalize by number of samples, per the paper).
    """
    raise NotImplementedError("Implement fast_rcnn_loss")


def rpn_loss(
    objectness: torch.Tensor,        # [num_sampled] logits
    pred_deltas: torch.Tensor,       # [num_sampled, 4]
    gt_labels: torch.Tensor,         # [num_sampled] 1=object, 0=background
    regression_targets: torch.Tensor,  # [num_sampled, 4]
) -> dict[str, torch.Tensor]:
    """RPN loss. Returns {"loss_rpn_obj","loss_rpn_box"}.

    TODO:
        1. loss_obj = F.binary_cross_entropy_with_logits(objectness, gt_labels.float()).
        2. loss_box = smooth_l1_loss over POSITIVE anchors only, normalized
           (paper normalizes by num anchors locations; dividing by #samples is fine).
    """
    raise NotImplementedError("Implement rpn_loss")
