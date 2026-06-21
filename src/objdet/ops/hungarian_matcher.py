"""Hungarian (bipartite) matching for the DETR family (DETR paper §3.1).

Unlike anchors, DETR predicts a *fixed set* of N queries and matches them 1:1 to
the ground-truth objects by solving a minimum-cost assignment. There is no NMS:
the set loss + one-to-one matching forces unique predictions.

The cost of matching prediction i to GT j combines:
    cost = w_class * (-prob_i[class_j])            # classification cost
         + w_l1    * ||box_i - box_j||_1           # L1 on normalized cxcywh boxes
         + w_giou  * (-generalized_iou(box_i, box_j))

Solve with scipy.optimize.linear_sum_assignment (CPU, per image).
"""

from __future__ import annotations

import torch


class HungarianMatcher:
    """Computes the optimal prediction<->GT assignment for a batch."""

    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0) -> None:
        # TODO: store the three cost weights (DETR defaults shown above).
        raise NotImplementedError("Implement HungarianMatcher.__init__")

    @torch.no_grad()
    def forward(self, outputs: dict, targets: list[dict]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return, per image, (pred_indices, gt_indices) of the optimal matching.

        Args:
            outputs: {"pred_logits": [B, N, num_classes+1],
                      "pred_boxes":  [B, N, 4] in normalized cxcywh}.
            targets: list of B dicts, each {"labels": [G], "boxes": [G, 4] cxcywh norm}.

        Returns:
            list of B tuples (row_idx, col_idx): query indices matched to GT indices.

        TODO (per image b):
            1. prob = outputs["pred_logits"][b].softmax(-1)         # [N, C+1]
            2. cost_class = -prob[:, target_labels]                 # [N, G]
            3. cost_bbox  = torch.cdist(pred_boxes, gt_boxes, p=1)  # [N, G]
            4. cost_giou  = -generalized_box_iou(xyxy(pred), xyxy(gt))
               (convert cxcywh->xyxy via ops.boxes.convert first).
            5. C = w_bbox*cost_bbox + w_class*cost_class + w_giou*cost_giou
            6. row, col = scipy.optimize.linear_sum_assignment(C.cpu())
            7. collect as (torch.as_tensor(row), torch.as_tensor(col)).
        """
        raise NotImplementedError("Implement HungarianMatcher.forward")

    __call__ = forward
