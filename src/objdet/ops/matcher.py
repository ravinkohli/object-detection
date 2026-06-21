"""IoU-based label assignment + minibatch sampling for anchors and proposals.

Two jobs, both used by the RPN and the R-CNN head:

1) MATCH: given anchors/proposals and GT boxes, label each as positive (assigned a
   GT to regress toward), negative (background), or ignore.
   Faster R-CNN rule (§3.1.2):
     - IoU >= high_thresh (0.7)  -> positive, assigned to that GT
     - IoU <  low_thresh  (0.3)  -> negative
     - in between                -> ignore (no loss)
     - also force the anchor with the highest IoU for each GT to be positive
       (guarantees every GT has at least one positive).

2) SAMPLE: positives are rare, so train on a fixed-size balanced minibatch
   (e.g. 256 anchors at 1:1 pos:neg for the RPN; 128 at 1:3 for the head).
"""

from __future__ import annotations

import torch


def match(
    iou: torch.Tensor,
    high_thresh: float,
    low_thresh: float,
    allow_low_quality: bool = True,
) -> torch.Tensor:
    """Assign each anchor/proposal a matched GT index or a negative/ignore label.

    Args:
        iou: [num_gt, num_anchors] IoU matrix (rows = GT, cols = anchors).
        high_thresh / low_thresh: the 0.7 / 0.3 thresholds above.
        allow_low_quality: force-match the best anchor per GT (the extra rule).

    Returns:
        matches: [num_anchors] long tensor where
                 >= 0 -> index of matched GT (positive),
                 -1   -> background (negative),
                 -2   -> ignore.

    TODO:
        1. best_gt_iou, best_gt_idx = iou.max(dim=0)   # per anchor, its best GT.
        2. Start matches = best_gt_idx; set < low_thresh -> -1, between -> -2.
        3. If allow_low_quality: for each GT find the anchor(s) with max IoU to it
           and force those to positive (assign that GT). (Be careful with ties.)
    """
    if iou.shape[0] == 0:
        return torch.full((iou.shape[1],), -1, dtype=torch.long, device=iou.device)
    best_gt_iou, best_gt_idx = iou.max(dim=0)
    matches = best_gt_idx.clone()
    matches[best_gt_iou < low_thresh] = -1
    between = (best_gt_iou >= low_thresh) & (best_gt_iou < high_thresh)
    matches[between] = -2
    if allow_low_quality:
        highest_per_gt, _ = iou.max(dim=1)
        gt_idx, anchor_idx = torch.where((iou == highest_per_gt[:, None]) & (highest_per_gt[:, None] > 0)) # ignore when iou==0
        matches[anchor_idx] = gt_idx
    return matches


def sample_minibatch(
    labels: torch.Tensor,
    batch_size: int,
    positive_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subsample positives/negatives to a fixed minibatch.

    Args:
        labels: [N] with >= 0 for positive, -1 for background and -2 for ignore
        batch_size: total samples to keep (e.g. 256 RPN, 128 head).
        positive_fraction: target fraction of positives (e.g. 0.5 RPN, 0.25 head).

    Returns:
        (pos_idx, neg_idx): index tensors of the chosen positives and negatives.
        If positives are scarce, fill the remainder with negatives.

    TODO:
        1. num_pos = min(#positives, round(batch_size * positive_fraction)).
        2. num_neg = batch_size - num_pos (capped at #negatives).
        3. Randomly permute and slice the positive/negative index pools.
    """
    positives = torch.where(labels >= 0)[0]
    negatives = torch.where(labels==-1)[0]
    num_pos = min(positives.shape[0], round(batch_size * positive_fraction))
    num_neg = min(negatives.shape[0], batch_size - num_pos)
    perm_pos = torch.randperm(positives.numel(), device=labels.device)
    perm_neg = torch.randperm(negatives.numel(), device=labels.device)
    return (positives[perm_pos[:num_pos]], negatives[perm_neg[:num_neg]])

