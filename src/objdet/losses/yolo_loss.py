"""YOLOv1 loss (Redmon et al., 2016, eq. 3).

A single sum-of-squared-errors loss over the SxSx(B*5+C) grid, with care taken so
that the many "no object" cells don't overwhelm the few that contain objects:

    L = lambda_coord * SUM_{obj boxes} [ (x-x̂)^2 + (y-ŷ)^2
                                         + (sqrt(w)-sqrt(ŵ))^2 + (sqrt(h)-sqrt(ĥ))^2 ]
      +              SUM_{obj boxes}   (C - Ĉ)^2          # confidence, responsible box
      + lambda_noobj SUM_{noobj boxes}(C - Ĉ)^2          # confidence, others
      +              SUM_{obj cells}   SUM_c (p_c - p̂_c)^2 # class probs

Key subtleties:
  - "responsible" box: of the B boxes in a cell that contains an object, only the one
    with the highest IoU to the GT is responsible for the coord + obj-confidence loss.
  - sqrt on (w,h) so errors on small boxes matter more.
  - lambda_coord = 5, lambda_noobj = 0.5 (paper defaults).
"""

from __future__ import annotations

import torch


def yolo_loss(
    pred: torch.Tensor,     # [B, S, S, B_boxes*5 + C]
    target: torch.Tensor,   # [B, S, S, 5 + C]  (built from GT: one object per cell)
    S: int = 7,
    B: int = 2,
    num_classes: int = 20,
    lambda_coord: float = 5.0,
    lambda_noobj: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Compute the YOLOv1 loss. Returns the total (and optionally the components).

    The ``target`` layout (one of several conventions — match it in YOLOv1.forward):
        [..., 0:4] = box (x,y,w,h) of the cell's object,
        [..., 4]   = 1 if the cell contains an object else 0,
        [..., 5:]  = one-hot class.

    TODO:
        1. obj_mask = target[..., 4] == 1   (cells with an object).
        2. For obj cells: compute IoU of each of the B predicted boxes with the GT box;
           pick the responsible box (argmax IoU).
        3. coord loss: lambda_coord * SSE on (x,y) and (sqrt w, sqrt h) of the
           responsible box only.
        4. obj-confidence loss: SSE between predicted conf of the responsible box and
           (ideally) its IoU with GT (paper) or 1.0 (simpler).
        5. noobj-confidence loss: lambda_noobj * SSE on confidences of all non-
           responsible / no-object boxes toward 0.
        6. class loss: SSE on class probs for obj cells.
        7. Sum, normalize by batch size. Return dict.
    """
    raise NotImplementedError("Implement yolo_loss")
