"""Bounding-box math — the foundation everything else builds on.

Conventions used throughout this repo:
  - "xyxy" = (x1, y1, x2, y2)  absolute pixel corners.
  - "xywh" = (x, y, w, h)      top-left + size (COCO annotation / result format).
  - "cxcywh" = (cx, cy, w, h)  center + size (DETR predicts this, normalized to [0,1]).

For the primitive numerical ops we deliberately lean on torchvision.ops (your
chosen "lean/pragmatic" boundary). The part that's genuinely detector-specific —
the box delta *parametrization* used to regress boxes — you implement here so you
understand what the network is actually predicting.

torchvision.ops you should wrap/use:
  nms, batched_nms, box_iou, generalized_box_iou, box_convert, clip_boxes_to_image, box_area
"""

from __future__ import annotations

import torch


# --- thin wrappers over torchvision.ops (so the rest of the code imports from here) ---

def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """IoU matrix [N, M] between two sets of xyxy boxes.

    TODO: return ``torchvision.ops.box_iou(boxes1, boxes2)``.
    (Implementing IoU by hand once is a great exercise — area of intersection over
    union — but the wrapper is what the rest of the code calls.)
    """
    raise NotImplementedError("Wrap torchvision.ops.box_iou")


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float) -> torch.Tensor:
    """Non-max suppression -> indices to keep. TODO: torchvision.ops.nms."""
    raise NotImplementedError("Wrap torchvision.ops.nms")


def batched_nms(boxes, scores, idxs, iou_thresh: float) -> torch.Tensor:
    """Class-aware NMS (boxes of different classes never suppress each other).

    TODO: torchvision.ops.batched_nms.
    """
    raise NotImplementedError("Wrap torchvision.ops.batched_nms")


def convert(boxes: torch.Tensor, in_fmt: str, out_fmt: str) -> torch.Tensor:
    """Format conversion among xyxy / xywh / cxcywh.

    TODO: torchvision.ops.box_convert(boxes, in_fmt, out_fmt).
    """
    raise NotImplementedError("Wrap torchvision.ops.box_convert")


def clip_to_image(boxes: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Clamp xyxy boxes to image bounds. ``size`` is (H, W).

    TODO: torchvision.ops.clip_boxes_to_image(boxes, size).
    """
    raise NotImplementedError("Wrap torchvision.ops.clip_boxes_to_image")


# --- the detector-specific part: box delta encode / decode -------------------
# This is the parametrization from R-CNN / Faster R-CNN §3.1.2. The network never
# regresses raw coordinates; it regresses *deltas* relative to a reference box
# (an anchor or a proposal):
#       tx = (gx - ax) / aw      ty = (gy - ay) / ah
#       tw = log(gw / aw)        th = log(gh / ah)
# where (ax,ay,aw,ah) is the anchor center+size and (gx,gy,gw,gh) the target.

def encode_boxes(
    reference: torch.Tensor,
    target: torch.Tensor,
    weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> torch.Tensor:
    """Encode target boxes as deltas relative to reference (anchor/proposal) boxes.

    Args:
        reference: [N, 4] xyxy anchors/proposals.
        target:    [N, 4] xyxy ground-truth boxes (already matched 1:1 to reference).
        weights:   (wx, wy, ww, wh) variance-normalization weights (Faster R-CNN
                   uses (1,1,1,1); some impls use (10,10,5,5)).

    Returns:
        [N, 4] deltas (tx, ty, tw, th).

    TODO:
        1. Convert both to center+size (cx, cy, w, h).
        2. Apply the four formulas above.
        3. Multiply by ``weights``. Return stacked [N,4].
    """
    raise NotImplementedError("Implement encode_boxes")


def decode_boxes(
    deltas: torch.Tensor,
    reference: torch.Tensor,
    weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> torch.Tensor:
    """Inverse of ``encode_boxes``: deltas + reference -> absolute xyxy boxes.

    Args:
        deltas:    [N, 4] predicted (tx, ty, tw, th).
        reference: [N, 4] xyxy anchors/proposals the deltas are relative to.

    Returns:
        [N, 4] xyxy predicted boxes.

    TODO:
        1. Divide deltas by ``weights``.
        2. Convert reference to center+size.
        3. gx = tx*aw + ax ; gw = exp(tw)*aw ; (same for y/h).
           Clamp tw/th before exp() to avoid overflow (e.g. clamp to log(1000/16)).
        4. Convert center+size back to xyxy. Return [N,4].
    """
    raise NotImplementedError("Implement decode_boxes")
