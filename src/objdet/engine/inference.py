"""Single-image / batch inference: model outputs -> clean detections.

This is the post-processing layer. Each detector head produces raw outputs in its
own form (R-CNN: per-class scores + deltas over proposals; YOLO: grid cells;
DETR-family: a fixed set of query predictions). This module turns any of those
into a uniform list of (boxes_xyxy, scores, labels) ready to draw or score.
"""

from __future__ import annotations

from typing import Any

import torch


@torch.no_grad()
def detect(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    score_thresh: float = 0.5,
    nms_thresh: float = 0.5,
    max_detections: int = 100,
) -> dict[str, torch.Tensor]:
    """Run a single image through the model and return final detections.

    Args:
        image: [3, H, W] preprocessed (resized + normalized) image tensor.

    Returns:
        {"boxes": [K,4] xyxy, "scores": [K], "labels": [K]} after thresholding + NMS.

    TODO:
        1. model.eval(); add batch dim; move to device.
        2. raw = model(image[None]).
        3. Decode raw -> absolute xyxy boxes + class scores. This step is
           model-specific; most models should expose a ``postprocess(raw, image_sizes)``
           method so this function stays generic. For R-CNN/YOLO apply:
              - score threshold,
              - per-class NMS (torchvision.ops.batched_nms),
              - keep top ``max_detections``.
           DETR-family: no NMS — take top-k over the set predictions by score.
        4. Return the dict (moved to CPU).
    """
    raise NotImplementedError("Implement detect")


def postprocess_detections(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    score_thresh: float,
    nms_thresh: float,
    max_detections: int,
) -> dict[str, torch.Tensor]:
    """Shared filter: score threshold -> per-class NMS -> top-k.

    Reused by R-CNN and YOLO inference. TODO:
        1. Keep boxes with score >= score_thresh.
        2. torchvision.ops.batched_nms(boxes, scores, labels, nms_thresh).
        3. Sort by score, keep top ``max_detections``.
    """
    raise NotImplementedError("Implement postprocess_detections")
