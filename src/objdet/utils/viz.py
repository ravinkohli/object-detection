"""Visualization: draw predicted/ground-truth boxes on an image.

Used by ``scripts/predict.py`` and (optionally) to log sample prediction images
to MLflow during training so you can *see* the model improve.
"""

from __future__ import annotations

from typing import Sequence

import torch


def draw_boxes(
    image: torch.Tensor,
    boxes: torch.Tensor,
    labels: Sequence[int] | None = None,
    scores: Sequence[float] | None = None,
    class_names: Sequence[str] | None = None,
) -> torch.Tensor:
    """Draw boxes on an image and return an annotated uint8 image tensor.

    Args:
        image:  [3, H, W] tensor. Either uint8 in [0,255] or float in [0,1]/normalized
                — convert as needed before drawing.
        boxes:  [N, 4] in xyxy (absolute pixel) coords.
        labels: optional length-N class ids.
        scores: optional length-N confidences (annotate "name 0.87").
        class_names: optional id -> name lookup for nicer captions.

    Returns:
        [3, H, W] uint8 tensor with boxes burned in.

    TODO:
        1. Convert ``image`` to uint8 [0,255] (undo normalization / scale).
        2. Build caption strings from labels/scores/class_names.
        3. Use ``torchvision.utils.draw_bounding_boxes`` (simplest), or matplotlib.
        4. Return the annotated tensor (save with torchvision.io.write_png or PIL).
    """
    raise NotImplementedError("Implement draw_boxes")
