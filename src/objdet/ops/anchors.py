"""Anchor generation for the RPN / Faster R-CNN (paper §3.1.1).

An "anchor" is a reference box of a fixed scale + aspect ratio, tiled densely over
every spatial location of the backbone feature map. The RPN then predicts, for
each anchor, (a) an objectness score and (b) a box delta refining it.

Worked example (single feature map, stride 16):
  - feature map H'xW', each cell maps back to a 16x16 region of the input.
  - at each cell place A anchors (e.g. 3 scales x 3 ratios = 9).
  - total anchors = H' * W' * A.
"""

from __future__ import annotations

import torch


def generate_anchors_single_level(
    feature_size: tuple[int, int],
    stride: int,
    scales: tuple[int, ...] = (128, 256, 512),
    aspect_ratios: tuple[float, ...] = (0.5, 1.0, 2.0),
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Tile anchors over one feature map.

    Args:
        feature_size: (H', W') of the feature map.
        stride: input pixels per feature cell (e.g. 16 for VGG conv5).
        scales: anchor box sizes in *input* pixels (sqrt-area).
        aspect_ratios: width/height ratios.

    Returns:
        [H'*W'*A, 4] anchors in xyxy (input-image pixel coords), A = len(scales)*len(ratios).

    TODO:
        1. Build the A "base" anchors centered at origin: for each (scale, ratio),
           w = scale * sqrt(ratio_w...) — standard derivation:
               h = scale / sqrt(ratio),  w = scale * sqrt(ratio)
           giving xyxy = (-w/2, -h/2, w/2, h/2).
        2. Build a grid of cell centers: x = (0..W'-1)*stride + stride/2, same for y
           (use torch.meshgrid).
        3. Broadcast-add base anchors to every center -> [H'*W'*A, 4].
        4. Return on ``device``. Keep the ordering consistent with how the RPN head
           reshapes its predictions (cells-major, then anchors) — document your choice.
    """
    raise NotImplementedError("Implement generate_anchors_single_level")


def generate_anchors_multi_level(
    feature_sizes: list[tuple[int, int]],
    strides: list[int],
    scales_per_level: list[tuple[int, ...]],
    aspect_ratios: tuple[float, ...] = (0.5, 1.0, 2.0),
    device: torch.device | str = "cpu",
) -> list[torch.Tensor]:
    """Anchors for an FPN-style multi-level backbone (one anchor set per level).

    Not needed for the plain VGG Faster R-CNN (single level) — implement when/if
    you add FPN. TODO: loop generate_anchors_single_level per level.
    """
    raise NotImplementedError("Implement generate_anchors_multi_level (optional, FPN)")
