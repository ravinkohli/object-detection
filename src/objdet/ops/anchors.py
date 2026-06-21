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
import itertools
from math import sqrt

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
    coords = []
    for scale, aspect_ratio in itertools.product(scales, aspect_ratios):
        w = scale * sqrt(aspect_ratio)
        h = scale / sqrt(aspect_ratio)
        coords.append([-w/2, -h/2, w/2, h/2])
    base = torch.as_tensor(coords, dtype=torch.float32, device=device)
    x_centers = torch.arange(feature_size[1], dtype=torch.float32, device=device)
    y_centers = torch.arange(feature_size[0], dtype=torch.float32, device=device)
    x_centers = x_centers * stride + stride/2
    y_centers = y_centers * stride + stride/2
    grid_y, grid_x = torch.meshgrid(y_centers, x_centers, indexing="ij")
    ys = grid_y.reshape(-1)
    xs = grid_x.reshape(-1)
    centers = torch.stack([xs, ys, xs, ys], dim=1)
    centers = centers.unsqueeze(1)
    anchors = centers + base[None]
    anchors = anchors.reshape(-1, 4)
    return anchors


def generate_anchors_multi_level(
    feature_sizes: list[tuple[int, int]],
    strides: list[int],
    scales_per_level: list[tuple[int, ...]],
    aspect_ratios: tuple[float, ...] = (0.5, 1.0, 2.0),
    device: torch.device | str = "cpu",
) -> list[torch.Tensor]:
    """Anchors for an FPN-style multi-level backbone (one anchor set per level).

    With FPN, each pyramid level handles a different object size, so you use ONE
    scale per level (not 3 scales on one map) and the same aspect ratios everywhere.

    Args:
        feature_sizes: [(H_l, W_l)] per level, e.g. for P2..P6.
        strides: input pixels per cell per level, e.g. [4, 8, 16, 32, 64].
        scales_per_level: one tuple of scales per level, typically a single value each,
                          e.g. [(32,), (64,), (128,), (256,), (512,)].
        aspect_ratios: shared across levels, e.g. (0.5, 1.0, 2.0).

    Returns:
        list of per-level anchor tensors [[H_l*W_l*A, 4], ...], A = scales_l * ratios.
        (Return a LIST, not concatenated, so each level's anchors stay aligned with
        that level's RPN head outputs. Concatenate later if/when you need a flat set.)

    TODO:
        1. assert len(feature_sizes) == len(strides) == len(scales_per_level).
        2. For each level l: call generate_anchors_single_level(feature_sizes[l],
           strides[l], scales_per_level[l], aspect_ratios, device).
        3. Return the list (one tensor per level).
    """
    raise NotImplementedError("Implement generate_anchors_multi_level (FPN)")
