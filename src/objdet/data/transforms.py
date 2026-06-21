"""Box-aware image transforms.

Detection transforms must transform the *boxes* alongside the image (a horizontal
flip flips the boxes too; a resize rescales them). torchvision.transforms.v2 has
box-aware transforms (it understands tv_tensors.BoundingBoxes) — you may use those,
or write simple ones here to see exactly what happens to the coordinates.

Each transform is callable as ``image, target = t(image, target)``.

Per-model expectations (set in configs):
  - YOLOv1 : square resize to 448x448.
  - Faster R-CNN : resize shorter side ~600/800, keep aspect ratio.
  - DETR family : resize + normalize; pad a batch to common size (mask the padding).
All: normalize with ImageNet mean/std if using a pretrained backbone, else any
fixed mean/std (or just /255).
"""

from __future__ import annotations

from typing import Any

import torch


class Compose:
    """Chain transforms. TODO: store list; call each in order on (image, target)."""

    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, image, target):
        # TODO: for t in self.transforms: image, target = t(image, target)
        raise NotImplementedError("Implement Compose.__call__")


class ToTensor:
    """PIL image -> float tensor [3,H,W] in [0,1]. Boxes untouched. TODO."""

    def __call__(self, image, target):
        raise NotImplementedError("Implement ToTensor")


class Resize:
    """Resize image (and scale boxes). TODO: handle fixed-square vs shorter-side modes."""

    def __init__(self, size: int | tuple[int, int], keep_aspect: bool = False) -> None:
        self.size = size
        self.keep_aspect = keep_aspect

    def __call__(self, image, target):
        # TODO: resize image; multiply target["boxes"] by (sx, sy); update sizes.
        raise NotImplementedError("Implement Resize")


class RandomHorizontalFlip:
    """Flip image and mirror boxes: x1' = W - x2, x2' = W - x1. TODO."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image, target):
        raise NotImplementedError("Implement RandomHorizontalFlip")


class Normalize:
    """Normalize a float image tensor by mean/std. Boxes untouched. TODO."""

    def __init__(self, mean: tuple[float, ...], std: tuple[float, ...]) -> None:
        self.mean, self.std = mean, std

    def __call__(self, image, target):
        raise NotImplementedError("Implement Normalize")


def build_transforms(cfg: dict[str, Any], train: bool):
    """Assemble a Compose pipeline from the config's data section.

    TODO: read cfg (image size, flip prob, normalize stats) and return Compose([...]).
    """
    raise NotImplementedError("Implement build_transforms")
