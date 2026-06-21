"""Box-aware image transforms, built on torchvision.transforms.v2.

The classic ``transforms.Compose`` is image-only (it can't carry boxes). The v2
API *is* detection-aware: if the target's boxes are wrapped as
``tv_tensors.BoundingBoxes``, every geometric transform (resize, flip, crop)
updates the boxes in lockstep with the image, and passes through non-tensor
fields (image_id, labels) untouched.

Integration: the dataset returns a PIL image + a target dict whose "boxes" is a
plain xyxy tensor. ``DetectionTransforms`` wraps those boxes as BoundingBoxes
(using the original H,W as the canvas), runs the v2 pipeline, then unwraps back
to a plain tensor so downstream code (matcher/losses) sees ordinary tensors.

Per-model expectations (set in configs):
  - YOLOv1     : square resize, e.g. data.image_size: 448.
  - Faster RCNN: keep aspect ratio, data.image_min_size / image_max_size.
  - DETR family: resize + normalize (batch padding handled at collate/model).
Normalization uses ImageNet mean/std (matches a pretrained backbone; harmless
for from-scratch since it's just a fixed affine).
"""

from __future__ import annotations

from typing import Any

import torch
from torchvision import tv_tensors
from torchvision.transforms import v2


class DetectionTransforms:
    """Wrap boxes as BoundingBoxes, run a v2 pipeline, unwrap to plain tensors."""

    def __init__(self, pipeline: v2.Compose) -> None:
        self.pipeline = pipeline

    def __call__(self, image, target: dict[str, Any]):
        target = dict(target)
        h, w = target["orig_size"]
        target["boxes"] = tv_tensors.BoundingBoxes(
            target["boxes"], format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=(h, w)
        )
        image, target = self.pipeline(image, target)
        target["boxes"] = target["boxes"].as_subclass(torch.Tensor)
        return image, target


def build_transforms(cfg: dict[str, Any], train: bool) -> DetectionTransforms:
    """Assemble a v2 detection pipeline from the config's data section.

    Reads (all optional, with sensible defaults):
        data.image_size            -> square resize (int or [H,W]); takes priority.
        data.image_min_size /
        data.image_max_size        -> keep-aspect resize (shorter side -> min, capped).
        data.hflip_prob            -> random horizontal flip prob (train only).
        data.normalize_mean / std  -> normalization stats.
    """
    data = cfg.get("data", cfg)
    mean = data.get("normalize_mean", [0.485, 0.456, 0.406])
    std = data.get("normalize_std", [0.229, 0.224, 0.225])

    steps: list = [v2.ToImage()]  # PIL/ndarray -> tv_tensors.Image (uint8)

    if train and data.get("hflip_prob", 0.5):
        steps.append(v2.RandomHorizontalFlip(p=data.get("hflip_prob", 0.5)))

    if "image_size" in data:
        size = data["image_size"]
        size = (size, size) if isinstance(size, int) else tuple(size)
        steps.append(v2.Resize(size, antialias=True))
    else:
        # keep aspect ratio: shorter side -> image_min_size, longer side capped at max
        steps.append(
            v2.Resize(
                data.get("image_min_size", 600),
                max_size=data.get("image_max_size", 1000),
                antialias=True,
            )
        )

    steps += [
        v2.ToDtype(torch.float32, scale=True),  # uint8 [0,255] -> float [0,1]
        v2.Normalize(mean=mean, std=std),
        # If you later add cropping (e.g. v2.RandomIoUCrop), append
        #   v2.SanitizeBoundingBoxes()
        # to drop boxes that fall out of view AND their matching labels.
    ]
    return DetectionTransforms(v2.Compose(steps))
