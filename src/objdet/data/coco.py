"""COCO detection dataset.

Wraps pycocotools to yield (image, target) pairs. We use pycocotools for parsing
the annotation JSON (reimplementing that teaches nothing about detection) but the
dataset/transform logic is yours.

Each item:
    image  : [3, H, W] float tensor (after transforms: resized + normalized).
    target : dict with at least
                "boxes"    : [G, 4] xyxy absolute pixel boxes (float),
                "labels"   : [G]    contiguous class ids (long) — NOT raw COCO ids,
                "image_id" : int    (needed for COCO eval),
                "orig_size": (H0, W0) original size (to map preds back for eval).

IMPORTANT id mapping: COCO category_ids are not contiguous (1..90 with gaps). Build
a {coco_cat_id -> contiguous 0..C-1} map (and its inverse for eval) once and reuse.

pycocotools: from pycocotools.coco import COCO; coco = COCO(ann_file).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import Dataset


class CocoDetectionDataset(Dataset):
    def __init__(
        self,
        images_dir: str | Path,
        ann_file: str | Path,
        image_ids: list[int] | None = None,   # restrict to a subset (see data/subset.py)
        transforms: Callable | None = None,
        cat_id_map: dict[int, int] | None = None,
    ) -> None:
        """TODO:
        1. self.coco = COCO(ann_file); store images_dir, transforms.
        2. self.ids = image_ids or list(self.coco.imgs.keys()).
           (Optionally drop images with zero annotations.)
        3. Build self.cat_id_map (coco->contiguous) and self.contiguous_to_coco
           (reuse cat_id_map if passed so train/val agree).
        """
        raise NotImplementedError("Implement CocoDetectionDataset.__init__")

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        """TODO:
        1. img_id = self.ids[index]; load image file (PIL) from images_dir.
        2. anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id, iscrowd=False)).
        3. boxes = [ann['bbox'] for ann in anns]  # COCO bbox is xywh -> convert to xyxy.
           labels = [self.cat_id_map[ann['category_id']] for ann in anns].
           Skip degenerate boxes (w<=0 or h<=0).
        4. Build target dict (boxes, labels, image_id, orig_size).
        5. If self.transforms: image, target = self.transforms(image, target).
        6. Return (image_tensor, target).
        """
        raise NotImplementedError("Implement CocoDetectionDataset.__getitem__")


def collate_fn(batch):
    """Detection batches have variable #boxes per image -> can't default-stack targets.

    TODO:
        - return (list_of_images, list_of_targets), or stack images to [B,3,H,W] only
          if your transforms pad/resize them all to the same size (DETR pads; YOLO
          uses fixed 448x448; Faster R-CNN typically keeps a list).
    """
    raise NotImplementedError("Implement collate_fn")
