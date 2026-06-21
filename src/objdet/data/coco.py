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

from PIL import Image
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO

from objdet.ops.boxes import convert


class CocoDetectionDataset(Dataset):
    """COCO detection dataset yielding (image, target) pairs.

    Args:
        images_dir: folder of image files (e.g. data/coco/val2017).
        ann_file:   COCO instances_*.json.
        image_ids:  restrict to these image ids (see data/subset.py); None = all.
        transforms: callable (image, target) -> (image, target).
        classes:    subset of category NAMES to keep. Labels become contiguous
                    0..C-1 in the order given (classes[0] -> 0). None = all 80.
        cat_id_map: reuse an existing {coco_cat_id -> contiguous} map so the val
                    split agrees with train. Overrides ``classes`` if both given.
    """

    def __init__(
        self,
        images_dir: str | Path,
        ann_file: str | Path,
        image_ids: list[int] | None = None,
        transforms: Callable | None = None,
        classes: list[str] | None = None,
        cat_id_map: dict[int, int] | None = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.transforms = transforms
        self.coco = COCO(ann_file)

        # category mapping: coco category_id -> contiguous label 0..C-1
        if cat_id_map is not None:
            self.cat_id_map = dict(cat_id_map)
        elif classes is not None:
            name_to_id = {c["name"]: c["id"] for c in self.coco.loadCats(self.coco.getCatIds())}
            self.cat_id_map = {name_to_id[name]: i for i, name in enumerate(classes)}
        else:
            self.cat_id_map = {c: i for i, c in enumerate(self.coco.cats.keys())}
        self.contiguous_to_coco = {v: k for k, v in self.cat_id_map.items()}
        self.cat_ids = list(self.cat_id_map.keys())  # coco ids we keep

        # keep only images with >=1 non-crowd annotation in our categories
        ids = image_ids or list(self.coco.imgs.keys())
        self.ids = [
            i for i in ids
            if len(self.coco.getAnnIds(imgIds=i, catIds=self.cat_ids, iscrowd=False)) > 0
        ]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
        img_id = self.ids[index]
        info = self.coco.loadImgs(img_id)[0]
        image = Image.open(self.images_dir / info["file_name"]).convert("RGB")

        # only annotations of our subset categories (drops other classes + crowd)
        anns = self.coco.loadAnns(
            self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids, iscrowd=False)
        )
        xywh = torch.as_tensor([a["bbox"] for a in anns], dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(
            [self.cat_id_map[a["category_id"]] for a in anns], dtype=torch.long
        )
        boxes = convert(xywh, in_fmt="xywh", out_fmt="xyxy")

        # drop degenerate boxes (zero / negative area)
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, labels = boxes[keep], labels[keep]

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": img_id,
            "orig_size": (info["height"], info["width"]),
        }
        return self.transforms(image, target) if self.transforms else (image, target)


def collate_fn(batch):
    """Collate (image, target) samples into a batch.

    Detection samples have a variable number of boxes per image, so the default
    collate (which stacks every field) fails on the target dict. We keep images
    and targets as parallel lists; this works for every detector here:
      - Faster R-CNN consumes a list of variable-size images directly.
      - YOLOv1 / DETR use fixed-size or padded images, so the model (or its own
        collate) can ``torch.stack(images)`` once it knows they share a shape.

    Args:
        batch: list of (image, target_dict).
    Returns:
        (images, targets): a list of B images and a list of B target dicts.
    """
    images, targets = zip(*batch)
    return list(images), list(targets)
