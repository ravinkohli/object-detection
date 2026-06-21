"""Build a small COCO subset so the full train->eval loop runs on a laptop GPU.

Default strategy (set in configs): pick a handful of classes and at most N images
that contain them, then split into train/val. This keeps everything "real COCO"
while making from-scratch training tractable on MPS.

Because the default download is val2017 only (~1GB, 5000 imgs), we carve BOTH the
train and val subsets out of val2017's annotations. Swap to train2017 later via the
download flag for a bigger run.
"""

from __future__ import annotations

from typing import Any


def select_image_ids(
    coco: Any,                          # pycocotools.coco.COCO
    class_names: list[str] | None = None,
    max_images: int | None = None,
    require_all: bool = False,
    seed: int = 0,
) -> list[int]:
    """Choose image ids for the subset.

    Args:
        coco: a loaded COCO handle.
        class_names: keep images containing these categories (None = all classes).
        max_images: cap the number of images.
        require_all: if True keep only images containing *every* listed class
                     (usually False = any).

    Returns:
        Sorted list of image ids.

    TODO:
        1. Map class_names -> category ids via coco.getCatIds(catNms=...).
        2. Gather image ids via coco.getImgIds(catIds=...) (union vs intersection
           depending on require_all).
        3. Deterministically shuffle (seeded) and truncate to max_images.
    """
    raise NotImplementedError("Implement select_image_ids")


def train_val_split(
    image_ids: list[int],
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[list[int], list[int]]:
    """Split image ids into (train_ids, val_ids) deterministically.

    TODO: seeded shuffle, slice by val_fraction, return the two lists.
    """
    raise NotImplementedError("Implement train_val_split")
