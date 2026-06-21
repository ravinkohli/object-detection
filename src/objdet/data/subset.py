"""Build a small COCO subset so the full train->eval loop runs on a laptop GPU.

Default strategy (set in configs): pick a handful of classes and at most N images
that contain them, then split into train/val. This keeps everything "real COCO"
while making from-scratch training tractable on MPS.

Because the default download is val2017 only (~1GB, 5000 imgs), we carve BOTH the
train and val subsets out of val2017's annotations. Swap to train2017 later via the
download flag for a bigger run.
"""

from __future__ import annotations

import random
from typing import Any

from sklearn.model_selection import train_test_split


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

    Note: pycocotools' ``getImgIds(catIds=[...])`` returns images containing ALL
    given categories (intersection). For the usual "any of these classes" behavior
    we union per-category instead.
    """
    if class_names:
        cat_ids = coco.getCatIds(catNms=class_names)
        if require_all:
            img_ids = set(coco.getImgIds(catIds=cat_ids))
        else:
            img_ids = set()
            for c in cat_ids:
                img_ids |= set(coco.getImgIds(catIds=[c]))
    else:
        img_ids = set(coco.getImgIds())

    img_ids = sorted(img_ids)
    # deterministic random sample (seeded local RNG, doesn't touch global state)
    random.Random(seed).shuffle(img_ids)
    if max_images is not None:
        img_ids = img_ids[:max_images]
    return sorted(img_ids)
    


def train_val_split(
    image_ids: list[int],
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[list[int], list[int]]:
    """Split image ids into (train_ids, val_ids) deterministically.

    Uses sklearn's train_test_split (shuffled, seeded by ``seed``).
    """
    train_ids, val_ids = train_test_split(
        image_ids, test_size=val_fraction, random_state=seed, shuffle=True
    )
    return train_ids, val_ids
