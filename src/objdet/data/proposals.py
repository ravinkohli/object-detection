"""Selective Search region proposals for Fast R-CNN (the historically faithful path).

Fast R-CNN does NOT generate its own proposals — it consumes ~2000 category-agnostic
region proposals per image produced by an external algorithm (Selective Search, the
2013 method). Faster R-CNN's whole contribution was replacing this with a learned RPN,
so implementing this makes the leap concrete.

Selective Search is slow (~1-2s/image), so we compute proposals ONCE and cache them
to disk (keyed by image_id). Training then just loads the cache.

OpenCV API (needs opencv-contrib-python):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(bgr_image)
    ss.switchToSelectiveSearchFast()   # or switchToSelectiveSearchQuality()
    rects = ss.process()               # array of [x, y, w, h]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def selective_search(image_bgr: np.ndarray, fast: bool = True, max_proposals: int = 2000) -> np.ndarray:
    """Run Selective Search on one image.

    Args:
        image_bgr: HxWx3 uint8 BGR image (OpenCV order!).
        fast: fast vs quality mode.
        max_proposals: keep at most this many (Fast R-CNN uses ~2000).

    Returns:
        [P, 4] xyxy proposals (int/float pixel coords).

    TODO:
        1. Create the SS segmenter, setBaseImage, switch mode, process().
        2. rects are xywh -> convert to xyxy; truncate to max_proposals.
    """
    raise NotImplementedError("Implement selective_search")


def cache_proposals(
    coco_ann_file: str,
    images_dir: str,
    out_path: str | Path,
    image_ids: list[int] | None = None,
    fast: bool = True,
    max_proposals: int = 2000,
) -> None:
    """Precompute + save proposals for a set of images.

    Driven by ``scripts/precompute_proposals.py``.

    TODO:
        1. Load COCO; resolve image_ids.
        2. For each image: read with cv2.imread (BGR); proposals = selective_search(...).
        3. Save a mapping {image_id: [P,4] array} to ``out_path`` (np.savez / pickle /
           one .npy per image). Show a tqdm progress bar — this takes a while.
    """
    raise NotImplementedError("Implement cache_proposals")


def load_proposals(cache_path: str | Path, image_id: int) -> np.ndarray:
    """Load cached proposals for one image. TODO: read from the cache format you chose."""
    raise NotImplementedError("Implement load_proposals")
