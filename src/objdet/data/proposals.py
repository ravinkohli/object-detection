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

import cv2
import numpy as np
from pycocotools.coco import COCO
import torch
from tqdm import tqdm

from objdet.ops.boxes import convert


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
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image_bgr)
    if fast:
        ss.switchToSelectiveSearchFast()
    else:
        ss.switchToSelectiveSearchQuality()
    rects = ss.process()
    boxes = torch.as_tensor(rects, dtype=torch.float32)
    boxes_xyxy = convert(boxes, in_fmt="xywh", out_fmt="xyxy")
    return boxes_xyxy[:max_proposals].numpy()


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
    c = COCO(coco_ann_file)
    image_ids = image_ids or c.getImgIds()
    infos = c.loadImgs(image_ids)
    out_path = Path(out_path)
    if not out_path.exists():
        out_path.mkdir(exist_ok=True)

    for info in tqdm(infos):
        out_file = Path(out_path) / f"{info['id']}.npy"
        if out_file.exists():
            continue
        image = cv2.imread((Path(images_dir) / info["file_name"]).as_posix())
        if image is None:
            continue
        proposals = selective_search(image, fast, max_proposals=max_proposals)
        np.save(out_file, proposals)
    return


def load_proposals(cache_path: str | Path, image_id: int) -> np.ndarray:
    """Load cached proposals for one image. TODO: read from the cache format you chose."""
    return np.load(Path(cache_path) / f"{image_id}.npy")
