"""COCO mAP evaluation via pycocotools.

Every model is scored the same way: run inference over the val set, dump
predictions in COCO detection JSON format, then let ``COCOeval`` compute
AP@[.5:.95], AP50, AP75, etc. This makes the six detectors directly comparable.

COCO detection result format (list of dicts):
    {"image_id": int, "category_id": int, "bbox": [x, y, w, h], "score": float}
Note: ``bbox`` is xywh (top-left + width/height), NOT xyxy — convert before dumping.

pycocotools refs:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    coco_gt: Any,  # pycocotools.coco.COCO ground-truth handle
) -> dict[str, float]:
    """Run COCO evaluation and return a dict of AP metrics.

    TODO:
        1. model.eval().
        2. For each batch: move images to device; preds = model(images)  # inference mode.
           Each prediction has boxes [N,4] xyxy, scores [N], labels [N].
        3. Convert boxes xyxy -> xywh; map your contiguous label ids back to COCO
           category_ids (keep the mapping from the dataset!).
        4. Append {"image_id","category_id","bbox","score"} per detection.
        5. coco_dt = coco_gt.loadRes(results)
        6. e = COCOeval(coco_gt, coco_dt, iouType="bbox"); e.evaluate(); e.accumulate(); e.summarize()
        7. Pull numbers out of ``e.stats`` (stats[0]=AP, stats[1]=AP50, stats[2]=AP75, ...).
        8. Return {"mAP": ..., "AP50": ..., "AP75": ...}.

    Gotcha: if a batch produces zero detections COCOeval still needs a (possibly
    empty) results list — handle that gracefully.
    """
    raise NotImplementedError("Implement evaluate")
