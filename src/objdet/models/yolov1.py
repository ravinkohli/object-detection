"""YOLOv1 (Redmon et al., 2016) — "You Only Look Once".

The first single-stage detector: no proposals, no anchors. The image is divided
into an S x S grid (S=7). Each cell predicts B bounding boxes (B=2) with a
confidence, plus C class probabilities (shared by the cell). Detection is reframed
as a single regression from pixels to a 7x7x(B*5 + C) tensor.

Output tensor layout per cell (the convention this repo uses):
    [x, y, w, h, conf]  * B   then   [class_1 ... class_C]
  - (x, y): box center RELATIVE TO THE CELL, in [0,1].
  - (w, h): box size relative to the WHOLE IMAGE, in [0,1] (paper uses sqrt in loss).
  - conf  : Pr(object) * IoU(pred, truth).
For COCO C would be 80; the paper trained on VOC (C=20). With S=7,B=2 the channel
count is 7*7*(2*5 + C).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class YOLOv1(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, S: int = 7, B: int = 2) -> None:
        super().__init__()
        # TODO:
        #   - self.backbone = backbone (YOLOBackbone -> [B,1024,7,7]).
        #   - detection head: a couple of FC layers mapping 1024*7*7 ->
        #     S*S*(B*5 + num_classes), reshaped to [B, S, S, B*5+C].
        #     (Original uses Linear(7*7*1024, 4096) -> LeakyReLU -> Dropout ->
        #      Linear(4096, S*S*(B*5+C)).)
        #   - store S, B, num_classes.
        raise NotImplementedError("Implement YOLOv1.__init__")

    def forward(self, images, targets=None):
        """
        Returns:
            train: {"loss": total} (or the component dict) via losses.yolo_loss.
            eval:  decoded detections per image.

        TODO:
            1. feats = self.backbone(images); flatten; head -> reshape [B,S,S,B*5+C].
            2. (train) build the SxSx(...) target tensor from GT boxes (assign each
               object to the cell containing its center) and call yolo_loss(pred, target).
            3. (eval) decode_predictions(...) below, then score-thresh + NMS.
        """
        raise NotImplementedError("Implement YOLOv1.forward")

    @torch.no_grad()
    def decode_predictions(self, pred: torch.Tensor, conf_thresh: float = 0.25):
        """Turn the [B,S,S,B*5+C] grid into absolute xyxy boxes + scores + labels.

        TODO:
            1. For each cell & each of the B boxes: convert (x,y) cell-relative ->
               image coords using the cell's (col,row) and stride; (w,h)*image_size.
            2. class_score = conf * class_prob; keep best class per box.
            3. Threshold by conf_thresh; convert center+size -> xyxy. (NMS done later.)
        """
        raise NotImplementedError("Implement YOLOv1.decode_predictions")
