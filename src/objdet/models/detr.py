"""DETR — DEtection TRansformer (Carion et al., 2020).

The first end-to-end set-prediction detector: no anchors, no NMS. A CNN backbone +
transformer encoder-decoder maps an image to a fixed set of N predictions; a
bipartite (Hungarian) matching + set loss trains them to align 1:1 with the GT.

Pipeline:
    image -> ResNet C5 -> 1x1 conv to d_model -> + pos enc
          -> transformer encoder/decoder with N object queries
          -> per-query: class head (Linear -> C+1) and box head (MLP -> 4, sigmoid, cxcywh)

Loss + matching live in ops.hungarian_matcher and losses.set_criterion.
"""

from __future__ import annotations

import torch.nn as nn

from .transformer import MLP, PositionEmbeddingSine, Transformer


class DETR(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, num_queries: int = 100, d_model: int = 256, cfg: dict | None = None) -> None:
        super().__init__()
        # TODO:
        #   - self.backbone = backbone (ResNetBackbone returning C5).
        #   - self.input_proj = Conv2d(backbone.out_channels, d_model, 1).
        #   - self.pos_enc = PositionEmbeddingSine(d_model // 2).
        #   - self.query_embed = nn.Embedding(num_queries, d_model).
        #   - self.transformer = Transformer(d_model, ...).
        #   - self.class_embed = nn.Linear(d_model, num_classes + 1)   # +1 = "no object".
        #   - self.bbox_embed  = MLP(d_model, d_model, 4, num_layers=3).
        raise NotImplementedError("Implement DETR.__init__")

    def forward(self, images, targets=None):
        """
        Returns:
            outputs = {"pred_logits": [B,N,C+1], "pred_boxes": [B,N,4] cxcywh in [0,1],
                       "aux_outputs": [...]}  (aux = per-decoder-layer outputs).
            train: also compute losses via losses.set_criterion (or return outputs and
                   let the engine call the criterion — pick one and be consistent).

        TODO:
            1. feat = backbone(images)[last]; src = input_proj(feat); pos = pos_enc(feat).
            2. hs = transformer(src, pos, query_embed.weight)  # [L,B,N,d].
            3. logits = class_embed(hs); boxes = bbox_embed(hs).sigmoid().
            4. outputs from the LAST layer; stash earlier layers as aux_outputs.
            5. (train) loss_dict = criterion(outputs, targets). Return accordingly.
        """
        raise NotImplementedError("Implement DETR.forward")

    def postprocess(self, outputs, target_sizes):
        """Set predictions -> detections (NO NMS; just top-k by score).

        TODO: softmax logits (drop the no-object class), take per-query best class +
        score, convert cxcywh[0,1] -> xyxy absolute using target_sizes, top-k.
        """
        raise NotImplementedError("Implement DETR.postprocess")
