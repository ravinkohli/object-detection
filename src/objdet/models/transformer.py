"""Transformer building blocks shared by the DETR family.

DETR adds a standard transformer encoder-decoder on top of CNN features:
  - flatten the [B,C,H,W] feature map to a sequence [B, H*W, C] of "tokens",
  - add 2D sinusoidal positional encodings (the transformer is permutation-invariant),
  - encoder: self-attention over image tokens -> "memory",
  - decoder: N learnable "object queries" cross-attend to memory, producing N output
    embeddings, each decoded into (class, box) by small heads.

For plain DETR you can build this from torch.nn.MultiheadAttention / nn.TransformerEncoderLayer.
DINO/RT-DETR swap the attention for deformable attention (ops.deformable_attn).
Reference: DETR (Carion et al., 2020), §3.2 + appendix architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PositionEmbeddingSine(nn.Module):
    """2D sinusoidal positional encoding over a feature map.

    TODO:
        - Given a [B,C,H,W] feature (and optional padding mask), produce a
          [B, C, H, W] (or [B, H*W, C]) position embedding: build normalized x/y
          coordinate grids, apply the sin/cos-of-geometric-frequencies formula
          (half the channels encode x, half encode y). See DETR's PositionEmbeddingSine.
    """

    def __init__(self, num_pos_feats: int = 128, temperature: float = 10000.0) -> None:
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError("Implement PositionEmbeddingSine.forward")


class TransformerEncoderLayer(nn.Module):
    """One encoder block: self-attn + FFN (pre/post-norm your choice).

    TODO: nn.MultiheadAttention(d_model, nhead) + 2-layer FFN + LayerNorms +
    residuals. Add positional encoding to Q and K (not to V) — DETR detail.
    """

    def __init__(self, d_model=256, nhead=8, dim_feedforward=2048, dropout=0.1) -> None:
        super().__init__()
        raise NotImplementedError("Implement TransformerEncoderLayer")

    def forward(self, src, pos):
        raise NotImplementedError


class TransformerDecoderLayer(nn.Module):
    """One decoder block: self-attn over queries + cross-attn to memory + FFN.

    TODO: two nn.MultiheadAttention (self then cross) + FFN + norms. Add query
    positional embeddings to the query self-attn, and image pos to the memory keys.
    """

    def __init__(self, d_model=256, nhead=8, dim_feedforward=2048, dropout=0.1) -> None:
        super().__init__()
        raise NotImplementedError("Implement TransformerDecoderLayer")

    def forward(self, tgt, memory, pos, query_pos):
        raise NotImplementedError


class Transformer(nn.Module):
    """Full encoder-decoder. Returns per-layer decoder outputs (for aux losses)."""

    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048) -> None:
        super().__init__()
        # TODO: stack N encoder + M decoder layers (nn.ModuleList). Return ALL decoder
        #       layer outputs stacked [num_layers, B, num_queries, d_model] so the
        #       criterion can apply auxiliary losses at every layer.
        raise NotImplementedError("Implement Transformer.__init__")

    def forward(self, src, pos_embed, query_embed):
        """
        Args:
            src:         [B, C, H, W] backbone features projected to d_model.
            pos_embed:   [B, C, H, W] positional encoding.
            query_embed: [num_queries, d_model] learnable object queries.
        Returns:
            hs: [num_decoder_layers, B, num_queries, d_model].
        TODO: flatten src/pos to sequences; run encoder; expand query_embed over batch;
              run decoder; reshape outputs.
        """
        raise NotImplementedError("Implement Transformer.forward")


class MLP(nn.Module):
    """Tiny multi-layer perceptron used for the box-regression head (DETR's MLP).

    TODO: ``num_layers`` Linear layers with ReLU between; final layer outputs 4
    (the cxcywh box, fed through sigmoid in the detector). Useful enough to keep here.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers) -> None:
        super().__init__()
        raise NotImplementedError("Implement MLP")

    def forward(self, x):
        raise NotImplementedError
