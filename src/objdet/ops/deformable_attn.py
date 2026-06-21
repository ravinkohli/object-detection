"""Multi-scale deformable attention — pure PyTorch (Deformable DETR, used by DINO & RT-DETR).

Standard attention attends to *all* spatial locations (O((HW)^2) — brutal for
detection feature maps). Deformable attention instead lets each query attend to a
small set of *learned sampling points* around a reference point, across multiple
feature levels. Reference: "Deformable DETR" (Zhu et al., 2021), eq. (2)-(3).

The official implementation ships a custom CUDA/C++ kernel (MSDeformAttn) that does
not exist for MPS. We implement the pure-PyTorch version using
``F.grid_sample`` — slower, but correct and MPS-friendly. That's the whole point of
this stub: get the mechanics right, accept the speed hit.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def ms_deform_attn_core_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """The core sampling+weighting op (grid_sample based).

    Args:
        value:                [B, sum(H_l*W_l), n_heads, head_dim] flattened multi-level features.
        value_spatial_shapes: [n_levels, 2] each (H_l, W_l).
        sampling_locations:   [B, n_queries, n_heads, n_levels, n_points, 2] in [0,1].
        attention_weights:    [B, n_queries, n_heads, n_levels, n_points] (softmax over levels*points).

    Returns:
        [B, n_queries, n_heads*head_dim].

    TODO:
        1. Split ``value`` per level using value_spatial_shapes; reshape each level
           to [B*n_heads, head_dim, H_l, W_l].
        2. Map sampling_locations from [0,1] to grid_sample's [-1,1] convention.
        3. For each level: F.grid_sample(value_l, grid_l, mode="bilinear",
           align_corners=False) -> sampled [B*n_heads, head_dim, n_queries, n_points].
        4. Stack levels, weight by attention_weights, sum over (levels, points).
        5. Reshape to [B, n_queries, n_heads*head_dim].
    """
    raise NotImplementedError("Implement ms_deform_attn_core_pytorch")


class MSDeformAttn(nn.Module):
    """Multi-scale deformable attention module (the nn wrapper around the core op)."""

    def __init__(self, d_model: int = 256, n_levels: int = 4, n_heads: int = 8, n_points: int = 4) -> None:
        super().__init__()
        # TODO:
        #   - assert d_model % n_heads == 0; store dims.
        #   - self.sampling_offsets = nn.Linear(d_model, n_heads*n_levels*n_points*2)
        #   - self.attention_weights = nn.Linear(d_model, n_heads*n_levels*n_points)
        #   - self.value_proj = nn.Linear(d_model, d_model)
        #   - self.output_proj = nn.Linear(d_model, d_model)
        #   - initialize sampling_offsets bias to a ring of directions (see paper repo).
        raise NotImplementedError("Implement MSDeformAttn.__init__")

    def forward(
        self,
        query: torch.Tensor,             # [B, n_queries, d_model]
        reference_points: torch.Tensor,  # [B, n_queries, n_levels, 2] in [0,1]
        input_flatten: torch.Tensor,     # [B, sum(HW), d_model] (the "value")
        input_spatial_shapes: torch.Tensor,
    ) -> torch.Tensor:
        """TODO:
        1. value = value_proj(input_flatten); reshape to [B, sum(HW), n_heads, head_dim].
        2. offsets = sampling_offsets(query).view(B, Nq, n_heads, n_levels, n_points, 2).
        3. attn = softmax(attention_weights(query).view(..., n_levels*n_points)) -> reshape.
        4. sampling_locations = reference_points[:,:,None,:,None,:] + offsets / normalizer.
        5. out = ms_deform_attn_core_pytorch(value, shapes, sampling_locations, attn).
        6. return output_proj(out).
        """
        raise NotImplementedError("Implement MSDeformAttn.forward")
