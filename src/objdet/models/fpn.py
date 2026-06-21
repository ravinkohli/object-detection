"""Feature Pyramid Network (Lin et al., 2017) — a neck for Faster R-CNN.

The backbone produces features at several strides:
    C2 (stride 4)  -> high-res, shallow/low-semantics
    C3 (stride 8)
    C4 (stride 16)
    C5 (stride 32) -> low-res, deep/high-semantics
FPN fuses them into a pyramid P2..P5 where EVERY level is both high-resolution AND
semantically strong, all with the same channel count (256). It does this with a
top-down pathway + lateral connections:

    P5 = smooth( lateral_1x1(C5) )
    P4 = smooth( lateral_1x1(C4) + upsample_x2(P5_premerge) )
    P3 = smooth( lateral_1x1(C3) + upsample_x2(P4_premerge) )
    P2 = smooth( lateral_1x1(C2) + upsample_x2(P3_premerge) )
    P6 = downsample(P5)     # extra coarse level for the RPN (large objects)

Where:
  - lateral_1x1 : 1x1 conv mapping each Ck's channels -> out_channels (256), so they
                  can be added together.
  - upsample_x2 : nearest-neighbor interpolation to match the finer level's H,W.
  - smooth      : 3x3 conv (padding 1) to remove the aliasing from upsampling.

Downstream changes this enables (see ops/anchors.py + the ROI head):
  - one anchor scale PER level (e.g. 32,64,128,256,512), 3 ratios each;
  - level-aware ROI pooling (torchvision.ops.MultiScaleRoIAlign) that routes each
    proposal to the pyramid level matching its size (FPN eq.1).

Reference: FPN paper §3 (the top-down architecture figure).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FPN(nn.Module):
    """Top-down feature pyramid with lateral connections."""

    def __init__(
        self,
        in_channels_list: list[int],   # channels of [C2, C3, C4, C5], e.g. resnet50 -> [256,512,1024,2048]
        out_channels: int = 256,
        add_p6: bool = True,           # extra coarsest level for the RPN
    ) -> None:
        super().__init__()
        # TODO:
        #   - self.lateral_convs : a ModuleList of 1x1 Conv2d(in_c, out_channels) — one
        #     per input level (so all levels become `out_channels` wide).
        #   - self.smooth_convs  : a ModuleList of 3x3 Conv2d(out_channels, out_channels,
        #     padding=1) — one per output level (P2..P5).
        #   - if add_p6: a way to make P6 from P5 (FPN/RetinaNet use a stride-2 3x3 conv,
        #     or a simple nn.MaxPool2d(1, stride=2)). Store add_p6.
        #   - self.out_channels = out_channels (the heads read this).
        #   - (optional) init convs with kaiming_uniform_ + zero bias.
        raise NotImplementedError("Implement FPN.__init__")

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Fuse backbone features into a pyramid.

        Args:
            features: ordered dict {"c2":..., "c3":..., "c4":..., "c5":...},
                      each [B, Ck, Hk, Wk] with Hk/Wk halving as k increases.
        Returns:
            {"p2":..., "p3":..., "p4":..., "p5":...(, "p6":...)} all [B, out_channels, ., .].

        TODO (top-down, process from DEEPEST to shallowest):
            1. laterals = [lateral_conv_k(Ck) for each level].
            2. Start prev = laterals[-1] (the C5 lateral) -> that's P5's pre-merge.
            3. For k from second-deepest down to shallowest:
                 up = F.interpolate(prev, size=laterals[k].shape[-2:], mode="nearest")
                 prev = laterals[k] + up        # merge top-down + lateral
               (keep the pre-merge maps; you smooth them next).
            4. P_k = smooth_conv_k(merged_k) for every level.
            5. if add_p6: P6 = downsample(P5). Append it.
            6. Return the dict in increasing-stride order (p2..p6).

        Gotcha: upsample to the EXACT finer-level size (use `size=...`, not
        `scale_factor=2`) — odd input sizes make 2*H' != the finer H'.
        """
        raise NotImplementedError("Implement FPN.forward")


class BackboneWithFPN(nn.Module):
    """Convenience wrapper: ResNetBackbone (multi-level) -> FPN. Optional but tidy.

    TODO: hold a multi-level backbone (ResNetBackbone returning c2..c5) + an FPN; in
    forward, run backbone then fpn and return the pyramid dict. Expose `out_channels`
    and the per-level `strides` (4,8,16,32,64) so the RPN/anchors know each level.
    """

    def __init__(self, backbone: nn.Module, fpn: FPN) -> None:
        super().__init__()
        raise NotImplementedError("Implement BackboneWithFPN (optional)")

    def forward(self, x):
        raise NotImplementedError
