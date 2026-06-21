"""Convolutional backbones (feature extractors) shared by the detectors.

"From scratch" default = random init. A config flag swaps in ImageNet-pretrained
weights so you can feel how much faster training converges with a warm start.

We lean on torchvision.models for the conv stacks (VGG16, ResNet50) — these are
generic classifiers, not detectors. We just chop off the classification head and
expose the feature map(s).

Which backbone for which detector (typical):
  - Fast/Faster R-CNN : VGG16 conv layers up to conv5_3 (stride 16) — classic choice.
  - YOLOv1            : its own 24-conv "DarkNet-like" stack (build by hand, see below).
  - DETR / DINO / RT-DETR : ResNet50 (DETR uses C5; deformable variants use C3-C5).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VGG16Backbone(nn.Module):
    """VGG16 feature extractor for R-CNN (output stride 16, 512 channels)."""

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        # TODO:
        #   - m = torchvision.models.vgg16(weights="IMAGENET1K_V1" if pretrained else None)
        #   - take m.features but DROP the final maxpool so stride stays 16 (conv5_3).
        #   - (classic trick) freeze conv1/conv2 to save memory; optional.
        #   - self.out_channels = 512.
        raise NotImplementedError("Implement VGG16Backbone.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B,3,H,W] -> [B,512,H/16,W/16]. TODO: return self.features(x)."""
        raise NotImplementedError("Implement VGG16Backbone.forward")


class ResNetBackbone(nn.Module):
    """ResNet feature extractor for the DETR family.

    Return C5 only for plain DETR, or {C3,C4,C5} for deformable (DINO/RT-DETR).
    """

    def __init__(self, name: str = "resnet50", pretrained: bool = False, return_layers: tuple[str, ...] = ("layer4",)) -> None:
        super().__init__()
        # TODO:
        #   - load torchvision.models.<name>(weights=... if pretrained else None).
        #   - use torchvision.models._utils.IntermediateLayerGetter to expose
        #     the requested layers (e.g. layer2->C3, layer3->C4, layer4->C5).
        #   - record out_channels per returned level (512/1024/2048 for resnet50).
        raise NotImplementedError("Implement ResNetBackbone.__init__")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """[B,3,H,W] -> {level_name: feature_map}. TODO."""
        raise NotImplementedError("Implement ResNetBackbone.forward")


class YOLOBackbone(nn.Module):
    """The 24 conv layers of YOLOv1 (Fig. 3 of the paper), built by hand.

    Input 448x448 -> 7x7x1024 feature map. Long but mechanical: a stack of
    Conv(+LeakyReLU(0.1)) blocks with periodic 2x2 maxpools. Build it from raw
    nn.Conv2d so you see the architecture; this one is NOT from torchvision.
    """

    def __init__(self) -> None:
        super().__init__()
        # TODO: transcribe the conv/maxpool table from YOLOv1 Fig.3 into nn.Sequential.
        #   Tip: write a small helper conv(in,out,k,s,p) -> Conv2d + LeakyReLU(0.1).
        raise NotImplementedError("Implement YOLOBackbone.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B,3,448,448] -> [B,1024,7,7]. TODO."""
        raise NotImplementedError("Implement YOLOBackbone.forward")


def build_backbone(cfg: dict) -> nn.Module:
    """Factory: pick a backbone class from cfg['model']['backbone']. TODO."""
    raise NotImplementedError("Implement build_backbone")
