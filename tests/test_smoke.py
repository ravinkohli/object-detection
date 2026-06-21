"""Smoke tests — a forward (and backward) pass on tiny dummy inputs per model.

These catch shape bugs without needing the dataset. Pattern to follow for each model
once it's implemented:

    model = FasterRCNN(backbone, num_classes=5)
    images = torch.randn(2, 3, 224, 224)
    targets = [{"boxes": torch.tensor([[10.,10.,50.,50.]]), "labels": torch.tensor([1])},
               {"boxes": torch.tensor([[20.,20.,80.,80.]]), "labels": torch.tensor([2])}]
    loss_dict = model(images, targets)        # train mode -> dict of scalar tensors
    loss = sum(loss_dict.values())
    loss.backward()                            # gradients flow, no shape errors
    model.eval(); dets = model(images)         # eval mode -> list of detections

Run: uv run pytest tests/test_smoke.py
"""

import pytest

pytestmark = pytest.mark.skip(reason="TODO: implement the models then enable per-model smoke tests")


def test_faster_rcnn_forward_backward():
    # TODO
    ...


def test_yolov1_forward_backward():
    # TODO
    ...


def test_detr_forward_backward():
    # TODO
    ...
