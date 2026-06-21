"""Tests for ops/boxes.py.

Run: uv run pytest tests/test_boxes.py
"""

import math

import torch

from objdet.ops import boxes


# --- encode / decode -------------------------------------------------------

def test_encode_decode_roundtrip_default_weights():
    # reference (anchors) and targets, all xyxy, deliberately different.
    reference = torch.tensor(
        [[10.0, 10.0, 50.0, 50.0],
         [0.0, 0.0, 100.0, 200.0],
         [30.0, 40.0, 90.0, 120.0]]
    )
    target = torch.tensor(
        [[12.0, 14.0, 60.0, 58.0],
         [5.0, 10.0, 110.0, 180.0],
         [25.0, 35.0, 100.0, 150.0]]
    )
    deltas = boxes.encode_boxes(reference, target)
    recovered = boxes.decode_boxes(deltas, reference)
    torch.testing.assert_close(recovered, target, rtol=1e-4, atol=1e-4)


def test_encode_decode_roundtrip_scaled_weights():
    # round-trip must hold for any weights, as long as encode/decode use the same.
    reference = torch.tensor([[10.0, 10.0, 50.0, 50.0], [0.0, 0.0, 80.0, 60.0]])
    target = torch.tensor([[14.0, 9.0, 55.0, 52.0], [3.0, 4.0, 70.0, 66.0]])
    weights = (10.0, 10.0, 5.0, 5.0)
    deltas = boxes.encode_boxes(reference, target, weights=weights)
    recovered = boxes.decode_boxes(deltas, reference, weights=weights)
    torch.testing.assert_close(recovered, target, rtol=1e-4, atol=1e-4)


def test_encode_identity_is_zero():
    # encoding a box against itself -> all deltas zero (center matches, log(1)=0).
    ref = torch.tensor([[10.0, 20.0, 40.0, 80.0]])
    deltas = boxes.encode_boxes(ref, ref)
    torch.testing.assert_close(deltas, torch.zeros_like(deltas), atol=1e-6, rtol=0)


def test_decode_zero_delta_returns_reference():
    # zero deltas -> boxes unchanged from the reference.
    ref = torch.tensor([[10.0, 20.0, 40.0, 80.0], [0.0, 0.0, 100.0, 50.0]])
    deltas = torch.zeros_like(ref)
    out = boxes.decode_boxes(deltas, ref)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


def test_weights_actually_scale_deltas():
    # bigger weights -> proportionally bigger encoded deltas.
    ref = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
    tgt = torch.tensor([[14.0, 12.0, 60.0, 55.0]])
    d1 = boxes.encode_boxes(ref, tgt, weights=(1.0, 1.0, 1.0, 1.0))
    d10 = boxes.encode_boxes(ref, tgt, weights=(10.0, 10.0, 10.0, 10.0))
    torch.testing.assert_close(d10, d1 * 10.0, rtol=1e-5, atol=1e-6)


# --- IoU -------------------------------------------------------------------

def test_iou_identical_is_one():
    b = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    iou = boxes.box_iou(b, b)
    torch.testing.assert_close(iou, torch.tensor([[1.0]]))


def test_iou_disjoint_is_zero():
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b = torch.tensor([[20.0, 20.0, 30.0, 30.0]])
    iou = boxes.box_iou(a, b)
    torch.testing.assert_close(iou, torch.tensor([[0.0]]))


def test_iou_known_half_overlap():
    # two unit-area-ish boxes overlapping in exactly half their union.
    # A = [0,0,2,2] (area 4), B = [1,0,3,2] (area 4), intersection = [1,0,2,2] area 2.
    # union = 4 + 4 - 2 = 6 -> IoU = 2/6 = 1/3.
    a = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
    b = torch.tensor([[1.0, 0.0, 3.0, 2.0]])
    iou = boxes.box_iou(a, b)
    torch.testing.assert_close(iou, torch.tensor([[1.0 / 3.0]]), rtol=1e-5, atol=1e-6)


def test_iou_matrix_shape():
    a = torch.rand(5, 4)
    a[:, 2:] += a[:, :2] + 1.0  # ensure x2>x1, y2>y1
    b = torch.rand(3, 4)
    b[:, 2:] += b[:, :2] + 1.0
    assert boxes.box_iou(a, b).shape == (5, 3)


# --- NMS -------------------------------------------------------------------

def test_nms_suppresses_overlap():
    # two heavily overlapping boxes; only the higher-scoring one survives.
    b = torch.tensor([[0.0, 0.0, 10.0, 10.0],
                      [1.0, 1.0, 11.0, 11.0]])
    scores = torch.tensor([0.9, 0.8])
    keep = boxes.nms(b, scores, iou_thresh=0.5)
    assert keep.tolist() == [0]


def test_nms_keeps_distinct_boxes():
    b = torch.tensor([[0.0, 0.0, 10.0, 10.0],
                      [100.0, 100.0, 110.0, 110.0]])
    scores = torch.tensor([0.5, 0.6])
    keep = boxes.nms(b, scores, iou_thresh=0.5)
    assert sorted(keep.tolist()) == [0, 1]


def test_batched_nms_respects_classes():
    # same overlapping geometry but different classes -> NOT suppressed.
    b = torch.tensor([[0.0, 0.0, 10.0, 10.0],
                      [1.0, 1.0, 11.0, 11.0]])
    scores = torch.tensor([0.9, 0.8])
    idxs = torch.tensor([0, 1])  # different class ids
    keep = boxes.batched_nms(b, scores, idxs, iou_thresh=0.5)
    assert sorted(keep.tolist()) == [0, 1]


# --- format conversion -----------------------------------------------------

def test_convert_roundtrip_xyxy_cxcywh():
    b = torch.tensor([[10.0, 20.0, 40.0, 80.0], [0.0, 0.0, 100.0, 50.0]])
    back = boxes.convert(boxes.convert(b, "xyxy", "cxcywh"), "cxcywh", "xyxy")
    torch.testing.assert_close(back, b, rtol=1e-5, atol=1e-5)


def test_clip_to_image():
    b = torch.tensor([[-5.0, -5.0, 50.0, 200.0]])  # spills past a 100x100 image
    clipped = boxes.clip_to_image(b, (100, 100))  # size = (H, W)
    assert clipped[0, 0].item() >= 0.0
    assert clipped[0, 1].item() >= 0.0
    assert clipped[0, 2].item() <= 100.0
    assert clipped[0, 3].item() <= 100.0
