"""Tests for ops/boxes.py — write these as you implement the box math.

Good first tests to add (you write the assertions):
  - encode_boxes then decode_boxes round-trips: decode(encode(gt, anchors), anchors)
    should recover `gt` (up to float tolerance) when anchors != gt.
  - convert() round-trips between xyxy <-> cxcywh <-> xywh.
  - box_iou: identical boxes -> 1.0; disjoint boxes -> 0.0; a known half-overlap case.
  - nms: two highly-overlapping boxes with different scores -> only the higher kept.

Run: uv run pytest tests/test_boxes.py
"""

import pytest

pytestmark = pytest.mark.skip(reason="TODO: implement ops/boxes.py then write these tests")


def test_encode_decode_roundtrip():
    # TODO
    ...


def test_iou_basic():
    # TODO
    ...


def test_nms_suppresses_overlap():
    # TODO
    ...
