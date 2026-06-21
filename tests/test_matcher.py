"""Tests for ops/matcher.py.

Run: uv run pytest tests/test_matcher.py
"""

import torch

from objdet.ops.matcher import match, sample_minibatch


def make_labels(num_pos: int, num_neg: int, num_ignore: int = 0) -> torch.Tensor:
    """Build a shuffled label vector: >=0 positive, -1 background, -2 ignore."""
    labels = torch.cat([
        torch.zeros(num_pos, dtype=torch.long),            # positives (GT idx 0)
        torch.full((num_neg,), -1, dtype=torch.long),      # background
        torch.full((num_ignore,), -2, dtype=torch.long),   # ignore
    ])
    return labels[torch.randperm(labels.numel())]


# Convention under test (from match's docstring):
#   matches[i] >= 0 -> anchor i is positive, value = matched GT index
#   matches[i] == -1 -> background (negative)
#   matches[i] == -2 -> ignore


def test_three_band_assignment():
    # iou[g, a]: 2 GTs x 4 anchors.
    iou = torch.tensor([[0.80, 0.40, 0.10, 0.05],
                        [0.20, 0.50, 0.60, 0.00]])
    m = match(iou, high_thresh=0.7, low_thresh=0.3, allow_low_quality=True)
    # a0: 0.80 >= 0.7 -> positive, GT0
    # a1: best 0.40 in [0.3,0.7) -> ignore
    # a2: best 0.60 in [0.3,0.7) -> ignore, BUT it's GT1's best anchor -> forced positive GT1
    # a3: best 0.05 < 0.3 -> background
    assert m.tolist() == [0, -2, 1, -1]


def test_no_low_quality_leaves_ignore():
    iou = torch.tensor([[0.80, 0.40, 0.10, 0.05],
                        [0.20, 0.50, 0.60, 0.00]])
    m = match(iou, 0.7, 0.3, allow_low_quality=False)
    # without the rescue, a2 stays ignore.
    assert m.tolist() == [0, -2, -2, -1]


def test_positive_keeps_correct_gt_index():
    # 3 GTs; each anchor strongly overlaps a different GT.
    iou = torch.tensor([[0.9, 0.0, 0.0],
                        [0.0, 0.8, 0.0],
                        [0.0, 0.0, 0.95]])
    m = match(iou, 0.7, 0.3, allow_low_quality=False)
    assert m.tolist() == [0, 1, 2]  # not all "0" — the GT index is preserved


def test_low_quality_rescues_below_high_thresh():
    # The single GT's best anchor (0.55) is below high_thresh=0.7.
    # Without rescue it'd be ignore; with rescue it must become positive.
    iou = torch.tensor([[0.55, 0.20, 0.10]])
    m_rescue = match(iou, 0.7, 0.3, allow_low_quality=True)
    m_norescue = match(iou, 0.7, 0.3, allow_low_quality=False)
    # a0=0.55 in [0.3,0.7) -> ignore band; a1=0.20 and a2=0.10 < 0.3 -> background.
    assert m_rescue.tolist() == [0, -1, -1]     # a0 forced positive by rescue
    assert m_norescue.tolist() == [-2, -1, -1]  # without rescue a0 only reaches ignore


def test_ties_force_all_max_anchors():
    # Two anchors tie for the GT's max IoU (0.5) -> BOTH should be forced positive.
    iou = torch.tensor([[0.5, 0.5, 0.1]])
    m = match(iou, 0.7, 0.3, allow_low_quality=True)
    assert m[0].item() == 0
    assert m[1].item() == 0
    assert m[2].item() == -1


def test_degenerate_gt_not_force_assigned():
    # GT1 has zero IoU with every anchor (highest_per_gt == 0).
    # The mask must prevent force-assigning all those zero-overlap anchors to GT1.
    iou = torch.tensor([[0.9, 0.4, 0.05],
                        [0.0, 0.0, 0.00]])
    m = match(iou, 0.7, 0.3, allow_low_quality=True)
    # No anchor should be matched to GT1.
    assert (m == 1).sum().item() == 0
    # a0 still positive to GT0; a1 ignore; a2 background.
    assert m.tolist() == [0, -2, -1]


def test_empty_gt_all_background():
    m = match(torch.zeros(0, 5), 0.7, 0.3)
    assert m.tolist() == [-1, -1, -1, -1, -1]


def test_return_dtype_is_long():
    # downstream code indexes GT boxes with these values -> must be integer/long.
    iou = torch.tensor([[0.8, 0.1]])
    assert match(iou, 0.7, 0.3).dtype == torch.long
    assert match(torch.zeros(0, 2), 0.7, 0.3).dtype == torch.long  # empty path too


def test_boundary_high_thresh_inclusive():
    # exactly == high_thresh should count as positive (>= convention).
    iou = torch.tensor([[0.7, 0.3]])
    m = match(iou, 0.7, 0.3, allow_low_quality=False)
    assert m[0].item() == 0     # 0.7 >= high -> positive
    assert m[1].item() == -2    # 0.3 in [0.3, 0.7) -> ignore (>= low, < high)


# --- sample_minibatch -------------------------------------------------------

def test_respects_size_and_fraction_when_plentiful():
    # Plenty of both -> exact target split.
    labels = make_labels(num_pos=500, num_neg=5000)
    pos, neg = sample_minibatch(labels, batch_size=256, positive_fraction=0.5)
    assert pos.numel() == 128          # round(256 * 0.5)
    assert neg.numel() == 128          # remainder
    assert pos.numel() + neg.numel() == 256


def test_fraction_quarter():
    labels = make_labels(num_pos=500, num_neg=5000)
    pos, neg = sample_minibatch(labels, batch_size=128, positive_fraction=0.25)
    assert pos.numel() == 32           # round(128 * 0.25)
    assert neg.numel() == 96


def test_samples_drawn_from_correct_pools():
    labels = make_labels(num_pos=200, num_neg=2000, num_ignore=100)
    pos, neg = sample_minibatch(labels, batch_size=256, positive_fraction=0.5)
    assert bool((labels[pos] >= 0).all())     # positives only
    assert bool((labels[neg] == -1).all())    # background only
    chosen = torch.cat([pos, neg])
    assert bool((labels[chosen] != -2).all())  # never sample an ignore


def test_indices_are_distinct():
    labels = make_labels(num_pos=200, num_neg=2000)
    pos, neg = sample_minibatch(labels, batch_size=256, positive_fraction=0.5)
    assert pos.unique().numel() == pos.numel()
    assert neg.unique().numel() == neg.numel()
    # pos and neg index sets must not overlap
    assert torch.isin(pos, neg).sum().item() == 0


def test_scarce_positives_backfill_with_negatives():
    # only 30 positives but we want 128 -> cap at 30, fill rest with negatives.
    labels = make_labels(num_pos=30, num_neg=2000)
    pos, neg = sample_minibatch(labels, batch_size=256, positive_fraction=0.5)
    assert pos.numel() == 30
    assert neg.numel() == 226
    assert pos.numel() + neg.numel() == 256


def test_fewer_total_than_batch_returns_all_available():
    # both pools small -> can't reach batch_size; return everything.
    labels = make_labels(num_pos=10, num_neg=20)
    pos, neg = sample_minibatch(labels, batch_size=256, positive_fraction=0.5)
    assert pos.numel() == 10
    assert neg.numel() == 20


def test_all_ignore_returns_empty():
    labels = make_labels(num_pos=0, num_neg=0, num_ignore=50)
    pos, neg = sample_minibatch(labels, batch_size=256, positive_fraction=0.5)
    assert pos.numel() == 0
    assert neg.numel() == 0


def test_sampling_is_randomized():
    # two draws on a large pool should (almost surely) differ -> confirms shuffling.
    labels = make_labels(num_pos=1000, num_neg=5000)
    torch.manual_seed(0)
    a, _ = sample_minibatch(labels, 256, 0.5)
    torch.manual_seed(1)
    b, _ = sample_minibatch(labels, 256, 0.5)
    assert not torch.equal(a.sort().values, b.sort().values)
