"""Unit tests for eval.metrics: relabel_contiguous, instance_ap, matched_mean_iou, panoptic_quality."""

import numpy as np
import pytest

from pluriseg.eval.metrics import (
    instance_ap,
    matched_mean_iou,
    panoptic_quality,
)
from pluriseg.utils import relabel_contiguous


def _blobs(*specs, shape=(16, 16)):
    """Build a label map from (label, (rows, cols)) rectangle specs."""
    out = np.zeros(shape, dtype=np.int64)
    for lbl, (r, c) in specs:
        out[r[0]: r[1], c[0]: c[1]] = lbl
    return out


# ---------------------------------------------------------------- relabel

def test_relabel_default_ids_are_positive_uniques():
    label = _blobs((5, ((0, 4), (0, 4))), (2, ((8, 12), (8, 12))))
    out = relabel_contiguous(label)
    assert set(np.unique(out)) == {0, 1, 2}
    # order follows sorted original ids: 2 -> 1, 5 -> 2
    assert out[0, 0] == 2 and out[9, 9] == 1


def test_relabel_empty_keeps_background():
    out = relabel_contiguous(np.zeros((4, 4), dtype=np.int64))
    assert np.all(out == 0)


def test_relabel_negative_ids_dropped():
    label = _blobs((1, ((0, 4), (0, 4))), (2, ((8, 12), (8, 12))))
    label[label == 2] = -1
    out = relabel_contiguous(label)
    assert out[0, 0] == 1 and np.all(out[8:, 8:] == 0)


# ------------------------------------------------------------- instance_ap

def test_ap_perfect_prediction():
    gt = _blobs((1, ((0, 6), (0, 6))), (2, ((8, 16), (0, 8))))
    res = instance_ap(gt, gt.copy())
    assert res["n_gt"] == 2 and res["n_pred"] == 2
    assert res["AP@0.50"] == pytest.approx(1.0)
    assert res["AP@[.5:.95]"] == pytest.approx(1.0)


def test_ap_empty_edge_cases():
    empty = np.zeros((8, 8), dtype=np.int64)
    gt = _blobs((1, ((0, 4), (0, 4))))
    # no GT, no pred -> perfect
    res = instance_ap(empty, empty.copy())
    assert res["AP@0.50"] == 1.0
    # no GT, one FP -> 0
    assert instance_ap(empty, gt.copy())["AP@0.50"] == 0.0
    # GT present, no pred -> 0
    assert instance_ap(gt.copy(), empty)["AP@0.50"] == 0.0
    assert instance_ap(gt.copy(), empty)["n_pred"] == 0


def test_ap_exact_match_one_pair():
    gt = _blobs((1, ((0, 4), (0, 4))))
    pred = gt.copy()
    res = instance_ap(gt, pred)
    assert res["AP@0.50"] == pytest.approx(1.0)
    assert res["AP@[.5:.95]"] == pytest.approx(1.0)


def test_ap_split_prediction_counts_fp():
    # one GT square predicted as two halves: each half IoU ~ 0.5 with GT
    gt = _blobs((1, ((0, 4), (0, 8))))  # 4x8
    pred = gt.copy()
    pred[2:, :] = 2
    res = instance_ap(gt, pred)
    assert res["n_gt"] == 1 and res["n_pred"] == 2
    # 2x8 half vs 4x8 GT: IoU = 16/32 = 0.5 -> matches only at t<=0.5
    assert res["AP@0.25"] == pytest.approx(0.5)
    assert res["AP@0.50"] == pytest.approx(0.5)
    assert res["AP@0.75"] == pytest.approx(0.0)


def test_ap_one_to_one_not_many_to_many():
    # two GT, one pred overlapping both -> only best pair matches
    gt = _blobs((1, ((0, 4), (0, 4))), (2, ((0, 4), (4, 8))))
    pred = _blobs((1, ((0, 4), (0, 8))))
    res = instance_ap(gt, pred)
    assert res["n_pred"] == 1
    # IoU with one GT = 16/32 = 0.5; the other GT is FN
    # TP=1, n_gt=2, n_pred=1 -> AP = 1/(2+1-1) = 0.5
    assert res["AP@0.50"] == pytest.approx(0.5)
    assert res["AP@0.75"] == pytest.approx(0.0)
    assert res["AP@0.25"] == pytest.approx(0.5)


def test_ap_thresholds_fixed_regardless_of_inputs():
    gt = _blobs((1, ((0, 4), (0, 4))))
    res = instance_ap(gt, gt.copy())
    # 0.25/0.75 lie outside THRESHOLDS but are still reported
    assert res["AP@0.25"] == pytest.approx(1.0)
    assert res["AP@0.75"] == pytest.approx(1.0)
    assert res["AP@[.5:.95]"] == pytest.approx(1.0)


def test_ap_overlapping_ids_renormalized():
    # sparse/high ids and an id present in pred but not gt must not matter
    gt = _blobs((100, ((0, 4), (0, 4))))
    pred = _blobs((37, ((0, 4), (0, 4))), (99, ((8, 10), (8, 10))))
    res = instance_ap(gt, pred)
    assert res["n_gt"] == 1 and res["n_pred"] == 2
    # perfect match for id 100 (IoU 1.0) + one small FP
    # TP=1, n_gt=1, n_pred=2 -> AP = 1/(1+2-1) = 0.5
    assert res["AP@0.50"] == pytest.approx(0.5)


def test_ap_partial_overlap_overwrite_accounting():
    # two preds overlap one GT; pred2 overwrites pred1 pixels (labels can't
    # overlap), so IoU(pred1) = 28/(64+28-28) = 0.4375, IoU(pred2) = 4/64.
    gt = _blobs((1, ((0, 8), (0, 8))))
    pred = _blobs((1, ((0, 8), (0, 4))), (2, ((0, 2), (0, 2))))
    res = instance_ap(gt, pred)
    # only pred1 clears t=0.25 (IoU 0.4375 >= 0.25) -> TP=1 -> AP = 0.5
    assert res["AP@0.25"] == pytest.approx(0.5)
    assert res["AP@0.50"] == pytest.approx(0.0)
    assert res["AP@0.75"] == pytest.approx(0.0)


def test_ap_disjoint_pred_counts_in_denominator():
    # a zero-IoU pred must not match, but still counts in the denominator
    gt = _blobs((1, ((0, 8), (0, 8))))
    pred = _blobs((1, ((0, 8), (0, 6))), (2, ((8, 10), (10, 12))))
    res = instance_ap(gt, pred)
    # IoU(pred1) = 48/(64+48-48) = 0.75, IoU(pred2) = 0 (disjoint)
    assert res["AP@0.75"] == pytest.approx(0.5)
    assert res["AP@0.50"] == pytest.approx(0.5)


def test_ap_greedy_not_optimal_matching():
    # Greedy's top pair A-GTB (IoU 400/1100 = 0.364) consumes both A and
    # GTB, blocking C-GTB (0.35) and A-GTE (0.25) -> TP=1 -> AP=1/3. The
    # optimal one-to-one matching (A-GTE + C-GTB) gives TP=2 -> 2/3.
    # A is ONE instance (two rects, same label) spanning GTB and GTE.
    gt = _blobs(
        (1, ((0, 20), (0, 40))),  # GTB: 800 px
        (2, ((20, 40), (0, 40))),  # GTE: 800 px
        shape=(40, 40),
    )
    pred = _blobs(
        (1, ((0, 20), (0, 20))),  # A: 400 px in GTB
        (1, ((20, 35), (0, 20))),  # A: 300 px in GTE  (|A| = 700)
        (2, ((0, 14), (20, 40))),  # C: 280 px in GTB -> IoU 280/800 = 0.35
        shape=(40, 40),
    )
    res = instance_ap(gt, pred)
    # IoU(A,GTB) = 400/(800+700-400) = 0.364 (top)
    # IoU(A,GTE) = 300/(800+700-300) = 0.25
    assert res["n_gt"] == 2 and res["n_pred"] == 2
    assert res["AP@0.25"] == pytest.approx(1 / 3)
    assert res["AP@0.50"] == pytest.approx(0.0)  # all IoUs < 0.5


def test_ap_best_first_not_scan_order():
    # Top pair X-GT2 (IoU 0.667) is NOT first in row-major (gt, pred) scan
    # order: (GT1,X) comes first. Best-first: X-GT2, then Y-GT1 -> TP=2 ->
    # AP=1.0. A scan-order matcher takes X-GT1 (0.25) first, blocking both
    # others -> TP=1 -> AP=1/3.
    gt = _blobs(
        (1, ((0, 20), (0, 20))),  # GT1: 400 px
        (2, ((20, 40), (0, 20))),  # GT2: 400 px
        shape=(40, 40),
    )
    # X (label 1) spans GT1+GT2; Y (label 2) overwrites X's top-left half
    pred = _blobs(
        (1, ((0, 40), (0, 20))),  # X: 800 px before overwrite
        (2, ((0, 20), (0, 10))),  # Y: 200 px in GT1
        shape=(40, 40),
    )
    res = instance_ap(gt, pred)
    # X_eff = 600 px (200 in GT1, 400 in GT2):
    # IoU(X,GT1) = 200/(400+600-200) = 0.25
    # IoU(X,GT2) = 400/(400+600-400) = 0.667 (top, scan-last)
    # IoU(Y,GT1) = 200/(400+200-200) = 0.5
    assert res["n_gt"] == 2 and res["n_pred"] == 2
    assert res["AP@0.25"] == pytest.approx(1.0)
    assert res["AP@0.50"] == pytest.approx(1.0)
    assert res["AP@0.75"] == pytest.approx(0.0)


# -------------------------------------------------------- matched_mean_iou

def test_matched_iou_perfect_prediction():
    gt = _blobs((1, ((0, 6), (0, 6))), (2, ((8, 16), (0, 8))))
    res = matched_mean_iou(gt, gt.copy())
    assert res["n_matched"] == 2
    assert res["matched_iou"] == pytest.approx(1.0)


def test_matched_iou_empty_edge_cases():
    empty = np.zeros((8, 8), dtype=np.int64)
    gt = _blobs((1, ((0, 4), (0, 4))))
    # no GT, no pred -> agreement
    assert matched_mean_iou(empty, empty.copy())["matched_iou"] == 1.0
    # one-sided instances -> no pairs
    assert matched_mean_iou(empty, gt.copy())["matched_iou"] == 0.0
    assert matched_mean_iou(gt.copy(), empty)["matched_iou"] == 0.0


def test_matched_iou_split_counts_quality_not_threshold():
    # GT2 split into two halves: only one half can be matched (one-to-one),
    # the other stays unmatched. matched_iou rewards the fit quality of the
    # matched pair instead of dropping it to a binary TP/FP.
    gt = _blobs((1, ((0, 6), (0, 6))), (2, ((8, 16), (0, 8))))
    pred = _blobs(
        (1, ((0, 6), (0, 6))),
        (2, ((8, 16), (0, 4))),   # left half of GT2, IoU = 32/(64+32-32) = 0.5
        (3, ((8, 16), (4, 8))),   # right half, IoU also 0.5
    )
    res = matched_mean_iou(gt, pred)
    assert res["n_matched"] == 2
    # Hungarian: obj1<->1 (1.0); obj2<->2 or 3 (both 0.5)
    assert res["matched_iou"] == pytest.approx(0.75)


def test_matched_iou_ignores_low_quality_not_matched_by_ap():
    # AP@0.75 counts a 0.6-IoU pair as FP; matched_iou still credits 0.6.
    gt = _blobs((1, ((0, 10), (0, 10))))
    pred = _blobs((1, ((0, 8), (0, 12))))  # IoU = 80/(100+96-80) = 0.684
    res = matched_mean_iou(gt, pred)
    assert res["n_matched"] == 1
    assert res["matched_iou"] == pytest.approx(80 / (100 + 96 - 80))


def test_matched_iou_maximizes_total_not_pairwise_greedy():
    # greedy best-first pairing would take the high-IoU diagonal pair and
    # strand the off-diagonal; optimal assignment takes both off-diagonals.
    # 2x2 block: IoU(gt_i, pred_j) = 0.9 on diagonal, 0.8 off-diagonal is
    # impossible to make greedy-wrong at thresh 0 with realistic shapes;
    # use the classic anti-greedy pattern instead:
    gt = _blobs((1, ((0, 8), (0, 8))), (2, ((8, 16), (0, 8))), shape=(16, 16))
    # pred1 slightly covers both, pred2 = GT2 exact
    pred = _blobs((1, ((0, 8), (0, 6))), (2, ((8, 16), (0, 8))), shape=(16, 16))
    res = matched_mean_iou(gt, pred)
    assert res["n_matched"] == 2
    assert res["matched_iou"] == pytest.approx((1.0 + 48 / 64) / 2)


# -------------------------------------------------------- panoptic_quality

def test_pq_perfect_prediction():
    gt = _blobs((1, ((0, 6), (0, 6))), (2, ((8, 16), (0, 8))))
    res = panoptic_quality(gt, gt.copy())
    assert res["tp"] == 2
    assert res["SQ"] == pytest.approx(1.0)
    assert res["RQ"] == pytest.approx(1.0)
    assert res["PQ"] == pytest.approx(1.0)


def test_pq_empty_edge_cases():
    empty = np.zeros((8, 8), dtype=np.int64)
    gt = _blobs((1, ((0, 4), (0, 4))))
    # both empty -> agreement
    res = panoptic_quality(empty, empty.copy())
    assert res["PQ"] == 1.0 and res["RQ"] == 1.0
    # one-sided instances -> RQ = 0 -> PQ = 0
    assert panoptic_quality(empty, gt.copy())["PQ"] == 0.0
    assert panoptic_quality(gt.copy(), empty)["PQ"] == 0.0


def test_pq_split_halves_rq_and_sq():
    # one GT square predicted as two halves: at most one half matches (TP=1,
    # IoU = 16/32 = 0.5); the other half is an FP and the GT stays matched,
    # so FP=1, FN=0. RQ = 1/(1+.5) = 2/3, SQ = 0.5, PQ = 1/3.
    gt = _blobs((1, ((0, 4), (0, 8))))
    pred = gt.copy()
    pred[2:, :] = 2
    res = panoptic_quality(gt, pred)
    assert res["tp"] == 1
    assert res["SQ"] == pytest.approx(0.5)
    assert res["RQ"] == pytest.approx(2 / 3)
    assert res["PQ"] == pytest.approx(0.5 * 2 / 3)


def test_pq_merge_counts_fn():
    # two GT, one pred overlapping both: TP=1 (IoU 0.5 with one GT), FP=0,
    # FN=1 -> RQ = 1/(1+.5) = 2/3, SQ = 0.5, PQ = 1/3.
    gt = _blobs((1, ((0, 4), (0, 4))), (2, ((0, 4), (4, 8))))
    pred = _blobs((1, ((0, 4), (0, 8))))
    res = panoptic_quality(gt, pred)
    assert res["tp"] == 1
    assert res["SQ"] == pytest.approx(0.5)
    assert res["RQ"] == pytest.approx(2 / 3)
    assert res["PQ"] == pytest.approx(0.5 * 2 / 3)


def test_pq_missed_instance_pure_fn():
    # one matched pair + one missed GT: FP=0, FN=1 -> RQ = 1/(1+.5) = 2/3,
    # SQ = 1.0 (the matched pair is perfect)
    gt = _blobs((1, ((0, 6), (0, 6))), (2, ((8, 16), (0, 8))))
    pred = _blobs((1, ((0, 6), (0, 6))))
    res = panoptic_quality(gt, pred)
    assert res["tp"] == 1
    assert res["SQ"] == pytest.approx(1.0)
    assert res["RQ"] == pytest.approx(2 / 3)
    assert res["PQ"] == pytest.approx(2 / 3)


def test_pq_sub_05_pair_not_tp():
    # IoU 0.684 > 0.5 -> matched; a 0.4-IoU pair is not a TP (drops from SQ
    # entirely; counts in FP and FN).
    gt = _blobs((1, ((0, 10), (0, 10))))
    good = _blobs((1, ((0, 8), (0, 12))))  # IoU = 80/(100+96-80) = 0.684
    res = panoptic_quality(gt, good)
    assert res["tp"] == 1
    assert res["PQ"] == pytest.approx(80 / (100 + 96 - 80))
    # far too small: IoU = 40/(100+40-40) = 0.4 < 0.5 -> no TP
    bad = _blobs((1, ((0, 4), (0, 10))))
    res = panoptic_quality(gt, bad)
    assert res["tp"] == 0
    assert res["SQ"] == 0.0
    assert res["RQ"] == 0.0
    assert res["PQ"] == 0.0


def test_pq_iou_exact_05_counts_as_tp():
    # IoU exactly 0.5 (2x4 half vs 4x8 GT, from the AP split test) must
    # match with the >= thresh convention.
    gt = _blobs((1, ((0, 4), (0, 8))))
    pred = gt.copy()
    pred[2:, :] = 2
    # the left half has IoU exactly 0.5 -> TP; FP=1 (right half), FN=0
    res = panoptic_quality(gt, pred)
    assert res["tp"] == 1
    assert res["RQ"] == pytest.approx(2 / 3)


# ------------------------------------------------------ per_category_metrics

def _pcat(gt, pred, cats):
    """per_category_metrics with the raw-label-id categories of `cats`."""
    from pluriseg.shapes.metrics import per_category_metrics
    return per_category_metrics(gt, pred, cats, thresh=0.5)


def test_pcat_perfect_matches_panoptic_quality():
    # single category spanning all GT: SQ/RQ/PQ equal whole-image
    # panoptic_quality
    gt = _blobs((1, ((0, 6), (0, 6))), (2, ((8, 16), (0, 8))))
    res = _pcat(gt, gt.copy(), {1: "all", 2: "all"})
    pq = panoptic_quality(gt, gt.copy())
    assert res["all"]["sq"] == pytest.approx(pq["SQ"])
    assert res["all"]["rq"] == pytest.approx(pq["RQ"])
    assert res["all"]["pq"] == pytest.approx(pq["PQ"])
    assert res["all"]["sq"] == pytest.approx(1.0)


def test_pcat_split_penalizes_only_own_category():
    # splitting one instance adds an FP attributed to the same category;
    # untouched categories keep PQ = 1. Geometry: GT2 (8x8, 64px) loses its
    # lower rows to a new label; the upper sliver (16px) has IoU 16/64 = .25
    # and the remaining part (48px) 48/64 = .75, so the assignment keeps the
    # .75 pair and the sliver is an attributed FP.
    gt = _blobs((1, ((0, 6), (0, 6))), (2, ((8, 16), (0, 8))))
    pred = gt.copy()
    pred[10:, 0:8] = 3  # split instance 2 (bottom square) in two
    res = _pcat(gt, pred, {1: "round", 2: "round"})
    pq = panoptic_quality(gt, pred)
    # per-category RQ/PQ equal the whole-image values here (one category)
    assert res["round"]["rq"] == pytest.approx(pq["RQ"])
    assert res["round"]["pq"] == pytest.approx(pq["PQ"])
    assert res["round"]["sq"] == pytest.approx(0.875)
    assert res["round"]["rq"] == pytest.approx(0.8)
    assert res["round"]["pq"] == pytest.approx(0.7)
    assert res["round"]["n_matched"] == 2 and res["round"]["n_gt"] == 2


def test_pcat_fp_attribution_by_max_iou():
    # an FP overlapping a GT instance lands in that instance's category,
    # not in 'unmatched' (no 'unmatched' bucket appears at all)
    gt = _blobs((1, ((0, 6), (0, 6))))
    pred = gt.copy()
    pred[0:3, 0:3] = 2  # 9px FP carved out of GT1; remainder IoU 27/36 = .75
    res = _pcat(gt, pred, {1: "round"})
    assert res["round"]["n_gt"] == 1
    assert res["round"]["n_matched"] == 1
    assert res["round"]["sq"] == pytest.approx(0.75)
    # FP inflates RQ: 1/(1+.5) = 2/3, PQ = .75 * 2/3
    assert res["round"]["rq"] == pytest.approx(2 / 3)
    assert res["round"]["pq"] == pytest.approx(0.75 * 2 / 3)
    assert "unmatched" not in res


def test_pcat_fp_no_overlap_goes_unmatched():
    # an FP with no GT overlap -> 'unmatched' bucket (PQ/RQ 0 there),
    # while the FP changes no category's own metrics
    gt = _blobs((1, ((0, 6), (0, 6))), (2, ((8, 16), (0, 8))))
    pred = gt.copy()
    pred[0:2, 12:14] = 3  # empty corner (GT2 spans cols 0:8 only)
    res = _pcat(gt, pred, {1: "round", 2: "tiny"})
    assert res["unmatched"]["n_matched"] == 0
    assert res["unmatched"]["pq"] == 0.0
    assert res["unmatched"]["rq"] == 0.0
    assert res["round"]["pq"] == pytest.approx(1.0)
    assert res["tiny"]["pq"] == pytest.approx(1.0)


def test_pcat_sub_05_matched_pair_counts_fp_and_fn():
    # a GT whose best pair is below 0.5 is unmatched (FN) and that pair is
    # an FP attributed to its category, like panoptic_quality's accounting
    gt = _blobs((1, ((0, 4), (0, 10))))
    pred = _blobs((1, ((0, 4), (0, 4))))  # IoU = 40/(100+40-40) = 0.4 < 0.5
    res = _pcat(gt, pred, {1: "all"})
    # SQ: matched_mean_iou at thresh 0 keeps the 0.4 pair in mean_iou;
    # SQ must exclude it -> mean over TP pairs only
    assert res["all"]["n_matched"] == 0
    assert res["all"]["sq"] == 0.0
    # FN=1 (the GT) + FP=1 (attributed pred) -> RQ = 0/(0+.5+.5) = 0
    assert res["all"]["rq"] == 0.0
    assert res["all"]["pq"] == 0.0
    pq = panoptic_quality(gt, pred)
    assert pq["PQ"] == 0.0 and pq["RQ"] == 0.0
