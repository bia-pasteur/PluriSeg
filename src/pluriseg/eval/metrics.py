"""Evaluation metrics: GT instance labels vs predicted instance labels.

Skeleton pass: signatures + docstrings only. The real implementations
(AP@0.5, AP@[.5:.95], per-image DSC/F1) land in a later pass; the stubs
below return placeholder structures so the validation loop can be wired
end-to-end without a working metric.
"""

from __future__ import annotations

import numpy as np

from scipy.optimize import linear_sum_assignment

from pluriseg.utils import relabel_contiguous


#: IoU thresholds for the reported metrics. Fixed for comparability across
#: heads/datasets; not exposed as a function argument.
THRESHOLDS = np.arange(0.5, 1.0, 0.05)


def iou_matrix(
    gt: np.ndarray,
    pred: np.ndarray,
) -> np.ndarray:
    """Pairwise IoU between instances of two label maps, shape (n_gt, n_pred).

    0 where no overlap. Instance extents are full (not restricted to the
    overlap mask), so IoU is correct when one object lies entirely inside
    a larger one.
    """
    g = relabel_contiguous(gt).ravel()
    p = relabel_contiguous(pred).ravel()
    n_gt, n_pred = int(g.max()), int(p.max())
    if n_gt == 0 or n_pred == 0:
        return np.zeros((n_gt, n_pred), dtype=np.float64)
    fg = (g > 0) & (p > 0)
    area_gt = np.bincount(g[g > 0], minlength=n_gt + 1)[1:].astype(np.float64)
    area_pred = np.bincount(
        p[p > 0], minlength=n_pred + 1)[1:].astype(np.float64)
    overlap = np.bincount(
        (g[fg] - 1) * n_pred + (p[fg] - 1), minlength=n_gt * n_pred
    ).reshape(n_gt, n_pred).astype(np.float64)
    union = area_gt[:, None] + area_pred[None, :] - overlap
    return np.where(overlap > 0, overlap / np.maximum(union, 1e-12), 0.0)


def instance_ap(
    gt: np.ndarray,
    pred: np.ndarray,
) -> dict[str, float]:
    """Instance-level average precision between GT and predicted label maps.

    Greedy one-to-one IoU matching, best-IoU pairs first; without confidence
    scores there is no PR curve, so AP at threshold t is TP/(TP+FP+FN)
    (cellpose/stardist convention).

    Args:
        gt: (H, W) integer instance label map (0 = background).
        pred: (H, W) integer instance label map (0 = background).

    Returns:
        dict with keys "AP@0.25", "AP@0.50", "AP@0.75", "AP@[.5:.95]"
        (mean over THRESHOLDS), "n_gt", "n_pred".
    """
    thresholds = THRESHOLDS

    g = relabel_contiguous(gt).ravel()
    p = relabel_contiguous(pred).ravel()
    n_gt, n_pred = int(g.max()), int(p.max())

    out: dict[str, float] = {"n_gt": float(n_gt), "n_pred": float(n_pred)}

    def _fill(value: float) -> dict[str, float]:
        out.update(
            {
                "AP@0.25": value,
                "AP@0.50": value,
                "AP@0.75": value,
                "AP@[.5:.95]": value,
            }
        )
        return out

    if n_gt == 0:
        # No instances to find: perfect if pred is empty, else all FPs.
        return _fill(1.0 if n_pred == 0 else 0.0)
    if n_pred == 0:
        return _fill(0.0)

    iou = iou_matrix(gt, pred)

    # Greedy one-to-one matching per threshold, best-IoU pairs first.
    # 0.25/0.5/0.75 are matched separately so they are always reported,
    # even though 0.25/0.75 fall outside THRESHOLDS.
    r, c = np.nonzero(iou > 0)
    order = np.argsort(-iou[r, c])
    r, c = r[order], c[order]

    extra = [t for t in (0.25, 0.50, 0.75) if t not in thresholds]
    all_t = np.sort(np.concatenate([thresholds, extra]))

    def _match(t: float) -> int:
        used_gt = np.zeros(n_gt, dtype=bool)
        used_pred = np.zeros(n_pred, dtype=bool)
        tp = 0
        for i, j in zip(r.tolist(), c.tolist()):
            if iou[i, j] < t:
                break
            if not used_gt[i] and not used_pred[j]:
                used_gt[i] = used_pred[j] = True
                tp += 1
        return tp

    ap = {t: _match(t) / (n_gt + n_pred - _match(t)) for t in all_t}

    out["AP@0.25"] = float(ap[0.25])
    out["AP@0.50"] = float(ap[0.50])
    out["AP@0.75"] = float(ap[0.75])
    out["AP@[.5:.95]"] = float(np.mean([ap[t] for t in thresholds]))
    return out


def matched_mean_iou(
    gt: np.ndarray,
    pred: np.ndarray,
    thresh: float = 0.0,
) -> dict[str, float]:
    """Mean IoU over Hungarian-matched instance pairs (quality metric).

    Pairs GT and predicted instances one-to-one (max-total-IoU assignment);
    pairs with IoU below `thresh` are dropped, and the score is the mean
    IoU of what remains. Because unmatched instances contribute nothing to
    the average, the score measures segmentation quality, not detection --
    read it together with "n_matched" vs "n_gt"/"n_pred". Default
    `thresh=0` keeps every matched pair; `thresh=0.5` gives the quality
    term of panoptic quality. Whole-image metric; per-category versions
    live in shapes/metrics.py.

    Empty GT and empty pred -> 1.0 (agreement); instances on one side only
    -> 0.0 (no pairs to average).

    Returns:
        dict with keys "matched_iou" (the score), "n_matched", "n_gt", "n_pred".
    """
    g = relabel_contiguous(gt).ravel()
    p = relabel_contiguous(pred).ravel()
    n_gt, n_pred = int(g.max()), int(p.max())

    out: dict[str, float] = {"n_gt": float(n_gt), "n_pred": float(n_pred)}
    if n_gt == 0 or n_pred == 0:
        out["matched_iou"] = 1.0 if n_gt == n_pred else 0.0
        out["n_matched"] = 0.0
        return out

    iou = iou_matrix(gt, pred)
    r, c = match_iou(gt, pred, thresh=thresh)
    vals = iou[r, c]
    out["n_matched"] = float(len(vals))
    out["matched_iou"] = float(vals.mean()) if len(vals) else 0.0
    return out


def panoptic_quality(
    gt: np.ndarray,
    pred: np.ndarray,
) -> dict[str, float]:
    """Panoptic quality (Kirillov 2019) between GT and predicted label maps.

    Hungarian one-to-one matching; pairs with IoU >= 0.5 count as TP
    (matches match_iou's >= convention; Kirillov's "> 0.5" differs only on
    the measure-zero IoU == 0.5 tie). SQ = mean IoU over TP pairs, RQ =
    TP/(TP + 0.5FP + 0.5FN), PQ = SQ * RQ. Unmatched instances penalize via
    RQ only, matched pairs below 0.5 do not contribute (they are neither TP
    nor FP/FN — they inflate both FP and FN side counts in this per-image
    aggregate).

    Empty GT and empty pred -> 1.0 (agreement, instance_ap convention);
    instances on one side only -> RQ = 0, so PQ = 0.

    Returns:
        dict with keys "PQ", "SQ", "RQ", "tp", "n_gt", "n_pred".
    """
    miou0 = matched_mean_iou(gt, pred)
    n_gt, n_pred = int(miou0["n_gt"]), int(miou0["n_pred"])
    if n_gt == 0 and n_pred == 0:
        return {"PQ": 1.0, "SQ": 1.0, "RQ": 1.0, "tp": 0.0,
                "n_gt": 0.0, "n_pred": 0.0}

    r, c = match_iou(gt, pred, thresh=0.5)
    iou = iou_matrix(gt, pred)
    tp = len(r)
    sq = float(iou[r, c].mean()) if tp else 0.0
    fp, fn = n_pred - tp, n_gt - tp
    rq = tp / (tp + 0.5 * fp + 0.5 * fn)
    return {"PQ": sq * rq, "SQ": sq, "RQ": float(rq), "tp": float(tp),
            "n_gt": float(n_gt), "n_pred": float(n_pred)}


def match_iou(
    a: np.ndarray,
    b: np.ndarray,
    thresh: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """One-to-one instance matching between two label maps (Hungarian).

    Maximizes total IoU; pairs with IoU < thresh are dropped afterwards
    (assignment is global-optimal, greedy-per-threshold is not).

    Returns (row, col) index arrays; instance ids are 0-based matrix indices
    (label id - 1).
    """
    iou = iou_matrix(a, b)
    if min(iou.shape) == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    r, c = linear_sum_assignment(-iou)
    keep = iou[r, c] >= thresh
    return r[keep], c[keep]


def dsc_f1(
    gt: np.ndarray,
    pred: np.ndarray,
) -> dict[str, float]:
    """Per-image semantic Dice/F1 (pixel-level, threshold=0.5 semantics)."""
    raise NotImplementedError("dsc_f1")
