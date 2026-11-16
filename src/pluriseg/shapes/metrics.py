import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from scipy.optimize import linear_sum_assignment

from pluriseg.eval.metrics import THRESHOLDS, iou_matrix

rprint = Console().print


def per_category_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
    categories: pd.Series,
    thresh: float = 0.5,
    ap_thresholds: tuple[float, ...] = (),
    ap_coco: bool = False,
) -> dict[str, dict[str, float]]:
    """Segmentation performance per GT shape category for one image.

    Matches pred instances to GT with Hungarian IoU (eval.metrics.match_iou
    semantics via linear_sum_assignment on iou_matrix); each matched pair
    inherits the GT instance's category from `categories` (a Series indexed
    by label_id, e.g. a categorizer().categorize(feats) column).

    Unmatched predictions are attributed to the GT instance with the max
    IoU over it (else the 'unmatched' bucket) so over-segmentation costs
    the category it came from.

    The assignment does not depend on the acceptance threshold, so all
    `ap_thresholds` are scored from the same matching; each adds an
    'ap@t' column next to the base 'ap' (= ap@thresh). If
    `ap_coco` is set, an 'ap@[0.5:0.95]' column holds the average of the
    COCO-style thresholds 0.50..0.95 in steps of 0.05.

    The same matching yields per-category panoptic quality (eval.metrics.
    panoptic_quality formulas applied per category): sq = mean IoU over
    matched pairs (= mean_iou), rq = TP / (TP + .5*FP + .5*FN) with FP
    from the attribution rule above and FN the category's GT left
    unmatched, pq = sq * rq. Sub-thresh matched pairs count on both the
    FP and FN sides, as in panoptic_quality; per-category RQ/PQ differ
    from the whole-image values (category-local denominators) and only
    sum back to panoptic_quality when a single category spans all GT.

    Returns:
        {category: {"n_gt", "n_matched", "mean_iou", "sq", "rq", "pq",
        "ap"[, "ap@t"...][, "ap@[0.5:0.95]"]}};
        'unmatched' holds FP predictions with no overlapping GT.
    """
    gt = np.asarray(gt)
    pred = np.asarray(pred)
    iou = iou_matrix(gt, pred)
    n_gt, n_pred = iou.shape

    # thresholds for per-threshold matched-GT counts
    # (dedup: THRESHOLDS and ap_thresholds may overlap)
    thresholds = tuple(dict.fromkeys(ap_thresholds + tuple(THRESHOLDS))) \
        if ap_coco else tuple(ap_thresholds)

    # iou_matrix relabels internally, so its row i is 0-based and ordered by
    # ascending original id; map back via unique positive ids of gt.
    gt_ids = np.unique(gt[gt > 0])
    if len(gt_ids) != n_gt:
        raise ValueError(
            "categories must be indexed by the raw label ids of `gt`"
        )
    cat_of_gt = {int(lbl): cat for lbl, cat in categories.items()}

    r, c = linear_sum_assignment(-iou)

    # per-threshold matched-GT mask: same assignment, other cutoff
    t_matched = {
        t: (iou[r, c] >= t) for t in thresholds
    }

    matched_pred = np.zeros(n_pred, dtype=bool)
    matched_gt = np.zeros(n_gt, dtype=bool)
    match_iou_val = {}
    for i, j in zip(r.tolist(), c.tolist()):
        # a pair counts as a match only if its IoU clears `thresh`
        if iou[i, j] < thresh:
            continue
        matched_gt[i] = True
        matched_pred[j] = True
        match_iou_val[i] = float(iou[i, j])

    # attribution of unmatched preds: argmax GT IoU, else 'unmatched'
    attribution = np.full(n_pred, "unmatched", dtype=object)
    for j in range(n_pred):
        if matched_pred[j]:
            continue
        col = iou[:, j]
        if col.max() > 0:
            attribution[j] = cat_of_gt.get(
                int(gt_ids[int(np.argmax(col))]), "unmatched"
            )

    stats: dict[str, dict[str, float]] = {}

    def _bucket(cat: str) -> dict[str, float]:
        return stats.setdefault(
            cat, {"n_gt": 0, "n_matched": 0, "iou_sum": 0.0,
                  "fp": 0, "t_matched": {t: 0 for t in thresholds}}
        )

    for i in range(n_gt):
        cat = cat_of_gt.get(int(gt_ids[i]), "unmatched")
        s = _bucket(cat)
        s["n_gt"] += 1
        if matched_gt[i]:
            s["n_matched"] += 1
            s["iou_sum"] += match_iou_val[i]
    for j in range(n_pred):
        if not matched_pred[j]:
            _bucket(attribution[j])["fp"] += 1
    # per-threshold matched-GT counts per category
    for t, mask in t_matched.items():
        for i in r[mask].tolist():
            _bucket(cat_of_gt.get(int(gt_ids[i]), "unmatched"))["t_matched"][t] += 1

    out = {}
    for cat, s in stats.items():
        denom = s["n_gt"] + s["fp"]
        miou = s["iou_sum"] / s["n_matched"] if s["n_matched"] else 0.0
        rq_denom = s["n_matched"] + 0.5 * s["fp"] \
            + 0.5 * (s["n_gt"] - s["n_matched"])
        rq = s["n_matched"] / rq_denom if rq_denom else 1.0
        out[cat] = {
            "n_gt": s["n_gt"],
            "n_matched": s["n_matched"],
            "mean_iou": miou,
            "sq": miou,
            "rq": rq,
            "pq": miou * rq,
            "ap": s["n_matched"] / denom if denom else 1.0,
            **{
                f"ap@{t}": s["t_matched"][t] / denom if denom else 1.0
                for t in ap_thresholds
            },
        }
        if ap_coco:
            out[cat]["ap@[0.5:0.95]"] = float(np.mean([
                s["t_matched"][t] / denom if denom else 1.0
                for t in THRESHOLDS]))
    return out


def per_category_table(rows: list[dict[str, dict[str, float]]]) -> None:
    """Aggregate a list of per_category_metrics outputs and rich-print it."""
    df = pd.DataFrame(
        [dict(category=cat, **m) for row in rows for cat, m in row.items()]
    )
    if df.empty:
        rprint("[yellow]per_category_table: no rows[/yellow]")
        return
    agg = df.groupby("category").agg(
        n_gt=("n_gt", "sum"),
        n_matched=("n_matched", "sum"),
        mean_iou=("mean_iou", "mean"),
        ap=("ap", "mean"),
        **({"ap@[0.5:0.95]": ("ap@[0.5:0.95]", "mean")}
           if "ap@[0.5:0.95]" in df.columns else {}),
        **({c: (c, "mean") for c in ("sq", "rq", "pq")
            if c in df.columns}),
    )
    table = Table(title="Per-category segmentation performance")
    for col in agg.reset_index().columns:
        table.add_column(str(col))
    for row in agg.reset_index().itertuples(index=False):
        table.add_row(*[f"{v:.3f}" if isinstance(v, float)
                      else str(v) for v in row])
    rprint(table)


def plot_category_radar(
    rows: list[dict[str, dict[str, float]]],
    metric: str = "mean_iou",  # or "ap"
    ax=None,
    color: str | None = None,  # one run's color, for multi-run overlays
    label: str = "",  # legend entry, e.g. the run name
):
    """Polar (radar) plot of a per-category metric, 'unmatched' skipped.

    Args:
        rows: list of per_category_metrics outputs (one per image), as
            produced by tools/shape_metrics.py.
        metric: column to plot, "mean_iou" or "ap".
        ax: optional polar axes to draw into (for overlays of several runs
            on one figure); created otherwise.
        color/label: line color and legend text, useful when overlaying
            several runs on the same axes.

    Returns:
        the (possibly newly created) axes, for further overlays or saving.
    """
    import matplotlib.pyplot as plt

    df = pd.DataFrame(
        [dict(category=cat, **m) for row in rows for cat, m in row.items()]
    )
    if df.empty:
        rprint("[yellow]plot_category_radar: no rows[/yellow]")
        return ax
    agg = df.groupby("category").agg(
        n_gt=("n_gt", "sum"), value=(metric, "mean")
    ).drop(index="unmatched", errors="ignore")
    cats = list(agg.index)
    values = agg["value"].to_numpy()

    # close the polygon: first angle repeats at the end
    n = len(cats)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    values = np.append(values, values[0])

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5), subplot_kw={"projection": "polar"})
    ax.plot(angles, values, color=color, label=label or None,
            marker="o", markersize=3)
    ax.fill(angles, values, color=color, alpha=0.15)
    # category names around the circle (with instance counts)
    labels = [f"{c}\nn={int(agg.loc[c, 'n_gt'])}" for c in cats]
    ax.set_xticks(angles[:-1], labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", ""], fontsize=6)
    if label:
        ax.legend(loc="lower right", bbox_to_anchor=(1.15, -0.1))
    return ax