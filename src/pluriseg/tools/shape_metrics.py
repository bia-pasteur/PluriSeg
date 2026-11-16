"""Per-category segmentation metrics for a pattern of trained runs.

compute: for every run matching a glob under the runs root, rebuilds the
rule-based categories on the test split's
GT instances (shared feature cache with pluriseg.shapes.features), then
scores the run's saved eval predictions (<cache_id>_pred.tif) with
shapes.metrics.per_category_metrics. Saves one raw-data CSV per run at
results/shape-metrics/{dataset}_{model}_{runname}_metrics.csv plus a
sidecar {name}.meta.yaml carrying dataset/model/run/fold/seed (filenames keep
single underscores; plot() reads the sidecar, so no name parsing), and
n_examples overlay crops per category per run under
results/shape-metrics/examples/ (image with GT boundaries dim, the
sampled instance's boundary yellow, prediction boundaries magenta —
same sampled instances for every run, so models compare directly).

plot: polar (radar) comparison of the per-object metrics pooled across
datasets — one figure per metric (AP@0.5, PQ, SQ, RQ), all models on the
same axis; each point is the pooled mean per (model, category) over the
best run of every (model, dataset) cell (drop_zeros removes failed runs,
one_seed picks the best-PQ run per cell, repeats compete but never
average); plus a rich report table and a {pattern-stem}_summary.csv (one
row per model x category, mean +- std across (dataset, repeat) cells).
by_dataset adds one radar set per dataset, pooled within that dataset
only (the per-modality standings next to the pooled figure). The pooled
figure and summary are always saved twice: image-weighted (default
names) and equal-dataset (_equal suffix — every dataset counts once).
"""

from fnmatch import fnmatch
from pathlib import Path

import imageio.v3 as iio
import matplotlib as mpl
import numpy as np
import pandas as pd
import re
import yaml
from rich import print as rprint
from rich.table import Table
from scipy.ndimage import binary_erosion

from pluriseg.configs import DatasetConfig
from pluriseg.display import plt_be_stilish
from pluriseg.paths import RESULTS, SPLITS, DEFAULT_ROOT_MAP
from pluriseg.shapes.cleanup import DEFAULT_OPENING_RADIUS, get_clean_label
from pluriseg.shapes.examples import pick_category_examples
from pluriseg.shapes.features import dataset_shape_features
from pluriseg.shapes.metrics import per_category_metrics, per_category_table
from pluriseg.shapes.rules import RuleShapeCategorizer, ShapeRule
from pluriseg.splits import load_split
from pluriseg.utils import (
    _expected_repeats, _parse_config, _repeat_col,
    _run_pq,
)
from pluriseg.utils import load_masks

#: radar metrics plot() draws; (title, column) pairs
RADAR_METRICS = [
    ("AP@0.5", "ap@0.5"),
    ("Panoptic Quality", "pq"),
    ("Segmentation Quality", "sq"),
    ("Recognition Quality", "rq"),
]

#: columns the report table shows, in order
METRIC_COLS = ("n_gt", "n_matched", "mean_iou", "sq", "rq", "pq", "ap")

#: pretty (short) display names per representation; anything unmapped keeps
#: its raw rep name
REP_DISPLAY = {
    "mask": "Binary mask",
    "stardist": "StarDist",
    "cellpose": "Cellpose",
    "omnipose": "Omnipose",
    "embedseg": "EmbedSeg",
    "chimera": "PluriSeg hybrid",
}

#: model colors: matplotlib's default tab10 cycle pinned at import time, so
#: the same model keeps the same color across figures regardless of global
#: rcParams changes
MODEL_COLORS = [mpl.colors.to_hex(f"C{i}") for i in range(10)]


def _rep_display(model: str) -> str:
    """Short rep name from a run's model dir name ({split}-{rep.name}):
    cellpose_all-mask -> Mask. Falls back to the full name if the suffix is
    not a known rep."""
    rep = model.rsplit("-", 1)[-1]
    return REP_DISPLAY.get(rep, model)


def _ap_cols(df: pd.DataFrame) -> list[str]:
    """AP columns available in a metrics CSV ('ap', any 'ap@t' and
    'ap@[0.5:0.95]')."""
    return [c for c in df.columns
            if c == "ap" or c.startswith("ap@")]


def _pooled_mean(pooled: pd.DataFrame, dataset_weight: str) -> pd.DataFrame:
    """Pooled mean per (model, category) over per-(image, category) rows.

    'images': every row counts equally — datasets contribute in proportion
    to the images carrying a category (livecell dominates most axes).
    'equal': each dataset's per-category mean first, then an unweighted
    mean across datasets — every dataset counts once regardless of size.
    """
    if dataset_weight == "images":
        return pooled.groupby(["model", "category"]).mean(numeric_only=True)
    if dataset_weight != "equal":
        raise ValueError(f"dataset_weight must be 'images' or 'equal', "
                         f"got {dataset_weight!r}")
    per_ds = pooled.groupby(["model", "category", "dataset"]).mean(
        numeric_only=True)
    return per_ds.groupby(["model", "category"]).mean()


def report(pooled: pd.DataFrame, out_dir: Path, stem: str,
           repeats: str = "seeds", n_repeats: int = None,
           dataset_weight: str = "images", n_expected_cells: int = None,
           dropped: dict[str, int] = None) -> pd.DataFrame:
    """Cross-run summary of the pooled per-object rows plot() loaded.

    One row per (model, category): the pooled mean of every metric column
    (the radar's points, weighted per `dataset_weight`) and its std across
    (dataset, repeat) cell means, plus n_combos (the model's surviving
    dataset x repeat cells) and n_dropped (failed runs removed by the drop
    filter). Prints a rich table (red n_combos below n_expected_cells) and
    saves out_dir/{stem}_summary.csv; 'unmatched' rows included (they hold
    the FP-only bucket).
    """
    if pooled.empty:
        rprint("[yellow]report: no rows[/yellow]")
        return pd.DataFrame()
    rcol = _repeat_col(repeats)
    metric_cols = [c for c in METRIC_COLS if c in pooled.columns]
    pooled_mean = _pooled_mean(pooled, dataset_weight)[metric_cols]
    combo_std = (pooled.groupby(["model", "category", "dataset", rcol])[
        metric_cols].mean()
        .groupby(level=["model", "category"]).std())
    cs = combo_std.reset_index().rename(
        columns={c: f"{c}_std" for c in metric_cols})
    out = pooled_mean.reset_index().merge(cs, on=["model", "category"])
    combos = {m: len(set(zip(g["dataset"], g[rcol])))
              for m, g in pooled.groupby("model")}
    out["n_combos"] = out["model"].map(combos).astype(int)
    out["n_dropped"] = out["model"].map(dropped or {}).fillna(0).astype(int)
    out_path = out_dir / f"{stem}_summary.csv"
    out.to_csv(out_path, index=False)
    rprint(f"[green]saved {out_path}[/green]")
    table = Table(title=f"Per-category metrics ({dataset_weight}) — pooled "
                  "mean ± std across "
                  f"(dataset, {repeats}), per model",
                  title_style="bold")
    for col in ("model", "category", "n_combos", "n_dropped", *metric_cols):
        table.add_column(col, overflow="fold")
    for _, r in out.iterrows():
        n_cell = str(r["n_combos"])
        if n_expected_cells and r["n_combos"] < n_expected_cells:
            n_cell = f"[red]{n_cell}[/red]"
        cells = []
        for c in METRIC_COLS:
            if c not in out.columns or pd.isna(r[c]):
                cells.append("--")
            else:
                std = r.get(f"{c}_std")
                cells.append(f"{r[c]:.3f}" if pd.isna(std)
                             else f"{r[c]:.3f} ± {std:.3f}")
        table.add_row(_rep_display(str(r["model"])), str(r["category"]),
                      n_cell, str(r["n_dropped"]), *cells)
    rprint(table)
    return out


def plot(
    pattern: str,  # glob matching *_metrics.csv files, e.g.
    # 'results/shape-metrics/cis_*' (quoted so the shell doesn't expand it)
    out: Path = None,  # output dir; defaults to next to the matched files
    repeats: str = "seeds",  # repeat axis: "seeds" (CIS batch) | "folds"
    n_repeats: int = None,  # expected repeats; None uses the mode's default
    drop_zeros: bool = False,  # drop runs with ap/pq <= drop_margin everywhere
    drop_margin: float = 0.0,  # metric floor at/below which a run counts as failed
    one_seed: bool = True,  # keep only the best run per (model, dataset)
    root_runs: Path = Path("runs"),  # for the eval PQ of one_seed
    fill_opacity: float = 0.05,  # opacity filling under each mean polygon; 0 = no fill
    by_dataset: bool = False,  # also emit one radar set per dataset
):
    """Radar plots comparing models over shape categories, pooled across
    datasets.

    One figure per RADAR_METRICS metric, all models on the same axis. The
    underlying data is the per-(image, category) rows of the surviving
    runs (model, dataset, repeat, run logged): drop_zeros first removes
    failed runs — mean ap and pq <= drop_margin in every category (NaN
    counts as 0), e.g. the collapsed embedseg runs; then one_seed keeps a
    single run per (model, dataset), the one with the best eval PQ (mean
    row of eval/metrics.csv, results_figure's criterion; its pooled shape
    PQ as fallback), so repeats only ever compete, never average — the
    legend then carries no daggers, and missing (model, dataset) cells are
    flagged in the report only. Without one_seed, a missing (dataset,
    repeat) cell counts as one dagger in the legend vs
    n_datasets x n_repeats. Each radar point is the pooled mean per (model,
    category), saved under both dataset weightings (see below). Axis
    labels' n = GT instances per category summed over all datasets present
    (model-independent; run completeness is carried by the daggers).
    Saved as PDF under `out`/plots (default: next to the matched files),
    named radar_{column}.pdf; the legend ships as its own PDF. Every PDF
    gets a .csv sidecar holding the exact plotted values (radar: one
    column per model over the categories; legend: model/color/daggers).
    The 'unmatched' category is plotted too. Also writes a cross-run
    report (report()).

    by_dataset additionally emits one radar set per dataset
    (radar_{column}_{dataset}.pdf, same one_seed/drop rules pooled within
    that dataset only): the per-modality standings next to the pooled
    figure, exposing the instance-count weighting the pooled mean hides.

    Every pooled figure and the summary are saved twice: image-weighted
    (default names; datasets contribute in proportion to the images
    carrying a category, e.g. livecell dominates most axes) and
    equal-dataset (`_equal` suffix; each dataset's per-category mean
    first, then across datasets — every dataset counts once, shared
    legend).
    """
    import matplotlib.pyplot as plt

    rcol = _repeat_col(repeats)
    files = sorted(Path().glob(pattern))
    if not files:
        rprint(f"[red]pattern matched no files: {pattern}[/red]")
        return
    rprint(f"{len(files)} file(s) matching {pattern}")
    # dataset/model/run/fold/seed come from each file's sidecar
    # {name}.meta.yaml (written by compute()); older CSVs without one fall
    # back to filename parsing, and runs predating the seed column use
    # their fold id as the repeat
    frames = []
    n_cats_by_ds: dict[str, dict[str, int]] = {}
    dropped: dict[str, int] = {}
    rules_file = None  # recovered from a sidecar, for the axis order below
    # one_seed bookkeeping: (model, dataset) -> ((eval_pq, last_epoch), run)
    best: dict[tuple, tuple] = {}
    n_runs_by_cell: dict[tuple, int] = {}
    for f in files:
        meta_file = f.with_suffix("").with_suffix(".meta.yaml")
        if meta_file.exists():
            meta = yaml.safe_load(meta_file.read_text())
            model, run, ds = meta["model"], meta["run"], meta.get("dataset")
            rid = meta.get("seed") if rcol == "seed" else meta.get("fold")
            if rid is None:
                rid = meta.get("fold")
            rules_file = rules_file or meta.get("rules_file")
        else:
            rprint(f"[yellow]no sidecar meta for {f.name}; "
                   f"falling back to filename parsing[/yellow]")
            ds, model, run = f.stem.removesuffix("_metrics").split("_", 2)
            m = re.search(r"-f(\d+)$", run)
            rid = int(m.group(1)) if m else None
        raw = pd.read_csv(f)
        model_dir = model
        # the runs model dir is {dataset}-{rep}; the pooled key is
        # the rep suffix so the radar pools each rep across datasets
        # (unmatched dir names keep the full name)
        model = (model.rsplit("-", 1)[-1]
                 if model.rsplit("-", 1)[-1] in REP_DISPLAY else model)
        if drop_zeros:
            # a failed run (e.g. embedseg predicting nothing) scores <=
            # drop_margin AP and PQ in every category; NaN counts as 0
            # here, matching the radar's fillna. Dropped at load, so its
            # cells vanish and the legend's daggers stay honest
            per_cat = raw.groupby("category")[["ap", "pq"]].mean()
            if per_cat.empty or (per_cat.fillna(0) <= drop_margin).all().all():
                rprint(f"[yellow]drop: failed run (ap/pq <= {drop_margin} "
                       f"in every category): {f.name}[/yellow]")
                dropped[model] = dropped.get(model, 0) + 1
                continue
        if one_seed:
            # rank for the one-seed selection below: eval PQ of the mean
            # row (results_figure's criterion), the run's pooled shape PQ
            # when eval/metrics.csv is missing; ties by highest last epoch
            run_path = root_runs / model_dir / run
            eval_pq = _run_pq(run_path)
            parsed = _parse_config(run_path)
            rank = (eval_pq if eval_pq is not None
                    else float(raw["pq"].mean()),
                    (parsed["last_epoch"] if parsed else None) or -1)
            cell = (model, ds)
            n_runs_by_cell[cell] = n_runs_by_cell.get(cell, 0) + 1
            if cell not in best or rank > best[cell][0]:
                best[cell] = (rank, run)
        if ds not in n_cats_by_ds:
            # GT is shared across a dataset's runs: the instance count per
            # category comes from the dataset's first surviving CSV
            n_cats_by_ds[ds] = raw.groupby("category")["n_gt"].sum().to_dict()
        frames.append(raw.assign(model=model, dataset=ds, run=run,
                                 **{rcol: rid}))
    if not frames:
        rprint("[red]no run survived the drop filter; nothing to plot[/red]")
        return
    pooled = pd.concat(frames, ignore_index=True)
    if one_seed:
        # one run per (model, dataset): keep only the best-ranked run, as
        # results_figure picks the figure's runs (ties by last epoch);
        # repeats compete, they never average
        keep = {cell: run for cell, (_, run) in best.items()}
        pooled = pooled[[(model, ds) in keep and keep[(model, ds)] == run
                         for model, ds, run in zip(pooled["model"],
                                                   pooled["dataset"],
                                                   pooled["run"])]]
        for (model, ds), (rank, run) in sorted(best.items()):
            if n_runs_by_cell.get((model, ds), 0) > 1:
                rprint(f"[cyan]best {model} on {ds}: {run} "
                       f"(PQ {rank[0]:.3f})[/cyan]")
    datasets = sorted(pooled["dataset"].unique())
    # expected cells: with one_seed a cell holds one run, so only a fully
    # absent dataset counts as missing (and the legend stays dagger-free);
    # without one_seed every (dataset, repeat) is expected
    n_expected_cells = (len(datasets) if one_seed else
                        len(datasets) * len(_expected_repeats(repeats,
                                                              n_repeats)))
    # axes follow REP_DISPLAY's order (unmatched reps go last, by name)
    models = sorted(pooled["model"].unique(), key=lambda m: (
        list(REP_DISPLAY).index(m) if m in REP_DISPLAY else len(REP_DISPLAY),
        m))
    # union of categories across runs (a model missing a category plots as 0)
    cats = sorted(set(pooled["category"]) - {"unmatched"})
    # axis order follows the rules YAML the CSVs were computed with (recovered
    # from a sidecar meta), rule order first and the rest last; without a
    # sidecar the axes fall back to alphabetical
    if rules_file and Path(rules_file).exists():
        rule_names = [r["name"] for r in
                      yaml.safe_load(Path(rules_file).read_text())]
        cats = ([c for c in rule_names if c in cats]
                + sorted(set(cats) - set(rule_names)))
    elif rules_file:
        rprint(f"[yellow]rules file not found ({rules_file}); "
               f"axes keep alphabetical order[/yellow]")
    # GT instance count per category, summed over every dataset present
    # (the pooled mean is computed over these objects)
    n_cats = {c: int(sum(n_cats_by_ds.get(d, {}).get(c, 0)
                         for d in datasets)) for c in cats}

    plt_be_stilish(grid=False)
    out_dir = (Path(out) if out else files[0].parent) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _radar_figures(pool: pd.DataFrame, cats_now: list[str],
                       n_now: dict[str, int], n_expected: int, tag: str,
                       legend: bool = True, weight: str = "images"):
        """One figure per RADAR_METRICS metric over `cats_now`, all models
        on the same axis; `tag` suffixes the filenames ('' = the pooled
        set). A model absent from `pool` is skipped; one missing a category
        plots as 0. `legend=False` skips the legend PDF (colors match the
        shared one). Each PDF ships a .csv sidecar with the exact plotted
        values (radar: category/n_gt/one column per model; legend:
        model/color/daggers), for LLM-readable figures."""
        piv = _pooled_mean(pool, weight)
        combos = {m: len(set(zip(g["dataset"], g[rcol])))
                  for m, g in pool.groupby("model")}
        daggers = {m: max(0, n_expected - c) for m, c in combos.items()}
        angles = np.linspace(0, 2 * np.pi, len(cats_now),
                             endpoint=False).tolist()
        angles += angles[:1]  # close the polygon
        pool_models = set(pool["model"])

        def _style(ax, title):
            # first category's axis points south (default polar puts it east)
            ax.set_theta_offset(-np.pi / 2)
            for r in (0.25, 0.5, 0.75):
                ax.plot(np.linspace(0, 2 * np.pi, 181), [r] * 181,
                        color="#e0e0e0", lw=0.6, zorder=0.5)
            ax.set_xticks(angles[:-1],
                          [f"{c.replace('_', '-')}\nn={n_now[c]}"
                           for c in cats_now],
                          fontsize=12)
            ax.tick_params(axis="x", pad=14)  # push labels off the rim
            ax.set_ylim(0, 1)
            ax.set_yticks([0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=16)
            ax.set_title(title, pad=30, fontweight="bold", fontsize=14)

        def _series(model: str, col: str) -> np.ndarray:
            """Closed polygon values for one model's column (missing cat = 0)."""
            v = np.array([piv[col].get((model, c), np.nan)
                          for c in cats_now])
            v = np.nan_to_num(v)
            return np.append(v, v[0])

        def _draw(ax, series, color, label=None, lw=2.2):
            ax.plot(angles, series, marker="o", markersize=8, lw=lw,
                    color=color, label=label)

        # colors follow MODEL_COLORS order (unmatched reps go last, by name)
        legend_done = False  # legend ships as its own PDF, from the first radar
        for title, colname in RADAR_METRICS:
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={
                                   "projection": "polar"})
            any_drawn = False
            # sidecar: the exact values the polygon draws (missing cat = 0)
            sidecar = pd.DataFrame({"category": cats_now,
                                    "n_gt": [n_now[c] for c in cats_now]})
            for i, model in enumerate(models):
                if model not in pool_models:
                    continue  # no surviving run here: skipped, not zero
                if colname not in piv.columns:
                    rprint(f"[yellow]no {colname} column for model {model}; "
                           f"re-run compute to add it[/yellow]")
                    continue
                color = MODEL_COLORS[i % len(MODEL_COLORS)]
                any_drawn = True
                mean = _series(model, colname)
                sidecar[_rep_display(model)] = mean[:-1]
                if fill_opacity > 0:
                    # fill under the mean polygon with the model's color
                    ax.fill(angles, mean, color=color, alpha=fill_opacity,
                            linewidth=0, zorder=0.5)
                # one dagger per missing (dataset, repeat) cell; daggers as a
                # latex superscript, same string as agglomerate_metrics. With
                # one_seed the legend stays dagger-free (one run per cell, the
                # report flags missing datasets instead)
                label = _rep_display(model)
                if not one_seed:
                    label += r"$^{" + r"\dagger" * daggers[model] + "}$"
                _draw(ax, mean, color, label=label, lw=3.2)
            if not any_drawn:
                plt.close(fig)
                rprint(f"[red]no model has column {colname}; skipping[/red]")
                continue
            _style(ax, title)
            if legend and not legend_done:
                legend_fig, legend_ax = plt.subplots(figsize=(2.4, 2.4))
                legend_ax.axis("off")
                legend_ax.legend(
                    *ax.get_legend_handles_labels(), loc="center",
                    fontsize=10, frameon=False)
                legend_fig.tight_layout()
                legend_fig.savefig(out_dir / f"legend{tag}.pdf")
                plt.close(legend_fig)
                # legend sidecar: model -> color, plus dagger counts
                drawn = [m for m in models if m in pool_models]
                pd.DataFrame({
                    "model": [_rep_display(m) for m in drawn],
                    "color": [MODEL_COLORS[i % len(MODEL_COLORS)]
                              for i, m in enumerate(models)
                              if m in pool_models],
                    "daggers": [daggers[m] for m in drawn],
                }).to_csv(out_dir / f"legend{tag}.csv", index=False)
                rprint(f"[green]saved {out_dir / f'legend{tag}.pdf'}[/green]")
                legend_done = True
            fig.tight_layout()
            fig.savefig(out_dir / f"radar_{colname}{tag}.pdf")
            sidecar.to_csv(out_dir / f"radar_{colname}{tag}.csv", index=False)
            plt.close(fig)
            rprint(f"[green]saved {out_dir / f'radar_{colname}{tag}.pdf'} "
                   f"(+ .csv sidecar)[/green]")

    # both weightings, always: image-weighted under the default names,
    # equal-dataset with an _equal suffix (shared legend, per-dataset
    # figures are single-dataset so weighting doesn't apply there)
    _radar_figures(pooled, cats, n_cats, n_expected_cells, "")
    _radar_figures(pooled, cats, n_cats, n_expected_cells, "_equal",
                   legend=False, weight="equal")
    if by_dataset:
        # one radar set per dataset: standings per imaging modality and
        # annotation style, exposing the instance-count weighting that the
        # pooled mean hides (livecell dominates most axes)
        for ds in datasets:
            ds_pool = pooled[pooled["dataset"] == ds]
            ds_cats = [c for c in cats if c in set(ds_pool["category"])]
            _radar_figures(
                ds_pool, ds_cats,
                {c: int(n_cats_by_ds.get(ds, {}).get(c, 0)) for c in ds_cats},
                1 if one_seed else len(_expected_repeats(repeats, n_repeats)),
                f"_{ds}")
    # the summary carries the pattern's name, not the first CSV's (it pools
    # every matched run)
    stem = re.sub(r"\W+", "_", f"{Path(pattern).parent.name}_"
                  f"{Path(pattern).stem}").strip("_")
    report(pooled, out_dir, stem, repeats=repeats, n_repeats=n_repeats,
           dataset_weight="images", n_expected_cells=n_expected_cells,
           dropped=dropped)
    report(pooled, out_dir, stem + "_equal", repeats=repeats,
           n_repeats=n_repeats, dataset_weight="equal",
           n_expected_cells=n_expected_cells, dropped=dropped)


def _best_ckpt(run_dir: Path) -> Path | None:
    """Newest-scored best-*.ckpt of a run (None when it has none); the
    freshness proxy for compute()'s skip-if-fresh logic."""
    ckpts = sorted(run_dir.glob("checkpoints/best-*.ckpt"))
    if not ckpts:
        return None

    def score(p: Path) -> float:
        m = re.search(r"=-?([\d.]+)\.ckpt$", p.name)
        return float(m.group(1)) if m else float("-inf")
    return max(ckpts, key=score)


def compute(
    pattern: str,  # glob under root_runs, e.g. "*/*" (model/run) or "*/run*"
    rules_file: Path,  # YAML with a list of ShapeRule dicts
    root_runs: Path = Path("runs"),
    root_data: Path = None,
    split_name: str = None,  # None: taken from the first run's config.yaml
    splits_dir: Path = SPLITS,
    n_folds: int = 1,
    ap_thresholds: list[float] = [0.25, 0.5, 0.75],  # ap@t columns to store
    opening_radius: int = DEFAULT_OPENING_RADIUS,
    n_examples: int = 4,  # overlay crops per category per run; 0 = off
    num_workers: int = 12,
    only: str = "",  # glob on run name or model/run path; empty = all
    force: bool = False,  # recompute even when a fresh CSV already exists
):
    """Score each matching run's eval predictions per GT shape category.

    Args:
        pattern: glob under root_runs; each match should be a
            runs/<model>/<run> directory with an eval/ subdir of
            <cache_id>_pred.tif files.
        rules_file: YAML list of rule dicts (ShapeRule fields: name,
            features mapping feature column -> (lo, hi), None = open);
            GT instances matching no rule get "other".
        ap_thresholds: IoU thresholds; each stores an 'ap@t' column next
            to the plain 'ap' (which is ap@0.5).
        root_data/split_name: the dataset and split the runs were trained
            on (must match their configs for cache ids to line up);
            split_name=None reads the split (n_folds and dataset root too)
            from each group's first config.yaml, so one command covers
            mixed splits (e.g. the CIS seed batch). The fold and seed come
            from each run's config.yaml, so runs of several folds/seeds
            can be scored in one go (test images per split+fold).
        opening_radius: disk radius for stray-component removal; must match
            what produced the feature cache.
        n_examples: per category, GT instances sampled once per split
            (seed 0) and saved as an overlay crop per run under
            results/shape-metrics/examples/{dataset}/{model}/{category}/
            (image, GT context boundary dim, the instance's boundary
            yellow, prediction boundaries magenta); the same instances for
            every run, so models compare directly. 0 disables the export.
        only: restrict to a select set of runs — fnmatch glob on the run
            name or the model/run path relative to root_runs.
        force: recompute runs whose CSV already exists; by default a run
            is skipped when its CSV is newer than its best checkpoint and
            the recorded rules/opening/thresholds match (so rescued runs
            with a replaced ckpt are recomputed, untouched ones are not).
    """
    runs = sorted(root_runs.glob(pattern))
    if not runs:
        rprint(f"[red]no runs match {root_runs}/{pattern}[/red]")
        return
    if only:
        runs = [r for r in runs
                if fnmatch(r.name, only)
                or fnmatch(str(r.relative_to(root_runs)), only)]
        if not runs:
            rprint(f"[red]no run matches --only {only!r}[/red]")
            return
    rprint(f"found {len(runs)} run(s) matching {pattern} in {root_runs}")

    # (split, fold) and seed per run, from its config.yaml (same convention
    # as the agglomerate tool); runs without a parsable config are skipped.
    # Runs are grouped per split: the CIS batch has five datasets at fold 0,
    # and a fold-0-only grouping would score every dataset's runs against
    # the first group's split
    runs_by_group: dict[tuple[str, int], list[tuple[Path, int]]] = {}
    for run_dir in runs:
        parsed = _parse_config(run_dir)
        if parsed is None or parsed["fold"] < 0:
            rprint(
                f"[yellow]skip (unparsable config.yaml): {run_dir}[/yellow]")
            continue
        runs_by_group.setdefault((parsed["dataset"], parsed["fold"]), []).append(
            (run_dir, parsed["seed"]))
    if not runs_by_group:
        rprint("[red]no run has a parsable config.yaml; nothing to score[/red]")
        return

    for (g_split, fold), group_runs in sorted(runs_by_group.items()):
        if split_name is not None and g_split != split_name:
            rprint(f"[yellow]skip {len(group_runs)} run(s): config split "
                   f"{g_split} != requested {split_name}[/yellow]")
            continue
        g_n_folds, g_root = n_folds, root_data
        if split_name is None:
            # split's n_folds + dataset root from the group's first
            # config.yaml; the config's own root wins when it exists on
            # this machine (cluster runs keep their pasteur path),
            # otherwise the split's DEFAULT_ROOT_MAP entry (local mirror)
            first = group_runs[0][0]
            ds_cfg = yaml.safe_load(
                (first / "config.yaml").read_text())["dataset"]
            g_n_folds = int(ds_cfg.get("n_folds", n_folds))
            cfg_root = ds_cfg.get("root")
            if g_root is None and cfg_root and Path(cfg_root).is_dir():
                g_root = Path(cfg_root)
        if g_root is None:  # split name -> dataset root, from tools.utils
            g_root = Path(DEFAULT_ROOT_MAP.get(
                g_split, DEFAULT_ROOT_MAP["*"]))
        rprint(f"[bold]{g_split} fold {fold}: {len(group_runs)} run(s)[/bold]")
        _compute_fold(
            group_runs, rules_file, g_root, g_split, splits_dir, fold,
            g_n_folds, ap_thresholds, opening_radius, num_workers,
            n_examples, force=force)


def _overlay_example(
    img: np.ndarray, gt: np.ndarray, pred: np.ndarray, label_id: int,
) -> np.ndarray | None:
    """Square (border-clipped) overlay crop around GT instance `label_id`,
    for visual checks.

    Image crop, contrast-stretched between its 2nd/98th percentiles, with
    all GT boundaries dim gray, the instance's own boundary yellow and all
    prediction boundaries magenta — a merged-blob GT against separated
    predictions (or vice versa) shows up directly. None when the label is
    missing from `gt` (stale cache).
    """
    mask = gt == label_id
    if not mask.any():
        rprint(f"[yellow]_overlay_example: label {label_id} not in the "
               f"cleaned map — stale cache? Skipping.[/yellow]")
        return None
    ys, xs = np.nonzero(mask)
    h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    pad = max(2, max(h, w) // 8)
    side = max(h, w) + 2 * pad
    y0 = max(0, min(int(ys.min()) - pad, img.shape[0] - side))
    x0 = max(0, min(int(xs.min()) - pad, img.shape[1] - side))
    win = (slice(y0, y0 + side), slice(x0, x0 + side))
    base = img[win].astype(float)
    if base.ndim == 3:  # RGB: collapse to a single luminance plane
        base = base.mean(axis=-1)
    lo, hi = np.percentile(base, (2, 98))
    out = np.repeat(
        np.clip((base - lo) / max(hi - lo, 1e-12), 0, 1)[..., None], 3,
        axis=2)
    for m, color in ((gt > 0, (0.55, 0.55, 0.55)),
                     (mask, (1.0, 1.0, 0.0)),
                     (pred > 0, (1.0, 0.0, 1.0))):
        edge = m & ~binary_erosion(m)
        out[edge[win]] = color
    return (out * 255).astype(np.uint8)


def _compute_fold(
    runs: list[tuple[Path, int]], rules_file: Path, root_data, split_name,
    splits_dir, fold: int, n_folds: int, ap_thresholds: list[float],
    opening_radius: int, num_workers: int, n_examples: int,
    force: bool = False,
):
    """compute() for one fold's runs — (run_dir, seed) pairs; the split +
    GT categories are per fold, the seed rides along into the sidecar."""
    cfg = DatasetConfig(
        root=str(root_data), split_name=split_name, splits_dir=str(splits_dir),
        fold=fold, n_folds=n_folds)
    dataset_name, parts = load_split(cfg)
    test_pairs = parts["test"]
    rprint(f"dataset={dataset_name} test images={len(test_pairs)}")

    # features + categories come from the shared GT cache, so this is
    # instant after pluriseg.shapes.features has run once for the dataset
    rprint("computing shape features...")
    df = dataset_shape_features(
        [(label, cache_id) for _, label, cache_id in test_pairs],
        dataset_name, opening_radius=opening_radius, num_workers=num_workers)
    rprint(f"instances={len(df)}")

    rprint(f"rules from {rules_file}...")
    rules = [ShapeRule(**r)
             for r in yaml.safe_load(rules_file.read_text())]
    df["category"] = RuleShapeCategorizer(rules).categorize(df)
    rprint(f"sizes={df['category'].value_counts().sort_index().to_dict()}")

    # category per (image, raw label id): exactly the index per_category_metrics
    # expects; instances dropped by the opening simply have no row here
    cats_by_image = {
        cache_id: g.set_index("label_id")["category"]
        for cache_id, g in df.groupby("cache_id")
    }

    # GT-side example selection, shared by every run of the split: the same
    # instances get an overlay per run (seed 0), so models compare directly
    examples_by_image: dict[str, list[tuple[str, int]]] = {}
    if n_examples > 0:
        for cat, picks in pick_category_examples(
                df, df["category"], n_per_cat=n_examples, seed=0).items():
            for cache_id, label_id in picks:
                examples_by_image.setdefault(cache_id, []).append(
                    (cat, label_id))
    ex_dir = RESULTS / "shape-metrics" / "examples" / dataset_name

    # one CSV per run: results/shape-metrics/{dataset}_{model}_{run}_metrics.csv,
    # with a sidecar {name}.meta.yaml (dataset/model/run) for plot()
    out_dir = RESULTS / "shape-metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (run_dir, seed) in enumerate(runs):
        rprint(f"[cyan]({i+1}/{len(runs)})[/cyan] {run_dir}")
        out = out_dir / f"{dataset_name}_{run_dir.name}_metrics.csv"
        meta_path = out.with_suffix("").with_suffix(".meta.yaml")
        if not force and out.exists() and meta_path.exists():
            ckpt = _best_ckpt(run_dir)
            fresh = ckpt is not None and \
                out.stat().st_mtime >= ckpt.stat().st_mtime
            if fresh:
                meta = yaml.safe_load(meta_path.read_text())
                fresh = (meta.get("rules_file") == str(rules_file)
                         and meta.get("opening_radius") == opening_radius
                         and list(meta.get("ap_thresholds", []))
                         == list(ap_thresholds))
            if fresh:
                rprint("[dim]skip (fresh CSV; --force true to recompute)"
                       "[/dim]")
                continue
        eval_dir = run_dir / "eval"
        if not eval_dir.is_dir():
            rprint("[yellow]skip (no eval/ dir)[/yellow]")
            continue
        pairs = [(ip, lf, cid) for ip, lf, cid in test_pairs
                 if (eval_dir / f"{cid}_pred.tif").exists()]
        if len(pairs) < len(test_pairs):
            rprint(f"[yellow]skip ({len(test_pairs) - len(pairs)}/"
                   f"{len(test_pairs)} preds missing in {eval_dir})[/yellow]")
            continue
        rows = []
        used_ids = []
        n_saved = 0
        for img_file, label_file, cache_id in pairs:
            # raw GT ids == feature-cache ids (the opening preserves ids), so
            # the categories join directly; get_clean_label warms the GT cache
            get_clean_label(dataset_name, cache_id, label_file,
                            opening_radius=opening_radius)
            pred = load_masks(eval_dir / f"{cache_id}_pred.tif")
            per_img = per_category_metrics(
                load_masks(label_file), pred,
                cats_by_image.get(cache_id, pd.Series(dtype=str)),
                thresh=0.5, ap_thresholds=tuple(ap_thresholds), ap_coco=True)
            rows.append(per_img)
            used_ids.append(cache_id)
            for cat, label_id in examples_by_image.get(cache_id, ()):
                gt = get_clean_label(dataset_name, cache_id, label_file,
                                     opening_radius=opening_radius)
                crop_dir = ex_dir / run_dir.parent.name / cat
                crop_dir.mkdir(parents=True, exist_ok=True)
                crop = _overlay_example(iio.imread(img_file), gt, pred,
                                        label_id)
                if crop is None:
                    continue
                iio.imwrite(
                    crop_dir / f"{run_dir.name}__{cache_id}_{label_id}.png",
                    crop)
                n_saved += 1
        if n_saved:
            rprint(f"[green]saved {n_saved} example overlay(s) "
                   f"under {ex_dir}[/green]")

        if not rows:
            rprint("[yellow]no predictions scored; skipping[/yellow]")
            continue
        per_category_table(rows)
        pd.DataFrame(
            [dict(image=cache_id, category=cat, **m)
             for cache_id, row in zip(used_ids, rows)
             for cat, m in row.items()]
        ).to_csv(out, index=False)
        rprint(f"[green]saved {out}[/green]")
        # sidecar metadata: the authoritative dataset/model/run triple, so
        # plot() never has to parse names (models/runs contain underscores)
        meta_path.write_text(yaml.safe_dump({
            "dataset": dataset_name,
            "model": run_dir.parent.name,
            "run": run_dir.name,
            "fold": fold,
            "seed": seed,
            "rules_file": str(rules_file),
            "ap_thresholds": list(ap_thresholds),
            "opening_radius": opening_radius,
        }, sort_keys=False))
        rprint(f"[green]saved {meta_path}[/green]")
