from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from rich import print as rprint
from rich.table import Table

from pluriseg.display import plt_be_stilish
from pluriseg.paths import RESULTS_METRICS
from pluriseg.utils import _expected_repeats, _parse_config, _repeat_col

LATEX_TABLE_COLS = ['cellpose', 'DSB2018', 'BBBC010', 'merged_all']
LATEX_TABLE_SUBCOLS = ['AP@0.50', 'PQ']
# table subcol -> eval metrics.csv column
LATEX_SUBCOL_TO_COLUMN = {}
LATEX_TABLE_ROWS = {
    'mask': r"     Binary mask \citeun",
    'stardist': r"     StarDist \citesd",
    'cellpose': r"     Cellpose \citecp",
    'embedseg': r"     EmbedSeg \citees",
    'omnipose': r"     Omnipose \citeop",
    'chimera': r"     PluriSeg hybrid"
}

#: pretty head names for the chimera per-head LaTeX rows (chimera/<head>)
HEAD_LABELS = {"mask": "Binary mask", "cellpose": "Cellpose", "omnipose": "Omnipose",
               "stardist": "StarDist", "embedseg": "EmbedSeg"}

#: dataset row labels for the vertical head-selector table (cite macros
#: defined in the paper preamble); unmapped datasets fall back to their name
DATASET_LABELS = {
    "cellpose": r"Cellpose dataset \citecp",
    "DSB2018": r"DSB2018 \citedsb",
    "BBBC010": r"BBBC010 \citebbbc",
    "merged_all": r"Merged",
}
#: pseudo-rep carrying the chimera gate's selection accuracy: its fold rows
#: store the accuracy in the PQ column (the gate's oracle criterion)
GATE_ACC_REP = "chimera-gate-acc"

#: how close a gate pick's PQ must be to the best head's PQ to count as a
#: good selection (ties always count)
HEAD_PERF_MARGIN = 0.02

#: printed-table cells flag as suspicious (yellow + warning) when their mean
#: is below or their std is above this; a pointer to drop_zeros/drop_margin
RICH_WARN_MARGIN = 0.1


FOLDS = ["f0", "f1", "f2", "f3", "f4"]
#: FOLDS entries ("f0", ...) as the fold numbers the tables expect
FOLD_NUMS = [int(f[1:]) for f in FOLDS if f.startswith("f")]


@dataclass
class TableLayout:
    """LaTeX table layout: which rep rows, dataset columns, metric subcols
    and pretty labels the tables carry. Defaults to the module constants;
    the CLI config file's `table:` block overrides any subset (a layout
    for a run set missing e.g. BBBC010 simply lists the datasets it has)."""
    rows: dict[str, str] = field(
        default_factory=lambda: dict(LATEX_TABLE_ROWS))
    datasets: list[str] = field(
        default_factory=lambda: list(LATEX_TABLE_COLS))
    dataset_labels: dict[str, str] = field(
        default_factory=lambda: dict(DATASET_LABELS))
    head_labels: dict[str, str] = field(
        default_factory=lambda: dict(HEAD_LABELS))
    subcols: list[str] = field(
        default_factory=lambda: list(LATEX_TABLE_SUBCOLS))
    #: table subcol -> eval metrics.csv column (empty = identity)
    subcol_to_column: dict[str, str] = field(default_factory=dict)
    #: decimals of the LaTeX mean / std cells (percent scale, separate so
    #: e.g. a small std can carry one more digit than its mean)
    mean_decimals: int = 1
    std_decimals: int = 1

    def col_of(self, subcol: str) -> str:
        """table subcol -> eval metrics.csv column"""
        return self.subcol_to_column.get(subcol, subcol)


def _pick_fractions(df: pd.DataFrame, heads: list[str]) -> dict[str, float]:
    """Per-head gate-pick fractions over a run's test images.

    Heads never picked still get 0.0 (a real absence, not a missing repeat).
    """
    if "gate_pick" not in df.columns or not df["gate_pick"].notna().any():
        return {}
    counts = df["gate_pick"].value_counts(normalize=True)
    return {f"pick_{h}": float(counts.get(h, 0.0)) for h in heads}


def _latex_rows(df: pd.DataFrame, table: TableLayout) -> list[tuple[str, str]]:
    """Ordered (rep key, LaTeX label) rows: table.rows with, after the
    chimera row, one row per chimera/<head> rep present in the data."""
    rows = []
    for key, label in table.rows.items():
        rows.append((key, label))
        if key == "chimera":
            heads = sorted({r.split("/", 1)[1] for r in df["rep"].unique()
                            if isinstance(r, str) and r.startswith("chimera/")})
            rows += [(f"chimera/{h}",
                      rf"     \hspace{{4mm}} PluriSeg ({table.head_labels.get(h, h)})")
                     for h in heads]
    return rows


def to_latex(df: pd.DataFrame, out: Path = None, n_repeats: int = None,
             head_metrics: bool = False, repeats: str = "seeds",
             table: TableLayout = None) -> None:
    """LaTeX table: one row per rep (table.rows keys, values as the row
    label), one cell per (dataset col, metric subcol) as mean +- std across
    repeats (seeds by default, folds with repeats="folds"). One superscript
    dagger per missing repeat (`*` is reserved for significance in
    biology); per (dataset, metric) column the best rep is bolded. The
    gate's selection accuracy (GATE_ACC_REP) is NOT rendered in this table:
    it is reported in the dedicated head-selector table built by
    to_latex_gate().

    Args:
        df: summarize()'s output (one row per rep, dataset, with
            '<col>_mean'/'<col>_std' columns and a 'n_repeats' count). Rows
            removed by drop_zero_repeats() are counted as missing.
        n_repeats: expected repeat count; None uses the mode's default
            (N_SEEDS for seeds, len(FOLDS) for folds).
        head_metrics: add the chimera/<head> per-head diagnostic rows;
            default keeps only the chimera row and the gate-accuracy row.
        repeats: which column is the repeat axis ("seeds" or "folds").
        table: the table layout (rows, dataset columns, labels, subcols);
            default TableLayout() (the built-in constants).
        out: where to save the .tex; default
            RESULTS_METRICS/all_metrics.tex.
    """
    if not out:
        out = RESULTS_METRICS / "all_metrics.tex"
    out = Path(out)
    table = table or TableLayout()
    n_expected = len(_expected_repeats(repeats, n_repeats))
    cols = table.datasets
    # dataset names differ in case between configs and the table layout;
    # match case-insensitively
    by_ds = {d.lower(): d for d in df["dataset"].unique()}
    rows_list = _latex_rows(df, table)
    if not head_metrics:
        rows_list = [(rep, label) for rep, label in rows_list
                     if not rep.startswith("chimera/")]
    # GATE_ACC_REP is deliberately not rendered here: the gate's selection
    # accuracy is reported in the dedicated to_latex_gate() table
    # per (col, subcol): collect each rep's parsed cell first, so the max
    # can be bolded; missing cells (None) never participate
    cells = {}  # (col, subcol) -> list[(rep, mean, std, n_missing)]
    for rep, _ in rows_list:
        for col in cols:
            ds = by_ds.get(col.lower())
            sel = df[(df["rep"] == rep) & (df["dataset"] == ds)] \
                if ds else df.iloc[0:0]
            n_missing = n_expected - int(sel["n_repeats"].iloc[0]) \
                if len(sel) and "n_repeats" in sel.columns else n_expected
            n_missing = max(n_missing, 0)
            for sub in table.subcols:
                mcol = table.col_of(sub)
                if not len(sel) or f"{mcol}_mean" not in sel.columns \
                        or pd.isna(sel[f"{mcol}_mean"].iloc[0]):
                    cells.setdefault((col, sub), []).append(
                        (rep, None, None, n_missing))
                else:
                    # percent scale, no '%' glyph: (x * 100) printed .1f;
                    # a single repeat has no std -> mean only
                    cells.setdefault((col, sub), []).append(
                        (rep, float(sel[f"{mcol}_mean"].iloc[0]),
                         sel[f"{mcol}_std"].iloc[0], n_missing))
    lines = [
        r"\begin{tabular}{l" + "c" * (len(cols)
                                      * len(table.subcols)) + "}",
        r"\toprule",
        # dataset names span their subcols on the top header line
        " & " + " & ".join(
            r"\multicolumn{%d}{c}{%s}" % (len(table.subcols), col)
            for col in cols) + r" \\",
        " & ".join(table.subcols * len(cols)) + r" \\",
        r"\midrule",
    ]
    chimera_rule_done = False
    for rep, r in rows_list:
        if ((rep == "chimera" or rep.startswith("chimera/"))
                and not chimera_rule_done):
            lines.append(r"\midrule")
            chimera_rule_done = True
        row = [r]
        for col in cols:
            for sub in table.subcols:
                m, std, n_missing = next(
                    t[1:] for t in cells[(col, sub)] if t[0] == rep)
                if m is None:
                    row.append("--")
                    continue
                cell = f"{m * 100:.{table.mean_decimals}f}"
                if not pd.isna(std):
                    cell += f" $\\pm$ {std * 100:.{table.std_decimals}f}"
                # one superscript dagger per missing repeat (e.g. one dropped
                # zero run); '*' is reserved for significance
                if n_missing:
                    cell += r"$^{" + r"\dagger" * n_missing + "}$"
                # bold the best rep per (col, subcol); per-head chimera rows
                # are diagnostics, not competing methods. The emptiness
                # guard also covers all-missing columns (max() on [] raises).
                vals = [t[1] for t in cells[(col, sub)]
                        if t[1] is not None and "/" not in t[0]]
                if vals and m == max(vals):
                    cell = r"\textbf{" + cell + "}"
                row.append(cell)
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines), encoding="utf-8")
    rprint(f"[green]saved {out}[/green]")


def to_latex_gate(df: pd.DataFrame, out: Path = None,
                  n_repeats: int = None, repeats: str = "seeds",
                  table: TableLayout = None) -> None:
    """Dedicated vertical LaTeX table for the gated chimera's head selector.

    Tabular body only (\\begin{table}/caption/label live in the main tex):
    one row per dataset — the margin-based accuracy (mean +- std across
    repeats, percent) and the most-picked head with its share, the two cells
    the main table's one-cell-per-metric layout cannot carry.

    Args:
        df: gate_selection_accuracy()'s repeat-level frame
            (rep=GATE_ACC_REP).
        n_repeats: expected repeat count; None uses the mode's default
            (N_SEEDS for seeds, len(FOLDS) for folds).
        repeats: which column is the repeat axis ("seeds" or "folds").
        table: the table layout (dataset columns and labels); default
            TableLayout() (the built-in constants).
        out: where to save the .tex; default
            RESULTS_METRICS/gate_selector.tex.
    """
    if not out:
        out = RESULTS_METRICS / "gate_selector.tex"
    out = Path(out)
    table = table or TableLayout()
    n_expected = len(_expected_repeats(repeats, n_repeats))
    s = summarize(df, repeats)
    by_ds = {d.lower(): d for d in s["dataset"].unique()}
    pick_cols = [c for c in s.columns
                 if c.startswith("pick_") and c.endswith("_mean")]

    lines = [
        # tabular body only: caption and label live in the main tex
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Dataset & Acc. & Most picked \\",
        r"\midrule",
    ]
    for col in table.datasets:
        ds = by_ds.get(col.lower())
        sel = (s[(s["rep"] == GATE_ACC_REP) & (s["dataset"] == ds)]
               if ds else s.iloc[0:0])
        if not len(sel) or pd.isna(sel["PQ_mean"].iloc[0]):
            acc = "--"
        else:
            acc = f"{sel['PQ_mean'].iloc[0] * 100:.{table.mean_decimals}f}"
            std = sel["PQ_std"].iloc[0]
            if not pd.isna(std):
                acc += f" $\\pm$ {std * 100:.{table.std_decimals}f}"
            n_missing = max(n_expected - int(sel["n_repeats"].iloc[0]), 0)
            if n_missing:
                acc += r"$^{" + r"\dagger" * n_missing + "}$"
        picks = {c.removeprefix("pick_").removesuffix("_mean"):
                 float(sel[c].iloc[0]) for c in pick_cols
                 if len(sel) and not pd.isna(sel[c].iloc[0])}
        if picks:
            h, v = max(picks.items(), key=lambda t: t[1])
            pick = f"{table.head_labels.get(h, h)} ({v * 100:.0f}\\%)"
        else:
            pick = "--"
        label = table.dataset_labels.get(ds or col, (ds or col))
        lines.append(f"{label} & {acc} & {pick} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines), encoding="utf-8")
    rprint(f"[green]saved {out}[/green]")


def summarize(df: pd.DataFrame, repeats: str = "seeds") -> pd.DataFrame:
    """Repeat-level table -> one row per (rep, dataset): mean and std of
    every metric across repeats (seeds or folds per `repeats`), as
    '<col>_mean' / '<col>_std' columns, plus 'n_repeats' counting the
    repeats present per row."""
    rcol = _repeat_col(repeats)
    metric_cols = [c for c in df.columns
                   if c not in ("rep", "dataset", "fold", "seed", "run_dir")]
    agg = df.groupby(["rep", "dataset"])[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    out = agg.reset_index()
    out.insert(2, "n_repeats", df.groupby(
        ["rep", "dataset"])[rcol].nunique().to_numpy())
    return out.sort_values(
        ["dataset", "rep"], kind="stable").reset_index(drop=True)


def report_missing(df: pd.DataFrame, n_repeats: int = None,
                   repeats: str = "seeds", table: TableLayout = None) -> Table:
    """Gaps vs the tex table's expectations, as a rich table: which
    (rep, dataset) lacks repeats of the active axis (seeds s0..sN or folds
    f0..f4), which repeat lacks a table metric column (or carries NaN), and
    which rep x dataset combos have no eval rows at all. Expected combos
    span every rep/dataset seen in the data plus the table's own rows/cols.
    Empty table = complete."""
    rcol = _repeat_col(repeats)
    table = table or TableLayout()
    expected = _expected_repeats(repeats, n_repeats)
    table_cols = [table.col_of(s) for s in table.subcols]
    rt = Table(title=f"missing {repeats} / metrics vs the LaTeX table")
    for c in ("rep", "dataset", rcol, "missing"):
        rt.add_column(c)
    # expect the union: table layout + whatever the runs actually produced,
    # minus the chimera/<head> diagnostics (not table expectations)
    reps = list(table.rows) + [
        r for r in df["rep"].unique()
        if r not in table.rows and not str(r).startswith("chimera/")]
    datasets = list(table.datasets) + [
        d for d in df["dataset"].unique() if d not in table.datasets]
    by_ds = {d.lower(): d for d in datasets}
    for rep in reps:
        for col in datasets:
            ds = by_ds.get(col.lower())
            g = df[(df["rep"] == rep) & (df["dataset"] == ds)
                   ] if ds else df.iloc[0:0]
            if not len(g):
                rt.add_row(rep, col, "--", "no eval rows")
                continue
            missing = [r for r in expected if r not in g[rcol].unique()]
            if missing:
                rt.add_row(rep, ds, "--",
                           f"{repeats} {missing}")
            for rep_id, gg in g.groupby(rcol):
                for tcol in table_cols:
                    if tcol not in gg.columns or gg[tcol].isna().any():
                        rt.add_row(rep, ds, str(rep_id), f"metric '{tcol}'")
    return rt


def drop_zero_repeats(df: pd.DataFrame, repeats: str = "seeds",
                      table: TableLayout = None,
                      margin: float = 0.0) -> pd.DataFrame:
    """Drop failed repeat rows of any rep: every table subcol is at or
    below `margin` (decoder collapse, e.g. the EmbedSeg dead basin; with
    the default margin 0.0 that means all-zero). They leave the printed
    table/LaTeX aggregation and are reported as missing (wedge
    superscripts) by summarize()/to_latex(); a partial failure (some
    metric above the margin) keeps the row. NaN metrics never count as
    failed.

    Args:
        df: agglomerate()'s repeat-level table.
        repeats: which column is the repeat axis ("seeds" or "folds").
        table: the table layout (its subcols decide which metrics judge a
            run); default TableLayout().
        margin: metric floor at or below which a repeat counts as failed;
            0.0 keeps only exactly-zero rows.

    Returns:
        The df without the failed repeats.
    """
    rcol = _repeat_col(repeats)
    table = table or TableLayout()
    tag = "s" if repeats == "seeds" else "f"
    table_cols = [table.col_of(s) for s in table.subcols]
    cols = [c for c in table_cols if c in df.columns]
    if not cols:
        return df
    failed = df[cols].le(margin).all(axis=1)
    for _, r in df[failed].iterrows():
        vals = ", ".join(f"{c}={r[c]:.3f}" for c in cols)
        rprint(f"[yellow]dropping failed {r['rep']} {r['dataset']} "
               f"{tag}{int(r[rcol])}: {vals} (all <= {margin})[/yellow]")
    return df[~failed].reset_index(drop=True)


#: folds-mode name kept for callers predating the seeds/folds flag
drop_zero_folds = drop_zero_repeats


def gate_selection_accuracy(runs: Path = None, pattern: str = "*/*",
                            repeats: str = "seeds",
                            margin: float = HEAD_PERF_MARGIN) -> pd.DataFrame:
    """Head-selection accuracy of gated chimera runs, per (dataset, repeat).

    For each matching run with gate.enabled, joins the gated decode's
    per-image PQ (eval/metrics.csv) with every head's per-image PQ
    (eval/metrics_head_<head>.csv). A pick counts as correct when its PQ is
    within `margin` of the best head's PQ — the gate's oracle criterion
    (models.gate.refresh scores heads by PQ). No eval re-run is needed.
    When the base CSV carries a per-image gate_pick column (written by
    evaluate()), the per-head pick fractions are reported too.

    Returns:
        One row per rep=GATE_ACC_REP, dataset, repeat (seed or fold per
        `repeats`) with the accuracy in the PQ column, per-head pick
        fractions in pick_<head> columns (other metric columns NaN), so it
        flows through summarize() into the LaTeX table as the "gate
        accuracy" row with its top pick.
    """
    rcol = _repeat_col(repeats)
    if not runs:
        runs = Path("runs")
    rows = []
    for run_dir in sorted(runs.glob(pattern)):
        if not run_dir.is_dir() or not (run_dir / "config.yaml").exists():
            continue
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        if not (cfg.get("gate") or {}).get("enabled"):
            continue  # ungated chimera decodes with its default head, not the gate
        parsed = _parse_config(run_dir)
        if not parsed or parsed["fold"] not in FOLD_NUMS:
            continue
        base_csv = run_dir / "eval" / "metrics.csv"
        head_csvs = sorted(run_dir.glob("eval/metrics_head_*.csv"))
        if not base_csv.exists() or not head_csvs:
            continue
        base = pd.read_csv(base_csv)
        base = base[base["image_source"] != "mean"]
        best = np.full(len(base), -np.inf)
        joinable = True
        for p in head_csvs:
            h = pd.read_csv(p)
            h = h[h["image_source"] != "mean"]
            if len(h) != len(base) or h["PQ"].isna().any():
                joinable = False
                break
            best = np.maximum(best, h.set_index("image_source")
                              .loc[base["image_source"], "PQ"].to_numpy())
        if not joinable:
            rprint(f"[yellow]gate acc: incomplete head CSVs, skipping "
                   f"{run_dir}[/yellow]")
            continue
        acc = float(np.mean(base["PQ"].to_numpy() >= best - margin))
        picks = _pick_fractions(
            base, [p.stem.removeprefix("metrics_head_") for p in head_csvs])
        rows.append({"rep": GATE_ACC_REP, "dataset": parsed["dataset"],
                     "fold": parsed["fold"], "seed": parsed.get("seed", 0),
                     "PQ": acc, **picks})
    if not rows:
        return pd.DataFrame(columns=["rep", "dataset", "fold", "seed", "PQ"])
    df = pd.DataFrame(rows)
    # several runs of one repeat: average their accuracy into one row
    return df.groupby(["rep", "dataset", rcol], as_index=False).mean(
        numeric_only=True)


def _rich_table(df: pd.DataFrame, repeats: str = "seeds",
                table: TableLayout = None, n_repeats: int = None,
                rep_df: pd.DataFrame = None) -> Table:
    """rich table of the summarized frame: 'mean ± std' per metric cell,
    one row per model rep (chimera/<head> diagnostics excluded), columns
    limited to the table layout's subcols; a red n_repeats marks rows
    missing repeats vs the expected count, a red n_dropped column (when
    present, filled by drop_zeros) counts dropped repeats, and cells with
    mean < RICH_WARN_MARGIN or std > RICH_WARN_MARGIN print yellow with a
    warning listing the per-repeat values and pointing to
    drop_zeros/drop_margin.

    Args:
        df: summarize()'s output (the frame shown).
        repeats: which column is the repeat axis ("seeds" or "folds").
        table: the table layout; default TableLayout().
        n_repeats: expected repeat count for the red n_repeats.
        rep_df: the repeat-level frame behind df (per-repeat values in the
            suspicious-value warning); default None omits them.
    """
    table = table or TableLayout()
    expected = len(_expected_repeats(repeats, n_repeats))
    metric_cols = [table.col_of(s) for s in table.subcols]
    df = df[~df["rep"].astype(str).str.startswith("chimera/")]
    has_dropped = "n_dropped" in df.columns
    rich_tbl = Table(title=f"eval metrics — mean ± std across {repeats}",
                     title_style="bold", expand=False)
    cols = ["rep", "dataset", "n_repeats"]
    if has_dropped:
        cols.append("n_dropped")
    cols += metric_cols
    for col in cols:
        rich_tbl.add_column(col, overflow="ellipsis")
    tag = "s" if repeats == "seeds" else "f"
    rcol = _repeat_col(repeats)
    odd = []
    for _, r in df.iterrows():
        n = int(r["n_repeats"])
        n_cell = str(n) if n >= expected else f"[red]{n}[/red]"
        cells = [str(r["rep"]), str(r["dataset"]), n_cell]
        if has_dropped:
            d = int(r["n_dropped"])
            cells.append(f"[red]{d}[/red]" if d else "0")
        for c in metric_cols:
            mean, std = r.get(f"{c}_mean"), r.get(f"{c}_std")
            if pd.isna(mean):
                cell = "--"
            elif pd.isna(std):
                cell = f"{mean:.3f}"
            else:
                cell = f"{mean:.3f} ± {std:.3f}"
            if not pd.isna(mean) and (mean < RICH_WARN_MARGIN
                                      or (not pd.isna(std)
                                          and std > RICH_WARN_MARGIN)):
                per = ""
                if rep_df is not None:
                    g = rep_df[(rep_df["rep"] == r["rep"])
                               & (rep_df["dataset"] == r["dataset"])]
                    if len(g) and c in g.columns:
                        per = " (" + ", ".join(
                            f"{tag}{int(rr[rcol])}={rr[c]:.3f}"
                            for _, rr in g.iterrows()) + ")"
                odd.append((r["rep"], r["dataset"], c, cell, per))
                cell = f"[yellow]{cell}[/yellow]"
            cells.append(cell)
        rich_tbl.add_row(*cells)
    if odd:
        vals = ", ".join(f"{rep}/{ds} {col} = {val}{per}"
                         for rep, ds, col, val, per in odd)
        rprint(f"[yellow]{len(odd)} suspicious value(s) (mean < "
               f"{RICH_WARN_MARGIN} or std > {RICH_WARN_MARGIN}): {vals} — "
               f"consider --drop_zeros true with --drop_margin above them"
               f"[/yellow]")
    return rich_tbl


def agglomerate(pattern: str = "*/*", runs: Path = None,
                repeats: str = "seeds") -> pd.DataFrame:
    """Collect every matching run's eval metrics into one DataFrame.

    Args:
        pattern: glob over `runs` for run dirs, e.g. "*/*" (group/name)
            or "*" (top-level dirs holding one run each).
        runs: parent dir to glob; default runs.
        repeats: which column is the repeat axis — "seeds" groups by
            (rep, dataset, seed) (seed read from the run config's
            train.seed_everything), "folds" by (rep, dataset, fold). Both
            columns are always emitted.

    Returns:
        One row per (rep, dataset, repeat): the AP@* columns averaged over
        all images of every matching run (the CSV's image_source="mean"
        row is dropped). Chimera runs' eval/metrics_head_<head>.csv files
        are also read, with rep "chimera/<head>" (per-head decodes).
        Runs without eval/metrics.csv are skipped with a warning.
    """
    rcol = _repeat_col(repeats)
    if not runs:
        runs = Path("runs")
    parts = []
    for run_dir in sorted(runs.glob(pattern)):
        if not run_dir.is_dir() or not (run_dir / "config.yaml").exists():
            # not a model run dir (e.g. runs/batch-1/BBBC010-cellpose)
            continue
        csv = run_dir / "eval" / "metrics.csv"
        if not csv.exists():
            rprint(f"[yellow]no eval/metrics.csv: {run_dir}[/yellow]")
            continue
        parsed = _parse_config(run_dir)
        if parsed is None or parsed["fold"] < 0:
            rprint(
                f"[yellow]skip (unparsable config.yaml): {run_dir}[/yellow]")
            continue
        rep, dataset, fold, seed = (
            parsed["rep"], parsed["dataset"], parsed["fold"],
            parsed.get("seed", 0))
        if fold not in FOLD_NUMS:
            # outside the FOLDS list: excluded from every agglomerated
            # table/result
            rprint(
                f"[yellow]skip (fold f{fold} not in FOLDS): {run_dir}[/yellow]")
            continue
        head_csvs = sorted(run_dir.glob("eval/metrics_head_*.csv"))
        for csv_path, rep_name in (
                [(csv, rep)]
                + [(p, f"{rep}/{p.stem.removeprefix('metrics_head_')}")
                   for p in head_csvs]):
            df = pd.read_csv(csv_path)
            # drop the writer's summary row: aggregation is done over images
            df = df[df["image_source"] != "mean"]
            if csv_path == csv:
                for k, v in _pick_fractions(
                        df, [p.stem.removeprefix("metrics_head_")
                             for p in head_csvs]).items():
                    df[k] = v
            df.insert(0, "fold", fold)
            df.insert(1, "seed", seed)
            df.insert(0, "dataset", dataset)
            df.insert(0, "rep", rep_name)
            df.insert(0, "run_dir", str(run_dir))
            parts.append(df)
    if not parts:
        rprint(f"[red]no eval metrics found under {runs}/{pattern}[/red]")
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    # one row per (rep, dataset, repeat): AP columns averaged over all
    # images of all runs sharing the group; per-image std is dropped --
    # the LaTeX table reports std across repeats instead. The inactive
    # repeat column rides along as first() (several runs can merge into
    # one repeat: e.g. folds f0..f4 of one experiment all parse seed 0).
    ap_cols = [c for c in df.columns if c.startswith(("AP@", "matched_iou@",
                                                      "pick_"))
               or c in ("SQ", "RQ", "PQ")]
    other = "fold" if rcol == "seed" else "seed"
    agg = df.groupby(["rep", "dataset", rcol]).agg(
        {**{c: "mean" for c in ap_cols}, other: "first"}).reset_index()
    rprint(f"agglomerated {len(df)} image row(s) from {len(parts)} run(s) "
           f"into {len(agg)} group(s) over {repeats}")
    return agg


def plot_metrics(df: pd.DataFrame, save_dir: Path = None,
                 table: TableLayout = None) -> None:
    """One boxplot PDF per metric column: x = dataset, hue = rep, y = metric.

    Args:
        df: agglomerate()'s repeat-level table (one row per rep, dataset,
            repeat) — boxes show the seed/fold distribution per
            (rep, dataset).
        save_dir: where the PDFs go; default RESULTS_METRICS/all_metrics.
        table: the table layout (its rows fix the rep order); default
            TableLayout().
    """
    if not save_dir:
        save_dir = RESULTS_METRICS / "all_metrics"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    plt_be_stilish()
    table = table or TableLayout()
    # reps in the tex-table order, extra reps appended at the end
    rep_order = list(table.rows) + [
        r for r in df["rep"].unique() if r not in table.rows]
    metric_cols = [c for c in df.columns
                   if c.startswith(("AP@", "matched_iou@"))
                   or c in ("SQ", "RQ", "PQ")]
    for col in metric_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x="dataset", y=col, hue="rep", ax=ax,
                    hue_order=rep_order)
        # col names carry '@' and '.'; keep the file name filesystem-safe
        name = col.replace("@", "_at_").replace(".", "").replace("[", "") \
            .replace("]", "").replace(":", "_")
        out = save_dir / f"{name}.pdf"
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
        rprint(f"[green]saved {out}[/green]")


def agglomerate_metrics(pattern: str = "*/*", runs: Path = None, latex: bool = False,
         drop_zeros: bool = False, head_metrics: bool = False,
         repeats: str = "seeds", n_repeats: int = None,
         table: TableLayout = None, head_perf_margin: float = HEAD_PERF_MARGIN,
         drop_margin: float = 0.0):
    """...
    head_metrics: LaTeX table only -- add the chimera/<head> per-head
        diagnostic rows (default off; chimera and its gate-accuracy row
        always show).
    repeats: which column is the repeat axis, "seeds" (default; the CIS
        batch) or "folds" (the older 5-fold runs). Both columns are always
        emitted; in seeds mode old fold-runs all parse as seed 0.
    n_repeats: expected repeat count for the missing-repeat reports;
        None uses the mode's default (3 seeds / 5 folds).
    table: the table layout (rows, dataset columns, labels, subcols);
        default TableLayout(). Set it in the config file's `table:` block
        for run sets the built-in columns don't cover.
    head_perf_margin: PQ margin counting a gate pick as correct (the
        gate's oracle criterion, gate_selection_accuracy()).
    drop_margin: with drop_zeros, the metric floor at or below which a
        repeat counts as failed and is dropped (0.0 = exactly-zero rows).
    """
    table = table or TableLayout()
    fold_df = agglomerate(pattern, runs, repeats)
    if not len(fold_df):
        return

    RESULTS_METRICS.mkdir(parents=True, exist_ok=True)
    csv_out = RESULTS_METRICS / "all_metrics.csv"
    fold_df.to_csv(csv_out, index=False)
    rprint(f"[green]saved {csv_out}[/green]")

    # drop_zeros affects the printed table, the missing report and the
    # LaTeX table: failed repeats count as missing (daggers); the CSVs and
    # the boxplots keep every repeat
    table_df = drop_zero_repeats(fold_df, repeats, table, drop_margin) \
        if drop_zeros else fold_df

    df = summarize(fold_df, repeats)
    if drop_zeros:
        shown = summarize(table_df, repeats)
        before = fold_df.groupby(["rep", "dataset"]).size()
        after = table_df.groupby(["rep", "dataset"]).size()
        dropped = (before - after.reindex(before.index).fillna(0)) \
            .astype(int).reset_index(name="n_dropped")
        shown = shown.merge(dropped, on=["rep", "dataset"], how="left")
    else:
        shown = df
    rprint(_rich_table(shown, repeats, table, n_repeats, table_df))
    rprint(report_missing(table_df, n_repeats=n_repeats, repeats=repeats,
                          table=table))

    RESULTS_METRICS.mkdir(parents=True, exist_ok=True)
    csv_out = RESULTS_METRICS / "all_metrics_accross_folds.csv"
    df.to_csv(csv_out, index=False)
    rprint(f"[green]saved {csv_out}[/green]")

    plot_metrics(fold_df, table=table)

    if latex:
        gate_df = gate_selection_accuracy(runs, pattern, repeats,
                                          margin=head_perf_margin)
        to_latex(summarize(
            pd.concat([table_df, gate_df], ignore_index=True), repeats),
            n_repeats=n_repeats, head_metrics=head_metrics, repeats=repeats,
            table=table)
        # the dedicated head-selector table (accuracy + most-picked head)
        if len(gate_df):
            to_latex_gate(gate_df, n_repeats=n_repeats, repeats=repeats,
                          table=table)

