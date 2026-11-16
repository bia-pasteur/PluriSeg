"""Whole-image evaluation loop: predict, save preds, per-image metrics CSV."""

from contextlib import nullcontext
from pathlib import Path
import re

import imageio.v3 as iio
import numpy as np
import pandas as pd
import torch
import yaml
from rich import print as rprint
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from pluriseg.configs import AugmentationConfig, DatasetConfig, TrainConfig
from pluriseg.data import SegmentationData
from pluriseg.eval.metrics import instance_ap, matched_mean_iou, panoptic_quality
from pluriseg.factory import build_model_and_dm
from pluriseg.model import UnifedSegmentationModel
from pluriseg.utils import save_masks



def evaluate(model: UnifedSegmentationModel, dm: SegmentationData,
             out_dir: Path | str = 'eval',
             enable_progress_bar: bool = True) -> pd.DataFrame:
    """Whole-image eval loop: predict, save preds, per-image metrics CSV.

    The caller owns checkpoint loading and model state; the model is put in
    eval mode here. Iterates test images one by one (no trainer.test); AP is
    computed on the original extent (pad cropped before metrics). Per-image
    rows are collected and saved to out_dir/metrics.csv with a mean row
    (image_source="mean"); preds are saved as <cache_id>_pred.tif.

    When the representation has sub-representations (ChimeraRep), every
    sub-head is additionally decoded per image and written to
    out_dir/metrics_head_<head>.csv (same schema, one mean row per file);
    those metrics are the sub-rep's standalone decode, while the base CSV
    keeps the run's default decode (gate pick).
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()

    dm.setup('test')
    ds = dm.test_ds
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'metrics.csv'

    n = len(ds)
    if enable_progress_bar:
        progress = Progress(
            TextColumn('[progress.description]{task.description}'),
            BarColumn(), TaskProgressColumn(), TimeElapsedColumn())
        rprint(f'[bold cyan]Evaluating[/] {n} images → [dim]{out_dir}[/]')
    else:
        rprint(f'[bold cyan]Evaluating[/] {n} images → [dim]{out_dir}[/]')
        progress = nullcontext()
    rows_by_head: dict[str, list] = {}
    with progress as p:
        task = p.add_task('eval', total=n) if enable_progress_bar else None
        for i in range(n):
            image_file, label_file, cache_id = ds.image_label_pairs[i]
            item = ds[i]
            image = item['image'].to(device)
            h, w = iio.imread(image_file).shape[:2]
            gt = item['label'].numpy().astype(np.uint32)[:h, :w]
            if getattr(model.rep, 'subs', None):
                decodes = model.predict_labels_all_heads(image)
            else:
                decodes = {'': model.predict_labels(image)}
            # the gate's pick for this image (gated chimera only; argmax of
            # the router logits set by the gated "" decode above)
            pick = (max(model.gated.last_gate, key=model.gated.last_gate.get)
                    if model.gated and model.gated.last_gate else None)
            for head, pred in decodes.items():
                ap = instance_ap(gt, pred)
                # matched_iou@0 = every matched pair (quality-only); PQ per
                # Kirillov 2019 (panoptic_quality handles SQ/RQ internally).
                miou0 = matched_mean_iou(gt, pred)
                pq = panoptic_quality(gt, pred)
                rows_by_head.setdefault(head, []).append(
                    {'image_source': str(image_file), **ap,
                     'matched_iou@0': miou0['matched_iou'],
                     'matched_iou@0.5': pq['SQ'],
                     'SQ': pq['SQ'], 'RQ': pq['RQ'], 'PQ': pq['PQ'],
                     **({'gate_pick': pick}
                        if head == '' and pick is not None else {})})
            save_masks(
                decodes[''], out_dir / f'{cache_id}_pred.tif',
            )
            if enable_progress_bar:
                p.advance(task)

    def _write(rows: list, csv: Path) -> pd.DataFrame:
        df = pd.DataFrame.from_records(rows)
        mean = df.mean(numeric_only=True)
        out = pd.concat(
            [df, pd.DataFrame([{'image_source': 'mean', **mean.to_dict()}])],
            ignore_index=True)
        out.to_csv(csv, index=False)
        return out

    df = pd.DataFrame.from_records(rows_by_head.get('', []))
    if df.empty:
        rprint('[yellow]No test images; nothing to save[/]')
        rprint(f'[bold green]EVAL DONE[/] — 0 images')
        return df
    df_out = _write(rows_by_head[''], csv_path)
    mean = df_out[df_out['image_source'] != 'mean'].mean(numeric_only=True)

    for head, head_rows in rows_by_head.items():
        if head == '':
            continue
        head_csv = out_dir / f'metrics_head_{head}.csv'
        hmean = _write(head_rows, head_csv)
        hmean = hmean[hmean['image_source'] != 'mean'].mean(numeric_only=True)
        rprint(
            f'[bold green]EVAL DONE[/] head [dim]{head}[/] — {n} images → '
            f'[dim]{head_csv}[/] | AP@0.50 {hmean["AP@0.50"]:.4f} | '
            f'PQ {hmean["PQ"]:.4f}')

    rprint(
        f'[bold green]EVAL DONE[/] — {n} images → [dim]{csv_path}[/] | '
        f'AP@0.50 {mean["AP@0.50"]:.4f} | AP@[.5:.95] {mean["AP@[.5:.95]"]:.4f} | '
        f'PQ {mean["PQ"]:.4f}')
    return df_out


def _config_head_names(run_dir: Path) -> list[str]:
    """Sub-head names configured in the run's config.yaml (chimera only;
    [] for any other or unparsable config)."""
    try:
        with open(run_dir / "config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return list(cfg["representation"]["params"]["sub_params"])
    except Exception:
        return []


def eval_(
    ckpt_path: str | Path,
    data: DatasetConfig | None = None,
    out_dir: str | None = None,
    enable_progress_bar: bool = True,
):
    """Standalone eval entrypoint: evaluate a checkpoint on a dataset.

    --ckpt_path accepts a checkpoint file or a run directory (the one holding
    config.yaml): in the latter case the best checkpoint (highest epoch) is
    used. rep/backbone/optimizer come from the checkpoint hparams, and
    data/augs/train from the config.yaml saved next to it. Passing --data
    overrides the saved dataset (e.g. another split or fold); only --data.*
    fields actually given are applied, the rest stay from config.yaml.

    out_dir defaults to the run directory (where config.yaml lives) + /eval.
    """
    ckpt_path = Path(ckpt_path)
    if ckpt_path.is_dir():
        run_dir = ckpt_path
        ckpts = sorted(run_dir.glob('checkpoints/best-*.ckpt'))
        if not ckpts:
            raise FileNotFoundError(
                f'no best-*.ckpt in {run_dir / "checkpoints"}')
        # filename is best-epoch=NNN-<monitor>=SCORE.ckpt: pick the best score
        def score(p: Path) -> float:
            m = re.search(r'=-?([\d.]+)\.ckpt$', p.name)
            return float(m.group(1)) if m else float('-inf')
        ckpt_path = max(ckpts, key=score)
        rprint(f'[bold cyan]Eval[/] run dir given → checkpoint: [dim]{ckpt_path}[/]')
    elif not ckpt_path.exists():
        raise FileNotFoundError(f'checkpoint not found: {ckpt_path}')

    cfg_dir = next((p for p in (ckpt_path, *ckpt_path.parents)
                    if (p / 'config.yaml').exists()), None)
    if cfg_dir is None:
        raise FileNotFoundError(
            f'no config.yaml found next to {ckpt_path}; pass --data explicitly')
    with open(cfg_dir / 'config.yaml', encoding='utf-8') as f:
        saved = yaml.safe_load(f)
    rprint(f'[bold cyan]Eval[/] config: [dim]{cfg_dir / "config.yaml"}[/]')

    train = TrainConfig(**saved['train'])
    augs = AugmentationConfig(**saved['augs'])
    if data is not None:
        saved_data = saved['dataset']
        given = {k: v for k, v in data.model_dump(
            exclude_defaults=True).items()}
        data = DatasetConfig(**{**saved_data, **given})
        rprint(
            f'[bold cyan]Eval[/] dataset overridden: [dim]root=[/]{data.root} '
            f'[dim]split=[/]{data.split_name or "auto"} '
            f'[dim]fold=[/]{data.fold}/{data.n_folds}')
    else:
        data = DatasetConfig(**saved['dataset'])
    rprint(
        f'[bold cyan]Eval[/] dataset: [dim]root=[/]{data.root} '
        f'[dim]split=[/]{data.split_name or "auto"} '
        f'[dim]fold=[/]{data.fold}/{data.n_folds} '
        f'[dim]patch=[/]{train.patch}')

    # gate comes back via hparams: save_hyperparameters() stored gate_cfg, so
    # load_from_checkpoint rebuilds GatedChimera with the checkpoint weights
    model = UnifedSegmentationModel.load_from_checkpoint(
        str(ckpt_path), weights_only=False)
    rep, backbone, optimizer = (
        model.hparams.rep_cfg, model.hparams.backbone_cfg,
        model.hparams.optimizer_cfg)
    model.debug = train.debug
    rprint(
        f'[bold cyan]Eval[/] model: [dim]rep=[/]{rep.name} '
        f'[dim]backbone=[/]{backbone.size} [dim]ckpt=[/]{ckpt_path}')

    dm = build_model_and_dm(rep, backbone, optimizer, data, augs, train)[1]

    out = Path(out_dir) if out_dir else cfg_dir / 'eval'
    rprint(f'[bold cyan]Eval[/] output: [dim]{out}[/]')
    evaluate(model, dm, out_dir=out,
             enable_progress_bar=enable_progress_bar)


REQUIRED_METRIC_COLUMNS = (
    "AP@0.25", "AP@0.50", "AP@0.75", "AP@[.5:.95]",
    "matched_iou@0", "matched_iou@0.5", "SQ", "RQ", "PQ",
)



def metrics_complete(run_dir: Path) -> bool:
    """True when run_dir/eval/metrics.csv exists and carries every required
    metric column (incomplete CSVs, e.g. pre-matched_iou, count as missing).

    Chimera runs additionally need one complete metrics_head_<head>.csv per
    configured sub-head (per-head metrics were added later; runs evaluated
    before that count as missing so batch re-evaluation backfills them)."""
    csv = run_dir / "eval" / "metrics.csv"
    if not csv.exists():
        return False
    try:
        header = pd.read_csv(csv, nrows=0)
    except Exception:
        return False
    if not all(c in header.columns for c in REQUIRED_METRIC_COLUMNS):
        return False
    for sname in _config_head_names(run_dir):
        head_csv = run_dir / "eval" / f"metrics_head_{sname}.csv"
        if not head_csv.exists():
            return False
        try:
            head_header = pd.read_csv(head_csv, nrows=0)
        except Exception:
            return False
        if not all(c in head_header.columns for c in REQUIRED_METRIC_COLUMNS):
            return False
    return True
