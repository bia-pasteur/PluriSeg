"""Training entrypoint.

Usage:
    python -m pluriseg.train --config configs/train/base.yaml
    python -m pluriseg.train --config configs/train/base.yaml --rep.name stardist \
        --rep.params.n_rays 32 --backbone.size medium

Function args are wired by jsonargparse (add_function_arguments): every pydantic
config block is one CLI group, and a single --config yaml maps 1:1 onto the
signature.
"""

import datetime
from pathlib import Path

import jsonargparse
import lightning as L
import torch
import yaml
from lightning.pytorch import callbacks, loggers
from rich import print as rprint
from rich.panel import Panel

from pluriseg.callbacks import (EpochLog, EpochTiming, GradNormMonitor,
                              NanGradGuard)
from pluriseg.configs import (AugmentationConfig, BackboneConfig, DatasetConfig,
                            GateConfig, OptimizerConfig, RepresentationConfig,
                            TrainConfig)
from pluriseg.eval.evaluate import evaluate
from pluriseg.factory import build_model_and_dm
from pluriseg.head_selector.train import train_selection_head, promote_ckpt
from pluriseg.utils import generate_name, git_info

torch.set_float32_matmul_precision('medium')
# Autotune cudnn conv algorithms per input shape. Train/val inputs are a fixed
# patch size, so the tune cost amortizes over an epoch; test decode feeds
# variable full-image shapes and pays a one-off tune per distinct shape.
torch.backends.cudnn.benchmark = True


def announce(model_name: str, run_name: str, dirpath: Path, representation, backbone,
             dataset, optimizer, patch: int, batch_size: int, val_batch_size: int,
             num_workers: int):
    """Pretty-print the run configuration panel."""
    if torch.cuda.is_available():
        devices = ', '.join(
            f'{torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 2**30:.0f} GiB)'
            for i in range(torch.cuda.device_count()))
        device_str = f'GPU: {devices}'
    else:
        device_str = 'CPU'
    rprint(
        f'[bold cyan]Device:[/] {device_str} [dim](torch {torch.__version__})[/]')
    try:
        rprint(Panel(
            f'[bold cyan]Save dir:[/] {dirpath}\n'
            f'[bold cyan]Git:[/] {git_info()}\n'
            f'[bold cyan]Representation:[/] [dim]name=[/]{representation.name} [dim]params=[/]{representation.params}\n'
            f'[bold cyan]Backbone:[/] [dim]size=[/]{backbone.size} [dim]in_ch=[/]{backbone.in_ch}\n'
            f'[bold cyan]Dataset:[/] [dim]root=[/]{dataset.root} [dim]split=[/]{dataset.split_name or "auto"} [dim]fold=[/]{dataset.fold}/{dataset.n_folds}\n'
            f'[bold cyan]Optimizer:[/] [dim]lr=[/]{optimizer.lr} [dim]wd=[/]{optimizer.wd} [dim]epochs=[/]{optimizer.epochs}\n'
            f'[bold cyan]Data:[/] [dim]patch=[/]{patch} [dim]batch=[/]{batch_size} [dim]val=[/]{val_batch_size} [dim]workers=[/]{num_workers}',
            title=f'[bold magenta]{model_name}[/] [dim] / [/] [bold green]{run_name}[/]',
            title_align='left',
            border_style='cyan',
        ))
    except Exception:
        # rich markup can choke on '[' in values (e.g. paths) — plain fallback
        print(
            f'Model: {model_name}  Run: {run_name}\nSave dir: {dirpath}\nRepresentation: {representation}')


def run_training(
    rep: RepresentationConfig,
    backbone: BackboneConfig,
    data: DatasetConfig,
    augs: AugmentationConfig,
    optimizer: OptimizerConfig,
    train: TrainConfig,
    gate: GateConfig = GateConfig(),
):
    L.seed_everything(train.seed_everything)

    if not train.model_name:
        train.model_name = data.split_name + '-' + rep.name

    if not train.run_name:
        timestamp = datetime.datetime.now().strftime(r"%y%m%d-%H%M%S")
        train.run_name = timestamp + '_' + generate_name()

    dirpath = Path(train.run_dir) / train.model_name / train.run_name

    dirpath.mkdir(parents=True, exist_ok=True)
    with open(dirpath / 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump({
            'git_commit': git_info(),
            'representation': rep.model_dump(),
            'backbone': backbone.model_dump(),
            'dataset': data.model_dump(),
            'augs': augs.model_dump(),
            'optimizer': optimizer.model_dump(),
            'train': train.model_dump(),
            'gate': gate.model_dump(),
        }, f, sort_keys=False)

    announce(train.model_name, train.run_name, dirpath, rep, backbone,
             data, optimizer, train.patch, train.batch_size,
             train.val_batch_size, train.num_workers)

    model, dm = build_model_and_dm(
        rep, backbone, optimizer, data, augs, train,
        timing_log_dir=str(dirpath / 'timings'),
        gate_cfg=gate,
    )

    ckpt_cb = callbacks.ModelCheckpoint(
        dirpath=dirpath / 'checkpoints',
        filename=f"best-{{epoch:03d}}-{{{train.monitor}:.2f}}",
        save_top_k=train.save_top_k,
        monitor=train.monitor,
        mode="max",
        save_weights_only=True,
    )
    last_cb = callbacks.ModelCheckpoint(
        dirpath=dirpath / 'checkpoints',
        filename="last",
        every_n_epochs=train.ckpt_every_n_epochs,
        save_top_k=0,
        save_last=True,
        save_weights_only=True,
    )
    # fallback checkpoint on the low-variance PQ metric: AP is a discrete
    # count over decoded instances (oscillates for cluster-based decodes like
    # embedseg), PQ is continuous — the test AP of both checkpoints picks the
    # winner afterwards. Redundant when the monitor already is val_pq
    # (Lightning rejects duplicate ModelCheckpoint state_keys), so skip it.
    ckpt_cb_iou = None
    if train.monitor != "val_pq":
        ckpt_cb_iou = callbacks.ModelCheckpoint(
            dirpath=dirpath / 'checkpoints',
            filename="pq-best-{epoch:03d}-{val_pq:.2f}",
            save_top_k=1,
            monitor="val_pq",
            mode="max",
            save_weights_only=True,
        )
    trainer = L.Trainer(
        max_epochs=optimizer.epochs,
        limit_train_batches=(
            train.batches_per_epoch if train.batches_per_epoch else 1.0),
        default_root_dir=train.run_dir,
        precision="bf16-mixed",
        gradient_clip_val=train.gradient_clip_val,
        callbacks=[
            last_cb,
            ckpt_cb,
            *[cb for cb in (ckpt_cb_iou,) if cb is not None],
            callbacks.LearningRateMonitor(logging_interval="epoch"),
            GradNormMonitor(),
            NanGradGuard(),
            EpochLog(),
            EpochTiming(),
        ],
        logger=loggers.CSVLogger(
            save_dir=train.run_dir, name=train.model_name, version=train.run_name
        ),
        log_every_n_steps=train.log_every_n_steps,
        check_val_every_n_epoch=train.check_val_every_n_epoch,
        enable_progress_bar=train.enable_progress_bar,
        detect_anomaly=train.debug,
    )
    trainer.fit(model, dm)

    best_path = ckpt_cb.best_model_path
    best = ckpt_cb.best_model_score
    if best is not None:
        best_str = f'{train.monitor}={best.item():.4f}' if hasattr(
            best, 'item') else f'{train.monitor}={best:.4f}'
    else:
        best_str = 'n/a'
    rprint(
        f'[bold green]Training finished[/] — best checkpoint: {best_path} ({best_str})')

    # Post-hoc gate: the router trains on no gate signal during fit; refit it
    # once on all val images scored under the FROZEN deployment checkpoint
    # (frozen weights make every scored entry valid forever). The refit ckpt
    # is promoted onto best-*.ckpt (original moved to pre-refit-*) so eval_
    # picks up the trained router, and the test eval below runs on
    # the same weights the bank was scored with.
    if gate.enabled and model.gated is not None:
        dep_path = Path(best_path) if best_path else dirpath / \
            'checkpoints' / 'last.ckpt'
        if not dep_path.exists():
            rprint(f'[red]post-gate: no checkpoint at {dep_path} — '
                   f'router stays untrained[/]')
        else:
            rprint(f'[bold cyan]Post-gate[/] refitting router on val bank '
                   f'under [dim]{dep_path.name}[/]')
            model.load_state_dict(
                torch.load(dep_path, weights_only=False)["state_dict"])
            stats = train_selection_head(
                model, dm, gate, optimizer,
                train.seed_everything
            )
            if stats is not None:
                rprint(
                    f'[bold green]post-gate refit[/] val routed PQ '
                    f'{stats["routed_pq"]:.3f} vs best-head '
                    f'{stats["best_pq"]:.3f} | argmax match '
                    f'{stats["argmax_match"]:.2f} | {gate.post_epochs} ep, '
                    f'{gate.post_seeds} seed(s)')
                ck = torch.load(dep_path, weights_only=False)
                for k, v in model.gated.router.state_dict().items():
                    key = f"gated.router.{k}"
                    assert key in ck["state_dict"] \
                        and ck["state_dict"][key].shape == v.shape, key
                    ck["state_dict"][key] = v
                refit_path = dirpath / 'checkpoints' / f'refit-{dep_path.name}'
                torch.save(ck, refit_path)
                promote_ckpt(dirpath, refit_path)
                rprint(f'[green]promoted {refit_path.name} → best-*.ckpt[/]')

    if train.test_after_training:
        try:
            dm.setup('test')
            rprint(
                f'[bold cyan]Test eval starting[/] — {len(dm.test_ds)} images')
            evaluate(model, dm, out_dir=dirpath / 'eval',
                     enable_progress_bar=train.enable_progress_bar)
            iou_path = ckpt_cb_iou.best_model_path if ckpt_cb_iou else ""
            if iou_path and iou_path != ckpt_cb.best_model_path:
                rprint(
                    f'[bold cyan]Test eval of IoU-fallback ckpt[/] — {iou_path}')
                iou_ckpt = torch.load(iou_path, weights_only=False)
                model.load_state_dict(iou_ckpt['state_dict'])
                evaluate(model, dm, out_dir=dirpath / 'eval-iou',
                         enable_progress_bar=train.enable_progress_bar)
        except Exception as e:
            rprint(f'[bold red]Test loop failed[/] ({type(e).__name__}: {e})')

    rprint(f'[bold green]ALL DONE[/] — {train.run_name}')


if __name__ == "__main__":
    jsonargparse.auto_cli(
        run_training,
        as_positional=False,
        default_config_files=['configs/train/base.yaml']
    )
