import gc
import os
import shutil
import time
import jsonargparse
from time import sleep

from pathlib import Path


from matplotlib import pyplot as plt
from rich.progress import track
import lightning as L
from pluriseg.configs import (AugmentationConfig, BackboneConfig, DatasetConfig, OptimizerConfig,
                            RepresentationConfig, TrainConfig)
from pluriseg.data import SegmentationData
from pluriseg.paths import RESULTS
from pluriseg.representations import REGISTRY
from pluriseg.timing import read_logs, print_table
from pluriseg.display import get_labels_cmap, plt_be_stilish
from rich import print as rprint


def plot_batches(
        data: SegmentationData,
):
    rep = data.rep
    train_dataloader = data.train_dataloader()

    t0 = time.perf_counter()
    batch = next(iter(train_dataloader))
    t_batch = time.perf_counter() - t0
    rprint(
        f"[timing]first batch: {t_batch*1000:.0f} ms (incl. worker startup)")
    batch_targets = batch['targets']
    batch_size = batch['image'].shape[0]

    r = batch_size
    c = 3 + rep.plt_encoded_axes
    fig, axs = plt.subplots(r, c, figsize=(6*c, 6*r),
                            sharex=True, sharey=True)
    for b in track(range(batch_size), description='Plotting samples'):
        s = Path(batch['image_source'][b])
        rprint(f"[b cyan][{b}][/b cyan] [dim]{s.parent}[/dim]/[b]{s.name}[/b]")
        axs[b, 0].imshow(batch['image'][b].permute(1, 2, 0).cpu().numpy())
        axs[b, 0].set_title('image')

        label = batch['label'][b].cpu().numpy()
        cmap = get_labels_cmap(label)
        axs[b, 1].imshow(label, cmap)
        axs[b, 1].set_title('GT labels')

        targets_torch = {
            c: batch_targets[c][b]
            for c in rep.channels.keys()
        }
        targets_np = {
            c: v.permute(1, 2, 0).cpu().numpy()
            for c, v in targets_torch.items()
        }
        rep.display_encoded_plt(
            encoded=targets_np,
            axes=axs[b, 2:2+rep.plt_encoded_axes]
        )
        preds = rep.encoded_to_perfect_pred(
            encoded=targets_np
        )
        label_decoded = rep.decode_preds(
            preds=preds
        )
        cmap = get_labels_cmap(label_decoded)
        axs[b, -1].imshow(label_decoded, cmap)
        axs[b, -1].set_title('decoded label')
    fig.tight_layout()
    return fig


def check_batches(
    rep: RepresentationConfig,
    backbone: BackboneConfig,
    data: DatasetConfig,
    augs: AugmentationConfig,
    optimizer: OptimizerConfig,
    train: TrainConfig,
    all_reps: bool = False,
):
    L.seed_everything(train.seed_everything)

    plt_be_stilish(grid=False)

    reps: list[str] = (
        list(REGISTRY.keys()) if all_reps else [rep.name]
    )

    for rep_name in track(reps, description='Reps'):
        rep_i = rep.model_copy(deep=True)
        rep_i.name = rep_name
        _run_one(rep_i, data, augs, train)


def _run_one(
    rep: RepresentationConfig,
    data: DatasetConfig,
    augs: AugmentationConfig,
    train: TrainConfig,
):
    # identical RNG state per rep: same shuffle order, same crops/augs
    L.seed_everything(train.seed_everything)

    rprint(f"Rep: {rep.name}")

    save_dir = Path(RESULTS) / 'batches' / \
        f"batches_{data.split_name}_{rep.name}.png"

    save_dir.parent.mkdir(parents=True, exist_ok=True)

    timings = save_dir.parent / f'timings-{data.split_name}_{rep.name}'
    shutil.rmtree(timings, ignore_errors=True)  # fresh CSVs per run

    dm = SegmentationData(
        dataset_cfg=data,
        augs=augs,
        rep_cfg=rep,
        patch=train.patch,
        batch_size=train.batch_size,
        val_batch_size=train.val_batch_size,
        num_workers=train.num_workers,
        timing_train_log_dir=timings
    )
    rprint("Loaded SegmentationData")

    dm.setup(stage='fit')
    rprint("SegmentationData setup done !")

    fig = plot_batches(dm)
    fig.savefig(save_dir)
    rprint(f"Saved result to {save_dir}")

    # per-worker CSVs are written at worker exit (atexit); the loader died
    # with plot_batches' frame, give the async worker shutdown a moment
    gc.collect()
    sleep(0.2)
    df = read_logs(timings)
    print_table(df, tag=f"{data.split_name}/{rep.name}")
