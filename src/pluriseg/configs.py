"""Pydantic config models for the PluriSeg benchmark. See configs/*.yaml for defaults."""

from dataclasses import fields
from typing import get_origin

import importlib

from fluometrik import Op
from fluometrik.ops import OPS
from pydantic import BaseModel, Field

from pluriseg.paths import CACHE, SPLITS


class OptimizerConfig(BaseModel):
    lr: float = 3e-4
    wd: float = 1e-5
    epochs: int = 200
    warmup_epochs: int = 5  # linear LR warm-up before cosine decay


class AffineTransformConfig(BaseModel):
    scale_range: tuple[float, float] = [0.8, 1.2]
    rotation: bool = True


class AugmentationConfig(BaseModel):
    flip_x: bool = True
    flip_y: bool = True
    rot90: bool = True
    affine: AffineTransformConfig | None = None
    # fluometrik pipeline as pure data: flat dicts with an "op" key holding
    # the class name from the fluometrik.OPS registry, e.g.
    # {"op": "Blur", "p": 0.4, "sigma": [0.37, 1.83]}. Ops are instantiated
    # in PatchDataset; empty = no photometric augmentation. Lossless under
    # model_dump()/yaml round-trip.
    photometric: list[dict] = Field(default_factory=list)


class BackboneConfig(BaseModel):
    size: str = "small"
    in_ch: int = 3


class RepresentationConfig(BaseModel):
    name: str = "mask"
    params: dict = Field(default_factory=dict)


class GateConfig(BaseModel):
    """Per-image gate head-pick for ChimeraRep"""

    enabled: bool = False
    hidden: int = 128
    # CE target temperature softmax(PQ / tau); lower = harder labels
    tau: float = 0.1
    # router input: gap | stats (GAP,GMP,std) | stats_grid (+2x2 grid)
    pool: str = "gap"
    post_lr: float = 1e-3  # router refit lr (full-batch AdamW)
    post_epochs: int = 500  # full-batch refit epochs
    post_seeds: int = 3  # fresh router inits; best val routed-PQ kept
    post_val_frac: float = 0.2  # bank fraction held out for refit selection


class DatasetConfig(BaseModel):
    # Single-root mode: explicit split file relative to root.
    root: str = ""
    # Split directory under splits_dir; may differ from the root dir name.
    split_name: str = ""
    splits_dir: str = str(SPLITS)
    fold: int = 0
    n_folds: int = 1
    cache_dir: str = str(CACHE)
    # Expected image channel count; 0 = use the image as-is. When set,
    # prepare_image normalizes any input to this many channels
    # (1 -> replicate x3, 4 (RGBA) -> drop alpha) so a shared backbone can
    # train on mixed sources. Asserted equal to BackboneConfig.in_ch.
    in_ch: int = 0
    # Balance train sampling across sources (prefix before '-' in cache_id):
    # WeightedRandomSampler with per-source weights 1/n_source. Val/test
    # keep the natural distribution regardless.
    balanced_sampling: bool = False
    # Pre-build rep-target cache entries (label-only) in DataModule.setup,
    # so the first epoch / first val doesn't stall on encode misses.
    warm_cache: bool = True


class TrainConfig(BaseModel):
    """Loose training scalars, grouped for a single config dump."""
    seed_everything: int = 0
    patch: int = 256
    batch_size: int = 4
    val_batch_size: int = 1
    num_workers: int = 4
    # Cap train batches per epoch (Lightning limit_train_batches) — fixes the
    # images-per-epoch budget across datasets of different sizes; each epoch
    # draws a fresh random subsample (shuffled loader). None = full epoch.
    batches_per_epoch: int | None = None
    run_dir: str = "runs"
    model_name: str | None = None
    run_name: str | None = None
    save_top_k: int = 2
    ckpt_every_n_epochs: int = 10
    check_val_every_n_epoch: int = 1
    log_every_n_steps: int = 10
    gradient_clip_val: float = 0.0
    enable_progress_bar: bool = True
    test_after_training: bool = True
    monitor: str = "val_ap75"
    debug: bool = False  # raise on NaN/Inf losses instead of warning


def load_photometric(specs: list[dict] | None) -> list[Op]:
    """Instantiate fluometrik Ops from flat specs via the fluometrik.OPS
    registry. Also accepts jsonargparse class_path/init_args entries (legacy
    configs). Unknown op names are dropped."""
    registry = {cls.__name__: cls for cls in OPS}

    ops = []
    for entry in specs or []:
        if 'op' in entry:
            cls = registry.get(entry['op'])
            if cls is None:
                continue
            kwargs = {k: v for k, v in entry.items() if k != 'op'}
            # yaml lists back to tuple ranges, per the dataclass annotations
            hints = {f.name: f.type for f in fields(cls)}
            kwargs = {
                k: tuple(v) if get_origin(hints.get(k)) is tuple else v
                for k, v in kwargs.items()
            }
            ops.append(cls(**kwargs))
        elif 'class_path' in entry:
            module, _, cls_name = entry['class_path'].rpartition('.')
            cls = registry.get(cls_name)
            if cls is None:
                cls = getattr(importlib.import_module(module), cls_name)
            ops.append(cls(**entry.get('init_args', {})))
    return ops
