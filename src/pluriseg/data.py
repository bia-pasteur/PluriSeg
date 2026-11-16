"""Data module: patch loading, representation encoding (+ on-disk cache), and augmentation.

Batch contract (both train and val):
    {"image":   (B, in_ch, H, W) float32,
     "targets": {name: (B, C, H, W) float32},   # encoded by the shared Representation
     "label":   (B, H, W) int64}                # raw instance labels (0 = bg)

The spatial shape is identical across image / targets / label in every batch:

  - train: a random patch×patch crop of a full image, then D4-augmented
    (crop first, transform within the patch — exact, no resampling).
  - val: one deterministic patch×patch crop per image (center-anchored;
    zero-padded bottom/right if the image is smaller than the patch), so all
    batches are plain patch stacks. AP is computed on the patch extent.
  - test: the full image, zero-padded to the next multiple of 32 (the
    backbone needs 5 downsamples); all three members padded identically, so
    preds and label share shape and decode_preds works directly. AP is computed on
    the padded extent — pad is GT-background, so spillover predictions cost
    precision (exactness returns with tiled inference).

Two Representation instances are built from the same rep_cfg (model + here).
encode_targets/transform are stateless, so the instances are interchangeable; the
config is the single source of truth.

On-disk target cache (only when rep.cacheable): <cache_dir>/<dataset_name>/<rep.name>-<rep.get_hash()>/<cache_id>.npz
(cache_id comes from the split file; it joins the regex capture groups, so
e.g. w1/w2 variants of the same well stay distinct)
with one array per channel name, shaped (H, W) for C=1 and (H, W, C) otherwise.
Written lazily on first miss; safe under dataloader workers (content is a
pure function of label + rep config).
"""

import collections
import os
import sys
from pathlib import Path
import warnings

import imageio.v3 as iio
import lightning as L
import numpy as np
import torch
import torch.utils.data as data
from fluometrik import Pipeline
from rich import print as rprint
from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                            TextColumn, TimeRemainingColumn)

from pluriseg.configs import (AugmentationConfig, DatasetConfig,
                            RepresentationConfig, load_photometric)
from pluriseg.representations import build_representation
from pluriseg.representations.base import Representation
from pluriseg.splits import load_split
from pluriseg.transforms import AffineTransformV2, OrthoTransform, normalize_percentile
from pluriseg.timing import Timer
from pluriseg.utils import load_masks

# Backbone downsamples 2^5; inputs must be divisible by this for UNet forward.
DIVISOR = 32


def pad_to_multiple(arr: np.ndarray, divisor: int = DIVISOR) -> np.ndarray:
    """Zero-pad (H, W) or (H, W, C) bottom/right to H, W multiples of divisor."""
    h, w = arr.shape[:2]
    ph = (divisor - h % divisor) % divisor
    pw = (divisor - w % divisor) % divisor
    if ph == 0 and pw == 0:
        return arr
    pad = [(0, ph), (0, pw)] + [(0, 0)] * (arr.ndim - 2)
    return np.pad(arr, pad, mode="constant")


def center_patch(arr: np.ndarray, patch: int) -> np.ndarray:
    """Deterministic center crop of patch×patch; zero-padded (centered) if the
    array is smaller than the patch."""
    h, w = arr.shape[:2]
    h_ins, w_ins = min(patch, h), min(patch, w)
    top, left = max((h - h_ins) // 2, 0), max((w - w_ins) // 2, 0)
    crop = arr[top:top + h_ins, left:left + w_ins]
    pad_h, pad_w = patch - h_ins, patch - w_ins
    if pad_h or pad_w:
        crop = np.pad(
            crop,
            [(pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)]
            + [(0, 0)] * (crop.ndim - 2),
            mode="constant",
        )
    return crop


def normalize_channels(image: np.ndarray, in_ch: int) -> np.ndarray:
    """Normalize channel count to `in_ch` (H, W[, C]) -> (H, W, in_ch).

    1 -> replicate x3, 4 (RGBA) -> drop alpha. Only 1/3/4 are meaningful for
    microscopy inputs; anything else passes through unchanged.
    """
    if image.ndim == 2:
        image = image[..., None]
    c = image.shape[-1]
    if in_ch == c:
        return image
    if in_ch == 3 and c == 1:
        return np.repeat(image, 3, axis=-1)
    if in_ch == 3 and c == 4:
        return image[..., :3]
    raise ValueError(
        f"normalize_channels: cannot map {c} channels -> {in_ch}")


def prepare_image(image: np.ndarray, in_ch: int = 0) -> np.ndarray:
    """Shared pre-geometry step used by __getitem__ and inference:
    (H,W)->(H,W,C), float32, percentile-normalized; optional channel
    normalization to `in_ch` (see normalize_channels)."""
    if in_ch:
        image = normalize_channels(image, in_ch)
    if image.ndim == 2:
        image = image[..., None]
    return normalize_percentile(image.astype(np.float32))


def to_chw(arr: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    """(H,W,C) -> (C,H,W) tensor. Used for image and encoded target maps."""
    return torch.from_numpy(arr).to(dtype).permute(2, 0, 1)


def numpy_image_to_model_input(image: np.ndarray, in_ch: int = 0) -> torch.Tensor:
    """Inference preprocessing: (H,W[,C]) raw image -> (C,H,W) tensor. The
    /32 pad lives in the model (predict_labels pads around the backbone).
    `in_ch` must match the value the training dataset used (DatasetConfig.in_ch)."""
    return to_chw(prepare_image(image, in_ch=in_ch), torch.float32)


def crop_output_to_input_shape(output: np.ndarray, input_image: np.ndarray) -> np.ndarray:
    """Truncate an (H', W') prediction back to the valid (H, W) region of the
    pre-padding input image."""
    if input_image.ndim == 3:
        # channel-last: C is the smaller dim (raise if ambiguous, C > min(H, W))
        H, W, C = input_image.shape
        if C > min(H, W):
            raise ValueError(
                f"cannot tell H, W apart in shape {input_image.shape}")
    elif input_image.ndim == 2:
        H, W = input_image.shape
    else:
        raise ValueError(
            f"input_image must be (H, W) or (H, W, C), got shape {input_image.shape}")
    if output.shape[:2] < (H, W):
        raise ValueError(
            f"output {output.shape[:2]} smaller than input region {(H, W)}")
    return output[:H, :W]


def _write_cache_file(cache_file: Path, maps: dict[str, np.ndarray]) -> None:
    """Atomic npz write: tmp name is unique per pid (must end in .npz or
    savez appends it), os.replace is atomic on POSIX, so concurrent jobs
    and dataloader workers can fill the same cache concurrently."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_file.with_suffix(f".tmp{os.getpid()}.npz")
    try:
        np.savez(tmp, **maps)
        os.replace(tmp, cache_file)
    finally:
        tmp.unlink(missing_ok=True)


def _warm_worker_init_fn(worker_id: int) -> None:
    """Cap intra-op threads in warm workers so n_workers * threads stays
    near the core count (the training loader's _worker_init_fn policy).
    Neutral for cellpose (measured); stardist's njit prange pool would
    otherwise scale with every worker."""
    torch.set_num_threads(1)
    try:
        import numba
        numba.set_num_threads(2)
    except (ImportError, ValueError):
        pass


class _CacheWarmDataset(data.Dataset):
    """Label-only encode+write for cache misses — the parallel twin of
    PatchDataset._load_maps' miss path, run in dataloader workers by
    warm_cache(). Returns the cache_id regardless of who wrote the file,
    so a worker racing another process just no-ops."""

    def __init__(self, items: list[tuple[str, str]], hash_dir: Path, rep: Representation):
        self.items = items
        self.hash_dir = hash_dir
        self.rep = rep

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> str:
        label_file, cache_id = self.items[idx]
        cache_file = self.hash_dir / f"{cache_id}.npz"
        if not cache_file.exists():  # may have raced another worker/process
            label = load_masks(label_file).astype(np.uint32)
            _write_cache_file(cache_file, self.rep.encode_targets(label))
        return cache_id


class PatchDataset(data.Dataset):
    """One full image per item; cropping/augmentation happens in __getitem__.

    train: random crop of `patch` px + random D4 + intensity shift/scale.
    val:   deterministic center patch crop, no augmentation.
    test:  full image padded to /32, no augmentation.
    """

    def __init__(
        self,
        image_label_pairs: list[tuple[str, str, str]],
        cache_dir: str,
        dataset_name: str,
        rep: Representation,
        cfg: DatasetConfig,
        augs: AugmentationConfig,
        patch: int,
        train: bool,
        mode: str = 'test',
        timing_log_dir: str = ""
    ):
        self.image_label_pairs = image_label_pairs
        if not self.image_label_pairs:
            warnings.warn(
                f"Creating a patch dataset with zero images "
                f"(dataset={dataset_name}, mode={mode}, train={train}) !"
            )
        self.dataset_name = dataset_name

        self.cache_dir = Path(cache_dir) / self.dataset_name
        os.makedirs(self.cache_dir, exist_ok=True)

        self.in_ch = cfg.in_ch
        self.rep = rep
        self.cfg = cfg
        self.augs = augs
        # photometric ops arrive as flat data specs; instantiate once here
        # (train-only: the pipeline is unused on val/test datasets)
        self.photometric_pipeline = Pipeline(
            steps=load_photometric(augs.photometric))
        self.patch = patch
        self.train = train
        self.mode = mode
        # Reproducible when PL_GLOBAL_SEED is set (Lightning does); the
        # datamodule's worker_init_fn then gives each worker its own seeded
        # generator whose state persists and advances across epochs.
        env_seed = os.environ.get("PL_GLOBAL_SEED")
        self.base_seed = int(env_seed) if env_seed is not None else None
        self.rng = np.random.default_rng(self.base_seed)
        self.timer = Timer(log_dir=timing_log_dir)
        if self.timer.enabled:
            # covers the main process (num_workers=0); dataloader workers get
            # theirs from _worker_init_fn (spawn re-pickles, skipping __init__)
            self.timer.install_exit_hook()

    def set_worker_seed(self, worker_id: int) -> None:
        """Called from the dataloader's worker_init_fn (main process only
        when num_workers == 0)."""
        if self.base_seed is None:
            return
        self.rng = np.random.default_rng([self.base_seed, worker_id])

    def __len__(self) -> int:
        return len(self.image_label_pairs)

    # -- loading --------------------------------------------------------------

    def _get_cache_file(self, cache_id: str) -> Path:
        return self.cache_dir / f"{self.rep.name}-{self.rep.get_hash()}" / f"{cache_id}.npz"

    def _load_maps(self, cache_id: str, label: np.ndarray) -> dict[str, np.ndarray]:
        """Encoded targets from cache when possible, else encode_targets (+ write)."""
        if not self.rep.cacheable:
            with self.timer.stage("encode"):
                return self.rep.encode_targets(label)
        cache_file = self._get_cache_file(cache_id)
        if cache_file.exists():
            with self.timer.stage("cache load"):
                with np.load(cache_file) as z:
                    return {k: z[k] for k in self.rep.channels.keys()}
        with self.timer.stage("encode"):
            maps = self.rep.encode_targets(label)
        with self.timer.stage("cache write"):
            _write_cache_file(cache_file, maps)
        return maps

    def warm_cache(self, n_workers: int = 0) -> None:
        """Pre-build every missing cache entry so the first epoch/val pass
        doesn't stall on encode misses. Label-only (no image load, no
        aug/crop work); entries already on disk are skipped; no-ops for
        non-cacheable reps. With n_workers > 0 the fill runs in dataloader
        worker processes (same fork machinery as __getitem__, same atomic
        writes), which is also faster than the lazy in-loader fill."""
        if not self.rep.cacheable or not self.image_label_pairs:
            return
        todo = [
            (label_file, cache_id)
            for _, label_file, cache_id in self.image_label_pairs
            if not self._get_cache_file(cache_id).exists()
        ]
        if not todo:
            return
        n_workers = min(n_workers, len(todo))
        rprint(f"[dim]warming cache: {len(todo)}/{len(self.image_label_pairs)} "
               f"{self.mode} images missing ({max(n_workers, 1)} worker(s))")
        warm_ds = _CacheWarmDataset(
            todo, self.cache_dir / f"{self.rep.name}-{self.rep.get_hash()}",
            self.rep)
        loader = data.DataLoader(
            warm_ds,
            batch_size=1,  # tick the bar per image, not per 16-item chunk
            num_workers=n_workers,
            collate_fn=lambda batch: batch,
            worker_init_fn=_warm_worker_init_fn,
        )
        done = 0
        is_tty = sys.stdout.isatty()
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),  # n/tot, not percent
            TimeRemainingColumn(),
            transient=is_tty,  # non-tty prints the final bar once, no live spam
        ) as prog:
            task = prog.add_task(f"warm cache {self.mode}", total=len(todo))
            for _ in loader:
                done += 1
                prog.update(task, completed=done)
        if not is_tty:
            rprint(f"[dim]cache warm: wrote {done} {self.mode} entries")

    # -- geometry & intensity ---------------------------------------------------

    def _deterministic_crop(self, *arrays: np.ndarray) -> list[np.ndarray]:
        """Patch centered on the image; zero-pads (centered) if the image is
        smaller than patch."""
        return [center_patch(a, self.patch) for a in arrays]

    def _random_crop(
        self,
        *arrays: np.ndarray,
        margin: int = 0,
        fill_value: float = 0.0,
    ) -> list[np.ndarray]:
        """Random patch crop; with margin > 0, crop patch + 2*margin so an
        affine warp can run on the wider context: the image is padded with
        `margin` of fill_value on all sides first, then the crop's top-left
        is uniform over (0..H-patch, 0..W-patch), so the inner patch
        crop[margin:margin+patch] stays within the original bounds.
        margin=0 reduces to the plain patch crop."""
        if margin:
            arrays = [
                np.pad(
                    a,
                    [(margin, margin)] * 2 + [(0, 0)] * (a.ndim - 2),
                    mode="constant",
                    constant_values=fill_value,
                )
                for a in arrays
            ]
        size = self.patch + 2 * margin
        h, w = arrays[0].shape[:2]
        ph, pw = max(size - h, 0), max(size - w, 0)
        if ph or pw:
            # undersized image: pad at a random offset so patch position varies
            top = int(self.rng.integers(0, ph + 1))
            left = int(self.rng.integers(0, pw + 1))
            arrays = [np.pad(
                a,
                [(top, ph - top), (left, pw - left)] + [(0, 0)] * (a.ndim - 2),
                mode="constant",
                constant_values=fill_value,
            ) for a in arrays]
            h, w = size, size
        top = int(self.rng.integers(0, h - size + 1))
        left = int(self.rng.integers(0, w - size + 1))
        return [a[top:top + size, left:left + size] for a in arrays]

    def _affine_margin(
        self, rot_deg: float | None = None, scale: float | None = None
    ) -> int:
        """Margin (px) so the affine warp of a patch+2*margin crop keeps the
        center patch fully covered with real data.

        The output center patch back-maps through the inverse affine to a
        source square of side patch/scale rotated by rot_deg; its bounding
        box peaks at patch*(|cos|+|sin|)/scale. Pass the sampled rot/scale
        to size the margin for that draw; without them, the config worst
        case is used (sqrt(2) at 45 deg if rotation is enabled, else 1,
        over the scale_range lower bound).
        """
        if not self.augs.affine:
            return 0
        if rot_deg is None or scale is None:
            f = np.sqrt(2.0) if self.augs.affine.rotation else 1.0
            scale = min(self.augs.affine.scale_range)
        else:
            r = np.deg2rad(rot_deg)
            f = abs(np.cos(r)) + abs(np.sin(r))
        return max(0, int(np.ceil(self.patch * (f / scale - 1) / 2)))

    def _photometric_augmentations(self, image: np.ndarray) -> np.ndarray:
        """Per-patch photometric ops; image only, never targets."""
        if not self.train or not self.photometric_pipeline:
            return image
        out, _ = self.photometric_pipeline(image, self.rng)
        return out

    def __getitem__(self, idx: int) -> dict:
        image_file, label_file, cache_id = self.image_label_pairs[idx]

        with self.timer.stage("img load"):
            image = prepare_image(iio.imread(image_file), in_ch=self.in_ch)
        with self.timer.stage("label load"):
            label = load_masks(label_file).astype(np.uint32)
        maps = self._load_maps(cache_id, label)

        if self.train:
            if self.augs.affine:
                # sample the affine draw first so the crop can be sized for it;
                # crop first, then transform image + encoded maps within the patch
                ta_rot = self.rng.uniform(
                    0, 360) if self.augs.affine.rotation else 0
                ta_scale = self.rng.uniform(*self.augs.affine.scale_range)
                margin = self._affine_margin(ta_rot, ta_scale)
            else:
                # if no affine, no margin, ta = placeholder
                margin = 0
                ta_rot, ta_scale = 0.0, 1.0
            with self.timer.stage("crop"):
                image, label, *map_vals = self._random_crop(
                    image, label, *maps.values(),
                    margin=margin
                )
            maps = dict(zip(maps.keys(), map_vals))

            if any([self.augs.flip_x, self.augs.flip_y, self.augs.rot90]):
                t = OrthoTransform(
                    flip_x=self.augs.flip_x and bool(self.rng.integers(0, 2)),
                    flip_y=self.augs.flip_y and bool(self.rng.integers(0, 2)),
                    rot90=int(
                        self.rng.integers(0, 4)
                    ) if self.augs.rot90 else 0,
                )
                if not t.is_identity():
                    with self.timer.stage("d4 aug"):
                        image = t.apply_to_image(image)
                        label = t.apply_to_image(label)
                        maps = self.rep.transform(maps, t)

            if self.augs.affine:
                h, w = image.shape[:2]
                # do affine on top (avoid doing flip in affine)
                ta = AffineTransformV2.from_params(
                    rot_deg=ta_rot,
                    scale=ta_scale,
                    center=(h//2, w//2)
                )
                if not ta.is_identity():
                    with self.timer.stage("affine aug"):
                        image, label = ta.apply_to_images(
                            [image, label], bilinear=[True, False])
                        maps = self.rep.transform_affine(maps, ta, label)
                if margin:
                    # drop the margin ring: center patch survived the warp on
                    # real data (margin sized by _affine_margin)
                    m2 = (h - self.patch) // 2
                    sl = slice(m2, m2 + self.patch)
                    image = image[sl, sl]
                    label = label[sl, sl]
                    maps = {k: v[sl, sl] for k, v in maps.items()}

            with self.timer.stage("photo aug"):
                image = self._photometric_augmentations(image)
        elif self.mode == "val":
            with self.timer.stage("crop"):
                image, label, *map_vals = self._deterministic_crop(
                    image, label, *maps.values())
            maps = dict(zip(maps.keys(), map_vals))
        else:
            # no pad here: the model pads around the backbone and decodes on
            # the original extent (predict_labels)
            pass

        # all channels go to targets (the loss may need non-predicted ones,
        # e.g. mask's distance weights); predict only drives the head.
        with self.timer.stage("to tensor"):
            targets = {}
            for k, c in self.rep.channels.items():
                targets[k] = to_chw(maps[k], c.torch_type)
            image_t = to_chw(image, torch.float32)
        self.timer.tick(tag=f"{self.dataset_name}-{self.mode}")
        return {
            "image": image_t,
            "targets": targets,
            "label": torch.from_numpy(label).long(),
            "image_source": str(image_file),
            "label_source": str(label_file)
        }


class SegmentationData(L.LightningDataModule):
    """Lightning datamodule wiring the shared rep_cfg to the datasets.

    Train: shuffled DataLoader, batch_size, drop_last. Val: plain loader over
    deterministic patches (batch_size = val_batch_size). Test: one full
    /32-padded image per batch.
    """

    def __init__(
        self,
        rep_cfg: RepresentationConfig,
        dataset_cfg: DatasetConfig,
        augs: AugmentationConfig,
        patch: int = 256,
        batch_size: int = 4,
        val_batch_size: int = 1,
        num_workers: int = 4,
        timing_train_log_dir: str = ""
    ):
        super().__init__()
        self.rep_cfg = rep_cfg
        self.dataset_cfg = dataset_cfg
        self.augs = augs
        self.patch = patch
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.rep: Representation | None = None
        self.train_ds: data.Dataset | None = None
        self.val_ds: data.Dataset | None = None
        self.test_ds: data.Dataset | None = None
        self.timing_train_log_dir = timing_train_log_dir

    def setup(self, stage: str | None = None):
        self.rep = build_representation(self.rep_cfg)
        dataset_name, split = load_split(self.dataset_cfg)
        if stage in ("fit", None):
            self.train_ds = PatchDataset(
                image_label_pairs=split["train"],
                cache_dir=self._cache_root(),
                dataset_name=dataset_name,
                rep=self.rep,
                cfg=self.dataset_cfg,
                augs=self.augs,
                patch=self.patch,
                train=True,
                mode="train",
                timing_log_dir=self.timing_train_log_dir
            )
            self.val_ds = PatchDataset(
                image_label_pairs=split["val"],
                cache_dir=self._cache_root(),
                dataset_name=dataset_name,
                rep=self.rep,
                cfg=self.dataset_cfg,
                augs=self.augs,
                patch=self.patch,
                train=False,
                mode="val",
            )
            if self.dataset_cfg.warm_cache:
                self.train_ds.warm_cache(n_workers=self.num_workers)
                self.val_ds.warm_cache(n_workers=self.num_workers)
        if stage in ("test", "predict", None):
            self.test_ds = PatchDataset(
                image_label_pairs=split["test"],
                cache_dir=self._cache_root(),
                dataset_name=dataset_name,
                rep=self.rep,
                cfg=self.dataset_cfg,
                augs=self.augs,
                patch=self.patch,
                train=False,
                mode="test",
            )
            if self.dataset_cfg.warm_cache:
                self.test_ds.warm_cache(n_workers=self.num_workers)

    def load_full_image(self, idx: int) -> tuple[torch.Tensor, np.ndarray]:
        """Whole-image loading for the gate's image-level oracle refresh:
        the same preprocessing as PatchDataset.__getitem__ (prepare_image)
        but without any crop, plus the full-resolution label."""
        image_file, label_file, _ = self.val_ds.image_label_pairs[idx]
        image = numpy_image_to_model_input(
            iio.imread(image_file), in_ch=self.val_ds.in_ch)
        label = load_masks(label_file).astype(np.uint32)
        return image, label

    def _cache_root(self) -> Path:
        return Path(self.dataset_cfg.cache_dir)

    @staticmethod
    def _worker_init_fn(worker_id: int) -> None:
        """Seed this worker's dataset RNG distinctly (no-op if unseeded);
        keep torch single-threaded in the worker (the warp/aug ops use torch
        CPU ops — 12 workers x intra-op threads would oversubscribe CPUs)."""
        info = data.get_worker_info()
        ds = info.dataset
        if hasattr(ds, "set_worker_seed"):
            ds.set_worker_seed(worker_id)
        torch.set_num_threads(1)
        # numba's prange pool is independent of torch's; 12 workers x 22 numba
        # threads would oversubscribe (d4/aug stage variance seen in chimera
        # smoke runs) — cap per-worker threads so worker_count * n <= ~cores
        try:
            import numba
            numba.set_num_threads(2)
        except (ImportError, ValueError):
            pass
        if hasattr(ds, "timer"):
            ds.timer.install_exit_hook()

    def train_dataloader(self) -> data.DataLoader:
        if len(self.train_ds) < self.batch_size:
            warnings.warn(
                f"batch_size={self.batch_size} exceeds the {len(self.train_ds)} "
                f"training images and drop_last=True: the train loader would "
                f"yield zero batches. Lowering batch_size to "
                f"{len(self.train_ds)} for this run."
            )
            self.batch_size = len(self.train_ds)
        sampler = None
        shuffle = True
        if self.dataset_cfg.balanced_sampling:
            groups = [pair[2].split("-", 1)[0] for pair in
                      self.train_ds.image_label_pairs]
            counts = collections.Counter(groups)
            if len(counts) < 2:
                rprint(
                    "[yellow bold]warning:[/] balanced_sampling with a single "
                    f"source ({list(counts)[0]!r}); falling back to shuffle")
            else:
                weights = [1.0 / counts[g] for g in groups]
                sampler = data.WeightedRandomSampler(
                    weights, num_samples=len(weights), replacement=True)
                shuffle = False
        return data.DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers, pin_memory=True, drop_last=True,
            worker_init_fn=self._worker_init_fn,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> data.DataLoader:
        if len(self.val_ds) < self.val_batch_size:
            warnings.warn(
                f"val_batch_size={self.val_batch_size} exceeds the "
                f"{len(self.val_ds)} validation images. Lowering "
                f"val_batch_size to {len(self.val_ds)} for this run."
            )
            self.val_batch_size = len(self.val_ds)
        return data.DataLoader(
            self.val_ds, batch_size=self.val_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.test_ds, batch_size=1, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )
