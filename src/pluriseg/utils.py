import subprocess
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import tifffile
import torch
import yaml
from rich import print as rprint
from pluriseg.configs import DatasetConfig, RepresentationConfig

from pluriseg.display import get_glasbey_colors, get_random_colors

_MISC_DIR = Path(__file__).parent / "misc"

#: paths already warned about in load_masks (module-level so per-run once)
_MASKS_WARNED: set[str] = set()


def generate_name(
    separator: str = "-",
    rng: np.random.Generator | None = None,
) -> str:
    """Generate a random experiment name of the form adjective-noun-function.

    Reproduces scripts/name_generator.sh, using the word lists in
    src/pluriseg/misc.

    Args:
        separator: string joining the three words.
        rng: optional numpy Generator; defaults to fresh randomness.

    Returns:
        Name string, e.g. "sunny-forest-reconstruct".
    """
    rng = rng if rng is not None else np.random.default_rng()

    def pick(wordlist: str) -> str:
        lines = (_MISC_DIR / wordlist).read_text().splitlines()
        return lines[rng.integers(len(lines))]

    return separator.join(
        pick(f)
        for f in ("english-adjectives.txt", "english-nouns.txt", "function-names.txt")
    )


def load_masks(path: str | Path) -> np.ndarray:
    """Load a (H, W) instance label map from file with basic sanity checks.

    Drops non-instance sentinel values (background floods written as the
    dtype's max, e.g. 65535 in uint16 cellpose masks) with a warning.

    Args:
        path: path to a label map readable by imageio (png/tif).

    Returns:
        (H, W) int64 array, background 0, sentinel ids removed.
    """
    import imageio.v3 as iio

    label = iio.imread(path).astype(np.int64)
    # sentinel = dtype max: background floods written by some tools (e.g.
    # uint16 cellpose masks with value 65535). Only 16/32-bit: 255 is a
    # legitimate instance id (uint8 masks often hold ~255 real instances).
    # A real id at the dtype max would need that many distinct labels, so
    # the sentinel is identified by: max value hit + label count far below.
    max_val = int(label.max())
    if max_val in (2 ** 16 - 1, 2 ** 32 - 1):
        n_labels = len(np.unique(label[label > 0]))
        if n_labels < max_val / 2:
            label[bad := label == max_val] = 0
            if str(path) not in _MASKS_WARNED:
                _MASKS_WARNED.add(str(path))
                rprint(
                    f"[yellow]load_masks {Path(path).name}: suspicious "
                    f"id {max_val} (dtype max, "
                    f"{bad.sum()} px = {bad.mean():.1%}, {n_labels} real "
                    f"labels) set to 0[/yellow]"
                )
    return label


def relabel_contiguous(label: np.ndarray) -> np.ndarray:
    """Map instance ids to contiguous labels; background stays 0. negative labels are mapped to 0

    Args:
        label: (H, W) integer label map (0 = background).

    Returns:
        (H, W) int64 array with instances renumbered 1..n in ascending
        original-id order and everything else set to 0.
    """
    ids = np.unique(label)
    ids = ids[ids > 0]
    out = np.zeros(label.shape, dtype=np.int64)
    if len(ids) == 0:
        return out
    lookup = np.zeros(int(ids.max()) + 1, dtype=np.int64)
    lookup[ids] = np.arange(1, len(ids) + 1)
    out.ravel()[:] = lookup[np.maximum(label.ravel(), 0)]
    return out


def remap(
    arr: np.ndarray,
    out_range: tuple[float, float] = (0.0, 1.0),
    in_range: tuple[float, float] | None = None,
    clip: bool = False,
) -> np.ndarray:
    """Linearly map arr from in_range to out_range.

    Args:
        arr: input array.
        out_range: output (min, max).
        in_range: input (min, max); defaults to (arr.min(), arr.max()).
        clip: if True, clip input values to in_range before mapping,
            so the output stays within out_range.

    Returns:
        float64 array with values linearly rescaled.
    """
    arr = arr.astype(np.float64)
    out_min, out_max = out_range
    if in_range is None:
        in_min, in_max = arr.min(), arr.max()
    else:
        in_min, in_max = in_range
    if clip:
        arr = np.clip(arr, in_min, in_max)
    scale = (out_max - out_min) / (in_max - in_min)
    return (arr - in_min) * scale + out_min


def save_masks(
        masks: np.ndarray,
        file: str | Path,
        tif_format: Literal['imagej', 'ome'] = 'ome',
        relabel: bool = True,
        n_bits: Literal[8, 16] = 8,
        fallback_to_16_bits: bool = True
):
    """Save a (H, W) label map as an ImageJ or OME TIFF.

    Args:
        masks: (H, W) integer label map (0 = background).
        file: output path.
        tif_format: 'imagej' saves as uint8, 'ome' as uint16; an OverflowError
            is raised if labels exceed the dtype range.
        relabel: if True, apply relabel_contiguous first so labels are 1..n
            and the dtype range is used efficiently.
        n_bits: pixel depth, 8 or 16.
        fallback_to_16_bits: if True (default), 8-bit images whose labels
            exceed uint8 are saved as 16-bit grayscale with a warning
            (16-bit colormap is not reliably rendered by viewers).
    """
    if masks.ndim != 2:
        raise ValueError(f"masks must be (H, W), got shape {masks.shape}")
    if not np.issubdtype(masks.dtype, np.integer):
        raise TypeError(f"masks must be integer dtype, got {masks.dtype}")
    if masks.min() < 0:
        raise ValueError(f"masks contains negative labels (min={masks.min()})")
    if relabel:
        masks = relabel_contiguous(masks)

    if n_bits == 8:
        mask_data_type = np.uint8
        use_colormap = True
    elif n_bits == 16:
        mask_data_type = np.uint16
        use_colormap = False  # 16-bit colormap is not reliably rendered by viewers
    else:
        raise ValueError(n_bits)

    axes = 'YX'

    write_params = dict(metadata={
        'axes': axes,
    })

    if tif_format == 'imagej':
        write_params.update(
            imagej=True
        )
    elif tif_format == 'ome':
        write_params.update(
            ome=True
        )
    else:
        raise ValueError(tif_format)

    max_label = int(masks.max())
    dtype_max = np.iinfo(mask_data_type).max
    if max_label > dtype_max:
        if fallback_to_16_bits and n_bits == 8:
            warnings.warn(
                f"labels max value {max_label} exceeds uint8 ({dtype_max}), "
                "falling back to 16 bits", stacklevel=2)
            mask_data_type = np.uint16
            use_colormap = False
        else:
            raise OverflowError(
                f"labels max value {max_label} exeed max {mask_data_type} ({dtype_max})")

    masks = masks.astype(mask_data_type)

    if use_colormap:
        lut_size = 2 ** (8 if tif_format == 'imagej' else n_bits)

        if lut_size <= 256:
            palette = get_glasbey_colors(n=lut_size).T
        else:
            palette = get_random_colors(n=lut_size).T
        label_colormap = (palette * 65535).astype(np.uint16)
        label_colormap[:, 0] = 0
        write_params.update(
            colormap=label_colormap,
            metadata={
                'axes': axes,
                'photometric': 'palette'
            }
        )

    tifffile.imwrite(
        file,
        data=masks,
        compression='zlib',
        **write_params
    )


def tensor_to_numpy(a: torch.Tensor) -> np.ndarray:
    assert a.ndim == 3
    return a.permute(1, 2, 0).detach().cpu().numpy()


def dict_of_tensors_to_numpy(d: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {
        k: tensor_to_numpy(v)
        for k, v in d.items()
    }


def git_info() -> str:
    """Short git commit hash + dirty flag, e.g. "9e6887b (dirty)". Falls back
    to "unknown" outside a repo or without git (cluster nodes)."""
    try:
        commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                                capture_output=True, text=True, timeout=5)
        if commit.returncode != 0:
            return 'unknown'
        dirty = subprocess.run(['git', 'diff', '--quiet', 'HEAD'],
                               capture_output=True, timeout=5)
        return commit.stdout.strip() + (' (dirty)' if dirty.returncode else '')
    except Exception:
        return 'unknown'


def _last_epoch(run_dir: Path) -> int | None:
    """Highest logged epoch in the run's metrics.csv; None if missing/empty."""
    csv = run_dir / "metrics.csv"
    if not csv.exists():
        return None
    try:
        epochs = pd.read_csv(csv, usecols=["epoch"])["epoch"].dropna()
    except (ValueError, pd.errors.EmptyDataError):
        return None
    return int(epochs.max()) if len(epochs) else None


def _parse_config(run_dir: Path) -> dict:
    """rep, dataset, fold, seed, epochs and last_epoch from the run's
    config.yaml.

    Returns:
        dict with keys rep, dataset, fold, seed, epochs, last_epoch;
        unparsable configs get fold=-1, seed=-1 (epochs=None). Missing
        config.yaml returns None. last_epoch is the highest epoch logged in
        run_dir/metrics.csv (None when absent).
    """
    config_file = run_dir / "config.yaml" if run_dir is not None else None
    if config_file is None or not config_file.exists():
        return None
    try:
        with open(config_file, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        rep = RepresentationConfig(**config["representation"]).name
        ds = DatasetConfig(**config["dataset"])
        opt = config.get("optimizer") or {}
        return {"rep": rep, "dataset": ds.split_name, "fold": int(ds.fold),
                "seed": int((config.get("train") or {}).get(
                    "seed_everything", 0)),
                "epochs": int(opt.get("epochs", 200)),
                "last_epoch": _last_epoch(run_dir)}
    except Exception:
        return {"rep": "", "dataset": "", "fold": -1, "seed": -1,
                "epochs": None, "last_epoch": _last_epoch(run_dir)}


REPEAT_MODES = ("seeds", "folds")


def _expected_repeats(repeats: str, n_repeats: int | None) -> list[int]:
    """Repeat ids the tables expect: seeds mode s0..sN (N_SEEDS default),
    folds mode f0..f4; n_repeats overrides the expected count."""
    if repeats == "seeds":
        return list(range(n_repeats))
    return list(range(n_repeats))


def _repeat_col(repeats: str) -> str:
    """Fold-level column holding the active repeat axis ('seed' or 'fold')."""
    if repeats not in REPEAT_MODES:
        raise ValueError(f"repeats must be one of {REPEAT_MODES}, "
                         f"got {repeats!r}")
    return "seed" if repeats == "seeds" else "fold"


def _run_pq(run_dir: Path) -> float | None:
    """Mean test PQ of the run's eval/metrics.csv ('mean' row); None when
    the CSV is missing or unparsable. Shared ranking criterion for the
    best-run picks (shape_metrics.plot one_seed)."""
    csv = run_dir / "eval" / "metrics.csv"
    if not csv.exists():
        return None
    try:
        df = pd.read_csv(csv)
        row = df[df["image_source"] == "mean"]
        return float(row["PQ"].iloc[0]) if len(row) else None
    except Exception:
        return None
