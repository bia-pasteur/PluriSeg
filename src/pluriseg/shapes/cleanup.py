from pathlib import Path

import imageio.v3 as iio
import numpy as np
from skimage.measure import label as cc_label  # pylint: disable=E0611
from skimage.measure import regionprops # pylint: disable=E0611
from skimage.morphology import disk, opening

from pluriseg.paths import CACHE_SHAPES
from pluriseg.utils import load_masks, save_masks


def _clean_label_path(
    cache_dir: str | Path, dataset_name: str, cache_id: str
) -> Path:
    return Path(cache_dir) / dataset_name / f"{cache_id}.tif"


# Caching layout: pluriseg-cache/shapes/<dataset>/<cache_id>_r<rad>open.csv
# (features) and <cache_id>_r<rad>clean.tif (cleaned labels). Cache ids are
# the split's per-image keys (source-prefixed); label stems alone can
# collide across sources (e.g. cellpose-train-015 vs cellpose-test-015).

#: Default disk radius for stray-component removal (binary opening).
DEFAULT_OPENING_RADIUS = 2


def filter_stray_components(
    label: np.ndarray,
    opening_radius: int = DEFAULT_OPENING_RADIUS,
) -> np.ndarray:
    """Remove stray connected components from each instance's mask.

    Per label id: binary-open the mask (disk of `opening_radius`), then
    keep only the connected components of the *original* mask that contain
    at least one surviving opened pixel. Components whose opened
    counterpart is empty (stray pixels, thin slivers) are dropped.

    Args:
        label: (H, W) integer instance label map, 0 = background.
        opening_radius: disk radius for the opening (0 keeps all).

    Returns:
        (H, W) same ids as input, with stray components removed (set to 0).
    """
    label = np.asarray(label)
    # nothing to do without an opening: keep ids, return a copy
    if opening_radius <= 0:
        return label.copy()
    out = np.zeros_like(label)  # stray ids are simply never written here
    kernel = disk(opening_radius)
    for lbl in np.unique(label[label > 0]):
        mask = label == lbl
        ys, xs = np.nonzero(mask)
        # crop to the bbox before opening: morphology cost scales with the
        # full frame, not the object, so this is the big win
        r0, c0 = max(0, ys.min() - 1), max(0, xs.min() - 1)
        crop = mask[r0:ys.max() + 2, c0:xs.max() + 2]
        opened = opening(crop, kernel) > 0
        if not opened.any():
            continue  # whole instance is stray
        # keep components of the original mask that touch the opened mask
        for cc in regionprops(cc_label(crop)):
            r1, c1, r2, c2 = cc.bbox
            if opened[r1:r2, c1:c2][cc.image].any():
                keep = out[r0 + r1:r0 + r2, c0 + c1:c0 + c2]
                keep[cc.image] = lbl
    return out


def get_clean_label(
    dataset_name: str,
    cache_id: str,
    label_path: str | Path,
    cache_dir: str | Path = CACHE_SHAPES,
    opening_radius: int = DEFAULT_OPENING_RADIUS,
) -> np.ndarray:
    """Label map with stray components removed, cached as a TIFF.

    Same role as the feature CSV cache but for the cleaned label itself,
    so plot_category_examples (and any other consumer) can reuse it
    without recomputing the per-instance morphology. Ids are unchanged
    from the source label file (no relabeling). The radius is part of the
    cache key: `<cache_id>_r<rad>clean.tif` (r0 = no filtering).
    """
    path = _clean_label_path(cache_dir, dataset_name, cache_id)
    # radius in the filename: different radius -> different cache file
    path = path.with_name(path.stem + f"_r{opening_radius}clean" + path.suffix)
    if path.exists():
        return iio.imread(path).astype(np.int64)  # cache hit
    label = load_masks(label_path)
    clean = filter_stray_components(label, opening_radius)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 16-bit up front: raw ids are not contiguous, so they routinely exceed
    # uint8, and save_masks' 8-bit fallback warning would spam per image
    save_masks(clean, path, n_bits=16, relabel=False)
    return clean


