from pathlib import Path

import numpy as np
import pandas as pd
from rich.progress import track
from skimage.filters import gaussian  # pylint: disable=E0611
from skimage.measure import label as cc_label  # pylint: disable=E0611
from skimage.measure import perimeter as _perimeter  # pylint: disable=E0611
from skimage.measure import regionprops  # pylint: disable=E0611
from skimage.morphology import closing, disk  # pylint: disable=E0611
from scipy.ndimage import distance_transform_edt

from pluriseg.paths import CACHE_SHAPES
from pluriseg.shapes.cleanup import (DEFAULT_OPENING_RADIUS,
                                   filter_stray_components, get_clean_label)

_EPS = 1e-12

#: Component stats (num_components, min_comp_rel_size) are computed after a
#: binary closing with this disk radius: gaps up to ~2x this many pixels
#: fuse, so multi_component only fires for genuinely separated objects
#: (hairline splits from the opening or mask noise don't count).
COMPONENT_CLOSE_RADIUS = 2
# All numeric descriptors written to the per-image CSVs. Rules configs may
# only reference columns from this tuple (validated by RuleShapeCategorizer).


def _roughness_column(sigma: float) -> str:
    """Column name for the roughness feature at smoothing sigma `sigma`
    (`roughness_s150` for 1.5, `roughness_s100` for 1.0); sigma is encoded
    in cents so cached CSVs and rules configs stay unambiguous about which
    smoothing produced it."""
    return f"roughness_s{round(sigma * 100):03d}"


def _smoothed_perimeter(mask: np.ndarray, sigma: float) -> float:
    """Perimeter of `mask` after gaussian smoothing at `sigma` + threshold."""


    g = gaussian(mask.astype(float), sigma) > 0.5
    return float(_perimeter(g))


#: Default smoothing sigmas for the roughness feature (one column each).
ROUGHNESS_SIGMAS: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)


def extract_shape_features(
    label: np.ndarray,
    opening_radius: int = DEFAULT_OPENING_RADIUS,
    roughness_sigmas: tuple[float, ...] | None = None,
) -> list[dict]:
    """Shape descriptors for every instance of a (H, W) label map.

    Args:
        label: (H, W) integer instance label map, 0 = background. Ids are
            kept as-is (no relabeling) and returned in 'label_id'.
        opening_radius: if > 0, features are computed on
            filter_stray_components(label, opening_radius) — the original
            mask minus stray components. Ids are preserved; an id whose
            mask vanishes entirely is omitted from the output.
        roughness_sigmas: sigmas of the gaussian smoothing behind the
            roughness columns (one column per sigma, named by
            _roughness_column); None = module default (ROUGHNESS_SIGMAS).

    Returns:
        list of dicts with keys 'label_id' + FEATURE_COLUMNS. Tiny regions
        (area < 5) get NaN for degenerate ratios.
    """
    label = np.asarray(label)
    if opening_radius > 0:
        label = filter_stray_components(label, opening_radius)
    rough_cols = [
        _roughness_column(s) for s in (roughness_sigmas or ROUGHNESS_SIGMAS)
        if s > 0
    ]
    # per-label smoothed mask for the roughness feature: smoothing the full
    # frame once would bleed across touching instances, so blur each label's
    # bbox crop in isolation (mask = label == id inside the crop)
    rough_masks = (
        {p.label: p.image.astype(float) for p in regionprops(label)}
        if rough_cols else {}
    )
    props = regionprops(label)
    out = []
    for p in props:
        # one dict per instance = one row of the cached CSV
        area = float(p.area)
        perimeter = float(p.perimeter) if p.perimeter > 0 else np.nan
        minor = float(p.axis_minor_length)
        major = float(p.axis_major_length)
        conv_area = float(p.area_convex)
        eigvals = p.inertia_tensor_eigvals
        # component sizes of this instance's mask, from its bbox crop
        # (p.image) after a binary closing: gaps up to ~2x
        # COMPONENT_CLOSE_RADIUS fuse, so hairline splits from the opening
        # or mask noise don't count as multi-component. The largest
        # component is the instance's body, the rest are satellites
        # (p.image components are contiguous within the crop but may
        # include touch-free neighbors — acceptable, the body still
        # dominates). Sizes are used by the min_comp_rel_size column below.
        if p.area:
            comp_labels = cc_label(closing(
                p.image, disk(COMPONENT_CLOSE_RADIUS)))
            comp_sizes = sorted(
                (cc.area for cc in regionprops(comp_labels)),
                reverse=True)
        else:
            comp_labels, comp_sizes = None, []
        row = {
            "label_id": int(p.label),
            "area": area,
            "perimeter": perimeter,
            "equivalent_diameter": float(p.equivalent_diameter_area),
            "feret_diameter": float(p.feret_diameter_max),
            "eccentricity": float(p.eccentricity),
            "solidity": float(p.solidity),
            "extent": float(p.extent),
            # 4*pi*A/P^2: 1.0 for a perfect circle.
            "circularity": 4.0 * np.pi * area / max(perimeter, _EPS) ** 2,
            # major/minor axis: 1.0 for a circle, larger = more elongated.
            "elongation": major / max(minor, _EPS),
            # convexity ratio duplicates solidity's idea without its
            # boundary complexity; keep both, they diverge on concave shapes.
            "convexity": area / max(conv_area, _EPS),
            "aspect_ratio": (p.bbox[2] - p.bbox[0]) / max(
                p.bbox[3] - p.bbox[1], 1.0
            ),
            "inertia_eig_ratio": (eigvals[0] / max(abs(eigvals[1]), _EPS)),
            # max of the distance transform (inradius): pi*r_max^2/area =
            # area fraction covered by the instance's largest inscribed
            # disk. 1.0 for a disk, ->0 for thin/elongated shapes (worms).
            # The crop is padded 1px so the EDT sees background at the
            # border; unpadded, masks filling their bbox get inflated
            # distances (>1.0 fracs measured at up to 13).
            "inscribed_disk_frac": (
                np.pi * float(distance_transform_edt(np.pad(p.image, 1)).max()) ** 2
                / area if p.area else np.nan
            ),
            # connected-component count of this instance's closed mask
            # (see COMPONENT_CLOSE_RADIUS), 0 for empty masks.
            "num_components": (
                int(comp_labels.max()) if comp_labels is not None else 0),
            # size of the largest satellite component relative to the whole
            # instance (1.0 when single-component, i.e. no satellites).
            # Rules use this instead of num_components alone so a mask with
            # a tiny speck off its side is not multi_component: e.g.
            # min_comp_rel_size >= 0.05 means every satellite covers >= 5%
            # of the instance area.
            "min_comp_rel_size": (
                comp_sizes[1] / area if len(comp_sizes) > 1 else 1.0
            ),
            # enclosed background components (holes): fill the mask and
            # count the resulting filled regions. p.image_filled works on
            # the same bbox crop as p.image, where the tight bbox means
            # holes can't touch the crop edge.
            "num_holes": int(cc_label(
                p.image_filled != p.image
            ).max()) if p.area_filled > p.area else 0,
            # perimeter ratio between raw and smoothed boundary: ~1 = smooth,
            # smaller = rugged (smoothing shortens a wiggly boundary).
            # bbox-crop smoothing keeps touching neighbors from bleeding.
            **{
                col: (
                    perimeter
                    / max(
                        _smoothed_perimeter(rough_masks[p.label], sigma),
                        _EPS,
                    )
                    if np.isfinite(perimeter) and perimeter > 0 else np.nan
                )
                for col, sigma in zip(rough_cols, ROUGHNESS_SIGMAS)
            },
            "log_area": np.log1p(area),
            "log_perimeter": (
                np.log1p(perimeter) if np.isfinite(perimeter) else np.nan
            ),
            "log_equivalent_diameter": np.log1p(
                float(p.equivalent_diameter_area)
            ),
            "log_feret_diameter": np.log1p(float(p.feret_diameter_max)),
        }
        out.append(row)
    return out


FEATURE_COLUMNS: tuple[str, ...] = (
    "area",
    "perimeter",
    "equivalent_diameter",
    "feret_diameter",
    "eccentricity",
    "solidity",
    "extent",
    "circularity",
    "elongation",
    "convexity",
    "aspect_ratio",
    "inertia_eig_ratio",
    "inscribed_disk_frac",
    "num_components",
    "min_comp_rel_size",
    "num_holes",
    "log_area",
    "log_perimeter",
    "log_equivalent_diameter",
    "log_feret_diameter",
) + tuple(_roughness_column(s) for s in ROUGHNESS_SIGMAS)


def _compute_one_image(
    args: tuple[str, str, str, str, int],
) -> pd.DataFrame:
    """Worker for dataset_shape_features: (cache_id, path, dataset, cache, r)."""
    cache_id, label_path, dataset_name, cache_dir, opening_radius = args
    df = get_shape_features(
        dataset_name, cache_id, label_path, cache_dir, opening_radius
    )
    df.insert(0, "cache_id", cache_id)
    return df


def dataset_shape_features(
    pairs: list[tuple[str | Path, str]],
    dataset_name: str,
    cache_dir: str | Path = CACHE_SHAPES,
    opening_radius: int = DEFAULT_OPENING_RADIUS,
    num_workers: int = 8,
    progress: bool = True,
) -> pd.DataFrame:
    """Features for a list of images, each via its own per-image CSV.

    Args:
        pairs: (label_path, cache_id) per image; cache_id is the split's
            per-image key and names the cache CSV.
        dataset_name: name of the dataset, part of the cache key.

    Adds a 'cache_id' column identifying which image each row came from.
    Images are processed in a process pool: most of the per-image cost
    (skimage morphology, regionprops) holds the GIL, so processes scale
    where threads do not (measured ~3x at 4 workers).
    """
    jobs = [
        (cache_id, str(p), dataset_name, str(cache_dir), opening_radius)
        for p, cache_id in pairs
    ]
    parts = []
    if num_workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        # process pool: skimage work holds the GIL, threads would not scale
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            it = pool.map(_compute_one_image, jobs)
            if progress:
                it = track(it, "getting shapes features", total=len(jobs))
            parts = list(it)
    else:
        if progress:
            jobs = track(jobs, "getting shapes features")
        parts = [_compute_one_image(job) for job in jobs]
    # single table; empty input still yields a correctly-shaped frame
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["cache_id", "label_id", *FEATURE_COLUMNS]
    )


def _cache_path(
    cache_dir: str | Path, dataset_name: str, cache_id: str
) -> Path:
    return Path(cache_dir) / dataset_name / f"{cache_id}.csv"


def get_shape_features(
    dataset_name: str,
    cache_id: str,
    label_path: str | Path,
    cache_dir: str | Path = CACHE_SHAPES,
    opening_radius: int = DEFAULT_OPENING_RADIUS,
) -> pd.DataFrame:
    """Per-instance shape features for one image, cached as a whole.

    Computes and writes the full image's CSV if missing; otherwise reads
    it. One call = one image; there is no partial-per-label computation.
    `opening_radius` is part of the cache key: a nonzero value writes
    `..._r<rad>open.csv` next to the plain cache; the component closing
    (COMPONENT_CLOSE_RADIUS) adds a `_c<rad>close` part, so pre-closing
    caches never read as hits.
    """
    suffix = (f"_r{opening_radius}open"
              f"_c{COMPONENT_CLOSE_RADIUS}close"
              if opening_radius > 0 else
              f"_r0open_c{COMPONENT_CLOSE_RADIUS}close")
    path = _cache_path(cache_dir, dataset_name, cache_id)
    path = path.with_name(path.stem + suffix + path.suffix)
    if path.exists():
        try:
            return pd.read_csv(path)  # cache hit
        except pd.errors.EmptyDataError:
            # 0-byte file left behind by an interrupted run: recompute
            path.unlink()
    clean = get_clean_label(
        dataset_name, cache_id, label_path, cache_dir, opening_radius
    )
    df = pd.DataFrame(extract_shape_features(clean))
    # atomic write: a crashed run can't leave a partial CSV that later reads
    # as an empty cache hit
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)
    return df