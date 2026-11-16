from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
from rich.console import Console

rprint = Console().print


def _crop_example(mask: np.ndarray) -> np.ndarray:
    """Square, padded bbox crop of a single-instance mask."""
    ys, xs = np.nonzero(mask)
    # crop to the mask extent, pad by 1/8 of the longer side (so the
    # mask never touches the crop edge), then pad to square
    h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    crop = np.pad(crop, max(2, max(h, w) // 8))
    side = max(crop.shape)
    return np.pad(
        crop,
        (
            (max(0, (side - crop.shape[0]) // 2),
             max(0, (side - crop.shape[0]) // 2)),
            (max(0, (side - crop.shape[1]) // 2),
             max(0, (side - crop.shape[1]) // 2)),
        ),
    )


def pick_category_examples(
    feats: pd.DataFrame,
    categories: pd.Series,
    n_per_cat: int = 4,
    seed: int = 0,
) -> dict[str, list[tuple[str, int]]]:
    """Sample example instances per category, no duplicate instances.

    Args:
        feats: feature table; needs 'cache_id' (a label_maps key) and
            'label_id' columns.
        categories: category per feature row, aligned with feats.
        n_per_cat: max examples per category (fewer if the category
            has fewer instances).
        seed: RNG seed for reproducible sampling.

    Returns:
        {category: [(cache_id, label_id), ...]} in sampling order.
    """
    df = feats.assign(_cat=categories)
    rng = np.random.default_rng(seed)
    out: dict[str, list[tuple[str, int]]] = {}
    for name, rows in df.dropna(subset=["_cat"]).groupby("_cat"):
        # one random draw per category; rows are distinct, so sampling
        # without replacement from the row indices guarantees that a
        # (cache_id, label_id) pair never repeats within a category
        idx = rng.choice(
            rows.index.to_numpy(),
            size=min(n_per_cat, len(rows)),
            replace=False,
        )
        out[name] = [
            (str(df.at[i, "cache_id"]), int(df.at[i, "label_id"]))
            for i in idx
        ]
    return out


def save_category_examples(
    label_maps: dict[str, np.ndarray],
    examples: dict[str, list[tuple[str, int]]],
    save_dir: str | Path,
) -> None:
    """Save individual example masks as PNGs, one subfolder per category.

    Writes `<category>/<cache_id>_<label_id>.png` (square, padded crops of
    the cleaned label maps) as black instances on a transparent
    background (RGBA) so the folder doubles as a browsable dataset.
    """
    for cat, pairs in examples.items():
        cat_dir = Path(save_dir) / str(cat)
        cat_dir.mkdir(parents=True, exist_ok=True)
        for key, label_id in pairs:
            mask = label_maps[key] == label_id
            if not mask.any():
                rprint(
                    f"[yellow]save_category_examples: label {label_id} not "
                    f"found in '{key}' map — stale cache? Skipping.[/yellow]"
                )
                continue
            crop = _crop_example(mask)
            rgba = np.zeros((*crop.shape, 4), np.uint8)
            rgba[crop, 3] = 255  # RGB stays black; opaque where the mask is
            iio.imwrite(cat_dir / f"{key}_{label_id}.png", rgba)


def plot_category_examples(
    label_maps: dict[str, np.ndarray],
    examples: dict[str, list[tuple[str, int]]],
) -> object:
    """Grid of instance masks, one row per category.

    Args:
        label_maps: {image key: (H, W) label map} the examples belong to.
            Pass the cleaned maps (get_clean_label) so what you see
            matches the features.
        examples: {category: [(cache_id, label_id), ...]} from
            pick_category_examples; rows follow list order, row order
            follows dict order.
    """
    import matplotlib.pyplot as plt

    names = list(examples)
    n_cols = max((len(v) for v in examples.values()), default=1)
    fig, axes = plt.subplots(
        len(names), n_cols, figsize=(2.2 * n_cols, 2.2 * len(names)),
        squeeze=False,
    )
    for i, name in enumerate(names):
        # plot each sampled example; pad rows shorter than the widest one
        for j, (key, label_id) in enumerate(examples[name]):
            mask = label_maps[key] == label_id
            if not mask.any():
                rprint(
                    f"[yellow]plot_category_examples: label {label_id} not "
                    f"found in '{key}' map — stale cache? Skipping.[/yellow]"
                )
                continue
            axes[i, j].imshow(_crop_example(mask), cmap="gray")
            # bold titles; long cache ids (e.g. dataset hashes) truncated
            key = key if len(key) <= 12 else f"{key[:9]}…"
            axes[i, j].set_title(
                f"{key} #{label_id}", fontsize=6, fontweight="bold"
            )
            axes[i, j].axis("off")
        for j in range(len(examples[name]), n_cols):
            axes[i, j].axis("off")
    for i, name in enumerate(names):
        # category label on the left of each row
        axes[i, 0].text(
            -0.35, 0.5, str(name), transform=axes[i, 0].transAxes,
            ha="right", va="center", fontsize=12, fontweight="bold",
            rotation=90,
        )
    fig.tight_layout()
    return fig