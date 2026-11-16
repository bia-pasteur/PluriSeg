
from functools import lru_cache
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import numpy as np
from glasbey import create_palette

from pluriseg.transforms import normalize_percentile


@lru_cache(maxsize=1)
def _glasbey_rgb(size: int = 256) -> np.ndarray:
    """Glasbey palette (most-distinct-first ordering), as (size, 3) floats."""
    return np.array(create_palette(
        palette_size=size,
        as_hex=False,
        optimize_palette=False,
        lightness_bounds=(30, 90),
        chroma_bounds=(10, 90),
        colorblind_safe=True
    ))


def get_labels_cmap(
        label: np.ndarray,           # (H, W) integer instance map
) -> ListedColormap:
    """Colormap sized to this label map's max id, so imshow(label, cmap=...)
    needs no data remapping. Slot 0 is black background, slots 1..max taken
    from the start of the Glasbey most-distinct palette (cached).
    """
    return get_glasbey_cmap(int(label.max()) + 1)


@lru_cache(maxsize=16)
def get_glasbey_cmap(n: int) -> ListedColormap:
    """ListedColormap sized to n labels: slot 0 black background, slots 1..n-1
    taken from the start of the Glasbey most-distinct palette (cached)."""
    rgb = _glasbey_rgb(n)
    palette = np.ones((n, 4))
    palette[0, :3] = 0
    palette[1:, :3] = rgb[: n - 1]
    return ListedColormap(palette)


@lru_cache(maxsize=16)
def get_glasbey_colors(n: int) -> np.ndarray:
    """"""
    rgb = _glasbey_rgb(n)
    palette = np.ones((n, 3))
    palette[0, :] = 0
    palette[1:, :] = rgb[: n - 1]
    return palette


def get_random_colors(n: int) -> np.ndarray:

    rng = np.random.default_rng(0)
    palette = np.ones((n, 3))
    palette[0, :] = 0
    palette[1:, :] = rng.random((n-1, 3))
    return palette


def label_edges(label: np.ndarray) -> np.ndarray:
    """(H, W) integer instance map -> (H, W) bool edge map marking every pixel
    with a 4-connected neighbor of a different id. Both sides of an instance
    boundary are marked (mirrors erode_labels' padded-shift pattern); edge
    mode padding means image borders are never edges."""
    H, W = label.shape
    pad = np.pad(label, 1, mode="edge")
    edge = np.zeros((H, W), bool)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        offset = pad[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
        edge |= offset != label
    return edge


def dx_to_circ(dP):
    """Converts the optic flow representation to a circular color representation.

    from: https://github.com/MouseLand/cellpose/blob/a54cb48849b7e225a81e8e43dcb042d42427f543/cellpose/plot.py#L25

    Args:
        dP (ndarray): Flow field components [dy, dx].

    Returns:
        ndarray: The circular color representation of the optic flow.


    """
    mag = 255 * \
        np.clip(normalize_percentile(np.sqrt(np.sum(dP**2, axis=0))))
    angles = np.arctan2(dP[1], dP[0]) + np.pi
    a = 2
    mag /= a
    rgb = np.zeros((*dP.shape[1:], 3), "uint8")
    rgb[..., 0] = np.clip(mag * (np.cos(angles) + 1), 0, 255).astype("uint8")
    rgb[..., 1] = np.clip(
        mag * (np.cos(angles + 2 * np.pi / 3) + 1), 0, 255).astype("uint8")
    rgb[..., 2] = np.clip(
        mag * (np.cos(angles + 4 * np.pi / 3) + 1), 0, 255).astype("uint8")

    return rgb


def plt_be_stilish(grid:bool=True):
    """Prettier matplotlib defaults for the metrics figures."""
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#fafafa",
        "axes.edgecolor": "#cccccc",
        "axes.grid": grid,
        "grid.color": "#e0e0e0",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 7.5,
        "lines.linewidth": 1.2,
    })
    plt_be_stilish.done = True
