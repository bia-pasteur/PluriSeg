"""Direct-mask representation: foreground + weighted cross-entropy.

Based on:
Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. “U-Net: Convolutional Networks for Biomedical Image Segmentation.” In Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, edited by Nassir Navab, Joachim Hornegger, William M. Wells, and Alejandro F. Frangi. Springer International Publishing, 2015. https://doi.org/10.1007/978-3-319-24574-4_28.

we split the weights between distance weights (cached) and class balance weight (recomputed) at loss time on the cropped patch so eq (2) remains correct

We create boundaries by eroding each label by one pixel only in boundaries : that is where one of the neighbors (8 connectivity) is another label (excluding bg).

Decode is the U-Net protocol as published: sigmoid threshold at 0.5 then
connected components — no watershed or post-filtering (instance separation
relies on the eroded seam surviving the threshold).

"""

import numpy as np
import scipy.ndimage
import skimage
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from pluriseg.representations.base import ChannelProps, Representation
from pluriseg.transforms import AffineTransformV2, OrthoTransform
from pluriseg.utils import tensor_to_numpy

_NEIGHBORS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_NEIGHBORS_8 = _NEIGHBORS_4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]


class MaskRep(Representation):
    name = "mask"
    channels = {
        'fg': ChannelProps("target", 1, logits=True),
        'distance_weights': ChannelProps("loss", 1),
    }
    cacheable = True
    plt_encoded_axes = 2
    plt_pred_axes = 1

    def __init__(
        self,
        w0: float = 10,  # from paper
        sigma: float = 5.0  # from paper
    ):
        super().__init__()
        self.w0 = w0
        self.sigma = sigma

    def _cache_params(self) -> dict:
        return {"w0": self.w0, "sigma": self.sigma}

    def encode_targets(self, label: np.ndarray) -> dict[str, np.ndarray]:

        # erode each label by one
        eroded_labels = erode_labels(label, 2)
        fg = eroded_labels > 0
        # compute weights
        # compute d1 and d2, wc and w0
        d1, d2, _ = nearest_and_second_label_distances(eroded_labels)

        wc = np.ones(fg.shape)
        n_pixels = fg.shape[0] * fg.shape[1]
        fg_freq = np.sum(fg) / n_pixels
        wc[fg] = 1 - fg_freq
        wc[~fg] = fg_freq

        # w0 = 0 in fg because d1 and d2 == np.inf

        weights = self.w0 * np.exp(
            - np.square(d1 + d2) / (2 * self.sigma**2)
        )

        return {
            "fg": fg[:, :, None],  # (H,W,1)
            "distance_weights": weights[:, :, None]  # (H,W,1)
        }

    def decode_preds(self, preds: dict[str, torch.Tensor], cfg: dict | None = None) -> np.ndarray:

        cfg = dict() if cfg is None else cfg
        threshold = cfg.get('threshold', 0.5)

        # connected components
        fg_proba = torch.sigmoid(preds['fg'][0]).float().cpu().numpy()

        fg_bin = fg_proba > threshold

        labels = skimage.measure.label(fg_bin).astype(np.uint32)

        return labels

    def _compute_class_weight(
        self,
        fg: torch.Tensor
    ) -> torch.Tensor:
        # compute the wc part on the fly, so (2) is respected for each patch
        fg_freq = fg.mean(dim=(-2, -1), keepdim=True)  # (B,1,1,1), per img
        wc = torch.where(fg > 0, 1 - fg_freq, fg_freq)
        return wc

    def loss(
        self,
        preds: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:

        fg = targets['fg']  # (B,1,H,W)
        wc = self._compute_class_weight(fg)

        loss = F.binary_cross_entropy_with_logits(
            input=preds['fg'],  # fg is logits (pre sigmoid)
            target=fg,
            weight=wc + targets['distance_weights'],
            reduction='mean'
        )

        return {
            'loss': loss
        }

    def transform(
        self, maps: dict[str, np.ndarray], t: OrthoTransform
    ) -> dict[str, np.ndarray]:
        return {name: t.apply_to_image(m) for name, m in maps.items()}

    def transform_affine(
        self,
        maps: dict[str, np.ndarray],
        t: AffineTransformV2,
        t_label: np.ndarray,
    ):
        # we can show that w' = w0·(w/w0)^{s²}
        w_t = t.apply_to_images(
            images=[maps['distance_weights']],
            bilinear=[True])[0]
        w_ts = self.w0 * np.power(
            np.maximum(w_t / self.w0, 0), #safety for interpolation
            t.scale**2
        )
        # bilinear warping of the pre-encoded fg can collapse the 1px ring
        # between touching instances (e.g. after downscale, two neighbours
        # merge in the target). The warped order=0 label preserves instance
        # identities, so re-encode the eroded mask from it — geometry stays
        # consistent with the warped label by construction.
        fg = (erode_labels(t_label) > 0)[:, :, None]
        return {
            'fg': fg.astype(maps['fg'].dtype),
            'distance_weights': w_ts
        }

    def display_encoded_plt(
        self,
        encoded: dict[str, np.ndarray],
        axes: list[plt.Axes],
        show_combined_w: bool = True,
        **kwargs
    ) -> None:

        axes[0].imshow(
            encoded['fg'][..., 0], 
            cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('Foreground')
        if show_combined_w:
            fg = torch.from_numpy(encoded['fg']).permute(
                (2, 0, 1)).unsqueeze(0).float()
            wc = self._compute_class_weight(fg)[0, 0].numpy()
            wd = encoded['distance_weights'][..., 0]

            axes[1].imshow(wc + wd, cmap='jet')
            axes[1].set_title('(Distance + Class) weights')
        else:
            axes[1].imshow(encoded['distance_weights'][..., 0], cmap='jet')
            axes[1].set_title('Distance weights')

    def display_preds_plt(
        self,
        preds: dict[str, torch.Tensor],
        axes: list[plt.Axes],
        **kwargs
    ):
        fg = tensor_to_numpy(torch.sigmoid(preds['fg']))
        axes[0].imshow(
            fg[..., 0], 
            cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('Foreground prob.')


def erode_labels(label: np.ndarray, connectivity: int = 2) -> np.ndarray:
    """Erode every instance of an integer label map by one pixel.

    Vectorized over all labels: a pixel keeps its label only if its whole
    4- (connectivity=1) or 8- (connectivity=2) neighborhood holds the same
    label, else it becomes 0 (background). 1px-wide structures vanish.
    Equivalent to a per-label scipy.ndimage.binary_erosion loop (verified),
    in O(8) shifts instead of O(n_instances * H * W).

    Args:
        label: (H, W) integer instance label map (0 = background).
        connectivity: 1 (cross) or 2 (square) structuring element.
    Returns:
        (H, W) eroded label map, same dtype; 0 = background/eroded ring.
    """
    if connectivity not in (1, 2):
        raise ValueError(f"connectivity must be 1 or 2, got {connectivity}")
    H, W = label.shape
    pad = np.pad(label, 1)
    keep = np.ones((H, W), bool)
    for dy, dx in _NEIGHBORS_4 if connectivity == 1 else _NEIGHBORS_8:
        offset = pad[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
        keep &= ((offset == label) | (offset == 0))
    return np.where(keep, label, 0)


def nearest_and_second_label_distances(
    label: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distance from every background pixel to the nearest instance (d1), the
    second-nearest *different* instance (d2), and which instance is nearest.

    The nearest-instance label comes from one index-returning EDT of the full
    label map. d2 has no single-EDT closed form — the second nearest is
    defined per-pixel against the instance that won d1 — so it falls to one
    EDT per instance label (EDT of "everything but this instance" measures,
    on background pixels, the distance to that instance), reduced with a
    running first/second minimum so the loop is order-independent. Cost is
    n_instances full-map EDTs; the accepted cost for dense label maps.

    Args:
        label: (H, W) integer instance label map (0 = background).
    Returns:
        (d1, d2, nearest) float32/int64 (H, W); d1 = d2 = +inf on instance
        pixels (undefined there; mask away downstream).
    """
    inds = scipy.ndimage.distance_transform_edt(
        label == 0, return_distances=False, return_indices=True)
    # indices point at the nearest instance pixel -> its label
    nearest = label[inds[0], inds[1]].astype(np.int64)

    # per-instance distance maps, reduced with a running first/second min:
    #   d2 <- min(d2, max(d, d1));  d1 <- min(d1, d)
    # keeps the two smallest distances per pixel regardless of loop order.
    d1 = np.full(label.shape, np.inf)
    d2 = np.full(label.shape, np.inf)
    for lab in np.unique(label[label != 0]):
        # on background pixels this EDT is the distance to instance `lab`
        d = scipy.ndimage.distance_transform_edt(label != lab)
        d2 = np.minimum(d2, np.maximum(d, d1))
        d1 = np.minimum(d1, d)
    d1[label != 0] = np.inf
    d2[label != 0] = np.inf
    return d1.astype(np.float32), d2.astype(np.float32), nearest
