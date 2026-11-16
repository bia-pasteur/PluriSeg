"""StarDist representation: object prob + n_rays radial distances to boundary.

Reference: Schmidt et al. 2018. Encode is a numba port of the reference
implementation `pluriseg.vendor.stardist.geom2d._py_star_dist` (exact; the C lib
is not required). Decode ports the NMS of stardist nms.py + stardist2d.cpp
(scipy KDTree + shapely instead of nanoflann + Clipper) and rasterizes with
`pluriseg.vendor.stardist.geom2d.polygons_to_label`.
"""

import colorsys
from typing import Iterable

from matplotlib import pyplot as plt
import numpy as np
import shapely
import torch
from matplotlib.collections import LineCollection
from numba import njit, prange
from scipy.spatial import KDTree
from torch import nn

from pluriseg.representations.base import ChannelProps, Representation
from pluriseg.transforms import AffineTransformV2, OrthoTransform
from pluriseg.vendor.stardist.geom2d import polygons_to_label
from pluriseg.vendor.stardist.utils import edt_prob as compute_edt_prob


@njit(cache=True, parallel=True)
def _nb_star_dist(a, n_rays, grid_y, grid_x):
    """Numba port of `c_star_dist` / `_py_star_dist`: per-pixel ray tracing to
    the instance boundary. `a` is a uint16 label map; returns
    ((H-1)//grid_y+1, (W-1)//grid_x+1, n_rays) float32 radii, 0 on background.
    Rows run in parallel (prange): each pixel's rays are independent, and the
    dataset aug path calls this per patch — a multicore dataloader worker
    amortizes the per-call threading overhead.
    """
    H, W = a.shape
    out_H = (H - 1) // grid_y + 1
    out_W = (W - 1) // grid_x + 1
    dst = np.empty((out_H, out_W, n_rays), np.float32)
    st_rays = np.float32((2 * np.pi) / n_rays)
    for i in prange(out_H):
        for j in range(out_W):
            value = a[i * grid_y, j * grid_x]
            if value == 0:
                for k in range(n_rays):
                    dst[i, j, k] = 0
            else:
                for k in range(n_rays):
                    phi = np.float32(k * st_rays)
                    dy = np.cos(phi)
                    dx = np.sin(phi)
                    x = np.float32(0.0)
                    y = np.float32(0.0)
                    while True:
                        x += dx
                        y += dy
                        ii = int(round(i * grid_y + x))
                        jj = int(round(j * grid_x + y))
                        if (ii < 0 or ii >= H or
                                jj < 0 or jj >= W or
                                value != a[ii, jj]):
                            # small correction as we overshoot the boundary
                            t_corr = 1.0 - 0.5 / max(abs(dx), abs(dy))
                            x -= t_corr * dx
                            y -= t_corr * dy
                            dst[i, j, k] = np.sqrt(x**2 + y**2)
                            break
    return dst


def _ind_prob_thresh(prob, prob_thresh, b=2):
    """prob > prob_thresh, excluding a b-pixel margin at the image boundary."""
    mask = prob > prob_thresh
    if b is not None and b > 0:
        allowed = np.zeros_like(mask)
        allowed[b:-b, b:-b] = True
        mask &= allowed
    return mask


def non_maximum_suppression(dist, prob, grid=(1, 1), b=2, nms_thresh=0.5, prob_thresh=0.5,
                            use_bbox=True, use_kdtree=True):
    """Greedy NMS of ray-convex polygons (port of stardist nms.py + stardist2d.cpp).

    Retains polygons whose overlap with a higher-prob polygon is <= nms_thresh,
    with overlap = Ainter / min(Ai, Aj).

    dist (H,W,n_rays) float; prob (H,W) float in [0,1]; grid (gy,gx) stride of
    the pred maps.

    Returns points (N,2) [row,col], scores (N,), dist (N,n_rays) for survivors.
    """
    dist = np.asarray(dist, np.float32)
    prob = np.asarray(prob, np.float32)
    dist = np.maximum(dist, 0)  # negative radii -> self-intersecting polygons

    mask = _ind_prob_thresh(prob, prob_thresh, b)
    points = np.stack(np.where(mask), axis=1).astype(np.float32)
    dist_ = dist[mask]
    scores = prob[mask]

    # sort scores descendingly
    ind = np.argsort(scores)[::-1]
    dist_, scores, points = dist_[ind], scores[ind], points[ind]
    points = points * np.array(grid).reshape((1, 2))

    inds = _non_max_suppression_inds(
        dist_, points, scores, thresh=nms_thresh,
        use_bbox=use_bbox, use_kdtree=use_kdtree)
    return points[inds], scores[inds], dist_[inds]


def _non_max_suppression_inds(dist, points, scores, thresh=0.5,
                              use_bbox=True, use_kdtree=True):
    """Suppression loop over score-sorted polygons (port of c_non_max_suppression_inds).

    dist (N, n_rays); points (N,2) [row,col]; scores (N,) sorted desc (unused
    beyond ordering, kept for API parity with stardist).
    Returns bool survivor mask.
    """
    n_poly = len(dist)
    if n_poly == 0:
        return np.zeros(0, bool)

    n_rays = dist.shape[1]
    phis = np.arange(n_rays) * (2 * np.pi / n_rays)
    # stardist convention: y = p + d*sin(phi), x = p + d*cos(phi)
    sin, cos = np.sin(phis), np.cos(phis)

    # cartesian vertices (N, n_rays, 2) -> shapely polygons, areas, bboxes.
    # clamp radii to >=0: negative/collapsed predictions (e.g. untrained model
    # at sanity check) make self-intersecting polygons that break shapely.
    dist = np.maximum(dist, 0)
    pts = points[:, None, :] + dist[..., None] * np.stack([sin, cos], -1)
    polys = shapely.polygons(pts)
    polys = shapely.make_valid(polys)
    areas = shapely.area(polys)
    bboxes = shapely.bounds(polys)  # (N,4) minx,miny,maxx,maxy
    radius_outer = dist.max(-1)

    tree = KDTree(points) if use_kdtree else None
    max_radius = radius_outer.max()

    suppressed = np.zeros(n_poly, bool)
    for i in range(n_poly - 1):
        if suppressed[i]:
            continue
        # centers farther than max_radius + r_i => no possible overlap
        if tree is not None:
            j_list = tree.query_ball_point(
                points[i], r=max_radius + radius_outer[i])
        else:
            j_list = range(i + 1, n_poly)
        for j in j_list:
            if j <= i or suppressed[j]:
                continue
            if use_bbox and not bbox_intersect(bboxes[i], bboxes[j]):
                continue
            inter = shapely.intersection(polys[i], polys[j]).area
            overlap = inter / min(areas[i] + 1e-10, areas[j] + 1e-10)
            if overlap > thresh:
                suppressed[j] = True
    return ~suppressed


def bbox_intersect(b1, b2):
    """shapely bounds (minx,miny,maxx,maxy) intersect (touching counts)."""
    return b1[0] <= b2[2] and b2[0] <= b1[2] and b1[1] <= b2[3] and b2[1] <= b1[3]


class StarDistRep(Representation):
    """Channels:
        edt_prob: object prob = EDT of each instance, normalized per instance
            to [0, 1] (value 1 at the centermost pixel, 0 at the boundary);
            predicted with logits.
        radii: (H, W, n_rays) radial distance from each pixel to the instance
            boundary along ray k at phi_k = 2*pi*k/n_rays, ray k pointing
            (sin phi_k, cos phi_k) in (row, col); 0 on background.
        dist_mask: 1 where the EDT > 0 (instance pixels) — masks the dist
            loss to the foreground.
    """
    name = "stardist"
    channels = {
        "edt_prob": ChannelProps("target", 1, logits=True),
        "radii": ChannelProps("target", 32),
        "dist_mask": ChannelProps("loss", 1),
    }
    version = "2"  # 2: added dist_mask channel
    cacheable = True  # radii invariant to flips/rot90 after channel permutation; prob/radii flip/rot with the image
    plt_encoded_axes = 2
    plt_pred_axes = 2

    def __init__(self, n_rays: int = 32, dist_loss: str = "mae",
                 loss_weights: tuple = (1., 0.2), background_reg: float = 1e-4):
        super().__init__()
        assert n_rays % 4 == 0, "n_rays must be a multiple of 4 for D4-exact transform"
        self.n_rays = n_rays
        self.channels = {
            "edt_prob": ChannelProps("target", 1, logits=True),
            "radii": ChannelProps("target", n_rays),
            "dist_mask": ChannelProps("loss", 1),
        }
        self.dist_loss_type = dist_loss
        self.loss_weights = loss_weights  # (prob, dist)
        self.background_reg = background_reg

        self.criterion_prob = nn.BCEWithLogitsLoss(reduction="mean")
        self.criterion_dist = nn.L1Loss(reduction="none")

    def _cache_params(self) -> dict:
        return {"n_rays": self.n_rays}

    def encode_targets(self, label: np.ndarray) -> dict[str, np.ndarray]:
        label = np.ascontiguousarray(label, dtype=np.uint16)
        dist = _nb_star_dist(label, self.n_rays, 1, 1)
        prob = compute_edt_prob(label)
        return {
            "edt_prob": prob[..., None],
            "radii": dist,
            "dist_mask": (prob > 0)[..., None].astype(np.float32),
        }

    def decode_preds(self, preds: dict[str, torch.Tensor], cfg: dict | None = None) -> np.ndarray:
        """preds (C,H,W) -> instance label map via NMS + polygon rasterization.

        Port of StarDist2D._instances_from_prediction: threshold prob, greedy
        NMS (polygons overlap IoU = Ainter/min(Ai,Aj) > nms_thresh are dropped,
        highest prob kept), then paint survivors by ascending prob.
        """
        cfg = dict() if cfg is None else cfg
        # numpy doesn't support bfloat16 (precision=bf16-mixed), cast first
        prob = torch.sigmoid(preds['edt_prob'][0]).float().cpu().numpy()
        dist = preds['radii'].float().cpu().numpy(
        ).transpose(1, 2, 0)  # (H,W,n_rays)

        points, scores, polys = non_maximum_suppression(
            dist, prob,
            b=cfg.get('b', 2),
            nms_thresh=cfg.get('nms_thresh', 0.5),
            prob_thresh=cfg.get('prob_thresh', 0.5),
        )
        if len(points) == 0:  # empty input crashes polygons_to_label
            return np.zeros(prob.shape, np.uint32)
        return polygons_to_label(polys, points, prob.shape, prob=scores).astype(np.uint32)

    def transform(
        self, maps: dict[str, np.ndarray], t: OrthoTransform
    ) -> dict[str, np.ndarray]:
        """Scalar channels flip/rot with the image; radii also permute rays.

        Ray k points along (sin φ_k, cos φ_k) in (row, col), φ_k = 2πk/n_rays.
        The transform's linear part maps it to ray i = σ·k + d with
        σ = (-1)^(flip_x+flip_y), d = n/2·flip_x - n/4·rot90 (exact because
        n_rays % 4 == 0), so out[..., i] = in[..., σ(i - d)].
        """
        out = {k: t.apply_to_image(v) for k, v in maps.items()}
        n = self.n_rays
        sigma = -1 if t.flip_x != t.flip_y else 1
        d = (n // 2 if t.flip_x else 0) - t.rot90 * (n // 4)
        perm = (sigma * (np.arange(n) - d)) % n
        # torch gather: numpy fancy indexing on the 67 MB radii array costs
        # ~86 ms; the same gather through torch is ~8 ms (tensor.copy_ loop
        # beats the numpy take machinery on this shape)
        if not np.array_equal(perm, np.arange(n)):
            out["radii"] = torch.from_numpy(
                np.ascontiguousarray(out["radii"])
            )[..., torch.from_numpy(perm)].numpy()
        return out

    def transform_affine(
        self,
        maps: dict[str, np.ndarray],
        t: AffineTransformV2,
        t_label: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """edt_prob warps unchanged: it is the per-instance ratio d/d_max and
        isotropic scale multiplies both by s, so the normalized field is
        scale-invariant (rotation/flip change nothing). radii re-encode from
        t_label: rays must stay axis-aligned, and the per-pixel ray tracing
        cannot be expressed as an in-place op. dist_mask: order=0 warp of
        the binary mask."""
        edt_prob, dist_mask = t.apply_to_images(
            images=[maps["edt_prob"], maps["dist_mask"]],
            bilinear=[True, False]
        )
        radii = _nb_star_dist(
            np.ascontiguousarray(t_label, dtype=np.uint16),
            self.n_rays, 1, 1)
        return {
            "edt_prob": edt_prob,
            "radii": radii,
            "dist_mask": dist_mask
        }

    def loss(self, preds, targets):
        """BCEWithLogits on prob + masked dist loss (port of stardist models/base.py:310-349).

        dist penalty: 'mae' (L1), 'mse' (L2) or 'iou' (ray-wise 1 - IoU of
        y_true², y_pred² shapes). Masked to fg via targets['dist_mask'],
        normalized by mask mean; background_reg adds reg_weight * mean(|pred| on
        background). Total = w_prob * prob + w_dist * dist.
        """
        w_prob, w_dist = self.loss_weights
        loss_prob = self.criterion_prob(preds["edt_prob"], targets["edt_prob"])

        mask = (targets["dist_mask"] > 0).to(preds["radii"].dtype)
        pred = preds["radii"]
        true = targets["radii"]
        if self.dist_loss_type == "mae":
            pen = (pred - true).abs()
        elif self.dist_loss_type == "mse":
            pen = (pred - true).square()
        elif self.dist_loss_type == "iou":
            # stardist masked_loss_iou: inter = mean(sign(p)·min(t,p)²), union = mean(max(t,p)²)
            inter = (pred.sign() * torch.minimum(true, pred).square()).mean(-1)
            union = torch.maximum(true, pred).square().mean(-1)
            pen = (1 - inter / (union + 1e-7))[..., None]
        else:
            raise ValueError(self.dist_loss_type)

        norm = mask.mean() + 1e-7
        dist = (mask * pen).mean() / norm
        reg = (1 - mask) * pred.abs()
        dist = dist + self.background_reg * reg.mean()

        return {
            "loss": w_prob * loss_prob + w_dist * dist,
            "loss_prob": loss_prob,
            "loss_dist": dist,
        }

    def display_encoded_plt(self, encoded, axes, prob_thresh=0.5, nms_thresh=0.5, seed=None, **kwargs):
        self._display_plt(
            prob=encoded['edt_prob'][..., 0],
            radii=encoded['radii'],
            axes=axes,
            prob_thresh=prob_thresh,
            nms_thresh=nms_thresh,
            seed=seed
        )

    def _display_plt(
        self,
        prob: np.ndarray,
        radii,
        axes: list[plt.Axes],
        prob_thresh=0.5,
        nms_thresh=0.5,
        seed=None,
    ):
        """Left: EDT prob map with the dist_mask (fg) as a red contour overlay.
        Right: star polygons at the NMS-selected centers (decode_preds' actual
        center choice)."""

        axes[0].imshow(prob, cmap='gray')
        axes[0].set_title('EDT prob')
        # if 'dist_mask' in maps:
        #     axes[0].contour(maps['dist_mask'][..., 0], levels=[0.5],
        #                     colors='r', linewidths=0.8)

        points, scores, dist = non_maximum_suppression(
            radii, prob,
            prob_thresh=prob_thresh, nms_thresh=nms_thresh)
        ys, xs = points[:, 0].astype(int), points[:, 1].astype(int)
        n = len(ys)

        # polar -> cartesian vertices; convention as stardist: row += r*sin(phi), col += r*cos(phi)
        phis = np.linspace(0, 2 * np.pi, self.n_rays, endpoint=False)
        rows = ys[:, None] + dist * np.sin(phis)
        cols = xs[:, None] + dist * np.cos(phis)

        rng = np.random.default_rng(seed)
        h, l, s = rng.uniform(0, 1, n), rng.uniform(.4,
                                                    1, n), rng.uniform(.2, .8, n)

        axes[1].imshow(np.zeros_like(prob), cmap='gray', vmin=0, vmax=1)
        for i in range(n):
            c = colorsys.hls_to_rgb(h[i], l[i], s[i])
            # rays from center to each vertex (cf. stardist draw_polygons show_dist)
            ray_lines = np.empty((self.n_rays, 2, 2))
            ray_lines[:, 0, 0] = cols[i]
            ray_lines[:, 0, 1] = rows[i]
            ray_lines[:, 1, 0] = xs[i]
            ray_lines[:, 1, 1] = ys[i]
            axes[1].add_collection(LineCollection(
                ray_lines, colors=c, linewidths=0.4, zorder=1))
            axes[1].plot(np.r_[cols[i], cols[i, 0]], np.r_[rows[i], rows[i, 0]],
                         '-', alpha=1, linewidth=0.6, zorder=2, color=c)
            axes[1].plot(xs[i], ys[i], '.', markersize=4 *
                         scores[i] + 1, color=c)
        axes[1].set_title(f'NMS star polygons (n={n})')

    def display_preds_plt(self, preds, axes, **kwargs):
        raise NotImplementedError
