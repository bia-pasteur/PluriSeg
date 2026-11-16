"""Omnipose representation: flow + Eikonal distance field + boundary.


Channels (omnipose's 3 classes + loss weight):
    flow: (dy, dx) gradient of the distance field, x5 scaling at encode
        (same convention as cellpose).
    dist: Eikonal distance field T; background sentinel -5 (dist_bg), so
        dist > 0 doubles as the foreground indicator.
    bd:   binary 1-connected boundary of each instance (affinity graph).
    weight: loss weight map, fg = 1 / bg = 0.5; not predicted.
"""

import warnings

import numpy as np
import torch

import edt as edt_lib

from pluriseg.display import dx_to_circ
from pluriseg.representations.base import ChannelProps, Representation
from pluriseg.transforms import AffineTransformV2, OrthoTransform
from pluriseg.utils import tensor_to_numpy
from pluriseg.vendor.omnipose.core import (compute_masks, divergence_torch,
                                         masks_to_flows, scale_to_tenths)
from pluriseg.vendor.omnipose.loss import (BatchMeanMSE, DerivativeLoss,
                                         SSL_Norm, WeightedMSELoss)

# batch_labels' dist_bg: dist target is -5 on background so that dist > 0
# doubles as the foreground indicator
DIST_BG = 5


class OmniposeRep(Representation):
    name = "omnipose"
    channels = {
        "flow": ChannelProps("target", 2),
        "dist": ChannelProps("target", 1),
        "weight": ChannelProps("loss", 1),
    }
    # flow rotates component-wise (D4); dist/bd/weight are scalars
    cacheable = True
    plt_encoded_axes = 2
    plt_pred_axes = 2

    def __init__(self):
        super().__init__()
        self.WeightedMSE = WeightedMSELoss()
        self.SSNLoss = SSL_Norm()
        self.DerivativeLoss = DerivativeLoss()
        self.MSELoss = BatchMeanMSE()

    def _cache_params(self) -> dict:
        return {}

    def encode_targets(self, label: np.ndarray) -> dict[str, np.ndarray]:
        # dists is precomputed here instead of letting masks_to_flows compute
        # it: on a fully-foreground crop (no background pixel and a single
        # instance, e.g. a random crop inside one large object) the EDT is inf
        # everywhere and get_niter casts inf to int (warning + garbage
        # n_iter, which silently degenerates the targets). Clamping to half
        # the patch diagonal keeps n_iter in the usual range (a 256px patch
        # would have max EDT ~128 → n_iter ~150). Passing dists also skips
        # the internal ncolor.format_labels; EDT is identical either way.
        dists = edt_lib.edt(label, parallel=-1).astype(np.float64)
        if not np.isfinite(dists).all():
            fill = 0.5 * max(label.shape)
            dists[~np.isfinite(dists)] = fill

        masks, dists, boundaries, T, mu = masks_to_flows(
            label,
            dists=dists,
            use_gpu=False,
            device='cpu'
        )

        # batch_labels: dist target gets the -5 background sentinel so that
        # dist > 0 doubles as the foreground indicator
        T = T.clone()
        T[T <= 0] = -DIST_BG

        fg = label > 0
        weight = (1+fg) / 2

        return {
            # batch_labels: flow target is mu*5 (same x5 convention as cellpose)
            'flow': (5 * mu).permute(1, 2, 0).numpy(),
            'dist': T[..., None].numpy(),
            'weight': weight[..., None]
        }

    def decode_preds(self, preds: dict[str, torch.Tensor], cfg: dict | None = None) -> np.ndarray:

        # numpy doesn't support bfloat16 (precision=bf16-mixed), cast first
        dP = preds['flow'].float().cpu().numpy()  # (2,H,W)
        dist = preds['dist'][0].float().cpu().numpy()  # (H,W)

        cfg = dict() if cfg is None else cfg

        # degenerate decode: with fewer fg pixels than get_masks' DBSCAN
        # min_samples (5), every point becomes an outlier and the
        # outlier-snapping call kneighbors(n_neighbors=5) on a fit of those
        # same points, raising ValueError — return an empty mask instead.
        if (dist > 0).sum() < 5:
            return np.zeros(dist.shape, np.uint32)

        # niter mirrors cellpose_omni.models.eval: 2*(dim+1)*mean(dist[fg]) —
        # the vendored compute_masks None-default (200) over-integrates and
        # splits instances.
        if cfg.get('niter') is None:
            fgb = dist > 0
            niter = int(2 * 3 * dist[fgb].mean()) if fgb.any() else 1
        else:
            niter = cfg['niter']

        with warnings.catch_warnings():
            # NaN in the predicted dist channel makes get_niter's int cast
            # emit garbage n_iter; fail loudly instead of decoding nonsense.
            # warnings.filterwarnings("error", category=RuntimeWarning,
            #                         message=r".*invalid value encountered in cast.*")
            # to catch the odd inf filter ?
            masks, _, _, _, _ = compute_masks(
                dP=dP,
                dist=dist,
                cluster=cfg.get('cluster', True),
                flow_threshold=cfg.get('flow_threshold', 0.4),
                niter=niter,
                min_size=cfg.get('min_size', 15),
                suppress=cfg.get('suppress', True),
            )

        return masks.astype(np.uint32)

    def transform(
        self, maps: dict[str, np.ndarray], t: OrthoTransform
    ) -> dict[str, np.ndarray]:
        # flow is (H, W, 2) = (dy, dx); convert to (2, H, W)
        f = maps["flow"].transpose(2, 0, 1)
        if t.flip_y:  # flip along H: dy negates
            f = np.stack([-f[0, ::-1], f[1, ::-1]])
        if t.flip_x:  # flip along W: dx negates
            f = np.stack([f[0, :, ::-1], -f[1, :, ::-1]])
        if t.rot90:
            # spatial rot90(k=1) is a visual CCW rotation: +x -> -y in array
            # coords, i.e. (dy, dx) -> (-dx, dy)
            f = np.rot90(f, k=t.rot90, axes=(1, 2))
            dy, dx = f[0].copy(), f[1].copy()
            for _ in range(t.rot90):
                dy, dx = -dx, dy
            f = np.stack([dy, dx])
        out = {
            "flow": f.transpose(1, 2, 0),
            "dist": t.apply_to_image(maps["dist"]),
            "weight": t.apply_to_image(maps["weight"]),
        }
        return out

    def transform_affine(
        self,
        maps: dict[str, np.ndarray],
        t: AffineTransformV2,
        t_label: np.ndarray,
    ) -> dict[str, np.ndarray]:
        # flow: same recipe as cellpose — rotate by the linear part, undo the
        # scale on the magnitude (mu is the Eikonal gradient, |mu| ~ 1,
        # scaled x5 at encode)
        f,dist = t.apply_to_images(
            images=[maps["flow"],maps["dist"]],
            bilinear=[True,True]
        )
        a, b = t.matrix[0, :2]  # y' = a*y + b*x
        d, e = t.matrix[1, :2]  # x' = d*y + e*x
        flow = np.stack([a * f[..., 0] + b * f[..., 1],
                         d * f[..., 0] + e * f[..., 1]], axis=-1) / t.scale

        # dist: distance field, values scale by s (d' = s*d); the bg
        # sentinel -5 stays at -5 (the loss weights bg, and a per-patch
        # varying sentinel would be unlearnable); border blends that dip
        # below zero snap to the sentinel, same border noise as fg maps
        dist = np.where(dist > 0, dist * t.scale, -DIST_BG)

        # weight is fg=1/bg=0.5: bilinear warping blends the two at the
        # boundary ring; it is a pure function of the label, so re-encode
        # from the warped label instead (same fix class as mask fg)
        weight = ((t_label > 0) + 1) / 2
        weight = weight[..., None].astype(maps["weight"].dtype)

        return {
            "flow": flow,
            "dist": dist,
            "weight": weight,
        }

    def loss(self, preds, targets):
        """Port of omnipose.core.loss (repo core.py:2753) minus AffinityLoss.
        Components in the repo's list order:

            flow_mse   WeightedMSELoss(flow, veci)  vs the x5-scaled target
            SSL        SSL_Norm orientation term    (1-cos^2)^2, weighted
            bd_loss    DerivativeLoss on the dist channel (repo's variable
                       name; no bd involved)
            norm_loss  SSL_Norm magnitude term      MSE(|pred|, |GT|)
            dist_loss  WeightedMSELoss(dist, dist_gt)
            lossDC     BatchMeanMSE(div GT flow, div pred flow)

        Each component is rescaled by scale_to_tenths (max_gain=1e12, as at
        the repo call site) then summed; loss_raw is the unrescaled mean.
        """
        flow = preds['flow']            # (B,2,H,W)
        veci = targets['flow']          # (B,2,H,W), x5-scaled
        dist = targets['dist'][:, 0]    # (B,H,W), bg sentinel -5
        w = targets['weight'][:, 0]     # (B,H,W)

        cellmask = dist > 0
        wt = torch.stack([w] * flow.shape[1], dim=1)  # (B,dim,H,W)

        flow_mse = self.WeightedMSE(flow, veci, wt)
        SSL, norm_loss = self.SSNLoss(flow, veci, dist, w, None)
        bd_loss = self.DerivativeLoss(preds['dist'], targets['dist'],
                                      w.unsqueeze(1), cellmask.unsqueeze(1))
        dist_loss = self.WeightedMSE(preds['dist'], targets['dist'],
                                     w.unsqueeze(1))
        lossDC = self.MSELoss(divergence_torch(veci), divergence_torch(flow))

        losses = [flow_mse, SSL, bd_loss, norm_loss, dist_loss, lossDC]
        loss_raw = sum(losses).detach() / len(losses)
        loss = sum(scale_to_tenths(l, max_gain=1e12) for l in losses)

        return {
            'loss': loss,
            'loss_raw': loss_raw,
            'loss_flow_mse': flow_mse.detach(),
            'loss_SSL': SSL.detach(),
            'loss_bd': bd_loss.detach(),
            'loss_norm': norm_loss.detach(),
            'loss_dist': dist_loss.detach(),
            'loss_DC': lossDC.detach(),
        }

    def _display_plt(self, dists: np.ndarray, flow: np.ndarray, axes):
        dists_abs = np.max(np.abs(dists))
        axes[0].imshow(dists, cmap='coolwarm', vmin=-dists_abs, vmax=dists_abs)
        axes[0].set_title('Distance')

        flow_rgb = dx_to_circ(flow.transpose((2, 0, 1)) / 5.)
        axes[1].imshow(flow_rgb)
        axes[1].set_title('Flow')

    def display_encoded_plt(self, encoded, axes, **kwargs) -> None:
        self._display_plt(
            encoded['dist'][..., 0],
            encoded['flow'],
            axes
        )

    def display_preds_plt(self, preds, axes, **kwargs):
        self._display_plt(
            tensor_to_numpy(preds['dist'])[..., 0],
            tensor_to_numpy(preds['flow']),
            axes
        )
