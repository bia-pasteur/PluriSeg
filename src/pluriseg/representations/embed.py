"""EmbedSeg representation: spatial embeddings + clustering.

Reference: Lalit et al. 2022, EmbedSeg (CC BY-NC 4.0). The network emits
per-pixel offsets, per-axis sigmas and seediness; embeddings
``tanh(offset) + xym`` of one instance agree on the absolute coordinates of
its center. Loss and clustering core are vendored from EmbedSeg
(``pluriseg.vendor.embedseg``).
"""

import numpy as np
import torch

from pluriseg.display import dx_to_circ, get_glasbey_colors
from pluriseg.utils import relabel_contiguous, tensor_to_numpy
from pluriseg.representations.base import ChannelProps, Representation
from pluriseg.transforms import AffineTransformV2, OrthoTransform
from pluriseg.vendor.embedseg.utils import Cluster
from pluriseg.vendor.embedseg.my_loss import SpatialEmbLoss


class EmbedsegRep(Representation):
    name = "embedseg"
    # 3: per-axis sigma targets (was isotropic); 2: abs coords/loss pairing
    version = "4"
    channels = {
        "offset": ChannelProps("target", 2),   # (dy, dx) toward center
        "sigma": ChannelProps("target", 2),    # per-axis, kept anisotropic
        # no GT upstream; trained on dist.detach
        "seed": ChannelProps("indirect", 1, logits=True),
        "instance": ChannelProps("loss", 1),  # loss only: relabeled ids
        "center": ChannelProps("loss", 1),    # loss only: centroid image
    }
    # offsets transform exactly under D4 (rotate/flip components)
    cacheable = True
    plt_encoded_axes = 2
    plt_pred_axes = 2

    def __init__(self, sigma_ceil: float | None = None, train_patch: int = 256):
        super().__init__()
        # sigma ceiling = smallest sane clustering ball; None = 1-px ball
        # (s_max = ln2*train_patch^2). Prevents degenerate sigma for
        # 1-px-wide instances (mean d^2 = 0 in the thin axis). Applied to
        # encode-side targets AND as a prediction clamp in loss()
        # (exp(10*sigma) overflows to inf above ~8.87 → NaN gradients).
        self.sigma_ceil = (
            np.log(np.log(2.0) * train_patch**2) / 10.0
            if sigma_ceil is None else sigma_ceil)
        self.train_patch = train_patch
        self.criterion = SpatialEmbLoss(foreground_weight=10)  # vendor 2D default
        # the vendored loss registers xym as a persistent buffer; rebuilds in
        # loss() replace it per input shape, so it must not enter state_dict
        # (checkpoint size mismatch on load)
        self.criterion.register_buffer(
            "xym", self.criterion.xym, persistent=False)
        # (H, W) the criterion's xym buffer matches
        self._criterion_shape = None

    def _cache_params(self) -> dict:
        return {"sigma_ceil": self.sigma_ceil, "train_patch": self.train_patch}

    def init_head(self, head) -> None:
        """Mirror BranchedERFNet.init_output with a safer sigma init: offsets
        start at tanh(0) + xym = absolute coordinates; sigma bias 0.3.
        Head channels in self.channels order: offset(2) sigma(2) seed(1)."""
        conv = head.net[-1]
        # Head emits non-LOSS channels in self.channels order, i.e. output
        # channels [0:2] = offset (dy, dx), [2:4] = sigma, [4] = seed.
        with torch.no_grad():
            # offset channels out=0: prediction 0 => spatial_emb =
            # tanh(0) + xym = the pixel's absolute coordinates, the loss's
            # only sane starting point (random offsets would scatter every
            # pixel's embedding to a random absolute position)
            conv.weight[:, 0:2].fill_(0)
            conv.bias[0:2].fill_(0)
            # sigma channels out=2: weight 0 + bias 0.3 => sigma_pred = 0.3,
            # s = exp(3) = 20. Upstream's bias 1 (s = exp(10) = 2.2e4) puts
            # dist = exp(-d^2*s) below float32 underflow beyond ~1 px of each
            # centroid — gradients survive only there and the escape race
            # (vs seed saturation) collapses ~50% of runs into the
            # loss == 2.0 dead basin. At
            # 0.3, worst-case d^2*s over a 384 patch (d <= 1.5 in
            # 1/train_patch units) is 45 < the underflow threshold 87, so
            # every instance pixel has live gradient from step 0; per-axis
            # sigma targets span ~0.25-0.75, so 0.3 starts at scale.
            conv.weight[:, 2:4].fill_(0)
            conv.bias[2:4].fill_(0.3)

    def encode_targets(self, label: np.ndarray) -> dict[str, np.ndarray]:
        """Instance ids -> offset / sigma / seed / instance / center maps.

        Offset target: center_xy - pixel_xy per instance (channel 0 = dy,
        channel 1 = dx, matching the framework (H, W, 2) convention), in
        absolute pixels scaled by 1/train_patch — crop-invariant, offsets
        bounded by object size / train_patch (tanh-safe whenever objects are
        smaller than the training patch).
        Sigma target: per-instance per-axis s_i = exp(10 sigma_i) =
        ln2 / mean(d_i^2), d_i the per-axis distance to the centroid (same
        1/train_patch scale); capped at `sigma_ceil` (default = a 1-px-radius
        ball) so 1-px-wide instances (mean d^2 = 0 in the thin axis) get a
        sane smallest ball instead of a degenerate sub-pixel one.
        Seed target: binary
        foreground (1 on instance, 0 on background) — the seediness branch
        predicts fg, and at perfect embeddings the loss's own target
        dist.detach equals 1 on the instance. Supersedes the scan §4
        dist-field seed target (decoded masks truncated at the RMS ball).
        """
        H, W = label.shape
        label = relabel_contiguous(label)
        max_label = int(label.max())

        scale = 1.0 / self.train_patch
        ym, xm = np.mgrid[0:H, 0:W] * scale

        offset = np.zeros((H, W, 2), np.float32)
        sigma = np.zeros((H, W, 2), np.float32)
        seed = np.zeros((H, W, 1), np.float32)
        instance = np.zeros((H, W, 1), np.float32)
        center = np.zeros((H, W, 1), np.float32)

        for k in range(1, max_label + 1):
            yy, xx = np.where(label == k)
            # centroid in normalized coords (matches the loss: center is read
            # off xym, and its fallback is the xym mean over the instance)
            cy, cx = ym[yy, xx].mean(), xm[yy, xx].mean()

            dy2 = (ym[yy, xx] - cy) ** 2
            dx2 = (xm[yy, xx] - cx) ** 2

            # sigma target: per-axis membership ball (dist = 0.5) hits the
            # mean per-axis radius; capped at sigma_ceil (1-px ball by
            # default) — mean d^2 = 0 for 1-px-wide instances must map to
            # the smallest sane ball, not a degenerate sub-pixel one
            s_y = min(np.log(np.log(2.0) / max(dy2.mean(), 1e-12)) / 10.0, self.sigma_ceil)
            s_x = min(np.log(np.log(2.0) / max(dx2.mean(), 1e-12)) / 10.0, self.sigma_ceil)

            offset[yy, xx, 0] = cy - ym[yy, xx]  # dy
            offset[yy, xx, 1] = cx - xm[yy, xx]  # dx
            # ch0 pairs with dx^2 in the loss (offset flip), so ch0 = s_x
            sigma[yy, xx, 0] = s_x
            sigma[yy, xx, 1] = s_y
            seed[yy, xx, 0] = 1.0

            instance[yy, xx, 0] = k
            center[int(round(yy.mean())), int(round(xx.mean())), 0] = 1.0

        return {
            "offset": offset,
            "sigma": sigma,
            "seed": seed,
            "instance": instance,
            "center": center,
        }

    def decode_preds(self, preds: dict[str, torch.Tensor], cfg: dict | None = None) -> np.ndarray:
        """Greedy seediness clustering over the fg subset.

        preds are (C, H, W) torch tensors in framework channel order
        (offset ch 0 = dy, ch 1 = dx); the vendored Cluster works in
        EmbedSeg's (x, y) order, hence the channel swap at the boundary.
        Grid spacing 1/train_patch matches encode/loss; the extent-absolute
        origin is invisible (clustering only uses within-image coordinate
        differences), matching upstream expand_grid semantics
        (repos/EmbedSeg/EmbedSeg/test.py:1013-1020).
        """
        cfg = dict() if cfg is None else cfg
        offset = preds["offset"].flip(0)  # (dy, dx) -> (x=dx, y=dy)
        H, W = offset.shape[1:]
        # spacing 1/train_patch, matching encode/loss; extent-absolute origin
        # is invisible (clustering only uses within-image coordinate
        # differences), so this matches upstream expand_grid semantics
        # (repos/EmbedSeg/EmbedSeg/test.py:1013-1020)
        cluster = Cluster(grid_y=H, grid_x=W,
                          pixel_y=H / self.train_patch,
                          pixel_x=W / self.train_patch,
                          device=offset.device)
        # seed chan is logits but cluster does sigmoid internally
        labels = cluster.cluster(
            torch.cat([offset, preds["sigma"], preds["seed"]], 0),
            n_sigma=2,
            seed_thresh=cfg.get("seed_thresh", 0.9),
            fg_thresh=cfg.get("fg_thresh", 0.5),
            min_mask_sum=cfg.get("min_mask_sum", 0),
            min_unclustered_sum=cfg.get("min_unclustered_sum", 0),
            min_object_size=cfg.get("min_object_size", 10),
        )
        return labels.numpy().astype(np.uint32)

    def transform(
        self, maps: dict[str, np.ndarray], t: OrthoTransform
    ) -> dict[str, np.ndarray]:
        """D4-exact: offset vectors rotate/flip component-wise; per-axis
        sigmas (ch0 = s_x, ch1 = s_y) keep their axis under flips and swap
        under rot90, like the squared distances they normalize; remaining
        channels are scalar maps."""
        f = maps["offset"].transpose(2, 0, 1)  # (2, H, W) = (dy, dx)
        s = maps["sigma"].transpose(2, 0, 1)   # (2, H, W) = (s_x, s_y)
        if t.flip_y:
            f = np.stack([-f[0, ::-1], f[1, ::-1]])
            s = np.stack([s[0, ::-1], s[1, ::-1]])
        if t.flip_x:
            f = np.stack([f[0, :, ::-1], -f[1, :, ::-1]])
            s = np.stack([s[0, :, ::-1], s[1, :, ::-1]])
        if t.rot90:
            # spatial rot90(k=1) is a visual CCW rotation: +x -> -y in array
            # coords, i.e. (dy, dx) -> (-dx, dy); sigmas swap axis pairing
            f = np.rot90(f, k=t.rot90, axes=(1, 2))
            s = np.rot90(s, k=t.rot90, axes=(1, 2))
            dy, dx = f[0].copy(), f[1].copy()
            sx, sy = s[0].copy(), s[1].copy()
            for _ in range(t.rot90):
                dy, dx = -dx, dy
                sx, sy = sy, sx
            f = np.stack([dy, dx])
            s = np.stack([sx, sy])
        out = {"offset": f.transpose(1, 2, 0), "sigma": s.transpose(1, 2, 0)}
        for k in ("seed", "instance", "center"):
            out[k] = t.apply_to_image(maps[k])
        return out

    def transform_affine(
        self,
        maps: dict[str, np.ndarray],
        t: AffineTransformV2,
        t_label: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Re-encode offset/sigma/center from the nearest-warped instance ids.

        The pre-encoded offset field only warps bilinearly-exact in an
        instance's interior: at the boundary ring the sample blends with
        background (0) or a neighbour's offsets, and since border pixels
        carry the largest d^2, the per-instance reduction reads a mean d^2
        that is systematically too small — sigma (s = ln(ln2/d^2)) inflates
        (median +16% over random affines, up to 2x for tiny instances).
        The warped instance map preserves ids exactly (nearest), so
        offset/sigma/center are re-encoded from it with the encode math,
        vectorized via bincounts over the instance ids — identical results
        to encode_targets on the warped label. t_label is unused: the
        instance channel carries the same ids.
        """
        instance, = t.apply_to_images(images=[maps["instance"]], bilinear=[False])
        inst = instance[..., 0]
        H, W = inst.shape
        scale = 1.0 / self.train_patch

        offset = np.zeros((H, W, 2), np.float32)
        sigma = np.zeros_like(maps["sigma"])
        center = np.zeros_like(maps["center"])
        seed = (inst > 0)[..., None].astype(maps["seed"].dtype)

        flat = np.flatnonzero(inst)
        if flat.size:
            ids = inst.reshape(-1)[flat].astype(np.int64)
            n_inst = int(ids.max()) + 1
            ys, xs = np.divmod(flat, W)
            n = np.bincount(ids, minlength=n_inst).astype(np.float64)
            # centroid in 1/train_patch units, per-axis d^2 to it (bincount
            # leaves 0/0 = nan at id gaps, never read back)
            with np.errstate(invalid="ignore", divide="ignore"):
                cy = np.bincount(ids, weights=ys, minlength=n_inst) / n * scale
                cx = np.bincount(ids, weights=xs, minlength=n_inst) / n * scale
            dy2 = np.divide(
                np.bincount(ids, weights=(cy[ids] - ys * scale) ** 2,
                            minlength=n_inst), n,
                out=np.zeros(n_inst), where=n > 0)
            dx2 = np.divide(
                np.bincount(ids, weights=(cx[ids] - xs * scale) ** 2,
                            minlength=n_inst), n,
                out=np.zeros(n_inst), where=n > 0)
            with np.errstate(invalid="ignore", divide="ignore"):
                s_y = np.minimum(
                    np.log(np.log(2.0) / dy2) / 10.0, self.sigma_ceil)
                s_x = np.minimum(
                    np.log(np.log(2.0) / dx2) / 10.0, self.sigma_ceil)
            offset[ys, xs, 0] = (cy[ids] - ys * scale).astype(np.float32)
            offset[ys, xs, 1] = (cx[ids] - xs * scale).astype(np.float32)
            sig = sigma.reshape(-1, 2)
            sig[flat, 1] = s_y[ids]
            sig[flat, 0] = s_x[ids]
            # center spikes at the rounded per-instance centroid (present ids
            # only: bincount leaves 0/0 = nan at id gaps)
            present = n > 0
            yi = np.rint(cy[present] / scale).astype(int)
            xi = np.rint(cx[present] / scale).astype(int)
            ok = (yi >= 0) & (yi < H) & (xi >= 0) & (xi < W)
            center[yi[ok], xi[ok], 0] = 1.0

        return {
            "offset": offset,
            "sigma": sigma,
            "seed": seed,
            "instance": instance,
            "center": center,
        }

    def loss(
        self,
        preds: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Thin wrapper over the vendored SpatialEmbLoss.

        Assembles the reference-layout prediction [offset(2), sigma(2),
        seed(1)] from the channel dicts (no activation — the loss applies
        tanh/sigmoid internally) and passes the loss-only target channels
        (instance ids, center image).

        The criterion's xym buffer is rebuilt (cached) whenever the input
        shape differs: the buffer is (2, H, W) with spacing 1/train_patch, so
        the loss is identical at patch and full-image extent — origin shift is
        invisible (the loss only uses within-image coordinate differences).
        The network emits (dy, dx); the vendored loss pairs prediction[0:2]
        with xym ch0=x, ch1=y, hence the flip, matching decode_preds.
        """
        # σ̂ clamped at the encode-side ceiling: exp(10·σ̂) overflows to inf
        # above σ̂ ≈ 8.87, and dist = exp(-inf) = 0 keeps the loss finite
        # while backward yields inf·0 = NaN (which corrupts the backbone);
        # targets never exceed sigma_ceil, so σ̂ above it is meaningless
        prediction = torch.cat([
            preds["offset"].flip(1),
            preds["sigma"].clamp(max=self.sigma_ceil),
            preds["seed"]], 1)
        H, W = prediction.shape[2:]
        if self._criterion_shape != (H, W):
            xym = SpatialEmbLoss(grid_y=H, grid_x=W,
                                 pixel_y=H / self.train_patch,
                                 pixel_x=W / self.train_patch).xym
            self.criterion.xym = xym.to(self.criterion.xym)
            self._criterion_shape = (H, W)
        instance = targets["instance"][:, 0]
        loss = self.criterion(
            prediction,
            instances=instance,
            labels=(instance > 0).to(instance.dtype),
            center_images=targets["center"][:, 0].bool(),
        )
        return {"loss": loss}

    def display_encoded_plt(self, encoded, axes, **kwargs):
        from matplotlib.patches import Ellipse
        axes[0].imshow(dx_to_circ(encoded["offset"].transpose((2, 0, 1))))
        axes[0].set_title("Offset (dy, dx)")
        ax = axes[1]
        sigma = encoded["sigma"]
        ax.imshow(encoded["seed"][..., 0], cmap="gray", vmin=0, vmax=1)
        # sigma ellipse, vendor convention (utils.prepare_embedding_for_train_image):
        # radius = sqrt(0.5/s) in pixels (xym spacing 1/train_patch);
        # ch0 pairs with x, ch1 with y
        centers = np.argwhere(encoded["center"][..., 0] > 0.5)
        colors = get_glasbey_colors(len(centers) + 1)[1:]  # [0] is black
        # a spike can land on background (rounded centroid of a concave
        # instance): read sigma at the nearest instance pixel instead
        fg = encoded["seed"][..., 0] > 0.5
        inds = np.stack(
            np.nonzero(np.asarray(fg)), axis=-1) if fg.any() else np.zeros((0, 2), int)
        for c, (y, x) in enumerate(centers):
            if fg[y, x]:
                yy, xx = int(y), int(x)
            elif len(inds):
                d2 = ((inds - (y, x)) ** 2).sum(-1)
                yy, xx = inds[int(np.argmin(d2))]
            else:
                continue
            s_x, s_y = np.exp(10 * sigma[yy, xx])
            a_x = np.sqrt(0.5 / s_x) * self.train_patch
            a_y = np.sqrt(0.5 / s_y) * self.train_patch
            ax.add_patch(Ellipse((x, y), 2 * a_x, 2 * a_y, fill=False,
                                 color=colors[c], lw=2))
        ax.set_title("Seed (fg) + 0.5 ellipse")

    def display_preds_plt(self, preds, axes, **kwargs):

        offset = tensor_to_numpy(preds['offset'])
        axes[0].imshow(dx_to_circ(offset.transpose((2, 0, 1))))
        axes[0].set_title("Offset (dy, dx)")

        seed = tensor_to_numpy(torch.sigmoid(preds['seed']))
        axes[1].imshow(seed[..., 0], cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Seed (fg)")

    def encoded_to_perfect_pred(
        self, encoded: dict[str, np.ndarray]
    ) -> dict[str, torch.Tensor]:
        """Base conversion with one override: the model computes
        tanh(offset_pred) + xym, so a perfect offset target c_k - xym must be
        mapped through the inverse tanh before feeding decode_preds()/loss().
        With absolute coords / train_patch, |offset| <= object extent /
        train_patch, so the clamp is a safety net only (tanh-safe unless an
        object exceeds the training patch)."""
        out = dict()
        for k, c in self.channels.items():
            t = torch.from_numpy(encoded[k]).to(
                c.torch_type).permute((2, 0, 1))
            if k == "offset":
                t = torch.atanh(t.clamp(-(1 - 1e-4), 1 - 1e-4))
            elif c.logits:
                t = torch.logit(t.clamp(1e-4, 1 - 1e-4))
            out[k] = t
        return out
