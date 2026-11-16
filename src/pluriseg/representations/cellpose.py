"""Cellpose-style flow representation: flow field + cell prob + boundary.

Stub: signatures only. Will also cover the Omnipose "flow+inside" variant
later (P2) by adding an extra channel.
"""

import logging

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn

from pluriseg.display import dx_to_circ
from pluriseg.utils import relabel_contiguous, tensor_to_numpy
from pluriseg.representations.base import ChannelProps, Representation
from pluriseg.transforms import AffineTransformV2, OrthoTransform
from pluriseg.vendor.cellpose.dynamics import compute_masks, masks_to_flows_gpu

dynamics_logger = logging.getLogger(__name__)


class CellposeRep(Representation):
    name = "cellpose"
    channels = {
        "flow": ChannelProps("target", 2),
        "cellprob": ChannelProps("target", 1, logits=True)
    }
    cacheable = True  # flow vectors rotate component-wise under D4 transforms
    plt_encoded_axes = 2
    plt_pred_axes = 2

    def __init__(self):
        super().__init__()

        self.criterion_flow = nn.MSELoss(reduction="mean")
        self.criterion_cp = nn.BCEWithLogitsLoss(reduction="mean")

    def _cache_params(self) -> dict:
        return {}

    def encode_targets(self, label: np.ndarray) -> dict[str, np.ndarray]:

        # cellpose vendor code assumes ids are 1..N (get_centers indexes slices
        # by i+1); relabel so non-contiguous inputs stay correct
        label = relabel_contiguous(label)

        flows, _ = masks_to_flows_gpu(
            label.astype(int),
            niter=None  # TODO some other value ?
        )
        fg = label > 0

        flows *= 5.
        # we apply flow sclaling at encode_targets to maintain consistency accross encoding and target

        return {
            # maintain (H,W,C) convention for numpy
            'flow': flows.transpose((1, 2, 0)),
            'cellprob': fg[..., None]
        }

    def decode_preds(self, preds: dict[str, torch.Tensor], cfg: dict | None = None) -> np.ndarray:

        # numpy doesn't support bfloat16 (precision=bf16-mixed), cast first
        dP = preds['flow'].float().cpu().numpy()
        # stay in (2,H,W) format, keep flow scaling: compute_masks divides by 5
        # (cellpose convention — network outputs are pre-scaled flows)

        # raw logits: compute_masks applies cellprob_threshold=0 to logits
        # (fg = logits > 0 = prob > 0.5); squashing to [0,1] first would make
        # threshold 0 select every pixel (whole-image integration)
        cellprob_logits = preds['cellprob'].float().cpu().numpy()[0]  # (H,W)

        cfg = dict() if cfg is None else cfg

        labels = compute_masks(
            dP,
            cellprob_logits,
            niter=cfg.get('niter', 200),
            cellprob_threshold=cfg.get('cellprob_threshold', 0),
            flow_threshold=cfg.get('flow_threshold', 0.6),
            do_3D=False,
            max_size_fraction=cfg.get('max_size_fraction', 0.8),
            device=torch.device("cpu")
        )

        return labels.astype(np.uint32)

    def transform(
        self, maps: dict[str, np.ndarray], t: OrthoTransform
    ) -> dict[str, np.ndarray]:
        # flow is (H, W, 2) = (dy, dx); convert to cellpose's (2, H, W)
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
        return {
            "flow": f.transpose(1, 2, 0),
            "cellprob": t.apply_to_image(maps["cellprob"]),
        }

    def transform_affine(
        self,
        maps: dict[str, np.ndarray],
        t: AffineTransformV2,
        t_label: np.ndarray,
    ) -> dict[str, np.ndarray]:
        # flow vectors (dy, dx): warp bilinear, then rotate by the linear
        # part and undo the scale on the magnitude (encode emits unit-norm x5);
        # cellprob is nearest-neighbor so labels stay integer. t_label unused:
        # flow and cellprob both warp in place.
        f, cp = t.apply_to_images(
            images=[maps["flow"], maps["cellprob"]],
            bilinear=[True, False]
        )
        a, b = t.matrix[0, :2]  # y' = a*y + b*x
        d, e = t.matrix[1, :2]  # x' = d*y + e*x
        f = np.stack([a * f[..., 0] + b * f[..., 1],
                      d * f[..., 0] + e * f[..., 1]], axis=-1) / t.scale
        return {
            "flow": f,
            "cellprob": cp,
        }

    def loss(self, preds, targets):
        """
        DOCSTRING

        Cellpose does:

        criterion = nn.MSELoss(reduction="mean")
        criterion2 = nn.BCEWithLogitsLoss(reduction="mean")
        veci = 5. * lbl[:, -2:]
        loss = criterion(y[:, -3:-1], veci)
        loss /= 2.
        loss2 = criterion2(y[:, -1], (lbl[:, -3] > 0.5).to(y.dtype))
        loss = loss + loss2

        https://github.com/MouseLand/cellpose/blob/a54cb48849b7e225a81e8e43dcb042d42427f543/cellpose/train.py#L33

        """

        veci = targets['flow']  # scaling done at encode
        loss_flow = self.criterion_flow(preds['flow'], veci)
        loss_cp = self.criterion_cp(preds['cellprob'], targets['cellprob'])
        loss = loss_cp + loss_flow / 2.0

        return {
            'loss': loss,
            'loss_flow': loss_flow,
            'loss_cp': loss_cp
        }

    def _display_plt(
        self,
        cellprob: np.ndarray,
        flow: np.ndarray,
        axes: list[plt.Axes]
    ):
        axes[0].imshow(cellprob, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('Cellprob')
        flow_rgb = dx_to_circ(flow.transpose((2, 0, 1)) / 5.)
        axes[1].imshow(flow_rgb)
        axes[1].set_title('Flow')

    def display_encoded_plt(self, encoded, axes, **kwargs):
        self._display_plt(
            encoded['cellprob'][..., 0],
            encoded['flow'],
            axes
        )

    def display_preds_plt(self, preds, axes, **kwargs):
        self._display_plt(
            tensor_to_numpy(torch.sigmoid(preds['cellprob']))[..., 0],
            tensor_to_numpy(preds['flow']),
            axes
        )
