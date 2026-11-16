"""Representation meta-class: the single source of truth linking heads, targets,
losses, decoding and augmentation.

One Representation instance is built in train.py and shared by the datamodule
(encode_targets / transform) and the Lightning model (channels / loss / decode_preds), so the
two sides can never desync.
"""

import hashlib
from abc import ABC, abstractmethod
from enum import Enum

import pydantic.dataclasses
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
import torch

from pluriseg.transforms import AffineTransformV2, OrthoTransform


class ChannelRole(Enum):
    """What happens to an encoded channel.

    Predicted channels (TARGET, INDIRECT) drive the head's conv stack;
    LOSS channels never reach the network — they are encode-derived
    bookkeeping consumed only by loss().
    """

    TARGET = "target"
    """Predicted and directly supervised: encode_targets() provides the GT target the
    loss minimizes toward (e.g. cellpose "flow", embed "offset", "sigma")."""

    INDIRECT = "indirect"
    """Predicted without a direct GT: the loss still optimizes it, but through
    its dependence on the other channels (target = f(predictions)), e.g. embed
    "seed" regressed onto the loss's own dist.detach."""

    LOSS = "loss"
    """Not predicted, not supervised: encode-derived map consumed by loss() only
    (e.g. embed "instance"/"center", stardist "dist_mask"). Decoded instances
    come from the predicted channels alone."""


@pydantic.dataclasses.dataclass(config=dict(arbitrary_types_allowed=True))
class ChannelProps:
    """Per-channel output spec. role decides what happens to the channel; see
    ChannelRole. role accepts a ChannelRole or its value string (coerced,
    so ChannelProps("target", 1) == ChannelProps(ChannelRole.TARGET, 1));
    anything else fails loudly at construction."""
    role: ChannelRole
    channels: int
    torch_type: torch.dtype = torch.float
    # Contract: a loss is minimized exactly when preds[c] == targets[c].
    # True when the network emits *logits* for this channel while encode_targets()
    # returns the *probabilities/labels*; the loss then applies the inverse
    # activation internally (e.g. BCEWithLogits). This is what lets
    # encoded_to_perfect_pred map targets -> valid "perfect" preds.
    logits: bool = False


class Representation(ABC, nn.Module):
    """Maps instance label maps <-> network output maps.

    Subclasses nn.Module so any submodules (e.g. loss criteria with
    buffers/parameters, EmbedSeg's SpatialEmbLoss) follow .to(device) on the
    owning LightningModule.
        name: registry key (e.g. "stardist").
        channels: output spec, name -> ChannelProps(role, channels,
            torch_type, logits). Only predicted roles (TARGET, INDIRECT)
            drive the head's conv stack; LOSS channels are loss plumbing
            (e.g. mask's "distance_weights", embed's "instance"/"center").
        cacheable: whether per-image targets can be precomputed on disk and
            geometrically augmented in place via `transform` (True), or must
            be re-encoded from the instance label at patch time (False).
        plt_encoded_axes: number of pyplot axes display_encoded_plt() needs.
        plt_pred_axes: number of pyplot axes display_preds_plt() needs.
        version: bump when encode_targets() output semantics change — it feeds
            get_hash(), which tags the on-disk target cache dir.
    """

    name: str
    channels: dict[str, ChannelProps]  # channels
    # {group_name: [channel names]}: predicted channels to give their own
    # conv stack in GroupedConvHead. None (default) = one stack for all
    # channels. Only needs to be set when a rep wants per-group stacks
    # (e.g. ChimeraRep, one stack per sub-rep).
    channel_groups: dict[str, list[str]] | None = None
    cacheable: bool
    plt_encoded_axes: int
    plt_pred_axes: int
    version: str = "1"

    @abstractmethod
    def _cache_params(self) -> dict:
        """Constructor params that change encode_targets() output, so changing
        them must invalidate the on-disk cache (empty dict if none). Consumed
        by get_hash()."""

    def init_head(self, head: nn.Module) -> None:
        """Optional hook to alter the conv head after construction (e.g.
        output-layer init). Default: no-op. `head` is the ConvHead (or
        GroupedConvHead) producing this rep's channels; non-LOSS channels
        appear in its output in `self.channels` order."""
        return

    def get_hash(self) -> str:
        """Stable short hash tagging the on-disk target cache.

        Covers class identity, the manual `version` bump and _cache_params();
        change any of those and the cache dir changes instead of serving
        stale entries.
        """
        payload = repr(
            (type(self).__name__, self.version, self._cache_params()))
        return hashlib.md5(payload.encode()).hexdigest()[:10]

    def _channels_repr(self) -> str:
        width = max(len(k) for k in self.channels)
        return "\n".join(
            f"  {k:<{width}}  {role:<8}  {c.channels}ch"
            + ("  (logits)" if c.logits else "")
            for k, c in self.channels.items()
            for role in (getattr(c.role, "value", c.role),)
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__} (name={self.name!r}, version={self.version})\n{self._channels_repr()}"

    def __rich_repr__(self) -> tuple:
        """rich-style repr; picked up by `rich.print` when rich is installed."""
        yield "name", self.name
        yield "version", self.version
        for k, c in self.channels.items():
            d = {"role": getattr(c.role, "value", c.role),
                 "channels": c.channels}
            if c.logits:
                d["logits"] = True
            yield k, d

    @abstractmethod
    def encode_targets(self, label: np.ndarray) -> dict[str, np.ndarray]:
        """Instance label map -> target maps.

        Args:
            label: (H, W) integer instance label map (0 = background).
        Returns:
            dict of name -> array; each entry's shape matches self.channels
            ((H, W, C) for C channels, C may be one).
        """

    @abstractmethod
    def decode_preds(self, preds: dict[str, torch.Tensor], cfg: dict | None = None) -> np.ndarray:
        """Predicted output maps of one image -> instance label map.

        Conventions: torch tensors are (C,H,W) (no batch), numpy is (H,W,C).
        Returns uint32 (H, W), 0 = background. Post-processing (NMS, flow
        dynamics, clustering, watershed) lives here or in postproc/ helpers
        it calls.
        """

    @abstractmethod
    def transform(
        self, maps: dict[str, np.ndarray], t: OrthoTransform
    ) -> dict[str, np.ndarray]:
        """Apply a geometric transform to already-encoded target maps.

        Implementations must be exact for the D4 group (flips + 90 deg
        rotations); raising NotImplementedError for anything else is the
        contract, not a bug. Subclasses override for representation-specific
        handling (e.g. flow vectors rotate component-wise; radii are
        invariant to flips/rotations).
        """

    @abstractmethod
    def transform_affine(
        self,
        maps: dict[str, np.ndarray],
        t: AffineTransformV2,
        t_label: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Apply a geometric transform to already-encoded target maps.

        t_label is the (H, W) instance label warped with the same transform
        (order=0); implementations that cannot warp some maps in place
        (position-dependent channels, recompute-dependent values) fall back
        to re-encoding from it.
        """

    @abstractmethod
    def loss(
        self,
        preds: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Preds vs targets -> dict of loss components including "loss"."""

    @abstractmethod
    def display_encoded_plt(self, encoded: dict[str, np.ndarray], axes: list[plt.Axes], **kwargs) -> None:
        """Render encoded representation maps in pyplot fig
        """

    @abstractmethod
    def display_preds_plt(self, preds: dict[str, torch.Tensor], axes: list[plt.Axes], **kwargs) -> None:
        """Render predicted representation maps in pyplot fig
        tensor (C,H,W) shape
        """

    def encoded_to_perfect_pred(
        self,
        encoded: dict[str, np.ndarray]
    ) -> dict[str, torch.Tensor]:
        """Convert encoded target maps to a "perfect prediction" for self.

        Useful for sanity checks / debugging decode_preds and loss: the model's output
        space is fed the ground-truth encoding itself. Channels flagged
        `logits=True` are mapped through an inverse sigmoid (clamped, so finite);
        others pass through unchanged. Subclasses override when a target needs an
        activation inverse beyond `logits` (e.g. embed's offset -> atanh).

        Args:
            encoded: output of `self.encode_targets`; each key must exist in self.channels.
        Returns:
            dict name -> (C, H, W) tensor (torch convention, no batch), cast to
            that channel's torch_type; ready to feed decode_preds() directly.
        """
        out = dict()
        for k, c in self.channels.items():
            t = torch.from_numpy(encoded[k]).to(
                c.torch_type).permute((2, 0, 1,))
            if c.logits:
                # clamp so logit stays finite at p = 0 / 1
                t = torch.logit(t.clamp(1e-4, 1 - 1e-4))
            out[k] = t

        return out
