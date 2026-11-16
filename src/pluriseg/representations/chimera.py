"""Chimera representation: all representations trained jointly on one shared trunk.

One conv stack per sub-representation (GroupedConvHead) reads the shared
backbone features and emits that sub-rep's channels, prefixed with the
sub-rep's `name` (e.g. `stardist_radii`) so that same-named channels
(cellpose/omnipose `flow`) never collide. Encoding, transforms and losses
are delegated to the sub-representations; decoding picks one
sub-representation head (see decode_preds).
"""

import torch.nn as nn

from pluriseg.representations.base import ChannelProps, Representation
from pluriseg.representations.cellpose import CellposeRep
from pluriseg.representations.embed import EmbedsegRep
from pluriseg.representations.mask import MaskRep
from pluriseg.representations.omnipose import OmniposeRep
from pluriseg.representations.stardist import StarDistRep
from pluriseg.transforms import AffineTransformV2, OrthoTransform

_SUB_CLASSES = {
    MaskRep.name: MaskRep,
    CellposeRep.name: CellposeRep,
    OmniposeRep.name: OmniposeRep,
    StarDistRep.name: StarDistRep,
    EmbedsegRep.name: EmbedsegRep,
}


class ChimeraRep(Representation):
    """Composite representation over all sub-reps in `_SUB_CLASSES`.

    sub_params: per-sub-rep constructor kwargs, e.g.
        {"stardist": {"n_rays": 32}, "embedseg": {"train_patch": 256}}
    loss_weights: optional per-sub-rep weight on its total loss (default 1).

    Loss components are reported as "loss_{sub}_{key}" (the sub-total loss as
    "loss_{sub}_loss"), so the Lightning logs read e.g.
    train_loss_cellpose_flow; the combined scalar is "loss". Every key follows
    the base contract: "loss" or "loss_*".
    """

    name = "chimera"
    version = "1"
    cacheable = True  # every sub-rep is cacheable; encoded maps are prefixed unions
    # one head stack per sub-rep; names computed in __init__ once channels exist
    channel_groups: dict[str, list[str]] | None = None

    def __init__(
            self, 
            sub_params: dict[str, dict] | None = None,
            loss_weights: dict[str, float] | None = None,
            default_head:str = MaskRep.name
        ):
        super().__init__()
        self.default_head = default_head
        self.subs = nn.ModuleDict({
            cname: cls(**(sub_params or {}).get(cname, {}))
            for cname, cls in _SUB_CLASSES.items()
        })
        self.loss_weights = loss_weights or {}

        # prefixed channel spec: per-channel props copied unchanged from the
        # sub-rep so ChannelRole/logits semantics survive the rename
        self.channels = {
            f"{sname}_{k}": ChannelProps(v.role, v.channels, v.torch_type, v.logits)
            for sname, sub in self.subs.items()
            for k, v in sub.channels.items()
        }
        # one group per sub-rep -> GroupedConvHead gives each its own stack
        self.channel_groups = {
            sname: [f"{sname}_{k}" for k in sub.channels]
            for sname, sub in self.subs.items()
        }
        self.plt_encoded_axes = sum(
            s.plt_encoded_axes for s in self.subs.values())
        self.plt_pred_axes = sum(s.plt_pred_axes for s in self.subs.values())

    # ---- prefix helpers -------------------------------------------------
    @staticmethod
    def _prefixed(sname: str, maps: dict) -> dict:
        return {f"{sname}_{k}": v for k, v in maps.items()}

    @staticmethod
    def _unprefixed(sname: str, maps: dict) -> dict:
        # sub names are mutually non-prefixing, so startswith is unambiguous
        p = sname + "_"
        return {k[len(p):]: v for k, v in maps.items() if k.startswith(p)}

    # ---- representation API ---------------------------------------------
    def _cache_params(self) -> dict:
        # nested: a change to any sub-rep param must invalidate the cache
        return {
            f"{sname}.{k}": v
            for sname, sub in self.subs.items()
            for k, v in sub._cache_params().items()
        }

    def encode_targets(self, label) -> dict:
        out = {}
        for sname, sub in self.subs.items():
            out.update(self._prefixed(sname, sub.encode_targets(label)))
        return out

    def transform(self, maps: dict, t: OrthoTransform) -> dict:
        out = {}
        for sname, sub in self.subs.items():
            out.update(self._prefixed(
                sname, sub.transform(self._unprefixed(sname, maps), t)))
        return out

    def transform_affine(self, maps: dict, t: AffineTransformV2,
                         t_label) -> dict:
        out = {}
        for sname, sub in self.subs.items():
            out.update(self._prefixed(
                sname, sub.transform_affine(
                    self._unprefixed(sname, maps), t, t_label)))
        return out

    def loss(self, preds: dict, targets: dict) -> dict:
        comps, total = {}, None
        for sname, sub in self.subs.items():
            w = self.loss_weights.get(sname, 1.0)
            sub_loss = sub.loss(self._unprefixed(sname, preds),
                                self._unprefixed(sname, targets))
            for k, v in sub_loss.items():
                # every key satisfies the base contract ("loss" or "loss_*"):
                # sub components are "loss_{sub}_{key}" (the sub's own total
                # included, as "loss_{sub}_loss") — "loss" is the combined
                comps[f"loss_{sname}_{k}"] = v.detach()
            total = w * \
                sub_loss["loss"] if total is None else total + \
                w * sub_loss["loss"]
        comps["loss"] = total
        return comps

    def encoded_to_perfect_pred(self, encoded: dict) -> dict:
        # delegate so sub-rep-specific inversions (e.g. embed offset -> atanh)
        # are applied, then re-prefix
        out = {}
        for sname, sub in self.subs.items():
            out.update(self._prefixed(
                sname, sub.encoded_to_perfect_pred(self._unprefixed(sname, encoded))))
        return out

    def decode_preds(self, preds: dict, cfg: dict | None = None):
        """Decode merged predictions into an instance label map.

        cfg["decode"] (default "head"):
            "head": single-head decode; cfg["decode_head"] names the sub-rep
                (e.g. "stardist"), defaulting to self.default_head; remaining
                keys are forwarded to it.
        """
        cfg = dict() if cfg is None else cfg
        mode = cfg.get("decode", "head")
        if mode == "head":
            sname = cfg.get("decode_head",self.default_head)
            if sname not in self.subs:
                raise ValueError(
                    f"unknown decode_head {sname!r}; expected one of {sorted(self.subs)}")
            sub_cfg = {k: v for k, v in cfg.items()
                       if k not in ("decode", "decode_head")}
            return self.subs[sname].decode_preds(
                self._unprefixed(sname, preds), sub_cfg)
        raise ValueError(f"unknown decode mode {mode!r}; expected 'head'")

    # ---- display ---------------------------------------------------------
    def _display(self, maps: dict, axes, which: str, **kwargs) -> None:
        """which: 'encoded' or 'preds'; axes sliced per sub-rep's declared count."""
        i = 0
        for sname, sub in self.subs.items():
            n = getattr(sub, f"plt_{which}_axes")
            fn = sub.display_encoded_plt if which == "encoded" else sub.display_preds_plt
            fn(self._unprefixed(sname, maps), axes[i:i + n], **kwargs)
            i += n

    def display_encoded_plt(self, encoded: dict, axes, **kwargs) -> None:
        self._display(encoded, axes, "encoded", **kwargs)

    def display_preds_plt(self, preds: dict, axes, **kwargs) -> None:
        self._display(preds, axes, "preds", **kwargs)
