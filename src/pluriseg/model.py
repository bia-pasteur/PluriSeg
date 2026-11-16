"""Lightning module: shared backbone + representation-driven head + shared rep losses."""

import warnings

import lightning as L
import numpy as np
from rich.panel import Panel
import torch
from pathlib import Path
from rich import print as rprint

from pluriseg.configs import (BackboneConfig, GateConfig, OptimizerConfig,
                            RepresentationConfig)
from pluriseg.eval.metrics import instance_ap, matched_mean_iou, panoptic_quality
from pluriseg.models.backbone import build_backbone
from pluriseg.models.gate import GatedChimera
from pluriseg.models.heads import ConvHead, GroupedConvHead
from pluriseg.representations import build_representation


def check_loss_finite(
    losses: dict[str, torch.Tensor],
    batch: dict,
    phase: str,
    debug: bool = False,
) -> None:
    """Sanity-check loss values; warn (or raise in debug mode) on NaN/Inf.

    Prints the loss dict (non-finite values in bold red), the offending batch
    item's file sources, and shape/range/non-finite counts for image / targets /
    label.

    Args:
        losses: loss dict as returned by a Representation.loss().
        batch: the dataloader batch (needs "image", "targets", "label",
            "image_source", "label_source").
        phase: "train" or "val"/"test", for the message.
        debug: raise ValueError instead of warning.
    """
    def fmt(name: str, t: torch.Tensor) -> str:
        """One line: name shape [min, max] n_bad, non-finite count in red."""
        t = t.detach().float()
        n_bad = int((~torch.isfinite(t)).sum())
        finite = t[torch.isfinite(t)]
        lo = f"{finite.min().item():.4g}" if finite.numel() else "—"
        hi = f"{finite.max().item():.4g}" if finite.numel() else "—"
        bad_s = f"[bold red]{n_bad}[/]" if n_bad else "0"
        return f"  {name}: {tuple(t.shape)} [{lo}, {hi}] bad={bad_s}"

    bad = {k: v for k, v in losses.items()
           if isinstance(v, torch.Tensor) and not torch.isfinite(v).all()}
    if not bad:
        return

    def fmt_loss(k: str, v: torch.Tensor) -> str:
        x = v.item() if v.numel(
        ) == 1 else f"{int((~torch.isfinite(v)).sum())}/{v.numel()} nonfinite"
        s = f"{x:.4g}" if isinstance(x, float) else str(x)
        return f"  {k}: [bold red]{s}[/]" if k in bad else f"  {k}: {s}"

    lines = (
        [f"non-finite: [bold red]{', '.join(sorted(bad))}[/]", "losses:"]
        + [fmt_loss(k, v) for k, v in losses.items()]
        + ["batch:"]
        + [fmt("image", batch["image"])]
        + [fmt(f"target/{k}", t) for k, t in batch["targets"].items()]
        + [fmt("label", batch["label"])]
        + ["sources:"]
        + [f"  {i}: {i_s}  <-  {l_s}" for i, (i_s, l_s) in
           enumerate(zip(batch["image_source"], batch["label_source"]))]
    )

    rprint(Panel("\n".join(lines),
                 title=f"[bold red]NaN/Inf {phase} losses[/]",
                 border_style="red"))
    if debug:
        raise ValueError(
            f"non-finite {phase} losses: {', '.join(sorted(bad))}")
    rprint("[bold yellow]Training continues — the NaN loss will propagate to "
           "the optimizer (debug mode raises instead).[/]")



class UnifedSegmentationModel(L.LightningModule):
    """Backbone + conv head, with loss/decode_preds supplied by the Representation.

    The model builds its own Representation from `rep_cfg`; the datamodule
    builds an equivalent instance from the same config. Keeping only the
    pydantic config in `__init__` means a checkpoint alone suffices to
    reconstruct the model (`load_from_checkpoint`), with channels, losses
    and decoding guaranteed consistent by construction.

    When the representation is a ChimeraRep and `gate_cfg.enabled`, a small
    GateRouter on the pooled backbone features picks one sub-representation
    head per image at decode time.
    """

    def __init__(
        self,
        rep_cfg: RepresentationConfig,
        backbone_cfg: BackboneConfig,
        optimizer_cfg: OptimizerConfig,
        gate_cfg: GateConfig | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()  # all configs serialize
        self.rep = build_representation(rep_cfg)
        self.backbone, backbone_out_ch = build_backbone(backbone_cfg)

        # Representations may declare channel_groups: predicted channels that
        # each get their own conv stack on the shared backbone features
        # (currently only ChimeraRep, one stack per sub-representation).
        # Everything else keeps the classic single ConvHead.
        groups = getattr(self.rep, "channel_groups", None)
        if groups is not None:
            self.head = GroupedConvHead(
                in_ch=backbone_out_ch,
                channels=self.rep.channels,
                groups=groups,
            )
        else:
            self.head = ConvHead(
                in_ch=backbone_out_ch,
                channels=self.rep.channels,
            )
        self.rep.init_head(self.head)
        self.optimizer_cfg = optimizer_cfg
        self.debug = False  # set by train.py from TrainConfig.debug

        head_names = list(getattr(self.rep, "subs", {}))
        if gate_cfg and gate_cfg.enabled and not head_names:
            raise ValueError(
                f"gate.enabled requires rep.name='chimera' (per-head decodes); "
                f"got rep.name={rep_cfg.name!r}")
        self.use_gate = bool(gate_cfg and gate_cfg.enabled and head_names)
        if self.use_gate:
            self.gated = GatedChimera(self.rep, gate_cfg, backbone_out_ch)
        else:
            self.gated = None

    # ---------------------------------------------------------------- forward

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(image)
        return self.head(features)

    def _predict_logits(self, image: torch.Tensor):
        """Pad to /32, forward, crop channels to the original extent.

        Returns (features, preds); preds channels are (C, h, w). Images too
        large for the GPU fall back to a fp32 CPU forward (identical
        numerics, just slower).
        """
        h, w = image.shape[-2:]
        ph = (-h) % 32
        pw = (-w) % 32
        padded = torch.nn.functional.pad(
            image.unsqueeze(0), (0, pw, 0, ph))
        try:
            features = self.backbone(padded)
            preds = self.head(features)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            device = next(self.backbone.parameters()).device
            self.backbone.to("cpu"), self.head.to("cpu")
            features = self.backbone(padded.to("cpu"))
            preds = self.head(features)
            self.backbone.to(device), self.head.to(device)
        # crop channels back to the original extent so decode runs there
        return features, {k: v[..., :h, :w] for k, v in preds.items()}

    def _check_inference_mode(self, image: torch.Tensor) -> None:
        assert image.ndim == 3, f"batch inference not implemented yet, got {image.shape}, expected (C,H,W)"
        if self.training:
            warnings.warn(
                "UnifedSegmentationModel.predict_labels() called in train mode; "
                "call model.eval() first", stacklevel=3)

    def predict_labels(self, image: torch.Tensor) -> np.ndarray:
        """Model input (C, H, W) -> predicted instance labels (H, W) uint32.

        Expects the model in eval mode (call `model.eval()` first); runs a
        no-grad forward and decodes the prediction. With the gate active the
        gate picks one sub-representation head; its logits land in
        `last_gate`.
        """
        self._check_inference_mode(image)
        with torch.no_grad():
            features, preds = self._predict_logits(image)
            if self.gated:
                return self.gated.decode(features, preds)
        return self.rep.decode_preds({k: v[0] for k, v in preds.items()})

    def predict_labels_all_heads(self, image: torch.Tensor) -> dict[str, np.ndarray]:
        """One forward, every decode: {"": default decode} plus, when the
        representation has sub-representations (ChimeraRep), one entry per
        sub-head name decoded exactly as that rep decodes standalone.

        """
        self._check_inference_mode(image)
        with torch.no_grad():
            features, preds = self._predict_logits(image)
            preds_b = {k: v[0] for k, v in preds.items()}
            out: dict[str, np.ndarray] = {
                "": (self.gated.decode(features, preds) if self.gated
                     else self.rep.decode_preds(preds_b))}
            for sname in getattr(self.rep, "subs", {}):
                out[sname] = self.rep.decode_preds(
                    preds_b, cfg={"decode": "head", "decode_head": sname})
            return out

    # ---------------------------------------------------------------- training

    def training_step(self, batch, batch_idx):
        batch_size = batch["image"].shape[0]
        features = self.backbone(batch["image"])
        preds = self.head(features)
        losses = self.rep.loss(preds, batch["targets"])

        check_loss_finite(losses, batch, "train", debug=self.debug)
        self.log_dict(
            {f"train_{k}": v for k, v in losses.items() if k != 'loss'},
            batch_size=batch_size
        )
        self.log_dict(
            {'train_loss': losses['loss']},
            batch_size=batch_size, prog_bar=True
        )

        return losses["loss"]

    # ---------------------------------------------------------------- eval

    def _eval_step(self, batch, phase: str) -> None:
        """Shared validation/test logic: losses + instance AP metrics.

        Whole-image batches (test mode) are padded to /32 around the
        backbone; head outputs are cropped back to the original extent so
        losses and decodes run there (same convention as _predict_logits).
        """
        images = batch["image"]
        bsz, _, h, w = images.shape
        ph, pw = (-h) % 32, (-w) % 32
        if ph or pw:
            images = torch.nn.functional.pad(images, (0, pw, 0, ph))
        features = self.backbone(images)
        preds = self.head(features)
        preds = {k: v[..., :h, :w] for k, v in preds.items()}
        losses = self.rep.loss(preds, batch["targets"])
        check_loss_finite(
            losses, batch, f"val/test ({phase})", debug=self.debug)
        self.log_dict(
            {f"{phase}_{k}": v for k, v in losses.items() if k != 'loss'},
            batch_size=bsz
        )
        self.log_dict(
            {f'{phase}_loss': losses['loss']},
            batch_size=bsz, prog_bar=True
        )

        # decode_preds + AP per image, average over the batch, log once per
        # step. The datamodule stores original (h, w) per item; preds are
        # cropped back from the /32-padded size before decoding.
        aps = []
        mious = []
        pqs = []
        for b in range(bsz):
            label_np = batch["label"][b].detach(
            ).cpu().numpy().astype(np.uint32)
            preds_b = {
                # (C, H, W), decode_preds' convention
                k: v[b] for k, v in preds.items()
            }
            if self.gated:
                pred_labels = self.gated.decode(features, preds, index=b)
            else:
                pred_labels = self.rep.decode_preds(preds_b)
            aps.append(instance_ap(label_np, pred_labels))
            mious.append(matched_mean_iou(
                label_np, pred_labels)["matched_iou"])
            pqs.append(panoptic_quality(label_np, pred_labels)["PQ"])
        for thr in (25, 50, 75):
            key = f"AP@{thr / 100:.2f}"
            self.log(
                f"{phase}_ap{thr}",
                sum(a[key] for a in aps) / bsz,
                batch_size=bsz
            )
        self.log(f"{phase}_matched_iou", sum(mious) / bsz,
                 batch_size=bsz)
        self.log(f"{phase}_pq", sum(pqs) / bsz,
                 batch_size=bsz)

    def validation_step(self, batch, batch_idx):
        self._eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._eval_step(batch, "test")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.optimizer_cfg.lr,
            weight_decay=self.optimizer_cfg.wd,
        )
        # linear warm-up to full LR, then cosine decay to lr/eta_min_factor
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1 / self.optimizer_cfg.warmup_epochs,
            total_iters=self.optimizer_cfg.warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=self.optimizer_cfg.epochs - self.optimizer_cfg.warmup_epochs,
            eta_min=self.optimizer_cfg.lr / 100,
        )
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[
                self.optimizer_cfg.warmup_epochs]
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "epoch",
            },
        }

def load_model(ckpt_path: str | Path) -> "UnifedSegmentationModel":
    """Load a model from a checkpoint file or a run directory.

    A run directory (containing `checkpoints/`) resolves to its best
    checkpoint (highest-val filename, last of the sorted matches). Missing
    path raises FileNotFoundError.
    """

    p = Path(ckpt_path)
    if not p.exists():
        raise FileNotFoundError(f"checkpoint not found: {p}")
    if p.is_dir():
        cands = sorted(p.glob("checkpoints/best-*.ckpt"))
        if not cands:
            raise FileNotFoundError(f"no checkpoints under {p}")
        p = cands[-1]
    rprint(f"[bold cyan]Loading checkpoint[/] {p}")
    # weights_only=False: hparams hold pydantic configs (torch >= 2.6
    # defaults to the restricted unpickler, which rejects them)
    return UnifedSegmentationModel.load_from_checkpoint(
        str(p), weights_only=False)