"""Per-image gate for ChimeraRep: router + wrapper owning the gate machinery.

GatedChimera holds the router, the head-pick decision and the post-refit
scoring, so the Lightning module only wires calls through. The router is
never trained during fit: head_selector.train scores every val image under
the frozen best checkpoint (the bank) and refits it offline (soft-target
CE on per-head PQ) — train.py's post-gate block.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pluriseg.configs import GateConfig
from pluriseg.eval.metrics import panoptic_quality
from pluriseg.representations.base import Representation

_POOL_MODES = ("gap", "stats", "stats_grid")


class GateRouter(nn.Module):
    """Pool backbone features -> MLP -> per-head logits.

    pool() maps (B, C, H, W) to a (B, D) descriptor; forward pools then
    classifies. Supervision is soft-target cross-entropy against per-head
    panoptic-quality scores (targets softmax(PQ / tau)).
    """

    _POOL_DIM = {"gap": 1, "stats": 3, "stats_grid": 7}  # x in_ch

    def __init__(self, in_ch: int, n_heads: int, hidden: int = 128,
                 pool: str = "gap"):
        super().__init__()
        if pool not in _POOL_MODES:
            raise ValueError(
                f"unknown gate.pool {pool!r}; expected one of {_POOL_MODES}")
        self.pool_mode = pool
        self.net = nn.Sequential(
            nn.Linear(self._POOL_DIM[pool] * in_ch, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_heads),
        )

    def pool(self, features: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, D) pooled descriptor.

        gap: GAP only. stats: [GAP, GMP, std]. stats_grid: stats plus the
        2x2 adaptive-avg-pool means (spatial layout, scale-free: identical
        code path for training patches and whole images).
        """
        if self.pool_mode == "gap":
            return features.mean(dim=(2, 3))
        stats = [features.mean(dim=(2, 3)),
                 torch.amax(features, dim=(2, 3)),
                 features.flatten(2).std(dim=2)]
        if self.pool_mode == "stats":
            return torch.cat(stats, dim=1)
        grid = F.adaptive_avg_pool2d(features, 2).flatten(1)
        return torch.cat([*stats, grid], dim=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, n_heads) logits."""
        return self.net(self.pool(features))

    def forward_pooled(self, pooled: torch.Tensor) -> torch.Tensor:
        """Pre-pooled descriptor (B, D) -> logits (replay-buffer path)."""
        return self.net(pooled)


class GatedChimera(nn.Module):
    """Gate wrapper for a ChimeraRep: pick one sub-rep head per image.

    Owns everything gate-related so the model module stays thin:
    - decode: argmax over router logits selects a sub-rep, decoded via the
      rep's per-head path;
    - scoring: forward + per-head PQ for one whole image under frozen
      weights — what head_selector._score_val_bank calls per val image.

    The wrapped rep is referenced, not registered: it is the owning model's
    `.rep`, so it must not appear twice in the module tree.
    """

    def __init__(self, rep: Representation, gate_cfg: GateConfig, in_ch: int):
        super().__init__()
        self.rep = rep
        self.cfg = gate_cfg
        self.router = GateRouter(
            in_ch=in_ch, n_heads=len(rep.subs), hidden=gate_cfg.hidden,
            pool=gate_cfg.pool)
        self.head_names = list(rep.subs)
        self.last_gate: dict[str, float] = {}  # logits per head at last decode

    # ------------------------------------------------------------- decode

    def pick(self, features: torch.Tensor, index: int = 0) -> str:
        """(B, C, H, W) -> argmax head name for batch item `index`.

        Logits land in `last_gate` (single-image use; the last call wins).
        """
        # pool on the features' device (parameter-free), classify on the
        # router's device — CPU-forwarded features must not break the pick
        pooled = self.router.pool(features)
        device = next(self.router.parameters()).device
        logits = self.router.forward_pooled(
            pooled.to(device))[index].detach().cpu().tolist()
        self.last_gate = dict(zip(self.head_names, logits))
        return max(self.head_names, key=lambda s: self.last_gate[s])

    def decode(self, features: torch.Tensor, preds: dict[str, torch.Tensor],
               index: int = 0) -> np.ndarray:
        """Gate-picked per-head decode for one image of a batch.

        preds is the model head output for the batch; index selects the
        image. Returns (H, W) instance labels from the picked sub-rep.
        """
        sname = self.pick(features, index)
        preds_b = {k: v[index] for k, v in preds.items()}
        return self.rep.decode_preds(
            preds_b, cfg={"decode": "head", "decode_head": sname})

    # ------------------------------------------------- post-refit scoring

    @staticmethod
    def _pad32(img: torch.Tensor) -> torch.Tensor:
        """Zero-pad (C, H, W) down/right to a multiple of 32."""
        ph, pw = (-img.shape[-2]) % 32, (-img.shape[-1]) % 32
        return F.pad(img, (0, pw, 0, ph)) if (ph or pw) else img

    @torch.no_grad()
    def compute_pooled_features_and_score(
        self, image: torch.Tensor, label: np.ndarray,
        backbone: nn.Module, head: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pooled descriptor + per-head PQ for one whole image under frozen
        weights.

        Pool runs on the padded features (the decode-time router pools the
        same extent via _predict_logits); head outputs are cropped back to
        the original extent before decode. Images too large for the GPU
        fall back to a fp32 CPU forward (identical numerics, just slower);
        pool() is parameter-free, so the pooled descriptor computes on CPU.
        """
        h, w = image.shape[-2:]
        image = self._pad32(image).unsqueeze(0)  # (1, C, H, W) /32-padded
        device = next(backbone.parameters()).device
        try:
            feats = backbone(image.to(device))
            head_res = head(feats)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            backbone.to("cpu"), head.to("cpu")
            feats = backbone(image)
            head_res = head(feats)
            backbone.to(device), head.to(device)
        pooled_feats = self.router.pool(feats.float())[0].cpu()
        pred_rep = {k: v[0, :, :h, :w].cpu() for k, v in head_res.items()}
        scores = [panoptic_quality(
            label, self.rep.decode_preds(
                pred_rep, cfg={"decode": "head", "decode_head": sname}))["PQ"]
            for sname in self.head_names]
        return pooled_feats, torch.tensor(scores)
