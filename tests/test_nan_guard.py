"""NanGradGuard: poisoned-batch protection (step skipped, grads zeroed).

Regression test: a finite loss with NaN gradients (bf16 exp(10·sigma)
overflow → inf·0 in backward) must not reach the optimizer — it would
corrupt the shared backbone permanently.
"""

import torch
import torch.nn as nn

from pluriseg.callbacks import NanGradGuard


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(1, 1)


def _trainer_pl_module():
    """Minimal stand-ins: only .global_step and .log are touched."""
    class Trainer:
        global_step = 21479

    class PlModule:
        def __init__(self):
            self.module = _Tiny()
            self.logged = []

        def parameters(self):
            return self.module.parameters()

        def log(self, name, value, **kwargs):
            self.logged.append((name, value))

    return Trainer(), PlModule()


def test_guard_skips_and_zeroes_on_nan_grad():
    """NaN grad → step skipped (grads zeroed), counter logged."""
    trainer, pl = _trainer_pl_module()
    guard = NanGradGuard()
    p = next(pl.module.parameters())
    p.grad = torch.tensor([float("nan")]).expand_as(p)
    guard.on_before_optimizer_step(trainer, pl, None)
    assert torch.isfinite(p.grad).all() and p.grad.abs().sum() == 0
    assert pl.logged and pl.logged[0][0] == "train_nan_batches"


def test_guard_passes_finite_grads():
    trainer, pl = _trainer_pl_module()
    guard = NanGradGuard()
    p = next(pl.module.parameters())
    p.grad = torch.ones_like(p)
    guard.on_before_optimizer_step(trainer, pl, None)
    assert not pl.logged
    assert p.grad.abs().sum() == 1.0


def test_overflow_scenario_backward_is_nan():
    """The pinned mechanism: exp(10·sigma) overflows to inf, dist=exp(-inf)=0
    keeps the loss finite, backward yields NaN (inf·0) — exactly what the
    guard must catch and what the sigma clamp prevents."""
    sigma = torch.tensor(9.0, dtype=torch.bfloat16, requires_grad=True)
    d2 = torch.tensor(2.0, dtype=torch.bfloat16)
    s = torch.exp(sigma * 10)
    dist = torch.exp(-d2 * s)
    loss = (1 - dist).sum()
    assert torch.isfinite(loss) and loss.item() == 1.0  # dist == 0
    loss.backward()
    assert sigma.grad.isnan().item()


def test_embedseg_sigma_pred_clamped():
    """EmbedsegRep.loss clamps predicted sigma at sigma_ceil: a σ̂ above the
    ceiling must not reach the vendored loss, where exp(10·σ̂) overflows."""
    import numpy as np

    from pluriseg.representations.embed import EmbedsegRep

    rep = EmbedsegRep(train_patch=256)
    H = W = 64
    label = np.zeros((H, W), np.uint32)
    label[16:48, 16:48] = 1
    encoded = rep.encode_targets(label)
    targets = {
        k: torch.from_numpy(v).permute(2, 0, 1).unsqueeze(0)
        for k, v in encoded.items()
    }
    preds = rep.encoded_to_perfect_pred(encoded)
    # poison sigma far beyond the ceiling (bf16 overflow threshold ~8.87)
    preds["sigma"] = preds["sigma"] + 100.0
    preds = {k: v.requires_grad_(True) for k, v in preds.items()}
    losses = rep.loss({k: v.unsqueeze(0) for k, v in preds.items()}, targets)
    assert torch.isfinite(losses["loss"])
    loss = losses["loss"]
    loss.backward()
    # the clamped sigma gradient is zero (no NaN path), other channels flow
    assert torch.isfinite(preds["sigma"].grad).all()
    assert preds["sigma"].grad.abs().sum() == 0
    assert torch.isfinite(preds["offset"].grad).all()
