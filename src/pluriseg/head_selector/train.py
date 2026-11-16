

from pathlib import Path

import numpy as np
import torch
from rich import print as rprint
from rich.progress import track

from pluriseg.configs import GateConfig, OptimizerConfig
from pluriseg.data import SegmentationData
from pluriseg.model import UnifedSegmentationModel
from pluriseg.models.gate import GateRouter


def refit_router(features: np.ndarray, scores: np.ndarray, hidden: int,
                 epochs: int, val_frac: float, seeds: int, seed: int,
                 lr: float = 1e-3, wd: float = 0.01, loss: str = "ce",
                 tau: float = 0.1):
    """Fit a fresh GateRouter on (pooled features, per-head PQ).

    Decision-focused by default: soft-target cross-entropy with target
    softmax(y / tau) over the per-head PQ (tau -> 0 hardens toward the
    oracle argmax); the router output is only ever argmax'd, so calibrated
    regression buys nothing — MSE stays available via loss="mse". Full-batch
    AdamW; every epoch the val routed-PQ is tracked and the best step kept;
    across `seeds` inits the best seed wins (val split shared across seeds).

    Returns (router module (eval, cpu), stats dict on the val split).
    """
    rng = np.random.default_rng(seed)
    n = len(features)
    perm = rng.permutation(n)
    # a 1-entry bank has no split: val tracks the single training entry
    if n >= 2:
        n_val = max(1, int(val_frac * n))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
    else:
        val_idx, tr_idx = perm, perm
    seeds = max(1, seeds)

    x = torch.from_numpy(features)
    y = torch.from_numpy(scores)
    q = torch.softmax(y / tau, dim=1)

    # (val routed-PQ, state_dict, seed idx)
    best: tuple[float, dict[str, torch.Tensor], int] | None = None
    best_rpq = float("-inf")
    for s in range(seeds):
        torch.manual_seed(seed + s)
        router = GateRouter(in_ch=features.shape[1], n_heads=scores.shape[1],
                            hidden=hidden)
        opt = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=wd)
        for _ in range(epochs):
            opt.zero_grad()
            out = router.net(x[tr_idx])
            if loss == "ce":
                loss_ = -(q[tr_idx] * torch.log_softmax(out, dim=1)
                          ).sum(1).mean()
            else:
                loss_ = torch.nn.functional.mse_loss(out, y[tr_idx])
            loss_.backward()
            opt.step()
            with torch.no_grad():
                routed = router.net(x[val_idx]).argmax(1)
                rpq = y[val_idx].gather(1, routed[:, None]).mean()
            rpq_val = float(rpq)
            if rpq_val > best_rpq:
                best_rpq = rpq_val
                best = (rpq_val, {k: v.detach().clone()
                                  for k, v in router.state_dict().items()}, s)

    router = GateRouter(in_ch=features.shape[1], n_heads=scores.shape[1],
                        hidden=hidden)
    router.load_state_dict(best[1])
    router.eval()
    with torch.no_grad():
        val_pred = router.net(x[val_idx])
        val_y = y[val_idx]
        routed = val_pred.argmax(1)
        stats = {
            "val_mse": float(torch.nn.functional.mse_loss(val_pred, val_y)),
            # fraction of val images where the routed head is the oracle-best
            "argmax_match": float(
                (routed == val_y.argmax(1)).float().mean()),
            # PQ of the routed head vs the oracle-best: the decode quality
            # the gate leaves on the table
            "routed_pq": float(
                val_y.gather(1, routed[:, None]).mean()),
            "best_pq": float(val_y.max(1).values.mean()),
            "best_seed": best[2],
        }
    return router, stats


def _score_val_bank(model: UnifedSegmentationModel, dm: SegmentationData,
                    data_path: Path | None = None) -> tuple | None:
    """Score every val image under the frozen model (the post-gate bank).

    One image at a time, in-process. Images without GT objects are skipped
    (PQ undefined). With data_path the bank is cached every 32 images in
    <run>/gate_post_data.npz (features, scores, heads, n_processed): an
    interrupted scoring resumes where it stopped.

    Returns (features (n, C) float32, scores (n, n_heads) float32), or
    None when the val bank is empty.
    """
    model.eval()
    if dm.val_ds is None:
        # load_full_image reads raw files (no rep cache); skip the train
        # warm-up entirely
        dm.dataset_cfg.warm_cache = False
        dm.setup("fit")
    n_val = len(dm.val_ds.image_label_pairs)
    head_names = list(model.gated.head_names)
    # interval to cache buffer to disk (in case things go wrong)
    cache_interval = 32
    feats: list = []
    scores: list = []
    start = 0
    if data_path is not None and data_path.exists():
        d = np.load(data_path, allow_pickle=True)
        if list(d["heads"]) == head_names and int(d["n_processed"]) <= n_val:
            feats = [torch.as_tensor(f) for f in d["features"]]
            scores = [torch.as_tensor(s) for s in d["scores"]]
            start = int(d["n_processed"])
            rprint(f"[cyan]resuming post bank at {start}/{n_val} "
                   f"({len(feats)} scored image(s))[/cyan]")

    for idx in track(range(start, n_val), "computing image features and scores on val data"):
        img, label = dm.load_full_image(idx)
        if label.max() == 0:
            continue

        f, s = model.gated.compute_pooled_features_and_score(
            image=img,
            label=label,
            backbone=model.backbone,
            head=model.head
        )
        feats.append(f)
        scores.append(s)

        if data_path is not None and idx % cache_interval == 0:
            np.savez(data_path,
                     features=torch.stack(
                         feats).numpy().astype(np.float32),
                     scores=torch.stack(scores).numpy().astype(np.float32),
                     heads=np.array(head_names),
                     n_processed=idx + 1)

    if not feats:
        return None
    if data_path is not None:
        np.savez(data_path,
                 features=torch.stack(feats).numpy().astype(np.float32),
                 scores=torch.stack(scores).numpy().astype(np.float32),
                 heads=np.array(head_names),
                 n_processed=n_val)
    return (torch.stack(feats).numpy().astype(np.float32),
            torch.stack(scores).numpy().astype(np.float32))


def train_selection_head(model: UnifedSegmentationModel, dm: SegmentationData,
                         gate_cfg: GateConfig, optimizer_cfg: OptimizerConfig, seed: int
                         ) -> dict | None:
    """Post-hoc gate refit: score all val images under the FROZEN model
    (deployment weights — the caller loads the best checkpoint first),
    refit the router with soft-target CE on per-head PQ, and load it into
    model.gated.router.

    Scoring delegates to _score_val_bank (sequential, resumable cache);
    the refit reuses refit_router. Frozen weights make every scored
    (image, label) entry valid forever, so one pass suffices.

    Returns the refit stats dict (val routed/best PQ, argmax match), or None
    when the val bank is empty.
    """
    bank = _score_val_bank(model, dm)
    if bank is None:
        rprint("[yellow][gate] no val images with GT — post refit skipped[/]")
        return None
    features, scores = bank
    router, stats = refit_router(
        features, scores, hidden=gate_cfg.hidden, epochs=gate_cfg.post_epochs,
        val_frac=gate_cfg.post_val_frac, seeds=gate_cfg.post_seeds, seed=seed,
        lr=gate_cfg.post_lr, wd=optimizer_cfg.wd, loss="ce", tau=gate_cfg.tau)
    model.gated.router.load_state_dict(router.state_dict())
    return stats


def promote_ckpt(run_dir: Path, refit_ckpt: Path) -> Path:
    """Make the refit ckpt the run's best: the original best-*.ckpt moves to
    pre-refit-*.ckpt and the refit takes the best-* name, so eval_ selects
    the refit router from now on. Returns the promoted ckpt path."""
    assert refit_ckpt.name.startswith("refit-"), refit_ckpt.name
    ckpts = run_dir / "checkpoints"
    name = refit_ckpt.name[len("refit-"):]
    pre = ckpts / f"pre-refit-{name}"
    if pre.exists():
        rprint(f"[yellow]{pre.name} already exists — true original kept, "
               f"current best overwritten by the new refit[/yellow]")
    elif (ckpts / name).exists():
        (ckpts / name).rename(pre)
        rprint(f"[yellow]promoted: {name} → {pre.name}[/yellow]")
    refit_ckpt.rename(ckpts / name)
    return ckpts / name
