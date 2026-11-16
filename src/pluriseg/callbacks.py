from rich import print as rprint
import time

import torch
from lightning.pytorch import callbacks


class GradNormMonitor(callbacks.Callback):
    """Log the total gradient norm every step (diagnostic for clip tuning)."""

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        grads = [p.grad for p in pl_module.parameters() if p.grad is not None]
        if grads:
            norm = torch.linalg.vector_norm( # pylint: disable=E1102
                torch.stack([g.norm() for g in grads])).item()
            pl_module.log("train_grad_norm", norm,
                          on_step=True, on_epoch=False, prog_bar=False)


class NanGradGuard(callbacks.Callback):
    """Skip the optimizer step when any gradient is non-finite.

    A finite loss can still produce NaN gradients (bf16 overflow: forward
    gives exp(10·sigma) = inf, dist = exp(-inf) = 0, backward gives
    inf·0 = NaN). Without a guard AdamW applies the NaN to the shared
    backbone and every head collapses permanently. gradient_clip_val does
    not help (clip on NaN stays NaN), so zero the grads and skip the step;
    `train_nan_batches` counts skips in metrics.csv.
    """

    def __init__(self):
        self._warned = False

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        grads = [p.grad for p in pl_module.parameters() if p.grad is not None]
        if not grads:
            return
        bad = sum(int((~torch.isfinite(g)).any()) for g in grads)
        if not bad:
            return
        for g in grads:
            g.zero_()
        pl_module.log("train_nan_batches", 1, on_step=True, on_epoch=True,
                      reduce_fx="sum")
        if not self._warned:
            rprint(f'[bold red]NanGradGuard[/] {bad} non-finite grad(s) at '
                   f'step {trainer.global_step}: step skipped, grads zeroed')
            self._warned = True


class EpochTiming(callbacks.Callback):
    """Log per-epoch wall time (train and val separately) as epoch_train_s /
    epoch_val_s, so metrics.csv carries timing per epoch."""

    def __init__(self):
        self._t_train = None
        self._t_val = None

    @staticmethod
    def _log(pl_module, key: str, dt: float):
        pl_module.log(key, dt, on_step=False, on_epoch=True,
                      prog_bar=False, sync_dist=True)

    def on_train_epoch_start(self, trainer, pl_module):
        self._t_train = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        if self._t_train is not None:
            self._log(pl_module, "epoch_train_s",
                      time.perf_counter() - self._t_train)
            self._t_train = None

    def on_validation_epoch_start(self, trainer, pl_module):
        self._t_val = time.perf_counter()

    def on_validation_epoch_end(self, trainer, pl_module):
        if self._t_val is not None:
            self._log(pl_module, "epoch_val_s",
                      time.perf_counter() - self._t_val)
            self._t_val = None


class EpochLog(callbacks.Callback):
    """Print epoch summary metrics to stdout (useful when the progress bar is off)."""

    def _print(self, trainer, phase):
        metrics = {k: v for k, v in trainer.callback_metrics.items()
                   if k.startswith(f'{phase}_') and v.numel() == 1}
        if not metrics:
            return
        epoch = trainer.current_epoch + 1
        mstr = '  '.join(f'{k}={v.item():.4g}' for k, v in sorted(metrics.items()))
        rprint(f'[bold cyan]epoch {epoch:03d}[/] {phase}  {mstr}')

    def on_train_epoch_end(self, trainer, pl_module):
        self._print(trainer, 'train')

    def on_validation_epoch_end(self, trainer, pl_module):
        self._print(trainer, 'val')

    def on_test_epoch_end(self, trainer, pl_module):
        self._print(trainer, 'test')