"""Per-stage timing to long-format CSV.

A `Timer` accumulates per-stage times for one item (`timer.stage(name)`),
buffers rows of (idx, pid, worker, stage, seconds), and on close writes
`<log_dir>/<tag>-p<pid>.csv` via pandas — each process (main or dataloader
worker) writes its own file, safe under fork/spawn. Aggregate later with e.g.
`df.groupby(["pid", "stage"])["seconds"].mean()`.

Enabled only when a log dir is given (`Timer(log_dir=...)`); disabled by
default, zero overhead.
"""

import atexit
import os
import time
from pathlib import Path

import pandas as pd


class stage_timer:
    """Context manager returned by Timer.stage."""

    __slots__ = ("timer", "name", "t0")

    def __init__(self, timer: "Timer", name: str):
        self.timer = timer
        self.name = name

    def __enter__(self):
        if self.timer.enabled:
            self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.timer.enabled and self.t0 is not None:
            dt = time.perf_counter() - self.t0
            self.timer._item[self.name] = self.timer._item.get(
                self.name, 0.0) + dt
            self.t0 = None
        return False


class Timer:
    """Per-item stage accumulator; one long-format CSV per process.

    Rows append to <log_dir>/<tag>-p<pid>.csv every `flush_every` items
    (header written once on file creation), so a killed run keeps its
    timings; close() does the final flush.
    """

    def __init__(self, log_dir: str = "", flush_every: int = 500):
        self.enabled = bool(log_dir)
        self.log_dir = Path(log_dir) if self.enabled else None
        self.flush_every = flush_every
        self._item: dict[str, float] = {}
        self._rows: list[tuple[int, int, int, str, float]] = []
        self._tag: str | None = None
        self._worker = -1
        self._idx = 0

    def _worker_info(self) -> None:
        try:
            import torch.utils.data as tud
            info = tud.get_worker_info()
            self._worker = info.id if info else -1
        except Exception:
            self._worker = -1

    def stage(self, name: str) -> stage_timer:
        return stage_timer(self, name)

    def tick(self, tag: str = "") -> None:
        """Close out the current item. `tag` names the log file; call it
        consistently (e.g. f"{dataset}-{mode}") for the whole run."""
        if not self.enabled:
            return
        if self._tag is None:
            self._tag = tag
            self._worker_info()
        for stage, seconds in self._item.items():
            self._rows.append(
                (self._idx, os.getpid(), self._worker, stage, seconds))
        self._idx += 1
        self._item = {}
        if self._idx % self.flush_every == 0:
            self._flush()

    def _flush(self) -> None:
        """Append buffered rows to <log_dir>/<tag>-p<pid>.csv."""
        if not self.enabled or not self._rows:
            return
        df = pd.DataFrame(
            self._rows, columns=["idx", "pid", "worker", "stage", "seconds"]
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / f"{self._tag}-p{os.getpid()}.csv"
        df.to_csv(path, index=False, mode="a", header=not path.exists())
        self._rows = []

    def close(self) -> None:
        """Final flush; safe to call multiple times."""
        self._flush()

    def install_exit_hook(self) -> None:
        """Write the CSV when the process exits (call from worker_init_fn so
        each worker dumps its own file)."""
        if self.enabled:
            atexit.register(self.close)


def read_logs(log_dir: str | Path) -> pd.DataFrame:
    """Load all timing CSVs from log_dir into one long DataFrame."""
    files = sorted(Path(log_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no timing CSVs under {log_dir}")
    return pd.concat(
        [pd.read_csv(f).assign(file=f.name) for f in files], ignore_index=True
    )


def compute_summary_table(
    df: pd.DataFrame
) -> pd.DataFrame:

    return df.groupby("stage")["seconds"].agg(["mean", "std", "sum", "count"])


def print_table(df: pd.DataFrame, tag: str = "timing") -> None:
    """Rich table aggregating all processes: per-stage mean/total ms and
    per-stage call counts; items = sum of per-process item counts."""
    from rich.console import Console
    from rich.table import Table

    g = compute_summary_table(df)
    n_items = df.groupby("pid")["idx"].nunique().sum()
    console = Console(highlight=False)
    table = Table(
        title=f"[timing]{tag}: {n_items} items, {g['sum'].sum():.2f}s total")
    table.add_column("stage", style="cyan")
    table.add_column("mean±std ms", justify="right")
    table.add_column("total s", justify="right")
    table.add_column("calls", justify="right")
    for stage, r in g.sort_values("sum", ascending=False).iterrows():
        table.add_row(
            stage,
            f"{1000 * r['mean']:.1f}±{1000 * r['std']:.1f}",
            f"{r['sum']:.2f}",
            str(int(r["count"])),
        )
    console.print(table)
