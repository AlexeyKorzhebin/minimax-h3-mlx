"""Memory accounting for the 48 GB budget.

Two numbers matter and they are not the same. MLX's own counters track what the *allocator* holds,
which is what the GPU wired limit applies to; process RSS is what the OS sees, and on Apple silicon
it includes the memory-mapped checkpoint pages a load has touched. A component can be freed by MLX
while its file pages linger in RSS, so an unload is only proven by watching the MLX active figure
fall — RSS is the sanity check, not the measurement.
"""

from __future__ import annotations

import gc
import os
import subprocess

from . import _upstream  # noqa: F401

import mlx.core as mx


def rss_gb() -> float:
    """Current resident set size of this process, in GB. 0.0 if it cannot be read."""
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return int(out) * 1024 / 1e9
    except Exception:
        return 0.0


def snapshot() -> dict[str, float]:
    """MLX allocator state plus process RSS, all in GB."""
    return {
        "mlx_active_gb": mx.get_active_memory() / 1e9,
        "mlx_peak_gb": mx.get_peak_memory() / 1e9,
        "mlx_cache_gb": mx.get_cache_memory() / 1e9,
        "rss_gb": rss_gb(),
    }


def format_snapshot(label: str, snap: dict[str, float] | None = None) -> str:
    snap = snap or snapshot()
    return (f"[mem] {label}: mlx active {snap['mlx_active_gb']:.2f} GB, "
            f"peak {snap['mlx_peak_gb']:.2f} GB, cache {snap['mlx_cache_gb']:.2f} GB, "
            f"rss {snap['rss_gb']:.2f} GB")


#: How much freed-buffer cache MLX may keep. Unbounded by default, which is right on a machine
#: with headroom and wrong here: the allocator reuses buffers rather than returning them, so a
#: long run drifts upward until the OS starts swapping. A 2h14m run showed 29.1 GB against roughly
#: 21.4 GB of weights plus activations — the rest was cache the run had no use for.
#:
#: Not zero: reallocating every intermediate would cost real time. Two gigabytes is enough for the
#: churn inside one forward while leaving the difference to the OS.
DEFAULT_CACHE_LIMIT_GB = 2.0


def limit_cache(gigabytes: float | None = None) -> int:
    """Bound MLX's freed-buffer cache. Returns the previous limit in bytes.

    `H3_CACHE_LIMIT_GB` overrides; 0 disables the cache entirely, which is the safest setting on a
    machine that is swapping and the slowest everywhere else.
    """
    import os

    if gigabytes is None:
        gigabytes = float(os.environ.get("H3_CACHE_LIMIT_GB", DEFAULT_CACHE_LIMIT_GB))
    return mx.set_cache_limit(int(gigabytes * 1e9))


def report() -> str:
    """One line of MLX's own accounting — the numbers `ps` cannot see.

    Process RSS omits Metal buffers entirely on Apple silicon: during a run holding 29 GB, `ps`
    reported 0.13 GB. Activity Monitor and MLX both see it; nothing else does.
    """
    s = snapshot()
    return (f"memory: {s['mlx_active_gb']:.2f} GB active, {s['mlx_cache_gb']:.2f} GB cached, "
            f"peak {s['mlx_peak_gb']:.2f} GB")


def release() -> None:
    """Actually give the memory back, in the order that makes it stick.

    Dropping the last *name* for a model is not enough. Module trees, closures and MLX graphs form
    reference cycles, so the buffers survive refcounting and only the cyclic collector reclaims
    them — measured: without the ``gc.collect()`` the 28.2 GB encoder stays resident right through
    the diffusion loop. Only then does clearing MLX's allocator cache return anything, since it can
    only release buffers nothing still refers to.
    """
    gc.collect()
    mx.clear_cache()


class PhaseTracker:
    """Records an MLX peak per named phase, for checking the run against its memory budget."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.phases: list[tuple[str, dict[str, float]]] = []

    def mark(self, label: str) -> dict[str, float]:
        snap = snapshot()
        self.phases.append((label, snap))
        if self.verbose:
            print(format_snapshot(label, snap), flush=True)
        mx.reset_peak_memory()
        return snap

    def worst(self) -> float:
        return max((s["mlx_peak_gb"] for _, s in self.phases), default=0.0)
