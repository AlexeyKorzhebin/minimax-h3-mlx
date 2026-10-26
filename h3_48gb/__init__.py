"""Fork-side patches for running MiniMax-H3 on a 48 GB Mac from the mere.run/Sawfwair build.

`upstream/` is never edited, so it can be fast-forwarded; everything here is applied from the
outside. Four changes, in descending order of how badly they fail without it:

1. :mod:`h3_48gb.adaln` — the mere.run transformer omits the AdaLN modulation path entirely
   (106 tensors), having precomputed it into ``adaln_cache.safetensors``. This serves the port's
   ``ModulationCache`` interface from that file, with the ``(step, variant, modality)`` layout
   established from the data and a hard failure on any schedule the table was not baked for.
2. :mod:`h3_48gb.text_encoder` — the port has no quantization path and would load mere.run's 8-bit
   conditioner as packed integers in bf16 slots, silently.
3. :mod:`h3_48gb.pipeline` — components load per phase instead of all at once (45.9 GB of
   weights resident, ~55 GB with activations at 1344x768/5 s; see ``docs/RESULTS.md``).
4. The same module unloads the 28.2 GB text encoder after its single call, materializing its output
   first so the release is real.
5. :mod:`h3_48gb.checkpoint` — a run is resumable. At 586 s per step over 30 steps a crash on the
   last one costs five hours, and fifteen for a ten-second clip; the state that carries a run
   forward is two arrays and fits in a file written between steps.
6. :mod:`h3_48gb.preview` — a run is *watchable*. The same five-hour clip produces no visible pixel
   until the very end; a wrong prompt or composition is otherwise discovered only after paying for
   the whole render. ``preview_every`` decodes one frame from the current latent every N steps and
   writes it next to where the finished clip will land.

Usage::

    from h3_48gb import LazyMiniMaxH3Pipeline

    pipe = LazyMiniMaxH3Pipeline.from_pretrained("~/models/h3-converted")
    result = pipe("a cat", num_inference_steps=31, height=512, width=512,
                  checkpoint_dir="~/video-out/checkpoints",
                  preview_every=5, preview_stem="~/video-out/h3-run")

Re-running the same call resumes where the last one stopped; a call with different parameters is
refused rather than silently continued. ``preview_every=0`` (the default) disables previews.
"""

from __future__ import annotations

from .cli import main

# Lazy imports: these modules require mlx, so they're imported only when needed
_LAZY_IMPORTS = {
    "AdaLNCacheFile": ("adaln", "AdaLNCacheFile"),
    "CachedModulation": ("adaln", "CachedModulation"),
    "ScheduleMismatch": ("adaln", "ScheduleMismatch"),
    "CheckpointCorrupt": ("checkpoint", "CheckpointCorrupt"),
    "CheckpointError": ("checkpoint", "CheckpointError"),
    "CheckpointingPipeline": ("checkpoint", "CheckpointingPipeline"),
    "CheckpointLocked": ("checkpoint", "CheckpointLocked"),
    "CheckpointMismatch": ("checkpoint", "CheckpointMismatch"),
    "CheckpointStore": ("checkpoint", "CheckpointStore"),
    "ResumableRun": ("checkpoint", "ResumableRun"),
    "ResumableScheduler": ("checkpoint", "ResumableScheduler"),
    "weights_fingerprint": ("checkpoint", "weights_fingerprint"),
    "load_dit_cached": ("dit", "load_dit_cached"),
    "PhaseTracker": ("memory", "PhaseTracker"),
    "format_snapshot": ("memory", "format_snapshot"),
    "snapshot": ("memory", "snapshot"),
    "LazyComponent": ("pipeline", "LazyComponent"),
    "LazyMiniMaxH3Pipeline": ("pipeline", "LazyMiniMaxH3Pipeline"),
    "LazyTextEncoder": ("pipeline", "LazyTextEncoder"),
    "PreviewInterceptor": ("preview", "PreviewInterceptor"),
    "emit_preview": ("preview", "emit_preview"),
    "min_preview_latent_frames": ("preview", "min_preview_latent_frames"),
    "QuantizedTextEncoder": ("text_encoder", "QuantizedTextEncoder"),
}


def __getattr__(name: str):
    """Lazy load heavy modules that require mlx.core.

    `UPSTREAM` and `ensure_on_path` are lazy too: importing `_upstream` puts the vendored checkout
    on `sys.path` and raises FileNotFoundError when there is none, and the CLI must import
    where `upstream/` does not exist -- a fresh clone.
    """
    if name in ("UPSTREAM", "ensure_on_path"):
        from . import _upstream

        return getattr(_upstream, name)
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        module = __import__(f"h3_48gb.{module_name}", fromlist=[attr_name])
        return getattr(module, attr_name)
    raise AttributeError(f"module 'h3_48gb' has no attribute '{name}'")


def __dir__():
    """Expose all public API symbols.

    Returns only the symbols listed in __all__ in a deduplicated, sorted order.
    Filters out all private names (starting with _) and dunders.
    """
    return sorted(__all__)

__all__ = [
    "AdaLNCacheFile",
    "CachedModulation",
    "CheckpointCorrupt",
    "CheckpointError",
    "CheckpointLocked",
    "CheckpointMismatch",
    "CheckpointStore",
    "CheckpointingPipeline",
    "LazyComponent",
    "LazyMiniMaxH3Pipeline",
    "LazyTextEncoder",
    "PhaseTracker",
    "PreviewInterceptor",
    "QuantizedTextEncoder",
    "ResumableRun",
    "ResumableScheduler",
    "ScheduleMismatch",
    "UPSTREAM",
    "emit_preview",
    "ensure_on_path",
    "format_snapshot",
    "load_dit_cached",
    "main",
    "min_preview_latent_frames",
    "snapshot",
    "weights_fingerprint",
]
