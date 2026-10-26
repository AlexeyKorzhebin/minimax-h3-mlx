"""Mid-generation preview frames: see the shot before spending the rest of a multi-hour render.

A 1344x768 clip runs 5-15.7 hours end to end (5s / 10s), all of it spent before
``MiniMaxH3Pipeline._decode_video`` produces a single visible pixel. If the prompt missed or the
composition is wrong, that is discovered only at the very end. This module decodes one frame from
the current (partially denoised) latent every few steps instead.

Two constraints shaped the design, both measured rather than assumed:

* **The VAE can't decode less than one chunk.** `VideoVAE.decode` (`upstream/minimax_h3_mlx/
  video_vae.py`) is a causal chunked decoder: its temporal padding assumes frame 0 of whatever it
  is given is the true start of the clip, and it refuses fewer than
  ``2 * chunk_tokens - token_drop`` latent frames outright (its own ``ValueError``). For the
  released config that floor is 7 latent frames -> about 22 pixel frames (~1s at 24 fps) — there is
  no cheaper *correct* VAE decode than that, and it must start at frame 0, not at the step's
  "interesting" moment. `render_preview_frame` decodes exactly that minimal prefix and keeps one
  representative frame from the middle of it.
* **Tiling cost is spatial, not temporal.** `VideoVAE._decode_clip` tiles over height/width
  regardless of how many latent frames it is given (256px tiles, 64px overlap -> 28 tiles at
  1344x768), so decoding the 7-frame floor still walks the full native canvas through the 36-layer
  ViT decoder. Measured on this machine (correctly-shaped random weights — the real checkpoint
  is not downloaded here; compute cost depends on shapes/dtypes, not weight values):
  weights resident 5.2 GB, decode itself 49s wall time, peak allocator usage 8.5 GB. See
  `docs/RESULTS.md` ("Previews") for how that compares to the diffusion loop's own budget, and for
  the same caveat about where those three numbers come from.

That decode happens with the video VAE loaded on demand and unloaded again immediately after,
rather than kept resident for the whole diffusion loop: keeping it resident would add its ~5.2 GB
on top of the DiT's *own* peak activation moment (up to ~42 GB projected at 1344x768/15s), which
is uncomfortably close to the 44 GB GPU limit. A preview's decode instead runs between steps, after
the previous step's activations have already been freed by `mx.eval` — so the concurrent peak is
DiT weights + AdaLN cache + the VAE's own 8.5 GB, comfortably under budget regardless of duration.
The trade-off is repeated load/unload across the run instead of one load; at the default cadence
of every 5 steps that is a handful of extra loads against a run measured in hours.

Every entry point but `emit_preview` raises on failure — that keeps them independently testable.
`emit_preview` is the only one the diffusion loop should call: it never raises, because a broken
preview must never cost the hours of compute already spent on the real generation (requirement 3
of the task this module implements).

`PreviewInterceptor` is how this hooks into the loop at all, and it deliberately does *not*
reimplement `MiniMaxH3Pipeline.__call__` the way :mod:`h3_48gb.checkpoint` reasons it would have to
for checkpointing. A preview only needs to *see* the state entering a step, and upstream's loop
already hands the full packed rows to the transformer every step — so wrapping `pipeline.dit`, the
same trick `h3_48gb.checkpoint.StepInterceptor` uses, is enough. `LazyMiniMaxH3Pipeline.__call__`
installs it immediately before delegating to `super().__call__()` and restores the original `dit`
in a `finally`, so it composes with checkpointing's own wrapping rather than fighting it: whichever
one is installed first ends up on the outside, and a step the checkpoint fast-forwards never
reaches this wrapper at all (there is nothing new to preview in state that was already on disk).
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

#: `__call__` keyword arguments this module owns, stripped before the upstream signature is bound
#: — mirrors `h3_48gb.checkpoint.CHECKPOINT_KWARGS` / `pop_checkpoint_kwargs`.
PREVIEW_KWARGS = ("preview_every", "preview_stem", "preview_decoder")


def pop_preview_kwargs(kwargs: dict) -> dict:
    """Remove and return this module's keyword arguments from a ``__call__`` kwargs dict."""
    return {name: kwargs.pop(name) for name in PREVIEW_KWARGS if name in kwargs}


def min_preview_latent_frames(video_vae_config) -> int:
    """The fewest latent frames `VideoVAE.decode` will accept, per its own floor.

    Mirrors the arithmetic in `VideoVAE.__init__` / the `ValueError` branch of `decode`
    (``tokens_chunk_size = ceil(clip_length / temporal_compression_ratio)``, floor
    ``2 * tokens_chunk_size - token_drop``) without importing VAE internals, so a preview request
    can be sized before the VAE is even loaded.
    """
    ratio_t = video_vae_config.temporal_compression_ratio
    chunk_tokens = math.ceil(video_vae_config.clip_length / ratio_t)
    return 2 * chunk_tokens - video_vae_config.token_drop


def render_preview_frame(
    pipeline,
    generated_video_rows: mx.array,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    patch_size,
) -> np.ndarray:
    """Decode the earliest ~1s of the current (still-noisy) video latent into one RGB frame.

    ``generated_video_rows`` is ``video_rows[n_cond_v:]`` from the diffusion loop — the same
    tensor the loop scheduler-steps every iteration, *before* the run has converged. Early previews
    are rough by construction; that is expected and is still enough to judge composition and
    whether the prompt landed.

    Loads ``pipeline.video_vae`` if it is not already resident and unloads it again before
    returning (successfully or not), via a `finally` block — see the module docstring for why a
    preview must not leave the VAE's ~5.2 GB pinned into the next diffusion step. Raises on any
    failure; callers that must not propagate errors should go through `emit_preview` instead.

    Returns:
        ``(H, W, 3)`` uint8 RGB — one frame, at the native pixel resolution the run is generating.
    """
    from minimax_h3_mlx.packing import PIXEL_MEAN, PIXEL_STD, unpatchify_video_tokens

    cfg = pipeline.video_vae.config
    needed = min_preview_latent_frames(cfg)
    if num_latent_frames < needed:
        raise ValueError(
            f"Not enough latent frames for a preview: have {num_latent_frames}, need {needed}."
        )

    latents = unpatchify_video_tokens(
        generated_video_rows, num_latent_frames, latent_height, latent_width,
        cfg.latent_channels, patch_size,
    )
    mean = mx.array(np.array(cfg.latents_mean, np.float32)).reshape(1, -1, 1, 1, 1)
    std = mx.array(np.array(cfg.latents_std, np.float32)).reshape(1, -1, 1, 1, 1)
    # Only the minimal decodable prefix. This used to claim a later window would decode a boundary
    # artifact because the VAE's temporal padding is causal -- but the *encoder* is causal; the
    # decoder is a non-causal ViT with full attention inside its 7-latent-frame window, and that
    # window is the same wherever it sits. Measured on four finished clips: at the decode seams
    # that coincide with nothing else -- frames 17, 51, 68 -- the pixel discontinuity is 1.00x the
    # local median, i.e. invisible. (Frame 34 is 5-11x, but it is both a seam and the Shot 2 cut,
    # so it cannot separate the two; the other three seams are what carry the point.) Any aligned
    # window would do; the prefix is chosen because it is the one that exists first.
    latents = (latents[:, :, :needed] * std + mean).astype(mx.float32)

    was_loaded = bool(getattr(pipeline.video_vae, "loaded", True))
    try:
        frames = np.array(pipeline.video_vae.decode(latents))
    finally:
        if not was_loaded and hasattr(pipeline.video_vae, "unload"):
            pipeline.video_vae.unload()

    pixel_mean = np.array(PIXEL_MEAN, np.float32).reshape(1, 3, 1, 1, 1)
    pixel_std = np.array(PIXEL_STD, np.float32).reshape(1, 3, 1, 1, 1)
    frames = frames * pixel_std + pixel_mean
    frames = np.clip(frames, 0.0, 1.0)[0].transpose(1, 2, 3, 0)  # -> (F, H, W, 3)
    frame_idx = frames.shape[0] // 2  # middle of the mini-clip, away from chunk-boundary blending
    return (frames[frame_idx] * 255.0 + 0.5).astype(np.uint8)


def render_tae_frame(
    pipeline,
    generated_video_rows: mx.array,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    patch_size,
) -> np.ndarray:
    """Decode one frame with TAE: no VAE, no chunk floor, no tiling.

    Unlike `render_preview_frame` this can decode *any* single frame, because TAE has no temporal
    state — so it takes the middle one, where the clip is most representative, rather than being
    forced to start at frame 0 by the real VAE's causal padding.

    The whole point of TAE is that it never *needs* the video VAE, so its absence, breakage, or
    (today) its normalization constants being irrelevant must never stop this from decoding. The
    optional read of `pipeline.video_vae.config` below is wrapped in its own narrow try/except for
    exactly that reason: `LazyComponent.config` is documented to resolve without a load, so reading
    it is normally free and worth doing for forward-compatibility with a future
    `TAE_EXPECTS_NORMALIZED = False` — but if `pipeline.video_vae` cannot be reached at all (no
    such attribute, or the attribute itself raises), this falls back to the identity transform
    (mean 0, std 1) and the checkpoint's own `LATENT_CHANNELS`, and still decodes with TAE.
    `hasattr` is deliberately not used here: it only swallows `AttributeError`, not an arbitrary
    exception a `video_vae` accessor might raise, so it would not provide the fallback this
    function needs.
    """
    from minimax_h3_mlx.packing import unpatchify_video_tokens

    from h3_48gb.tae import LATENT_CHANNELS, TAE_WEIGHTS_PATH, decode_latent_frame, load_tae

    latent_channels = LATENT_CHANNELS
    mean = mx.zeros((1, latent_channels, 1, 1, 1))
    std = mx.ones((1, latent_channels, 1, 1, 1))
    try:
        cfg = pipeline.video_vae.config
        latent_channels = cfg.latent_channels
        mean = mx.array(np.array(cfg.latents_mean, np.float32)).reshape(1, -1, 1, 1, 1)
        std = mx.array(np.array(cfg.latents_std, np.float32)).reshape(1, -1, 1, 1, 1)
    except Exception:  # noqa: BLE001 - the config is a bonus, not a requirement; see docstring
        pass

    latents = unpatchify_video_tokens(
        generated_video_rows, num_latent_frames, latent_height, latent_width,
        latent_channels, patch_size,
    )
    decoder = load_tae(TAE_WEIGHTS_PATH)
    return decode_latent_frame(decoder, latents, mean, std, num_latent_frames // 2)


def render_latent_fallback(
    generated_video_rows: mx.array,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    latent_channels: int,
    patch_size,
) -> np.ndarray:
    """A free, VAE-less approximation: per-pixel energy across latent channels, as grayscale.

    Not a color reconstruction — unlike e.g. Stable Diffusion's well-known 4-channel "latents to
    RGB" trick, there is no known fixed linear map from this VAE's 24 latent channels to RGB — but
    the spatial energy pattern already shows silhouette, motion and composition, which is what a
    sanity-check preview is for. No VAE, no forward pass: one reshape and one norm, at latent
    resolution (1/16 of native per axis). This is the fallback `emit_preview` reaches for when the
    real VAE decode fails or the video VAE cannot be loaded at all.

    Returns:
        ``(H', W', 3)`` uint8 grayscale-as-RGB, at *latent* resolution (small — callers should
        upscale for viewing if they want it pixel-matched to the real frame).
    """
    from minimax_h3_mlx.packing import unpatchify_video_tokens

    latents = unpatchify_video_tokens(
        generated_video_rows, num_latent_frames, latent_height, latent_width,
        latent_channels, patch_size,
    )
    frame_idx = num_latent_frames // 2  # the fallback has no chunk-decode floor, so use the middle
    frame = np.array(latents[0, :, frame_idx]).astype(np.float32)  # (C, H', W')
    energy = np.sqrt(np.mean(frame * frame, axis=0))  # (H', W')
    lo, hi = np.percentile(energy, 1), np.percentile(energy, 99)
    scaled = np.clip((energy - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    gray = (scaled * 255.0 + 0.5).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def save_preview_jpeg(frame: np.ndarray, dest: Path, quality: int = 85) -> None:
    """Write an ``(H, W, 3)`` uint8 array as a JPEG, creating the parent directory if needed."""
    from PIL import Image

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame, mode="RGB").save(dest, format="JPEG", quality=quality)


def preview_path(stem: str | Path, step: int) -> Path:
    """``<stem>-preview-step07.jpg`` — matches the naming of the final ``<stem>.mp4``."""
    return Path(f"{stem}-preview-step{step:02d}.jpg")


#: The valid `decoder` choices, for the error message on an unknown one. The renderer for each is
#: looked up as a plain global inside `emit_preview` itself, not built into a dict here — tests
#: monkeypatch `render_preview_frame` / `render_tae_frame` as module attributes (same pattern
#: `h3_48gb.checkpoint` and others use), and a dict built once at import time would keep pointing
#: at the original functions instead of picking up the patch.
DECODER_NAMES = ("vae", "tae", "latent")


#: Decoders whose "weights absent" notice has already been printed this process. Absent weights
#: are a supported configuration, not an incident, so the notice is worth exactly one line.
_ANNOUNCED: dict[str, bool] = {}


def emit_preview(
    pipeline,
    generated_video_rows: mx.array,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    patch_size,
    stem: str | Path,
    step: int,
    verbose: bool = True,
    decoder: str = "vae",
) -> bool:
    """Best-effort preview: try the requested decoder, fall back to the latent heatmap, never raise.

    This is the only function in this module the diffusion loop should call directly. A generation
    can run for over fifteen hours; nothing a preview does is worth risking that run, so every
    failure here is caught, logged to stderr (regardless of ``verbose`` — a silently-broken preview
    defeats its own purpose), and swallowed.

    ``decoder`` selects the renderer: ``"vae"`` (default, unchanged) is the real video VAE;
    ``"tae"`` is the 9.8 MB approximation with no chunk floor; ``"latent"`` skips straight to the
    VAE-free heat map that is otherwise only the fallback. Whichever primary decoder is chosen, on
    failure this falls back to the latent heat map before giving up — and when the primary choice
    is ``"tae"`` or ``"latent"``, that fallback also never touches ``pipeline.video_vae``, so a
    TAE preview never costs anything from the real VAE even when it is failing.

    The one deliberate exception to "never raises": an unknown ``decoder`` name raises
    ``ValueError`` immediately, before any work starts. That is a programming error caught ahead of
    time, not the mid-run decode failure this function exists to swallow — silently decoding with
    something other than what the caller asked for would be worse than raising.

    Returns:
        Whether a JPEG was written (for tests and for the caller's own logging; the diffusion loop
        does not need to act on this).
    """
    if decoder not in DECODER_NAMES:
        raise ValueError(
            f"Unknown preview decoder {decoder!r}; expected one of {sorted(DECODER_NAMES)}."
        )

    dest = preview_path(stem, step)
    started = time.perf_counter()
    # Looked up as plain globals, not through a dict built once at import time, so that tests
    # monkeypatching `render_preview_frame` / `render_tae_frame` as module attributes still take
    # effect (see the note on `DECODER_NAMES`). "latent" has no primary renderer of its own — it
    # *is* the fallback below, so it goes straight there.
    render = {"vae": render_preview_frame, "tae": render_tae_frame, "latent": None}[decoder]
    if render is not None:
        try:
            frame = render(
                pipeline, generated_video_rows, num_latent_frames, latent_height, latent_width,
                patch_size,
            )
            save_preview_jpeg(frame, dest)
            if verbose:
                print(f"  [preview] step {step}: wrote {dest.name} via {decoder} "
                      f"({time.perf_counter() - started:.1f}s)", flush=True)
            return True
        except FileNotFoundError as exc:
            # Not a failure: the TAE weights are optional and previews default to TAE, so a reader
            # who never downloaded them takes this path on every preview of every run. Said once
            # per process, because six identical stack-adjacent lines per run is how a log stops
            # being read.
            if not _ANNOUNCED.get(decoder):
                _ANNOUNCED[decoder] = True
                print(f"  [preview] {decoder} weights not found ({exc}); using the latent heat map "
                      "for previews. See README's Previews section to fetch them.",
                      file=sys.stderr, flush=True)
        except Exception as exc:  # noqa: BLE001 - a broken preview must never break the real run
            print(f"  [preview] step {step}: {decoder} decode failed ({exc!r}), "
                  f"falling back to a latent-only preview", file=sys.stderr, flush=True)

    try:
        if decoder == "vae":
            # Unchanged from before `decoder` existed: the VAE path is allowed to read its own
            # config for the fallback's channel count, same as it always has.
            latent_channels = pipeline.video_vae.config.latent_channels
        else:
            # "tae" and "latent" must never touch the video VAE, including in their own fallback
            # — that is the whole point of choosing them. Use TAE's own constant instead, which is
            # tested elsewhere to match the video VAE's `latent_channels` exactly.
            from h3_48gb.tae import LATENT_CHANNELS
            latent_channels = LATENT_CHANNELS
        frame = render_latent_fallback(
            generated_video_rows, num_latent_frames, latent_height, latent_width,
            latent_channels, patch_size,
        )
        save_preview_jpeg(frame, dest)
        if verbose:
            print(f"  [preview] step {step}: wrote {dest.name} (latent fallback, "
                  f"{time.perf_counter() - started:.1f}s)", flush=True)
        return True
    except Exception as exc:  # noqa: BLE001 - see above
        print(f"  [preview] step {step}: fallback preview also failed ({exc!r}), skipping",
              file=sys.stderr, flush=True)
        return False


# -- the loop hook ---------------------------------------------------------------------------

class PreviewInterceptor:
    """Wraps ``pipeline.dit`` so a preview is rendered every ``every`` completed steps.

    Installed as ``pipeline.dit`` for the duration of one ``__call__`` and restored afterwards —
    see the module docstring for why this is enough to hook the loop without reimplementing it.
    It never changes what the transformer computes or returns: this is an observer, not a filter.

    ``n_cond_v`` — the number of leading rows that are keyframe conditioning rather than generated
    video — has to be supplied by the caller because it is not derivable from anything this wrapper
    sees; it is cheap to compute from the same public call arguments upstream uses
    (``len(keyframe_anchors) * rows_per_frame``), which is what `LazyMiniMaxH3Pipeline.__call__`
    does before installing this wrapper.
    """

    def __init__(
        self,
        dit,
        pipeline,
        every: int,
        stem: str | Path,
        n_cond_v: int,
        num_latent_frames: int,
        latent_height: int,
        latent_width: int,
        patch_size,
        verbose: bool = True,
        decoder: str = "vae",
    ):
        self._dit = dit
        self._pipeline = pipeline
        self._every = every
        self._stem = stem
        self._n_cond_v = n_cond_v
        self._num_latent_frames = num_latent_frames
        self._latent_height = latent_height
        self._latent_width = latent_width
        self._patch_size = patch_size
        self._verbose = verbose
        self._decoder = decoder
        self._forward_index = -1

    def __getattr__(self, item):
        # Only reached for attributes not found on this wrapper itself (normal attribute lookup
        # already covers everything set in `__init__`). Forwarded so `dit.config.patch_size` and
        # similar still resolve through this wrapper without a load, same contract as
        # `h3_48gb.pipeline.LazyComponent`.
        if item.startswith("__"):
            raise AttributeError(item)
        return getattr(self._dit, item)

    def _step_index(self) -> int:
        """The forward's true position in the whole run.

        If `h3_48gb.checkpoint` is also active and has already fast-forwarded past some steps,
        `ResumableRun.forward_index` (advanced by the outer `StepInterceptor` before it ever calls
        into this wrapper) is the real count; this wrapper's own counter only sees the forwards
        that actually reach it, which would otherwise undercount by however many steps were
        skipped. Read defensively — `_resume_run` is `h3_48gb.checkpoint` internals, not a contract
        this module depends on, so any shape change just falls back to independent counting rather
        than breaking.
        """
        run = getattr(self._pipeline, "_resume_run", None)
        forward_index = getattr(run, "forward_index", None)
        if isinstance(forward_index, int) and forward_index >= 0:
            return forward_index
        self._forward_index += 1
        return self._forward_index

    def __call__(self, video_latents, audio_latents, *args, **kwargs):
        step = self._step_index()
        if step > 0 and step % self._every == 0:
            generated = video_latents[0, self._n_cond_v :]
            emit_preview(
                self._pipeline, generated, self._num_latent_frames, self._latent_height,
                self._latent_width, self._patch_size, self._stem, step, verbose=self._verbose,
                decoder=self._decoder,
            )
        return self._dit(video_latents, audio_latents, *args, **kwargs)
