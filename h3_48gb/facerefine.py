"""The windowed video-to-video refine engine -- the GPU half of the face-refine pipeline.

It takes the 448x288 crops `facepaste.crop_window` cut around a tracked face (Task 2), denoises
them in overlapping 56-frame windows on a *partial* sigma schedule, crossfades the windows back
into one continuous clip, and hands the result back for `facepaste.paste_back` to composite onto
the source frames. Nothing here knows about faces, files or ffmpeg: in goes a `(F, H, W, 3)` uint8
stack, out comes a `(F, H, W, 3)` uint8 stack of the same shape.

**None of the v2v mechanics below were invented here.** They are the three experiment rounds
recorded in `docs/FEASIBILITY-face-refine.md`, ported from the scripts that ran them
(`~/h3-outputs/face-refine-exp/v2v_face_refine.py`, `.../bake_partial.py`,
`.../round2/overlap/crossfade.py`). Every constant in this module is a measurement, and the
docstrings say which round measured it. Re-deriving any of them from taste would silently undo a
day of GPU time.

**Why a partial schedule and not keyframe conditioning.** Keyframe rows are noised once to
`t = 0.999` and then *held* there for every step -- they anchor a clip, they are never denoised
(see `h3_48gb/pipeline.py`'s module docstring, and § A of the feasibility doc). Face-refine needs
the opposite: the crop's own latent has to live in the *generated* row block and be denoised from
a low starting sigma down to 0. That is `scale_noise(x0, t = 1 - sigma)` into `video_rows`, with
`n_cond_v = 0` -- simpler than i2v, not harder -- against an AdaLN table baked for a grid that
starts at that sigma instead of at 1.0 (`ensure_partial_table`).

**Why sigma is capped at 0.25.** Round 2 ran the same crop at 0.15 / 0.25 / 0.40 on a genuinely
soft face. At 0.25 eyelids, lashes, brows, nostrils and teeth assemble out of mush and the face
stays the same face. At 0.40 the model *invents*: it opened an eye with a full iris where the
source had a squint, and the "engraved" over-sharpened look of round 1 came back. 0.25 is not a
default a caller may raise -- above it the pass stops being a refine and starts being a
generation, so `refine_clip` raises rather than obeying.

**Why windows overlap and crossfade instead of butting together.** A production shot is 3-7
windows. Round 3 put two windows over the same footage (0-55 and 28-83, bit-identical input in
the overlap) and measured what they disagree on: the *weight* and contrast of invented detail,
not its position -- geometry is pinned by the shared source latent, so the windows are
co-registered by construction. A hard cut still ticks (+1.41 sigma on a motion-compensated step
metric); a 12-frame pixel crossfade removes it completely, with no ghosting, because averaging
two versions of one eyebrow gives an eyebrow rather than two. Hence `crossfade >= 12`, an overlap
of 12-16 frames is enough, and the resulting step of 42 costs 35% less GPU than the 28 of the
experiment.

**Why the phases are ordered encode-all / denoise-all / decode-all rather than per window.**
`pipeline._decode_video` releases the transformer before it decodes -- correct for a single long
generation, where the decode is the memory peak, and wrong here, where it would make every window
after the first pay a full DiT load. Rows are tiny (a 56-frame window is ~0.5 MB of latent rows),
so the whole plan's worth fits in memory with room to spare, and running each stage to completion
for every window before moving on costs exactly one text-encoder load, one VAE-encode residency,
one DiT residency and one VAE-decode residency for a clip of any length. The peak is unchanged:
it is the 28.2 GB text encoder (round 1 measured 28.4 GB total), which is released before the
first latent is encoded.

**Ordering discipline around the RNG.** This codebase has a documented history of seeds drawn on
the wrong side of a lazy module build (`h3_48gb/pipeline.py:293-331`: two identical keyframes
encoding 0.83 apart). Both loops below therefore materialize the component they need *before*
seeding -- `video_vae.load()` then `mx.random.seed(KEYFRAME_ENCODE_SEED)`, `dit.load()` then
`mx.random.seed(seed)` -- because building a module draws from MLX's global stream. The denoise
seed is re-set per window on purpose (round 3 ran both windows on one seed): every window then
sees the same noise realization over co-registered content, and a window's output does not depend
on how many windows preceded it in the plan.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

#: Denoise strength ceiling, measured in round 2. See the module docstring: above this the pass
#: invents expression rather than resolving detail. `refine_clip` raises instead of clipping --
#: silently running a different strength than asked for is how a user concludes "face-refine is
#: broken" about a parameter they never got.
SIGMA_CEILING = 0.25
DEFAULT_SIGMA = 0.25

#: 56 = 17*3 + 5, exactly on the VAE's native `17n + 5` pixel-frame grid -> 17 latent frames, no
#: padding anywhere in the stack (round 1). A window off this grid makes `video_latent_num_frames`
#: raise, which is the correct outcome and the reason `plan_windows` checks it up front.
WINDOW_FRAMES = 56
FRAMES_PER_CHUNK = 17          # the VAE's own chunk length; the grid is `17n + 5`
LATENTS_PER_CHUNK = 5          # ... and its remainder, i.e. the shortest legal clip

#: Round 3: an overlap of 12-16 frames is enough for the crossfade, and stepping 42 instead of the
#: experiment's 28 costs 35% less GPU on the same footage. 56 - 42 = 14 frames of overlap.
WINDOW_STEP = 42
CROSSFADE_FRAMES = 12

#: The partial table's grid: four sigmas from the requested one down to 0, i.e. three forwards
#: (~17 s each on a 56-frame crop; the grid point at 0 is spent on the terminal sigma, see
#: `supported_num_inference_steps`).
GRID_POINTS = 4

#: Where the baked tables, the AdaLN time curve and the Turbo LoRA all live.
DEFAULT_ADALN_DIR = Path.home() / "models/turbo"
TURBO_LORA_NAME = "minimax_h3_turbo_v4_step600_ema.safetensors"
CURVE_NAME = "adaln_curve.safetensors"
SILU_GRID_NAME = "h3_silu_temb_grid.safetensors"

#: The video VAE's spatial compression. 448/16 = 28 and 288/16 = 18, which is why `facepaste`'s
#: `out_size` is what it is -- both sides land exactly on the latent grid with no edge padding.
VAE_SPATIAL_RATIO = 16

#: Shifts the two modalities' sigma grids are placed with. Video 12, audio 3 -- the pair the model
#: was trained on, and the reason `partial_sigmas` builds one *unshifted* grid and shifts it twice
#: rather than building two grids: the (video_t, audio_t) pairs must stay on the training diagonal.
VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0

#: The conditioning text. The refine has no idea what the shot is about, and the crop is a face by
#: construction, so the prompt describes *that* and nothing else -- the experiments' own prompts
#: (round 1's blonde against a blue sky, round 2's `village` / `ridge`) were per-scene because a
#: human wrote one per clip. A caller that knows the source clip's own prompt should pass it:
#: conditioning that matches the footage is strictly better than this generic stand-in, and the
#: `prompt` argument exists for exactly that.
DEFAULT_PROMPT = (
    "integrated_multimodal_description: [Shot 1] Live-action, cinematic, an extreme close-up of a "
    "human face, sharply in focus. Fine skin texture, individual eyelashes, eyebrow hairs and "
    "strands of hair are clearly resolved; the eyes are sharp and alive.\n\n"
    "overall_soundscape: quiet room tone, a faint breath of air."
)


@dataclass(frozen=True)
class Window:
    """One v2v pass over `frames[start:start + length]` of the crop stack.

    `length` is always on the native `17n + 5` grid; `start` is not constrained beyond staying
    inside the clip. Frozen and comparable so `plan_windows` can be asserted against a literal
    list in tests -- the plan is the part of this module a reader has to be able to check by eye.
    """

    start: int
    length: int

    @property
    def end(self) -> int:
        """One past the last frame, i.e. a Python slice bound."""
        return self.start + self.length


# -- the native frame grid -------------------------------------------------------------------

def native_frame_count(num_frames: int) -> int:
    """The largest ``17n + 5`` count that fits in `num_frames`.

    Rounded *down*, never up: rounding up would need frames the clip does not have, and padding
    them (repeating the last frame, which is what the VAE itself does internally) would feed the
    refine a freeze-frame and then paste its invention over real footage. `plan_windows` uses this
    as its effective window length on a clip shorter than `window`, then covers the rest of the
    clip with further windows at this same truncated length -- see its docstring.
    """
    if num_frames < LATENTS_PER_CHUNK:
        raise ValueError(
            f"A refine window needs at least {LATENTS_PER_CHUNK} frames (the shortest length on "
            f"the VAE's native 17n+5 grid), got {num_frames}."
        )
    chunks = (num_frames - LATENTS_PER_CHUNK) // FRAMES_PER_CHUNK
    return chunks * FRAMES_PER_CHUNK + LATENTS_PER_CHUNK


# -- window planning -------------------------------------------------------------------------

def plan_windows(num_frames: int, window: int = WINDOW_FRAMES, step: int = WINDOW_STEP,
                 crossfade: int = CROSSFADE_FRAMES) -> list[Window]:
    """Lay passes over `num_frames`, stepping `step`, tail pinned to the end. Covers every frame.

    One algorithm regardless of how `num_frames` compares to `window`: the effective window length
    is ``w = min(window, native_frame_count(num_frames))`` -- `window` itself on a clip that has
    enough frames for it, the clip's own truncated-to-the-grid length on one that does not (round
    1's rule: truncate down rather than pad a freeze-frame up to the grid). Every window in the
    returned plan is `w` frames long.

    Starts land at ``0, advance, 2*advance, ...`` for as long as a full `w`-frame window still fits
    strictly inside the clip, then one final window is **pinned to the tail**
    (``start = num_frames - w``), where ``advance = min(step, w)``: on a clip long enough for a
    full `window`, `w` equals `window` and `advance` equals `step` (already required to be no
    larger), so nothing changes there; on a shorter clip `w` can be well under `step`, and
    advancing by the raw `step` would then jump clean over frames no window's span reaches --
    capping the advance at `w` is what keeps consecutive windows touching or overlapping instead
    of leaving a gap. Pinning rather than padding the tail is why every window is a full-length
    native pass; the price is that the last overlap is usually wider than ``window - step``, never
    narrower, which the crossfade only benefits from. The pinned window can never duplicate its
    predecessor: the loop only emits a start `s` while ``s + w < num_frames``, i.e. while
    ``s < num_frames - w``, so the pinned start is strictly greater than every start already
    emitted -- and because consecutive starts are never more than `w` apart while the tail start is
    always within `w` of the last emitted start, the plan covers every frame from 0 to `num_frames`
    with no gap, on a clip of any length >= 5.

    A middle window whose predecessor already overlaps the pinned tail by a full crossfade is
    redundant -- it can add as little as one new frame for a full window's GPU cost -- so it is
    dropped: while there are at least three starts and ``(starts[-3] + w) - starts[-1] >=
    crossfade``, `starts[-2]` is removed. (`plan_windows(99)` would otherwise land a window at 42
    that contributes past nothing the 0- and 43-start windows do not already cover between them;
    dropping it gives ``[0, 43]`` with a 13-frame overlap, still >= `crossfade`.)
    """
    if window % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK:
        raise ValueError(
            f"`window` must be on the VAE's native grid (17n + 5: 5, 22, 39, 56, 73 ...), got "
            f"{window}. Off-grid windows cannot be packed into whole latent frames."
        )
    if not 1 <= step <= window:
        raise ValueError(f"`step` must be between 1 and `window` ({window}), got {step}.")
    if crossfade < 0:
        raise ValueError(f"`crossfade` cannot be negative, got {crossfade}.")
    if crossfade > window - step:
        raise ValueError(
            f"A crossfade of {crossfade} frames does not fit the {window - step}-frame overlap a "
            f"window of {window} stepped by {step} leaves. Lower `crossfade` or `step`."
        )

    w = min(window, native_frame_count(num_frames))
    # `step` is validated against `window`, not against `w`: on a clip shorter than `window` the
    # effective window `w` can be far smaller than `step`, and advancing by the raw `step` would
    # then jump clean over frames no window's span reaches -- a real gap, not merely a seam wider
    # than the crossfade wants. Capping the advance at `w` is a no-op whenever `w == window`
    # (`step <= window` is already required), so the >= 56-frame regime is unaffected.
    advance = min(step, w)

    starts: list[int] = []
    start = 0
    while start + w < num_frames:
        starts.append(start)
        start += advance
    starts.append(num_frames - w)                # the tail window, pinned to the last frame

    while len(starts) >= 3 and (starts[-3] + w) - starts[-1] >= crossfade:
        del starts[-2]

    return [Window(s, w) for s in starts]


# -- crossfade compositing -----------------------------------------------------------------------

def _fade_weights(fade: int) -> np.ndarray:
    """The reference ramp, verbatim from `round2/overlap/crossfade.py`: ``(i + 1) / (fade + 1)``.

    Note what it is *not*: it never reaches 0 or 1 inside the ramp. The frame before the ramp is
    pure A and the frame after it is pure B, so the endpoints are already covered; spending ramp
    frames repeating them would make the transition effectively two frames shorter than asked for.
    """
    return (np.arange(1, fade + 1, dtype=np.float32) / np.float32(fade + 1))


def composite_windows(source: np.ndarray, refined: Sequence[np.ndarray], plan: Sequence[Window],
                      crossfade: int = CROSSFADE_FRAMES) -> np.ndarray:
    """Blend the per-window refined frames into one clip the shape of `source`.

    Windows are laid down in order onto an accumulator that starts as `source`, so:

    * frames no window covers keep their source pixels **byte for byte**. `plan_windows` itself
      now always covers every frame (>= 5), so this only matters for a plan a caller hand-builds
      with a gap in it;
    * the first window overwrites its whole span;
    * every later window hands over from the accumulated result across a `crossfade`-frame ramp
      **centred in the overlap** -- the placement round 3 measured, where an overlap of 28 and a
      fade of 12 put the ramp on frames 36..47 -- then owns the rest of its span outright.

    Compositing against the *accumulator* rather than pairwise against the previous window is what
    makes a pinned tail window safe when it overlaps two predecessors at once (it does, on clip
    lengths where the last regular window nearly reaches the end): by the time it is laid down,
    everything behind it is already one settled image.

    The ramp shrinks to fit an overlap narrower than `crossfade` instead of reading outside the
    window. On clips of a window or longer `plan_windows` never produces such an overlap, but on
    sub-window clips it legitimately can: a 6-frame clip fits only 5-frame windows, and no
    algorithm can carve a 12-frame ramp out of them. There the shrink is the designed behaviour,
    not a fallback -- the R3 "crossfade >= 12" requirement is about seams between full windows,
    and a clip that short has no full windows to seam.
    """
    if len(refined) != len(plan):
        raise ValueError(f"{len(refined)} refined windows for a plan of {len(plan)}.")
    for window, frames in zip(plan, refined):
        if frames.shape[0] != window.length:
            raise ValueError(
                f"Window {window} was refined into {frames.shape[0]} frames, not {window.length}.")
        if frames.shape[1:] != source.shape[1:]:
            raise ValueError(
                f"Window {window} came back as {frames.shape[1:]}, but the source frames are "
                f"{source.shape[1:]}.")

    out = source.astype(np.float32)
    covered_end = 0
    for index, (window, frames) in enumerate(zip(plan, refined)):
        block = frames.astype(np.float32)
        if index == 0:
            out[window.start:window.end] = block
            covered_end = window.end
            continue

        overlap = max(0, min(covered_end, window.end) - window.start)
        fade = min(crossfade, overlap)
        fade_start = window.start + (overlap - fade) // 2
        if fade:
            weight = _fade_weights(fade).reshape(-1, 1, 1, 1)
            held = out[fade_start:fade_start + fade]
            incoming = block[fade_start - window.start:fade_start - window.start + fade]
            out[fade_start:fade_start + fade] = (1.0 - weight) * held + weight * incoming
        out[fade_start + fade:window.end] = block[fade_start + fade - window.start:]
        covered_end = max(covered_end, window.end)

    # `+ 0.5` then truncate is round-to-nearest, the same convention `pipeline._decode_video` uses
    # on its way out of the VAE -- and it leaves untouched frames bit-identical, because an exact
    # integer plus a half still truncates to itself.
    return (np.clip(out, 0.0, 255.0) + 0.5).astype(np.uint8)


# -- the partial AdaLN table -----------------------------------------------------------------

def partial_table_name(sigma: float, points: int = GRID_POINTS, turbo: bool = True) -> str:
    """The file name `bake_partial.py` gave the tables already sitting in `~/models/turbo/`.

    Two decimals of sigma, zero padded: 0.25 -> ``adaln_face_s025_4pt_turbo.safetensors``. The
    name is the cache key, so `refine_clip` quantizes its sigma to the same two decimals before
    anything else -- otherwise 0.249 and 0.250 would share a table and the run's *effective*
    strength (which comes from the grid inside the file, adopted verbatim by `_build_schedules`)
    would quietly differ from the one asked for.
    """
    tag = f"s{int(round(sigma * 100)):03d}_{points}pt" + ("_turbo" if turbo else "")
    return f"adaln_face_{tag}.safetensors"


def _unshift(sigma_shifted: float, shift: float) -> float:
    """Invert ``s' = shift*s / (1 + (shift-1)*s)``."""
    return sigma_shifted / (shift - (shift - 1.0) * sigma_shifted)


def partial_sigmas(sigma_start: float, points: int, shift: float,
                   video_shift: float = VIDEO_SHIFT) -> np.ndarray:
    """`points` sigmas from `sigma_start` (given in the **video** shifted domain) down to 0.

    Ported from `bake_partial.py`. The grid is built in the *unshifted* domain and shifted per
    modality afterwards, which is the whole trick: video (shift 12) and audio (shift 3) then come
    off one base grid and every ``(video_t, audio_t)`` pair sits where a training step sat. Build
    the two grids independently and the joint forward is fed a pair the model has never seen.

    For sigma 0.25 this is ``[0.250, 0.180, 0.098, 0.0]``, the grid the shipped
    `adaln_face_s025_4pt_turbo.safetensors` was baked for.
    """
    base_start = np.float32(_unshift(sigma_start, video_shift))
    base = np.linspace(base_start, 0.0, points, dtype=np.float32)
    shift32 = np.float32(shift)
    shifted = (shift32 * base) / (np.float32(1.0) + np.float32(shift - 1.0) * base)
    return shifted.astype(np.float32)


def _bake_inputs_present(adaln_dir: Path, lora: Path) -> None:
    """Fail before the baker does, naming the file that is missing and where it comes from.

    Same discipline as `cli.py`'s `checkpoint_not_found`: a `FileNotFoundError` thrown from inside
    `mx.load` three frames deep says which path was missing but not what it *is* or how to get it.
    """
    missing = [path for path in (adaln_dir / CURVE_NAME, adaln_dir / SILU_GRID_NAME, lora)
               if not path.exists()]
    if missing:
        raise RuntimeError(
            "Cannot bake the partial AdaLN table for face-refine: "
            + ", ".join(str(path) for path in missing) + " missing. "
            f"{CURVE_NAME} and {SILU_GRID_NAME} are the pruned base's folded time curve and its "
            "silu grid (see docs/FEASIBILITY-turbo-tae.md and scripts/fetch_adaln_curve.py); "
            f"{TURBO_LORA_NAME} is the Turbo LoRA whose AdaLN half is folded into the table. All "
            "three normally live in ~/models/turbo/."
        )


def _bake_partial_table(sigma: float, dest: Path, points: int, lora: Path, strength: float,
                        verbose: bool) -> None:
    """Bake one partial table by driving `scripts/bake_adaln.py` through its `tail-split` seam.

    `bake()` is not hard-coded to full 1.0-starting schedules -- it takes whatever
    `tail_split_sigmas` returns -- so a partial grid needs no change to the script at all, only
    the substitution `bake_partial.py` already used for three rounds of experiments. The
    substitution is a module global, restored in a `finally` so a caller that later bakes an
    ordinary tail-split table in the same process gets the real function back.

    `CURVE` / `SILU_GRID` are redirected at `adaln_dir` alongside the grid, so a caller pointing
    `adaln_dir` somewhere other than `~/models/turbo` gets a self-contained directory rather than
    a table baked from tensors somewhere else.
    """
    bake_adaln = _load_bake_adaln()
    original_sigmas = bake_adaln.tail_split_sigmas
    original_curve, original_grid = bake_adaln.CURVE, bake_adaln.SILU_GRID

    def custom(steps, shift, split):        # the signature of `tail_split_sigmas`
        return partial_sigmas(sigma, points, shift)

    if verbose:
        print(f"  baking {dest.name}: video grid "
              f"{[round(float(v), 3) for v in partial_sigmas(sigma, points, VIDEO_SHIFT)]}, "
              f"audio grid "
              f"{[round(float(v), 3) for v in partial_sigmas(sigma, points, AUDIO_SHIFT)]}")
    bake_adaln.tail_split_sigmas = custom
    bake_adaln.CURVE = dest.parent / CURVE_NAME
    bake_adaln.SILU_GRID = dest.parent / SILU_GRID_NAME
    try:
        bake_adaln.bake(points, dest, schedule="tail-split", lora=lora, strength=strength)
    finally:
        bake_adaln.tail_split_sigmas = original_sigmas
        bake_adaln.CURVE, bake_adaln.SILU_GRID = original_curve, original_grid


def _load_bake_adaln():
    """Import `scripts/bake_adaln.py`, which is a script directory rather than a package.

    Loaded by path instead of by `sys.path` surgery so importing this module never changes how
    anything else in the process resolves imports.
    """
    import importlib.util
    import sys

    if "bake_adaln" in sys.modules:
        return sys.modules["bake_adaln"]
    path = Path(__file__).resolve().parent.parent / "scripts" / "bake_adaln.py"
    spec = importlib.util.spec_from_file_location("bake_adaln", path)
    if spec is None or spec.loader is None:                      # pragma: no cover - unreachable
        raise RuntimeError(f"Cannot load the AdaLN baker from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules["bake_adaln"] = module
    spec.loader.exec_module(module)
    return module


def ensure_partial_table(sigma: float, checkpoint: str | Path | None = None,
                         adaln_dir: str | Path = DEFAULT_ADALN_DIR, *,
                         points: int = GRID_POINTS, turbo_lora: str | Path | None = None,
                         strength: float = 1.0, verbose: bool = True) -> Path:
    """Return the partial AdaLN table for `sigma`, baking it into `adaln_dir` if it is not there.

    Baking costs 0.7 s of CPU and writes 87 MB, so this is a cache, not a build step -- the four
    tables the experiments produced (s015, s025, s030, s040, s050) are already on disk and a
    default run finds one rather than making one.

    `checkpoint` is accepted and unused: the table is reconstructed from the pruned base's folded
    time curve (`adaln_curve.safetensors`), never from the checkpoint's own weights -- the whole
    point of the curve is that the 13B modulation path this build ships without does not have to
    exist. It stays in the signature because it is how Task 4 was specified to call this, because
    every other entry point in this module takes it, and because a future non-Turbo table would
    need the DiT's real `time_embedder` and therefore the checkpoint.

    `strength` is the LoRA strength baked into the table's modulation rows -- distinct from, but
    meant to travel with, the strength `apply_backbone_lora` applies to the backbone weights
    themselves at denoise time. `refine_clip` has no `turbo_strength` parameter and always calls
    this with the default `strength=1.0`, which is also what the backbone always gets, so the two
    never disagree; a caller baking a table directly (as the bare-metal LoRA-strength experiments
    did) is the only path that can set this to anything else, and the resulting file's name still
    only encodes `sigma`, not `strength` -- do not point two different strengths at one `adaln_dir`
    for the same sigma.
    """
    adaln_dir = Path(adaln_dir).expanduser()
    dest = adaln_dir / partial_table_name(sigma, points)
    if dest.exists():
        return dest

    lora = Path(turbo_lora).expanduser() if turbo_lora else adaln_dir / TURBO_LORA_NAME
    _bake_inputs_present(adaln_dir, lora)
    adaln_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    _bake_partial_table(sigma, dest, points, lora, strength, verbose)
    if verbose:
        print(f"  baked {dest.name} in {time.perf_counter() - started:.1f}s")
    return dest


# -- the engine ------------------------------------------------------------------------------

def _validate_request(crops, sigma: float) -> float:
    """Everything that can be refused without a GPU, refused before one is touched.

    Loading the checkpoint costs 20 s and 28 GB; a bad sigma or a mis-shaped crop stack must not
    get that far. Returns the sigma quantized to the two decimals `partial_table_name` keys on --
    see there for why the quantization has to happen before the table is chosen.
    """
    if isinstance(sigma, bool):
        # `bool` is an `int` subclass in Python -- `isinstance(True, (int, float))` is true, so
        # this has to be checked and refused *before* the numeric check below would silently wave
        # `sigma=True` through as `1.0`.
        raise ValueError(f"`sigma` must be a positive denoise strength, got {sigma!r}.")
    if isinstance(sigma, (np.floating, np.integer)):
        # A caller doing sigma arithmetic in numpy (e.g. off an array of candidate strengths) gets
        # `np.float32`/`np.float64` back, not a Python `float`; cast before the `isinstance` check
        # below rejects it for a reason that has nothing to do with the value being invalid.
        sigma = float(sigma)
    if not isinstance(sigma, (int, float)) or not sigma > 0.0:
        raise ValueError(f"`sigma` must be a positive denoise strength, got {sigma!r}.")
    if sigma > SIGMA_CEILING:
        raise ValueError(
            f"`sigma` is capped at {SIGMA_CEILING} for face-refine, got {sigma}. Round 2 measured "
            "0.40 inventing expression -- opening an eye the source had squinting -- and the "
            "engraved over-sharpening of round 1 coming back. Above the cap this stops being a "
            "refine, so it is refused rather than clipped."
        )
    quantized = round(float(sigma), 2)
    if not quantized > 0.0:
        # Checked again *after* rounding: `sigma=0.004` clears the raw `> 0.0` check above but
        # rounds to the `0.00` table name, which would silently run a no-op refine instead of the
        # weak-but-real one the caller asked for.
        raise ValueError(
            f"`sigma` rounds to 0.0 at the two-decimal precision the table cache keys on, got "
            f"{sigma!r}. Use a sigma of at least 0.005."
        )
    if not isinstance(crops, np.ndarray) or crops.ndim != 4 or crops.shape[-1] != 3:
        raise ValueError(
            f"`crops` must be a (frames, height, width, 3) array, got "
            f"{getattr(crops, 'shape', type(crops).__name__)}.")
    if crops.dtype != np.uint8:
        raise ValueError(f"`crops` must be uint8 RGB, as `facepaste.crop_window` returns it, got "
                         f"{crops.dtype}.")
    height, width = crops.shape[1:3]
    if height % VAE_SPATIAL_RATIO or width % VAE_SPATIAL_RATIO:
        raise ValueError(
            f"The crop is {width}x{height}; both sides must be multiples of "
            f"{VAE_SPATIAL_RATIO} (the video VAE's spatial compression). `facepaste`'s default "
            "448x288 is.")
    return quantized


def _encode_window(pipe, frames: np.ndarray, mx, packing) -> "object":
    """Real pixels -> normalized latents, with the posterior discipline `_encode_keyframes` uses.

    Ported from the reference. Three details that are not obvious and all three matter: the
    posterior is *sampled*, not taken at its mode; the sample is rounded through float16 because
    the diffusers reference's own numerical path does; and the seed is the fixed
    `KEYFRAME_ENCODE_SEED`, independent of the request seed, so the same crop always encodes to
    the same latent no matter what the caller asked for.

    The caller must have loaded the VAE already -- building it draws from the global RNG stream,
    which would move the posterior sample if it happened after the seed.
    """
    cfg = pipe.video_vae.config
    pixel_mean = np.array(packing.PIXEL_MEAN, np.float32).reshape(1, 3, 1, 1, 1)
    pixel_std = np.array(packing.PIXEL_STD, np.float32).reshape(1, 3, 1, 1, 1)
    pixels = frames.astype(np.float32).transpose(3, 0, 1, 2)[None]      # (1, 3, F, H, W)
    pixels = (pixels / 255.0 - pixel_mean) / pixel_std

    mx.random.seed(packing.KEYFRAME_ENCODE_SEED)
    moments = pipe.video_vae.encode(mx.array(pixels))
    channels = cfg.latent_channels
    mean, logvar = moments[:, :channels], moments[:, channels:]
    logvar = mx.clip(logvar, -30.0, 20.0)
    latent = mean + mx.exp(0.5 * logvar) * mx.random.normal(mean.shape)
    latent = latent.astype(mx.float16).astype(mx.float32)               # the reference round trip
    latents_mean = mx.array(np.array(cfg.latents_mean, np.float32)).reshape(1, -1, 1, 1, 1)
    latents_std = mx.array(np.array(cfg.latents_std, np.float32)).reshape(1, -1, 1, 1, 1)
    normalized = (latent - latents_mean) / latents_std
    # Materialize before the caller drops the VAE: an unevaluated graph over its parameters pins
    # all 5.21 GB of them no matter what is unloaded (`pipeline._release_vae_after`).
    mx.eval(normalized)
    return normalized


def refine_clip(crops: np.ndarray, *, sigma: float = DEFAULT_SIGMA, seed: int = 42,
                window: int = WINDOW_FRAMES, step: int = WINDOW_STEP,
                crossfade: int = CROSSFADE_FRAMES, checkpoint: str | Path,
                adaln_dir: str | Path = DEFAULT_ADALN_DIR, prompt: str | None = None,
                turbo_lora: str | Path | None = None,
                verbose: bool = True,
                progress: Callable[[str], None] | None = None) -> np.ndarray:
    """Refine a stack of face crops, window by window, and return frames of the same shape.

    `crops` is `facepaste.crop_window`'s own output: `(frames, 288, 448, 3)` uint8 RGB, one crop
    per source frame, cut with a rectangle that is a function of the *frame* index and not of the
    window (round 3: window-local geometry destroys the co-registration the crossfade depends on).
    The return value is the same shape and dtype, ready for `facepaste.paste_back`.

    `sigma` is the denoise strength, capped at 0.25 (see `SIGMA_CEILING`) and quantized to two
    decimals to match the table cache. `seed` is one seed for the whole pass, re-set per window.
    `checkpoint` is the model directory (`~/models/h3-8bit-full`); `adaln_dir` holds the partial
    tables, the time curve and the Turbo LoRA. `prompt` defaults to a generic close-up-of-a-face
    description -- pass the source clip's own prompt when it is known.

    The backbone LoRA is always applied at strength 1.0 -- no round of experiments varied it, and
    `ensure_partial_table`'s own `strength` (which bakes the LoRA's AdaLN half into the table
    itself) is likewise always called at 1.0 here, so the two never disagree. There is no
    `turbo_strength` parameter; pass a different LoRA via `turbo_lora` instead of trying to dial
    this one down.

    Raises `ValueError` for anything refusable without a GPU and `RuntimeError` when the table on
    disk does not describe the schedule its name claims.
    """
    sigma = _validate_request(crops, sigma)
    plan = plan_windows(len(crops), window=window, step=step, crossfade=crossfade)
    table = ensure_partial_table(sigma, checkpoint, adaln_dir, turbo_lora=turbo_lora,
                                 verbose=verbose)

    def say(message: str) -> None:
        if verbose:
            print(message, flush=True)
        if progress is not None:
            progress(message)

    say(f"face-refine: {len(crops)} frames, {len(plan)} window(s) of {plan[0].length} "
        f"(step {step}, crossfade {crossfade}), sigma {sigma}, table {table.name}")

    refined = _run_windows(crops, plan, table, sigma=sigma, seed=seed, checkpoint=checkpoint,
                           adaln_dir=Path(adaln_dir).expanduser(), prompt=prompt or DEFAULT_PROMPT,
                           turbo_lora=turbo_lora, turbo_strength=1.0, verbose=verbose,
                           say=say)
    return composite_windows(crops, refined, plan, crossfade=crossfade)


def _reset_window_schedule(sched, grid_sigmas: list[float]) -> None:
    """Clear a scheduler's step counter so a new window starts its own Euler walk at index 0.

    `_run_windows` builds `video_sched`/`audio_sched` once, in phase 2 before the loop over
    windows -- `_ensure_cache`'s AdaLN table only needs to be built once, and that is what forces
    the schedulers to be built early too. But `MiniMaxH3Scheduler._step_index` is cleared only by
    `set_timesteps` (`upstream/minimax_h3_mlx/scheduler.py`), not by finishing a run of `.step()`
    calls, so without this reset window 2 continues stepping from window 1's final index, walks
    off the end of the sigma grid (`sigmas[step_index + 1]` past the last entry), and divides
    ``0.0 / 0.0`` into every latent -- Task 5's D1: a black rectangle from the second window on.
    `set_timesteps(sigmas=...)` is exactly the call `MiniMaxH3Pipeline._build_schedules` already
    makes from the cached grid (`h3_48gb/pipeline.py`), so calling it again per window reuses that
    same logic instead of introducing a second way to seed a schedule.
    """
    sched.set_timesteps(sigmas=grid_sigmas)


def _run_windows(crops, plan, table, *, sigma, seed, checkpoint, adaln_dir, prompt, turbo_lora,
                 turbo_strength, verbose, say) -> list[np.ndarray]:
    """The GPU half: one text encode, one VAE-encode pass, one denoise pass, one decode pass.

    Split out of `refine_clip` so that everything above it -- validation, planning, table
    selection -- is reachable in a test without MLX, the checkpoint or 28 GB of RAM. Every heavy
    import lives inside this function for the same reason.
    """
    from . import _upstream  # noqa: F401  (puts upstream/ on sys.path)

    import mlx.core as mx
    from minimax_h3_mlx import packing

    from . import memory
    from .pipeline import LazyMiniMaxH3Pipeline
    from .turbo import apply_backbone_lora

    started = time.perf_counter()
    lora = Path(turbo_lora).expanduser() if turbo_lora else adaln_dir / TURBO_LORA_NAME
    if not lora.exists():
        # Checked before the checkpoint is even opened: `load_with_lora` below only runs when
        # `pipe.dit.load()` fires in phase 2, by which point the text encoder (28 GB, ~9 s) and the
        # VAE encode have already happened. A missing LoRA is a request-shape problem, not a GPU
        # one, so it fails in milliseconds here instead of ~30 s and 21 GB in.
        raise RuntimeError(
            f"Turbo LoRA not found at {lora}. Face-refine's backbone pass needs it; pass "
            "`turbo_lora=` to point at a different file, or place "
            f"{TURBO_LORA_NAME} in `adaln_dir`."
        )
    pipe = LazyMiniMaxH3Pipeline.from_pretrained(str(checkpoint), verbose=verbose,
                                                 adaln_cache=table)
    inner = pipe.dit.__dict__["_loader"]

    def load_with_lora():
        dit = inner()
        apply_backbone_lora(dit, lora, strength=turbo_strength, verbose=verbose)
        return dit

    pipe.dit.__dict__["_loader"] = load_with_lora

    # -- the schedule the table was baked for ---------------------------------------------------
    grids = pipe._cached_sigma_grids()
    if grids is None:
        raise RuntimeError(f"{table} is not readable as an AdaLN cache; delete it and let "
                           "`ensure_partial_table` bake it again.")
    video_sched, audio_sched = pipe._build_schedules(len(grids[0]))
    sigma_start_v = float(video_sched.sigmas[0].item())
    if abs(sigma_start_v - sigma) > 1e-3:
        raise RuntimeError(
            f"{table.name} is named for sigma {sigma} but its video grid starts at "
            f"{sigma_start_v:.4f}. The name is the cache key, so a mismatched file would run a "
            "strength nobody asked for; delete it and let it be baked again."
        )
    say(f"  schedule: {[round(float(v), 4) for v in video_sched.sigmas.tolist()]} "
        f"({len(grids[0]) - 1} forwards per window)")

    # -- geometry from the crop, not from a duration --------------------------------------------
    length = plan[0].length
    if any(w.length != length for w in plan):        # every window shares one packed layout
        raise RuntimeError(f"Mixed window lengths in one plan: {plan}.")
    height, width = crops.shape[1:3]
    num_latent_frames = packing.video_latent_num_frames(length)
    ratio = pipe.video_vae.config.spatial_compression_ratio
    latent_height, latent_width = height // ratio, width // ratio
    num_audio_latents = packing.audio_latent_num_frames(length)
    patch_size = pipe.dit.config.patch_size

    # -- conditioning: encoded once, reused by every window --------------------------------------
    # The text encoder is 28.2 GB and the run's memory peak; per-window encoding would pay it once
    # per window for a prompt that never changes. `LazyTextEncoder` releases it on the way out.
    prompt_embeds, text_token_tags = pipe.text_encoder.encode(prompt, None)
    layout = packing.build_packed_sequence(text_token_tags, num_latent_frames, latent_height,
                                           latent_width, num_audio_latents, patch_size, ())
    say(f"  packed sequence: {layout.sequence_length:,} rows ({len(text_token_tags):,} text)")

    # -- phase 1: encode every window's pixels ---------------------------------------------------
    pipe.video_vae.load()                    # before any seeding; see the module docstring
    latents = []
    for window in plan:
        clean = _encode_window(pipe, crops[window.start:window.end], mx, packing)
        if clean.shape[2] != num_latent_frames:
            raise RuntimeError(f"The VAE returned {clean.shape[2]} latent frames for a "
                               f"{window.length}-frame window; packing expects {num_latent_frames}.")
        latents.append(clean)
    pipe.video_vae.unload()
    memory.release()
    say(f"  encoded {len(latents)} window(s) to {tuple(latents[0].shape)}")

    # -- phase 2: denoise every window ------------------------------------------------------------
    pipe.dit.load()                          # before the seed, for the same reason as the VAE
    timestep_table, row_plan = pipe._row_timestep_plan(layout, video_sched.timesteps,
                                                       audio_sched.timesteps)
    pipe._ensure_cache(timestep_table, True, verbose)
    embeds = prompt_embeds.astype(mx.bfloat16)
    audio_shape = (num_audio_latents * packing.AUDIO_CHANNELS,
                   pipe.audio_vae.config.latent_channels)

    denoised = []
    for index, window in enumerate(plan, 1):
        # Announced *before* the forward passes, not after: a caller timing an ETA off this
        # message (Task 6) needs to see window `k` start, not learn about it only once it has
        # already finished.
        say(f"  window {index}/{len(plan)} frames {window.start}-{window.end - 1}")
        tick = time.perf_counter()
        # Both schedulers walk the same 4-point grid on every window; reset the step counter each
        # time so window 2+ does not pick up where the previous window's Euler walk left off (D1).
        _reset_window_schedule(video_sched, grids[0])
        _reset_window_schedule(audio_sched, grids[1])
        # One seed for the whole pass, re-set per window: the same noise realization over
        # co-registered content, and a window that does not depend on its position in the plan.
        mx.random.seed(seed)
        noise = mx.random.normal(latents[index - 1].shape).astype(mx.float32)
        noised = video_sched.scale_noise(latents[index - 1], 1.0 - sigma_start_v, noise)
        video_rows = packing.patchify_video_latents(noised, patch_size)
        # Pure noise for audio. Round 1 compared it against the source clip's own encoded audio
        # and the difference was inside the run-to-run noise floor (1.39x/28.89 dB against
        # 1.40x/28.92 dB), so the audio VAE is never loaded and the decoded audio is discarded.
        # The rows themselves cannot be dropped: the two modalities are denoised in one forward.
        audio_rows = mx.random.normal(audio_shape).astype(mx.float32)
        mx.eval(video_rows, audio_rows)

        for stepno, t in enumerate(video_sched.timesteps.tolist()):
            video_pred, audio_pred = pipe.dit(
                video_rows[None].astype(mx.bfloat16),
                audio_rows[None].astype(mx.bfloat16),
                embeds,
                timestep_table,
                row_plan[stepno],
                layout.token_tags,
                layout.position_ids,
                layout.video_indices,
                layout.audio_indices,
                layout.text_indices,
                modulation_cache=pipe._cache,
            )
            video_rows = video_sched.step(video_pred[0].astype(mx.float32), float(t), video_rows)
            audio_rows = audio_sched.step(audio_pred[0].astype(mx.float32),
                                          float(audio_sched.timesteps[stepno].item()), audio_rows)
            mx.eval(video_rows, audio_rows)
        denoised.append(video_rows)
        say(f"  window {index}/{len(plan)} denoised in {time.perf_counter() - tick:.0f}s")

    # -- phase 3: decode -------------------------------------------------------------------------
    # `_decode_video` releases the transformer on its first call; every window after that decodes
    # with it already gone, which is why the denoise loop above had to finish first.
    refined = []
    for window, rows in zip(plan, denoised):
        frames = pipe._decode_video(rows, num_latent_frames, latent_height, latent_width)
        if frames.shape[0] < window.length:
            raise RuntimeError(f"The VAE decoded {frames.shape[0]} frames for a "
                               f"{window.length}-frame window.")
        refined.append(frames[:window.length])
    memory.release()
    say(f"face-refine: {len(plan)} window(s) in {(time.perf_counter() - started) / 60:.1f} min, "
        f"peak {mx.get_peak_memory() / 1e9:.1f} GB")
    return refined
