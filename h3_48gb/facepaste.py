"""Crop a window around a tracked face and paste a refined result back -- the CPU-only geometry
half of the face-refine pipeline, sitting between `facetrack.FaceTrack` (Task 1) and the GPU v2v
engine that actually improves the face (Task 3, a later module).

**Why the crop rectangle needs its own record, separate from `FaceTrack.box`.** `FaceTrack.box`
is a *smooth float function of frame index* by design (see `facetrack.py`'s module docstring) --
that smoothness is the whole point of the track, and this module must not break it by rounding
inconsistently from one caller to the next. But the pixels `crop_window` actually reads are never
that exact float rectangle: they are `scale` times it, expanded to `out_size`'s own aspect ratio
(see "why the window is expanded to `out_size`'s aspect" below), shifted and possibly shrunk to fit
inside the frame (`_clip_to_frame`), and then rounded to integers to become real array slice indices
(`_round_rect` plus `_clamp_int_origin`). `CropGeometry` is the record of that *real* rectangle, per
frame -- the one `paste_back` must use to land refined pixels back where they were actually cut
from, not the geometric ideal `FaceTrack.box` still reports for the same frame.

**Why the window is expanded to `out_size`'s aspect ratio (review finding I3).** A YuNet face box
is roughly portrait (about 1:1.25, tall relative to its width), but `out_size` is a fixed landscape
resolution (e.g. `448x288`, 1.556:1) chosen for the refine model, not for the face's own shape. The
first version of this module resized the portrait-scaled window straight into that landscape
`out_size`, which is a non-uniform stretch -- it visibly doubled the face's apparent width. The fix
(`_expand_to_aspect`) widens or heightens the *shorter* side of the `scale`-times-box window,
symmetric about its own center, until its aspect ratio matches `out_size`'s, before any clipping or
resizing happens -- so the eventual `cv2.resize` to `out_size` is always a uniform scale, and the
face's proportions survive. This runs before `_clip_to_frame`, which still may need to shrink the
now-wider-or-taller window if it no longer fits the source frame.

**Why the rounding is two functions, not "just round it" (review finding I1).** The first version
of `_round_rect` rounded the crop rectangle's two *edges* (`x0` and `x0 + w`) independently, on the
theory that this would keep a smoothly moving rectangle's width from jittering. It does the
opposite: a rectangle whose *position* drifts smoothly, even with a perfectly constant size, sees
its two edges' fractional parts cross integer rounding boundaries at different frames, so the
integer *width* recovered as their difference flips by +-1 on roughly every other frame -- 28
changes across 59 transitions on a real smooth track, measured on review, reproduced here as 44
changes across 60 frames in `test_crop_window_rect_size_is_stable_on_a_smooth_track_not_edge_
rounding_noise`. That is a ~0.9% frame-to-frame "breathing" scale jitter feeding straight into the
refine model's input, which breaks Task 3's co-registration between adjacent crop windows. The fix
is `_round_rect` rounding `x0`, `y0`, `w`, and `h` each exactly ONCE (not derived from independently
rounded edges), followed by `_clamp_int_origin`, which shifts the rounded rectangle's *origin* (only
the origin, never its already-rounded size) back inside the frame bounds -- needed because rounding
`w`/`h` and `x0`/`y0` independently can still push the result up to one pixel past a bound a
float-clipped rectangle already respected (see `_clamp_int_origin`'s own docstring for the exact
argument and a constructed overflow case).

**Why `_clip_to_frame` shifts instead of intersecting (controller decision, kept as-is).** A crop
rectangle that spills off a frame edge is pulled back inside by *shifting* it (keeping its full
`w x h` size, as long as that size itself fits the frame) rather than by intersecting it with the
frame bounds (which would shrink it, changing its scale). This was a deliberate choice, not an
oversight: a face-refine pipeline that will eventually do video-to-video across many adjacent crop
windows (Task 3) needs those windows to carry a *consistent scale* far more than it needs their
framing to be perfectly centered on the face at the very edge of a shot -- a window that randomly
shrinks by a few percent whenever the face nears an edge would be a second, unrelated source of the
same kind of scale jitter I1 already had to fix. Shrinking only happens when the window is larger
than the whole frame and there is genuinely no shift that could make it fit (see `_clip_to_frame`'s
own docstring).

**Why the paste is masked and fades, not a flat rectangle swap.** A hard-edged rectangle paste
would show a visible seam at the crop boundary the instant the refined pixels differ from the
source even slightly -- the entire reason ComfyUI-style face-restoration nodes paste through a
feathered mask instead of a bare crop (see the design spec's ComfyUI-FaceRefine reference).
`_feather_mask` softens the outer `feather` fraction of each side into a linear ramp -- except on
whichever side(s) of the rectangle coincide with the frame's own border (review Minor finding): a
rect edge that *is* the picture's own edge is not a seam, and feathering it anyway would leave the
outer ~10% of a full-frame paste un-refined for no reason. `_fade_multipliers` additionally
collapses that mask toward zero, over `FADE_FRAMES` frames, on any stretch where `FaceTrack.detected`
says there is no real evidence nearby -- an extrapolated box far from the last real detection is a
guess, and guessing at where to paste face-refined pixels is worse than leaving the original frame
alone.

**Why the downscale/upscale interpolation is chosen per-direction, not fixed (review finding I2).**
`cv2.INTER_AREA` is a real area-averaging algorithm only when *shrinking* an image -- asked to
*enlarge* one, it silently falls back to the same bit-exact output as `cv2.INTER_NEAREST` (verified
directly). `paste_back`'s downscale of `refined` (shaped like `out_size`) into the crop rect's own
`(w, h)` is only a downscale when the rect is smaller than `out_size`; on a 1080p source the rect is
usually native-resolution and therefore *larger* than `out_size`, making that resize an upscale --
which the original single-`INTER_AREA` choice silently turned into nearest-neighbor across the
entire pasted region. `_paste_interpolation` picks `INTER_AREA` for a shrink and `INTER_LANCZOS4`
for a stretch, by comparing the source and destination areas.

Deliberately no dependency on `mlx`, same reasoning as `facetrack.py` and `queue.py`: this is a
CPU-only geometry/blending step, and importing it must never pull in the 33B-parameter transformer
stack. See `test_facepaste_module_does_not_import_mlx`.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

#: How many frames `paste_back`'s fade_out ramps the blend down over, once a frame leaves the
#: neighborhood of a real detection (`FaceTrack.detected` false). Fixed, not a parameter: the
#: design spec's parameter table gives this as a measured constant ("линейный fade_out вклейки за
#: 6 кадров"), the same status as `facetrack`'s `_SCORE_THRESHOLD` -- a number from experiments,
#: not something a caller is expected to retune per clip.
FADE_FRAMES = 6


@dataclass(frozen=True)
class CropGeometry:
    """The pixel-exact crop rectangle `crop_window` actually used on every frame -- see the
    module docstring's "why the crop rectangle needs its own record" for why this is not simply
    recomputed from `FaceTrack.box` at paste time.

    `rects` is `(n_frames, 4)` int32, one `(x, y, w, h)` row per frame index -- top-left corner
    and extent, in the source frame's own pixel coordinates, the same convention `FaceTrack.box`
    uses. `out_size` is `(width, height)` the crop was resized to (`crop_window`'s own `out_size`
    argument, echoed back so `paste_back` knows what shape `refined` frames must be without
    threading a second parameter through every caller).
    """

    rects: np.ndarray
    out_size: tuple[int, int]

    @property
    def n_frames(self) -> int:
        return int(self.rects.shape[0])

    def _check_index(self, frame_idx: int) -> None:
        if not 0 <= frame_idx < self.n_frames:
            raise IndexError(f"frame_idx {frame_idx} out of range [0, {self.n_frames})")

    def rect(self, frame_idx: int) -> tuple[int, int, int, int]:
        """`(x, y, w, h)`, the real integer pixel rectangle `crop_window` read frame `frame_idx`
        from -- always inside the source frame's bounds (see `_clip_to_frame`).
        """
        self._check_index(frame_idx)
        x, y, w, h = self.rects[frame_idx]
        return (int(x), int(y), int(w), int(h))


def _clip_to_frame(cx: float, cy: float, w: float, h: float,
                    frame_w: int, frame_h: int) -> tuple[float, float, float, float]:
    """The `w x h` rectangle centered on `(cx, cy)`, shrunk to fit inside a `frame_w x frame_h`
    frame if it is larger than the frame in either dimension, then shifted (never shrunk further)
    so both edges land inside `[0, frame_w]` / `[0, frame_h]`.

    Shrinking happens before shifting on purpose: a `scale`-times-the-face-box rectangle can
    legitimately be bigger than the whole frame (a face that nearly fills the shot, scaled up by
    2.75x), and there is no shift that fits an oversized rectangle inside a frame it cannot fit
    in -- the only honest answer there is the whole frame. A rectangle that already fits just
    gets pushed back inside the bounds, keeping its own size, which is the "клип у краёв кадра"
    (clip at the frame edges) the brief asks for: the crop follows the face right up to the edge
    of frame instead of the box wandering off it.

    This is a *shift*, not an *intersection*, and that is deliberate (see the module docstring's
    "why `_clip_to_frame` shifts instead of intersecting"): a rectangle that already fits keeps its
    exact size no matter how close to the frame edge it is pushed, so the crop window's scale stays
    constant across a whole clip -- which matters for Task 3's video-to-video co-registration
    between adjacent windows -- and only shrinks in the one case (bigger than the whole frame) where
    no shift could possibly make it fit anyway.
    """
    w = min(w, float(frame_w))
    h = min(h, float(frame_h))
    x0 = cx - w / 2.0
    y0 = cy - h / 2.0
    x0 = min(max(x0, 0.0), frame_w - w)
    y0 = min(max(y0, 0.0), frame_h - h)
    return x0, y0, w, h


def _expand_to_aspect(w: float, h: float, target_w: int, target_h: int) -> tuple[float, float]:
    """Expand the *shorter* side of a `w x h` rectangle outward -- symmetric about its own center,
    since the caller always feeds the result straight into `_clip_to_frame`, which centers on
    `(cx, cy)` -- until its aspect ratio matches `target_w / target_h`. The longer side is never
    touched.

    See the module docstring's "why the window is expanded to `out_size`'s aspect" for the bug this
    fixes: without it, a portrait face-box window resized straight into a landscape `out_size`
    stretches non-uniformly and visibly distorts the face.
    """
    target_aspect = target_w / target_h
    current_aspect = w / h
    if current_aspect < target_aspect:
        # Too narrow for the target aspect: widen `w`, keep `h`.
        return h * target_aspect, h
    if current_aspect > target_aspect:
        # Too wide (short) for the target aspect: heighten `h`, keep `w`.
        return w, w / target_aspect
    return w, h


def _round_rect(x0: float, y0: float, w: float, h: float) -> tuple[int, int, int, int]:
    """`(x0, y0, w, h)` rounded to integer pixel indices: `round()` (Python's own round-half-to-
    even) applied to `x0`, `y0`, `w`, and `h` -- each rounded exactly ONCE, not derived from
    independently rounding two edges (`x0` and `x0 + w`) and taking their difference.

    See the module docstring's "why the rounding is two functions" note for the bug this fixes:
    rounding `x0` and `x0 + w` independently let the two edges' rounding *error* drift apart from
    one frame to the next even when `w` itself was not changing, producing visible width jitter on
    an otherwise perfectly smooth track. Rounding `w` once and reusing it removes that source of
    jitter entirely -- the rect's integer size now only changes on the rare frames where the
    underlying float `w`/`h` itself crosses a rounding boundary.

    `round()`'s ties-to-even policy (not "always round half up") is deliberate too: a track's box
    drifts continuously across many frames and will cross a `.5` boundary many times over a long
    clip, and "always round up" would accumulate a directional bias in the crop's average position
    that ties-to-even does not.

    The result can land up to half a pixel past an ideal `x0 + w`/`y0 + h` bound if the input was
    exactly frame-edge-exact before rounding -- `_clamp_int_origin` is what pulls the *origin* back
    inside the frame afterward, using this already-rounded `w`/`h`; this function does not attempt
    that itself since it is never given the frame size to clamp against.
    """
    return round(x0), round(y0), round(w), round(h)


def _clamp_int_origin(ix0: int, iy0: int, iw: int, ih: int,
                       frame_w: int, frame_h: int) -> tuple[int, int]:
    """Shift an already-rounded `(ix0, iy0, iw, ih)` rectangle's *origin* (never its size) so both
    edges land inside `[0, frame_w] x [0, frame_h]`.

    This exists because `_round_rect` rounds `x0`/`y0` and `w`/`h` independently: `round(x0) <= x0 +
    0.5` and `round(w) <= w + 0.5`, so `round(x0) + round(w)` can land up to one pixel past
    `frame_w` even when the original float rectangle fit inside `[0, frame_w]` exactly (see
    `test_clamp_int_origin_pulls_the_origin_back_inside_bounds_after_rounding_overflow` for a
    constructed case with `x0 = 1.5, w = 7.5, frame_w = 9`, where the ties-to-even rule rounds both
    values up and the origin needs pulling back by one pixel).

    Assumes `iw <= frame_w` and `ih <= frame_h`: true here because `_clip_to_frame` already shrank
    `w`/`h` to fit inside the frame in float before `_round_rect` ever saw them, and rounding a
    value `<= frame_w` (`frame_w` itself an integer) cannot round above `frame_w`. That keeps the
    clamp range `[0, frame_w - iw]` from ever being empty.
    """
    ix0 = min(max(ix0, 0), frame_w - iw)
    iy0 = min(max(iy0, 0), frame_h - ih)
    return ix0, iy0


def crop_window(frames, track, scale: float = 2.75,
                 out_size: tuple[int, int] = (448, 288)) -> tuple[np.ndarray, CropGeometry]:
    """Crop a `scale`-times-the-face-box window around `track`'s smoothed box on every frame of
    `frames`, expanded to `out_size`'s own aspect ratio (`_expand_to_aspect`, see the module
    docstring), Lanczos-resize each crop to `out_size`, and return the stacked crops alongside the
    `CropGeometry` `paste_back` will need to undo this.

    `frames` is consumed one frame at a time via plain iteration -- a list, a NumPy `(n, H, W, 3)`
    array, or any other single-pass iterable -- since nothing here needs random access into it:
    `track.box(i)` already carries everything about frame `i`'s geometry, computed once up front
    in Task 1. `out_size` is `(width, height)`, matching this fork's own `WxH` convention (e.g.
    `h3-<tag>-448x288.mp4`) and `cv2.resize`'s own `dsize` argument, which it is passed to
    directly.

    The crop window starts as `scale * (w, h)` centered on `track.box(i)`'s own center -- not
    `scale * out_size` or any other independent size, because the window must track the face's
    actual size in the source frame -- and is then widened or heightened (never cropped) to match
    `out_size`'s own aspect ratio before clipping, so the final resize to `out_size` is a uniform
    scale rather than a distorting stretch.

    Raises `ValueError` if `frames` yields a different number of frames than `track.n_frames`:
    `track.box`/`track.detected` are only meaningful over `[0, track.n_frames)`, and a caller that
    hands this function frames from a different pass over the source video (a re-decode with a
    dropped frame, say) needs to hear about the mismatch here, not get a silently truncated or
    out-of-range result.
    """
    out_w, out_h = out_size
    n_frames = track.n_frames
    rects = np.empty((n_frames, 4), dtype=np.int32)
    crops = np.empty((n_frames, out_h, out_w, 3), dtype=np.uint8)

    consumed = 0
    for frame_idx, frame in enumerate(frames):
        if frame_idx >= n_frames:
            raise ValueError(
                f"frames has more frames than track.n_frames={n_frames}"
            )
        frame_h, frame_w = frame.shape[:2]
        x, y, w, h = track.box(frame_idx)
        cx, cy = x + w / 2.0, y + h / 2.0
        exp_w, exp_h = _expand_to_aspect(scale * w, scale * h, out_w, out_h)
        cx0, cy0, cw, ch = _clip_to_frame(cx, cy, exp_w, exp_h, frame_w, frame_h)
        ix, iy, iw, ih = _round_rect(cx0, cy0, cw, ch)
        if iw <= 0 or ih <= 0:
            raise ValueError(
                f"frame {frame_idx}: crop rect collapsed to {iw}x{ih}, nothing to crop"
            )
        ix, iy = _clamp_int_origin(ix, iy, iw, ih, frame_w, frame_h)
        rects[frame_idx] = (ix, iy, iw, ih)
        patch = frame[iy:iy + ih, ix:ix + iw]
        crops[frame_idx] = cv2.resize(patch, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
        consumed = frame_idx + 1

    if consumed != n_frames:
        raise ValueError(f"frames has {consumed} frames but track.n_frames={n_frames}")

    return crops, CropGeometry(rects=rects, out_size=(out_w, out_h))


def _feather_ramp(length: int, feather: float, fade_start: bool = True,
                   fade_end: bool = True) -> np.ndarray:
    """A 1-D alpha ramp of `length` samples, rising linearly to a flat 1.0 over the outer
    `feather` fraction of `length`, on each end that `fade_start`/`fade_end` says should fade.

    Sampled at pixel *centers* (`i + 0.5`), not integer pixel indices, so the ramp is symmetric
    around the rectangle's true center regardless of `length`'s parity -- an integer-index ramp
    would be off-center by half a pixel on an even-length axis. `feather <= 0` (or a `length` too
    short for any band) returns a flat 1.0 everywhere: a zero-width soft border is just a hard
    rectangle, which is a valid, if unrequested, choice for a caller rather than a special case
    this function needs to refuse.

    `fade_start`/`fade_end` (both default `True`, reproducing the original always-feather-both-ends
    behavior) let a caller suppress the ramp on one or both ends -- used when that end of the crop
    rectangle coincides with the *frame's own* border rather than an internal seam (review Minor
    finding, see the module docstring's "why the paste is masked and fades" section): there is
    nothing to hide there, so ramping it down anyway would leave that edge of a full-frame paste
    un-refined for no reason.
    """
    if length <= 0:
        return np.zeros(0, dtype=np.float64)
    if not fade_start and not fade_end:
        return np.ones(length, dtype=np.float64)
    band = feather * length
    if band < 1e-9:
        return np.ones(length, dtype=np.float64)
    idx = np.arange(length, dtype=np.float64) + 0.5
    start_ramp = idx / band if fade_start else np.full(length, np.inf)
    end_ramp = (length - idx) / band if fade_end else np.full(length, np.inf)
    return np.clip(np.minimum(start_ramp, end_ramp), 0.0, 1.0)


def _feather_mask(height: int, width: int, feather: float, *, fade_top: bool = True,
                   fade_bottom: bool = True, fade_left: bool = True,
                   fade_right: bool = True) -> np.ndarray:
    """A `(height, width)` 2-D alpha mask: the outer product of the two axes' `_feather_ramp`s.

    The outer product (rather than, say, `min` of the two axis ramps) gives the corners a smooth
    radial-looking falloff instead of a sharp diagonal crease -- both are "a feathered rectangle",
    but the product keeps the corner alpha continuous in both partial derivatives, which reads as
    softer over a real face crop's rounded features.

    `fade_top`/`fade_bottom`/`fade_left`/`fade_right` (all default `True`) are forwarded to the two
    axis ramps to suppress feathering on whichever side(s) of the rectangle are actually the frame's
    own border -- see `_feather_ramp`'s docstring. With all four `False` (a rect that fills the
    whole frame on every side) this returns a flat matrix of `1.0`.
    """
    row_ramp = _feather_ramp(height, feather, fade_start=fade_top, fade_end=fade_bottom)
    col_ramp = _feather_ramp(width, feather, fade_start=fade_left, fade_end=fade_right)
    return np.outer(row_ramp, col_ramp)


def _fade_multipliers(track) -> np.ndarray:
    """One multiplier per frame, in `[0, 1]`: `1.0` wherever `track.detected(i)` is true, ramping
    linearly down to `0.0` over `FADE_FRAMES` frames of distance from the nearest frame where it
    is, and staying `0.0` beyond that.

    "Distance" is to the nearest true frame in *either* direction, computed with two linear
    sweeps (forward, then backward, each one only ever lowering a running distance) rather than
    a single-direction "frames since the last detection": `FaceTrack.detected`'s own true regions
    already sit on both sides of every real YuNet anchor (`facetrack.py`'s `_build_track`, `±every`
    around each anchor), so a stretch with no real evidence can end at a true region approached
    from *either* side -- the tail past the last detection, but also, symmetrically, the run-up to
    a detection that comes later. Fading symmetrically means the paste eases in on approach to a
    detection and eases out on departure from one, rather than snapping to full strength the
    instant `detected` flips true -- see
    `test_fade_multipliers_ramps_symmetrically_around_a_gap_between_two_detections` for the case of
    a gap *between* two real detections, ramping down leaving one and back up approaching the
    other.

    `1.0 - distance / FADE_FRAMES` is the requested "linear fade_out over 6 frames": at
    `distance=0` (a true frame itself) the multiplier is the same `1.0` the mask would already
    give it, at `distance=FADE_FRAMES` it reaches exactly `0.0`, and it is clamped there for any
    larger distance so a long stretch with no nearby evidence at all is left completely untouched
    rather than asymptotically approaching zero without ever reaching it.
    """
    n = track.n_frames
    detected = np.array([track.detected(i) for i in range(n)], dtype=bool)

    far = n + FADE_FRAMES + 1  # larger than any real distance this loop can produce
    distance = np.where(detected, 0, far).astype(np.int64)
    for i in range(1, n):
        if distance[i] > distance[i - 1] + 1:
            distance[i] = distance[i - 1] + 1
    for i in range(n - 2, -1, -1):
        if distance[i] > distance[i + 1] + 1:
            distance[i] = distance[i + 1] + 1

    return np.clip(1.0 - distance / FADE_FRAMES, 0.0, 1.0)


def _paste_interpolation(src_w: int, src_h: int, dst_w: int, dst_h: int) -> int:
    """The `cv2.resize` interpolation flag for resizing an `src_w x src_h` patch to `dst_w x
    dst_h`: `cv2.INTER_AREA` when the target is smaller or equal by area (a shrink -- cv2's own
    docs recommend area-averaging over Lanczos here, since Lanczos can ring on a strong downscale),
    `cv2.INTER_LANCZOS4` when the target is strictly larger (a stretch).

    See the module docstring's "why the downscale/upscale interpolation is chosen per-direction"
    for the bug this fixes: `cv2.INTER_AREA` silently produces bit-identical output to
    `cv2.INTER_NEAREST` when asked to enlarge an image rather than shrink one, so a fixed
    `INTER_AREA` choice in `paste_back` turned every upscaling paste (the common case on a
    higher-resolution source, where the crop rect is larger than `out_size`) into blocky
    nearest-neighbor resampling.
    """
    return cv2.INTER_AREA if dst_w * dst_h <= src_w * src_h else cv2.INTER_LANCZOS4


def paste_back(frames, refined, geometry: CropGeometry, track, feather: float = 0.10) -> np.ndarray:
    """Paste `refined` (frames shaped like `crop_window`'s own `out_size` output, one per frame
    of `frames`) back into `frames`, through a feathered mask confined to each frame's
    `geometry.rect`, faded by `_fade_multipliers` wherever `track` has no nearby real detection.

    Every returned frame is a *new* array, and every pixel strictly outside `geometry.rect(i)` is
    copied byte-for-byte from `frames[i]` -- the module docstring's "outside the mask, bit-exact
    source" guarantee holds unconditionally, not just approximately, because those pixels are
    literally never written to. Inside the rect, `refined[i]` is first resized (`cv2.resize`, with
    `_paste_interpolation`'s direction-aware choice of `INTER_AREA`/`INTER_LANCZOS4`, see the module
    docstring) back to the rect's own `(w, h)`, then alpha-blended against the source pixels with
    `_feather_mask(h, w, feather, ...) * _fade_multipliers(track)[i]` as the per-pixel weight --
    `_feather_mask` is told which of the rect's four sides coincide with the frame's own border
    (`geometry.rect`'s `x == 0`, `y == 0`, `x + w == frame_w`, `y + h == frame_h`) so those sides are
    not feathered (review Minor finding).

    A frame whose fade multiplier is exactly `0.0` skips the resize/blend entirely and is a
    straight copy -- not merely "blended with weight zero", which would still visit every pixel of
    the rect and could still be off by a rounding fraction from the source through the float
    arithmetic. This is what makes the fade_out tail genuinely, bit-exactly untouched rather than
    just very close to it.

    Raises `ValueError` if `frames`, `refined` or `geometry` disagree with `track.n_frames` on how
    many frames there are (the same "surface a mismatch here, not as a confusing index error three
    lines down" reasoning as `crop_window`); if any frame's shape differs from `frames[0]`'s; if any
    `refined[i]`'s shape does not match `geometry.out_size` (review finding I5 -- a caller that
    mixes up `(W, H)` and `(H, W)` for either `frames` or `refined` would otherwise get a silently
    distorted paste instead of an error); or if `geometry.rect(i)` does not fit inside frame `i`'s
    own shape.
    """
    n_frames = track.n_frames
    if len(frames) != n_frames:
        raise ValueError(f"frames has {len(frames)} frames but track.n_frames={n_frames}")
    if len(refined) != n_frames:
        raise ValueError(f"refined has {len(refined)} frames but track.n_frames={n_frames}")
    if geometry.n_frames != n_frames:
        raise ValueError(
            f"geometry has {geometry.n_frames} frames but track.n_frames={n_frames}"
        )

    out_w, out_h = geometry.out_size
    expected_refined_shape = (out_h, out_w, 3)
    frame_shape = np.asarray(frames[0]).shape

    fade = _fade_multipliers(track)
    out = np.empty((n_frames, *frame_shape), dtype=np.uint8)

    for frame_idx in range(n_frames):
        source = np.asarray(frames[frame_idx])
        if source.shape != frame_shape:
            raise ValueError(
                f"frame {frame_idx} has shape {source.shape}, but frame 0 has shape "
                f"{frame_shape} -- paste_back requires every frame to share one shape"
            )
        refined_frame = np.asarray(refined[frame_idx])
        if refined_frame.shape != expected_refined_shape:
            raise ValueError(
                f"refined[{frame_idx}] has shape {refined_frame.shape}, expected "
                f"{expected_refined_shape} to match geometry.out_size={geometry.out_size}"
            )
        x, y, w, h = geometry.rect(frame_idx)
        frame_h, frame_w = source.shape[:2]
        if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
            raise ValueError(
                f"frame {frame_idx}: rect {(x, y, w, h)} does not fit inside frame shape "
                f"{source.shape[:2]} (H, W) -- geometry/frames shape mismatch"
            )

        out_frame = source.copy()
        weight = fade[frame_idx]
        if weight > 0.0:
            interp = _paste_interpolation(out_w, out_h, w, h)
            patch = cv2.resize(refined_frame, (w, h), interpolation=interp)
            mask = _feather_mask(
                h, w, feather,
                fade_top=(y != 0), fade_bottom=(y + h != frame_h),
                fade_left=(x != 0), fade_right=(x + w != frame_w),
            )
            mask = (mask * weight)[:, :, None]
            region = out_frame[y:y + h, x:x + w].astype(np.float64)
            blended = region * (1.0 - mask) + patch.astype(np.float64) * mask
            out_frame[y:y + h, x:x + w] = np.clip(blended + 0.5, 0, 255).astype(np.uint8)
        out[frame_idx] = out_frame

    return out
