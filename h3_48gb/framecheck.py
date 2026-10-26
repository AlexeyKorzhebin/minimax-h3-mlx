"""Detect VAE-decode corruption in a frame: zero-filled tiles and tile-seam blocking artifacts.

Боевые ворота 2026-08-19's root cause (chunk-recon investigation, `~/h3-outputs/chunk-
recon/`): `VideoVAE.decode` (`upstream/minimax_h3_mlx/video_vae.py`) chains 13 chunks x 15 tiles —
up to 195 ViT-decoder forwards — into a single lazy MLX graph and materializes it once, at the very
end, with one `np.array()` call. Under allocator/memory pressure, a fraction of that graph's
intermediate buffers could come back as an unwritten (zero) or garbled buffer instead of the tile
actually computed — non-deterministically, invisible to any shape/dtype check, because the corrupt
buffer is still a well-formed array. Two visible symptoms: a tile (or a whole frame) decodes to a
flat fill color, or a tile lands correctly but disagrees with its neighbour hard enough at the seam
to be visible as blocking.

`patches/0003-vae-decode-eval.patch` (`h3_48gb.pipeline.vae_decode_eval_patch_applied`) bounds the
graph at the source, which should make both symptoms unreachable. This module is defense in depth
on top of that, not a redundant check to delete once the patch lands — a decode is expensive
(minutes) and feeds a scene chain (`h3_48gb.assemble`) a silently-corrupt keyframe would poison, so
every frame that reaches a caller is verified rather than trusted.

Both thresholds and the seam-score formula come from the investigation's own scripts, merged: the
tile-seam geometry originally came from `check_mp4.py`'s hardcoded five-column list (calibrated by
eye against one canvas, 896x512); the fixed `2.5` threshold and the zero-fill fraction floor are
`decode_after_unload.py`'s (the investigation's final, most-refined pass, run against the real
allocator state a production decode sees) — see each constant's own docstring for its exact
provenance.

**2026-08-24 fix (`.superpowers/fixes-2026-08-24/task-1-report.md`):** the hardcoded five-column /
two-row list was `check_mp4.py`'s calibration for 896x512 alone. On any other canvas, part of it
fell outside the frame and the surviving coordinates stopped being tile seams while still being
scored as if they were one — this is what stalled боевые ворота 2026-08-20 (two scene chains, ~3h
GPU, on clean 448x288 clips). `tile_seam_score` now computes seam positions from the frame's own
size, using the same tiling arithmetic `video_vae.py::_split_tiles` uses (`_vae_tile_starts`
below, ported to plain Python -- this module must stay `mlx`-free, see `video_vae.py:26-27`), and
returns the clean value `1.0` outright on any canvas too small to give the ratio enough points to
average over (`MIN_SEAM_POINTS`) rather than measuring noise. Measured: 448x288 never has enough
points and is why `zero_fill_fraction` is now that canvas's *only* corruption signal; 896x512 and
1344x768 both clear the floor and keep catching real corruption with wide margin. Full numbers in
the report above.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: RGB a zero (unwritten) decode buffer resolves to once `PIXEL_MEAN`/`PIXEL_STD` unnormalize it
#: and the pipeline's `* 255.0 + 0.5` round-to-uint8 runs — measured directly off a corrupt clip's
#: flat-filled frames (chunk-recon investigation, `decode_after_unload.py`'s `FILL`), not derived
#: symbolically here, so this matches what was actually observed rather than what algebra predicts.
FILL_COLOR = (124, 116, 104)

#: How close an RGB pixel must be to `FILL_COLOR`, per channel, to count as "filled" — matches
#: `decode_after_unload.py`'s own `<= 2` (uint8 rounding slop, not a loose match).
FILL_TOLERANCE = 2

#: Fraction of a frame's pixels that must read as `FILL_COLOR`-filled before the frame counts as
#: zero-fill corrupted. `decode_after_unload.py`'s own "partial" floor (`0.005`, ~0.5% of an
#: 896x512 frame is ~2300 pixels — well under one VAE tile, which covers several percent) — low
#: enough to catch a single unwritten tile without needing the whole frame to be flat, but high
#: enough that no legitimate frame accidentally lands enough pixels on this one exact RGB triple.
ZERO_FILL_FRACTION_THRESHOLD = 0.005

#: The video VAE's own tiling geometry (`upstream/minimax_h3_mlx/video_vae.py`'s
#: `VideoAutoencoder.__init__`, ~459-463): every decode tiles the frame into `VAE_TILE_SIZE`
#: squares overlapping by at least `VAE_TILE_MIN_OVERLAP`. `framecheck` cannot import `video_vae`
#: to read these off the real config (`video_vae.py:26-27` pulls `mlx` at module scope, and this
#: module has to stay numpy-only so `h3_48gb.assemble`'s worker process can import it without a
#: GPU) — these are this module's own copy, fixed for this project's one VAE.
VAE_TILE_SIZE = 256
VAE_TILE_MIN_OVERLAP = 64

#: The step tile-boundary slack is handed out in, so every tile boundary stays latent-aligned
#: (`_split_tiles`'s `remaining // ratio` loop) — `VideoVAEConfig.spatial_compression_ratio`,
#: product of `spatial_downsample_factors`. Same value in the dataclass default (`video_vae.py`'s
#: `[2,2,2,2,1,1]`) and in the real weights' config (`~/models/h3-converted/video_vae/source/
#: config.json`'s `space_down`): 16. Matters for the exact positions (measured different results
#: at 8 vs 16 on 1344x768), so it is its own named constant, not folded into the arithmetic.
VAE_SPATIAL_COMPRESSION_RATIO = 16


def _vae_tile_starts(length: int) -> list[int]:
    """Port of `VideoVAE._split_tiles`'s tile-start arithmetic (plain Python/no `mlx`, this
    module's one allowed dependency is `numpy`) — the smallest number of `VAE_TILE_SIZE` tiles
    that covers `length` while keeping every overlap at least `VAE_TILE_MIN_OVERLAP`, with the
    leftover slack spread round-robin over the overlaps in `VAE_SPATIAL_COMPRESSION_RATIO` steps.

    Returns `[0]` (one tile, no seam at all) when `length` doesn't need tiling.
    """
    tile = VAE_TILE_SIZE
    if tile >= length:
        return [0]
    ratio = VAE_SPATIAL_COMPRESSION_RATIO
    min_overlap = VAE_TILE_MIN_OVERLAP
    num_tiles = math.ceil(length / tile)
    while tile * num_tiles - min_overlap * (num_tiles - 1) - length < 0:
        num_tiles += 1
    overlaps = [min_overlap] * (num_tiles - 1)
    remaining = tile * num_tiles - sum(overlaps) - length
    for i in range(remaining // ratio):
        overlaps[i % (num_tiles - 1)] += ratio
    starts = [0]
    for i in range(num_tiles - 1):
        starts.append(starts[-1] + tile - overlaps[i])
    return starts


def _tile_seam_positions(length: int) -> tuple[int, ...]:
    """Every coordinate along one axis (a frame's width or height) where two VAE decode tiles
    meet — both edges of each overlap region: `_vae_tile_starts(length)[i + 1]` (where the next
    tile starts) and `_vae_tile_starts(length)[i] + VAE_TILE_SIZE` (where the previous tile ends),
    for every adjacent pair. Empty when `length` needs no tiling (`_vae_tile_starts` returns a
    single start) — nothing to measure is not evidence of corruption, same contract as
    `tile_seam_score`'s own "no configured seam fits" case.

    For 896x512 (`h3_48gb.assemble.DEFAULT_SCENE_CANVAS`) this gives `(160, 256, 320, 416, 480,
    576, 640, 736)` on the width axis and `(128, 256, 384)` on the height axis — a strict superset
    of the tuples this replaced (`(160, 320, 480, 640, 736)` / `(128, 256)`, removed
    2026-08-24): those were `check_mp4.py`'s by-eye reading of the same picture, and turned out
    (measured, see the module docstring's fix note) to mix "overlap starts" for columns with
    "overlap starts only, no ends" for rows — no single rule reproduced both, which is why this
    computes every edge instead of guessing which subset someone meant.
    """
    starts = _vae_tile_starts(length)
    if len(starts) < 2:
        return ()
    positions = set(starts[1:])
    positions.update(s + VAE_TILE_SIZE for s in starts[:-1])
    return tuple(sorted(positions))


#: Below this many total measurement points (`_tile_seam_positions(width)` columns plus
#: `_tile_seam_positions(height)` rows, combined), `tile_seam_score` returns the clean value
#: outright instead of computing a ratio — too few points for an average to mean anything rather
#: than just amplify whatever one of them happens to land on.
#:
#: Measured (`.superpowers/fixes-2026-08-24/task-1-report.md` has the full tables, including the
#: fix-round-1 review that raised this from 8): 896x512 gets 11 points (8 columns + 3 rows) and
#: 1344x768 gets 18 (12 + 6) — every clean frame across every available clip at those canvases
#: (732 and 970 frames respectively, all clips on disk, not just a 60-frame sample) scored under
#: 1.6, while the two reference corrupt frames still scored 19-32x over threshold. 448x288 gets
#: only 4 (2 + 2) — and at 4 points, false positives were real and frequent: up to seam_score
#: 3720+ across the 23 available clean 448x288 clips and the two gates-2026-08-20
#: `assembly/final.mp4` outputs, the same failure mode that stalled боевые ворота 2026-08-20 on
#: clean footage.
#:
#: **8-10 points is an unmeasured band, not a verified-safe one.** No real canvas in this
#: project's rotation lands there, so it was never checked against real decode output; a
#: fix-round-1 review measured it on resampled proxy material instead (896x512/1344x768 clips
#: resized to 672x384/896x448/768x432, all landing on 8 or 10 points) and found real false
#: positives there too: 672x384 (8 points) 2/5023 frames over threshold (max 3.29), 896x448 (10
#: points) 1/3456 (max 2.62) — against 0 anywhere at 11+ points on the same material. Not a
#: native decode, but the brief's own criterion is that a single false frame stalls a whole scene
#: chain, so an unverified 8-10-point band is not a safe place to leave the check on. The floor is
#: therefore set at 11 — the smallest point count actually measured clean, not the smallest
#: theoretically distinguishable one. Below it, only `zero_fill_fraction` (canvas-size-independent,
#: catches the same corruption `patches/0003` targets) still watches for corruption.
MIN_SEAM_POINTS = 11

#: Above this, `tile_seam_score`'s ratio of seam-adjacent pixel deltas to a same-tile baseline a
#: few pixels over reads as a real discontinuity rather than picture detail. Chosen empirically by
#: the investigation (`decode_after_unload.py`): clean frames scored <= 1.86, corrupted
#: (tile-boundary garbage) frames scored >= 2.80 across its sample; `2.5` sits in the gap with
#: margin on both sides. That sample was narrower than the docstring here used to claim — the
#: full `corruption-map.csv` (1600 frames) has a real grey zone between 2.80 and 5.59 among frames
#: `decode_after_unload.py` itself didn't call zero-filled — but the reference corrupt frames and
#: every clean frame measured for this fix (see `MIN_SEAM_POINTS`) both sit with wide margin on
#: their respective sides of `2.5`, so it stays unchanged rather than being retuned without cause.
TILE_SEAM_SCORE_THRESHOLD = 2.5


def zero_fill_fraction(frame: np.ndarray) -> float:
    """Fraction of `frame`'s (H, W, 3) pixels within `FILL_TOLERANCE` of `FILL_COLOR`, per channel."""
    diff = np.abs(frame.astype(np.int16) - np.array(FILL_COLOR, dtype=np.int16))
    return float((diff <= FILL_TOLERANCE).all(axis=-1).mean())


def tile_seam_score(frame: np.ndarray) -> float:
    """Energy at the VAE's own tile seams, relative to a same-tile baseline a few pixels away.

    ``1.0`` means the seam pixels differ exactly as much as any other pixels a few columns/rows
    over — a clean image, where nothing marks a tile boundary as special. Values well above 1.0
    mean the discontinuity is concentrated exactly at the seam, which only tile-level decode damage
    produces (legitimate picture content has no reason to break precisely on a VAE tile boundary).
    Ported from chunk-recon's `check_mp4.py` / `decode_after_unload.py` `seam_score`.

    Seam positions are computed from `frame`'s own size (`_tile_seam_positions`), not a fixed
    table for one canvas — see the module docstring's 2026-08-24 fix note. Returns ``1.0`` (the
    "clean" value) if `frame` is too small for any seam to fall inside it, or too small overall
    for the ratio to mean anything (`MIN_SEAM_POINTS`) — nothing reliable to measure is not
    evidence of corruption.
    """
    g = frame.astype(np.float32).mean(axis=-1)
    h, w = g.shape
    columns = _tile_seam_positions(w)
    rows = _tile_seam_positions(h)
    if len(columns) + len(rows) < MIN_SEAM_POINTS:
        return 1.0
    seam, base = [], []
    for x in columns:
        if x - 4 >= 0 and x < w:
            seam.append(np.abs(g[:, x] - g[:, x - 1]).mean())
            base.append(np.abs(g[:, x - 3] - g[:, x - 4]).mean())
    for y in rows:
        if y - 4 >= 0 and y < h:
            seam.append(np.abs(g[y] - g[y - 1]).mean())
            base.append(np.abs(g[y - 3] - g[y - 4]).mean())
    if not seam:
        return 1.0
    baseline = float(np.mean(base)) or 1e-6
    return float(np.mean(seam)) / baseline


def is_frame_corrupt(frame: np.ndarray) -> bool:
    """Whether a single (H, W, 3) uint8 frame trips zero-fill or tile-seam detection.

    The check `h3_48gb.assemble`'s keyframe/freeze-frame extraction runs on one still image at a
    time (there is no clip of neighbouring frames to compare against there).
    """
    return (zero_fill_fraction(frame) > ZERO_FILL_FRACTION_THRESHOLD
            or tile_seam_score(frame) > TILE_SEAM_SCORE_THRESHOLD)


@dataclass(frozen=True)
class FrameCorruption:
    """One corrupt frame `find_corrupt_frames` found, and which detector(s) tripped."""

    index: int
    zero_fill: bool
    seam_score: float

    @property
    def seam(self) -> bool:
        return self.seam_score > TILE_SEAM_SCORE_THRESHOLD


def find_corrupt_frames(frames: np.ndarray) -> list[FrameCorruption]:
    """Every frame in `frames` (N, H, W, 3) uint8 that trips zero-fill or tile-seam detection.

    Used on a decoded clip's full frame stack (`h3_48gb.pipeline._validate_decoded_frames`), where
    there are many frames to report on at once.
    """
    bad = []
    for i in range(frames.shape[0]):
        frame = frames[i]
        zf = zero_fill_fraction(frame) > ZERO_FILL_FRACTION_THRESHOLD
        seam_score = tile_seam_score(frame)
        if zf or seam_score > TILE_SEAM_SCORE_THRESHOLD:
            bad.append(FrameCorruption(index=i, zero_fill=zf, seam_score=seam_score))
    return bad


#: sglang (spec §4.1.7, final review 2026-10-07 C1): a frame counts as broken only when ONE colour
#: -- any colour, not `FILL_COLOR` -- covers at least this share of it. Both MLX detectors are
#: calibrations of the MLX port's VAE and do not transfer: `FILL_COLOR` is an ordinary sand-grey
#: pixel on sglang (597 of 9475 frames of 60 clean h3-bench clips were over the 0.5 % zero-fill
#: floor, the fight-armored-40 arena at 0.0040 of 0.005), and the seam detector is tuned to the MLX
#: tiling. "Filled entirely" is the coordinator's ruling: an unwritten frame is flat end to end,
#: and a night scene or a sky must not come anywhere near this line.
FLAT_FRAME_FRACTION_THRESHOLD = 0.95


def dominant_color_fraction(frame: np.ndarray) -> float:
    """Share of `frame`'s (H, W, 3) pixels within `FILL_TOLERANCE` of the frame's own per-channel
    median colour. When one colour covers more than half of the frame the median *is* that colour
    (to within the tolerance), so a flat frame of any colour reads close to 1.0."""
    pixels = frame.reshape(-1, 3).astype(np.int16)
    median = np.median(pixels, axis=0).round().astype(np.int16)
    return float((np.abs(pixels - median) <= FILL_TOLERANCE).all(axis=-1).mean())


def is_flat_frame(frame: np.ndarray) -> bool:
    """The sglang check: the frame is filled with one colour (`FLAT_FRAME_FRACTION_THRESHOLD`)."""
    return dominant_color_fraction(frame) >= FLAT_FRAME_FRACTION_THRESHOLD


class CorruptFramesError(RuntimeError):
    """Raised when one or more frames fail `is_frame_corrupt`/`find_corrupt_frames` validation."""
