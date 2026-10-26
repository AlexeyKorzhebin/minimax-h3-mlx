"""Crop window / paste back -- the CPU geometry half of the face-refine pipeline.

`crop_window` and `paste_back` never touch a real detector: every track here is built directly
with `facetrack._build_track` (the same seam `test_facetrack.py` uses to test track math without
`cv2`'s YuNet), so this module's own tests are about the crop/paste geometry and blending, not
about detection at all.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.cv
pytest.importorskip("cv2", reason="cv: needs opencv-python and scipy, absent here")
pytest.importorskip("scipy", reason="cv: needs opencv-python and scipy, absent here")


import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from h3_48gb import facepaste as fp
from h3_48gb import facetrack as ft


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------

def _dense_track(n_frames: int, box: tuple[float, float, float, float]) -> ft.FaceTrack:
    """A track holding the same `box` on *every* frame, with `every=1` so `detected()` is True
    everywhere -- the track shape the round-trip and clip tests want: fade never engages, so a
    disagreement between the crop and the paste is purely about the crop/resize/blend geometry
    itself, not about the fade_out ramp `_sparse_track` below is built to exercise.
    """
    samples = [(i, box) for i in range(n_frames)]
    track = ft._build_track(samples, n_frames=n_frames, every=1)
    assert track is not None
    return track


def _sparse_track(n_frames: int, box: tuple[float, float, float, float],
                   anchor: int, every: int) -> ft.FaceTrack:
    """A track with exactly one real detection, at frame `anchor` -- `detected(i)` is True only
    within `every` frames of `anchor` (see `facetrack.FaceTrack.detected`'s own docstring: the
    "near" window is `±every` around each anchor, not just the anchor frame itself) and False
    everywhere else. Callers that want a *single*-frame True island, to keep "distance from
    frame_idx to `anchor`" arithmetic simple in a test, pass `every=0`.
    """
    track = ft._build_track([(anchor, box)], n_frames=n_frames, every=every)
    assert track is not None
    return track


def _synthetic_frames(n_frames: int, height: int, width: int, seed: int = 20260817) -> np.ndarray:
    """`n_frames` RGB uint8 frames with real texture (a smooth gradient plus noise), not a flat
    color: a flat-color frame round-trips a Lanczos resize *exactly*, which would make the
    round-trip test's "close but not bit-exact inside the mask" claim vacuously true instead of
    actually exercising the resize.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    gradient = (
        (xx.astype(np.float64) / max(width - 1, 1)) * 120.0
        + (yy.astype(np.float64) / max(height - 1, 1)) * 80.0
    )
    frames = np.empty((n_frames, height, width, 3), dtype=np.uint8)
    for i in range(n_frames):
        noise = rng.integers(0, 40, size=(height, width, 3))
        channel_shift = np.array([0, 20, 50]) + i  # a little per-frame, per-channel drift
        frame = gradient[:, :, None] + noise + channel_shift[None, None, :]
        frames[i] = np.clip(frame, 0, 255).astype(np.uint8)
    return frames


# --------------------------------------------------------------------------------------------
# `_round_rect` / `_clip_to_frame` -- private rectangle math
# --------------------------------------------------------------------------------------------

def test_round_rect_is_deterministic_and_matches_python_round():
    # x0=10.5 rounds to 10 (round-half-to-even: 10 is even), y0=3.2 rounds to 3. w=20.0 and
    # h=10.0 are already integral and round to themselves.
    assert fp._round_rect(10.5, 3.2, 20.0, 10.0) == (10, 3, 20, 10)


def test_round_rect_rounds_width_and_height_once_not_via_independent_edges():
    # Regression check for review finding I1: rounding x0 and x0+w as two independent edges
    # (round(0.3)=0, round(0.3+10.4)=round(10.7)=11 -> width 11) used to disagree with rounding w
    # directly (round(10.4)=10). The fix rounds w/h once, matching the direct value.
    ix0, iy0, iw, ih = fp._round_rect(0.3, 0.3, 10.4, 10.4)
    assert (ix0, iy0, iw, ih) == (0, 0, 10, 10)


def test_clamp_int_origin_pulls_the_origin_back_inside_bounds_after_rounding_overflow():
    # x0=1.5 and w=7.5 both round UP under ties-to-even (2 and 8 are the even neighbors), even
    # though the float rect (x0=1.5, x0+w=9.0) exactly fits a frame_w=9 frame: round(x0) +
    # round(w) == 10, one pixel past frame_w -- exactly the scenario _clamp_int_origin exists to
    # fix (see _round_rect's own docstring for why rounding x0/w independently can do this).
    ix0, iy0, iw, ih = fp._round_rect(1.5, 0.0, 7.5, 5.0)
    assert ix0 + iw == 10  # confirm the overflow really happens before clamping
    cx0, cy0 = fp._clamp_int_origin(ix0, iy0, iw, ih, frame_w=9, frame_h=100)
    assert cx0 + iw <= 9
    assert cx0 >= 0
    assert cy0 == iy0  # y axis was already in bounds, untouched


def test_clip_to_frame_keeps_a_rect_that_already_fits_centered():
    x0, y0, w, h = fp._clip_to_frame(cx=50.0, cy=50.0, w=20.0, h=10.0,
                                      frame_w=200, frame_h=200)
    assert (x0, y0, w, h) == pytest.approx((40.0, 45.0, 20.0, 10.0))


def test_clip_to_frame_shifts_a_rect_that_would_spill_off_the_top_left():
    x0, y0, w, h = fp._clip_to_frame(cx=2.0, cy=2.0, w=20.0, h=20.0,
                                      frame_w=200, frame_h=200)
    assert x0 == pytest.approx(0.0)
    assert y0 == pytest.approx(0.0)
    assert (w, h) == pytest.approx((20.0, 20.0))


def test_clip_to_frame_shifts_a_rect_that_would_spill_off_the_bottom_right():
    x0, y0, w, h = fp._clip_to_frame(cx=198.0, cy=198.0, w=20.0, h=20.0,
                                      frame_w=200, frame_h=200)
    assert x0 + w == pytest.approx(200.0)
    assert y0 + h == pytest.approx(200.0)


def test_clip_to_frame_shrinks_a_rect_wider_than_the_whole_frame():
    x0, y0, w, h = fp._clip_to_frame(cx=50.0, cy=50.0, w=500.0, h=500.0,
                                      frame_w=100, frame_h=80)
    assert (w, h) == pytest.approx((100.0, 80.0))
    assert x0 == pytest.approx(0.0)
    assert y0 == pytest.approx(0.0)


# --------------------------------------------------------------------------------------------
# `_feather_ramp` / `_feather_mask`
# --------------------------------------------------------------------------------------------

def test_feather_ramp_is_flat_one_in_the_middle_and_near_zero_at_the_edges():
    ramp = fp._feather_ramp(100, feather=0.10)
    assert ramp[0] < 0.1
    assert ramp[-1] < 0.1
    assert ramp[50] == pytest.approx(1.0)
    # Monotonic rise from the left edge into the flat interior.
    assert np.all(np.diff(ramp[:10]) >= 0)


def test_feather_ramp_is_all_ones_when_feather_is_zero():
    ramp = fp._feather_ramp(50, feather=0.0)
    assert np.all(ramp == 1.0)


def test_feather_ramp_does_not_fade_a_side_told_it_is_the_frame_boundary():
    # fade_start=False: the start side should stay flat 1.0 (it's the frame's own edge, not a
    # seam), while the end side still ramps down as normal.
    ramp = fp._feather_ramp(100, feather=0.10, fade_start=False, fade_end=True)
    assert ramp[0] == pytest.approx(1.0)
    assert ramp[50] == pytest.approx(1.0)
    assert ramp[-1] < 0.1


def test_feather_mask_is_the_outer_product_of_the_two_axis_ramps():
    mask = fp._feather_mask(20, 30, feather=0.10)
    assert mask.shape == (20, 30)
    expected = np.outer(fp._feather_ramp(20, 0.10), fp._feather_ramp(30, 0.10))
    assert np.allclose(mask, expected)


def test_feather_mask_is_flat_when_no_side_needs_fading():
    # Minor finding: a rect that fills the whole frame on every side (all four fade_* False)
    # should get a flat 1.0 mask -- there is no seam anywhere to hide.
    mask = fp._feather_mask(20, 30, feather=0.10, fade_top=False, fade_bottom=False,
                             fade_left=False, fade_right=False)
    assert np.all(mask == 1.0)


# --------------------------------------------------------------------------------------------
# `_expand_to_aspect`
# --------------------------------------------------------------------------------------------

def test_expand_to_aspect_widens_a_portrait_box_to_a_landscape_target():
    w, h = fp._expand_to_aspect(40.0, 60.0, target_w=448, target_h=288)
    assert h == pytest.approx(60.0)  # longer side, unchanged
    assert w / h == pytest.approx(448 / 288, rel=1e-6)


def test_expand_to_aspect_heightens_a_landscape_box_to_a_portrait_target():
    w, h = fp._expand_to_aspect(60.0, 40.0, target_w=288, target_h=448)
    assert w == pytest.approx(60.0)  # longer side, unchanged
    assert w / h == pytest.approx(288 / 448, rel=1e-6)


def test_expand_to_aspect_is_a_no_op_when_the_aspect_already_matches():
    w, h = fp._expand_to_aspect(100.0, 50.0, target_w=200, target_h=100)
    assert (w, h) == pytest.approx((100.0, 50.0))


# --------------------------------------------------------------------------------------------
# `_paste_interpolation`
# --------------------------------------------------------------------------------------------

def test_paste_interpolation_picks_area_for_a_shrink_and_lanczos_for_a_stretch():
    # Shrinking (destination area <= source area): INTER_AREA, cv2's own recommendation.
    assert fp._paste_interpolation(src_w=100, src_h=100, dst_w=50, dst_h=50) == cv2.INTER_AREA
    assert fp._paste_interpolation(src_w=100, src_h=100, dst_w=100, dst_h=100) == cv2.INTER_AREA
    # Stretching (destination area > source area): INTER_LANCZOS4, not INTER_AREA -- INTER_AREA
    # degenerates to nearest-neighbor on an upscale (review finding I2).
    assert fp._paste_interpolation(src_w=50, src_h=50, dst_w=100, dst_h=100) == cv2.INTER_LANCZOS4


def test_inter_area_is_bit_identical_to_nearest_when_enlarging():
    # Documents the exact cv2 behavior _paste_interpolation exists to route around: INTER_AREA
    # asked to enlarge silently falls back to nearest-neighbor rather than doing real
    # interpolation.
    frames = _synthetic_frames(1, height=8, width=8)[0]
    area_up = cv2.resize(frames, (32, 32), interpolation=cv2.INTER_AREA)
    nearest_up = cv2.resize(frames, (32, 32), interpolation=cv2.INTER_NEAREST)
    assert np.array_equal(area_up, nearest_up)


# --------------------------------------------------------------------------------------------
# `_fade_multipliers` -- the linear fade_out ramp over FADE_FRAMES
# --------------------------------------------------------------------------------------------

def test_fade_multipliers_is_one_at_and_near_the_one_real_detection():
    track = _sparse_track(n_frames=40, box=(0.0, 0.0, 10.0, 10.0), anchor=20, every=2)
    fade = fp._fade_multipliers(track)
    assert fade[20] == pytest.approx(1.0)
    assert fade[18] == pytest.approx(1.0)  # within `every` of the anchor: detected() is True
    assert fade[22] == pytest.approx(1.0)


def test_fade_multipliers_ramps_down_linearly_over_fade_frames_and_then_holds_zero():
    track = _sparse_track(n_frames=40, box=(0.0, 0.0, 10.0, 10.0), anchor=0, every=0)
    fade = fp._fade_multipliers(track)
    # detected() is True only at frame 0 here (every=0: a single-frame island); frame i>0 is
    # exactly i frames past the nearest True frame.
    for distance in range(1, fp.FADE_FRAMES + 1):
        frame_idx = distance  # frame 0 is the True anchor itself
        expected = max(0.0, 1.0 - distance / fp.FADE_FRAMES)
        assert fade[frame_idx] == pytest.approx(expected), f"frame {frame_idx}"
    assert fade[fp.FADE_FRAMES] == pytest.approx(0.0)
    assert fade[fp.FADE_FRAMES + 5] == pytest.approx(0.0)
    # Strictly decreasing across the ramp itself.
    ramp = fade[0:fp.FADE_FRAMES + 1]
    assert np.all(np.diff(ramp) <= 0)


def test_fade_multipliers_ramps_symmetrically_around_a_gap_between_two_detections():
    """Regression / spec-lock for review finding I4(b), closing the executor's sомнение #1: with
    two real detections bracketing a gap wider than `2 * FADE_FRAMES`, the fade multiplier ramps
    DOWN approaching the gap leaving the earlier anchor, and ramps back UP approaching the later
    anchor -- symmetric on both sides of the hole, not just decaying away from the first detection.
    """
    n_frames = 60
    box = (0.0, 0.0, 10.0, 10.0)
    gap = 2 * fp.FADE_FRAMES + 10  # wide enough that the fade genuinely reaches 0 in the middle
    anchor_a, anchor_b = 5, 5 + gap
    track = ft._build_track([(anchor_a, box), (anchor_b, box)], n_frames=n_frames, every=0)
    fade = fp._fade_multipliers(track)

    assert fade[anchor_a] == pytest.approx(1.0)
    assert fade[anchor_b] == pytest.approx(1.0)
    mid = (anchor_a + anchor_b) // 2
    assert fade[mid] == pytest.approx(0.0)

    # Ramping down leaving anchor_a...
    left_ramp = fade[anchor_a: anchor_a + fp.FADE_FRAMES + 1]
    assert np.all(np.diff(left_ramp) <= 0)
    # ...and ramping back up approaching anchor_b, the mirror image.
    right_ramp = fade[anchor_b - fp.FADE_FRAMES: anchor_b + 1]
    assert np.all(np.diff(right_ramp) >= 0)
    # Equal distance from either anchor gives the equal multiplier -- true symmetry, not just
    # "both ends happen to ramp".
    for d in range(0, fp.FADE_FRAMES + 1):
        assert fade[anchor_a + d] == pytest.approx(fade[anchor_b - d]), f"d={d}"


# --------------------------------------------------------------------------------------------
# `crop_window`
# --------------------------------------------------------------------------------------------

def test_crop_window_produces_the_requested_out_size_for_every_frame():
    frames = _synthetic_frames(6, height=120, width=160)
    track = _dense_track(6, box=(60.0, 40.0, 30.0, 30.0))
    crops, geometry = fp.crop_window(frames, track, scale=2.0, out_size=(64, 48))
    assert crops.shape == (6, 48, 64, 3)
    assert crops.dtype == np.uint8
    assert geometry.n_frames == 6
    assert geometry.out_size == (64, 48)


def test_crop_window_clips_a_rect_that_would_spill_off_the_frame_edge():
    frames = _synthetic_frames(3, height=100, width=100)
    # Face box hard in the top-left corner: ideal scale=3 crop is far larger than fits there.
    track = _dense_track(3, box=(2.0, 2.0, 20.0, 20.0))
    crops, geometry = fp.crop_window(frames, track, scale=3.0, out_size=(64, 64))
    for frame_idx in range(3):
        x, y, w, h = geometry.rect(frame_idx)
        assert x >= 0 and y >= 0
        assert x + w <= 100
        assert y + h <= 100
        assert w > 0 and h > 0
    assert crops.shape == (3, 64, 64, 3)


def test_crop_window_rejects_too_few_frames():
    frames = _synthetic_frames(4, height=80, width=80)
    track = _dense_track(6, box=(30.0, 30.0, 10.0, 10.0))  # track expects 6 frames, only 4 given
    with pytest.raises(ValueError):
        fp.crop_window(frames, track)


def test_crop_window_rejects_too_many_frames():
    frames = _synthetic_frames(8, height=80, width=80)
    track = _dense_track(6, box=(30.0, 30.0, 10.0, 10.0))  # track expects 6 frames, 8 given
    with pytest.raises(ValueError):
        fp.crop_window(frames, track)


def test_crop_window_rect_matches_out_size_aspect_after_expansion():
    """Regression for review finding I3: the crop window used to inherit the YuNet box's own
    aspect ratio (roughly portrait, ~1:1.25) straight into `_round_rect`, then got stretched
    non-uniformly by `cv2.resize` into a landscape `out_size` (e.g. 448x288, 1.556:1) -- doubling
    the face's apparent width. `crop_window` must expand the *window* to `out_size`'s aspect
    first, so the resize to `out_size` is a uniform scale, not a stretch.
    """
    height, width = 800, 800
    out_size = (448, 288)  # (w, h) -- 1.5556:1, landscape
    target_aspect = out_size[0] / out_size[1]
    for box in [(300.0, 300.0, 40.0, 50.0),   # portrait-ish, like a real face box
                (300.0, 300.0, 60.0, 40.0),    # landscape-ish
                (300.0, 300.0, 45.0, 45.0)]:   # square
        frames = _synthetic_frames(1, height=height, width=width)
        track = _dense_track(1, box=box)
        _, geometry = fp.crop_window(frames, track, scale=2.0, out_size=out_size)
        x, y, w, h = geometry.rect(0)
        assert abs(w / h - target_aspect) < 0.02, f"box={box}: rect aspect {w / h} != {target_aspect}"


def test_crop_window_rect_long_side_tracks_scale_times_the_box_and_stays_centered():
    """Regression / spec check for review finding I4(a): the crop rect's *long* side should track
    `scale *` the face box's own long dimension -- the short side is now dictated by `out_size`'s
    aspect (I3) -- and the rect stays centered on the box's own center, up to rounding and the
    aspect expansion.
    """
    scale = 2.75
    out_size = (448, 288)  # landscape -- a portrait box's height is the side that survives
    box = (300.0, 260.0, 40.0, 55.0)  # (x, y, w, h): h=55 is the long side
    height, width = 1000, 1000
    frames = _synthetic_frames(1, height=height, width=width)
    track = _dense_track(1, box=box)
    _, geometry = fp.crop_window(frames, track, scale=scale, out_size=out_size)
    x, y, w, h = geometry.rect(0)
    bx, by, bw, bh = box
    box_cx, box_cy = bx + bw / 2.0, by + bh / 2.0
    rect_cx, rect_cy = x + w / 2.0, y + h / 2.0
    assert abs(rect_cx - box_cx) < 1.5
    assert abs(rect_cy - box_cy) < 1.5
    expected_long = scale * bh
    assert abs(h - expected_long) < 1.5


def test_crop_window_rect_size_is_stable_on_a_smooth_track_not_edge_rounding_noise():
    """RED-FIRST regression probe for review finding I1 (run against the pre-fix code to confirm
    it fails): `_round_rect` used to round `x0` and `x0+w` as two independent edges, so a rect
    whose *position* drifts smoothly -- even with a perfectly constant box size -- sees its integer
    *width* flip by +-1 whenever the two edges' fractional parts cross an integer boundary at
    different frames (measured at 28 changes across 59 transitions on a real smooth track). A track
    this smooth should hold its rect size for long stretches.
    """
    n_frames = 60
    height, width = 400, 400
    frames = _synthetic_frames(n_frames, height=height, width=width)
    box_w, box_h = 41.3, 51.7  # constant size -- only the *position* drifts below
    samples = []
    for i in range(n_frames):
        cx = 150.0 + 0.37 * i
        cy = 150.0 + 0.31 * i
        samples.append((i, (cx - box_w / 2.0, cy - box_h / 2.0, box_w, box_h)))
    track = ft._build_track(samples, n_frames=n_frames, every=1)
    _, geometry = fp.crop_window(frames, track, scale=2.75, out_size=(64, 64))
    widths = geometry.rects[:, 2]
    heights = geometry.rects[:, 3]
    width_changes = int(np.sum(np.diff(widths) != 0))
    height_changes = int(np.sum(np.diff(heights) != 0))
    assert width_changes <= 3, f"rect width changed {width_changes} times across {n_frames} frames"
    assert height_changes <= 3, f"rect height changed {height_changes} times across {n_frames} frames"


# --------------------------------------------------------------------------------------------
# `paste_back` -- round trip, edge clip, fade_out
# --------------------------------------------------------------------------------------------

def test_paste_back_round_trip_is_bit_exact_outside_the_crop_rect():
    frames = _synthetic_frames(4, height=120, width=160)
    track = _dense_track(4, box=(60.0, 40.0, 30.0, 30.0))
    crops, geometry = fp.crop_window(frames, track, scale=2.0, out_size=(64, 48))
    result = fp.paste_back(frames, crops, geometry, track, feather=0.10)
    assert result.shape == frames.shape
    for frame_idx in range(4):
        x, y, w, h = geometry.rect(frame_idx)
        outside = np.ones(frames[frame_idx].shape[:2], dtype=bool)
        outside[y:y + h, x:x + w] = False
        assert np.array_equal(result[frame_idx][outside], frames[frame_idx][outside])


def test_paste_back_round_trip_is_close_inside_the_crop_rect():
    frames = _synthetic_frames(4, height=120, width=160)
    track = _dense_track(4, box=(60.0, 40.0, 30.0, 30.0))
    crops, geometry = fp.crop_window(frames, track, scale=2.0, out_size=(64, 48))
    result = fp.paste_back(frames, crops, geometry, track, feather=0.10)
    for frame_idx in range(4):
        x, y, w, h = geometry.rect(frame_idx)
        # The center of the rect is fully inside the feather mask's flat interior (mask == 1),
        # so it went through a full-strength Lanczos up/down round trip: close, not bit-exact.
        cy, cx = y + h // 2, x + w // 2
        original = frames[frame_idx][cy, cx].astype(np.int16)
        pasted = result[frame_idx][cy, cx].astype(np.int16)
        assert np.abs(original - pasted).max() < 20


def test_paste_back_rejects_length_mismatch():
    frames = _synthetic_frames(4, height=80, width=80)
    track = _dense_track(4, box=(30.0, 30.0, 10.0, 10.0))
    crops, geometry = fp.crop_window(frames, track, scale=2.0, out_size=(32, 32))
    with pytest.raises(ValueError):
        fp.paste_back(frames[:3], crops, geometry, track)


def test_paste_back_rejects_refined_frame_shape_that_does_not_match_geometry_out_size():
    """Regression for review finding I5: paste_back used to trust `refined`'s shape blindly, so a
    caller that mixed up width/height for `refined` would silently get a distorted paste instead
    of an error.
    """
    frames = _synthetic_frames(3, height=80, width=80)
    track = _dense_track(3, box=(30.0, 30.0, 10.0, 10.0))
    crops, geometry = fp.crop_window(frames, track, scale=2.0, out_size=(48, 32))  # (w, h)
    assert crops.shape == (3, 32, 48, 3)
    # Width/height swapped on the refined frames -- exactly the "перепутанные (W,H)/(H,W)"
    # mistake the finding warns about.
    swapped = np.zeros((3, 48, 32, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        fp.paste_back(frames, swapped, geometry, track)


def test_paste_back_rejects_a_frame_with_a_different_shape_than_frame_zero():
    """Regression for review finding I5: paste_back used to assume every frame shares frame 0's
    shape (it builds the output array from that shape alone) without ever checking later frames
    against it.
    """
    frames_list = list(_synthetic_frames(3, height=80, width=80))
    track = _dense_track(3, box=(30.0, 30.0, 10.0, 10.0))
    crops, geometry = fp.crop_window(frames_list, track, scale=2.0, out_size=(32, 32))
    frames_list[1] = frames_list[1][:60, :60]  # a differently-shaped frame slipped in
    with pytest.raises(ValueError):
        fp.paste_back(frames_list, crops, geometry, track)


def test_paste_back_does_not_feather_the_side_that_is_the_frame_boundary():
    """Regression for the Minor finding and review finding I4(c): a rect clipped flush against the
    frame's own edge should not get a soft ramp on that side -- there is no seam to hide there, it
    is the picture's own border. Feathering it anyway leaves the outer ~10% of a real full-frame
    paste un-refined.
    """
    height, width = 60, 60
    frames = _synthetic_frames(1, height=height, width=width)
    # Face box hard in the top-left corner -- the clipped rect touches x=0 and y=0.
    track = _dense_track(1, box=(1.0, 1.0, 6.0, 6.0))
    crops, geometry = fp.crop_window(frames, track, scale=3.0, out_size=(32, 32))
    x, y, w, h = geometry.rect(0)
    assert x == 0 and y == 0  # touches the top-left frame boundary, as intended by the setup
    refined = np.full_like(crops, 255)
    result = fp.paste_back(frames, refined, geometry, track, feather=0.10)
    # The very corner pixel of the rect is right at the frame's edge: full paste strength is
    # expected there (no ramp), not the near-zero a normal 10% feather would give it.
    corner_pixel = result[0][y, x]
    assert np.abs(corner_pixel.astype(np.int64) - 255).max() < 5, (
        "edge pixel should be nearly full-strength refined (255), not faded toward the source"
    )


def test_paste_back_feathers_nothing_when_the_rect_fills_the_whole_frame():
    """Minor finding, end-to-end: when the crop rect covers the entire frame (all four sides are
    the frame's own border), nothing should fade -- the mask is 1.0 everywhere.
    """
    height, width = 40, 40
    frames = _synthetic_frames(1, height=height, width=width)
    # A window far bigger than the frame in every direction: `_clip_to_frame` shrinks it down to
    # exactly the whole frame.
    track = _dense_track(1, box=(15.0, 15.0, 10.0, 10.0))
    crops, geometry = fp.crop_window(frames, track, scale=10.0, out_size=(32, 32))
    x, y, w, h = geometry.rect(0)
    assert (x, y, w, h) == (0, 0, width, height)
    refined = np.full_like(crops, 255)
    result = fp.paste_back(frames, refined, geometry, track, feather=0.10)
    for corner in [(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)]:
        pixel = result[0][corner]
        assert np.abs(pixel.astype(np.int64) - 255).max() < 5, f"corner {corner} not full-strength"


def test_paste_back_fade_out_shrinks_the_blend_and_then_stops_pasting_entirely():
    height, width = 100, 100
    n_frames = 40
    # The *same* source content on every frame (not `_synthetic_frames(n_frames, ...)`, which
    # deliberately drifts color per frame): isolates the measured deviation to the fade weight
    # itself, since the box/geometry are also constant (`_sparse_track` extrapolates one box
    # everywhere) and only `_fade_multipliers` changes across frames.
    one_frame = _synthetic_frames(1, height=height, width=width)[0]
    frames = np.tile(one_frame[None], (n_frames, 1, 1, 1))
    track = _sparse_track(n_frames, box=(30.0, 30.0, 20.0, 20.0), anchor=0, every=0)
    crops, geometry = fp.crop_window(frames, track, scale=1.5, out_size=(48, 48))
    # A "refined" result that is as different from the source as possible (pure white), so any
    # surviving blend strength is visible as a jump toward 255 at the rect's center pixel.
    refined = np.full_like(crops, 255)
    result = fp.paste_back(frames, refined, geometry, track, feather=0.10)

    x, y, w, h = geometry.rect(0)
    cy, cx = y + h // 2, x + w // 2
    deviations = [
        int(result[i][cy, cx].astype(np.int16).sum())
        - int(frames[i][cy, cx].astype(np.int16).sum())
        for i in range(n_frames)
    ]
    # Frame 0 (the real detection) blends at full strength: a large deviation toward white.
    assert deviations[0] > 100
    # Deviation shrinks monotonically across the fade window...
    ramp = deviations[0:fp.FADE_FRAMES + 1]
    assert all(a >= b for a, b in zip(ramp, ramp[1:]))
    # ...and every frame FADE_FRAMES or more past the anchor is untouched: bit-exact original.
    for i in range(fp.FADE_FRAMES, n_frames):
        assert np.array_equal(result[i], frames[i]), f"frame {i} should be untouched"


# --------------------------------------------------------------------------------------------
# Module hygiene
# --------------------------------------------------------------------------------------------

def test_facepaste_module_does_not_import_mlx():
    """`h3_48gb.facepaste` is a CPU-only module, same reasoning and same test shape as
    `facetrack.py`'s `test_facetrack_module_does_not_import_mlx` -- see that module's docstring.
    """
    code = "import sys; import h3_48gb.facepaste; print('mlx' in sys.modules)"
    project_root = Path(__file__).resolve().parent.parent
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            cwd=str(project_root))
    assert result.returncode == 0, f"subprocess failed: {result.stderr}"
    assert result.stdout.strip() == "False", "importing h3_48gb.facepaste must not import mlx"
