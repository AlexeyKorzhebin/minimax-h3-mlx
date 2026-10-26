"""Unit tests for `h3_48gb.framecheck`'s two corruption detectors, on synthetic frames.

No `ffmpeg`/`ffprobe`, no MLX -- pure numpy, matching the module's own "no `mlx` import, ever"
input contract, since it has to be importable from `h3_48gb.assemble` (worker-process, no GPU).

A couple of tests below also read a real reference frame off disk
(`~/h3-outputs/gates-phase2-frames/`), the 2026-08-24 fix's own measurement material --
`pytest.skip`-gated on the file existing, same convention as `test_vae_decode_parity.py`'s
`RECON_DIR` check, so the suite still passes on a machine without that research directory.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from h3_48gb import framecheck

CANVAS = (512, 896)  # (H, W) -- DEFAULT_SCENE_CANVAS, so every configured seam falls in range

GATES_PHASE2_DIR = Path.home() / "h3-outputs/gates-phase2-frames"


def _clean_frame(seed: int = 0) -> np.ndarray:
    """Smooth, low-variance noise around mid-grey -- far from `FILL_COLOR` and with no artificial
    discontinuity at any of `framecheck`'s configured tile seams.
    """
    rng = np.random.default_rng(seed)
    h, w = CANVAS
    frame = 128.0 + rng.normal(0.0, 5.0, size=(h, w, 3))
    return np.clip(frame, 0, 255).astype(np.uint8)


# -- zero-fill --------------------------------------------------------------------------------


def test_zero_fill_fraction_is_zero_on_a_clean_frame():
    frame = _clean_frame()
    assert framecheck.zero_fill_fraction(frame) < framecheck.ZERO_FILL_FRACTION_THRESHOLD


def test_zero_fill_rectangle_trips_the_threshold_and_is_frame_corrupt():
    """A single unwritten VAE tile's worth of `FILL_COLOR`, not the whole frame -- the "or a large
    rectangle" half of the task's own criterion, not just the all-flat-frame case.
    """
    frame = _clean_frame()
    frame[100:300, 100:300] = framecheck.FILL_COLOR  # 200x200 = 40,000 px >> 0.5% of 458,752

    fraction = framecheck.zero_fill_fraction(frame)

    assert fraction > framecheck.ZERO_FILL_FRACTION_THRESHOLD
    assert framecheck.is_frame_corrupt(frame)


def test_zero_fill_fraction_tolerates_the_documented_slop():
    """`FILL_TOLERANCE` (2) is uint8 rounding slop, not a loose match -- a fill color nudged by
    exactly the tolerance still counts; nudged one past it does not.
    """
    within = np.full((64, 64, 3), 0, dtype=np.uint8)
    within[:] = np.array(framecheck.FILL_COLOR) + framecheck.FILL_TOLERANCE
    assert framecheck.zero_fill_fraction(within) == pytest.approx(1.0)

    outside = np.full((64, 64, 3), 0, dtype=np.uint8)
    outside[:] = np.array(framecheck.FILL_COLOR) + framecheck.FILL_TOLERANCE + 1
    assert framecheck.zero_fill_fraction(outside) == pytest.approx(0.0)


@pytest.mark.cv
def test_is_frame_corrupt_catches_every_flat_frame_from_the_chunk_recon_calibration():
    """`chunk-recon/corruption-map.csv` is the investigation's own frame-by-frame calibration of
    the 2026-08-19 corrupted scenes (1600 frames, 8 real 896x512 clips). `flat > 0` marks a frame
    the investigation itself identified as having an unwritten (zero-fill) tile -- the `flat`
    column is boolean `0`/`1`, and 36 of the 1600 rows read `1` (measured directly off the file).
    `zero_fill_fraction`/`is_frame_corrupt` are untouched by the 2026-08-24 tile-seam fix; this
    pins that they still catch every one of those 36 frames, reading the actual pixels off the
    real clips rather than trusting the csv's own `seam_score` column (computed with the old,
    now-removed hardcoded columns). Every one of the 36 measured `True` for *both* detectors here
    (`tile_seam_score` also flags them, since they're all on the native 896x512 canvas), so this
    asserts `zero_fill_fraction` itself, not just `is_frame_corrupt`'s either/or -- otherwise a
    broken `zero_fill_fraction` could hide behind the seam detector and this test would never
    notice.
    """
    cv2 = pytest.importorskip("cv2", reason="cv: needs opencv-python and scipy, absent here")
    import csv as csv_module

    csv_path = Path.home() / "h3-outputs/chunk-recon/corruption-map.csv"
    clips_dir = Path.home() / "h3-outputs/projects/example-project"
    if not csv_path.is_file() or not clips_dir.is_dir():
        pytest.skip(f"chunk-recon investigation artifacts not present under {csv_path.parent}")

    with csv_path.open() as f:
        rows = list(csv_module.DictReader(f))
    flat_rows = [r for r in rows if r["flat"] == "1"]
    assert len(flat_rows) == 36  # measured directly off the csv's boolean `flat` column

    by_clip: dict[str, list[int]] = {}
    for r in flat_rows:
        by_clip.setdefault(r["clip"], []).append(int(r["frame_idx"]))

    clip_paths = {p.stem: p for p in clips_dir.rglob("*896x512.mp4")}
    checked = 0
    for clip, indices in by_clip.items():
        path = clip_paths.get(clip)
        if path is None:
            pytest.skip(f"clip {clip} referenced by corruption-map.csv not found under {clips_dir}")
        cap = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release()
        for idx in indices:
            frame = frames[idx]
            assert framecheck.is_frame_corrupt(frame), f"{clip} frame {idx} (flat=1) not caught"
            assert framecheck.zero_fill_fraction(frame) > framecheck.ZERO_FILL_FRACTION_THRESHOLD, (
                f"{clip} frame {idx} (flat=1) not caught by zero_fill_fraction specifically")
            checked += 1
    assert checked == 36


# -- tile-seam score ----------------------------------------------------------------------------


def test_tile_seam_score_is_near_one_on_a_clean_frame():
    frame = _clean_frame()
    assert framecheck.tile_seam_score(frame) <= framecheck.TILE_SEAM_SCORE_THRESHOLD
    assert not framecheck.is_frame_corrupt(frame)


def test_tile_seam_score_flags_an_artificial_seam_at_a_computed_column():
    """A hard step exactly at one of 896x512's *computed* seam columns (320, one of
    `_tile_seam_positions(896)`) -- everything from that column on is brightened by 60 levels --
    must read as far more discontinuous at the seam than a few pixels either side of it.
    """
    frame = _clean_frame()
    x = 320
    assert x in framecheck._tile_seam_positions(896)
    shifted = frame.astype(np.int16)
    shifted[:, x:, :] = np.clip(shifted[:, x:, :] + 60, 0, 255)
    frame = shifted.astype(np.uint8)

    score = framecheck.tile_seam_score(frame)

    assert score > framecheck.TILE_SEAM_SCORE_THRESHOLD
    assert framecheck.is_frame_corrupt(frame)


def test_tile_seam_score_flags_an_artificial_seam_at_a_computed_row():
    frame = _clean_frame()
    y = 256
    assert y in framecheck._tile_seam_positions(512)
    shifted = frame.astype(np.int16)
    shifted[y:, :, :] = np.clip(shifted[y:, :, :] + 60, 0, 255)
    frame = shifted.astype(np.uint8)

    assert framecheck.tile_seam_score(frame) > framecheck.TILE_SEAM_SCORE_THRESHOLD


def test_tile_seam_positions_for_896x512_is_a_superset_of_the_historical_calibration():
    """`_tile_seam_positions` replaces the old hardcoded `TILE_SEAM_COLUMNS`/`TILE_SEAM_ROWS`
    (removed 2026-08-24). It isn't required to reproduce them exactly -- measured (see the
    module's `_tile_seam_positions` docstring), the historical columns were "overlap starts plus
    the last overlap's end" while the historical rows were "overlap starts only, no ends", two
    different partial rules for the same picture -- but at 896x512, the canvas they were
    calibrated for, every historical coordinate must still be a position this computes.
    """
    historical_columns = {160, 320, 480, 640, 736}
    historical_rows = {128, 256}

    assert historical_columns <= set(framecheck._tile_seam_positions(896))
    assert historical_rows <= set(framecheck._tile_seam_positions(512))


def test_tile_seam_positions_for_1344x768_are_pinned_to_the_measured_geometry():
    """`VAE_SPATIAL_COMPRESSION_RATIO` (fix-round-1 review, I2) has no test coverage from the
    896x512 tests above: that canvas's overlap boundaries happen to land on the same positions
    whether the ratio is 16 or 8, so a wrong ratio would sail through every 896x512-only
    assertion. 1344x768 IS sensitive -- measured (task-1-report.md): mutating the ratio 16 -> 8
    shifts 6 of its 12 columns (528, 704, 784, 896, 960, 1152 move to 536, 720, 792, 904, 976,
    1160) and 2 of its 6 rows (160, 416 move to 168, 424). Pinning the exact tuple for this canvas
    is what actually exercises the ratio, not just the tiling arithmetic's shape.
    """
    assert framecheck._tile_seam_positions(1344) == (
        176, 256, 352, 432, 528, 608, 704, 784, 896, 960, 1088, 1152)
    assert framecheck._tile_seam_positions(768) == (160, 256, 336, 416, 512, 592)


def test_vae_spatial_compression_ratio_matches_the_real_weights_config():
    """`VAE_SPATIAL_COMPRESSION_RATIO` is this module's own hardcoded copy of
    `VideoVAEConfig.spatial_compression_ratio` (`framecheck` cannot import `video_vae`, see the
    module docstring's mlx-free contract) -- if the real weights' config ever disagreed, seam
    positions would silently stop matching the actual decode tiling with nothing to notice. Checks
    it directly against `~/models/h3-converted/video_vae/source/config.json`'s `space_down`,
    skip-gated the same way `test_vae_decode_parity.py` gates on a real checkpoint.
    """
    import json

    config_path = Path.home() / "models/h3-converted/video_vae/source/config.json"
    if not config_path.is_file():
        pytest.skip(f"no checkpoint config at {config_path}")
    config = json.loads(config_path.read_text())
    ratio = 1
    for factor in config["space_down"]:
        ratio *= factor

    assert framecheck.VAE_SPATIAL_COMPRESSION_RATIO == ratio


def test_tile_seam_score_ignores_a_historical_grid_column_on_a_non_native_canvas():
    """The false alarm that stalled боевые ворота 2026-08-20: a 448x288 clip with a real, legitimate
    high-contrast vertical edge sitting at x=160 -- one of the *old* hardcoded `TILE_SEAM_COLUMNS`
    -- is not a VAE tile seam at this width at all (448's own tile boundary is at x=192,
    `_tile_seam_positions(448) == (192, 256)`), and must not be flagged as corrupt.

    Must fail against the pre-fix code: the old `tile_seam_score` scored every frame against the
    fixed `(160, 320, 480, 640, 736)` columns regardless of the frame's actual width, so this exact
    step at x=160 read as a seam-level discontinuity there.
    """
    rng = np.random.default_rng(7)
    h, w = 288, 448
    frame = 128.0 + rng.normal(0.0, 5.0, size=(h, w, 3))
    frame = np.clip(frame, 0, 255)
    x = 160
    assert x not in framecheck._tile_seam_positions(w)
    frame[:, x:, :] = np.clip(frame[:, x:, :] + 60, 0, 255)
    frame = frame.astype(np.uint8)

    assert framecheck.tile_seam_score(frame) == 1.0
    assert not framecheck.is_frame_corrupt(frame)


def test_tile_seam_score_is_disabled_below_min_seam_points_on_448x288():
    """448x288 gets only 4 total measurement points (`_tile_seam_positions(448)` has 2,
    `_tile_seam_positions(288)` has 2) -- below `MIN_SEAM_POINTS` (11, raised from an initial 8
    by a fix-round-1 review that found the 8-10 band unmeasured and unsafe: see the constant's own
    docstring). `tile_seam_score` must return the clean value unconditionally there, regardless of
    frame content, rather than compute a ratio over too few points to be meaningful.
    `zero_fill_fraction` is untouched and still the corruption signal that canvas relies on.
    """
    assert len(framecheck._tile_seam_positions(448)) + len(framecheck._tile_seam_positions(288)) < framecheck.MIN_SEAM_POINTS

    rng = np.random.default_rng(11)
    frame = np.clip(128.0 + rng.normal(0.0, 40.0, size=(288, 448, 3)), 0, 255).astype(np.uint8)

    assert framecheck.tile_seam_score(frame) == 1.0


def test_tile_seam_score_stays_disabled_at_the_min_seam_points_boundary_896x448():
    """Guards the exact floor `MIN_SEAM_POINTS` sits at (11): 896x448 -- the fix-round-1 review's
    own proxy canvas for the unmeasured 8-10 band -- gives exactly 10 total points
    (`_tile_seam_positions(896)` has 8 columns, `_tile_seam_positions(448)` has 2 rows), one below
    the floor. The only other MIN_SEAM_POINTS test runs at 4 points, nowhere near either boundary,
    so a regression back to `MIN_SEAM_POINTS = 8` or `= 10` would enable the check here and this
    test would be the one to notice: 896x448 is exactly one of the canvases the review measured a
    real false positive on (proxy material, max seam_score 2.62 there) before the floor was raised.
    """
    assert len(framecheck._tile_seam_positions(896)) + len(framecheck._tile_seam_positions(448)) == 10
    assert 10 < framecheck.MIN_SEAM_POINTS

    rng = np.random.default_rng(13)
    frame = np.clip(128.0 + rng.normal(0.0, 40.0, size=(448, 896, 3)), 0, 255).astype(np.uint8)

    assert framecheck.tile_seam_score(frame) == 1.0


@pytest.mark.cv
def test_tile_seam_score_catches_the_reference_corrupt_frame_from_the_2026_08_20_incident():
    """The actual corrupt keyframe (`CLIP-ARTIFACT-seam1-corrupt-*`, 896x512) that came out of the
    boевые ворота 2026-08-19 investigation, saved to `gates-phase2-frames/` -- measured (task-1
    fix): computed-position seam score 19.27 / 19.86, ~8x `TILE_SEAM_SCORE_THRESHOLD` (2.5), on the
    two variants (`FINAL-f226` and `SOURCE-scene1-frame0`). Real corruption clears the new
    computed-geometry threshold with the same wide margin the old hardcoded columns did.
    """
    cv2 = pytest.importorskip("cv2", reason="cv: needs opencv-python and scipy, absent here")
    path = GATES_PHASE2_DIR / "CLIP-ARTIFACT-seam1-corrupt-SOURCE-scene1-frame0.png"
    if not path.is_file():
        pytest.skip(f"gates-phase2 reference frame not present under {GATES_PHASE2_DIR}")
    bgr = cv2.imread(str(path))
    frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    assert framecheck.tile_seam_score(frame) > framecheck.TILE_SEAM_SCORE_THRESHOLD
    assert framecheck.is_frame_corrupt(frame)


def test_tile_seam_score_returns_the_clean_value_when_no_configured_seam_fits():
    """A frame smaller than every configured seam position: nothing to measure is not evidence of
    corruption -- must degrade to the clean value (1.0), not raise or return a spurious score.
    """
    tiny = np.zeros((8, 8, 3), dtype=np.uint8)
    assert framecheck.tile_seam_score(tiny) == 1.0
    assert not framecheck.is_frame_corrupt(tiny)


# -- find_corrupt_frames / batch API --------------------------------------------------------------


def test_find_corrupt_frames_reports_only_the_bad_ones_with_their_reason():
    clean = _clean_frame(seed=1)
    zero_filled = _clean_frame(seed=2)
    zero_filled[:, :] = framecheck.FILL_COLOR
    seam = _clean_frame(seed=3)
    seam_i16 = seam.astype(np.int16)
    seam_i16[:, 640:, :] = np.clip(seam_i16[:, 640:, :] + 80, 0, 255)
    seam = seam_i16.astype(np.uint8)

    frames = np.stack([clean, zero_filled, seam, clean])
    bad = framecheck.find_corrupt_frames(frames)

    assert {b.index for b in bad} == {1, 2}
    by_index = {b.index: b for b in bad}
    assert by_index[1].zero_fill
    assert by_index[2].seam
    assert not by_index[2].zero_fill


def test_find_corrupt_frames_is_empty_for_an_all_clean_batch():
    frames = np.stack([_clean_frame(seed=i) for i in range(5)])
    assert framecheck.find_corrupt_frames(frames) == []
