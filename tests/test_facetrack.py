"""Face detection and track math -- YuNet mocked out for everything except one real-frame test.

Track math (`_build_track`, `_associate_tracks`, `_select_subject_track`, `FaceTrack.box`/
`detected`) is tested directly, without touching `cv2` or a real detector at all: `detect_track`'s
own wiring (load the detector, sample every Nth frame, hand samples to `_build_track`) is tested
separately with the detector monkeypatched to return a fixed list of boxes per call. Only
`test_detect_track_finds_a_real_face_in_a_real_frame` below exercises the real YuNet ONNX model,
against a real frame cut from ~/h3-outputs/face-refine-exp/round2/SOURCE-crop.mp4 --
synthetic drawn "faces" are not something YuNet detects (see the task brief), so there is no
substitute for a real photograph here.
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

from h3_48gb import facetrack as ft


# --------------------------------------------------------------------------------------------
# `_associate_tracks` / `_select_subject_track`
# --------------------------------------------------------------------------------------------

def test_associate_tracks_groups_by_nearest_center_into_separate_tracks():
    detections = [
        (0, [(0.0, 0.0, 10.0, 10.0), (100.0, 100.0, 10.0, 10.0)]),
        (5, [(2.0, 1.0, 10.0, 10.0), (98.0, 102.0, 10.0, 10.0)]),
    ]
    tracks = ft._associate_tracks(detections)
    assert len(tracks) == 2
    assert sorted(len(track) for track in tracks) == [2, 2]


def test_select_subject_track_keeps_the_track_with_the_larger_median_area():
    detections = [
        (0, [(0.0, 0.0, 10.0, 10.0), (500.0, 500.0, 30.0, 30.0)]),
        (5, [(2.0, 1.0, 10.0, 10.0), (498.0, 502.0, 30.0, 30.0)]),
    ]
    subject = ft._select_subject_track(detections)
    assert len(subject) == 2
    for _, box in subject:
        assert box[2] == pytest.approx(30.0)


def test_select_subject_track_is_empty_with_no_detections():
    assert ft._select_subject_track([]) == []


# --------------------------------------------------------------------------------------------
# `_build_track` -- pure math, no detector
# --------------------------------------------------------------------------------------------

def test_build_track_is_none_with_no_samples():
    assert ft._build_track([], n_frames=20, every=5) is None


def test_single_detection_gives_a_constant_box_across_every_frame():
    samples = [(3, (10.0, 20.0, 30.0, 40.0))]
    track = ft._build_track(samples, n_frames=8, every=5)
    assert track is not None
    assert track.n_frames == 8
    assert track.median_area == pytest.approx(30.0 * 40.0)
    for frame_idx in range(8):
        x, y, w, h = track.box(frame_idx)
        assert (x, y, w, h) == pytest.approx((10.0, 20.0, 30.0, 40.0))


def test_linear_interpolation_between_two_detections():
    # every=1 keeps the smoothing window below the 5-sample floor (see `_smooth`), so this test
    # sees the raw interpolation, not a savgol-adjusted version of it.
    samples = [(0, (0.0, 0.0, 10.0, 10.0)), (4, (40.0, 20.0, 10.0, 10.0))]
    track = ft._build_track(samples, n_frames=5, every=1)
    x, y, w, h = track.box(2)  # exact midpoint in frame index
    # centers: (5, 5) at frame 0 -> (45, 25) at frame 4; midpoint center is (25, 15)
    assert (x, y, w, h) == pytest.approx((20.0, 10.0, 10.0, 10.0))


def test_constant_extrapolation_holds_each_tails_own_anchor_value():
    samples = [(3, (0.0, 0.0, 10.0, 10.0)), (7, (100.0, 0.0, 10.0, 10.0))]
    track = ft._build_track(samples, n_frames=10, every=1)
    for frame_idx in (0, 1, 2):  # before the first detection
        x, _, _, _ = track.box(frame_idx)
        assert x == pytest.approx(0.0)
    for frame_idx in (8, 9):  # after the last detection
        x, _, _, _ = track.box(frame_idx)
        assert x == pytest.approx(100.0)


def test_constant_extrapolation_holds_in_working_mode_with_savgol_active():
    # Same constant-tails claim as `test_constant_extrapolation_holds_each_tails_own_anchor_value`,
    # but with `every=5` and enough frames that `_smooth`'s window (2*every+1 = 11) actually
    # engages savgol -- that test uses `every=1`, which keeps the window below the 5-sample floor
    # and never exercises smoothing at all. With savgol active, `np.interp`'s exact constant tails
    # get run through the filter too: far from the anchors the fit over a locally-constant window
    # reproduces the constant exactly, but close to the ramp-to-flat kink the quadratic fit can
    # overshoot slightly (verified on this exact data: up to ~0.73, well under 1% of the 0-100
    # range) -- both facts are asserted here, not just the middle case that was already covered.
    samples = [(20, (0.0, 0.0, 10.0, 10.0)), (40, (100.0, 0.0, 10.0, 10.0))]
    track = ft._build_track(samples, n_frames=60, every=5)
    # Far tails (more than half a smoothing window away from the nearest anchor): exact.
    for frame_idx in range(0, 15):
        x, _, _, _ = track.box(frame_idx)
        assert x == pytest.approx(0.0, abs=1e-6)
    for frame_idx in range(45, 60):
        x, _, _, _ = track.box(frame_idx)
        assert x == pytest.approx(100.0, abs=1e-6)
    # Near the kink: bounded overshoot, not an exact hold, but nowhere near sweeping off to the
    # other anchor's value.
    for frame_idx in range(0, 60):
        x, _, _, _ = track.box(frame_idx)
        assert -1.5 <= x <= 101.5


def test_median_area_uses_raw_detections_not_the_smoothed_track():
    # Two small detections flank one huge spike in the middle: the RAW median (of just the three
    # sample areas 100, 10000, 100) is 100. But a broad, far-apart triangle -- size ramping
    # linearly from 10 up to 100 and back down over the full 0-100 frame span -- spends most of
    # its per-frame area values well above 100 before ever reaching the two small endpoints, so a
    # buggy implementation that computed median_area from the smoothed/interpolated per-frame
    # series instead of the raw samples would land far from 100 (numerically, ~3025 -- verified by
    # direct computation on this exact data). Only reading the median off the three raw sample
    # areas lands on exactly 100.
    samples = [
        (0, (0.0, 0.0, 10.0, 10.0)),      # area 100
        (50, (0.0, 0.0, 100.0, 100.0)),   # area 10000
        (100, (0.0, 0.0, 10.0, 10.0)),    # area 100
    ]
    track = ft._build_track(samples, n_frames=101, every=5)
    assert track.median_area == pytest.approx(100.0)


def test_smoothing_reduces_jitter_between_dense_noisy_detections():
    # A real face would not zigzag every frame; this is exactly the shape of noise `_smooth`
    # exists to remove. Raw interpolation reproduces every zigzag exactly (interpolation is exact
    # at its own anchors); after Savitzky-Golay smoothing the amplitude at those same points must
    # come down substantially.
    n = 25
    every = 3  # window = min(n, 2*every+1) = 7 >= 5, so `_smooth` actually runs
    ys = [0.0 if i % 2 == 0 else 20.0 for i in range(n)]
    samples = [(i, (0.0, ys[i], 10.0, 10.0)) for i in range(n)]
    track = ft._build_track(samples, n_frames=n, every=every)
    smoothed_ys = [track.box(i)[1] for i in range(n)]
    raw_amplitude = max(ys) - min(ys)
    smoothed_amplitude = max(smoothed_ys) - min(smoothed_ys)
    assert smoothed_amplitude < raw_amplitude * 0.5


def test_detected_is_true_within_one_sampling_interval_of_a_real_detection():
    samples = [(10, (0.0, 0.0, 10.0, 10.0))]
    track = ft._build_track(samples, n_frames=30, every=5)
    assert track.detected(10) is True   # the detection itself
    assert track.detected(5) is True    # exactly `every` frames away
    assert track.detected(15) is True   # exactly `every` frames away
    assert track.detected(4) is False   # every + 1 away
    assert track.detected(16) is False
    assert track.detected(0) is False


def test_box_and_detected_raise_index_error_out_of_range():
    track = ft._build_track([(0, (0.0, 0.0, 1.0, 1.0))], n_frames=5, every=1)
    for bad in (-1, 5, 100):
        with pytest.raises(IndexError):
            track.box(bad)
        with pytest.raises(IndexError):
            track.detected(bad)


# --------------------------------------------------------------------------------------------
# `detect_track` -- wiring, with the detector mocked (a list of boxes per call, as the brief asks)
# --------------------------------------------------------------------------------------------

def _frame_stamped(value: int) -> np.ndarray:
    """A tiny RGB frame whose pixels all equal `value`, so a fake `_detect_faces` can recover
    which frame it was called on without any real detection happening.
    """
    return np.full((2, 2, 3), value, dtype=np.uint8)


def test_detect_track_wires_sampling_selection_and_track_math_together(monkeypatch):
    fake_boxes_by_frame = {
        0: [(10.0, 10.0, 20.0, 20.0)],
        5: [(15.0, 10.0, 20.0, 20.0), (100.0, 100.0, 5.0, 5.0)],  # second is smaller: ignored
        10: [(20.0, 10.0, 20.0, 20.0)],
        # frames 1-4, 6-9, 11 are never sampled at every=5 and must never reach the detector
    }
    monkeypatch.setattr(ft, "_load_detector", lambda: object())

    def fake_detect_faces(detector, frame):
        frame_idx = int(frame[0, 0, 0])
        assert frame_idx in fake_boxes_by_frame, f"detector called on unsampled frame {frame_idx}"
        return fake_boxes_by_frame[frame_idx]

    monkeypatch.setattr(ft, "_detect_faces", fake_detect_faces)

    frames = [_frame_stamped(i) for i in range(12)]
    track = ft.detect_track(iter(frames), every=5)

    assert track is not None
    assert track.n_frames == 12
    x, y, w, h = track.box(5)
    assert (w, h) == pytest.approx((20.0, 20.0))  # the larger of the two frame-5 candidates


def test_detect_track_is_none_when_nothing_is_ever_detected(monkeypatch):
    monkeypatch.setattr(ft, "_load_detector", lambda: object())
    monkeypatch.setattr(ft, "_detect_faces", lambda detector, frame: [])
    frames = [_frame_stamped(i) for i in range(9)]
    assert ft.detect_track(iter(frames), every=5) is None


def test_detect_track_rejects_every_less_than_one():
    with pytest.raises(ValueError):
        ft.detect_track(iter([]), every=0)


def test_track_does_not_switch_between_two_similarly_sized_faces(monkeypatch):
    """Two static faces of nearly equal size (80px and 78px), far apart in frame, with YuNet-style
    detection jitter (sigma=3px) on both position and size, sampled every 5 frames across 100
    frames: `detect_track` must settle on ONE face's identity for the whole clip.

    Picking the larger detection independently in each *sampled frame* (no association across
    frames -- the pre-fix behavior) flips between the two faces roughly every other sample, since
    jitter alone is enough to make either one look larger on a given frame -- and interpolating/
    smoothing across that mixed selection sweeps the box across the entire distance between the
    two faces. A correct implementation associates detections into per-face candidate tracks first
    and keeps only the one with the larger median area, so the box stays near one face throughout.
    """
    rng = np.random.default_rng(20260817)
    n_frames = 100
    every = 5
    face_a = (50.0, 50.0, 80.0, 80.0)      # top-left corner, 80x80 -- center (90, 90)
    face_b = (500.0, 400.0, 78.0, 78.0)    # far away, 78x78 -- center (539, 439)

    def jittered(box):
        x, y, w, h = box
        return (
            x + rng.normal(0, 3), y + rng.normal(0, 3),
            w + rng.normal(0, 3), h + rng.normal(0, 3),
        )

    fake_boxes_by_frame = {
        frame_idx: [jittered(face_a), jittered(face_b)]
        for frame_idx in range(0, n_frames, every)
    }

    monkeypatch.setattr(ft, "_load_detector", lambda: object())

    def fake_detect_faces(detector, frame):
        frame_idx = int(frame[0, 0, 0])
        return fake_boxes_by_frame.get(frame_idx, [])

    monkeypatch.setattr(ft, "_detect_faces", fake_detect_faces)

    frames = [_frame_stamped(i) for i in range(n_frames)]
    track = ft.detect_track(iter(frames), every=every)

    assert track is not None
    sample_frames = range(0, n_frames, every)
    centers_x = [track.box(i)[0] + track.box(i)[2] / 2.0 for i in sample_frames]
    # If the track stayed on one face throughout, every sampled center clusters tightly around
    # either ~90 or ~539. If it kept switching identities (the bug), centers span nearly the full
    # ~450px distance between the two faces.
    spread = max(centers_x) - min(centers_x)
    assert spread < 50.0, f"track center x spread {spread:.1f}px across the clip -- looks like it switched faces"


# --------------------------------------------------------------------------------------------
# The missing-model-file refusal
# --------------------------------------------------------------------------------------------

def test_load_detector_missing_model_raises_runtime_error_naming_the_path_and_the_fix(
    monkeypatch, tmp_path
):
    missing = tmp_path / "no-such-model.onnx"
    monkeypatch.setattr(ft, "YUNET_MODEL_PATH", missing)
    with pytest.raises(RuntimeError) as excinfo:
        ft._load_detector()
    message = str(excinfo.value)
    assert str(missing) in message
    assert ft.YUNET_DOWNLOAD_URL in message


class _ExplodingIter:
    """An iterator that fails the test if `detect_track` ever pulls a frame from it -- used to
    prove the missing-model check happens before any frame is touched.
    """

    def __iter__(self):
        return self

    def __next__(self):
        raise AssertionError("frames_iter must not be consumed before the detector is validated")


def test_detect_track_validates_the_model_before_touching_frames_iter(monkeypatch, tmp_path):
    missing = tmp_path / "no-such-model.onnx"
    monkeypatch.setattr(ft, "YUNET_MODEL_PATH", missing)
    with pytest.raises(RuntimeError):
        ft.detect_track(_ExplodingIter(), every=5)


# --------------------------------------------------------------------------------------------
# Module hygiene
# --------------------------------------------------------------------------------------------

def test_facetrack_module_does_not_import_mlx():
    """`h3_48gb.facetrack` is a CPU-only module (see its module docstring) -- importing it must
    never pull in the MLX stack. Run in a subprocess so this test's own process (which may have
    already imported mlx via another test module) cannot hide a leak.
    """
    code = "import sys; import h3_48gb.facetrack; print('mlx' in sys.modules)"
    project_root = Path(__file__).resolve().parent.parent
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            cwd=str(project_root))
    assert result.returncode == 0, f"subprocess failed: {result.stderr}"
    assert result.stdout.strip() == "False", "importing h3_48gb.facetrack must not import mlx"


# --------------------------------------------------------------------------------------------
# Real YuNet, real frame -- the one test synthetic boxes cannot stand in for
# --------------------------------------------------------------------------------------------

def test_detect_track_finds_a_real_face_in_a_real_frame():
    if not ft.YUNET_MODEL_PATH.exists():
        pytest.skip(f"YuNet weights not present at {ft.YUNET_MODEL_PATH}")

    image_path = Path(__file__).parent / "data" / "face_frame.png"
    frame_bgr = cv2.imread(str(image_path))
    assert frame_bgr is not None, f"could not read {image_path}"
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width = frame_rgb.shape[:2]

    # The same real frame repeated: this test is about the detector genuinely finding a face in
    # real pixels and the plumbing turning that into a usable track, not about real motion.
    frames = [frame_rgb for _ in range(12)]
    track = ft.detect_track(iter(frames), every=5)

    assert track is not None
    assert track.n_frames == 12
    assert track.median_area > 0
    assert track.detected(0) is True
    for frame_idx in range(12):
        x, y, w, h = track.box(frame_idx)
        assert w > 0 and h > 0
        # The box should sit within (or, allowing for a little smoothing overshoot, barely
        # outside) the frame it was detected in -- not some detector-coordinate nonsense.
        assert -w < x < width
        assert -h < y < height
