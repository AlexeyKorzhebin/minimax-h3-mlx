"""The windowed v2v engine: window planning, crossfade compositing, partial-table selection.

Everything in here except the last test runs on the CPU with no weights on disk. That is the
point: the parts of `facerefine` that are easy to get silently wrong -- where the windows land,
which frames the crossfade ramp covers, which table name a sigma maps to -- are pure arithmetic,
and pinning them costs milliseconds instead of the 2.1 GPU-minutes one window actually takes.

The one GPU test is marked `slow` and skipped unless `H3_GPU_SMOKE=1` is set, because collecting
it into an ordinary `pytest tests/ -q` run would put a 2-minute, 28 GB job in the middle of a
suite that otherwise finishes in seconds.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from h3_48gb import facerefine
from h3_48gb.facerefine import (
    CROSSFADE_FRAMES,
    SIGMA_CEILING,
    WINDOW_FRAMES,
    WINDOW_STEP,
    Window,
    composite_windows,
    ensure_partial_table,
    native_frame_count,
    partial_sigmas,
    partial_table_name,
    plan_windows,
    refine_clip,
)


# -- the native 17k+5 grid -----------------------------------------------------------------------

def test_native_frame_count_rounds_a_length_down_to_the_grid_never_up():
    # 17k + 5: 5, 22, 39, 56, 73 ... A count rounded *up* would need frames that do not exist.
    assert native_frame_count(56) == 56
    assert native_frame_count(55) == 39
    assert native_frame_count(40) == 39
    assert native_frame_count(39) == 39
    assert native_frame_count(38) == 22
    assert native_frame_count(5) == 5


def test_native_frame_count_rejects_a_clip_below_the_shortest_native_length():
    with pytest.raises(ValueError, match="5"):
        native_frame_count(4)


def test_the_default_window_is_the_one_the_experiments_ran():
    # 56 = 17*3 + 5 -> 17 latent frames. Documented in docs/FEASIBILITY-face-refine.md.
    assert (WINDOW_FRAMES, WINDOW_STEP, CROSSFADE_FRAMES) == (56, 42, 12)
    assert WINDOW_FRAMES % 17 == 5


# -- window planning -----------------------------------------------------------------------------

def test_plan_windows_on_exactly_one_window_is_one_window():
    assert plan_windows(56) == [Window(0, 56)]


def test_plan_windows_reproduces_the_round_three_overlap_experiment():
    """84 frames -> windows 0-55 and 28-83, the exact pair round 3 measured the seam on."""
    plan = plan_windows(84)
    assert plan == [Window(0, 56), Window(28, 56)]
    assert plan[0].end - plan[1].start == 28          # the 28-frame overlap of the experiment


def test_plan_windows_covers_every_frame_of_a_long_clip_and_pins_the_last_window_to_the_tail():
    plan = plan_windows(361)
    assert [w.start for w in plan] == [0, 42, 84, 126, 168, 210, 252, 294, 305]
    assert all(w.length == 56 for w in plan)
    assert plan[-1].end == 361                        # the tail window is pinned, not padded
    covered = np.zeros(361, bool)
    for w in plan:
        covered[w.start:w.end] = True
    assert covered.all()


def test_plan_windows_pinned_tail_overlaps_more_never_less():
    plan = plan_windows(361)
    overlaps = [plan[i].end - plan[i + 1].start for i in range(len(plan) - 1)]
    assert min(overlaps) >= CROSSFADE_FRAMES
    assert overlaps[-1] == 45                         # 294+56 - 305, the squeezed tail


@pytest.mark.parametrize("num_frames", [56, 57, 70, 84, 97, 98, 99, 145, 240, 243, 361, 720])
def test_plan_windows_always_covers_the_whole_clip_with_room_for_the_crossfade(num_frames):
    plan = plan_windows(num_frames)
    assert plan[0].start == 0 and plan[-1].end == num_frames
    starts = [w.start for w in plan]
    assert starts == sorted(set(starts))              # strictly increasing, no duplicate window
    for i in range(len(plan) - 1):
        assert plan[i].end - plan[i + 1].start >= CROSSFADE_FRAMES


def test_plan_windows_on_a_clip_shorter_than_the_window_covers_it_fully_on_the_native_grid():
    """Fix round 1: a clip shorter than `window` used to get *one* window truncated down to the
    native grid and leave the tail frames un-refined -- 16 of 55, 16 of 38 (42% of the clip). The
    effective window length is still `native_frame_count(num_frames)`, but the plan now lays down
    as many of them as it takes to reach the last frame, so nothing is left un-refined."""
    assert plan_windows(39) == [Window(0, 39)]
    assert plan_windows(22) == [Window(0, 22)]
    assert plan_windows(38) == [Window(0, 22), Window(16, 22)]     # overlap 6, per the brief
    assert plan_windows(40) == [Window(0, 39), Window(1, 39)]
    assert plan_windows(55) == [Window(0, 39), Window(16, 39)]     # the 16-frame gap is gone


@pytest.mark.parametrize("num_frames", [6, 11, 21, 33, 38])
def test_short_clip_overlap_below_crossfade_is_a_known_limit_composite_shrinks_the_ramp(num_frames):
    """Re-review of fix round 1: on sub-window clips the effective window can be so short that no
    plan can hold a full `CROSSFADE_FRAMES` overlap (a 6-frame clip fits only 5-frame windows).
    That is a geometric limit, not a planner bug -- what must hold instead is that
    `composite_windows` takes such a plan as-is and shrinks the ramp, because the R3 crossfade
    requirement is about seams between full windows, which a clip this short does not have."""
    plan = plan_windows(num_frames)
    assert plan[0].start == 0 and plan[-1].end == num_frames
    source = np.zeros((num_frames, 4, 4, 3), dtype=np.uint8)
    refined = [np.full((w.length, 4, 4, 3), 255, dtype=np.uint8) for w in plan]
    out = composite_windows(source, refined, plan)
    assert out.shape == source.shape
    assert (out == 255).all()                         # every frame covered, no source leaks


def test_plan_windows_never_leaves_a_frame_uncovered_at_any_length_from_five_up():
    """The regression this fix round exists for, pinned directly: at every legal clip length,
    every frame belongs to at least one window. (Failed on length 55 -- and the whole 5..120
    sweep -- against the pre-fix `plan_windows`, which covered only `native_frame_count(55) == 39`
    of the 55 frames.)"""
    for num_frames in range(5, 121):
        plan = plan_windows(num_frames)
        covered = np.zeros(num_frames, dtype=bool)
        for w in plan:
            covered[w.start:w.end] = True
        assert covered.all(), f"{num_frames} frames: {plan} leaves a gap"


def test_plan_windows_rejects_a_clip_shorter_than_the_shortest_native_window():
    with pytest.raises(ValueError):
        plan_windows(4)


def test_plan_windows_rejects_a_window_length_off_the_native_grid():
    with pytest.raises(ValueError, match="17"):
        plan_windows(200, window=55)


def test_plan_windows_rejects_a_step_that_leaves_no_overlap_for_the_crossfade():
    with pytest.raises(ValueError, match="crossfade"):
        plan_windows(200, window=56, step=50, crossfade=12)


def test_plan_windows_rejects_a_nonpositive_or_oversized_step():
    with pytest.raises(ValueError):
        plan_windows(200, step=0)
    with pytest.raises(ValueError):
        plan_windows(200, step=57)


def test_plan_windows_drops_a_penultimate_window_the_tail_already_makes_redundant():
    """99 frames would otherwise plan [0, 42, 43]: the window at 42 sits almost on top of the
    pinned tail at 43 and adds about one new frame for a full window's 2.2 GPU-minutes. Once its
    predecessor (start 0) already reaches within `crossfade` of the tail, the middle window is
    dropped."""
    plan = plan_windows(99)
    assert plan == [Window(0, 56), Window(43, 56)]
    assert plan[0].end - plan[1].start == 13                   # still >= CROSSFADE_FRAMES

    plan = plan_windows(141)
    assert plan == [Window(0, 56), Window(42, 56), Window(85, 56)]
    overlaps = [plan[i].end - plan[i + 1].start for i in range(len(plan) - 1)]
    assert all(o >= CROSSFADE_FRAMES for o in overlaps)
    covered = np.zeros(141, bool)
    for w in plan:
        covered[w.start:w.end] = True
    assert covered.all()


# -- crossfade compositing -----------------------------------------------------------------------

def _ramp(fade: int) -> np.ndarray:
    """The reference's own weights (round2/overlap/crossfade.py): (i + 1) / (fade + 1)."""
    return np.array([(i + 1) / (fade + 1) for i in range(fade)], np.float32)


def _flat(num_frames: int, value: int) -> np.ndarray:
    return np.full((num_frames, 6, 8, 3), value, np.uint8)


def test_composite_of_a_single_window_is_that_window_verbatim():
    source = _flat(56, 7)
    refined = [_flat(56, 200)]
    out = composite_windows(source, refined, plan_windows(56))
    assert out.dtype == np.uint8 and out.shape == source.shape
    assert (out == 200).all()


def test_composite_passes_frames_no_window_covers_through_byte_for_byte():
    """`plan_windows` itself no longer produces a plan with a gap (fix round 1), but
    `composite_windows` still has to behave on one, because nothing stops a caller from
    hand-building one -- so the property is pinned directly against a hand-built plan rather than
    through `plan_windows`."""
    rng = np.random.default_rng(0)
    source = rng.integers(0, 256, (40, 6, 8, 3), dtype=np.uint8)
    plan = [Window(0, 39)]                              # frame 39 covered by no window
    refined = [_flat(39, 200)]
    out = composite_windows(source, refined, plan)
    assert (out[39] == source[39]).all()               # the un-refined tail frame is untouched
    assert (out[:39] == 200).all()


def test_composite_hands_over_through_the_reference_ramp_centred_in_the_overlap():
    source = _flat(84, 0)
    a, b = _flat(56, 10), _flat(56, 200)
    plan = plan_windows(84)
    out = composite_windows(source, [a, b], plan, crossfade=12)

    # overlap 28..55, fade 12 frames -> starts at 28 + (28 - 12)//2 = 36, exactly as the
    # experiment's FADE_START.
    assert (out[:36] == 10).all()
    assert (out[48:] == 200).all()
    weights = _ramp(12)
    expected = np.floor((1 - weights) * 10 + weights * 200 + 0.5).astype(np.uint8)
    assert (out[36:48, 0, 0, 0] == expected).all()


def test_composite_never_leaves_a_hard_cut_when_the_overlap_is_narrower_than_the_crossfade():
    """A pinned tail can overlap by less than `crossfade` only if a caller shrinks it; the fade
    then shrinks with the overlap rather than reading outside the window."""
    source = _flat(70, 0)
    plan = [Window(0, 56), Window(52, 18)]
    out = composite_windows(source, [_flat(56, 10), _flat(18, 200)], plan, crossfade=12)
    assert (out[:52] == 10).all()
    assert (out[56:] == 200).all()
    middle = out[52:56, 0, 0, 0].astype(int)
    assert 10 < middle.min() and middle.max() < 200     # a real ramp, not a cut
    assert list(middle) == sorted(middle)


def test_composite_rejects_a_window_whose_frames_do_not_match_the_plan():
    with pytest.raises(ValueError):
        composite_windows(_flat(84, 0), [_flat(56, 1), _flat(55, 2)], plan_windows(84))


def test_composite_tail_window_overlapping_two_predecessors_hands_off_cleanly():
    """The 361-frame plan's pinned tail (start 305) overlaps *two* earlier windows at once: 45
    frames of window 294-350 and, past that, 3 frames of window 252-308. Compositing against the
    running accumulator rather than pairwise against only the immediately preceding window is what
    the module docstring says makes this safe -- pin it: no frame is dropped, none is left at a
    value from neither neighbour, and the run is monotonic through each handoff (values only ever
    move from one window's flat value toward the next's, never overshoot and back)."""
    plan = plan_windows(361)
    assert [w.start for w in plan] == [0, 42, 84, 126, 168, 210, 252, 294, 305]      # unchanged

    source = _flat(361, 0)
    values = [10 * (i + 1) for i in range(len(plan))]          # 10, 20, ..., 90: one per window
    refined = [_flat(w.length, v) for w, v in zip(plan, values)]
    out = composite_windows(source, refined, plan, crossfade=CROSSFADE_FRAMES)

    assert out.shape == source.shape and out.dtype == np.uint8
    trace = out[:, 0, 0, 0].astype(int)
    # No composite value falls outside the range the contributing windows could produce -- a
    # dropped accumulator write or a mis-registered fade would show up as over/undershoot.
    assert trace.min() >= min(values) and trace.max() <= max(values)
    # Frames the tail window (value 90) owns outright, past every fade, land exactly on it --
    # including the stretch inside the triple-overlap region that is closest to the tail's own
    # start, which only a correct accumulator hand-off reaches undamaged.
    assert (trace[350:361] == values[-1]).all()
    # Monotonic end to end: every 3-window-deep region is still just a chain of two-window ramps,
    # so the trace never reverses direction against the window values' own ascending order.
    diffs = np.diff(trace)
    assert (diffs >= 0).all()


# -- the partial AdaLN table ---------------------------------------------------------------------

def test_partial_table_name_is_the_two_decimal_sigma_the_experiments_baked():
    assert partial_table_name(0.25) == "adaln_face_s025_4pt_turbo.safetensors"
    assert partial_table_name(0.15) == "adaln_face_s015_4pt_turbo.safetensors"
    assert partial_table_name(0.2) == "adaln_face_s020_4pt_turbo.safetensors"


def test_partial_sigma_grid_matches_the_table_baked_for_sigma_025():
    """The s0.25 grid documented in the plan: [0.250, 0.180, 0.098, 0] (the grid is really 0.1805,
    rounded to three decimals for the docstring)."""
    grid = partial_sigmas(0.25, 4, 12.0)
    assert grid[0] == pytest.approx(0.25, abs=1e-6)
    assert [round(float(v), 3) for v in grid] == [0.250, 0.180, 0.098, 0.0]
    assert grid[-1] == 0.0
    assert list(grid) == sorted(grid, reverse=True)


def test_partial_sigma_grid_is_shifted_per_modality_off_one_base_grid():
    """Video (shift 12) and audio (shift 3) must sit on the same unshifted base grid, which is
    what keeps the pair on the (video_t, audio_t) diagonal every training step sat on."""
    video, audio = partial_sigmas(0.25, 4, 12.0), partial_sigmas(0.25, 4, 3.0)

    def unshift(s, shift):
        return s / (shift - (shift - 1.0) * s)

    assert [unshift(float(v), 12.0) for v in video] == pytest.approx(
        [unshift(float(a), 3.0) for a in audio], abs=1e-7)


_SHIPPED_S025_TABLE = facerefine.DEFAULT_ADALN_DIR / partial_table_name(0.25)


@pytest.mark.skipif(not _SHIPPED_S025_TABLE.exists(),
                    reason=f"no {_SHIPPED_S025_TABLE} on this machine")
def test_partial_sigmas_matches_the_grid_baked_into_the_shipped_s025_table():
    """`partial_sigmas` isn't just consistent with itself -- it is the function `bake_partial.py`
    used to bake the table that is actually sitting in `~/models/turbo/`, so it has to reproduce
    that table's grids bit for bit, not just to three decimals."""
    import mlx.core as mx

    table = mx.load(str(_SHIPPED_S025_TABLE))
    video = partial_sigmas(0.25, facerefine.GRID_POINTS, facerefine.VIDEO_SHIFT)
    assert np.array(table["video_sigmas"]).tolist() == video.tolist()      # bit for bit
    if "audio_sigmas" in table:
        audio = partial_sigmas(0.25, facerefine.GRID_POINTS, facerefine.AUDIO_SHIFT)
        assert np.array(table["audio_sigmas"]).tolist() == audio.tolist()  # bit for bit


def test_ensure_partial_table_returns_an_existing_table_and_bakes_nothing(tmp_path, monkeypatch):
    table = tmp_path / partial_table_name(0.25)
    table.write_bytes(b"not really a table")
    monkeypatch.setattr(facerefine, "_bake_partial_table",
                        lambda *a, **k: pytest.fail("baked over an existing table"))
    assert ensure_partial_table(0.25, None, tmp_path) == table


def test_ensure_partial_table_bakes_the_four_point_grid_when_the_table_is_missing(tmp_path,
                                                                                 monkeypatch):
    calls = []

    def fake_bake(sigma, dest, points, lora, strength, verbose):
        calls.append((sigma, dest, points, lora, strength))
        Path(dest).write_bytes(b"baked")

    monkeypatch.setattr(facerefine, "_bake_partial_table", fake_bake)
    lora = tmp_path / facerefine.TURBO_LORA_NAME
    lora.write_bytes(b"lora")
    monkeypatch.setattr(facerefine, "_bake_inputs_present", lambda adaln_dir, lora: None)

    out = ensure_partial_table(0.15, None, tmp_path)
    assert out == tmp_path / "adaln_face_s015_4pt_turbo.safetensors"
    assert calls == [(0.15, out, 4, lora, 1.0)]


def test_ensure_partial_table_says_what_is_missing_instead_of_failing_inside_the_baker(tmp_path):
    with pytest.raises(RuntimeError) as excinfo:
        ensure_partial_table(0.25, None, tmp_path)
    assert "adaln_curve.safetensors" in str(excinfo.value) or \
           facerefine.TURBO_LORA_NAME in str(excinfo.value)


# -- refine_clip's guard rails (no GPU, no weights) ----------------------------------------------

def _gray_crops(num_frames: int = 56) -> np.ndarray:
    return np.full((num_frames, 288, 448, 3), 128, np.uint8)


def test_refine_clip_refuses_a_sigma_above_the_measured_ceiling():
    with pytest.raises(ValueError, match="0.25"):
        refine_clip(_gray_crops(), sigma=0.4, checkpoint=Path("/nonexistent"))


def test_refine_clip_refuses_a_sigma_at_or_below_zero():
    with pytest.raises(ValueError):
        refine_clip(_gray_crops(), sigma=0.0, checkpoint=Path("/nonexistent"))


def test_the_sigma_ceiling_is_the_round_two_measurement():
    assert SIGMA_CEILING == 0.25


def test_refine_clip_refuses_crops_that_are_not_a_uint8_rgb_stack():
    with pytest.raises(ValueError):
        refine_clip(np.zeros((56, 288, 448), np.uint8), checkpoint=Path("/nonexistent"))
    with pytest.raises(ValueError):
        refine_clip(np.zeros((56, 288, 448, 3), np.float32), checkpoint=Path("/nonexistent"))


def test_refine_clip_refuses_a_crop_whose_sides_are_not_multiples_of_the_vae_ratio():
    with pytest.raises(ValueError, match="16"):
        refine_clip(np.zeros((56, 100, 448, 3), np.uint8), checkpoint=Path("/nonexistent"))


def test_validate_request_rejects_a_sigma_that_rounds_to_zero_only_after_rounding():
    """0.004 clears the raw `sigma > 0.0` check but rounds to the `0.00` table name -- checked
    again after quantizing so it does not silently become a no-op refine."""
    with pytest.raises(ValueError):
        facerefine._validate_request(_gray_crops(), 0.004)
    # A sigma that survives rounding still works.
    assert facerefine._validate_request(_gray_crops(), 0.006) == pytest.approx(0.01)


def test_validate_request_accepts_numpy_floats():
    assert facerefine._validate_request(_gray_crops(), np.float32(0.25)) == pytest.approx(0.25)
    assert facerefine._validate_request(_gray_crops(), np.float64(0.15)) == pytest.approx(0.15)


def test_validate_request_rejects_a_bool_sigma():
    """`bool` is an `int` subclass, so `sigma=True` would otherwise sail through as `1.0`."""
    with pytest.raises(ValueError):
        facerefine._validate_request(_gray_crops(), True)
    with pytest.raises(ValueError):
        facerefine._validate_request(_gray_crops(), False)


def test_refine_clip_has_no_turbo_strength_parameter():
    """No round of experiments varied the backbone LoRA strength -- YAGNI'd out of the public
    signature. The table's own `strength` still exists on `ensure_partial_table`, always called
    at 1.0 from `refine_clip`."""
    import inspect
    assert "turbo_strength" not in inspect.signature(refine_clip).parameters
    with pytest.raises(TypeError):
        refine_clip(_gray_crops(), checkpoint=Path("/nonexistent"), turbo_strength=0.5)


def test_refine_clip_happy_path_on_cpu_with_the_gpu_half_mocked(monkeypatch, tmp_path):
    """Everything except `_run_windows` runs for real: validate -> plan -> ensure_table ->
    composite, on a clip long enough to need more than one window (84 frames -> 2 windows). Proves
    the wiring, not the v2v mechanics -- those are the GPU smoke test's job."""
    crops = _gray_crops(84)
    fake_table = tmp_path / "fake_table.safetensors"
    monkeypatch.setattr(facerefine, "ensure_partial_table", lambda *a, **k: fake_table)

    def fake_run_windows(crops_, plan, table, **kwargs):
        assert table == fake_table
        # One grey level per window, so the composite's crossfade is exercised for real.
        return [np.full((w.length,) + crops_.shape[1:], 100 + 50 * i, np.uint8)
                for i, w in enumerate(plan)]

    monkeypatch.setattr(facerefine, "_run_windows", fake_run_windows)

    out = refine_clip(crops, checkpoint=Path("/nonexistent"))
    assert out.shape == crops.shape
    assert out.dtype == np.uint8


# -- D1 (Task 5 gate): scheduler state must reset between windows -------------------------------

_D1_GRID = [0.25, 0.1805, 0.0984, 0.0]     # the shipped s0.25 4-point grid; sigmas[-1] == 0.0


def _run_one_window(sched, num_steps: int):
    """One window's worth of Euler steps, standing in for the inner loop of `_run_windows`'s
    phase 2 -- a fixed model output plays the transformer's role, since the bug lives entirely in
    the scheduler's own step-counter bookkeeping and never touches the DiT."""
    import mlx.core as mx

    x = mx.ones((2, 4))
    for t in sched.timesteps.tolist()[:num_steps]:
        x = sched.step(mx.zeros((2, 4)) + 0.1, float(t), x)
    return x


@pytest.mark.mlx
def test_run_windows_scheduler_state_is_finite_after_two_windows():
    """D1 regression (Task 5's integration gate). `_run_windows` builds `video_sched` once, in
    phase 2, *before* the loop over windows (`facerefine.py`'s phase-2 comment), and
    `MiniMaxH3Scheduler._step_index` is cleared only by `set_timesteps` -- never by finishing a
    window's steps. Two windows run back to back on one scheduler therefore walk the first
    window's `_step_index` (3, for this 4-value grid) straight into the second window's first
    `.step()` call, which reads `sigmas[3] == 0.0` over `sigmas[4]` (past the end of the array;
    MLX returns 0.0 there) -- `0.0 / 0.0` is NaN, and every latent from window 2 on is NaN, which
    decodes to a black rectangle. `facerefine._reset_window_schedule` is `_run_windows`'s fix:
    called at the top of each window's loop body, right where this test calls it between windows
    one and two."""
    from h3_48gb import _upstream
    _upstream.ensure_on_path()
    from minimax_h3_mlx.scheduler import MiniMaxH3Scheduler
    import mlx.core as mx

    # First, the bug itself, pinned directly: two windows run back to back on one scheduler with
    # no reset in between -- exactly what `_run_windows` did before this fix -- go non-finite.
    # This is the scheduler's own documented step-counter behaviour (`upstream/` is not touched by
    # this fix), so it is expected to hold forever, not just until the next refactor.
    broken = MiniMaxH3Scheduler()
    broken.set_timesteps(sigmas=_D1_GRID)          # mirrors `_build_schedules`, run once
    _run_one_window(broken, len(_D1_GRID) - 1)     # window 1: leaves `_step_index` at 3
    nan_out = _run_one_window(broken, len(_D1_GRID) - 1)   # window 2, no reset
    assert not bool(mx.all(mx.isfinite(nan_out)).item()), \
        "the un-reset scheduler no longer reproduces D1 -- has the upstream scheduler changed?"

    # Now the fix: the same two windows, with `facerefine._reset_window_schedule` called where
    # `_run_windows`'s loop calls it, at the top of each window's iteration.
    sched = MiniMaxH3Scheduler()
    sched.set_timesteps(sigmas=_D1_GRID)
    _run_one_window(sched, len(_D1_GRID) - 1)      # window 1
    facerefine._reset_window_schedule(sched, _D1_GRID)   # the fix, applied where window 2 starts
    out = _run_one_window(sched, len(_D1_GRID) - 1)      # window 2

    assert bool(mx.all(mx.isfinite(out)).item()), \
        "window 2's output is not finite -- the scheduler carried window 1's step index over"


@pytest.mark.mlx
def test_reset_window_schedule_clears_the_step_index_before_a_window_starts():
    """The other half of D1: `_reset_window_schedule` must actually clear `_step_index`, not just
    happen to leave the arithmetic finite for this one grid. `step_index` is `None` only right
    after `set_timesteps`; a window that has already taken a step moves it forward, and only a
    reset -- not merely re-reading the same grid -- brings it back to the start."""
    from h3_48gb import _upstream
    _upstream.ensure_on_path()
    from minimax_h3_mlx.scheduler import MiniMaxH3Scheduler

    sched = MiniMaxH3Scheduler()
    sched.set_timesteps(sigmas=_D1_GRID)
    assert sched.step_index is None

    _run_one_window(sched, len(_D1_GRID) - 1)
    assert sched.step_index == len(_D1_GRID) - 1      # advanced past window 1, not reset by itself

    facerefine._reset_window_schedule(sched, _D1_GRID)
    assert sched.step_index is None                   # a fresh window, ready to start at index 0


# -- the one GPU test ----------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("H3_GPU_SMOKE") != "1",
                    reason="2 GPU-minutes and 28 GB; run with H3_GPU_SMOKE=1")
def test_refine_clip_smoke_on_one_window_of_grey_frames():
    checkpoint = Path.home() / "models/h3-8bit-full"
    if not checkpoint.is_dir():
        pytest.skip(f"no checkpoint at {checkpoint}")
    crops = _gray_crops(56)
    out = refine_clip(crops, sigma=0.25, checkpoint=checkpoint)
    assert out.shape == crops.shape
    assert out.dtype == np.uint8
