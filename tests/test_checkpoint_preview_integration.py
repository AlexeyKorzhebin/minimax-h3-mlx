"""Prove `h3_48gb.checkpoint` and `h3_48gb.preview` work *together*, not just separately.

    ./.venv/bin/pytest tests/test_checkpoint_preview_integration.py -q

Both features were built concurrently against the same seam -- `pipeline.dit` -- and are on by
default in `run_bench.py` (`--preview-every` defaults to 5, checkpointing is opt-out via
`--no-checkpoint`), so their combination is the default path a real run takes, not an edge case.
`tests/test_checkpoint.py` proves checkpointing alone is exact; `test_preview.py` proves preview
alone is safe. Neither exercises the other, which is exactly the gap the checkpoint author's own
note flags:

    its `PreviewInterceptor` also wraps `self.dit` (`preview.py:276`). Both are
    attribute-forwarding proxies so nesting works, but whoever installs second must wrap rather
    than replace.

`h3_48gb.pipeline.LazyMiniMaxH3Pipeline.__call__` installs `PreviewInterceptor` first (wrapping
the raw `dit`), then delegates to `CheckpointingPipeline.__call__` (via the MRO), which installs
`StepInterceptor` second (wrapping whatever `self.dit` currently is -- the preview interceptor).
So at the moment the denoising loop actually calls `self.dit(...)`, the chain is
`StepInterceptor(PreviewInterceptor(real_dit))`: checkpoint outermost, preview innermost. That is
what makes a fast-forwarded step invisible to preview (`StepInterceptor` returns before ever
calling into its `_inner`) without preview needing to know checkpointing exists at all. Verifying
that this is exactly what happens -- and that it is not an accident of the order two lines happen
to be written in -- is what this file is for.

Reuses `tests/test_checkpoint.py`'s stub infrastructure (`ToyPipeline`, `_ToyDiT`, `run`,
`identical`) rather than inventing a second set, and `test_preview.py`'s tiny real `VideoVAE` so a
preview genuinely decodes instead of only ever exercising the VAE-less fallback. Nothing here loads
the real 33B transformer or a real-sized VAE -- one native step costs 586s, which has no place in a
test suite.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.mlx
pytest.importorskip("mlx.core", reason="mlx: needs the MLX stack, absent here")


import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from h3_48gb import _upstream  # noqa: E402,F401

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from minimax_h3_mlx.config import DiTConfig, PipelineConfig  # noqa: E402

import h3_48gb.pipeline as pipeline_mod  # noqa: E402
from h3_48gb.checkpoint import CheckpointStore, StepInterceptor  # noqa: E402
from h3_48gb.pipeline import LazyMiniMaxH3Pipeline  # noqa: E402
from h3_48gb.preview import PreviewInterceptor, pop_preview_kwargs, preview_path  # noqa: E402

from test_checkpoint import (  # noqa: E402
    CANVAS,
    DURATION,
    STEPS,
    Interrupt,
    ToyPipeline,
    _Config,
    _ToyDiT,
    _ToyTextEncoder,
    identical,
    run as run_toy,
)
from test_preview import tiny_video_vae  # noqa: E402


# -- shared geometry ------------------------------------------------------------------------------

def _preview_geometry(pipeline):
    """The same derivation `LazyMiniMaxH3Pipeline._install_preview` does, sized for the fixed
    CANVAS/DURATION request `run_toy` (== `test_checkpoint.run`) issues."""
    from minimax_h3_mlx.packing import FPS, align_num_frames, video_latent_num_frames

    ratio = pipeline.video_vae.config.spatial_compression_ratio
    latent_height = latent_width = CANVAS // ratio
    num_frames = align_num_frames(int(round(DURATION * FPS)))
    num_latent_frames = video_latent_num_frames(num_frames)
    patch_size = pipeline.dit.config.patch_size
    return num_latent_frames, latent_height, latent_width, patch_size


class CountingPreviewInterceptor(PreviewInterceptor):
    """`PreviewInterceptor`, instrumented with a call counter.

    A fast-forwarded step must never reach this wrapper at all -- not "reach it and be a no-op",
    literally never call `__call__`. Counting invocations is the only way to distinguish "correctly
    invisible" from "visible but happens to render nothing", which a JPEG-existence check alone
    cannot do.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.invocations = 0

    def __call__(self, *args, **kwargs):
        self.invocations += 1
        return super().__call__(*args, **kwargs)


class PreviewFirstPipeline(ToyPipeline):
    """`ToyPipeline` plus a preview interceptor installed in the same order and the same place
    `LazyMiniMaxH3Pipeline.__call__` installs one: before delegating to `CheckpointingPipeline`'s
    `__call__` (reached via `super()`, since this class does not otherwise override it), which then
    wraps a `StepInterceptor` *around* whatever `self.dit` already is. That reproduces the
    production nesting exactly -- see the module docstring.

    `video_vae` is swapped for a real (tiny, 2-decoder-layer) `VideoVAE` -- the same one
    `test_preview.py` builds -- instead of `ToyPipeline`'s config-only stand-in, so a preview here
    genuinely decodes through `render_preview_frame` rather than only ever falling back to the
    VAE-less heatmap.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.video_vae = tiny_video_vae()
        self.preview_interceptor: CountingPreviewInterceptor | None = None

    def __call__(self, *args, **kwargs):
        preview_opts = pop_preview_kwargs(kwargs)
        every = int(preview_opts.get("preview_every", 0) or 0)
        stem = preview_opts.get("preview_stem")
        original_dit = self.dit
        if every:
            num_latent_frames, latent_height, latent_width, patch_size = _preview_geometry(self)
            self.preview_interceptor = CountingPreviewInterceptor(
                original_dit, self, every, stem, self.keyframe_rows, num_latent_frames,
                latent_height, latent_width, patch_size, verbose=False,
            )
            self.dit = self.preview_interceptor
        try:
            return super().__call__(*args, **kwargs)
        finally:
            self.dit = original_dit


def _eligible_preview_steps(every: int) -> list[int]:
    """Forward indices at which `PreviewInterceptor` fires: `step > 0 and step % every == 0`."""
    return [i for i in range(STEPS - 1) if i > 0 and i % every == 0]


class RealLazyPipeline(LazyMiniMaxH3Pipeline):
    """`h3_48gb.pipeline.LazyMiniMaxH3Pipeline` itself -- the actual class both features compose
    through in production -- fed the same toy stand-ins `ToyPipeline` uses.

    `PreviewFirstPipeline` above hand-mirrors the install order `LazyMiniMaxH3Pipeline.__call__`
    uses; this class instead runs the *real* `__call__` / `_install_preview` / `_build_schedules`,
    so a future change to the real order in `h3_48gb/pipeline.py` would be caught here even if
    `PreviewFirstPipeline`'s mirror silently drifted from it.
    """

    def __init__(self, fail_at: int | None = None, keyframe_rows: int = 0):
        dit_config = DiTConfig(patch_size=(1, 2, 2), latents_dim=4, audio_latents_dim=6, text_dim=8)
        audio_vae = _Config(config=_Config(latent_channels=6, sampling_rate=32000,
                                           latents_mean=(0.0,) * 6, latents_std=(1.0,) * 6))
        super().__init__(
            _ToyDiT(dit_config, fail_at=fail_at), _ToyTextEncoder(dit_config.text_dim),
            tiny_video_vae(), audio_vae, PipelineConfig(), verbose=False,
        )
        self.keyframe_rows = keyframe_rows
        self.captured: dict[str, np.ndarray] = {}

    def _ensure_cache(self, timesteps, drop_adaln, verbose):
        return None

    def _encode_keyframes(self, images, height, width):
        patch = self.dit.config.patch_size
        dim = self.dit.config.latents_dim * patch[0] * patch[1] * patch[2]
        rows = mx.arange(self.keyframe_rows * dim, dtype=mx.float32).reshape(self.keyframe_rows, dim)
        return rows / 1000.0

    def _decode_video(self, rows, *args, **kwargs):
        self.captured["video"] = np.array(rows, copy=True)
        return np.zeros((1, 1, 1, 3), dtype=np.uint8)

    def _decode_audio(self, rows, *args, **kwargs):
        self.captured["audio"] = np.array(rows, copy=True)
        return np.zeros((2, 1), dtype=np.float32)


def test_real_lazy_pipeline_writes_previews_and_checkpoints_in_one_run(tmp_path):
    """Requirement 1, against the real production class rather than a mirror of it."""
    path = tmp_path / "run.safetensors"
    stem = tmp_path / "clip"
    pipeline = RealLazyPipeline()

    run_toy(pipeline, checkpoint_path=path, keep_checkpoint=True,
            preview_every=2, preview_stem=str(stem))

    loaded = CheckpointStore(path).read()
    assert loaded.completed_steps == loaded.total_forwards == STEPS - 1

    expected = _eligible_preview_steps(2)
    for step in expected:
        assert preview_path(stem, step).exists(), f"no preview JPEG for step {step}"


def test_real_lazy_pipeline_resume_is_identical_and_hides_fast_forwarded_steps(tmp_path, monkeypatch):
    """Requirement 2 and the correct half of requirement 3, against the real production class:
    resuming with both features enabled reproduces an uninterrupted both-enabled run bit for bit,
    re-enters the transformer only for the steps actually left to do, and -- verified here by
    swapping in a counting subclass at the point `h3_48gb.pipeline` constructs it -- never lets a
    fast-forwarded step reach the preview wrapper.
    """
    created: list[CountingPreviewInterceptor] = []

    def make_counting(*args, **kwargs):
        interceptor = CountingPreviewInterceptor(*args, **kwargs)
        created.append(interceptor)
        return interceptor

    monkeypatch.setattr(pipeline_mod, "PreviewInterceptor", make_counting)

    reference = run_toy(RealLazyPipeline(), preview_every=2, preview_stem=str(tmp_path / "ref"))

    path = tmp_path / "run.safetensors"
    stem = tmp_path / "clip"
    crashed = RealLazyPipeline(fail_at=4)
    try:
        run_toy(crashed, checkpoint_path=path, preview_every=2, preview_stem=str(stem))
        raise AssertionError("expected a crash")
    except Interrupt:
        pass
    assert path.exists()
    assert created[1].invocations == 5, \
        "the crashed run should have reached the preview wrapper for forwards 0..4"

    resumed_pipeline = RealLazyPipeline()
    resumed = run_toy(resumed_pipeline, checkpoint_path=path, preview_every=2, preview_stem=str(stem))

    assert identical(reference, resumed), "resumed latents differ from the both-enabled reference run"
    assert resumed_pipeline.dit.calls == 4, \
        f"resume re-entered the transformer {resumed_pipeline.dit.calls} times instead of 4"
    assert not path.exists()
    assert created[2].invocations == 4, (
        f"the resumed run reached the preview wrapper {created[2].invocations} times; only the 4 "
        "forwards actually re-run (4, 5, 6, 7) should, not the 4 fast-forwarded ones (0, 1, 2, 3)"
    )


# -- 1. both features fire in the same run ---------------------------------------------------------

def test_previews_and_checkpoints_both_happen_in_one_run(tmp_path):
    path = tmp_path / "run.safetensors"
    stem = tmp_path / "clip"
    pipeline = PreviewFirstPipeline()

    run_toy(pipeline, checkpoint_path=path, keep_checkpoint=True,
            preview_every=2, preview_stem=str(stem))

    loaded = CheckpointStore(path).read()
    assert loaded.completed_steps == loaded.total_forwards == STEPS - 1, \
        "the checkpoint did not record a complete run"

    expected = _eligible_preview_steps(2)
    assert expected, "test is vacuous if no step in this schedule is preview-eligible"
    for step in expected:
        assert preview_path(stem, step).exists(), f"no preview JPEG for step {step}"
    written = sorted(tmp_path.glob("clip-preview-step*.jpg"))
    assert len(written) == len(expected), f"expected {len(expected)} previews, found {written}"


# -- 2. resume with both enabled: bit-identical, and skipped steps stay skipped ---------------------

def test_resume_with_both_enabled_is_identical_and_skips_completed_forwards(tmp_path):
    reference_pipeline = PreviewFirstPipeline()
    reference = run_toy(reference_pipeline, preview_every=2, preview_stem=str(tmp_path / "ref"))

    path = tmp_path / "run.safetensors"
    stem = tmp_path / "clip"
    crashed = PreviewFirstPipeline(fail_at=4)
    try:
        run_toy(crashed, checkpoint_path=path, preview_every=2, preview_stem=str(stem))
        raise AssertionError("expected a crash")
    except Interrupt:
        pass
    assert path.exists(), "no checkpoint survived the crash"
    # The crash happens mid-forward, after the step-4 preview (2 is also eligible and precedes it).
    assert preview_path(stem, 2).exists()
    assert preview_path(stem, 4).exists()

    resumed_pipeline = PreviewFirstPipeline()
    resumed = run_toy(resumed_pipeline, checkpoint_path=path, preview_every=2, preview_stem=str(stem))

    assert identical(reference, resumed), "resumed latents differ from the both-enabled reference run"
    assert resumed_pipeline.dit.calls == 4, \
        f"resume re-entered the transformer {resumed_pipeline.dit.calls} times instead of 4"
    assert not path.exists(), "the checkpoint outlived a successful run"

    # Only the forwards that actually ran for real (4, 5, 6, 7) may reach the preview wrapper --
    # the fast-forwarded 0..3 must not, even though step 2 is preview-eligible.
    assert resumed_pipeline.preview_interceptor.invocations == 4, (
        f"the preview wrapper was reached {resumed_pipeline.preview_interceptor.invocations} times "
        "during resume; a fast-forwarded step must never reach it"
    )
    for step in _eligible_preview_steps(2):
        assert preview_path(stem, step).exists(), f"missing preview for step {step} after resume"


# -- 3. install order: proxy-level proof, both orders ------------------------------------------------
#
# `LazyMiniMaxH3Pipeline` cannot be asked to install the two interceptors in the other order --
# the order is fixed by which `__call__` runs first, not by an argument -- so "test both orders"
# is done directly against `StepInterceptor` and `PreviewInterceptor`, the real classes, wired up
# by hand in each arrangement.

class _FakeRun:
    """The minimum `StepInterceptor` needs: `begin_forward` advances a counter, `should_skip`
    decides whether a given forward index is being fast-forwarded. Mirrors
    `h3_48gb.checkpoint.ResumableRun`'s contract without dragging in a real checkpoint file."""

    def __init__(self, start_step: int):
        self.start_step = start_step
        self.forward_index = -1

    def begin_forward(self, video_rows, audio_rows) -> int:
        self.forward_index += 1
        return self.forward_index

    def should_skip(self, index: int) -> bool:
        return index < self.start_step


class _FakePipeline:
    """Stands in for the pipeline `PreviewInterceptor` reads `_resume_run` off of."""

    def __init__(self, resume_run):
        self._resume_run = resume_run


def test_checkpoint_outside_preview_hides_fast_forwarded_steps_from_it():
    """The production order: preview installs first, checkpoint installs second and wraps it, so
    `StepInterceptor` is what the loop actually calls. A fast-forwarded step must reach neither the
    real transformer nor the preview wrapper.
    """
    real_dit_calls = []

    def real_dit(video_latents, audio_latents):
        real_dit_calls.append(True)
        return "video", "audio"

    fake_run = _FakeRun(start_step=3)
    preview = CountingPreviewInterceptor(
        real_dit, _FakePipeline(fake_run), every=1, stem="x", n_cond_v=0, num_latent_frames=7,
        latent_height=8, latent_width=8, patch_size=(1, 2, 2), verbose=False,
    )
    outer = StepInterceptor(preview, fake_run)  # checkpoint installs second -> wraps preview

    for _ in range(6):
        outer(mx.zeros((1, 4, 4)), mx.zeros((1, 2, 2)))

    assert len(real_dit_calls) == 3, "the real transformer ran on a fast-forwarded step"
    assert preview.invocations == 3, "a fast-forwarded step reached the preview wrapper"


def test_preview_outside_checkpoint_leaks_fast_forwarded_steps_into_it():
    """The reversed order: checkpoint installs first, preview installs second and wraps *it*. The
    real transformer still must never run on a fast-forwarded step (`StepInterceptor`'s own skip
    check is unaffected by what wraps it) -- but the preview wrapper, now on the outside, runs
    *before* that skip check, so it sees every step, including the ones being fast-forwarded. This
    is the concrete shape of the risk the checkpoint author's note warns about: reversing the order
    does not corrupt data, it silently defeats the point of a fast-forward by attempting a preview
    (VAE load, chunked decode, tens of seconds and several GB on real hardware) on steps that are
    supposed to cost nothing.
    """
    real_dit_calls = []

    def real_dit(video_latents, audio_latents):
        real_dit_calls.append(True)
        return "video", "audio"

    fake_run = _FakeRun(start_step=3)
    inner = StepInterceptor(real_dit, fake_run)  # checkpoint installs first
    outer = CountingPreviewInterceptor(  # preview installs second -> wraps checkpoint
        inner, _FakePipeline(fake_run), every=1, stem="x", n_cond_v=0, num_latent_frames=7,
        latent_height=8, latent_width=8, patch_size=(1, 2, 2), verbose=False,
    )

    for _ in range(6):
        outer(mx.zeros((1, 4, 4)), mx.zeros((1, 2, 2)))

    assert len(real_dit_calls) == 3, "the real transformer must never run on a fast-forwarded step"
    assert outer.invocations == 6, (
        f"expected every one of the 6 steps to reach the preview wrapper when it wraps the "
        f"checkpoint interceptor instead of being wrapped by it, got {outer.invocations}; if this "
        "assertion is what fails, the risk this test documents may have been fixed some other way"
    )


def test_both_nesting_orders_stay_attribute_transparent():
    """'Both are attribute-forwarding proxies so nesting works' -- checked directly: `.config`
    access must resolve through either order, since `LazyComponent`'s no-load contract and the
    pipeline's own early geometry reads depend on exactly this."""

    class FakeConfig:
        patch_size = (1, 2, 2)

    class FakeDit:
        config = FakeConfig()

        def __call__(self, *a, **k):
            return "ok"

    dit = FakeDit()
    fake_pipeline = _FakePipeline(_FakeRun(start_step=0))

    checkpoint_outside = StepInterceptor(
        PreviewInterceptor(dit, fake_pipeline, every=1, stem="x", n_cond_v=0, num_latent_frames=7,
                           latent_height=8, latent_width=8, patch_size=(1, 2, 2), verbose=False),
        _FakeRun(start_step=0),
    )
    preview_outside = PreviewInterceptor(
        StepInterceptor(dit, _FakeRun(start_step=0)), fake_pipeline, every=1, stem="x", n_cond_v=0,
        num_latent_frames=7, latent_height=8, latent_width=8, patch_size=(1, 2, 2), verbose=False,
    )

    assert checkpoint_outside.config.patch_size == (1, 2, 2)
    assert preview_outside.config.patch_size == (1, 2, 2)


# -- 4. independent failure -------------------------------------------------------------------------

def test_preview_failure_does_not_corrupt_or_prevent_checkpointing(tmp_path, monkeypatch):
    """Force every preview attempt -- the real decode *and* the VAE-less fallback -- to fail, and
    show the denoising/checkpointing side neither notices nor is corrupted: same bytes as an
    otherwise-identical run with preview simply turned off, a fully complete checkpoint, and no
    stray files.

    The reference run below still uses `PreviewFirstPipeline` (real tiny `VideoVAE`, `preview_every`
    left at 0) rather than plain `ToyPipeline`: swapping the video VAE also changes
    `spatial_compression_ratio`, which feeds the packed-sequence geometry for the *whole* denoising
    loop, not just preview -- comparing against `ToyPipeline`'s different-ratio stub would be
    comparing two different-shaped runs and would fail for a reason that has nothing to do with
    preview.
    """
    import h3_48gb.preview as preview_mod

    def always_raise(*args, **kwargs):
        raise RuntimeError("simulated total preview failure")

    monkeypatch.setattr(preview_mod, "render_preview_frame", always_raise)
    monkeypatch.setattr(preview_mod, "render_latent_fallback", always_raise)

    reference = run_toy(PreviewFirstPipeline())  # same geometry, preview simply off

    path = tmp_path / "run.safetensors"
    stem = tmp_path / "clip"
    pipeline = PreviewFirstPipeline()
    run_toy(pipeline, checkpoint_path=path, preview_every=1, preview_stem=str(stem),
            keep_checkpoint=True)

    assert identical(reference, pipeline.captured), \
        "a totally broken preview changed the denoised output"
    loaded = CheckpointStore(path).read()
    assert loaded.completed_steps == loaded.total_forwards == STEPS - 1, \
        "a broken preview left the checkpoint incomplete"
    assert not list(tmp_path.glob("clip-preview-step*.jpg")), \
        "a broken preview still left a JPEG behind"


def test_checkpoint_failure_does_not_erase_or_block_already_written_previews(tmp_path, monkeypatch):
    """A checkpoint write failure is meant to be loud (see `test_checkpoint.py`'s atomicity tests --
    it is never swallowed), but it must not reach backwards and undo a preview that already made it
    to disk earlier in the same step, and it must not be preview's problem to guard against."""
    real_write = CheckpointStore.write
    calls = {"n": 0}

    def flaky_write(self, arrays, meta):
        calls["n"] += 1
        if calls["n"] == 2:  # let the first checkpoint through; fail the second
            raise OSError("simulated disk failure")
        return real_write(self, arrays, meta)

    monkeypatch.setattr(CheckpointStore, "write", flaky_write)

    path = tmp_path / "run.safetensors"
    stem = tmp_path / "clip"
    pipeline = PreviewFirstPipeline()
    try:
        run_toy(pipeline, checkpoint_path=path, preview_every=1, preview_stem=str(stem))
        raise AssertionError("expected the injected checkpoint failure to propagate")
    except OSError:
        pass

    # Forward index 1's preview runs (inside the DiT call) strictly before the checkpoint write
    # that then fails, so it must already be on disk regardless of what happens to that write.
    assert preview_path(stem, 1).exists(), \
        "an unrelated checkpoint failure erased or blocked a preview already written"
    loaded = CheckpointStore(path).read()
    assert loaded.completed_steps == 1, "the one checkpoint write that succeeded was not the one kept"
