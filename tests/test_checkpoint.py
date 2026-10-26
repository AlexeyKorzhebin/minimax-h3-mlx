#!/usr/bin/env python3
"""Prove that an interrupted run resumes to a bit-identical result, and refuses when it must not.

    ./.venv/bin/pytest tests/test_checkpoint.py -q
    ./.venv/bin/python tests/test_checkpoint.py        # same tests, no pytest needed

The claim under test is exactness, not similarity: a run stopped after step k and resumed must
produce the *same bytes* as a run that was never stopped. "Looks the same" would hide an ulp of
drift per step, which is precisely the failure mode a resume can introduce — see
`ResumableScheduler` on why a zero velocity through the real Euler step is not the identity.

Nothing here loads the real model. A 33B transformer at 586 s per step cannot be part of a test
suite, so `_ToyDiT` stands in: a cheap, deterministic, *state-dependent* function of the packed
rows and the timestep. Everything else on the resume path is the real thing — upstream's
``MiniMaxH3Pipeline.__call__``, upstream's ``MiniMaxH3Scheduler``, upstream's packing, and the
fork's checkpoint writer, reader, proxies and identity check. The point of the stub is that the
transformer is the one component whose *value* the resume logic never inspects; substituting it
changes the numbers, not the control flow.

`test_tampered_checkpoint_changes_the_result` is the control: it perturbs one element of the saved
latents and shows the comparison notices. Without it, a test that compared two runs of an
insensitive stub would pass for the wrong reason.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.mlx
pytest.importorskip("mlx.core", reason="mlx: needs the MLX stack, absent here")


import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from h3_48gb import _upstream  # noqa: E402,F401

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from minimax_h3_mlx.config import DiTConfig, PipelineConfig  # noqa: E402
from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline  # noqa: E402
from minimax_h3_mlx.scheduler import MiniMaxH3Scheduler  # noqa: E402

from h3_48gb.checkpoint import (  # noqa: E402
    CheckpointCorrupt,
    CheckpointingPipeline,
    CheckpointMismatch,
    CheckpointStore,
    ResumableScheduler,
)

# Small enough that a whole 8-step run is milliseconds, large enough that the packed layout is real:
# 5 s at 24 fps snaps to 124 frames -> 37 latent frames, a 32x32 canvas -> 4x4 latents -> 2x2 patches.
CANVAS = 32
DURATION = 5.0
STEPS = 9  # grid points; the schedule drives STEPS - 1 = 8 transformer evaluations
SEED = 20250807

# A real keyframe, not a sentinel. `ToyPipeline._encode_keyframes` never looks at the pixels, so an
# `object()` used to do — but `_image_digest` does look at it, and for a bare object numpy yields a
# 0-d object array whose `tobytes()` is the pointer address. Run identity then varied with heap
# layout, so `test_resume_with_conditioning_rows` passed alone and failed as soon as another test
# module was collected first. A fixed ndarray digests to the same value in every process.
KEYFRAME = np.arange(CANVAS * CANVAS * 3, dtype=np.uint8).reshape(CANVAS, CANVAS, 3)


class Interrupt(RuntimeError):
    """Raised from the toy transformer to simulate a crash mid-run."""


# -- stand-ins -------------------------------------------------------------------------------------

class _Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _ToyDiT:
    """A deterministic velocity field with no learned weights.

    It has to depend on the incoming rows — a constant would make a broken resume indistinguishable
    from a correct one — and on the timestep, or every step would be the same map. Both inputs are
    read here, and the arithmetic is done in the bfloat16 the real transformer is handed, so the
    stub exercises the same dtype boundary the real one does.
    """

    def __init__(self, config: DiTConfig, fail_at: int | None = None):
        self.config = config
        self.fail_at = fail_at
        self.calls = 0
        #: The MLX RNG key seen at the top of every step, for `test_denoising_loop_draws_no_randomness`.
        self.rng_keys: list[np.ndarray] = []

    def __call__(self, video_latents, audio_latents, text_embeds, timestep, timestep_indices,
                 token_tags, position_ids, video_indices, audio_indices, text_indices,
                 modulation_cache=None, mask=None):
        if self.fail_at is not None and self.calls == self.fail_at:
            raise Interrupt(f"simulated crash at forward {self.calls}")
        self.calls += 1
        self.rng_keys.append(np.concatenate([np.array(k).reshape(-1) for k in mx.random.state]))

        def bf16(value):
            return mx.array(value, dtype=mx.bfloat16)

        level = mx.mean(timestep[timestep_indices[video_indices]]).astype(mx.bfloat16)
        text = mx.mean(text_embeds.astype(mx.bfloat16))
        video = mx.tanh(video_latents * bf16(1.37) + level) * bf16(0.5) + text
        audio = mx.tanh(audio_latents * bf16(0.91) - level) * bf16(0.5) + text
        return video, audio


class _ToyTextEncoder:
    """Deterministic prompt conditioning, so a replayed encode is checkable against a live one."""

    def __init__(self, text_dim: int, tokens: int = 12):
        self.text_dim = text_dim
        self.tokens = tokens
        self.calls = 0

    def encode(self, prompt: str, images=None):
        self.calls += 1
        # A digest, not ``hash()``: Python randomizes string hashing per process, which would make
        # this fixture differ between runs and quietly weaken every comparison in the file.
        seed = int.from_bytes(hashlib.blake2b(prompt.encode(), digest_size=4).digest(), "little")
        rng = np.random.default_rng(seed)
        embeds = mx.array(rng.standard_normal((1, self.tokens, self.text_dim), dtype=np.float32))
        tags = np.ones(self.tokens, dtype=np.int64)  # every row is text; no keyframe vision block
        return embeds.astype(mx.bfloat16), tags


class ToyPipeline(CheckpointingPipeline, MiniMaxH3Pipeline):
    """Upstream's ``__call__`` and scheduler, driven by stand-ins for the four heavy components.

    ``_ensure_cache`` is disabled because the modulation table is a property of the real weights and
    has nothing to do with resumption; the decoders are replaced by capture hooks so the test can
    compare *latents* rather than pixels, which is where bit-exactness has to hold.
    """

    def __init__(self, fail_at: int | None = None, keyframe_rows: int = 0):
        dit_config = DiTConfig(patch_size=(1, 2, 2), latents_dim=4, audio_latents_dim=6, text_dim=8)
        video_vae = _Config(config=_Config(latent_channels=4, spatial_compression_ratio=8,
                                           latents_mean=(0.0,) * 4, latents_std=(1.0,) * 4))
        audio_vae = _Config(config=_Config(latent_channels=6, sampling_rate=32000,
                                           latents_mean=(0.0,) * 6, latents_std=(1.0,) * 6))
        super().__init__(
            _ToyDiT(dit_config, fail_at=fail_at),
            _ToyTextEncoder(dit_config.text_dim),
            video_vae,
            audio_vae,
            PipelineConfig(),
        )
        self.keyframe_rows = keyframe_rows
        self.captured: dict[str, np.ndarray] = {}

    def checkpoint_identity_extra(self) -> dict:
        return {"weights": "toy-v1"}

    def _ensure_cache(self, timesteps, drop_adaln, verbose):
        return None

    def _encode_keyframes(self, images, height, width):
        """Deterministic conditioning rows, standing in for the seeded video-VAE posterior draw."""
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


def run(pipeline: ToyPipeline, checkpoint_path=None, **overrides) -> dict[str, np.ndarray]:
    """One generation with the fixed geometry the whole file shares."""
    kwargs = dict(
        prompt="a jeweled hummingbird beside a red orchid",
        duration_seconds=DURATION,
        num_inference_steps=STEPS,
        seed=SEED,
        height=CANVAS,
        width=CANVAS,
        verbose=False,
    )
    kwargs.update(overrides)
    if checkpoint_path is not None:
        kwargs["checkpoint_path"] = str(checkpoint_path)
    pipeline(**kwargs)
    return pipeline.captured


def identical(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> bool:
    """Bit-for-bit, via the raw buffers — ``==`` on float arrays would forgive a NaN or a -0.0."""
    if set(a) != set(b):
        return False
    return all(
        a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and a[k].tobytes() == b[k].tobytes()
        for k in a
    )


# -- the determinism argument ------------------------------------------------------------------

def test_sampler_has_no_state_beyond_the_rows():
    """The Euler step is a pure function of ``(model_output, timestep, sample)`` plus a step index.

    This is the premise the whole feature rests on, so it is asserted rather than asserted about:
    the same inputs at the same index give the same bytes, and no ``mx.random`` draw happens.
    """
    scheduler = MiniMaxH3Scheduler(shift=12.0)
    scheduler.set_timesteps(9)
    rng = np.random.default_rng(0)
    sample = mx.array(rng.standard_normal((16, 4), dtype=np.float32))
    velocity = mx.array(rng.standard_normal((16, 4), dtype=np.float32))
    timestep = float(scheduler.timesteps[3].item())

    before = [np.array(k) for k in mx.random.state]
    first = MiniMaxH3Scheduler(shift=12.0)
    first.set_timesteps(9)
    first._step_index = 3
    a = np.array(first.step(velocity, timestep, sample))

    second = MiniMaxH3Scheduler(shift=12.0)
    second.set_timesteps(9)
    second._step_index = 3
    b = np.array(second.step(velocity, timestep, sample))
    after = [np.array(k) for k in mx.random.state]

    assert a.tobytes() == b.tobytes(), "the scheduler step is not reproducible"
    assert all(np.array_equal(x, y) for x, y in zip(before, after)), \
        "the scheduler step consumed the MLX RNG"


def test_denoising_loop_draws_no_randomness():
    """The MLX RNG key is identical at the top of every step, so the loop consumes no randomness.

    This is the second half of the determinism argument: all sampling in ``__call__`` happens
    *before* the loop (``mx.random.seed(KEYFRAME_ENCODE_SEED)`` for the keyframe posterior,
    ``mx.random.seed(seed)`` for the three noise tensors), and nothing after it draws. Watching the
    key directly beats reading the source, because it also covers whatever the transformer and the
    scheduler do underneath.

    If this ever fails, the checkpoint's ``rng_state`` stops being a formality —
    `ResumableRun._restore_rng` already restores it and `rng_note` would say so out loud.
    """
    pipeline = ToyPipeline()
    run(pipeline)
    keys = pipeline.dit.rng_keys
    assert len(keys) == STEPS - 1, "the transformer was not called once per step"
    assert all(np.array_equal(keys[0], k) for k in keys), \
        "the MLX RNG key moved during the denoising loop — the loop is stochastic"


def test_uninterrupted_run_is_reproducible():
    """Two full runs of the same request agree byte for byte. Without this, nothing else means much."""
    assert identical(run(ToyPipeline()), run(ToyPipeline()))


def test_resume_reproduces_an_uninterrupted_run(tmp_path):
    """The headline: crash at step 4 of 8, resume, and land on the same bytes."""
    reference = run(ToyPipeline())

    path = tmp_path / "run.safetensors"
    crashed = ToyPipeline(fail_at=4)
    try:
        run(crashed, checkpoint_path=path)
        raise AssertionError("the toy transformer was supposed to crash")
    except Interrupt:
        pass
    assert path.exists(), "no checkpoint survived the crash"

    resumed_pipeline = ToyPipeline()
    resumed = run(resumed_pipeline, checkpoint_path=path)

    assert identical(reference, resumed), "resumed latents differ from the uninterrupted run"
    assert resumed_pipeline.dit.calls == 4, \
        f"resume re-ran {resumed_pipeline.dit.calls} forwards instead of the 4 that were left"
    assert not path.exists(), "the checkpoint outlived a successful run"


def test_resume_at_every_step_boundary(tmp_path):
    """Not just one seam. Every ``k`` from 1 to the last step has to reconstruct the same run.

    Off-by-one is the natural bug here — the checkpoint records *completed* steps, the scheduler
    counts the step it is about to take — and only sweeping the boundary catches it.
    """
    reference = run(ToyPipeline())
    for k in range(1, STEPS - 1):
        path = tmp_path / f"seam-{k}.safetensors"
        try:
            run(ToyPipeline(fail_at=k), checkpoint_path=path)
            raise AssertionError("expected a crash")
        except Interrupt:
            pass
        resumed_pipeline = ToyPipeline()
        assert identical(reference, run(resumed_pipeline, checkpoint_path=path)), \
            f"resuming after {k} completed steps diverged"
        assert resumed_pipeline.dit.calls == (STEPS - 1) - k


def test_resume_after_the_last_step_skips_the_transformer_entirely(tmp_path):
    """A crash in the VAE decode must not cost the diffusion.

    The final checkpoint is written after the last step, before decoding — which on a 48 GB machine
    is its own memory cliff. Resuming from it runs zero forwards.
    """
    path = tmp_path / "post-loop.safetensors"

    class DecodeCrash(ToyPipeline):
        def _decode_video(self, rows, *args, **kwargs):
            raise Interrupt("simulated OOM in the video VAE")

    try:
        run(DecodeCrash(), checkpoint_path=path)
        raise AssertionError("expected a decode crash")
    except Interrupt:
        pass

    resumed_pipeline = ToyPipeline()
    assert identical(run(ToyPipeline()), run(resumed_pipeline, checkpoint_path=path))
    assert resumed_pipeline.dit.calls == 0, "the transformer ran on a fully-completed checkpoint"


def test_resume_with_conditioning_rows(tmp_path):
    """The keyframe path: conditioning rows are rebuilt, generated rows come from the checkpoint."""
    reference = run(ToyPipeline(keyframe_rows=4), images=[KEYFRAME], keyframe_anchors=("first",))

    path = tmp_path / "fl2va.safetensors"
    try:
        run(ToyPipeline(fail_at=3, keyframe_rows=4), checkpoint_path=path,
            images=[KEYFRAME], keyframe_anchors=("first",))
        raise AssertionError("expected a crash")
    except Interrupt:
        pass

    resumed = run(ToyPipeline(keyframe_rows=4), checkpoint_path=path,
                  images=[KEYFRAME], keyframe_anchors=("first",))
    assert identical(reference, resumed)


def test_conditioning_rows_that_do_not_reproduce_are_caught(tmp_path):
    """The one thing a resume rebuilds rather than restores has to be checked, not trusted.

    Generated rows come out of the file, but keyframe conditioning rows are re-derived from the
    images and the seed, which assumes the video VAE encode is reproducible. That assumption is
    verified at the seam instead of being taken on faith: if the rebuilt rows differ from the ones
    the checkpoint recorded, the run stops rather than denoising against the wrong anchor.
    """
    path = tmp_path / "cond.safetensors"
    try:
        run(ToyPipeline(fail_at=3, keyframe_rows=4), checkpoint_path=path,
            images=[KEYFRAME], keyframe_anchors=("first",))
        raise AssertionError("expected a crash")
    except Interrupt:
        pass

    class DriftingKeyframes(ToyPipeline):
        def _encode_keyframes(self, images, height, width):
            return super()._encode_keyframes(images, height, width) + 0.25

    try:
        run(DriftingKeyframes(keyframe_rows=4), checkpoint_path=path,
            images=[KEYFRAME], keyframe_anchors=("first",))
        raise AssertionError("conditioning rows that changed were accepted")
    except CheckpointMismatch as exc:
        assert "conditioning rows" in str(exc)


def test_tampered_checkpoint_changes_the_result(tmp_path):
    """The control. Nudge a single element of the saved latents and the comparison must notice.

    A determinism test whose comparison cannot fail proves nothing; this is what gives the others
    their teeth. The nudge is small but real rather than one ulp on purpose: the transformer is
    handed its rows in bfloat16, so a float32 ulp is genuinely rounded away before it can propagate
    — a control built on it would be testing float32 rounding, not the resume path.
    """
    reference = run(ToyPipeline())
    path = tmp_path / "tampered.safetensors"
    try:
        run(ToyPipeline(fail_at=4), checkpoint_path=path)
        raise AssertionError("expected a crash")
    except Interrupt:
        pass

    store = CheckpointStore(path)
    loaded = store.read()
    arrays = dict(loaded.arrays)
    poked = np.array(arrays["video_gen"])
    poked[0, 0] += np.float32(1e-2)
    arrays["video_gen"] = mx.array(poked)
    store.write(arrays, loaded.meta)

    tampered = run(ToyPipeline(), checkpoint_path=path)
    assert not identical(reference, tampered), \
        "a perturbed checkpoint produced identical output — the comparison is not sensitive"
    # `_ToyDiT` is elementwise, so exactly one element moves and it moves by something other than
    # the injected amount: the perturbation went through four more Euler steps to get here.
    differing = np.flatnonzero(tampered["video"] != reference["video"])
    assert differing.tolist() == [0]
    assert tampered["video"].ravel()[0] != reference["video"].ravel()[0] + np.float32(1e-2)


# -- refusals --------------------------------------------------------------------------------------

def test_foreign_parameters_are_refused(tmp_path):
    """A checkpoint from another request is an error, never a silent continuation."""
    path = tmp_path / "shared.safetensors"
    try:
        run(ToyPipeline(fail_at=4), checkpoint_path=path)
        raise AssertionError("expected a crash")
    except Interrupt:
        pass

    for label, override in (
        ("prompt", {"prompt": "a different clip entirely"}),
        ("seed", {"seed": SEED + 1}),
        ("steps", {"num_inference_steps": STEPS + 2}),
        ("geometry", {"height": CANVAS * 2, "width": CANVAS * 2}),
        ("duration", {"duration_seconds": DURATION + 1.0}),
    ):
        try:
            run(ToyPipeline(), checkpoint_path=path, **override)
            raise AssertionError(f"a checkpoint with a different {label} was accepted")
        except CheckpointMismatch as exc:
            assert label.split()[0] in str(exc) or "different run" in str(exc)


def test_different_weights_are_refused(tmp_path):
    """Same request, different model. The latents are not interchangeable and must not be reused."""
    path = tmp_path / "weights.safetensors"
    try:
        run(ToyPipeline(fail_at=4), checkpoint_path=path)
        raise AssertionError("expected a crash")
    except Interrupt:
        pass

    class Rebuilt(ToyPipeline):
        def checkpoint_identity_extra(self) -> dict:
            return {"weights": "toy-v2"}

    try:
        run(Rebuilt(), checkpoint_path=path)
        raise AssertionError("a checkpoint from different weights was accepted")
    except CheckpointMismatch as exc:
        assert "toy-v1" in str(exc) and "toy-v2" in str(exc)


def test_hashed_names_keep_unrelated_runs_apart(tmp_path):
    """With ``checkpoint_dir`` the file is named after the run, so two requests never collide."""
    for seed in (1, 2):
        try:
            run(ToyPipeline(fail_at=4), checkpoint_dir=str(tmp_path), seed=seed)
            raise AssertionError("expected a crash")
        except Interrupt:
            pass
    files = sorted(p.name for p in tmp_path.glob("h3-*.safetensors"))
    assert len(files) == 2, f"expected one checkpoint per request, got {files}"


def test_tag_keeps_otherwise_identical_runs_apart(tmp_path):
    """`tag` never reached `request_identity` (upstream's `__call__` has no `tag` parameter, so it
    never landed in `bound.arguments`) until `CHECKPOINT_KWARGS`/`request_identity` grew an explicit
    `tag` seam for it. Same prompt/geometry/duration/steps/seed as `test_hashed_names_keep_
    unrelated_runs_apart`, differing only by `tag` this time — must still land in two files, not
    one shared, silently-overwritten one.
    """
    for tag in ("a", "b"):
        try:
            run(ToyPipeline(fail_at=4), checkpoint_dir=str(tmp_path), tag=tag)
            raise AssertionError("expected a crash")
        except Interrupt:
            pass
    files = sorted(p.name for p in tmp_path.glob("h3-*.safetensors"))
    assert len(files) == 2, f"expected one checkpoint per tag, got {files}"


def test_a_keyframe_numpy_cannot_describe_is_refused():
    """An un-digestible keyframe must raise, not hash its own pointer address.

    ``np.asarray(object())`` is a 0-d *object* array; its ``tobytes()`` is the address of the boxed
    Python object, which differs between processes and repeats whenever the allocator reuses an
    address. Hashing it produced a run identity that depended on heap layout — checkpoints that
    refused to resume the run that wrote them, and, in principle, two different images colliding on
    one file. Both are silent. The refusal is the fix.
    """
    from h3_48gb.checkpoint import _image_digest

    try:
        _image_digest(object())
        raise AssertionError("an object-dtype keyframe was digested instead of refused")
    except TypeError as exc:
        assert "object array" in str(exc)

    # The supported forms still work, and digest by content: same pixels -> same digest, one
    # changed pixel -> a different one.
    other = KEYFRAME.copy()
    other[0, 0, 0] ^= 1
    assert _image_digest(KEYFRAME) == _image_digest(KEYFRAME.copy())
    assert _image_digest(KEYFRAME) != _image_digest(other)


def test_corrupt_checkpoint_is_quarantined_and_the_run_restarts(tmp_path):
    """Truncated data has nothing to resume from, so it is moved aside and the run starts over.

    Loudly: the file is kept under a ``.corrupt-*`` name rather than deleted, because a checkpoint
    that will not load is evidence about something, and fifteen hours of compute is not the moment
    to discover it was thrown away.
    """
    path = tmp_path / "broken.safetensors"
    try:
        run(ToyPipeline(fail_at=4), checkpoint_path=path)
        raise AssertionError("expected a crash")
    except Interrupt:
        pass

    raw = path.read_bytes()
    path.write_bytes(raw[: len(raw) - 64])

    fresh = ToyPipeline()
    assert identical(run(ToyPipeline()), run(fresh, checkpoint_path=path))
    assert fresh.dit.calls == STEPS - 1, "a corrupt checkpoint was partially believed"
    assert list(tmp_path.glob("broken.safetensors.corrupt-*")), "the corrupt file was not kept"


def test_corrupt_checkpoint_can_be_made_fatal(tmp_path):
    path = tmp_path / "strict.safetensors"
    try:
        run(ToyPipeline(fail_at=4), checkpoint_path=path)
        raise AssertionError("expected a crash")
    except Interrupt:
        pass
    path.write_bytes(b"this is not a safetensors file")

    try:
        run(ToyPipeline(), checkpoint_path=path, on_corrupt="raise")
        raise AssertionError("on_corrupt='raise' did not raise")
    except CheckpointCorrupt:
        pass


def test_a_foreign_safetensors_file_is_not_mistaken_for_a_checkpoint(tmp_path):
    path = tmp_path / "someone-elses.safetensors"
    mx.save_safetensors(str(path), {"weight": mx.zeros((4, 4))})
    try:
        run(ToyPipeline(), checkpoint_path=path, on_corrupt="raise")
        raise AssertionError("a foreign safetensors file was accepted as a checkpoint")
    except CheckpointCorrupt as exc:
        assert "not a MiniMax-H3 checkpoint" in str(exc)


# -- atomicity --------------------------------------------------------------------------------------

def test_a_write_that_dies_midway_leaves_the_previous_checkpoint_intact(tmp_path):
    """The reason for the temp-file-and-rename: a half-written file must never become the checkpoint.

    The failure is injected inside ``mx.save_safetensors``, i.e. after the temporary file exists and
    before the rename — the exact window a power cut would hit.
    """
    path = tmp_path / "atomic.safetensors"
    try:
        run(ToyPipeline(fail_at=4), checkpoint_path=path)
        raise AssertionError("expected a crash")
    except Interrupt:
        pass
    good = path.read_bytes()
    good_meta = CheckpointStore(path).read().meta["completed_steps"]

    store = CheckpointStore(path)
    real_save = mx.save_safetensors

    def half_written(file, arrays, metadata=None):
        real_save(file, arrays, metadata)
        with open(str(file), "r+b") as fh:  # `file` already carries the .safetensors suffix
            fh.truncate(32)
        raise Interrupt("power cut mid-write")

    mx.save_safetensors = half_written
    try:
        store.write({"video_gen": mx.zeros((2, 2)), "audio_gen": mx.zeros((2, 2))}, {"format": 1})
        raise AssertionError("the injected failure did not propagate")
    except Interrupt:
        pass
    finally:
        mx.save_safetensors = real_save

    assert path.read_bytes() == good, "a failed write damaged the previous checkpoint"
    assert CheckpointStore(path).read().meta["completed_steps"] == good_meta
    assert not list(tmp_path.glob(".*.tmp-*.safetensors")), "the temporary file was left behind"


def test_a_leftover_temporary_file_is_never_read_and_is_cleaned_up(tmp_path):
    """A temp file from a killed process must not be picked up, and must not survive success."""
    path = tmp_path / "leftover.safetensors"
    try:
        run(ToyPipeline(fail_at=4), checkpoint_path=path)
        raise AssertionError("expected a crash")
    except Interrupt:
        pass
    stray = tmp_path / f".{path.stem}.tmp-999999.safetensors"
    stray.write_bytes(b"garbage from a process that was killed")

    resumed = ToyPipeline()
    assert identical(run(ToyPipeline()), run(resumed, checkpoint_path=path))
    assert resumed.dit.calls == 4, "the stray temp file disturbed the resume"
    assert not stray.exists(), "the stray temp file survived a successful run"


# -- housekeeping -----------------------------------------------------------------------------------

def test_checkpoint_is_kept_on_request(tmp_path):
    path = tmp_path / "kept.safetensors"
    run(ToyPipeline(), checkpoint_path=path, keep_checkpoint=True)
    assert path.exists()
    loaded = CheckpointStore(path).read()
    assert loaded.completed_steps == loaded.total_forwards == STEPS - 1


def test_resume_false_starts_over(tmp_path):
    path = tmp_path / "ignored.safetensors"
    try:
        run(ToyPipeline(fail_at=4), checkpoint_path=path)
        raise AssertionError("expected a crash")
    except Interrupt:
        pass
    fresh = ToyPipeline()
    assert identical(run(ToyPipeline()), run(fresh, checkpoint_path=path, resume=False))
    assert fresh.dit.calls == STEPS - 1


def test_prompt_conditioning_is_replayed_not_recomputed(tmp_path):
    """The 28.2 GB encoder is the single most expensive thing to redo, so the checkpoint carries it."""
    path = tmp_path / "prompt.safetensors"
    try:
        run(ToyPipeline(fail_at=4), checkpoint_path=path)
        raise AssertionError("expected a crash")
    except Interrupt:
        pass
    assert CheckpointStore(path).read().meta["has_prompt_embeds"]

    resumed = ToyPipeline()
    run(resumed, checkpoint_path=path)
    assert resumed.text_encoder.calls == 0, "the text encoder ran on a resume"


def test_prompt_replay_can_be_turned_off(tmp_path):
    path = tmp_path / "no-prompt.safetensors"
    try:
        run(ToyPipeline(fail_at=4), checkpoint_path=path, checkpoint_prompt_embeds=False)
        raise AssertionError("expected a crash")
    except Interrupt:
        pass
    assert not CheckpointStore(path).read().meta["has_prompt_embeds"]

    resumed = ToyPipeline()
    assert identical(run(ToyPipeline()), run(resumed, checkpoint_path=path))
    assert resumed.text_encoder.calls == 1


def test_without_a_checkpoint_argument_nothing_is_written(tmp_path):
    """The feature is opt-in and leaves no trace when it is off."""
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        run(ToyPipeline())
    finally:
        os.chdir(cwd)
    assert not list(tmp_path.iterdir())


def test_the_scheduler_stays_a_real_scheduler():
    """`ResumableScheduler` is installed by class rebinding, so nothing downstream can tell.

    `LazyMiniMaxH3Pipeline._ensure_cache` passes the schedulers straight to
    ``AdaLNCacheFile.build``; a wrapper object would have broken that quietly.
    """
    scheduler = MiniMaxH3Scheduler(shift=12.0)
    scheduler.set_timesteps(9)
    sigmas = np.array(scheduler.sigmas)
    scheduler.__class__ = ResumableScheduler
    assert isinstance(scheduler, MiniMaxH3Scheduler)
    assert np.array_equal(np.array(scheduler.sigmas), sigmas)
    assert scheduler.shift == 12.0


def _read_checkpoint_meta(path: Path) -> dict:
    """Parse the safetensors JSON header by hand, the way `runs.scan` and an operator's `head` do."""
    with open(path, "rb") as fh:
        length = int.from_bytes(fh.read(8), "little")
        header = json.loads(fh.read(length))
    return json.loads(header["__metadata__"]["h3_checkpoint"])


def test_metadata_is_readable_without_mlx(tmp_path):
    """The identity lives in the safetensors JSON header, so an operator can inspect it with `head`."""
    path = tmp_path / "meta.safetensors"
    run(ToyPipeline(), checkpoint_path=path, keep_checkpoint=True)
    meta = _read_checkpoint_meta(path)
    assert meta["identity"]["request"]["seed"] == SEED
    assert meta["completed_steps"] == STEPS - 1
    # `h3_48gb.runs.scan` reads a live run's speed from these two fields without importing MLX —
    # `test_a_resumed_run_rates_only_this_session` in test_runs.py depends on both being present
    # and correct on a real, written-by-the-pipeline checkpoint, not just on a hand-built dict.
    assert meta["started_at"] is not None
    assert meta["completed_at_start"] == 0


def test_resumed_checkpoint_records_previous_session_progress(tmp_path):
    """A resume must record how many forwards the *previous* session already paid for.

    `runs.seconds_per_forward` divides this session's elapsed wall time by
    `completed_steps - completed_at_start` — the forwards *this* session actually did. If a
    resumed run wrote `completed_at_start=0` instead of the forward count at load time, a run
    resumed at step 4 of 8 that then does one more forward would look like it did all 5 in this
    session, reporting a rate and an ETA several times too optimistic.
    """
    path = tmp_path / "resume.safetensors"
    try:
        run(ToyPipeline(fail_at=4), checkpoint_path=path)
        raise AssertionError("expected a crash")
    except Interrupt:
        pass

    before_resume = _read_checkpoint_meta(path)
    assert before_resume["completed_steps"] == 4
    assert before_resume["completed_at_start"] == 0, "the first session started at forward 0"

    run(ToyPipeline(), checkpoint_path=path, keep_checkpoint=True)
    after_resume = _read_checkpoint_meta(path)
    assert after_resume["completed_steps"] == STEPS - 1
    assert after_resume["completed_at_start"] == 4, (
        "a resumed session must record the forwards already done when it began, not zero")


def test_started_at_is_constant_across_every_write_in_one_session(tmp_path, monkeypatch):
    """`started_at` marks when *this session* began, once — not the moment of each write.

    `test_metadata_is_readable_without_mlx` only checks `started_at is not None`, which a mutation
    that overwrites `started_at` with `written_at` (or blanks it to `""`) on every write would
    still pass. This pins the actual value across several writes in one uninterrupted session,
    using a fake clock so the assertion does not depend on every write happening to land in the
    same wall-clock second (STEPS - 1 = 8 writes is comfortably faster than that on a real machine,
    which would otherwise mask exactly the "written_at every time" mutation this test exists to
    catch).
    """
    import h3_48gb.checkpoint as checkpoint_mod

    clock = iter(f"2026-08-10T21:00:{i:02d}" for i in range(60))
    monkeypatch.setattr(checkpoint_mod.time, "strftime", lambda _fmt: next(clock))

    metas: list[dict] = []
    original_write = checkpoint_mod.CheckpointStore.write

    def spy_write(self, arrays, meta):
        metas.append(dict(meta))
        original_write(self, arrays, meta)

    monkeypatch.setattr(checkpoint_mod.CheckpointStore, "write", spy_write)

    path = tmp_path / "session.safetensors"
    run(ToyPipeline(), checkpoint_path=path, keep_checkpoint=True)  # one full, uncrashed session

    assert len(metas) == STEPS - 1, "expected one checkpoint write per forward"
    # Call 0 of the fake clock is spent on `ResumableRun.started_at`'s default factory, at
    # construction, before any write; calls 1..8 are `written_at` on each of the eight writes.
    assert metas[0]["started_at"] == "2026-08-10T21:00:00"
    assert {m["started_at"] for m in metas} == {"2026-08-10T21:00:00"}, (
        "started_at must be the same value on every write in a session, not re-stamped per write")
    assert [m["written_at"] for m in metas] == [
        f"2026-08-10T21:00:{i:02d}" for i in range(1, len(metas) + 1)
    ], "written_at must advance on every write, or the fixture cannot tell it apart from started_at"


def test_written_at_is_stamped_after_the_step_is_materialized(tmp_path, monkeypatch):
    """The clock must be read after the arrays are real, not while they are still a graph.

    MLX is lazy. Until something forces it, `arrays` here is an unevaluated graph over the forward
    pass that just "finished", and `mx.save_safetensors` is what forces it — so without an explicit
    `mx.eval` the entire step executes *inside* the write, and a `written_at` taken beforehand
    records when the step began.

    Measured on the 15 s run of 2026-08-11, before the fix: the checkpoint file's mtime was
    12:29:52 and the timestamp inside it read 12:11:46 — eighteen minutes early, one whole step.
    `h3 status` divides `written_at - started_at` by the step count, so it announced "19 s per
    forward, 2 minutes left" for a run doing 1086 s per forward with an hour and three quarters
    left. Nothing raised; the number was merely wrong, which is the failure mode this whole
    subcommand exists to avoid.

    `test_started_at_is_constant_across_every_write_in_one_session` cannot catch this: it pins the
    values a fake clock hands out, and a fake clock advances the same way whether it is read before
    or after materialization. The ordering is the invariant, so the ordering is what is asserted.

    Who calls `mx.eval` is part of the assertion, and deliberately so. The first version of this
    test only required *some* eval immediately before each stamp, and passed with the fix reverted:
    the pipeline's own per-step `mx.eval` in `__call__` lands there, one iteration late, and
    satisfied the pattern without materializing anything the pending write depends on. The
    invariant is not "an eval happened nearby" but "`_write` evaluated its own arrays before
    reading the clock", so the frame that made the call is what is checked.
    """
    import sys

    import h3_48gb.checkpoint as checkpoint_mod

    order: list[tuple[str, str]] = []
    real_eval = checkpoint_mod.mx.eval

    def spy_eval(*args, **kwargs):
        order.append(("eval", sys._getframe(1).f_code.co_name))
        return real_eval(*args, **kwargs)

    clock = iter(f"2026-08-10T21:00:{i:02d}" for i in range(60))

    def spy_strftime(_fmt):
        order.append(("stamp", sys._getframe(1).f_code.co_name))
        return next(clock)

    monkeypatch.setattr(checkpoint_mod.mx, "eval", spy_eval)
    monkeypatch.setattr(checkpoint_mod.time, "strftime", spy_strftime)

    run(ToyPipeline(), checkpoint_path=tmp_path / "session.safetensors", keep_checkpoint=True)

    write_stamps = [i for i, event in enumerate(order) if event == ("stamp", "_write")]
    assert len(write_stamps) == STEPS - 1, (
        f"expected one written_at per forward, got {len(write_stamps)}")
    for i in write_stamps:
        assert order[i - 1] == ("eval", "_write"), (
            "written_at was stamped while the step was still an unevaluated graph — the timestamp "
            f"will name the moment the step started, not the moment the checkpoint landed; the "
            f"call before it was {order[i - 1]}")


def test_writer_holds_an_exclusive_lock_for_the_life_of_the_run(tmp_path):
    """The liveness signal `h3_48gb.runs._writer_alive` reads is a real, held `flock` on a
    companion file next to the checkpoint — proved here with the reader's own probe technique
    (a non-blocking shared `flock` attempt), not a mocked stand-in, while a real run is mid-flight.

    The companion file must survive the run ending, unlocked but present -- see
    `CheckpointStore.release_lock`'s docstring for why deleting it here would be exactly the race
    task 3's fix exists to close: a reader (or the next writer for the same identity) that has the
    path open at the moment of an unlink can end up holding a lock on an inode nobody can reach by
    that name any more.
    """
    import fcntl
    from h3_48gb.checkpoint import CheckpointStore

    path = tmp_path / "live.safetensors"
    store = CheckpointStore(path)
    observed: list[bool] = []

    class ObserveMidRun(ToyPipeline):
        def _decode_video(self, rows, *args, **kwargs):
            assert store.lock_path.exists(), "no companion lock file while the run is in progress"
            probe = open(store.lock_path, "rb")
            try:
                try:
                    fcntl.flock(probe.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                    observed.append(True)  # acquired -- would mean no writer holds it
                    fcntl.flock(probe.fileno(), fcntl.LOCK_UN)
                except OSError:
                    observed.append(False)  # blocked -- the writer's own exclusive lock is held
            finally:
                probe.close()
            return super()._decode_video(rows, *args, **kwargs)

    run(ObserveMidRun(), checkpoint_path=path, keep_checkpoint=True)

    assert observed == [False], "a non-blocking shared probe must fail while the run is live"
    assert store.lock_path.exists(), "the companion lock file must survive the run ending"
    # And it must actually be unlocked now -- proof `release_lock` dropped the flock, not just
    # that the file happens to still be there.
    probe = open(store.lock_path, "rb")
    try:
        fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(probe.fileno(), fcntl.LOCK_UN)
    except OSError:
        raise AssertionError("the lock must be released once the run ends, even though the file "
                              "stays")
    finally:
        probe.close()


def test_lock_is_released_but_kept_after_a_crash_python_catches(tmp_path):
    """A crash that unwinds through Python (unlike a hard process kill) still runs
    `CheckpointStore.release_lock` from the `finally` wrapping the whole call, so a graceful
    failure drops its own flock exactly like a graceful success does. The checkpoint itself is
    untouched — it is still there for the next resume — and so, deliberately, is the lock file:
    see `release_lock`'s docstring for why removing it would reopen the very race it exists to
    close.
    """
    import fcntl
    from h3_48gb.checkpoint import CheckpointStore

    path = tmp_path / "crash.safetensors"
    store = CheckpointStore(path)
    try:
        run(ToyPipeline(fail_at=3), checkpoint_path=path)
        raise AssertionError("expected a crash")
    except Interrupt:
        pass

    assert path.exists(), "the checkpoint should survive a crash, for the next resume"
    assert store.lock_path.exists(), "a Python-level crash must not delete the lock file"
    probe = open(store.lock_path, "rb")
    try:
        fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(probe.fileno(), fcntl.LOCK_UN)
    except OSError:
        raise AssertionError("a Python-level crash must still release the flock")
    finally:
        probe.close()


def test_a_second_writer_is_refused_the_lock(tmp_path):
    """Two writers on the same checkpoint must not both proceed -- a shared lock (the bug this
    fixes) lets both hold it at once; the fix is an exclusive lock, and the second `acquire_lock`
    must fail loudly instead of silently succeeding beside the first.
    """
    from h3_48gb.checkpoint import CheckpointError, CheckpointLocked, CheckpointStore

    path = tmp_path / "race.safetensors"
    first = CheckpointStore(path)
    first.acquire_lock()
    try:
        second = CheckpointStore(path)
        try:
            second.acquire_lock()
        except CheckpointLocked:
            pass
        except CheckpointError as exc:
            raise AssertionError(f"expected CheckpointLocked specifically, got {type(exc)}: {exc}")
        else:
            raise AssertionError(
                "a second writer must not acquire the lock while the first still holds it")
    finally:
        first.release_lock()

    # And once the first writer releases, a second writer must be able to acquire it -- this is
    # not a permanent refusal, only a concurrent one.
    third = CheckpointStore(path)
    third.acquire_lock()
    third.release_lock()


def test_the_lock_is_held_before_the_checkpoint_is_read_for_resume(tmp_path, monkeypatch):
    """The lock must be acquired before `_load_for_resume` reads anything off disk -- taking it
    any later would let two processes both read the same `completed_steps` and then race each
    other to `os.replace`, walking the checkpoint's progress backwards. Proved here by having
    `_load_for_resume` itself probe the lock with an independent file handle: if the real writer's
    own lock is not already held by the time this runs, the probe would succeed.
    """
    import fcntl

    import h3_48gb.checkpoint as checkpoint_mod

    path = tmp_path / "order.safetensors"
    # A first, uninterrupted session, so the second run below actually has something to resume.
    run(ToyPipeline(), checkpoint_path=path, keep_checkpoint=True)

    observed: list[str] = []
    original_load = checkpoint_mod._load_for_resume

    def spy(store, *args, **kwargs):
        probe = open(store.lock_path, "rb")
        try:
            try:
                fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                observed.append("free")  # would mean nothing holds the lock yet
                fcntl.flock(probe.fileno(), fcntl.LOCK_UN)
            except OSError:
                observed.append("held")  # the running writer's own lock is already in place
        finally:
            probe.close()
        return original_load(store, *args, **kwargs)

    monkeypatch.setattr(checkpoint_mod, "_load_for_resume", spy)

    run(ToyPipeline(), checkpoint_path=path, keep_checkpoint=True)

    assert observed == ["held"], (
        "the lock must already be held by the time the checkpoint is read for resume, "
        f"got {observed}")


def test_a_real_write_is_visible_to_the_reader_end_to_end(tmp_path):
    """Exercises the whole liveness contract through both real modules at once: a genuine
    pipeline run under `run()` (through `checkpoint.py`'s writer), and the genuine reader entry
    point `h3_48gb.runs.scan()` -- not a hand-built lock path.

    `test_writer_holds_an_exclusive_lock_for_the_life_of_the_run` above proves the writer's own
    flock, but does so by reading `store.lock_path` -- the writer's own property -- so the writer
    and the probe travel together if that suffix ever changes. `test_runs.py`'s fixtures build the
    reader's lock path as a literal `".lock"` suffix, independent of `CheckpointStore.lock_path` --
    correct in isolation, but they never touch a real writer either. This is the one test in the
    suite that would turn red if `checkpoint.py`'s writer and `runs.py`'s reader ever disagreed on
    the filename convention, because it is the only one that drives a real instance of each and
    asks whether they still agree.
    """
    from h3_48gb import runs as runs_mod

    outdir = tmp_path / "outdir"
    path = outdir / "checkpoints" / "h3-live.safetensors"
    observed: list[str | None] = []

    class ObserveMidRun(ToyPipeline):
        def _decode_video(self, rows, *args, **kwargs):
            states = {r.outdir: r.state for r in runs_mod.scan(outdir)}
            observed.append(states.get(outdir))
            return super()._decode_video(rows, *args, **kwargs)

    run(ObserveMidRun(), checkpoint_path=path, keep_checkpoint=True)

    assert observed == ["in_flight"], (
        f"runs.scan() must see the real writer as in_flight mid-run, got {observed}")


def test_session_timing_is_not_part_of_the_identity():
    """A run resumed tomorrow must still match the checkpoint it wrote today.

    `started_at` changes every session by construction. If it reached `identity_digest`, every
    resume would be a fresh run and the checkpoint would be dead weight.
    """
    from h3_48gb.checkpoint import identity_digest

    identity = {"request": {"prompt": "a cat", "seed": 7}}
    before = identity_digest(identity)
    identity_with_timing = dict(identity, started_at="2026-08-10T21:00:00",
                                completed_at_start=5)

    assert identity_digest(identity) == before
    assert identity_digest(identity_with_timing) != before, (
        "the fixture is pointless if the digest ignores extra keys")


# -- standalone runner ------------------------------------------------------------------------------

def _main() -> int:
    import inspect
    import shutil
    import tempfile
    import traceback

    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failed = []
    for name, fn in tests:
        scratch = Path(tempfile.mkdtemp(prefix="h3-ckpt-"))
        try:
            needs = inspect.signature(fn).parameters
            fn(scratch) if "tmp_path" in needs else fn()
            print(f"  ok    {name}")
        except Exception:
            failed.append(name)
            print(f"  FAIL  {name}")
            traceback.print_exc()
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
