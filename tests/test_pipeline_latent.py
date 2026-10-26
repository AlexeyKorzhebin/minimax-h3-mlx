#!/usr/bin/env python3
"""Prove that `MiniMaxH3Pipeline.__call__` accepts raw latent rows as conditioning, and that
adding that parameter did not move a single existing run's identity digest.

    ./.venv/bin/pytest tests/test_pipeline_latent.py -q

`docs/FEASIBILITY-latent-handoff.md` §1.2: the tail of a previous scene is already in exactly the
space `_encode_keyframes` produces — patchified, normalized `(latent - mean) / std`, float32 — so
handing it back needs no VAE encode at all, only a way in. `condition_latent_rows` is that way in.
It bypasses `_encode_keyframes` entirely (and with it the float16 round trip, which reproduces a
rounding *inside* the reference's VAE encode and has nothing to say about rows that were never
encoded), but it does **not** bypass the noise augmentation: the rows land in the same slot the
packed layout marks as `t = 0.999`, so they are noised with `scale_noise(..., KEYFRAME_NOISE_AUG)`
exactly like an image keyframe's rows. That is what `test_latent_rows_reach_the_transformer_*`
pins, on values.

Four refusals, all loud, all here: an empty block, a shape that does not match the request's
geometry, a dtype that is not float32, and `images` passed alongside. The shape one is the
load-bearing check — `video_rows = mx.concatenate([condition_rows, video_rows])` (upstream :289)
would happily accept a block of the wrong length and desynchronize it from
`layout.num_condition_video_rows`, so every row of the sequence would be denoised against the
wrong position, timestep and mask, silently. The empty one guards a shape the check above cannot
see through: `(0, PATCH_DIM)` matches `expected` whenever `keyframe_anchors` is also empty, yet
still draws a real `mx.random.normal` sample and shifts every later noise draw off a bare run's.

The identity tests are the other half of the task, and they are about a *hazard*, not a feature:
`request_identity` (`h3_48gb/checkpoint.py:150`) hashes `bound.arguments` wholesale, so a new
parameter — even one left at its `None` default — would change the digest of **every run this
fork has ever checkpointed** and orphan every checkpoint on disk. `_IDENTITY_EXCLUDED` keeps that
from happening; `BASELINE_IDENTITY_DIGEST` below was captured from a real run *before* the
parameter existed, and `test_identity_digest_of_a_latent_free_run_is_unchanged` compares against
it byte for byte. The price is `test_identity_is_blind_to_the_latent_tail`, which pins the
resulting hole rather than hiding it — see that test's docstring and BACKLOG.md.

Nothing here loads the real model: `ToyPipeline` (tests/test_checkpoint.py) drives upstream's real
`__call__`, real packing and real scheduler with stand-ins for the four heavy components.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.mlx
pytest.importorskip("mlx.core", reason="mlx: needs the MLX stack, absent here")


import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from h3_48gb import _upstream  # noqa: E402,F401

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import minimax_h3_mlx.pipeline as pipeline_module  # noqa: E402
from minimax_h3_mlx.packing import KEYFRAME_NOISE_AUG  # noqa: E402
from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline  # noqa: E402
from minimax_h3_mlx.scheduler import MiniMaxH3Scheduler  # noqa: E402

from h3_48gb.checkpoint import (  # noqa: E402
    _IDENTITY_EXCLUDED,
    _META_KEY,
    identity_digest,
    request_identity,
)

from test_checkpoint import CANVAS, DURATION, KEYFRAME, SEED, STEPS, ToyPipeline, run  # noqa: E402

# The geometry `run()` fixes, unrolled: a 32x32 canvas over a spatial compression of 8 is a 4x4
# latent, and a 2x2 patch makes that 4 rows per latent frame, each of `latents_dim * 1 * 2 * 2` = 16
# values. Two conditioning blocks is the smallest count that can catch an off-by-one-block bug.
ROWS_PER_FRAME = 4
PATCH_DIM = 16
BLOCKS = 2
LATENT_SHAPE = (BLOCKS * ROWS_PER_FRAME, PATCH_DIM)

# Integer anchors (patches/0004-latent-anchors.patch): pixel-frame indices 0 and 1 are latent
# frames 0 and 1. Strings would work too, but the whole point of the feature is anchoring a tail
# where the previous scene left off, so the test exercises the path a tail actually takes.
ANCHORS = (0, 1)

#: Captured from a real `run(ToyPipeline(), checkpoint_dir=...)` **before** `condition_latent_rows`
#: was added to upstream's signature, by reading `identity_digest` back out of the checkpoint the
#: run left on disk. If adding the parameter shifts any existing run's identity, this is the value
#: that stops it — do not "update" it to whatever the code now produces.
BASELINE_IDENTITY_DIGEST = "d343d8709d7a35d2d9e6806a72da2090"


def latent_rows(scale: float = 1.0) -> mx.array:
    """A deterministic stand-in for a saved tail: distinct per row *and* per column, so a block
    that is transposed, rolled or replaced by `_encode_keyframes` output cannot compare equal."""
    values = np.arange(LATENT_SHAPE[0] * LATENT_SHAPE[1], dtype=np.float32)
    values = values.reshape(LATENT_SHAPE) / 97.0 - 0.5
    return mx.array(values * scale)


class _RecordingPipeline(ToyPipeline):
    """`ToyPipeline` that keeps the rows the transformer saw on its first forward, and counts
    `_encode_keyframes` calls so "the VAE encode was bypassed" is asserted, not assumed."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.first_video_latents: mx.array | None = None
        self.encode_calls = 0
        self.dit = _DiTRecorder(self.dit, self)

    def _encode_keyframes(self, images, height, width):
        self.encode_calls += 1
        return super()._encode_keyframes(images, height, width)


class _DiTRecorder:
    """Transparent proxy; `__getattr__` forwards `config` and everything else upstream reads."""

    def __init__(self, dit, owner: _RecordingPipeline):
        self._dit = dit
        self._owner = owner

    def __getattr__(self, name):
        return getattr(self._dit, name)

    def __call__(self, video_latents, *args, **kwargs):
        if self._owner.first_video_latents is None:
            self._owner.first_video_latents = video_latents
        return self._dit(video_latents, *args, **kwargs)


def _noised_like_a_keyframe(rows: mx.array) -> mx.array:
    """The conditioning block upstream must build from `rows`, derived independently of it.

    The draw order is load-bearing and reproduced here rather than trusted: `mx.random.seed(seed)`
    then the conditioning noise *first* (upstream :272-280), before video and audio latents.
    """
    mx.random.seed(SEED)
    noise = mx.random.normal(rows.shape).astype(mx.float32)
    scheduler = MiniMaxH3Scheduler(shift=ToyPipeline().config.sigma_shift_video)
    return scheduler.scale_noise(rows, KEYFRAME_NOISE_AUG, noise)


def _checkpoint_metadata(path: Path) -> dict:
    """The checkpoint's metadata, read with plain stdlib file I/O.

    Deliberately not `CheckpointStore.read()`: that materializes every tensor through `mx.load`,
    which memory-maps the file, and under the memory pressure of the full suite that read failed
    once with `[read] Unable to read from file` while the same test passed in isolation. The claim
    being made here is about a digest recorded on disk, so nothing about it needs the arrays or
    MLX. safetensors is a `<u64 header length><JSON header>` prefix and the free-form
    `__metadata__` map is inside that header.
    """
    with path.open("rb") as handle:
        length = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(length))
    return json.loads(header["__metadata__"][_META_KEY])


def _f32_bytes(array: mx.array) -> bytes:
    """Bit-for-bit comparison material. `==` on floats would forgive a NaN or a -0.0."""
    return np.array(array.astype(mx.float32)).tobytes()


# -- the feature -----------------------------------------------------------------------------

def test_latent_rows_reach_the_transformer_as_the_conditioning_block():
    """The rows handed in are the rows the model sees, noised exactly like an image keyframe's.

    The comparison is made in the bfloat16 the transformer is actually called with (upstream :306
    casts), because that is the boundary the claim is about; both sides are widened back to
    float32 for a byte comparison, which is lossless in that direction.
    """
    pipeline = _RecordingPipeline()
    rows = latent_rows()
    run(pipeline, condition_latent_rows=rows, keyframe_anchors=ANCHORS)

    assert pipeline.encode_calls == 0, "the VAE-encode path must not be touched by a latent handoff"
    seen = pipeline.first_video_latents
    assert seen is not None, "the transformer was never called"

    conditioning = seen[0, : LATENT_SHAPE[0]]
    expected = _noised_like_a_keyframe(rows).astype(mx.bfloat16)
    assert _f32_bytes(conditioning) == _f32_bytes(expected)

    # And the block is not the raw input either: the noise augmentation really happened.
    assert _f32_bytes(conditioning) != _f32_bytes(rows.astype(mx.bfloat16))


def test_latent_rows_and_an_image_keyframe_produce_different_conditioning():
    """A control for the test above: `_encode_keyframes` output is not accidentally equal to it."""
    latent_pipeline = _RecordingPipeline()
    run(latent_pipeline, condition_latent_rows=latent_rows(), keyframe_anchors=ANCHORS)

    image_pipeline = _RecordingPipeline(keyframe_rows=LATENT_SHAPE[0])
    run(image_pipeline, images=[KEYFRAME, KEYFRAME], keyframe_anchors=ANCHORS)

    assert image_pipeline.encode_calls == 1
    a = latent_pipeline.first_video_latents[0, : LATENT_SHAPE[0]]
    b = image_pipeline.first_video_latents[0, : LATENT_SHAPE[0]]
    assert _f32_bytes(a) != _f32_bytes(b)


def test_condition_rows_are_materialized_inside_call_before_the_first_forward(monkeypatch):
    """spec §4.2: the latent path skips `_encode_keyframes` and, with it, every `mx.eval` inside
    the video VAE — nothing forces `condition_rows` before the transformer sees it unless
    `__call__` does that itself (upstream :320-325). Watching *some* `mx.eval` happen nearby is
    not enough — the off-by-one trap: "some eval happened just before" can be satisfied by
    the *previous* iteration's eval — so the calling frame is asserted, not just the sequence.
    """
    real_eval = mx.eval
    log: list[tuple[str, str, tuple]] = []

    def spy(*args, **kwargs):
        frame = inspect.currentframe().f_back
        log.append((frame.f_code.co_name, frame.f_code.co_filename, args))
        return real_eval(*args, **kwargs)

    monkeypatch.setattr(mx, "eval", spy)

    pipeline = _RecordingPipeline()
    rows = latent_rows()

    dit_seen: list[int] = []
    inner = pipeline.dit

    class _DitCallWatcher:
        """Records how many `mx.eval` calls preceded the first transformer forward, then
        forwards to `inner` (itself `_RecordingPipeline`'s own `_DiTRecorder`) unchanged."""

        def __init__(self, dit):
            self._dit = dit

        def __getattr__(self, name):
            return getattr(self._dit, name)

        def __call__(self, *args, **kwargs):
            dit_seen.append(len(log))
            return self._dit(*args, **kwargs)

    pipeline.dit = _DitCallWatcher(inner)

    run(pipeline, condition_latent_rows=rows, keyframe_anchors=ANCHORS)

    assert dit_seen, "the transformer was never called"
    before_first_forward = log[: dit_seen[0]]
    hits = [
        call
        for call in before_first_forward
        if call[0] == "__call__"
        and call[1] == pipeline_module.__file__
        and len(call[2]) == 1
        and isinstance(call[2][0], mx.array)
        and tuple(call[2][0].shape) == tuple(rows.shape)
    ]
    assert hits, (
        "no `mx.eval` on the conditioning rows was issued by MiniMaxH3Pipeline.__call__ before "
        "the first forward — the latent graph reaches the transformer unmaterialized (spec §4.2). "
        f"Calls seen before the forward: {[(c[0], Path(c[1]).name) for c in before_first_forward]}"
    )
    # ...and it is the noised block, not the raw input: the eval must sit after `scale_noise`.
    evaled = np.array(hits[-1][2][0])
    assert evaled.tobytes() != np.array(rows).tobytes()


# -- the four refusals -----------------------------------------------------------------------

def test_an_empty_latent_tail_is_refused():
    """The shape check alone lets an empty tail through when `keyframe_anchors` is also empty:
    `expected` then collapses to `(0, PATCH_DIM)` and a `(0, PATCH_DIM)` array matches it. But
    `condition_rows` is not `None` in that case, so step 4's `mx.random.normal(condition_rows.
    shape)` still draws a real (if empty) sample off the request's generator, desyncing every
    later noise draw from a run that passed no `condition_latent_rows` at all — and, worse,
    sharing that run's identity digest (`test_identity_is_blind_to_the_latent_tail`), so nothing
    downstream can even tell the two apart.
    """
    pipeline = _RecordingPipeline()
    empty = mx.zeros((0, PATCH_DIM), dtype=mx.float32)
    with pytest.raises(ValueError, match=r"`condition_latent_rows` must not be empty"):
        run(pipeline, condition_latent_rows=empty, keyframe_anchors=())


def test_a_latent_of_the_wrong_shape_is_refused():
    """Wrong geometry is the failure that cannot be allowed to pass: a block of the wrong length
    concatenates cleanly and desynchronizes the whole sequence from `layout.position_ids`."""
    pipeline = _RecordingPipeline()
    wrong = mx.concatenate([latent_rows(), latent_rows()])  # one block too many
    with pytest.raises(ValueError, match=r"`condition_latent_rows` must be shaped"):
        run(pipeline, condition_latent_rows=wrong, keyframe_anchors=ANCHORS)


def test_a_latent_of_the_wrong_width_is_refused():
    """The row count can be right while the patch dimension is not — a tail saved at a different
    canvas resolution. Checked separately so a shape check that only compares `shape[0]` is red."""
    pipeline = _RecordingPipeline()
    narrow = latent_rows()[:, : PATCH_DIM // 2]
    with pytest.raises(ValueError, match=r"`condition_latent_rows` must be shaped"):
        run(pipeline, condition_latent_rows=narrow, keyframe_anchors=ANCHORS)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_a_latent_that_is_not_float32_is_refused(dtype):
    """float32 is what the checkpoint writes (`h3_48gb/checkpoint.py:618-640`) and what the packed
    sequence is built in; a lower-precision tail would silently halve the precision of the seam.

    Both half-precision dtypes MLX offers are exercised, not just the `float16` `_encode_keyframes`
    itself happens to round through: a check narrow enough to only reject `float16` would wave a
    `bfloat16` tail straight through.
    """
    pipeline = _RecordingPipeline()
    with pytest.raises(ValueError, match=r"`condition_latent_rows` must be float32"):
        run(pipeline, condition_latent_rows=latent_rows().astype(dtype),
            keyframe_anchors=ANCHORS)


def test_a_latent_together_with_images_is_refused():
    """Both sources fill the same slot: `_encode_keyframes` would append its own rows and the
    conditioning block would be twice `layout.num_condition_video_rows` (spec §1.2)."""
    pipeline = _RecordingPipeline(keyframe_rows=ROWS_PER_FRAME)
    with pytest.raises(ValueError, match=r"`condition_latent_rows` and `images` cannot be combined"):
        run(pipeline, condition_latent_rows=latent_rows(), images=[KEYFRAME],
            keyframe_anchors=ANCHORS)
    # The refusal must land before the 28.2 GB text encode, not after it (review: on the real
    # pipeline this is the difference between failing instantly and paying for a peak that only
    # gets thrown away).
    assert pipeline.text_encoder.calls == 0, "the text encoder ran before the conflict was refused"


# -- run identity ----------------------------------------------------------------------------

def test_the_new_parameter_is_excluded_from_run_identity():
    """The declaration itself, so the reason is greppable from the exclusion set."""
    assert "condition_latent_rows" in _IDENTITY_EXCLUDED
    assert "condition_latent_rows" in inspect.signature(MiniMaxH3Pipeline.__call__).parameters


def test_identity_digest_of_a_latent_free_run_is_unchanged(tmp_path):
    """Every run that predates this feature must still hash to the same digest.

    Not a synthetic `request_identity` call: the run goes through `CheckpointingPipeline.__call__`,
    which is what binds upstream's signature and applies its defaults — the exact place a new
    parameter leaks in. The digest is read back off disk from the checkpoint the run left, and the
    file's *name* is derived from it too, so both are checked.
    """
    pipeline = ToyPipeline()
    run(pipeline, checkpoint_dir=str(tmp_path), keep_checkpoint=True)

    files = sorted(tmp_path.glob("h3-*.safetensors"))
    assert [f.name for f in files] == [f"h3-{BASELINE_IDENTITY_DIGEST}.safetensors"]

    meta = _checkpoint_metadata(files[0])
    assert meta["identity_digest"] == BASELINE_IDENTITY_DIGEST
    assert "condition_latent_rows" not in meta["identity"]["request"]


def test_identity_is_blind_to_the_latent_tail():
    """The cost of the exclusion, pinned rather than left to be discovered.

    Two runs that differ **only** in the latent tail they continue from share one identity, so they
    share one checkpoint file: resuming scene N after its predecessor was regenerated would keep
    denoising against the old context and `_check_seam` could not see it (spec §4.2). Until task 4
    of the handoff wave (`FORMAT_VERSION` bump plus a real content digest of the tail), latent
    scenes must not be resumed — BACKLOG.md says so where an operator will read it.

    When task 4 lands, this test is the one that must go red and be rewritten. That is deliberate.
    """
    def digest_for(**overrides) -> str:
        kwargs = dict(
            prompt="a jeweled hummingbird beside a red orchid",
            duration_seconds=DURATION,
            num_inference_steps=STEPS,
            seed=SEED,
            height=CANVAS,
            width=CANVAS,
            keyframe_anchors=ANCHORS,
            verbose=False,
        )
        kwargs.update(overrides)
        bound = inspect.signature(MiniMaxH3Pipeline.__call__).bind(None, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        return identity_digest(request_identity(arguments, {"weights": "toy-v1"}))

    without = digest_for()
    one_tail = digest_for(condition_latent_rows=latent_rows())
    other_tail = digest_for(condition_latent_rows=latent_rows(scale=-3.0))

    assert without == one_tail == other_tail


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
