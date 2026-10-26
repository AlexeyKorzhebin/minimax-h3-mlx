#!/usr/bin/env python3
"""Prove phase-scoped residency: configs are free, weights are not loaded early, and the text
encoder's memory is really released.

    ./.venv/bin/python test_lazy_pipeline.py

Two things are easy to get wrong and neither raises:

* a proxy that loads on *any* touch defeats the point — the pipeline reads
  ``dit.config.patch_size`` and ``video_vae.config.spatial_compression_ratio`` long before the
  first weight is needed, so ``.config`` must not trigger a load;
* dropping a reference to the encoder frees nothing while a lazy MLX graph still reads its
  parameters. The unload is only real if the output is materialized first.

The second one is measured against MLX's allocator on arrays big enough for the difference to be
unambiguous, with a deliberately-lazy control that shows the measurement has teeth.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.mlx
pytest.importorskip("mlx.core", reason="mlx: needs the MLX stack, absent here")


import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from h3_48gb import _upstream  # noqa: E402,F401

import mlx.core as mx  # noqa: E402

from h3_48gb.memory import PhaseTracker, release, snapshot  # noqa: E402
from h3_48gb.pipeline import LazyComponent, LazyTextEncoder  # noqa: E402

#: Big enough that allocator noise cannot explain the drop: 256 MB per fake "encoder".
WEIGHT_ELEMENTS = 64 * 1024 * 1024
WEIGHT_GB = WEIGHT_ELEMENTS * 4 / 1e9


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name} FAILED {detail}")
    print(f"  ok  {name}")


class FakeConfig:
    def __init__(self, value):
        self.patch_size = value


class FakeModel:
    """Holds a real MLX array so its residency is visible to the allocator."""

    def __init__(self, config, elements=WEIGHT_ELEMENTS):
        self.config = config
        self.weight = mx.zeros((elements,), dtype=mx.float32)
        mx.eval(self.weight)
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return x * 2

    def encode(self, prompt, images=None):
        # Deliberately lazy: the returned graph still reads `self.weight`.
        return self.weight[:4] + len(prompt), ["tag"]


def test_config_is_free() -> None:
    loads = []
    proxy = LazyComponent("toy", FakeConfig((1, 2, 2)),
                          lambda: (loads.append(1), FakeModel(FakeConfig((9, 9, 9))))[1],
                          verbose=False)
    check("not loaded on construction", not proxy.loaded)
    check("config resolves without loading", proxy.config.patch_size == (1, 2, 2))
    check("still not loaded after reading the config", not proxy.loaded and not loads)


def test_load_on_first_real_use() -> None:
    loads = []
    proxy = LazyComponent("toy", FakeConfig((1, 2, 2)),
                          lambda: (loads.append(1), FakeModel(FakeConfig((1, 2, 2))))[1],
                          verbose=False)
    check("calling the proxy loads it", int(proxy(mx.array([2.0]))[0]) == 4 and len(loads) == 1)
    check("it is marked loaded", proxy.loaded)
    check("a second call does not reload", proxy(mx.array([1.0])) is not None and len(loads) == 1)
    check("attribute access forwards", proxy.calls == 2, f"got {proxy.calls}")
    check("config now comes from the loaded object", proxy.config.patch_size == (1, 2, 2))


def test_private_attributes_forward() -> None:
    """`_encode_keyframes` reaches into ``video_vae._encode_clip``; a blanket underscore guard
    would break keyframe conditioning with an AttributeError."""

    class WithPrivate(FakeModel):
        def _encode_clip(self, x):
            return x + 1

    proxy = LazyComponent("vae", FakeConfig(None),
                          lambda: WithPrivate(FakeConfig(None), elements=16), verbose=False)
    check("single-underscore methods forward", int(proxy._encode_clip(mx.array([1.0]))[0]) == 2)

    # Dunder probes (copy, pickle, repr protocols) must never trigger a load, whether Python finds
    # them on `object` or not.
    fresh = LazyComponent("vae", FakeConfig(None), lambda: None, verbose=False)
    for dunder in ("__deepcopy__", "__getstate__", "__setstate__", "__iter__"):
        try:
            getattr(fresh, dunder)
        except AttributeError:
            pass
        check(f"{dunder} does not trigger a load", not fresh.loaded)
    try:
        fresh.__deepcopy__
        raise AssertionError("an absent dunder should not be forwarded")
    except AttributeError:
        check("an absent dunder raises instead of forwarding to the loader", True)


def test_unload_releases_memory() -> None:
    mx.clear_cache()
    baseline = mx.get_active_memory()
    proxy = LazyComponent("weights", FakeConfig(None), lambda: FakeModel(FakeConfig(None)),
                          verbose=False)
    proxy.load()
    loaded = mx.get_active_memory()
    grew = (loaded - baseline) / 1e9
    check(f"loading is visible to the allocator (+{grew:.2f} GB)", grew > WEIGHT_GB * 0.8)

    proxy.unload()
    after = mx.get_active_memory()
    freed = (loaded - after) / 1e9
    check(f"unloading releases it (-{freed:.2f} GB)", freed > WEIGHT_GB * 0.8)
    check("and the proxy reports itself unloaded", not proxy.loaded)


class FakeEncoderWithUnload(FakeModel):
    """Like the real `QuantizedTextEncoder`, which drops its own sub-modules on unload."""

    def unload(self) -> int:
        released = self.weight.nbytes
        self.weight = None
        return released


def test_encoder_unloads_after_encode() -> None:
    """The whole point of task 4: 28.2 GB must not survive into the diffusion loop.

    Run twice — once on a component that only gets dropped by reference (which is what catches a
    stray local in the proxy) and once on one that also releases its own sub-modules.
    """
    for factory, label in ((FakeModel, "by reference alone"),
                           (FakeEncoderWithUnload, "with an unload() hook")):
        mx.clear_cache()
        baseline = mx.get_active_memory()
        proxy = LazyTextEncoder("text encoder", None,
                                lambda f=factory: f(FakeConfig(None)), verbose=False)
        embeds, tags = proxy.encode("hello")
        after = mx.get_active_memory()

        check(f"the encoder is gone once encode returns ({label})", not proxy.loaded)
        check(f"its memory went with it, {label} ({(after - baseline) / 1e9:+.2f} GB)",
              (after - baseline) / 1e9 < WEIGHT_GB * 0.2)
        check(f"the embeddings survive and are usable ({label})",
              embeds.shape == (4,) and float(embeds[0]) == 5.0, f"got {embeds}")
        check(f"the tags come back too ({label})", tags == ["tag"])


def test_unload_without_eval_would_not_free() -> None:
    """The control that gives the previous test its meaning.

    Dropping every reference to the model while a lazy graph still reads its weight frees nothing.
    This is what the pipeline would do without the `mx.eval` before the unload.
    """
    # Both halves use the same release primitive, so the only variable under test is the `mx.eval`.
    release()
    baseline = mx.get_active_memory()
    model = FakeModel(FakeConfig(None))
    lazy, _ = model.encode("hello")   # graph over model.weight, never evaluated
    loaded = mx.get_active_memory()
    del model
    release()
    still = mx.get_active_memory()
    check(f"an unevaluated graph pins the weights ({(still - baseline) / 1e9:.2f} GB still held)",
          (still - baseline) > 0.8 * (loaded - baseline),
          f"held {(still - baseline) / 1e9:.2f} of {(loaded - baseline) / 1e9:.2f} GB")

    mx.eval(lazy)
    del lazy
    release()
    check(f"and materializing then releasing does free them "
          f"({(mx.get_active_memory() - baseline) / 1e9:+.2f} GB)",
          (mx.get_active_memory() - baseline) / 1e9 < WEIGHT_GB * 0.2)


def test_reload_after_unload() -> None:
    calls = []
    proxy = LazyTextEncoder("text encoder", None,
                            lambda: (calls.append(1), FakeModel(FakeConfig(None), 16))[1],
                            verbose=False)
    proxy.encode("a")
    proxy.encode("b")
    check("a second request reloads rather than failing", len(calls) == 2, f"got {len(calls)}")

    kept = LazyTextEncoder("text encoder", None,
                           lambda: FakeModel(FakeConfig(None), 16),
                           verbose=False, unload_after_encode=False)
    kept.encode("a")
    check("unload_after_encode=False keeps it resident", kept.loaded)


def test_phase_tracker() -> None:
    tracker = PhaseTracker(verbose=False)
    tracker.mark("start")
    big = mx.zeros((WEIGHT_ELEMENTS,), dtype=mx.float32)
    mx.eval(big)
    tracker.mark("allocated")
    del big
    mx.clear_cache()
    tracker.mark("freed")
    check("three phases recorded", len(tracker.phases) == 3)
    check("the allocation shows up as a peak", tracker.phases[1][1]["mlx_peak_gb"] > WEIGHT_GB * 0.8,
          f"got {tracker.phases[1][1]['mlx_peak_gb']:.2f} GB")
    check("snapshot reports rss too", snapshot()["rss_gb"] > 0.0)


def test_keyframes_load_the_vae_before_upstream_seeds() -> None:
    """Constructing the VAE draws from the global RNG, so it must not happen after `seed(42)`.

    Upstream builds every component in `from_pretrained`; here the VAE is a proxy whose first
    real use is *inside* `_encode_keyframes`, after that seed. Building it there moves the stream
    by some 560 parameter draws, so the posterior sample lands elsewhere and a cold run differs
    from a warm one — measured at 0.87 before this override existed.
    """
    from h3_48gb.pipeline import LazyMiniMaxH3Pipeline
    from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline

    events: list[str] = []

    class ProxyVAE:
        def load(self):
            events.append("vae-load")

        def unload(self):
            events.append("vae-unload")

    class Config:
        sigma_shift_video = 12.0
        sigma_shift_audio = 3.0

    pipe = LazyMiniMaxH3Pipeline(object(), object(), ProxyVAE(), object(), Config(), verbose=False)

    original = MiniMaxH3Pipeline._encode_keyframes
    MiniMaxH3Pipeline._encode_keyframes = lambda self, images, h, w: (
        events.append("upstream-encode") or "rows")
    try:
        returned = pipe._encode_keyframes(["frame"], 512, 512)
    finally:
        MiniMaxH3Pipeline._encode_keyframes = original

    check("vae is loaded before upstream runs, and released after",
          events == ["vae-load", "upstream-encode", "vae-unload"], f"got {events}")
    check("upstream's return value is passed through", returned == "rows", f"got {returned!r}")


def test_both_consumers_get_the_same_prepared_keyframe() -> None:
    """The vision tower and the VAE must not be shown different pictures.

    Upstream hands the raw image to the text encoder and prepares it only for the VAE, so
    Qwen3-VL describes the original while the conditioning latent comes from the canvas version.
    Nothing raises — but the vision-token count changes, and `packing` derives the rotary clock
    of every audio and video row from it, so the whole timeline shifts.
    """
    import functools
    import inspect

    from PIL import Image

    from h3_48gb.pipeline import LazyMiniMaxH3Pipeline
    from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline

    class Config:
        sigma_shift_video = 12.0
        sigma_shift_audio = 3.0

    pipe = LazyMiniMaxH3Pipeline(object(), object(), object(), object(), Config(), verbose=False)
    pipe.supported_num_inference_steps = lambda: None

    original = MiniMaxH3Pipeline.__call__
    seen: dict = {}

    # `functools.wraps` is load-bearing: `__call__` binds against
    # `inspect.signature(MiniMaxH3Pipeline.__call__)`, so a spy with a bare `*args` signature
    # would swallow every argument into `args` and the assertion below would pass vacuously.
    @functools.wraps(original)
    def spy(self, *args, **kwargs):
        bound = inspect.signature(original).bind(self, *args, **kwargs)
        bound.apply_defaults()
        seen["sizes"] = [image.size for image in bound.arguments["images"]]
        return "ok"

    MiniMaxH3Pipeline.__call__ = spy
    try:
        for source, canvas in (((1536, 1024), (576, 384)),   # 3:2 landscape
                               ((896, 1152), (448, 576)),    # 7:9 portrait
                               ((576, 384), (576, 384))):    # already the canvas
            pipe(prompt="x", images=[Image.new("RGB", source)], keyframe_anchors=("first",),
                 height=canvas[1], width=canvas[0], seed=7)
            check(f"{source[0]}x{source[1]} reaches upstream as {canvas[0]}x{canvas[1]}",
                  seen["sizes"] == [canvas], f"got {seen['sizes']}")
    finally:
        MiniMaxH3Pipeline.__call__ = original


def test_each_keyframe_gets_its_own_seed() -> None:
    """Two identical keyframes must encode identically, and upstream seeds outside its loop.

    The reference builds a fresh generator per image, so keyframe 2 draws the same noise as
    keyframe 1. Upstream seeds once per request, so keyframe 2 continues the stream — measured
    0.83 apart on two byte-identical frames. Encoding one at a time puts upstream's own seed in
    front of every draw; verified against the real VAE at max|d| = 0.0.
    """
    from PIL import Image

    from h3_48gb.pipeline import LazyMiniMaxH3Pipeline
    from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline

    class ProxyVAE:
        def load(self):
            pass

        def unload(self):
            pass

    class Config:
        sigma_shift_video = 12.0
        sigma_shift_audio = 3.0

    pipe = LazyMiniMaxH3Pipeline(object(), object(), ProxyVAE(), object(), Config(), verbose=False)
    batches: list[int] = []

    def like_upstream(self, images, h, w):
        """Upstream's shape: seed once, then draw once per image from the shared stream.

        Reproducing the seed and the draw — rather than counting calls — is what makes this test
        about the noise instead of about batch sizes.
        """
        batches.append(len(images))
        mx.random.seed(42)
        return mx.concatenate([mx.random.normal((2, 4)) for _ in images])

    original = MiniMaxH3Pipeline._encode_keyframes
    MiniMaxH3Pipeline._encode_keyframes = like_upstream
    try:
        canvas = Image.new("RGB", (576, 384))
        rows = pipe._encode_keyframes([canvas, canvas], 384, 576)
        first, second = rows[:2], rows[2:]
        check("two identical keyframes draw identical noise",
              float(mx.abs(first - second).max()) == 0.0,
              f"max|d| = {float(mx.abs(first - second).max())}")
        check("two keyframes are encoded one at a time", batches == [1, 1], f"got {batches}")

        # The control: upstream's own arrangement, one call for both, is what this fixes — and it
        # must visibly fail the assertion above, or that assertion proves nothing.
        upstream_rows = like_upstream(pipe, [canvas, canvas], 384, 576)
        check("and upstream's single call would not have",
              float(mx.abs(upstream_rows[:2] - upstream_rows[2:]).max()) > 0.0)

        batches.clear()
        pipe._encode_keyframes([canvas], 384, 576)
        check("a single keyframe still goes through in one call", batches == [1], f"got {batches}")

        # The one-at-a-time path loses upstream's stretch/cover-crop distinction, so it may only
        # run on frames already on the canvas. If that stops holding, it must fail, not stretch.
        try:
            pipe._encode_keyframes([Image.new("RGB", (800, 600)), canvas], 384, 576)
            raise AssertionError("unprepared keyframes should have been refused")
        except RuntimeError as exc:
            check("unprepared keyframes are refused, not silently stretched",
                  "canvas" in str(exc), f"got {exc}")
    finally:
        MiniMaxH3Pipeline._encode_keyframes = original


def main() -> int:
    tests = [
        test_config_is_free,
        test_load_on_first_real_use,
        test_private_attributes_forward,
        test_unload_releases_memory,
        test_encoder_unloads_after_encode,
        test_unload_without_eval_would_not_free,
        test_reload_after_unload,
        test_phase_tracker,
        test_keyframes_load_the_vae_before_upstream_seeds,
        test_both_consumers_get_the_same_prepared_keyframe,
        test_each_keyframe_gets_its_own_seed,
        test_no_component_outlives_its_phase,
        test_the_allocator_cache_is_bounded,
        test_constructing_a_pipeline_applies_the_limit,
        test_the_default_limit_leaves_room_to_work,
        test_the_transformer_is_released_before_decoding,
    ]
    for test in tests:
        print(f"{test.__name__}:")
        test()
    print(f"\n{len(tests)} test groups passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


class FakeDiT:
    """Only `.config.patch_size` and `.unload()` are read by `_decode_video`."""

    def __init__(self, events: list[str], patch_size=(1, 1, 1)):
        self._events = events
        self.config = FakeConfig(patch_size)

    def unload(self):
        self._events.append("unload:dit")


class FakeVideoVAEConfig:
    latent_channels = 2
    latents_mean = [0.0, 0.0]
    latents_std = [1.0, 1.0]


class FakeVideoVAE:
    """`.config` and `.decode()` are read directly by `_decode_video` now that it no longer
    delegates to `MiniMaxH3Pipeline._decode_video` (see `h3_48gb/pipeline.py`) — the numpy tail
    was inlined and rewritten to finish in uint8 on the MLX side. `.decode()` ignores its input
    and hands back a fixed tiny tensor, same trick `StubVAE` in
    `tests/test_decode_video_uint8.py` uses to isolate this from the real 5.21 GB VAE.

    That tensor must not be all zeros: `_decode_video` unnormalizes it with `PIXEL_MEAN`/
    `PIXEL_STD` before `_validate_decoded_frames` (`h3_48gb/pipeline.py`) runs `framecheck` over
    the result, and a zero latent unnormalizes to exactly `(124, 116, 104)` — `framecheck.
    FILL_COLOR`, the flat-fill signature of an unwritten VAE tile (`h3_48gb/framecheck.py`).
    `find_corrupt_frames` correctly reads that as corruption and raises, which is what actually
    broke these two tests for about a month: not a pipeline defect, a fixture that predates the
    corruption check (patch 0003) and happened to fabricate the one RGB triple that check exists
    to catch. `mx.ones` unnormalizes to a color nowhere near `FILL_COLOR`, and the frame is 1x1 so
    `tile_seam_score` never has enough points to fire either way (`MIN_SEAM_POINTS`) — this frame
    passes validation honestly, not by dodging it."""

    def __init__(self, events: list[str]):
        self._events = events
        self.config = FakeVideoVAEConfig()

    def load(self):
        self._events.append("load:video_vae")

    def decode(self, latents):
        self._events.append("decode:video")
        return mx.ones((1, 3, 1, 1, 1))

    def unload(self):
        self._events.append("unload:video_vae")


def test_the_transformer_is_released_before_decoding() -> None:
    """Decoding must not pay for the transformer it will never touch again.

    Upstream keeps every component resident to the end of `__call__`, so the video VAE's tiling
    runs on top of 11.34 GB of transformer plus its LoRA and modulation table — 12.1 GB that the
    decode cannot use. On a long clip the decode is where the peak lands (243 frames at 896x576
    tile 28 ways each), which is exactly the wrong place to be carrying dead weight.

    The `mx.eval` before the unload is the same requirement `LazyTextEncoder` documents: an
    unevaluated graph over a module's parameters pins all of them regardless of what is dropped.
    """
    from minimax_h3_mlx.packing import patchify_video_latents

    from h3_48gb.pipeline import LazyMiniMaxH3Pipeline
    from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline

    events: list[str] = []

    class Proxy:
        def __init__(self, name):
            self.name = name

        def unload(self):
            events.append(f"unload:{self.name}")

    class Config:
        sigma_shift_video = 12.0
        sigma_shift_audio = 3.0

    dit = FakeDiT(events)
    video_vae = FakeVideoVAE(events)
    pipe = LazyMiniMaxH3Pipeline(dit, object(), video_vae, object(), Config(), verbose=False)
    pipe._cache = "a table"

    # `rows` for a single (1, 1, 1) latent voxel with 2 channels -- `FakeVideoVAE.decode` ignores
    # it, but `unpatchify_video_tokens` inside `_decode_video` still runs on it and needs a shape
    # that round-trips through `patch_size`.
    rows = patchify_video_latents(mx.zeros((1, 2, 1, 1, 1)), (1, 1, 1))

    original_audio = MiniMaxH3Pipeline._decode_audio
    MiniMaxH3Pipeline._decode_audio = lambda self, rows, *a, **k: events.append("decode:audio")
    try:
        pipe._decode_video(rows, 1, 1, 1)
        pipe._decode_audio(mx.zeros((4, 8)))
    finally:
        MiniMaxH3Pipeline._decode_audio = original_audio

    check("the transformer goes before the video decode, not after",
          events[:2] == ["unload:dit", "decode:video"], f"got {events}")
    check("the video VAE goes before the audio decode",
          events[2:] == ["unload:video_vae", "decode:audio"], f"got {events}")
    check("the modulation table is dropped with the transformer", pipe._cache is None)


def test_no_component_outlives_its_phase() -> None:
    """Every phase should hold only what it needs, and nothing that is merely convenient.

    Four unloads make that true, and each was absent at some point: the text encoder after
    `encode` (28.2 GB), the video VAE after keyframe encoding (5.21 GB across hours of
    diffusion), the transformer before decoding (11.34 GB plus LoRA and modulation table), and
    the video VAE again before the audio VAE loads. None of them raises when missing — the run
    just needs more memory than the machine has, which is the whole problem this fork exists for.
    """
    from minimax_h3_mlx.packing import patchify_video_latents

    from h3_48gb.pipeline import LazyMiniMaxH3Pipeline
    from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline

    events: list[str] = []

    class Proxy:
        def __init__(self, name):
            self.name = name
        def load(self):
            events.append(f"load:{self.name}")
        def unload(self):
            events.append(f"unload:{self.name}")

    class Config:
        sigma_shift_video = 12.0
        sigma_shift_audio = 3.0

    # `_decode_video` no longer delegates to `MiniMaxH3Pipeline._decode_video` (see
    # `h3_48gb/pipeline.py` — the numpy postprocessing tail was inlined and rewritten to finish
    # in uint8 on the MLX side), so `dit` and `video_vae` need enough of a real interface
    # (`.config`, `video_vae.decode()`) to survive that method's own body, not just a plain
    # load/unload `Proxy`.
    dit = FakeDiT(events)
    video = FakeVideoVAE(events)
    pipe = LazyMiniMaxH3Pipeline(dit, object(), video, Proxy("audio_vae"),
                                 Config(), verbose=False)

    rows = patchify_video_latents(mx.zeros((1, 2, 1, 1, 1)), (1, 1, 1))

    originals = (MiniMaxH3Pipeline._encode_keyframes, MiniMaxH3Pipeline._decode_audio)
    MiniMaxH3Pipeline._encode_keyframes = lambda self, i, h, w: mx.zeros((2, 4))
    MiniMaxH3Pipeline._decode_audio = lambda self, r, *a, **k: "audio"
    try:
        pipe._encode_keyframes([object()], 512, 512)
        pipe._decode_video(rows, 1, 1, 1)
        pipe._decode_audio(mx.zeros((2, 4)), 1)
    finally:
        (MiniMaxH3Pipeline._encode_keyframes, MiniMaxH3Pipeline._decode_audio) = originals

    check("the video VAE is released after keyframes, not held through diffusion",
          events[:2] == ["load:video_vae", "unload:video_vae"], f"got {events}")
    check("the transformer is released before decoding", "unload:dit" in events, f"got {events}")
    check("the video VAE is released again before the audio VAE runs",
          events.count("unload:video_vae") == 2, f"got {events}")


def test_the_allocator_cache_is_bounded() -> None:
    """MLX keeps freed buffers forever by default, which a 48 GB machine cannot afford.

    Measured on a real run: 29.1 GB held against roughly 21.4 GB of weights plus activations. The
    difference is cache the run has no use for, and it is what pushes the machine into swap — one
    step took 818 s where its neighbours took 568.
    """
    from h3_48gb import memory

    previous = memory.limit_cache(1.0)
    try:
        restored = memory.limit_cache(2.0)
        check("the limit is actually applied", restored == int(1.0 * 1e9), f"got {restored}")
        check("and reports MLX's own numbers, which ps cannot see",
              "active" in memory.report() and "cached" in memory.report())
    finally:
        memory.limit_cache(previous / 1e9)


def test_constructing_a_pipeline_applies_the_limit() -> None:
    """Pinning the rule is not pinning the call site.

    An earlier version of this file tested `limit_cache` alone; removing the pipeline's call to it
    left every test green. The same gap once shipped a modulation table nobody was transposing.
    """
    from h3_48gb import memory
    from h3_48gb.pipeline import LazyMiniMaxH3Pipeline

    calls: list = []
    original = memory.limit_cache
    memory.limit_cache = lambda *a, **k: calls.append(a) or 0
    try:
        class Config:
            sigma_shift_video = 12.0
            sigma_shift_audio = 3.0

        LazyMiniMaxH3Pipeline(object(), object(), object(), object(), Config(), verbose=False)
    finally:
        memory.limit_cache = original

    check("the pipeline bounds the allocator cache when it is built", len(calls) == 1,
          f"limit_cache was called {len(calls)} times")


def test_the_default_limit_leaves_room_to_work() -> None:
    """Zero would return every intermediate to the OS and reallocate it — correct and slow."""
    from h3_48gb.memory import DEFAULT_CACHE_LIMIT_GB

    check("the default is neither unbounded nor zero",
          0 < DEFAULT_CACHE_LIMIT_GB <= 8, f"got {DEFAULT_CACHE_LIMIT_GB}")
