#!/usr/bin/env python3
"""Prove the preview feature is cheap, correct-shaped, and — above everything else — safe.

    ./.venv/bin/python test_preview.py

The whole point of a mid-generation preview is that it must never cost the run it is trying to
save the user from wasting: a run can be fifteen hours long, and nothing a preview does is worth
risking that. So the tests here are ordered by what would be worst to get wrong first:

* a broken VAE decode must fall back to the VAE-less latent heatmap, not raise;
* a *totally* broken video VAE (fallback included) must still leave `PreviewInterceptor` calling
  through to the real transformer with its result untouched — the generation must not even notice;
* only once that is nailed down do the shape/geometry tests matter.

Everything here uses a deliberately tiny synthetic `VideoVAE` (2 decoder layers, dim 16, an 8x8
latent grid) — enough to exercise the real module tree and the real chunking arithmetic without
the multi-second cost a realistically-sized decode has (see `docs/RESULTS.md`, "Previews", for what
that costs at native resolution, which is not what this file is measuring).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.mlx
pytest.importorskip("mlx.core", reason="mlx: needs the MLX stack, absent here")


import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from h3_48gb import _upstream  # noqa: E402,F401

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from h3_48gb import preview  # noqa: E402
from h3_48gb.pipeline import LazyMiniMaxH3Pipeline  # noqa: E402
from minimax_h3_mlx.packing import build_packed_sequence, patchify_video_latents  # noqa: E402
from minimax_h3_mlx.video_vae import VideoVAE, VideoVAEConfig  # noqa: E402

PATCH_SIZE = (1, 2, 2)


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name} FAILED {detail}")
    print(f"  ok  {name}")


# -- fixtures --------------------------------------------------------------------------------

def tiny_video_vae_config() -> VideoVAEConfig:
    """Real chunking arithmetic (clip_length 17, temporal ratio 4, token_drop 3 -> the same 7-latent-
    frame floor the release build has), a toy decoder otherwise."""
    return VideoVAEConfig(
        in_channels=3,
        out_channels=3,
        latent_channels=4,
        block_out_channels=(8, 16),
        layers_per_block=1,
        spatial_downsample_factors=(2, 2),
        temporal_downsample_factors=(1, 4),
        norm_num_groups=4,
        decoder_num_layers=2,
        decoder_num_attention_heads=2,
        decoder_attention_head_dim=8,
        decoder_num_register_tokens=1,
        decoder_ffn_mult=2,
        decoder_rope_theta=100.0,
        decoder_rope_dim_ratio=0.75,
        decoder_norm_eps=1e-5,
        clip_length=17,
        token_drop=3,
        latents_mean=(0.0,) * 4,
        latents_std=(1.0,) * 4,
    )


def tiny_video_vae() -> VideoVAE:
    return VideoVAE(tiny_video_vae_config())


def random_generated_rows(num_latent_frames: int, latent_height: int, latent_width: int,
                          channels: int = 4):
    latents = mx.random.normal((1, channels, num_latent_frames, latent_height, latent_width))
    return patchify_video_latents(latents.astype(mx.float32), PATCH_SIZE)


class FakePipeline:
    """Stands in for `LazyMiniMaxH3Pipeline`: only `.video_vae` is ever touched by this module."""

    def __init__(self, video_vae):
        self.video_vae = video_vae


class BrokenVideoVAE:
    """Simulates a video VAE that cannot even report its config — the worst case: both the real
    decode *and* the latent-only fallback (which also reads `.config.latent_channels`) fail."""

    @property
    def config(self):
        raise RuntimeError("simulated: the video VAE could not be loaded")

    def decode(self, z):
        raise RuntimeError("simulated: decode failed")


class DecodeFailsVideoVAE:
    """A video VAE whose config is fine but whose decode raises — the fallback should still work."""

    def __init__(self, config):
        self.config = config

    def decode(self, z):
        raise RuntimeError("simulated: decode failed")


# -- correctness -------------------------------------------------------------------------------

def test_min_preview_latent_frames_matches_released_config() -> None:
    # The released config (VideoVAEConfig defaults in `video_vae_config`/`load_video_vae`):
    # clip_length=17, temporal_compression_ratio=4, token_drop=3 -> chunk_tokens=5, floor=7. This is
    # also the number `VideoVAE.decode` itself reports in its `ValueError` message.
    cfg = VideoVAEConfig(temporal_downsample_factors=(1, 2, 2, 1, 1, 1), clip_length=17, token_drop=3)
    check("floor is 7 latent frames for the released geometry",
          preview.min_preview_latent_frames(cfg) == 7,
          f"got {preview.min_preview_latent_frames(cfg)}")


def test_render_preview_frame_shape_and_content() -> None:
    vae = tiny_video_vae()
    cfg = vae.config
    needed = preview.min_preview_latent_frames(cfg)
    latent_h, latent_w = 8, 8
    rows = random_generated_rows(needed, latent_h, latent_w)
    pipe = FakePipeline(vae)

    frame = preview.render_preview_frame(pipe, rows, needed, latent_h, latent_w, PATCH_SIZE)
    native_h, native_w = latent_h * cfg.spatial_compression_ratio, latent_w * cfg.spatial_compression_ratio
    check("preview frame is at native pixel resolution",
          frame.shape == (native_h, native_w, 3), f"got {frame.shape}")
    check("preview frame is uint8", frame.dtype == np.uint8, f"got {frame.dtype}")
    check("preview frame is not degenerate (has some variance)", float(frame.std()) > 0.0)


def test_render_preview_frame_rejects_too_few_frames() -> None:
    vae = tiny_video_vae()
    needed = preview.min_preview_latent_frames(vae.config)
    rows = random_generated_rows(needed - 1, 8, 8)
    pipe = FakePipeline(vae)
    try:
        preview.render_preview_frame(pipe, rows, needed - 1, 8, 8, PATCH_SIZE)
        raise AssertionError("expected a ValueError for too few latent frames")
    except ValueError as exc:
        check("clear error naming the shortfall", "need at least" in str(exc) or "need" in str(exc),
              f"got {exc!r}")


def test_render_latent_fallback_is_vae_free_and_cheap() -> None:
    latent_h, latent_w, channels, num_frames = 8, 8, 4, 9
    rows = random_generated_rows(num_frames, latent_h, latent_w, channels)
    frame = preview.render_latent_fallback(rows, num_frames, latent_h, latent_w, channels, PATCH_SIZE)
    check("fallback frame is at latent resolution", frame.shape == (latent_h, latent_w, 3),
          f"got {frame.shape}")
    check("fallback frame is uint8", frame.dtype == np.uint8)
    check("fallback frame is grayscale (R == G == B)",
          bool(np.array_equal(frame[..., 0], frame[..., 1])) and bool(np.array_equal(frame[..., 1], frame[..., 2])))


# -- resilience: the requirement that matters most ----------------------------------------------

def test_emit_preview_falls_back_when_decode_fails() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="h3-preview-"))
    try:
        cfg = tiny_video_vae_config()
        needed = preview.min_preview_latent_frames(cfg)
        rows = random_generated_rows(needed, 8, 8)
        pipe = FakePipeline(DecodeFailsVideoVAE(cfg))

        wrote = preview.emit_preview(pipe, rows, needed, 8, 8, PATCH_SIZE, tmp / "run", 5, verbose=False)
        check("emit_preview reports success via the fallback", wrote is True)
        dest = preview.preview_path(tmp / "run", 5)
        check("a JPEG landed at the expected <stem>-preview-stepNN.jpg path", dest.exists(),
              f"expected {dest}")
        check("the file is non-empty", dest.stat().st_size > 0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_emit_preview_never_raises_even_when_everything_fails() -> None:
    """The core safety requirement: a fully broken preview path must not raise, must not write a
    file, and must return False — quietly, so the caller (a run worth hours of compute) proceeds."""
    tmp = Path(tempfile.mkdtemp(prefix="h3-preview-"))
    try:
        pipe = FakePipeline(BrokenVideoVAE())
        rows = random_generated_rows(7, 8, 8)
        try:
            wrote = preview.emit_preview(pipe, rows, 7, 8, 8, PATCH_SIZE, tmp / "run", 12, verbose=False)
        except Exception as exc:  # noqa: BLE001 - this is exactly what must never happen
            raise AssertionError(f"emit_preview raised instead of swallowing the failure: {exc!r}")
        check("emit_preview reports failure rather than raising", wrote is False)
        dest = preview.preview_path(tmp / "run", 12)
        check("no partial/broken file was left behind", not dest.exists())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_preview_interceptor_never_breaks_the_forward() -> None:
    """`PreviewInterceptor` wraps the DiT; a preview failure at every single step must still let
    every real forward run and return its actual result, untouched."""
    calls = []

    def fake_dit(video_latents, audio_latents, *args, **kwargs):
        calls.append((video_latents.shape, audio_latents.shape))
        return "video-pred", "audio-pred"

    tmp = Path(tempfile.mkdtemp(prefix="h3-preview-"))
    try:
        pipe = FakePipeline(BrokenVideoVAE())
        wrapped = preview.PreviewInterceptor(
            fake_dit, pipe, every=1, stem=tmp / "run", n_cond_v=0, num_latent_frames=7,
            latent_height=8, latent_width=8, patch_size=PATCH_SIZE, verbose=False,
        )

        num_steps = 6
        for i in range(num_steps):
            video_latents = mx.zeros((1, 7 * 4 * 4, 4 * PATCH_SIZE[0] * PATCH_SIZE[1] * PATCH_SIZE[2]))
            audio_latents = mx.zeros((1, 3, 8))
            result = wrapped(video_latents, audio_latents)
            check(f"step {i}: the real forward's result passes through untouched",
                  result == ("video-pred", "audio-pred"), f"got {result}")

        check("every step actually reached the wrapped transformer despite every preview failing",
              len(calls) == num_steps, f"got {len(calls)}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_preview_interceptor_forwards_config_access() -> None:
    """`dit.config.patch_size` etc. must still resolve through the wrapper — the pipeline's own
    early geometry reads (and `h3_48gb.pipeline.LazyComponent`'s no-load contract) depend on it."""

    class FakeConfig:
        patch_size = (1, 2, 2)

    class FakeDit:
        config = FakeConfig()

        def __call__(self, *a, **k):
            return None

    wrapped = preview.PreviewInterceptor(
        FakeDit(), FakePipeline(BrokenVideoVAE()), every=5, stem="x", n_cond_v=0,
        num_latent_frames=7, latent_height=8, latent_width=8, patch_size=PATCH_SIZE, verbose=False,
    )
    check("config forwards through the wrapper", wrapped.config.patch_size == (1, 2, 2))


def test_install_preview_geometry_matches_build_packed_sequence() -> None:
    """`LazyMiniMaxH3Pipeline._install_preview` re-derives `n_cond_v` from the request arguments
    instead of reading it off upstream's `layout` (there is no seam to read it from without
    duplicating the loop — see the module docstring). This checks that re-derivation against the
    real `build_packed_sequence`, upstream's own source of truth for that number, on both a plain
    text-to-video request and a keyframe-conditioned one."""

    class FakeVideoVAEConfig:
        spatial_compression_ratio = 16

    class FakeDitConfig:
        patch_size = PATCH_SIZE

    stub = object.__new__(LazyMiniMaxH3Pipeline)
    stub.video_vae = type("V", (), {"config": FakeVideoVAEConfig()})()
    stub.dit = type("D", (), {"config": FakeDitConfig()})()

    for keyframe_anchors in ((), ("first",), ("first", "last")):
        arguments = {
            "height": 768, "width": 1344, "aspect": (16, 9), "duration_seconds": 5.0,
            "keyframe_anchors": keyframe_anchors, "verbose": False,
        }
        interceptor = stub._install_preview(lambda *a, **k: None, 5, "stem", arguments)

        # Upstream's own geometry, independently computed, to check `_install_preview` against.
        ratio = FakeVideoVAEConfig.spatial_compression_ratio
        latent_height, latent_width = 768 // ratio, 1344 // ratio
        from minimax_h3_mlx.packing import align_num_frames, video_latent_num_frames, FPS
        num_frames = align_num_frames(int(round(5.0 * FPS)))
        num_latent_frames = video_latent_num_frames(num_frames)
        num_audio_latents = 1  # irrelevant to `num_condition_video_rows`
        text_tags = [1] * 8  # irrelevant to `num_condition_video_rows`
        layout = build_packed_sequence(text_tags, num_latent_frames, latent_height, latent_width,
                                       num_audio_latents, PATCH_SIZE, keyframe_anchors)

        check(f"n_cond_v matches build_packed_sequence for keyframe_anchors={keyframe_anchors!r}",
              interceptor._n_cond_v == layout.num_condition_video_rows,
              f"got {interceptor._n_cond_v}, expected {layout.num_condition_video_rows}")
        check(f"num_latent_frames matches for keyframe_anchors={keyframe_anchors!r}",
              interceptor._num_latent_frames == num_latent_frames)
        check(f"latent_height/width match for keyframe_anchors={keyframe_anchors!r}",
              (interceptor._latent_height, interceptor._latent_width) == (latent_height, latent_width))


def test_install_preview_requires_a_stem() -> None:
    class FakeVideoVAEConfig:
        spatial_compression_ratio = 16

    class FakeDitConfig:
        patch_size = PATCH_SIZE

    stub = object.__new__(LazyMiniMaxH3Pipeline)
    stub.video_vae = type("V", (), {"config": FakeVideoVAEConfig()})()
    stub.dit = type("D", (), {"config": FakeDitConfig()})()
    arguments = {"height": 768, "width": 1344, "aspect": (16, 9), "duration_seconds": 5.0,
                 "keyframe_anchors": (), "verbose": False}
    try:
        stub._install_preview(lambda *a, **k: None, 5, None, arguments)
        raise AssertionError("expected a ValueError when preview_every > 0 but preview_stem is None")
    except ValueError as exc:
        check("the error names the missing argument", "preview_stem" in str(exc), f"got {exc!r}")


def test_pop_preview_kwargs() -> None:
    kwargs = {"prompt": "x", "preview_every": 5, "preview_stem": "y", "seed": 0}
    popped = preview.pop_preview_kwargs(kwargs)
    check("only the two preview kwargs are removed", set(popped) == {"preview_every", "preview_stem"})
    check("everything else is left alone", kwargs == {"prompt": "x", "seed": 0}, f"got {kwargs}")


def main() -> int:
    tests = [
        test_min_preview_latent_frames_matches_released_config,
        test_render_preview_frame_shape_and_content,
        test_render_preview_frame_rejects_too_few_frames,
        test_render_latent_fallback_is_vae_free_and_cheap,
        test_emit_preview_falls_back_when_decode_fails,
        test_emit_preview_never_raises_even_when_everything_fails,
        test_preview_interceptor_never_breaks_the_forward,
        test_preview_interceptor_forwards_config_access,
        test_install_preview_geometry_matches_build_packed_sequence,
        test_install_preview_requires_a_stem,
        test_pop_preview_kwargs,
    ]
    for test in tests:
        print(f"{test.__name__}:")
        test()
    print(f"\n{len(tests)} test groups passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
