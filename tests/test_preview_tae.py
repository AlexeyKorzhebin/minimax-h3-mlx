import pytest

pytestmark = pytest.mark.mlx
pytest.importorskip("mlx.core", reason="mlx: needs the MLX stack, absent here")

import sys
from pathlib import Path

import mlx.core as mx
import pytest

from PIL import Image

from h3_48gb.preview import emit_preview, preview_path
from h3_48gb.tae import SPATIAL_RATIO, TAE_WEIGHTS_PATH


class _ExplodingPipeline:
    """Any decoder that touches the VAE through this will fail; TAE must not need it."""

    class _Config:
        latent_channels = 24
        latents_mean = [0.0] * 24
        latents_std = [1.0] * 24
        clip_length = 17
        token_drop = 3

    @property
    def video_vae(self):
        raise AssertionError("a TAE preview must not load the video VAE")


def _rows(num_latent_frames=8, latent_h=8, latent_w=12, patch=(1, 2, 2)):
    per_frame = (latent_h // patch[1]) * (latent_w // patch[2])
    width = 24 * patch[0] * patch[1] * patch[2]
    return mx.zeros((num_latent_frames * per_frame, width))


@pytest.mark.skipif(not TAE_WEIGHTS_PATH.exists(),
                    reason=f"no TAE weights at {TAE_WEIGHTS_PATH}")
def test_a_tae_preview_never_touches_the_video_vae(tmp_path, monkeypatch):
    """The whole point: 9.8 MB instead of 5.21 GB, and no chunk floor.

    Checked by the frame's *size*, not by a file merely existing. Both decoders write a JPEG, so
    "a JPEG appeared" is satisfied by the latent fallback — which is what this test would silently
    become if TAE broke, or if the weights were absent, or if reading the VAE config ever started
    loading 5.21 GB. TAE upsamples the 8x12 latent by 16 to 192x128; the fallback draws the latent
    grid itself, at 12x8.
    """
    written = emit_preview(
        _ExplodingPipeline(), _rows(), 8, 8, 12, (1, 2, 2),
        stem=tmp_path / "run", step=3, decoder="tae",
    )
    assert written
    dest = preview_path(tmp_path / "run", 3)
    assert dest.exists()
    with Image.open(dest) as frame:
        assert frame.size == (12 * SPATIAL_RATIO, 8 * SPATIAL_RATIO), (
            f"expected a TAE frame at {12 * SPATIAL_RATIO}x{8 * SPATIAL_RATIO}, got {frame.size} "
            "— this is the latent fallback, so TAE did not run")


def test_a_missing_tae_file_falls_back_instead_of_raising(tmp_path, monkeypatch):
    from h3_48gb import tae

    monkeypatch.setattr(tae, "TAE_WEIGHTS_PATH", tmp_path / "absent.safetensors")
    written = emit_preview(
        _ExplodingPipeline(), _rows(), 8, 8, 12, (1, 2, 2),
        stem=tmp_path / "run", step=4, decoder="tae",
    )
    # The latent fallback needs no VAE weights either, so a frame still lands.
    assert written


def test_a_corrupt_tae_file_falls_back_instead_of_raising(tmp_path, monkeypatch):
    from h3_48gb import tae

    corrupt = tmp_path / "corrupt.safetensors"
    corrupt.write_bytes(b"not a safetensors file")
    monkeypatch.setattr(tae, "TAE_WEIGHTS_PATH", corrupt)
    written = emit_preview(
        _ExplodingPipeline(), _rows(), 8, 8, 12, (1, 2, 2),
        stem=tmp_path / "run", step=5, decoder="tae",
    )
    assert written


def test_an_unknown_decoder_name_is_refused_loudly(tmp_path):
    """A typo must not silently fall back to a different decoder than the caller asked for."""
    with pytest.raises(ValueError) as excinfo:
        emit_preview(_ExplodingPipeline(), _rows(), 8, 8, 12, (1, 2, 2),
                     stem=tmp_path / "run", step=1, decoder="taa")
    assert "taa" in str(excinfo.value)


def test_the_default_is_still_the_real_vae(tmp_path):
    """An experimental decoder must not become the default by being merged."""
    import inspect

    from h3_48gb.preview import emit_preview as fn

    assert inspect.signature(fn).parameters["decoder"].default == "vae"


def test_the_latent_decoder_is_reachable_and_needs_no_weights(tmp_path):
    """The third state ships too, and is the only one that works with nothing on disk at all."""
    written = emit_preview(
        _ExplodingPipeline(), _rows(), 8, 8, 12, (1, 2, 2),
        stem=tmp_path / "run", step=7, decoder="latent",
    )
    assert written
    dest = preview_path(tmp_path / "run", 7)
    with Image.open(dest) as frame:
        assert frame.size == (12, 8), (
            f"the latent heat map draws the latent grid itself; got {frame.size}")


# -- the wire from the CLI flag to the actual decode ----------------------------------------------

def test_the_decoder_choice_survives_the_whole_wire(tmp_path):
    """Four one-line mutations between the flag and the decode used to survive the whole suite.

    Each silently reverts to the real VAE: 49.3 s and 8.46 GB per preview instead of 0.125 s and
    2.06 — five minutes added to a run, with nothing to indicate it. The pieces (RunSpec, the
    parser, emit_preview) were each covered; the wire between them was not.

    Driven through `LazyMiniMaxH3Pipeline.__call__` so every hand-off is exercised:
    pop_preview_kwargs -> _install_preview -> PreviewInterceptor -> emit_preview.
    """
    import functools
    import inspect

    import mlx.core as mx

    from h3_48gb import preview as preview_module
    from h3_48gb.pipeline import LazyMiniMaxH3Pipeline
    from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline

    seen = []

    def spy_emit(pipeline, rows, *args, **kwargs):
        seen.append(kwargs.get("decoder"))
        return True

    class Config:
        sigma_shift_video = 12.0
        sigma_shift_audio = 3.0

    class FakeDiT:
        class config:
            patch_size = (1, 2, 2)

        def __call__(self, video, audio, *args, **kwargs):
            return video, audio

    class FakeVAEConfig:
        spatial_compression_ratio = 16
        latent_channels = 24

    class FakeVAE:
        config = FakeVAEConfig()

    original_call = MiniMaxH3Pipeline.__call__

    @functools.wraps(original_call)
    def run_two_steps(self, *args, **kwargs):
        # Stand in for the denoising loop: call the (intercepted) DiT the way it does.
        rows = mx.zeros((8 * 24, 96))
        for _ in range(2):
            self.dit(rows[None], rows[None])
        return "ok"

    for requested in ("tae", "latent", "vae"):
        pipe = LazyMiniMaxH3Pipeline(FakeDiT(), object(), FakeVAE(), object(), Config(),
                                     verbose=False)
        pipe.supported_num_inference_steps = lambda: None
        MiniMaxH3Pipeline.__call__ = run_two_steps
        real_emit = preview_module.emit_preview
        preview_module.emit_preview = spy_emit
        try:
            pipe(prompt="x", duration_seconds=1.0, num_inference_steps=31, seed=0,
                 height=128, width=192,
                 preview_every=1, preview_stem=tmp_path / "run", preview_decoder=requested)
        finally:
            MiniMaxH3Pipeline.__call__ = original_call
            preview_module.emit_preview = real_emit

        assert seen, f"no preview was emitted for decoder={requested!r}"
        assert set(seen) == {requested}, (
            f"asked for {requested!r}, emit_preview received {set(seen)!r} — the flag is dropped "
            "somewhere between pop_preview_kwargs and the interceptor")
        seen.clear()


def test_absent_weights_are_announced_once_not_every_preview(tmp_path, monkeypatch, capsys):
    """Missing TAE weights are a supported configuration, not an incident.

    Previews default to TAE, so a reader who never downloaded the 9.8 MB file takes this path on
    every preview of every run. Six identical lines per run is how a log stops being read.
    """
    from h3_48gb import preview as preview_module
    from h3_48gb import tae

    monkeypatch.setattr(tae, "TAE_WEIGHTS_PATH", tmp_path / "absent.safetensors")
    monkeypatch.setattr(preview_module, "_ANNOUNCED", {})

    for step in (1, 2, 3):
        assert emit_preview(
            _ExplodingPipeline(), _rows(), 8, 8, 12, (1, 2, 2),
            stem=tmp_path / "run", step=step, decoder="tae", verbose=False,
        )

    complaints = [line for line in capsys.readouterr().err.splitlines() if "not found" in line]
    assert len(complaints) == 1, f"expected one notice for three previews, got {len(complaints)}"
    assert "README" in complaints[0], "the notice must say where to get the weights"


def test_the_latent_decoder_is_not_just_a_failed_vae(tmp_path, capsys):
    """`latent` must go straight to the heat map, not attempt a decoder and fall back to it.

    The frame cannot tell these apart: a VAE that raises falls back to the same 12x8 heat map, so
    asserting the size passes either way — verified by mutation. What distinguishes them is that
    the direct path has nothing to report. A fallback always announces itself on stderr, because a
    silently-degraded preview defeats its own purpose.
    """
    written = emit_preview(
        _ExplodingPipeline(), _rows(), 8, 8, 12, (1, 2, 2),
        stem=tmp_path / "run", step=11, decoder="latent", verbose=False,
    )
    assert written
    with Image.open(preview_path(tmp_path / "run", 11)) as frame:
        assert frame.size == (12, 8), f"expected the latent grid at 12x8, got {frame.size}"

    stderr = capsys.readouterr().err
    assert "failed" not in stderr and "not found" not in stderr, (
        f"`latent` should reach the heat map directly, but something fell back to it: {stderr!r}")
