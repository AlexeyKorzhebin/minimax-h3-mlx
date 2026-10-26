"""`_decode_video`'s postprocessing tail must be byte-identical to upstream's numpy path.

The fix (`h3_48gb/pipeline.py`, `LazyMiniMaxH3Pipeline._decode_video`) moves the pixel-mean/std
unnormalize, the [0, 1] clip and the round-to-uint8 out of numpy — where upstream reassigns
`frames` to a brand-new full-size float32 array four times over (`upstream/minimax_h3_mlx/
pipeline.py:368-374`) — into the still-lazy MLX graph, so the *only* array that crosses the
`np.array()` barrier is the final uint8 buffer, a quarter the size of one of those float32
copies. See `docs/FEASIBILITY-chunked-vae.md` for the measured +10.8 GB RSS this removes.

This isolates just the changed tail: everything before `self.video_vae.decode(...)` — unpatchify,
the mean/std unnormalize of the *latents* — is shared, untouched code, so `StubVAE.decode` ignores
its input and hands back a fixed random tensor instead of paying for the real 5.21 GB video VAE or
a converted checkpoint. `upstream_postprocess` below is upstream's numpy path, copied verbatim, and
is the ground truth both the old and the new code were checked against.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.mlx
pytest.importorskip("mlx.core", reason="mlx: needs the MLX stack, absent here")


import mlx.core as mx
import numpy as np
import pytest

# Importing `h3_48gb.pipeline` first puts the vendored `upstream/` checkout on `sys.path` (see
# `h3_48gb._upstream.ensure_on_path`, run as an import-time side effect) -- `minimax_h3_mlx` is not
# installed, so importing it first would fail.
from h3_48gb.pipeline import LazyMiniMaxH3Pipeline

from minimax_h3_mlx.packing import PIXEL_MEAN, PIXEL_STD, patchify_video_latents


def upstream_postprocess(raw: np.ndarray) -> np.ndarray:
    """`upstream/minimax_h3_mlx/pipeline.py:368-374`, copied verbatim.

    `raw` is what `video_vae.decode(...)` returns before any of this fork's changes: ImageNet-
    normalized RGB, `(1, 3, F, H, W)` float32.
    """
    pixel_mean = np.array(PIXEL_MEAN, np.float32).reshape(1, 3, 1, 1, 1)
    pixel_std = np.array(PIXEL_STD, np.float32).reshape(1, 3, 1, 1, 1)
    frames = raw * pixel_std + pixel_mean
    frames = np.clip(frames, 0.0, 1.0)[0].transpose(1, 2, 3, 0)  # -> (F, H, W, 3)
    return (frames * 255.0 + 0.5).astype(np.uint8)


class StubVAE:
    """Hands back a fixed tensor instead of running the real video VAE."""

    def __init__(self, raw: mx.array, config):
        self._raw = raw
        self.config = config

    def decode(self, latents):  # noqa: ARG002 -- the point of the stub is to ignore this
        return self._raw

    def unload(self):
        pass


class _Config:
    def __init__(self, latent_channels: int):
        self.latent_channels = latent_channels
        self.latents_mean = [0.0] * latent_channels
        self.latents_std = [1.0] * latent_channels


class _DiT:
    """Only `.config.patch_size` is read; `.unload()` is called and must be a no-op."""

    def __init__(self, patch_size):
        self.config = type("C", (), {"patch_size": patch_size})()

    def unload(self):
        pass


class Bare(LazyMiniMaxH3Pipeline):
    """Only `_decode_video` — constructing the real pipeline needs 46 GB of weights."""

    def __init__(self, raw: mx.array, latent_channels: int, patch_size):
        self.video_vae = StubVAE(raw, _Config(latent_channels))
        self.dit = _DiT(patch_size)
        self._cache = None
        self._cache_timesteps = None


def decode_video(raw_np: np.ndarray) -> np.ndarray:
    """Drives the real `_decode_video` with a stubbed VAE that returns `raw_np`."""
    patch_size = (1, 1, 1)
    latent_channels = 4
    num_latent_frames, latent_height, latent_width = 2, 2, 2
    latents = mx.zeros((1, latent_channels, num_latent_frames, latent_height, latent_width))
    rows = patchify_video_latents(latents, patch_size)

    pipe = Bare(mx.array(raw_np), latent_channels, patch_size)
    return pipe._decode_video(rows, num_latent_frames, latent_height, latent_width)


def _random_raw(rng: np.random.Generator, shape) -> np.ndarray:
    # Deliberately wide (mean 0, std ~2.5) so unnormalized values land well outside [0, 1] in both
    # directions and the clip is actually exercised, not a no-op.
    return (rng.standard_normal(shape) * 2.5).astype(np.float32)


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 1234])
def test_decode_video_matches_upstream_bytes(seed):
    rng = np.random.default_rng(seed)
    raw = _random_raw(rng, (1, 3, 5, 6, 7))

    got = decode_video(raw)
    want = upstream_postprocess(raw)

    assert got.dtype == np.uint8
    assert got.shape == want.shape
    np.testing.assert_array_equal(got, want)


def test_decode_video_matches_upstream_bytes_at_rounding_edges():
    """Values placed exactly at (k + 0.5) / 255-ish boundaries, where a rounding-mode mismatch
    between MLX and numpy would show up first."""
    # For each of the 3 channels, sweep unnormalized pixel values so that
    # value * pixel_std + pixel_mean lands near every half-integer over [0, 255] after *255.
    pixel_mean = np.array(PIXEL_MEAN, np.float32).reshape(3, 1)
    pixel_std = np.array(PIXEL_STD, np.float32).reshape(3, 1)
    targets = (np.arange(0, 256, dtype=np.float32) - 0.5) / 255.0  # the exact halfway points
    targets = np.tile(targets, (3, 1))  # (3, 256)
    raw_flat = (targets - pixel_mean) / pixel_std  # invert the unnormalize
    raw = raw_flat.reshape(1, 3, 1, 1, 256).astype(np.float32)

    got = decode_video(raw)
    want = upstream_postprocess(raw)

    assert got.dtype == np.uint8
    np.testing.assert_array_equal(got, want)
