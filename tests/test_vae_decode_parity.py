"""GPU parity smoke test for `patches/0003-vae-decode-eval.patch`: decoding the same latents that
produced боевые ворота 2026-08-19's corrupted clip must be byte-identical to the pre-patch
reference decode saved during the chunk-recon investigation
(`~/h3-outputs/chunk-recon/run1/`).

`mx.eval` is a pure materialization -- it changes *when* MLX computes a value, never *what* the
value is -- so this is the test that actually pins that claim down, rather than just asserting it
in a docstring.

Mirrors `test_facerefine.py`'s own "the one GPU test" convention: `@pytest.mark.slow`, gated behind
`H3_GPU_SMOKE=1` so the ordinary test run never pays its GPU minutes and ~10.6 GB peak.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

RECON_DIR = Path.home() / "h3-outputs/chunk-recon/run1"
CHECKPOINT = Path.home() / "models/h3-8bit-full"


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("H3_GPU_SMOKE") != "1",
                     reason="GPU minutes and ~10.6 GB peak; run with H3_GPU_SMOKE=1")
def test_patched_decode_matches_the_pre_patch_reference_bit_for_bit():
    """`frames-decode0.npy` is `chunk-recon/analyze_latents.py`'s own decode of `latents-
    recon0.npz` from *before* this patch bounded `VideoVAE.decode`'s graph -- the investigation's
    own reference, captured under a stable allocator (not the churned-memory state that actually
    triggered corruption). Decoding the same latents through the now-patched `VideoVAE.decode`
    must produce the exact same bytes.
    """
    latents_path = RECON_DIR / "latents-recon0.npz"
    reference_path = RECON_DIR / "frames-decode0.npy"
    if not latents_path.is_file() or not reference_path.is_file():
        pytest.skip(f"chunk-recon investigation artifacts not present under {RECON_DIR}")
    if not CHECKPOINT.is_dir():
        pytest.skip(f"no checkpoint at {CHECKPOINT}")

    import mlx.core as mx

    from h3_48gb.pipeline import _load_video_vae, vae_decode_eval_patch_applied, video_vae_config
    from minimax_h3_mlx.packing import PIXEL_MEAN, PIXEL_STD, unpatchify_video_tokens

    assert vae_decode_eval_patch_applied(), (
        "upstream/ is unpatched; run "
        "`git -C upstream apply ../patches/0003-vae-decode-eval.patch`")

    d = np.load(latents_path)
    rows = mx.array(d["rows"])
    nlf, lh, lw = int(d["num_latent_frames"]), int(d["latent_height"]), int(d["latent_width"])
    patch_size = tuple(int(x) for x in d["patch_size"])

    cfg = video_vae_config(CHECKPOINT / "video_vae")
    latents = unpatchify_video_tokens(rows, nlf, lh, lw, cfg.latent_channels, patch_size)
    mean = mx.array(np.array(cfg.latents_mean, np.float32)).reshape(1, -1, 1, 1, 1)
    std = mx.array(np.array(cfg.latents_std, np.float32)).reshape(1, -1, 1, 1, 1)
    latents = (latents * std + mean).astype(mx.float32)
    mx.eval(latents)

    vae = _load_video_vae(CHECKPOINT / "video_vae")
    frames = vae.decode(latents)
    pixel_mean = mx.array(np.array(PIXEL_MEAN, np.float32)).reshape(1, 3, 1, 1, 1)
    pixel_std = mx.array(np.array(PIXEL_STD, np.float32)).reshape(1, 3, 1, 1, 1)
    frames = frames * pixel_std + pixel_mean
    frames = mx.clip(frames, 0.0, 1.0)
    frames = (frames * 255.0 + 0.5).astype(mx.uint8)
    frames = frames[0].transpose(1, 2, 3, 0)
    got = np.array(frames)

    want = np.load(reference_path)
    assert got.shape == want.shape
    np.testing.assert_array_equal(got, want)
