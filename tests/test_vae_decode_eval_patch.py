"""`h3_48gb.pipeline.vae_decode_eval_patch_applied` -- the honest check for
`patches/0003-vae-decode-eval.patch`, mirroring `h3_48gb.dit.attention_levers_patch_applied`'s and
`h3_48gb.text_encoder.keyframe_scatter_patch_applied`'s own tests.

Also covers `_validate_decoded_frames`/`framecheck.CorruptFramesError`, the P0 fix's second half
(defense in depth on top of the patch: a decoded clip is verified before `_decode_video` returns
it).
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.mlx
pytest.importorskip("mlx.core", reason="mlx: needs the MLX stack, absent here")


from pathlib import Path

import numpy as np
import pytest

# Importing `h3_48gb.pipeline` first puts the vendored `upstream/` checkout on `sys.path` -- same
# reason `test_decode_video_uint8.py` does this before anything from `minimax_h3_mlx`.
from h3_48gb import framecheck
from h3_48gb.pipeline import _validate_decoded_frames, vae_decode_eval_patch_applied

PATCH_PATH = Path(__file__).resolve().parent.parent / "patches/0003-vae-decode-eval.patch"


def _patch_halves() -> tuple[str, str]:
    """`(removed, added)` -- the pre-patch and post-patch source lines of `patches/0003-vae-decode-
    eval.patch`, taken from the patch file itself rather than retyped so the two cannot drift.
    """
    patch = PATCH_PATH.read_text()
    removed = "\n".join(line[1:] for line in patch.splitlines()
                         if line.startswith("-") and not line.startswith("---"))
    added = "\n".join(line[1:] for line in patch.splitlines()
                       if line.startswith("+") and not line.startswith("+++"))
    return removed, added


def test_the_vendored_checkout_carries_the_vae_decode_eval_patch():
    """Not a unit test of the detector -- a statement about this working tree."""
    assert vae_decode_eval_patch_applied(), (
        "upstream/ is unpatched; run "
        "`git -C upstream apply ../patches/0003-vae-decode-eval.patch`")


def test_detector_reads_the_unpatched_source_as_unpatched():
    removed, _ = _patch_halves()
    # The pre-patch source has neither marker in either method's old body.
    assert not vae_decode_eval_patch_applied(tile_source=removed, chunk_source=removed)


def test_detector_reads_the_patched_source_as_patched():
    """The patch's added lines quote the old unpatched line in a comment -- exactly the trap
    `keyframe_scatter_patch_applied`'s own docstring documents for patch 0001, and the reason the
    comment-stripping in `vae_decode_eval_patch_applied` exists.
    """
    _, added = _patch_halves()
    assert vae_decode_eval_patch_applied(tile_source=added, chunk_source=added)


def test_detector_requires_both_markers():
    """Half a patch (only the tile-level eval, or only the chunk-level eval) must not read as
    fully applied -- `decode()`'s own loop can still leave an unbounded lazy graph if only one of
    the two insertion points landed.
    """
    _, added = _patch_halves()
    unrelated = "def encode(self):\n    return 1\n"
    assert not vae_decode_eval_patch_applied(tile_source=added, chunk_source=unrelated)
    assert not vae_decode_eval_patch_applied(tile_source=unrelated, chunk_source=added)


def test_detector_ignores_unrelated_source():
    unrelated = "def encode(self):\n    return 1\n"
    assert not vae_decode_eval_patch_applied(tile_source=unrelated, chunk_source=unrelated)


# -- _validate_decoded_frames / CorruptFramesError -----------------------------------------------


def _clean_frames(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frames = 128.0 + rng.normal(0.0, 5.0, size=(n, 64, 64, 3))
    return np.clip(frames, 0, 255).astype(np.uint8)


def test_validate_decoded_frames_passes_a_clean_clip():
    _validate_decoded_frames(_clean_frames(6))  # must not raise


def test_validate_decoded_frames_raises_on_a_zero_filled_frame():
    frames = _clean_frames(6)
    frames[3, :, :] = framecheck.FILL_COLOR

    with pytest.raises(framecheck.CorruptFramesError, match="frame indices"):
        _validate_decoded_frames(frames)


def test_validate_decoded_frames_names_every_bad_frame_index():
    frames = _clean_frames(6)
    frames[1, :, :] = framecheck.FILL_COLOR
    frames[4, :, :] = framecheck.FILL_COLOR

    with pytest.raises(framecheck.CorruptFramesError) as excinfo:
        _validate_decoded_frames(frames)

    message = str(excinfo.value)
    assert "1" in message and "4" in message
    assert "2 of 6" in message
