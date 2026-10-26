#!/usr/bin/env python3
"""Prove that `build_packed_sequence` accepts integer keyframe anchors on their real values.

    ./.venv/bin/pytest tests/test_packing_latent.py -q

`docs/FEASIBILITY-latent-handoff.md` §1.1: the new integer anchor is a *pixel-frame* index fed
straight into ``anchor_time = num_text + (5/3) * anchor`` -- the same formula the existing
``"first"``/``"last"`` literals reduce to, with ``pixel_index = 0`` for ``"first"`` and
``pixel_index = num_pixel_frames - 1`` for ``"last"``. It is **not** a latent-frame array index:
`_temporal_position_grid(F, origin)[k]` is not `origin + 5/3 * k` for `k > 1`, because the VAE's
17-pixel-frames-to-5-latent-frames grouping (`(1, 4, 4, 4, 4)` in `_ROPE_FRAMES_PER_LATENT`) makes
consecutive latent frames land on the *non-consecutive* pixel indices ``0, 1, 5, 9, 13, 17, 18,
...``. To anchor at latent frame ``k`` one must pass ``anchor = pixel_index(k)`` -- that pixel
index, not ``k`` itself; passing ``k`` directly anchors on pixel frame ``k``, a different (earlier)
point in time for every ``k >= 2``. Getting this backwards is exactly the kind of mistake the brief
calls out by name ("коллизия имён в спеке"), and it is also the bug this test suite is built to
catch: swap `pixel_index(k)` for a plain `k` anywhere below and every value-level assertion here
must go red.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.mlx
pytest.importorskip("mlx.core", reason="mlx: needs the MLX stack, absent here")


import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from h3_48gb import _upstream  # noqa: E402,F401

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from minimax_h3_mlx.packing import (  # noqa: E402
    _ROPE_FRAME_RESCALE,
    _ROPE_FRAMES_PER_LATENT,
    _temporal_position_grid,
    _temporal_position_span,
    build_packed_sequence,
)

# A small, cheap geometry: 12 latent frames (m=2: one seed frame plus two full 5-latent VAE
# chunks), a 2x2 patch on an 8x8 latent canvas -> 4x4 = 16 rows per frame, no audio.
NUM_TEXT = 6
NUM_LATENT_FRAMES = 12
LATENT_HEIGHT = 8
LATENT_WIDTH = 8
PATCH_SIZE = (1, 2, 2)
ROWS_PER_FRAME = (LATENT_HEIGHT // PATCH_SIZE[1]) * (LATENT_WIDTH // PATCH_SIZE[2])
TEXT_TAGS = [1] * NUM_TEXT

# `docs/FEASIBILITY-latent-handoff.md` §1.3: the pixel-index value of latent frame k, i.e. the
# value one must pass as the integer anchor to land on that latent frame -- 0, 1, 5, 9, 13, 17,
# 18, ... Independently derived from `_ROPE_FRAMES_PER_LATENT` (not from `_temporal_position_grid`
# itself), so this is a real, separate check of the grid's shape, not a tautology.
def _pixel_index_of_latent_frame(k: int) -> int:
    total = 0
    for i in range(k):
        total += _ROPE_FRAMES_PER_LATENT[i % len(_ROPE_FRAMES_PER_LATENT)]
    return total


# 39 pixel frames (17 * 2 + 5) is the smallest chunk-aligned pixel-frame count whose VAE encode
# yields exactly NUM_LATENT_FRAMES=12 latent frames (5 * 2 + 2); the literal "last" branch anchors
# at pixel index `num_pixel_frames - 1`.
NUM_PIXEL_FRAMES_FOR_GEOMETRY = 39
LAST_PIXEL_INDEX = NUM_PIXEL_FRAMES_FOR_GEOMETRY - 1


def _build(anchors):
    return build_packed_sequence(
        text_token_tags=TEXT_TAGS,
        num_latent_frames=NUM_LATENT_FRAMES,
        latent_height=LATENT_HEIGHT,
        latent_width=LATENT_WIDTH,
        num_audio_latents=0,
        patch_size=PATCH_SIZE,
        keyframe_anchors=anchors,
    )


def _condition_block(layout, block_index: int, num_blocks: int = 1) -> np.ndarray:
    """`position_ids[lo:hi]` for conditioning block `block_index`, as float64 (values already
    carry whatever float32 rounding `build_packed_sequence` applied on its way out)."""
    position_ids = np.array(layout.position_ids.tolist(), dtype=np.float64)
    lo = NUM_TEXT + block_index * ROWS_PER_FRAME
    hi = NUM_TEXT + (block_index + 1) * ROWS_PER_FRAME
    return position_ids[lo:hi]


def _float32_of(value: float) -> float:
    """Round a float64 expected value through float32, matching the cast at the end of
    `build_packed_sequence` (`position_ids.astype(np.float32)`, packing.py) -- comparisons against
    the packed output must clear this rounding, not just the formula."""
    return float(np.float32(value))


def test_integer_anchor_block_position_is_num_text_plus_scaled_pixel_index():
    """(a) position_ids[lo:hi, 0] is the constant num_text + (5/3) * pixel_index per block, and it
    differs between two blocks anchored on different latent frames."""
    pixel_a, pixel_b = _pixel_index_of_latent_frame(0), _pixel_index_of_latent_frame(3)
    assert pixel_a != pixel_b  # sanity: the two anchors below must be genuinely different

    layout = _build((pixel_a, pixel_b))

    for block_index, pixel_index in enumerate((pixel_a, pixel_b)):
        expected = _float32_of(NUM_TEXT + _ROPE_FRAME_RESCALE * pixel_index)
        column0 = _condition_block(layout, block_index)[:, 0]
        assert np.all(column0 == expected), (
            f"block {block_index} (pixel_index={pixel_index}): expected constant {expected!r} in "
            f"position_ids column 0, got {column0!r}"
        )

    # The two blocks must differ from one another -- a bug that ignores the anchor value and
    # anchors every block identically (e.g. always at pixel_index 0) would pass the per-block check
    # above but fail this one.
    col_a = _condition_block(layout, 0)[:, 0]
    col_b = _condition_block(layout, 1)[:, 0]
    assert not np.any(col_a == col_b)


def test_integer_anchor_spatial_columns_match_frame_grid():
    """(b) position_ids[lo:hi, 1:] for a conditioning block equals the same (h, w) frame_grid that
    every target video frame carries."""
    layout = _build((_pixel_index_of_latent_frame(2),))
    block_spatial = _condition_block(layout, 0)[:, 1:]

    # A target frame's spatial columns, read straight from the packed video rows (any frame will
    # do -- the spatial grid is frame-independent). No audio rows in this geometry, so the video
    # block starts right after the one conditioning block.
    position_ids = np.array(layout.position_ids.tolist(), dtype=np.float64)
    video_start = NUM_TEXT + ROWS_PER_FRAME
    target_spatial = position_ids[video_start : video_start + ROWS_PER_FRAME, 1:]

    assert np.array_equal(block_spatial, target_spatial)


def test_integer_anchor_matches_temporal_position_grid_at_its_index():
    """(c) the block's time column equals `_temporal_position_grid(num_latent_frames, num_text)[k]`
    for an anchor on latent frame k -- checked at several k spanning more than one VAE chunk, not
    just k=0 (where the distinction from a plain array index is invisible)."""
    grid = _temporal_position_grid(NUM_LATENT_FRAMES, float(NUM_TEXT))

    for k in (0, 1, 2, 5, 9, 11):
        pixel_index = _pixel_index_of_latent_frame(k)
        layout = _build((pixel_index,))
        column0 = _condition_block(layout, 0)[:, 0]
        expected = _float32_of(grid[k])
        assert np.all(column0 == expected), (
            f"anchor for latent frame k={k} (pixel_index={pixel_index}): expected {expected!r} "
            f"from _temporal_position_grid[{k}], got {column0!r}"
        )


def test_integer_anchor_zero_matches_first_bit_for_bit():
    """Anchoring on latent frame 0 (pixel_index 0) must be bit-identical to the literal `"first"`
    branch."""
    layout_first = _build(("first",))
    layout_zero = _build((0,))

    pos_first = np.array(layout_first.position_ids.tolist(), dtype=np.float32)
    pos_zero = np.array(layout_zero.position_ids.tolist(), dtype=np.float32)

    assert np.array_equal(pos_first, pos_zero), "anchor 0 must match \"first\" bit-for-bit"
    # Tighten to the exact bit pattern of the conditioning block itself, not just array equality.
    lo, hi = NUM_TEXT, NUM_TEXT + ROWS_PER_FRAME
    assert pos_first[lo:hi, 0].tobytes() == pos_zero[lo:hi, 0].tobytes()


def test_integer_anchor_last_pixel_index_discrepancy_against_last_literal_is_measured():
    """`anchor = num_pixel_frames - 1` (the pixel index the "last" literal reduces to, per
    docs/FEASIBILITY-latent-handoff.md §1.1) and the literal `"last"` compute the anchor time via
    two different float64 formulas: `_temporal_position_span`'s pairwise numpy sum (the "last"
    branch, untouched) vs. a single direct multiplication (the new integer branch). The module's
    own docstring warns these summation orders can differ in the last ulp from 16 latent frames
    onward. The brief demands this discrepancy be *measured*, not assumed -- so this test computes
    both formulas directly in float64 (bypassing the `astype(float32)` cast `build_packed_sequence`
    applies on the way out) and prints the true float64 gap, then separately confirms the packed
    float32 output the pipeline actually consumes is unaffected (0 differing bits).
    """
    last_via_span = (
        float(NUM_TEXT) + _temporal_position_span(NUM_LATENT_FRAMES) - _ROPE_FRAME_RESCALE
    )
    last_via_direct = float(NUM_TEXT) + _ROPE_FRAME_RESCALE * float(LAST_PIXEL_INDEX)
    float64_gap = abs(last_via_span - last_via_direct)

    # Now the actual packed output, cast to float32 as build_packed_sequence does.
    layout_last = _build(("last",))
    layout_index = _build((LAST_PIXEL_INDEX,))
    pos_last = np.array(layout_last.position_ids.tolist(), dtype=np.float32)
    pos_index = np.array(layout_index.position_ids.tolist(), dtype=np.float32)

    lo, hi = NUM_TEXT, NUM_TEXT + ROWS_PER_FRAME
    float32_diff_bits = int(
        np.sum(pos_last[lo:hi, 0].view(np.uint32) != pos_index[lo:hi, 0].view(np.uint32))
    )

    print(
        f"\n[measured] float64 |'last' pairwise-sum formula - anchor({LAST_PIXEL_INDEX}) direct "
        f"formula| = {float64_gap!r}; float32-cast packed output differing elements = "
        f"{float32_diff_bits} (of {hi - lo})"
    )

    # The float64 gap is real (ulp-level, not zero) -- this is the discrepancy the brief wants
    # measured rather than postulated away.
    assert float64_gap > 0.0
    # The brief states this is EXPECTED to collapse to 0 bits on the float32-cast output for this
    # repo's ulp regime -- record it as an explicit assertion so a future change to either formula
    # is caught, not silently accepted.
    assert float32_diff_bits == 0


def test_invalid_anchor_still_raises():
    """Non-string, non-integer anchors must still fail loudly -- the new branch must not swallow
    garbage silently."""
    with pytest.raises(ValueError):
        _build((3.5,))
    with pytest.raises(ValueError):
        _build(("middle",))
    # `bool` is an `int` subclass: without the explicit guard `True` would silently become
    # pixel index 1 (review of task 1, P2-1 -- the one uncovered line of the new branch).
    with pytest.raises(ValueError):
        _build((True,))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
