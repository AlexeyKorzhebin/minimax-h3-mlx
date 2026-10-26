#!/usr/bin/env python3
"""Prove the QKV row permutation in ``convert_sawfwair.py`` is the right one, and reversible.

Run directly (no pytest needed)::

    ./.venv/bin/python test_qkv_permutation.py

The claim under test: mere.run stores ``attn.qkv_proj`` as three global slabs
``[all-q; all-k; all-v]``, and the port consumes it as
``qkv_proj(x).reshape(B, S, heads, 3, head_dim)`` indexed on axis -2 (``dit.py`` line 152), i.e.
per-head interleave ``[h0: q,k,v][h1: q,k,v]...``. Everything below is on a synthetic
3-head / 2-channel matrix small enough to check by eye.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from convert_sawfwair import (  # noqa: E402
    interleaved_to_slabs_index,
    permute_qkv_rows,
    slabs_to_interleaved_index,
)

HEADS, HEAD_DIM, IN_FEATURES = 3, 2, 4
INNER = HEADS * HEAD_DIM
ROWS = 3 * INNER


def slab_row(component: int, head: int, channel: int) -> int:
    """Row index of (component, head, channel) in the mere.run ``[all-q; all-k; all-v]`` layout."""
    return component * INNER + head * HEAD_DIM + channel


def interleaved_row(component: int, head: int, channel: int) -> int:
    """Row index of the same element in the port's per-head interleaved layout."""
    return head * 3 * HEAD_DIM + component * HEAD_DIM + channel


def tag(component: int, head: int, channel: int) -> int:
    """A value that uniquely identifies where a row came from."""
    return component * 100 + head * 10 + channel


def build_slab_matrix(width: int) -> np.ndarray:
    """A ``[3*inner, width]`` matrix whose every row is stamped with its (component, head, channel)."""
    out = np.zeros((ROWS, width), dtype=np.int64)
    for component in range(3):
        for head in range(HEADS):
            for channel in range(HEAD_DIM):
                out[slab_row(component, head, channel), :] = tag(component, head, channel)
    return out


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name} FAILED {detail}")
    print(f"  ok  {name}")


def test_index_is_a_permutation() -> None:
    forward = slabs_to_interleaved_index(HEADS, HEAD_DIM)
    check("forward index is a bijection over all rows",
          sorted(forward.tolist()) == list(range(ROWS)), f"got {forward.tolist()}")
    check("forward index length", forward.shape == (ROWS,), f"got {forward.shape}")


def test_rows_land_where_the_port_reads_them() -> None:
    """The heart of it: every (component, head, channel) must move slab -> interleaved position."""
    matrix = build_slab_matrix(IN_FEATURES)
    moved = permute_qkv_rows(matrix, HEADS, HEAD_DIM)

    check("shape is preserved", moved.shape == matrix.shape, f"got {moved.shape}")
    for component in range(3):
        for head in range(HEADS):
            for channel in range(HEAD_DIM):
                want = tag(component, head, channel)
                dst = interleaved_row(component, head, channel)
                check(f"c={component} h={head} d={channel} -> row {dst}",
                      bool((moved[dst] == want).all()),
                      f"row {dst} holds {moved[dst].tolist()}, wanted {want}")


def test_explicit_expected_layout() -> None:
    """Spell the whole 18-row result out, so a wrong-but-plausible permutation cannot pass."""
    moved = permute_qkv_rows(build_slab_matrix(1), HEADS, HEAD_DIM)[:, 0].tolist()
    expected = [
        # head 0: q(0,1)   k(0,1)   v(0,1)
        0, 1, 100, 101, 200, 201,
        # head 1
        10, 11, 110, 111, 210, 211,
        # head 2
        20, 21, 120, 121, 220, 221,
    ]
    check("full 18-row layout matches the hand-written expectation",
          moved == expected, f"\n    got      {moved}\n    expected {expected}")


def test_round_trip() -> None:
    forward = slabs_to_interleaved_index(HEADS, HEAD_DIM)
    backward = interleaved_to_slabs_index(HEADS, HEAD_DIM)
    check("inverse index is a bijection",
          sorted(backward.tolist()) == list(range(ROWS)))
    check("backward(forward) is the identity",
          (forward[backward] == np.arange(ROWS)).all())
    check("forward(backward) is the identity",
          (backward[forward] == np.arange(ROWS)).all())

    rng = np.random.default_rng(0)
    original = rng.integers(0, 2**31, size=(ROWS, IN_FEATURES), dtype=np.int64)
    check("permute then invert restores the original bytes",
          bool((original[forward][backward] == original).all()))


def test_quantized_triplet_moves_together() -> None:
    """``weight``, ``scales`` and ``biases`` share axis 0, so one index serves all three.

    This is why no dequantization is needed: MLX affine quantization groups along the *input*
    axis, so ``scales``/``biases`` are ``[rows, in_features/group_size]`` — a row keeps its own
    scale and bias when the rows are reordered.
    """
    groups = IN_FEATURES // 2  # pretend group_size = 2
    packed = build_slab_matrix(IN_FEATURES // 2).astype(np.uint32)   # stand-in for the U32 weight
    scales = build_slab_matrix(groups).astype(np.float32) / 1000.0
    biases = build_slab_matrix(groups).astype(np.float32) / -1000.0

    mw = permute_qkv_rows(packed, HEADS, HEAD_DIM)
    ms = permute_qkv_rows(scales, HEADS, HEAD_DIM)
    mb = permute_qkv_rows(biases, HEADS, HEAD_DIM)

    check("weight keeps its dtype", mw.dtype == np.uint32, f"got {mw.dtype}")
    for row in range(ROWS):
        check(f"row {row} keeps its own scale/bias",
              mw[row, 0] == round(ms[row, 0] * 1000) == round(-mb[row, 0] * 1000),
              f"weight {mw[row, 0]}, scale {ms[row, 0]}, bias {mb[row, 0]}")


def test_matches_the_ports_forward_pass() -> None:
    """End-to-end: a real projection, sliced the mere.run way and the port's way, must agree.

    ``dit.py`` does ``qkv_proj(x).reshape(B, S, heads, 3, head_dim)`` and takes q/k/v off axis -2.
    Slicing the *slab* matrix into thirds must give the same q, k, v.
    """
    rng = np.random.default_rng(7)
    weight_slabs = rng.standard_normal((ROWS, IN_FEATURES)).astype(np.float64)
    x = rng.standard_normal((2, 5, IN_FEATURES)).astype(np.float64)  # (B, S, in_features)

    # Reference: the mere.run layout, sliced into three contiguous slabs.
    ref = x @ weight_slabs.T
    ref_q, ref_k, ref_v = (ref[..., i * INNER:(i + 1) * INNER].reshape(2, 5, HEADS, HEAD_DIM)
                           for i in range(3))

    # Port: the permuted layout, reshaped and indexed on axis -2.
    weight_interleaved = permute_qkv_rows(weight_slabs, HEADS, HEAD_DIM)
    qkv = (x @ weight_interleaved.T).reshape(2, 5, HEADS, 3, HEAD_DIM)
    got_q, got_k, got_v = qkv[..., 0, :], qkv[..., 1, :], qkv[..., 2, :]

    for label, want, got in (("q", ref_q, got_q), ("k", ref_k, got_k), ("v", ref_v, got_v)):
        check(f"port-style {label} matches the slab slice",
              bool(np.allclose(want, got, rtol=0, atol=0)),
              f"max abs diff {np.abs(want - got).max()}")

    # And the wrong permutation must be visibly wrong — guards against a no-op passing the suite.
    naive = (x @ weight_slabs.T).reshape(2, 5, HEADS, 3, HEAD_DIM)
    check("unpermuted weights do NOT satisfy the port's reshape",
          not np.allclose(ref_q, naive[..., 0, :]))


def test_real_model_dimensions() -> None:
    """The shipped sizes: 56 heads x 128 = 7168 inner, 21504 rows."""
    heads, head_dim = 56, 128
    forward = slabs_to_interleaved_index(heads, head_dim)
    check("21504 rows", forward.shape == (3 * heads * head_dim,), f"got {forward.shape}")
    check("still a bijection at full size",
          bool((np.sort(forward) == np.arange(3 * heads * head_dim)).all()))
    # v-slab row 0 (component 2, head 0, channel 0) sits at 2*7168 = 14336 in the source and must
    # land at 2*128 = 256 in the destination.
    check("v[h0,d0]: source row 14336 -> destination row 256", forward[256] == 14336,
          f"got {forward[256]}")
    # q of head 55 channel 127 is source row 7167, destination row 55*384 + 127 = 21247.
    check("q[h55,d127]: source row 7167 -> destination row 21247", forward[21247] == 7167,
          f"got {forward[21247]}")


def main() -> int:
    tests = [
        test_index_is_a_permutation,
        test_explicit_expected_layout,
        test_rows_land_where_the_port_reads_them,
        test_round_trip,
        test_quantized_triplet_moves_together,
        test_matches_the_ports_forward_pass,
        test_real_model_dimensions,
    ]
    for test in tests:
        print(f"{test.__name__}:")
        test()
    print(f"\n{len(tests)} test groups passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
