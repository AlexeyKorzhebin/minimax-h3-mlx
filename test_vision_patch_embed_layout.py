#!/usr/bin/env python3
"""Prove the vision patch-embed conv weight lands in the right layout, not just a matching shape.

Run directly (no pytest needed)::

    ./.venv/bin/python test_vision_patch_embed_layout.py

The bug being fixed: the converted checkpoint stores ``model.visual.patch_embed.proj.weight`` in
PyTorch's conv layout, ``(out, in, D, H, W)``. ``mlx.nn.Conv3d`` expects channels-last,
``(out, D, H, W, in)``. The real checkpoint's kernel is ``(D, H, W) = (2, 16, 16)`` — H and W equal
— which means a shape check alone cannot tell the correct permutation, ``(0, 2, 3, 4, 1)``, apart
from a wrong one that also swaps H and W, e.g. ``(0, 4, 3, 2, 1)``: both produce a
``(out, 2, 16, 16, in)`` tensor. A wrong permutation would not raise anywhere — it would silently
scramble the patch embedding, and the only symptom would be a worse clip, hours later.

So every test below either uses dimensions that are all pairwise distinct (so a wrong permutation
changes the *shape*, not just the values) or checks values against a reference computed with plain
Python indexing rather than another call to ``.transpose()`` (so a bug shared between the function
under test and the check itself cannot cancel out).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.mlx
pytest.importorskip("mlx.core", reason="mlx: needs the MLX stack, absent here")


import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from h3_48gb import _upstream  # noqa: E402,F401

import mlx.core as mx  # noqa: E402

from h3_48gb.text_encoder import (  # noqa: E402
    VISION_CONV3D_WEIGHTS,
    to_mlx_conv3d_layout,
)

# All-distinct so any axis transposition, not just an H/W swap, changes the output shape.
OUT, IN, D, H, W = 4, 3, 2, 5, 7


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name} FAILED {detail}")
    print(f"  ok  {name}")


def test_output_shape_is_channels_last() -> None:
    source = mx.zeros((OUT, IN, D, H, W))
    result = to_mlx_conv3d_layout(source)
    check("out_channels, D, H, W, in_channels, in that order",
          result.shape == (OUT, D, H, W, IN), f"got {result.shape}")


def test_real_checkpoint_shape_is_unambiguous_only_by_luck() -> None:
    """The real checkpoint's H == W == 16, which the tests above deliberately avoid relying on."""
    source = mx.zeros((1152, 3, 2, 16, 16))
    result = to_mlx_conv3d_layout(source)
    check("matches mlx.nn.Conv3d's own weight shape for this checkpoint's config",
          result.shape == (1152, 2, 16, 16, 3), f"got {result.shape}")


def test_values_land_at_the_permuted_index_not_just_the_right_shape() -> None:
    """The actual trap: pin values, computed independently of `.transpose`.

    Every element of `source` is a unique integer encoding its own (o, i, d, h, w) index, built with
    plain nested loops. `expected[o, d, h, w, c]` is filled in the same way, reading straight from
    that formula rather than by calling `.transpose()` on `source` again — so this cannot pass by
    the two implementations sharing the same bug.
    """
    source_np = np.arange(OUT * IN * D * H * W, dtype=np.float32).reshape(OUT, IN, D, H, W)
    expected_np = np.empty((OUT, D, H, W, IN), dtype=np.float32)
    for o in range(OUT):
        for i in range(IN):
            for d in range(D):
                for h in range(H):
                    for w in range(W):
                        expected_np[o, d, h, w, i] = source_np[o, i, d, h, w]

    result = np.asarray(to_mlx_conv3d_layout(mx.array(source_np)))
    check("every element lands at its permuted index, not merely the right shape",
          np.array_equal(result, expected_np))


def test_a_plausible_wrong_permutation_is_actually_distinguishable() -> None:
    """Guard the guard: confirm the wrong permutation this test suite worries about really differs.

    If this failed, the value-pinning test above would not be exercising anything: it would mean
    `(0, 2, 3, 4, 1)` and the wrong candidate `(0, 4, 3, 2, 1)` produce identical arrays, and no test
    here could tell them apart no matter how it was written.
    """
    source_np = np.arange(OUT * IN * D * H * W, dtype=np.float32).reshape(OUT, IN, D, H, W)
    correct = np.asarray(to_mlx_conv3d_layout(mx.array(source_np)))
    wrong = source_np.transpose(0, 4, 3, 2, 1)  # swaps H and W as well as moving `in` to the end
    check("the wrong permutation this suite guards against is not secretly identical",
          correct.shape != wrong.shape or not np.array_equal(correct, wrong),
          f"correct.shape={correct.shape} wrong.shape={wrong.shape}")


def test_h_equals_w_like_the_real_checkpoint_still_catches_a_wrong_permutation() -> None:
    """The exact collision the real checkpoint has: kernel (D, H, W) = (2, 16, 16), H == W.

    Reproduced here at a smaller size (H = W = 5) so it runs in milliseconds. With H == W, the
    correct permutation `(0, 2, 3, 4, 1)` and the wrong one that also swaps H and W,
    `(0, 2, 4, 3, 1)`, produce the *same shape* — `(out, D, H, W, in)` either way, since H and W are
    interchangeable in the shape tuple. Only a value check, not a shape check, can catch this one.
    """
    o, i, d, hw = 4, 3, 2, 5
    source_np = np.arange(o * i * d * hw * hw, dtype=np.float32).reshape(o, i, d, hw, hw)
    correct = np.asarray(to_mlx_conv3d_layout(mx.array(source_np)))
    wrong = source_np.transpose(0, 2, 4, 3, 1)  # swaps H and W; same resulting shape as correct
    check("same shape as the correct permutation (this is why shape alone can't catch it)",
          correct.shape == wrong.shape, f"{correct.shape} vs {wrong.shape}")
    check("but the values differ, because this source has no H/W symmetry to hide behind",
          not np.array_equal(correct, wrong))


def test_only_the_conv_stem_is_marked_for_transposition() -> None:
    """Everything else in the vision tower is attention/linear and layout-agnostic."""
    check("exactly one vision weight needs this treatment",
          VISION_CONV3D_WEIGHTS == frozenset({"patch_embed.proj.weight"}),
          f"got {VISION_CONV3D_WEIGHTS}")


def test_bias_is_left_alone() -> None:
    """The stem's bias is `(out_channels,)` — 1-D, so no axis order applies to it at all."""
    check("patch_embed.proj.bias is not in the set that gets transposed",
          "patch_embed.proj.bias" not in VISION_CONV3D_WEIGHTS)


def test_the_loader_actually_applies_the_transpose() -> None:
    """`to_mlx_conv3d_layout` being correct is worth nothing if the loader stops calling it.

    Deleting the call used to leave all 159 tests green, because every test here exercised the
    pure function directly. `prepare_loaded_tensor` is the loader's own per-tensor step, so this
    fails if the rule is dropped, moved, or scoped to the wrong bucket.
    """
    import mlx.core as mx

    from h3_48gb.text_encoder import VISION_CONV3D_WEIGHTS, prepare_loaded_tensor

    source = mx.random.normal((4, 3, 2, 16, 16))          # (out, in, D, H, W), PyTorch layout
    path = next(iter(VISION_CONV3D_WEIGHTS))

    out = prepare_loaded_tensor("vision", path, source, mx.float32)
    check("the vision conv weight is transposed to channels-last",
          out.shape == (4, 2, 16, 16, 3), f"got {out.shape}")
    # Values, not just shape: H and W are both 16 in the real checkpoint, so a permutation that
    # swaps them produces this exact shape too.
    check("and it is the right permutation",
          float(mx.abs(out - source.transpose(0, 2, 3, 4, 1)).max()) == 0.0)

    same = prepare_loaded_tensor("language", path, source, mx.float32)
    check("the same path in another bucket is left alone", same.shape == source.shape,
          f"got {same.shape}")
    other = prepare_loaded_tensor("vision", "blocks.0.mlp.fc1.weight", source, mx.float32)
    check("other vision weights are left alone", other.shape == source.shape, f"got {other.shape}")

    packed = mx.zeros((4, 8), dtype=mx.uint32)
    kept = prepare_loaded_tensor("vision", "blocks.0.attn.qkv.weight", packed, mx.bfloat16)
    check("packed quantized storage is never cast", kept.dtype == mx.uint32, f"got {kept.dtype}")


def test_the_real_loader_transposes_on_the_way_in() -> None:
    """The rule being right is worth nothing if `_load_weights` stops calling it.

    `test_the_loader_actually_applies_the_transpose` pins `prepare_loaded_tensor`, but replacing
    the call site with `loaded[key]` still left every test green — the point of application was
    unguarded. This drives the real `_load_weights` over a two-tensor synthetic checkpoint, so
    the call site itself is covered without materializing 28.2 GB.
    """
    import tempfile
    from pathlib import Path as _Path

    import mlx.core as mx
    import mlx.nn as nn

    from h3_48gb.text_encoder import QuantizedTextEncoder

    class Leaf(nn.Module):
        def __init__(self, shape):
            super().__init__()
            self.weight = mx.zeros(shape)

    class PatchEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = Leaf((4, 2, 16, 16, 3))      # channels-last, as mlx.nn.Conv3d builds it

    class Vision(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = PatchEmbed()

    class Language(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = Leaf((8,))

    encoder = object.__new__(QuantizedTextEncoder)
    encoder._recipe = None
    encoder.num_layers = 1
    encoder.language, encoder.vision = Language(), Vision()
    encoder.quantized_layers = {"language": 0, "vision": 0}

    source = mx.random.normal((4, 3, 2, 16, 16))     # (out, in, D, H, W), as the checkpoint stores it
    with tempfile.TemporaryDirectory() as tmp:
        directory = _Path(tmp)
        mx.save_safetensors(str(directory / "model.safetensors"), {
            "model.visual.patch_embed.proj.weight": source,
            "model.language_model.norm.weight": mx.zeros((8,)),
        })
        encoder._load_weights(directory, mx.float32, False)
    loaded = encoder.vision.patch_embed.proj.weight

    check("the loader stores the conv weight channels-last", loaded.shape == (4, 2, 16, 16, 3),
          f"got {loaded.shape}")
    check("with the right permutation, not merely the right shape",
          float(mx.abs(loaded - source.transpose(0, 2, 3, 4, 1)).max()) == 0.0)


def main() -> int:
    tests = [
        test_output_shape_is_channels_last,
        test_real_checkpoint_shape_is_unambiguous_only_by_luck,
        test_values_land_at_the_permuted_index_not_just_the_right_shape,
        test_a_plausible_wrong_permutation_is_actually_distinguishable,
        test_h_equals_w_like_the_real_checkpoint_still_catches_a_wrong_permutation,
        test_only_the_conv_stem_is_marked_for_transposition,
        test_bias_is_left_alone,
        test_the_loader_actually_applies_the_transpose,
        test_the_real_loader_transposes_on_the_way_in,
    ]
    for test in tests:
        print(f"{test.__name__}:")
        test()
    print(f"\n{len(tests)} test groups passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
