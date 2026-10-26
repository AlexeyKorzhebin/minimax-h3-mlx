#!/usr/bin/env python3
"""Prove the text-encoder quantization is selective and that a mismatched recipe is caught.

    ./.venv/bin/python test_text_encoder_quant.py

The bug being fixed is silent: the port builds plain `nn.Linear`, so mere.run's packed uint32
weight matches the expected ``...weight`` key and is written into a bf16 slot while ``scales`` and
``biases`` are counted as "skipped". Nothing raises. These tests pin the two properties that make
that impossible — the right layers and only the right layers get quantized, and every ``scales``
in the checkpoint must find a home.

No mlx_vlm and no checkpoint required: the recipe handling and the key audit are exercised on a
toy module tree.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.mlx
pytest.importorskip("mlx.core", reason="mlx: needs the MLX stack, absent here")


import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from h3_48gb import _upstream  # noqa: E402,F401

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from h3_48gb.text_encoder import (  # noqa: E402
    LANGUAGE_PREFIX,
    VISION_PREFIX,
    audit_keys,
    describe_dropped,
    quantize_selected,
    split_recipe,
)

GROUP, BITS, WIDTH = 64, 8, 128


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name} FAILED {detail}")
    print(f"  ok  {name}")


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_qkv = nn.Linear(WIDTH, 3 * WIDTH, bias=False)
        self.attn_proj = nn.Linear(WIDTH, WIDTH, bias=False)
        self.mlp_fc1 = nn.Linear(WIDTH, 2 * WIDTH, bias=False)
        # Left dense in the real checkpoint, like `visual.blocks.N.mlp.linear_fc2`.
        self.mlp_fc2 = nn.Linear(2 * WIDTH, WIDTH, bias=False)


class Tower(nn.Module):
    def __init__(self, depth: int = 2):
        super().__init__()
        self.blocks = [Block() for _ in range(depth)]
        self.norm = nn.RMSNorm(WIDTH)


def quantized_paths(module: nn.Module) -> set[str]:
    return {k.rsplit(".", 1)[0] for k, _ in tree_flatten(module.parameters())
            if k.endswith(".scales")}


def test_recipe_split() -> None:
    recipe = [
        f"{LANGUAGE_PREFIX}layers.0.self_attn.q_proj",
        f"{LANGUAGE_PREFIX}layers.0.mlp.down_proj",
        f"{VISION_PREFIX}blocks.3.attn.qkv",
        "something.else.entirely",
    ]
    buckets = split_recipe(recipe)
    check("language paths are made relative to the language sub-tree",
          buckets["language"] == {"layers.0.self_attn.q_proj", "layers.0.mlp.down_proj"},
          f"got {buckets['language']}")
    check("vision paths are made relative to the vision sub-tree",
          buckets["vision"] == {"blocks.3.attn.qkv"}, f"got {buckets['vision']}")
    check("paths outside both sub-trees are ignored",
          "something.else.entirely" not in buckets["language"] | buckets["vision"])


def test_only_listed_layers_are_quantized() -> None:
    """A blanket `nn.quantize` would be wrong; the recipe is the whole point."""
    tower = Tower()
    wanted = {"blocks.0.attn_qkv", "blocks.0.mlp_fc1", "blocks.1.attn_proj"}
    converted = quantize_selected(tower, wanted, BITS, GROUP)
    mx.eval(tower.parameters())

    check("every listed layer was converted", converted == len(wanted), f"got {converted}")
    check("exactly the listed layers carry scales",
          quantized_paths(tower) == wanted, f"got {quantized_paths(tower)}")
    check("the dense layer stays dense",
          isinstance(tower.blocks[0].mlp_fc2, nn.Linear)
          and not isinstance(tower.blocks[0].mlp_fc2, nn.QuantizedLinear))
    check("a quantized weight is packed uint32",
          tower.blocks[0].attn_qkv.weight.dtype == mx.uint32,
          f"got {tower.blocks[0].attn_qkv.weight.dtype}")
    check("an 8-bit packed row is a quarter as wide",
          tower.blocks[0].attn_qkv.weight.shape == (3 * WIDTH, WIDTH // 4),
          f"got {tower.blocks[0].attn_qkv.weight.shape}")
    check("scales carry one group per group_size inputs",
          tower.blocks[0].attn_qkv.scales.shape == (3 * WIDTH, WIDTH // GROUP),
          f"got {tower.blocks[0].attn_qkv.scales.shape}")


def test_empty_recipe_quantizes_nothing() -> None:
    tower = Tower()
    converted = quantize_selected(tower, set(), BITS, GROUP)
    mx.eval(tower.parameters())
    check("an empty recipe is a no-op", converted == 0 and quantized_paths(tower) == set())


def test_audit_catches_orphaned_scales() -> None:
    """The exact silent failure: the tree is not quantized, so `scales` has nowhere to go."""
    tower = Tower()
    expected = {"language": {k for k, _ in tree_flatten(tower.parameters())}, "vision": set()}
    checkpoint = list(expected["language"]) + [
        "blocks.0.attn_qkv.scales", "blocks.0.attn_qkv.biases", "lm_head.weight",
    ]

    def wanted(key):
        return None if key.startswith("lm_head") else ("language", key)

    kept, skipped, dropped = audit_keys(checkpoint, wanted, expected)
    check("the deliberately unused head is skipped, not dropped", skipped == 1, f"got {skipped}")
    check("the orphaned quantization tensors are dropped",
          sorted(dropped) == ["blocks.0.attn_qkv.biases", "blocks.0.attn_qkv.scales"],
          f"got {dropped}")
    message = describe_dropped(dropped)
    check("and the error names them as quantization tensors",
          message is not None and "quantization tensors" in message, f"got {message!r}")
    check("every real parameter is still kept",
          set(kept["language"]) == expected["language"])


def test_audit_is_clean_on_a_matching_recipe() -> None:
    tower = Tower()
    quantize_selected(tower, {"blocks.0.attn_qkv", "blocks.1.mlp_fc1"}, BITS, GROUP)
    mx.eval(tower.parameters())
    expected = {"language": {k for k, _ in tree_flatten(tower.parameters())}, "vision": set()}

    kept, skipped, dropped = audit_keys(expected["language"], lambda k: ("language", k), expected)
    check("a matching recipe drops nothing", dropped == [] and skipped == 0, f"{dropped} {skipped}")
    check("describe_dropped stays quiet", describe_dropped(dropped) is None)
    check("scales and biases are among the kept keys",
          any(k.endswith(".scales") for k in kept["language"])
          and any(k.endswith(".biases") for k in kept["language"]))


def test_converted_recipe_is_selective() -> None:
    """Against the real converted checkpoint, if it is on disk."""
    import os

    root = Path(os.environ.get("H3_CHECKPOINT", Path.home() / "models/h3-converted"))
    recipe_path = root / "text_encoder" / "quant_config.json"
    if not recipe_path.exists():
        print("  --  no converted text_encoder/quant_config.json, skipping")
        return
    with open(recipe_path) as fh:
        recipe = json.load(fh)
    buckets = split_recipe(recipe["quantized_modules"])
    dense = set(recipe.get("dense_linear_modules", []))

    check("the recipe is 8-bit, group 64",
          (recipe["bits"], recipe["group_size"]) == (8, 64),
          f"got {recipe['bits']}/{recipe['group_size']}")
    check("both sub-trees are covered",
          bool(buckets["language"]) and bool(buckets["vision"]),
          f"{len(buckets['language'])} language, {len(buckets['vision'])} vision")
    check("the recipe is not a blanket rule: some linears stay dense", bool(dense), f"{dense}")
    overlap = dense & set(recipe["quantized_modules"])
    check("no module is listed as both quantized and dense", not overlap, f"{overlap}")
    fc2 = [m for m in dense if m.endswith("mlp.linear_fc2")]
    check("the vision mlp.linear_fc2 layers are the dense ones",
          len(fc2) >= 1, f"dense modules: {sorted(dense)[:6]}")


def main() -> int:
    tests = [
        test_recipe_split,
        test_only_listed_layers_are_quantized,
        test_empty_recipe_quantizes_nothing,
        test_audit_catches_orphaned_scales,
        test_audit_is_clean_on_a_matching_recipe,
        test_converted_recipe_is_selective,
    ]
    for test in tests:
        print(f"{test.__name__}:")
        test()
    print(f"\n{len(tests)} test groups passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
