"""The DiT loader's contract about the AdaLN modulation path.

The loader deletes that path from the module tree before anything can evaluate it — 13B parameters
this build never allocates — and serves the modulation from `adaln_cache.safetensors` instead. Two
kinds of checkpoint then exist in the wild:

* mere.run's, which omits those tensors, which is where the design came from;
* pipenetwork's, which ships them, and would otherwise be rejected as 206 unexpected keys.

Both must load, and a genuinely stray tensor must still fail. These tests build a doll-sized DiT
rather than the real 20B one; the loader does not care how large the tree is, only which keys it
holds.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.mlx
pytest.importorskip("mlx.core", reason="mlx: needs the MLX stack, absent here")


import json

import mlx.core as mx
import numpy as np
import pytest

from h3_48gb import _upstream  # noqa: F401
from h3_48gb.dit import attention_levers_patch_applied, load_dit_cached, strip_modulation_path

from minimax_h3_mlx.config import DiTConfig
from minimax_h3_mlx.dit import MiniMaxH3DiT
from mlx.utils import tree_flatten

TINY = dict(hidden_size=64, num_layers=2, token_refiner_num_layers=1, num_attention_heads=2,
            attention_head_dim=32, ffn_hidden_size=128, text_dim=48, time_embed_hidden_size=64,
            time_embed_dim=32, adaln_out_features=1152, final_adaln_out_features=128)

#: `load_dit_cached` splits the fused QKV by default, which needs the levers patch. On a checkout
#: that never applied it these tests would fail deep inside the loader, and the failure would read
#: like a broken loader rather than like a missing setup step. Skip with the command instead.
needs_levers_patch = pytest.mark.skipif(
    not attention_levers_patch_applied(),
    reason=("upstream/ is missing patches/0002-attention-memory-levers.patch, which "
            "`load_dit_cached` needs for its default `split_qkv=True`. Apply it with:  "
            "git -C upstream apply ../patches/0002-attention-memory-levers.patch"),
)


def _write_checkpoint(tmp_path, extra: dict[str, np.ndarray] | None = None):
    """A directory the loader will accept, plus whatever `extra` tensors the test wants in it."""
    (tmp_path / "config.json").write_text(json.dumps(TINY))
    model = MiniMaxH3DiT(DiTConfig(**TINY))
    strip_modulation_path(model)
    tensors = {key: np.zeros(value.shape, np.float32)
               for key, value in tree_flatten(model.parameters())}
    tensors.update(extra or {})
    mx.save_safetensors(str(tmp_path / "model.safetensors"),
                        {k: mx.array(v) for k, v in tensors.items()})
    return tmp_path


def _modulation_keys(tmp_path) -> list[str]:
    """Exactly the keys `strip_modulation_path` removes, for this config."""
    (tmp_path / "config.json").write_text(json.dumps(TINY))
    return strip_modulation_path(MiniMaxH3DiT(DiTConfig(**TINY)))


@needs_levers_patch
def test_a_build_without_the_modulation_path_loads(tmp_path):
    """mere.run's shape: the tensors are simply absent."""
    assert load_dit_cached(_write_checkpoint(tmp_path)) is not None


def test_a_build_loads_unsplit_when_the_carve_is_declined(tmp_path):
    """The escape hatch, and the only path through the loader that needs no `upstream/` patch.

    Deliberately *not* marked with `needs_levers_patch`: it is what an unpatched checkout is told
    to use, so it is exactly the case that must keep working there.
    """
    model = load_dit_cached(_write_checkpoint(tmp_path), split_qkv=False)
    assert model is not None
    assert all(hasattr(block.attn, "qkv_proj") for block in model.blocks)
    assert not any(hasattr(block.attn, "q_proj") for block in model.blocks)


@needs_levers_patch
def test_a_loaded_build_has_its_qkv_carved_by_default(tmp_path):
    """The default is the split one — every consumer downstream is written against that."""
    model = load_dit_cached(_write_checkpoint(tmp_path))
    attentions = [b.attn for b in model.blocks] + [b.attn for b in model.token_refiner.blocks]
    assert attentions and all(attn.is_split for attn in attentions)
    assert not any(hasattr(attn, "qkv_proj") for attn in attentions)


@needs_levers_patch
def test_a_build_that_ships_the_modulation_path_also_loads(tmp_path):
    """pipenetwork's shape. Before this, 206 real tensors read as 206 unexpected keys.

    They are not merely tolerated — they must be *skipped*, because allocating them is the 13B the
    whole cached-table design exists to avoid.
    """
    keys = _modulation_keys(tmp_path)
    assert keys, "the fixture is pointless if the config strips nothing"
    shipped = {k: np.zeros((2, 2), np.float32) for k in keys}

    model = load_dit_cached(_write_checkpoint(tmp_path, extra=shipped))

    live = {key for key, _ in tree_flatten(model.parameters())}
    assert not (live & set(keys)), "the modulation path was allocated after all"


def test_a_stray_tensor_still_fails(tmp_path):
    """The permissiveness above is scoped to the modulation path and nothing else."""
    with pytest.raises(KeyError, match="unexpected"):
        load_dit_cached(_write_checkpoint(
            tmp_path, extra={"blocks.0.attn.nonsense": np.zeros((2, 2), np.float32)}))


def test_a_missing_tensor_still_fails(tmp_path):
    """And the check has teeth in the other direction too."""
    _write_checkpoint(tmp_path)
    loaded = dict(mx.load(str(tmp_path / "model.safetensors")))
    # `mx.load` is lazy mmap-views into the file; saving over the SAME path below would rewrite
    # the bytes those views still point at. That poisons the process, and the abort strikes much
    # later, in whatever unrelated test materializes an `mx.load` next (found 2026-08-26: full
    # suite died with `Fatal Python error: Aborted` inside test_latent_tail's round-trip; this
    # pair reproduces it two tests long). Materialize before overwriting.
    mx.eval(*loaded.values())
    victim = next(k for k in loaded if k.startswith("blocks.0."))
    del loaded[victim]
    mx.save_safetensors(str(tmp_path / "model.safetensors"), loaded)

    with pytest.raises(KeyError, match="missing"):
        load_dit_cached(tmp_path)
