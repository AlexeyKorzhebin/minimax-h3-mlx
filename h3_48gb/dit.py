"""Load the mere.run DiT, which ships without its AdaLN modulation path.

`minimax_h3_mlx.load.load_dit` cannot be used as-is for two independent reasons:

* it is strict, and this checkpoint is missing 106 tensors on purpose;
* it ends with ``mx.eval(model.parameters())``, which would **materialize** the randomly
  initialized ``adaln_proj`` projections it could not fill — 13B parameters, ~26 GB of garbage.

MLX builds parameters lazily, so `MiniMaxH3DiT(config)` costs nothing until something evaluates it.
This loader exploits that: the absent modules are removed from the tree *before* any evaluation,
so the 13B is never allocated at all, and what is left is loaded strictly against the reduced key
set — a genuinely missing weight still raises.

What replaces them:

* ``time_embedder`` -> a parameter-free stub. ``temb`` is computed at the top of the forward pass
  and read only by ``final_layer.norm_out``, which is redirected below, so its value is never used.
* ``blocks.N.adaln_proj`` -> a stub that raises. The stock forward reads it only when no modulation
  cache is passed; that path cannot work here and must fail loudly rather than run on noise.
* ``final_layer.adaln_proj`` -> removed, and ``FinalLayer.norm_out`` is redirected to the cached
  shift/scale by rebinding the instance's class. Nothing in the upstream forward is copied.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from . import _upstream  # noqa: F401

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from minimax_h3_mlx.config import DiTConfig
from minimax_h3_mlx.dit import Attention, FinalLayer, MiniMaxH3DiT
from minimax_h3_mlx.load import SKIP_KEYS, is_fp32_key, shard_paths

#: Module names the mere.run build folds into `adaln_cache.safetensors`.
CACHE_COVERED = ("time_embedder", "adaln_proj")


def attention_levers_patch_applied() -> bool:
    """Whether the vendored `upstream/` carries `patches/0002-attention-memory-levers.patch`.

    Checked by name rather than by reading source: the carve is a *method* the patch adds, so its
    absence is unambiguous, and a check that cannot be satisfied by a comment cannot be fooled by
    one either.
    """
    return hasattr(Attention, "split_projections")


def split_fused_attention(model: MiniMaxH3DiT, verbose: bool = False) -> int:
    """Carve every attention's fused ``qkv_proj`` into ``q_proj`` / ``k_proj`` / ``v_proj``.

    Done once, here, rather than per forward: the gather touches the *weights*, so paying for it
    at load time costs one pass over 6 GB and nothing afterwards. The saving is on the activation
    side — see `Attention.split_projections` for why the fused staging is the expensive part —
    and it is worth ~3 GB of peak on a long native forward.

    Resident weights do not grow. The carve allocates the three parts and then drops the fused
    projection, so the transient overshoot is one layer's worth (~150 MB at the shipped size),
    not the whole stack's.

    Returns the number of attention layers split.
    """
    if not attention_levers_patch_applied():
        raise RuntimeError(
            "`upstream/minimax_h3_mlx/dit.py` has no `Attention.split_projections`, so "
            "patches/0002-attention-memory-levers.patch is not applied to this checkout. Apply it "
            "(`git -C upstream apply ../patches/0002-attention-memory-levers.patch`, see README) "
            "or load with `split_qkv=False` — unsplit is correct, just ~3 GB heavier at the peak."
        )

    attentions = [block.attn for block in model.blocks]
    refiner = getattr(model, "token_refiner", None)
    if refiner is not None:
        attentions += [block.attn for block in refiner.blocks]

    for attention in attentions:
        attention.split_projections()
    from . import memory

    memory.release()
    if verbose:
        print(f"  fused QKV split into q/k/v on {len(attentions)} attention layers")
    return len(attentions)


class _NoTimestepEmbedder(nn.Module):
    """Stands in for the omitted timestep MLP.

    Returns a correctly shaped zero so the stock forward pass keeps type-checking; the value is
    inert because the only consumer, ``final_layer.norm_out``, reads the cached table instead.
    """

    def __init__(self, time_embed_dim: int):
        super().__init__()
        self._dim = int(time_embed_dim)

    def __call__(self, sinusoid: mx.array) -> mx.array:
        return mx.zeros((sinusoid.shape[0], self._dim), dtype=mx.float32)


class _AbsentModulation(nn.Module):
    """Stands in for an omitted ``adaln_proj``; raises rather than returning noise."""

    def __init__(self, where: str):
        super().__init__()
        self._where = where

    def __call__(self, *_args, **_kwargs):
        raise RuntimeError(
            f"{self._where} was not shipped by this build — mere.run precomputed the AdaLN "
            "modulation into adaln_cache.safetensors and dropped the projections. Pass a "
            "`modulation_cache` (see h3_48gb.adaln.AdaLNCacheFile); the uncached path cannot run."
        )


class CachedFinalLayer(FinalLayer):
    """`FinalLayer` whose modulation comes from the cache instead of ``adaln_proj``.

    Installed by rebinding an existing instance's ``__class__``: no new fields, no re-initialization,
    and the loaded ``norm`` / ``video_out`` / ``audio_out`` stay exactly as they were. The cached
    tensors are stored under leading-underscore names, which MLX's ``valid_parameter_filter``
    excludes from ``parameters()``, so they never reach ``update()`` or a strict key diff.
    """

    def set_modulation(self, shift: mx.array, scale: mx.array) -> None:
        self._mod_shift = shift
        self._mod_scale = scale

    def norm_out(self, x: mx.array, temb: mx.array, timestep_indices: mx.array) -> mx.array:
        shift = getattr(self, "_mod_shift", None)
        if shift is None:
            raise RuntimeError(
                "The output layer's modulation table has not been installed. Build it with "
                "AdaLNCacheFile.build(...) and call `set_modulation` before the first forward."
            )
        scale = self._mod_scale
        x = self.norm(x)
        return x * (1.0 + scale[timestep_indices]) + shift[timestep_indices]


def strip_modulation_path(model: MiniMaxH3DiT) -> list[str]:
    """Remove the omitted modules *before* anything evaluates their lazy initializers.

    Returns the parameter keys that disappeared, for the caller to report.
    """
    dropped: list[str] = []

    def keys_of(prefix: str, module: nn.Module) -> list[str]:
        return [f"{prefix}.{k}" for k, _ in tree_flatten(module.parameters())]

    dropped += keys_of("time_embedder", model.time_embedder)
    model.pop("time_embedder", None)
    model.time_embedder = _NoTimestepEmbedder(model.config.time_embed_dim)

    for index, block in enumerate(model.blocks):
        dropped += keys_of(f"blocks.{index}.adaln_proj", block.adaln_proj)
        block.pop("adaln_proj", None)
        block.adaln_proj = _AbsentModulation(f"blocks.{index}.adaln_proj")

    dropped += keys_of("final_layer.adaln_proj", model.final_layer.adaln_proj)
    model.final_layer.pop("adaln_proj", None)
    model.final_layer.__class__ = CachedFinalLayer

    return sorted(dropped)


def load_dit_cached(
    model_dir: str | Path,
    dtype: mx.Dtype | None = None,
    verbose: bool = False,
    split_qkv: bool = True,
) -> MiniMaxH3DiT:
    """Load a converted mere.run transformer directory.

    Mirrors `minimax_h3_mlx.load.load_dit` — same config file, same ``quant_config.json`` replay,
    same mixed float32/bfloat16 handling — but against the reduced module tree.
    """
    model_dir = Path(model_dir)
    config = DiTConfig.from_json(model_dir / "config.json")
    model = MiniMaxH3DiT(config)

    quant_path = model_dir / "quant_config.json"
    if quant_path.exists():
        from minimax_h3_mlx.quantize import QuantConfig, apply_quantization_structure

        with open(quant_path) as fh:
            recipe = json.load(fh)
        apply_quantization_structure(
            model,
            QuantConfig(
                bits=recipe["bits"],
                group_size=recipe["group_size"],
                quantize_adaln=recipe.get("quantize_adaln", False),
                adaln_bits=recipe.get("adaln_bits") or 8,
            ),
        )
        if verbose:
            print(f"  quantized structure: {recipe['bits']}-bit, group {recipe['group_size']}")

    dropped = strip_modulation_path(model)
    if verbose:
        print(f"  modulation path not in this build: {len(dropped)} tensors never allocated")

    # Some builds *do* ship the modulation path — pipenetwork's, for one, which carries all 206
    # `adaln_proj` tensors the mere.run monolith omitted. Those keys are not a mismatch to report:
    # they are the 13B this loader deliberately refuses to allocate, and the cached table stands in
    # for them. Skipping them here is what lets an 8-bit build load into 21.5 GB instead of 35.3.
    # They are kept distinct from `unexpected` so a genuinely stray tensor still fails loudly.
    replaced_by_cache = set(dropped)
    expected = {key for key, _ in tree_flatten(model.parameters())}
    weights: dict[str, mx.array] = {}
    unexpected: list[str] = []
    ignored = 0

    for shard in shard_paths(model_dir):
        started = time.perf_counter()
        loaded = mx.load(str(shard))
        for key, tensor in loaded.items():
            if key in SKIP_KEYS:
                continue
            if key not in expected:
                if key in replaced_by_cache:
                    ignored += 1
                else:
                    unexpected.append(key)
                continue
            if dtype is not None and tensor.dtype not in (mx.uint32, mx.int32):
                # Never cast a quantized layer's packed storage: it is uint32 bit-packing, not a
                # number, and `astype` would convert it arithmetically.
                with mx.stream(mx.cpu):
                    tensor = tensor.astype(dtype)
                    mx.eval(tensor)
            elif is_fp32_key(key) and tensor.dtype != mx.float32:
                tensor = tensor.astype(mx.float32)
            weights[key] = tensor
        if verbose:
            gb = sum(t.nbytes for t in loaded.values()) / 1e9
            print(f"  {shard.name}: {len(loaded)} tensors, {gb:.2f} GB, "
                  f"{time.perf_counter() - started:.1f}s")

    if verbose and ignored:
        print(f"  this build ships the modulation path: {ignored} tensors skipped, "
              "the cached table is used instead")

    missing = sorted(expected - weights.keys())
    if missing or unexpected:
        raise KeyError(
            f"Checkpoint/module mismatch: {len(missing)} missing (e.g. {missing[:4]}), "
            f"{len(unexpected)} unexpected (e.g. {unexpected[:4]}). The AdaLN modulation path is "
            "expected to be absent and is already excluded; anything listed here is a real gap."
        )

    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())

    # After the weights are real: the carve gathers rows out of them, so it cannot run against the
    # lazy tree, and running it here means every consumer — pipeline, LoRA, checkpoint resume —
    # sees the same split model without having to know the option exists.
    if split_qkv:
        split_fused_attention(model, verbose=verbose)
    return model
