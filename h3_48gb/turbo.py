"""Turbo LoRA as a runtime side path over 4-bit weights.

The Turbo LoRA (`larryvrh/MiniMax-H3-Turbo-Lora`) trains MiniMax-H3 to sample in 4-8 steps instead
of 30. It arrives in two halves, and they must be applied in completely different ways:

* **The AdaLN half** (102 of 518 tensors) lives in the 2688-dim ``silu(t_emb)`` space that this
  build's pruned base folded away. It is baked into the modulation table instead — see
  ``scripts/bake_adaln.py``. That table is bf16, so folding costs nothing at run time.
* **The backbone half** (416 tensors: ``attn.qkv_proj``, ``attn.out_proj``, ``mlp.fc1``,
  ``mlp.fc2``) is what this module applies, and it **must not be merged into the weights**. Those
  weights are 4-bit: the update's rms is 3.37e-05 against a quantization step of 1.19e-02, so it is
  0.3% of one step and requantization would discard 99.7% of the training. It is added at run time
  as ``y = quantized(x) + strength * (x @ A.T) @ B.T`` — about 1.5% more work in the linear layers
  and none at all in attention, which is where the time actually goes.

**The QKV rows have to be permuted.** This fork's ``attn.qkv_proj`` is in the release's per-head
interleave (``[h0:q,k,v][h1:q,k,v]...``), restored by ``convert_sawfwair.py`` from the global slabs
mere.run had rewritten it into. The LoRA is in slabs. Measured rather than assumed: grouping
``lora_B``'s row norms by slab separates q/k/v where grouping by head does not — in block 49 the v
slab is half the magnitude of q and k (0.0051 against 0.0102 and 0.0108), and that structure
vanishes entirely under the per-head grouping (0.0086/0.0087/0.0088). Applying an unpermuted
``lora_B`` raises nothing; it just scrambles which head each correction lands on.

**...unless the projection has been split.** `h3_48gb.dit.split_fused_attention` carves the fused
layer into ``q_proj``/``k_proj``/``v_proj`` at load time, and those carved rows come out in slab
order — so against a split attention the same adapter is three contiguous *slices* of the
unpermuted ``lora_B``, with no permutation at all. Both spellings are supported and are checked
against each other bit-for-bit (`tests/test_attention_levers.py`), because the two are exactly the
kind of pair that silently diverges.
"""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

#: The four projections the backbone half touches, as they are named in both trees.
BACKBONE_TARGETS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")

#: Only this one needs its rows permuted; the other three are 1:1 between the two layouts.
PERMUTED_TARGET = "attn.qkv_proj"


def slabs_to_interleaved(num_heads: int, head_dim: int) -> mx.array:
    """Row index mapping ``[all-q; all-k; all-v]`` -> ``[h0:q,k,v][h1:q,k,v]...``.

    The same permutation `convert_sawfwair.py` applies to the checkpoint itself, expressed here
    over row indices so it can be applied to a LoRA factor.
    """
    index = mx.arange(3 * num_heads * head_dim).reshape(3, num_heads, head_dim)
    return index.transpose(1, 0, 2).reshape(-1)


class LoRALinear(nn.Module):
    """A quantized linear with a full-precision low-rank correction added to its output.

    Deliberately not a merge. See the module docstring for the arithmetic that rules merging out.
    """

    def __init__(self, base: nn.Module, lora_a: mx.array, lora_b: mx.array, strength: float = 1.0):
        super().__init__()
        self.base = base
        self.lora_a = lora_a
        self.lora_b = lora_b
        self.strength = strength

    def __call__(self, x: mx.array) -> mx.array:
        # `out + strength * (low @ B.T)` materializes a full output-sized delta before adding it:
        # 2.16 GB for `mlp.fc1` at 37,657 rows, live only long enough to be discarded. `addmm`
        # accumulates into `out` instead.
        return self.apply_rows(x, self.prepare_rows(x))

    # -- the row-chunking protocol `minimax_h3_mlx.dit.prepare_rows` defines ----------------------
    #
    # `x @ A.T` is the one narrow matmul in the model — rank 64 out of a 7168-wide input — and MLX
    # splits a narrow reduction differently at different row counts, so computing it per chunk
    # moves the result by a bf16 ULP against computing it whole. It is also the one thing cheap
    # enough to hoist: `(seq, 64)` is 9 MB where the buffers being chunked are gigabytes. So the
    # low-rank read happens once over the whole sequence and the chunks consume slices of it,
    # which is what keeps `q-chunk` and `mlp-chunk` bit-identical rather than merely close.

    def prepare_rows(self, x: mx.array) -> mx.array:
        return x @ self.lora_a.T

    def apply_rows(self, x: mx.array, low: mx.array) -> mx.array:
        out = self.base(x)
        return mx.addmm(out, low.astype(out.dtype), self.lora_b.T.astype(out.dtype),
                        alpha=self.strength)


def _resolve(root: nn.Module, path: str):
    """``blocks.0.attn.qkv_proj`` -> (owning module, attribute name)."""
    parts = path.split(".")
    node = root
    for part in parts[:-1]:
        node = node[int(part)] if part.isdigit() else getattr(node, part)
    return node, parts[-1]


#: The three sub-projections a split attention carries, in the order a slab-ordered `lora_B` is in.
SPLIT_PARTS = ("q_proj", "k_proj", "v_proj")


def _is_split(attention: nn.Module) -> bool:
    """Whether `Attention.split_projections` has carved this layer (see `h3_48gb.dit`)."""
    return getattr(attention, "is_split", False)


def wrap_qkv_lora(attention: nn.Module, a: mx.array, b: mx.array, strength: float,
                  permutation: mx.array, permute: bool = True) -> int:
    """Attach a fused-QKV LoRA to `attention`, split or not. Returns layers wrapped.

    The two layouts need opposite treatment and both are easy to get silently wrong:

    * **Fused.** ``lora_B`` is in the release's ``[all-q; all-k; all-v]`` slab order and the
      projection is in per-head interleave, so the rows must be permuted. Applying an unpermuted
      ``lora_B`` raises nothing — it just scrambles which head each correction lands on.
    * **Split.** ``q_proj``'s rows were gathered out of the interleave as ``[h0, h1, ...]``, which
      *is* the q slab. So the three parts are contiguous thirds of the unpermuted ``lora_B`` and
      the permutation must **not** be applied. Same numbers as the fused path, reached by slicing
      instead of by gathering: this is the simplification the split buys.

    ``lora_A`` is shared by all three — it is the same rank-r read of the same input — so the
    three parts differ only in ``lora_B``.

    ``permute=False`` is a fused-layout debugging switch (it is what proves, in the tests, that the
    permutation changes anything at all). It has no meaning against a split attention — there is no
    permutation being applied for it to turn off — so asking for it here raises rather than being
    quietly ignored, which would hand back a model that looks like the one that was asked for.
    """
    if not _is_split(attention):
        base = attention.qkv_proj
        attention.qkv_proj = LoRALinear(base, a, b[permutation] if permute else b, strength)
        return 1

    if not permute:
        raise ValueError(
            "`permute_qkv=False` is meaningless against a split attention: the carved q/k/v rows "
            "are already in the adapter's own slab order, so there is no permutation to skip. "
            "Load with `split_qkv=False` if the unpermuted fused layout is what you meant to test."
        )

    rows = b.shape[0] // 3
    for index, part in enumerate(SPLIT_PARTS):
        base = getattr(attention, part)
        setattr(attention, part,
                LoRALinear(base, a, b[index * rows:(index + 1) * rows], strength))
    return len(SPLIT_PARTS)


def apply_backbone_lora(dit, weights_path: Path | str, strength: float = 1.0,
                        num_heads: int = 56, head_dim: int = 128, verbose: bool = True,
                        permute_qkv: bool = True) -> dict:
    """Wrap every backbone projection of `dit` with its LoRA correction.

    Returns a report: how many layers were wrapped, how many were permuted, and the added bytes.
    Raises if the LoRA names a layer this model does not have, or if a shape disagrees — a LoRA
    silently applied to three quarters of the blocks would look like a weak LoRA, not like a bug.
    """
    lora = mx.load(str(weights_path))
    permutation = slabs_to_interleaved(num_heads, head_dim)

    wrapped = permuted = 0
    added_bytes = 0
    missing: list[str] = []

    # `token_refiner` is part of the backbone too — the author's node says so explicitly
    # ("attn/mlp/refiner"). Missing it leaves the text conditioning un-adapted while all 50
    # blocks downstream expect the adapted form, and nothing raises: the run just gets worse.
    paths = [f"blocks.{b}.{t}" for b in range(len(dit.blocks)) for t in BACKBONE_TARGETS]
    refiner = getattr(dit, "token_refiner", None)
    if refiner is not None:
        paths += [f"token_refiner.blocks.{b}.{t}"
                  for b in range(len(refiner.blocks)) for t in BACKBONE_TARGETS]

    for path in paths:
        target = path.rsplit(".", 2)[-2] + "." + path.rsplit(".", 1)[-1]
        key_a, key_b = f"{path}.lora_A.weight", f"{path}.lora_B.weight"
        if key_a not in lora or key_b not in lora:
            missing.append(path)
            continue

        a, b = lora[key_a], lora[key_b]
        if target == PERMUTED_TARGET:
            if b.shape[0] != 3 * num_heads * head_dim:
                raise ValueError(
                    f"{key_b} has {b.shape[0]} rows, expected 3 * {num_heads} * {head_dim} = "
                    f"{3 * num_heads * head_dim}; the QKV permutation would be wrong.")
            attention, _ = _resolve(dit, path)
            wrapped += wrap_qkv_lora(attention, a, b, strength, permutation,
                                     permute=permute_qkv)
            permuted += int(permute_qkv and not _is_split(attention))
            added_bytes += a.nbytes + b.nbytes
            continue

        owner, attribute = _resolve(dit, path)
        base = getattr(owner, attribute)
        setattr(owner, attribute, LoRALinear(base, a, b, strength))
        wrapped += 1
        added_bytes += a.nbytes + b.nbytes

    if missing:
        raise KeyError(
            f"the LoRA is missing {len(missing)} backbone projections this model has, e.g. "
            f"{missing[:3]} — a partial application looks like a weak LoRA, not like a failure.")

    # The check above is one-directional and that is how `token_refiner` was silently skipped for
    # a whole afternoon: every projection the loop visited was present in the LoRA, so nothing
    # complained, while eight of the LoRA's own targets were never applied. Check both ways.
    offered = {k.rsplit(".lora_", 1)[0] for k in lora}
    unapplied = sorted(offered - set(paths) - {n for n in offered if "adaln_proj" in n})
    if unapplied:
        raise KeyError(
            f"the LoRA carries {len(unapplied)} backbone targets this run never applied: "
            f"{unapplied[:4]} — a partial application is indistinguishable from a weak LoRA.")

    split = sum(1 for path in paths if path.endswith(PERMUTED_TARGET)
                and _is_split(_resolve(dit, path)[0]))
    report = {"wrapped": wrapped, "permuted": permuted, "split": split,
              "added_gb": added_bytes / 1e9, "strength": strength}
    if verbose:
        layout = f"{permuted} QKV permuted" if not split else f"{split} QKV split into q/k/v"
        print(f"  turbo: {wrapped} projections wrapped ({layout}), "
              f"+{report['added_gb']:.2f} GB, strength {strength}")
    return report


# -- LightX2V's adapter, which is shaped differently ---------------------------------------------

#: peft stores `alpha` separately from the rank and scales the update by `alpha / rank`, so this
#: number cannot be read off the checkpoint — it lives in the training config. At rank 128 that
#: makes the effective scale 0.0625.
#:
#: **8, from the authors' own inference script**, `DEFAULT_LORA_ALPHA` in
#: github.com/ModelTC/Minimax-H3-Turbo/blob/main/inference_minimax_h3.py, which computes
#: `effective_scale = lora_scale * lora_alpha / rank`. Kijai's ComfyUI conversion guesses 16 and
#: says so ("this is mostly a guess since I don't know how it's intended to be applied"). Taking
#: that guess drove the motion to 2.1x the reference here — exactly the factor two the wrong alpha
#: predicts, which is how the error was caught.
LIGHTX2V_ALPHA = 8.0

#: diffusers naming -> this fork's. The three attention projections have no single counterpart:
#: they are split there and fused here, handled by `SplitQKVLoRALinear` below.
LIGHTX2V_RENAMES = {
    "attn.to_out.0": "attn.out_proj",
    "ff.net.0.proj": "mlp.fc1",
    "ff.net.2": "mlp.fc2",
}


class SplitQKVLoRALinear(nn.Module):
    """A fused QKV projection carrying three separate low-rank corrections.

    LightX2V trains `to_q`, `to_k` and `to_v` as independent adapters, each with its own A. They
    cannot be concatenated into one pair — that would need a block-diagonal B and would inflate
    the rank from 128 to 384, which is the size blow-up Kijai reports when converting this adapter
    for a fused base.

    Computing the three separately and concatenating the *outputs* is the same arithmetic at a
    third of the memory. The concatenation is in slab order, so it is permuted into this fork's
    per-head interleave afterwards — the same reordering the fused adapter needs.
    """

    def __init__(self, base: nn.Module, factors, permutation: mx.array, scale: float,
                 num_heads: int = 56, head_dim: int = 128):
        super().__init__()
        self.base = base
        self.q_a, self.q_b = factors["q"]
        self.k_a, self.k_b = factors["k"]
        self.v_a, self.v_b = factors["v"]
        self.scale = scale
        self.num_heads = num_heads
        self.head_dim = head_dim

    def __call__(self, x: mx.array) -> mx.array:
        out = self.base(x)
        shape = (*x.shape[:-1], self.num_heads, self.head_dim)
        # Stacking on the head axis *is* the slab -> per-head interleave permutation, so the
        # gather this used to do is unnecessary: `[q|k|v]` reshaped per head and stacked at axis
        # -2 lands each head's q, k and v adjacent, which is the layout `dit.py` reads. Verified
        # elementwise against the explicit permutation. Saves a full 21,504-wide copy per block.
        delta = mx.stack([
            ((x @ self.q_a.T) @ self.q_b.T).reshape(shape),
            ((x @ self.k_a.T) @ self.k_b.T).reshape(shape),
            ((x @ self.v_a.T) @ self.v_b.T).reshape(shape),
        ], axis=-2).reshape(*x.shape[:-1], 3 * self.num_heads * self.head_dim)
        return out + self.scale * delta.astype(out.dtype)


def apply_lightx2v_lora(dit, weights_path: Path | str, strength: float = 1.0,
                        num_heads: int = 56, head_dim: int = 128, verbose: bool = True) -> dict:
    """Apply LightX2V's adapter, converting its layout on the way in.

    Three differences from `apply_backbone_lora`, all of them silent if missed: peft's
    `alpha / rank` scaling, diffusers key names, and split-vs-fused attention projections.
    """
    lora = mx.load(str(weights_path))
    targets = sorted({k.rsplit(".lora_", 1)[0] for k in lora})
    rank = lora[f"{targets[0]}.lora_A.default.weight"].shape[0]
    scale = strength * LIGHTX2V_ALPHA / rank
    permutation = slabs_to_interleaved(num_heads, head_dim)

    def factors(name):
        return (lora[f"{name}.lora_A.default.weight"], lora[f"{name}.lora_B.default.weight"])

    wrapped = 0
    for source_block, our_block in ([(f"transformer_blocks.{i}", f"blocks.{i}")
                                     for i in range(len(dit.blocks))] +
                                    [(f"token_refiner.refiner_blocks.{i}", f"token_refiner.blocks.{i}")
                                     for i in range(len(getattr(dit, "token_refiner").blocks))]):
        qkv = {part: factors(f"{source_block}.attn.to_{part}") for part in ("q", "k", "v")
               if f"{source_block}.attn.to_{part}.lora_A.default.weight" in lora}
        if len(qkv) == 3:
            attention, _ = _resolve(dit, f"{our_block}.attn.qkv_proj")
            if _is_split(attention):
                # Against a split attention this adapter needs no conversion at all: it was
                # trained as three separate `to_q`/`to_k`/`to_v` and the model now *has* three
                # separate projections in the same order, so each is a plain `LoRALinear`.
                # `SplitQKVLoRALinear` below exists only for the fused layout.
                for part, name in zip(("q", "k", "v"), SPLIT_PARTS):
                    a, b = qkv[part]
                    setattr(attention, name, LoRALinear(getattr(attention, name), a, b, scale))
            else:
                # `num_heads`/`head_dim` were accepted here and then not forwarded, so the
                # wrapper always used its 56x128 defaults. Invisible at the shipped size and
                # fatal at any other — the reshape simply cannot run.
                attention.qkv_proj = SplitQKVLoRALinear(
                    attention.qkv_proj, qkv, permutation, scale, num_heads, head_dim)
            wrapped += 1
        for source_suffix, our_suffix in LIGHTX2V_RENAMES.items():
            key = f"{source_block}.{source_suffix}"
            if f"{key}.lora_A.default.weight" not in lora:
                continue
            a, b = factors(key)
            owner, attribute = _resolve(dit, f"{our_block}.{our_suffix}")
            setattr(owner, attribute, LoRALinear(getattr(owner, attribute), a, b, scale))
            wrapped += 1

    report = {"wrapped": wrapped, "rank": rank, "scale": round(scale, 4), "targets": len(targets)}
    if verbose:
        print(f"  lightx2v: {wrapped} projections, rank {rank}, "
              f"scale {scale:.3f} (alpha {LIGHTX2V_ALPHA}/rank {rank} x strength {strength})")
    return report
