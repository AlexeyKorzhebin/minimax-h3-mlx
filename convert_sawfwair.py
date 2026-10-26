#!/usr/bin/env python3
"""Convert the mere.run/Sawfwair MiniMax-H3 build into the layout the MLX port reads.

The mere.run artifact ships five monolithic safetensors plus one shared ``config.json``. The port
(`PipeNetwork/minimax-h3-mlx`) reads the *upstream* release layout: one directory per component,
each with its own config files, and — for a quantized DiT — a ``quant_config.json`` recording the
recipe so the module tree can be rebuilt with `QuantizedLinear` layers before the weights land.

Three things actually change; everything else is a byte copy.

1. **QKV layout.** mere.run rewrote every ``attn.qkv_proj`` from the release's per-head interleave
   into three global slabs (``[all-q; all-k; all-v]``); its own receipt says so
   (``"qkv_layout": "global-qkv-slabs"``, 52 matrices) and the scale statistics confirm it — grouping
   the per-row mean |scale| by slab separates q/k/v ~100x more sharply than grouping by head. The
   port consumes the projection as ``reshape(B, S, heads, 3, head_dim)`` (``dit.py`` line 152), so
   the rows have to go back. Quantization groups run along **axis 1** (``scales`` is
   ``[21504, 84]`` = 5376/64 groups per row), so a row carries its own scales and biases with it and
   the permutation is exact — no dequantize/requantize, no precision loss.

2. **Key renames in the text encoder.** mere.run stores ``model.layers.*`` / ``visual.*``; the port's
   `MiniMaxH3TextEncoder._wanted` matches ``model.language_model.*`` / ``model.visual.*``. It also
   expects ``norm.weight`` to exist (it loads it purely to keep the module tree complete and never
   applies it — H3 conditions on the *unnormalized* hidden state); mere.run dropped it, so a
   ones-vector is synthesized in its place.

3. **Config synthesis.** mere.run flattened every per-component config into one file and dropped the
   rest. Each component config is rebuilt here from the shared ``config.json`` plus the port's own
   dataclass defaults, and every field that *can* be cross-checked against a tensor shape is checked
   (``--check`` additionally diffs the written key sets against the port's module trees).

Known gaps this script cannot paper over — see ``conversion_report.json`` after a run:

* The transformer monolith **omits the whole AdaLN modulation path** (50 x ``blocks.N.adaln_proj``,
  ``final_layer.adaln_proj`` and ``time_embedder`` — 106 tensors), because mere.run precomputed it
  into ``adaln_cache.safetensors`` for a fixed 30-step schedule. ``load_dit(strict=True)`` will
  therefore fail, and `ModulationCache.build` / `final_layer_modulation` cannot run. The cache is
  copied through to ``transformer/adaln_cache.safetensors`` so the pipeline patch can consume it.
* The port's text encoder loader has **no quantization support** at all: it builds plain
  `nn.Linear` layers, so mere.run's 8-bit weights would be loaded as packed uint32 into a bf16 slot
  and the ``scales``/``biases`` silently dropped. The exact recipe is written to
  ``text_encoder/quant_config.json`` for a patched loader.

Usage::

    ./.venv/bin/python convert_sawfwair.py --out ~/models/MiniMax-H3-MLX-mere
    ./.venv/bin/python convert_sawfwair.py --out ... --components transformer --check

The script is idempotent and resumable at shard granularity: a shard whose header already matches
the plan is left alone, so an interrupted run can simply be re-issued.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

DEFAULT_SOURCE = Path.home() / "Library/Application Support/MereRun/models/video-minimax-h3-fl2va-mlx"
DEFAULT_PORT = Path(__file__).resolve().parent / "upstream"

SHARD_BYTES = 2 * 1024**3  # keep peak RSS low; the port reads any shard count via the index
COPY_CHUNK = 64 * 1024**2

DTYPE_SIZE = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
    "U32": 4, "I32": 4, "F32": 4,
    "U64": 8, "I64": 8, "F64": 8,
}
# Raw numpy view for a safetensors dtype. bfloat16 has no numpy equivalent, but the permutation is
# a row gather, which is dtype-agnostic — uint16 is a faithful bit container.
DTYPE_VIEW = {
    "BOOL": "|b1", "U8": "|u1", "I8": "|i1", "F8_E4M3": "|u1", "F8_E5M2": "|u1",
    "U16": "<u2", "I16": "<i2", "F16": "<f2", "BF16": "<u2",
    "U32": "<u4", "I32": "<i4", "F32": "<f4",
    "U64": "<u8", "I64": "<i8", "F64": "<f8",
}


# --------------------------------------------------------------------------------------------
# QKV row permutation
# --------------------------------------------------------------------------------------------

def slabs_to_interleaved_index(num_heads: int, head_dim: int) -> np.ndarray:
    """Row gather taking ``[all-q; all-k; all-v]`` to per-head ``[h0: q,k,v][h1: q,k,v]...``.

    Source row for component ``c`` (0=q, 1=k, 2=v), head ``h``, channel ``d`` sits at
    ``c * heads * head_dim + h * head_dim + d``; the destination wants it at
    ``h * 3 * head_dim + c * head_dim + d``. Returned array ``idx`` is the gather form:
    ``dst = src[idx]``.
    """
    total = 3 * num_heads * head_dim
    return (
        np.arange(total, dtype=np.int64)
        .reshape(3, num_heads, head_dim)
        .transpose(1, 0, 2)
        .reshape(total)
    )


def interleaved_to_slabs_index(num_heads: int, head_dim: int) -> np.ndarray:
    """Inverse of :func:`slabs_to_interleaved_index`."""
    total = 3 * num_heads * head_dim
    return (
        np.arange(total, dtype=np.int64)
        .reshape(num_heads, 3, head_dim)
        .transpose(1, 0, 2)
        .reshape(total)
    )


def permute_qkv_rows(array: np.ndarray, num_heads: int, head_dim: int) -> np.ndarray:
    """Apply the slab -> per-head-interleave permutation along axis 0.

    Works uniformly on ``weight`` (packed uint32), ``scales`` and ``biases``: all three are indexed
    by output row on axis 0, and the quantization groups run along axis 1, so a row travels with its
    own scale and bias.
    """
    expected = 3 * num_heads * head_dim
    if array.shape[0] != expected:
        raise ValueError(f"QKV axis 0 is {array.shape[0]}, expected 3 * {num_heads} * {head_dim} = {expected}")
    return array[slabs_to_interleaved_index(num_heads, head_dim)]


# --------------------------------------------------------------------------------------------
# Streaming safetensors I/O
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Entry:
    """One tensor as it exists in a source file."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int  # absolute byte offset in the file
    nbytes: int


def read_header(path: Path) -> tuple[dict[str, str], dict[str, Entry]]:
    """Parse a safetensors header without touching the tensor data."""
    with open(path, "rb") as fh:
        (length,) = struct.unpack("<Q", fh.read(8))
        raw = json.loads(fh.read(length))
    base = 8 + length
    meta = raw.pop("__metadata__", {}) or {}
    entries = {}
    for name, item in raw.items():
        start, end = item["data_offsets"]
        entries[name] = Entry(name, item["dtype"], tuple(item["shape"]), base + start, end - start)
    return meta, entries


def read_tensor(path: Path, entry: Entry) -> np.ndarray:
    """Read one tensor into a numpy array with a bit-faithful view dtype."""
    with open(path, "rb") as fh:
        fh.seek(entry.offset)
        raw = fh.read(entry.nbytes)
    if len(raw) != entry.nbytes:
        raise IOError(f"short read for {entry.name} in {path}")
    return np.frombuffer(raw, dtype=np.dtype(DTYPE_VIEW[entry.dtype])).reshape(entry.shape)


@dataclass
class OutTensor:
    """One tensor to write, sourced either by byte copy or from memory."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    src: tuple[Path, int] | None = None          # (file, absolute offset) for a straight copy
    producer: Callable[[], np.ndarray] | None = None  # materialize on demand instead

    @classmethod
    def copy_of(cls, entry: Entry, path: Path, name: str | None = None) -> "OutTensor":
        return cls(name or entry.name, entry.dtype, entry.shape, entry.nbytes, src=(path, entry.offset))

    @classmethod
    def computed(cls, name: str, dtype: str, shape: tuple[int, ...],
                 producer: Callable[[], np.ndarray]) -> "OutTensor":
        nbytes = DTYPE_SIZE[dtype] * math.prod(shape)
        return cls(name, dtype, tuple(shape), nbytes, producer=producer)

    def write_to(self, out) -> None:
        if self.producer is not None:
            array = np.ascontiguousarray(self.producer())
            if array.nbytes != self.nbytes:
                raise ValueError(f"{self.name}: produced {array.nbytes} bytes, planned {self.nbytes}")
            out.write(array.tobytes())
            return
        path, offset = self.src
        with open(path, "rb") as fh:
            fh.seek(offset)
            remaining = self.nbytes
            while remaining:
                chunk = fh.read(min(COPY_CHUNK, remaining))
                if not chunk:
                    raise IOError(f"short read for {self.name} in {path}")
                out.write(chunk)
                remaining -= len(chunk)


def _header_bytes(tensors: list[OutTensor], metadata: dict[str, str]) -> bytes:
    header: dict[str, object] = {}
    if metadata:
        header["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    cursor = 0
    for t in tensors:
        header[t.name] = {"dtype": t.dtype, "shape": list(t.shape),
                          "data_offsets": [cursor, cursor + t.nbytes]}
        cursor += t.nbytes
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    blob += b" " * ((8 - len(blob) % 8) % 8)  # keep the data section 8-byte aligned
    return struct.pack("<Q", len(blob)) + blob


def shard_matches(path: Path, tensors: list[OutTensor]) -> bool:
    """True when an existing file already holds exactly this planned tensor set."""
    if not path.exists():
        return False
    try:
        _, entries = read_header(path)
    except Exception:
        return False
    if len(entries) != len(tensors):
        return False
    for t in tensors:
        got = entries.get(t.name)
        if got is None or got.dtype != t.dtype or got.shape != tuple(t.shape) or got.nbytes != t.nbytes:
            return False
    expected_size = _len_header(path) + sum(t.nbytes for t in tensors)
    return path.stat().st_size == expected_size


def _len_header(path: Path) -> int:
    with open(path, "rb") as fh:
        (length,) = struct.unpack("<Q", fh.read(8))
    return 8 + length


def write_safetensors(path: Path, tensors: list[OutTensor], metadata: dict[str, str],
                      dry_run: bool = False) -> None:
    """Write one safetensors file, streaming tensor by tensor."""
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    with open(tmp, "wb") as out:
        out.write(_header_bytes(tensors, metadata))
        for t in tensors:
            t.write_to(out)
        out.flush()
        os.fsync(out.fileno())
    tmp.replace(path)


def plan_shards(tensors: list[OutTensor], cap: int) -> list[list[OutTensor]]:
    """Greedy, deterministic packing over sorted names — so a resumed run rebuilds the same plan."""
    ordered = sorted(tensors, key=lambda t: t.name)
    shards: list[list[OutTensor]] = [[]]
    sizes = [0]
    for t in ordered:
        if sizes[-1] and sizes[-1] + t.nbytes > cap:
            shards.append([])
            sizes.append(0)
        shards[-1].append(t)
        sizes[-1] += t.nbytes
    return shards


def write_sharded(out_dir: Path, tensors: list[OutTensor], metadata: dict[str, str],
                  cap: int, dry_run: bool, verbose: bool = True) -> dict:
    """Write a sharded component plus its ``model.safetensors.index.json``."""
    shards = plan_shards(tensors, cap)
    total = len(shards)
    weight_map: dict[str, str] = {}
    written = skipped = 0
    for index, shard in enumerate(shards, start=1):
        name = f"model-{index:05d}-of-{total:05d}.safetensors"
        path = out_dir / name
        for t in shard:
            weight_map[t.name] = name
        if shard_matches(path, shard):
            skipped += 1
        else:
            started = time.perf_counter()
            write_safetensors(path, shard, metadata, dry_run)
            written += 1
            if verbose and not dry_run:
                gb = sum(t.nbytes for t in shard) / 1e9
                print(f"    {name}: {len(shard)} tensors, {gb:.2f} GB, "
                      f"{time.perf_counter() - started:.1f}s", flush=True)
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "model.safetensors.index.json", "w") as fh:
            json.dump({"metadata": {"total_size": sum(t.nbytes for t in tensors), **metadata},
                       "weight_map": weight_map}, fh, indent=2)
    return {"shards": total, "written": written, "skipped": skipped,
            "tensors": len(tensors), "bytes": sum(t.nbytes for t in tensors)}


def write_single(path: Path, tensors: list[OutTensor], metadata: dict[str, str],
                 dry_run: bool, verbose: bool = True) -> dict:
    """Write a component the port loads as one file (both VAEs)."""
    ordered = sorted(tensors, key=lambda t: t.name)
    if shard_matches(path, ordered):
        return {"written": 0, "skipped": 1, "tensors": len(ordered),
                "bytes": sum(t.nbytes for t in ordered)}
    started = time.perf_counter()
    write_safetensors(path, ordered, metadata, dry_run)
    if verbose and not dry_run:
        gb = sum(t.nbytes for t in ordered) / 1e9
        print(f"    {path.name}: {len(ordered)} tensors, {gb:.2f} GB, "
              f"{time.perf_counter() - started:.1f}s", flush=True)
    return {"written": 1, "skipped": 0, "tensors": len(ordered),
            "bytes": sum(t.nbytes for t in ordered)}


def write_json(path: Path, payload: object, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def link_or_copy(src: Path, dst: Path, dry_run: bool) -> None:
    """Hardlink when possible (the 7 MB tokenizer lands in three places), else copy."""
    if dry_run or not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def to_float_list(array: np.ndarray, dtype: str) -> list[float]:
    """Decode a stats vector to python floats (``latents_mean`` / ``latents_std``)."""
    if dtype == "BF16":
        return (array.astype(np.uint32) << 16).view(np.float32).astype(np.float64).tolist()
    return array.astype(np.float64).tolist()


def expect(condition: bool, message: str, problems: list[str]) -> None:
    if not condition:
        problems.append(message)


# --------------------------------------------------------------------------------------------
# transformer
# --------------------------------------------------------------------------------------------

QKV_SUFFIXES = (".attn.qkv_proj.weight", ".attn.qkv_proj.scales", ".attn.qkv_proj.biases")

# Tensors the port's DiT holds but the mere.run monolith omits, because they were folded into
# `adaln_cache.safetensors`. Reported, never fabricated.
ADALN_PATH_MARKERS = ("adaln_proj", "time_embedder")


def convert_transformer(source: Path, out_root: Path, shared: dict, cap: int,
                        dry_run: bool) -> dict:
    src = source / "transformer.safetensors"
    out_dir = out_root / "transformer"
    meta, entries = read_header(src)
    problems: list[str] = []

    tcfg = shared["transformer"]
    heads = int(tcfg["num_attention_heads"])
    head_dim = int(tcfg["attention_head_dim"])
    hidden = int(tcfg["hidden_size"])
    inner = heads * head_dim

    expect(meta.get("qkv_layout") == "global-qkv-slabs",
           f"source qkv_layout is {meta.get('qkv_layout')!r}, expected 'global-qkv-slabs'; "
           "the permutation below would be wrong", problems)

    qkv_keys = sorted(k for k in entries if k.endswith(QKV_SUFFIXES))
    expect(len(qkv_keys) == 3 * 52,
           f"found {len(qkv_keys)} qkv tensors, expected 156 (52 matrices x weight/scales/biases)",
           problems)
    for key in qkv_keys:
        expect(entries[key].shape[0] == 3 * inner,
               f"{key} axis 0 is {entries[key].shape[0]}, expected {3 * inner}", problems)

    # Cross-check the config against the tensors it describes.
    expect(entries["blocks.0.attn.qkv_proj.scales"].shape[1] == hidden // int(shared["quantization"]["group_size"]),
           "qkv scales group count disagrees with hidden_size / group_size", problems)
    expect(entries["condition_proj.weight"].shape == (hidden, int(tcfg["text_dim"])),
           "condition_proj shape disagrees with hidden_size/text_dim", problems)
    expect(entries["audio_patch_proj.weight"].shape == (hidden, int(tcfg["audio_latents_dim"])),
           "audio_patch_proj shape disagrees with audio_latents_dim", problems)
    expect(entries["blocks.0.mlp.fc1.weight"].shape[0] == 2 * int(tcfg["ffn_hidden_size"]),
           "mlp.fc1 rows are not 2 * ffn_hidden_size (fused SwiGLU)", problems)
    num_blocks = 1 + max(int(k.split(".")[1]) for k in entries if k.startswith("blocks."))
    expect(num_blocks == int(tcfg["num_layers"]),
           f"file has {num_blocks} blocks, config says {tcfg['num_layers']}", problems)

    permutation = slabs_to_interleaved_index(heads, head_dim)

    tensors: list[OutTensor] = []
    for name, entry in entries.items():
        if name.endswith(QKV_SUFFIXES):
            def producer(path=src, e=entry):
                return read_tensor(path, e)[permutation]
            tensors.append(OutTensor.computed(name, entry.dtype, entry.shape, producer))
        else:
            tensors.append(OutTensor.copy_of(entry, src))

    quant = shared["quantization"]
    stats = write_sharded(out_dir, tensors, {
        "format": "mlx",
        "converted_from": "mere.run/Sawfwair video-minimax-h3-fl2va-mlx",
        "qkv_layout": "per-head-interleaved",
    }, cap, dry_run)

    write_json(out_dir / "config.json", dit_config(tcfg), dry_run)
    write_json(out_dir / "quant_config.json", {
        "bits": int(quant["bits"]),
        "group_size": int(quant["group_size"]),
        "quantize_adaln": False,
        "adaln_bits": None,
        "mode": quant.get("mode", "affine"),
        "quantized_layers": {str(int(quant["bits"])): 4 * 52},
        "note": "reconstructed from the mere.run config.json; verified against the shipped tensor "
                "shapes with minimax_h3_mlx.quantize.apply_quantization_structure",
    }, dry_run)

    # The AdaLN path was precomputed away upstream; pass the cache through untouched.
    cache_src = source / "adaln_cache.safetensors"
    cache_meta: dict[str, str] = {}
    if cache_src.exists():
        link_or_copy(cache_src, out_dir / "adaln_cache.safetensors", dry_run)
        cache_meta, cache_entries = read_header(cache_src)
        cache_meta = {**cache_meta, "tensors": str(len(cache_entries))}

    missing = missing_dit_keys(entries)
    warnings = [
        f"{missing['count']} tensors of the AdaLN modulation path are absent from the mere.run "
        "monolith (50 x blocks.N.adaln_proj, final_layer.adaln_proj, time_embedder). "
        "load_dit(strict=True) WILL FAIL on this directory, and ModulationCache.build / "
        "final_layer_modulation cannot run. mere.run precomputed the table into "
        "adaln_cache.safetensors (copied alongside) for a 30-step schedule at shifts 12.0/3.0; "
        "the pipeline has to be taught to read it. See conversion_report.json."
    ]
    return {"component": "transformer", **stats, "problems": problems, "warnings": warnings,
            "qkv_permuted": len(qkv_keys), "adaln_cache": cache_meta,
            "missing_for_strict_load": missing}


def missing_dit_keys(entries: dict[str, Entry]) -> dict[str, object]:
    """The AdaLN/timestep tensors the port expects but mere.run folded into the cache."""
    blocks = 1 + max(int(k.split(".")[1]) for k in entries if k.startswith("blocks."))
    keys = [f"blocks.{i}.adaln_proj.linear.{p}" for i in range(blocks) for p in ("weight", "bias")]
    keys += [f"final_layer.adaln_proj.linear.{p}" for p in ("weight", "bias")]
    keys += [f"time_embedder.{m}.{p}" for m in ("proj_in", "proj_out") for p in ("weight", "bias")]
    return {
        "count": len(keys),
        "reason": "mere.run precomputed the AdaLN modulation into adaln_cache.safetensors and "
                  "omitted the projections; load_dit(strict=True) will fail and "
                  "ModulationCache.build / final_layer_modulation cannot run",
        "keys": keys,
    }


def dit_config(tcfg: dict) -> dict:
    """`DiTConfig` fields, taken from the mere.run config where it carries them.

    ``time_embed_hidden_dim`` is renamed to the port's ``time_embed_hidden_size``; the remaining
    fields are not present in the mere.run file and come from the port's dataclass defaults, which
    were themselves read off the release.
    """
    return {
        "hidden_size": int(tcfg["hidden_size"]),
        "num_layers": int(tcfg["num_layers"]),
        "token_refiner_num_layers": 2,
        "num_attention_heads": int(tcfg["num_attention_heads"]),
        "attention_head_dim": int(tcfg["attention_head_dim"]),
        "ffn_hidden_size": int(tcfg["ffn_hidden_size"]),
        "latents_dim": int(tcfg["latents_dim"]),
        "audio_latents_dim": int(tcfg["audio_latents_dim"]),
        "patch_size": [1, 2, 2],
        "text_dim": int(tcfg["text_dim"]),
        "timestep_input_dim": 256,
        "time_embed_hidden_size": int(tcfg.get("time_embed_hidden_dim", tcfg["hidden_size"])),
        "time_embed_dim": int(tcfg["time_embed_dim"]),
        "adaln_out_features": 96768,
        "final_adaln_out_features": 10752,
        "rope_inv_freq_len": int(tcfg["rope_inv_freq_len"]),
        "rope_theta": 10000.0,
        "norm_eps": 1e-5,
        "qk_norm_eps": 1e-5,
        "final_norm_eps": 1e-5,
    }


# --------------------------------------------------------------------------------------------
# text encoder
# --------------------------------------------------------------------------------------------

def rename_text_encoder_key(key: str) -> str | None:
    """mere.run key -> the path `MiniMaxH3TextEncoder._wanted` matches on."""
    if key.startswith("model.layers.") or key.startswith("model.embed_tokens"):
        return "model.language_model." + key[len("model."):]
    if key.startswith("model.norm."):
        return "model.language_model." + key[len("model."):]
    if key.startswith("visual."):
        return "model." + key
    if key.startswith("model.visual."):
        return key
    return None


def convert_text_encoder(source: Path, out_root: Path, shared: dict, cap: int,
                         dry_run: bool, upstream_config: Path | None) -> dict:
    src = source / "text_encoder.safetensors"
    out_dir = out_root / "text_encoder"
    meta, entries = read_header(src)
    problems: list[str] = []

    tensors: list[OutTensor] = []
    unmapped: list[str] = []
    for name, entry in entries.items():
        renamed = rename_text_encoder_key(name)
        if renamed is None:
            unmapped.append(name)
            continue
        tensors.append(OutTensor.copy_of(entry, src, renamed))
    expect(not unmapped, f"{len(unmapped)} text-encoder keys had no mapping, e.g. {unmapped[:4]}",
           problems)

    ecfg = shared["text_encoder"]
    hidden = int(ecfg["hidden"])

    # `_load_weights` requires `norm.weight` to be present even though `_hidden_states` never applies
    # it (H3 reads the pre-norm state). mere.run dropped it; a ones-vector is the identity.
    names = {t.name for t in tensors}
    synthesized: list[str] = []
    norm_key = "model.language_model.norm.weight"
    if norm_key not in names:
        bf16_one = np.uint16(0x3F80)
        tensors.append(OutTensor.computed(
            norm_key, "BF16", (hidden,), lambda: np.full(hidden, bf16_one, dtype="<u2")))
        synthesized.append(norm_key)

    layers = 1 + max(int(k.split(".")[2]) for k in entries if k.startswith("model.layers."))
    expect(layers == int(ecfg["layers"]),
           f"file has {layers} language layers, config says {ecfg['layers']}", problems)
    expect(entries["model.embed_tokens.weight"].shape[1] == hidden,
           "embed_tokens width disagrees with text_encoder.hidden", problems)
    expect(entries["model.layers.0.self_attn.q_proj.scales"].shape[0]
           == int(ecfg["heads"]) * int(ecfg["head_dim"]),
           "q_proj rows disagree with heads * head_dim", problems)
    expect(entries["model.layers.0.self_attn.k_proj.scales"].shape[0]
           == int(ecfg["kv_heads"]) * int(ecfg["head_dim"]),
           "k_proj rows disagree with kv_heads * head_dim", problems)

    stats = write_sharded(out_dir, tensors, {
        "format": "mlx",
        "converted_from": "mere.run/Sawfwair video-minimax-h3-fl2va-mlx",
    }, cap, dry_run)

    warnings: list[str] = []
    if upstream_config is not None and upstream_config.exists():
        with open(upstream_config) as fh:
            config = json.load(fh)
        config_origin = str(upstream_config)
    else:
        config = qwen3_vl_config(shared, entries)
        config_origin = "synthesized from the mere.run config.json and the shipped tensor shapes"
        warnings.append(
            "text_encoder/config.json was synthesized: mere.run does not ship the upstream one. "
            "Vision num_heads, mrope_section and deepstack_visual_indexes are the published "
            "Qwen3-VL-32B values, not derived — pass --text-encoder-config "
            "<FL2VA/text_encoder/config.json> if you have the release.")
    quant = shared["text_encoder_quantization"]
    config["quantization"] = {"group_size": int(quant["group_size"]), "bits": int(quant["bits"])}
    write_json(out_dir / "config.json", config, dry_run)

    quantized = sorted({k.rsplit(".", 1)[0] for k in entries if k.endswith(".scales")})
    dense_linears = sorted({
        k.rsplit(".", 1)[0] for k in entries
        if k.endswith(".weight") and len(entries[k].shape) == 2 and entries[k].dtype != "U32"
        and f"{k.rsplit('.', 1)[0]}.scales" not in entries
    })
    write_json(out_dir / "quant_config.json", {
        "bits": int(quant["bits"]),
        "group_size": int(quant["group_size"]),
        "mode": quant.get("mode", "affine"),
        "note": "the port's MiniMaxH3TextEncoder builds plain nn.Linear layers and has no "
                "quantization path; nn.quantize must be applied with this recipe before "
                "_load_weights or the packed uint32 weights land in bf16 slots and the "
                "scales/biases are silently dropped",
        "quantized_modules": [rename_text_encoder_key(m) or m for m in quantized],
        "dense_linear_modules": [rename_text_encoder_key(m) or m for m in dense_linears],
    }, dry_run)

    # `tokenizer` is looked up at <root>/tokenizer (falling back to the encoder dir) and the image
    # processor at <root>/processor.
    for name in ("tokenizer.json", "tokenizer_config.json"):
        for target in (out_root / "tokenizer", out_root / "processor", out_dir):
            link_or_copy(source / name, target / name, dry_run)
    if not (out_root / "processor" / "preprocessor_config.json").exists():
        write_json(out_root / "processor" / "preprocessor_config.json",
                   qwen3_vl_preprocessor_config(), dry_run)
        write_json(out_dir / "preprocessor_config.json", qwen3_vl_preprocessor_config(), dry_run)

    return {"component": "text_encoder", **stats, "problems": problems, "warnings": warnings,
            "renamed": len(tensors) - len(synthesized), "synthesized": synthesized,
            "config_origin": config_origin,
            "quantized_modules": len(quantized)}


def qwen3_vl_config(shared: dict, entries: dict[str, Entry]) -> dict:
    """Rebuild the Qwen3-VL-32B ``config.json`` mere.run dropped.

    Everything derivable is derived from the shipped tensors; the rest is the published Qwen3-VL
    architecture. ``num_hidden_layers`` is declared as the **full** 64 on purpose: the port refuses
    to run on a stack that has only the 50 layers it reads, because the last hidden state of a
    truncated stack is post-norm and is not the conditioning H3 expects. It then rebuilds the config
    with 50 layers itself and skips the rest.
    """
    ecfg = shared["text_encoder"]
    vocab, hidden = entries["model.embed_tokens.weight"].shape
    vision_hidden = entries["visual.patch_embed.proj.weight"].shape[0]
    _, in_channels, temporal_patch, patch, _ = entries["visual.patch_embed.proj.weight"].shape
    vision_depth = 1 + max(int(k.split(".")[2]) for k in entries if k.startswith("visual.blocks."))
    vision_intermediate = entries["visual.blocks.0.mlp.linear_fc1.bias"].shape[0]
    # The patch merger concatenates a merge x merge block before projecting, so its *input* width is
    # `vision_hidden * merge**2`. The weight is packed uint32, so the true input width is recovered
    # from the scales: one group per `group_size` input features.
    group_size = int(shared["text_encoder_quantization"]["group_size"])
    merge_in = entries["visual.merger.linear_fc1.scales"].shape[1] * group_size
    spatial_merge = int(round(math.sqrt(merge_in / vision_hidden)))
    deepstack = 1 + max(int(k.split(".")[2]) for k in entries
                        if k.startswith("visual.deepstack_merger_list."))
    out_hidden = entries["visual.merger.linear_fc2.bias"].shape[0]
    positions = entries["visual.pos_embed.weight"].shape[0]

    # Which encoder layers feed the deepstack merge is not recoverable from shapes — only how many.
    # Qwen3-VL's 27-layer tower publishes [8, 16, 24]; anything else falls back to even spacing.
    if (vision_depth, deepstack) == (27, 3):
        deepstack_indexes = [8, 16, 24]
    else:
        step = max(vision_depth // (deepstack + 1), 1)
        deepstack_indexes = [min(step * (i + 1), vision_depth - 1) for i in range(deepstack)]

    head_dim = int(ecfg["head_dim"])
    # M-RoPE splits head_dim/2 across (t, h, w); Qwen3-VL-32B ships [24, 20, 20].
    mrope_section = [24, 20, 20]
    if sum(mrope_section) != head_dim // 2:
        mrope_section = [head_dim // 2 - 2 * (head_dim // 6), head_dim // 6, head_dim // 6]

    return {
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "model_type": "qwen3_vl",
        "image_token_id": 151655,
        "video_token_id": 151656,
        "vision_start_token_id": 151652,
        "vision_end_token_id": 151653,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "text_config": {
            "model_type": "qwen3_vl_text",
            "vocab_size": int(vocab),
            "hidden_size": int(hidden),
            "intermediate_size": int(ecfg["intermediate"]),
            # The full released depth, not the 50 layers shipped here — see the docstring.
            "num_hidden_layers": 64,
            "num_attention_heads": int(ecfg["heads"]),
            "num_key_value_heads": int(ecfg["kv_heads"]),
            "head_dim": head_dim,
            "hidden_act": "silu",
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            "rope_theta": float(ecfg["theta"]),
            "rope_scaling": {"rope_type": "default", "mrope_section": mrope_section,
                             "mrope_interleaved": True},
            "attention_bias": False,
            "attention_dropout": 0.0,
            "use_cache": True,
            "tie_word_embeddings": False,
        },
        "vision_config": {
            "model_type": "qwen3_vl",
            "depth": int(vision_depth),
            "hidden_size": int(vision_hidden),
            "intermediate_size": int(vision_intermediate),
            "num_heads": 16,
            "in_channels": int(in_channels),
            "patch_size": int(patch),
            "temporal_patch_size": int(temporal_patch),
            "spatial_merge_size": int(spatial_merge),
            "out_hidden_size": int(out_hidden),
            "num_position_embeddings": int(positions),
            "deepstack_visual_indexes": deepstack_indexes,
            "hidden_act": "gelu_pytorch_tanh",
            "initializer_range": 0.02,
        },
        "_synthesized_by": "convert_sawfwair.py",
        "_synthesized_note": "mere.run dropped the upstream text_encoder/config.json. Derived "
                             "fields come from the shipped tensor shapes; num_heads (vision), "
                             "mrope_section and deepstack_visual_indexes are the published "
                             "Qwen3-VL-32B values. Pass --text-encoder-config to use the real one.",
    }


def qwen3_vl_preprocessor_config() -> dict:
    """Qwen3-VL image processor settings; only needed for keyframe (fl2va) conditioning.

    `min_pixels` / `max_pixels` are **not** guesses: they match MiniMaxAI/MiniMax-H3, which
    ships the same two numbers in `FL2VA/processor`, `FL2VA/text_encoder` and `processor`.

    They are spelled differently here on purpose. The official file writes them as
    ``size: {"shortest_edge": 65536, "longest_edge": 16777216}``, and mlx-vlm's processor has no
    `size` key — handed that config it silently keeps Qwen2-VL's defaults and caps a keyframe at
    1,003,520 pixels. Copying the official file verbatim would therefore change the conditioning
    without raising anything: a 4K keyframe would yield 943 image tokens instead of 8160, and
    since `num_image_tokens` sets the rotary clock of every media row, the whole timeline shifts.
    `tests/test_image_processor.py` pins both the values and this trap.
    """
    return {
        "image_processor_type": "Qwen2VLImageProcessorFast",
        "processor_class": "Qwen3VLProcessor",
        "do_convert_rgb": True,
        "do_normalize": True,
        "do_rescale": True,
        "do_resize": True,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "min_pixels": 65536,
        "max_pixels": 16777216,
        "patch_size": 16,
        "temporal_patch_size": 2,
        "merge_size": 2,
        "resample": 3,
        "rescale_factor": 0.00392156862745098,
        "_synthesized_by": "convert_sawfwair.py",
    }


# --------------------------------------------------------------------------------------------
# VAEs
# --------------------------------------------------------------------------------------------

STATS_KEYS = ("latents_mean", "latents_std")


def convert_video_vae(source: Path, out_root: Path, dry_run: bool) -> dict:
    src = source / "video_vae.safetensors"
    out_dir = out_root / "video_vae"
    _, entries = read_header(src)
    problems: list[str] = []

    stats_values = {k: to_float_list(read_tensor(src, entries[k]), entries[k].dtype)
                    for k in STATS_KEYS if k in entries}
    # `load_video_vae` reads the statistics from the wrapper config and rejects unexpected keys.
    tensors = [OutTensor.copy_of(e, src) for n, e in entries.items() if n not in STATS_KEYS]

    ch = entries["encoder.conv_in.weight"].shape[0]
    in_channels = entries["encoder.conv_in.weight"].shape[1]
    z_channels = entries["post_quant_conv.weight"].shape[0]
    levels = 1 + max(int(k.split(".")[2]) for k in entries if k.startswith("encoder.down."))
    ch_mult = [entries[f"encoder.down.{i}.block.0.conv2.weight"].shape[0] // ch for i in range(levels)]
    num_res_blocks = 1 + max(int(k.split(".")[4]) for k in entries
                             if k.startswith("encoder.down.") and ".block." in k)
    decoder_layers = 1 + max(int(k.split(".")[2]) for k in entries
                             if k.startswith("decoder.transformer_blocks."))
    decoder_dim = entries["decoder.x_embedder.weight"].shape[0]

    # Not derivable from shapes: which of the six levels downsample in space vs time, and the ViT
    # head split. Both come from the port's defaults (read off the release); the *presence* of a
    # downsample per level and the head product are checked against the file.
    space_down = [2, 2, 2, 2, 1, 1]
    time_down = [1, 2, 2, 1, 1, 1]
    heads, dim_head = 32, 64

    has_downsample = [any(k.startswith(f"encoder.down.{i}.downsample.") for k in entries)
                      for i in range(levels)]
    expected_downsample = [s > 1 or t > 1 for s, t in zip(space_down, time_down)]
    expect(has_downsample == expected_downsample,
           f"downsample presence {has_downsample} disagrees with the assumed "
           f"space_down/time_down {space_down}/{time_down}", problems)
    expect(heads * dim_head == decoder_dim,
           f"assumed ViT heads*dim_head != decoder width {decoder_dim}", problems)
    expect(entries["encoder.conv_out.weight"].shape[0] == 2 * z_channels,
           "encoder.conv_out does not emit 2 * z_channels (mean/logvar)", problems)
    out_ch = entries["decoder.proj_out.weight"].shape[0] // (
        math.prod(time_down) * math.prod(space_down) ** 2)
    expect(out_ch == in_channels, f"decoder output channels {out_ch} != input {in_channels}", problems)

    write_json(out_dir / "config.json", {
        "vae_clip_length": 17,
        "vae_token_drop": 3,
        **{k: v for k, v in stats_values.items()},
        "_synthesized_by": "convert_sawfwair.py",
        "_synthesized_note": "latents_mean/latents_std lifted out of the mere.run monolith, where "
                             "they were stored as tensors; clip_length/token_drop are the port's "
                             "defaults (mere.run dropped the upstream wrapper config).",
    }, dry_run)
    write_json(out_dir / "source" / "config.json", {
        "ch": int(ch),
        "in_channels": int(in_channels),
        "out_ch": int(out_ch),
        "z_channels": int(z_channels),
        "ch_mult": [int(m) for m in ch_mult],
        "num_res_blocks": int(num_res_blocks),
        "space_down": space_down,
        "time_down": time_down,
        "vit_decoder_kwargs": {
            "num_layers": int(decoder_layers),
            "heads": heads,
            "dim_head": dim_head,
            "rope_theta": 100.0,
            "rope_dim_ratio": 0.75,
        },
        "_synthesized_by": "convert_sawfwair.py",
    }, dry_run)

    stats = write_single(out_dir / "source" / "model.safetensors", tensors, {
        "format": "mlx",
        "converted_from": "mere.run/Sawfwair video-minimax-h3-fl2va-mlx",
    }, dry_run)
    return {"component": "video_vae", **stats, "problems": problems,
            "stats_lifted_to_config": sorted(stats_values)}


def convert_audio_vae(source: Path, out_root: Path, dry_run: bool) -> dict:
    src = source / "audio_vae.safetensors"
    out_dir = out_root / "audio_vae"
    _, entries = read_header(src)
    problems: list[str] = []

    stats_values = {k: to_float_list(read_tensor(src, entries[k]), entries[k].dtype)
                    for k in STATS_KEYS if k in entries}
    tensors = [OutTensor.copy_of(e, src) for n, e in entries.items() if n not in STATS_KEYS]

    encoder_dim = entries["encoder.block.0.weight"].shape[0]
    stages = sorted(int(k.split(".")[2]) for k in entries
                    if k.startswith("encoder.block.") and k.endswith(".block.4.weight"))
    encoder_rates = [entries[f"encoder.block.{i}.block.4.weight"].shape[2] // 2 for i in stages]
    latent_dim = entries["encoder.block.7.weight"].shape[0]
    latent_channels = entries["mean_proj.weight"].shape[0]
    decoder_dim = entries["decoder.conv_pre.weight"].shape[0]
    up_kernels = [entries[f"decoder.ups.{i}.0.weight"].shape[2]
                  for i in range(1 + max(int(k.split(".")[2]) for k in entries if ".ups." in k))]

    # Upsample rate is not recoverable from a transposed-conv kernel alone (rate 5 -> kernel 9,
    # rate 2 -> kernel 4), so the port's defaults are used and checked two ways: the kernels must
    # match, and the decoder must invert the encoder's total hop.
    decoder_rates = [5, 5, 2, 2, 2, 2, 2]
    decoder_kernels = [9, 9, 4, 4, 4, 4, 4]
    expect(up_kernels == decoder_kernels,
           f"decoder upsample kernels {up_kernels} disagree with the assumed rates {decoder_rates}",
           problems)
    expect(math.prod(decoder_rates) == math.prod(encoder_rates),
           f"decoder hop {math.prod(decoder_rates)} != encoder hop {math.prod(encoder_rates)}",
           problems)
    expect(not any(k.endswith(("weight_g", "weight_v")) for k in entries),
           "weight-norm pairs are still present; the port expects them already folded", problems)

    write_json(out_dir / "config.json", {
        **{k: v for k, v in stats_values.items()},
        "_synthesized_by": "convert_sawfwair.py",
        "_synthesized_note": "latents_mean/latents_std lifted out of the mere.run monolith.",
    }, dry_run)
    write_json(out_dir / "metadata.json", {
        "metadata": {
            "kwargs": {
                "encoder_dim": int(encoder_dim),
                "encoder_rates": [int(r) for r in encoder_rates],
                "latent_dim": int(latent_dim),
                "vae_latent_channels": int(latent_channels),
                "decoder_dim": int(decoder_dim),
                "decoder_rates": decoder_rates,
                "sample_rate": 32000,
            }
        },
        "_synthesized_by": "convert_sawfwair.py",
    }, dry_run)

    stats = write_single(out_dir / "model.safetensors", tensors, {
        "format": "mlx",
        "converted_from": "mere.run/Sawfwair video-minimax-h3-fl2va-mlx",
    }, dry_run)
    return {"component": "audio_vae", **stats, "problems": problems,
            "stats_lifted_to_config": sorted(stats_values)}


# --------------------------------------------------------------------------------------------
# checks against the port's module trees
# --------------------------------------------------------------------------------------------

def check_against_port(out_root: Path, port: Path, components: Iterable[str]) -> dict:
    """Diff the written key sets against the port's parameter trees.

    MLX parameters are lazy, so a full 33B `MiniMaxH3DiT` can be instantiated for its *names and
    shapes* without materializing a byte.
    """
    sys.path.insert(0, str(port))
    import mlx.core as mx  # noqa: F401  (imported for its side effect of proving MLX is present)
    from mlx.utils import tree_flatten

    from minimax_h3_mlx.config import DiTConfig
    from minimax_h3_mlx.dit import MiniMaxH3DiT
    from minimax_h3_mlx.quantize import QuantConfig, apply_quantization_structure

    report: dict[str, object] = {}

    def diff(label: str, expected: dict[str, tuple], got: dict[str, tuple],
             ignore_missing: set[str] = frozenset(), ignore_extra: set[str] = frozenset(),
             shape_transform: Callable[[str, tuple], tuple] | None = None) -> dict:
        missing = sorted(set(expected) - set(got) - ignore_missing)
        extra = sorted(set(got) - set(expected) - ignore_extra)
        bad = []
        for key in set(expected) & set(got):
            shape = shape_transform(key, got[key]) if shape_transform else got[key]
            if tuple(expected[key]) != tuple(shape):
                bad.append([key, list(expected[key]), list(got[key])])
        result = {"expected": len(expected), "written": len(got), "missing": missing[:20],
                  "missing_count": len(missing), "extra": extra[:20], "extra_count": len(extra),
                  "shape_mismatch": bad[:20], "shape_mismatch_count": len(bad)}
        report[label] = result
        return result

    def written(paths: Iterable[Path]) -> dict[str, tuple]:
        out: dict[str, tuple] = {}
        for path in paths:
            _, entries = read_header(path)
            out.update({k: e.shape for k, e in entries.items()})
        return out

    if "transformer" in components:
        cfg_path = out_root / "transformer" / "config.json"
        quant_path = out_root / "transformer" / "quant_config.json"
        model = MiniMaxH3DiT(DiTConfig.from_json(cfg_path))
        with open(quant_path) as fh:
            recipe = json.load(fh)
        apply_quantization_structure(model, QuantConfig(
            bits=recipe["bits"], group_size=recipe["group_size"],
            quantize_adaln=recipe.get("quantize_adaln", False),
            adaln_bits=recipe.get("adaln_bits") or 8))
        expected = {k: tuple(v.shape) for k, v in tree_flatten(model.parameters())}
        index = out_root / "transformer" / "model.safetensors.index.json"
        with open(index) as fh:
            shards = sorted({v for v in json.load(fh)["weight_map"].values()})
        diff("transformer", expected,
             written(out_root / "transformer" / n for n in shards))

    if "video_vae" in components:
        from minimax_h3_mlx.load import load_video_vae  # noqa: F401 (import check only)
        from minimax_h3_mlx.video_vae import VideoVAE, VideoVAEConfig
        with open(out_root / "video_vae" / "config.json") as fh:
            wrapper = json.load(fh)
        with open(out_root / "video_vae" / "source" / "config.json") as fh:
            src_cfg = json.load(fh)
        ch = src_cfg["ch"]
        cfg = VideoVAEConfig(
            in_channels=src_cfg["in_channels"], out_channels=src_cfg["out_ch"],
            latent_channels=src_cfg["z_channels"],
            block_out_channels=tuple(ch * m for m in src_cfg["ch_mult"]),
            layers_per_block=src_cfg["num_res_blocks"],
            spatial_downsample_factors=tuple(src_cfg["space_down"]),
            temporal_downsample_factors=tuple(src_cfg["time_down"]),
            decoder_num_layers=src_cfg["vit_decoder_kwargs"]["num_layers"],
            decoder_num_attention_heads=src_cfg["vit_decoder_kwargs"]["heads"],
            decoder_attention_head_dim=src_cfg["vit_decoder_kwargs"]["dim_head"],
            decoder_rope_theta=src_cfg["vit_decoder_kwargs"]["rope_theta"],
            decoder_rope_dim_ratio=src_cfg["vit_decoder_kwargs"]["rope_dim_ratio"],
            clip_length=wrapper.get("vae_clip_length", 17),
            token_drop=wrapper.get("vae_token_drop", 3))
        expected = {k: tuple(v.shape) for k, v in tree_flatten(VideoVAE(cfg).parameters())}
        got = written([out_root / "video_vae" / "source" / "model.safetensors"])
        # The loader moves 5-D conv weights to channels-last on the way in.
        diff("video_vae", expected, got, ignore_extra={"decoder.mask_token"},
             shape_transform=lambda k, s: (s[0], *s[2:], s[1]) if len(s) == 5 else s)

    if "audio_vae" in components:
        from minimax_h3_mlx.audio_vae import AudioVAE, AudioVAEConfig
        with open(out_root / "audio_vae" / "metadata.json") as fh:
            kwargs = json.load(fh)["metadata"]["kwargs"]
        cfg = AudioVAEConfig(
            encoder_dim=kwargs["encoder_dim"], encoder_rates=tuple(kwargs["encoder_rates"]),
            latent_dim=kwargs["latent_dim"], latent_channels=kwargs["vae_latent_channels"],
            decoder_dim=kwargs["decoder_dim"], decoder_rates=tuple(kwargs["decoder_rates"]),
            sampling_rate=kwargs["sample_rate"])
        expected = {k: tuple(v.shape) for k, v in tree_flatten(AudioVAE(cfg).parameters())}
        got = {k: v for k, v in written([out_root / "audio_vae" / "model.safetensors"]).items()
               if not k.endswith(".filter")}

        def audio_shape(key: str, shape: tuple) -> tuple:
            if key.endswith(".weight") and len(shape) == 3:
                return (shape[1], shape[2], shape[0]) if ".ups." in key else (shape[0], shape[2], shape[1])
            if key.endswith(".alpha") and len(shape) == 3:
                return (shape[0], shape[2], shape[1])
            return shape

        diff("audio_vae", expected, got, shape_transform=audio_shape)

    return report


# --------------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------------

ALL_COMPONENTS = ("transformer", "text_encoder", "video_vae", "audio_vae")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="the mere.run/Sawfwair model directory (read-only)")
    parser.add_argument("--out", type=Path, required=True, help="output checkpoint root")
    parser.add_argument("--components", nargs="+", default=list(ALL_COMPONENTS),
                        choices=list(ALL_COMPONENTS))
    parser.add_argument("--shard-bytes", type=int, default=SHARD_BYTES,
                        help="shard size cap; also the converter's peak write buffer")
    parser.add_argument("--text-encoder-config", type=Path, default=None,
                        help="upstream FL2VA/text_encoder/config.json to use verbatim instead of "
                             "the synthesized one (strongly preferred if you have it)")
    parser.add_argument("--port", type=Path, default=DEFAULT_PORT,
                        help="the minimax-h3-mlx checkout, for --check")
    parser.add_argument("--check", action="store_true",
                        help="after writing, diff the key sets against the port's module trees")
    parser.add_argument("--dry-run", action="store_true", help="plan and validate, write nothing")
    args = parser.parse_args()

    source: Path = args.source.expanduser()
    out_root: Path = args.out.expanduser()
    if not source.is_dir():
        print(f"source not found: {source}", file=sys.stderr)
        return 1
    with open(source / "config.json") as fh:
        shared = json.load(fh)

    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
    write_json(out_root / "model_index.json", {
        "_class_name": "MiniMaxH3Pipeline",
        "_minimax_h3": {
            "partition": shared.get("partition", "fl2va"),
            "tasks": shared.get("tasks", ["t2va", "fl2va"]),
            "sigma_shift_scales": shared.get("sigma_shift_scales", {"video": 12.0, "audio": 3.0}),
            "fps": shared.get("fps", 24),
        },
        "_converted_from": "mere.run/Sawfwair video-minimax-h3-fl2va-mlx",
    }, args.dry_run)

    results = []
    started = time.perf_counter()
    for component in ALL_COMPONENTS:
        if component not in args.components:
            continue
        print(f"[{component}]", flush=True)
        if component == "transformer":
            results.append(convert_transformer(source, out_root, shared, args.shard_bytes, args.dry_run))
        elif component == "text_encoder":
            results.append(convert_text_encoder(source, out_root, shared, args.shard_bytes,
                                                args.dry_run, args.text_encoder_config))
        elif component == "video_vae":
            results.append(convert_video_vae(source, out_root, args.dry_run))
        elif component == "audio_vae":
            results.append(convert_audio_vae(source, out_root, args.dry_run))
        last = results[-1]
        print(f"  {last['tensors']} tensors, {last['bytes'] / 1e9:.2f} GB, "
              f"{last.get('written', 0)} file(s) written, {last.get('skipped', 0)} up to date",
              flush=True)
        for problem in last["problems"]:
            print(f"  !! {problem}", file=sys.stderr, flush=True)
        for warning in last.get("warnings", ()):
            print(f"  ~~ {warning}", file=sys.stderr, flush=True)

    report = {
        "source": str(source),
        "out": str(out_root),
        "components": list(args.components),
        "seconds": round(time.perf_counter() - started, 1),
        "results": results,
    }
    if args.check and not args.dry_run:
        print("[check] diffing against the port's module trees", flush=True)
        report["check"] = check_against_port(out_root, args.port.expanduser(), args.components)
        for label, result in report["check"].items():
            print(f"  {label}: expected {result['expected']}, written {result['written']}, "
                  f"missing {result['missing_count']}, extra {result['extra_count']}, "
                  f"shape-mismatch {result['shape_mismatch_count']}", flush=True)

    write_json(out_root / "conversion_report.json", report, args.dry_run)
    problems = sum(len(r["problems"]) for r in results)
    print(f"\ndone in {report['seconds']}s; {problems} validation problem(s); "
          f"report at {out_root / 'conversion_report.json'}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
