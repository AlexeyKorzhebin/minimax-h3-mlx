#!/usr/bin/env python3
"""Bit-parity of the attention memory levers against the pre-patch DiT, on the real checkpoint.

**Run this before landing any change to a chunked path, and after any MLX upgrade.** The unit
tests in `tests/test_attention_levers.py` run on doll-sized random weights and cannot see the one
thing that actually threatens these levers: MLX picks GEMM and attention kernels by *shape*, and
at doll size it picks different ones than it does at 7168 channels and 56 heads. The bug this
script exists to catch — a narrow matmul splitting its reduction differently at different row
counts, putting the LoRA a bf16 ULP off — was invisible to every unit test and obvious here.

    ./.venv/bin/python scripts/levers_parity.py --blocks 4 --lora ~/models/turbo/…safetensors

How it works: two loads in one process, the second only after the first is released.

  reference   loaded with `split_qkv=False`, with `apply_rotary` / `Attention.__call__` /
              `FeedForward.__call__` monkeypatched back to frozen copies of the upstream
              originals, and the LoRA applied to the fused `qkv_proj` with its row permutation.
  candidate   loaded as the repo loads it — carved projections, all four levers — with the LoRA
              applied the split way.

Anything but `max|delta| = 0.0` is a failure. Not "close": these levers are only allowed to exist
because they change nothing, and nothing means no bit. Exit status is 1 if any output differs.

`--query-chunk` / `--ffn-chunk` default to 2048 / 1500 rather than to the shipped 8192, and that
is load-bearing. A parity canvas is ~6.2k rows, so at 8192 both chunked branches fall through to
their unchunked fallbacks and the run silently proves nothing about the two chunking levers.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dit_probe as probe  # noqa: E402

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from minimax_h3_mlx.config import DiTConfig  # noqa: E402

from h3_48gb import memory  # noqa: E402
from h3_48gb.dit import load_dit_cached  # noqa: E402


def run(stock: bool, args, geo) -> tuple[np.ndarray, np.ndarray]:
    """Load, apply the LoRA, run one forward, release everything, return the outputs."""
    probe.install(stock)
    started = time.perf_counter()
    model = load_dit_cached(args.transformer, verbose=False, split_qkv=not stock)

    # The LoRA goes on the *whole* stack before any truncation: `apply_backbone_lora` checks both
    # directions, and a truncated stack looks to it like an adapter carrying targets nobody applied.
    if args.lora:
        from h3_48gb.turbo import apply_backbone_lora

        apply_backbone_lora(model, args.lora, strength=args.strength, verbose=args.verbose)
    if args.blocks:
        model.blocks = model.blocks[: args.blocks]

    cache = probe.synthetic_modulation(args.adaln, args.timesteps, model.config.hidden_size,
                                       len(model.blocks))
    model.final_layer.set_modulation(cache.final_shift, cache.final_scale)

    layout = geo["layout"]
    mx.random.seed(args.seed)
    inputs = probe.random_inputs(model, layout, args.timesteps, spread_levels=False)
    video_out, audio_out = probe.forward(model, layout, inputs, cache)
    outputs = (np.array(video_out.astype(mx.float32)), np.array(audio_out.astype(mx.float32)))

    print(f"  {'reference (stock)' if stock else 'candidate (levers)':<20} "
          f"{time.perf_counter() - started:5.0f}s", flush=True)
    del video_out, audio_out, inputs, model, cache
    memory.release()
    return outputs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    probe.add_common_arguments(ap)
    ap.set_defaults(query_chunk=2048, ffn_chunk=1500, height=512, width=512)
    ap.add_argument("--blocks", type=int, default=4,
                    help="transformer blocks to run; 0 = all 50 (slower, and the real gate)")
    ap.add_argument("--seconds", type=float, default=2.4)
    ap.add_argument("--lora", type=str, default="",
                    help=f"LoRA to apply to both sides; pass {probe.DEFAULT_LORA} for Turbo")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    memory.limit_cache(2.0)
    probe.apply_chunk_overrides(args)

    patch_size = DiTConfig.from_json(args.transformer / "config.json").patch_size
    geo = probe.geometry(args.seconds, args.height, args.width, patch_size, args.text_rows)
    rows = geo["rows"]

    print(f"parity: {args.width}x{args.height} {args.seconds}s -> {rows:,} rows, "
          f"blocks={args.blocks or 50}, lora={args.lora or 'none'} x{args.strength}", flush=True)
    for name, width in (("attention", updit_chunk("QUERY_CHUNK")),
                        ("MLP", updit_chunk("FFN_ROW_CHUNK"))):
        entered = "WILL be entered" if width < rows else "will NOT be entered — proves nothing"
        print(f"  chunked {name} path at {width:,} rows: {entered}", flush=True)

    reference = run(True, args, geo)
    candidate = run(False, args, geo)

    failed = False
    for name, want, got in (("video", reference[0], candidate[0]),
                            ("audio", reference[1], candidate[1])):
        identical = bool((got == want).all())
        print(f"{name}: max|delta| = {np.abs(got - want).max():.3e}  "
              f"(scale {np.abs(want).max():.3e})  bit-identical: {identical}", flush=True)
        failed |= not identical

    print("PARITY FAILED — the levers changed a value" if failed
          else "PARITY OK — bit-identical", flush=True)
    return 1 if failed else 0


def updit_chunk(name: str) -> int:
    from minimax_h3_mlx import dit as updit

    return getattr(updit, name, 1 << 30)


if __name__ == "__main__":
    raise SystemExit(main())
