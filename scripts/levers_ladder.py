#!/usr/bin/env python3
"""Peak-memory ladder: the stock DiT against the four attention memory levers, by clip length.

This is the instrument behind `docs/RESULTS.md`, "Attention memory levers". **Re-run it after any
change to the chunked paths, to the chunk widths, or to MLX** — the levers' whole value is a
number, and a refactor that quietly re-materializes one buffer gives back several gigabytes while
every test stays green.

    ./.venv/bin/python scripts/levers_ladder.py --durations 2.4 5 10 15

One process, one weight load, three variants:

  stock             unsplit projections plus frozen copies of the pre-patch upstream code.
  levers            the repo as it stands: carved QKV, rope-lean, q-chunk, mlp-chunk.
  levers+fc2chunk   `fc2` chunked as well — what the shipped code deliberately refuses to do,
                    because its LoRA reads the activation being built and so cannot be made
                    bit-exact. Measured anyway, so the price of that exactness is a number.

Two blocks by default, not fifty: the peak of a DiT forward is block-local — each block's
intermediates die with the block — so a 2-block forward reproduces the 50-block peak at a
twenty-fifth of the cost. `--blocks 0` runs all fifty if you want to check that claim.

Peaks are MLX's own allocator counter, never RSS. The two are not interchangeable: RSS on Apple
silicon includes memory-mapped checkpoint pages, and the wired limit applies to the allocator.

**The long durations are heavy on purpose.** Stock at 15 s peaks near 45 GB, above this machine's
40.2 GB recommended working set, so it runs on swap and takes minutes. That is the baseline being
measured, not a malfunction.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dit_probe as probe  # noqa: E402

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

from minimax_h3_mlx import dit as updit  # noqa: E402

from h3_48gb import memory  # noqa: E402
from h3_48gb.dit import load_dit_cached, split_fused_attention  # noqa: E402


def fc2_chunked_feed_forward(self, x):
    """`FeedForward` with `fc2` inside the chunk loop — the variant that is NOT shipped.

    Kept here rather than in `dit.py` because it is a measurement, not an option: with a LoRA
    attached it lands a bf16 ULP away from the unchunked forward, and no configuration flag should
    be able to make the model stop being bit-exact.
    """
    chunk = updit.FFN_ROW_CHUNK
    if x.shape[-2] <= chunk:
        return probe.stock_feed_forward(self, x)
    prepared = updit.prepare_rows(self.fc1, x)
    outs = []
    for start in range(0, x.shape[-2], chunk):
        stop = min(start + chunk, x.shape[-2])
        fused = updit.apply_rows(self.fc1, x, prepared, start, stop)
        gate, value = fused[..., : self._ffn], fused[..., self._ffn:]
        outs.append(self.fc2(nn.silu(gate) * value))
    return mx.concatenate(outs, axis=-2)


def install(variant: str) -> None:
    probe.install(stock=variant == "stock")
    if variant == "levers+fc2chunk":
        updit.FeedForward.__call__ = fc2_chunked_feed_forward


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    probe.add_common_arguments(ap)
    ap.add_argument("--durations", type=float, nargs="+", default=[2.4, 5.0, 10.0, 15.0])
    ap.add_argument("--blocks", type=int, default=2, help="0 = all 50")
    ap.add_argument("--fc2chunk-at", type=float, nargs="*", default=[10.0],
                    help="durations to also measure with fc2 chunked; empty to skip")
    ap.add_argument("--out", type=Path, default=None, help="write the raw numbers here as JSON")
    args = ap.parse_args()

    idle, why = probe.queue_is_idle()
    print(f"gpu: {why}", flush=True)
    if not idle:
        print("refusing to measure — peaks measured against another job are fiction", flush=True)
        return 2

    memory.limit_cache(2.0)
    probe.apply_chunk_overrides(args)
    install("stock")

    model = load_dit_cached(args.transformer, verbose=False, split_qkv=False)
    cache = probe.synthetic_modulation(args.adaln, args.timesteps, model.config.hidden_size,
                                       len(model.blocks))
    model.final_layer.set_modulation(cache.final_shift, cache.final_scale)
    resident = mx.get_active_memory() / 1e9
    print(f"resident weights + adaln: {resident:.2f} GB", flush=True)

    all_blocks = model.blocks
    model.blocks = all_blocks[: args.blocks] if args.blocks else all_blocks
    runs: list[dict] = []

    def measure(variant: str, seconds: float) -> None:
        geo = probe.geometry(seconds, args.height, args.width, model.config.patch_size,
                             args.text_rows)
        install(variant)
        mx.clear_cache()
        mx.reset_peak_memory()
        try:
            mx.random.seed(0)
            inputs = probe.random_inputs(model, geo["layout"], args.timesteps)
            mx.reset_peak_memory()
            started = time.perf_counter()
            video_out, audio_out = probe.forward(model, geo["layout"], inputs, cache)
            elapsed = time.perf_counter() - started
            peak = mx.get_peak_memory() / 1e9
            del video_out, audio_out, inputs
        except Exception as exc:  # noqa: BLE001
            print(f"  {variant:<16} {seconds:>5} s  FAILED {type(exc).__name__}: "
                  f"{str(exc)[:70]}", flush=True)
            runs.append({"variant": variant, "seconds": seconds, "rows": geo["rows"],
                         "error": f"{type(exc).__name__}: {exc}"})
            memory.release()
            return
        runs.append({"variant": variant, "seconds": seconds, "rows": geo["rows"],
                     "peak_gb": peak, "resident_gb": resident, "activation_gb": peak - resident,
                     "bytes_per_row": (peak - resident) * 1e9 / geo["rows"],
                     "forward_s": elapsed, "blocks": args.blocks or 50})
        print(f"  {variant:<16} {seconds:>5} s  {geo['rows']:>8,} rows  peak {peak:6.2f} GB  "
              f"act {peak - resident:6.2f}  "
              f"{(peak - resident) * 1e9 / geo['rows']:>8,.0f} B/row  {elapsed:6.1f}s", flush=True)
        memory.release()

    print("\n=== stock (fused QKV, no levers) ===", flush=True)
    for seconds in args.durations:
        measure("stock", seconds)

    print("\n=== carving the fused QKV ===", flush=True)
    split_fused_attention(model, verbose=True)
    memory.release()
    print(f"resident after the carve: {mx.get_active_memory() / 1e9:.2f} GB "
          f"(was {resident:.2f} — the carve must not grow it)", flush=True)

    print("\n=== levers (rope-lean + qkv-split + q-chunk + mlp-chunk) ===", flush=True)
    for seconds in args.durations:
        measure("levers", seconds)

    if args.fc2chunk_at:
        print("\n=== levers + fc2 chunked (NOT shipped: not bit-exact under a LoRA) ===",
              flush=True)
        for seconds in args.fc2chunk_at:
            measure("levers+fc2chunk", seconds)

    print("\n=== ladder ===", flush=True)
    print(f"{'seconds':>8} {'rows':>9} {'stock':>9} {'levers':>9} {'delta':>8} "
          f"{'stock s':>9} {'levers s':>9}")
    indexed = {(r["variant"], r["seconds"]): r for r in runs}

    def cell(row, key):
        if row is None:
            return f"{'-':>9}"
        return f"{row[key]:9.2f}" if key in row else f"{'OOM':>9}"

    for seconds in args.durations:
        stock, levers = indexed.get(("stock", seconds)), indexed.get(("levers", seconds))
        rows = (stock or levers or {}).get("rows", 0)
        delta = (f"{levers['peak_gb'] - stock['peak_gb']:8.2f}"
                 if stock and levers and "peak_gb" in stock and "peak_gb" in levers
                 else f"{'-':>8}")
        print(f"{seconds:>8} {rows:>9,} {cell(stock, 'peak_gb')} {cell(levers, 'peak_gb')} "
              f"{delta} {cell(stock, 'forward_s')} {cell(levers, 'forward_s')}")

    if args.out:
        args.out.write_text(json.dumps(
            {"args": {k: str(v) for k, v in vars(args).items()},
             "resident_gb": resident, "runs": runs}, indent=2))
        print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
