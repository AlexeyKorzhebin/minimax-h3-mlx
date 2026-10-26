#!/usr/bin/env python3
"""Pull the AdaLN time curve out of Comfy-Org's pruned base without downloading 40 GB.

This fork's checkpoint has no modulation path — mere.run dropped `time_embedder` and all 50
`adaln_proj` (13B parameters) and shipped a table baked for one 31-point grid instead. The pruned
Comfy-Org build keeps the same information folded into an `adaln_t_table` of 1025x8 plus per-block
projections from those 8 dims: **87.3 MB inside a 40 GB file**.

safetensors puts a JSON header up front listing every tensor's byte range, so the 103 tensors that
matter can be fetched with HTTP range requests and nothing else. They turn out to be contiguous,
so this is one request.

    ./.venv/bin/python scripts/fetch_adaln_curve.py
    ./.venv/bin/python scripts/bake_adaln.py 8      # then bake any grid you like
"""
from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

URL = ("https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/"
       "diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors")
DEST = Path.home() / "models/turbo/adaln_curve.safetensors"

#: The bundled silu(time_embedder(t)) grid, needed only to fold a LoRA's AdaLN half into the
#: table. Ships with Larryvrh's ComfyUI node; `bake_adaln.py --lora` expects it beside the curve.
GRID_URL = ("https://raw.githubusercontent.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/"
            "main/h3_silu_temb_grid.safetensors")


def _curl(args: list[str]) -> bytes:
    result = subprocess.run(["curl", "-sL", *args], capture_output=True, check=True)
    return result.stdout


def main() -> int:
    if DEST.exists():
        print(f"{DEST} already present ({DEST.stat().st_size / 1e6:.1f} MB)")
        return 0

    header_len = struct.unpack("<Q", _curl(["-r", "0-7", URL])[:8])[0]
    header = json.loads(_curl(["-r", f"8-{header_len + 7}", URL]))
    header.pop("__metadata__", None)
    prefix = 8 + header_len

    wanted = {k: v for k, v in header.items()
              if "adaln" in k.lower() or "t_table" in k.lower()}
    if not wanted:
        sys.exit("no AdaLN tensors in that checkpoint's header — wrong file?")
    total = sum(v["data_offsets"][1] - v["data_offsets"][0] for v in wanted.values())
    print(f"{len(wanted)} tensors, {total / 1e6:.1f} MB of a 40 GB file")

    lo = min(v["data_offsets"][0] for v in wanted.values())
    hi = max(v["data_offsets"][1] for v in wanted.values())
    if hi - lo > total * 1.5:
        sys.exit(f"tensors are not contiguous ({(hi - lo) / 1e6:.0f} MB span); "
                 "fetch them in runs instead of one range")
    blob = _curl(["-r", f"{prefix + lo}-{prefix + hi - 1}", URL])
    if len(blob) != hi - lo:
        sys.exit(f"short read: {len(blob)} of {hi - lo} bytes")

    out_header, offset = {}, 0
    payload = bytearray()
    for name, meta in sorted(wanted.items(), key=lambda kv: kv[1]["data_offsets"][0]):
        a, b = meta["data_offsets"]
        raw = blob[a - lo:b - lo]
        out_header[name] = {"dtype": meta["dtype"], "shape": meta["shape"],
                            "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
        payload += raw

    packed = json.dumps(out_header).encode()
    packed += b" " * ((8 - len(packed) % 8) % 8)
    DEST.parent.mkdir(parents=True, exist_ok=True)
    with open(DEST, "wb") as fh:
        fh.write(struct.pack("<Q", len(packed)))
        fh.write(packed)
        fh.write(payload)
    print(f"wrote {DEST} ({DEST.stat().st_size / 1e6:.1f} MB)")

    grid = DEST.parent / "h3_silu_temb_grid.safetensors"
    if not grid.exists():
        grid.write_bytes(_curl([GRID_URL]))
        print(f"wrote {grid} ({grid.stat().st_size / 1e6:.1f} MB) — needed only for --lora")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
