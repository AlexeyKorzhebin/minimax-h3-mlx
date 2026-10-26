#!/usr/bin/env python3
"""Run an H3 generation with per-phase timing.

    ./.venv/bin/python run_bench.py --width 512 --height 512 --duration 2.4 --tag smoke

Writes a JSON report next to the clip with phase timings and memory peaks, so runs
can be compared against each other instead of relying on impressions.
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from h3_48gb import LazyMiniMaxH3Pipeline, snapshot  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(Path.home() / "models/h3-converted"))
    ap.add_argument("--prompt", default="a jeweled hummingbird hovering beside a red orchid, "
                                        "cinematic natural light")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--duration", type=float, default=2.4)
    # The modulation cache is baked for 31 grid points (30 forwards). Any other number fails
    # loudly instead of silently computing something else.
    ap.add_argument("--steps", type=int, default=31)
    ap.add_argument("--seed", type=int, default=314159)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--outdir", default=str(Path.home() / "video-out"))
    # Resumption is on by default and costs one small file write per step. At 586 s per step there
    # is no run long enough for the write to matter and none short enough for a lost run not to.
    ap.add_argument("--checkpoint-dir", default=None,
                    help="where to keep the resume checkpoint (default: <outdir>/checkpoints)")
    ap.add_argument("--no-checkpoint", action="store_true",
                    help="disable checkpointing; a crash then costs the whole run")
    ap.add_argument("--restart", action="store_true",
                    help="ignore any existing checkpoint and start from step 0")
    # Every fifth step by default: at 586 s/step that is a preview roughly every 49 minutes,
    # cheap enough against a run measured in hours that there is little reason to turn it off.
    # 0 disables previews entirely.
    ap.add_argument("--preview-every", type=int, default=5,
                    help="decode a preview JPEG every N steps, 0 to disable (default: 5)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = outdir / f"h3-{args.tag}-{args.width}x{args.height}"
    ckpt_dir = None if args.no_checkpoint else Path(args.checkpoint_dir or outdir / "checkpoints")

    print(f"=== {args.width}x{args.height}, {args.duration}s, {args.steps} grid points ===",
          flush=True)
    print(f"memory before start: {snapshot()}", flush=True)

    t0 = time.perf_counter()
    pipe = LazyMiniMaxH3Pipeline.from_pretrained(args.checkpoint)
    t_open = time.perf_counter() - t0
    print(f"configs read in {t_open:.2f}s, memory: {snapshot()}", flush=True)

    t1 = time.perf_counter()
    result = pipe(
        args.prompt,
        duration_seconds=args.duration,
        num_inference_steps=args.steps,
        seed=args.seed,
        height=args.height,
        width=args.width,
        # Same arguments -> same checkpoint file, so re-running this command after a crash picks up
        # where it stopped. The file is removed once the run finishes. `tag` is part of that
        # identity too (see h3_48gb.checkpoint.request_identity) so this stays interoperable with
        # `h3 generate`/`h3 resume`, which also pass it: a preview run started here can be picked
        # up by `h3 resume` later only if every identity-bearing flag, tag included, matches.
        checkpoint_dir=str(ckpt_dir) if ckpt_dir else None,
        resume=not args.restart,
        tag=args.tag,
        # `<stem>-preview-step07.jpg`, next to where `<stem>.mp4` will land once the run finishes.
        preview_every=args.preview_every,
        preview_stem=str(stem) if args.preview_every else None,
    )
    total = time.perf_counter() - t1

    # Raw arrays to disk BEFORE any encoding: an error writing the mp4 must not cost
    # hours of compute. It has cost that once already — all that survived the clip was
    # a traceback.
    np.savez_compressed(str(stem) + "-raw.npz", video=result.video, audio=result.audio,
                        sample_rate=result.sample_rate)
    print(f"raw data saved: {stem}-raw.npz", flush=True)

    from minimax_h3_mlx.packing import FPS  # noqa: E402
    from minimax_h3_mlx.media import save_mp4, save_wav  # noqa: E402

    # Signature: save_mp4(path, video, fps, audio, sample_rate) — fps is third; get that
    # wrong and the audio array lands in fps, the sample rate lands in audio, and the
    # failure only surfaces later, in save_wav.
    save_mp4(str(stem) + ".mp4", result.video, FPS, result.audio, result.sample_rate)
    save_wav(str(stem) + ".wav", result.audio, result.sample_rate)

    report = {
        "tag": args.tag,
        "canvas": f"{args.width}x{args.height}",
        "duration_seconds": args.duration,
        "grid_points": args.steps,
        "seed": args.seed,
        "open_seconds": round(t_open, 2),
        "generate_seconds": round(total, 1),
        "seconds_per_step": round(result.seconds_per_step, 1),
        "frames": int(result.video.shape[0]),
        "audio_samples": int(result.audio.shape[-1]),
        "phases": getattr(pipe, "phase_report", None),
    }
    (Path(str(stem) + ".json")).write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"\n=== DONE in {total / 60:.1f} min ===", flush=True)
    print(f"  {result.seconds_per_step:.1f}s per step, {report['frames']} frames", flush=True)
    print(f"  video:  {stem}.mp4", flush=True)
    print(f"  report: {stem}.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
