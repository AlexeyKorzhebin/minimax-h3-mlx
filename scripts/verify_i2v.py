#!/usr/bin/env python3
"""Does a keyframe actually condition the clip, or is it silently ignored?

Runs the same prompt and seed twice at 512x512 — once with a keyframe, once without — and
compares each clip's first frame against the image. The control run is the point: if an
unconditioned clip scores as close to the image as the conditioned one, then what we measured
was agreement with the prompt, not conditioning.

About 50 minutes for the pair. Nothing here is imported by the package.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

OUT = Path.home() / "models/video-out/i2v-check"
PROMPT = "a red vintage car parked on a wet street at night, neon reflections"


def make_keyframe(path: Path) -> None:
    """A frame with structure a generator would not produce by chance."""
    img = Image.new("RGB", (512, 512), (18, 18, 28))
    px = img.load()
    for y in range(512):
        for x in range(512):
            if (x // 64 + y // 64) % 2 == 0:
                px[x, y] = (200, 40, 40)
    img.save(path)


def run(tag: str, image: Path | None, canvas: tuple[int, int], prompt: str,
        duration: float) -> Path:
    width, height = canvas
    cmd = ["./.venv/bin/python", "-m", "h3_48gb", "generate", prompt,
           "--width", str(width), "--height", str(height), "--duration", str(duration),
           "--steps", "31", "--seed", "20260807", "--tag", tag, "--outdir", str(OUT)]
    if image is not None:
        cmd += ["--image", str(image)]
    subprocess.run(cmd, check=True)
    return OUT / f"h3-{tag}-{width}x{height}.mp4"


def first_frame(clip: Path) -> np.ndarray:
    out = clip.with_suffix(".frame0.png")
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(clip), "-vframes", "1",
                    "-y", str(out)], check=True)
    return np.asarray(Image.open(out).convert("RGB"), dtype=np.float64)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(((a - b) ** 2).mean())
    return float("inf") if mse == 0 else 20 * np.log10(255.0) - 10 * np.log10(mse)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--image", type=Path, default=None,
                    help="keyframe to condition on; the synthetic checkerboard if omitted")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--duration", type=float, default=2.4)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--tag", default="i2v", help="prefix for both runs' tags")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    canvas = (args.width, args.height)
    if args.image is None:
        key = OUT / "keyframe.png"
        make_keyframe(key)
    else:
        key = args.image

    # Compare against the keyframe as the clip's first frame can possibly show it: on the canvas,
    # at the canvas size. Scoring a 1536x1024 source against a 576x384 frame would measure the
    # resize, not the conditioning.
    with Image.open(key) as raw:
        prepared = ImageOps.exif_transpose(raw).convert("RGB").resize(canvas, Image.LANCZOS)
    reference = np.asarray(prepared, dtype=np.float64)

    conditioned = first_frame(run(f"{args.tag}", key, canvas, args.prompt, args.duration))
    control = first_frame(run(f"{args.tag}-control", None, canvas, args.prompt, args.duration))

    report = {
        "conditioned_psnr": round(psnr(reference, conditioned), 2),
        "control_psnr": round(psnr(reference, control), 2),
        "conditioned_corr": round(float(np.corrcoef(reference.ravel(), conditioned.ravel())[0, 1]), 4),
        "control_corr": round(float(np.corrcoef(reference.ravel(), control.ravel())[0, 1]), 4),
    }
    report["verdict"] = (
        "conditioning works"
        if report["conditioned_psnr"] > report["control_psnr"] + 3
        else "INCONCLUSIVE: the keyframe did not move the first frame measurably"
    )
    (OUT / f"verdict-{args.tag}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["verdict"] == "conditioning works" else 1


if __name__ == "__main__":
    raise SystemExit(main())
