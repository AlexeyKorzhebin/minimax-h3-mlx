#!/usr/bin/env python3
"""Walk the whole image-to-video path in minutes instead of half an hour.

Three blockers on the `--image` path were found one at a time, each hidden behind the one
before it, and each cost a full 25-minute run to surface. This script exists so the fourth
one costs a few minutes: it runs the *real* pipeline end to end — encoder, keyframe encode,
packing, forward, decode — but denoises only a couple of steps instead of thirty.

**It executes the shipped code, not a copy of it.** Every earlier attempt to shortcut a run
by re-implementing the loop would have verified the copy, which is exactly the class of test
this project keeps getting burned by. So the only thing overridden here is where the loop
stops.

Why the loop can be cut but the schedule cannot: `AdaLNCacheFile.check_schedule` compares the
sigma grid elementwise and refuses anything but the baked 31 points, because serving the
nearest row instead would silently denoise on the wrong modulation. So the schedule is built
in full, the cache is built in full, and only the *timestep list the loop iterates* is
truncated afterwards — inside `_ensure_cache`, the one seam between plan construction and the
loop. Sigmas stay untouched, so step i still moves sigma[i] -> sigma[i+1]; the run simply
stops partway rather than taking a distorted final jump to zero.

WHAT THIS PROVES: every shape, layout and dtype on the i2v path is consistent, and no stage
raises. That is what the last three failures were.

WHAT THIS CANNOT PROVE: that conditioning actually *works*. The clip is deliberately left
half-denoised, so its frames mean nothing. Whether the keyframe steers the result is
`verify_i2v.py`'s job, and it needs the full 31 steps and its unconditioned control.

Run the control too:

    ./.venv/bin/python scripts/canary_i2v.py              # with a keyframe
    ./.venv/bin/python scripts/canary_i2v.py --no-image   # same path, text only

If both fail the same way, the bug is not in the i2v path — it is in this script or in the
run's geometry. That comparison is the point of `--no-image`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import mlx.core as mx  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

from h3_48gb.pipeline import LazyMiniMaxH3Pipeline  # noqa: E402

DEFAULT_CHECKPOINT = Path.home() / "models/h3-converted"
PROMPT = "a red vintage car parked on a wet street at night, neon reflections"


class ShapeProbe:
    """Print the shapes crossing into the first forward, then get out of the way.

    Wraps the DiT proxy rather than replacing it: `_ensure_cache` reaches for
    `dit.final_layer` and the geometry code for `dit.config`, so everything not defined here
    forwards to the real component.
    """

    def __init__(self, inner, milestones: dict):
        self._inner = inner
        self._milestones = milestones
        self._calls = 0

    def __getattr__(self, item):
        return getattr(self._inner, item)

    def __call__(self, video_rows, audio_rows, embeds, *args, **kwargs):
        self._calls += 1
        if self._calls == 1:
            print(f"[canary] first forward: video {tuple(video_rows.shape)}  "
                  f"audio {tuple(audio_rows.shape)}  embeds {tuple(embeds.shape)}", flush=True)
            self._milestones["first_forward"] = True
        return self._inner(video_rows, audio_rows, embeds, *args, **kwargs)


class CanaryPipeline(LazyMiniMaxH3Pipeline):
    """The real pipeline, stopped early, with a milestone recorded at each stage boundary.

    The overrides call `super()` and record — none of them reimplements a stage. If a stage
    raises, its milestone is simply absent, which is how the report names the failing stage
    without parsing the traceback.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.canary_steps = 2
        self.milestones: dict[str, object] = {}
        self.timings: dict[str, float] = {}

    def _stage(self, name: str, fn):
        started = time.perf_counter()
        result = fn()
        self.timings[name] = round(time.perf_counter() - started, 1)
        self.milestones[name] = True
        print(f"[canary] {name} ok in {self.timings[name]}s "
              f"(peak {mx.get_peak_memory() / 1e9:.2f} GB)", flush=True)
        return result

    def _encode_keyframes(self, images, height, width):
        rows = self._stage("keyframe_encode", lambda: super(CanaryPipeline, self)._encode_keyframes(
            images, height, width))
        print(f"[canary]   conditioning rows: {tuple(rows.shape)}", flush=True)
        self.milestones["condition_rows"] = tuple(rows.shape)
        return rows

    def _ensure_cache(self, timestep_table, drop_adaln, verbose):
        """Build the full cache the checked schedule requires, then shorten only the loop."""
        super()._ensure_cache(timestep_table, drop_adaln, verbose)
        self.milestones["adaln_cache"] = True

        video, audio = self._schedules
        full = int(video.timesteps.shape[0])
        if self.canary_steps >= full:
            return
        video.timesteps = video.timesteps[:self.canary_steps]
        audio.timesteps = audio.timesteps[:self.canary_steps]
        print(f"[canary] denoising truncated to {self.canary_steps} of {full} steps "
              f"(sigma grid untouched, so the cache stays valid)", flush=True)

    def _decode_video(self, rows, *args, **kwargs):
        video = self._stage("video_decode", lambda: super(CanaryPipeline, self)._decode_video(
            rows, *args, **kwargs))
        self.milestones["video_shape"] = tuple(video.shape)
        return video

    def _decode_audio(self, rows, *args, **kwargs):
        audio = self._stage("audio_decode", lambda: super(CanaryPipeline, self)._decode_audio(
            rows, *args, **kwargs))
        self.milestones["audio_shape"] = tuple(audio.shape)
        return audio


def make_keyframe(path: Path, width: int, height: int) -> None:
    """A frame with structure a generator would not produce by chance."""
    from verify_i2v import make_keyframe as build

    build(path)
    if (width, height) != (512, 512):
        Image.open(path).resize((width, height), Image.BICUBIC).save(path)


def load_image(path: Path):
    """Same load the CLI performs, EXIF orientation included."""
    with Image.open(path) as raw:
        return ImageOps.exif_transpose(raw).convert("RGB")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--image", type=Path, default=None,
                    help="keyframe to condition on; a synthetic one is generated if omitted")
    ap.add_argument("--no-image", action="store_true",
                    help="control run: the identical path with no keyframe at all")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--duration", type=float, default=2.4)
    ap.add_argument("--canary-steps", type=int, default=2,
                    help="forwards to run before decoding (default 2)")
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--report", type=Path, default=None, help="write the JSON report here too")
    args = ap.parse_args()

    outdir = Path.home() / "models/video-out/canary"
    outdir.mkdir(parents=True, exist_ok=True)

    images = None
    anchors: tuple[str, ...] = ()
    if not args.no_image:
        path = args.image
        if path is None:
            path = outdir / f"keyframe-{args.width}x{args.height}.png"
            make_keyframe(path, args.width, args.height)
        # The CLI refuses an unpatched checkout before loading anything; the canary bypasses the
        # CLI, so without this it would spend 20 s reaching a failure the CLI names instantly.
        from h3_48gb.text_encoder import keyframe_scatter_patch_applied

        if not keyframe_scatter_patch_applied():
            print("[canary] upstream/ is unpatched — a keyframe run would die inside the encoder. "
                  "Apply patches/0001-keyframe-masked-scatter.patch (see README).")
            return 1

        images = [load_image(path)]
        anchors = ("first",)
        print(f"[canary] keyframe: {path} {images[0].size}", flush=True)
    else:
        print("[canary] control run: no keyframe", flush=True)

    mx.reset_peak_memory()
    started = time.perf_counter()

    pipe = CanaryPipeline.from_pretrained(args.checkpoint, verbose=True)
    pipe.canary_steps = args.canary_steps
    steps = pipe.supported_num_inference_steps() or 31

    # The encoder is reached through the lazy proxy's `encode`; wrap that one method so the
    # milestone lands even though the proxy is not a CanaryPipeline.
    encoder = pipe.text_encoder
    inner_encode = encoder.encode

    def encode(prompt, imgs=None):
        return pipe._stage("text_encode", lambda: inner_encode(prompt, imgs))

    encoder.encode = encode
    pipe.dit = ShapeProbe(pipe.dit, pipe.milestones)

    failure = None
    try:
        pipe(prompt=PROMPT, duration_seconds=args.duration, num_inference_steps=steps,
             seed=args.seed, height=args.height, width=args.width,
             images=images, keyframe_anchors=anchors, verbose=True)
        pipe.milestones["complete"] = True
    except BaseException as exc:  # noqa: BLE001 - the failure is the deliverable
        failure = {"type": type(exc).__name__, "message": str(exc)[:600]}
        traceback.print_exc()

    report = {
        "mode": "control (no keyframe)" if args.no_image else "conditioned",
        "canvas": f"{args.width}x{args.height}",
        "duration_seconds": args.duration,
        "forwards_run": args.canary_steps,
        "reached": {k: v for k, v in pipe.milestones.items()},
        "stage_seconds": pipe.timings,
        "total_seconds": round(time.perf_counter() - started, 1),
        "peak_gb": round(mx.get_peak_memory() / 1e9, 2),
        "failure": failure,
    }
    # Name the stage that did not happen. The order is the pipeline's own.
    expected = ["text_encode", "keyframe_encode", "adaln_cache", "first_forward",
                "video_decode", "audio_decode", "complete"]
    if args.no_image:
        expected.remove("keyframe_encode")
    report["first_missing_stage"] = next((s for s in expected if s not in pipe.milestones), None)

    print("\n" + json.dumps(report, indent=2), flush=True)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2))

    # A stage can be skipped without anything raising — a pipeline that quietly dropped the
    # keyframe completes happily, and reporting that as "structurally sound" would be the exact
    # false negative this script exists to prevent.
    if failure is None and report["first_missing_stage"] is not None:
        missing = report["first_missing_stage"]
        print(f"\n[canary] the run completed but never reached `{missing}`. Nothing raised, so "
              "some stage was skipped rather than failing — that is a defect, not a pass.")
        return 1

    if failure is None:
        if args.no_image:
            # Saying "the i2v path is sound" here would be false: this run carried no keyframe.
            # All a clean control proves is that the canary itself reaches the end.
            print("\n[canary] the control reached the end, so the canary itself is sound. It "
                  "exercised no keyframe code at all — run without --no-image for that.")
        else:
            print("\n[canary] the i2v path is structurally sound. This says nothing about whether "
                  "the keyframe steers the clip — that needs verify_i2v.py and its control.")
        return 0
    print(f"\n[canary] BLOCKED at: {report['first_missing_stage']}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
