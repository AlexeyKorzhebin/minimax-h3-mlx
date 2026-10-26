"""Bake an AdaLN cache for any step count from the pruned base's 8-dim time curve.

This fork could only ever run `--steps 31`, because mere.run shipped a table baked for that one
grid and dropped the modulation path (`time_embedder` + 50 `adaln_proj`) that would let any other
grid be computed. Comfy-Org's pruned base keeps that path in a folded form — an `adaln_t_table` of
1025 rows by 8, plus per-block projections from those 8 dims — which is 87.3 MB rather than the
26 GB the full path would cost, and reproduces this build's own baked table to 1.8e-3 .. 6.0e-3
relative error (against 0.165 for its Q4 weights, so two orders of magnitude below the noise the
run already carries).

Output is byte-compatible with `adaln_cache.safetensors`, so nothing downstream needs to change:
`supported_num_inference_steps()` reads the sigma count straight out of it.
"""
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

CURVE = Path.home() / "models/turbo/adaln_curve.safetensors"
#: silu(time_embedder(t)) on linspace(0, 1, 1025), bundled with Larryvrh's ComfyUI node. The Turbo
#: LoRA's AdaLN update lives in this 2688-dim space, which the pruned base folded away — so the
#: grid is what makes that half of the LoRA applicable at all.
SILU_GRID = Path.home() / "models/turbo/h3_silu_temb_grid.safetensors"
NUM_BLOCKS = 50


def _interp(table: np.ndarray, t: float) -> np.ndarray:
    pos = float(np.clip(t, 0.0, 1.0)) * (table.shape[0] - 1)
    i = min(int(np.floor(pos)), table.shape[0] - 2)
    return table[i] + (pos - i) * (table[i + 1] - table[i])


def beta_sigmas(steps: int, shift: float, alpha: float = 0.6, beta: float = 0.6):
    """ComfyUI's `beta` scheduler, which is what people actually run this LoRA with.

    Places sigmas by the beta distribution's quantile function rather than uniformly, which
    concentrates steps at both ends of the trajectory. That matters here: this fork measured that
    the last third of the schedule carries 86% of the denoising, and `simple` spends its final
    step on one jump from 0.667 straight to 0 — beta splits that into 0.719 -> 0.421 -> 0.
    """
    from scipy.stats import beta as B

    ts = 1.0 - np.linspace(0, 1, steps, endpoint=False)
    base = B.ppf(ts, alpha, beta)
    shifted = shift * base / (1.0 + (shift - 1.0) * base)
    return np.concatenate([shifted, [0.0]]).astype(np.float32)


def tail_split_sigmas(steps: int, shift: float, split: int):
    """`simple`, with only its last interval subdivided into ``split`` Euler steps.

    Every sigma before the split is **bit-identical** to `simple`'s, because the same float32
    `linspace` and the same shift produce them. That is the whole reason this schedule exists:
    a run on it reproduces a `simple` run of the same seed exactly up to sigma 0.667, so what
    differs in the output is the tail and nothing else. Comparing 8 steps against 16 cannot say
    that — a different grid from step one sends the trajectory somewhere else, and the two clips
    disagree on composition long before they disagree on detail.

    The subdivision is placed in the *unshifted* domain and then shifted, so the extra points sit
    where the schedule's own geometry puts them rather than at arbitrary sigma values.
    """
    from minimax_h3_mlx.scheduler import _linspace_1_to_0

    if split < 1:
        raise ValueError(f"`split` must be at least 1 (1 leaves `simple` untouched), got {split}.")
    base = _linspace_1_to_0(int(steps))
    last = base[-2]                       # the base sigma the final jump starts from
    extra = (last * np.arange(split - 1, 0, -1, dtype=np.float32) / np.float32(split))
    base = np.concatenate([base[:-1], extra, np.array([0.0], np.float32)]).astype(np.float32)
    shift32 = np.float32(shift)
    shifted = (shift32 * base) / (np.float32(1.0) + np.float32(shift - 1.0) * base)
    values: list[float] = []
    for v in shifted.tolist():            # the shift collides float32 neighbours near sigma 1
        if not values or v != values[-1]:
            values.append(v)
    return np.array(values, np.float32)


def bake(steps: int, dest: Path, shift_video: float = 12.0, shift_audio: float = 3.0,
         lora: Path | None = None, strength: float = 1.0, schedule: str = "simple",
         tail_split: int = 1):
    """Bake the table, optionally folding in the Turbo LoRA's AdaLN half.

    Folding is legitimate *here* and nowhere else: the table is bf16, not the 4-bit weights the
    backbone lives in. Merging a LoRA into 4-bit storage would requantize away 99.7% of the update
    (its rms is 0.3% of one quantization step), which is why the backbone half must stay a runtime
    side path. The table has no such problem, so its half costs nothing at run time.
    """
    from h3_48gb import _upstream  # noqa: F401
    from minimax_h3_mlx.scheduler import MiniMaxH3Scheduler

    video = MiniMaxH3Scheduler(shift=shift_video)
    audio = MiniMaxH3Scheduler(shift=shift_audio)
    if schedule == "beta":
        vs, as_ = beta_sigmas(steps, shift_video), beta_sigmas(steps, shift_audio)
        video.set_timesteps(sigmas=vs.tolist())
        audio.set_timesteps(sigmas=as_.tolist())
    elif schedule == "tail-split":
        vs = tail_split_sigmas(steps, shift_video, tail_split)
        as_ = tail_split_sigmas(steps, shift_audio, tail_split)
        video.set_timesteps(sigmas=vs.tolist())
        audio.set_timesteps(sigmas=as_.tolist())
    else:
        video.set_timesteps(steps)
        audio.set_timesteps(steps)
        vs = np.array(video.sigmas.tolist(), np.float32)
        as_ = np.array(audio.sigmas.tolist(), np.float32)
    forwards = len(vs) - 1                    # the grid spends one point on the terminal sigma
    print(f"schedule={schedule}: {len(vs)} sigmas, {forwards} forwards")

    curve = mx.load(str(CURVE))
    table = np.array(curve["adaln_t_table"], np.float32)
    out = {"video_sigmas": mx.array(vs), "audio_sigmas": mx.array(as_)}

    lora_w = grid = None
    if lora is not None:
        lora_w = mx.load(str(lora))
        grid = np.array(mx.load(str(SILU_GRID))["silu_t_emb_grid"].astype(mx.float32))
        print(f"folding {lora.name} at strength {strength}")

    # The nine slices of a baked row are three modulation variants at three times: the video
    # clock, the audio clock, and the conditioning anchor pinned at 0.999.
    times = np.empty((forwards, 3), np.float32)
    for s in range(forwards):
        times[s] = (1.0 - vs[s], 1.0 - as_[s], max(1.0 - vs[s], 0.999))
    rows = np.stack([[_interp(table, t) for t in step] for step in times])   # [S, 3, 8]
    silu = (np.stack([[_interp(grid, t) for t in step] for step in times])
            if grid is not None else None)                                   # [S, 3, 2688]

    def lora_delta(prefix: str, clock: int, step: int):
        """B @ A @ silu(t_emb) — the LoRA's contribution to this projection at this time."""
        if lora_w is None:
            return 0.0
        a = np.array(lora_w[f"{prefix}.lora_A.weight"].astype(mx.float32))
        b = np.array(lora_w[f"{prefix}.lora_B.weight"].astype(mx.float32))
        return strength * (b @ (a @ silu[step, clock]))

    for block in range(NUM_BLOCKS):
        w = np.array(curve[f"blocks.{block}.adaln_proj.linear.weight"].astype(mx.float32))
        b = np.array(curve[f"blocks.{block}.adaln_proj.linear.bias"].astype(mx.float32))
        width = w.shape[0] // 3
        mods = np.empty((forwards, 9, width), np.float32)
        for s in range(forwards):
            for clock in range(3):
                value = w @ rows[s, clock] + b
                value = value + lora_delta(f"blocks.{block}.adaln_proj.linear", clock, s)
                mods[s, clock * 3:(clock + 1) * 3] = value.reshape(3, width)
        out[f"blocks.{block}.modulations"] = mx.array(mods).astype(mx.bfloat16)

    # `final_modulations` is [steps, 3, 2 * hidden]: the 3 is the timestep variant
    # (video, audio, conditioning) — the same axis as the blocks' — and each variant carries the
    # FULL shift+scale vector. An earlier version evaluated only the video clock and then sliced
    # its 10752 outputs into three chunks of 3584, which is neither the right clock for variants
    # 1 and 2 nor the right width for any of them. It did not raise: the output layer indexes this
    # table by timestep alone, so a wrong-width row is silently read as a valid one.
    w = np.array(curve["final_layer.adaln_proj.linear.weight"].astype(mx.float32))
    b = np.array(curve["final_layer.adaln_proj.linear.bias"].astype(mx.float32))
    final = np.stack([
        np.stack([w @ rows[s, clock] + b
                  + lora_delta("final_layer.adaln_proj.linear", clock, s)
                  for clock in range(3)])
        for s in range(forwards)])
    out["final_modulations"] = mx.array(final).astype(mx.bfloat16)

    dest.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(dest), out)
    # `len(vs)`, not `steps`: tail-split and beta both add points, and the grid count is what
    # `--steps` has to be given as, so printing the argument back would misreport it.
    print(f"{dest} — {len(vs)} grid points ({forwards} forwards), {dest.stat().st_size/1e6:.0f} MB")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("steps", type=int, nargs="?", default=8)
    ap.add_argument("--lora", type=Path, default=None)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--schedule", choices=("simple", "beta", "tail-split"), default="simple")
    ap.add_argument("--tail-split", type=int, default=2,
                    help="with --schedule tail-split: how many Euler steps replace the final jump")
    a = ap.parse_args()
    suffix = ("_turbo" if a.lora else "")
    if a.schedule == "beta":
        suffix += "_beta"
    elif a.schedule == "tail-split":
        suffix += f"_tail{a.tail_split}"
    bake(a.steps, a.out or Path.home() / f"models/turbo/adaln_cache_{a.steps}{suffix}.safetensors",
         lora=a.lora, strength=a.strength, schedule=a.schedule, tail_split=a.tail_split)
