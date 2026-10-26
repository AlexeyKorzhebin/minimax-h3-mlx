#!/usr/bin/env python3
"""Prove the cached AdaLN table is addressed exactly the way the port addresses its own.

    ./.venv/bin/python test_adaln_indexing.py

A wrong ``(step, variant, modality)`` mapping does not raise anything — it denoises on somebody
else's modulation and returns a plausible, wrong clip. So this does not test the adapter against a
restatement of its own assumptions; it builds a **reference** with the port's real
`AdaLayerNormModulation` and a synthetic mere.run cache from the *same weights*, and requires the
two tables to be identical element for element.

Then it does the same with the layout deliberately wrong (modality-major instead of variant-major,
and video/audio swapped) and requires detection, so a no-op adapter cannot pass.

Everything runs on an 8-wide, 2-block toy at float32 — no checkpoint, no MLX device pressure.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.mlx
pytest.importorskip("mlx.core", reason="mlx: needs the MLX stack, absent here")


import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from h3_48gb import _upstream  # noqa: E402,F401

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

from minimax_h3_mlx.config import MODALITY_NUM, DiTConfig  # noqa: E402
from minimax_h3_mlx.dit import AdaLayerNormModulation, timestep_embedding  # noqa: E402
from minimax_h3_mlx.packing import KEYFRAME_NOISE_AUG  # noqa: E402
from minimax_h3_mlx.scheduler import MiniMaxH3Scheduler  # noqa: E402

from h3_48gb.adaln import (  # noqa: E402
    ROWS_PER_STEP,
    VARIANT_AUDIO,
    VARIANT_CONDITION,
    VARIANT_VIDEO,
    AdaLNCacheFile,
    ScheduleMismatch,
    final_rows,
    modulation_rows,
    timestep_sources,
)

HIDDEN, BLOCKS, TIME_EMBED_DIM, TIME_INPUT_DIM = 8, 2, 6, 4
GRID = 6  # sigma grid points -> GRID - 1 forwards

CONFIG = DiTConfig(
    hidden_size=HIDDEN,
    num_layers=BLOCKS,
    time_embed_dim=TIME_EMBED_DIM,
    timestep_input_dim=TIME_INPUT_DIM,
    adaln_out_features=6 * MODALITY_NUM * HIDDEN,
    final_adaln_out_features=2 * HIDDEN,
)


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name} FAILED {detail}")
    print(f"  ok  {name}")


# -- the toy model ------------------------------------------------------------------------------

def build_toy():
    """A timestep MLP plus one `AdaLayerNormModulation` per block, and the output projection.

    These stand in for the tensors mere.run deleted. Both the reference and the synthetic cache are
    produced from *these* weights, so any disagreement is an indexing bug and nothing else.
    """
    mx.random.seed(11)

    def embedder(t: mx.array) -> mx.array:
        # Any injective-enough map from timestep to embedding; the real one is a two-layer MLP.
        sinusoid = timestep_embedding(t, TIME_INPUT_DIM)
        return nn.silu(sinusoid @ _EMB_W) + _EMB_B

    blocks = [AdaLayerNormModulation(CONFIG) for _ in range(BLOCKS)]
    final = nn.Linear(TIME_EMBED_DIM, 2 * HIDDEN, bias=True)
    mx.eval([b.parameters() for b in blocks], final.parameters())
    return embedder, blocks, final


_EMB_W = mx.random.normal((TIME_INPUT_DIM, TIME_EMBED_DIM), key=mx.random.key(3))
_EMB_B = mx.random.normal((TIME_EMBED_DIM,), key=mx.random.key(4))


def reference_tables(embedder, blocks, final, table: mx.array):
    """What `ModulationCache.build` would produce for a global timestep table."""
    temb = embedder(table)
    tables = [tuple(t.astype(mx.float32) for t in block(temb)) for block in blocks]
    h = final(nn.silu(temb))
    return tables, h[:, :HIDDEN].astype(mx.float32), h[:, HIDDEN:].astype(mx.float32)


def write_cache(path: Path, embedder, blocks, final, video_sched, audio_sched,
                variant_major: bool = True, swap_video_audio: bool = False) -> None:
    """Write a synthetic ``adaln_cache.safetensors`` in mere.run's per-step form.

    mere.run evaluates the projection on the step's three timestep variants at once and reshapes
    ``(3, 6*MODALITY_NUM*hidden)`` to ``(9, 6*hidden)`` — which makes the row index
    ``variant * MODALITY_NUM + modality``. ``variant_major=False`` writes the transpose instead,
    to prove the test can tell the difference.
    """
    steps = int(video_sched.timesteps.shape[0])
    vt = video_sched.timesteps.tolist()
    at = audio_sched.timesteps.tolist()

    per_step_t = []
    for i in range(steps):
        levels = [vt[i], at[i], max(vt[i], KEYFRAME_NOISE_AUG)]
        if swap_video_audio:
            levels[0], levels[1] = levels[1], levels[0]
        per_step_t.append(levels)
    t_all = mx.array(np.array(per_step_t, dtype=np.float32).reshape(-1))  # (steps*3,)
    temb = embedder(t_all)                                               # (steps*3, D)

    tensors: dict[str, mx.array] = {}
    for index, block in enumerate(blocks):
        rows = mx.concatenate(block(temb), axis=-1)  # (steps*3*MODALITY_NUM, 6*hidden)
        rows = rows.reshape(steps, ROWS_PER_STEP, 6 * HIDDEN)
        if not variant_major:
            rows = rows.reshape(steps, MODALITY_NUM, MODALITY_NUM, 6 * HIDDEN)
            rows = rows.transpose(0, 2, 1, 3).reshape(steps, ROWS_PER_STEP, 6 * HIDDEN)
        tensors[f"blocks.{index}.modulations"] = rows.astype(mx.bfloat16)

    h = final(nn.silu(temb)).reshape(steps, MODALITY_NUM, 2 * HIDDEN)
    tensors["final_modulations"] = h.astype(mx.bfloat16)
    tensors["time_embeddings"] = temb.reshape(steps, MODALITY_NUM, TIME_EMBED_DIM)
    tensors["video_sigmas"] = video_sched.sigmas
    tensors["audio_sigmas"] = audio_sched.sigmas
    mx.save_safetensors(str(path), tensors, metadata={"format": "mlx"})


def schedules(grid: int = GRID):
    video = MiniMaxH3Scheduler(shift=12.0)
    audio = MiniMaxH3Scheduler(shift=3.0)
    video.set_timesteps(grid)
    audio.set_timesteps(grid)
    return video, audio


def global_table(video_sched, audio_sched) -> mx.array:
    """The table `_row_timestep_plan` builds: every distinct level the run presents, sorted."""
    values = set()
    for t, at in zip(video_sched.timesteps.tolist(), audio_sched.timesteps.tolist()):
        values.add(float(np.float32(t)))
        values.add(float(np.float32(at)))
        values.add(float(np.float32(max(t, KEYFRAME_NOISE_AUG))))
    return mx.array(np.array(sorted(values), dtype=np.float32))


# -- tests --------------------------------------------------------------------------------------

def test_row_index_arithmetic() -> None:
    """The gather index, spelled out against hand-computed slots."""
    video, audio = schedules()
    sources = timestep_sources(video, audio)
    vt = [float(np.float32(t)) for t in video.timesteps.tolist()]
    at = [float(np.float32(t)) for t in audio.timesteps.tolist()]

    check("video step 2 resolves to (2, VARIANT_VIDEO)",
          sources[vt[2]][0] == (2, VARIANT_VIDEO), f"got {sources[vt[2]]}")
    check("audio step 3 resolves to (3, VARIANT_AUDIO)",
          any(slot == (3, VARIANT_AUDIO) for slot in sources[at[3]]), f"got {sources[at[3]]}")
    check("the conditioning level resolves to variant 2",
          sources[float(np.float32(KEYFRAME_NOISE_AUG))][0][1] == VARIANT_CONDITION)

    table = [vt[2]]
    rows = modulation_rows(table, sources)
    base = 2 * ROWS_PER_STEP + VARIANT_VIDEO * MODALITY_NUM
    check("one video timestep expands to its three modality rows",
          rows.tolist() == [base, base + 1, base + 2], f"got {rows.tolist()}")

    frows = final_rows([at[3]], sources)
    check("the output-layer table has no modality axis",
          frows.tolist() == [3 * MODALITY_NUM + VARIANT_AUDIO], f"got {frows.tolist()}")

    # The port reads row `timestep_index * MODALITY_NUM + tag`; the gather must place the modality
    # for tag t at exactly that offset.
    two = modulation_rows([vt[1], at[1]], sources)
    for k, expected_slot in enumerate([(1, VARIANT_VIDEO), (1, VARIANT_AUDIO)]):
        for tag in range(MODALITY_NUM):
            want = expected_slot[0] * ROWS_PER_STEP + expected_slot[1] * MODALITY_NUM + tag
            check(f"table row {k}, tag {tag} -> cache row {want}",
                  two[k * MODALITY_NUM + tag] == want, f"got {two[k * MODALITY_NUM + tag]}")


def test_matches_the_ports_own_table() -> None:
    """The whole point: cache-served tables must equal what the port would have computed."""
    embedder, blocks, final = build_toy()
    video, audio = schedules()
    table = global_table(video, audio)
    ref_tables, ref_shift, ref_scale = reference_tables(embedder, blocks, final, table)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "adaln_cache.safetensors"
        write_cache(path, embedder, blocks, final, video, audio)
        cache = AdaLNCacheFile(path).build(table, video, audio, dtype=mx.float32)

    check("the cache covers every block", len(cache.tables) == BLOCKS, f"got {len(cache.tables)}")
    check("timestep count matches", cache.num_timesteps == table.shape[0])

    worst = 0.0
    for index in range(BLOCKS):
        got, want = cache.get(index), ref_tables[index]
        check(f"block {index} yields six modulation tensors", len(got) == 6)
        for param in range(6):
            check(f"block {index} param {param} shape",
                  got[param].shape == want[param].shape,
                  f"{got[param].shape} vs {want[param].shape}")
            diff = float(mx.abs(got[param] - want[param]).max())
            worst = max(worst, diff)
    # bfloat16 is the storage of the real cache; the reference is float32, so allow the rounding.
    check(f"every block's modulation matches the port's (max diff {worst:.2e})", worst < 5e-2)

    for label, got, want in (("shift", cache.final_shift, ref_shift),
                             ("scale", cache.final_scale, ref_scale)):
        check(f"output-layer {label} shape", got.shape == want.shape, f"{got.shape} vs {want.shape}")
        check(f"output-layer {label} matches",
              float(mx.abs(got - want).max()) < 5e-2,
              f"max diff {float(mx.abs(got - want).max()):.2e}")


def test_detects_a_transposed_layout() -> None:
    """Write the cache modality-major and require the adapter to notice."""
    embedder, blocks, final = build_toy()
    video, audio = schedules()
    table = global_table(video, audio)
    ref_tables, _, _ = reference_tables(embedder, blocks, final, table)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "adaln_cache.safetensors"
        write_cache(path, embedder, blocks, final, video, audio, variant_major=False)
        try:
            cache = AdaLNCacheFile(path).build(table, video, audio, dtype=mx.float32)
        except ScheduleMismatch as exc:
            check("modality-major layout is rejected by the duplicate check",
                  "layout assumed" in str(exc), str(exc))
            return
    worst = max(float(mx.abs(cache.get(i)[p] - ref_tables[i][p]).max())
                for i in range(BLOCKS) for p in range(6))
    check(f"modality-major layout produces a different table (max diff {worst:.2e})", worst > 1e-2)


def test_detects_swapped_video_audio() -> None:
    """Write the cache with the video and audio variants exchanged and require detection."""
    embedder, blocks, final = build_toy()
    video, audio = schedules()
    table = global_table(video, audio)
    ref_tables, _, _ = reference_tables(embedder, blocks, final, table)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "adaln_cache.safetensors"
        write_cache(path, embedder, blocks, final, video, audio, swap_video_audio=True)
        try:
            cache = AdaLNCacheFile(path).build(table, video, audio, dtype=mx.float32)
        except ScheduleMismatch as exc:
            check("swapped variants are rejected by the duplicate check", True, str(exc))
            return
    worst = max(float(mx.abs(cache.get(i)[p] - ref_tables[i][p]).max())
                for i in range(BLOCKS) for p in range(6))
    check(f"swapped video/audio produces a different table (max diff {worst:.2e})", worst > 1e-2)


def test_rejects_the_wrong_schedule() -> None:
    """A table baked for one schedule must refuse every other, rather than serve a near row."""
    embedder, blocks, final = build_toy()
    video, audio = schedules()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "adaln_cache.safetensors"
        write_cache(path, embedder, blocks, final, video, audio)

        other_v, other_a = schedules(GRID + 2)
        try:
            AdaLNCacheFile(path).check_schedule(other_v, other_a)
            raise AssertionError("a different step count was accepted")
        except ScheduleMismatch as exc:
            check("a different step count raises", "evaluations" in str(exc), str(exc))

        shifted = MiniMaxH3Scheduler(shift=7.0)
        shifted.set_timesteps(GRID)
        try:
            AdaLNCacheFile(path).check_schedule(shifted, audio)
            raise AssertionError("a different sigma shift was accepted")
        except ScheduleMismatch as exc:
            check("a different sigma shift raises", "shift" in str(exc), str(exc))

        # The reference-audio conditioning level (1.0) is the one the cache genuinely cannot serve.
        table = mx.array(np.array([1.0], dtype=np.float32))
        try:
            AdaLNCacheFile(path).build(table, video, audio, dtype=mx.float32)
            raise AssertionError("an unavailable timestep was accepted")
        except ScheduleMismatch as exc:
            check("an unavailable timestep raises", "does not hold" in str(exc), str(exc))


#: Where a real ``adaln_cache.safetensors`` may live. `H3_ADALN_CACHE` overrides.
REAL_CACHE_CANDIDATES = (
    Path.home() / "models/h3-converted/transformer/adaln_cache.safetensors",
    Path.home() / ("Library/Application Support/MereRun/models/"
                   "video-minimax-h3-fl2va-mlx/adaln_cache.safetensors"),
)


def find_real_cache() -> Path | None:
    import os

    override = os.environ.get("H3_ADALN_CACHE")
    candidates = (Path(override),) if override else REAL_CACHE_CANDIDATES
    return next((p for p in candidates if p.exists()), None)


def test_real_cache_layout() -> None:
    """The shipped cache, if present: variant order and the step-0 collapse that proves it."""
    source = find_real_cache()
    if source is None:
        print("  --  no real cache found, skipping (set H3_ADALN_CACHE to point at one)")
        return
    print(f"  --  using {source}")

    raw = mx.load(str(source))
    te = np.array(raw["time_embeddings"])
    vs = np.array(raw["video_sigmas"])
    as_ = np.array(raw["audio_sigmas"])

    check("variant 2 is a constant level",
          float(np.abs(te[:, 2] - te[0, 2]).max()) == 0.0)
    check("variants 0 and 1 coincide at step 0 (both sigmas are 1.0)",
          vs[0] == as_[0] == 1.0 and float(np.abs(te[0, 0] - te[0, 1]).max()) == 0.0)

    # Where the two schedules meet at a shared sigma, variant 0 must be the video one.
    meets = [(i, j) for i in range(len(vs) - 1) for j in range(len(as_) - 1)
             if vs[i] == as_[j] and i != j]
    check("the schedules share at least one sigma away from step 0", bool(meets), f"{meets}")
    for i, j in meets:
        same = float(np.abs(te[i, 0] - te[j, 1]).max())
        swapped = float(np.abs(te[i, 1] - te[j, 0]).max())
        check(f"sigma {vs[i]:.6f}: variant 0 is video, variant 1 is audio",
              same == 0.0 and swapped > 0.0, f"same={same}, swapped={swapped}")

    mod = np.array(raw["blocks.0.modulations"][0].astype(mx.float32))
    equal = [[float(np.abs(mod[a] - mod[b]).max()) == 0.0 for b in range(9)] for a in range(9)]
    groups = sorted(tuple(b for b in range(9) if equal[a][b]) for a in range(9))
    # Variants 0 and 1 share a timestep at step 0, so rows differing only in variant collapse.
    check("step-0 rows collapse as variant*3 + modality predicts",
          (0, 3) in groups and (1, 4) in groups and (2, 5) in groups,
          f"groups {sorted(set(groups))}")
    check("step-0 rows do NOT collapse as modality*3 + variant would predict",
          (0, 1, 2) not in groups and (3, 4, 5) not in groups,
          f"groups {sorted(set(groups))}")


def main() -> int:
    tests = [
        test_row_index_arithmetic,
        test_matches_the_ports_own_table,
        test_detects_a_transposed_layout,
        test_detects_swapped_video_audio,
        test_rejects_the_wrong_schedule,
        test_real_cache_layout,
    ]
    for test in tests:
        print(f"{test.__name__}:")
        test()
    print(f"\n{len(tests)} test groups passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
