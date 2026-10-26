"""Serve the port's AdaLN modulation from mere.run's precomputed table.

The mere.run transformer omits the whole modulation path — 50 x ``blocks.N.adaln_proj``,
``final_layer.adaln_proj`` and ``time_embedder``, 106 tensors — because it precomputed the table
into ``adaln_cache.safetensors``. `ModulationCache.build` therefore cannot run, and this module
replaces it.

Layout of the cache (established from the data, not from documentation):

    blocks.{i}.modulations   bf16  [steps, 9, 6*hidden]
    final_modulations        bf16  [steps, 3, 2*hidden]
    time_embeddings          f32   [steps, 3, time_embed_dim]
    video_sigmas             f32   [steps + 1]
    audio_sigmas             f32   [steps + 1]

The ``3`` is a **timestep variant**, in the order ``(video, audio, conditioning)``; the ``9`` is
``variant * 3 + modality``. Both were proven rather than assumed:

* ``final_modulations`` has no modality axis at all — the port indexes the output layer's table by
  ``timestep_indices`` alone (`FinalLayer.norm_out`) — which fixes the meaning of the ``3``.
* At step 0 the video and audio sigmas are both exactly 1.0, so variants 0 and 1 share a timestep
  embedding (verified: ``max|te[0,0] - te[0,1]| == 0``). The 9 modulation rows then collapse into
  the equality pattern ``{0,3} {1,4} {2,5} {6} {7} {8}`` — which is what ``variant * 3 + modality``
  predicts, and *not* what ``modality * 3 + variant`` predicts (that would give ``{0,1} {3,4}
  {6,7}``).
* Variant 2 is bit-identical across all 30 steps, i.e. a fixed level: the conditioning
  ``max(t, KEYFRAME_NOISE_AUG) == 0.999``, which is the only constant level `build_row_timesteps`
  ever assigns (``condition_audio_timestep`` is dead in this build — `build_packed_sequence`
  hard-codes ``num_condition_audio_rows=0``).
* Video vs audio is settled by a collision: the two schedules share sigma ``0.857143`` at video
  step 20 and audio step 10, and ``te[20, 0] == te[10, 1]`` exactly while ``te[20, 1]`` vs
  ``te[10, 0]`` differ by 0.49.

That maps 1:1 onto the port, which reshapes ``adaln_proj``'s ``(T, 6*3*hidden)`` output to
``(T*3, 6*hidden)`` and addresses it with ``timestep_indices * MODALITY_NUM + token_tags``. Both
sides are the *same reshape of the same projection*, so ``modality`` and mere.run's inner index are
the same axis by construction.

The table is baked for one schedule. Rows are therefore located **structurally** — from the port's
own ``scheduler.timesteps`` arrays, by exact float32 equality — never by a nearest-value search, so
a schedule the cache cannot serve raises :class:`ScheduleMismatch` instead of quietly resolving to
the closest row.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import _upstream  # noqa: F401  (puts the vendored checkout on sys.path)

import mlx.core as mx

from minimax_h3_mlx.config import MODALITY_NUM
from minimax_h3_mlx.packing import KEYFRAME_NOISE_AUG

VARIANT_VIDEO, VARIANT_AUDIO, VARIANT_CONDITION = 0, 1, 2
VARIANT_NUM = 3
#: Rows per step in ``blocks.{i}.modulations``: one per (variant, modality) pair.
ROWS_PER_STEP = VARIANT_NUM * MODALITY_NUM

#: mere.run rebuilt the sigma grid in a slightly different order of operations, so its stored
#: sigmas differ from the port's by up to ~1.2e-7 (about one float32 ulp). The tightest gap between
#: two *distinct* schedule entries is 2.9e-3, so this tolerance cannot merge neighbours; it is
#: checked against the observed grid at load time anyway.
SIGMA_ATOL = 1e-6


class ScheduleMismatch(RuntimeError):
    """The requested sampling schedule is not the one the cached table was baked for."""


def _key(value: float) -> float:
    """Normalize a timestep to the float32 value the packing code will actually produce.

    `build_row_timesteps` writes every level into a float32 array, so the entries of the port's
    global timestep table are float32 values widened back to double. Keys must be built the same
    way or a Python-double level like ``KEYFRAME_NOISE_AUG`` will never match.
    """
    return float(np.float32(value))


class CachedModulation:
    """Drop-in replacement for `ModulationCache`, plus the output layer's table.

    Exposes exactly what `MiniMaxH3DiT.__call__` asks of a modulation cache — ``get(block_index)``
    returning the six modulation tensors — and additionally carries the ``final_layer`` shift and
    scale, which the stock code recomputes from ``adaln_proj`` weights this build does not have.
    """

    def __init__(
        self,
        tables: list[tuple[mx.array, ...]],
        timesteps: mx.array,
        final_shift: mx.array,
        final_scale: mx.array,
    ):
        self.tables = tables
        self.timesteps = timesteps
        self.final_shift = final_shift
        self.final_scale = final_scale

    def get(self, block_index: int) -> tuple[mx.array, ...]:
        return self.tables[block_index]

    @property
    def num_timesteps(self) -> int:
        return int(self.timesteps.shape[0])

    def nbytes(self) -> int:
        return (
            sum(t.nbytes for table in self.tables for t in table)
            + self.final_shift.nbytes
            + self.final_scale.nbytes
        )


def timestep_sources(video_sched, audio_sched) -> dict[float, list[tuple[int, int]]]:
    """Map every timestep the pipeline can present to the cache slots that hold it.

    Built from the port's *own* schedule arrays, so the keys are exactly the values
    `build_row_timesteps` will put in the global table — no tolerance, no search.

    A value can appear twice (the two schedules meet wherever their shifted sigmas coincide, e.g.
    at ``t = 0`` on the first step), which is why the values are lists: the caller verifies the
    duplicate slots really do hold the same modulation.
    """
    sources: dict[float, list[tuple[int, int]]] = {}
    for step, t in enumerate(video_sched.timesteps.tolist()):
        sources.setdefault(_key(t), []).append((step, VARIANT_VIDEO))
    for step, t in enumerate(audio_sched.timesteps.tolist()):
        sources.setdefault(_key(t), []).append((step, VARIANT_AUDIO))
    # The conditioning level is `max(t, KEYFRAME_NOISE_AUG)`, and every t in the schedule is below
    # 0.999, so it is the constant the cache stores in variant 2. Any step's variant-2 row will do.
    sources.setdefault(_key(KEYFRAME_NOISE_AUG), []).append((0, VARIANT_CONDITION))
    return sources


def modulation_rows(table_values: list[float], sources: dict[float, list[tuple[int, int]]]) -> np.ndarray:
    """Gather index of shape ``(T * MODALITY_NUM,)`` into the flattened ``(steps * 9, ...)`` table.

    Destination row ``k * MODALITY_NUM + m`` is what the port addresses with
    ``timestep_indices * MODALITY_NUM + token_tags``; it is served by cache row
    ``step * ROWS_PER_STEP + variant * MODALITY_NUM + m``.
    """
    rows = np.empty(len(table_values) * MODALITY_NUM, dtype=np.int64)
    for k, value in enumerate(table_values):
        step, variant = _resolve(value, sources)
        base = step * ROWS_PER_STEP + variant * MODALITY_NUM
        for m in range(MODALITY_NUM):
            rows[k * MODALITY_NUM + m] = base + m
    return rows


def final_rows(table_values: list[float], sources: dict[float, list[tuple[int, int]]]) -> np.ndarray:
    """Gather index of shape ``(T,)`` into the flattened ``(steps * 3, 2*hidden)`` output table.

    The output layer's modulation has no modality axis — `FinalLayer.norm_out` indexes it with
    ``timestep_indices`` directly — so one row per distinct timestep.
    """
    rows = np.empty(len(table_values), dtype=np.int64)
    for k, value in enumerate(table_values):
        step, variant = _resolve(value, sources)
        rows[k] = step * VARIANT_NUM + variant
    return rows


def _resolve(value: float, sources: dict[float, list[tuple[int, int]]]) -> tuple[int, int]:
    slots = sources.get(_key(value))
    if not slots:
        raise ScheduleMismatch(
            f"The packed sequence presents timestep {value!r}, which the precomputed AdaLN table "
            "does not hold. The table covers only this schedule's video and audio levels plus the "
            f"conditioning level {KEYFRAME_NOISE_AUG}. A reference-audio conditioning level (1.0) "
            "would land here — this build's cache cannot serve it."
        )
    return slots[0]


class AdaLNCacheFile:
    """``adaln_cache.safetensors`` as mere.run writes it."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"No AdaLN cache at {self.path}. The mere.run transformer omits the modulation "
                "projections entirely, so the cache is not optional for this build."
            )
        self._raw = mx.load(str(self.path))
        self.video_sigmas = np.array(self._raw["video_sigmas"], dtype=np.float32)
        self.audio_sigmas = np.array(self._raw["audio_sigmas"], dtype=np.float32)
        shape = self._raw["blocks.0.modulations"].shape
        self.num_steps, rows_per_step, self.modulation_width = shape
        self.num_blocks = sum(
            1 for k in self._raw if k.startswith("blocks.") and k.endswith(".modulations")
        )
        if rows_per_step != ROWS_PER_STEP:
            raise ValueError(
                f"{self.path.name}: expected {ROWS_PER_STEP} rows per step "
                f"(variant x modality), got {rows_per_step}."
            )

    @property
    def hidden_size(self) -> int:
        return self.modulation_width // 6

    def check_schedule(self, video_sched, audio_sched, atol: float = SIGMA_ATOL) -> None:
        """Refuse any schedule the table was not baked for, loudly.

        Silently serving the nearest row would not raise anything — it would just denoise on the
        wrong modulation and produce a plausible, wrong clip. So the sigma grids are compared
        elementwise, and the tolerance is justified against the grid's own minimum gap.
        """
        for label, sched, stored, shift in (
            ("video", video_sched, self.video_sigmas, 12.0),
            ("audio", audio_sched, self.audio_sigmas, 3.0),
        ):
            got = np.array(sched.sigmas, dtype=np.float32)
            if sched.shift != shift:
                raise ScheduleMismatch(
                    f"The cached AdaLN table was baked at {label} sigma shift {shift}, but the "
                    f"pipeline is running shift {sched.shift}."
                )
            if got.shape != stored.shape:
                raise ScheduleMismatch(
                    f"The cached AdaLN table holds {stored.shape[0] - 1} {label} evaluations "
                    f"({stored.shape[0]} sigmas); this run asks for {got.shape[0] - 1} "
                    f"({got.shape[0]} sigmas). This build only supports "
                    f"num_inference_steps={stored.shape[0]} — note the port's scheduler spends one "
                    "grid point on the terminal sigma, so N grid points drive N-1 forwards."
                )
            worst = float(np.abs(got - stored).max())
            if worst > atol:
                raise ScheduleMismatch(
                    f"The {label} sigma grid differs from the cached one by {worst:.3e} "
                    f"(tolerance {atol:.1e}). The cached AdaLN table cannot be used for this run."
                )
            # A tolerance is only safe while it stays far below the distance between neighbours.
            gaps = np.diff(stored)
            margin = float(np.abs(gaps[gaps != 0]).min())
            if margin <= 10 * atol:
                raise ScheduleMismatch(
                    f"The {label} sigma grid has neighbours {margin:.3e} apart, too close to the "
                    f"{atol:.1e} matching tolerance to identify rows unambiguously."
                )

    def build(
        self,
        timestep_table: mx.array,
        video_sched,
        audio_sched,
        dtype: mx.Dtype = mx.bfloat16,
        verbose: bool = False,
    ) -> CachedModulation:
        """Gather the table the port's global timestep list addresses.

        The cache is keyed by ``(step, variant)`` while the port works from one global list of
        distinct timesteps; this rewrites the former into the latter, block by block so that only
        one block's raw rows are resident at a time.
        """
        if self._raw is None:
            raise RuntimeError(
                f"{self.path.name} has already been consumed; construct a new AdaLNCacheFile to "
                "build another table."
            )
        self.check_schedule(video_sched, audio_sched)

        values = [float(v) for v in timestep_table.tolist()]
        sources = timestep_sources(video_sched, audio_sched)
        self._verify_duplicates(sources)

        rows = mx.array(modulation_rows(values, sources))
        frows = mx.array(final_rows(values, sources))
        hidden = self.hidden_size

        tables: list[tuple[mx.array, ...]] = []
        for block in range(self.num_blocks):
            flat = self._raw[f"blocks.{block}.modulations"].reshape(-1, self.modulation_width)
            picked = flat[rows]
            table = tuple(
                picked[:, i * hidden : (i + 1) * hidden].astype(dtype) for i in range(6)
            )
            mx.eval(table)
            tables.append(table)

        final = self._raw["final_modulations"].reshape(-1, 2 * hidden)[frows]
        final_shift = final[:, :hidden].astype(dtype)
        final_scale = final[:, hidden:].astype(dtype)
        mx.eval(final_shift, final_scale)

        cache = CachedModulation(tables, timestep_table, final_shift, final_scale)
        if verbose:
            print(f"  adaln cache: {len(values)} timesteps from {self.path.name}, "
                  f"{cache.nbytes() / 1e6:.0f} MB")
        # The raw table is several times the gathered one; let it go.
        self._raw = None
        return cache

    def _verify_duplicates(self, sources: dict[float, list[tuple[int, int]]]) -> None:
        """Where two schedules meet, the slots they point at must hold the same modulation.

        The modulation is a deterministic function of the timestep, so equal timesteps must give
        equal rows. If they do not, the ``(step, variant)`` mapping is wrong — exactly the failure
        that would otherwise be silent.
        """
        flat = self._raw["blocks.0.modulations"].reshape(-1, self.modulation_width)
        for slots in sources.values():
            if len(slots) < 2:
                continue
            first = slots[0][0] * ROWS_PER_STEP + slots[0][1] * MODALITY_NUM
            for step, variant in slots[1:]:
                other = step * ROWS_PER_STEP + variant * MODALITY_NUM
                a = flat[first : first + MODALITY_NUM]
                b = flat[other : other + MODALITY_NUM]
                if not bool(mx.array_equal(a, b)):
                    raise ScheduleMismatch(
                        "Two schedule entries share a timestep but the cached modulations differ "
                        f"({slots[0]} vs {(step, variant)}). The (step, variant) layout assumed "
                        "for this cache is wrong; refusing to run."
                    )
