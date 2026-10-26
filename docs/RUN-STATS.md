# Run stats

A flat log of every archived MiniMax H3 generation run this fork could find evidence of, machine-
collected from raw logs and JSON — not hand-curated, and not a substitute for
[`RESULTS.md`](RESULTS.md)'s hand-verified, methodology-annotated measurements. Where the two
disagree, `RESULTS.md` is the one that was re-derived from first principles after a locale bug
corrupted a memory CSV (see its "A correction" section); this file just says what the logs say.

**Every peak in this file predates 2026-08-17.** On that day four bit-identical rewrites of the
DiT's Q/K/V and MLP staging landed (`RESULTS.md`, "Attention memory levers"), and the diffusion
peak of a long native forward fell by about a fifth — 37.2 GB to 32.3 GB at 10 s, more in absolute
terms the longer the clip. Nothing about the *outputs* changed, so the rows below remain valid as a
record of what was rendered and how long it took; their memory column is simply no longer what the
same request would cost today. The pre-run estimate the queue shows is likewise still fitted to
pre-lever runs — re-fitting it needs whole-run measurements, which these are not.

## How this was collected

`scripts/collect_run_stats.py` walks four sources and merges rows by `--tag`:

1. `~/h3-outputs/queue/done/*.json` + `queue/logs/*.log` — the web queue's own
   bookkeeping. Exact canvas/duration/steps (from `args`), exact start/finish timestamps, and a
   *pre-run estimate* of peak memory (never a measurement — always rendered as `~N ГБ (оценка)`).
   Logs with no matching `done/*.json` are orphaned/failed runs and still get a row, marked ⚠️.
2. `~/h3-outputs/_logs/*.log` — hand-run shell queues (
   d896, ...). Exact canvas and step count come straight out of the log text; the *requested* clip
   length does not appear in these older logs at all, only the aligned frame count does, so
   duration is back-computed as `round(frames / 24 fps)` and marked `~N`. This is an approximation
   of the aligned clip, not necessarily the original `--duration` — `RESULTS.md`'s own per-run
   table shows one case (512x512) where the true request was 2.4 s and the aligned frame count
   rounds to 3 s, a discrepancy this heuristic cannot see.
3. `~/h3-outputs/*/h3-*.json` (searched recursively) — structured per-run reports next to
   the mp4. The best single source when present: exact `canvas`/`duration_seconds`/`grid_points`
   and an exact `generate_seconds`. Wins ties against the other two sources for every field it
   has.
4. `~/h3-outputs/memory.tsv` (global) and any `_logs/*-mem.tsv` (per-run) — the *only*
   measured peak-memory source, a `tag -> peak_gb` tracker. When a tag has no tracked peak, the
   estimate from source 1 is used if available; otherwise the cell is an honest `—`.

One gotcha the script works around: tags/filenames like `s20260821` or `d896-20260909` look like
dates but are **seeds** — checked against file mtimes, which land three weeks or more before the
"date" embedded in the tag. Dates for shell-log-only rows therefore come from file mtime (marked
`~YYYY-MM-DD`), never from parsing the tag.

## Reading the table

- `~` before a date or duration means it's derived/approximate (mtime-based date, or a duration
  back-computed from frame count), not a value read directly off a `--duration`/`created_at`
  field.
- `(оценка)` on a peak-memory cell means it's the queue's pre-run *estimate*, not something a
  memory tracker observed running.
- `—` means the source data plainly doesn't have that number. Never filled in by guesswork.
- ⚠️ marks a run that crashed or has no completion record (currently: `kot-italy`, which hit
  `blocks.0.adaln_proj was not shipped by this build` before reaching the diffusion loop).
- "шаги" is normalized to the `--steps` / `grid_points` convention (the count passed on the
  command line) across all three sources — the shell logs' own `step N/M` lines count *forwards*,
  which is one less, and the script adds 1 back before printing.

## How to update

Re-run the script and replace the table below:

```
python3 scripts/collect_run_stats.py
```

It's idempotent — same inputs always produce the same output, sorted by date then tag — so
diffing two runs shows exactly what changed (new runs appended, or a previously-unmeasured peak
now tracked). Pass `--testvideo`, `--queue`, `--shell-logs`, or `--memory-tsv` to point at a
different tree; defaults match the layout under `~/h3-outputs`.

---

<!-- reports: 18, queue: 14, shell logs: 55, merged rows: 69 -->
| дата | тег | канвас | длительность, с | шаги | время счёта | пик памяти |
|---|---|---|---|---|---|---|
| ~2026-08-10 | centaur-4bit | 896x576 | ~10 | 8 | 71.4 мин | 27.0 ГБ |
| ~2026-08-10 | centaur-8bit | 896x576 | ~10 | 8 | 74.0 мин | 28.0 ГБ |
| ~2026-08-10 | d896-16 | 896x512 | ~3 | 16 | 25.4 мин | 27.0 ГБ |
| ~2026-08-10 | d896-20260909 | 896x512 | ~3 | 8 | 12.5 мин | 27.0 ГБ |
| ~2026-08-10 | d896-20260912 | 896x512 | ~3 | 8 | 12.4 мин | 27.0 ГБ |
| ~2026-08-10 | d896-20260913 | 896x512 | ~3 | 8 | 12.4 мин | 27.0 ГБ |
| ~2026-08-10 | d896-31 | 896x512 | ~3 | 31 | 46.8 мин | 27.0 ГБ |
| ~2026-08-10 | d896-lora045 | 896x512 | ~3 | 8 | 12.6 мин | 27.0 ГБ |
| ~2026-08-10 | d896-lora100 | 896x512 | ~3 | 8 | 12.6 мин | 27.0 ГБ |
| ~2026-08-10 | key-16 | 512x512 | ~3 | 16 | 14.3 мин | 28.0 ГБ |
| ~2026-08-10 | key-8 | 512x512 | ~3 | 8 | 7.5 мин | 28.0 ГБ |
| ~2026-08-10 | key-tail3 | 512x512 | ~3 | 10 | 9.1 мин | 28.0 ГБ |
| ~2026-08-10 | q8bit | 896x512 | ~3 | 8 | 12.6 мин | 27.0 ГБ |
| ~2026-08-10 | q8bit-lora100 | 896x512 | ~3 | 8 | 12.8 мин | 27.0 ГБ |
| ~2026-08-10 | res-640 | 640x640 | ~3 | 8 | 10.3 мин | 27.0 ГБ |
| ~2026-08-10 | res-768 | 768x768 | ~3 | 8 | 16.3 мин | 27.0 ГБ |
| ~2026-08-10 | res-896w | 896x512 | ~3 | 8 | 12.5 мин | 27.0 ГБ |
| ~2026-08-10 | s20260909-512 | 512x512 | ~3 | 8 | 6.8 мин | 27.0 ГБ |
| ~2026-08-10 | s20260909-768 | 768x768 | ~3 | 8 | 16.4 мин | 27.0 ГБ |
| ~2026-08-10 | s20260912-512 | 512x512 | ~3 | 8 | 6.9 мин | 27.0 ГБ |
| ~2026-08-10 | s20260912-768 | 768x768 | ~3 | 8 | 16.4 мин | 27.0 ГБ |
| ~2026-08-10 | s20260913-512 | 512x512 | ~3 | 8 | 6.8 мин | 27.0 ГБ |
| ~2026-08-10 | s20260913-768 | 768x768 | ~3 | 8 | 16.3 мин | 27.0 ГБ |
| ~2026-08-10 | tango-50 | 512x512 | ~3 | 50 | 39.3 мин | 27.0 ГБ |
| ~2026-08-11 | centaur-8bit-20 | 896x576 | ~10 | 20 | 188.0 мин | 29.0 ГБ |
| ~2026-08-11 | centaur-8bit-lora100 | 896x576 | ~10 | 8 | 71.2 мин | 30.0 ГБ |
| ~2026-08-11 | centaur-8bit-lora100-15s | 896x576 | ~15 | 8 | 148.8 мин | 33.0 ГБ |
| ~2026-08-11 | centaur-8bit-lora100-s20260811 | 896x576 | ~10 | 8 | 71.6 мин | 30.0 ГБ |
| ~2026-08-11 | centaur-8bit-lora100-s20260812 | 896x576 | ~10 | 8 | 72.3 мин | 30.0 ГБ |
| ~2026-08-11 | centaur-8bit-seed2 | 896x576 | ~10 | 8 | 73.3 мин | 28.0 ГБ |
| ~2026-08-11 | centaur-battle-anatomy-2-s20260821-448x288 | 448x288 | ~10 | 8 | 13.4 мин | 27.0 ГБ |
| ~2026-08-11 | centaur-battle-anatomy-s20260821 | 448x288 | ~10 | 8 | 13.3 мин | 27.0 ГБ |
| ~2026-08-11 | centaur-battle-anatomy-s20260822 | 448x288 | ~10 | 8 | 13.8 мин | 27.0 ГБ |
| ~2026-08-11 | centaur-battle-anatomy-s20260823 | 448x288 | ~10 | 8 | 14.1 мин | 27.0 ГБ |
| ~2026-08-11 | centaur-v3-s20260821 | 448x288 | ~10 | 8 | 14.1 мин | 27.0 ГБ |
| ~2026-08-11 | greek-896-l060 | 896x576 | ~10 | 8 | 81.2 мин | 30.0 ГБ |
| ~2026-08-11 | greek-896-skin | 896x576 | ~10 | 8 | 80.9 мин | 30.0 ГБ |
| ~2026-08-11 | greek-l060-s20260821 | 448x288 | ~10 | 8 | 13.8 мин | 27.0 ГБ |
| ~2026-08-11 | greek-l100-s20260821 | 448x288 | ~10 | 8 | 13.8 мин | 27.0 ГБ |
| ~2026-08-11 | greek-warrior-battle-s20260821 | 448x288 | ~10 | 8 | 13.1 мин | 27.0 ГБ |
| ~2026-08-11 | greek-warrior-battle-s20260821-896x576 | 896x576 | ~10 | 8 | 78.3 мин | 30.0 ГБ |
| ~2026-08-11 | greek-warrior-battle-s20260822 | 448x288 | ~10 | 8 | 13.5 мин | 27.0 ГБ |
| ~2026-08-11 | greek-warrior-battle-s20260822-896x576 | 896x576 | ~10 | 8 | 82.7 мин | 30.0 ГБ |
| ~2026-08-11 | greek-warrior-battle-s20260823 | 448x288 | ~10 | 8 | 13.4 мин | 27.0 ГБ |
| ~2026-08-11 | greek-warrior-battle-s20260823-896x576 | 896x576 | ~10 | 8 | 81.1 мин | 30.0 ГБ |
| ~2026-08-12 | ballad-2 | 896x576 | 10 | 8 | 80.5 мин | 30.0 ГБ |
| ~2026-08-12 | ballad-3 | 896x576 | 10 | 8 | 79.7 мин | 30.0 ГБ |
| ~2026-08-12 | centaur-official | 448x288 | 10 | 8 | 13.3 мин | 27.0 ГБ |
| ~2026-08-12 | greek-896-overcast | 896x576 | ~10 | 8 | 77.3 мин | 30.0 ГБ |
| ~2026-08-12 | greek-official | 896x576 | 10 | 8 | 77.3 мин | 30.0 ГБ |
| ~2026-08-12 | overcast-1088 | 1088x704 | ~10 | 8 | 139.6 мин | 33.0 ГБ |
| ~2026-08-12 | overcast-15s | 896x576 | ~15 | 8 | 140.7 мин | 33.0 ГБ |
| ~2026-08-12 | overcast-s22 | 896x576 | ~10 | 8 | 76.1 мин | 30.0 ГБ |
| ~2026-08-12 | overcast-s23 | 896x576 | ~10 | 8 | 75.3 мин | 30.0 ГБ |
| ~2026-08-12 | tango-arc | 896x576 | 10 | 8 | 72.6 мин | 30.0 ГБ |
| 2026-08-13 | kot-italy ⚠️ | 448x288 | — | — | ошибка | — |
| 2026-08-13 | kot-italy2 | 448x288 | 10 | 8 | 13.4 мин | ~24.7 ГБ (оценка) |
| 2026-08-13 | run | 896x576 | 10 | 8 | 77.9 мин | ~31.5 ГБ (оценка) |
| 2026-08-14 | galloping-nude-women-whe | 896x576 | 10 | 8 | 83.4 мин | ~31.5 ГБ (оценка) |
| 2026-08-14 | medium-wide-shot-capture | 896x576 | 15 | 8 | 144.6 мин | ~35.7 ГБ (оценка) |
| 2026-08-14 | nude-horsewomen-field | 896x576 | 10 | 8 | 80.0 мин | ~31.5 ГБ (оценка) |
| 2026-08-14 | office-sexy-arc-shots | 896x576 | 10 | 8 | 77.9 мин | ~31.5 ГБ (оценка) |
| 2026-08-15 | athletic-nude-women-sovi-s1 | 448x288 | 15 | 8 | 20.5 мин | ~25.9 ГБ (оценка) |
| 2026-08-15 | athletic-nude-women-sovi-s2 | 448x288 | 15 | 8 | 21.1 мин | ~25.9 ГБ (оценка) |
| 2026-08-15 | athletic-nude-women-sovi-s3 | 896x576 | 15 | 8 | 144.8 мин | ~35.7 ГБ (оценка) |
| 2026-08-15 | athletic-nude-women-sovi-s4 | 896x576 | 15 | 8 | 145.4 мин | ~35.7 ГБ (оценка) |
| 2026-08-15 | athletic-nude-women-sovi-s5 | 448x288 | 15 | 8 | 21.4 мин | ~25.9 ГБ (оценка) |
| 2026-08-16 | athletic-nude-women-sovi-s6 | 896x576 | 15 | 8 | 148.4 мин | ~35.7 ГБ (оценка) |
| 2026-08-16 | athletic-nude-women-sovi-s7 | 1344x768 | 15 | 8 | 472.1 мин | ~48.7 ГБ (оценка) |
