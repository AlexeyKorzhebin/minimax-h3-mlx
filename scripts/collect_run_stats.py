#!/usr/bin/env python3
"""Collect wall-clock / memory stats for every archived MiniMax H3 run and print a markdown table.

Four independent, differently-shaped sources exist for the same underlying fact (one generate()
call, one clip out) and none of them is complete on its own:

  1. ``queue/done/*.json`` + ``queue/logs/*.log`` — the web queue's own bookkeeping. Has exact
     `--width/--height/--duration/--steps` from `args`, exact `started_at`/`finished_at`, and an
     *estimated* `peak_gb` computed before the run (a prediction, not a measurement — kept and
     labelled as such, never presented as if it were observed).
  2. ``_logs/*.log`` — hand-run shell queues (...). One log per tag,
     no JSON alongside. Canvas and step count are exact (parsed straight out of the log), but the
     *requested* clip length is not printed anywhere in these older logs — only the aligned frame
     count is, so duration is back-computed as ``round(frames / 24)`` and flagged approximate.
  3. ``<subdir>/h3-*.json`` — structured post-run reports sitting next to the mp4. The best single
     source when present: exact canvas/duration/steps and an exact `generate_seconds`. No memory
     figures.
  4. ``memory.tsv`` (global) and any ``_logs/*-mem.tsv`` (per-run) — `tag -> peak_gb` samples from
     the background memory tracker. The only *measured* peak-memory source; only a minority of
     tags were tracked this way.

`docs/RESULTS.md` is a fifth, narrative source: it documents the same MacBook's runs with careful
methodology write-ups (see "A correction: how these numbers were re-derived" in that file for a
cautionary tale about locale bugs corrupting a memory CSV). This script does not re-parse it —
its numbers are prose-embedded and already hand-verified — but its existence is why this script
never invents a number it cannot cite a source line for: where a source is silent, the table says
so with an em dash, not a guess.

A run's identity is its `--tag`. When more than one source has the same tag, fields are merged
with a priority order per source (report json > queue done json > shell log), not a blind
overwrite — e.g. a tag with both a queue `done.json` (which knows the exact date) and a report
json (which knows the exact `generate_seconds`) keeps both, taking the best available value for
each column independently.

One gotcha worth documenting here because it is easy to get backwards: tags/filenames like
``s20260821`` or ``d896-20260909`` look like dates but are **seeds**, not dates — this was checked
against file mtimes (e.g. `centaur-battle-anatomy-s20260821.log` has an mtime of 2026-08-11, three
weeks before the "20260821" in its own name). Dates for `_logs`-only runs therefore come from file
mtime, not from parsing digits out of the tag, and are marked accordingly.

Usage::

    python3 scripts/collect_run_stats.py                  # markdown table to stdout
    python3 scripts/collect_run_stats.py --testvideo ~/h3-outputs
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

FPS = 24  # h3_48gb / upstream minimax_h3_mlx.packing.FPS — used only to back out an approximate
# requested duration from a frame count when no explicit --duration is available (source 2).

CANVAS_RE = re.compile(r"canvas (\d+x\d+), (\d+) frames")
DONE_RE = re.compile(r"done in ([\d.]+) min -> (\S+)")
STEP_RE = re.compile(r"step (\d+)/(\d+)")


@dataclass
class Run:
    tag: str
    date: str = "—"
    date_approx: bool = False
    canvas: str = "—"
    duration_s: Optional[float] = None
    duration_approx: bool = False
    steps: Optional[int] = None
    compute_min: Optional[float] = None
    peak_gb: Optional[float] = None
    peak_kind: str = ""  # "measured" | "estimate" | ""
    sources: list = field(default_factory=list)
    status: str = "ok"  # "ok" | "failed"


# ---------------------------------------------------------------------------
# Source 4: memory.tsv (global) and per-run *-mem.tsv — tag -> measured peak_gb
# ---------------------------------------------------------------------------

def parse_mem_tsv(path: Path) -> dict[str, float]:
    """Parse a `time\\ttag\\tphase\\tfootprint_gb\\tpeak_gb\\tfree_gb\\tswap_gb` TSV.

    Returns the max observed peak_gb per tag. Works for both the global tracker (which has a
    header row) and per-run trackers (which do not) — a row is skipped only if its numeric field
    doesn't parse, which quietly handles the header either way.
    """
    peaks: dict[str, float] = {}
    if not path.exists():
        return peaks
    for line in path.read_text(errors="replace").splitlines():
        cols = line.split("\t")
        if len(cols) < 5:
            continue
        tag = cols[1].strip()
        try:
            peak = float(cols[4])
        except ValueError:
            continue
        if not tag or tag == "tag":
            continue
        if peak <= 0:
            continue
        peaks[tag] = max(peaks.get(tag, 0.0), peak)
    return peaks


def collect_measured_peaks(logs_dir: Path, memory_tsv: Path) -> dict[str, float]:
    peaks = parse_mem_tsv(memory_tsv)
    for tsv in sorted(logs_dir.glob("*-mem.tsv")):
        for tag, peak in parse_mem_tsv(tsv).items():
            peaks[tag] = max(peaks.get(tag, 0.0), peak)
    return peaks


# ---------------------------------------------------------------------------
# Source 1: queue/done/*.json + queue/logs/*.log
# ---------------------------------------------------------------------------

def _args_to_dict(args: list[str]) -> dict[str, str]:
    """Turn an argv-style list into {flag: value}, tolerating leading positionals (subcommand,
    inline prompt text) that don't start with `--`."""
    out: dict[str, str] = {}
    i = 0
    while i < len(args):
        tok = args[i]
        if tok.startswith("--") and i + 1 < len(args):
            out[tok[2:]] = args[i + 1]
            i += 2
        else:
            i += 1
    return out


def parse_queue(queue_dir: Path) -> list[Run]:
    runs: list[Run] = []
    done_dir = queue_dir / "done"
    logs_dir = queue_dir / "logs"
    done_ids = set()

    for jf in sorted(done_dir.glob("*.json")):
        data = json.loads(jf.read_text())
        done_ids.add(data.get("id", jf.stem))
        a = _args_to_dict(data.get("args", []))
        tag = a.get("tag", jf.stem)
        canvas = f"{a['width']}x{a['height']}" if "width" in a and "height" in a else "—"
        duration_s = float(a["duration"]) if "duration" in a else None
        steps = int(a["steps"]) if "steps" in a else None

        compute_min = None
        m = DONE_RE.search(data.get("log_tail", ""))
        if m:
            compute_min = float(m.group(1))
        elif data.get("started_at") and data.get("finished_at"):
            from datetime import datetime
            fmt = "%Y-%m-%dT%H:%M:%S"
            dt = datetime.strptime(data["finished_at"], fmt) - datetime.strptime(data["started_at"], fmt)
            compute_min = dt.total_seconds() / 60.0

        peak_gb = None
        peak_kind = ""
        est = data.get("estimate") or {}
        if "peak_gb" in est:
            peak_gb = float(est["peak_gb"])
            peak_kind = "estimate"

        status = "ok" if data.get("exit_code", 0) == 0 else "failed"
        date = data.get("created_at", "")[:10] or "—"

        runs.append(Run(
            tag=tag, date=date, canvas=canvas, duration_s=duration_s, steps=steps,
            compute_min=compute_min, peak_gb=peak_gb, peak_kind=peak_kind,
            sources=["queue/done"], status=status,
        ))

    # Orphan logs: a log with no matching done/*.json (the run failed before the queue worker
    # could write its result). Still worth a row — canvas is usually in the log even when the run
    # crashed before printing "done in ...", and the id-prefixed timestamp is a real date, not a
    # seed, so it doesn't need the mtime fallback other sources need.
    for lf in sorted(logs_dir.glob("*.log")):
        run_id = lf.stem
        if run_id in done_ids:
            continue
        text = lf.read_text(errors="replace")
        m = re.match(r"(\d{4})(\d{2})(\d{2})-", run_id)
        date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else "—"
        cm = CANVAS_RE.search(text)
        canvas = cm.group(1) if cm else "—"
        # tag: strip the leading timestamp and trailing 4-char random suffix that the queue
        # appends to the id, e.g. "20260813-110657-kot-italy-fl3p" -> "kot-italy"
        tag_guess = re.sub(r"^\d{8}-\d{6}-", "", run_id)
        tag_guess = re.sub(r"-[a-z0-9]{4}$", "", tag_guess)
        runs.append(Run(tag=tag_guess, date=date, canvas=canvas, sources=["queue/logs (orphan)"],
                         status="failed"))
    return runs


# ---------------------------------------------------------------------------
# Source 3: <TestVideo subdir>/h3-*.json — structured per-run reports
# ---------------------------------------------------------------------------

def parse_reports(testvideo_dir: Path) -> list[Run]:
    runs = []
    for jf in sorted(testvideo_dir.rglob("h3-*.json")):
        try:
            data = json.loads(jf.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if "tag" not in data or "canvas" not in data:
            continue  # not a run report shaped file
        compute_min = None
        if "generate_seconds" in data:
            compute_min = float(data["generate_seconds"]) / 60.0
        runs.append(Run(
            tag=data["tag"],
            canvas=data.get("canvas", "—"),
            duration_s=data.get("duration_seconds"),
            steps=data.get("grid_points"),
            compute_min=compute_min,
            # No date field in the report itself; fall back to the report file's own mtime.
            date=_mtime_date(jf), date_approx=True,
            sources=[f"report:{jf.relative_to(testvideo_dir)}"],
        ))
    return runs


# ---------------------------------------------------------------------------
# Source 2: _logs/*.log — hand-run shell queues
# ---------------------------------------------------------------------------

def _mtime_date(p: Path) -> str:
    import datetime
    return datetime.date.fromtimestamp(p.stat().st_mtime).isoformat()


def parse_shell_logs(logs_dir: Path) -> list[Run]:
    runs = []
    for lf in sorted(logs_dir.glob("*.log")):
        text = lf.read_text(errors="replace")
        cm = CANVAS_RE.search(text)
        dm = DONE_RE.search(text)
        if not cm or not dm:
            continue  # not a generation run (baking, fetch, web/worker service logs, ...)
        canvas, frames = cm.group(1), int(cm.group(2))
        steps = None
        step_matches = STEP_RE.findall(text)
        if step_matches:
            # The log's "step N/M" counts *forwards*, which is (--steps - 1) — e.g. a run started
            # with --steps 8 logs "step 7/7". Cross-checked against kot-italy2, which has both a
            # queue done.json (--steps 8) and this same log shape (max denominator 7). +1 here
            # converts back to the --steps / grid_points convention the other two sources use, so
            # the merged table's "шаги" column means the same thing regardless of source.
            steps = max(int(denom) for _, denom in step_matches) + 1
        runs.append(Run(
            tag=lf.stem,
            date=_mtime_date(lf), date_approx=True,
            canvas=canvas,
            duration_s=round(frames / FPS), duration_approx=True,
            steps=steps,
            compute_min=float(dm.group(1)),
            sources=[f"_logs/{lf.name}"],
        ))
    return runs


# ---------------------------------------------------------------------------
# Merge (report > queue done > shell log, field by field) and render
# ---------------------------------------------------------------------------

def merge(report_runs, queue_runs, shell_runs, measured_peaks: dict[str, float]) -> list[Run]:
    by_tag: dict[str, list[Run]] = {}
    # Priority low -> high so the last append per tag in each field-pick loop is the strongest.
    for bucket in (shell_runs, queue_runs, report_runs):
        for r in bucket:
            by_tag.setdefault(r.tag, []).append(r)

    merged = []
    for tag, candidates in by_tag.items():
        out = Run(tag=tag)
        out.status = "failed" if any(c.status == "failed" for c in candidates) else "ok"
        for c in candidates:  # later (higher-priority) candidates win ties
            if c.canvas != "—":
                out.canvas = c.canvas
            if c.duration_s is not None:
                out.duration_s, out.duration_approx = c.duration_s, c.duration_approx
            if c.steps is not None:
                out.steps = c.steps
            if c.compute_min is not None:
                out.compute_min = c.compute_min
            if c.date != "—":
                # A queue-sourced exact date always wins over an mtime-derived one.
                if out.date == "—" or out.date_approx or not c.date_approx:
                    out.date, out.date_approx = c.date, c.date_approx
            out.sources.extend(c.sources)

        if tag in measured_peaks:
            out.peak_gb, out.peak_kind = measured_peaks[tag], "measured"
        else:
            for c in candidates:
                if c.peak_gb is not None:
                    out.peak_gb, out.peak_kind = c.peak_gb, c.peak_kind
        merged.append(out)

    merged.sort(key=lambda r: (r.date, r.tag))
    return merged


def fmt_peak(r: Run) -> str:
    if r.peak_gb is None:
        return "—"
    if r.peak_kind == "estimate":
        return f"~{r.peak_gb:.1f} ГБ (оценка)"
    return f"{r.peak_gb:.1f} ГБ"


def fmt_duration(r: Run) -> str:
    if r.duration_s is None:
        return "—"
    s = f"{r.duration_s:g}"
    return f"~{s}" if r.duration_approx else s


def fmt_steps(r: Run) -> str:
    return "—" if r.steps is None else str(r.steps)


def fmt_compute(r: Run) -> str:
    if r.compute_min is None:
        return "—" if r.status != "failed" else "ошибка"
    return f"{r.compute_min:.1f} мин"


def fmt_date(r: Run) -> str:
    if r.date == "—":
        return "—"
    return f"~{r.date}" if r.date_approx else r.date


def render_markdown(runs: list[Run]) -> str:
    lines = [
        "| дата | тег | канвас | длительность, с | шаги | время счёта | пик памяти |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        tag = r.tag + (" ⚠️" if r.status == "failed" else "")
        lines.append(
            f"| {fmt_date(r)} | {tag} | {r.canvas} | {fmt_duration(r)} | {fmt_steps(r)} "
            f"| {fmt_compute(r)} | {fmt_peak(r)} |"
        )
    return "\n".join(lines)


def main() -> None:
    home = Path.home()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--testvideo", type=Path, default=home / "h3-outputs",
                     help="root of the TestVideo tree (default: ~/h3-outputs)")
    ap.add_argument("--queue", type=Path, default=None,
                     help="queue dir (default: <testvideo>/queue)")
    ap.add_argument("--shell-logs", type=Path, default=None,
                     help="dir of hand-run shell-queue logs (default: <testvideo>/_logs)")
    ap.add_argument("--memory-tsv", type=Path, default=None,
                     help="global memory tracker TSV (default: <testvideo>/memory.tsv)")
    args = ap.parse_args()

    testvideo = args.testvideo.expanduser()
    queue_dir = (args.queue or testvideo / "queue").expanduser()
    shell_logs_dir = (args.shell_logs or testvideo / "_logs").expanduser()
    memory_tsv = (args.memory_tsv or testvideo / "memory.tsv").expanduser()

    report_runs = parse_reports(testvideo)
    queue_runs = parse_queue(queue_dir) if queue_dir.exists() else []
    shell_runs = parse_shell_logs(shell_logs_dir) if shell_logs_dir.exists() else []
    measured_peaks = collect_measured_peaks(shell_logs_dir, memory_tsv)

    merged = merge(report_runs, queue_runs, shell_runs, measured_peaks)

    print(f"<!-- reports: {len(report_runs)}, queue: {len(queue_runs)}, "
          f"shell logs: {len(shell_runs)}, merged rows: {len(merged)} -->")
    print(render_markdown(merged))


if __name__ == "__main__":
    main()
