"""`runs.scan` reads what is on disk without importing the pipeline.

The fixtures write safetensors files by hand -- an 8-byte little-endian header length, then the
JSON header, then nothing. That is a valid safetensors file with no tensors, and it is all `scan`
reads, which is the point: a run's progress must be knowable without loading 21 GB of weights.
"""
import contextlib
import fcntl
import json
import struct
from pathlib import Path

from h3_48gb.runs import Run, _parse, scan


def write_checkpoint(path: Path, *, completed: int, total: int, written_at: str,
                     started_at: str | None = None, completed_at_start: int = 0,
                     identity: dict | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "format": 1,
        "identity": identity or {"request": {"prompt": "a cat", "seed": 7}},
        "identity_digest": "deadbeef",
        "completed_steps": completed,
        "total_forwards": total,
        "written_at": written_at,
        "started_at": started_at,
        "completed_at_start": completed_at_start,
    }
    header = json.dumps({"__metadata__": {"h3_checkpoint": json.dumps(meta)}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header)
    return path


def lock_path_for(checkpoint: Path) -> Path:
    """Where a writer's companion flock file lives, mirroring `checkpoint.CheckpointStore.lock_path`
    -- the two are never allowed to import each other (`checkpoint.py` needs `mlx`; `runs.py`
    cannot), so this is the naming convention duplicated on the test side to drive `runs._writer_alive`
    directly, the same way `runs.scan` derives it from a bare checkpoint path.
    """
    return checkpoint.with_name(checkpoint.name + ".lock")


def write_free_lock_file(checkpoint: Path) -> Path:
    """A companion lock file nobody currently holds.

    This is what a writer that crashed hard looks like from the outside: the file is still there,
    but `fcntl.flock` released whatever it held the instant the crashed process's file descriptor
    closed -- the kernel does that unconditionally, which is the entire mechanism `_writer_alive`
    leans on. No process needs to actually crash to produce this state; an untouched fresh file is
    indistinguishable from one a dead writer left behind, which is the point.
    """
    path = lock_path_for(checkpoint)
    path.touch()
    return path


@contextlib.contextmanager
def held_lock(checkpoint: Path):
    """A companion lock file with a live exclusive flock on it, for the lifetime of the `with`
    block.

    What a running writer looks like from the outside -- see `checkpoint.CheckpointStore.acquire_lock`
    for the real thing this stands in for. Exclusive, matching the real writer: `_writer_alive`'s
    probe takes a *shared* lock, and two shared locks never conflict with each other, so a fixture
    that held a shared lock here would let the probe through and silently stop testing anything.
    """
    path = lock_path_for(checkpoint)
    handle = open(path, "wb")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    try:
        yield path
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def test_a_checkpoint_is_read_into_a_run(tmp_path):
    write_checkpoint(tmp_path / "run-a" / "checkpoints" / "h3-abc.safetensors",
                     completed=3, total=7, written_at="2026-08-10T21:00:00")

    runs = scan(tmp_path)

    assert len(runs) == 1
    run = runs[0]
    assert run.completed == 3
    assert run.total == 7
    assert run.fraction == 3 / 7
    assert run.identity_digest == "deadbeef"


def test_a_resumed_run_rates_only_this_session(tmp_path):
    """Divide this session's elapsed time by this session's forwards, not by all of them.

    A run resumed at forward 5 of 7 that has since done one more forward has spent 600 s on one
    forward. Dividing by `completed` instead of by `completed - completed_at_start` reports 100 s
    and an ETA five times too optimistic.
    """
    write_checkpoint(tmp_path / "r" / "checkpoints" / "h3-a.safetensors",
                     completed=6, total=7,
                     started_at="2026-08-10T21:00:00", completed_at_start=5,
                     written_at="2026-08-10T21:10:00")

    run = scan(tmp_path)[0]

    assert run.seconds_per_forward == 600.0
    assert run.eta_seconds == 600.0


def test_no_rate_before_the_first_forward_of_a_session(tmp_path):
    """Between a resume and its first write there is nothing to divide by."""
    write_checkpoint(tmp_path / "r" / "checkpoints" / "h3-a.safetensors",
                     completed=5, total=7,
                     started_at="2026-08-10T21:00:00", completed_at_start=5,
                     written_at="2026-08-10T21:00:00")

    run = scan(tmp_path)[0]

    assert run.seconds_per_forward is None
    assert run.eta_seconds is None


def test_a_corrupt_checkpoint_does_not_hide_its_neighbours(tmp_path):
    """One bad file must cost one run, not the whole scan.

    The failure this pins is not hypothetical: a checkpoint is written by rename, so a file caught
    mid-write is a real possibility on a machine running a queue.
    """
    (tmp_path / "run-bad" / "checkpoints").mkdir(parents=True)
    (tmp_path / "run-bad" / "checkpoints" / "h3-bad.safetensors").write_bytes(b"\x01\x02\x03")
    write_checkpoint(tmp_path / "run-good" / "checkpoints" / "h3-ok.safetensors",
                     completed=1, total=7, written_at="2026-08-10T21:00:00")

    runs = {r.outdir.name: r for r in scan(tmp_path)}

    assert runs["run-bad"].state == "unreadable"
    assert runs["run-bad"].error
    assert runs["run-good"].completed == 1, "a corrupt neighbour swallowed a good run"


def write_raw_header(path: Path, header_obj) -> Path:
    """Like `write_checkpoint`, but for headers that are not the well-formed shape a healthy
    writer produces -- the fixtures below need to control the JSON precisely, down to putting a
    non-string or non-object where the format normally never would.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    header = json.dumps(header_obj).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header)
    return path


def test_metadata_stored_as_an_object_does_not_hide_its_neighbours(tmp_path):
    """`h3_checkpoint` is always a JSON *string* in a real checkpoint -- `mx.save_safetensors`
    metadata values are strings, so `checkpoint.py` does `json.dumps(meta)` before writing it.
    A file where it is instead the object directly (hand-edited, or a future writer's mistake)
    makes the inner `json.loads(raw[_META_KEY])` raise TypeError, not the ValueError family
    `scan` used to expect for bad JSON.
    """
    write_raw_header(tmp_path / "run-bad" / "checkpoints" / "h3-bad.safetensors",
                      {"__metadata__": {"h3_checkpoint": {"completed_steps": 1}}})
    write_checkpoint(tmp_path / "run-good" / "checkpoints" / "h3-ok.safetensors",
                     completed=1, total=7, written_at="2026-08-10T21:00:00")

    runs = {r.outdir.name: r for r in scan(tmp_path)}

    assert runs["run-bad"].state == "unreadable"
    assert runs["run-bad"].error
    assert runs["run-good"].completed == 1, "a structurally-odd neighbour swallowed a good run"


def test_a_null_completed_steps_does_not_hide_its_neighbours(tmp_path):
    """`completed_steps: null` parses as valid JSON -- `int(None)` is what actually raises, and it
    raises TypeError, not ValueError, so it must reach `scan` as one bad run, not a dead scan.
    """
    meta = {
        "format": 1,
        "identity": {"request": {"prompt": "a cat", "seed": 7}},
        "identity_digest": "deadbeef",
        "completed_steps": None,
        "total_forwards": 7,
        "written_at": "2026-08-10T21:00:00",
    }
    write_raw_header(tmp_path / "run-bad" / "checkpoints" / "h3-bad.safetensors",
                      {"__metadata__": {"h3_checkpoint": json.dumps(meta)}})
    write_checkpoint(tmp_path / "run-good" / "checkpoints" / "h3-ok.safetensors",
                     completed=1, total=7, written_at="2026-08-10T21:00:00")

    runs = {r.outdir.name: r for r in scan(tmp_path)}

    assert runs["run-bad"].state == "unreadable"
    assert runs["run-bad"].error
    assert runs["run-good"].completed == 1, "a null completed_steps neighbour swallowed a good run"


def test_a_non_object_header_does_not_hide_its_neighbours(tmp_path):
    """The 8-byte length is followed by *some* JSON value in principle, and `scan` assumes it is
    always an object. A top-level array makes `header.get` raise AttributeError before any
    metadata lookup even starts -- the earliest possible point in `read_checkpoint_meta`.
    """
    write_raw_header(tmp_path / "run-bad" / "checkpoints" / "h3-bad.safetensors",
                      ["not", "an", "object"])
    write_checkpoint(tmp_path / "run-good" / "checkpoints" / "h3-ok.safetensors",
                     completed=1, total=7, written_at="2026-08-10T21:00:00")

    runs = {r.outdir.name: r for r in scan(tmp_path)}

    assert runs["run-bad"].state == "unreadable"
    assert runs["run-bad"].error
    assert runs["run-good"].completed == 1, "a non-object header neighbour swallowed a good run"


def test_a_held_lock_reads_in_flight_no_matter_how_old_the_checkpoint_is(tmp_path, monkeypatch):
    """Liveness is the lock, full stop -- not age, not rate, not any threshold built from either.

    A real forward can run up to half an hour, and a checkpoint can go quiet for a VAE decode on
    top of that; no fixed window distinguishes a live run sitting on a slow step from an abandoned
    one without guessing. `_now` is pinned absurdly far in the future here specifically to prove
    the state ignores it: the old window-based `_state_of` would have called this run `stale` (or
    even `unknown`) at this age. The lock does not need to guess.
    """
    import h3_48gb.runs as runs_mod

    checkpoint = write_checkpoint(tmp_path / "old-but-alive" / "checkpoints" / "h3-a.safetensors",
                     completed=2, total=7, started_at="2026-08-10T21:00:00",
                     completed_at_start=0, written_at="2026-08-10T21:20:00")

    monkeypatch.setattr(runs_mod, "_now", lambda: _parse("2026-08-11T21:20:00"))  # 24 h later

    with held_lock(checkpoint):
        assert {r.outdir.name: r.state for r in scan(tmp_path)}["old-but-alive"] == "in_flight"


def test_an_unlocked_lock_file_reads_stale_no_matter_how_fresh_the_checkpoint_is(tmp_path, monkeypatch):
    """The mirror image: a checkpoint written a second ago whose writer is already confirmed gone
    (lock file present, nothing holding it) is `stale`, not `in_flight` just because it is young.
    """
    import h3_48gb.runs as runs_mod

    checkpoint = write_checkpoint(tmp_path / "fresh-but-dead" / "checkpoints" / "h3-a.safetensors",
                     completed=2, total=7, started_at="2026-08-10T21:00:00",
                     completed_at_start=0, written_at="2026-08-10T21:20:00")
    write_free_lock_file(checkpoint)

    monkeypatch.setattr(runs_mod, "_now", lambda: _parse("2026-08-10T21:20:01"))  # 1 s later

    assert {r.outdir.name: r.state for r in scan(tmp_path)}["fresh-but-dead"] == "stale"


def test_no_lock_file_at_all_reads_unknown_regardless_of_rate(tmp_path, monkeypatch):
    """A checkpoint with no companion lock file -- an old-format file predating this feature, or a
    writer that already cleaned up after a graceful exit -- has nothing to say about its writer.
    `unknown` must not depend on whether `started_at`/rate happen to be present either: liveness is
    the lock alone.
    """
    import h3_48gb.runs as runs_mod

    write_checkpoint(tmp_path / "old-format" / "checkpoints" / "h3-a.safetensors",
                     completed=8, total=19, started_at=None,
                     completed_at_start=0, written_at="2026-08-10T23:23:36")
    write_checkpoint(tmp_path / "healthy" / "checkpoints" / "h3-b.safetensors",
                     completed=2, total=7, started_at="2026-08-10T21:00:00",
                     completed_at_start=0, written_at="2026-08-10T21:20:00")

    monkeypatch.setattr(runs_mod, "_now", lambda: _parse("2026-08-10T23:40:20"))  # ~1004 s later

    runs = {r.outdir.name: r for r in scan(tmp_path)}

    assert runs["old-format"].state == "unknown"
    assert runs["healthy"].state == "unknown"  # no lock file for this one either
    assert runs["healthy"].completed == 2, "an unknown neighbour must not hide a healthy run"


def test_a_non_busy_kernel_error_reads_unknown_not_in_flight(tmp_path, monkeypatch):
    """`flock` can fail for reasons that have nothing to do with another holder -- `ENOTSUP` on a
    filesystem that does not implement `flock` at all, `EIO`, `ENOLCK`. Reading any `OSError`
    whatsoever as "someone else holds it" turns those into a permanent false `in_flight`: on a
    filesystem where `flock` never works, this bug would make `h3 watch` hang forever on a run
    that may already be finished, or may never have started.
    """
    import errno

    import h3_48gb.runs as runs_mod

    checkpoint = write_checkpoint(tmp_path / "weird-fs" / "checkpoints" / "h3-a.safetensors",
                     completed=2, total=7, started_at="2026-08-10T21:00:00",
                     completed_at_start=0, written_at="2026-08-10T21:20:00")
    write_free_lock_file(checkpoint)  # a lock file must exist, or `_writer_alive` never calls flock

    def fake_flock(fd, operation):
        raise OSError(errno.ENOTSUP, "Operation not supported")

    monkeypatch.setattr(runs_mod.fcntl, "flock", fake_flock)

    assert {r.outdir.name: r.state for r in scan(tmp_path)}["weird-fs"] == "unknown"


def test_scan_finds_a_run_from_its_lock_file_before_any_checkpoint_exists(tmp_path):
    """The window between a writer taking its lock and its first checkpoint write -- a single
    forward can run for minutes -- must not read as nothing running (task 6). `scan` has to find
    this run from the lock file alone, with no `completed`/`total` to report -- `None`, not `0`,
    since there is genuinely no data yet rather than a checkpoint claiming zero progress.
    """
    import fcntl

    checkpoints = tmp_path / "warming-up" / "checkpoints"
    checkpoints.mkdir(parents=True)
    handle = open(checkpoints / "h3-abc.safetensors.lock", "wb")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    try:
        runs = scan(tmp_path)
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()

    assert len(runs) == 1, f"expected exactly one run found by its lock file alone, got {runs}"
    run = runs[0]
    assert run.outdir.name == "warming-up"
    assert run.state == "in_flight"
    assert run.completed is None
    assert run.total is None


def test_a_lock_file_flock_cannot_judge_is_not_reported_as_in_flight(tmp_path, monkeypatch):
    """The eleventh mutation on `scan`'s second pass (task 6, lock file with no checkpoint yet):
    `if _writer_alive(checkpoint) is not True: continue` reads `None` (unknown -- `flock` could not
    judge this lock at all, see `_writer_alive`) exactly like `False` (confirmed gone): both skip,
    so nothing false gets reported. Rewritten as `is False`, `None` stops being caught by the
    `continue` and falls through into `runs.append(Run(..., state="in_flight"))` a few lines below
    -- a lock file on a filesystem where `flock` never works (`ENOTSUP`, the same failure
    `test_a_non_busy_kernel_error_reads_unknown_not_in_flight` covers for a checkpoint that already
    has data) would then read as a permanently running writer, with nothing on disk that could ever
    change to clear it -- and `h3 watch` would poll it forever. See
    `test_watch_terminates_when_flock_cannot_judge_a_lock_only_run` (test_cli.py) for the same
    guarantee end to end through `run_watch`.
    """
    import errno

    import h3_48gb.runs as runs_mod

    checkpoints = tmp_path / "warming-up" / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "h3-abc.safetensors.lock").write_bytes(b"")  # no checkpoint file at all

    def fake_flock(fd, operation):
        raise OSError(errno.ENOTSUP, "Operation not supported")

    monkeypatch.setattr(runs_mod.fcntl, "flock", fake_flock)

    runs = scan(tmp_path)
    assert runs == [], (
        f"an unjudged lock file with no checkpoint must not be reported as any run "
        f"(certainly not in_flight), got {runs}")


def test_a_missing_written_at_reads_exactly_unknown_without_a_lock_file(tmp_path):
    """Pins task 3's fix precisely: a checkpoint missing `written_at` (the field the old writer set
    unconditionally, and the old `_state_of` treated as "not from this fork, so it must be dead")
    must read as `unknown`, not merely "some non-`in_flight` value" -- a test that only asserted
    `!= "in_flight"` would keep passing if this regressed to `"stale"`, which is exactly the
    overclaim task 3 exists to remove: nothing here proves the writer is gone, only that there is
    no lock file to check.
    """
    checkpoint = write_checkpoint(tmp_path / "no-written-at" / "checkpoints" / "h3-a.safetensors",
                                  completed=2, total=7, written_at=None)

    assert not lock_path_for(checkpoint).exists()
    run = scan(tmp_path)[0]
    assert run.state == "unknown"


def test_a_bad_started_at_does_not_hide_its_neighbours(tmp_path):
    """Valid JSON, unparsable `started_at`: `_state_of` reads `age_seconds` and
    `seconds_per_forward`, and the latter calls `_parse(started_at)` -- which raises ValueError on
    a string `datetime.fromisoformat` cannot read. State computation must happen inside the same
    protected `try` that builds the `Run`, or one such file aborts the whole scan -- exactly the
    regression task 1 fixed for metadata parsing itself.
    """
    write_checkpoint(tmp_path / "run-bad" / "checkpoints" / "h3-bad.safetensors",
                     completed=2, total=7, started_at="не-дата",
                     completed_at_start=0, written_at="2026-08-10T21:20:00")
    write_checkpoint(tmp_path / "run-good" / "checkpoints" / "h3-ok.safetensors",
                     completed=1, total=7, written_at="2026-08-10T21:00:00")

    runs = {r.outdir.name: r for r in scan(tmp_path)}

    assert runs["run-bad"].state == "unreadable"
    assert runs["run-bad"].error
    assert runs["run-good"].completed == 1, "a bad started_at neighbour swallowed a good run"


def test_subdirectories_are_found(tmp_path):
    """The reason `scan` recurses at all: runs land in one directory per experiment."""
    write_checkpoint(tmp_path / "13-ladder" / "checkpoints" / "h3-a.safetensors",
                     completed=2, total=7, written_at="2026-08-10T21:00:00")
    write_checkpoint(tmp_path / "18-baseline" / "checkpoints" / "h3-b.safetensors",
                     completed=5, total=7, written_at="2026-08-10T21:00:00")

    assert {r.outdir.name for r in scan(tmp_path)} == {"13-ladder", "18-baseline"}


def test_runs_does_not_import_mlx():
    """The module's whole value is starting instantly and testing without weights.

    Checked by reading the source rather than by inspecting `sys.modules`, because another test in
    the same session will already have imported mlx and the check would pass vacuously.
    """
    import h3_48gb.runs

    source = Path(h3_48gb.runs.__file__).read_text()
    for forbidden in ("import mlx", "minimax_h3_mlx", "h3_48gb.pipeline"):
        assert forbidden not in source, f"runs.py must not depend on {forbidden}"
