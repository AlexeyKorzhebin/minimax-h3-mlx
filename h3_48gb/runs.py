"""Read runs off disk without loading a model.

A run leaves two kinds of artifact under `--outdir`: a resume checkpoint while it is in flight,
and a `<stem>.json` report once it finishes. Both are readable without MLX -- a safetensors file
is an 8-byte length, a JSON header, then tensor bytes this module never touches -- so a caller can
ask "how far along is it" for the price of a few hundred bytes, on a run this process did not
launch and does not own the terminal of.

Deliberately no dependency on MLX, the minimax-h3-mlx package, or h3_48gb's own pipeline module:
the point is that `h3 status` starts instantly and that these tests need no weights.
"""
from __future__ import annotations

import errno
import fcntl
import json
import struct
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

#: safetensors headers here are JSON and small. A junk file read as one yields an absurd length
#: prefix, and reading that many bytes raises MemoryError rather than returning garbage -- which
#: no reasonable `except` clause on a parser would list. Bounding the read turns a crash into the
#: `unreadable` state every caller already handles. Mirrors `_MAX_HEADER_BYTES` in cli.py.
_MAX_HEADER_BYTES = 8 << 20

_META_KEY = "h3_checkpoint"

#: The suffix a live writer's companion lock file carries, appended to the checkpoint's own name.
#: Must match `CheckpointStore.lock_path` in `checkpoint.py` exactly -- the two are never allowed
#: to share an import (`checkpoint.py` pulls in `mlx`; this module may not), so this is the one
#: piece of the contract duplicated instead of shared. A test that reconstructs this suffix by
#: reading `CheckpointStore.lock_path` (or this very constant) cannot catch the two sides
#: diverging, because it would drift along with whichever side changed. The one test that can is
#: `test_checkpoint.py::test_a_real_write_is_visible_to_the_reader_end_to_end`, which drives a real
#: writer through `checkpoint.py` and a real reader through this module's own `scan` and asserts
#: the two still agree -- see that test's docstring for the mutation that would slip past every
#: other lock test in the suite without it.
_LOCK_SUFFIX = ".lock"

#: `fcntl.flock`'s non-blocking failure mode when *another* holder has the file: `EWOULDBLOCK`
#: and `EAGAIN` are the same errno on most platforms but not guaranteed to be, so both are listed;
#: `EACCES` covers `fcntl`-based lock emulations (some platforms map lock contention there instead
#: of `EAGAIN`). Anything else -- `ENOTSUP` on a filesystem that does not implement `flock` at all,
#: `EIO`, `ENOLCK` -- means the probe could not test anything, not that someone is holding it; see
#: `_writer_alive`.
_BUSY_ERRNOS = frozenset({errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES})


def _parse(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp)


def _now() -> datetime:
    """Indirected so tests can pin it."""
    return datetime.now()


@dataclass
class Run:
    """One run, as far as the files on disk can describe it."""

    outdir: Path
    tag: str | None = None
    stem: str | None = None
    state: str = "unreadable"
    completed: int | None = None
    total: int | None = None
    identity_digest: str | None = None
    identity: dict = field(default_factory=dict)
    error: str | None = None
    started_at: str | None = None
    completed_at_start: int = 0
    written_at: str | None = None

    @property
    def fraction(self) -> float | None:
        if not self.total:
            return None
        return (self.completed or 0) / self.total

    @property
    def seconds_this_session(self) -> float | None:
        if not (self.started_at and self.written_at):
            return None
        return (_parse(self.written_at) - _parse(self.started_at)).total_seconds()

    @property
    def forwards_this_session(self) -> int:
        return (self.completed or 0) - self.completed_at_start

    @property
    def seconds_per_forward(self) -> float | None:
        """None, not a crash, in the window between a resume and its first write."""
        elapsed, done = self.seconds_this_session, self.forwards_this_session
        if elapsed is None or done <= 0:
            return None
        return elapsed / done

    @property
    def eta_seconds(self) -> float | None:
        rate = self.seconds_per_forward
        if rate is None or self.total is None:
            return None
        return rate * (self.total - (self.completed or 0))

    @property
    def age_seconds(self) -> float | None:
        if not self.written_at:
            return None
        return (_now() - _parse(self.written_at)).total_seconds()

    def as_dict(self) -> dict:
        """Flat JSON, absolute paths, computed fields included so a caller need not recompute."""
        return {
            "outdir": str(self.outdir), "tag": self.tag, "state": self.state,
            "completed": self.completed, "total": self.total, "fraction": self.fraction,
            "started_at": self.started_at, "written_at": self.written_at,
            "age_seconds": self.age_seconds,
            "seconds_per_forward": self.seconds_per_forward,
            "eta_seconds": self.eta_seconds,
            "identity_digest": self.identity_digest, "error": self.error,
        }


def read_checkpoint_meta(path: Path) -> dict:
    """The `h3_checkpoint` metadata block, or raise ValueError describing why not."""
    size = path.stat().st_size
    with open(path, "rb") as fh:
        length = struct.unpack("<Q", fh.read(8))[0]
        if length > min(_MAX_HEADER_BYTES, size):
            raise ValueError(f"header length {length} exceeds the file")
        header = json.loads(fh.read(length))
    raw = header.get("__metadata__", {})
    if _META_KEY not in raw:
        raise ValueError(f"no {_META_KEY!r} metadata")
    return json.loads(raw[_META_KEY])


def _writer_alive(checkpoint: Path) -> bool | None:
    """Whether a live writer holds the checkpoint's companion lock file.

    ``True``  -- another process holds the lock right now: some writer has this checkpoint open.
    ``False`` -- the companion file exists, but nothing holds its lock. A writer held it at some
                 point and is now gone -- a fact, not a guess: the kernel drops an `fcntl.flock`
                 lock the instant the holding file descriptor closes, for any reason, including a
                 crash that never ran a line of the writer's own cleanup code.
    ``None``  -- either there is no companion file to check at all (this checkpoint predates the
                 code that writes one), or there is one but `flock` could not judge it -- some
                 kernel/filesystem error unrelated to another holder (`ENOTSUP` on a filesystem
                 that does not implement `flock`, `EIO`, `ENOLCK`, ...). Either way there is
                 nothing on disk this probe can vouch for, so there is nothing to say.

    The probe takes its own non-blocking *shared* lock and releases it immediately. The writer's
    own lock is exclusive (`checkpoint.CheckpointStore.acquire_lock`), and an exclusive lock blocks
    a conflicting shared attempt exactly as reliably as it blocks another exclusive one -- shared is
    what a reader actually wants, since it does not also exclude a second, concurrent reader.
    Failure with an errno that means "something else is holding it" (`EWOULDBLOCK`/`EAGAIN`, or
    `EACCES` on platforms that report contention that way) is proof of a live holder; failure with
    any other errno is not evidence of anything and must not be read as one -- see `_BUSY_ERRNOS`.
    No timeout, no stored pid to go stale, no constant to tune.
    """
    lock_path = checkpoint.with_name(checkpoint.name + _LOCK_SUFFIX)
    try:
        handle = open(lock_path, "rb")
    except FileNotFoundError:
        return None
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError as exc:
            return True if exc.errno in _BUSY_ERRNOS else None
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return False
    finally:
        handle.close()


def _state_of(run: Run, checkpoint: Path) -> str:
    """"unreadable" for a Run this can't be reached on -- construction already raised for that.

    Liveness comes from the companion lock file alone -- see `_writer_alive`. No age, no measured
    rate, no threshold of either: a checkpoint can go quiet for a slow forward or a long VAE decode
    while its writer is very much alive, and a run can crash the instant after a write. Telling
    those apart from silence and a clock is exactly the thing no fixed window can do without
    guessing -- which is why an earlier version of this function tried anyway (a four-hour ceiling
    on `rate is None`, and a `3 * rate + 120s` window otherwise) and still got a live run called
    dead. The lock does not need to guess: `age_seconds` stays in `Run.as_dict()` as information
    for a human, never as an input to this decision.
    """
    alive = _writer_alive(checkpoint)
    if alive is True:
        return "in_flight"
    if alive is False:
        return "stale"
    return "unknown"


def scan(root: Path) -> list[Run]:
    """Every run under `root`, found recursively. Never raises for a file it finds."""
    root = Path(root)
    runs: list[Run] = []
    found: set[Path] = set()
    for checkpoint in sorted(root.rglob("checkpoints/h3-*.safetensors")):
        found.add(checkpoint)
        outdir = checkpoint.parent.parent
        try:
            meta = read_checkpoint_meta(checkpoint)
            run = Run(
                outdir=outdir,
                completed=int(meta.get("completed_steps", 0)),
                total=int(meta.get("total_forwards", 0)) or None,
                identity_digest=meta.get("identity_digest"),
                identity=meta.get("identity", {}),
                started_at=meta.get("started_at"),
                completed_at_start=int(meta.get("completed_at_start", 0)),
                written_at=meta.get("written_at"),
            )
            # `_state_of` itself no longer touches these -- liveness now comes from the lock file
            # alone -- but `age_seconds` and `seconds_per_forward` still call `_parse` on them
            # lazily, whenever a caller (`Run.as_dict`, `format_status`) reads those properties
            # outside of any try at all. Validating both here, inside the same protected try that
            # builds `run`, turns a checkpoint with well-formed JSON but an unparsable timestamp
            # into one "unreadable" run instead of a `ValueError` raised later out of a caller that
            # has no reason to expect one -- the same isolation `read_checkpoint_meta` gives the
            # JSON itself, extended to the two fields this module parses as datetimes.
            if run.started_at is not None:
                _parse(run.started_at)
            if run.written_at is not None:
                _parse(run.written_at)
            run.state = _state_of(run, checkpoint)
        except (OSError, ValueError, TypeError, AttributeError, KeyError, struct.error,
                MemoryError) as exc:
            # The bytes on disk are untrusted: a queue writes checkpoints by rename, so a reader
            # can see a truncated file (OSError/struct.error) or a header cut off mid-line
            # (ValueError from json.loads). But well-formed-JSON-with-a-surprising-shape is just
            # as real -- a review of this function found three cases live JSON never produces
            # from a healthy writer but a hand-edited or future-format file can: `h3_checkpoint`
            # nested as an object instead of the JSON string mx.save_safetensors always writes
            # (TypeError from the inner json.loads), `completed_steps: null` (TypeError from
            # int(None)), and a top-level header that parses to a list, not an object
            # (AttributeError from header.get). All of those are shape problems in one file, not
            # bugs in this loop, so they get folded into the same "unreadable" outcome. The catch
            # stays enumerated rather than `except Exception` on purpose: a typo in the Run(...)
            # call above -- a bad keyword, a NameError from a future edit -- must still crash
            # loudly instead of being misreported as a corrupt checkpoint.
            runs.append(Run(outdir=outdir, error=f"{type(exc).__name__}: {exc}"))
            continue
        runs.append(run)

    # A writer takes its companion lock at the very start of generation, well before the first
    # checkpoint write -- a single forward can run for minutes (see `checkpoint.py`'s module
    # docstring), so `checkpoints/` can hold nothing but a lock file for most of an hour of a run
    # that is very much alive. Without this second pass, the loop above finds nothing during that
    # whole window, and `h3 watch` would report "nothing running" and exit on it. `completed` and
    # `total` stay `None` here (`Run`'s own defaults) rather than `0`: there genuinely is no
    # progress data yet, and `0` would claim one where the truth is "not written yet".
    for lock_file in sorted(root.rglob(f"checkpoints/h3-*.safetensors{_LOCK_SUFFIX}")):
        checkpoint = lock_file.with_name(lock_file.name[: -len(_LOCK_SUFFIX)])
        if checkpoint in found:
            continue  # already reported above, with real progress data
        if _writer_alive(checkpoint) is not True:
            # No confirmed live holder and no checkpoint to read: either a writer crashed before
            # ever writing (an unheld lock file, nothing more to say than that), or the checkpoint
            # was written in the gap between the two `rglob` calls above and will be picked up,
            # with real data, on the next scan. Either way there is nothing to add here.
            continue
        runs.append(Run(outdir=checkpoint.parent.parent, state="in_flight"))
    return runs
