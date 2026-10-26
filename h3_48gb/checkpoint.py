"""Crash-safe checkpoints between denoising steps, and resumption from them.

A 5-second clip at 1344x768 costs ~586 s per step over 30 steps; ten seconds costs 15.7 hours. A
crash on step 29 currently throws away every one of them. Nothing about the loop makes that
necessary: the state carried from one step to the next is just two arrays — the video rows and the
audio rows — a few tens of megabytes next to hours of compute.

Determinism is what makes resumption *exact* rather than approximate, so it is worth stating what
was checked rather than assumed:

* All randomness in a run is drawn **before** the loop. ``MiniMaxH3Pipeline.__call__`` seeds twice
  — ``KEYFRAME_ENCODE_SEED`` for the keyframe posterior sample and the request ``seed`` for the
  conditioning noise, the video latents and the audio latents — and never touches
  ``mx.random`` again. ``dit.py`` has no dropout and no sampling; the only ``mx.random`` calls in
  the transformer and the VAEs are weight initializers that run at construction and are then
  overwritten by the checkpoint.
* ``MiniMaxH3Scheduler.step`` is rectified-flow Euler with ``eta = 0``: pure float32 arithmetic on
  ``(model_output, timestep, sample)`` plus one integer of state, ``_step_index``. No noise term.

So the state above really is the whole state, and `test_checkpoint.py` proves it bit-for-bit
against an interrupted run. The RNG key is still snapshotted into every checkpoint and restored at
the resume seam — belt and braces, and it turns "the loop draws no randomness" from an assumption
into something the code re-checks on every resume (see `ResumableRun.rng_note`).

Nothing here reimplements ``__call__``. Three seams carry the whole feature:

* the **DiT proxy** sees the full packed rows at the top of every step, and returns a zero velocity
  for steps that are being fast-forwarded — which keeps the transformer's weights off the machine
  until the first step that actually needs them;
* the **scheduler**, whose class is rebound to :class:`ResumableScheduler`, returns its input
  unchanged for fast-forwarded steps and the checkpointed rows at the seam. It has to be the
  scheduler and not just a zero velocity from the DiT: a zero velocity makes the Euler update
  ``r*x + (1-r)*x``, and ``r + (1 - r)`` is not exactly ``1`` in float32, so the state would drift
  by an ulp per skipped step;
* the **text-encoder proxy**, which stores the prompt conditioning in the checkpoint so a resume
  never loads the 28.2 GB encoder again.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import _upstream  # noqa: F401

import mlx.core as mx
import numpy as np

from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline
from minimax_h3_mlx.scheduler import MiniMaxH3Scheduler

#: Bumped whenever the on-disk layout changes in a way older files cannot satisfy.
FORMAT_VERSION = 1

#: The single safetensors metadata key everything else is nested under.
_META_KEY = "h3_checkpoint"

#: Request arguments that do not change the result and so do not belong in the run identity.
#:
#: ``condition_latent_rows`` is the one entry here that does NOT satisfy that description, and it
#: is a deliberate, temporary hole — read this before resuming anything that carries a latent tail.
#: `request_identity` hashes ``bound.arguments`` wholesale, so merely *adding* the parameter to
#: upstream's signature — even left at its ``None`` default — would change the digest of every run
#: this fork has ever made and orphan every checkpoint on disk. Excluding it keeps existing runs
#: byte-identical (tests/test_pipeline_latent.py pins the digest against a value captured before
#: the parameter existed), at the price that **the digest is blind to the tail**: two scenes that
#: differ only in the latent they continue from share one identity and therefore one checkpoint
#: file. Regenerate scene N-1, and an interrupted scene N resumes against the old context while
#: `_check_seam` sees a perfect match, because the run rebuilds exactly what is in the file now
#: (docs/FEASIBILITY-latent-handoff.md §4.2). Until task 4 of the latent-handoff wave — a
#: `FORMAT_VERSION` bump plus a real content digest of the tail, alongside `_checkpoint_path_for`
#: in cli.py — latent scenes must not be resumed. BACKLOG.md carries the same warning.
_IDENTITY_EXCLUDED = frozenset({"self", "verbose", "drop_adaln", "condition_latent_rows"})

#: Fork-only keyword arguments of ``__call__``, stripped before the upstream signature is bound.
CHECKPOINT_KWARGS = (
    "checkpoint_dir",
    "checkpoint_path",
    "resume",
    "keep_checkpoint",
    "on_corrupt",
    "checkpoint_prompt_embeds",
    "tag",
)


class CheckpointError(Exception):
    """Base class for every refusal raised by this module."""


class CheckpointMismatch(CheckpointError):
    """The checkpoint on disk belongs to a different run than the one being started."""


class CheckpointCorrupt(CheckpointError):
    """The checkpoint could not be read, or read back inconsistent."""


class CheckpointLocked(CheckpointError):
    """Another process already holds the exclusive lock this checkpoint needs to be written."""


def pop_checkpoint_kwargs(kwargs: dict) -> dict:
    """Remove and return this module's keyword arguments from a ``__call__`` kwargs dict.

    ``LazyMiniMaxH3Pipeline.__call__`` binds the *upstream* signature to validate the step count,
    and that bind rejects anything upstream does not declare.
    """
    return {name: kwargs.pop(name) for name in CHECKPOINT_KWARGS if name in kwargs}


# -- run identity --------------------------------------------------------------------------------

def _image_digest(image: Any) -> str:
    """A content digest for one keyframe, whatever form it arrives in.

    The ``dtype == object`` guard is not defensive tidying. ``np.asarray`` accepts *anything*: hand
    it a value it cannot describe numerically and it returns a 0-d object array whose ``tobytes()``
    is the **pointer address** of the boxed Python object. That hashes cleanly, so a run's identity
    would silently become a function of heap layout — the same image digesting differently between
    two processes, and two different images colliding whenever the allocator reused an address.
    A checkpoint keyed on that resumes the wrong run, or refuses to resume the right one, with no
    symptom pointing back here. Refuse instead: this codebase's rule everywhere else is that an
    unsupported input fails loudly rather than producing a plausible-looking wrong answer.
    """
    if isinstance(image, (str, Path)):
        path = Path(image)
        if path.exists():
            return "file:" + hashlib.blake2b(path.read_bytes(), digest_size=16).hexdigest()
        return "path:" + str(path)
    array = np.asarray(image)
    if array.dtype == object:
        raise TypeError(
            f"cannot digest a keyframe of type {type(image).__name__!r}: numpy describes it as an "
            "object array, whose bytes are a pointer address, not content — a run identity built "
            "from that would depend on heap layout. Pass a path, or an array of a numeric dtype "
            "(e.g. `np.asarray(PIL.Image.open(path))`)."
        )
    return (f"array:{array.shape}:{array.dtype}:"
            + hashlib.blake2b(array.tobytes(), digest_size=16).hexdigest())


def _jsonable(value: Any) -> Any:
    """Normalize a request argument into something stable under ``json.dumps(sort_keys=True)``."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items())}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return repr(value)


def request_identity(arguments: dict, extra: dict | None = None, tag: str | None = None) -> dict:
    """The fields that decide whether two runs are the same run.

    Geometry is deliberately *not* recomputed here. Canvas size, frame alignment and the latent
    grid are pure functions of these arguments plus the model configs, so duplicating that
    derivation would only create a second place for it to drift. What guards against drift instead
    is the row-shape check at the resume seam (:meth:`ResumableRun.adopt_rows`), which compares the
    checkpointed rows against the ones the live run actually built.

    ``tag`` is not part of upstream's ``__call__`` signature — it never reaches ``arguments`` — so
    it is threaded in as its own field instead. Without it, two runs that differ only by ``--tag``
    would collide on the same checkpoint file, which is not what a caller naming a run with `--tag`
    would expect. ``None`` omits the field entirely, so identities computed before this parameter
    existed (and callers that never pass a tag at all, e.g. ``run_bench.py``) are unaffected.
    """
    fields = {
        "format": FORMAT_VERSION,
        "request": {
            name: _jsonable(value)
            for name, value in sorted(arguments.items())
            if name not in _IDENTITY_EXCLUDED and name != "images"
        },
    }
    images = arguments.get("images")
    fields["request"]["images"] = [_image_digest(im) for im in images] if images else None
    if extra:
        fields["model"] = _jsonable(extra)
    if tag is not None:
        fields["tag"] = tag
    return fields


def identity_digest(identity: dict) -> str:
    raw = json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.blake2b(raw, digest_size=16).hexdigest()


def identity_diff(stored: dict, wanted: dict) -> list[str]:
    """Human-readable ``field: stored -> wanted`` lines for every leaf that differs.

    Leaves are elided past a line's worth: the weights fingerprint is a list of every shard in the
    checkpoint, and printing two of them would bury the one field the reader is looking for.
    """
    lines: list[str] = []

    def short(value: Any) -> str:
        text = repr(value)
        return text if len(text) <= 120 else f"{text[:117]}..."

    def walk(prefix: str, a: Any, b: Any) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                walk(f"{prefix}.{key}" if prefix else key, a.get(key, "<absent>"), b.get(key, "<absent>"))
            return
        if a != b:
            lines.append(f"  {prefix or '<root>'}: checkpoint has {short(a)}, this run has {short(b)}")

    walk("", stored, wanted)
    return lines


def weights_fingerprint(*roots: str | Path) -> dict:
    """A cheap identity for a set of weight directories: relative path and byte size per file.

    Cheap on purpose — hashing 40 GB of shards to decide whether to resume would cost more than a
    denoising step. Sizes catch a swapped or reconverted checkpoint, which is the failure that
    actually happens; they will not catch an in-place edit that preserves every file size, and that
    is an accepted limit of the check.
    """
    fingerprint: dict[str, list] = {}
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        entries = []
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in {".safetensors", ".json"}:
                entries.append([path.relative_to(root).as_posix(), path.stat().st_size])
        fingerprint[root.name] = entries
    return fingerprint


# -- storage -------------------------------------------------------------------------------------

@dataclass
class LoadedCheckpoint:
    """A checkpoint read back off disk and validated against the run that wants it."""

    path: Path
    arrays: dict[str, mx.array]
    meta: dict

    @property
    def completed_steps(self) -> int:
        return int(self.meta["completed_steps"])

    @property
    def total_forwards(self) -> int:
        return int(self.meta["total_forwards"])

    @property
    def prompt(self) -> tuple[mx.array, np.ndarray] | None:
        if "prompt_embeds" not in self.arrays:
            return None
        tags = np.asarray(self.arrays["text_token_tags"], dtype=np.int64)
        return self.arrays["prompt_embeds"], tags

    @property
    def rng_state(self) -> list[mx.array] | None:
        packed = self.arrays.get("rng_state")
        if packed is None:
            return None
        return [packed[i] for i in range(packed.shape[0])]


class CheckpointStore:
    """One checkpoint file, written atomically.

    ``mx.save_safetensors`` appends ``.safetensors`` when the name does not already end in it, so
    the temporary name has to keep that suffix or the rename would move the wrong path. The
    temporary file is hidden and lives in the destination directory, which is what makes
    ``os.replace`` atomic — a rename across filesystems is not.

    The temporary name carries the writing process's pid, so two processes never fight over the
    same partial file. Two processes targeting the *same* identity at once — two accidental
    invocations of the same command, say — are refused outright by :meth:`acquire_lock` instead:
    on a 48 GB machine, two identical requests running concurrently do not fit in memory anyway,
    and racing each other to `os.replace` the same checkpoint could walk its `completed_steps`
    backwards.
    """

    def __init__(self, path: str | Path):
        path = Path(path)
        if path.suffix != ".safetensors":
            path = path.with_name(path.name + ".safetensors")
        self.path = path
        self.bytes_written = 0
        self._lock_fd: int | None = None

    @property
    def lock_path(self) -> Path:
        """The liveness marker: a stable companion file a live writer holds an exclusive flock on.

        Named after the checkpoint itself rather than kept anywhere else, so a reader who only has
        the checkpoint path (``h3_48gb.runs`` never sees a ``CheckpointStore``) can derive it with
        the same one-line rule this property encodes: see ``runs._writer_alive``. Stable, and never
        deleted by this class (see :meth:`release_lock`): a reader must always find the same path
        to probe, whether the last writer exited cleanly, crashed, or has not written its first
        checkpoint yet.
        """
        return self.path.with_name(self.path.name + ".lock")

    # -- liveness

    def acquire_lock(self) -> None:
        """Take an exclusive, non-blocking flock on the companion file, held until
        :meth:`release_lock`. Raises :class:`CheckpointLocked` if another process holds it.

        Exclusive, not shared: two writers on the same identity racing to `os.replace` the same
        checkpoint is exactly the failure this lock exists to rule out, not merely to detect after
        the fact — a shared lock would let both hold it at once and neither would ever find out.
        An exclusive lock makes the second writer's attempt fail right here, before it reads a
        single byte of the checkpoint the first writer may already be resuming from. That ordering
        is why ``CheckpointingPipeline.__call__`` calls this *before* ``_load_for_resume``: taking
        the lock after the read would still let two processes both load the same
        ``completed_steps`` and then race each other to write, which is the same bug with the
        refusal arriving one step too late to prevent it.

        No timeout and no PID recorded anywhere: `fcntl.flock` is released by the kernel the moment
        this process's file descriptor closes, for any reason, including a crash that never runs a
        line of Python cleanup. That is the whole mechanism this feature, and
        ``runs._writer_alive``'s probe of it, rely on.
        """
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            raise CheckpointLocked(
                f"{self.lock_path} is already locked: another process is writing {self.path}, or "
                "(rarely) this filesystem does not honor flock. Refusing to start a second writer "
                "on the same checkpoint rather than risk two processes racing to replace the same "
                "file."
            ) from exc
        self._lock_fd = fd

    def release_lock(self) -> None:
        """Drop the exclusive flock. Idempotent, and safe with no lock held.

        Meant to run from a ``finally`` wrapping the whole generation, so it fires on success and
        on any exception that unwinds through Python. The companion file itself is deliberately
        *never* removed here — only the flock on it is.

        An earlier version also unlinked the file, on the theory that a lock file left behind
        unlocked already reads to a reader as "the writer is gone", so deleting it too was just
        tidying up. It was not: `unlink` drops the directory *name*, not the inode, and this
        process is not the only one that can have that name open at the moment it runs — a reader
        already mid-probe in `h3_48gb.runs._writer_alive`, or the next writer for the same
        identity starting up, can hold their own file descriptor on the very inode being unlinked.
        Whoever opens the path *after* the unlink gets a freshly created, lock-free file — plausible
        proof of "no writer", even while the unlinked inode's flock is still live under a
        descriptor nobody watching the path can reach any more. Leaving the file in place removes
        that window: a reader always opens the one inode a writer could actually be holding a lock
        on, and its answer — locked or not — was never ambiguous in the first place.
        """
        if self._lock_fd is not None:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(self._lock_fd)
                self._lock_fd = None

    # -- writing

    def _temp_path(self) -> Path:
        return self.path.with_name(f".{self.path.stem}.tmp-{os.getpid()}.safetensors")

    def write(self, arrays: dict[str, mx.array], meta: dict) -> None:
        """Write, flush to stable storage, then rename over the previous checkpoint.

        Order matters: without the ``fsync`` the rename can be durable while the bytes it points at
        are not, and a power loss would leave a checkpoint whose header promises data the file does
        not hold. Any failure removes the temporary file and leaves the previous checkpoint — which
        is a step behind but complete — exactly where it was.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self._temp_path()
        try:
            mx.save_safetensors(str(temp), arrays, metadata={_META_KEY: json.dumps(meta)})
            fd = os.open(temp, os.O_RDONLY)
            try:
                os.fsync(fd)
                self.bytes_written = os.fstat(fd).st_size
            finally:
                os.close(fd)
            os.replace(temp, self.path)
            dir_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise

    # -- reading

    def exists(self) -> bool:
        return self.path.exists()

    def read(self) -> LoadedCheckpoint:
        """Read the file back, or raise :class:`CheckpointCorrupt`.

        Every array is materialized here rather than left lazy: ``mx.load`` memory-maps the file,
        so a truncation past the header would otherwise surface as a fault somewhere deep in the
        denoising loop instead of as a refusal at startup.
        """
        try:
            arrays, raw_meta = mx.load(str(self.path), return_metadata=True)
            mx.eval(list(arrays.values()))
        except CheckpointError:
            raise
        except Exception as exc:
            raise CheckpointCorrupt(f"{self.path} could not be read: {exc}") from exc

        if _META_KEY not in raw_meta:
            raise CheckpointCorrupt(
                f"{self.path} is a safetensors file but not a MiniMax-H3 checkpoint: it carries no "
                f"{_META_KEY!r} metadata."
            )
        try:
            meta = json.loads(raw_meta[_META_KEY])
        except json.JSONDecodeError as exc:
            raise CheckpointCorrupt(f"{self.path}: checkpoint metadata is not valid JSON: {exc}") from exc

        required = {"format", "identity", "identity_digest", "completed_steps", "total_forwards"}
        missing = sorted(required - meta.keys())
        if missing:
            raise CheckpointCorrupt(f"{self.path}: checkpoint metadata is missing {missing}.")
        if int(meta["format"]) != FORMAT_VERSION:
            raise CheckpointCorrupt(
                f"{self.path} was written by format version {meta['format']}, this build reads "
                f"version {FORMAT_VERSION}."
            )
        if identity_digest(meta["identity"]) != meta["identity_digest"]:
            raise CheckpointCorrupt(
                f"{self.path}: the recorded identity digest does not match the recorded identity."
            )
        for name in ("video_gen", "audio_gen"):
            if name not in arrays:
                raise CheckpointCorrupt(f"{self.path}: no {name!r} rows in the checkpoint.")
        if not 0 <= int(meta["completed_steps"]) <= int(meta["total_forwards"]):
            raise CheckpointCorrupt(
                f"{self.path}: completed_steps={meta['completed_steps']} is outside "
                f"0..{meta['total_forwards']}."
            )
        return LoadedCheckpoint(self.path, arrays, meta)

    # -- lifecycle

    def quarantine(self) -> Path:
        """Move an unreadable checkpoint aside instead of deleting it, and return the new path."""
        target = self.path.with_name(f"{self.path.name}.corrupt-{int(time.time())}")
        os.replace(self.path, target)
        return target

    def discard(self) -> None:
        """Drop the checkpoint after a successful run. Leftover temporaries go with it."""
        self.path.unlink(missing_ok=True)
        for stale in self.path.parent.glob(f".{self.path.stem}.tmp-*.safetensors"):
            stale.unlink(missing_ok=True)


# -- the run controller --------------------------------------------------------------------------

def _rows_equal(a: mx.array, b: mx.array) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(a, b).item())


@dataclass
class ResumableRun:
    """Shared state between the three proxies for the duration of one ``__call__``.

    A checkpoint is written once both modalities have reported their stepped rows, so the file on
    disk is always a whole step — never video from step ``i`` beside audio from step ``i - 1``.
    """

    store: CheckpointStore
    identity: dict
    loaded: LoadedCheckpoint | None = None
    verbose: bool = True
    store_prompt_embeds: bool = True

    total_forwards: int | None = None
    forward_index: int = -1
    writes: int = 0
    #: Set when the RNG key at the resume seam differs from the one the checkpoint recorded, which
    #: would mean the denoising loop consumes randomness. It never has; see the module docstring.
    rng_note: str | None = None

    #: When *this session* began, and how many forwards a resumed checkpoint already carried.
    #: Both describe the session rather than the run so that a rate computed from them measures
    #: the machine's current speed: dividing this session's elapsed time by every forward,
    #: including ones a previous session paid for, reports a resumed run several times too fast.
    #: Metadata, never identity -- neither enters `identity_digest`.
    started_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    completed_at_start: int = 0

    _rows: dict[str, mx.array] = field(default_factory=dict)
    _stepped: dict[str, mx.array] = field(default_factory=dict)
    _prompt: tuple[mx.array, np.ndarray] | None = None
    _seam_checked: bool = False

    @property
    def start_step(self) -> int:
        return self.loaded.completed_steps if self.loaded is not None else 0

    def should_skip(self, index: int) -> bool:
        return index < self.start_step

    # -- prompt conditioning

    def replay_prompt(self) -> tuple[mx.array, np.ndarray] | None:
        return self.loaded.prompt if self.loaded is not None else None

    def note_prompt(self, embeds: mx.array, token_tags: np.ndarray) -> None:
        self._prompt = (embeds, np.asarray(token_tags))

    # -- schedule

    def note_total_forwards(self, count: int) -> None:
        """Record how many transformer evaluations this schedule really drives.

        Not the same as ``num_inference_steps``: the sigma shift collapses float32 duplicates near
        ``sigma = 1``, and the terminal sigma costs one grid point, so the count is derived rather
        than requested. A checkpoint written under a different count is not resumable into this run.
        """
        self.total_forwards = int(count)
        if self.loaded is None:
            return
        if self.loaded.total_forwards != self.total_forwards:
            raise CheckpointMismatch(
                f"{self.store.path} was written for a schedule of {self.loaded.total_forwards} "
                f"transformer evaluations, but this run drives {self.total_forwards}. The run "
                "identity matched, so this is a scheduler change rather than a different request; "
                "delete the checkpoint to start over."
            )
        if self.loaded.completed_steps > self.total_forwards:
            raise CheckpointCorrupt(
                f"{self.store.path} claims {self.loaded.completed_steps} completed steps out of "
                f"{self.total_forwards}."
            )

    # -- per-step hooks

    def begin_forward(self, video_rows: mx.array, audio_rows: mx.array) -> int:
        """Called by the DiT proxy with the full packed rows entering a step."""
        self.forward_index += 1
        self._rows = {"video": video_rows, "audio": audio_rows}
        if self.loaded is not None and self.forward_index == self.start_step:
            self._check_seam()
        return self.forward_index

    def _check_seam(self) -> None:
        """Verify what the resumed run rebuilt for itself against what the checkpoint recorded.

        Only the conditioning rows are checked, because only they are rebuilt: the generated rows
        come straight out of the checkpoint. They are compared in bfloat16 — the precision the
        transformer is handed them at — which is the criterion that matters, since conditioning
        rows reach the model and nothing else. They are never decoded and never written back.
        """
        for modality, rows in self._rows.items():
            stored = self.loaded.arrays.get(f"{modality}_cond")
            n_cond = 0 if stored is None else stored.shape[0]
            n_gen = self.loaded.arrays[f"{modality}_gen"].shape[0]
            if rows.shape[0] != n_cond + n_gen:
                raise CheckpointMismatch(
                    f"{modality} rows: the checkpoint holds {n_cond} conditioning + {n_gen} "
                    f"generated rows, this run built {rows.shape[0]}. The geometry of the two runs "
                    "differs — check duration, canvas size and keyframes."
                )
            if n_cond and not _rows_equal(rows[:n_cond].astype(stored.dtype), stored):
                raise CheckpointMismatch(
                    f"{modality} conditioning rows were rebuilt differently from the ones in "
                    f"{self.store.path}. Keyframe encoding is seeded and should reproduce exactly; "
                    "a difference here means the keyframe images, the canvas or the video VAE are "
                    "not the ones the checkpoint was made with."
                )
        self._restore_rng()
        self._seam_checked = True

    def _restore_rng(self) -> None:
        saved = self.loaded.rng_state
        if not saved:
            return
        live = list(mx.random.state)
        matched = len(live) == len(saved) and all(_rows_equal(a, b) for a, b in zip(live, saved))
        if not matched:
            self.rng_note = (
                "the MLX RNG key at the resume seam differs from the one recorded in the "
                "checkpoint, which means the denoising loop now consumes randomness; the recorded "
                "key was restored, so the resumed run still follows the original draw sequence"
            )
            if self.verbose:
                print(f"  note: {self.rng_note}", flush=True)
        mx.random.state = saved

    def adopt_rows(self, modality: str, sample: mx.array) -> mx.array:
        """The checkpointed generated rows, handed back at the seam step in place of an Euler step."""
        stored = self.loaded.arrays[f"{modality}_gen"]
        if stored.shape != sample.shape or stored.dtype != sample.dtype:
            raise CheckpointMismatch(
                f"{modality} rows in {self.store.path} are {stored.shape} {stored.dtype}, but this "
                f"run generates {sample.shape} {sample.dtype}. The two runs do not share a geometry."
            )
        return stored

    def record(self, modality: str, index: int, stepped: mx.array) -> None:
        """Collect one modality's stepped rows; write the checkpoint once both have arrived."""
        if self.should_skip(index):
            return
        self._stepped[modality] = stepped
        if len(self._stepped) < len(self._rows):
            return
        self._write(completed=index + 1)
        self._stepped = {}

    def _write(self, completed: int) -> None:
        arrays: dict[str, mx.array] = {}
        row_counts = {}
        for modality, full in self._rows.items():
            generated = self._stepped[modality]
            n_cond = full.shape[0] - generated.shape[0]
            if n_cond < 0:
                raise RuntimeError(
                    f"{modality}: the scheduler stepped {generated.shape[0]} rows out of a packed "
                    f"sequence of {full.shape[0]} — the pipeline layout is not what this expects."
                )
            if n_cond:
                # As handed to the transformer, i.e. bfloat16: conditioning rows only ever reach
                # the model, so that is the precision at which reproducing them matters.
                arrays[f"{modality}_cond"] = full[:n_cond]
            arrays[f"{modality}_gen"] = generated
            row_counts[modality] = [n_cond, generated.shape[0]]

        if self.store_prompt_embeds and self._prompt is not None:
            embeds, tags = self._prompt
            arrays["prompt_embeds"] = embeds
            arrays["text_token_tags"] = mx.array(np.asarray(tags, dtype=np.int32))

        state = list(mx.random.state)
        if state:
            arrays["rng_state"] = mx.stack([k.reshape(-1) for k in state])

        # Materialize before stamping the clock, because until this line `arrays` is a graph over
        # the step that just "finished" -- MLX has not run it yet. `save_safetensors` below would
        # force it anyway, which means the whole forward pass executes *inside* the write, and a
        # `written_at` taken before that records when the step began rather than when the
        # checkpoint landed. Measured on the 15 s run of 2026-08-11: the file's mtime was 12:29:52
        # and the timestamp inside it said 12:11:46, eighteen minutes early -- one whole step. That
        # is not cosmetic: `h3 status` divides by this interval, and it reported "19 s per forward,
        # 2 minutes left" for a run with 1086 s per forward and an hour and three quarters to go.
        # Same discipline as the unload path, for the same reason: eval, then act on the result.
        mx.eval(*arrays.values())

        meta = {
            "format": FORMAT_VERSION,
            "identity": self.identity,
            "identity_digest": identity_digest(self.identity),
            "completed_steps": int(completed),
            "total_forwards": int(self.total_forwards),
            "row_counts": row_counts,
            "has_prompt_embeds": "prompt_embeds" in arrays,
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "started_at": self.started_at,
            "completed_at_start": self.completed_at_start,
        }
        self.store.write(arrays, meta)
        self.writes += 1
        if self.verbose:
            print(f"  checkpoint: {completed}/{self.total_forwards} steps, "
                  f"{self.store.bytes_written / 1e6:.1f} MB -> {self.store.path}", flush=True)


# -- proxies -------------------------------------------------------------------------------------

class ResumableScheduler(MiniMaxH3Scheduler):
    """A scheduler that can fast-forward over steps a checkpoint already covers.

    Installed by rebinding an existing scheduler's ``__class__`` — the same trick
    :class:`h3_48gb.dit.CachedFinalLayer` uses — so the object stays a genuine
    ``MiniMaxH3Scheduler`` for everything that inspects it, and no upstream construction logic is
    duplicated here.

    Returning ``sample`` unchanged for a skipped step is what keeps the fast-forward exact. The
    tempting alternative — feed the real ``step`` a zero velocity — computes
    ``ratio * x + (1 - ratio) * x``, and float32 does not guarantee that back to ``x``.
    """

    def attach(self, run: ResumableRun, modality: str) -> None:
        self._run = run
        self._modality = modality

    def step(self, model_output: mx.array, timestep: float, sample: mx.array) -> mx.array:
        run = self._run
        index = self._step_index if self._step_index is not None else self.index_for_timestep(timestep)
        if index != run.forward_index:
            raise RuntimeError(
                f"{self._modality} scheduler is at step {index} while the transformer proxy counted "
                f"{run.forward_index}. Checkpointing assumes exactly one forward per step."
            )
        if run.should_skip(index):
            self._step_index = index + 1
            return run.adopt_rows(self._modality, sample) if index == run.start_step - 1 else sample

        stepped = super().step(model_output, timestep, sample)
        run.record(self._modality, index, stepped)
        return stepped


class _Proxy:
    """Attribute-forwarding wrapper. Dunders are not forwarded, so copy/pickle probes cannot recurse."""

    _OWN: frozenset[str] = frozenset()

    def __getattr__(self, item):
        if item in type(self)._OWN or item.startswith("__"):
            raise AttributeError(item)
        return getattr(self.__dict__["_inner"], item)


class StepInterceptor(_Proxy):
    """The DiT, wrapped so every step is seen before it runs.

    For a step the checkpoint already covers, the forward is replaced by a zero velocity of the
    right shape. The scheduler discards it, so the value is irrelevant — what matters is that the
    transformer's ``__call__`` is never entered, which means fast-forwarding 29 of 30 steps reads
    no weights and costs no time.
    """

    _OWN = frozenset({"_inner", "_run"})

    def __init__(self, dit, run: ResumableRun):
        self.__dict__["_inner"] = dit
        self.__dict__["_run"] = run

    def __call__(self, video_latents, audio_latents, *args, **kwargs):
        run = self.__dict__["_run"]
        index = run.begin_forward(video_latents[0], audio_latents[0])
        if run.should_skip(index):
            return mx.zeros_like(video_latents), mx.zeros_like(audio_latents)
        return self.__dict__["_inner"](video_latents, audio_latents, *args, **kwargs)


class PromptRelay(_Proxy):
    """The text encoder, wrapped so its output lands in the checkpoint and can be replayed.

    Prompt conditioning is a few megabytes and costs 28.2 GB of residency plus minutes of compute to
    produce. Putting it in the checkpoint is what makes a resume cheap rather than merely correct:
    the encoder is never loaded again.
    """

    _OWN = frozenset({"_inner", "_run"})

    def __init__(self, text_encoder, run: ResumableRun):
        self.__dict__["_inner"] = text_encoder
        self.__dict__["_run"] = run

    def encode(self, prompt, images=None):
        run = self.__dict__["_run"]
        replayed = run.replay_prompt()
        if replayed is not None:
            if run.verbose:
                print("  prompt conditioning replayed from the checkpoint "
                      "(the text encoder is not loaded)", flush=True)
            run.note_prompt(*replayed)
            return replayed
        embeds, token_tags = self.__dict__["_inner"].encode(prompt, images)
        # Materialize before it is handed on: the checkpoint writer would force this anyway, and
        # doing it here keeps the encoder's unload (LazyTextEncoder.encode) the last word.
        mx.eval(embeds)
        run.note_prompt(embeds, token_tags)
        return embeds, token_tags


# -- the mixin -----------------------------------------------------------------------------------

class CheckpointingPipeline:
    """Mixin adding checkpoint/resume to :class:`minimax_h3_mlx.pipeline.MiniMaxH3Pipeline`.

    Mix in *before* the pipeline::

        class MyPipeline(CheckpointingPipeline, MiniMaxH3Pipeline): ...

    Then pass ``checkpoint_dir=`` (or ``checkpoint_path=``) to ``__call__``. Without either, the
    call is upstream's, untouched.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._resume_run: ResumableRun | None = None

    # -- hooks subclasses fill in

    def checkpoint_identity_extra(self) -> dict:
        """Everything outside the request that changes the result — weights, above all."""
        return {}

    # -- schedule

    def _build_schedules(self, num_inference_steps: int):
        video, audio = super()._build_schedules(num_inference_steps)
        run = getattr(self, "_resume_run", None)
        if run is not None:
            for scheduler, modality in ((video, "video"), (audio, "audio")):
                scheduler.__class__ = ResumableScheduler
                scheduler.attach(run, modality)
            run.note_total_forwards(len(video.timesteps))
        return video, audio

    # -- generation

    def __call__(self, *args, **kwargs):
        options = pop_checkpoint_kwargs(kwargs)
        if options.get("checkpoint_dir") is None and options.get("checkpoint_path") is None:
            return super().__call__(*args, **kwargs)

        import inspect

        bound = inspect.signature(MiniMaxH3Pipeline.__call__).bind(self, *args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        verbose = bool(arguments.get("verbose", True))

        identity = request_identity(arguments, self.checkpoint_identity_extra(), tag=options.get("tag"))
        store = _resolve_store(options, identity)

        # Acquired here — right after `store` is known, before anything reads the checkpoint off
        # disk — and released in the `finally` below. This is also the earliest a lock file could
        # exist, which matters on its own: `_load_for_resume` and the loop it precedes can run for
        # a long time (a single forward is minutes; see the module docstring) before the first
        # checkpoint write, and `h3_48gb.runs.scan` needs a lock file on disk through that whole
        # window to tell a live run apart from nothing running. Taking the lock any later than this
        # would also reopen the race it exists to close: two processes both calling
        # `_load_for_resume` before either held the lock could both read the same `completed_steps`
        # and then race each other to `os.replace`, leaving the checkpoint a step behind whichever
        # wrote last. See `CheckpointStore.acquire_lock` for why the lock is exclusive, not shared,
        # and `h3_48gb.runs._writer_alive` for the liveness signal it gives a reader.
        store.acquire_lock()
        try:
            loaded = _load_for_resume(
                store,
                identity,
                resume=options.get("resume", True),
                on_corrupt=options.get("on_corrupt", "restart"),
                verbose=verbose,
            )
            run = ResumableRun(
                store=store,
                identity=identity,
                loaded=loaded,
                verbose=verbose,
                store_prompt_embeds=bool(options.get("checkpoint_prompt_embeds", True)),
                completed_at_start=loaded.completed_steps if loaded is not None else 0,
            )

            original_dit, original_text_encoder = self.dit, self.text_encoder
            self._resume_run = run
            self.dit = StepInterceptor(original_dit, run)
            self.text_encoder = PromptRelay(original_text_encoder, run)
            try:
                result = super().__call__(*args, **kwargs)
            finally:
                self.dit, self.text_encoder = original_dit, original_text_encoder
                self._resume_run = None

            # Only on success. A crash anywhere above — including in the VAE decode, which is its
            # own memory cliff on a 48 GB machine — leaves the checkpoint for the next attempt.
            if not options.get("keep_checkpoint", False):
                store.discard()
            elif verbose:
                print(f"  checkpoint kept at {store.path}", flush=True)
            return result
        finally:
            store.release_lock()


def _resolve_store(options: dict, identity: dict) -> CheckpointStore:
    """An explicit path wins; otherwise the file is named after the run so runs cannot collide."""
    explicit = options.get("checkpoint_path")
    if explicit is not None:
        return CheckpointStore(explicit)
    return CheckpointStore(Path(options["checkpoint_dir"]) / f"h3-{identity_digest(identity)}.safetensors")


def _load_for_resume(
    store: CheckpointStore,
    identity: dict,
    resume: bool,
    on_corrupt: str,
    verbose: bool,
) -> LoadedCheckpoint | None:
    """Find and validate a checkpoint, or decide there is nothing to resume from.

    Parameter mismatch is a hard refusal, never a silent continuation: quietly denoising someone
    else's latents for fifteen hours is a worse outcome than an error message. Unreadable *data*
    is different — there is nothing to continue from either way — so by default the file is moved
    aside with a loud warning and the run starts over. ``on_corrupt="raise"`` turns that into a
    refusal too.
    """
    if on_corrupt not in ("restart", "raise"):
        raise ValueError(f"`on_corrupt` must be 'restart' or 'raise', got {on_corrupt!r}.")
    if not store.exists():
        if verbose:
            print(f"checkpointing to {store.path} (no checkpoint to resume from)", flush=True)
        return None
    if not resume:
        if verbose:
            print(f"resume disabled; {store.path} will be overwritten from step 0", flush=True)
        return None

    try:
        loaded = store.read()
    except CheckpointCorrupt as exc:
        if on_corrupt == "raise":
            raise
        moved = store.quarantine()
        print(f"WARNING: {exc}\n         moved aside to {moved}; starting from step 0.", flush=True)
        return None

    if loaded.meta["identity_digest"] != identity_digest(identity):
        differences = identity_diff(loaded.meta["identity"], identity) or ["  (no leaf differs — "
                                                                          "the digest itself is inconsistent)"]
        raise CheckpointMismatch(
            f"{store.path} belongs to a different run and will not be resumed:\n"
            + "\n".join(differences)
            + "\n  Pass a different `checkpoint_path`, or delete this file to start over."
        )

    if loaded.completed_steps == 0:
        if verbose:
            print(f"{store.path} holds no completed steps; starting from step 0", flush=True)
        return None
    if verbose:
        print(f"resuming {store.path}: {loaded.completed_steps}/{loaded.total_forwards} steps "
              f"already done, {loaded.total_forwards - loaded.completed_steps} to go", flush=True)
    return loaded
