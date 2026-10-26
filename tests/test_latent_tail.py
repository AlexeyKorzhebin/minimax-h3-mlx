#!/usr/bin/env python3
"""Cut a scene's latent tail off at the one seam where it is still raw, and hand it back later.

    ./.venv/bin/pytest tests/test_latent_tail.py -q

`docs/FEASIBILITY-latent-handoff.md` §1.2-§1.3. Task 1 taught `packing.py` to anchor a
conditioning block at an arbitrary pixel-frame index; task 2 taught upstream's `__call__` to
accept those blocks as raw rows (`condition_latent_rows`). This file is the two ends that make
those usable from the command line: `--save-latent-tail N` on the scene that produces the tail,
`--latent <path>` on the scene that continues from it.

**Where the write happens, and why exactly there.** `LazyMiniMaxH3Pipeline._decode_video` opens
with `mx.eval(rows)` and then drops the transformer. Those rows are the finished video latent,
still patchified and still normalized -- the same space `_encode_keyframes` produces, so the tail
needs no conversion at either end (spec §1.2, "Ни одного преобразования между сохранением и
приёмом нет"). One line later `unpatchify_video_tokens` and the mean/std denormalize have moved it
out of that space, and the VAE has begun turning it into pixels. So the seam is *between*
`mx.eval(rows)` and the unpatchify, and `test_the_tail_is_written_after_the_rows_are_materialized`
is what holds it there -- not by checking that some `mx.eval` happened earlier (the previous
step's own eval satisfies that, one iteration late; `h3_48gb/checkpoint.py`'s `written_at` bug was
exactly that off-by-one) but by identifying the *frame* that called `mx.eval` and the *object* it
was handed.

**Why the kwargs are fork-only.** `save_latent_tail` / `latent_tail_stem` are popped in
`LazyMiniMaxH3Pipeline.__call__` before upstream's signature is bound, exactly as
`h3_48gb.preview.PREVIEW_KWARGS` are. `request_identity` hashes `bound.arguments` wholesale, so
anything surviving that bind joins the checkpoint identity -- and *where a scene writes its tail*
must not decide which checkpoint it resumes from.
`test_saving_a_latent_tail_does_not_move_the_run_identity` compares the two digests directly.

Nothing here loads the real model: `Bare`/`StubVAE` (tests/test_decode_video_uint8.py) drive the
real `_decode_video` with a stand-in for the 5.21 GB video VAE.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.mlx
pytest.importorskip("mlx.core", reason="mlx: needs the MLX stack, absent here")


import inspect
import json
import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

# Importing `h3_48gb.pipeline` first puts the vendored `upstream/` checkout on `sys.path`.
import h3_48gb.pipeline as pipeline_mod
from h3_48gb.pipeline import LazyMiniMaxH3Pipeline

from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline

from h3_48gb.checkpoint import CheckpointingPipeline, identity_digest, request_identity
from h3_48gb.cli import BAKED_GRID_POINTS, CliError, ERROR_CODES, RunSpec

from test_cli import _StubResult, bake_adaln_table
from test_decode_video_uint8 import Bare, StubVAE, _Config

# The toy geometry every stand in this file shares. Deliberately tiny and deliberately *not*
# square in the latent-frame axis: 12 source latent frames, of which the last 7 are the tail, so
# a slice that keeps the wrong number of frames or counts from the wrong end cannot pass by
# accident.
LATENT_CHANNELS = 4
PATCH_SIZE = (1, 2, 2)
LATENT_HEIGHT = LATENT_WIDTH = 4
ROWS_PER_FRAME = (LATENT_HEIGHT // PATCH_SIZE[1]) * (LATENT_WIDTH // PATCH_SIZE[2])  # 4
ROW_DIM = LATENT_CHANNELS * PATCH_SIZE[0] * PATCH_SIZE[1] * PATCH_SIZE[2]            # 16
SOURCE_LATENT_FRAMES = 12
TAIL_LATENT_FRAMES = 7          # m = 1: the Motion-Context default, 22 pixel frames
CANVAS = (64, 64)               # (width, height); a multiple of 32, so `RunSpec` accepts it

#: `0, 1, 5, 9, 13, 17, 18` -- where the seven tail frames land on the *next* scene's own start
#: grid. Written out longhand rather than computed, so a change to the grid has to be typed here.
EXPECTED_PINNED = (0, 1, 5, 9, 13, 17, 18)


# -- stands ------------------------------------------------------------------------------------

class _WatchedVAE(StubVAE):
    """`StubVAE`, plus a record of whether the decode has started yet."""

    def __init__(self, raw, config):
        super().__init__(raw, config)
        self.decoded = False

    def decode(self, latents):
        self.decoded = True
        return super().decode(latents)


class _WatchedDiT:
    """`_DiT` from tests/test_decode_video_uint8.py, plus a record of the unload."""

    def __init__(self, patch_size):
        self.config = type("C", (), {"patch_size": patch_size})()
        self.unloaded = False

    def unload(self):
        self.unloaded = True


class _Stand(Bare):
    """`Bare` with components that say when they were touched."""

    def __init__(self, raw, latent_channels, patch_size):
        super().__init__(raw, latent_channels, patch_size)
        self.video_vae = _WatchedVAE(raw, _Config(latent_channels))
        self.dit = _WatchedDiT(patch_size)


def _source_rows(seed: int = 0) -> mx.array:
    """A finished video latent: `(F * rows_per_frame, row_dim)` float32, frame-major."""
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((SOURCE_LATENT_FRAMES * ROWS_PER_FRAME, ROW_DIM))
    return mx.array(values.astype(np.float32))


def _stand(tmp_path: Path, tail_frames: int = TAIL_LATENT_FRAMES, stem: str = "scene-a"):
    """A pipeline that will write a tail, and the rows to hand it."""
    rng = np.random.default_rng(7)
    raw = mx.array((rng.standard_normal((1, 3, 5, 6, 7)) * 2.5).astype(np.float32))
    pipe = _Stand(raw, LATENT_CHANNELS, PATCH_SIZE)
    pipe._latent_tail = {
        "latent_frames": tail_frames,
        "stem": tmp_path / stem,
        "canvas": CANVAS,
        "verbose": False,
    }
    return pipe


def _decode(pipe, rows):
    return pipe._decode_video(rows, SOURCE_LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH)


def _spec(tmp_path: Path, **overrides) -> RunSpec:
    base = dict(prompt="a cat", width=CANVAS[0], height=CANVAS[1], duration=1.0,
                steps=BAKED_GRID_POINTS, seed=0,
                checkpoint=bake_adaln_table(tmp_path / "ckpt"), outdir=tmp_path, tag="t")
    base.update(overrides)
    return RunSpec(**base)


# -- (a) the round trip ------------------------------------------------------------------------

def test_a_saved_tail_reloads_as_the_exact_slice_of_the_latent_it_was_cut_from(tmp_path):
    """Save, load, compare bytes -- against the slice, not against "some rows of about that size".

    The slice is the whole feature. `patchify_video_latents` is frame-major, so latent frame `f`
    owns rows ``[f*R, (f+1)*R)`` and the tail of `L` frames is ``rows[(F - L) * R:]``. Keeping
    `L - 1` frames, or counting from the head, or slicing on the wrong axis all produce a file of
    plausible shape that conditions the next scene on the wrong moment -- and nothing downstream
    can tell, because a latent has no timestamp in it.
    """
    from h3_48gb.cli import load_latent_tail

    rows = _source_rows()
    _decode(_stand(tmp_path), rows)

    path = tmp_path / f"scene-a{pipeline_mod.LATENT_TAIL_SUFFIX}"
    assert path.exists(), f"no tail was written; {sorted(p.name for p in tmp_path.iterdir())}"

    loaded, anchors = load_latent_tail(_spec(tmp_path, latent=path))

    want = np.array(rows)[(SOURCE_LATENT_FRAMES - TAIL_LATENT_FRAMES) * ROWS_PER_FRAME:]
    got = np.array(loaded)
    assert loaded.dtype == mx.float32, f"the tail came back as {loaded.dtype}, not float32"
    assert got.shape == (TAIL_LATENT_FRAMES * ROWS_PER_FRAME, ROW_DIM), (
        f"the tail is {got.shape}, not {(TAIL_LATENT_FRAMES * ROWS_PER_FRAME, ROW_DIM)} -- "
        f"{TAIL_LATENT_FRAMES} latent frames of {ROWS_PER_FRAME} rows")
    assert got.tobytes() == want.tobytes(), (
        "the reloaded tail is not the last "
        f"{TAIL_LATENT_FRAMES} latent frames of the latent it was cut from")
    assert anchors == EXPECTED_PINNED, (
        f"the anchors handed to the next scene are {anchors}, not {EXPECTED_PINNED}")


def test_the_pinned_anchors_are_the_next_scenes_own_rotary_grid(tmp_path):
    """`pinned_pixel_indices` is only correct if `packing` agrees, so ask `packing`.

    The metadata carries pixel indices; `build_packed_sequence` turns each into
    ``anchor_time = num_text + _ROPE_FRAME_RESCALE * anchor`` (patch 0004). For the tail to sit on
    the next scene's timeline exactly where its own latent frames would have sat, those anchor
    times must reproduce `_temporal_position_grid`'s first `L` entries -- which is what makes a
    `5m + 2` tail free of any resampling (spec §1.3).
    """
    from minimax_h3_mlx.packing import _ROPE_FRAME_RESCALE, _temporal_position_grid

    from h3_48gb.cli import load_latent_tail

    _decode(_stand(tmp_path), _source_rows())
    _, anchors = load_latent_tail(
        _spec(tmp_path, latent=tmp_path / f"scene-a{pipeline_mod.LATENT_TAIL_SUFFIX}"))

    num_text = 11.0
    want = _temporal_position_grid(TAIL_LATENT_FRAMES, num_text)
    got = [num_text + _ROPE_FRAME_RESCALE * anchor for anchor in anchors]
    # `atol=0` holds exactly here (m=1, integer anchors, one multiply-add each) because both sides
    # are built from the same handful of float64 Python floats with no summation involved. It is
    # not a general guarantee: `_temporal_position_grid` accumulates the frame grid with a running
    # numpy sum, and numpy's pairwise summation reassociates terms once there are enough of them --
    # measured to disagree with a naive left-to-right sum by ~2.8e-14 at m>=12 (`TAIL_LATENT_FRAMES`
    # here is 7, i.e. m=1, well under that). A future parametrization over larger m must not raise
    # this atol to paper over that; it must compare the float32 values the pipeline actually
    # produces (the mx.float32 cast that already happens on the write/read path erases a 2.8e-14
    # float64 difference), not this test's float64 intermediate.
    np.testing.assert_allclose(got, want, rtol=0, atol=0,
                               err_msg="the pinned indices do not land on packing's own grid")


def test_the_tail_metadata_describes_the_geometry_the_next_scene_must_match(tmp_path):
    """Enough to refuse a mismatch, and no guesswork on the reading side (spec §1.3)."""
    _decode(_stand(tmp_path), _source_rows())

    _, metadata = mx.load(
        str(tmp_path / f"scene-a{pipeline_mod.LATENT_TAIL_SUFFIX}"), return_metadata=True)
    meta = json.loads(metadata[pipeline_mod.LATENT_TAIL_META_KEY])

    assert meta["format"] == pipeline_mod.LATENT_TAIL_FORMAT
    assert meta["canvas"] == list(CANVAS)
    assert meta["latent_frames"] == TAIL_LATENT_FRAMES
    assert meta["m"] == 1
    assert meta["pixel_frames"] == 22, "seven latent frames are 17*1 + 5 = 22 pixel frames"
    assert meta["rows_per_frame"] == ROWS_PER_FRAME
    assert meta["row_dim"] == ROW_DIM
    assert meta["dtype"] == "float32"
    assert meta["source_latent_frames"] == SOURCE_LATENT_FRAMES
    assert meta["pinned_pixel_indices"] == list(EXPECTED_PINNED)


def test_the_write_is_atomic(tmp_path):
    """A half-written tail must never appear under the name the next scene reads.

    `CheckpointStore.write`'s sequence (`h3_48gb/checkpoint.py:368-394`): temp file, fsync,
    `os.replace`, fsync the directory, and unlink the temp on any failure. The failure this pins
    is the one that matters -- a crash mid-write leaves *nothing* at the destination rather than a
    truncated safetensors header, and leaves no debris behind either.
    """
    real_save = pipeline_mod.mx.save_safetensors
    seen: list[str] = []

    def exploding_save(file, arrays, *args, **kwargs):
        seen.append(str(file))
        real_save(file, arrays, *args, **kwargs)     # write real bytes, then die
        raise OSError("disk full, halfway through")

    pipe = _stand(tmp_path)
    pipeline_mod.mx.save_safetensors = exploding_save
    try:
        with pytest.raises(OSError):
            _decode(pipe, _source_rows())
    finally:
        pipeline_mod.mx.save_safetensors = real_save

    destination = tmp_path / f"scene-a{pipeline_mod.LATENT_TAIL_SUFFIX}"
    assert seen and seen[0] != str(destination), (
        f"the tail was written straight to its final name ({seen}); a crash mid-write would "
        f"leave a truncated file where the next scene reads a latent")
    assert not destination.exists(), "a failed write left a file at the destination"
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".")] == [], (
        "a failed write left its temporary file behind")


def test_the_write_fsyncs_the_temp_file_then_the_destination_directory(tmp_path, monkeypatch):
    """Durability rests on two fsyncs, in a specific order, of two different things.

    `write_safetensors_atomically` (`h3_48gb/pipeline.py`, following `CheckpointStore.write`'s own
    sequence, `h3_48gb/checkpoint.py:379-407`): fsync the temp file's data *before* `os.replace`
    (so the rename cannot become durable while the bytes it points at are not), then fsync the
    directory *after* (so the rename itself -- a changed directory entry -- survives a power loss
    too). A bare count of two fsyncs would pass a bug that fsyncs the temp file twice, or the
    directory twice, just as easily as the real sequence -- this checks *what* each fsync'd fd
    pointed at and *when*, recovering each fd's path from the `os.open` call that produced it
    (macOS has no `/proc/self/fd`), the same technique
    `tests/test_queue.py::test_claim_fsyncs_both_directories_the_rename_touches` uses for the
    queue's own durable write.
    """
    paths_by_fd: dict[int, str] = {}
    real_open, real_fsync, real_replace = os.open, os.fsync, os.replace
    events: list[tuple] = []

    def spy_open(path, *a, **k):
        fd = real_open(path, *a, **k)
        paths_by_fd[fd] = str(path)
        return fd

    def spy_fsync(fd):
        events.append(("fsync", paths_by_fd.get(fd)))
        return real_fsync(fd)

    def spy_replace(src, dst):
        events.append(("replace", str(src), str(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(pipeline_mod.os, "open", spy_open)
    monkeypatch.setattr(pipeline_mod.os, "fsync", spy_fsync)
    monkeypatch.setattr(pipeline_mod.os, "replace", spy_replace)

    _decode(_stand(tmp_path), _source_rows())

    fsync_indices = [i for i, e in enumerate(events) if e[0] == "fsync"]
    replace_indices = [i for i, e in enumerate(events) if e[0] == "replace"]
    assert len(fsync_indices) == 2, f"expected exactly two fsyncs, got {len(fsync_indices)}: {events}"
    assert len(replace_indices) == 1, f"expected exactly one replace, got {events}"
    before, after = fsync_indices
    replace_at = replace_indices[0]
    assert before < replace_at < after, (
        f"the two fsyncs must straddle os.replace -- one before (the file), one after (the "
        f"directory): {events}")

    before_path = events[before][1]
    after_path = events[after][1]
    assert before_path is not None and not Path(before_path).is_dir(), (
        f"the fsync before replace must be a regular file, got {before_path}")
    assert Path(before_path).name.startswith("."), (
        f"the fsync before replace must be the temp file, got {before_path}")
    assert after_path is not None and Path(after_path).is_dir(), (
        f"the fsync after replace must be a directory, got {after_path}")
    assert after_path == str(tmp_path), (
        f"the fsync after replace must be the tail's own destination directory, got {after_path}")


# -- (a1) canvas: axes, not just magnitude ------------------------------------------------------

def test_a_non_square_canvas_round_trips_and_a_transposed_one_is_refused(tmp_path):
    """Every other test in this file saves at 64x64, where `canvas` swapped to `(height, width)`
    is invisible -- transposing two equal numbers changes nothing. This saves at 128x64 instead,
    and builds the request through `_latent_tail_request` (rather than hand-building the dict, as
    `_stand` does for the square fixtures) because that is the method whose `canvas` entry a
    transposition bug would actually live in -- `__call__` calls it exactly this way. The tail
    must reload on its own 128x64 canvas and be refused on the transposed 64x128 one.
    """
    from h3_48gb.cli import load_latent_tail

    width, height = 128, 64                                    # non-square, both multiples of 32
    latent_height, latent_width = height // 16, width // 16    # 4, 8 (spatial_compression_ratio=16)
    rows_per_frame = (latent_height // PATCH_SIZE[1]) * (latent_width // PATCH_SIZE[2])

    pipe = _stand(tmp_path, stem="scene-wide")
    request = pipe._latent_tail_request(
        {"save_latent_tail": TAIL_LATENT_FRAMES, "latent_tail_stem": str(tmp_path / "scene-wide")},
        {"height": height, "width": width, "duration_seconds": 5.0, "verbose": False})
    pipe._latent_tail = request

    rng = np.random.default_rng(11)
    rows = mx.array(rng.standard_normal(
        (SOURCE_LATENT_FRAMES * rows_per_frame, ROW_DIM)).astype(np.float32))
    pipe._decode_video(rows, SOURCE_LATENT_FRAMES, latent_height, latent_width)

    path = tmp_path / f"scene-wide{pipeline_mod.LATENT_TAIL_SUFFIX}"
    _, metadata = mx.load(str(path), return_metadata=True)
    meta = json.loads(metadata[pipeline_mod.LATENT_TAIL_META_KEY])
    assert meta["canvas"] == [width, height], (
        f"the tail's own metadata canvas is {meta['canvas']}, not [{width}, {height}]")

    loaded, _ = load_latent_tail(_spec(tmp_path, latent=path, width=width, height=height))
    assert loaded.shape == (TAIL_LATENT_FRAMES * rows_per_frame, ROW_DIM)

    with pytest.raises(CliError) as excinfo:
        load_latent_tail(_spec(tmp_path, latent=path, width=height, height=width))
    assert excinfo.value.code == "latent_canvas_mismatch", (
        f"a tail saved at {width}x{height} was accepted on the transposed {height}x{width} canvas "
        f"({excinfo.value.code})")


# -- (a2) a failed clip leaves no artifact for the next scene to pick up ------------------------

def test_a_validation_failure_removes_the_tail_this_call_just_wrote(tmp_path):
    """The tail is written at the seam (spec §1.2), *before* `_validate_decoded_frames` runs --
    that check needs decoded pixels, which do not exist yet at the seam -- so a scene whose clip
    fails corruption validation still produces a tail file for one instant. Left alone, that file
    would sit on disk exactly where the next scene's `--latent` looks, and a later run could chain
    conditioning off a clip nobody ever trusted. `_WatchedVAE` is handed an all-zero raw decode:
    after the pixel-mean/std unnormalize that is a flat `FILL_COLOR` frame,
    `framecheck.zero_fill_fraction`'s own trigger for the real corruption bug this check defends
    against (`_validate_decoded_frames`'s docstring).
    """
    from h3_48gb import framecheck

    pipe = _stand(tmp_path)
    pipe.video_vae = _WatchedVAE(mx.zeros((1, 3, 5, 6, 7)), _Config(LATENT_CHANNELS))

    path = tmp_path / f"scene-a{pipeline_mod.LATENT_TAIL_SUFFIX}"
    with pytest.raises(framecheck.CorruptFramesError):
        _decode(pipe, _source_rows())

    assert not path.exists(), (
        "a scene that failed corruption validation left its latent tail on disk -- a later "
        "--latent run could chain onto a clip nobody ever trusted")


# -- (b) the ordering --------------------------------------------------------------------------

def _enclosing_decode_frame(frame):
    """The `_decode_video` frame this call is nested inside, or None."""
    code = LazyMiniMaxH3Pipeline._decode_video.__code__
    while frame is not None:
        if frame.f_code is code:
            return frame
        frame = frame.f_back
    return None


def test_the_tail_is_written_after_the_rows_are_materialized_in_the_same_decode(tmp_path,
                                                                               monkeypatch):
    """The write must not be what forces the latent, and it must be *this* decode's eval.

    MLX is lazy. If `mx.save_safetensors` is reached while `rows` is still a graph, the write is
    what executes it -- the same trap that made `written_at` eighteen minutes early
    (`h3_48gb/checkpoint.py`, `test_written_at_is_stamped_after_the_step_is_materialized`). Here
    the graph in question is the transformer's last forward, so the cost is not a wrong timestamp
    but the transformer's whole peak held open across a file write.

    "Some `mx.eval` happened earlier" is not the invariant and would pass with the fix reverted:
    upstream's loop evaluates the rows every step, so the *previous* step's eval sits right there
    in the trace. The invariant is that **this** `_decode_video` call evaluated **this** `rows`
    object before writing, so the frame object and the array identity are what is asserted.

    The two component checks are the other half of "not in the graph": the DiT is still loaded
    (the write happens at the seam, not after the unload stranded a graph over dropped weights)
    and the VAE has not begun decoding (nothing of the decode can be in the array being saved).
    """
    events: list[tuple] = []
    real_eval = pipeline_mod.mx.eval
    real_save = pipeline_mod.mx.save_safetensors

    def spy_eval(*args, **kwargs):
        events.append(("eval", sys._getframe(1), args, None))
        return real_eval(*args, **kwargs)

    def spy_save(file, arrays, *rest, **kwargs):
        pipe = spy_save.pipe
        events.append(("save", _enclosing_decode_frame(sys._getframe(1)), tuple(arrays.values()),
                       {"dit_unloaded": pipe.dit.unloaded, "vae_decoded": pipe.video_vae.decoded}))
        return real_save(file, arrays, *rest, **kwargs)

    monkeypatch.setattr(pipeline_mod.mx, "eval", spy_eval)
    monkeypatch.setattr(pipeline_mod.mx, "save_safetensors", spy_save)

    pipe = _stand(tmp_path)
    spy_save.pipe = pipe
    rows = _source_rows()
    _decode(pipe, rows)

    saves = [(i, event) for i, event in enumerate(events) if event[0] == "save"]
    assert len(saves) == 1, f"expected exactly one tail write, got {len(saves)}"
    save_index, (_, decode_frame, saved_arrays, state) = saves[0]

    assert decode_frame is not None, (
        "the tail was written from outside `_decode_video` -- the only place the rows are still "
        "raw patchified latent")
    materialized = [
        i for i, (kind, frame, args, _) in enumerate(events[:save_index])
        if kind == "eval" and frame is decode_frame and any(a is rows for a in args)
    ]
    assert materialized, (
        "the tail was written before this `_decode_video` call evaluated the very rows it is "
        "cutting from: the write itself would force the transformer's last forward. Evals seen "
        f"before it: {[(f.f_code.co_name, len(a)) for _, f, a, _ in events[:save_index]]}")
    assert state == {"dit_unloaded": False, "vae_decoded": False}, (
        f"at the moment of the write the components were {state}: the tail must be cut at the "
        "seam -- transformer still resident, VAE not yet started -- so the array being written "
        "is a slice of materialized rows and nothing else")
    assert len(saved_arrays) == 1 and saved_arrays[0].shape[0] == \
        TAIL_LATENT_FRAMES * ROWS_PER_FRAME, (
        f"the write handed over {[a.shape for a in saved_arrays]}, not one tail block")


# -- (c) the two conditioning paths are exclusive ----------------------------------------------

def test_a_latent_tail_and_a_keyframe_cannot_be_combined(tmp_path):
    """One conditioning slot, two claimants -- refused at the CLI, before any weight loads.

    Upstream refuses it too (patch 0005), but only after the text encoder has run: both would
    build a conditioning block and `mx.concatenate` would hand the layout twice the rows it
    planned for. Refusing here costs nothing and names the flag the operator typed.
    """
    tail = tmp_path / "prev-latent-tail.safetensors"
    tail.write_bytes(b"never opened -- the conflict is refused before the file is read")
    image = tmp_path / "frame.png"
    from PIL import Image
    Image.new("RGB", CANVAS, (200, 30, 30)).save(image)

    with pytest.raises(CliError) as excinfo:
        _spec(tmp_path, image=image, latent=tail)

    assert excinfo.value.code == "latent_with_image"
    assert "latent_with_image" in ERROR_CODES


def test_a_missing_latent_tail_is_refused_by_path(tmp_path):
    with pytest.raises(CliError) as excinfo:
        _spec(tmp_path, latent=tmp_path / "absent.safetensors")
    assert excinfo.value.code == "latent_not_found"


def test_a_truncated_latent_tail_file_is_refused_as_unreadable(tmp_path):
    """A crash mid-write outside `write_safetensors_atomically`'s own protection -- a disk fill
    that hit `mx.save_safetensors` itself, or a copy interrupted onto this path by hand -- must
    read as `latent_unreadable`, not surface a raw safetensors parse error to the operator.
    """
    from h3_48gb.cli import load_latent_tail

    _decode(_stand(tmp_path), _source_rows())
    path = tmp_path / f"scene-a{pipeline_mod.LATENT_TAIL_SUFFIX}"
    good = path.read_bytes()
    path.write_bytes(good[: len(good) // 2])          # truncate mid-file: a broken header/tensor

    with pytest.raises(CliError) as excinfo:
        load_latent_tail(_spec(tmp_path, latent=path))
    assert excinfo.value.code == "latent_unreadable", excinfo.value.code


def test_a_latent_tail_with_hand_edited_anchors_is_refused(tmp_path):
    """`pinned_pixel_indices` must be the *one* set of anchors `latent_tail_pixel_indices` computes
    for that many latent frames (spec §1.3) -- anything else pins the tail's rows to a moment on
    the next scene's own timeline that formula never produces, and nothing about the rows
    themselves could ever reveal that: a latent has no timestamp of its own to check against.

    Forged the way a hand-edited or foreign-build file would look: real rows and metadata that
    passes every other check, with only the anchors disturbed -- one nudged off the grid, and the
    whole tail shifted late, the two shapes of corruption a bit-flip or a manual edit produce.
    """
    from h3_48gb.cli import load_latent_tail

    _decode(_stand(tmp_path), _source_rows())
    good_path = tmp_path / f"scene-a{pipeline_mod.LATENT_TAIL_SUFFIX}"
    arrays, metadata = mx.load(str(good_path), return_metadata=True)
    meta = json.loads(metadata[pipeline_mod.LATENT_TAIL_META_KEY])
    assert meta["pinned_pixel_indices"] == list(EXPECTED_PINNED)

    forged = tmp_path / "forged-latent-tail.safetensors"
    for label, anchors in [
        ("last entry nudged off grid", [0, 1, 5, 9, 13, 17, 19]),
        ("whole tail shifted late", [index + 3 for index in EXPECTED_PINNED]),
    ]:
        bad_meta = {**meta, "pinned_pixel_indices": anchors}
        mx.save_safetensors(str(forged), {"video_tail": arrays["video_tail"]},
                            metadata={pipeline_mod.LATENT_TAIL_META_KEY: json.dumps(bad_meta)})
        with pytest.raises(CliError) as excinfo:
            load_latent_tail(_spec(tmp_path, latent=forged))
        assert excinfo.value.code == "latent_anchors_off_grid", (
            f"{label}: got {excinfo.value.code!r}, not latent_anchors_off_grid")


def test_only_a_5m_plus_2_tail_can_start_the_next_scene(tmp_path):
    """`L = 5m + 2` is the grid, and the CLI and the pipeline must agree on it exactly.

    A tail of any other length does not line up with the next scene's own latent frames, so its
    rows would need resampling that nothing in this fork does. The two implementations are
    separate on purpose -- `RunSpec.__post_init__` must not import mlx -- so they are compared
    here rather than trusted to stay in step.
    """
    for latent_frames in range(1, 25):
        try:
            _spec(tmp_path, save_latent_tail=latent_frames)
            cli_ok = True
        except CliError as exc:
            assert exc.code == "latent_tail_off_grid", exc.code
            cli_ok = False
        try:
            pipeline_mod.latent_tail_pixel_frames(latent_frames)
            pipeline_ok = True
        except ValueError:
            pipeline_ok = False
        assert cli_ok == pipeline_ok == (latent_frames % 5 == 2), (
            f"L={latent_frames}: cli accepted={cli_ok}, pipeline accepted={pipeline_ok}, "
            f"but the grid is 2, 7, 12, 17, ...")

    assert pipeline_mod.latent_tail_pixel_frames(2) == 5
    assert pipeline_mod.latent_tail_pixel_frames(7) == 22
    assert pipeline_mod.latent_tail_pixel_frames(12) == 39
    assert pipeline_mod.latent_tail_pixel_frames(17) == 56


def test_both_flags_reach_the_run_spec(tmp_path):
    """A flag that stops at the parser is not a feature."""
    from h3_48gb.cli import build_parser, spec_from_args

    tail = tmp_path / "prev-latent-tail.safetensors"
    tail.write_bytes(b"never opened here")
    table = bake_adaln_table(tmp_path / "ckpt")
    spec = spec_from_args(build_parser().parse_args([
        "generate", "a cat", "--outdir", str(tmp_path), "--checkpoint", str(table),
        "--adaln-cache", str(Path(table) / "transformer/adaln_cache.safetensors"),
        "--steps", str(BAKED_GRID_POINTS), "--width", "64", "--height", "64",
        "--latent", str(tail), "--save-latent-tail", "7",
    ]))
    assert spec.latent == tail
    assert spec.save_latent_tail == 7


def test_the_latent_tail_reaches_the_pipeline(tmp_path):
    """`--latent` must arrive as `condition_latent_rows` *and* replace the keyframe anchors."""
    from h3_48gb.cli import run_generate

    _decode(_stand(tmp_path), _source_rows())
    tail = tmp_path / f"scene-a{pipeline_mod.LATENT_TAIL_SUFFIX}"

    seen: dict = {}

    def factory(_checkpoint):
        def pipe(**kwargs):
            seen.update(kwargs)
            return _StubResult()
        return pipe

    run_generate(_spec(tmp_path, latent=tail, save_latent_tail=7),
                 pipeline_factory=factory,
                 save_mp4_fn=lambda *a, **k: None, save_wav_fn=lambda *a, **k: None,
                 verbose=False)

    rows = seen["condition_latent_rows"]
    assert rows is not None and rows.shape == (TAIL_LATENT_FRAMES * ROWS_PER_FRAME, ROW_DIM)
    assert seen["keyframe_anchors"] == EXPECTED_PINNED, (
        f"the anchors reaching the pipeline are {seen['keyframe_anchors']}, not the tail's own "
        f"{EXPECTED_PINNED} -- the rows would be pinned to the wrong moment")
    assert seen["images"] is None
    assert seen["save_latent_tail"] == 7
    assert seen["latent_tail_stem"] == str(_spec(tmp_path).output_stem())


# -- (c1) --latent is a geometry anchor too, exactly like --image -------------------------------

def test_the_canvas_follows_a_latent_tail_when_none_is_given(tmp_path):
    """`--latent` must resolve the canvas the same way `--image` does: with neither --width nor
    --height given, the run's canvas comes from the tail's own metadata, not the 896x512 default.
    Without this, `h3 generate --latent prev-tail.safetensors` would fail `latent_canvas_mismatch`
    against a canvas the operator never asked for (spec §1.3).
    """
    from h3_48gb.cli import build_parser, spec_from_args

    _decode(_stand(tmp_path), _source_rows())
    tail = tmp_path / f"scene-a{pipeline_mod.LATENT_TAIL_SUFFIX}"
    table = bake_adaln_table(tmp_path / "ckpt-canvas-from-latent")

    spec = spec_from_args(build_parser().parse_args([
        "generate", "a cat", "--outdir", str(tmp_path), "--checkpoint", str(table),
        "--adaln-cache", str(Path(table) / "transformer/adaln_cache.safetensors"),
        "--steps", str(BAKED_GRID_POINTS), "--latent", str(tail),
    ]))
    assert (spec.width, spec.height) == CANVAS, (
        f"the run's canvas is {(spec.width, spec.height)}, not the tail's own {CANVAS}")


def test_half_a_canvas_with_a_latent_tail_is_refused(tmp_path):
    """Symmetric with `--image` (`partial_canvas_with_image`): one axis given, the other derived
    from the tail, would silently produce a canvas of neither the requested nor the tail's own
    shape -- and unlike a keyframe, a raw latent tail has no resample path to fit it to one anyway.
    """
    from h3_48gb.cli import build_parser, spec_from_args

    _decode(_stand(tmp_path), _source_rows())
    tail = tmp_path / f"scene-a{pipeline_mod.LATENT_TAIL_SUFFIX}"
    table = bake_adaln_table(tmp_path / "ckpt-partial-canvas")

    with pytest.raises(CliError) as excinfo:
        spec_from_args(build_parser().parse_args([
            "generate", "a cat", "--outdir", str(tmp_path), "--checkpoint", str(table),
            "--adaln-cache", str(Path(table) / "transformer/adaln_cache.safetensors"),
            "--steps", str(BAKED_GRID_POINTS), "--latent", str(tail), "--width", "640",
        ]))
    assert excinfo.value.code == "partial_canvas_with_latent"


def test_an_explicit_canvas_still_wins_over_a_latent_tail(tmp_path):
    """Explicit --width/--height must be the answer `resolve_canvas` gives even when --latent could
    derive a different one -- substituting the tail's canvas here would hide a real mismatch rather
    than let the existing, unchanged `latent_canvas_mismatch` refusal (`load_latent_tail`) catch it.
    """
    from h3_48gb.cli import build_parser, load_latent_tail, spec_from_args

    _decode(_stand(tmp_path), _source_rows())
    tail = tmp_path / f"scene-a{pipeline_mod.LATENT_TAIL_SUFFIX}"
    table = bake_adaln_table(tmp_path / "ckpt-explicit-canvas")

    spec = spec_from_args(build_parser().parse_args([
        "generate", "a cat", "--outdir", str(tmp_path), "--checkpoint", str(table),
        "--adaln-cache", str(Path(table) / "transformer/adaln_cache.safetensors"),
        "--steps", str(BAKED_GRID_POINTS), "--latent", str(tail),
        "--width", "128", "--height", "128",
    ]))
    assert (spec.width, spec.height) == (128, 128), (
        "an explicit --width/--height must win over the tail's own canvas")

    with pytest.raises(CliError) as excinfo:
        load_latent_tail(spec)
    assert excinfo.value.code == "latent_canvas_mismatch"


# -- (d) identity ------------------------------------------------------------------------------

def _bare_lazy_pipeline():
    """Enough of `LazyMiniMaxH3Pipeline` for its own `__call__` to run, and nothing more.

    `__init__` builds a `PhaseTracker` and bounds the MLX allocator cache; neither has anything to
    do with argument binding, which is all this exercises.
    """
    pipe = object.__new__(LazyMiniMaxH3Pipeline)
    pipe._adaln_cache_path = None      # -> `supported_num_inference_steps()` is None
    pipe._supported_steps = None
    pipe._latent_tail = None
    pipe.dit = None                    # saved and restored around the (uninstalled) preview seam
    return pipe


def test_saving_a_latent_tail_does_not_move_the_run_identity(tmp_path, monkeypatch):
    """Two runs, identical but for `--save-latent-tail`, must hash to one digest.

    `request_identity` (`h3_48gb/checkpoint.py:150`) hashes `bound.arguments` wholesale. A
    fork-only keyword that survived the bind would give every tail-saving scene a *different*
    identity from the same scene without the flag -- so a run interrupted with the flag on could
    not be resumed without it, and vice versa, with `checkpoint_not_found` as the only symptom.
    `PREVIEW_KWARGS` solved this once already; this is the same solution, asserted the same way.

    The digest is taken where the checkpointing layer takes it -- off the arguments that actually
    crossed into `CheckpointingPipeline.__call__` -- rather than off a hand-built dict, because
    the pop happens in between.
    """
    digests: list[str] = []
    arguments_seen: list[dict] = []

    def fake_call(self, *args, **kwargs):
        bound = inspect.signature(MiniMaxH3Pipeline.__call__).bind(None, *args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        arguments_seen.append(arguments)
        digests.append(identity_digest(request_identity(arguments, {"weights": "toy-v1"})))
        return None

    monkeypatch.setattr(CheckpointingPipeline, "__call__", fake_call)

    request = dict(prompt="a jeweled hummingbird beside a red orchid", duration_seconds=5.0,
                   num_inference_steps=8, seed=0, height=64, width=64, verbose=False)
    _bare_lazy_pipeline()(**request)
    _bare_lazy_pipeline()(**request, save_latent_tail=7,
                          latent_tail_stem=str(tmp_path / "scene-a"))

    assert digests[0] == digests[1], (
        "`--save-latent-tail` changed the run identity: every scene that saves a tail would "
        "resume from a different checkpoint than the same scene without the flag")
    for arguments in arguments_seen:
        assert not set(arguments) & set(pipeline_mod.LATENT_TAIL_KWARGS), (
            f"a fork-only keyword survived the bind into {sorted(arguments)}")


def test_a_tail_request_is_refused_before_the_run_rather_than_after_it(tmp_path):
    """A stem that is missing, or a tail longer than the clip, must not surface hours later."""
    request = dict(prompt="x", duration_seconds=5.0, num_inference_steps=8, seed=0,
                   height=64, width=64, verbose=False)

    with pytest.raises(ValueError, match="latent_tail_stem"):
        _bare_lazy_pipeline()(**request, save_latent_tail=7)

    with pytest.raises(ValueError, match="37"):     # 5 s -> 124 frames -> 37 latent frames
        _bare_lazy_pipeline()(**request, save_latent_tail=42,
                              latent_tail_stem=str(tmp_path / "scene-a"))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
