"""`h3 face-refine` -- the CLI glue over Task 1-3's already-reviewed face-refine engine.

Nothing here touches a GPU: `facerefine.refine_clip` and `facetrack.detect_track` are always
mocked or injected (real `refine_clip` needs 28 GB and a checkpoint; real `detect_track` needs
YuNet weights on disk), the same "mock the GPU/weights seam, run everything else for real" shape
`test_facerefine.py`'s own `test_refine_clip_happy_path_on_cpu_with_the_gpu_half_mocked` uses.
`facepaste.crop_window`/`paste_back` run for real -- pure CPU geometry, no external weights.

ffmpeg itself is a real subprocess in a few tests (reading/writing an actual tiny clip): it is not
a GPU dependency, and exercising the real pipe end to end is worth more than mocking it away
entirely. The command-*construction* functions (`ffmpeg_read_frames_cmd`/`ffmpeg_write_cmd`) are
also asserted on directly, with no process involved, per the brief's "сборка команды ffmpeg".
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.cv
pytest.importorskip("cv2", reason="cv: needs opencv-python and scipy, absent here")
pytest.importorskip("scipy", reason="cv: needs opencv-python and scipy, absent here")


import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from h3_48gb import cli
from h3_48gb import facepaste as fp
from h3_48gb import facerefine
from h3_48gb import facetrack as ft
from h3_48gb.cli import (
    CliError,
    DEFAULT_CHECKPOINT,
    DEFAULT_FACE_CROP_SCALE,
    DEFAULT_FACE_CROP_SIZE,
    DEFAULT_FACE_PASTE_FEATHER,
    DEFAULT_FACE_TRACK_EVERY,
    DEFAULT_TURBO_LORA,
    FACETRACK_VERSION,
    VideoInfo,
    build_parser,
    face_refine_output_path,
    ffmpeg_read_frames_cmd,
    ffmpeg_write_cmd,
    main,
    probe_video,
    read_video_frames,
    resolve_face_refine_prompt,
    run_face_refine,
    write_faces_mp4,
)

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
requires_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg/ffprobe not on PATH")


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------

def _dense_track(n_frames: int, box=(4.0, 4.0, 8.0, 8.0)) -> ft.FaceTrack:
    """A track with a real detection on every frame -- `detected()` is True everywhere, so
    `paste_back` never fades and a face-refine round trip is purely about the wiring under test.
    """
    samples = [(i, box) for i in range(n_frames)]
    track = ft._build_track(samples, n_frames=n_frames, every=1)
    assert track is not None
    return track


def _make_clip(path: Path, *, width=64, height=48, fps=8, frames=8, audio=True) -> None:
    """A tiny real .mp4 via ffmpeg's `lavfi` test sources -- cheap (a fraction of a second),
    no GPU, and a real file `probe_video`/`read_video_frames`/`write_faces_mp4` can be pointed at.
    """
    duration = frames / fps
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", f"testsrc=size={width}x{height}:rate={fps}:duration={duration}",
    ]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd += [str(path)]
    subprocess.run(cmd, check=True, capture_output=True)


def _fake_refine_clip(crops, **kwargs):
    """Stands in for `facerefine.refine_clip`: returns the crops unchanged, so a face-refine round
    trip through `run_face_refine` is a no-op end to end and easy to assert on pixel-for-pixel.
    """
    return crops


# --------------------------------------------------------------------------------------------
# argparse: flags, defaults
# --------------------------------------------------------------------------------------------

def test_face_refine_is_a_registered_subcommand():
    parser = build_parser()
    args = parser.parse_args(["face-refine", "clip.mp4"])
    assert args.command == "face-refine"
    assert args.input == Path("clip.mp4")


def test_face_refine_defaults_match_facerefine_and_the_battle_recipe():
    """Every default the brief pins: sigma/window/step/crossfade off `facerefine`'s own constants,
    checkpoint/turbo-lora off the battle recipe (not the old CLI defaults), adaln-dir off
    `facerefine.DEFAULT_ADALN_DIR` (~/models/turbo), --out/--prompt unset.
    """
    parser = build_parser()
    args = parser.parse_args(["face-refine", "clip.mp4"])
    assert args.sigma == facerefine.DEFAULT_SIGMA == 0.25
    assert args.seed == 42
    assert args.window == facerefine.WINDOW_FRAMES == 56
    assert args.step == facerefine.WINDOW_STEP == 42
    assert args.crossfade == facerefine.CROSSFADE_FRAMES == 12
    assert args.every == DEFAULT_FACE_TRACK_EVERY == 5
    assert args.scale == DEFAULT_FACE_CROP_SCALE == 2.75
    assert args.feather == DEFAULT_FACE_PASTE_FEATHER == 0.10
    assert args.checkpoint == DEFAULT_CHECKPOINT
    assert args.adaln_dir == facerefine.DEFAULT_ADALN_DIR
    assert args.turbo_lora == DEFAULT_TURBO_LORA
    assert args.out is None
    assert args.prompt is None
    assert args.json is False


def test_face_refine_flags_override_every_default():
    parser = build_parser()
    args = parser.parse_args([
        "face-refine", "clip.mp4",
        "--out", "out.mp4", "--sigma", "0.15", "--seed", "7",
        "--window", "39", "--step", "22", "--crossfade", "6",
        "--every", "3", "--scale", "2.0", "--feather", "0.2",
        "--checkpoint", "/tmp/ckpt", "--adaln-dir", "/tmp/adaln",
        "--turbo-lora", "/tmp/lora.safetensors", "--prompt", "a face",
        "--json",
    ])
    assert args.out == Path("out.mp4")
    assert args.sigma == 0.15
    assert args.seed == 7
    assert args.window == 39
    assert args.step == 22
    assert args.crossfade == 6
    assert args.every == 3
    assert args.scale == 2.0
    assert args.feather == 0.2
    assert args.checkpoint == Path("/tmp/ckpt")
    assert args.adaln_dir == Path("/tmp/adaln")
    assert args.turbo_lora == Path("/tmp/lora.safetensors")
    assert args.prompt == "a face"
    assert args.json is True


def test_face_refine_input_is_required():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["face-refine"])


def test_face_refine_force_flag_defaults_to_false_and_can_be_set():
    parser = build_parser()
    assert parser.parse_args(["face-refine", "clip.mp4"]).force is False
    assert parser.parse_args(["face-refine", "clip.mp4", "--force"]).force is True


def test_restated_cli_defaults_agree_with_the_modules_they_are_restated_from():
    """`DEFAULT_FACE_TRACK_EVERY`/`DEFAULT_FACE_CROP_SCALE`/`DEFAULT_FACE_CROP_SIZE`/
    `DEFAULT_FACE_PASTE_FEATHER` exist in `cli.py` only because `facetrack`/`facepaste` are too
    heavy to import eagerly (see their docstring) -- this pins them against the real signatures so
    a future default change in either module cannot silently drift from what this CLI advertises.
    """
    import inspect

    detect_sig = inspect.signature(ft.detect_track)
    assert detect_sig.parameters["every"].default == DEFAULT_FACE_TRACK_EVERY

    crop_sig = inspect.signature(fp.crop_window)
    assert crop_sig.parameters["scale"].default == DEFAULT_FACE_CROP_SCALE
    assert crop_sig.parameters["out_size"].default == DEFAULT_FACE_CROP_SIZE

    paste_sig = inspect.signature(fp.paste_back)
    assert paste_sig.parameters["feather"].default == DEFAULT_FACE_PASTE_FEATHER


# --------------------------------------------------------------------------------------------
# output naming
# --------------------------------------------------------------------------------------------

def test_output_path_defaults_to_stem_faces_next_to_the_input():
    assert face_refine_output_path(Path("/a/b/clip.mp4"), None) == Path("/a/b/clip-faces.mp4")


def test_output_path_honours_out_when_given():
    assert face_refine_output_path(Path("/a/b/clip.mp4"), Path("/elsewhere/x.mp4")) == \
        Path("/elsewhere/x.mp4")


# --------------------------------------------------------------------------------------------
# prompt: --prompt overrides, else the source clip's own sidecar json, else None
# --------------------------------------------------------------------------------------------

def test_prompt_from_task_json_reads_the_sidecar_prompt_field(tmp_path):
    clip = tmp_path / "h3-run-896x512.mp4"
    clip.write_bytes(b"not a real mp4")
    (tmp_path / "h3-run-896x512.json").write_text(json.dumps({"prompt": "a blonde woman smiling"}))
    prompt, source = resolve_face_refine_prompt(clip, None)
    assert prompt == "a blonde woman smiling"
    assert source == "task_json"


def test_prompt_flag_overrides_the_task_json_prompt(tmp_path):
    clip = tmp_path / "h3-run-896x512.mp4"
    clip.write_bytes(b"x")
    (tmp_path / "h3-run-896x512.json").write_text(json.dumps({"prompt": "from the task"}))
    prompt, source = resolve_face_refine_prompt(clip, "from the flag")
    assert prompt == "from the flag"
    assert source == "cli"


def test_prompt_is_none_with_no_sidecar_json_at_all(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    prompt, source = resolve_face_refine_prompt(clip, None)
    assert prompt is None
    assert source == "default"


def test_prompt_is_none_when_the_sidecar_has_no_prompt_key(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    (tmp_path / "clip.json").write_text(json.dumps({"tag": "run", "frames": 40}))
    prompt, source = resolve_face_refine_prompt(clip, None)
    assert prompt is None
    assert source == "default"


def test_prompt_is_none_when_the_sidecar_is_not_valid_json(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    (tmp_path / "clip.json").write_text("{not json")
    prompt, source = resolve_face_refine_prompt(clip, None)
    assert prompt is None
    assert source == "default"


def test_prompt_is_none_when_the_sidecar_prompt_is_empty_or_not_a_string(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    (tmp_path / "clip.json").write_text(json.dumps({"prompt": "   "}))
    assert resolve_face_refine_prompt(clip, None) == (None, "default")

    (tmp_path / "clip.json").write_text(json.dumps({"prompt": 5}))
    assert resolve_face_refine_prompt(clip, None) == (None, "default")


# --------------------------------------------------------------------------------------------
# ffmpeg: command construction (no process involved)
# --------------------------------------------------------------------------------------------

def test_ffmpeg_read_frames_cmd_decodes_to_rawvideo_rgb24_on_stdout():
    cmd = ffmpeg_read_frames_cmd(Path("/a/clip.mp4"))
    assert cmd[0] == "ffmpeg"
    assert "-i" in cmd and cmd[cmd.index("-i") + 1] == "/a/clip.mp4"
    # M8: this used to be `cmd[-3:] == [...] or "pipe:1" in cmd` -- the `or` meant the assertion
    # could never fail (the right side is true whenever the left side is), so the exact-tail shape
    # was never actually checked. Strict now.
    assert cmd[-3:] == ["-pix_fmt", "rgb24", "pipe:1"]
    assert "rawvideo" in cmd
    assert "rgb24" in cmd


def test_ffmpeg_read_frames_cmd_disables_autorotate_and_maps_the_first_video_stream():
    """I1: without `-noautorotate`, a clip carrying a rotation display-matrix (any portrait phone
    video) decodes already transposed -- `ffprobe` (via `probe_video`) reports the stream's stored
    geometry, but the rawvideo bytes on the pipe would be the rotated one, same byte count, wrong
    width/height, and `read_video_frames`'s `reshape` would silently scramble the frame instead of
    raising. Verified empirically against ffmpeg 8.1.2 in review: a 64x32 source with a 90-degree
    display matrix decodes to 32x64 without this flag, 64x32 with it.
    """
    cmd = ffmpeg_read_frames_cmd(Path("/a/clip.mp4"))
    assert "-noautorotate" in cmd
    # Must land before `-i`: it is a per-input decoding option, not a global one.
    assert cmd.index("-noautorotate") < cmd.index("-i")
    # M1: explicit stream selection, the same reasoning `ffmpeg_write_cmd`'s own `-map` already
    # documents -- "the first video stream" by construction, not by ffmpeg's own heuristics.
    assert "-map" in cmd and cmd[cmd.index("-map") + 1] == "0:v:0"


def test_ffmpeg_write_cmd_copies_audio_from_the_source_when_present():
    info = VideoInfo(width=64, height=48, fps="24/1", has_audio=True)
    cmd = ffmpeg_write_cmd(Path("/out/clip-faces.mp4"), Path("/src/clip.mp4"), info)
    assert cmd[0] == "ffmpeg"
    assert "-s" in cmd and cmd[cmd.index("-s") + 1] == "64x48"
    assert "-r" in cmd and cmd[cmd.index("-r") + 1] == "24/1"
    assert "pipe:0" in cmd
    assert str(Path("/src/clip.mp4")) in cmd
    assert "-map" in cmd
    maps = [cmd[i + 1] for i, tok in enumerate(cmd) if tok == "-map"]
    assert maps == ["0:v:0", "1:a:0"]
    assert "-c:a" in cmd and cmd[cmd.index("-c:a") + 1] == "copy"
    assert cmd[-1] == "/out/clip-faces.mp4"


def test_ffmpeg_write_cmd_omits_audio_mapping_when_the_source_has_none():
    info = VideoInfo(width=64, height=48, fps="30/1", has_audio=False)
    cmd = ffmpeg_write_cmd(Path("/out/clip-faces.mp4"), Path("/src/clip.mp4"), info)
    assert "-map" not in cmd
    assert "-c:a" not in cmd
    assert str(Path("/src/clip.mp4")) not in cmd  # never opened as a second input


def test_ffmpeg_write_cmd_caps_duration_at_the_shorter_stream_when_muxing_audio():
    """M5: `-shortest` when there is a source audio track -- the piped video's own duration
    (`len(frames) / fps`) need not match the source audio's exactly (a face track that drops
    trailing frames, or a source whose audio simply runs longer), and without `-shortest` ffmpeg
    pads the shorter stream instead of ending where it does.
    """
    info = VideoInfo(width=64, height=48, fps="24/1", has_audio=True)
    cmd = ffmpeg_write_cmd(Path("/out/clip-faces.mp4"), Path("/src/clip.mp4"), info)
    assert "-shortest" in cmd

    silent_info = VideoInfo(width=64, height=48, fps="24/1", has_audio=False)
    silent_cmd = ffmpeg_write_cmd(Path("/out/clip-faces.mp4"), Path("/src/clip.mp4"), silent_info)
    assert "-shortest" not in silent_cmd  # nothing to be shorter than


# --------------------------------------------------------------------------------------------
# ffmpeg: real read/write round trip on a tiny synthetic clip
# --------------------------------------------------------------------------------------------

@requires_ffmpeg
def test_probe_and_read_a_real_tiny_clip(tmp_path):
    clip = tmp_path / "src.mp4"
    _make_clip(clip, width=64, height=48, fps=8, frames=8, audio=True)

    info = probe_video(clip)
    assert (info.width, info.height) == (64, 48)
    assert info.has_audio is True
    assert info.fps in ("8/1", "8")  # ffprobe's own r_frame_rate spelling

    frames = read_video_frames(clip, info)
    assert frames.shape == (8, 48, 64, 3)
    assert frames.dtype == np.uint8


@requires_ffmpeg
def test_probe_reports_no_audio_when_the_source_has_none(tmp_path):
    clip = tmp_path / "silent.mp4"
    _make_clip(clip, width=32, height=32, fps=6, frames=6, audio=False)
    info = probe_video(clip)
    assert info.has_audio is False


def test_probe_video_falls_back_to_25fps_when_ffprobe_reports_0_over_0(tmp_path, monkeypatch):
    """M2: `probe_video`'s original fallback (`video.get("r_frame_rate") or "25/1"`) only caught an
    absent/empty key -- ffprobe answers the literal string `"0/0"` (truthy) for a stream whose rate
    it could not determine, which would reach `ffmpeg_write_cmd`'s `-r` unfiltered and ask ffmpeg
    to mux at zero frames per second.
    """
    import json as json_mod

    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")

    class _FakeProc:
        returncode = 0
        stdout = json_mod.dumps({
            "streams": [{"codec_type": "video", "width": 32, "height": 32, "r_frame_rate": "0/0"}],
        }).encode()
        stderr = b""

    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: _FakeProc())
    info = probe_video(clip)
    assert info.fps == "25/1"


@requires_ffmpeg
def test_write_faces_mp4_round_trips_frame_count_and_keeps_the_source_audio(tmp_path):
    clip = tmp_path / "src.mp4"
    _make_clip(clip, width=64, height=48, fps=8, frames=8, audio=True)
    info = probe_video(clip)
    frames = read_video_frames(clip, info)

    out = tmp_path / "out.mp4"
    write_faces_mp4(out, frames, clip, info)
    assert out.is_file()

    out_info = probe_video(out)
    assert (out_info.width, out_info.height) == (64, 48)
    assert out_info.has_audio is True
    out_frames = read_video_frames(out, out_info)
    assert out_frames.shape[0] == frames.shape[0]


@requires_ffmpeg
def test_write_faces_mp4_raises_and_leaves_no_output_file_when_ffmpeg_fails(tmp_path):
    """M9c: a real ffmpeg failure at write time (here: an output directory that does not exist,
    which ffmpeg cannot create) must raise and must not leave a half-written or otherwise present
    file at the destination -- a caller checking `out_path.exists()` after catching the error must
    see the honest "nothing was written" it would see for any other refusal.
    """
    clip = tmp_path / "src.mp4"
    _make_clip(clip, width=32, height=32, fps=4, frames=4, audio=False)
    info = probe_video(clip)
    frames = read_video_frames(clip, info)

    bad_out = tmp_path / "does" / "not" / "exist" / "out.mp4"
    with pytest.raises(RuntimeError):
        write_faces_mp4(bad_out, frames, clip, info)
    assert not bad_out.exists()


# --------------------------------------------------------------------------------------------
# run_face_refine: input missing
# --------------------------------------------------------------------------------------------

def test_run_face_refine_refuses_a_missing_input(tmp_path):
    with pytest.raises(CliError) as excinfo:
        run_face_refine(tmp_path / "absent.mp4")
    assert excinfo.value.code == "face_refine_input_not_found"
    assert "absent.mp4" in excinfo.value.message


# --------------------------------------------------------------------------------------------
# I5: --sigma/--every validated before any I/O, against facerefine's own SIGMA_CEILING
# --------------------------------------------------------------------------------------------

def test_run_face_refine_refuses_sigma_above_the_ceiling_before_touching_the_input(tmp_path):
    """Checked before `input_path.is_file()`: an absent input still reports `sigma_out_of_range`,
    not `face_refine_input_not_found` -- proof the check runs before any I/O, per the brief.
    """
    absent = tmp_path / "absent.mp4"
    with pytest.raises(CliError) as excinfo:
        run_face_refine(absent, sigma=facerefine.SIGMA_CEILING + 0.01)
    assert excinfo.value.code == "sigma_out_of_range"
    assert not absent.exists()


def test_run_face_refine_refuses_zero_and_negative_sigma(tmp_path):
    absent = tmp_path / "absent.mp4"
    for bad in (0.0, -0.1):
        with pytest.raises(CliError) as excinfo:
            run_face_refine(absent, sigma=bad)
        assert excinfo.value.code == "sigma_out_of_range"


def test_run_face_refine_sigma_ceiling_itself_is_accepted_not_refused():
    """The bound is read off `facerefine.SIGMA_CEILING`, closed at the top (`<=`, not `<`): passing
    exactly the ceiling must clear the sigma check and fail on the next thing instead (the absent
    input), not on `sigma_out_of_range` -- proof the two bounds (this one and `refine_clip`'s own)
    cannot silently drift apart into an off-by-epsilon disagreement.
    """
    with pytest.raises(CliError) as excinfo:
        run_face_refine(Path("/does/not/exist-clip.mp4"), sigma=facerefine.SIGMA_CEILING)
    assert excinfo.value.code == "face_refine_input_not_found"


def test_run_face_refine_refuses_every_less_than_one_before_touching_the_input(tmp_path):
    absent = tmp_path / "absent.mp4"
    with pytest.raises(CliError) as excinfo:
        run_face_refine(absent, every=0)
    assert excinfo.value.code == "invalid_every"
    assert not absent.exists()

    with pytest.raises(CliError) as excinfo:
        run_face_refine(absent, every=-3)
    assert excinfo.value.code == "invalid_every"


# --------------------------------------------------------------------------------------------
# I3: --checkpoint/--turbo-lora/--adaln-dir validated before decode
# --------------------------------------------------------------------------------------------

def test_run_face_refine_refuses_a_missing_checkpoint_before_decode(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")

    def fake_probe(path):
        raise AssertionError("probe_video must not run before --checkpoint is validated")

    with pytest.raises(CliError) as excinfo:
        run_face_refine(clip, checkpoint=tmp_path / "no-such-checkpoint", probe_fn=fake_probe)
    assert excinfo.value.code == "checkpoint_not_found"


def test_run_face_refine_refuses_a_missing_turbo_lora_before_decode(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")

    def fake_probe(path):
        raise AssertionError("probe_video must not run before --turbo-lora is validated")

    with pytest.raises(CliError) as excinfo:
        run_face_refine(clip, turbo_lora=tmp_path / "no-such-lora.safetensors", probe_fn=fake_probe)
    assert excinfo.value.code == "lora_not_found"


def test_run_face_refine_refuses_a_missing_adaln_dir_before_decode(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")

    def fake_probe(path):
        raise AssertionError("probe_video must not run before --adaln-dir is validated")

    with pytest.raises(CliError) as excinfo:
        run_face_refine(clip, adaln_dir=tmp_path / "no-such-adaln-dir", probe_fn=fake_probe)
    assert excinfo.value.code == "adaln_dir_not_found"


def test_run_face_refine_accepts_none_turbo_lora_without_checking_a_path(tmp_path):
    """`turbo_lora=None` means "no LoRA"; it must not be treated as a missing-file refusal."""
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    n_frames = 4
    frames = np.zeros((n_frames, 32, 32, 3), dtype=np.uint8)
    fake_info = VideoInfo(width=32, height=32, fps="4/1", has_audio=False)
    track = _dense_track(n_frames, box=(4.0, 4.0, 10.0, 10.0))

    report = run_face_refine(
        clip, turbo_lora=None,
        probe_fn=lambda path: fake_info,
        read_frames_fn=lambda path, info: frames,
        write_output_fn=lambda *a, **k: None,
        detect_track_fn=lambda f, every: track,
        refine_clip_fn=_fake_refine_clip,
        verbose=False,
    )
    assert report["turbo_lora"] is None


# --------------------------------------------------------------------------------------------
# I2: --out safety -- refuse out==input, refuse an existing --out without --force
# --------------------------------------------------------------------------------------------

def test_run_face_refine_refuses_when_out_resolves_to_the_input(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"original bytes, must survive the refusal untouched")
    with pytest.raises(CliError) as excinfo:
        run_face_refine(clip, out=clip)
    assert excinfo.value.code == "face_refine_output_is_input"
    assert clip.read_bytes() == b"original bytes, must survive the refusal untouched"


def test_run_face_refine_refuses_when_out_resolves_to_the_input_via_a_relative_spelling(
    tmp_path, monkeypatch,
):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(CliError) as excinfo:
        run_face_refine(clip, out=Path("clip.mp4"))
    assert excinfo.value.code == "face_refine_output_is_input"


def test_run_face_refine_refuses_an_existing_out_without_force(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    out_path = face_refine_output_path(clip, None)
    out_path.write_bytes(b"a previous face-refine result, must not be silently replaced")

    with pytest.raises(CliError) as excinfo:
        run_face_refine(clip)
    assert excinfo.value.code == "output_exists"
    assert out_path.read_bytes() == b"a previous face-refine result, must not be silently replaced"


def test_run_face_refine_force_overwrites_an_existing_out(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    out_path = face_refine_output_path(clip, None)
    out_path.write_bytes(b"stale")

    n_frames = 4
    frames = np.zeros((n_frames, 32, 32, 3), dtype=np.uint8)
    fake_info = VideoInfo(width=32, height=32, fps="4/1", has_audio=False)
    track = _dense_track(n_frames, box=(4.0, 4.0, 10.0, 10.0))
    written = {}

    report = run_face_refine(
        clip, force=True,
        probe_fn=lambda path: fake_info,
        read_frames_fn=lambda path, info: frames,
        write_output_fn=lambda p, *a: written.setdefault("out_path", Path(p)),
        detect_track_fn=lambda f, every: track,
        refine_clip_fn=_fake_refine_clip,
        verbose=False,
    )
    assert report["ok"] is True
    assert written["out_path"] == out_path


# --------------------------------------------------------------------------------------------
# M6: --out under a directory that does not exist yet gets one, rather than failing the run
# --------------------------------------------------------------------------------------------

def test_run_face_refine_creates_the_parent_directory_of_a_custom_out(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    custom_out = tmp_path / "brand" / "new" / "dir" / "result.mp4"
    assert not custom_out.parent.exists()

    n_frames = 4
    frames = np.zeros((n_frames, 32, 32, 3), dtype=np.uint8)
    fake_info = VideoInfo(width=32, height=32, fps="4/1", has_audio=False)
    track = _dense_track(n_frames, box=(4.0, 4.0, 10.0, 10.0))

    run_face_refine(
        clip, out=custom_out,
        probe_fn=lambda path: fake_info,
        read_frames_fn=lambda path, info: frames,
        write_output_fn=lambda *a, **k: None,
        detect_track_fn=lambda f, every: track,
        refine_clip_fn=_fake_refine_clip,
        verbose=False,
    )
    assert custom_out.parent.is_dir()


# --------------------------------------------------------------------------------------------
# run_face_refine: "no face" is an honest refusal, no output file
# --------------------------------------------------------------------------------------------

def test_run_face_refine_refuses_honestly_when_no_face_is_found(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"not a real mp4, never opened by ffmpeg because probe/read are stubbed")
    out_path = face_refine_output_path(clip, None)

    fake_info = VideoInfo(width=64, height=48, fps="8/1", has_audio=False)
    fake_frames = np.zeros((8, 48, 64, 3), dtype=np.uint8)

    with pytest.raises(CliError) as excinfo:
        run_face_refine(
            clip,
            probe_fn=lambda path: fake_info,
            read_frames_fn=lambda path, info: fake_frames,
            detect_track_fn=lambda frames, every: None,
            refine_clip_fn=_fake_refine_clip,
            verbose=False,
        )
    assert excinfo.value.code == "face_not_found"
    assert "не найдено" in excinfo.value.message
    assert not out_path.exists()


def test_main_face_refine_no_face_exits_nonzero_and_writes_nothing(tmp_path, monkeypatch, capsys):
    """The CLI-boundary version of the refusal above: `main()` returns non-zero, the honest
    sentence reaches stderr under human mode, and (per the brief) the output file never appears.
    """
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    out_path = face_refine_output_path(clip, None)

    fake_info = VideoInfo(width=32, height=32, fps="6/1", has_audio=False)
    fake_frames = np.zeros((6, 32, 32, 3), dtype=np.uint8)
    monkeypatch.setattr(cli, "probe_video", lambda path: fake_info)
    monkeypatch.setattr(cli, "read_video_frames", lambda path, info: fake_frames)
    monkeypatch.setattr(ft, "detect_track", lambda frames, every=5: None)

    code = main(["face-refine", str(clip)])
    assert code == 1
    assert not out_path.exists()
    captured = capsys.readouterr()
    assert "не найдено" in captured.err


def test_main_face_refine_no_face_json_reports_the_stable_code(tmp_path, monkeypatch, capsys):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")

    fake_info = VideoInfo(width=32, height=32, fps="6/1", has_audio=False)
    fake_frames = np.zeros((6, 32, 32, 3), dtype=np.uint8)
    monkeypatch.setattr(cli, "probe_video", lambda path: fake_info)
    monkeypatch.setattr(cli, "read_video_frames", lambda path, info: fake_frames)
    monkeypatch.setattr(ft, "detect_track", lambda frames, every=5: None)

    code = main(["face-refine", str(clip), "--json"])
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    # M11: was `payload == {..., "message": payload["error"]["message"], ...}` -- comparing a
    # field against itself is a tautology that can never fail. Explicit expected values instead.
    assert payload["ok"] is False
    assert payload["error"]["code"] == "face_not_found"
    assert "не найдено" in payload["error"]["message"]
    assert payload["error"]["detail"] == {"input": str(clip), "every": 5, "frames": 6}


def test_main_face_refine_happy_path_prints_the_done_line_and_returns_zero(tmp_path, monkeypatch, capsys):
    """The full `main(["face-refine", ...])` path, with every heavy/external seam
    (`probe_video`/`read_video_frames`/`write_faces_mp4`, `facetrack.detect_track`,
    `facerefine.refine_clip`) monkeypatched at the module level -- argparse's own wiring
    (`args.sigma`, `args.checkpoint`, ...) reaching `run_face_refine` is what this test is for,
    not the pipeline itself (covered directly against `run_face_refine` above).
    """
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    n_frames = 6
    fake_info = VideoInfo(width=32, height=32, fps="6/1", has_audio=False)
    fake_frames = np.zeros((n_frames, 32, 32, 3), dtype=np.uint8)
    track = _dense_track(n_frames, box=(4.0, 4.0, 10.0, 10.0))

    monkeypatch.setattr(cli, "probe_video", lambda path: fake_info)
    monkeypatch.setattr(cli, "read_video_frames", lambda path, info: fake_frames)
    written = {}
    monkeypatch.setattr(cli, "write_faces_mp4",
                        lambda out_path, frames, source_path, info: written.setdefault("out_path", out_path))
    monkeypatch.setattr(ft, "detect_track", lambda frames, every=5: track)
    monkeypatch.setattr(facerefine, "refine_clip", lambda crops, **kwargs: crops)

    code = main(["face-refine", str(clip)])
    assert code == 0
    out = capsys.readouterr().out
    assert "done in" in out
    assert str(face_refine_output_path(clip, None)) in out
    assert written["out_path"] == face_refine_output_path(clip, None)


# --------------------------------------------------------------------------------------------
# run_face_refine: happy path, fully injected (no GPU, no ffmpeg)
# --------------------------------------------------------------------------------------------

def test_run_face_refine_happy_path_wires_detect_crop_refine_paste_and_writes_the_output(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")

    # A frame much larger than the face box's scaled-up crop window, so the pasted rect provably
    # does not cover the whole frame -- otherwise "changed but not the whole frame" below could
    # not tell a correct local paste apart from a bug that overwrote everything.
    width, height, n_frames = 320, 240, 12
    frames = np.random.default_rng(0).integers(0, 255, (n_frames, height, width, 3), dtype=np.uint8)
    fake_info = VideoInfo(width=width, height=height, fps="8/1", has_audio=True)
    track = _dense_track(n_frames, box=(30.0, 30.0, 20.0, 20.0))

    written = {}

    def fake_write(out_path, out_frames, source_path, info):
        written["out_path"] = Path(out_path)
        written["frames"] = np.asarray(out_frames).copy()
        written["source_path"] = Path(source_path)
        written["info"] = info

    refine_calls = []

    def fake_refine(crops, **kwargs):
        refine_calls.append(kwargs)
        # A distinctive, uniform "refined" output -- easy to tell apart from the random source, so
        # the assertions below can check the paste actually landed without reimplementing
        # facepaste's own crop-rect geometry (that geometry is `test_facepaste.py`'s job).
        return np.full_like(crops, 30)

    report = run_face_refine(
        clip,
        probe_fn=lambda path: fake_info,
        read_frames_fn=lambda path, info: frames,
        write_output_fn=fake_write,
        detect_track_fn=lambda f, every: track,
        refine_clip_fn=fake_refine,
        prompt="a test prompt",
        verbose=False,
    )

    assert report["ok"] is True
    assert report["input"] == str(clip)
    assert report["output"] == str(face_refine_output_path(clip, None))
    assert report["frames"] == n_frames
    assert report["prompt"] == "a test prompt"
    assert report["prompt_source"] == "cli"
    assert report["audio"] is True

    assert len(refine_calls) == 1
    assert refine_calls[0]["sigma"] == facerefine.DEFAULT_SIGMA
    assert refine_calls[0]["prompt"] == "a test prompt"
    assert refine_calls[0]["checkpoint"] == DEFAULT_CHECKPOINT
    assert refine_calls[0]["adaln_dir"] == facerefine.DEFAULT_ADALN_DIR

    assert written["out_path"] == face_refine_output_path(clip, None)
    assert written["source_path"] == clip
    assert written["frames"].shape == frames.shape
    assert written["frames"].dtype == np.uint8
    # `paste_back`'s own contract (tested for real in `test_facepaste.py`): the pasted rect changed
    # (the refine result reached the output at all) but the whole frame did not (the crop/paste
    # geometry kept the effect local, rather than -- say -- `write_output_fn` being handed the raw
    # refined crop instead of a properly pasted-back full frame).
    changed = np.any(written["frames"] != frames, axis=-1)
    assert changed.any()
    assert not changed.all()


def test_run_face_refine_respects_out_override(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    custom_out = tmp_path / "elsewhere" / "result.mp4"
    custom_out.parent.mkdir()

    n_frames = 8
    frames = np.zeros((n_frames, 32, 32, 3), dtype=np.uint8)
    fake_info = VideoInfo(width=32, height=32, fps="4/1", has_audio=False)
    track = _dense_track(n_frames, box=(4.0, 4.0, 10.0, 10.0))
    written = {}

    report = run_face_refine(
        clip, out=custom_out,
        probe_fn=lambda path: fake_info,
        read_frames_fn=lambda path, info: frames,
        write_output_fn=lambda out_path, *a: written.__setitem__("out_path", Path(out_path)),
        detect_track_fn=lambda f, every: track,
        refine_clip_fn=_fake_refine_clip,
        verbose=False,
    )
    assert report["output"] == str(custom_out)
    assert written["out_path"] == custom_out


def test_run_face_refine_passes_window_step_crossfade_sigma_seed_through(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    n_frames = 22
    frames = np.zeros((n_frames, 32, 32, 3), dtype=np.uint8)
    fake_info = VideoInfo(width=32, height=32, fps="4/1", has_audio=False)
    track = _dense_track(n_frames, box=(4.0, 4.0, 10.0, 10.0))
    refine_calls = []

    def fake_refine(crops, **kwargs):
        refine_calls.append(kwargs)
        return crops

    run_face_refine(
        clip, sigma=0.15, seed=99, window=22, step=10, crossfade=4,
        probe_fn=lambda path: fake_info,
        read_frames_fn=lambda path, info: frames,
        write_output_fn=lambda *a, **k: None,
        detect_track_fn=lambda f, every: track,
        refine_clip_fn=fake_refine,
        verbose=False,
    )
    assert refine_calls[0]["sigma"] == 0.15
    assert refine_calls[0]["seed"] == 99
    assert refine_calls[0]["window"] == 22
    assert refine_calls[0]["step"] == 10
    assert refine_calls[0]["crossfade"] == 4


# --------------------------------------------------------------------------------------------
# M4: the report (and the "done" line) distinguish the whole pass from the refine call alone
# --------------------------------------------------------------------------------------------

def test_run_face_refine_report_distinguishes_total_pass_time_from_refine_time(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    n_frames = 4
    frames = np.zeros((n_frames, 32, 32, 3), dtype=np.uint8)
    fake_info = VideoInfo(width=32, height=32, fps="4/1", has_audio=False)
    track = _dense_track(n_frames, box=(4.0, 4.0, 10.0, 10.0))

    def slow_probe(path):
        import time as time_mod
        # Long enough that `round(..., 1)` cannot round it away to the same 0.0 as the instant
        # refine stub -- `total_seconds` and `refine_seconds` must differ at the report's own
        # precision, not merely in theory.
        time_mod.sleep(0.15)
        return fake_info

    report = run_face_refine(
        clip,
        probe_fn=slow_probe,
        read_frames_fn=lambda path, info: frames,
        write_output_fn=lambda *a, **k: None,
        detect_track_fn=lambda f, every: track,
        refine_clip_fn=_fake_refine_clip,  # instant -- no sleep
        verbose=False,
    )
    assert "total_seconds" in report and "refine_seconds" in report
    # The stub refine is instant; `slow_probe` is not -- `total_seconds` (the whole pass) must
    # exceed `refine_seconds` (just the `refine_clip` call), proof it is not just an alias for it.
    assert report["total_seconds"] > report["refine_seconds"]


def test_main_face_refine_done_line_reports_total_time_alongside_refine_time(tmp_path, monkeypatch, capsys):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    n_frames = 4
    fake_info = VideoInfo(width=32, height=32, fps="4/1", has_audio=False)
    fake_frames = np.zeros((n_frames, 32, 32, 3), dtype=np.uint8)
    track = _dense_track(n_frames, box=(4.0, 4.0, 10.0, 10.0))

    monkeypatch.setattr(cli, "probe_video", lambda path: fake_info)
    monkeypatch.setattr(cli, "read_video_frames", lambda path, info: fake_frames)
    monkeypatch.setattr(cli, "write_faces_mp4", lambda *a, **k: None)
    monkeypatch.setattr(ft, "detect_track", lambda frames, every=5: track)
    monkeypatch.setattr(facerefine, "refine_clip", lambda crops, **kwargs: crops)

    code = main(["face-refine", str(clip)])
    assert code == 0
    out = capsys.readouterr().out
    assert "done in" in out
    assert "refine" in out


# --------------------------------------------------------------------------------------------
# M9b: at least one progress-stage line reaches stdout under verbose=True
# --------------------------------------------------------------------------------------------

def test_run_face_refine_prints_progress_stage_lines_when_verbose(tmp_path, capsys):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    n_frames = 4
    frames = np.zeros((n_frames, 32, 32, 3), dtype=np.uint8)
    fake_info = VideoInfo(width=32, height=32, fps="4/1", has_audio=False)
    track = _dense_track(n_frames, box=(4.0, 4.0, 10.0, 10.0))

    run_face_refine(
        clip,
        probe_fn=lambda path: fake_info,
        read_frames_fn=lambda path, info: frames,
        write_output_fn=lambda *a, **k: None,
        detect_track_fn=lambda f, every: track,
        refine_clip_fn=_fake_refine_clip,
        verbose=True,
    )
    out = capsys.readouterr().out
    assert "face-refine: reading" in out
    assert "face-refine: detecting the face track" in out
    assert "face-refine: writing" in out


# --------------------------------------------------------------------------------------------
# I4: checkpoint_identity_extra's provenance is written to a sidecar next to the output
# --------------------------------------------------------------------------------------------

def test_run_face_refine_writes_a_provenance_sidecar_next_to_the_output(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip bytes for the sidecar's own source digest")
    n_frames = 4
    frames = np.zeros((n_frames, 32, 32, 3), dtype=np.uint8)
    fake_info = VideoInfo(width=32, height=32, fps="4/1", has_audio=False)
    track = _dense_track(n_frames, box=(4.0, 4.0, 10.0, 10.0))

    report = run_face_refine(
        clip, sigma=0.1234,
        probe_fn=lambda path: fake_info,
        read_frames_fn=lambda path, info: frames,
        write_output_fn=lambda *a, **k: None,
        detect_track_fn=lambda f, every: track,
        refine_clip_fn=_fake_refine_clip,
        verbose=False,
    )
    out_path = Path(report["output"])
    sidecar = out_path.with_suffix(".json")
    assert sidecar.is_file()
    on_disk = json.loads(sidecar.read_text())
    assert on_disk == report
    extra = on_disk["checkpoint_identity_extra"]
    assert extra["source_digest"].startswith("sha256:")
    assert extra["facetrack_version"] == FACETRACK_VERSION
    # M3-quantized: 0.1234 rounds to 0.12, matching the two-decimal precision `refine_clip`'s own
    # `_validate_request` keys its partial-table cache on -- the report should say what actually
    # ran, not the raw flag value that would mislead a later comparison into seeing a difference
    # where the table used was identical.
    assert on_disk["sigma"] == 0.12
    assert extra["sigma"] == 0.12


def test_run_face_refine_sidecar_survives_a_json_round_trip_for_a_later_prompt_chain(tmp_path):
    """The whole point of I4: a face-refine pass *over* a face-refine output must find its own
    prompt via `resolve_face_refine_prompt` -- which only works if the sidecar this test writes is
    readable back as `<stem>.json` next to `<stem>.mp4`, with a `"prompt"` field.
    """
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    n_frames = 4
    frames = np.zeros((n_frames, 32, 32, 3), dtype=np.uint8)
    fake_info = VideoInfo(width=32, height=32, fps="4/1", has_audio=False)
    track = _dense_track(n_frames, box=(4.0, 4.0, 10.0, 10.0))

    report = run_face_refine(
        clip, prompt="a specific face, lit from the left",
        probe_fn=lambda path: fake_info,
        read_frames_fn=lambda path, info: frames,
        write_output_fn=lambda *a, **k: None,
        detect_track_fn=lambda f, every: track,
        refine_clip_fn=_fake_refine_clip,
        verbose=False,
    )
    out_path = Path(report["output"])
    chained_prompt, chained_source = resolve_face_refine_prompt(out_path, None)
    assert chained_prompt == "a specific face, lit from the left"
    assert chained_source == "task_json"


def test_run_face_refine_falls_back_to_task_json_prompt_when_no_flag_given(tmp_path):
    clip = tmp_path / "h3-run-896x512.mp4"
    clip.write_bytes(b"x")
    (tmp_path / "h3-run-896x512.json").write_text(json.dumps({"prompt": "the source clip's shot"}))

    n_frames = 8
    frames = np.zeros((n_frames, 32, 32, 3), dtype=np.uint8)
    fake_info = VideoInfo(width=32, height=32, fps="4/1", has_audio=False)
    track = _dense_track(n_frames, box=(4.0, 4.0, 10.0, 10.0))
    refine_calls = []

    def fake_refine(crops, **kwargs):
        refine_calls.append(kwargs)
        return crops

    report = run_face_refine(
        clip,
        probe_fn=lambda path: fake_info,
        read_frames_fn=lambda path, info: frames,
        write_output_fn=lambda *a, **k: None,
        detect_track_fn=lambda f, every: track,
        refine_clip_fn=fake_refine,
        verbose=False,
    )
    assert report["prompt"] == "the source clip's shot"
    assert report["prompt_source"] == "task_json"
    assert refine_calls[0]["prompt"] == "the source clip's shot"


def test_run_face_refine_prompt_is_none_with_no_flag_and_no_task_json(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    n_frames = 8
    frames = np.zeros((n_frames, 32, 32, 3), dtype=np.uint8)
    fake_info = VideoInfo(width=32, height=32, fps="4/1", has_audio=False)
    track = _dense_track(n_frames, box=(4.0, 4.0, 10.0, 10.0))
    refine_calls = []

    def fake_refine(crops, **kwargs):
        refine_calls.append(kwargs)
        return crops

    report = run_face_refine(
        clip,
        probe_fn=lambda path: fake_info,
        read_frames_fn=lambda path, info: frames,
        write_output_fn=lambda *a, **k: None,
        detect_track_fn=lambda f, every: track,
        refine_clip_fn=fake_refine,
        verbose=False,
    )
    assert report["prompt"] is None
    assert report["prompt_source"] == "default"
    assert refine_calls[0]["prompt"] is None


# --------------------------------------------------------------------------------------------
# checkpoint_identity_extra
# --------------------------------------------------------------------------------------------

def test_report_carries_checkpoint_identity_extra_with_source_digest_and_facetrack_version(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"some real bytes so the digest is not of an empty file")
    n_frames = 8
    frames = np.zeros((n_frames, 32, 32, 3), dtype=np.uint8)
    fake_info = VideoInfo(width=32, height=32, fps="4/1", has_audio=False)
    track = _dense_track(n_frames, box=(4.0, 4.0, 10.0, 10.0))

    report = run_face_refine(
        clip, sigma=0.2, window=22, step=10, crossfade=4,
        probe_fn=lambda path: fake_info,
        read_frames_fn=lambda path, info: frames,
        write_output_fn=lambda *a, **k: None,
        detect_track_fn=lambda f, every: track,
        refine_clip_fn=_fake_refine_clip,
        verbose=False,
    )
    extra = report["checkpoint_identity_extra"]
    assert extra["sigma"] == 0.2
    assert extra["window"] == 22
    assert extra["step"] == 10
    assert extra["crossfade"] == 4
    assert extra["facetrack_version"] == FACETRACK_VERSION
    assert extra["source_digest"].startswith("sha256:")
    assert extra["source_digest"].endswith(f":{clip.stat().st_size}")


def test_source_digest_changes_when_the_file_content_changes(tmp_path):
    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    a.write_bytes(b"one clip's worth of bytes")
    b.write_bytes(b"a different clip's worth of bytes")
    assert cli._source_digest(a) != cli._source_digest(b)


def test_source_digest_is_stable_for_the_same_content():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "clip.mp4"
        path.write_bytes(b"deterministic content")
        assert cli._source_digest(path) == cli._source_digest(path)


# --------------------------------------------------------------------------------------------
# M10: `cli.FACETRACK_VERSION` is a hand-bumped copy of `facetrack.TRACK_ALGO_VERSION`
# --------------------------------------------------------------------------------------------

def test_facetrack_version_marker_is_pinned_to_the_module_it_describes():
    """Task 4 may not otherwise touch `facetrack.py`, so `cli.FACETRACK_VERSION` cannot read the
    real version off the module the way `--sigma`/`--window`/etc. read off `facerefine`'s own
    constants -- it is a separate, hand-maintained copy instead. This pins the two together so a
    bump on one side that forgets the other (a real change to `facetrack`'s detection/track
    semantics, with the CLI-side marker left stale) is a red test, not a silently wrong
    `checkpoint_identity_extra.facetrack_version` in every report from then on.
    """
    assert cli.FACETRACK_VERSION == ft.TRACK_ALGO_VERSION
