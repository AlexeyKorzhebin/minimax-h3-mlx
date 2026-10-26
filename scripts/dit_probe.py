"""Shared harness for the two DiT memory probes next to it.

`levers_parity.py` and `levers_ladder.py` both need the same three things: a packed sequence of
the *real* shape for a given canvas and duration, a modulation table of the real footprint, and a
frozen copy of the pre-patch upstream code to compare against. They live here so the two scripts
cannot drift apart — a parity checker and a benchmark that disagree about what "stock" means are
worse than neither.

Nothing here writes to the repo or to the checkpoint. Import it as a module; it has no CLI.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from h3_48gb import _upstream  # noqa: F401,E402

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

from minimax_h3_mlx import dit as updit  # noqa: E402
from minimax_h3_mlx.config import MODALITY_NUM, TAG_TEXT  # noqa: E402
from minimax_h3_mlx.packing import (  # noqa: E402
    FPS,
    align_num_frames,
    audio_latent_num_frames,
    build_packed_sequence,
    video_latent_num_frames,
)

from h3_48gb.adaln import CachedModulation  # noqa: E402

#: The 8-bit transformer these numbers were all measured against.
DEFAULT_TRANSFORMER = Path.home() / "models/h3-8bit-full/transformer"

#: Any AdaLN table of the right shape — only its dtype and footprint matter to a memory probe.
DEFAULT_ADALN = Path.home() / "models/turbo/adaln_8_plain.safetensors"

#: The Turbo LoRA the few-step runs use. Parity with it attached is not optional: it is the one
#: thing in the model with a narrow matmul, and narrow matmuls are what row-chunking can break.
DEFAULT_LORA = Path.home() / "models/turbo/minimax_h3_turbo_v4_step600_ema.safetensors"

#: Video VAE spatial downsampling (space_down = [2,2,2,2,1,1]).
SPATIAL_RATIO = 16

#: Text rows in a typical real request, and the number every measurement in `docs/RESULTS.md` used.
TEXT_ROWS = 371


def add_common_arguments(parser) -> None:
    """Canvas, checkpoint and chunk-size flags both scripts share."""
    parser.add_argument("--transformer", type=Path, default=DEFAULT_TRANSFORMER,
                        help=f"8-bit DiT directory (default: {DEFAULT_TRANSFORMER})")
    parser.add_argument("--adaln", type=Path, default=DEFAULT_ADALN,
                        help=f"AdaLN table to size the modulation cache (default: {DEFAULT_ADALN})")
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--text-rows", type=int, default=TEXT_ROWS)
    parser.add_argument("--timesteps", type=int, default=19,
                        help="distinct noise levels present in one packed forward")
    parser.add_argument("--query-chunk", type=int, default=0,
                        help="override dit.QUERY_CHUNK (0 = leave the shipped 8192)")
    parser.add_argument("--ffn-chunk", type=int, default=0,
                        help="override dit.FFN_ROW_CHUNK (0 = leave the shipped 8192)")


def apply_chunk_overrides(args) -> None:
    """Lower the chunk widths so a small parity canvas actually enters the chunked paths.

    This is not a tuning knob, it is a correctness one. At the shipped 8192 a 512x512 parity run
    is ~6.2k rows, `S <= chunk` holds, and both chunked branches fall through to the unchunked
    ones — so a parity run at the defaults proves nothing whatsoever about chunking. The original
    reconnaissance probe measured `max|delta| = 0` for the chunked variants exactly that way.
    """
    if args.query_chunk:
        updit.QUERY_CHUNK = args.query_chunk
    if args.ffn_chunk:
        updit.FFN_ROW_CHUNK = args.ffn_chunk


def geometry(seconds: float, height: int, width: int, patch_size, text_rows: int = TEXT_ROWS,
             keyframes: int = 0) -> dict:
    """The packed layout a real request of this shape would build."""
    num_frames = align_num_frames(int(round(seconds * FPS)))
    num_latent_frames = video_latent_num_frames(num_frames)
    latent_h, latent_w = height // SPATIAL_RATIO, width // SPATIAL_RATIO
    layout = build_packed_sequence(
        np.full(text_rows, TAG_TEXT, dtype=np.int64),
        num_latent_frames, latent_h, latent_w, audio_latent_num_frames(num_frames), patch_size,
        keyframe_anchors=("first",) * keyframes,
    )
    return {"seconds": seconds, "num_frames": num_frames, "rows": layout.sequence_length,
            "layout": layout}


def synthetic_modulation(adaln_path, num_timesteps: int, hidden: int,
                         num_blocks: int) -> CachedModulation:
    """The real table's tensors, gathered as the pipeline gathers them.

    Values are the file's own first ``T * 3`` rows rather than schedule-resolved ones: a memory
    probe needs the right dtype and shape, not the right sigma, and the footprint is identical.
    """
    raw = mx.load(str(adaln_path))
    width = raw["blocks.0.modulations"].shape[-1]
    rows = mx.array(np.arange(num_timesteps * MODALITY_NUM, dtype=np.int64))
    tables = []
    for block in range(num_blocks):
        picked = raw[f"blocks.{block}.modulations"].reshape(-1, width)[rows]
        table = tuple(picked[:, i * hidden:(i + 1) * hidden].astype(mx.bfloat16) for i in range(6))
        mx.eval(table)
        tables.append(table)
    final = raw["final_modulations"].reshape(-1, 2 * hidden)[
        mx.array(np.arange(num_timesteps, dtype=np.int64))]
    shift, scale = final[:, :hidden].astype(mx.bfloat16), final[:, hidden:].astype(mx.bfloat16)
    mx.eval(shift, scale)
    del raw

    from h3_48gb import memory

    memory.release()
    return CachedModulation(tables, mx.zeros((num_timesteps,)), shift, scale)


def random_inputs(model, layout, num_timesteps: int, spread_levels: bool = True) -> tuple:
    """Inputs of the right shapes and dtypes, deterministic under a seeded `mx.random`.

    ``spread_levels`` puts three distinct noise levels in one forward — video, audio and
    conditioning rows — which is what a real request looks like. A parity run turns it off so the
    two sides cannot differ by anything but the code under test.
    """
    cfg = model.config
    video = mx.random.normal((1, int(layout.video_indices.shape[0]),
                              cfg.video_patch_dim)).astype(mx.bfloat16)
    audio = mx.random.normal((1, int(layout.audio_indices.shape[0]),
                              cfg.audio_latents_dim)).astype(mx.bfloat16)
    text = mx.random.normal((1, int(layout.text_indices.shape[0]),
                             cfg.text_dim)).astype(mx.bfloat16)
    timestep = mx.array(np.linspace(1.0, 0.0, num_timesteps, dtype=np.float32))

    indices = np.zeros(layout.sequence_length, dtype=np.int32)
    if spread_levels:
        indices[np.asarray(layout.audio_indices.tolist())] = min(1, num_timesteps - 1)
        indices[np.asarray(layout.video_indices.tolist())[:layout.num_condition_video_rows]] = \
            min(2, num_timesteps - 1)
    inputs = (video, audio, text, timestep, mx.array(indices))
    mx.eval(*inputs)
    return inputs


def forward(model, layout, inputs, cache):
    """One packed forward. Returns the two velocity tensors, already evaluated."""
    video, audio, text, timestep, timestep_indices = inputs
    video_out, audio_out = model(
        video, audio, text, timestep, timestep_indices,
        layout.token_tags, layout.position_ids,
        layout.video_indices, layout.audio_indices, layout.text_indices,
        modulation_cache=cache,
    )
    mx.eval(video_out, audio_out)
    return video_out, audio_out


# ------------------------------------------------------------------ frozen pre-patch upstream

# Verbatim copies of `minimax_h3_mlx/dit.py` as it stood before
# `patches/0002-attention-memory-levers.patch`. Duplicated on purpose: reading them out of the
# patched module would compare the change against itself, which is the failure mode this whole
# harness exists to avoid.


def stock_apply_rotary(x, cos, sin):
    rotary_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    cos = cos.astype(x.dtype)[None, None, :, :]
    sin = sin.astype(x.dtype)[None, None, :, :]
    half = rotary_dim // 2
    x1, x2 = x_rot[..., :half], x_rot[..., half:]
    rotated = mx.concatenate([-x2, x1], axis=-1)
    out = x_rot * cos + rotated * sin
    if x_pass.shape[-1] == 0:
        return out
    return mx.concatenate([out, x_pass], axis=-1)


def stock_attention(self, x, rotary=None, mask=None):
    B, S, _ = x.shape
    qkv = self.qkv_proj(x).reshape(B, S, self.heads, 3, self.head_dim)
    q, k, v = qkv[:, :, :, 0], qkv[:, :, :, 1], qkv[:, :, :, 2]
    q = self.q_norm(q).transpose(0, 2, 1, 3)
    k = self.k_norm(k).transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)
    if rotary is not None:
        q = updit.apply_rotary(q, *rotary)
        k = updit.apply_rotary(k, *rotary)
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
    out = out.transpose(0, 2, 1, 3).reshape(B, S, self.heads * self.head_dim)
    return self.out_proj(out.astype(x.dtype))


def stock_feed_forward(self, x):
    fused = self.fc1(x)
    gate, value = fused[..., : self._ffn], fused[..., self._ffn:]
    return self.fc2(nn.silu(gate) * value)


PATCHED = (updit.apply_rotary, updit.Attention.__call__, updit.FeedForward.__call__)


def install(stock: bool) -> None:
    """Swap the three patched entry points for their frozen originals, or back.

    Note what this does *not* cover: the QKV carve is a property of the loaded model, not of the
    module, so a stock run must also be loaded with `split_qkv=False`.
    """
    if stock:
        updit.apply_rotary = stock_apply_rotary
        updit.Attention.__call__ = stock_attention
        updit.FeedForward.__call__ = stock_feed_forward
    else:
        updit.apply_rotary, updit.Attention.__call__, updit.FeedForward.__call__ = PATCHED


def queue_is_idle(port: int = 8765) -> tuple[bool, str]:
    """Whether the web queue is paused with nothing running — a GPU measurement needs that.

    An unreachable queue is reported as idle, loudly: not everyone runs the web UI, and refusing
    to measure because a localhost port is closed would be its own kind of wrong. But "assume
    idle" is the dangerous default, so it says so rather than passing silently.

    The request deliberately bypasses the system proxy. With one configured — this machine has one
    — a plain `urlopen` of `localhost:8765` is intercepted and comes back `502 Bad Gateway`, which
    reads as "unreachable" and hands back "idle" while a 40 GB job is running on the GPU. Found by
    noticing this function disagreed with `curl` about a queue that was plainly up.
    """
    import json
    import urllib.error
    import urllib.request

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"http://localhost:{port}/api/state", timeout=2) as response:
            state = json.load(response)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return True, (f"queue unreachable ({type(exc).__name__}) — ASSUMING nothing else is on "
                      "the GPU, which is an assumption and not a check")
    running = state.get("queue", {}).get("running", [])
    if running:
        return False, f"the queue is running {len(running)} job(s) — the GPU is not yours"
    if not state.get("paused"):
        return True, "queue is idle but NOT paused — a job could start mid-measurement"
    return True, "queue paused, nothing running"
