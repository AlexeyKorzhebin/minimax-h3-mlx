"""Phase-scoped component residency for a 48 GB machine.

`MiniMaxH3Pipeline.from_pretrained` loads the encoder, the DiT and both VAEs up front and holds all
four for the whole run — 45.9 GB of weights, and ~55 GB once the diffusion phase's own activations
are added, against 48 GB of physical memory. But the components are used in
strict phases: the text encoder is read exactly once (`pipeline.py` line 238), the video VAE
encodes keyframes and then is idle until the final decode, and the DiT only matters in between.

Most of this reimplements nothing. Two seams are enough for residency:

* Every component is replaced by a **proxy** that already knows its config — which is what the
  early lines of `__call__` actually need (``dit.config.patch_size``,
  ``video_vae.config.spatial_compression_ratio``, ``audio_vae.config.latent_channels``) — and loads
  its weights on the first access to anything else.
* The text-encoder proxy owns the unload. ``self.text_encoder.encode(...)`` is the encoder's only
  call site, so the proxy runs it, **materializes the result**, and then drops the weights. The
  `mx.eval` is not optional: MLX is lazy, and a returned graph still referencing the encoder's
  parameters pins all 28.2 GB no matter what is deleted.

`_ensure_cache` is overridden as well, because this build's AdaLN table is precomputed rather than
computable — see :mod:`h3_48gb.adaln`.

`__call__` stays a thin wrapper too, for the same reason :mod:`h3_48gb.checkpoint` gives for its own
seams: wrapping `self.dit` sees everything a per-step hook needs (the full packed rows entering
every forward) without duplicating the denoising loop itself. `preview_every` / `preview_stem`
install :class:`h3_48gb.preview.PreviewInterceptor` the same way `checkpoint_dir` installs
`h3_48gb.checkpoint.StepInterceptor` — nested around it if both are active, so a step the checkpoint
fast-forwards is invisible to the preview too. See :mod:`h3_48gb.preview` for the hook itself and
for why a preview can only ever decode the VAE's minimum chunk, not an arbitrary frame.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from . import _upstream  # noqa: F401

import mlx.core as mx
import numpy as np

from minimax_h3_mlx.config import DiTConfig, PipelineConfig
# `_ROPE_FRAMES_PER_LATENT` is private to `packing`, and imported anyway: it *is* the latent
# frame grid (`1, 4, 4, 4, 4` pixel frames per latent frame), and `pinned_pixel_indices` has to
# be that same grid or the tail lands on the wrong moment of the next scene. Re-deriving it from
# the public `_temporal_position_grid` would mean dividing rotary times by `5/3` and rounding —
# the same numbers, arrived at through a float round trip, for the sake of not naming a private.
from minimax_h3_mlx.packing import (
    FRAMES_PER_CHUNK,
    LATENTS_PER_CHUNK,
    PIXEL_MEAN,
    PIXEL_STD,
    _ROPE_FRAMES_PER_LATENT,
    unpatchify_video_tokens,
)
from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline

from . import framecheck, memory
from .adaln import AdaLNCacheFile, ScheduleMismatch
from .checkpoint import CheckpointingPipeline, pop_checkpoint_kwargs, weights_fingerprint
from .dit import load_dit_cached
from .preview import PreviewInterceptor, pop_preview_kwargs
from .text_encoder import QuantizedTextEncoder


def _file_identity(path: Path | None) -> list | None:
    """``[name, size]`` — enough to tell a rebaked file from the one a checkpoint was written under.

    Deliberately not a content hash: see `checkpoint_identity_extra`.
    """
    if path is None:
        return None
    path = Path(path)
    return [path.name, path.stat().st_size if path.exists() else None]


# -- the latent tail: a scene's last few latent frames, kept for the next one -------------------
#
# `docs/FEASIBILITY-latent-handoff.md` §1.2-§1.3. The finished video latent is patchified and
# normalized exactly as `_encode_keyframes` produces conditioning rows, so the tail of scene N is
# already in the space scene N+1 wants — no VAE round trip at either end, and no conversion in
# between. What that costs is a slice and a file.

#: `__call__` keyword arguments this feature owns, stripped before the upstream signature is
#: bound — the same trick, for the same two reasons, as `h3_48gb.preview.PREVIEW_KWARGS`:
#: upstream's `__call__` would reject an unknown keyword, and `request_identity` hashes
#: `bound.arguments` wholesale, so anything that survives the bind joins the checkpoint identity.
#: Where a scene writes its tail must not decide which checkpoint it resumes from.
LATENT_TAIL_KWARGS = ("save_latent_tail", "latent_tail_stem")

#: Appended to the run's output stem. `<stem>-latent-tail.safetensors`, per spec §1.3.
LATENT_TAIL_SUFFIX = "-latent-tail.safetensors"

#: The safetensors metadata key the tail's JSON description lives under. In the *same* file as
#: the array rather than a sidecar `.json`, and that is the whole reason: two files cannot be
#: renamed into place together, so a sidecar reintroduces exactly the torn state the atomic write
#: below exists to prevent — a tail whose geometry says one thing and whose rows say another.
LATENT_TAIL_META_KEY = "h3_latent_tail"

#: Bumped when the metadata below stops meaning what a previous version's reader assumed.
LATENT_TAIL_FORMAT = 1


def pop_latent_tail_kwargs(kwargs: dict) -> dict:
    """Remove and return this feature's keyword arguments from a ``__call__`` kwargs dict."""
    return {name: kwargs.pop(name) for name in LATENT_TAIL_KWARGS if name in kwargs}


def latent_tail_pixel_frames(latent_frames: int) -> int:
    """Pixel frames a tail of ``latent_frames`` latent frames covers: ``17m + 5`` for ``5m + 2``.

    The grid is not a convention, it is the video VAE's own chunking (``FRAMES_PER_CHUNK`` /
    ``LATENTS_PER_CHUNK``, packing.py:46-47): the first latent frame of a clip carries one pixel
    frame and every later one carries four, so ``L`` latent frames cover ``4L - 3`` pixel frames —
    which is a whole number of chunks, and therefore a valid clip start, only when ``L = 5m + 2``.
    That is what makes the tail free: the next scene's own latent grid reproduces it frame for
    frame, with no resampling anywhere (spec §1.3, and the same ``5, 22, 39, 56`` window list
    Motion-Context arrived at).

    Raises:
        ValueError: for any other length. Loudly, and here rather than at the end of a run: a tail
            off this grid cannot be handed to the next scene at all.
    """
    remainder = (latent_frames - 2) % LATENTS_PER_CHUNK
    if latent_frames < 2 or remainder:
        raise ValueError(
            f"A latent tail must be 5m + 2 latent frames (2, 7, 12, 17, ...) so it lands on the "
            f"next scene's own latent grid, got {latent_frames}."
        )
    return FRAMES_PER_CHUNK * ((latent_frames - 2) // LATENTS_PER_CHUNK) + LATENTS_PER_CHUNK


def latent_tail_pixel_indices(latent_frames: int) -> list[int]:
    """Where the tail's frames sit on the *next* scene's rotary clock: ``0, 1, 5, 9, 13, 17, ...``

    One entry per latent frame, in packed order, and each one is what `build_packed_sequence`
    wants as a `keyframe_anchors` element (patch 0004): a **pixel**-frame index, from which it
    computes ``anchor_time = num_text + 5/3 * index``. Written into the tail's metadata rather
    than recomputed on the reading side so this arithmetic lives in one place (spec §1.3).
    """
    indices, position = [], 0
    for frame in range(latent_frames):
        indices.append(position)
        position += _ROPE_FRAMES_PER_LATENT[frame % len(_ROPE_FRAMES_PER_LATENT)]
    return indices


def write_safetensors_atomically(path: Path, arrays: dict, metadata: dict) -> None:
    """Write, flush to stable storage, then rename over whatever was there.

    `CheckpointStore.write`'s sequence (`h3_48gb/checkpoint.py:368-394`), and deliberately not
    `h3_48gb.queue.write_text_durably`, which is a text protocol. Order matters: without the
    ``fsync`` the rename can be durable while the bytes it points at are not, and a power loss
    would leave a file whose safetensors header promises data that is not there. Any failure
    removes the temporary file and leaves the previous file — a scene behind, but whole — exactly
    where it was.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.stem}.tmp-{os.getpid()}{path.suffix}")
    try:
        mx.save_safetensors(str(temp), arrays, metadata=metadata)
        fd = os.open(temp, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(temp, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


# -- configs read without touching the weights -------------------------------------------------

def video_vae_config(model_dir: Path):
    """`VideoVAEConfig` from a converted ``video_vae/`` directory, mirroring `load_video_vae`."""
    from minimax_h3_mlx.video_vae import VideoVAEConfig

    with open(model_dir / "config.json") as fh:
        wrapper = json.load(fh)
    with open(model_dir / "source" / "config.json") as fh:
        source = json.load(fh)
    ch = source["ch"]
    return VideoVAEConfig(
        in_channels=source["in_channels"],
        out_channels=source["out_ch"],
        latent_channels=source["z_channels"],
        block_out_channels=tuple(ch * m for m in source["ch_mult"]),
        layers_per_block=source["num_res_blocks"],
        spatial_downsample_factors=tuple(source["space_down"]),
        temporal_downsample_factors=tuple(source["time_down"]),
        decoder_num_layers=source["vit_decoder_kwargs"]["num_layers"],
        decoder_num_attention_heads=source["vit_decoder_kwargs"]["heads"],
        decoder_attention_head_dim=source["vit_decoder_kwargs"]["dim_head"],
        decoder_rope_theta=source["vit_decoder_kwargs"]["rope_theta"],
        decoder_rope_dim_ratio=source["vit_decoder_kwargs"]["rope_dim_ratio"],
        clip_length=wrapper.get("vae_clip_length", 17),
        token_drop=wrapper.get("vae_token_drop", 3),
        latents_mean=tuple(wrapper.get("latents_mean", ())),
        latents_std=tuple(wrapper.get("latents_std", ())),
    )


def audio_vae_config(model_dir: Path):
    """`AudioVAEConfig` from a converted ``audio_vae/`` directory, mirroring `load_audio_vae`."""
    from minimax_h3_mlx.audio_vae import AudioVAEConfig

    with open(model_dir / "metadata.json") as fh:
        kwargs = json.load(fh)["metadata"]["kwargs"]
    with open(model_dir / "config.json") as fh:
        wrapper = json.load(fh)
    return AudioVAEConfig(
        encoder_dim=kwargs["encoder_dim"],
        encoder_rates=tuple(kwargs["encoder_rates"]),
        latent_dim=kwargs["latent_dim"],
        latent_channels=kwargs["vae_latent_channels"],
        decoder_dim=kwargs["decoder_dim"],
        decoder_rates=tuple(kwargs["decoder_rates"]),
        sampling_rate=kwargs["sample_rate"],
        latents_mean=tuple(wrapper.get("latents_mean", ())),
        latents_std=tuple(wrapper.get("latents_std", ())),
    )


# -- lazy components ---------------------------------------------------------------------------

class LazyComponent:
    """A component that is its config until something needs its weights.

    ``config`` resolves without loading. Any other attribute, or calling the proxy, triggers the
    load. That is exactly the distinction the pipeline's ``__call__`` draws: geometry is computed
    from configs long before the first weight is read.
    """

    def __init__(self, name: str, config, loader, verbose: bool = True):
        self.__dict__["_name"] = name
        self.__dict__["_config"] = config
        self.__dict__["_loader"] = loader
        self.__dict__["_obj"] = None
        self.__dict__["_verbose"] = verbose

    # -- state

    @property
    def loaded(self) -> bool:
        return self._obj is not None

    @property
    def config(self):
        return self._config if self._obj is None else self._obj.config

    def load(self):
        if self._obj is None:
            started = time.perf_counter()
            before = mx.get_active_memory()
            self.__dict__["_obj"] = self._loader()
            if self._verbose:
                grew = (mx.get_active_memory() - before) / 1e9
                print(f"  loaded {self._name}: {grew:+.2f} GB in "
                      f"{time.perf_counter() - started:.1f}s", flush=True)
        return self._obj

    def unload(self) -> None:
        if self._obj is None:
            return
        before = mx.get_active_memory()
        obj = self._obj
        self.__dict__["_obj"] = None
        unload = getattr(obj, "unload", None)
        if callable(unload):
            unload()
        del obj
        memory.release()
        if self._verbose:
            freed = (before - mx.get_active_memory()) / 1e9
            print(f"  unloaded {self._name}: {freed:.2f} GB released", flush=True)

    # -- transparency

    #: The proxy's own state. Everything else forwards — including single-underscore names, because
    #: `_encode_keyframes` reaches straight into ``video_vae._encode_clip``.
    _OWN = frozenset({"_name", "_config", "_loader", "_obj", "_verbose", "_unload_after_encode"})

    def __getattr__(self, item):
        # Only reached for attributes not found normally. Dunders and the proxy's own fields must
        # not trigger a load, or `__init__` and any copy/pickle probe would recurse.
        if item in LazyComponent._OWN or item.startswith("__"):
            raise AttributeError(item)
        return getattr(self.load(), item)

    def __call__(self, *args, **kwargs):
        return self.load()(*args, **kwargs)


class LazyTextEncoder(LazyComponent):
    """The text encoder, which unloads itself the moment its one job is done.

    28.2 GB resident for a single call at the top of the run. `encode` is the only entry point the
    pipeline uses, so wrapping it is enough to bound the encoder's lifetime — and it is the only
    place where the `mx.eval` that makes the release real can be placed correctly.
    """

    def __init__(self, *args, unload_after_encode: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__["_unload_after_encode"] = unload_after_encode

    def encode(self, prompt, images=None):
        hidden_states, token_tags = self.load().encode(prompt, images)
        # Materialize before releasing anything: a lazy graph over the encoder's parameters keeps
        # every one of them alive, and the unload below would free nothing at all.
        mx.eval(hidden_states)
        # Note the absence of a local holding the encoder. Binding `encoder = self.load()` above
        # and unloading here would free nothing until this frame returned, because that local is
        # another live reference — measured, not assumed (see test_lazy_pipeline.py).
        if self._unload_after_encode:
            self.unload()
        return hidden_states, token_tags


def _validate_decoded_frames(frames: np.ndarray) -> None:
    """Raise `framecheck.CorruptFramesError` if any frame in `frames` (F, H, W, 3) uint8 trips
    `framecheck`'s zero-fill/tile-seam corruption detectors.

    P0 fix for боевые ворота 2026-08-19's root cause (see `framecheck`'s own module docstring):
    `patches/0003-vae-decode-eval.patch` bounds `VideoVAE.decode`'s lazy graph, which should make
    the corruption this catches unreachable in practice — but a decode is expensive (minutes) and
    feeds a scene chain (`h3_48gb.assemble`) a silently-corrupt keyframe would poison, so this
    stays as defense in depth rather than a redundant check to delete once the patch lands.
    Costs ~2.3 s per clip (measured by the investigation), against a decode that runs minutes.

    A caught failure here means the job goes to `failed/` (the existing job-failure path already
    does this for any raised exception — see `h3_48gb.worker`), not a corrupt video written to
    disk and fed silently into the next scene.
    """
    bad = framecheck.find_corrupt_frames(frames)
    if not bad:
        return
    indices = ", ".join(str(b.index) for b in bad[:20])
    more = "" if len(bad) <= 20 else f" (+{len(bad) - 20} more)"
    kinds = sorted({("zero-fill" if b.zero_fill else "tile-seam") for b in bad})
    raise framecheck.CorruptFramesError(
        f"{len(bad)} of {frames.shape[0]} decoded frames failed corruption validation "
        f"({'/'.join(kinds)}): frame indices {indices}{more}. This clip cannot be trusted for "
        "output or for chaining the next scene off a keyframe -- failing the job rather than "
        "emitting a corrupt video."
    )


# -- the pipeline ------------------------------------------------------------------------------

class LazyMiniMaxH3Pipeline(CheckpointingPipeline, MiniMaxH3Pipeline):
    """`MiniMaxH3Pipeline` with phase-scoped residency, a precomputed AdaLN table and resumable runs."""

    #: The in-flight `--save-latent-tail` request, or `None`. A class attribute as well as an
    #: instance one so `_decode_video` can be driven by a stand that never ran `__init__` (the
    #: `Bare` stub in tests/test_decode_video_uint8.py) without it having to know this exists.
    _latent_tail: dict | None = None

    def __init__(self, dit, text_encoder, video_vae, audio_vae, config, adaln_cache_path=None,
                 verbose: bool = True, weights_id: dict | None = None):
        super().__init__(dit, text_encoder, video_vae, audio_vae, config)
        self._adaln_cache_path = Path(adaln_cache_path) if adaln_cache_path else None
        #: Set by `h3_48gb.cli.install_turbo_lora`; part of the checkpoint identity because the
        #: adapter and its strength change every latent the run produces.
        self._turbo_lora: Path | None = None
        self._turbo_strength: float | None = None
        self._schedules = None
        self._cached_grids: tuple[list[float], list[float]] | None = None
        self._supported_steps: int | None = None
        self._weights_id = weights_id or {}
        self._tracker = memory.PhaseTracker(verbose=verbose)
        # Bound the allocator's cache before anything is loaded: unbounded, it drifts upward across
        # a multi-hour run until the machine swaps, and swapping showed up as a step taking 818 s
        # where its neighbours took 568.
        memory.limit_cache()

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str | Path,
        transformer_dir: str | Path | None = None,
        dtype: mx.Dtype = mx.bfloat16,
        load_vision: bool = True,
        unload_after_encode: bool = True,
        verbose: bool = True,
        adaln_cache: str | Path | None = None,
    ) -> "LazyMiniMaxH3Pipeline":
        """Read every config; read no weights.

        ``load_vision`` defaults to True here, unlike upstream: keyframe conditioning is the point
        of the fl2va partition, and the tower is only 1.1 GB — and it is released with the rest of
        the encoder after the single `encode` call.
        """
        root = Path(checkpoint_dir)
        dit_path = Path(transformer_dir) if transformer_dir else root / "transformer"
        config = PipelineConfig.from_model_index(root / "model_index.json")

        dit_config = DiTConfig.from_json(dit_path / "config.json")
        # An alternate table swaps the whole sampling schedule — that is how few-step runs work.
        # Passing the path beats the alternative of building a symlink tree per step count, which
        # is what this required before and is easy to get subtly wrong.
        cache_path = Path(adaln_cache) if adaln_cache else dit_path / "adaln_cache.safetensors"

        if verbose:
            print(f"loading MiniMax-H3 configs from {root} (weights are deferred)")

        components = {
            "dit": LazyComponent(
                "transformer", dit_config,
                lambda: load_dit_cached(dit_path, verbose=verbose), verbose),
            "text_encoder": LazyTextEncoder(
                "text encoder", None,
                lambda: QuantizedTextEncoder(root / "text_encoder", dtype=dtype,
                                             load_vision=load_vision, verbose=verbose),
                verbose=verbose, unload_after_encode=unload_after_encode),
            "video_vae": LazyComponent(
                "video vae", video_vae_config(root / "video_vae"),
                lambda: _load_video_vae(root / "video_vae"), verbose),
            "audio_vae": LazyComponent(
                "audio vae", audio_vae_config(root / "audio_vae"),
                lambda: _load_audio_vae(root / "audio_vae"), verbose),
        }
        return cls(components["dit"], components["text_encoder"], components["video_vae"],
                   components["audio_vae"], config,
                   adaln_cache_path=cache_path if cache_path.exists() else None,
                   verbose=verbose,
                   # Which weights produced a checkpoint is part of its identity: resuming a run
                   # onto a reconverted transformer would denoise into a different model.
                   weights_id=weights_fingerprint(dit_path, root / "text_encoder",
                                                  root / "video_vae", root / "audio_vae"))

    # -- keyframe conditioning ------------------------------------------------------------------

    def _encode_keyframes(self, images, height, width):
        """Materialize the video VAE *before* upstream seeds the keyframe generator.

        Upstream constructs every component in `from_pretrained`, long before
        `_encode_keyframes` calls `mx.random.seed(42)` and draws the posterior sample. Here the
        VAE is a proxy, so the first touch of `video_vae._encode_clip` builds it — which happens
        *after* that seed, and building it draws from MLX's global RNG for some 560 parameter
        tensors. The sample therefore comes off a different point in the stream, and the
        conditioning latents depend on whether the VAE happened to be resident already: a cold
        and a warm run of the same request differed by 0.87.

        Loading here rather than in `__call__` keeps the peak where it was — by this point the
        28.2 GB text encoder has been released.
        """
        self.video_vae.load()
        encode = super()._encode_keyframes
        if len(images) < 2:
            return self._release_vae_after(encode(images, height, width))

        # Seed 42 once per keyframe, not once per request. Upstream seeds outside its loop, so the
        # second keyframe's posterior sample continues the stream; the reference builds a fresh
        # `torch.Generator().manual_seed(...)` for every image
        # (`reference/diffusers/modular/encoders.py:297`), which means two identical keyframes must
        # encode identically. They did not: 0.83 apart. Encoding one at a time puts upstream's own
        # seed call in front of each draw.
        #
        # Safe only because `__call__` has already put every keyframe on the canvas: upstream
        # stretches image 0 and cover-crops the rest, a distinction that is lost when each is
        # passed as index 0. Prepared frames come back untouched, so the distinction is moot — but
        # if that ever stops being true, this must fail rather than silently stretch keyframe 2.
        wrong_size = [(index, image.size) for index, image in enumerate(images)
                      if image.size != (width, height)]
        if wrong_size:
            raise RuntimeError(
                f"Keyframes must already be on the {width}x{height} canvas before they are encoded "
                f"one at a time, but {wrong_size} are not. See `__call__`, which prepares them."
            )
        return self._release_vae_after(
            mx.concatenate([encode([image], height, width) for image in images]))

    def _release_vae_after(self, rows: mx.array) -> mx.array:
        """Materialize the conditioning rows, then drop the video VAE until the final decode.

        Encoding keyframes is the VAE's first job and the decode is its second, with the entire
        diffusion loop in between — hours, on a long clip. Holding 5.21 GB across it buys nothing.

        The `mx.eval` is what makes the release real rather than nominal: until the rows are
        materialized they are a lazy graph over the VAE's parameters, and dropping the module
        would free none of them. Same trap `LazyTextEncoder` documents for the 28.2 GB encoder.
        """
        mx.eval(rows)
        self.video_vae.unload()
        memory.release()
        return rows

    # -- decoding: the diffusion phase is over, so stop paying for it ---------------------------

    def _decode_video(self, rows, num_latent_frames, latent_height, latent_width) -> np.ndarray:
        """Release the DiT before decoding, then decode straight to uint8 inside the MLX graph.

        By the time this runs the last forward is done and the transformer will not be touched
        again — but upstream keeps it resident to the end of the call, so decoding pays for
        11.34 GB of transformer, 0.62 GB of LoRA and 0.13 GB of modulation table it cannot use.
        That is 12.1 GB on top of the video VAE's 5.21, and decoding is where the peak lands on a
        long clip: 243 frames at 896x576 tile 28 ways each.

        `video_rows` and `audio_rows` are already materialized — upstream's loop calls
        `mx.eval(video_rows, audio_rows)` every step — so dropping the transformer here cannot
        strand a lazy graph. That distinction is the whole reason the text encoder's unload works
        (see `LazyTextEncoder`): an unevaluated graph over a module's parameters pins all of them
        no matter what is deleted.

        The body below is upstream's `_decode_video` (`upstream/minimax_h3_mlx/pipeline.py:359`)
        with one change: upstream calls `np.array()` on the VAE's raw float32 output and then runs
        the pixel-mean/std unnormalize, the [0, 1] clip and the *0.5 uint8 round entirely in numpy,
        on the host. Every one of those reassigns `frames` to a brand-new full-size array — on a
        10 s clip at the native canvas that is four float32 copies of the whole video (measured
        +10.8 GB RSS beyond the VAE's own 15.4 GB MLX peak; see
        `docs/FEASIBILITY-chunked-vae.md` and the probe it was measured with). None of that
        arithmetic needs numpy: `mx.clip`, multiply and `.astype(mx.uint8)` do the same thing on
        the still-lazy MLX graph, so the *only* array that ever crosses into numpy is the final
        uint8 frame buffer — a quarter the size of one of the float32 copies upstream made, and
        there is exactly one of it. `test_decode_video_matches_upstream_bytes` pins this against
        upstream's own numpy path byte-for-byte on a random small latent.
        """
        mx.eval(rows)
        # The one seam where the latent is still raw. `rows` has just been materialized, so the
        # write cannot drag the transformer's last forward into itself, and the two lines below
        # have not yet unpatchified or denormalized it out of the space the next scene wants. That
        # seam is *before* `_validate_decoded_frames` below, though -- the corruption check needs
        # decoded pixels, which do not exist yet -- so a tail written here can still turn out to
        # belong to a clip the validator later condemns. `tail_path` is only set when *this* call
        # wrote a file, so the failure handler below removes exactly that file and nothing a prior,
        # successful run left behind.
        tail_path = None
        if self._latent_tail is not None:
            tail_path = self._save_latent_tail(rows, num_latent_frames, latent_height, latent_width)
        self.dit.unload()
        self._cache = None
        self._cache_timesteps = None
        memory.release()

        cfg = self.video_vae.config
        latents = unpatchify_video_tokens(
            rows, num_latent_frames, latent_height, latent_width, cfg.latent_channels,
            self.dit.config.patch_size,
        )
        mean = mx.array(np.array(cfg.latents_mean, np.float32)).reshape(1, -1, 1, 1, 1)
        std = mx.array(np.array(cfg.latents_std, np.float32)).reshape(1, -1, 1, 1, 1)
        latents = latents * std + mean

        if not vae_decode_eval_patch_applied():
            raise RuntimeError(
                "`upstream/minimax_h3_mlx/video_vae.py` has no per-tile/per-chunk `mx.eval` in "
                "`VideoVAE._decode_clip`/`decode`, so patches/0003-vae-decode-eval.patch is not "
                "applied to this checkout. Apply it (`git -C upstream apply "
                "../patches/0003-vae-decode-eval.patch`, see README) before decoding -- an "
                "unbounded decode graph can non-deterministically leave a tile unwritten or "
                "corrupt a tile seam under allocator pressure (боевые ворота 2026-08-19)."
            )
        frames = self.video_vae.decode(latents.astype(mx.float32))
        # The VAE decodes into ImageNet-normalized RGB over a [0, 1] base range.
        pixel_mean = mx.array(np.array(PIXEL_MEAN, np.float32)).reshape(1, 3, 1, 1, 1)
        pixel_std = mx.array(np.array(PIXEL_STD, np.float32)).reshape(1, 3, 1, 1, 1)
        frames = frames * pixel_std + pixel_mean
        frames = mx.clip(frames, 0.0, 1.0)
        # Same `* 255.0 + 0.5` round-to-nearest upstream does in numpy, then the narrowing cast —
        # both still lazy, both still on the MLX side of the `np.array()` barrier below.
        frames = (frames * 255.0 + 0.5).astype(mx.uint8)
        frames = frames[0].transpose(1, 2, 3, 0)  # -> (F, H, W, 3)
        frames = np.array(frames)
        try:
            _validate_decoded_frames(frames)
        except Exception:
            # Хвост уже лежит на диске (шов записи — до декода, у валидатора там ещё нет
            # пикселей), а клип только что осуждён: не убрать файл — значит оставить артефакт,
            # который позже `--latent` подберёт как вход следующей сцены (ревью задачи 3, P2-6).
            # Только файл ЭТОГО вызова — прошлые успешные хвосты не трогаются. Best-effort:
            # ошибка уборки не должна заслонить настоящую ошибку валидации.
            if tail_path is not None:
                try:
                    tail_path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise
        return frames

    def _save_latent_tail(self, rows, num_latent_frames, latent_height, latent_width) -> Path:
        """Cut the last `L` latent frames off the finished latent and write them beside the clip.

        **The slice** (spec §1.3). `rows` is `video_rows[n_cond_v:]` — the generated video latent,
        patchified frame-major by `patchify_video_latents` (its `transpose(0,2,4,6,1,3,5,7)`), so
        latent frame `f` owns rows ``[f*R, (f+1)*R)`` and the whole tail is one contiguous slice:

            R = rows_per_frame = (latent_height // ph) * (latent_width // pw)
            L = 5m + 2                        latent frames kept, m >= 0
              covers 17m + 5 pixel frames     of this clip's own timeline
            tail = rows[(F - L) * R:]         shape (L*R, video_patch_dim), F = num_latent_frames

        `R` is taken from `rows.shape[0] // F` rather than recomputed from the canvas: it is then
        the layout's own answer, and a disagreement with `F` (a remainder) is a refusal rather
        than a silently misaligned cut. `L`'s `5m + 2` shape is enforced by
        `latent_tail_pixel_frames`, which explains why it is the only grid that works.

        **The geometry in the metadata** comes from the request (`_latent_tail_request`, off
        `bound.arguments`, the same source `_install_preview` reads), because the canvas the
        operator asked for is what the next scene has to match — `latent_height`/`latent_width`
        here are its 16x-divided shadow.

        Returns:
            The path written, `<stem>-latent-tail.safetensors`.
        """
        request = self._latent_tail
        latent_frames = int(request["latent_frames"])
        rows_per_frame, remainder = divmod(int(rows.shape[0]), int(num_latent_frames))
        if remainder:
            raise RuntimeError(
                f"{rows.shape[0]} generated rows do not divide into {num_latent_frames} latent "
                f"frames — the packed layout is not what this expects, and any tail cut from it "
                f"would start mid-frame."
            )
        if latent_frames > num_latent_frames:
            raise ValueError(
                f"A tail of {latent_frames} latent frames was asked for, but this clip has only "
                f"{num_latent_frames}."
            )
        pixel_frames = latent_tail_pixel_frames(latent_frames)

        tail = rows[(num_latent_frames - latent_frames) * rows_per_frame:]
        # Materialize the slice for the same reason `mx.eval(rows)` above exists: the write must
        # not be the thing that runs it. `rows` is already real, so this is one slice's worth of
        # work and nothing of the transformer is left in the graph.
        mx.eval(tail)

        width, height = request["canvas"]
        meta = {
            "format": LATENT_TAIL_FORMAT,
            "canvas": [int(width), int(height)],
            "latent_frames": latent_frames,
            "m": (latent_frames - 2) // LATENTS_PER_CHUNK,
            "pixel_frames": pixel_frames,
            "rows_per_frame": rows_per_frame,
            "row_dim": int(rows.shape[1]),
            "latent_height": int(latent_height),
            "latent_width": int(latent_width),
            "patch_size": [int(p) for p in self.dit.config.patch_size],
            "dtype": str(tail.dtype).rsplit(".", 1)[-1],
            "source_latent_frames": int(num_latent_frames),
            # What the next scene passes as `keyframe_anchors` — written, not recomputed there.
            "pinned_pixel_indices": latent_tail_pixel_indices(latent_frames),
        }

        path = Path(f"{request['stem']}{LATENT_TAIL_SUFFIX}")
        write_safetensors_atomically(path, {"video_tail": tail},
                                     {LATENT_TAIL_META_KEY: json.dumps(meta)})
        if request.get("verbose"):
            print(f"latent tail: {latent_frames} latent frames ({pixel_frames} pixel frames) "
                  f"-> {path}", flush=True)
        return path

    def _decode_audio(self, rows, *args, **kwargs):
        """Release the video VAE before the audio VAE loads — they are never needed together."""
        mx.eval(rows)
        self.video_vae.unload()
        memory.release()
        return super()._decode_audio(rows, *args, **kwargs)

    # -- schedule / modulation ------------------------------------------------------------------

    def _cached_sigma_grids(self):
        """The sigma grids the cached table was baked for, or None without a cache.

        Only the two sigma tensors are touched; the modulation blocks beside them in the same
        file are never read.
        """
        if self._adaln_cache_path is None:
            return None
        if self._cached_grids is None:
            raw = mx.load(str(self._adaln_cache_path))
            self._cached_grids = ([float(v) for v in raw["video_sigmas"].tolist()],
                                  [float(v) for v in raw["audio_sigmas"].tolist()])
        return self._cached_grids

    def _build_schedules(self, num_inference_steps: int):
        """Keep the schedulers: `_ensure_cache` needs them to locate rows in the cached table.

        The cache is authoritative about *which* sigma grid it was baked for, not only about the
        modulation rows it holds. Upstream rebuilds a uniform ``linspace(1, 0, N)`` grid here, and
        `AdaLNCacheFile.check_schedule` would then reject every non-uniform schedule the baker can
        produce — but a non-uniform grid is exactly the point of baking one: at 8 steps the uniform
        grid spends its last forward jumping from sigma 0.667 straight to 0, and fine detail is
        what an Euler step that long loses. Adopting the stored grid costs a uniform cache nothing:
        the baker wrote it from this same scheduler in float32, so the values round-trip exactly
        (`test_simple_cache_grid_is_unchanged`).
        """
        schedules = super()._build_schedules(num_inference_steps)
        grids = self._cached_sigma_grids()
        if grids is not None and len(grids[0]) == num_inference_steps:
            # Both modalities or neither. The two grids are stepped in lockstep by a `zip` over the
            # timestep lists: a longer audio grid is silently truncated and the audio latents finish
            # above sigma 0, a shorter one runs the loop off the end of the plan. `check_schedule`
            # cannot catch either, because by then both schedulers hold exactly what the table says.
            if len(grids[1]) != len(grids[0]):
                raise ScheduleMismatch(
                    f"{self._adaln_cache_path.name} holds {len(grids[0])} video sigmas and "
                    f"{len(grids[1])} audio sigmas. The two modalities are stepped together, so "
                    "the grids must be the same length; rebake the table."
                )
            video, audio = schedules
            video.set_timesteps(sigmas=grids[0])
            audio.set_timesteps(sigmas=grids[1])
        # A video-length mismatch is left alone on purpose: `check_schedule` reports it with the
        # grid sizes and the N-vs-N-1 convention spelled out, which beats anything raised here.
        self._schedules = schedules
        return schedules

    def _ensure_cache(self, timesteps: mx.array, drop_adaln: bool, verbose: bool):
        """Serve the modulation table from mere.run's cache instead of computing it.

        `ModulationCache.build` reads ``dit.time_embedder`` and every block's ``adaln_proj``; this
        build ships neither. ``drop_adaln`` is moot — the projections were never allocated.
        """
        if self._adaln_cache_path is None:
            return super()._ensure_cache(timesteps, drop_adaln, verbose)
        if self._schedules is None:
            raise RuntimeError("`_build_schedules` must run before the modulation table is built.")

        key = tuple(round(float(v), 9) for v in timesteps.tolist())
        if self._cache is not None and self._cache_timesteps == key:
            return

        started = time.perf_counter()
        video_sched, audio_sched = self._schedules
        cache_file = AdaLNCacheFile(self._adaln_cache_path)
        cache = cache_file.build(timesteps, video_sched, audio_sched, dtype=mx.bfloat16,
                                 verbose=False)

        # The DiT's output layer has no `adaln_proj` either; hand it the cached shift/scale.
        final_layer = self.dit.final_layer
        if not hasattr(final_layer, "set_modulation"):
            raise RuntimeError(
                "The DiT was not loaded by h3_48gb.dit.load_dit_cached, so its output layer still "
                "expects `final_layer.adaln_proj` — which this checkpoint does not carry."
            )
        final_layer.set_modulation(cache.final_shift, cache.final_scale)

        self._cache = cache
        self._cache_timesteps = key
        if verbose:
            print(f"  adaln cache: {len(key)} timesteps, {cache.nbytes() / 1e6:.0f} MB "
                  f"from {self._adaln_cache_path.name} in {time.perf_counter() - started:.1f}s")

    # -- guard rails ----------------------------------------------------------------------------

    def supported_num_inference_steps(self) -> int | None:
        """The one step count this build's precomputed table can serve, if it has one.

        The port's scheduler spends one grid point on the terminal sigma, so ``N`` grid points drive
        ``N - 1`` forwards — the cache's ``video_sigmas`` length *is* the argument to pass.
        """
        if self._adaln_cache_path is None:
            return None
        if self._supported_steps is None:
            raw = mx.load(str(self._adaln_cache_path))
            self._supported_steps = int(raw["video_sigmas"].shape[0])
        return self._supported_steps

    def checkpoint_identity_extra(self) -> dict:
        """Everything outside the request that steers the run, so a resume cannot cross streams.

        The AdaLN table used to be identified by filename alone. That is not enough once tables
        are baked rather than shipped: two tables can share a name and differ in step count, in
        sigma grid, or in whether a LoRA was folded into them — and resuming across that produces
        a clip denoised half one way and half the other, with nothing raised.

        Same for the runtime adapter. `turbo_strength` changes every latent it touches, so a run
        resumed at a different strength is not the run that was interrupted.

        Path plus byte size, as `weights_fingerprint` does, and for the same reason: hashing the
        contents would cost more than the step it protects, while size catches the failure that
        actually happens — a rebaked or swapped file.

        Size alone is not enough for the table, though, now that the run samples on whatever grid
        the table carries: a uniform 9-point table and a tail-split-to-9 one are the same shape and
        therefore the same size, so a swap under the same name would resume a half-finished clip
        onto a different trajectory with nothing raised. The grid itself is a few dozen floats and
        is already read to build the schedule, so it goes into the identity directly.
        """
        return {
            "weights": self._weights_id,
            "sigma_shift_video": self.config.sigma_shift_video,
            "sigma_shift_audio": self.config.sigma_shift_audio,
            "adaln_cache": _file_identity(self._adaln_cache_path),
            "sigma_grid": self._cached_sigma_grids(),
            "turbo_lora": _file_identity(self._turbo_lora),
            "turbo_strength": self._turbo_strength,
        }

    def __call__(self, *args, **kwargs):
        """Validate the schedule, install the checkpoint/preview seams, then run upstream's loop.

        A wrong step count would otherwise be caught only once the encoder had been loaded and run
        — 28.2 GB and several minutes into the request.
        """
        # `pop`/re-add rather than `**kwargs` passthrough: the bind below is against *upstream's*
        # signature, which knows nothing about the checkpoint/preview arguments and would reject
        # them.
        checkpoint_kwargs = pop_checkpoint_kwargs(kwargs)
        preview_options = pop_preview_kwargs(kwargs)
        latent_tail_options = pop_latent_tail_kwargs(kwargs)
        preview_every = int(preview_options.get("preview_every", 0) or 0)
        preview_stem = preview_options.get("preview_stem")
        preview_decoder = preview_options.get("preview_decoder", "vae") or "vae"
        supported = self.supported_num_inference_steps()

        import inspect

        bound = inspect.signature(MiniMaxH3Pipeline.__call__).bind(self, *args, **kwargs)
        bound.apply_defaults()
        if supported is not None:
            requested = int(bound.arguments["num_inference_steps"])
            if requested != supported:
                raise ScheduleMismatch(
                    f"This build's AdaLN table is precomputed for num_inference_steps={supported} "
                    f"({supported - 1} forwards) at sigma shifts 12.0/3.0, but the request asks "
                    f"for {requested}. mere.run dropped the modulation projections, so no other "
                    "schedule can be evaluated — pass "
                    f"num_inference_steps={supported}."
                )

        if preview_every < 0:
            raise ValueError(f"`preview_every` must be >= 0 (0 disables previews), got {preview_every}.")

        # Hand both consumers the same picture. Upstream passes the raw `images` to the text
        # encoder (its line 238) and only later prepares them for the VAE (its line 192), so
        # Qwen3-VL describes the original while the conditioning latent is made from the canvas
        # version. The reference prepares once, before either consumer
        # (`reference/diffusers/modular/before_encoder.py:190`). Mismatched, nothing raises: the
        # number of vision tokens changes, and `packing` derives every media row's rotary clock
        # from it. Preparing here is safe to repeat — `prepare_keyframe_image` returns an image
        # that already is the canvas untouched, so upstream's own call becomes a no-op.
        images = bound.arguments.get("images")
        if images:
            from minimax_h3_mlx.packing import prepare_keyframe_image, resolve_canvas_size

            height, width = bound.arguments["height"], bound.arguments["width"]
            if height is None or width is None:
                height, width = resolve_canvas_size(*bound.arguments["aspect"])
            bound.arguments["images"] = [
                prepare_keyframe_image(image, height, width, stretch=index == 0)
                for index, image in enumerate(images)
            ]
            args, kwargs = bound.args[1:], bound.kwargs

        # Sized here, off the same `bound.arguments` the preview reads, so a tail that cannot be
        # written (no stem, off-grid, longer than the clip) is refused now rather than at the end
        # of a multi-hour run — the tail is the *point* of a chained scene, and discovering it was
        # never going to be written after the diffusion loop costs the whole scene.
        latent_tail = self._latent_tail_request(latent_tail_options, bound.arguments)

        original_dit = self.dit
        original_latent_tail = self._latent_tail
        self._latent_tail = latent_tail
        if preview_every:
            self.dit = self._install_preview(original_dit, preview_every, preview_stem,
                                             bound.arguments, decoder=preview_decoder)
        try:
            return super().__call__(*args, **kwargs, **checkpoint_kwargs)
        finally:
            self.dit = original_dit
            self._latent_tail = original_latent_tail

    def _latent_tail_request(self, options: dict, arguments: dict) -> dict | None:
        """Validate `--save-latent-tail` against this request's geometry, or return `None`.

        Re-derives the canvas and the latent frame count from the request exactly as
        `_install_preview` does, and for the same reason: there is nowhere inside upstream's
        `__call__` to reach for them without duplicating the loop. Only the canvas travels into
        the metadata — the row geometry `_save_latent_tail` writes comes from the rows themselves.
        """
        latent_frames = int(options.get("save_latent_tail", 0) or 0)
        if latent_frames == 0:
            return None

        from minimax_h3_mlx.packing import (
            FPS, align_num_frames, resolve_canvas_size, video_latent_num_frames,
        )

        latent_tail_pixel_frames(latent_frames)     # raises on anything off the 5m + 2 grid
        stem = options.get("latent_tail_stem")
        if stem is None:
            raise ValueError(
                "`save_latent_tail` > 0 requires `latent_tail_stem`: the tail is written to "
                "`<latent_tail_stem>-latent-tail.safetensors`."
            )

        height, width = arguments["height"], arguments["width"]
        if height is None or width is None:
            height, width = resolve_canvas_size(*arguments["aspect"])
        num_frames = align_num_frames(int(round(arguments["duration_seconds"] * FPS)))
        num_latent_frames = video_latent_num_frames(num_frames)
        if latent_frames > num_latent_frames:
            raise ValueError(
                f"`save_latent_tail={latent_frames}` is longer than the clip: "
                f"{arguments['duration_seconds']} s is {num_frames} frames, which the video VAE "
                f"encodes as {num_latent_frames} latent frames."
            )
        return {
            "latent_frames": latent_frames,
            "stem": Path(stem),
            "canvas": (int(width), int(height)),
            "verbose": bool(arguments.get("verbose", True)),
        }

    def _install_preview(self, dit, every: int, stem, arguments: dict,
                         decoder: str = "vae") -> PreviewInterceptor:
        """Size a `PreviewInterceptor` from the same public geometry helpers upstream uses.

        Deliberately re-derives ``latent_height``/``latent_width``/``num_latent_frames``/
        ``n_cond_v`` from the request arguments rather than reaching into upstream's `__call__` for
        them — there is nowhere to reach into without duplicating the loop (see the module
        docstring). These are pure functions of the request plus the VAE/DiT configs, called here
        exactly as `MiniMaxH3Pipeline.__call__` calls them, so this can drift only if upstream's
        canvas/frame-count math changes shape entirely — which `test_preview.py` would catch via the
        shape check inside `preview.render_preview_frame`.
        """
        from minimax_h3_mlx.packing import FPS, align_num_frames, resolve_canvas_size, video_latent_num_frames

        if stem is None:
            raise ValueError(
                "`preview_every` > 0 requires `preview_stem`: previews are written to "
                "`<preview_stem>-preview-stepNN.jpg`."
            )

        height, width = arguments["height"], arguments["width"]
        if height is None or width is None:
            height, width = resolve_canvas_size(*arguments["aspect"])
        num_frames = align_num_frames(int(round(arguments["duration_seconds"] * FPS)))
        num_latent_frames = video_latent_num_frames(num_frames)
        ratio = self.video_vae.config.spatial_compression_ratio
        latent_height, latent_width = height // ratio, width // ratio
        patch_size = self.dit.config.patch_size
        rows_per_frame = (latent_height // patch_size[1]) * (latent_width // patch_size[2])
        n_cond_v = len(arguments["keyframe_anchors"]) * rows_per_frame

        return PreviewInterceptor(
            dit, self, every, stem, n_cond_v, num_latent_frames, latent_height, latent_width,
            patch_size, verbose=bool(arguments.get("verbose", True)), decoder=decoder,
        )


def vae_decode_eval_patch_applied(tile_source: str | None = None, chunk_source: str | None = None) -> bool:
    """Whether the vendored `upstream/` carries `patches/0003-vae-decode-eval.patch`.

    Same discipline as `h3_48gb.dit.attention_levers_patch_applied` /
    `h3_48gb.text_encoder.keyframe_scatter_patch_applied`, but textual like the latter rather than
    structural like the former: patch 0003 inserts `mx.eval` statements into two *existing*
    methods (`VideoVAE._decode_clip`, `VideoVAE.decode`) instead of adding a new one, so there is
    no new attribute for `hasattr` to find. Reads the live source of both methods, strips comment
    lines first (the patch quotes the very lines it replaces, which would otherwise fool a naive
    substring search — the same trap `keyframe_scatter_patch_applied`'s own docstring documents),
    and requires each method to carry its own materialization call.

    `tile_source`/`chunk_source` override the two methods' live source, the same seam
    `keyframe_scatter_patch_applied`'s own `source` parameter provides — what lets a test assert
    the detector's own behaviour against known patched/unpatched text without needing to flip
    `upstream/`'s checkout back and forth.
    """
    import inspect

    def has_marker(source: str, marker: str) -> bool:
        code = [line for line in source.splitlines() if not line.lstrip().startswith("#")]
        return any(marker in line for line in code)

    if tile_source is None or chunk_source is None:
        from minimax_h3_mlx.video_vae import VideoVAE

        if tile_source is None:
            tile_source = inspect.getsource(VideoVAE._decode_clip)
        if chunk_source is None:
            chunk_source = inspect.getsource(VideoVAE.decode)
    return has_marker(tile_source, "mx.eval(decoded_tile)") and has_marker(chunk_source, "mx.eval(chunk)")


def _load_video_vae(model_dir: Path):
    from minimax_h3_mlx.load import load_video_vae

    return load_video_vae(model_dir)


def _load_audio_vae(model_dir: Path):
    from minimax_h3_mlx.load import load_audio_vae

    return load_audio_vae(model_dir)
