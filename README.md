# minimax-h3-mlx

> **Status: Archived. Not maintained. Provided as-is.** This is a one-commit snapshot of a working
> port, published for reference. No issues, pull requests or support.

A fork of [PipeNetwork/minimax-h3-mlx](https://github.com/PipeNetwork/minimax-h3-mlx) (Apache-2.0),
the MLX port of [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) — a 33B diffusion
transformer that generates synchronized video and audio — that makes it run on a **Mac with 48 GB of
unified memory**. The upstream port and the reference Swift runtime both assume far more memory than
that; this fork changes nothing about the model, only when each of its parts is resident, and adds
the checkpointing, previews and CLI a multi-hour local render needs in practice.

Headline numbers (all measured, details in [`docs/RESULTS.md`](docs/RESULTS.md)):

* Peak memory during the multi-hour diffusion phase: **~21 GB** (MLX accounting); text-encoding
  peak 28.2 GB for about 10 seconds; the upstream layout would need ~55 GB.
* 512x512, 2.4 s clip: **24 min** at 31 steps; 8 steps + Turbo LoRA: 6.6 min.
* 1344x768 (native), 5 s clip: **299 min** at 31 steps.

This snapshot contains the MLX port only (code, converter, CLI, patches, tests, technical notes).
It contains no model weights.

## The memory problem, in numbers

The upstream pipeline loads every component at `from_pretrained` and keeps all of them resident for
the whole run: text encoder, DiT, video VAE, audio VAE. This machine has **48 GB**. mere.run, the
Swift runtime whose model build this fork's converter targets, will not even attempt it: it refuses
outright with *"Requires at least 96 GB unified memory; detected 48 GB"*, and, separately, its
admission control demands 32 GB free before it will admit any job at all.

The four components are needed in disjoint phases, though — the text encoder runs once, at the very
start, and is dead weight for the rest of the render. Loading them one phase at a time and
discarding each as its phase ends turns one number into three:

| Peak | Phase | Measured by |
|---|---|---|
| **~55 GB** | all four components resident at once, at the moment diffusion peaks | 45.9 GB of weights (this fork's own run log, below) + 9.3 GB of activations at 1344x768/5 s (upstream's measurement) |
| **28.2 GB** | peak during the text-encoding phase — which lasts about **10 seconds** | MLX's `get_active_memory`, printed by the run itself as `loaded text encoder: +28.22 GB in 8.3s` |
| **~21 GB** | for the entire multi-hour diffusion phase, which is >99% of the wall clock | MLX's own accounting: 12.09 GB of weights plus ~9.3 GB of activations. An earlier version of this table said 10.1–11.5 GB from process RSS — that instrument cannot see Metal allocations at all, see [`docs/MEMORY.md`](docs/MEMORY.md) |

The 45.9 GB is the sum of the four `loaded <component>: +N GB` lines MLX prints during a real run
(28.22 + 11.34 + 5.21 + 0.61) plus the 0.56 GB AdaLN cache; the 9.3 GB activation figure is
upstream's own, from its README's sequence-length table. **The three figures come from two
different instruments, and the honest version of that is in
[`docs/RESULTS.md`](docs/RESULTS.md)** — including the fact that the RSS trace never sampled the
encoding phase at all, so the 28.2 GB and the 11.5 GB were never observed by the same tool.

## The four modules, and the five source patches

Everything this fork adds lives in `h3_48gb/`, applied to upstream from the outside — the four
modules below. `upstream/` itself is a vendored clone, pinned to commit `fcd9e9b`, and carries
exactly five source edits, all applied during setup below:

* `patches/0001-keyframe-masked-scatter.patch`, without which keyframe runs die inside the text
  encoder. Text-only runs never reach it.
* `patches/0002-attention-memory-levers.patch`, four bit-identical rewrites of the DiT's Q/K/V and
  MLP staging. Measured worth: 8.65 GB off the peak of a 15 s native forward, 4.92 GB at 10 s (see
  `docs/RESULTS.md`, "Attention memory levers"). Nothing about the outputs changes.
* `patches/0003-vae-decode-eval.patch`, a bit-identical `mx.eval` after every VAE decode tile and
  chunk. Without it, `VideoVAE.decode` chains up to 195 ViT-decoder forwards into a single lazy
  graph and materializes all of it in one `np.array()` call at the very end — under allocator
  pressure, a fraction of that graph's tiles could come back as an unwritten (zero) or garbled
  buffer instead of what was actually computed, non-deterministically (боевые ворота 2026-08-19's
  root cause: corrupted keyframes and tile-seam blocking in generated clips). `h3_48gb.framecheck`
  is defense in depth on top of this — every decoded clip, automatic keyframe and freeze-frame pad
  is still checked for the two symptoms before it is trusted.
* `patches/0004-latent-anchors.patch`, a bit-identical generalization of `build_packed_sequence`'s
  `keyframe_anchors`: alongside the existing `"first"`/`"last"` literals (left byte-for-byte
  unchanged), an integer anchor now conditions on an arbitrary pixel-frame index of the rotary
  time axis — the mechanism a latent-tail handoff between scenes needs to anchor mid-clip, not
  just at the first or last frame (`docs/FEASIBILITY-latent-handoff.md` §1.1). Without it, an
  integer anchor fails loudly with `ValueError` from `packing.py`; nothing yet in this fork passes
  one.
* `patches/0005-condition-latent-rows.patch`, a `condition_latent_rows` parameter on
  `MiniMaxH3Pipeline.__call__`: conditioning rows supplied already patchified and normalized, in
  place of encoding keyframe images through the video VAE — the other half of a latent-tail
  handoff (`docs/FEASIBILITY-latent-handoff.md` §1.2). The rows are noised to `t = 0.999` exactly
  like an image keyframe's, since they occupy the same slot of the packed layout; the float16
  round trip inside `_encode_keyframes` is deliberately not applied, as it reproduces a rounding
  that happens inside the reference's VAE encode. Three loud refusals — a shape that does not
  match the request's geometry, a dtype that is not float32, and `images` passed alongside — and
  nothing changes for a run that does not pass the parameter: without it the new kwarg is a
  `TypeError`, and with it the argument is excluded from the run identity so every existing
  checkpoint keeps its digest (`h3_48gb/checkpoint.py`, `_IDENTITY_EXCLUDED`, which also carries
  the standing warning that the digest is blind to the tail until the handoff wave's task 4).

Patching from the outside keeps the two trees separable, but the pin is not optional: this fork
rebinds `FinalLayer.__class__`, binds `inspect.signature(MiniMaxH3Pipeline.__call__)` in three
places, and `docs/DESIGN.md` cites upstream by line number. Later upstream commits are untested
here, and any of those three couplings can break silently on one.

1. **`h3_48gb/adaln.py`** — the mere.run build ships no AdaLN modulation path at all (106 tensors
   absent: 50x `blocks.N.adaln_proj`, `final_layer.adaln_proj`, `time_embedder`); it precomputed
   them into `adaln_cache.safetensors` for one fixed schedule, so this module serves the port's
   `ModulationCache` interface from that file instead of computing it. That used to mean **only
   `num_inference_steps=31`** was servable; `scripts/bake_adaln.py` lifts it by reconstructing the
   table for any grid from the pruned base's folded time curve (87 MB — see "Few-step sampling"
   below). The shipped checkpoint still carries the 31-point table, so without that step —
   any other schedule fails loudly, not silently.
2. **`h3_48gb/text_encoder.py`** — the port builds plain `nn.Linear` layers, so mere.run's 8-bit
   conditioner would load as packed uint32 into bf16 slots and silently produce garbage; this
   quantizes selectively per the converter's manifest (439 modules — `visual.blocks.N.mlp.linear_fc2`
   is deliberately left dense, since mere.run shipped it that way).
3. **`h3_48gb/pipeline.py`** — components load per phase instead of all at once at
   `from_pretrained`.
4. **`h3_48gb/pipeline.py`** (same module) — the text encoder is unloaded immediately after its one
   call, `encode()`. Its output is materialized first (`mx.eval`; MLX is lazy, and an unevaluated
   graph keeps the encoder's weights alive), then `gc.collect()` runs before `mx.clear_cache()`,
   because module trees form reference cycles that plain refcounting does not break.

Alongside the patches, **`convert_sawfwair.py`** turns a user's own Sawfwair (mere.run) download
into the directory layout the port reads — including undoing mere.run's QKV re-layout, from its
`[all-q; all-k; all-v]` slabs back to the port's per-head interleave, across 52 matrices.
Quantization survives the move unchanged: grouping runs along input features, so each row carries
its own scales and biases with it as it moves.

Beyond the original plan, this fork also adds crash-safe checkpoint/resume (`h3_48gb/checkpoint.py`),
in-flight frame previews so a multi-hour render is watchable instead of opaque (`h3_48gb/preview.py`),
and a CLI (`h3_48gb/cli.py`) with `generate`, `resume`, `list`, `status`, `watch`, `doctor` and
`face-refine` subcommands, each able to emit machine-readable JSON for scripting.

The full memory-phase breakdown is in [`docs/MEMORY.md`](docs/MEMORY.md) — which instrument to
trust (three disagree by two orders of magnitude), what each phase holds, why every unload is
preceded by an `mx.eval`, and the allocator cache that quietly held 8 GB until it was bounded. The
QKV/quantization mapping behind the patches above is in [`docs/DESIGN.md`](docs/DESIGN.md).

The 31-step limitation in patch 1 is a property of *this* build, not of H3:
[`docs/FEASIBILITY-turbo-tae.md`](docs/FEASIBILITY-turbo-tae.md) is a study of two upstream
artifacts and finds the full modulation path exists in 8-dimensional curve form inside ComfyUI's
pruned checkpoint — 87 MB, reproducing this fork's own baked table to 2.2e-3 rel-L2. Nothing there
is implemented, and every number in it came from safetensors headers rather than from a run, but it
maps the way out. Known future work, not a permanent ceiling.

## Quickstart

```bash
# 0. Dependencies: an Apple Silicon Mac, Python 3.12, and ffmpeg on PATH.
#    upstream/ is a clone of PipeNetwork/minimax-h3-mlx that this fork patches from the outside;
#    it is not part of this repository, so clone it yourself. Pin the commit: this fork rebinds
#    `FinalLayer.__class__` and binds `inspect.signature(MiniMaxH3Pipeline.__call__)` in three
#    places, and docs/DESIGN.md cites upstream by line number. fcd9e9b is the only revision it
#    has been tested against.
git clone https://github.com/PipeNetwork/minimax-h3-mlx upstream
git -C upstream checkout fcd9e9b
for p in patches/000*.patch; do git -C upstream apply "../$p"; done   # the five patches above

python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/pip install -e .   # installs the `h3` console script

# 1. Get the weights (not included here). Use the mere.run app to download the MiniMax H3 (FL2VA)
#    build: mere.run refuses to *run* it on a 48 GB machine, but downloading is unaffected. It
#    lands at ~/Library/Application Support/MereRun/models/video-minimax-h3-fl2va-mlx/

# 2. Convert that download into the layout this port reads. --check cross-validates the written
#    key sets against the port's own module trees before you trust a multi-hour run to it.
./.venv/bin/python convert_sawfwair.py --out ~/models/h3-converted --check

# 3. Verify the converted checkpoint has everything needed, including the AdaLN cache.
h3 doctor --checkpoint ~/models/h3-converted

# 4. Generate. --steps defaults to 31, the only schedule the shipped AdaLN table covers
#    (see "Few-step sampling" for 8 steps). Previews are on by default (every 5 steps).
#    Wrap long runs in `caffeinate -dimsu`: a render keeps the GPU busy while the machine looks
#    idle, and macOS may sleep in the middle of it (see docs/MEMORY.md).
caffeinate -dimsu h3 generate \
    "a jeweled hummingbird hovering beside a red orchid, cinematic natural light" \
    --checkpoint ~/models/h3-converted --width 1344 --height 768
```

Tests: `pytest -q`. Tests that need MLX or real weights are skipped when those are absent.

## Previews, few-step sampling, CLI

### Previews

A preview used to cost 49.3 s and 8.46 GB, because the real video VAE is causal and chunked: it
cannot decode fewer than 7 latent frames and tiles 28 times at 1344x768 regardless. That is why
previews were opt-in.

This fork decodes them with **TAE** instead — a 9.8 MB 2D decoder, no chunk floor, no tiling — at
**0.125 s and 2.06 GB**, which is 394x faster. Six previews of a 30-step run cost 0.8 s, so they
are on by default. Measurements and the quality comparison are in
[`docs/RESULTS.md`](docs/RESULTS.md).

```bash
# Optional: the TAE weights (9.8 MB). Without them previews fall back to a VAE-free latent
# heat map — coarse but free, and nothing breaks.
mkdir -p ~/models/tae
curl -L -o ~/models/tae/taeh3.safetensors \
  https://huggingface.co/Kijai/MiniMax-H3-TAE/resolve/main/vae_approx/taeh3.safetensors
```

`--preview-decoder` chooses between `tae` (default), `vae` (the real one — exact, and 394x
slower) and `latent` (the heat map, no weights at all). TAE is an approximation for watching
progress; the delivered clip is always decoded by the real VAE.

### Few-step sampling

A 5-second clip at native resolution takes 299 minutes at the shipped 31-step schedule. Eight
steps take a quarter of that at the same quality — **6.6 min against 24.5 at 512x512** — once the
AdaLN table is baked for the shorter grid:

```bash
# 87 MB of folded time curve, pulled out of Comfy-Org's 40 GB pruned base by byte range.
# scripts/fetch_adaln_curve.py does this; the table itself is then local and reusable.
./.venv/bin/python scripts/bake_adaln.py 8 --out ~/models/turbo/adaln_cache_8.safetensors

# Point a checkpoint at it (symlink everything else) and run with --steps 8.
h3 generate "..." --checkpoint ~/models/h3-8step --steps 8
```

Eight steps reproduce the reference frame closely but carry **43% of its motion** — the clip looks
right and moves wrong. The [Turbo LoRA](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)
restores it, applied at **strength 0.45** rather than the 1.0 its author recommends for a bf16
base (at 1.0 it overshoots to 213% and over-sharpens). Numbers and method in
[`docs/RESULTS.md`](docs/RESULTS.md).

`h3 resume` continues an interrupted run from its checkpoint rather than restarting it; `h3 list`
lists finished runs under `--outdir`; `h3 status` reports what is running under an outdir and how
far along it is; `h3 watch` redraws that same status until nothing is running. Every subcommand
except `watch` accepts `--json` for a machine-readable report instead of human-readable text
(the command prints incrementally, and `--json` requires exactly one document on stdout).

Checkpointing is on by default and costs one small file per step. `--checkpoint-dir` moves it,
`--no-checkpoint` turns it off, and `--restart` ignores whatever is on disk and starts from step 0
— which is the way out of `checkpoint_mismatch`, a refusal whose file is named after a digest you
cannot compute by hand.

Both `checkpoint_locked` (the refusal that stops a second writer from opening a checkpoint another
process already has open) and `status`/`watch`'s ability to tell a live run from a dead one rest on
the same mechanism: an `fcntl.flock` held on a companion `.lock` file next to the checkpoint
(`CheckpointStore.acquire_lock`, probed read-only by `runs._writer_alive`). `flock` is a reliable,
kernel-enforced exclusion primitive on a local disk, but on network filesystems (NFS, SMB, most
cloud-sync mounts) it is not: depending on protocol version and mount options, a lock one client
takes may exclude only other processes on that same client, not a second machine writing the same
path at all. Put a checkpoint directory on one of those and both halves of this feature degrade at
once: a second real writer will believe it acquired an exclusive lock when it did not. Both write
atomically (via temporary file, `fsync`, and `os.replace`), but the last to replace the file wins.
Progress can rollback: if the later writer happens to have fewer `completed_steps` than the earlier
one, its state will overwrite the later progress. The atomic write ensures no file corruption, but
the stale process's checkpoint will silently become live. Meanwhile, `status`/`watch` probing that
same unreliable lock keep reporting `unknown` or stale `in_flight` instead of detecting the switch. Keep
`--checkpoint-dir` (and `--outdir`, for the in-progress lock files `watch` also reads) on a
genuinely local filesystem.

Two scripts sit outside the CLI and are used for measurement rather than for generating clips:
`run_bench.py` runs one generation and writes a JSON report of per-phase timings and memory peaks
next to the clip, and `night_queue.sh` drives a series of them overnight (light geometry first,
heaviest last) with `memwatch.sh` sampling process RSS and system memory beside each one. Everything
in "Measured results" below came out of those three.

## Measured results

All runs at the one schedule the baked AdaLN table supports: 31 grid points (30 forwards), sigma
shifts 12.0/3.0. Full per-run detail, the scaling curve and the memory profile are in
[`docs/RESULTS.md`](docs/RESULTS.md).

| Resolution | Requested | Frames (at 24 fps) | Per step | Total | Peak RSS | Swap |
|---|---|---|---|---|---|---|
| 512x512 | 2.4 s | 73 (3.04 s) | 46 s | 24 min | 11.5 GB | none consumed |
| 1344x768 (native) | 5 s | 124 (5.17 s) | 586 s | 299 min | 10.1 GB | none consumed |
| 1344x768 (native) | 10 s | 243 (10.13 s) | 1881 s | ~15.7 h (extrapolated) | not measured | not measured |

"Requested" is the `--duration` asked for; the port snaps it to the latent grid, so the clip that
comes out is slightly longer. **"none consumed" is not "zero swap".** The machine had 9–12 GB of
swap in use throughout, from everything else running on it; what the re-parsed logs show is that
each run *reduced* it, monotonically (12.01 → 9.36 GB over the smoke run, 10.94 → 8.98 GB over five
hours of the native run) rather than adding to it. That is the stronger claim, and it is the one the data
supports. The 10-second run's memory was not measured: its monitor attached to the wrong process.
All of these come from re-parsing the archived CSVs; the originally published figures were
corrupted by a locale bug in `memwatch.sh`, described in [`docs/RESULTS.md`](docs/RESULTS.md).

The 10-second run was measured over 2 steps and then abandoned deliberately, not crashed — see
`docs/RESULTS.md` for why scaling is worse than linear (attention is dense; MiniMax has not
released the sparse-attention implementation, so this is an attention-FLOPs bottleneck, not a
memory one, and quantization does not help it).

## Licences

* **Code: Apache-2.0.** Everything in this repository (`h3_48gb/`, `convert_sawfwair.py`, scripts,
  patches, tests) is under the Apache License 2.0; see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
  The patches in `patches/` are derivative of PipeNetwork/minimax-h3-mlx, also Apache-2.0.
* **Weights: MiniMax H3 Community License.** No weights are included or redistributed; you
  download and convert your own copy and are responsible for complying with the licence of the
  original model: <https://huggingface.co/MiniMaxAI/MiniMax-H3> (read the licence text there; the
  summary below is not legal advice and may be out of date). Its conditions, as understood when
  this snapshot was made:
  * *Territorial exclusions*: use, distribution and display are excluded in the United States, the
    European Union, the United Kingdom and the Republic of Korea.
  * *Commercial threshold*: commercial use above US$20 million in annual revenue needs separate
    permission from MiniMax.
  * *Attribution*: redistribution must carry a copy of the agreement, mark modified files, and
    display "Powered by MiniMax H3".
  Clips and frames the model produces fall under the same terms; this repository therefore ships no model output.

## Credits

* [PipeNetwork/minimax-h3-mlx](https://github.com/PipeNetwork/minimax-h3-mlx) (Apache-2.0): the MLX
  port of H3 this project patches (pinned at `fcd9e9b`).
* [MiniMax](https://huggingface.co/MiniMaxAI/MiniMax-H3): the H3 model.
* mere.run: the Swift runtime whose model build `convert_sawfwair.py` converts.
* [Kijai/MiniMax-H3-TAE](https://huggingface.co/Kijai/MiniMax-H3-TAE) (TAE preview decoder),
  [larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora) (Turbo
  LoRA) and Comfy-Org's pruned base (AdaLN time curve).
