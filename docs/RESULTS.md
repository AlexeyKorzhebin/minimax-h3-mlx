# Measured results

Everything below was run on a MacBook Pro M4 Pro, 48 GB unified memory. The runs through "Memory
profile" below used the checkpoint's shipped AdaLN table — `num_inference_steps=31` (31 grid
points, 30 forwards), sigma shifts 12.0/3.0. That grid is what this checkpoint ships baked, not a
hard ceiling: `scripts/bake_adaln.py` reconstructs the AdaLN table for any grid, and the "Few-step
sampling" section below measures 4, 8 and 16-step runs baked exactly that way. Memory was watched
with `memwatch.sh` (RSS via `ps`, wired/compressed via `vm_stat`,
swap via `sysctl vm.swapusage`) at a **10-second** sample interval for the duration of each run,
driven by `night_queue.sh`, which passes that interval explicitly.

`iogpu.wired_limit_mb` was raised to 45056 (44 GB) for these runs. That is **methodology, not a
prerequisite**: it was set to leave headroom for the heaviest configuration in the planned series
(15 s at native resolution, ~42 GB projected), which was never actually run. Nothing that *was* run
came close — the largest MLX allocation observed at any point is the 28.22 GB text encoder, and the
diffusion phase holds about 11.5 GB — so reproducing the runs in the table does not require raising
the limit. It resets on reboot in any case.

## A correction: how these numbers were re-derived

The figures in this file were published once from a corrupted instrument, and the corrected values
below come from re-parsing the archived CSVs rather than from re-running anything.

`memwatch.sh` used bash `printf '%.2f'`, which honours `LC_NUMERIC`. Under the author's
`LANG=ru_RU.UTF-8` every `%.2f` emitted a comma decimal, so each row of a nominally 6-column CSV
arrived with **11 comma-separated fields** — and bash's `printf` additionally *rejected* `bc`'s
dot-decimal output as malformed, writing `0,00` for `wired_gb` and `compressed_gb` in all 2,490
samples of all three runs. `night_queue.sh` then read column 2 as "peak RSS" (the integer part of
the RSS) and column 5 as "swap" (the decimal part of the always-zero wired figure), which is why
every run reported a suspiciously round RSS and exactly `0.0` swap.

The fix is `export LC_ALL=C` at the top of `memwatch.sh` (and of `night_queue.sh`, whose summary
`awk`s format numbers too); the CSV is now 6 fields under both `C` and `ru_RU.UTF-8`, verified in
both. Two consequences worth naming:

* **`wired_gb` and `compressed_gb` recorded nothing at all** for the archived runs. Those two
  columns are unrecoverable — there is no partial value to salvage, only zeros — so nothing in this
  file is based on them.
* Every RSS and swap figure below was recovered by splitting the 11 fields back into 6.

## Per-run results

| Resolution | Requested | Frames (at 24 fps) | Per step | Total | Peak RSS | Swap |
|---|---|---|---|---|---|---|
| 512x512 | 2.4 s | 73 (3.04 s) | 46.1 s | 24.5 min | 11.54 GB | none consumed; 12.01 → 9.36 GB |
| 1344x768 (native) | 5 s | 124 (5.17 s) | 585.8 s | 299.3 min | 10.14 GB | none consumed; 10.94 → 8.98 GB |
| 1344x768 (native) | 10 s | 243 (10.13 s) | 1881 s (2 steps measured) | ~15.7 h (extrapolated) | not measured | not measured |

"Requested" is the `--duration` passed on the command line. The port snaps it to the latent grid,
so the clip that comes out is longer than the one asked for — 2.4 s of request became 73 frames,
which is 3.04 s at 24 fps.

The 10-second run has no memory figures because its monitor attached to the wrong process:
`h3-mem-native10.csv` records an RSS of 0.09 GB at t=221 s, dropping to 0.04 GB at t=321 s, while the run's own log shows
it still working through steps of 1865 s and 1881 s. Nothing can be recovered from that trace, and
it is not reported. (The cause — `night_queue.sh` slept 10 s and then `pgrep`'d for a command line
— is fixed; the script now attaches to `$!` immediately.)

The 10-second run at native resolution was measured for 2 steps and then **abandoned
deliberately** — at 1881 s/step the full 30-forward run projects to roughly 15.7 hours, which was
not worth running to completion just to confirm a number the first 2 steps already established.
It was a stop, not a crash: no error, no sign the process would not have completed if left running.

## Scaling bends upward, but not everywhere

Per-step cost across the three geometries: **46 s -> 586 s -> 1881 s**.

| Step up | Tokens | Per-step cost | Exponent |
|---|---|---|---|
| 512x512/2.4s -> 1344x768/5s | x6.7 (3.9x pixels, 1.7x frames) | **x12.7** | tokens^1.34 |
| 1344x768/5s -> 1344x768/10s | x2.0 (frames only) | **x3.2** | tokens^1.68 |

Two corrections to what this section used to say. It reported the first step up as "roughly 3.2x",
which is wrong — 586/46 is 12.7, and the 3.2 belongs to the second row only. And it called the
curve "worse than a naive quadratic": both step ups are in fact *sub*-quadratic, at exponents 1.34
and 1.68 over token count. Growth is still faster than linear, and the second step up — where only
the sequence gets longer, with geometry fixed — is the steeper of the two, which is what the
mechanism predicts:

- **Attention is dense.** The whole packed sequence (text + keyframe conditions + audio rows +
  video rows) attends over itself with plain full self-attention — no cross-attention, no sparsity,
  no per-modality block weights (see `upstream/README.md`'s architecture table). MiniMax has not
  released a sparse-attention implementation for H3, so there is no cheaper path available to this
  port even in principle.
- **The bottleneck is attention FLOPs, not memory.** This is why quantization — which this fork
  already applies to the text encoder, and which helps residency — does not help throughput here.
  Quantizing the DiT's own weights would shrink what is resident, not the O(sequence_length^2)
  attention cost that dominates wall time at native resolution and longer clips.

Anyone tuning clip length or resolution against a time budget should expect the cost curve to bend
upward faster than the pixel or frame count alone would suggest, for exactly this reason: doubling
a clip's length roughly triples the cost of every step.

## Memory profile

### Three numbers, two instruments

The honest shape of this fork's memory behaviour is not one peak but three, and they were not all
measured the same way. Stating that plainly matters more than a tidy headline:

| Phase | Figure | Where it comes from |
|---|---|---|
| All four components resident | **~55 GB** | 45.9 GB of weights + 9.3 GB of activations (below) |
| Text-encoding phase, ~10 s long | **28.2 GB** | MLX's `get_active_memory`, printed by the run itself |
| Diffusion phase, the other 99% of the wall clock | **11.54 / 10.14 GB** | process RSS via `ps`, sampled every 10 s |

**The 45.9 GB** is derived from this fork's own run log (`h3-gen-native5.log`), which prints each
component's cost as MLX's allocator sees it at load: text encoder 28.22 + transformer 11.34 + video
VAE 5.21 + audio VAE 0.61 = 45.38 GB, plus the 0.56 GB AdaLN cache. **The 9.3 GB** is not this
fork's measurement — it is upstream's own, from its README's sequence-length table for
1344x768 / 5 s on an M3 Ultra. Their sum, 55.2 GB, is where the "55.5 GB" this project has quoted
since its planning notes comes from; the weights half of it is now re-derived here, the activation
half is still upstream's number and is labelled as such.

**The 28.2 GB is not an RSS measurement, and the RSS trace never saw that phase.** MLX prints
`loaded text encoder: +28.22 GB in 8.3s`, and that is what the figure is: the allocator's own
accounting for the encoder's weights. Meanwhile `night_queue.sh` slept 10 s before attaching
`memwatch.sh` — and the encoder finished loading in 8.3 s and was unloaded immediately after — so
the first RSS sample of every archived run was taken after the encoding phase was already over.
There is no measurement of process RSS during text encoding in any of the three runs. The
instrumentation gap is now narrowed (the monitor attaches to `$!` immediately), but since `night_queue.sh`
still uses a 10-second sample interval and the encoder finishes in 8.3 s, catching that phase remains
probabilistic. Closing it would require either a shorter sampling interval or attaching the monitor before the script launches the process; retroactively, it does not produce a sample anyway, and the runs were not repeated: a native step costs 586 s.

So: 28.2 GB and 11.5 GB come from different tools and were never observed together. What can be
said with both in hand is that the encoder's 28.22 GB is *allocated and released* before the
diffusion loop starts, and that during the loop the process's RSS is flat at 9.04 GB for 1,751
consecutive samples with a 10.14 GB peak. What cannot be said is that any instrument watched the
transition.

### Why the diffusion peak is what it is

| Phase | Resident | Notes |
|---|---|---|
| Text encoding | text encoder, Q8, 28.2 GB | Runs once, at the very start; unloaded immediately after (`LazyTextEncoder.encode`, `h3_48gb/pipeline.py`) |
| Diffusion | DiT + LoRA + AdaLN cache + activations | Text encoder is gone; the video VAE unloaded after keyframe encoding (`_release_vae_after`) and does not reload until decode |
| Video decode | video VAE + activations | DiT is unloaded first (`_decode_video`), so this phase never holds the transformer alongside the VAE |
| Audio decode | audio VAE | The video VAE is unloaded first (`_decode_audio`); the two VAEs are never resident together |

The unload is why the diffusion phase costs 10–11.5 GB rather than that plus the encoder's 28.2 GB.
It does **not** explain the 10.14 GB figure on its own: the DiT alone loads at 11.34 GB, more than
the peak RSS ever reached during the native run. Two things account for that, and neither is the
unload:

- **RSS is not the allocator's view.** MLX's `+11.34 GB` counts the buffers it allocated; `ps`
  reports pages this process has resident. Weights read from a memory-mapped safetensors file are
  file-backed and clean, so the kernel can evict them from the resident set under pressure and
  fault them back on use without any swap traffic at all. RSS legitimately sits below allocated
  weight for a memory-mapped, mostly-read-only working set.
- **Which is why RSS is the sanity check and not the measurement**, exactly as
  `h3_48gb/memory.py` says in its module docstring. The claim RSS supports is "this process is not
  thrashing and is not driving the machine into swap", which the traces do support. The claim it
  does not support is "the model's working set is 10.14 GB".

Confirming the unload actually reclaims memory (rather than merely dropping a Python name) took two
steps, both in `h3_48gb/pipeline.py` and `h3_48gb/memory.py`:

- **`mx.eval(prompt_embeds)` before anything else.** MLX is lazy; a computation graph that still
  references the encoder's parameters keeps all 28.2 GB alive no matter what else is deleted, until
  something forces evaluation.
- **`gc.collect()` before `mx.clear_cache()`.** Module trees and MLX graphs form reference cycles,
  so plain refcounting does not reclaim them — measured directly: without the `gc.collect()` call,
  the encoder stayed resident straight through the diffusion loop. Only after the cyclic collector
  runs does clearing MLX's allocator cache actually give memory back, since it can only release
  buffers nothing still refers to.

### Swap

This project previously claimed **zero swap**. That was false, and an artifact of the locale bug
above: the column being read as swap was the always-zero decimal half of the `wired_gb` field.

What the re-parsed CSVs actually show is better than the claim it replaces. The machine had
**8.98–12.01 GB of swap in use throughout**, from everything else running on it — `sysctl
vm.swapusage` is machine-wide and has no per-process view, so none of it is attributable to these
runs either way. The signal is the direction, and in both measured runs it points down:

| Run | Swap at first sample | At the end of the diffusion phase | Change |
|---|---|---|---|
| smoke, 512x512, 24 min | 12.01 GB (t=0) | 9.36 GB (t=1454 s) | **-2.65 GB** |
| native5, 1344x768, 5 h | 10.94 GB (t=0) | 8.98 GB (t=17,570 s) | **-1.96 GB** |

Swap in use fell **monotonically** across both diffusion phases — checked sample by sample, not
just endpoint to endpoint: of the smoke run's 147 samples, not one is higher than the sample before
it until the process has already exited; of the native5 run's 1,791 samples, the first increase of
any kind is at t=17,581 s. A run that needed more memory than the machine had would push this
number up; instead the machine **reclaimed** swap while generating. That is the claim the data
supports: *these runs did not consume swap, and the machine gave some back while they ran.*

The caveat, so the table is not read as more than it is. Swap does rise at the very end of the
native5 trace, from 8.98 GB to 22.75 GB across its last 38 samples. That rise begins at t=17,581 s
— and 30 steps at 585.8 s each is 17,574 s, so it starts at the moment the final diffusion step
completes and the video VAE decode, the raw `.npz` write and the ffmpeg encode begin. The process's
own RSS is 1.28 GB at that moment, dropping to ~0.9 GB as the decode phase progresses. It describes the decode-and-teardown tail (and whatever
else the overnight queue was starting), not the phase the table is about. Swap during the diffusion
phase itself never exceeded its opening 10.94 GB.

`sysctl vm.swapusage` is machine-wide, so none of these movements can be attributed to the run with
certainty in either direction — which is exactly why the claim above is about direction and bound,
not attribution.

mere.run's own admission control demands 32 GB free before it will start a job at all; this fork's
peak footprint fits inside that headroom.

## Previews

`h3_48gb/preview.py` decodes one frame from the current, partially-denoised latent every N steps,
so a run measured in hours is watchable rather than opaque. It is on by default —
`--preview-every` defaults to 5 and `--preview-decoder` to `tae` (see "TAE previews" below for
why); `--preview-every 0` disables it. Two constraints, both measured rather than assumed:

- **One preview costs 49.3 s and 8.46 GB peak** at native (1344x768) resolution. **Measured with
  correctly-shaped random weights, not with the real checkpoint** — see `h3_48gb/preview.py`, which
  makes the same disclosure. The argument for why that is still a fair measurement is that VAE
  decode cost is a function of shapes and dtypes, not of weight values: the same 28 tiles walk the
  same 36-layer decoder either way. The argument against is that it was never confirmed against the
  real weights, so treat it as an estimate with a sound basis rather than as a measurement of the
  shipped path. The video VAE is
  loaded on demand for the decode and unloaded again immediately after — it is not kept resident for
  the whole diffusion loop, which would stack its ~5.2 GB on top of the DiT's own peak activation
  moment.
- **The VAE cannot decode fewer than 7 latent frames, but it does not have to start at frame 0 to
  do it correctly.** `VideoVAE.decode` (`upstream/minimax_h3_mlx/video_vae.py`) is a chunked
  decoder that refuses fewer than `2 * chunk_tokens - token_drop` latent frames outright — for the
  released config (`clip_length=17`, `token_drop=3`) that floor is 7 latent frames, about 22 pixel
  frames (~1 s at 24 fps), and there is no cheaper *correct* decode than that. What is causal is
  the **encoder** (`CausalConv3d`); the decoder is a non-causal 36-layer ViT with full attention
  inside each 7-latent-frame window, so which window a preview decodes does not matter for
  correctness. Concretely, each decode window covers latent frames `z[5i : 5i+7]` and emits 17
  pixel frames starting at pixel frame `17i`, cross-faded over 5 frames with its neighbouring
  window. Measured directly on four finished clips: at the seams that coincide with nothing else —
  pixel frames 17, 51 and 68 — the frame-to-frame pixel discontinuity is 1.00x the local median,
  which is to say invisible. Frame 34 is 5-11x, but frame 34 is *both* a seam and the Shot 2 cut,
  so it cannot tell the two apart; the other three seams are what carry the point. The minimal
  prefix is what a preview decodes because it is the window that exists first during denoising,
  not because any other window would be wrong.

## Honest limits

- **Only the loaded AdaLN table's own grid works, and a run must pass `--adaln-cache` to use any
  grid but the checkpoint's shipped one.** `h3_48gb/adaln.py` serves whichever table is loaded —
  the checkpoint's shipped 31-point grid (30 forwards) at sigma shifts 12.0/3.0 by default — and
  raises `ScheduleMismatch` before the first forward for any other `--steps`, by design, with no
  fallback to a nearest-value lookup. This used to be a hard limitation of *this build*, full stop:
  [`FEASIBILITY-turbo-tae.md`](FEASIBILITY-turbo-tae.md) located the full modulation path in
  8-dimensional curve form inside ComfyUI's pruned checkpoint (87 MB, reproducing this fork's baked
  table to 2.2e-3 rel-L2), and `scripts/bake_adaln.py` now reconstructs a table for any grid from
  it — see "Few-step sampling" below, where 4, 8 and 16-step tables are baked and run this way.
  What remains true: a single run is still limited to whichever one grid its loaded table was baked
  for.
- **A 10-second clip at native resolution is a ~15.7-hour commitment**, extrapolated from 2
  measured steps at 1881 s each. Nothing about that run failed; it simply was not run to completion.
  `h3_48gb/checkpoint.py` makes this tractable in practice — a crash or an intentional stop costs
  only the steps since the last checkpoint, not the whole run — but the wall-clock cost itself is
  real and this fork does nothing to reduce it.
- **No sparse attention.** MiniMax has not released one for H3, and full dense attention over the
  packed sequence is what makes the scaling above worse than linear. Quantization, which this fork
  already uses for the text encoder, addresses memory residency, not this cost, and would not help
  even if applied to the DiT.
- **Test suite: 115 tests**, none of which run a real generation (a single native step costs 586 s).
  Reproduce with:

  ```bash
  ./.venv/bin/python -m pytest -q
  ```

  This file previously documented `pytest tests/ test_preview.py -q` and claimed 71 tests. That
  command collected 76, and hid the four root-level files — `test_adaln_indexing.py`,
  `test_lazy_pipeline.py`, `test_qkv_permutation.py`, `test_text_encoder_quant.py` — which are 27
  more tests and which cover three of the four core patches (76 + 27 = 103). The remaining 12 tests
  were added in this precision-sweep commit to verify measured values against the actual data. A bare `pytest` used to fail instead,
  with five collection errors from `upstream/tests/*` importing torch; `norecursedirs` in
  `pyproject.toml` excludes the vendored clone, so the plain command is now the whole suite.

## Keyframe conditioning: verified, and it works

`--image`/`--end-image` (added on `feat/image-to-video`) parse, validate, and bind into checkpoint
identity — that much is covered by unit tests. Whether a keyframe actually **conditions** the clip,
as opposed to being silently accepted and ignored, has never been checked against the real
checkpoint, because every existing test replaces the pipeline with a fake
(`_default_pipeline_factory` is monkeypatched in `tests/test_cli.py` and friends). `scripts/verify_i2v.py`
was written to close that gap: generate the same prompt and seed twice at 512x512 — once with a
keyframe, once without — and compare each clip's first frame against the image by PSNR and
correlation, with the unconditioned control as the baseline that makes the comparison meaningful.
It has now run, twice, and the answer is yes — see **The measurement** below. Getting there took
three blockers, each hidden behind the one before it, and each costing a 25-minute run to surface.
All three are recorded here so the history is legible instead of overwritten.

**Blocker 1 — `torchvision`, fixed.** `h3_48gb/image_processor.py`'s `TorchFreeProcessor` (landed on
this branch, verified against real `transformers` output in `tests/fixtures/processor/`) now stands
in for the composite `AutoProcessor`, so `upstream/minimax_h3_mlx/text_encoder.py`'s
`build_request()` no longer touches `transformers.AutoProcessor.from_pretrained` for an image
request. Confirmed directly: a conditioned run now gets past `build_request()` — token ids, tags,
`pixel_values` and `image_grid_thw` all come back — and reaches `self.vision(...)`, the Qwen3-VL
vision tower's own forward pass.

**Blocker 2 — the vision tower's conv weight layout, fixed.** The next failure was inside the vision
tower itself, in `mx.conv3d`:

```
ValueError: [conv] Expect the input channels in the input and weight array to match but got
shapes - input: (4032,2,16,16,3) and weight: (1152,3,2,16,16)
```

Confirmed by inspecting the checkpoint directly: `~/models/h3-converted/text_encoder/model-00014-of-00014.safetensors`
stores `model.visual.patch_embed.proj.weight` as `(1152, 3, 2, 16, 16)` —
`(out_channels, in_channels, kD, kH, kW)`, PyTorch/HF's conv layout — while `mlx.nn.Conv3d` expects
channels-last, `(out_channels, kD, kH, kW, in_channels)`, exactly as its own constructor builds
`self.weight` (`mlx/nn/layers/convolution.py`). `upstream/minimax_h3_mlx/load.py`'s VideoVAE loader
already applies this exact transpose to every 5-D conv weight it loads
(`tensor.transpose(0, 2, 3, 4, 1)`, with a comment explaining why); `upstream/minimax_h3_mlx/text_encoder.py`'s
`_load_weights()` never got the same treatment for the vision tower's one convolution (everything
else in `VisionModel` is attention and linear layers, layout-agnostic). No test or run before this
task ever asked the real checkpoint to encode an image, so this was never exercised.

Fixed in this fork's own loader — `h3_48gb/text_encoder.py`'s `QuantizedTextEncoder._load_weights()`
now applies `to_mlx_conv3d_layout()` to `patch_embed.proj.weight` specifically, at load time, before
the module tree is used. `upstream/` is unmodified. Two things were added, deliberately, because a
wrong permutation here would not raise — it would silently scramble the patch embedding and show up
only as a worse clip, hours later, with the real checkpoint's kernel shape `(D, H, W) = (2, 16, 16)`
giving a wrong H/W-swapping permutation the exact same output *shape* as the right one:

- `test_vision_patch_embed_layout.py` pins values, not just shape — an independently-computed
  per-index reference (plain nested loops, not another call to `.transpose()`), plus a case with
  `H == W` matching the real checkpoint's collision, and a case that confirms a plausible wrong
  permutation is actually distinguishable from the right one (otherwise the value check would prove
  nothing). 7 tests, all passing.
- Verified against the real checkpoint directly (not via the CLI, to avoid paying for a full
  diffusion run to check one loader): `QuantizedTextEncoder(...).encode(prompt, [image])` now runs
  the real image through `patch_embed` without the shape error — strictly further than blocker 2
  allowed.

**Blocker 3 — fixed, as a patch against `upstream/`.** Past the fixed conv layer, `encode()` failed
one call later, still inside `upstream/minimax_h3_mlx/text_encoder.py`:

```
ValueError: [broadcast_shapes] Shapes (1,1026,1) and (1,1008,5120) cannot be broadcast.
```

at

```python
inputs_embeds = mx.where(image_mask[..., None], hidden.astype(inputs_embeds.dtype)[None], inputs_embeds)
```

`image_mask` has one entry per token in the *full* prompt (label + `<|vision_start|>` + one
`<|image_pad|>` per merged patch + `<|vision_end|>` + the text prompt itself — 1026 tokens for this
run); `hidden`, the vision tower's output, has one row per merged image patch only (1008 for this
run's 1344x768 keyframe). `mx.where` requires elementwise-broadcastable shapes, so this only works
when the whole prompt is image tokens with nothing else — never true in practice, since there is
always at least a `<Picture i>:` label and the prompt text. Confirmed this is a real bug rather than
a misuse: `mlx_vlm`'s own `qwen3_vl.py` (`.venv/lib/python3.12/site-packages/mlx_vlm/models/qwen3_vl/qwen3_vl.py`)
implements the equivalent operation correctly, as `masked_scatter()` — flatten, index the True
positions, assign, reshape — specifically because a boolean-masked `mx.where` can't do this.
`upstream/minimax_h3_mlx/text_encoder.py`'s hand-rolled version does not use it. The processor's own
`pixel_values`/`image_grid_thw` were checked directly and are internally consistent (4032 raw
patches, 1008 after the checkpoint's `merge_size=2`) — the mismatch is between the image-token count
and the full sequence length, not a processor bug.

This one is a control-flow bug in the middle of `encode()`, not a weight layout fixable before the
call, so no loader-side workaround exists. Copying `encode()` into a subclass to change one line
would have meant carrying 40 lines of upstream logic that then drift apart silently. It is carried
as `patches/0001-keyframe-masked-scatter.patch` instead — `upstream/` is gitignored, so an edit made
there would exist on one machine and nowhere else. `README.md`'s setup applies it; `--image` without
it is refused up front with `upstream_patch_missing`, before any weight loads, rather than crashing
28.2 GB into a run. Text-only runs never reach the line.

### Five silent divergences, found by reading against upstream's own reference

With the run finally completing, the remaining defects were the dangerous kind: none of them raise.
They were found by comparing this fork against `upstream/reference/diffusers/`, which ships in the
vendored checkout, and each is now fixed and pinned by a test.

| What was wrong | How it showed | Where |
|---|---|---|
| Seed 42 taken once per request, not once per keyframe | Two byte-identical keyframes encoded 0.83 apart; the reference makes them identical | `h3_48gb/pipeline.py` |
| Our lazy VAE construction consumed the RNG stream between `seed(42)` and the draw | A cold and a warm run of the same request differed by 0.87 | `h3_48gb/pipeline.py` |
| The vision tower saw the raw image, the VAE the canvas version | Vision-token count changes, which shifts the rotary clock of every media row | `h3_48gb/pipeline.py` |
| The canvas ignored the keyframe's aspect | A portrait photo was stretched into 16:9, since keyframe 0 is stretched, not fitted | `h3_48gb/cli.py` |
| `preprocessor_config.json` was synthesized by the converter | Values turned out correct — verified against MiniMaxAI/MiniMax-H3 — but only in the one spelling mlx-vlm reads | `convert_sawfwair.py` |

That last one is worth stating plainly, because the obvious "fix" is a regression: the official file
writes the pixel budget as `size: {shortest_edge, longest_edge}`, a key mlx-vlm's processor does not
have. Handed the official config verbatim, it silently keeps Qwen2-VL's defaults and caps a keyframe
at 1,003,520 pixels instead of 16,777,216 — a 4K keyframe would then yield 943 image tokens where
the released model produces 8160, with no exception raised.

### A canary, so the next blocker costs minutes

`scripts/canary_i2v.py` walks the entire i2v path — encoder, keyframe encode, packing, forward,
decode — with the denoising loop cut to two forwards. **3 minutes instead of 25, and 8 seconds to
reach a failure inside the encoder.** It calls the real `__call__` rather than reimplementing the
loop, because a canary that reimplements the pipeline verifies the copy. Only the loop's timestep
list is truncated: the schedule and the AdaLN table are built in full, since `check_schedule`
compares the sigma grid elementwise and refuses anything else. `--no-image` runs the identical path
without a keyframe, which is what distinguishes "the i2v path is broken" from "the canary is broken".

### The measurement

Both pairs are the same prompt and seed run twice, once with a keyframe and once without. The
control is the point: without it, a high score would only show that the clip agrees with the prompt.

| Keyframe | Canvas | Conditioned | Control | Conditioned corr | Control corr |
|---|---|---|---|---|---|
| Synthetic checkerboard 512x512 | 512x512 | **34.81 dB** | 10.35 dB | **0.999** | 0.041 |
| Photograph 1536x1024 (3:2) | 576x384 | **26.30 dB** | 9.81 dB | **0.979** | 0.314 |

The two runs are deliberately different tests. In the first, the prompt (*"a red vintage car parked
on a wet street at night"*) **contradicts** the keyframe, so a first frame that reproduces the
checkerboard can only come from the conditioning — which is why the control scores 0.041. In the
second the prompt agrees with the image, which is the harder case: the model would draw a dragon
over mountains regardless, and the control's 0.314 correlation is exactly that shared composition.
The conditioned run still clears it by 16.5 dB.

The photograph scores lower than the checkerboard for a reason that is not a defect: a keyframe
makes a round trip through the VAE, and flat colour blocks survive that far better than scales,
cloud and foliage.

The second pair also ran on the fixed code and on a 3:2 source — the case that, before the canvas
fix, would have been stretched into 16:9 without comment.


## How this fork compares to another Mac port

A ComfyUI-based port (`Bambushu/minimax-h3-mac`) published per-step figures for the same model on
comparable geometry, which is a rare chance to check whether this fork leaves performance on the
table. Measured with the canary on that port's own geometry:

| | this fork | Bambushu port |
|---|---|---|
| Canvas / duration | 1344x768, 3 s (73 frames) | 768x1376, 3 s |
| Packed sequence | 22,434 rows | "about 22k tokens" |
| Chip | **M4 Pro**, 20 GPU cores | **M5 Pro** |
| DiT precision | 4-bit | int8 |
| **Per step** | **262 s** | **131.6 s** |

Both figures are sampling only, excluding load and decode. Ours is the per-step time the pipeline
itself prints, and the two forwards measured 262.1 s and 261.6 s — within 0.2% of each other, so
neither carries hidden warm-up or load cost. The 2x gap is a chip generation and a
quantization choice, not headroom in this code: M5's GPU carries neural accelerators per core, and
int8 is a different trade than the 4-bit weights this fork uses to fit 48 GB at all. Attention here
already runs through `mx.fast.scaled_dot_product_attention`, so there is no obvious slow path to
reclaim.

One thread is worth pulling, though: that port's *whole-clip* time for a 5-second native render is
116.8 minutes against this fork's 299 — a 2.56x gap, wider than the 1.99x per step. Some of it is
outside sampling. This fork's video decode at 1344x768 measures 208 s, which is where to look first.


## TAE previews: 394x faster, and now on by default

An in-flight preview used to cost **49.3 s and 8.46 GB** — not because decoding is expensive, but
because the real video VAE's decoder is chunked: it cannot decode fewer than 7 latent frames, and
it tiles 28 times at 1344x768 no matter how little is asked of it. Every preview therefore decoded
about a second of video to show one frame, loading a 5.21 GB module to do it.

`Kijai/MiniMax-H3-TAE` is a 9.8 MB 2D decoder — no temporal state, no chunk floor, no tiling.
Ported in `h3_48gb/tae.py`. Measured on the same geometry as the figure above:

| | time | peak memory |
|---|---|---|
| Real video VAE | 49.3 s | 8.46 GB |
| **TAE** | **0.125 s** | **2.06 GB** |
| | **394x faster** | **4.1x smaller** |

Three consecutive calls measured 0.195 s, 0.126 s and 0.125 s — the first carries compilation, so
0.125 s is the steady-state figure.

### Which normalization, and why one metric was not enough

TAE was trained by a third party, and nothing said whether it expects the latent as the sampler
holds it (normalized) or after `latents * std + mean`. The spec required this be settled by
measurement. It was, and **a single metric would have settled it backwards**:

| Compared against | normalized | denormalized |
|---|---|---|
| the real VAE's decode (PSNR) | 15.11 dB | **17.50 dB** |
| the real VAE's decode (correlation) | **0.927** | 0.883 |
| the real VAE's decode (gradient correlation, i.e. structure) | **0.508** | 0.457 |
| **the source image (PSNR)** | **21.78 dB** | 16.15 dB |
| **the source image (correlation)** | **0.939** | 0.847 |

The denormalized form wins on PSNR against the VAE and loses everywhere else: it carries a colour
shift that happens to land near the reference's tone, which intensity-based PSNR rewards and
structure does not.

What settles it is the second reference. The latent under test was produced by **encoding a known
image**, so that image is ground truth independent of the VAE — and against it the normalized form
scores 21.78 dB / 0.939 where the real VAE itself manages 14.72 / 0.941. The small decoder lands
closer to the original than the VAE does, because it skips the encoder round trip and the tiling.

`TAE_EXPECTS_NORMALIZED = True`.

Two ports were wrong before this was measured, and neither raised anything:

- the decoder carried TAESD's `+ 0.5` output offset, which belongs to Stable Diffusion's latent
  space. With it, frames came out at mean brightness 225/255 and scored 5 dB. Without it, 17.5 dB.
- eight separate mutations — swapping decoder slots, swapping convolutions inside a block, breaking
  the upsample, dropping either ReLU, dropping the input clamp — passed a suite that only checked
  shapes. `tests/fixtures/tae/golden_frame.npz` pins the output by value; every one of those
  mutations now fails.

### Quality, judged by eye

Decoding a real latent: the dragon is fully recognisable — pose, wings, colour, the mountains
below, the composition. Detail is smeared and textures go waxy. That is exactly the trade a preview
wants, and it clears the bar TAE's own author set ("beats latent2rgb") by a wide margin.

### Consequence: previews are on by default

`--preview-every` now defaults to 5 and `--preview-decoder` to `tae`. Six previews of a 30-step run
cost **0.8 s** with TAE against 295.8 s with the real VAE — which is why previews were opt-in
before and why that no longer makes sense. The real VAE stays one flag away for a preview that must
be exact; TAE is an approximation for watching progress, never for the delivered clip. Without the
weights file the default degrades to the VAE-free latent heat map, so it costs nothing to a reader
who never downloads it.


## What `--image` + `--end-image` interpolates, and what it does not

Both keyframes anchor their ends firmly. Measured on two 896x1152 stills at a 448x576 canvas,
with the cross-comparison as the control:

| | first frame | last frame |
|---|---|---|
| against the **first** keyframe | **27.55 dB / 0.987** | 13.41 dB / 0.660 |
| against the **last** keyframe | 13.63 dB / 0.661 | **27.78 dB / 0.986** |

A 14 dB gap each way, so neither anchor is being ignored. What happens *between* them depends
entirely on whether the two frames belong to the same scene.

**Unrelated stills produce a cut, not a transition.** Two different scenes — figures in snow, and
an empty courtyard — gave a clip that holds the first still for half its length, jumps in a single
frame, and holds the second:

```
0.9 s -> 1.0 s   mean pixel change  0.1     (held)
1.0 s -> 1.1 s   mean pixel change 32.6     (the cut)
1.1 s -> 1.2 s   mean pixel change  0.2     (held)
```

**Frames from one scene interpolate properly.** The control: first and last frame of an existing
dragon clip, same scene with the camera pushed in, correlation 0.554 between them. The result
changes continuously — 21 to 43 mean pixel change between every pair of sampled frames, no jump
anywhere — and the midpoint is a coherent frame of that scene at an intermediate camera distance.

So `fl2va` interpolates the ends of one shot. It does not cut between shots, and asking it to
leaves it nothing to interpolate through: there is no continuous path from two people standing in
snow to a courtyard without them. Pick keyframes that are the start and end of a single move.

Worth noting how this was nearly missed: the end-point measurements above are excellent, and they
were all that was checked at first. The clip between them was never looked at until a viewer
watched it. Metrics aimed at the ends cannot see the middle.


## Few-step sampling: 3.7x faster, whole-frame metrics unchanged on the scenes measured

This fork shipped with a hard limit of `--steps 31`, recorded everywhere in these docs as
immovable: mere.run's build dropped the modulation path (`time_embedder` + 50 `adaln_proj`, 13B
parameters) and shipped a table baked for that one grid. Nothing else could be evaluated.

Comfy-Org's pruned base keeps that path folded into an `adaln_t_table` of 1025x8 plus per-block
projections — **87.3 MB, pulled out of a 40 GB file by byte range**. `scripts/bake_adaln.py`
reconstructs the table for any grid from it, reproducing the shipped 31-step table to
4.3e-3..7.7e-3 relative error, two orders below the 0.165 the Q4 weights already carry.

| | wall clock | motion vs reference | frame sharpness |
|---|---|---|---|
| 31 steps (reference) | 24.5 min | 100% | 2.23 |
| 8 steps, no LoRA | **6.6 min** | 43% | 2.11 |
| 8 steps + Turbo LoRA at 0.45 | **6.6 min** | **117%** | 3.10 |
| 8 steps + Turbo LoRA at 1.0 | 6.6 min | 213% | 5.61 |

512x512, 2.4 s, same prompt and seed throughout — one scene. See "Strength 0.45 across scenes"
below for how far the motion figure generalizes; the sharpness figures were not re-checked on
other scenes.

### What the Turbo LoRA actually does here

Not what its name suggests. It was fetched to enable 4-step sampling; the step count turned out to
be free once the table could be baked. What the LoRA supplies is **motion**.

Eight steps alone reproduce the reference frame almost exactly — sharpness 2.11 against 2.23,
correlation 0.950 — while carrying **43% of its motion**. The clip looks right and moves wrong: a
dragon that beats its wings half as hard. Frame-level metrics cannot see this at all, and did not:
it was caught by a human watching the clip.

The LoRA restores it, and overshoots at the strength its author recommends. At 1.0 the motion runs
to 213% of the reference and the frame goes visibly over-sharp. At **0.45** the motion lands at
117% and the frame correlation is 0.952 — marginally better than without it. The author's 1.0 is
tuned for a bf16 base and his own sampler; ours is 4-bit with a different one.

`h3_48gb/turbo.py` applies the backbone half at run time (a merge into 4-bit storage would
requantize the update away) and `scripts/bake_adaln.py` folds the AdaLN half into the table, where
merging is legitimate because the table is bf16.

### The bug that cost a day, and how it hid

An earlier version of the bake evaluated the output layer on the video clock only, then sliced its
10752 outputs into three chunks of 3584. The correct shape is three timestep variants each
carrying the full shift+scale vector.

**It raised nothing.** The output layer indexes that table by timestep alone, so a wrong-width row
reads as a valid one. Every few-step experiment that day ran on a corrupted final modulation, and
the results led to a confident, wrong conclusion — that the Turbo LoRA was fundamentally
incompatible with a 4-bit base, backed by eight separate ruled-out hypotheses.

Two things found it, neither of them the eight hypotheses:

- an independent review (codex) read the baker and spotted the shape error directly;
- the verification that should have caught it compared the bake against the shipped table **block
  by block** and never checked `final_modulations` at all. Checking part of a structure and
  concluding the whole is sound is the recurring failure in this project.

`tests/test_bake_adaln.py` now compares every tensor the loader reads, names `final_modulations`
explicitly among the parametrized cases, and asserts the three variants are not slices of one
evaluation. All three assertions go red against the old code.

### How few steps is too few

| steps | wall clock | motion vs reference | sharpness | verdict |
|---|---|---|---|---|
| 31 (reference) | 24.5 min | 100% | 2.23 | |
| 8 + LoRA 0.45 | 6.8 min | 117% | 3.10 | **best whole-frame match, this scene** |
| 4 + LoRA 0.45 | 3.5 min | 86% | 3.49 | visibly short on detail |

Four steps hold their motion and cost half of eight, but detail suffers in a way a viewer notices
immediately, on this scene. That matches what users of this LoRA report independently ("8 steps,
quality is much better"), which is corroborating, not confirming — it is the same whole-frame-metric
generalization this section is qualifying. **Eight looks like the better trade of the two on the
scenes measured here**; four is for iterating on a prompt, not for output.

### Strength 0.45 across scenes

Tuned on one smooth flying shot, then checked against 31-step references on three more:

| scene | motion vs its own reference |
|---|---|
| dragon in flight | 117% |
| galloping horse | 94% |
| static portrait | 97% |

The LoRA's author warns specifically about smearing under fast motion at low step counts. The
gallop's motion ratio, 94%, is the closest of the three to its own reference — but that is a
whole-frame average, and a whole-frame average cannot see local smearing (a blurred leg inside an
otherwise-sharp frame moves this metric very little). This rules out a global motion deficit on the
gallop; it does not rule out local smearing, which would need a per-region check that was not done
here.

A fourth scene (two people dancing) also came out poorly at 31 steps — but that comparison is
confounded, not clean: 8-step and 31-step runs use different sigma grids, so the same seed does not
walk the same trajectory, and the two clips do not share a composition to begin with. In the
31-step clip the woman's head is simply turned away in the early frames; that is a different
composition, not the same shot rendered with worse detail. Whether the poor face is a property of
few-step sampling or of this prompt is therefore **not established** by this comparison. What is
established: naming the shot ("medium shot from the waist up") fixed the face at 512x512 without
needing a larger canvas, which is worth knowing before spending four hours on a native render. See
"Prompting" below.

### Prompting: name the shot

The model picks a wide shot when the prompt does not say otherwise, and a wide shot at 512x512
leaves a face perhaps forty pixels across. That is not a *canvas* resolution limit — a close-up
portrait at the same 512x512 renders faces well — but it is an effective-object-resolution problem:
the video VAE downsamples 16x spatially and the DiT patches another 2x on top of that
(`patch_size=(1,2,2)`), so one DiT token covers 32 pixels per side of canvas. A 40 px face spans
about 1.25 DiT tokens across, which is barely anything for attention to work with regardless of how
many pixels the rest of the canvas has. It is a framing default that starves the face of tokens,
not a ceiling on canvas resolution.

Two fixes to the same prompt, tested against each other:

- naming the shot (`medium shot from the waist up`) — fixed it, at 512x512, in 6.8 min
- moving to 768x768 without naming the shot — also fixed it, but took 16.6 min

Same result, one costs 2.4x more. Name the shot instead of buying pixels. Also avoid contradicting
yourself about light: the failing prompt said `dim ballroom` and `warm stage light` in one breath.

### 16 grid points on the tango clip, and the confound that still isn't removed

Four runs of the tango prompt (the "two people dancing" scene above), same seed, differing only in
grid points and whether the Turbo LoRA is applied:

| grid points | forwards | wall clock |
|---|---|---|
| 8 | 7 | 6.8 min |
| 16 | 15 | 13.2 min |
| 31 (reference) | 30 | 25 min |

512x512, 2.4 s.

16 grid points is visibly the best of the four tango runs, by eye, both in the wide opening shot
and in the Shot 2 close-ups. Frame 0 in particular improved going from 8 to 16 grid points — but
the woman's face still carries visible distortion even at 16.

Per-frame whole-frame sharpness (Laplacian variance), frame 0 / mean across the clip:

| run | frame 0 | mean |
|---|---|---|
| 31 steps | 85.3 | 133.1 |
| 16 steps | 108.4 | 138.5 |
| 8 steps, no LoRA | 99.2 | 118.9 |
| 8 steps + Turbo LoRA 0.45 | 99.8 | 131.8 |

The visible step at frame 34 in every one of these clips is the Shot 2 cut. The decode window is
ruled out separately, by the seams that land on nothing else — see "Previews" above; frame 34 is
itself both a seam and the cut, so on its own it proves neither.

All four of these runs carry the same confound as the original dancing-scene comparison: 8, 16 and
31 grid points are three different sigma grids, and a different grid moves the sampling trajectory,
not just its fidelity to one trajectory — the same seed does not compose the same shot on a
different grid. None of the sharpness numbers or the by-eye ranking above separate "fewer steps
render this composition worse" from "a different grid happened to compose the shot differently."

### Splitting the tail does not sharpen anything — it softens

`--schedule tail-split --tail-split K` was built to remove exactly that confound. It keeps
`simple`'s grid bit-identical up to the final interval and subdivides only the tail, so a
tail-split run and a `simple` run of the same seed share one trajectory until sigma 0.667. The
hypothesis it was built to test: `simple` at 8 grid points spends its last forward jumping from
0.667 straight to 0, the last third of the schedule carries most of the denoising, and fine detail
— of which a small face is the densest kind — should be what an Euler step that long loses.

Same prompt, same seed 20260909, 512x512, 2.4 s, no LoRA:

| grid | forwards | tail | wall clock | sharpness f0 | mean | mean pixel distance from the baseline clip |
|---|---|---|---|---|---|---|
| `simple` 8 | 7 | 0.667 → 0 | 6.8 min | 99.2 | 118.9 | — |
| tail-split 2 | 8 | 0.667 → 0.480 → 0 | 7.6 min | 89.4 | 110.4 | 3.5 |
| tail-split 3 | 9 | 0.667 → 0.558 → 0.375 → 0 | 8.4 min | 86.8 | 104.6 | 4.3 |
| 16 grid points | 15 | (uniform) | 13.2 min | 108.4 | 138.5 | **34.3** |

The last column is what the schedule was built for. The tail-split clips sit 3.5 and 4.3 grey
levels from the baseline — the same shot, the same poses, differing in the tail. The 16-step clip
sits at 34.3: a different clip of the same prompt. That is the confound, measured.

The hypothesis is refused. More tail steps make the image *softer*, monotonically, and the faces
do not improve: side by side at frames 0, 8 and 16 the three clips differ by less than the JPEG
they are viewed through. The single long jump to zero was not costing detail — if anything it was
adding acutance the better-integrated tail does not, which is the usual few-step-flow trade of
crispness against fidelity, in the direction opposite to the one assumed.

So the gap between 8 and 16 grid points is not tail discretization. What 16 steps bought on this
prompt was a *different composition* — one that happens to put both faces frontal and larger — and
composition is worth more here than integration accuracy. That is consistent with the token-
footprint argument above, and it is why the keyframe result below matters more than either.

### A keyframe fixes the weak opening outright

If the weak start is a framing default rather than a transient of the sampler, then taking the
framing away from the model should end it. Handing the last frame of a finished tango clip back in
as `--image` does exactly that: at 8 grid points with the Turbo LoRA at 0.45, 7.8 min, frame 0 of
the new clip differs from the supplied keyframe by **6.2 grey levels of 255**. The clip opens on
that composition — two faces, close, large — and the first frames carry none of the smear that
started this whole investigation.

This is worth more than any schedule change measured here, and it also supplies the composition
control that the step-count comparison never had: with the opening frame fixed from outside,
two runs can finally be compared on rendering rather than on which shot the model chose.

### LightX2V at the authors' alpha

The second Turbo adapter, run as its authors run it — `DEFAULT_LORA_ALPHA = 8`, strength 1.0,
4 grid points (3 forwards):

| | wall clock | sharpness f0 | mean | mean frame-to-frame motion |
|---|---|---|---|---|
| LightX2V, alpha 8 | 3.8 min | 61.2 | 105.6 | 7.13 |
| 8 grid points, no LoRA | 6.8 min | 99.2 | 118.9 | 6.26 |
| 16 grid points | 13.2 min | 108.4 | 138.5 | 8.46 |

Motion is back in range. At the alpha 16 that a widely-used ComfyUI conversion guesses at — and
says it is guessing at — the same adapter drove motion to 13.5 against a 6.4 reference, 2.1x, which
is precisely the factor a doubled alpha predicts through `scale = strength * alpha / rank`. Reading
the authors' inference script settled it in a minute; the guess had cost a day of runs.

At three forwards the close-ups hold up and the wide shots are visibly softer than anything else
measured here. It is the fastest usable setting in this fork, and the least detailed.


## Predicting wall clock: use the packed sequence length, not pixels or seconds

Attention is quadratic in the packed sequence and everything else in the block is linear, so the
cost of one forward is a quadratic *with a linear term* — not a power law. Fitting a single
exponent is what went wrong for months.

Five points, each a single measured forward rather than a wall clock, so loading and decoding
cannot contaminate them:

| canvas / duration | packed rows | s / forward |
|---|---|---|
| 512x512, 2.4 s | 6,671 | 51.3 |
| 640x640, 2.4 s | 9,319 | 77.3 |
| 896x512, 2.4 s | 10,375 | 89.7 |
| 768x768, 2.4 s | 13,191 | 118.6 |
| 1344x768, 10 s | 73,061 | 1842.0 |

```
rows    ≈ (5.53 + 1.641 * (seconds - 2.4)) * (W/16) * (H/16) + 81 * seconds + text_rows
seconds ≈ 5.699e-3 * rows + 2.671e-7 * rows**2          # one forward
```

Every point lands within **3%**, and the row estimate reproduces the native run's 73,061 to within
0.9%. The quadratic term is 21% of the cost at 512x512 and 77% at native resolution, which is why
no single exponent ever fit both ends.

**The old formula overestimated the native run by 2.9x, and the reason was one bad number.** The
table this replaces listed 512x512 / 2.4 s at 9,906 rows; the model above puts it at 6,157, and the
run's own log says 6,671 with its keyframe and longer prompt included. The other two old points —
19,242 for 512x512 / 10 s and 37,657 for 896x576 / 10 s — both reproduce to within 1.6%, so the
9,906 was simply wrong, and fitting a power law through it forced the exponent up to 2.25.

Two cautions on using this:

- It predicts **diffusion only**. Loading is a couple of minutes and decoding scales with pixels
  and frames on its own; at 512x512 / 2.4 s the gap between the two is under a minute, at native
  resolution it is not.
- The one point it does not explain is the 896x576 / 10 s run at 133 min wall clock, against 69 min
  of predicted diffusion. That run dates from before `limit_cache`, and it is the era where one
  step took 818 s where its neighbours took 568. Treat it as unmeasured rather than as evidence
  against the model.

What this buys, at 8 steps, on 1344x768:

| duration | rows | diffusion |
|---|---|---|
| 2.4 s | 22,791 | 31 min |
| 3 s | 26,809 | 40 min |
| 5 s | 40,202 | 77 min |
| 10 s | 73,686 | 218 min |

Native resolution is affordable at short durations and only at short durations. The 10 s version is
not four times the 5 s version, it is nearly three times, and that gap is the quadratic term
arriving.


## Damaged faces are a resolution problem, not a step-count problem

The complaint that started this was a distorted face in a waist-up two-shot. Three things were
tried against it and two of them did nothing:

- **More steps: no effect on the face.** 8 versus 31 grid points leaves the noise floor identical
  (flat-region residual 0.72 vs 0.69, temporal flicker 8.55 vs 8.55). The `tail-split` runs, which
  hold the composition bit-identical and subdivide only the final Euler step, made the image
  *softer* as the tail was split further — 99.2, 89.4, 86.8 sharpness at frame 0 for splits of
  1, 2 and 3.

  **Read that first line narrowly: it is about faces, not about grain.** Those two metrics measure
  the residual in flat regions and the frame-to-frame second difference, and both are dominated by
  what is in the shot rather than by noise. Measured properly — see "Grain is a real defect and
  steps do fix it" below — step count halves the grain between 8 and 31. The face damage is what
  steps do not fix.
- **A pinned keyframe: no effect on rendering.** With composition fixed for the first ~8 frames,
  8 / 16 / tail-3 differ by 5% on the face band (93.8 / 98.5 / 98.0) — which is to say they do not
  differ. What the keyframe *does* fix is the weak opening, which is a separate problem.
- **More pixels: this is the one.**

The first resolution ladder — 512, 640, 768, 896x512 on one seed — could not settle it, because
changing the canvas changes the latent shape and therefore the scene: four canvases produced four
different mise-en-scenes, not one at four sizes. Averaging the framing lottery over three seeds at
each of two canvases does settle it:

| canvas | seed 20260909 | 20260912 | 20260913 | mean | spread |
|---|---|---|---|---|---|
| 512x512 | 81.4 | 40.2 | 44.0 | **55.2** | 18.6 |
| 768x768 | 55.8 | 69.3 | 63.8 | **63.0** | **5.5** |

768 is better on the mean and, more tellingly, **three times more consistent**. At 512 the outcome
swings by a factor of two between seeds — sometimes a usable face, sometimes a mangled one. At 768
every seed lands in a narrow band. Looking at the sheet
(`_сравнения/сиды-512-против-768.png`) the difference is not subtle: at 512 two of the three seeds
render the man's face with visibly wrong eyes, and at 768 none of them do.

Read the metric with care — the face band still contains background, so the absolute numbers move
with how cluttered the theatre is behind the dancers. The spread is the robust part of this table,
and the eye agrees with it.

The cost is 16.3 min against 6.8. On the model above that is what buying 2.25x the pixels costs.

That is **not** cheaper than doubling the steps — 512x512 at 16 steps is 13.2 min, three minutes
less. An earlier version of this paragraph claimed otherwise and was simply wrong on the
arithmetic. What the three-seed table supports is narrower: doubling the steps does not fix faces
and 768 does, so the pixels are worth their extra three minutes *for this defect*. Against the
31-step run (23 min) resolution is both cheaper and better.


## Grain is a real defect and steps do fix it — but pixels fix it cheaper

Three metrics were tried against the grain question and the first two were worthless. Both the
flat-region residual and the frame-to-frame second difference are dominated by scene content: they
scored our clips *cleaner* than a reference clip that is visibly grainier, and they scored a
50-step run as noisier than a 31-step one when the 50-step run had simply resolved more true
detail — wall lamps and balcony ornament that the 31-step run left as mush — and that new detail
moved with the camera.

What works needs no motion model and no flat-region hunt. Real picture structure has a 1/f-ish
spectrum and almost nothing at a two-pixel period; grain is flat to Nyquist. So take the radially
averaged power at period 2-3 px over the power at period 8-16 px:

| clip | grain |
|---|---|
| 512x512, 8 steps | **6.57** |
| 512x512, 31 steps | 3.15 |
| 512x512, 50 steps | 3.39 |
| **768x768, 8 steps** | **2.66** |
| reference `soldiers.mp4` (608x352) | 2.19 |
| reference `rapidsave` (1920x1080, upscaled) | 0.96 |

Four things fall out, and the first is a correction:

- **Steps halve the grain between 8 and 31.** An earlier version of this file said step count did
  not touch grain. That was the bad metric talking.
- **Our 8-step 512x512 output really is about three times grainier than a reference clip from the
  hosted model.** The first two metrics said the opposite, which is why the complaint went
  uncorroborated for as long as it did. Trust the eye over a metric that disagrees with it.
- **31 to 50 buys nothing** (+7.7%, marginally worse). Saturation arrives before 31, so upstream's
  "16-31 points are undersampled, use 50" does not hold here. That run cost 39.3 min against 6.8.
- **Pixels beat steps on grain too, and cost less.** 768x768 at 8 steps scores 2.66 — cleaner than
  512x512 at *31* steps (3.15) — for 16.3 min against 23. This is the same conclusion the face
  measurement reached, arrived at independently and by a metric that shares none of its
  assumptions.

Note what is *not* settled by that table: 2.66 still sits above the reference's 2.19, and the
reference was downscaled to 608x352, which suppresses grain on its own. The next section settles it.


## The two bands are scales, not noise and detail

Everything below this heading was first written calling the 2-3 px band "grain" and the 8-16 px
band "structure". That labelling is wrong and a ground-truth measurement disproved it.

The hosted model rendered the same prompt at 896x576 (three seeds) and at 1248x832. Measured on
equal *fractions* of each frame resampled to one size — so the bands refer to scene scale rather
than pixel pitch — the larger render carries **90% more** 2-3 px energy with its 8-16 px energy
unchanged (+3%). A model does not double its noise by rendering bigger; it resolves finer texture.
The 2-3 px band is therefore **fine-scale content**, which happens to include grain, and the
8-16 px band is **mid-scale content**. Read every table below with those names.

What survives the relabelling, and is the load-bearing result:

| clip (centaur prompt, 10 s, 20 steps) | fine 2-3 px | mid 8-16 px |
|---|---|---|
| reference, 896x576, **4 samples** | **0.425 ± 0.038** | **574 ± 42** |
| reference, 1248x832, 2 samples | **0.849 ± 0.068** | 593 ± 23 |
| ours, 4-bit, 8 steps, 896x576 | 0.377 (**-11%**) | 349 (**-39%**) |

Same prompt, same canvas, same duration, same frame count. **Our fine scale sits inside the
reference's own spread; our mid scale is 39% below it, 5.3 sigma out.** The deficit is narrow and
specific, not a general softness.

Two things this table settles that nothing earlier could:

- **Resolution buys fine scale always, and mid scale only when steps are short.** At 20 steps,
  going from 896x576 to 1248x832 doubles fine (0.425 → 0.849) and moves mid by 3%. At **8** steps
  the same jump moves mid from 490 to 601, +23%. So rendering bigger substitutes for steps when
  steps are the binding constraint — which is exactly the regime this fork lives in — and stops
  paying once they are not.
- **Only two candidates remain**: DiT precision and step count. The 2x2 that separates them is
  what the night queue exists for.

Two caveats, both load-bearing:

- The reference clips are 20-step renders and ours is 8-step, so that last row confounds precision
  with step count.
- **The reference is not the bf16 release.** It is a ComfyUI build:
  `minimax_h3_fl2va_pruned_int8_convrot` (DiT at int8, rotation-based),
  `qwen3vl_32b_minimax_h3_nvfp4_awq` (text encoder at **4-bit**),
  `minimax_h3_video_vae_fp16`, `minimax_h3_audio_vae_fp32`. Both VAEs match ours in format, and our
  8-bit text encoder is *more* precise than theirs. The comparison is therefore our 4-bit affine
  DiT against their int8 rotation-quantized one — a bit-width gap, not a quantized-versus-full-
  precision one, which makes "8-bit closes it" a sharper prediction than it looked.


## It was the 4-bit DiT, and 8-bit costs nothing

Decompose the ratio and the question changes shape. The numerator is fine-scale energy, the
denominator mid-scale energy, and a ratio can fall either way — opposite news:

| clip | grain 2-3 px | structure 8-16 px |
|---|---|---|
| ours, 896x512, 8 steps | 0.74 | 138 |
| reference `soldiers.mp4` | **1.67** | **768** |

**The reference has more than twice our absolute noise.** It reads as clean because it carries five
and a half times the detail, and noise disappears into detail. So the complaint that our output
"looks grainy" was never about noise: it was about a *detail deficit*, with a constant noise floor
showing through the gap.

That reframes the step curve too. At 896x512, on one seed, no LoRA:

| steps | grain | structure | ratio | time |
|---|---|---|---|---|
| 8 | 0.741 | 138 | 5.36 | 12.5 min |
| 16 | 0.641 | 275 | 2.33 | 25.4 min |
| 31 | 0.624 | 368 | 1.69 | 46 min |

Steps barely touch the noise — 14% and then 3%, saturating almost immediately. What they buy is
detail, which doubles from 8 to 16.

And the detail deficit had a cause. Same seed, same canvas, same 8 steps, same table, no LoRA;
the only difference is the precision of the DiT's weights:

| DiT | grain | structure | ratio | motion | wall clock | peak |
|---|---|---|---|---|---|---|
| 4-bit (mere.run) | 0.741 | 138 | 5.36 | 4.50 | 12.5 min | 27 GB |
| **8-bit (pipenetwork)** | **0.540** | **242** | **2.24** | **6.03** | **12.6 min** | **27 GB** |

27% less noise, 75% more detail, a third more motion, for **six seconds** of extra wall clock and
**no extra peak memory at all** — the peak is the text encoder on both, and diffusion at 8 bits
runs at 24 GB against 4-bit's 14, still under it. The two clips differ by 25 grey levels of 255, so
4-bit was not merely blurring a shared trajectory; it was denoising to a different one.

An 8-bit DiT at 8 steps therefore matches what previously took 16 steps (2.33) or a Turbo LoRA
(2.37), and it is free. Resident cost is 21.35 GB against 11.34 — measured, and within 0.6% of
pipenetwork's own `gb_resident_after_adaln_drop: 21.47`.

Loading that build needed one fix: it *ships* the 206 AdaLN modulation tensors the mere.run
monolith omits, and `load_dit_cached` rejected them as unexpected keys. They are now skipped by
name, which is also what keeps the build resident in 21.5 GB instead of 35.3.

## Words do not move the centaur

> **Retracted 2026-08-12.** The words were in the wrong place, not powerless. All three revisions
> kept the anatomy in a `Characters:` block — a section that does not exist in MiniMax's own prompt
> format, which the guides in `docs/upstream-guides/` define and which the model was trained to
> parse. The same anatomy moved inside `[Shot 1]` produced a correct centaur on the first run.
> See "The centaur moves when the words are where the model looks" below. The observations about
> repetition spreading concepts across figures stand on their own evidence and are not retracted.

Three prompt revisions, ten drafts, and the model's idea of a centaur did not budge. Recorded here
because the negative result is the useful part: it says which lever to stop pulling.

| version | what it added | what came back |
|---|---|---|
| `centaur-battle-anatomy` | where the human torso joins the horse, six limbs, "exactly one head, and it is human" | a recognisable centaur — and a horse's mane growing out of the human back |
| `centaur-battle-anatomy-2` | no mane, no horse neck; human back all the way down to the buttocks | the horse's back returned, running up to the neck |
| `centaur-battle-anatomy-3` | the whole creature in one sentence; W1 declared "entirely human, two-legged, no animal parts" | W1 clean; the centaur still seats the human torso too low |

Two things fall out of this, and they point in different directions.

**Detail made it worse, not better.** Version 2's paragraph says "horse" and its relatives a dozen
times, and the result was horse-ness spread across the scene rather than attached to one figure —
the observer saw a horse's back on *both* characters. A diffusion text encoder has no syntax tree
that binds an adjective to a noun three clauses away; repetition is a vote for the concept, not for
its placement. Version 3 said the same thing in one sentence and one figure got cleaner, which is
the opposite of what more specification is supposed to do.

**The remaining error is a prior, not a misunderstanding.** Every version puts the human torso lower
into the horse's body than the description asks. That is not the model failing to parse a clause; it
is the model having a fixed idea of where the seam goes and returning to it. Words are the wrong
instrument for that, and the levers that might work — anchoring the first frame with `--image`, or
an adapter trained on the shape — are a different kind of work, untried and out of scope here.

### What this changes about measurement

`prompts/centaur-battle.txt` stays exactly as it is and stays the **measurement fixture**: all eight
reference clips in `reference/` were rendered from it, and it is the only thing that makes
"are we behind the other build" answerable. Nothing about its anatomy needs fixing for that job —
the reference build gets the same creature wrong in its own way, and both sides are compared on the
same words.

`prompts/greek-warrior-battle.txt` becomes the **working prompt** for questions about quality: same
scene, same choreography, same second figure, with the centaur replaced by a body the model has seen
a million times. Use it when the question is about skin, faces, motion or the recipe.

One consequence for the numbers already recorded: the 27% seed-to-seed spread measured on the
centaur is inflated by the model's own uncertainty about the creature, and should not be read as the
recipe's instability alone. The Greek's spread, measured on the same canvas, is the number to
compare against the reference's 4-9%.

## The centaur moves when the words are where the model looks

2026-08-12. The discovery of MiniMax's own prompt-writing guides (`docs/upstream-guides/`) exposed
the flaw in the centaur experiments above: every revision carried its anatomy in a `Characters:`
block placed before any shot — a section the documented format does not have. The format introduces
a subject inside the shot where it first appears, and that is presumably where the model learned to
look for descriptions.

`prompts/centaur-official.txt` is the same creature with the anatomy folded into `[Shot 1]`,
rendered as a 448x288 draft — 8 steps, seed 20260821, the standard 8-bit + Turbo recipe, 13.3
minutes. The topology came back correct on the first run: one horse body, the human torso rising
from the withers, one head and it is human (bearded, holding the bow the prompt gave him), the
second figure fully human with no horse parts bleeding onto her. The seam sits where a centaur's
seam goes — the exact error called "a prior, not a misunderstanding" above.

What survives from the retracted section: the repetition effect (a dozen "horse" mentions spreading
horse-ness across both figures) was observed directly and is consistent with how the format's own
rules behave — introduce once, restate only what would drift. What does not survive is the
conclusion that words were the wrong instrument. They were the right instrument aimed at a section
the parser never reads.

One run, one seed, draft resolution — the usual caveats. But the old conclusion rested on runs with
the same budget, and a single clean counterexample is enough to retire "stop pulling this lever."

The companion test — the same-seed Greek battle rewritten into the documented format
(`prompts/greek-official.txt`, 896x576, versus `23-пластик-896/h3-greek-896-overcast`) — came back
at minimum no worse on stills, and follows its own shot list more visibly: the scripted third-shot
spear spin appears on schedule. Whether motion quality moved is for the viewing, not for stills.

## A line of dialogue survives a cut between independent renders

2026-08-12, the ballad (`docs/ballad-scenario.md`): three ten-second clips — the battle, the
mercy, the embrace — chained by i2v conditioning, each keyframe taken from the penultimate
second of the previous clip, one seed for all three, concatenated without re-encoding into
`27-баллада/ballad-full-30s.mp4`. This was the plan's montage experiment and its speech
experiment folded into one story, and every mechanism under test worked:

- **Speech, in two languages.** Her Russian «Сдавайся!» and his English "Never!" land in their
  scripted windows (voiced harmonics at 4.1–4.9 s and 6.0–6.7 s of clip 2, mouths open on the
  matching frames).
- **A line split across the splice.** Clip 2 ends on "I will not" (voiced segment right at the
  9.5 s boundary), marked `<scenetrans>` in both prompts; clip 3 opens with "hurt you." inside
  its first 0.7 s. The clips were rendered independently — nothing but the prompt text and the
  keyframe connects them — and the phrase reads as one sentence across the cut.
- **Identity across two i2v hops.** Faces, builds and the overcast light hold through both
  splices; the final wide embrace is recognisably the same two people who crossed weapons
  twenty seconds and two renders earlier.

The viewer's verdict on the assembled thirty seconds: it came out as planned. Worth keeping
for next time: clips 2 and 3 deliberately end in near-static poses so their final frames make
clean conditioning targets, and the penultimate-second rule (`-sseof -1.5`) dodged the motion
blur of true last frames — neither choice has been varied yet, both are recorded as fixed.

## The plastic skin was the lighting, not the recipe

A viewer described skin on the 896x576 Greek clips as plastic — overexposed, oiled. Four runs at
one seed and one canvas separated the causes, and the answer costs nothing to apply.

| arm | lighting | LoRA | prompt | skin |
|---|---|---|---|---|
| control | golden hour | 1.0 | plain | plastic |
| weakened adapter | golden hour | 0.6 | plain | still plastic, and grain returned |
| skin detail | golden hour | 1.0 | pores, texture, beads of sweat | still plastic, plus a sword that lost its blade |
| **overcast** | **flat overcast** | 1.0 | short matte phrase | **clean** |

**The recipe was never the problem.** Weakening the Turbo LoRA to 0.6 did not remove the sheen and
brought back the grain that eight steps only survive *because* of the adapter — the adapter is what
makes the short schedule converge, so turning it down trades one defect for a worse one. Asking for
pores at full strength did not help either, and the extra text cost the sword its blade: the same
attention-spreading that put a horse's back on both figures of the centaur prompt.

**The lighting was.** "Epic golden-hour lighting" is a low sun, and a low sun is specular highlights
across skin — that reading *is* the oiled look. Worse, the prompt asked for it twice: `sweat
glistening` is a literal request for a sheen over every lit surface. Flat overcast daylight removes
both by construction rather than by persuasion.

Three things changed in the winning arm at once — the light, the `sweat glistening` phrase, and a
one-sentence matte instruction — so which of the three carries the weight is not separated. All
three are the same lever, and it is a free one: no steps, no bits, no pixels, no wall clock.

### What to carry forward

- `prompts/greek-warrior-battle-overcast.txt` is the prompt to copy for anything where skin is on
  screen. Keep the golden-hour variant only when the sheen is wanted.
- **Never write `sweat glistening`** unless the sheen is the point.
- Skin detail belongs in one sentence, not a paragraph. The 485-word version produced high-frequency
  noise and dropped objects; the 399-word version did not. Asking for detail finer than a pixel —
  "individual beads of sweat" at 896x576 — sets a target the canvas cannot hold, and the model
  answers with artefacts.
- **A canvas that hides a defect is not a canvas that fixes it.** The same control clip looked clean
  at 448x288, where a face is fifty pixels tall and there is no skin to be plastic. Drafts answer
  questions about meaning and composition; questions about surface must be asked at the working
  canvas. This cost one wasted pair of drafts.

## The seed-spread number is withdrawn

Every "our spread is N% against the reference's M%" line written on 2026-08-11 and 08-12 should be
read as retired. The comparison was unsound at both ends, and the person watching the clips is the
one who found it.

**Our end: the metric measures staging, not stability.** The spectral band ratio is a global
statistic over the frame — how much of it is sky, how much is bodies, how much is ground. Seeds of
this prompt stage the scene differently: the figures swap sides, one seed frames them close and
another goes wide. That moves the mid-scale band by a factor while the *quality* is identical, and
the viewer's verdict on the night's four clips was "excellent in every one of them". So a 20% spread
across seeds is a measurement of how freely the prompt lets the model direct, not of how steadily
the recipe renders.

**Their end: nothing says the reference trio are three seeds.** `reference/README.md` records their
seeds as "not recorded", and then asserts they provide "a measured seed-to-seed spread". That
sentence was an assumption written as a fact, and it has been withdrawn. Three renders of one seed
would produce exactly the 1% we measured, and we cannot rule it out.

Two withdrawals in one: the numbers on both sides of the comparison, and the claim that the
comparison was possible.

### What replaces it

Nothing, for now, and saying so is better than quoting a number that measures the wrong thing.
This is the fourth metric this project has had to retire — flat-region residual and the temporal
second difference both said our clips were *cleaner* than the reference; motion compensation
produced a residual larger than the plain frame difference; the spectral bands matched the eye on
grain but were blind to anatomy, and are now shown to be sensitive to staging as well.

If a number is wanted later, it has to be invariant to staging. Two candidates, neither tried:
measure inside a detected face or body crop rather than over the whole frame, so composition cannot
move it; or compare seeds under a prompt that pins the camera to a single static shot, so there is
nothing left for the model to direct. Until one of those exists, the honest instrument for questions
about surface quality is a person looking at the clip at the working canvas.

## Attention memory levers: 44.8 GB to 36.2 GB at 15 s, and not one bit changed

Four rewrites of how the DiT stages its Q/K/V and its MLP, landed 2026-08-17 as
`patches/0002-attention-memory-levers.patch` plus a load-time carve in `h3_48gb/dit.py`. None of
them is an approximation, a lower precision, or a dropped term: every one is an arithmetic
identity, and the gate each had to pass was a forward compared to the unpatched code at `atol=0,
rtol=0` — bit for bit, not "within tolerance".

| clip | packed rows | stock peak | levers peak | delta | stock fwd | levers fwd |
|-----:|------------:|-----------:|------------:|------:|----------:|-----------:|
| 2.4 s | 22,791 | 26.39 GB | 25.40 GB | −0.98 | 10.4 s | 10.3 s |
| 5 s | 38,081 | 29.64 GB | 28.00 GB | −1.65 | 22.5 s | 22.5 s |
| 10 s | 73,757 | 37.23 GB | 32.31 GB | −4.92 | 70.2 s | 67.9 s |
| 15 s | 109,433 | 44.81 GB | 36.16 GB | **−8.65** | 138.8 s | 138.0 s |

1344x768, two transformer blocks, 19 distinct noise levels, 8-bit DiT (21.54 GB resident weights
plus AdaLN table). Two blocks rather than fifty because the peak of a DiT forward is block-local —
each block's intermediates die with the block — so a 2-block forward reproduces the 50-block peak
at a twenty-fifth of the cost. Peaks are MLX's own allocator counter (`mx.get_peak_memory`), not
RSS; the two instruments are not interchangeable and this table is entirely the first one.

**The saving grows with the clip, and that is the point.** It is not a constant overhead being
removed: the chunked paths hold one 8192-row slice regardless of how long the sequence is, so the
share of *activations* they remove rises from 20% at 2.4 s (4.85 GB above resident weights, down
to 3.87) to 37% at 15 s (23.27 GB, down to 14.62). Against the whole peak — activations plus the
21.54 GB of resident weights, which no lever touches — that is 3.7% and 19.3%. The 15 s forward is
the case the whole exercise was for — at 44.81 GB it was above this machine's 40.2 GB recommended
working set and ran on swap; at 36.16 GB it is under it.

**Wall clock is unchanged to within a percent**, and slightly better at 10 s (70.2 -> 67.9 s). The
chunk loops add kernel launches and re-read `k`/`v` per slice; that cost lands inside the noise of
what the narrower buffers buy back.

### The four

* **rope-lean** (−1.02 GB at 10 s). `apply_rotary` split the head into a rotated and a
  passed-through part, rotated one, and concatenated them back: five full-size temporaries for
  elementwise work. Padding `cos`/`sin` to the full `head_dim` with 1 and 0 makes the pass-through
  channels fall out of the same expression — `x*1 + rotated*0` is exact at any float width — and
  two temporaries do the job of five.
* **qkv-split** (−1.64 GB). The fused `attn.qkv_proj` is carved into `q_proj`/`k_proj`/`v_proj` at
  load time by gathering output rows. Fused, the `(seq, 3*inner_dim)` projection — 4.8 GB at 73k
  rows — must be entirely live before q, k or v can be sliced out of it. Split, k and v are built
  and reduced to their final form before q is projected at all. Exact under quantization because
  MLX groups affine quantization along the *input* axis: `scales`/`biases` are
  `(rows, in/group_size)`, so a row carries its own scale and a row gather moves it intact.
  Resident weights do not grow — the fused layer is dropped.
* **q-chunk** (0.00 GB on its own, then −2.26 GB jointly with the next one). Attention over 8192
  query rows at a time, `k`/`v` whole. Softmax normalizes within a query row, so a slice of query
  rows is a self-contained problem. On its own it moved the peak by *nothing*, because until the
  MLP horn came down the peak was the MLP's — the discovery probe had measured its −3.3 GB on top
  of `mlp-chunk` and the plan's task order inverted that. It did make the forward faster.
* **mlp-chunk** (−2.26 GB with the above). `fc1`'s fused `[gate; value]` output is twice the FFN
  width — 57 KB per row against the hidden stream's 14 KB — and has to be live alongside the
  activation it feeds. Built a chunk at a time, only the activation stays full-size.

### The bit that nearly wasn't exact

Chunking a linear layer by rows is arithmetically free and *not* automatically bit-free. MLX picks
a GEMM tiling by shape, and for a narrow output it splits the reduction differently at different
row counts. Measured here: `x @ lora_A.T` at 7168-wide input and rank 64 differs by one bf16 ULP
between a 6247-row call and the concatenation of 2048-row calls. Every spelling tried — `addmm`
into zeros, float32 accumulation — has the same property, so it is the kernel and not the
expression. The first chunked run with the Turbo LoRA attached came back at `max|Δ| = 0.536` on a
scale of 154, which is exactly one bf16 ULP, and exactly not zero.

The LoRA's rank-r read is the only narrow matmul in the model, and it is also the only thing small
enough to hoist: `(seq, 64)` is 9 MB where the buffers being chunked are gigabytes. So `dit.py`
grew a two-function protocol — a layer may declare a whole-sequence `prepare_rows` and consume
slices of it in `apply_rows` — and `LoRALinear` implements it. The `addmm` left in the loop *is*
row-stable, as is the 8-bit `QuantizedLinear` base; both were checked rather than assumed. With
that, everything returns to 0.0.

The same reasoning is why **`fc2` still runs once over the whole activation** instead of per chunk.
Its LoRA reads the activation, which is the very thing the chunking is building, so its narrow read
cannot be hoisted. Chunking it too was measured — 30.78 GB against 32.31 at 10 s — so the exactness
costs **1.53 GB**, paid deliberately.

### What was checked

* `tests/test_attention_levers.py`, 38 tests: each lever against a frozen copy of the stock
  upstream code at `atol=0, rtol=0`; the carve under 8-bit quantization; the Turbo LoRA over split
  and fused projections against each other; LightX2V likewise; the two chunk levers composed.
  Guards throughout against a test passing for the wrong reason — that the rotation happens at all,
  that the QKV permutation still matters, that the chunk loop is actually entered, that a masked
  call is never chunked.
* Real weights, `h3-8bit-full`, all 50 blocks at 512x512, chunk sizes lowered to 2048/1500 so the
  chunked paths are genuinely entered: `max|Δ| = 0.000e+00` on both outputs, with the Turbo LoRA
  (`minimax_h3_turbo_v4_step600_ema`, strength 1.0) and without it.
* One MLX fact is pinned on purpose: below ~512 query rows MLX takes a different attention kernel
  whose accumulation order differs, so chunking a 17-row sequence moves by a ULP. The lever never
  enters that regime (it engages above 8192 rows), but a doll-sized parity test written without
  knowing this would have passed for the wrong reason.

### How to re-check it

Both instruments are in the repo, not in a scratch directory, because the thing they measure is
exactly the thing a later refactor gives back without failing a test.

    ./.venv/bin/python scripts/levers_parity.py --blocks 0 \
        --lora ~/models/turbo/minimax_h3_turbo_v4_step600_ema.safetensors
    ./.venv/bin/python scripts/levers_ladder.py --durations 2.4 5 10 15

`scripts/levers_parity.py` is **mandatory before any change to a chunked path, and after any MLX
upgrade**. It loads the real checkpoint twice in one process — once unsplit with frozen copies of
the pre-patch upstream code, once as the repo now loads it — and compares outputs bit for bit. It
defaults its chunk widths to 2048/1500 rather than the shipped 8192 on purpose: a parity canvas is
~6.2k rows, so at 8192 both chunked branches fall through to their unchunked fallbacks and the run
proves nothing about chunking at all. That is precisely how the reconnaissance probe reported
`max|Δ| = 0` for variants whose chunk loops had never executed.

`scripts/levers_ladder.py` produces the table above, plus the `fc2`-chunked variant that is not
shipped. It refuses to measure while the queue has a job running, since a peak measured against
another process is fiction. `scripts/dit_probe.py` holds what the two share — packed geometry, the
synthetic modulation table, and the frozen pre-patch code — so a benchmark and a parity checker
cannot end up disagreeing about what "stock" means.

### Still open

The queue's pre-run memory estimate (`h3_48gb/web.py`, `peak_gb = 9.3*rows/37657 + weights`) is
fitted to whole-run measurements from 2026-08-11 and has **not** been re-fitted. It is now
pessimistic for the diffusion phase — but a run's peak is not necessarily its diffusion peak (the
largest single allocation this project has observed is the 28.22 GB text encoder), so re-fitting it
honestly needs end-to-end runs, which the probe above is not.
