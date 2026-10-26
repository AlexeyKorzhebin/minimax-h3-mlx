# Design notes

Facts established by reading the code and the safetensors headers directly, not from
documentation.

## What blocks a run

`pipeline.py::from_pretrained` (lines 92-113 of the upstream port, at commit `fcd9e9b` — the pinned
revision this fork is written against) loads the text encoder, the DiT and both VAEs one after
another and keeps all of them resident until generation finishes: **45.9 GB of weights**, and
**~55 GB** at the moment diffusion peaks, on a machine with **48 GB** physical. The mere.run runtime
refuses this twice: a declared-support threshold of 96 GB (liftable with `--allow-unsupported`),
and, separately, admission control that requires 32 GB free before it will admit any job at all.

The 45.9 GB is the sum of the per-component figures MLX itself prints during a real run
(28.22 + 11.34 + 5.21 + 0.61 GB, plus the 0.56 GB AdaLN cache); the additional 9.3 GB is upstream's
own measured activation working set for 1344x768 / 5 s. See `docs/RESULTS.md` for both derivations
and for what was and was not observed directly.

The four components are needed strictly by phase, though:

| Phase | Component | Where in `__call__` |
|---|---|---|
| 1. Text encoding | `text_encoder` | line 238, called once |
| 2. Keyframes | `video_vae.encode` | line 270, only with `--image` |
| 3. Diffusion | `dit` | lines 306-318 |
| 4. Decode | `video_vae`, `audio_vae` | lines 368, 382 |

## Sawfwair weight layout vs. what the port expects

| | Sawfwair (as downloaded) | Port expects |
|---|---|---|
| Structure | monolithic `<component>.safetensors` files | `FL2VA/<component>/` directories, sharded |
| Quantization | MLX affine: `weight` U32 + `scales`/`biases` BF16 | same ✅ |
| Tensor names | `blocks.N.attn.qkv_proj.*`, `model.layers.N.mlp.*` | same ✅ |
| QKV layout | `[all-q; all-k; all-v]` slabs | **per-head interleave** `[h0:q,k,v][h1:q,k,v]...` ❌ |
| Quantization recipe | a field inside `config.json` | separate `quant_config.json` ❌ |

The QKV permutation is cheap: `scales [21504, 84]` means grouping runs along input features
(5376 / 64 = 84 groups), not along rows. Rows move together with their own `scales` and `biases`,
so no dequantization is required. 56 heads x 128 = 7168 = `inner_dim`.

## The work

1. **Converter** (`convert_sawfwair.py`): unpack the monoliths into the port's directory
   structure, permute QKV into the per-head interleave, and generate `quant_config.json`. Avoid
   copying weights that need no change — tensors that don't require permutation are written
   as-is.
2. **Lazy loading** in `pipeline.py`: `from_pretrained` reads only configs; weights load on
   demand. Configs are needed early (`dit.config.patch_size`, `video_vae.config`), so they are
   split from the weights.
3. **Encoder eviction** after line 238. Critical detail: `mx.eval(prompt_embeds)` must run before
   eviction — MLX is lazy, and without materializing the output the graph keeps the encoder's
   weights alive, so nothing is actually freed.
4. **Verification**: 512x512, 56 frames — the geometry with the largest memory headroom. Only
   attempt native 1344x768 after that succeeds.

## Expected memory use after the patches

These were the plan's projections, written before any run. They are kept as written, with what
actually happened beside them:

| Phase | Resident | Projected peak | Outcome |
|---|---|---|---|
| Encoding | encoder Q8, 28.2 GB | ~29 GB | MLX reported exactly 28.22 GB allocated and released; **process RSS was never sampled during this phase** — see `docs/RESULTS.md` |
| Diffusion, 512x512 | DiT 11.3 + VAE 5.8 + cache 0.9 + activations ~2 | ~20 GB | RSS said 11.54 GB; RSS is blind to Metal, see below |
| Diffusion, 5 s at 1344x768 | same + activations 9.3 | ~27 GB | RSS said 10.14 GB; Activity Monitor on a comparable run said 29.13 GB |
| Diffusion, 15 s at 1344x768 | same + activations 24.4 | ~42 GB | never attempted; extrapolates past 35 hours per clip |

**The RSS column above is not a measurement of what the run holds, and the projections were
closer to right than it made them look.** Metal allocations do not appear in a process's resident
set on Apple silicon: during a run Activity Monitor showed at 29.13 GB, `ps` reported 0.13 GB. The
10–11.5 GB figures are whatever fraction of the allocation happened to be file-backed resident
pages at sampling time, which is not a quantity anyone should plan against.

The projections' error was in the other direction and smaller: they omitted MLX's freed-buffer
cache, roughly 8 GB on a long run until it was bounded. Corrected accounting, the instrument that
sees it, and the four unloads that keep each phase to its own budget are in
[`docs/MEMORY.md`](MEMORY.md), which supersedes this section.

The GPU limit was raised to 44 GB (`iogpu.wired_limit_mb=45056`; resets on reboot) to leave headroom
for the 15 s row above. Since that row was never run and nothing else came near 12 GB, this is
methodology rather than a prerequisite: reproducing the runs in `docs/RESULTS.md` does not require
changing it.

## What this fork does not solve

Speed. The bottleneck is dense-attention FLOPs — MiniMax has not released a sparse-attention
implementation. The Spectrum accelerator (ComfyUI-Spectrum-MiniMax-H3) gets 30-35% by skipping
steps, but needs 2.2-6.1 GB for prediction history, degrades fast motion, and is written for
ComfyUI/PyTorch. Worth revisiting once the pipeline itself works, not before.

See `docs/RESULTS.md` for the measured numbers this plan predicted, and the README's "The four
patches" section for how the patches above map onto `h3_48gb/`.
