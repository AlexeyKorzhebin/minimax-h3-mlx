# Feasibility: Turbo LoRA (4-step) and TAE previews

Assessment of two upstream artifacts against this fork's converted checkpoint
(`~/models/h3-converted`, Q4 affine / group 64, per-head-interleaved QKV). Nothing here was
implemented; every number below came from safetensors headers read over HTTP range requests, from
the published sources of the ComfyUI nodes, and from arithmetic against tensors already on disk.
No multi-GB file was downloaded — the largest read was 87 MB of range-selected tensors.

**Verdicts up front.**

| | Verdict | Download | Work |
|---|---|---|---|
| **A. Turbo LoRA** | Feasible | ~837 MB | ~20–30 h |
| **B. TAE preview** | Feasible, cheap | 9.8 MB | ~10–15 h |

The AdaLN blocker that motivated this study **does not cost 26 GB and does not require the official
checkpoint.** The whole modulation path exists in 8-dimensional curve form inside ComfyUI's pruned
checkpoint, weighs **87.3 MB**, and reproduces our own baked table to 2.2e-3 rel-L2 — slightly
*better* than the 8-bit AdaLN quantization this project already calls measured-safe and ships.

---

## A. Turbo LoRA — 4-step distillation

### Verdict: feasible, at ~837 MB of downloads and ~20–30 hours of work

Native 1344x768 / 5 s goes from 299 min to **39 min (4 forwards)** or **78 min (8 forwards)**.
Two findings are hard requirements rather than preferences: use the **original** LoRA, not the
ComfyUI pruned conversion; and apply it as a **runtime side-path**, never merged into the Q4
weights. Both are justified with measurements below.

### Evidence

#### 1. The LoRA matches our module tree exactly — one layout fix aside

Header of `larryvrh/MiniMax-H3-Turbo-Lora/minimax_h3_turbo_4step*.safetensors`: 518 tensors,
780 MB, all bf16, metadata

```
{"dtype": "bfloat16", "sampler_steps": "4",
 "application": "W_eff = W + lora_B @ lora_A", "base_model": "MiniMax-H3"}
```

259 A/B pairs, in two groups:

| group | pairs | rank | shapes |
|---|---|---|---|
| backbone | 208 | 64 | `blocks.N.attn.qkv_proj` A[64,5376] B[21504,64]; `attn.out_proj` A[64,7168] B[5376,64]; `mlp.fc1` A[64,5376] B[28672,64]; `mlp.fc2` A[64,14336] B[5376,64] — x50, plus the same four x2 in `token_refiner.blocks.N` |
| modulation | 51 | 16 | `blocks.N.adaln_proj.linear` A[16,2688] B[96768,16]; `final_layer.adaln_proj.linear` B[10752,16] |

Every one of those names and dimensions is ours. Our `blocks.0.attn.qkv_proj.weight` is
`[21504, 672]` u32 = `[21504, 5376]` dequantized; `mlp.fc1` `[28672, 5376]`; `mlp.fc2`
`[5376, 14336]`; `attn.out_proj` `[5376, 7168]`. The adaln pairs match
`config.json`'s `time_embed_dim: 2688`, `adaln_out_features: 96768`,
`final_adaln_out_features: 10752`.

**Which variant to use: the original.** `drbaph/MiniMax-H3-Turbo-Lora-ComfyUI` is a namespace
conversion (`blocks.*` → `diffusion_model.blocks.*`) that **deletes all 102 AdaLN tensors**. Its own
file metadata says so:

> `"warning": "All AdaLN LoRA adapters were removed because the source targets input dimension 2688
> while the pruned model targets dimension 8. Four-step distillation behaviour may be degraded or
> broken."`, `"retained_tensor_count": "416"`, `"removed_pair_count": "51"`.

The `diffusion_model.` prefix it adds is trivial to strip, and it offers nothing we cannot get from
the original. Take the original and keep the 51 adaln pairs — §4 shows how to apply them.

#### 2. QKV row order differs, and the fix already exists in this repo

Measured, not assumed. I dequantized rows of our `blocks.0.attn.qkv_proj` (implementing MLX's
affine 4-bit unpacking in numpy) and range-read the same rows from
`Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors`, then correlated:

```
comfy row     0 -> our row     0   (corr 0.995)     Q head 0
comfy row   128 -> our row   384   (corr 0.995)     Q head 1
comfy row  7168 -> our row   128   (corr 0.995)     K head 0
comfy row 14336 -> our row   256   (corr 0.995)     V head 0
```

ComfyUI (and therefore the LoRA) uses `[all-Q; all-K; all-V]` slabs. We use the port's per-head
interleave. So `qkv_proj.lora_B`'s 21504 rows must be permuted for all 52 matrices (50 blocks +
2 token-refiner) — **the identical permutation `convert_sawfwair.py` already applies to the base
weights** when undoing mere.run's layout. Reuse, not new work.

`out_proj`, `mlp.fc1` and `mlp.fc2` were probed the same way and match 1:1 (including fc1's
gate/value halves at rows 0 and 14336). Only `qkv_proj` needs the permutation.

#### 3. LoRA on MLX Q4: the side-path works, merging destroys it

MLX 0.32.0 and mlx-lm 0.31.3 are installed and support both routes:

- `mx.quantize` / `mx.dequantize` / `nn.QuantizedLinear.from_linear(linear, group_size, bits, mode)`
- `mlx_lm.tuner.lora.LoRALinear.from_base(...)` — runtime side-path, explicitly branches on
  `isinstance(linear, nn.QuantizedLinear)`
- `LoRALinear.fuse(dequantize=False)` — merge: `mx.dequantize` → add `scale * lora_b.T @ lora_a.T`
  → `nn.QuantizedLinear.from_linear` again

**Merging is not viable at 4 bits.** Measured on `blocks.0.attn.qkv_proj` with the recommended
`minimax_h3_turbo_4step_ema_ckpt850` weights:

| quantity | value |
|---|---|
| base weight (dequantized) rms | 1.19e-01 |
| LoRA delta `B@A` rms / max | 3.37e-05 / 3.09e-04 |
| Q4 group-64 step (median \|scale\|) | 1.19e-02 |
| Q4 rounding-error rms (step/sqrt 12) | 3.44e-03 |
| **delta rms / quantization step** | **2.8e-03** |
| delta rms / base rms | 2.8e-04 |

Each weight's LoRA update is ~0.3 % of one quantization bin, so requantization rounds ~99.7 % of it
away. larryvrh warns that merging is "softer on quantized (`int8` / `fp8` / pruned) bases"; at 4
bits it is not softer, it is **absent**. The `low_vram` merge switch in his node is therefore not an
option for us.

The runtime side-path costs, per forward:

- **Memory**: 620 MB resident bf16 for the 208 backbone pairs. (The 51 adaln pairs are consumed once
  at table-build time — §4 — and never stay resident.) Plus one output-sized temporary per patched
  layer; the worst is `fc2` at ~46 k tokens x 5376 bf16 ≈ 0.5 GB. Against a current peak of
  10–11 GB under a 44 GB limit, this is comfortable.
- **Compute**: rank 64 adds `r*(in+out)` = 5.96 MMAC/token/block against 385 MMAC/token/block for
  the base linears — **+1.5 % on the linear layers and 0 % on attention**, which `docs/RESULTS.md`
  establishes as the wall-time bottleneck at native resolution. Wall-clock overhead is under 1.5 %.

This also matches the reference: larryvrh's node defaults to bypass and calls it "exactly like the
standalone `generate.py` reference".

#### 4. The AdaLN table: 87.3 MB, not 26 GB

**The official route, as asked.** `MiniMaxAI/MiniMax-H3`'s
`transformer/diffusion_pytorch_model.safetensors.index.json`: 638 tensors, `total_size`
66,280,430,080 (66.28 GB), 14 shards. The 104 modulation tensors are

- `transformer_blocks.N.adaln_proj.linear.weight` `[96768, 2688]` bf16 + bias, x50
- `norm_out.linear.weight` `[10752, 2688]` bf16 + bias (the final layer's)
- `time_embedder.linear_1` `[5376, 256]` f32, `time_embedder.linear_2` `[2688, 5376]` f32, + biases

and they are spread across **all 14 shards** (6–10 tensors each). So there is no shard subset:
whole-shard downloading is the full **66.28 GB**. Byte-exact range reads of only those 104 tensors
come to **26.14 GB**. Both are unattractive, and neither is necessary.

**The route that actually works.** `Comfy-Org/MiniMax-H3`'s
`diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors` ships the same modulation path in
8-dimensional curve form:

| tensor | shape | dtype |
|---|---|---|
| `adaln_t_table` | `[1025, 8]` | F32 |
| `blocks.N.adaln_proj.linear.weight` (x50) | `[96768, 8]` | F16 |
| `blocks.N.adaln_proj.linear.bias` (x50) | `[96768]` | F16 |
| `final_layer.adaln_proj.linear.weight` | `[10752, 8]` | F16 |
| `final_layer.adaln_proj.linear.bias` | `[10752]` | F16 |

**103 tensors, 87.3 MB**, all range-readable. That checkpoint's module naming is byte-for-byte ours
(`blocks.N.attn.qkv_proj`, `mlp.fc1`, `final_layer.*`, `token_refiner.blocks.N.*`, `rope.inv_freq`)
— mere.run's build is the same pruned/curve family; it simply dropped these tensors and baked a
fixed table instead.

**Verified numerically against our own cache.** Reconstructing

```
modulation(t) = lerp(adaln_t_table, t * 1024) @ W.T + b        # W = [96768, 8]
```

at `t = 1 - sigma`, reshaped `[3 modalities, 6*hidden]`, and comparing against
`adaln_cache.safetensors` over all 30 steps x {video, audio} plus the 0.999 conditioning level:

| block | rel-L2 |
|---|---|
| 0 | 2.19e-03 |
| 17 | 2.31e-03 |
| 33 | 2.23e-03 |
| 49 | 2.28e-03 |
| `final_layer` | 2.01e-03 |

Uniform; no blow-up at any step or level.

**That error is inside this project's own published tolerance.**
`upstream/minimax_h3_mlx/quantize.py` records measured AdaLN quantization results:

| adaln bits | table rel-L2 | core velocity rel-L2 |
|---:|---:|---:|
| 8 | **0.0025** | 0.0329 |
| 6 | 0.0031 | 0.0611 |
| 4 | 0.0077 | 0.1649 |

8-bit (0.0025) is described there as "measured-safe" and is what the published builds use. The
curve form's 0.0022 is slightly better than that, and ~75x below the 0.1649 velocity error our own
4-bit core already carries.

A useful side effect: this reconstruction independently re-derives the `(step, variant)` layout
that `h3_48gb/adaln.py` reverse-engineered from the data — the two agree to 2e-3 with no fitting.

#### 5. Applying the AdaLN LoRA — free, at table-build time

The 51 adaln pairs live in the 2688-dim `silu(t_emb)` space, which the curve form has collapsed to
8 dims, so they cannot be folded into the `[96768, 8]` weights. larryvrh solved this by shipping
`h3_silu_temb_grid.safetensors` in the node repo — **5.5 MB**, one tensor `silu_t_emb_grid`
`[1025, 2688]` bf16, metadata `{"grid": "linspace(0,1,1025)", "desc": "silu(time_embedder(t))
aligned with adaln_t_table rows"}` — and adding `B @ (A @ silu_temb)` to each adaln projection on
every forward (`_make_adaln_forward`, `_inject_adaln_egrid`).

We can do this more cheaply than he can. Because we precompute the entire modulation table anyway,
the term is added **once per distinct timestep at build time**:

```
modulation(t) = lerp(adaln_t_table, t) @ W.T + b  +  B @ (A @ lerp(silu_t_emb_grid, t))
```

Zero runtime cost, zero extra residency, and the 51 adaln LoRA pairs are dropped after the build.
This is strictly better than the ComfyUI path, which pays the injection on every forward.

#### 6. Sampler and sigma schedule: already correct, no port needed

The node hard-codes

```python
SHIFT_V, SHIFT_A = 12.0, 3.0
```

— **exactly our shifts.** Its `_turbo_sampler` steps video on its own sigma and audio on a
re-shifted clock with a Jacobian ("slope") correction, because ComfyUI wraps the model behind a
single sigma and a stock sampler "over-steps the audio at 4 steps and the audio breaks".

The port does not have that problem. It already runs **two independent `MiniMaxH3Scheduler`s**
(`video_sched.step(...)` / `audio_sched.step(...)`, `upstream/minimax_h3_mlx/pipeline.py` around
lines 322–327), each on its own sigma grid. The two reduce to the same update:

- port: `x_next = r*x + (1-r)*x0`, `x0 = x + sigma*v`, `r = sigma_next/sigma` → `x + (sigma - sigma_next)*v`
- node: `out = (x - denoised)/sigma = -v`, then `x + (sigma_next - sigma)*out` → `x + (sigma - sigma_next)*v`

The node's `slope` division is purely the clock conversion the port never has to make. So **no
sampler work is required** — only verification.

`MiniMaxH3Scheduler.set_timesteps` already accepts any `num_inference_steps >= 2`, and `set_shift`
already exists. The scheduler was never the blocker; only the table was.

One note on schedules: drbaph's README suggests audio shift 4–6 and 6–10 steps. That is a
third-party retune of a conversion whose adaln adapters were deleted — not the author's setting.
larryvrh's node uses 3.0, which is what our table (baked and curve alike) is consistent with.

#### 7. What the speedup actually is

The port spends one grid point on the terminal sigma, so N grid points drive N-1 forwards: ComfyUI's
"4 steps" is `--steps 5` here. At the measured 586 s/forward for native 1344x768 / 5 s:

| forwards | `--steps` | total | vs 299 min |
|---:|---:|---:|---:|
| 4 | 5 | 39 min | 7.7x |
| 6 | 7 | 59 min | 5.1x |
| 8 | 9 | 78 min | 3.8x |
| 30 | 31 | 299 min | 1.0x |

The author reports `ckpt850` is sharp at 4; earlier checkpoints needed 6–8.

### Work required

| # | Task | Est. |
|---|---|---|
| 1 | `h3_48gb/curve_adaln.py`: range-fetch the 103 curve tensors, build the modulation table for an arbitrary schedule, serve the existing `CachedModulation` interface. Keep `adaln_cache.safetensors` as the preferred source when the schedule matches it. | 5–7 h |
| 2 | Fold the 51 adaln LoRA pairs into the build via the silu grid (§5). | 2 h |
| 3 | LoRA loader: strip/normalize names, permute `qkv_proj.lora_B` rows for 52 matrices reusing `convert_sawfwair.py`'s permutation, assert alignment against base rows. | 3–4 h |
| 4 | Runtime side-path module over `nn.QuantizedLinear` for the 208 backbone pairs; must survive per-phase lazy loading and checkpoint/resume. | 4–6 h |
| 5 | CLI: `--lora`, allow `--steps` other than 31, `doctor` checks for the curve tensors and LoRA. | 3–4 h |
| 6 | Tests (permutation correctness, table fidelity vs the baked cache, no-LoRA regression). | 4–5 h |
| 7 | Validation: 512x512 4-forward run (~3 min) then native 4- and 8-forward runs (39 / 78 min). | ~2.5 h wall |
| | **Total** | **~21–28 h** |

Downloads: **744 MB** (`minimax_h3_turbo_4step_ema_ckpt850.safetensors`) + **87.3 MB** (curve
tensors, range-read) + **5.5 MB** (silu grid) = **~837 MB**.

### Risks

- **Nobody has run this LoRA on a 4-bit base.** It was trained and validated against bf16 / int8 /
  fp8. The delta is 100x below Q4's own rounding noise in rms; the side-path preserves it exactly,
  but whether a coherent rank-64 update still steers a Q4 model into the 4-step regime is unverified
  and cannot be settled analytically. **This is the single biggest risk.** Mitigation: the 512x512
  4-forward run costs ~3 minutes and answers it before any native run.
- **The LoRA is an explicitly unfinished preview.** The author reports "plastic-looking skin and
  over-sharp grain" at `ckpt850`, has paused training, and states "functionality and compatibility
  are not guaranteed". Expect a quality regression against the 30-forward baseline, traded for 7.7x
  wall time.
- **Silent-failure surface.** A wrong `qkv_proj.lora_B` permutation produces a plausible but wrong
  clip, not an error — the same failure class `docs/DESIGN.md` already guards against. It must be
  asserted structurally, the way the correlation probe above did, not eyeballed.
- **Switching the AdaLN source affects the validated 31-step path too.** Keep `adaln_cache.safetensors`
  as the preferred source whenever the schedule matches it, and fall back to the curve form only for
  schedules it cannot serve. That keeps the existing measured results reproducible bit-for-bit.
- **Licensing.** The LoRA is Apache-2.0, but the curve tensors come from Comfy-Org's redistribution
  of MiniMax H3 weights, still under the MiniMax H3 Community License with its territorial
  exclusions. Same posture as the README: fetch your own, redistribute nothing.

---

## B. TAE — tiny autoencoder for previews

### Verdict: feasible and cheap — 9.8 MB, ~10–15 hours, removes the 7-frame floor entirely

Preview goes from **49.3 s / 8.46 GB** to an estimated **0.1–0.4 s / under 1 GB**, and can decode
any single frame rather than a mandatory 7-latent-frame prefix from the clip's start.

### Evidence

#### Architecture: ComfyUI's stock TAESD decoder, widened, with a fourth upsample

`Kijai/MiniMax-H3-TAE/vae_approx/taeh3.safetensors`: 81 tensors, **9.78 MB**, all F32,
**2.45 M parameters**, **decoder only** (no encoder).

It is `comfy/taesd/taesd.py`'s `Decoder`, whose reference source I read to confirm the module
shapes: `Block = Sequential(conv3x3, ReLU, conv3x3, ReLU, conv3x3)`, plus
`skip = Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else Identity`, plus `fuse = ReLU`;
`Clamp = tanh(x/3)*3`. The tensor names match exactly (`N.conv.0/2/4.weight`, `13.skip.weight`), and
**index 13 is the only one carrying a `skip`** — the only stage where `n_in != n_out` (96→64). That
single detail pins the whole layout:

```
0  Clamp                       12 conv(96,96,bias=False)
1  conv(24 -> 96)              13 Block(96 -> 64)   <- only skip
2  ReLU                        14,15 Block(64,64)
3,4,5   Block(96,96)           16 Upsample x2
6  Upsample x2                 17 conv(64,64,bias=False)
7  conv(96,96,bias=False)      18,19 Block(64,64)
8,9,10  Block(96,96)           20 Upsample x2
11 Upsample x2                 21 conv(64,64,bias=False)
                               22 Block(64,64)
                               23 conv(64 -> 3)
```

Reconstructing the parameter count from that layout gives **2.44 M** against the header's 2.45 M
(the gap is biases I skipped) — the architecture is fully determined, not guessed.

#### It does map H3's latents to RGB, at the right ratio

- **Input 24 channels** = `config.json`'s `latents_dim: 24`. Confirmed by `1.weight [96, 24, 3, 3]`.
- **Four 2x upsamples = 16x spatial**, exactly the video VAE's `spatial_compression_ratio`
  (`upstream/minimax_h3_mlx/video_vae.py`: `spatial_downsample_factors = (2, 2, 2, 2, 1, 1)`).
  1344/16 = 84, 768/16 = 48 — integer, as required.
- **Output 3 channels** = RGB, at full 1344x768.

#### No 7-frame floor — it has no temporal dimension at all

Every layer is a 2D convolution. There is no causal padding, no chunking, no
`2 * chunk_tokens - token_drop` constraint. One latent frame in, one RGB frame out. This removes
both current constraints at once: `h3_48gb/preview.py` presently decodes the minimal 7-latent-frame
prefix and must start at frame 0; with the TAE, any frame index is decodable independently, so a
preview can show the moment that is actually interesting.

#### Cost

At native 1344x768, latent grid 84x48, summed over the layout above:

- **265.4 GMAC = 0.53 TFLOP per frame**
- largest activation `1344x768x64` = 132 MB in fp16 (264 MB f32)
- weights 9.8 MB f32 / 4.9 MB f16

At 1.5–6 TFLOPS effective for 3x3 convs on this GPU that is **0.09–0.35 s per frame**. Even
allowing a 10x miss on convolution efficiency it stays in the low seconds — against 49.3 s today.
Peak memory is dominated by a handful of 132 MB activations, so **under 1 GB**, against 8.46 GB
today. Decoding all ~31 latent frames of a 5 s clip for an animated preview would cost roughly
3–10 s.

#### Portability to MLX

Straightforward. It ships as safetensors F32 with PyTorch conv layout `[O, I, kH, kW]`; MLX's
`nn.Conv2d` wants `[O, kH, kW, I]` with NHWC activations, so a transpose at load. Five module kinds
total (Clamp, conv3x3, ReLU, Block, nearest Upsample). No PyTorch dependency is needed — numpy is
enough to read and transpose the weights.

#### A better but heavier alternative

Kijai's own README says: *"Quickly trained 2D tine VAE for MiniMax-H3. Not the greatest outcome,
still beats latent2rgb for preview purposes."* He points to madebyollin's proper one
(`madebyollin/taehv`, `safetensors/taeh3.safetensors`): 128 tensors, 22.7 MB, F16, encoder +
decoder. Its header shows it is TAEHV — temporally aware:

- final `decoder.22.weight [12, 64, 3, 3]` = 3 RGB x **4 temporal**, matching the VAE's
  `temporal_compression_ratio` of 4
- blocks take `2C` inputs (`decoder.3.conv.0.weight [256, 512, 3, 3]`) — the MemBlock pattern that
  concatenates the previous frame's memory
- 1x1 `TGrow` convs at `decoder.7/13/19` doing 2x temporal upsampling twice

Higher fidelity and four pixel frames per latent frame, but it needs the recurrent memory and
streaming machinery ported, and ships an encoder we do not need. **Recommendation: port Kijai's
2D decoder first** (it is a preview, and it is a third of the work), and treat the TAEHV port as a
later upgrade if preview quality proves insufficient.

### Work required

| # | Task | Est. |
|---|---|---|
| 1 | `h3_48gb/taeh3.py`: MLX TAESD decoder (Clamp / Block / Upsample / conv), weight loader with the `[O,I,kH,kW]` → `[O,kH,kW,I]` transpose. | 4–6 h |
| 2 | Calibrate the latent convention against the real VAE (see risks) — decode one real latent both raw and mean/std-normalized, compare. | 2–3 h |
| 3 | Wire into `preview.py` as the default path; keep the real-VAE decode as an option and the existing latent2rgb fallback. Drop the 7-frame floor from the preview request path. | 3–4 h |
| 4 | Tests (shape/ratio, golden-frame regression, fallback ordering). | 2–3 h |
| | **Total** | **~11–16 h** |

Download: **9.8 MB**.

### Risks

- **The latent convention is the one real unknown.** The TAE was trained against ComfyUI's H3 latent
  tensor; our pipeline carries `latents_mean` / `latents_std` (24-vectors) in
  `video_vae/config.json`. Whether the TAE expects raw VAE latents or normalized ones is not
  documented anywhere I could find. It is cheap and decisive to settle empirically — decode one real
  latent both ways and compare against the real VAE's output — but it must be done before trusting
  any preview. A wrong convention gives a plausible, wrongly-colored image, not an error.
- **Preview quality is explicitly approximate.** The author's own assessment is "not the greatest
  outcome". It should be used to judge composition, motion and color, not fine detail. Keep the
  real-VAE decode reachable for when a trustworthy frame is wanted.
- **Decoder-only.** Kijai's file cannot encode. We do not need encoding, but it forecloses any
  future use as a cheap latent round-trip.
- **Architecture is inferred from tensor shapes**, not from a config file — though the `13.skip`
  argument and the 2.44 M vs 2.45 M parameter check make it as close to certain as an inference
  gets. Confirm against a real decode before building on it.

---

## Appendix: where the numbers came from

| Claim | Source |
|---|---|
| LoRA tensor names / shapes / metadata | safetensors headers, HTTP range read (57,480 B and 52,936 B) |
| Pruned conversion deletes adaln | its own `__metadata__` `warning` field |
| QKV slab vs interleave | correlation of dequantized base rows vs range-read ComfyUI rows |
| Merge destroys the delta at Q4 | `B@A` rms vs `scales` median from our own shards |
| 66.28 GB / 26.14 GB / all-14-shards | `MiniMaxAI/MiniMax-H3` index + all 14 shard headers, range-read |
| 87.3 MB curve path | `Comfy-Org/MiniMax-H3` pruned bf16 header |
| Curve fidelity 2.0–2.3e-3 | reconstruction vs local `adaln_cache.safetensors`, blocks 0/17/33/49 + final |
| Tolerance comparison | `upstream/minimax_h3_mlx/quantize.py` measured table |
| Shifts 12.0 / 3.0, sampler equivalence | `ComfyUI-MiniMax-H3-Turbo/__init__.py` vs `upstream/.../pipeline.py`, `scheduler.py` |
| TAE architecture | its header + `comfy/taesd/taesd.py` reference source |
| TAE cost | MAC count over the reconstructed layout at 84x48 latent |
