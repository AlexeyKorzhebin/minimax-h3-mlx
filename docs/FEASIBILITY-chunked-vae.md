# Feasibility: chunked / streaming video VAE decode and encode

Assessment of porting Kijai's chunked VAE I/O (ComfyUI core, merged 2026-08-09) into this fork's
MLX video VAE. Nothing here was implemented. Every number below came from the merged PR's own diff
and benchmark table, from this repository's existing measurements (`docs/RESULTS.md`,
`docs/MEMORY.md`, `h3_48gb/preview.py`), and from arithmetic against the released VAE config. No
run was made for this study.

**Verdict up front.**

| | Verdict | Work | Peak saving, native 10 s |
|---|---|---|---|
| **Decode** | Feasible, and the mechanism is already half here | ~13–18 h | **~20–26 GB** |
| **Encode** | Feasible, low value for our current paths | ~2–3 h | ~1–3 GB, v2v only |

The headline is not the same as Kijai's. He saved 2.9 GB of VRAM by removing three full-video
copies from an *eager* framework. **We stand to save an order of magnitude more, because MLX's
laziness means our existing chunking currently saves nothing at all** — `VideoVAE.decode` builds all
14 chunks and all 392 tile decodes as one unevaluated graph and hands it to a single `np.array()`,
which then re-fuses into one allocation event. The chunk loop is already written; it simply has no
`mx.eval` in it, so it is decorative.

**The second finding is arguably more urgent than the first.** Of the ~30 GB the decode contributes
to the `metal::malloc 51.4 GB > 48 GB` crash, roughly **12 GB is plain numpy** in
`MiniMaxH3Pipeline._decode_video` — four full-size float32 temporaries applied to the whole video
after the VAE has already returned. On unified memory that is the same 48 GB the GPU is competing
for. It is four lines and it is the cheapest thing on this list to delete.

---

## 1. What Kijai actually changed

PR [Comfy-Org/ComfyUI#15446](https://github.com/Comfy-Org/ComfyUI/pull/15446) "Optimize MiniMax-H3
VAE", merged 2026-08-09, +68 −45 across `comfy/ldm/minimax/vae.py` and `comfy/sd.py`. Its own
benchmark, 175 frames at 1344x768:

| | before | after |
|---|---:|---:|
| decode VRAM peak alloc | 3485 MB | **607 MB** |
| decode RAM peak | +7089 MB | **+2320 MB** |
| decode time | 14.6 s | 13.9 s |
| encode VRAM peak alloc | 5714 MB | **2402 MB** |
| encode time | 40.5 s | 40.9 s |

"Outputs verified bitwise identical to master (end-user pixel space) across latent lengths, speed
remains about same."

### 1a. There is no feature caching, no causal-conv state passing, and no new blending

This is the single most important thing the diff settles, and it settles it in our favour. The task
brief asked how causal-conv state crosses a chunk boundary. **It doesn't, and it never did.**

- The **decoder is a non-causal ViT**, not a CNN. There is no temporal convolution state to carry.
  Continuity across chunks comes from *re-decoding an overlap*: each chunk is decoded from
  `tokens_chunk_size + token_overlap` latent frames instead of `tokens_chunk_size`, and the
  duplicated pixel frames are linearly cross-faded. That is `token_drop`'s entire purpose.
- The **encoder is causal 3D conv**, but chunks are encoded *independently*, each with its own
  zero-padded causal lead-in. No state crosses a clip boundary there either; the model was trained
  that way.
- **Spatial tiling is untouched by the PR.** 256 px tiles, 64 px minimum overlap, linear blend —
  before and after.

The PR did not change the chunk plan, the overlap, the blend, or the tiling by one line. It changed
**where the results are written**.

### 1b. The four changes that matter

1. **A preallocated output buffer on the host.** `decode_temporal` used to lazily allocate `dec` on
   the *GPU* at the first chunk and assemble the whole video there. Now `decode_output_shape()`
   derives the frame count from the chunk plan up front, the buffer is allocated on
   `intermediate_device()` (CPU), and each chunk is `copy_()`d straight into its slice. The full
   video never sits in VRAM.
2. **Per-chunk finalization.** The new `_finalize_pixels` applies `*pixel_std + pixel_mean` and
   `clamp(0,1)` to *one chunk*, inside the write. Previously that ran once over the assembled
   full-length tensor — and in float32, so it also created a second full-size copy via `.float()`.
3. **`process_output` set to identity** in `sd.py`. Decode now emits `[0,1]` directly instead of
   `[-1,1]`-then-rescale, deleting one more whole-video pass in the VAE wrapper.
4. **Encode streams its input.** `encode(x, device=None)` keeps the pixel video on CPU and moves one
   17-frame clip to the GPU at a time; normalization and tail padding moved inside the loop so the
   full padded copy is never built. Plus a `sd.py` change capping the memory *estimate* to one
   chunk, so ComfyUI's model manager stops evicting models it did not need to evict.

### 1c. The deferred write is subtle and must be copied exactly

`write_part` is not called on chunk *i* during iteration *i*. The last `frame_overlap` frames of
chunk *i* are held in `dec_overlap` and only written after chunk *i+1* has been blended into them.
Writing each chunk eagerly as it is produced would quantize the overlap region before the blend and
put a visible seam every 20 pixel frames. This is the one piece of ordering that must survive the
port intact.

---

## 2. What our VAE does today

`upstream/minimax_h3_mlx/video_vae.py` (656 lines). Causal 3D CNN encoder, 36-layer ViT decoder at
dim 2048, channels-last internally.

### 2a. The chunk plan, confirmed by arithmetic

From the released config (`clip_length` 17, `temporal_compression_ratio` 4, `token_drop` 3):

```
tokens_chunk_size  = ceil(17/4)   = 5      frame_pre_padding = (-17) % 4 = 3
token_overlap      = (-3) % 5     = 2      frame_overlap     = 2*4 - 3   = 5
```

For a native 10 s clip at 1344x768:

| quantity | value |
|---|---:|
| pixel frames | 243 |
| encode clips (17 frames each, padded from 243 to 255) | 15 |
| latent frames after `token_drop` | **72** |
| decode chunks | **14** |
| latent frames fed per chunk decode (`5 + 2`) | 7 |
| pixel frames produced per chunk (`7 * 4`) | 28 |
| spatial tiles per chunk decode (1344 → 7, 768 → 4) | **28** |
| tile decodes in one full video decode | **392** |

This reproduces the brief's "72 latent frames into 243" exactly, and the 28 tiles quoted in
`h3_48gb/preview.py`.

### 2b. So we already chunk and already tile — and it buys nothing

`VideoVAE.decode` (lines 596–656) is a correct, complete temporal chunk loop with overlap and linear
cross-fade. It has the same `overlap` deferral Kijai has. What it lacks is any evaluation barrier:

```python
decoded, overlap = [], None
for i in range(num_chunks):
    clip = self._decode_clip(...)      # lazy
    ...
    decoded.append(chunk)              # lazy
out = mx.concatenate(decoded, axis=1)  # lazy
return out.transpose(0, 4, 1, 2, 3)    # lazy
```

and the caller, `upstream/minimax_h3_mlx/pipeline.py:368`, is

```python
frames = np.array(self.video_vae.decode(latents.astype(mx.float32)))
```

One `np.array()` on a graph containing 392 tile decodes, 14 stitches, 13 blends, a 14-way
concatenate and a transpose. **In an eager framework a chunk loop streams by construction. In MLX a
chunk loop without `mx.eval` is a no-op**: all 14 branches are live simultaneously at the
concatenate, and nothing forces the scheduler to run them one at a time.

This project already knows this. `h3_48gb/pipeline.py:337` documents the identical trap for the
encoder — *"until the rows are materialized they are a lazy graph over the VAE's parameters, and
dropping the module would free none of them"* — and `_release_vae_after` exists solely to place the
`mx.eval`. The decode path never got the same treatment.

### 2c. And then there are four more full-video copies, in numpy

`upstream/minimax_h3_mlx/pipeline.py:368–374`:

```python
frames = np.array(...)                                  # 3.01 GB   f32 host copy
frames = frames * pixel_std + pixel_mean                # +3.01 GB
frames = np.clip(frames, 0.0, 1.0)[0].transpose(...)    # +3.01 GB
return (frames * 255.0 + 0.5).astype(np.uint8)          # +3.01 GB f32 temp, +0.75 GB u8
```

This is precisely the redundancy Kijai deleted with `_finalize_pixels` and the identity
`process_output` — except we have four passes where ComfyUI had one, and on unified memory there is
no "it's only host RAM" consolation.

---

## 3. The numbers for us

### 3a. The one hard measurement we already own

`docs/RESULTS.md` ("TAE previews") and `h3_48gb/preview.py` record a preview decode at 1344x768,
which is *exactly one chunk* (7 latent frames, 28 tiles): **49.3 s, 8.46 GB peak, 5.21 GB of that
being resident weights** (`docs/MEMORY.md`, phase residency).

That gives the number the whole estimate turns on:

> **one decode chunk's transient working set ≈ 8.46 − 5.21 = 3.25 GB, at any clip length.**

A timing reconciliation worth noting: 14 × 49.3 s = 690 s, but the measured full decode is 208 s
(`docs/RESULTS.md:446`). The preview figure loads and unloads the 5.21 GB module around each call
(`h3_48gb/preview.py:133–138`), so ~34 s of it is load. Per-chunk *compute* is then **208 / 14 ≈
14.9 s**, which is self-consistent. Worth confirming, since it is the basis for saying chunking
costs no time.

### 3b. Where the ~30 GB goes

The crash recorded in `BACKLOG.md:44` is `metal::malloc 51.4 GB > 48 GB` with 21 GB of resident
8-bit DiT — leaving **~30.4 GB attributable to the decode**. Reconstructing that for 243 frames
(full-video float32 = 3.01 GB, one chunk float32 = 0.347 GB):

| term | now | after |
|---|---:|---:|
| VAE weights (resident) | 5.21 | 5.21 |
| live chunk transient | 3.25 (× up to 14 if branches interleave) | **3.25** |
| 14 chunk outputs held for the concatenate | 4.86 | 0 |
| concatenate result | 3.01 | 0 |
| `transpose` materialization | 3.01 | 0 |
| `np.array()` host copy | 3.01 | 0 |
| numpy denorm / clip / scale temporaries | ~12.0 | 0 |
| output buffer | 3.01 (f32) | **0.75 (u8)** |
| **total** | **~37, ≥31 if the scheduler serializes perfectly** | **~9.2** |

The observed 30.4 GB sits inside that band, which is the corroboration this estimate needed —
the mechanism is not merely plausible, it accounts for the crash quantitatively.

**Expected saving on a native 10–15 s clip: ~20–26 GB.** Stated conservatively against the
serialize-perfectly lower bound, ~21 GB; against the realistic figure, ~26 GB.

### 3c. The part that actually matters: length independence

Everything in the "now" column except the weights and one chunk scales linearly with clip length.
Summed, the length-dependent terms are **~107 MB per output frame** (≈ 8.6 full-video float32
copies at 12.4 MB/frame). After the change the only length-dependent term is a uint8 host buffer at
**3.1 MB/frame** — a **34x reduction in the slope**.

| clip | frames | decode peak now (est.) | after (est.) |
|---|---:|---:|---:|
| 5 s | 121 | ~21 GB | ~8.8 GB |
| 10 s | 243 | ~37 GB | ~9.2 GB |
| **15 s** | **~360** | **~47 GB** | **~9.6 GB** |

**A 15 s native decode does not fit in 48 GB today even with nothing else resident.** That, not the
20 GB on the 10 s case, is the real argument: chunking is what makes long native clips decodable at
all, which is exactly Kijai's "peak stops depending on length" claim landing harder on our side.

---

## 4. Mapping: 1:1, adapt, invent

| Kijai's piece | Ours |
|---|---|
| temporal chunk plan, overlap, cross-fade | **already have it**, line-for-line equivalent |
| spatial tiling | **already have it**, same 256/64 |
| causal state / feature cache across chunks | **does not exist in either** — nothing to port |
| `_decode_temporal_chunks` factored out | trivial refactor of our lines 610–621 |
| `decode_output_shape` (frame count *before* decoding) | **must be written.** We compute the trailing pad only *after* the fact (lines 648–655); preallocation needs it forward |
| `_finalize_pixels` per chunk | **must be written**, and should absorb all four numpy passes |
| deferred `write_part` / `dec_overlap` ordering | our `overlap` variable already defers — keep it, just write instead of append |
| `output_buffer` on `intermediate_device()`, `.copy_()` into a slice | **adapt.** `mx.array` is immutable; no in-place slice write into a device buffer. The natural sink is `np.empty((F,H,W,3), np.uint8)` filled per chunk |
| `encode(x, device=...)`, input stays on CPU | **worth zero for us.** Unified memory: our numpy array and our `mx.array` are the same physical bytes. There is no transfer to avoid |
| `sd.py` memory-estimate capping | **no analogue.** We have no model manager that preemptively evicts |
| — | **`mx.eval` per chunk. Nothing in Kijai's diff corresponds to this, and it is the entire load-bearing change for us.** |

Two consequences of unified memory, stated plainly, since the brief asked:

- **Kijai's VRAM/RAM split is meaningless here.** His decode win reads as "3.49 → 0.61 GB VRAM plus
  7.09 → 2.32 GB RAM"; for us those are one pool and the honest statement is a single ~10.6 → ~2.9 GB
  figure on his geometry. Our own gain is larger for reasons that are ours, not his.
- **His encode win is almost entirely a discrete-GPU artifact.** Moving one clip at a time across
  PCIe saves 3.3 GB on his card and nothing on ours. Our encode gain is only the per-clip `mx.eval`
  (15 clip subgraphs not co-existing) and not materializing the padded full-video copy — and it
  applies to v2v only, since `_encode_keyframes`
  (`upstream/minimax_h3_mlx/pipeline.py:191–207`) calls `_encode_clip` on single frames and never
  goes through `encode()`. Low priority.

---

## 5. Where the code should live

There is no VAE file in `h3_48gb/` — the video VAE exists only under `upstream/`, and
`upstream/tests/test_video_vae_parity.py` diffs `model.decode(...)` against diffusers at 2e-4. The
fork's established pattern is to subclass and override in `h3_48gb/` (`LazyMiniMaxH3Pipeline`
overrides `_decode_video`; `adaln.py`, `tae.py` sit beside it), keeping `upstream/` pristine.

So: a `h3_48gb/chunked_vae.py` holding a `VideoVAE` subclass with the streaming decode, plus an
override of `_decode_video` in `h3_48gb/pipeline.py` for the numpy tail. `decode()` must keep
returning one whole array so the parity test and `preview.py` are unaffected — the streaming form
is the primitive, the array form a thin wrapper.

---

## 6. Work required

| # | Task | Est. |
|---|---|---|
| 1 | Factor `_decode_temporal_chunks(z_len)` out of `decode`, and add the forward frame-count function (Kijai's `_decode_temporal_frame_plan`). Prove it equals the existing post-hoc `pad_frames` arithmetic for every latent length 7…200. | 3–4 h |
| 2 | `decode_streaming(z, sink)`: per chunk decode → blend with the deferred overlap → `mx.eval` → hand the finalized chunk to `sink`. Keep `decode()` as a wrapper so the parity test and preview are untouched. | 4–6 h |
| 3 | Fold denorm + clip + scale + transpose + uint8 into the per-chunk finalize; rewrite `_decode_video` to preallocate one `np.empty((F,H,W,3), np.uint8)` and fill it. Delete the four numpy temporaries. | 3–4 h |
| 4 | Bit-exactness test: old path vs new on a fixed latent, `np.array_equal` on the uint8 output — *plus* a targeted seam test on the 5 blended frames at each of the 13 chunk boundaries. | 3–4 h |
| | **Critical path (unblocks kot-1344)** | **13–18 h** |
| 5 | Encode side: per-clip `mx.eval`, pad inside the loop. v2v only. | 2–3 h |
| 6 | Per-chunk `PhaseTracker.mark`; assert the chunk peak is *flat* across chunk index — length independence is the actual claim, so test it as one. | 2 h |
| 7 | Validation: 512x512 decode (minutes), then native 1344x768 / 10 s under `memwatch.sh`. | 2–3 h wall |
| | **Total** | **~20–27 h** |

No downloads. No new weights. Nothing to fetch.

---

## 7. Risks

- **The crash is unattributed, and one plausible cause is not chunking at all.**
  `h3_48gb/pipeline.py:363–368` unloads the DiT *before* decoding, precisely so decode does not pay
  for it — yet `BACKLOG.md:44` records 21 GB of DiT resident at the OOM. Either the crash was a
  *preview* decode (which runs mid-loop with the DiT deliberately resident, per
  `h3_48gb/preview.py:27–32`), or the 8-bit build's `unload()` is not releasing what it claims. **If
  it is the latter, that is a bigger and far cheaper win than this whole document**, and it should be
  measured before a line of chunking is written. This is the top risk not because the analysis is
  wrong but because the fix might be aimed at the wrong 20 GB.
- **The deferred-overlap write is a silent-failure surface.** Get the ordering wrong and the output
  is a plausible video with a faint seam every 20 frames — not an error. Same failure class
  `docs/DESIGN.md` guards against, and the reason task 4 tests the boundary frames specifically
  rather than trusting an aggregate diff.
- **Bitwise identity is *not* free for us the way it was for Kijai.** He only moved allocations. We
  are additionally collapsing four numpy float32 passes into one per-chunk finalize. The map is
  elementwise so it *should* be exact, but float32 op ordering (`(x*s+m)` clipped then scaled, vs.
  fused) can differ in the last ulp and then round differently at the uint8 boundary. Expect to
  either prove exactness or accept a documented ≤1 LSB tolerance.
- **`mx.eval` barriers could cost wall time.** 14 forced synchronizations prevent cross-chunk
  scheduling. Each chunk is 28 tiles × 36 ViT layers, so there should be ample work to hide latency,
  and the current 208 s may itself contain memory-pressure stalls that removing 25 GB would relieve —
  `docs/RESULTS.md:446` already flags decode as the place to look for the 2.56x gap against the other
  port. But "same speed" is Kijai's measurement on an eager framework, not ours, and it must be
  re-measured, not assumed.
- **The numpy sink does not generalize.** Writing the output host-side is right today (the result
  goes straight to ffmpeg) but forecloses keeping decoded frames on device for any future GPU-side
  consumer. Acceptable; worth knowing it is a one-way door.
- **`decode()` must keep its exact current signature and return.**
  `upstream/tests/test_video_vae_parity.py:128` and `h3_48gb/preview.py:135` both call it and both
  expect one array. The streaming form has to be additive.

---

## 8. What to measure first

In order, because each answer changes what the next task should be:

1. **Instrument the crash before fixing it.** `PhaseTracker.mark` around the final decode *and* each
   preview decode on a kot-1344-shaped run; confirm which one OOMs and whether `dit.unload()` at
   `h3_48gb/pipeline.py:364` actually returns the 21 GB. Cheap, and it decides everything above.
2. **Confirm the 3.25 GB per-chunk transient with the real checkpoint**, and confirm it is *flat* as
   chunk count grows. Decode 7, 12, 17 and 22 latent frames and watch `mx.get_peak_memory()`. If the
   peak is flat, the whole estimate in §3b holds; if it grows, the excess is inside a chunk and
   chunking will not help.
3. **Delete the four numpy passes on their own, first.** It is a self-contained change worth ~12 GB
   and it needs none of the chunking machinery. Measure it in isolation before building task 2 on
   top of it.
4. **Time one chunk in isolation** to settle 14.9 s vs 49.3 s (§3a), so the "chunking costs no time"
   claim rests on a measurement rather than a subtraction.

---

## Appendix: where the numbers came from

| Claim | Source |
|---|---|
| PR identity, diff, +68/−45, merged 2026-08-09 | `gh pr view/diff 15446 --repo Comfy-Org/ComfyUI` |
| 3485→607 MB, 5714→2402 MB, +7089→+2320 MB, timings | PR #15446 description (author's own benchmark, 175x1344x768) |
| No feature cache / no causal state across chunks | the diff itself — chunk plan, overlap and blend are byte-identical before and after |
| chunk_tokens 5, token_overlap 2, frame_overlap 5 | `upstream/minimax_h3_mlx/video_vae.py:453–457`, recomputed |
| 243 frames → 72 latents → 14 chunks; 28 tiles | arithmetic over `VideoVAE.decode` / `_split_tiles`, matching `h3_48gb/preview.py:18` |
| one-chunk decode 49.3 s / 8.46 GB | `docs/RESULTS.md` ("TAE previews", 451–463) |
| VAE weights 5.21 GB resident | `docs/MEMORY.md` phase residency, 69–81 |
| full decode 208 s at 1344x768 | `docs/RESULTS.md:446` |
| `metal::malloc 51.4 GB > 48 GB`, 21 GB resident DiT | `BACKLOG.md:44` |
| MLX laziness pins module parameters | `h3_48gb/pipeline.py:331–344`, `357–368` (this repo's own prior finding) |
| four full-video numpy passes | `upstream/minimax_h3_mlx/pipeline.py:368–374` |
| keyframe encode bypasses `encode()` | `upstream/minimax_h3_mlx/pipeline.py:191–207` |

---

## Поправка по замеру 2026-08-14: чанкование НЕ оправдано

Контрольный замер на синтетическом латенте (скрипт `decode-peak-probe.py`, нативные 1344×768,
полный путь `_decode_video` с постобработкой) опроверг оценку выше:

| длительность | пик MLX | прирост RSS (numpy) | время |
|---|---|---|---|
| 2,4 с (73 кадра) | 9,6 ГБ | +3,2 ГБ | 199 с |
| 5 с (124 кадра) | 11,0 ГБ | +5,5 ГБ | 348 с |
| 10 с (243 кадра) | 15,4 ГБ | +10,8 ГБ | 695 с |

Оценка «37–47 ГБ» не подтвердилась: MLX освобождает промежуточные буферы по счётчику ссылок
даже при одном `np.array()` на весь граф — «все 14 веток живы одновременно» оказалось неверным
выводом. Выигрыш Kijai — специфика дискретной GPU (VRAM↔RAM), в unified-памяти его нет.
Вдобавок «вчерашний OOM kot-1344» оказался демо-фикстурой живой проверки UI, а не реальным
прогоном — атрибуции падения не существует, падения не было.

Что реально растёт с длиной — numpy-хвост постобработки (+~1,07 ГБ RSS на секунду ролика,
float32-копии всего видео). Лечится переносом clip/scale/uint8 внутрь MLX-графа до
`np.array()` — минимальная правка вместо 13–18 часов порта. Порт чанкования снят с плана;
вернуться к нему стоит только если появятся ролики 30+ секунд (экстраполяция пика MLX
~0,64 ГБ/с + фикс-хвоста упирается в 48 ГБ около 45–50 с).

Урок в копилку методологии (рядом с «выпечкой финального слоя» из RESULTS): оценка стояла на
двух непроверенных опорах — экстраполяции транзиента превью и фикстурной ошибке, — и замер
за 20 минут снял обе. Мерить до того, как планировать часы.
