# Feasibility: face-refine pass (ComfyUI-H3-FaceRefine recipe)

Spike for the BACKLOG.md item "Face-refine проход — рецепт из ComfyUI-H3-FaceRefine" (source:
https://github.com/Carasibana/ComfyUI-H3-FaceRefine — take the idea, not the code). Read-only:
every claim below comes from reading `h3_48gb/`, `upstream/minimax_h3_mlx/`, `scripts/bake_adaln.py`
and the files already on disk under `~/models/turbo/`. No GPU was touched, nothing was run, nothing
was downloaded — the AdaLN curve tensors this plan depends on were already fetched for the Turbo
LoRA work (`docs/FEASIBILITY-turbo-tae.md`) and sit at `~/models/turbo/adaln_curve.safetensors`.

**Verdict up front: feasible, and the pre-requisite the backlog worried about is not a blocker.**

| stage | verdict | new code |
|---|---|---|
| A. v2v hook (noised real latent as the seed) | feasible, no blocker | ~1 new seam, everything it composes already exists |
| B. Partial-schedule AdaLN table | feasible, free (curve already on disk) | small script addition |
| C. `video_vae.encode` on a real crop | already works, already parity-tested | none |
| D. CV wrapper (detect/track/crop/paste) | feasible, standard OpenCV | all new — most self-contained stage |

Total estimate **~38–58 h** across the four stages below, plus a first experiment (§ "first cheap
experiment") that costs minutes, not hours, and should be run before committing to the rest.

---

## A. The v2v hook — is "inject a real latent" actually blocked?

**No.** The backlog's own framing — "у нас есть кейфрейм-механика — близко, но не то же" — is
correct that keyframe conditioning is the wrong shape for this job, but every primitive a real
img2img/v2v hook needs already exists in the codebase, individually built and tested for a
different purpose. The work is composition, not invention.

### Why keyframe conditioning is the wrong shape

`_encode_keyframes` (`upstream/minimax_h3_mlx/pipeline.py:165-207`) encodes an image into
**conditioning rows** that are noised once to `t = max(t, KEYFRAME_NOISE_AUG) = 0.999`
(`__call__`, lines 274-280) and then **held fixed at that near-clean level for every step** — they
are never denoised, only re-imposed (`video_rows[:n_cond_v]` is never written back, see
`h3_48gb/pipeline.py`'s module docstring: "only generated rows are ever written back"). That is
correct for "this frame anchors the clip" and wrong for face-refine, which needs the crop to be
**denoised from a partial noise level down to 0**, i.e. live in the *generated* row block, not the
*conditioning* block.

### The primitives the generated-row path needs, and where each already exists

1. **Forward-noise a real latent to an arbitrary level.**
   `MiniMaxH3Scheduler.scale_noise` (`upstream/minimax_h3_mlx/scheduler.py:115-122`) is exactly
   `x_t = t*x0 + (1-t)*noise` in this model's convention. It is already called once, in
   `__call__` (upstream `pipeline.py:276-280`), to noise keyframe anchors to `t=0.999`. Calling it
   at a lower `t` (i.e. a higher starting sigma) to noise a *real* clip's latent to
   denoise-strength 0.3–0.5 is the same function, same call site pattern, different argument.

2. **A schedule that doesn't start at sigma 1.0.**
   `MiniMaxH3Scheduler.set_timesteps(sigmas=...)` (`scheduler.py:68-103`) only requires "at least
   two strictly decreasing values ending at 0.0" (line 95-98) — it does **not** require the first
   value to be 1.0. A schedule like `[0.45, 0.28, 0.12, 0.0]` is legal today, with zero changes.

3. **Adopting a non-uniform, non-1.0-start grid at the pipeline level, for free.**
   This is the strongest finding. `h3_48gb/pipeline.py`'s `_build_schedules` override
   (lines 428-459) already works by building a throwaway uniform grid and then **wholesale
   replacing it** with whatever grid the AdaLN cache file was baked for, whenever the lengths
   match:
   ```python
   grids = self._cached_sigma_grids()
   if grids is not None and len(grids[0]) == num_inference_steps:
       video.set_timesteps(sigmas=grids[0])
       audio.set_timesteps(sigmas=grids[1])
   ```
   Nothing here assumes the cache's grid starts at 1.0 or is uniform — it already has to handle
   non-uniform grids (the `tail-split` schedules `scripts/bake_adaln.py` bakes today are
   non-uniform by construction). A partial, low-starting-sigma grid is the same kind of object as
   the tail-split grids this code already serves. **No change needed here at all** — only a new
   cache file with the right grid (§ B).

4. **Encoding the real footage into a latent.** § C — already works, already tested.

5. **Packing a latent into transformer rows.** `patchify_video_latents`
   (`upstream/minimax_h3_mlx/packing.py:169-...`) — already the function `__call__` uses for the
   ordinary noise seed; reused unchanged.

### What is genuinely new

- A sibling entry point to `__call__` (or a new pipeline method) that, instead of

  ```python
  latents = mx.random.normal((1, C, num_latent_frames, latent_height, latent_width))
  video_rows = patchify_video_latents(latents, patch_size)
  ```

  does

  ```python
  real_latent = normalize(video_vae.encode(crop_pixels))          # § C, existing function
  noised = scheduler.scale_noise(real_latent, t=1 - sigma_start, noise)  # existing function, new t
  video_rows = patchify_video_latents(noised, patch_size)          # existing function
  ```

  with `n_cond_v = 0` (no keyframe anchors — simpler than i2v, not harder).

- Geometry for this call **cannot** come from `duration_seconds → align_num_frames`, because the
  crop's frame count is already fixed by the source clip, not chosen by the request. This needs
  its own small geometry derivation from the crop's own `(F, H, W)` rather than reuse of
  `__call__`'s duration math. (Note: a 243-frame crop, as the user asked to be evaluated,
  satisfies `243 = 17*14 + 5` — the pipeline's own native `17n+5` grid — so
  `video_latent_num_frames(243) = 72` exactly, no padding/truncation surprises at that layer. This
  looks like a deliberate choice on the CV side, not a coincidence, and is worth keeping.)

- RNG-draw-order discipline. This codebase has a documented history of exactly this class of bug
  — two supposedly-identical keyframe encodes differing by 0.87 and then 0.83 because a seed was
  drawn in the wrong place relative to a loop (`h3_48gb/pipeline.py:293-331`). The new noise draw
  for the v2v seed needs the same care, though it does not need to be bit-reproducible against
  anything upstream — there is no upstream reference for this feature, so "matches itself under
  `h3 resume`" is the only bar, and the checkpoint module's own `test_checkpoint.py` pattern shows
  how this project verifies that class of claim.

- If a face-refine sub-run is meant to be checkpoint/resumable in its own right (worthwhile for a
  long clip), it needs its own `checkpoint_identity_extra()` fields (crop source digest, denoise
  strength, starting sigma) so it can never collide with or be silently resumed against the outer
  generation's checkpoint. `h3_48gb/checkpoint.py`'s identity system was built to make exactly
  this kind of mismatch a loud `CheckpointMismatch` rather than a silent wrong resume — it just
  needs new fields, not new mechanism.

**One clarification on the backlog's own phrasing:** the checkpoint/resume machinery
(`ResumableScheduler` / `StepInterceptor` in `h3_48gb/checkpoint.py`) is not the vehicle for this
feature, even though it is the existing code that most resembles "start mid-schedule with
externally-supplied rows." It exists to resume *the same run* after a crash, and its identity
checks (`_check_seam`, `request_identity`, weight fingerprints) are built around that guarantee —
repurposing its on-disk format to smuggle in a foreign real-video latent would be fighting the
abstraction, not reusing it. What *is* worth reusing is the proof it constitutes: this codebase
already runs and unit-tests "denoise correctly from a pre-set intermediate state, for some steps
skipped/pre-seeded" as a supported code path. That derisks the new hook considerably even though
the new hook should be its own small piece of code, not a repurposed checkpoint file.

---

## B. Schedule: does the 8-step AdaLN table reach denoise 0.3–0.5?

**No — but baking a table that does costs nothing new.** Two separate findings:

### The current baked table has no usable point in that range

Computed directly from `MiniMaxH3Scheduler`'s own formula (`scripts/bake_adaln.py`'s `simple` path
== `upstream/minimax_h3_mlx/scheduler.py:_linspace_1_to_0` + shift), the 8-step video grid
(`shift=12.0`, what `adaln_8_l100.safetensors` / `adaln_8_plain.safetensors` were baked for) is:

```
[1.0, 0.9863, 0.9677, 0.9412, 0.9, 0.8276, 0.6667, 0.0]
```

There is **no grid point between 0.667 and 0** — the schedule's own last interval is famously the
one `h3_48gb/pipeline.py`'s docstring calls out ("spends its last forward jumping from sigma 0.667
straight to 0"). Two consequences for a v2v pass wanting denoise 0.3–0.5:

- `AdaLNCacheFile.check_schedule` (`h3_48gb/adaln.py:209-247`) matches grids by **exact float
  equality** against the baked table, on purpose (§ its own docstring: "never by a nearest-value
  search, so a schedule the cache cannot serve raises `ScheduleMismatch` instead of quietly
  resolving to the closest row"). A request to start at sigma 0.4 against this table fails loudly,
  not approximately.
- Even the nearest usable point, 0.667, would run the whole refine in **one** Euler step straight
  to 0 — a denoise strength of ~0.667, not the requested 0.3–0.5, and one step across that big a
  gap is exactly the kind of jump the tail-split schedule (below) exists to avoid.

### Baking a dedicated partial table is already-generic, already-cheap machinery

`scripts/bake_adaln.py`'s `bake()` function is not hard-coded to full 1.0-starting schedules — it
already takes an arbitrary `video.set_timesteps(sigmas=...)` list for its `tail-split` and `beta`
modes (lines 60-104), interpolating `~/models/turbo/adaln_curve.safetensors`'s
`[1025, 8]` `adaln_t_table` at whatever `t = 1 - sigma` values the requested grid implies
(`_interp`, lines 27-29). Nothing in `bake()` assumes the grid starts at 1.0. A short partial
schedule for face-refine — e.g. 4–6 points from sigma 0.5 (or 0.3) down to 0 — bakes exactly the
same way as today's `simple`/`beta`/`tail-split` presets, needs one new small preset function (a
few lines, mirroring `tail_split_sigmas`), and costs:

- **No download** — the curve tensors (87.3 MB) and the silu grid (5.5 MB) are already on disk.
- **No GPU** — baking is pure CPU interpolation over an 8-dimensional curve, the same operation
  that already produced the three tables currently in `~/models/turbo/` in seconds each.
- Optionally folds the Turbo LoRA's AdaLN half (`bake()`'s `lora=` argument already supports this,
  and `h3_48gb/turbo.py` already implements the backbone-half runtime side path) so the face-refine
  pass itself can run in as few as 2–4 forwards on a small crop, not just be schedule-partial.

The one thing worth auditing before relying on this: a few call sites read `video_sigmas[0]`
implicitly as "1.0" in spirit even though nothing enforces it (e.g. `BAKED_GRID_POINTS` in
`h3_48gb/cli.py:41`, and human-facing messaging like `check_schedule`'s "N grid points drive N-1
forwards" framing). None of this breaks a partial table technically — `_build_schedules` adopts
whatever grid the file holds, verbatim — but CLI/doctor messaging that currently assumes "steps"
means "grid points from full noise" should say "from denoise-strength start" for a partial table,
or a user will misread `--steps 4` as a from-scratch 4-step run.

---

## C. `video_vae.encode()` on a 448×288, 243-frame crop

**Already works, and already parity-tested — nothing to build here.**

- `VideoVAE.encode()` (`upstream/minimax_h3_mlx/video_vae.py:571-594`) takes arbitrary
  `(B, C, F, H, W)` pixels, no special-casing for keyframes: it pads `F` up to a multiple of
  `clip_length=17` by repeating the last frame, runs `_encode_clip` per 17-frame chunk (which
  itself tiles spatially above `tile_sample_min_height/width=256` — a 448×288 crop is tiled
  automatically, same as any native-resolution decode already does), then drops
  `token_drop=3` trailing latent frames. This is the **full multi-chunk path**, distinct from the
  single-frame `_encode_clip` call `_encode_keyframes` uses today — the pipeline just never wires
  it up, because nothing has needed a multi-frame *input* before.
- 448 / 16 = 28 and 288 / 16 = 18 — both exact multiples of `spatial_compression_ratio = 16`, so no
  edge-padding surprises at the spatial boundary.
- **Already numerically pinned against the diffusers reference**: `upstream/tests/
  test_video_vae_parity.py` pushes the MLX model's own parameters through
  `convert_video_vae_key` into the reference implementation and compares outputs — this exercises
  `_encode_clip`, which `encode()` calls per-chunk, so the multi-chunk path is not an unverified
  code path, only an unused one.
- What face-refine needs to add here is the same posterior-sampling discipline
  `_encode_keyframes` already implements once (`upstream/minimax_h3_mlx/pipeline.py:165-207`):
  sample from `(mean, logvar)`, not take the mode; round through float16 to match the reference's
  own numerical path; normalize with `video_vae.config.latents_mean/std`. All three lines already
  exist for the single-frame case and generalize to a multi-frame tensor without new math.

---

## D. CV wrapper: YOLO detect, smoothing, crop, feathered paste

**Nothing exists in this repo yet for this stage** — it is the one genuinely from-scratch piece,
and also the most self-contained: it needs no MLX, no model weights, and can be built and unit
tested with zero GPU runs, matching this spike's own constraint.

- `opencv-python` (5.0.0.93) and `scipy` (1.18.0) are **already installed** in `.venv` — not
  because of any existing face-refine work, they are simply already present (not even listed in
  `pyproject.toml`'s dependencies, so worth adding explicitly once used). `scipy.signal
  .savgol_filter` for the center/size smoothing the recipe calls for is free.
- `ultralytics` (and the torch it drags in) is **not installed**, and this project has otherwise
  kept torch out of its runtime on purpose — `upstream/requirements.txt` comments it out with
  "Validation only — the port itself needs neither torch nor diffusers." The backlog names
  "ultralytics, Mac ok" explicitly, and CPU/MPS YOLO inference on a face-sized crop is genuinely
  fast (well under a second per frame), so this is not a hard problem — just the single new
  heavyweight dependency in an otherwise torch-free project. Worth trying first, since it needs
  zero new dependencies: `cv2.FaceDetectorYN` (YuNet), which ships inside `opencv-python` itself as
  an ONNX face detector, sub-10 ms/frame on CPU. It only does faces (not general YOLO
  person/object detection), but that is exactly what this pass needs.
- The rest — sliding-window (`savgol`) smoothing of the crop's center and size across frames
  separately (the recipe's "раздельные окна центр/размер"), rectangular crop extraction, feathered
  paste-back of **only the face box**, not the whole refined crop (the README detail the backlog
  flags as non-obvious and worth keeping: "SAM-маска обычно ХУЖЕ прямоугольника — шов прячется в
  волосах"), simple colour-match at the seam, and `fade_out` on frames with no detection — is
  standard OpenCV/NumPy, independent of the DiT/VAE entirely. It is gradable stage-by-stage
  (detect → track/smooth → crop → paste) with synthetic or pre-existing footage, before any face-
  refine generation exists to test it against.
- `docs/RESULTS.md`'s own "seed-spread number is withdrawn" postmortem already independently
  arrived at wanting "measure inside a detected face or body crop rather than over the whole
  frame" as a future metric — the face-detect step this stage needs is the same primitive that
  would unblock that, a second use for the same new code.

---

## Work plan, by stage

| # | Stage | Task | Est. |
|---|---|---|---|
| 1 | v2v hook | New geometry-from-crop entry point; real-latent encode+noise seeding of `video_rows` (composes §A.1/2/5, all existing functions); RNG-order discipline; checkpoint-identity fields if resumable | 8–14 h |
| 2 | AdaLN table | New partial-sigma preset in `scripts/bake_adaln.py` (mirrors `tail_split_sigmas`); bake + numerically spot-check 2–3 candidate denoise-start tables against the curve; audit `--steps`/doctor messaging for the "partial, not full" case | 4–6 h |
| 3 | CV wrapper | Face/YuNet or YOLO detect over source clip; savgol-smoothed center/size tracking; crop extraction at working resolution; feathered face-box paste-back + colour match; `fade_out` on undetected frames | 10–14 h |
| 4 | Splice + colour | Align the (denoise-shortened) refined crop's own timeline back onto host frame indices; blend at crop edges vs. face-box only; verify no drift over a 243-frame / ~10 s span | 4–6 h |
| 5 | CLI / web integration | `h3 face-refine` subcommand or `--face-refine` flag; web-morda control for denoise strength; dry-run/doctor gate mirroring the existing `schedule_not_baked` guard so a missing partial table fails before 20 s of weight-loading, not after | 6–10 h |
| 6 | Tests + first validation | Unit tests per stage (permutation/shape checks in the style of `test_checkpoint.py`); first real GPU run per "first cheap experiment" below | 4–6 h dev + minutes of wall time |
| | **Total** | | **~38–58 h** |

---

## First cheap experiment (run before committing to the rest)

The one question code-reading cannot answer: this checkpoint's distilled schedule and its Turbo
LoRA were fit to **the model's own generation trajectory** — real footage encoded through the same
VAE may sit at a different point in latent space (different noise statistics, different
high-frequency content) than anything the 8-step/curve-baked modulation table has ever seen at
that sigma. That could make "denoise a real crop" behave differently from "denoise a partially-
generated one," and only a real forward pass answers it.

Recommended minimal experiment, in order of cost:

1. **Bake one partial table** — e.g. 4 points from sigma 0.4 to 0, shift 12.0/3.0, optionally with
   the Turbo LoRA folded in. CPU only, seconds.
2. **Encode + noise one real crop** — take a face crop from an already-rendered clip
   (`docs/media/`), run it through `video_vae.encode()` + `scale_noise(t=0.6)` in a throwaway
   script. VAE-only, no DiT, cheap (this alone validates § C's claims against real data rather than
   just against the parity test's synthetic tensors).
3. **One short v2v forward pass** — 448×288 crop, 3–4 forwards, and look at whether the result
   reads as "the same face, more detail" rather than "a different face" or "no visible change."
   This is the actual go/no-go signal for the other ~30–45 h in the table above.

---

## Risks

- **Real-footage-vs-generated-latent mismatch** (above) — the one risk that genuinely cannot be
  resolved by more reading; the cheap experiment exists specifically to retire it early.
- **`check_schedule`'s exact-match gate is strict by design** (good: it already refuses to
  silently serve a mismatched table — see `h3_48gb/adaln.py:169-178`'s `ScheduleMismatch`) but a
  few user-facing messages (`BAKED_GRID_POINTS`, doctor/dry-run text) implicitly read "steps" as
  "from full noise." A partial table must not be silently presented to a user as a normal `--steps
  4` full run.
- **Audio can't be dropped structurally.** Video and audio are denoised jointly in one forward
  (`h3_48gb/pipeline.py`'s own module docstring); a face-refine pass pays for an audio track it
  will discard. Cheap at 3–4 forwards on a short crop, but not zero, and not currently possible to
  disable without touching `build_packed_sequence`.
- **New dependency weight** if the literal `ultralytics`/torch route is taken on an otherwise
  torch-free project; `cv2.FaceDetectorYN` is a lighter substitute worth trying first for the
  face-only case.
- **Checkpoint-identity interplay.** A face-refine sub-run must carry its own
  `checkpoint_identity_extra()` fields (crop source digest, denoise strength, starting sigma) so
  it can never be mistaken for, or accidentally resumed against, the outer generation's own
  checkpoint. This needs explicit new fields, not new mechanism — but forgetting them is exactly
  the silent-wrong-resume failure class `h3_48gb/checkpoint.py` was built to rule out everywhere
  else.
- **Geometry derivation duplication.** The v2v entry point cannot reuse `__call__`'s
  `duration_seconds → align_num_frames` path (§A) and must derive latent geometry from the crop's
  own shape instead — a second place this math can drift from `packing.py`'s, the same class of
  risk `h3_48gb/pipeline.py`'s `_install_preview` docstring already flags for its own re-derivation
  of the same geometry, with the same mitigation available (a shape-check test, as
  `test_preview.py` does there).

---

## Эксперимент 2026-08-17: вердикт РАБОТАЕТ

Дешёвый эксперимент из §«первый эксперимент» проведён; риск №1 («real-footage-vs-generated-latent
mismatch») снят. Дистиллированное расписание с Turbo LoRA ведёт себя в v2v на VAE-латентах
вменяемо во всём диапазоне sigma 0.30–0.50: то же лицо, больше деталей; расплава, подмены
личности, дрейфа и мерцания нет.

**Механика (подтверждена):**
- Партиальная таблица бейкается без доработки `scripts/bake_adaln.py`: `bake()` принимает
  произвольные сигмы через ветку tail-split (обёртка подменяет `tail_split_sigmas`). Сетка
  ставится в несдвинутом домене и сдвигается на модальность — video (shift 12) и audio (shift 3)
  остаются на тренировочной диагонали. Бейк 0.7 с CPU, 87 МБ (3 форварда). Таблицы с LoRA 1.0:
  `~/models/turbo/adaln_face_s0{30,40,50}_4pt_turbo.safetensors`; сетка s0.40 = [0.400, 0.304, 0.176, 0].
- Кроп 448×288, 56 кадров (17·3+5 — ровно на нативной сетке, 17 латент-кадров). Путь §A работает:
  encode → сэмпл постериора → f16 round-trip → нормализация → scale_noise(t=1−sigma) →
  patchify → цикл с n_cond_v=0 → decode.
- **Аудио засевать не обязательно**: чистый шум vs реальное аудио — различие в пределах шума
  (1.39×/28.89 dB против 1.40×/28.92 dB). Этап 1 может не делать audio_vae.encode.

**Числа** (лапласиан-деталь ×, PSNR к исходнику, межкадровая дельта ×):
только VAE 1.09×/38.8 dB/0.99× · s0.30 1.32×/30.6/1.11× · **s0.40 1.40×/28.9/1.13× (оптимум)** ·
s0.50 1.47×/27.5/1.17× (появляется «хрусткость»). Прирост детали — от DiT, не от кодека;
PSNR второй половины клипа выше первой — накопительного дрейфа нет.

**Ресурсы:** 2.1–2.2 мин на проход (3 форварда × 17 с), пик 28.4 ГБ — и это текстовый энкодер,
не диффузия (последовательность всего 2445 строк).

**Поправки к плану этапов:**
- Детектор: `cv2.CascadeClassifier` удалён в OpenCV 5.0.0; `cv2.FaceDetectorYN` требует внешний
  ONNX (YuNet) — «нулевых новых зависимостей» не будет: качать веса YuNet или ultralytics.
- Источник эксперимента — собственная генерация через h264+VAE-энкод; структурная часть риска
  закрыта, статистика реальной камеры — нет (при необходимости нужен внешний клип).

**Артефакты:** `~/h3-outputs/face-refine-exp/` — COMPARE-src-vs-sigma040.mp4,
COMPARE-all5.mp4, STILLS-face-zoom-3frames.png, temporal_check.png, refine_s0{30,40,50}.mp4,
SOURCE-crop.mp4 + скрипты bake_partial.py / v2v_face_refine.py / vae_roundtrip.py / metrics.py.

## Раунд 2 (2026-08-17, плохое лицо): рефайн работает там, где нужен

Пользователь глазами опроверг вывод раунда 1 на хорошем лице: там исходник лучше рефайна
(sigma добавляют жёсткость/«гравюру», лапласиан меряет перешарп как «деталь»). Раунд 2 на
целевом кейсе — мыльное лицо ~40–75 px с малого канваса, кроп 168×108 + лансцош ×2.67 до
448×288 — развернул картину: **рефайн очевидно лучше исходника**. Два источника (sovi-s1
статичная камера; centaur-official бегущая женщина, трекинг-кроп).

- **sigma 0.25 — оптимум и жёсткий потолок**: из каши собираются веки, ресницы, брови, нос,
  губы, зубы; лицо остаётся тем же. 0.15 — мягче, тоже выигрыш. 0.40 — «сочиняет»
  (открывает глаз с радужкой там, где был прищур; возвращается гравюрность раунда 1).
- **Рефайн лечит мыло, но не уродство**: деформированный рот исходника становится резким
  деформированным ртом. В UI нужна честная оговорка.
- **Лапласиан бесполезен как критерий** (голый VAE даёт ×1.28 без единой новой структуры;
  ×1.40 в раунде 1 значило «хуже», в раунде 2 — «радикально лучше»). Решают глаза; полезны
  межкадровая дельта (×1.00–1.05 — мерцания нет) и PSNR как мера ухода от исходника
  (24–27 dB осмысленно, 22 dB на 0.40 — уже сочинение).
- Стоимость: 2.1 мин на окно 56 кадров, пик 28.4 ГБ (текстовый энкодер). Шот 5 с ≈ 3 окна
  ≈ 6 мин на лицо; ролик 30 с с 1–2 лицами — 40–70 мин GPU: отдельный проход, не бесплатный.
- Новые таблицы: ~/models/turbo/adaln_face_s015_4pt_turbo.safetensors и ..._s025_...

**Решающий вопрос перед ~40 ч обвязки — согласованность между окнами** (продакшн-шот =
3–7 окон; совпадёт ли сочинённая структура на стыке). Тест: два перекрывающихся окна
(0–55 и 28–83) при s0.25, сравнение зоны перекрытия + жёсткий стык. Если щёлкает и не
лечится кроссфейдом — пункт закрывается.

Артефакты: ~/h3-outputs/face-refine-exp/round2/ (STILLS-face-zoom-source{1,2}.png,
STILLS-vae-control.png, STILLS-temporal.png, COMPARE*-all4.mp4, v2v_r2.py).

## Раунд 3 (2026-08-17, стык окон): кроссфейд лечит — СТРОИТЬ

Два перекрывающихся окна (0–55 и 28–83, перекрытие 28 кадров, вход в зоне перекрытия
побитово одинаков) при sigma 0.25: окна сочиняют порознь только «вес»/контраст деталей
(55% расхождения — низкочастотная светотень), но не их положение — геометрию задаёт общий
исходный латент, окна ко-регистрированы по построению. Жёсткий стык слабо щёлкает
(+1.41σ motion-compensated step, ниже худшего собственного перехода окна; глазами — шаг
локального контраста, не подмена структуры). **Кроссфейд 12 кадров по декодированным
пикселям лечит полностью** (стык становится ровнее самих окон, гостинга нет — усреднение
даёт бровь, а не две брови), прирост рефайна при этом сохраняется.

Требования к боевой обвязке из раунда:
- Кроссфейд ≥12 кадров обязателен; перекрытие 12–16 достаточно (шаг окна 40–44, −35% GPU
  против шага 28).
- Прямоугольник кропа — функция номера КАДРА, не окна (иначе ко-регистрация рассыпается);
  на сильном движении с трекинг-кропом проверить отдельно.
- sigma 0.20–0.25, потолок 0.25; честная оговорка в UI (лечит мыло, не деформацию).
- Краёв окна обрезать не нужно (краевых эффектов нет).

Артефакты: ~/h3-outputs/face-refine-exp/round2/overlap/ (METRICS.txt, STILLS-*,
SPLICE-hard.mp4, SPLICE-crossfade.mp4, скрипты mc_step.py / crossfade.py и др.).

**Итог трёх раундов: ценность подтверждена, блокирующих рисков нет — строить боевой
face-refine (этапы 1–3 из плана выше, ~38–58 ч, с параметрами из раундов 2–3).**

---

## Итог 2026-08-18: ЗАПАРКОВАНО после боевых ворот

Конвейер построен целиком (facetrack + facepaste + facerefine + CLI `h3 face-refine`,
коммиты d17ff736..d300b1e5, 1213 тестов) и прошёл два раунда интеграционных ворот.
Механика подтверждена: движок стабильно воспроизводит эталон экспериментов
(×1.46 детали на кропе, PSNR 26.6 dB), кадр вне лица побитово чист, аудио синхронно,
стыки окон лечатся кроссфейдом (блокер D1 — несброс _step_index шедулера между окнами —
найден воротами и закрыт d300b1e5).

**Причина парковки — D2, дефект формата доставки, а не движка**: дорисованная деталь
живёт на кропе 448×288, а вклейка возвращает её в рамку лица нативного кадра с даунскейлом
2–7×, который прирост уничтожает. Замеры (лапласиан лицевой рамки, результат/исходник):
448×288 (лицо 47 px) — 0.62; 896×576 (60 px) — 0.67; 1344×768 (81 px) — 1.00 (нейтрально,
глазами худшие кадры пятнистые). Гипотеза «на большом канвасе ценность выживет» проверена
и не подтвердилась. Дополнительно D3: трекер подменяет субъекта на смене плана (нужна
отсечка сцен). A/B промпта: при sigma 0.25 промпт не влияет (38.4 dB).

**Условие разморозки**: апскейл-выдача — проход отдаёт ролик в увеличенном разрешении,
лицо вклеивается в нативном масштабе кропа (прирост сохраняется по построению). Оценка
4–6 ч + ворота. До этого веб-кнопку (Task 6) не делать.

Артефакты ворот: ~/h3-outputs/face-refine-exp/production-test/ (+ round2/).
