# Memory

This fork exists because MiniMax-H3 does not fit in 48 GB the way upstream loads it. Everything
below is what keeps it fitting, in the order it matters.

## Measure it with the right instrument

Three tools report this process's memory and they disagree by two orders of magnitude. During a
live run that Activity Monitor showed at **29.13 GB**:

| Tool | Reported | Why |
|---|---|---|
| `ps -o rss=` | **0.13 GB** | MLX allocates through Metal; those buffers are not resident pages of the process |
| `vm_stat` "wired" | 12.3 GB | system-wide wired memory, not this process, and not all of the allocation |
| Activity Monitor | **29.13 GB** | counts unified-memory allocations — the real figure |
| `mx.get_active_memory()` | matches Activity Monitor | MLX's own accounting, available inside the process |
| `footprint -p <pid>` | matches Activity Monitor | the same figure from outside the process, scriptable, **and it reports the peak** |

`footprint -p` is the one to reach for from a script. It breaks the allocation down by category —
during a native run, 24 of 25 GB sat under `IOAccelerator (graphics)`, which is exactly the part
`ps` cannot see — and its `phys_footprint_peak` line is the number a multi-hour run has to be
judged on, since the peak is what decides whether the machine swaps. `/tmp/memtrack.sh` samples it
once a minute into `~/h3-outputs/memory.tsv` alongside the run's current phase, so a peak
can be attributed to the phase that caused it rather than to the run as a whole.

Measured peaks, 48 GB machine, seven tracked runs:

| Run | Peak | Steady through diffusion |
|---|---|---|
| 1344x768, 10 s, Turbo LoRA | 27-28 GB | 24-27 GB |
| 768x768, 2.4 s | 27 GB | 15 GB |
| 896x512, 2.4 s | 27 GB | 14 GB |
| 640x640, 2.4 s | 27 GB | 14 GB |
| 512x512, 2.4 s (x3) | 27 GB | 13 GB |

**The peak is the text encoder, on every canvas.** It does not move: 27 GB whether the run is
512x512 or 1344x768, because the encoder's 28.22 GB of weights dwarf anything diffusion holds. On
the small canvases diffusion runs at 13-15 GB, half the peak. This was predicted from the residency
table below and is now measured — and it means the canvas you can afford is not decided by the
peak at all, only by how long the diffusion takes.

The corollary is worth stating plainly: **nothing between 512x512 and 1344x768 is memory-limited.**
Every canvas tried fits in the same 27 GB. The constraint on this machine is wall clock.

The precision question is where that corollary stops applying, and the distinction matters. An
8-bit DiT is 21.35 GB resident against the 4-bit build's 11.34 — nearly twice — and on a
896x512 / 2.4 s run the peak does not move: **27 GB either way**, because the text encoder still
dominates and diffusion at 8 bits reaches only 24 GB.

That is a statement about *short clips on a middling canvas*, not a general one. The ten extra
gigabytes are free only while diffusion stays under the encoder's 27 GB, and the activations
section below projects a 15 s native clip past 24 GB of activations on its own. Against a 21.35 GB
DiT that is ~46 GB on a 48 GB machine: at that size the 8-bit build *is* the peak, and the headroom
it costs is the binding constraint rather than a footnote. An earlier version of this paragraph
called the diffusion-phase cost "true and irrelevant"; it is true and conditionally relevant, and
the condition is exactly the long native runs this fork most wants to reach.

`memory.report()` prints MLX's numbers, and that is the measurement of *this process's* footprint.
`ps -o rss=` on this process is not a sanity check for that — it is blind to Metal-backed
allocations, which is exactly the 0.13 GB row above. That blindness is specific to reading a single
MLX process's own memory, though: summing `ps -eo rss,command` across *other* processes is a
legitimate way to see what else is competing for the machine, which is exactly what the
`ps -eo rss,command` snippet under "What else competes for the 48 GB" below does — those other
processes are not running MLX, so their RSS is a real measurement, not a blind one. An earlier
version of `docs/DESIGN.md` built a whole table on *this process's* RSS and concluded diffusion
peaked at 10–11.5 GB. It does not — that was the fraction of the allocation that happened to be
file-backed resident pages.

## Phase residency

The four components are needed in disjoint phases. Loading all of them costs 45.9 GB of weights
before a single activation; loading each as its phase begins and dropping it as the phase ends
costs whatever the largest single phase costs.

| Phase | Held | Released at the end by |
|---|---|---|
| Text encoding, ~10 s | encoder **28.22 GB** | `LazyTextEncoder.encode` |
| Keyframe encoding | video VAE **5.21 GB** | `_release_vae_after` |
| Diffusion, hours | DiT 11.34 + LoRA 0.62 + table 0.13 = **12.09 GB** | `_decode_video` |
| Video decode | video VAE **5.21 GB** | `_decode_audio` |
| Audio decode | audio VAE **0.61 GB** | end of run |

Each unload is preceded by `mx.eval` on whatever the phase produced, and that ordering is the
whole mechanism. MLX is lazy: until the result is materialized it is a graph over the module's
parameters, and dropping the module frees nothing. `test_unload_without_eval_would_not_free`
measures this against the allocator, with a deliberately-lazy control to show the test has teeth.

`release()` then runs `gc.collect()` before `mx.clear_cache()`, in that order. Module trees form
reference cycles that refcounting does not break — measured: without the collect, the 28.2 GB
encoder survives into the diffusion loop.

Three of these unloads were added after the fact, and none of them announced its absence:

- the video VAE sat through the entire diffusion loop after encoding keyframes;
- the transformer sat through decoding, which is where the peak lands on a long clip;
- the video VAE was still resident when the audio VAE loaded.

Nothing raised. The run simply needed more memory than the machine had, which reads as slowness,
not as a bug.

## The allocator cache

MLX keeps freed buffers rather than returning them, because reuse beats reallocation. Unbounded,
that drifts upward across a long run: 29.13 GB observed against 12.09 of weights and roughly 9.3
of activations, so about **8 GB was cache the run had no use for**.

`memory.limit_cache()` caps it at 2 GB, applied when the pipeline is constructed. Override with
`H3_CACHE_LIMIT_GB`; zero disables the cache entirely, which is right on a machine already
swapping and wrong everywhere else.

The cost of not doing this is not an error, it is variance. In the run that exposed it, one step
took **818 s where its neighbours took 568** — the machine had 0.36 GB free and 3.98 GB swapped.
Freeing 1 GB elsewhere brought the next step back to 586 s.

## What else competes for the 48 GB

Worth checking before a multi-hour run, because the model's own budget assumes it has the machine:

```bash
ps -eo rss,command | awk '{s[$2]+=$1} END {for (a in s) if (s[a] > 500000) print s[a]/1048576" GB", a}'
```

In one measured case a browser held 4.05 GB across six processes and four editor sessions held
1.7 GB. Closing the editors alone moved free memory from 0.36 GB to 1.68 GB and swap from 3.98 to
3.56, and the step time recovered.

## The machine must not sleep

A multi-hour run needs `caffeinate -dimsu` around it. Nothing in this fork takes a sleep
assertion, and the generate process does not look like activity to macOS: it holds the GPU at full
power while the keyboard, the display, and the network sit idle, which is exactly the profile
`powerd` reads as "nobody is here".

The consequence is not a slow run, it is a kernel panic. Measured, on the 1344x768 10 s run of
2026-08-10:

```
02:20:51  DarkWake from Deep Idle             <- it had already been asleep
02:26:56  Sleep: 'Dark Wake Thermal Emergency'
02:34:58  DarkWake from Deep Idle
02:35:43  Sleep: 'Maintenance Sleep'
02:36:28  panic: GFX NMI FIQ - agx_power(6) - failed to transition to state 0 (_iopStatus=7)
```

Forty-five seconds after entering sleep, the GPU firmware failed to power down and took the
kernel with it. The run died at step 2 of 7 and every log in `/tmp` died with the reboot.

Two things fall out of this that are worth knowing separately:

- **A gap in `memory.tsv` means the machine slept.** The tracker's `sleep 60` does not advance
  through system sleep, so the gaps at 02:04-02:11, 02:11-02:21 and 02:26-02:35 are not a stalled
  sampler — the last of them matches the 02:26:56-02:34:58 sleep to the minute. Read a gap as an
  event, not as missing data.
- **Sustained native-resolution load can trip a thermal emergency.** The 02:26 sleep was
  `Dark Wake Thermal Emergency`: the machine woke for maintenance with the GPU job still resident,
  could not shed the heat in dark wake, and bailed out. Staying awake avoids this too, because an
  awake machine runs its fans; it does not make the thermal ceiling go away.

Memory was not involved and the tracker proves it: 28 GB peak of 48, 14-22 GB free, swap flat at
3.31 GB through the last sample before the panic. The failure mode this section describes looks
nothing like the allocator problems above and will not be caught by watching them.

Logs therefore do not belong in `/tmp`. `~/h3-outputs/_logs/` survives a reboot;
`/tmp` does not, and a panic is exactly when the log matters most.

## What is not solved

**Activations scale with the packed sequence, not with anything convenient.** At 37,657 rows they
are roughly 9.3 GB, and the sequence grows with pixels *and* frames — so a 15 s clip at native
resolution is projected past 24 GB of activations on top of the 12.09 of weights. That is the
ceiling this fork does not lift; see `docs/RESULTS.md` for the wall-clock consequence.

**`iogpu.wired_limit_mb`** was raised to 44 GB during early measurement and resets on reboot.
Nothing that has actually run came near needing it, so reproducing anything here does not require
changing it. Methodology, not a prerequisite.
