# Latency Pass — MLX-accelerated transcription/refine + paid-Gemini parallelism

- **Date:** 2026-07-05
- **Status:** Design approved (brainstorming); pending spec review → writing-plans
- **Owner:** clips/ pipeline
- **Approach:** A (MLX-local ASR + paid-Gemini parallelism), no new cloud keys

## 1. Problem

A cold first run of a 1-hour YouTube video through the clipper takes ~10 min; a warm
re-clip (cached structure) ~7 min. The user's goal: **under a minute per hour-long video.**
The user confirmed output *quality is acceptable* (the eval judge over-flags); latency is the
product blocker.

## 2. Goal & non-goals

**Goal.** Maximum *zero-quality-loss* speedup on this machine (Apple M2, 8 cores). Concretely:
- **Warm re-clip → ~60-70s** (under a minute) — the sub-60s number that is actually reachable at full quality.
- **Cold new video → ~250-330s** (~4-5 min → ~4 min). Explicitly **accepted as not <60s**.

**Preserve every shipped quality feature:** multimodal perception, Pro topic selection
(`gemini-3.1-pro-preview`), precise silence-snapped cuts, discourse-onset boundaries, and — where
we keep supadata — the YouTube-caption transcript text that feeds understanding/titles.

**Non-goals.**
- Cold <60s (physically blocked: video download ~65s + Gemini vision ~90-120s each already exceed 60s).
- Dropping any quality feature (no `PRECISE_BOUNDARIES=0`, no `REFINE_WHISPER_MODEL=base`, no `ANALYSIS_PROFILE=fast`, no `MULTIMODAL=0`).
- Changing *which* clips are selected or their content. All changes are hardware/schedule/caching or an A/B-gated ASR swap.

## 3. Constraints (user-chosen)

- **Zero quality loss**, *proven* by A/B gates — not assumed.
- **No new cloud STT keys.** Groq would be simpler/equally fast but was declined; stay fully local. (Groq/parakeet reach the same endpoint; the local MLX path is self-contained, no per-use cost, works without a valid Groq key.)
- **Paid Gemini tier is available** → free-tier RPM=30 no longer serializes fan-outs.
- **Hardware:** Apple M2, 8 cores, MLX-capable GPU. `faster-whisper` here is CPU int8 (no Metal).

## 4. Root cause (code-verified; W0 confirms with measured numbers)

The dominant cost is the **boundary-refine pass**: `faster-whisper medium` on **CPU int8** re-transcribing
~50 clip-edge windows (~330s), while the **M2 GPU sits idle**.

- `orchestrator.py:331,378` — `refine_clip_boundaries` runs **only** when `transcript.source == "supadata"`.
- It exists because `transcribe_supadata` (`transcribe.py:185-197`) **synthesizes** per-word timings
  (proportional within each ~180-char caption chunk); they are too coarse to cut on.
- `boundary.py::_whisper_window` does the expensive per-window transcription; `_energy_min_snap`
  does the cheap RMS silence placement. **Only the transcription is the cost.**

Estimated current budget (hosted config: `OUTPUT_MODE=cut SCENE_DETECTION=0`, defaults otherwise).
**All to be replaced by W0 measurements.**

| Stage | Warm | Cold | Runs when |
|---|---|---|---|
| Transcribe (supadata) | ~2 (cached) | ~15 | always |
| Punctuation restore | ~2 (cached) | ~40 | supadata + PUNCTUATION |
| Video download | — | ~65 | MULTIMODAL/cut, cold |
| Vision keyframe captioning | — | ~90-120 | MULTIMODAL, cold |
| Extract units + deps | — | ~45 | full, cold |
| Assemble (topic: Pro select + flash author + judge) | ~85 | ~85 | CLIP_ENGINE=topic |
| **Refine (local CPU medium-Whisper)** | **~330** | **~330** | PRECISE_BOUNDARIES + supadata |
| Cut (ffmpeg VideoToolbox) | small | small | output_mode=cut |
| **Total** | **~420** | **~600** | |

The refine pass is the #1 rock in **both** warm and cold.

## 5. Design — workstreams

### W0 — Profiling + validation harness (FIRST, non-negotiable)

Everything downstream is gated on real numbers and a precision oracle.

1. **Per-stage profile.** Run one real cold + one warm hour-long video with `PROFILE_TIMINGS_FILE`
   set (machinery exists in `orchestrator.py::_record`). Produces the measured budget that replaces
   §4's estimates and identifies the true sink. Use a known long lecture; record cold (fresh
   `work/<id>` wipe) and warm (cached structure).
2. **Boundary-precision A/B oracle.** For a fixed set of clips (a golden video, e.g.
   `output/4yvfd8aoUBc`), compare final cut times from the current path (supadata + CPU-refine) vs
   each candidate (MLX-refine; later parakeet). Metrics per edge:
   - Δ start_ms, Δ end_ms vs the current shipped cut.
   - "Lands in inter-word silence?" using `_energy_min_snap` RMS as an independent oracle.
   - **Gate:** a candidate ships only if edge deltas are human-imperceptible (target ≤ ~50 ms and
     no edge moves onto a word) on the golden set.
3. **Wall-clock A/B.** Standard per-lever timing on one real video (the config's existing
   methodology). Structure REBUILD is the dominant run-to-run noise — time *stages*, not just E2E.

Deliverable: `backend/eval/` additions (boundary A/B harness reusing `golden`/`run_eval` plumbing)
and a short measured-baseline note committed alongside.

### W1 — MLX-accelerate the refine windows (core win, keeps supadata text)

Swap the refine transcription CPU→GPU **without changing the algorithm**.

- New config `REFINE_ASR_BACKEND = faster_whisper | mlx_whisper` (default `faster_whisper` until W0's
  boundary A/B passes, then flip default to `mlx_whisper`; keep `faster_whisper` as revert switch).
- MLX backend inside `boundary.py::_whisper_window` (or a small `refine_asr.py` seam): `mlx-whisper`
  (`mlx-community/whisper-medium` or `-large-v3`) with `word_timestamps=True`, on the M2 GPU.
  `_energy_min_snap` and all boundary logic **unchanged**.
- **Concurrency:** the GPU is single-stream, so `REFINE_WORKERS` thread-parallelism does not help and
  may contend. Run windows serially on GPU (each ~5-10× faster than CPU), or batch multiple windows
  via `lightning-whisper-mlx` batched decoding if W0 shows it helps. **Prewarm** the MLX model during
  the download/vision stage so its load is hidden.
- Keeps supadata YouTube-caption **text** → zero downstream change. Truest zero-quality-loss lever.
- Expected: refine ~330s → **~40-80s**, boundary-equivalent (gated by W0).

### W2 — Exploit paid Gemini tier (parallelism)

- Free-tier RPM=30 no longer applies; the vision / understanding / assembly fan-outs can run parallel
  instead of serializing under reactive 429 backoff (`gemini_client.with_backoff`).
- Make worker counts env-tunable (most already are) and **raise where W0 measures faster**:
  `VISION_WORKERS`, `UNDERSTAND_WORKERS`, `PUNCT_WORKERS`.
- **Keep `JUDGE_WORKERS=4`.** The config's own concurrency probe found the judge model *soft-throttles*
  even on paid tier and raising to 8 measured net-slower (`config.py:355-363`). Re-verify with W0; do
  not blind-raise.
- All Gemini calls are temp-0 and keyed → schedule-only, **output-neutral**.
- Expected: vision ~-30-50s, understanding ~-15s, assembly window fan-out faster.

### W3 — Persistent caches

- **(a) Disk verdict cache.** The judge verdict cache is in-memory (dies with the job). Add a disk
  cache keyed by `(unit_ids, text_hash, judge_model)` under `work/<id>/`. Re-runs skip judging.
  Byte-identical (temp-0, keyed).
- **(b) Assembled-clip result cache.** The topic engine is **query-independent and deterministic**
  given `(structure, settings)`, so a warm re-clip re-does the whole assemble+refine for the same
  output. Cache the final `clips_spec` (post-refine) per `(structure_hash, settings_hash)` →
  a true repeat becomes near-instant. Invalidated on structure/settings change. Byte-identical.

### W4 — Small cold-path wins

- **Prewarm** MLX/Whisper models during earlier stages.
- **`aria2c` external downloader** for yt-dlp multi-connection fetch (~-15s on the part of the download
  that spills past the speculative prefetch window). Same bytes.
- Multimodal + 720p kept. (480p is a separate quality call — deferred, not in scope.)

### W5 — Parakeet global transcript (optional, GATED phase)

Only pursued if W1-W4 do not reach a fresh warm-ish run <~70s and the user wants to push further.

- New `TRANSCRIBER = parakeet` using `parakeet-mlx` (`mlx-community/parakeet-tdt-0.6b-v3`),
  `chunk_duration≈120s, overlap≈15s` for the hour; `AlignedResult.tokens` give per-word start/end +
  confidence → real global timestamps.
- `source = "parakeet"` ⇒ `refine_clip_boundaries` **skips entirely** (existing gate) → the ~330s rock
  is *deleted*, not just accelerated. Optionally still run the cheap `_energy_min_snap` on parakeet
  timings for sub-frame placement.
- **Extra gate (beyond the boundary A/B):** a **WER/text A/B** — parakeet ASR text vs supadata
  YouTube captions on the golden set — because it changes the transcript feeding
  understanding/titles/selection. Ship only if text quality ≥ captions (parakeet-tdt v3 is SOTA
  English; usually ≥ auto-captions, possibly < human-uploaded captions).
- **Supadata remains the default/fallback** (graceful degrade preserved).

## 6. Data flow

Unchanged except the transcription hardware (and, under W5, the transcript source):

```
supadata transcript (text + coarse timing)          [W5: parakeet text + real timing]
  → build_sentences (punctuation)
  → understand (content_map ∥ perceive → structure)  [W2 parallelism]
  → assemble (topic engine)                          [W2 parallelism, W3a verdict cache]
  → refine: ASR windows + _energy_min_snap           [W1 MLX GPU]  [W5: skipped]
  → cut (ffmpeg VideoToolbox)
        ↑ W3b caches the post-refine clips_spec per (structure, settings)
```

## 7. New / changed config knobs

| Knob | Default | Purpose |
|---|---|---|
| `REFINE_ASR_BACKEND` | `faster_whisper` → `mlx_whisper` after W0 | W1 refine backend + revert switch |
| `MLX_WHISPER_MODEL` | `mlx-community/whisper-medium` | W1 model (or `-large-v3`) |
| `TRANSCRIBER=parakeet` | (opt-in) | W5 global ASR |
| `PARAKEET_MODEL` | `mlx-community/parakeet-tdt-0.6b-v3` | W5 model |
| `VERDICT_CACHE_DISK` | on | W3a |
| `CLIP_RESULT_CACHE` | on | W3b |
| `YTDLP_EXTERNAL_DL` | `aria2c` if present | W4 |
| `VISION_WORKERS` / `UNDERSTAND_WORKERS` / `PUNCT_WORKERS` | raised per W0 | W2 |
| `JUDGE_WORKERS` | 4 (unchanged) | judge soft-throttle |

MLX/parakeet deps are Apple-Silicon-only → **lazy-import + guard** so non-mac/CI paths fall back to
`faster_whisper` and the suite stays importable. Add to `requirements.txt` under an
Apple-Silicon-optional section (or a separate `requirements-mac.txt`).

## 8. Error handling / degrade / invariants

- MLX refine backend fails/unavailable → fall back to CPU `faster_whisper` refine (existing path).
- Parakeet fails/unavailable → fall back to supadata (existing path).
- `_run_full` → `_run_fast` graceful degrade **unchanged**.
- `unverified_kill = 0` invariant holds (no judge-logic change).
- Existing 760-test suite stays green; snapshot to `.backup/` before the boundary/transcribe edits.

## 9. Testing & validation

- **W0 boundary A/B oracle** gates W1 and W5 (edge Δ ≤ ~50 ms, no edge on a word).
- **W5 WER/text A/B** gates W5 additionally.
- **Wall-clock A/B** per lever (stage-timed) confirms each win and catches regressions (e.g. a worker
  bump that soft-throttles net-slower).
- New unit tests: `REFINE_ASR_BACKEND` switch + fallback; disk verdict cache hit/miss; result-cache
  invalidation on structure/settings change; MLX/parakeet lazy-import guard.
- Mutation-check the new cache-invalidation and fallback branches (project norm).

## 10. Projected budget (W0 confirms/replaces)

| | Today | W1-W4 (supadata text kept) | +W5 (parakeet, gated) |
|---|---|---|---|
| **Warm re-clip** | ~420s | ~110-130s (near-instant on a true repeat via W3b) | **~55-70s** |
| **Cold new video** | ~600s | ~280-330s | ~230-280s |

**Honest caveat.** Strict zero-*text*-change (W1-W4) lands warm ~2 min unless it's a true repeat
(W3b → near-instant). A *fresh* warm-ish run strictly <60s leans on **W5 (parakeet)**, which is
zero-loss only if it passes both A/B gates. Cold stays ~4-5 min → ~4 min; **cold <60s is out of
scope** (vision + download floors), and the user accepted this.

## 11. Rollout order

W0 (measure + oracle) → W2 + W3 + W4 (safe, output-neutral, bank immediately) → W1 (MLX refine,
gated by boundary A/B) → W5 (parakeet, gated by boundary + WER A/B, only if needed). Each phase is a
revert-switchable config flag.

## 12. Decisions log

- Target = max zero-quality-loss speedup; warm ~under a minute, cold ~4 min accepted. *(user)*
- No new cloud STT keys; local MLX only (Groq declined despite equal speed / less code). *(user)*
- Paid Gemini tier available → exploit parallelism. *(user)*
- W5 (parakeet) included as a **gated** final phase, not default. *(user)*
