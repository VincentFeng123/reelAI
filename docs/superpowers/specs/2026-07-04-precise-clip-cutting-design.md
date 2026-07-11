# Precise clip cutting — silence-snapped, window-extending boundaries — design

**Date:** 2026-07-04 · **Status:** approved-pending-user-review · **Source:** user report
("the whisper thing sometimes cuts off — I need a lot better whisper refining and on-the-dot
cutting"). Diagnosis from reading `boundary.py`/`cut.py`/`transcribe.py`/config + a 22-source
adversarially-verified deep-research pass on ASR word-timestamp precision.

---

## 1. The problem (grounded)

The precise-boundary pass (`boundary.py`) cuts **exactly at Whisper's word timestamps**, which
are structurally imprecise, and then adds almost no cushion:

- **`tail_pad_s = 0.05`** (50 ms) is the only cushion; there is **no lead pad** at the start.
- Whisper word timestamps are a **post-hoc DTW-on-cross-attention alignment quantized to a ~20 ms
  grid, with tens-to->100 ms real error** (verified; native faster-whisper's tokenizer absorbs
  inter-word pauses into adjacent words). Cutting on `word.end` with a 50 ms margin against a
  100 ms+ error → the last word's release is clipped. Same at the start.
- **`WHISPER_MODEL = small`, int8, no `vad_filter`** on the refine window — the least-precise tier.
- **`_pick_end` hard-caps the search at `pad` (10 s)**: if no period-terminated sentence end is
  within 10 s of the rough end it returns the coarse rough end → a mid-sentence, imprecise cut.

Research also **refuted** a "faster-whisper truncates final words" bug — the cut-off is purely
quantization + approximate DTW + no trailing pad, so the fix is padding + silence-snapping, not a
different model.

## 2. Core principle

> **The cut always lands in the inter-word silence, never on a word.** Because we cut in a gap,
> a ±100 ms timestamp error is harmless — it moves the cut within the silence, not into speech.

When no silence (or no sentence-ending period) is found in the current window, **grow the window
and re-transcribe** until one is found (bounded), rather than accept an imprecise cut.

## 3. Decisions (locked with user)

1. **Lightweight** — faster-whisper only, no forced-alignment dependency (WhisperX/MFA/stable-ts
   `refine()` don't integrate cleanly with the CTranslate2 model anyway). Silence-snapping is
   implemented in-pipeline (word gaps + energy minimum).
2. **Window extends when silence/period not found** — direction-aware, doubling, bounded by a
   search cap and by `max_clip_duration_s`.
3. Fallback pads: `lead_pad ≈ 0.06 s`, `tail_pad ≈ 0.15 s` (research: ~100 ms trailing catches
   ~98–99 % of end-clipping; 150 ms for real/noisy audio). Refine model → `medium`.

## 4. Components

All changes are in `backend/pipeline/boundary.py` + `backend/config.py`; `cut.py` is verified,
not changed (see §5).

### 4a. Richer refine window — `boundary._whisper_window`
- Transcribe the edge window with a dedicated refine model `config.REFINE_WHISPER_MODEL`
  (default `medium`; the window is small so cost is modest), `vad_filter=True`
  (`speech_pad_ms≈200`), `condition_on_previous_text=False`, `temperature=0`, `beam_size=5`.
- Return the window's word list (each `{word,start,end}`, already produced) AND keep the extracted
  wav path so the energy-minimum step (4c) can read samples without re-extracting.

### 4b. Silence-aware boundary pick — `boundary._pick_start` / `_pick_end`
After snapping to the sentence's first word `S` / last word `E` (existing sentence logic):
- **Start cut** = `max(S.start - lead_pad, midpoint(prev_word.end, S.start))` — into the gap
  before `S`, never into `prev_word`. If `prev_word` is absent/far, `S.start - lead_pad`.
- **End cut** = `min(E.end + tail_pad, midpoint(E.end, next_word.start))` — into the gap after
  `E`, never into `next_word`. If `next_word` is absent/far, `E.end + tail_pad`.
- The direction-safety invariants stay: start never moves later than `rough+1s`; end never earlier
  than `rough-1s`.

### 4c. Energy-minimum refinement (the "on the dot" point)
Within the chosen gap `[a,b]`, snap the cut to the **lowest-RMS 10 ms frame** of the window audio
(read the already-extracted wav; compute short-frame RMS over `[a,b]`). This lands the cut at the
acoustically quietest instant in the pause. Pure-numpy, no new dependency; skipped gracefully if
the wav read fails (falls back to the gap midpoint / pad).

### 4d. Adaptive window extension — the key robustness addition
`_pick_end` (and symmetrically `_pick_start`) currently returns the coarse rough boundary when no
period is within `pad`. Replace with a search that **extends the window** when the target isn't
found:
- Target for the END: a **period-terminated sentence end** with a **usable trailing silence gap**
  at/after the rough end. Target for the START: a sentence start with a leading gap at/before it.
- If not found in the current window, **re-transcribe a larger window** — forward for the end
  (`win_end += grow_step`), backward for the start (`win_start -= grow_step`) — doubling the search
  half-width: `pad → 2·pad → 4·pad …`, capped at `config.MAX_BOUNDARY_SEARCH_S` (default 45 s) and
  never extending the clip past `max_clip_duration_s`.
- Only after the max-extended window still yields nothing: fall back to the padded rough boundary,
  flagged `boundary_search_exhausted` (surfaced, never drops the clip).
- **Asymmetry to note:** for the END, extension may move the cut to a *later* clean sentence+gap
  boundary (bounded by `max_clip_duration_s`) — the clip end genuinely advances to the nearest
  clean stop. For the START, the chosen first sentence does NOT change; backward extension only
  widens the transcription window so `prev_word` becomes visible and the leading gap can be
  measured — it never re-selects an earlier sentence (that would change the clip's content).

### 4e. Config — `backend/config.py`
`tail_pad_s: 0.05 → 0.15`; new `lead_pad_s: 0.06`; new `REFINE_WHISPER_MODEL` (default `medium`,
env-overridable, falls back to `WHISPER_MODEL` if unset); new `REFINE_VAD` (default on);
`MAX_BOUNDARY_SEARCH_S = 45.0`; `SILENCE_MIN_GAP_S = 0.12` (a gap must be at least this wide to
count as a clean cut site). `BOUNDARY_PAD_S` stays the initial window half-width.

## 5. ffmpeg (verify, don't blind-change)

Research did **not** cover ffmpeg cutting, so treat it empirically. Current cut is re-encode with
`-ss` before `-i` + `-t dur` (`cut.py`) — frame-accurate for video. Because cuts now land in
silence, any ±1-frame or AAC-priming (~1024–2112 sample) error is absorbed by the gap. Verification
(§6) measures trailing speech energy of a real cut mp4; only if that shows truncation do we adjust
ffmpeg (candidate: `-ss` after `-i` for sample-accurate audio, or a small extra end margin).

## 6. Testing & verification

- **Unit tests** (offline, synthetic word lists) for the silence-snap: cut always lands in a gap,
  never inside a word; asymmetric pad; `prev/next` word absent; gap narrower than
  `SILENCE_MIN_GAP_S` → extension path; direction-safety invariants hold.
- **Extension tests**: no period in the initial window → window grows → period found in the
  extension; exhaustion → flagged fallback, clip still ships.
- **Energy-minimum test**: on a synthesized tone+silence wav, the cut snaps to the silent frame.
- **Real-cut A/B** (needs the local Whisper + ffmpeg, run in the hosting env): for a sample video,
  cut clips old vs new and measure (a) fraction of cut points landing in silence (≈0 % → ≈100 %),
  (b) RMS energy of the final 100 ms of the clip (should drop to near-silence), (c) first 100 ms
  likewise. Acceptance: no clip's last/first 100 ms contains speech-level energy.

## 7. Scope / non-goals

- No forced-alignment dependency (locked). No change to full transcription (supadata/small stays).
- ffmpeg unchanged unless §6 proves truncation. Single implementation plan.
