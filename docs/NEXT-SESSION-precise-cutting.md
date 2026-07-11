# Next-session prompt — precise, silence-snapped clip cutting (2026-07-04)

Paste the block below into a fresh Claude Code session. Self-contained: assumes zero prior context.

---

You are implementing **precise, silence-snapped clip cutting** for the VidScout clipper so clips
never cut off the first or last word ("on-the-dot cutting"). This continues prior design work.

## 0. READ FIRST (source of truth)
- Spec: `clips/docs/superpowers/specs/2026-07-04-precise-clip-cutting-design.md` — READ IT FULLY
  before doing anything. It is the approved design; this prompt is the operational handoff.
- This handoff: `clips/docs/NEXT-SESSION-precise-cutting.md`.

## 1. Environment
- Repo root: `/Users/vincentfeng/Documents/practice/clips` — a **git repo, currently on `main`**
  (the "discourse-onset" clip-start work merged just before this, merge `1dff8b3`; do NOT regress it).
- **First step:** `cd /Users/vincentfeng/Documents/practice/clips && git checkout -b precise-cutting`.
- Python venv: `clips/.venv`. Run tests with it: `.venv/bin/python -m pytest backend/<path> -v`
  (NOT bare pytest). Run from the `clips/` dir.
- End every commit message with exactly:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- Local faster-whisper runs CPU/int8. The refine model change (→ `medium`) triggers a **one-time
  ~1.5 GB model download** on first run; account for it in latency expectations.
- The full app is **hosted live** via `/Users/vincentfeng/Documents/practice/host.sh` (clipper
  :8000 private + Node :3000 + Cloudflare tunnel). To see changes live, restart the clipper AFTER
  implementing (see §6). NOTE: `./host.sh` may get auto-backgrounded by the harness and trap the
  daemons under the session; if `ps -o ppid= -p <pid>` shows a non-1 parent, relaunch the 3
  services with a Python `os.fork()`+`os.setsid()` double-fork so they reparent to launchd (PPID 1)
  and survive session cleanup. (There is a working pattern from the last session.)

## 2. The problem (already diagnosed — do not re-diagnose)
`backend/pipeline/boundary.py` refines each clip boundary with faster-whisper on a small audio
window, then **cuts exactly at Whisper's word timestamps** with `tail_pad_s=0.05` (50 ms) and **no
lead pad**. Whisper word timestamps are a post-hoc DTW alignment quantized to ~20 ms with
**tens-to->100 ms real error**, so cutting on `word.end` clips the last word; same at the start.
`_pick_end` also hard-caps the period search at `pad` (10 s) → coarse cut when no period is near.

## 3. Research grounding (verified — do NOT re-run research)
- Native faster-whisper word timestamps: ~20 ms floor, tens-to->100 ms real error. **Never cut on
  `word.start`/`word.end`.**
- A ~100 ms trailing pad catches ~98–99 % of end-clipping; ~50 ms catches ~82–89 %. Use 150 ms for
  real/noisy audio.
- `vad_filter=True` pads speech segments ±400 ms (`speech_pad_ms`, tunable) — useful silence margin.
- The "faster-whisper truncates final words" theory was **REFUTED** — it's quantization + no pad.
- **Locked decision: LIGHTWEIGHT** — no forced-alignment dependency (WhisperX/MFA; stable-ts
  `refine()` doesn't work on CTranslate2 models). Implement silence-snapping in-pipeline
  (word-gaps + energy minimum).
- **ffmpeg cutting was NOT covered by research** — treat it empirically (verify a real cut; only
  change `cut.py` if a measurement proves truncation).

## 4. What to build (per the spec §4)
Core principle: **the cut lands in the inter-word silence, never on a word.**

**`backend/config.py`** (current anchors):
- L127 `"tail_pad_s": 0.05` → `0.15`.
- Add `"lead_pad_s": 0.06` to DEFAULTS.
- Add `REFINE_WHISPER_MODEL = os.environ.get("REFINE_WHISPER_MODEL", "medium")` near L55
  (`WHISPER_MODEL`); the refine pass uses this, full transcription keeps `WHISPER_MODEL`.
- Add `REFINE_VAD` (default on), `MAX_BOUNDARY_SEARCH_S = 45.0`, `SILENCE_MIN_GAP_S = 0.12`.
- Keep `BOUNDARY_PAD_S` (L63, =10) as the INITIAL window half-width; keep
  `DEFAULTS["max_clip_duration_s"]` (L130, =180) as the extension ceiling.

**`backend/pipeline/boundary.py`** (current anchors):
- `_whisper_window` (L71): transcribe with `config.REFINE_WHISPER_MODEL`, `vad_filter=REFINE_VAD`
  (`speech_pad_ms≈200`), `condition_on_previous_text=False`, `temperature=0`, `beam_size=5`.
  Return the words AND keep the extracted wav path so the energy step can read samples.
  (Uses `transcribe._get_whisper()`; you'll likely need a SEPARATE refine-model singleton keyed by
  `REFINE_WHISPER_MODEL` — mind the `REFINE_WORKERS=4` thread pool / `num_workers` thread-safety;
  pysbd is thread-local, the model is a threadsafe CTranslate2 singleton.)
- `_pick_start` (L99) / `_pick_end` (L114): silence-aware placement —
  start cut = `max(S.start - lead_pad, midpoint(prev_word.end, S.start))`;
  end cut = `min(E.end + tail_pad, midpoint(E.end, next_word.start))`; then snap to the
  lowest-RMS 10 ms frame within that gap (read the window wav; pure numpy). Preserve the existing
  direction-safety invariants (start never later than rough+1s; end never earlier than rough-1s).
- **Window extension** (spec §4d): if no period-terminated sentence end WITH a usable trailing gap
  (≥ `SILENCE_MIN_GAP_S`) is found, re-transcribe a LARGER window (end: `win_end += grow`; start:
  `win_start -= grow`), doubling `pad→2·pad→4·pad`, capped at `MAX_BOUNDARY_SEARCH_S` and never
  past `max_clip_duration_s`. Exhausted → padded fallback flagged `boundary_search_exhausted`
  (never drop the clip). ASYMMETRY: the END may advance to a later clean boundary; the START keeps
  its sentence and only widens the window to see `prev_word`.
- `_refine_one` (L151) / `refine_clip_boundaries` (L179): thread `lead_pad`, `tail_pad`, the refine
  model, and the wav-for-energy through.

**`backend/pipeline/cut.py`**: no change unless §6 verification shows truncation. (`build_cmd`
L50: re-encode, `-ss` before `-i`, `-t dur`, `cut_end = end + tail_pad`.)

## 5. Where the refine actually runs (so you can test end-to-end)
`orchestrator.py` L331 (full path) & L378 (fast path):
`if config.PRECISE_BOUNDARIES and transcript.get("source") == "supadata": refine_clip_boundaries(...)`.
So the refine only fires with `PRECISE_BOUNDARIES=1` (default) AND a supadata transcript. The host
runs `OUTPUT_MODE=cut` so users get the actual cut mp4s (`cut.py`) — the boundary times you produce
are what get cut.

## 6. Testing & verification (spec §6)
- **Unit tests** (offline, synthetic word lists), TDD: cut always lands in a gap, never inside a
  word; asymmetric pad; `prev`/`next` word absent; gap < `SILENCE_MIN_GAP_S` → extension path;
  extension finds a period beyond the initial window; exhaustion → flagged fallback, clip ships;
  energy-minimum snaps to the silent frame on a tone+silence wav. Don't regress
  `backend/pipeline/tests/test_bnd1_boundary_guards.py` or the onset-guard tests.
- **Real-cut A/B** (needs local Whisper + ffmpeg; run in the host env): cut a sample video's clips
  old vs new; measure (a) fraction of cut points landing in silence (≈0 %→≈100 %), (b) RMS of the
  clip's final 100 ms and (c) first 100 ms (both should drop to near-silence). Acceptance: no
  clip's first/last 100 ms contains speech-level energy.
- Run the full suite green before finishing: `.venv/bin/python -m pytest backend/ -q` (baseline
  was 701 passed).

## 7. Process
Follow the superpowers flow: invoke **writing-plans** to turn the spec into a task-by-task TDD plan
(save under `docs/superpowers/plans/`), then **subagent-driven-development** to execute with a
fresh implementer + reviewer per task and a final whole-branch review. (The spec + this handoff are
the inputs.) When done, offer merge via **finishing-a-development-branch**, then restart the host to
go live.

## 8. Open question the user should confirm early
When the speaker runs straight into the next sentence with NO pause at the chosen end, the
extension can make the clip a bit longer (advancing to the next clean stop, bounded by 180 s) — OR
you take the best-available cut at the original sentence end. Confirm which the user prefers before
finalizing `_pick_end`'s extension policy.
