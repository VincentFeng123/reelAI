# Robustness pkg 3 — quick wins (7 surgical fixes) — design

**Date:** 2026-07-02 · **Status:** approved-pending-user-review · **Source:** S-effort
high/medium findings from `docs/clip-editing-robustness-audit.md`; package 3 of 3 (executed
before pkg 2 per user sequencing).

## Decisions (locked with user)

Seven independent, surgical fixes following the audit proposals and the codebase's
degrade-and-surface ethos. No structural refactors (job namespacing, disk lifecycle, clip
identity → pkg 2). Each fix independently testable offline.

## Fixes

### 1. Precise-boundary resilience — `backend/pipeline/boundary.py`

`refine_clip_boundaries` wraps its `_ensure_audio(url, video_id)` call in try/except; on
exception it returns the input `clips` list UNCHANGED (already sentence-snapped + judged) —
never raises. Covers both orchestrator call sites at once. The per-clip try/except stays.

### 2. Relevance-failure surfacing — `backend/pipeline/assemble/candidates.py` + `__init__.py`

- `score_topic_relevance` return becomes `tuple[dict[str, float], bool]` (scores, degraded).
  A failed batch retries ONCE; if the retry also fails, every unit in the batch defaults 0.5
  (unchanged) and `degraded=True`. Empty topic → `({all 1.0}, False)` (unchanged fast path).
- `assemble_clips` (sole caller) unpacks the tuple; when degraded and a topic was given,
  `notes` gains the suffix `" (topic filtering degraded — clips selected by role priority)"`.

### 3. Atomic cut/export writes — `backend/pipeline/cut.py` + `export.py`

- New helper in `export.py` (imported by cut.py): `finalize_output(tmp: Path, out: Path,
  rc: int, err: str) -> None` — on `rc == 0 and tmp.exists() and tmp.stat().st_size > 0`:
  `tmp.replace(out)` (atomic same-fs); otherwise `tmp.unlink(missing_ok=True)` and raise
  `PipelineError(f"Clip cut failed: {tail of err or rc}")`.
- Both ffmpeg invocations target `tmp = out.with_name(out.stem + ".tmp.mp4")` (the `.mp4`
  extension is kept so ffmpeg muxes correctly). After the process exits, call
  `finalize_output(tmp, out, rc, err)`.
- `export_clip` cache hit tightens to `out.exists() and out.stat().st_size > 0`.

### 4. Export cache key includes the range — `backend/pipeline/export.py`

`fname = f"clip_{n}_{_slug(facet)}_{int(round(start*1000))}_{int(round(end*1000))}.mp4"`.
`served` derives from the same fname. Different boundaries → different file; stale content
can never be served for a new range. Old un-keyed files become orphans (disk lifecycle → pkg 2).
`cut_clips`' per-job filenames unchanged (job namespacing → pkg 2).

### 5. Self-defined prerequisite hints — `backend/pipeline/assemble/sequence.py` + caller

- `sequence_clips(specs, graph, units_by_id)` and `attach_prerequisites(specs, graph,
  units_by_id)` gain the units map (assemble's step 7 passes its existing `units_by_id`).
- `attach_prerequisites` computes per clip: `intro = ∪ concepts_introduced(s.units)`,
  `req = ∪ concepts_required(s.units)`, `unmet = req − intro`. An earlier clip becomes a
  hint iff `∪ concepts_introduced(other.units) ∩ unmet ≠ ∅`. `graph.defines_needed_for` is
  no longer consulted here (graph param retained for signature stability this round).

### 6. Zip endpoint guard — `backend/main.py`

- Pure helper `_zipable_files(clips: list[dict], folder: Path) -> list[Path]`: entries with
  truthy `path` whose file exists under `folder` (match by `Path(c["path"]).name`).
- `download_zip`: compute the list BEFORE constructing the response; if it is empty and
  `folder/clips.json` doesn't exist → `raise HTTPException(409, "No rendered clips to zip —
  use output_mode=cut or export clips first")`. The generator iterates the precomputed list
  (+ clips.json when present) — no `Path(None)` possible.

### 7. Direction-safe Whisper picks — `backend/pipeline/boundary.py`

- `_pick_end`: when no valid end exists at/after `rough − 1.0` inside the window, return
  `rough` (NEVER `max(ends)` — that truncates judged content by up to pad seconds).
- `_pick_start`: when no sentence start exists at/before `rough + 1.0`, return `rough`
  (never `min(starts)` later than that).
- `_pick_start` gains `keep_first: bool = False`: the `sents[0]` leading-fragment drop
  applies only when the Whisper window actually cut into speech; `refine_clip_boundaries`
  passes `keep_first=(s0 - pad <= 0.0)` (window clamped at video start → sents[0] is real).

## Out of scope

Job/output namespacing, orphan cleanup, per-video locking, job persistence/cancel (pkg 2);
whisper-pass drop ledgering + shrink-aware metadata repair (pkg-2 backlog, already documented);
`refine.py` remains frozen (untouched by this package).

## Testing (offline; new `backend/pipeline/tests/` package for boundary/export/cut helpers,
mirroring existing conventions; assemble tests extend existing files)

1. `refine_clip_boundaries` with `_ensure_audio` monkeypatched to raise → returns input list
   unchanged (identity of dicts preserved).
2. `score_topic_relevance` mocked llm: fail-then-succeed → retried batch scores land,
   degraded False; fail-fail → all 0.5 + degraded True; empty topic → no LLM call.
   Assemble notes suffix appears when degraded (unit-level: build notes string logic or
   integration via mocked pipeline as in test_integrity).
3. `finalize_output`: rc 0 + real tmp file → renamed; rc 1 → tmp gone, PipelineError, out
   absent; rc 0 + zero-byte tmp → error + cleanup.
4. Export fname embeds range (pure string check via a small `_export_fname` helper or direct
   call inspection).
5. `attach_prerequisites`: clip requiring X with own intro X → no hint; requiring Y introduced
   only by earlier clip → hint with that clip's sequence_index; later clips never hinted.
6. `_zipable_files`: skips None paths, skips missing files, keeps existing; 409 logic —
   pure helper tested directly (no TestClient needed).
7. `_pick_end`/`_pick_start` with synthetic sentences: fallback returns rough (not max/min);
   normal path unchanged; `keep_first=True` uses sents[0].

## Verification

Full suite green (baseline 133) + compile; CLI smoke on a cached video (no behavior change
expected on the happy path). No eval run (no selection-quality change).
