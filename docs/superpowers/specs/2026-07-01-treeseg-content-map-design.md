# TreeSeg content map — embedding-based divisive segmentation

**Date:** 2026-07-01 · **Status:** approved-pending-user-review · **Roadmap:** RESEARCH.md Tier-2 "TreeSeg content_map" (arXiv:2407.12028)

## Problem

`build_content_map` finds topic boundaries with a per-chunk LLM pass (`CM_SYSTEM`). Research
(RESEARCH.md Area 3, ✓ VERIFIED): embedding/fine-tuned segmenters beat zero-shot LLM boundaries
(Pk 0.14 vs 0.31). Worse, the LLM partition is **non-deterministic**: the step-1 trustworthy-eval
work measured the structure rebuild as the dominant eval noise (clip count flips 1↔2, comprehension
0.5↔1.0 across identical-code runs), and the topic partition re-drawn each run is a driver of it.
Deterministic boundaries attack both boundary quality and rebuild noise at the root.

## Decisions (locked with user)

1. **Boundaries: 100% embeddings** (deterministic). **LLM: labels only** — one cheap pass titling
   already-fixed segments. Labels never change the partition.
2. **Granularity: target length + coherence floor.** `K = clamp(round(duration / TREESEG_TARGET_TOPIC_SEC),
   TREESEG_MIN_TOPICS, TREESEG_MAX_TOPICS)`; splitting stops early when best gain < floor; never
   splits below min size.
3. **Pause/discourse prior blended in:** small additive bonus at candidate boundaries from
   `segment.gap_before` / `segment.discourse_hits` (both deterministic).
4. **Legacy LLM path kept** as `CONTENT_MAP_ENGINE="llm"` and as the graceful-degrade fallback.
5. **Subtopics:** the treeseg engine emits none (splitting stops at the topic cut, and nothing
   consumes `subtopics()` downstream; `topics()` returns topic-level nodes either way). The legacy
   fallback keeps emitting them.
6. **Baseline strategy:** no upfront baseline run needed — either engine measurable any time via
   the env flag + the step-1 `--runs N` harness.

## Architecture

### New module `backend/pipeline/understand/treeseg.py` (pure, LLM-free, offline-testable)

- `embed_sentences(sentences) -> np.ndarray` — lazy module-singleton
  `SentenceTransformer(config.BI_ENCODER)` (`all-MiniLM-L6-v2`, already cached locally),
  L2-normalized float32, `device=config.TORCH_DEVICE`. Model-load or encode failure raises →
  caller falls back to legacy.
- `divisive_segments(emb, *, target_k, min_size, coherence_floor, priors) -> list[tuple[int, int]]`
  — bisecting/divisive segmentation on contiguous spans:
  - Best cut `k` in a span maximizes between-segment scatter
    `‖S_L‖²/n_L + ‖S_R‖²/n_R − ‖S‖²/n` (Ward-style bisecting objective; O(n) per span via vector
    prefix sums), plus `TREESEG_PAUSE_PRIOR × normalized_prior[k]` at candidate boundaries.
  - Max-gain priority queue over spans; split until `target_k` segments **or** best gain <
    `coherence_floor`. Cuts creating a side `< min_size` sentences are skipped.
  - Deterministic: ties broken by lowest index; no randomness anywhere.
  - Returns the flat cut AND records split order/gains (the tree), so one run yields both the
    topic cut and a coarser chapter cut.
- `chapter_cut(split_history, n_topics) -> groupings` — the shallower cut of the same tree: merge
  adjacent topics in inverse split order until `≤ max(1, round(n_topics / CHAPTER_MAX_TOPICS))`
  chapters (reuses the existing `CHAPTER_MAX_TOPICS` intent; no second pass, no LLM).

### Refactor `backend/pipeline/understand/content_map.py`

- `build_content_map(sentences, settings, progress)` dispatches on
  `settings.get("content_map_engine", config.CONTENT_MAP_ENGINE)`:
  - `"treeseg"` (default): `embed_sentences` → `divisive_segments` → `chapter_cut` →
    `_label_segments` (LLM) → assemble `ContentMap` (same `ContentNode` tree shape: video →
    chapter → topic; no subtopic nodes — see Decisions #5).
  - `"llm"`: the current implementation, moved verbatim to `_build_content_map_llm`.
  - Any exception in the treeseg path (embed failure, degenerate input) → log-and-fall-back to
    `_build_content_map_llm`. Fallback recorded (see Degrade marker).
- **Edge cases:** `n == 0` → video-only node (unchanged); `n < TREESEG_MIN_TOPIC_SENTS × 2` or
  `K == 1` → single topic spanning all sentences (no embeddings needed); coverage is always
  gapless `[0, n-1]` non-overlapping (invariant test).

### Labeling pass `_label_segments` (LLM, cheap, non-structural)

- One `llm_json` call (temperature 0.1) with each segment's first/last sentences + a middle sample
  (bounded chars) → `{title, summary, keywords}` per segment, aligned by index. Batched
  `LABEL_BATCH` segments per call when many.
- Alignment: response list clamped/padded to the segment count — labels can never re-partition.
- LLM failure → deterministic fallback titles (`Topic N` + top TF-IDF-ish keywords from the
  segment's own tokens; plain Python, no dependency), summary empty. Partition intact.
- Consumers unaffected: `extract_units` reads only `sentence_range` (hard) + `title` (soft prompt
  hint) + `summary` (failure fallback text) — all preserved.

### Cache invalidation (must-do)

- **`models.py: SCHEMA_VERSION 3 → 4`** ("4: content map boundaries now embedding-derived;
  cached topic partitions/sentence_ranges from the LLM engine are stale"). Both `structure.json`
  and `perception.json` gate on it → auto-rebuild once, including the step-1 freeze caches.

### Degrade marker

- New field `ContentMap.engine: str = ""` records which engine produced the map: `"treeseg"`,
  `"llm"` (explicitly configured), or `"llm-fallback"` (treeseg failed → legacy). Adding the field
  is safe — `SCHEMA_VERSION` is being bumped anyway, and it serializes into `structure.json` for
  inspection. `build_structure` appends `"content_map"` to `Structure.degraded` when
  `engine == "llm-fallback"` (mirrors the perception-degrade pattern; one-line change in
  `build.py`). `/health` untouched (engine is config-visible).

### Config (`config.py`, env-gated)

```python
CONTENT_MAP_ENGINE = os.environ.get("CONTENT_MAP_ENGINE", "treeseg")   # "treeseg" | "llm"
TREESEG_TARGET_TOPIC_SEC = float(os.environ.get("TREESEG_TARGET_TOPIC_SEC", "120"))
TREESEG_MIN_TOPICS = 2
TREESEG_MAX_TOPICS = 24
TREESEG_MIN_TOPIC_SENTS = 3
TREESEG_COHERENCE_FLOOR = float(os.environ.get("TREESEG_COHERENCE_FLOOR", "0.0"))  # 0 = target-K driven
TREESEG_PAUSE_PRIOR = float(os.environ.get("TREESEG_PAUSE_PRIOR", "0.15"))
TREESEG_LABEL_BATCH = 12
```
Reuses `BI_ENCODER`, `TORCH_DEVICE`, `CHAPTER_MAX_TOPICS`. `DEFAULTS` gains
`"content_map_engine": None` (None → inherit config) for per-job override. No new dependencies
(numpy + sentence-transformers already installed; model already cached).

## Out of scope (separate roadmap items)

Golden boundary set + Pk/WindowDiff (needs human labels, roadmap #6); judge rubric/1-10 scale
(#4); Maverick coref (#3); OCR/Texo (#5); VAD/multimodal boundary refinement (research lead).
Legacy helpers (`_normalize_topics`, `_group_chapters`, `_split_subtopics`, `CM_SYSTEM`) survive
only inside the fallback path — not deleted.

## Testing (TDD, offline, no LLM/network)

New `backend/pipeline/understand/tests/` (first tests for this package; mirrors
`punctuation/tests` convention):

1. **Algorithm (fake embedder — block-structured vectors):** two orthogonal blocks → first cut on
   the block boundary; nested blocks → nested cuts in split order; uniform vectors → no split
   past the floor; `target_k` honored; `min_size` respected (no slivers); gapless coverage;
   determinism (identical input → identical output, twice).
2. **Pause prior:** tie between two candidate cuts broken by the boundary with the larger gap.
3. **ContentMap assembly:** node ids/levels/parent-child links well-formed; `topics()` covers
   `[0, n-1]` gapless; chapters partition topics; no subtopic nodes under the treeseg engine;
   `engine` field set correctly in all three cases.
4. **Labeling:** mock `llm_json` — titles land on the right segments; short/long response clamped;
   LLM raise → partition unchanged + fallback titles.
5. **Fallback:** embedder raises → legacy path output + degraded marker.
6. **Regression:** full backend suite stays green; compile clean.

## Verification & measurement (uses step-1 harness)

1. **Determinism proof:** build the content map twice on a cached video (`CONTENT_MAP_ENGINE=treeseg`)
   → byte-identical topic boundaries; same run with `llm` engine differs (illustrative).
2. **A/B:** `python -m backend.eval.run_eval <vids> --runs 3` with `CONTENT_MAP_ENGINE=llm` vs
   `treeseg` (no `--freeze` — structure is the thing changing). Headline: comprehension_rate
   mean ± std, plus **n_clips run-to-run std** (expected to shrink toward 0 under treeseg).
3. Smoke one real cached video end-to-end (`backend.cli`, full profile, offline).

## Risks

- MiniLM embeddings on noisy ASR text may under-segment monotone lectures → the pause prior and
  target-K guard against degenerate one-topic outputs; fallback path covers hard failure.
- Chapter titles come from the labeling pass (first topic's title today) — cosmetic parity, judged
  by eval, not asserted in tests.
- First model load ~1–2 s per process (then cached in-process; already on disk).
