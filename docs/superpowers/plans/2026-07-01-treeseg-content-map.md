# TreeSeg Content Map Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace LLM topic-boundary finding with deterministic embedding-based divisive segmentation (TreeSeg, arXiv:2407.12028); LLM labels the fixed segments only.

**Architecture:** New pure module `treeseg.py` (embeddings → Ward-style bisecting splits + pause prior → topic cut + chapter cut from one tree). `content_map.py` dispatches on `CONTENT_MAP_ENGINE` ("treeseg" default, legacy "llm" kept as explicit engine AND as fallback), then runs a cheap batched LLM labeling pass that can never re-partition. `SCHEMA_VERSION` 3→4 invalidates all cached structures/perceptions.

**Tech Stack:** Python 3.12, numpy 2.2.6, sentence-transformers 3.4.1 (`all-MiniLM-L6-v2`, already cached locally), pydantic, pytest 9.1.1.

**Spec:** `docs/superpowers/specs/2026-07-01-treeseg-content-map-design.md`

## Global Constraints

- Run everything from `/Users/vincentfeng/Documents/practice/clips` with `.venv/bin/python`.
- **No git repo here** — "Checkpoint" steps (full test suite + compile) replace commit steps.
- Tests are offline: no network, no real LLM, no real SentenceTransformer load (fake embedder / mocked `llm_json`).
- Determinism is a hard requirement of `divisive_segments`: no randomness; ties break to the lowest index.
- Gapless invariant: topic segments partition `[0, n-1]`, non-overlapping, sorted.
- `ContentMap.topics()` consumers (`units.extract_units`) read only `sentence_range` (hard), `title` (prompt hint), `summary` (fallback text) — all must stay populated.
- Config values verbatim from the spec: `CONTENT_MAP_ENGINE="treeseg"`, `TREESEG_TARGET_TOPIC_SEC=120`, `TREESEG_MIN_TOPICS=2`, `TREESEG_MAX_TOPICS=24`, `TREESEG_MIN_TOPIC_SENTS=3`, `TREESEG_COHERENCE_FLOOR=0.0`, `TREESEG_PAUSE_PRIOR=0.15`, `TREESEG_LABEL_BATCH=12`.

---

### Task 1: Config block + schema bump + `ContentMap.engine`

**Files:**
- Modify: `backend/config.py` (after the `BRIDGE_MAX_PER_UNIT` line, ~line 191; plus `DEFAULTS` dict ~line 135)
- Modify: `backend/pipeline/understand/models.py` (`SCHEMA_VERSION` ~line 18; `ContentMap` ~line 104)
- Create: `backend/pipeline/understand/tests/__init__.py` (empty)
- Test: `backend/pipeline/understand/tests/test_models_v4.py`

**Interfaces:**
- Produces: `config.CONTENT_MAP_ENGINE: str`, `config.TREESEG_TARGET_TOPIC_SEC: float`, `config.TREESEG_MIN_TOPICS: int`, `config.TREESEG_MAX_TOPICS: int`, `config.TREESEG_MIN_TOPIC_SENTS: int`, `config.TREESEG_COHERENCE_FLOOR: float`, `config.TREESEG_PAUSE_PRIOR: float`, `config.TREESEG_LABEL_BATCH: int`; `DEFAULTS["content_map_engine"] = None`; `ContentMap.engine: str = ""`; `SCHEMA_VERSION == 4`.

- [ ] **Step 1: Write the failing test**

```python
# backend/pipeline/understand/tests/test_models_v4.py
"""Schema-v4 groundwork: engine field, version bump, cache gating, config block."""
from __future__ import annotations

import json

from backend import config
from backend.pipeline.understand.models import (
    SCHEMA_VERSION, ContentMap, Structure, load_structure, save_structure,
)


def test_schema_version_is_4():
    assert SCHEMA_VERSION == 4


def test_content_map_engine_field_defaults_empty():
    assert ContentMap().engine == ""


def test_engine_field_roundtrips_through_structure_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORK_DIR", tmp_path)
    st = Structure(video_id="vidA", content_map=ContentMap(engine="treeseg"))
    save_structure(st)
    loaded = load_structure("vidA")
    assert loaded is not None
    assert loaded.content_map.engine == "treeseg"


def test_v3_cache_is_invalidated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORK_DIR", tmp_path)
    st = Structure(video_id="vidB")
    save_structure(st)
    p = tmp_path / "vidB" / "structure.json"
    data = json.loads(p.read_text())
    data["schema_version"] = 3
    p.write_text(json.dumps(data))
    assert load_structure("vidB") is None


def test_treeseg_config_block():
    assert config.CONTENT_MAP_ENGINE in ("treeseg", "llm")
    assert config.TREESEG_TARGET_TOPIC_SEC == 120.0
    assert config.TREESEG_MIN_TOPICS == 2
    assert config.TREESEG_MAX_TOPICS == 24
    assert config.TREESEG_MIN_TOPIC_SENTS == 3
    assert config.TREESEG_COHERENCE_FLOOR == 0.0
    assert config.TREESEG_PAUSE_PRIOR == 0.15
    assert config.TREESEG_LABEL_BATCH == 12
    assert "content_map_engine" in config.DEFAULTS and config.DEFAULTS["content_map_engine"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest backend/pipeline/understand/tests/test_models_v4.py -q`
Expected: FAIL — `SCHEMA_VERSION == 3`, `ContentMap` has no `engine`, config attrs missing.

- [ ] **Step 3: Implement**

In `backend/pipeline/understand/models.py` replace the `SCHEMA_VERSION` block:

```python
SCHEMA_VERSION = 4   # 4: content-map boundaries are embedding-derived (TreeSeg); cached topic
                     #    partitions / sentence_ranges from the LLM engine are stale; rebuild.
                     # 3: punctuation-restoration stage changes sentence segmentation
                     # 2: Phase-2 perception (visual_events populated, visual_dependencies linked)
```

In `ContentMap` add the field (after `nodes`):

```python
class ContentMap(BaseModel):
    root_id: str = "video"
    nodes: list[ContentNode] = Field(default_factory=list)
    engine: str = ""                           # "treeseg" | "llm" | "llm-fallback" (see content_map)
```

In `backend/config.py`, after the `BRIDGE_MAX_PER_UNIT = 6` line add:

```python
# Content-map engine: "treeseg" = deterministic embedding-based divisive segmentation
# (arXiv:2407.12028) with a cheap LLM labeling pass; "llm" = the legacy per-chunk LLM
# boundary pass (also the graceful-degrade fallback when treeseg fails).
CONTENT_MAP_ENGINE = os.environ.get("CONTENT_MAP_ENGINE", "treeseg")   # "treeseg" | "llm"
TREESEG_TARGET_TOPIC_SEC = float(os.environ.get("TREESEG_TARGET_TOPIC_SEC", "120"))
TREESEG_MIN_TOPICS = 2
TREESEG_MAX_TOPICS = 24
TREESEG_MIN_TOPIC_SENTS = 3
TREESEG_COHERENCE_FLOOR = float(os.environ.get("TREESEG_COHERENCE_FLOOR", "0.0"))  # 0 = target-K driven
TREESEG_PAUSE_PRIOR = float(os.environ.get("TREESEG_PAUSE_PRIOR", "0.15"))
TREESEG_LABEL_BATCH = 12
```

In `config.DEFAULTS` add (next to `"multimodal": None`):

```python
    "content_map_engine": None,             # None → inherit config.CONTENT_MAP_ENGINE
```

Create empty `backend/pipeline/understand/tests/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest backend/pipeline/understand/tests/test_models_v4.py -q`
Expected: 5 passed.

- [ ] **Step 5: Checkpoint**

Run: `.venv/bin/python -m pytest backend -q && .venv/bin/python -m compileall -q backend`
Expected: all pass (63 existing + 5 new), compile clean.

---

### Task 2: `treeseg.py` — pure segmentation algorithm

**Files:**
- Create: `backend/pipeline/understand/treeseg.py`
- Create: `backend/pipeline/understand/tests/conftest.py`
- Test: `backend/pipeline/understand/tests/test_treeseg.py`

**Interfaces:**
- Consumes: `segment.gap_before(sentences, i) -> float`, `segment.discourse_hits(sentences) -> set[int]` (existing).
- Produces:
  - `boundary_priors(sentences, weight: float) -> np.ndarray`  # shape (n+1,), prior bonus at cut index k
  - `divisive_segments(emb: np.ndarray, *, target_k: int, min_size: int, coherence_floor: float = 0.0, priors: np.ndarray | None = None) -> tuple[list[tuple[int, int]], list[int]]`  # (sorted inclusive segments, boundary indices in split order)
  - `chapter_cut(split_order: list[int], segments: list[tuple[int, int]], max_per_chapter: int) -> list[tuple[int, int]]`  # inclusive topic-index ranges
  - `embed_sentences(sentences) -> np.ndarray` (Task 3 fills the real body; declare here)

- [ ] **Step 1: Write the shared test fixtures**

```python
# backend/pipeline/understand/tests/conftest.py
"""Offline fixtures: fabricated Sentences + block-structured fake embeddings (no model, no LLM)."""
from __future__ import annotations

import numpy as np

from backend.pipeline.sentences import Sentence


def make_sents(n: int, *, sec: float = 2.0, gap_at: tuple[int, ...] = (), gap: float = 6.0,
               texts: list[str] | None = None) -> list[Sentence]:
    """n sentences, `sec` seconds each; a `gap`-second pause BEFORE each index in gap_at."""
    sents, t = [], 0.0
    for i in range(n):
        if i in gap_at:
            t += gap
        s, e = t, t + sec
        sents.append(Sentence(idx=i, text=(texts[i] if texts else f"sentence number {i}"),
                              start=s, end=e, terminator=".", ends_with_period=True,
                              word_start_idx=i, word_end_idx=i, align_confidence=1.0))
        t = e + 0.1
    return sents


def block_emb(sizes: list[int], dim: int = 8) -> np.ndarray:
    """Unit vectors; block j points along axis j%dim → maximal between-block scatter at
    block boundaries. Deterministic by construction."""
    rows = []
    for j, sz in enumerate(sizes):
        v = np.zeros(dim)
        v[j % dim] = 1.0
        rows += [v] * sz
    return np.asarray(rows, dtype=np.float64)
```

- [ ] **Step 2: Write the failing algorithm tests**

```python
# backend/pipeline/understand/tests/test_treeseg.py
"""divisive_segments / boundary_priors / chapter_cut — pure, deterministic, offline."""
from __future__ import annotations

import numpy as np

from backend.pipeline.understand.treeseg import boundary_priors, chapter_cut, divisive_segments

from .conftest import block_emb, make_sents


def _coverage_ok(segments, n):
    assert segments == sorted(segments)
    assert segments[0][0] == 0 and segments[-1][1] == n - 1
    for (a0, a1), (b0, b1) in zip(segments, segments[1:]):
        assert b0 == a1 + 1


def test_two_blocks_cut_on_boundary():
    emb = block_emb([5, 5])
    segs, order = divisive_segments(emb, target_k=2, min_size=2)
    assert segs == [(0, 4), (5, 9)]
    assert order == [5]


def test_three_blocks_found():
    segs, _ = divisive_segments(block_emb([4, 4, 4]), target_k=3, min_size=2)
    assert segs == [(0, 3), (4, 7), (8, 11)]


def test_target_k_stops_splitting():
    segs, _ = divisive_segments(block_emb([3, 3, 3, 3]), target_k=2, min_size=2)
    assert len(segs) == 2


def test_uniform_never_splits_past_floor():
    emb = np.tile(np.array([1.0, 0, 0, 0]), (12, 1))          # identical vectors → gain 0
    segs, order = divisive_segments(emb, target_k=4, min_size=2, coherence_floor=1e-6)
    assert segs == [(0, 11)] and order == []


def test_min_size_respected():
    segs, _ = divisive_segments(block_emb([2, 10]), target_k=4, min_size=3)
    assert all(b - a + 1 >= 3 for a, b in segs)


def test_coverage_and_determinism():
    emb = block_emb([5, 4, 6, 5])
    r1 = divisive_segments(emb, target_k=4, min_size=2)
    r2 = divisive_segments(emb, target_k=4, min_size=2)
    assert r1 == r2
    _coverage_ok(r1[0], 20)


def test_degenerate_inputs():
    assert divisive_segments(block_emb([3]), target_k=1, min_size=2) == ([(0, 2)], [])
    assert divisive_segments(block_emb([2]), target_k=4, min_size=2) == ([(0, 1)], [])  # n < 2*min
    assert divisive_segments(np.zeros((0, 4)), target_k=2, min_size=2) == ([], [])


def test_prior_places_cut_on_uniform_embeddings():
    emb = np.tile(np.array([1.0, 0, 0, 0]), (12, 1))          # no embedding signal at all
    pr = np.zeros(13)
    pr[7] = 0.5
    segs, order = divisive_segments(emb, target_k=2, min_size=2, priors=pr)
    assert order == [7] and segs == [(0, 6), (7, 11)]


def test_boundary_priors_gap_and_discourse():
    sents = make_sents(10, gap_at=(5,), texts=[
        "alpha one", "alpha two", "alpha three", "alpha four", "alpha five",
        "beta one", "beta two", "So let us move on", "beta four", "beta five"])
    pr = boundary_priors(sents, weight=0.15)
    assert pr.shape == (11,)
    assert pr[5] > 0 and pr[7] > 0                            # pause seam + discourse marker
    assert pr[5] > pr[2] and pr[7] > pr[2]                    # plain boundary scores lower
    assert boundary_priors(sents, weight=0.0).sum() == 0.0


def test_chapter_cut_groups_by_earliest_splits():
    segs = [(0, 3), (4, 7), (8, 11), (12, 15), (16, 19), (20, 23)]
    order = [12, 4, 8, 16, 20]                                # first split at 12 = coarsest seam
    ranges = chapter_cut(order, segs, max_per_chapter=3)      # round(6/3)=2 chapters
    assert ranges == [(0, 2), (3, 5)]                         # boundary 12 = topic index 3


def test_chapter_cut_small_counts():
    segs = [(0, 5), (6, 11)]
    assert chapter_cut([6], segs, max_per_chapter=5) == [(0, 1)]   # round(2/5)=0 → 1 chapter
    assert chapter_cut([], [(0, 9)], max_per_chapter=5) == [(0, 0)]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest backend/pipeline/understand/tests/test_treeseg.py -q`
Expected: FAIL — `ModuleNotFoundError`/`ImportError: treeseg`.

- [ ] **Step 4: Implement `treeseg.py`**

```python
# backend/pipeline/understand/treeseg.py
"""Deterministic embedding-based divisive topic segmentation (TreeSeg, arXiv:2407.12028).

Boundaries come 100% from sentence embeddings: bisecting splits that maximize the Ward-style
between-segment scatter gain, plus a small deterministic pause/discourse prior at candidate
boundaries. One tree yields both the topic cut and (via the earliest splits) the chapter cut.
No LLM, no randomness — same input, same output. Labeling happens later in content_map.
"""
from __future__ import annotations

import heapq

import numpy as np

from ... import config
from ..sentences import Sentence
from .segment import discourse_hits, gap_before

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(config.BI_ENCODER, device=config.TORCH_DEVICE)
    return _model


def embed_sentences(sentences: list[Sentence]) -> np.ndarray:
    """L2-normalized float32 sentence embeddings (all-MiniLM-L6-v2, locally cached).
    Raises on model-load/encode failure — the caller falls back to the legacy engine."""
    texts = [(s.text or "") for s in sentences]
    emb = _get_model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(emb, dtype=np.float32)


def boundary_priors(sentences: list[Sentence], weight: float) -> np.ndarray:
    """Prior bonus for cutting BEFORE sentence k (shape n+1): long pauses and discourse
    markers ("so", "next", …) make a boundary slightly more attractive. Deterministic."""
    n = len(sentences)
    pr = np.zeros(n + 1, dtype=np.float64)
    if weight <= 0.0 or n < 2:
        return pr
    hits = discourse_hits(sentences)
    gaps = np.array([gap_before(sentences, i) for i in range(n)], dtype=np.float64)
    gmax = float(gaps.max()) or 1.0
    for i in range(1, n):
        pr[i] = weight * (min(gaps[i] / gmax, 1.0) + (1.0 if i in hits else 0.0)) / 2.0
    return pr


def divisive_segments(emb: np.ndarray, *, target_k: int, min_size: int,
                      coherence_floor: float = 0.0,
                      priors: np.ndarray | None = None) -> tuple[list[tuple[int, int]], list[int]]:
    """Bisecting segmentation of [0, n-1] into ≤ target_k contiguous segments.

    A cut at k splits [a,b] into [a,k-1],[k,b]; its gain is the between-segment scatter
    ‖S_L‖²/n_L + ‖S_R‖²/n_R − ‖S‖²/n (O(1) per candidate via prefix sums) + priors[k].
    Highest-gain span splits first; stops at target_k or when the best gain < coherence_floor.
    Ties break to the earliest index. Returns (sorted segments, boundaries in split order)."""
    n = int(emb.shape[0])
    if n == 0:
        return [], []
    if target_k <= 1 or n < 2 * min_size:
        return [(0, n - 1)], []
    prefix = np.zeros((n + 1, emb.shape[1]), dtype=np.float64)
    np.cumsum(emb, axis=0, out=prefix[1:])
    pri = priors if priors is not None else np.zeros(n + 1, dtype=np.float64)

    def best_cut(a: int, b: int):
        total = prefix[b + 1] - prefix[a]
        base = float(total @ total) / (b - a + 1)
        best_gain = best_k = None
        for k in range(a + min_size, b - min_size + 2):
            left = prefix[k] - prefix[a]
            right = total - left
            gain = (float(left @ left) / (k - a) + float(right @ right) / (b - k + 1)
                    - base + float(pri[k]))
            if best_gain is None or gain > best_gain + 1e-12:   # strict → earliest k wins ties
                best_gain, best_k = gain, k
        return (best_gain, best_k) if best_k is not None else None

    segments = [(0, n - 1)]
    split_order: list[int] = []
    heap: list[tuple[float, int, int, int]] = []
    first = best_cut(0, n - 1)
    if first:
        heapq.heappush(heap, (-first[0], 0, n - 1, first[1]))
    while heap and len(segments) < target_k:
        neg_gain, a, b, k = heapq.heappop(heap)
        if -neg_gain < coherence_floor:
            break
        segments.remove((a, b))
        segments.extend([(a, k - 1), (k, b)])
        split_order.append(k)
        for x, y in ((a, k - 1), (k, b)):
            c = best_cut(x, y)
            if c:
                heapq.heappush(heap, (-c[0], x, y, c[1]))
    segments.sort()
    return segments, split_order


def chapter_cut(split_order: list[int], segments: list[tuple[int, int]],
                max_per_chapter: int) -> list[tuple[int, int]]:
    """Chapter grouping from the SAME tree: the earliest splits are the coarsest seams.
    Returns inclusive (first_topic_idx, last_topic_idx) ranges covering all topics."""
    n_topics = len(segments)
    if n_topics == 0:
        return []
    n_chapters = max(1, round(n_topics / max_per_chapter))
    if n_chapters >= n_topics:
        return [(i, i) for i in range(n_topics)]
    bounds = sorted(split_order[: n_chapters - 1])
    starts = [s0 for s0, _ in segments]
    ranges, t0 = [], 0
    for b in bounds:
        t = starts.index(b)              # splits nest → every early boundary survives in the cut
        ranges.append((t0, t - 1))
        t0 = t
    ranges.append((t0, n_topics - 1))
    return ranges
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest backend/pipeline/understand/tests/test_treeseg.py -q`
Expected: 11 passed.

- [ ] **Step 6: Checkpoint**

Run: `.venv/bin/python -m pytest backend -q && .venv/bin/python -m compileall -q backend`
Expected: all pass, compile clean.

---

### Task 3: `content_map.py` — engine dispatch, treeseg assembly, labeling, fallback

**Files:**
- Modify: `backend/pipeline/understand/content_map.py`
- Test: `backend/pipeline/understand/tests/test_content_map_treeseg.py`

**Interfaces:**
- Consumes: Task 2's `treeseg` functions; existing `TopicLLM`, `ContentMap`, `ContentNode`, `render_sentences`, `backend.llm.llm_json`.
- Produces: `build_content_map(sentences, settings, progress) -> ContentMap` (signature unchanged) with `engine` set to `"treeseg"` / `"llm"` / `"llm-fallback"`; internal `_build_content_map_llm` (the moved legacy body), `_build_content_map_treeseg`, `_label_segments(segments, sentences) -> list[TopicLLM]`, `SegLabelLLM` / `SegLabelsLLM` pydantic models.

- [ ] **Step 1: Write the failing tests**

```python
# backend/pipeline/understand/tests/test_content_map_treeseg.py
"""Engine dispatch + treeseg ContentMap assembly + labeling alignment + fallback. Offline."""
from __future__ import annotations

import numpy as np
import pytest

import backend.llm as llm_mod
import backend.pipeline.understand.content_map as cm_mod
from backend.pipeline.understand.content_map import SegLabelLLM, SegLabelsLLM, build_content_map

from .conftest import block_emb, make_sents


@pytest.fixture
def two_block_embedder(monkeypatch):
    calls = {"n": 0}

    def fake_embed(sentences):
        calls["n"] += 1
        return block_emb([len(sentences) // 2, len(sentences) - len(sentences) // 2])

    monkeypatch.setattr(cm_mod, "embed_sentences", fake_embed)
    return calls


@pytest.fixture
def label_llm(monkeypatch):
    def fake_llm_json(system, user, schema, **kw):
        assert schema is SegLabelsLLM
        idxs = [int(t) for t in __import__("re").findall(r"\[(\d+)\]", user)]
        return SegLabelsLLM(labels=[
            SegLabelLLM(index=i, title=f"Label {i}", summary=f"About {i}", keywords=[f"k{i}"])
            for i in sorted(set(idxs))])

    monkeypatch.setattr(llm_mod, "llm_json", fake_llm_json)


def _mk_settings(engine=None):
    return {"content_map_engine": engine}


def test_treeseg_assembly(two_block_embedder, label_llm):
    sents = make_sents(12, sec=20.0)                     # ≈241 s → K = round(241/120) = 2
    cm = build_content_map(sents, _mk_settings())
    assert cm.engine == "treeseg"
    topics = cm.topics()
    assert [t.sentence_range for t in topics] == [(0, 5), (6, 11)]
    assert [t.title for t in topics] == ["Label 0", "Label 1"]
    assert topics[0].summary == "About 0" and topics[0].keywords == ["k0"]
    assert not cm.subtopics()                            # treeseg emits no subtopic nodes
    chapters = cm.chapters()
    assert chapters and all(ch.parent_id == "video" for ch in chapters)
    for t in topics:
        assert t.parent_id in {ch.node_id for ch in chapters}
        assert t.node_id in next(ch for ch in chapters if ch.node_id == t.parent_id).children_ids


def test_label_failure_keeps_partition(two_block_embedder, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("label LLM down")
    monkeypatch.setattr(llm_mod, "llm_json", boom)
    sents = make_sents(12, sec=20.0)                     # ≈241 s → K = 2 (see test above)
    cm = build_content_map(sents, _mk_settings())
    assert cm.engine == "treeseg"                        # label failure ≠ engine fallback
    topics = cm.topics()
    assert [t.sentence_range for t in topics] == [(0, 5), (6, 11)]
    assert all(t.title for t in topics)                  # deterministic fallback titles


def test_embed_failure_falls_back_to_llm_engine(monkeypatch):
    def boom(sentences):
        raise RuntimeError("no model")
    monkeypatch.setattr(cm_mod, "embed_sentences", boom)

    def fake_llm_json(system, user, schema, **kw):       # legacy path's topic pass
        from backend.pipeline.understand.content_map import ContentMapLLM, TopicLLM
        assert schema is ContentMapLLM
        return ContentMapLLM(topics=[TopicLLM(title="Legacy", sentence_start=0, sentence_end=11)])
    monkeypatch.setattr(llm_mod, "llm_json", fake_llm_json)

    cm = build_content_map(make_sents(12, sec=30.0), _mk_settings())
    assert cm.engine == "llm-fallback"
    assert cm.topics()[0].title == "Legacy"


def test_llm_engine_when_configured(two_block_embedder, monkeypatch):
    def fake_llm_json(system, user, schema, **kw):
        from backend.pipeline.understand.content_map import ContentMapLLM, TopicLLM
        return ContentMapLLM(topics=[TopicLLM(title="Legacy", sentence_start=0, sentence_end=11)])
    monkeypatch.setattr(llm_mod, "llm_json", fake_llm_json)
    cm = build_content_map(make_sents(12, sec=30.0), _mk_settings(engine="llm"))
    assert cm.engine == "llm"
    assert two_block_embedder["n"] == 0                  # embeddings never touched


def test_empty_sentences_video_only():
    cm = build_content_map([], _mk_settings())
    assert [n.level for n in cm.nodes] == ["video"]


def test_tiny_video_single_topic_no_embeddings(two_block_embedder, label_llm):
    sents = make_sents(4, sec=30.0)                      # n=4 < 2×TREESEG_MIN_TOPIC_SENTS
    cm = build_content_map(sents, _mk_settings())
    assert cm.engine == "treeseg"
    assert [t.sentence_range for t in cm.topics()] == [(0, 3)]
    assert two_block_embedder["n"] == 0                  # early-out before embedding
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest backend/pipeline/understand/tests/test_content_map_treeseg.py -q`
Expected: FAIL — `ImportError: SegLabelLLM` (module not yet refactored).

- [ ] **Step 3: Implement the refactor**

In `backend/pipeline/understand/content_map.py`:

a) Extend the module docstring's first line context (keep spec §2 reference) and add imports:

```python
import re
from collections import Counter

from .treeseg import boundary_priors, chapter_cut, divisive_segments, embed_sentences
```

b) Rename the existing `build_content_map` body to `_build_content_map_llm` **unchanged** (drop only its `n == 0` early-return, which moves to the dispatcher).

c) Add the labeling models + prompt + helpers:

```python
class SegLabelLLM(BaseModel):
    index: int = 0
    title: str = ""
    summary: str = ""
    keywords: list[str] = Field(default_factory=list)


class SegLabelsLLM(BaseModel):
    labels: list[SegLabelLLM] = Field(default_factory=list)


LABEL_SYSTEM = (
    "You are labeling ALREADY-SEGMENTED topics of a video transcript. For each numbered "
    "segment excerpt, return its index with a short specific title, a one-line summary, and "
    "a few keywords. Do NOT merge, split, re-order, or invent segments. "
    "Output only the structured result."
)

_STOP = frozenset(
    "the a an and or of to in on for with is are was were be this that it its as at by "
    "we you i so now what when where which have has had will would can could about".split())


def _excerpt(sentences: list[Sentence], s0: int, s1: int, cap: int = 420) -> str:
    """First / middle / last sentences of the segment, bounded — enough to label, cheap to send."""
    idxs = sorted({max(s0, min(s1, i)) for i in (s0, s0 + 1, (s0 + s1) // 2, s1 - 1, s1)})
    return " … ".join(p for p in ((sentences[i].text or "").strip() for i in idxs) if p)[:cap]


def _fallback_title(sentences: list[Sentence], s0: int, s1: int, ti: int) -> str:
    """Deterministic keyword title when the label LLM fails: top content words of the segment."""
    text = " ".join((sentences[i].text or "") for i in range(s0, s1 + 1)).lower()
    words = [w for w in re.findall(r"[a-zA-Z][a-zA-Z'-]{3,}", text) if w not in _STOP]
    common = [w for w, _ in Counter(words).most_common(3)]
    return " ".join(common).title() or f"Topic {ti + 1}"


def _label_segments(segments: list[tuple[int, int]], sentences: list[Sentence]) -> list[TopicLLM]:
    """One cheap LLM pass per TREESEG_LABEL_BATCH segments. Labels align by index and are
    clamped to the fixed partition — they can never re-segment. Any failure degrades to
    deterministic keyword titles; the partition is untouched."""
    from ...llm import llm_json
    out = [TopicLLM(title="", sentence_start=s0, sentence_end=s1) for s0, s1 in segments]
    for base in range(0, len(segments), config.TREESEG_LABEL_BATCH):
        batch = segments[base: base + config.TREESEG_LABEL_BATCH]
        listing = "\n".join(
            f"[{base + j}] ({sentences[s0].start:.0f}-{sentences[s1].end:.0f}s) "
            f"{_excerpt(sentences, s0, s1)}"
            for j, (s0, s1) in enumerate(batch))
        try:
            res = llm_json(LABEL_SYSTEM,
                           f"SEGMENT EXCERPTS:\n{listing}\n\n"
                           f"Label every segment [{base}]–[{base + len(batch) - 1}].",
                           SegLabelsLLM, temperature=0.1)
        except Exception:
            res = SegLabelsLLM()
        for lab in res.labels:
            i = int(lab.index)
            if base <= i < base + len(batch) and lab.title.strip():
                out[i].title = lab.title.strip()
                out[i].summary = lab.summary.strip()
                out[i].keywords = [k for k in lab.keywords if k]
    for ti, ((s0, s1), t) in enumerate(zip(segments, out)):
        if not t.title:
            t.title = _fallback_title(sentences, s0, s1, ti)
    return out
```

d) Add the treeseg builder + assembler:

```python
def _assemble_treeseg(sentences: list[Sentence], segments: list[tuple[int, int]],
                      labels: list[TopicLLM], chapters: list[tuple[int, int]]) -> ContentMap:
    n = len(sentences)
    video = ContentNode(node_id="video", level="video", title="",
                        start=sentences[0].start, end=sentences[-1].end, sentence_range=(0, n - 1))
    nodes: list[ContentNode] = [video]
    for ci, (t0, t1) in enumerate(chapters):
        cid = f"c{ci}"
        cs0, cs1 = segments[t0][0], segments[t1][1]
        chapter = ContentNode(
            node_id=cid, level="chapter", parent_id="video",
            title=labels[t0].title or f"Chapter {ci + 1}",
            start=float(sentences[cs0].start), end=float(sentences[cs1].end),
            sentence_range=(cs0, cs1))
        video.children_ids.append(cid)
        nodes.append(chapter)
        for ti in range(t0, t1 + 1):
            s0, s1 = segments[ti]
            t = labels[ti]
            tid = f"{cid}.t{ti}"
            topic = ContentNode(
                node_id=tid, level="topic", parent_id=cid,
                title=t.title or f"Topic {ti + 1}", summary=t.summary,
                start=float(sentences[s0].start), end=float(sentences[s1].end),
                sentence_range=(s0, s1), keywords=t.keywords)
            chapter.children_ids.append(tid)
            nodes.append(topic)
    return ContentMap(root_id="video", nodes=nodes, engine="treeseg")


def _build_content_map_treeseg(sentences: list[Sentence], settings: dict,
                               progress: ProgressCb = None) -> ContentMap:
    n = len(sentences)
    duration = float(sentences[-1].end) - float(sentences[0].start)
    k = int(round(duration / config.TREESEG_TARGET_TOPIC_SEC))
    k = max(config.TREESEG_MIN_TOPICS, min(config.TREESEG_MAX_TOPICS, k))
    if n < 2 * config.TREESEG_MIN_TOPIC_SENTS or k <= 1:
        segments, split_order = [(0, n - 1)], []          # too small to segment — one topic
    else:
        emb = embed_sentences(sentences)                   # raises → caller falls back to legacy
        priors = boundary_priors(sentences, config.TREESEG_PAUSE_PRIOR)
        segments, split_order = divisive_segments(
            emb, target_k=k, min_size=config.TREESEG_MIN_TOPIC_SENTS,
            coherence_floor=config.TREESEG_COHERENCE_FLOOR, priors=priors)
    if progress:
        progress(0.6, f"Segmented into {len(segments)} topics")
    labels = _label_segments(segments, sentences)
    chapters = chapter_cut(split_order, segments, config.CHAPTER_MAX_TOPICS)
    if progress:
        progress(1.0, f"Labeled {len(segments)} topics")
    return _assemble_treeseg(sentences, segments, labels, chapters)
```

e) New dispatcher (public API, signature unchanged):

```python
def build_content_map(sentences: list[Sentence], settings: dict,
                      progress: ProgressCb = None) -> ContentMap:
    """Topic partition + labels. Engine "treeseg" (default): deterministic embedding
    boundaries + LLM labels; "llm": the legacy per-chunk LLM pass. Treeseg failure
    degrades to the legacy engine (engine="llm-fallback") — never crashes the job."""
    if not sentences:
        return ContentMap(nodes=[ContentNode(node_id="video", level="video")])
    engine = str(settings.get("content_map_engine") or config.CONTENT_MAP_ENGINE)
    if engine == "treeseg":
        try:
            return _build_content_map_treeseg(sentences, settings, progress)
        except Exception:
            cm = _build_content_map_llm(sentences, settings, progress)
            cm.engine = "llm-fallback"
            return cm
    cm = _build_content_map_llm(sentences, settings, progress)
    cm.engine = "llm"
    return cm
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest backend/pipeline/understand/tests/test_content_map_treeseg.py -q`
Expected: 6 passed.

- [ ] **Step 5: Checkpoint**

Run: `.venv/bin/python -m pytest backend -q && .venv/bin/python -m compileall -q backend`
Expected: all pass, compile clean.

---

### Task 4: `build.py` degrade marker

**Files:**
- Modify: `backend/pipeline/understand/build.py` (lines 34–50)
- Test: append to `backend/pipeline/understand/tests/test_content_map_treeseg.py`

**Interfaces:**
- Consumes: `ContentMap.engine` (Task 1), `build_content_map` (Task 3).
- Produces: `Structure.degraded` contains `"content_map"` iff `engine == "llm-fallback"`.

- [ ] **Step 1: Write the failing test** (append to `test_content_map_treeseg.py`)

```python
def test_build_structure_records_content_map_fallback(monkeypatch):
    from backend.adapters import generic
    from backend.pipeline.understand import build as build_mod
    from backend.pipeline.understand.models import ContentMap, ContentNode

    def fake_cm(sentences, settings, progress=None):
        return ContentMap(nodes=[ContentNode(node_id="video", level="video",
                                             sentence_range=(0, len(sentences) - 1))],
                          engine="llm-fallback")
    monkeypatch.setattr(build_mod, "build_content_map", fake_cm)
    monkeypatch.setattr(build_mod, "extract_units", lambda *a, **kw: [])
    monkeypatch.setattr(build_mod, "build_dependency_graph",
                        lambda units, settings, progress=None: __import__(
                            "backend.pipeline.understand.models",
                            fromlist=["DependencyGraph"]).DependencyGraph())

    sents = make_sents(4)
    st = build_mod.build_structure("vidX", {"title": "t"}, sents,
                                   generic.GenericAdapter(), None, {})
    assert "content_map" in st.degraded
```

Note: `detection=None` is acceptable — `Structure.detection` has a default factory and
`build_structure` passes it through; if pydantic rejects `None`, construct
`from backend.adapters.detect import DetectionResult` and pass `DetectionResult()` instead.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest backend/pipeline/understand/tests/test_content_map_treeseg.py::test_build_structure_records_content_map_fallback -q`
Expected: FAIL — `"content_map" not in []`.

- [ ] **Step 3: Implement** — in `build.py`, replace the return-block tail:

```python
    visual_events = list(perception.visual_events) if perception else []
    degraded = list(perception.degraded) if perception else []
    if content_map.engine == "llm-fallback":
        degraded.append("content_map")             # treeseg failed; legacy LLM engine used
    return Structure(
        video_id=video_id,
        title=transcript.get("title", "") or "",
        duration=float(transcript.get("duration", 0.0) or 0.0),
        detection=detection,
        content_map=content_map,
        units=units,
        dependencies=dependencies,
        visual_events=visual_events,
        has_perception=bool(visual_events),
        degraded=degraded,
    )
```

(If Step 1's `detection=None` fails validation, keep the test's `DetectionResult()` variant.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest backend/pipeline/understand/tests/ -q`
Expected: all pass (22 in this package).

- [ ] **Step 5: Checkpoint**

Run: `.venv/bin/python -m pytest backend -q && .venv/bin/python -m compileall -q backend`
Expected: all pass, compile clean.

---

### Task 5: Real-world verification (LLM + model; not unit tests)

**Files:** none (measurement only; uses the step-1 eval harness).

- [ ] **Step 1: Determinism proof** — run twice, boundaries must be byte-identical:

```bash
cd /Users/vincentfeng/Documents/practice/clips
for i in 1 2; do .venv/bin/python - <<'EOF'
import json
from backend import config
from backend.pipeline.sentences import build_sentence_index
from backend.pipeline.understand.content_map import build_content_map
t = json.loads((config.WORK_DIR / "NjvwWiCYLl4" / "transcript.json").read_text())
sents = build_sentence_index(t)
cm = build_content_map(sents, dict(config.DEFAULTS))
print(cm.engine, [n.sentence_range for n in cm.topics()])
EOF
done
```

Expected: two identical `treeseg [...]` boundary lines (titles may differ; boundaries may not).

- [ ] **Step 2: A/B eval** (structures rebuild once under schema v4):

```bash
CONTENT_MAP_ENGINE=llm .venv/bin/python -m backend.eval.run_eval uqwC41RDPyg NjvwWiCYLl4 yfajBIaDf1Q --runs 3
CONTENT_MAP_ENGINE=treeseg .venv/bin/python -m backend.eval.run_eval uqwC41RDPyg NjvwWiCYLl4 yfajBIaDf1Q --runs 3
```

Expected: treeseg comprehension_rate mean within noise of (or above) llm; **n_clips run-to-run
std strictly smaller** under treeseg. Judge calls remain the residual noise.

- [ ] **Step 3: End-to-end smoke** on one cached video:

Run: `PRECISE_BOUNDARIES=0 .venv/bin/python -m backend.cli "https://youtu.be/NjvwWiCYLl4" "" full`
Expected: completes; prints topic count; clips produced; no `content_map` in degraded.

- [ ] **Step 4: Update docs** — RESEARCH.md roadmap row "TreeSeg content_map" → shipped;
  clipper memory file gains the TreeSeg entry (engine flag, determinism result, A/B numbers).

---

## Self-Review (done)

- **Spec coverage:** decisions 1–6 → Tasks 2/3 (boundaries+labels), granularity/K → Task 3 builder, pause prior → Tasks 2/3, legacy engine+fallback → Task 3, no subtopics → Task 3 test, baseline via env flag → Task 5; SCHEMA_VERSION+engine field → Task 1; degrade marker → Task 4; config → Task 1; edge cases → Task 2 (`test_degenerate_inputs`) + Task 3 (`test_empty_sentences`, `test_tiny_video`); measurement → Task 5. No gaps.
- **Placeholders:** none — every code step is complete.
- **Type consistency:** `divisive_segments` returns `(segments, split_order)` everywhere; `chapter_cut(split_order, segments, max_per_chapter)` matches Task 2 definition at its Task 3 call site; `SegLabelsLLM`/`SegLabelLLM` names match between Tasks 3's test and implementation; `ContentMap.engine` values `"treeseg"/"llm"/"llm-fallback"` consistent across Tasks 1/3/4.
