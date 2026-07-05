# Topic‑first clipping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new `CLIP_ENGINE=topic` clip engine that ships one ≤~60 s self‑contained window per *selected substantive teaching topic*, reusing the existing precise cutter, and route the live product to it (revertible via `CLIP_ENGINE=unit`).

**Architecture:** A new `assemble/topics.py` replaces the unit‑anchored front of assembly. It (1) runs one batched Gemini call to keep substantive teaching topics and drop filler, (2) runs one Gemini call per kept topic to pick the best `≤ CLIP_MAX_S` window (opens on framing, closes on a terminator), (3) emits clip **spec dicts** in the exact shape the orchestrator already consumes. `assemble_topic_clips()` mirrors `assemble_clips()`'s signature and 3‑tuple return, so `refine_clip_boundaries` (Whisper silence‑snap), `_build_embed_clips`, `cut_clips`, and `finish` are untouched.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, `backend.llm.llm_json` (Gemini `gemini-2.5-flash`), pytest. Run tests with `clips/.venv/bin/python -m pytest backend -q`.

## Global Constraints

- **Offline tests only.** Every test monkeypatches the LLM (`backend.pipeline.assemble.topics.llm_json`); no test performs network I/O. The suite is 735 green today.
- **Authoring LLM = Gemini via `llm_json`.** Call `from ...llm import llm_json` → `llm_json(system: str, user: str, SchemaClass, temperature=..., model=None)` returns a validated pydantic instance; it raises on final failure. Default model is `gemini-2.5-flash`. **Never** call `gemini_client.generate_json` directly.
- **`Sentence` field names (dataclass, `backend/pipeline/sentences.py`):** `idx:int, text:str, start:float, end:float, terminator:str ('.'/'?'/'!'/''), ends_with_period:bool (True for any terminator), word_start_idx:int, word_end_idx:int, align_confidence:float, warnings:tuple`. It is `idx` (NOT `index`); there is no `tokenIds`. Never infer a terminator from `text[-1]` — read `ends_with_period`/`terminator`.
- **`ContentNode.sentence_range` is half‑open `(i0, i1)`** (i0 inclusive, i1 exclusive), same as `Unit.sentence_range`.
- **Spec dict contract:** `_build_embed_clips` (orchestrator.py:41) reads every key via `.get()` **except `c["start"]` and `c["end"]`** (mandatory floats). `final_quality` may be `None`. `cut_clips` reads `start,end,cut_end(optional),facet,reason`. So a topic spec MUST set `start`+`end`; everything else is defaulted downstream.
- **`assemble_topic_clips` must be a SYNC callable** returning exactly `tuple[list[dict], str, list[Rejection]]` and must fill the caller‑owned `stats` dict. `rejections` must be `Rejection` objects (they carry a `.stage` attr the orchestrator reads).
- **Invariant preserved trivially:** the topic engine never runs the killing judge in v1 — nothing is dropped by an LLM verdict, so `unverified_kill = 0` holds by construction.
- **Rollout:** default `CLIP_ENGINE=topic`; `CLIP_ENGINE=unit` instantly reverts to the legacy engine. Deleting the unit machinery is a SEPARATE follow‑up (see "Later"), gated on a live real‑video pass.
- **After code changes:** restart uvicorn (`.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000`); rebuild the frontend only if a `.tsx` changed (this plan changes none).

## File Structure

- **Create** `backend/pipeline/assemble/topics.py` — the whole topic engine: LLM schemas (`TopicJudgment`, `TopicSelection`, `WindowChoice`), data (`TopicPick`, `Window`), and functions `select_topics`, `extract_best_window`, `assemble_topic_clips`, helpers `_topic_prompt`, `_window_prompt`, `_snap_end_to_terminator`, `_fit_budget`.
- **Create** `backend/pipeline/assemble/tests/test_topics_select.py` — `select_topics` unit tests.
- **Create** `backend/pipeline/assemble/tests/test_topics_window.py` — `extract_best_window` unit tests.
- **Create** `backend/pipeline/assemble/tests/test_topics_assemble.py` — `assemble_topic_clips` integration (mocked LLM) tests.
- **Create** `backend/tests/test_clip_engine_switch.py` — orchestrator/CLI routing test.
- **Modify** `backend/config.py` — add the `CLIP_ENGINE` block + one `DEFAULTS` key.
- **Modify** `backend/orchestrator.py:314-319` — engine switch at the `assemble_clips` call site.
- **Modify** `backend/cli.py` — mirror the switch (dev parity).

---

### Task 1: Config knobs + `CLIP_ENGINE` flag

**Files:**
- Modify: `backend/config.py` (add a block after line 218 `FEED_DEFAULT_PROFILE`; add one key to `DEFAULTS` ~line 179)
- Test: `backend/tests/test_config_topic.py` (create)

**Interfaces:**
- Produces: `config.CLIP_ENGINE:str`, `config.TOPIC_MAX_CLIPS:int`, `config.CLIP_TARGET_S:float`, `config.CLIP_MAX_S:float`, `config.TOPIC_INFORMATIVENESS_MIN:float`, `config.TOPIC_BOUNDARY_WINDOW:int`; `config.DEFAULTS["clip_engine"] = None`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_config_topic.py
from backend import config

def test_topic_engine_defaults():
    assert config.CLIP_ENGINE in ("topic", "unit")
    assert config.CLIP_ENGINE == "topic"          # default routes to the new engine
    assert config.TOPIC_MAX_CLIPS == 10
    assert config.CLIP_MAX_S == 75.0
    assert config.CLIP_TARGET_S == 58.0
    assert config.TOPIC_INFORMATIVENESS_MIN == 0.5
    assert config.TOPIC_BOUNDARY_WINDOW == 3
    assert "clip_engine" in config.DEFAULTS and config.DEFAULTS["clip_engine"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd clips && .venv/bin/python -m pytest backend/tests/test_config_topic.py -q`
Expected: FAIL (`AttributeError: module 'backend.config' has no attribute 'CLIP_ENGINE'`).

- [ ] **Step 3: Add the config block**

In `backend/config.py`, after the `FEED_DEFAULT_PROFILE = ...` line (~218), add:

```python
# ── Topic-first clip engine (CLIP_ENGINE=topic) ─────────────────────────────
# "topic": select substantive teaching topics from the content_map, then ship ONE
# best <=CLIP_MAX_S self-contained window per topic. "unit": legacy unit-anchored
# assemble_clips (revert switch). See docs/superpowers/specs/2026-07-04-topic-first-clipping-design.md
CLIP_ENGINE = os.environ.get("CLIP_ENGINE", "topic")            # "topic" | "unit"
TOPIC_MAX_CLIPS = int(os.environ.get("TOPIC_MAX_CLIPS", "10"))  # max clips (one window per kept topic)
CLIP_TARGET_S = float(os.environ.get("CLIP_TARGET_S", "58"))    # window length aim
CLIP_MAX_S = float(os.environ.get("CLIP_MAX_S", "75"))          # hard-ish ceiling (finish the sentence)
TOPIC_INFORMATIVENESS_MIN = float(os.environ.get("TOPIC_INFORMATIVENESS_MIN", "0.5"))
TOPIC_BOUNDARY_WINDOW = int(os.environ.get("TOPIC_BOUNDARY_WINDOW", "3"))  # sentences of slack each side
```

Then in the `DEFAULTS` dict (after `"edge_probe": None,` ~line 178) add:

```python
    "clip_engine": None,                    # None → inherit config.CLIP_ENGINE ("topic"|"unit")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd clips && .venv/bin/python -m pytest backend/tests/test_config_topic.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/config.py backend/tests/test_config_topic.py
git commit -m "feat(config): add CLIP_ENGINE=topic knobs (target/max/informativeness/window)"
```

---

### Task 2: `topics.py` schemas + data types

**Files:**
- Create: `backend/pipeline/assemble/topics.py`
- Test: `backend/pipeline/assemble/tests/test_topics_types.py` (create)

**Interfaces:**
- Produces: pydantic `TopicJudgment{node_id:str, type:str, informativeness:float, self_contained:float, why:str}`, `TopicSelection{topics:list[TopicJudgment]}`, `WindowChoice{start_idx:int, end_idx:int, title:str, why:str}`; dataclasses `TopicPick{node:ContentNode, type:str, informativeness:float, self_contained:float, why:str, warnings:tuple}`, `Window{node_id:str, start_idx:int, end_idx:int, start_s:float, end_s:float, title:str, facet:str, why:str, warnings:tuple}`.

- [ ] **Step 1: Write the failing test**

```python
# backend/pipeline/assemble/tests/test_topics_types.py
from backend.pipeline.assemble import topics as T

def test_schemas_and_dataclasses_exist():
    sel = T.TopicSelection(topics=[T.TopicJudgment(node_id="c0.t1", type="teaching",
                                                   informativeness=0.9, self_contained=0.8, why="core")])
    assert sel.topics[0].node_id == "c0.t1"
    ch = T.WindowChoice(start_idx=3, end_idx=9, title="Reflex arc", why="mechanism+example")
    assert (ch.start_idx, ch.end_idx) == (3, 9)
    w = T.Window(node_id="c0.t1", start_idx=3, end_idx=9, start_s=10.0, end_s=65.0,
                 title="Reflex arc", facet="teaching", why="", warnings=())
    assert w.end_s - w.start_s == 55.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd clips && .venv/bin/python -m pytest backend/pipeline/assemble/tests/test_topics_types.py -q`
Expected: FAIL (`ModuleNotFoundError: backend.pipeline.assemble.topics`).

- [ ] **Step 3: Create the module skeleton with schemas + types**

```python
# backend/pipeline/assemble/topics.py
"""Topic-first clip assembly (CLIP_ENGINE=topic).

One clip per SELECTED substantive teaching topic: a batched LLM selection judge drops
filler (intro/outro/transition/promo/tangent), then per kept topic an LLM picks the best
self-contained <=CLIP_MAX_S window (opens on framing, closes on a terminator). The chosen
sentence spans become clip spec dicts fed to the SAME precise cutter the unit engine uses.
Spec: docs/superpowers/specs/2026-07-04-topic-first-clipping-design.md
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel, Field

from ... import config
from ...llm import llm_json
from ..sentences import Sentence
from ..understand.models import ContentNode, Structure
from .integrity import Rejection


# ── LLM schemas ──────────────────────────────────────────────────────────────
class TopicJudgment(BaseModel):
    node_id: str
    type: str = "teaching"          # teaching|intro|outro|transition|admin|promo|tangent
    informativeness: float = 0.0    # 0..1, standalone value
    self_contained: float = 0.0     # 0..1
    why: str = ""


class TopicSelection(BaseModel):
    topics: list[TopicJudgment] = Field(default_factory=list)


class WindowChoice(BaseModel):
    start_idx: int
    end_idx: int
    title: str = ""
    why: str = ""


# ── internal data ────────────────────────────────────────────────────────────
@dataclass
class TopicPick:
    node: ContentNode
    type: str
    informativeness: float
    self_contained: float
    why: str
    warnings: tuple[str, ...] = ()


@dataclass
class Window:
    node_id: str
    start_idx: int
    end_idx: int
    start_s: float
    end_s: float
    title: str
    facet: str
    why: str
    warnings: tuple[str, ...] = ()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd clips && .venv/bin/python -m pytest backend/pipeline/assemble/tests/test_topics_types.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline/assemble/topics.py backend/pipeline/assemble/tests/test_topics_types.py
git commit -m "feat(topics): topic-engine LLM schemas + TopicPick/Window types"
```

---

### Task 3: `select_topics` — the selection judge (drops filler)

**Files:**
- Modify: `backend/pipeline/assemble/topics.py`
- Test: `backend/pipeline/assemble/tests/test_topics_select.py` (create)

**Interfaces:**
- Consumes: `Structure.content_map.topics() -> list[ContentNode]`, `llm_json(...) -> TopicSelection`.
- Produces: `select_topics(structure: Structure, sentences: list[Sentence], settings: dict) -> tuple[list[TopicPick], list[TopicPick]]` returning `(kept, dropped)`. `kept` is chronological (by `node.start`), capped at `max_clips`; `dropped` is every non‑kept topic.

- [ ] **Step 1: Write the failing test**

```python
# backend/pipeline/assemble/tests/test_topics_select.py
import backend.pipeline.assemble.topics as T
from backend.pipeline.understand.models import ContentMap, ContentNode, Structure
from backend.pipeline.sentences import Sentence


def _sent(idx, text, start, end, term="."):
    return Sentence(idx, text, start, end, term, term in ".?!", idx, idx, 1.0, ())


def _structure(nodes):
    return Structure(video_id="vid", content_map=ContentMap(nodes=nodes, engine="treeseg"))


def _node(nid, title, i0, i1, start, end):
    return ContentNode(node_id=nid, level="topic", title=title, summary=title,
                       start=start, end=end, sentence_range=(i0, i1), keywords=[])


# 4 topics: intro (drop), reflex arc (keep), sponsor promo (drop), pitch (keep)
NODES = [
    _node("t0", "Intro", 0, 1, 0.0, 30.0),
    _node("t1", "Reflex arc", 1, 2, 30.0, 120.0),
    _node("t2", "Subscribe promo", 2, 3, 120.0, 140.0),
    _node("t3", "Pitch and frequency", 3, 4, 140.0, 220.0),
]
SENTS = [_sent(0, "Welcome to the channel.", 0, 30),
         _sent(1, "A reflex arc lets you react without thinking.", 30, 120),
         _sent(2, "Smash subscribe and hit the bell.", 120, 140),
         _sent(3, "Frequency determines the pitch of a sound.", 140, 220)]

JUDGMENTS = T.TopicSelection(topics=[
    T.TopicJudgment(node_id="t0", type="intro", informativeness=0.1, self_contained=0.2),
    T.TopicJudgment(node_id="t1", type="teaching", informativeness=0.9, self_contained=0.8),
    T.TopicJudgment(node_id="t2", type="promo", informativeness=0.0, self_contained=0.1),
    T.TopicJudgment(node_id="t3", type="teaching", informativeness=0.7, self_contained=0.7),
])


def test_select_keeps_teaching_drops_filler(monkeypatch):
    monkeypatch.setattr(T, "llm_json", lambda *a, **k: JUDGMENTS)
    kept, dropped = T.select_topics(_structure(NODES), SENTS, {})
    assert [p.node.node_id for p in kept] == ["t1", "t3"]     # chronological
    assert {p.node.node_id for p in dropped} == {"t0", "t2"}


def test_select_respects_max_clips(monkeypatch):
    monkeypatch.setattr(T, "llm_json", lambda *a, **k: JUDGMENTS)
    kept, _ = T.select_topics(_structure(NODES), SENTS, {"max_clips": 1})
    assert [p.node.node_id for p in kept] == ["t3"]           # highest informativeness first, then chrono


def test_select_never_zero_on_llm_failure(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("llm down")
    monkeypatch.setattr(T, "llm_json", boom)
    kept, _ = T.select_topics(_structure(NODES), SENTS, {})
    assert len(kept) >= 1                                     # unknown ⇒ neutral-teaching, never empty
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd clips && .venv/bin/python -m pytest backend/pipeline/assemble/tests/test_topics_select.py -q`
Expected: FAIL (`AttributeError: module ... has no attribute 'select_topics'`).

- [ ] **Step 3: Implement `select_topics` + `_topic_prompt`**

Append to `backend/pipeline/assemble/topics.py`:

```python
_TEACHING = "teaching"

SELECT_SYSTEM = (
    "You curate short, self-contained TEACHING clips from a video's topic outline. For EACH "
    "topic decide its type and how informative it is ON ITS OWN, judged RELATIVE to the whole "
    "video. type is one of: teaching (explains a concept / mechanism / definition-with-"
    "explanation / worked idea / self-contained argument or story), intro, outro, transition, "
    "admin, promo (subscribe / patreon / 'grab the packet'), tangent. informativeness and "
    "self_contained are 0..1. A topic that only welcomes, thanks, recaps, or promotes is NOT "
    "teaching. Return exactly one entry per topic id, using the ids given."
)


def _topic_prompt(topics: list[ContentNode], sentences: list[Sentence]) -> str:
    lines = []
    n = len(sentences)
    for node in topics:
        i0, i1 = node.sentence_range
        first = sentences[i0].text if 0 <= i0 < n else ""
        last = sentences[i1 - 1].text if 0 < i1 <= n else ""
        kw = ", ".join(node.keywords[:8])
        lines.append(
            f"[{node.node_id}] title={node.title!r}"
            + (f" keywords=[{kw}]" if kw else "")
            + (f"\n    summary: {node.summary}" if node.summary else "")
            + f"\n    opens: {first!r}\n    closes: {last!r}"
        )
    return "TOPICS:\n" + "\n".join(lines)


def select_topics(structure: Structure, sentences: list[Sentence],
                  settings: dict) -> tuple[list[TopicPick], list[TopicPick]]:
    """Keep substantive teaching topics; drop filler. Returns (kept, dropped).

    kept is chronological (by node.start) and capped at max_clips; dropped is every
    non-kept topic (for the rejection ledger). Never returns an empty kept list when
    the video has topics (LLM failure ⇒ neutral-teaching fallback)."""
    topics = structure.content_map.topics() or structure.content_map.chapters()
    if not topics:
        return [], []
    thr = float(settings.get("informativeness_min") or config.TOPIC_INFORMATIVENESS_MIN)
    cap = int(settings.get("max_clips") or config.TOPIC_MAX_CLIPS)
    by_id = {node.node_id: node for node in topics}
    try:
        sel = llm_json(SELECT_SYSTEM, _topic_prompt(topics, sentences),
                       TopicSelection, temperature=0.1)
        judged = {j.node_id: j for j in sel.topics if j.node_id in by_id}
    except Exception:
        judged = {}

    picks: list[TopicPick] = []
    for nid, node in by_id.items():
        j = judged.get(nid)
        if j is None:                       # unknown ⇒ neutral teaching (never silently lost)
            picks.append(TopicPick(node, _TEACHING, 0.5, 0.5, ""))
        else:
            picks.append(TopicPick(node, (j.type or _TEACHING).strip().lower(),
                                   float(j.informativeness), float(j.self_contained), j.why))

    kept = [p for p in picks if p.type == _TEACHING and p.informativeness >= thr]
    if not kept:                            # never zero on a real teaching video
        kept = sorted(picks, key=lambda p: p.informativeness, reverse=True)[:max(1, min(cap, len(picks)))]
        kept = [TopicPick(p.node, p.type, p.informativeness, p.self_contained, p.why,
                          ("low_confidence_selection",)) for p in kept]

    kept.sort(key=lambda p: p.informativeness, reverse=True)
    kept = kept[:cap]
    kept.sort(key=lambda p: p.node.start)   # chronological for downstream
    kept_ids = {p.node.node_id for p in kept}
    dropped = [p for p in picks if p.node.node_id not in kept_ids]
    return kept, dropped
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd clips && .venv/bin/python -m pytest backend/pipeline/assemble/tests/test_topics_select.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline/assemble/topics.py backend/pipeline/assemble/tests/test_topics_select.py
git commit -m "feat(topics): select_topics — batched LLM keeps teaching, drops filler"
```

---

### Task 4: `extract_best_window` — best ≤CLIP_MAX_S self‑contained window

**Files:**
- Modify: `backend/pipeline/assemble/topics.py`
- Test: `backend/pipeline/assemble/tests/test_topics_window.py` (create)

**Interfaces:**
- Consumes: `TopicPick`, `list[Sentence]`, `llm_json(...) -> WindowChoice`.
- Produces: `extract_best_window(pick: TopicPick, sentences: list[Sentence], settings: dict) -> Optional[Window]`; helpers `_window_prompt(sentences, lo, hi) -> str`, `_snap_end_to_terminator(sentences, a, b, warnings) -> int`, `_fit_budget(sentences, a, b, max_s, warnings) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# backend/pipeline/assemble/tests/test_topics_window.py
import backend.pipeline.assemble.topics as T
from backend.pipeline.understand.models import ContentNode
from backend.pipeline.sentences import Sentence


def _sent(idx, text, start, end, term="."):
    return Sentence(idx, text, start, end, term, term in ".?!", idx, idx, 1.0, ())


def _pick(i0, i1, start, end, title="Topic"):
    node = ContentNode(node_id="t1", level="topic", title=title, summary=title,
                       start=start, end=end, sentence_range=(i0, i1), keywords=[])
    return T.TopicPick(node, "teaching", 0.9, 0.8, "")


# 6 sentences, ~10s each. Topic node covers [1,6): first sentence (idx1) dangles.
SENTS = [
    _sent(0, "A reflex arc lets you react without thinking.", 0, 10),
    _sent(1, "These neurons carry the signal.", 10, 20),       # dangling opener
    _sent(2, "The pathway runs sensory to motor.", 20, 30),
    _sent(3, "Touch a hot stove and your hand pulls back.", 30, 40),
    _sent(4, "That is the reflex arc at work.", 40, 50),
    _sent(5, "Anyway, moving on to something else.", 50, 60),
]


def test_window_moves_start_off_dangling_opener(monkeypatch):
    # LLM chooses to open at the framing sentence (idx0), close at idx4
    monkeypatch.setattr(T, "llm_json", lambda *a, **k: T.WindowChoice(start_idx=0, end_idx=4, title="Reflex arc"))
    w = T.extract_best_window(_pick(1, 6, 10.0, 60.0), SENTS, {})
    assert w.start_idx == 0 and w.end_idx == 4
    assert w.start_s == 0.0 and w.end_s == 50.0


def test_window_truncates_to_budget(monkeypatch):
    # LLM over-reaches (0..5 = 60s); CLIP_MAX_S small ⇒ walk end back to a terminator within budget
    monkeypatch.setattr(T, "llm_json", lambda *a, **k: T.WindowChoice(start_idx=0, end_idx=5))
    w = T.extract_best_window(_pick(0, 6, 0.0, 60.0), SENTS, {"clip_max_s": 35.0})
    assert w.end_s - w.start_s <= 35.0
    assert SENTS[w.end_idx].ends_with_period
    assert "window_truncated_to_budget" in w.warnings


def test_window_snaps_end_to_terminator(monkeypatch):
    seq = [_sent(0, "First point.", 0, 10),
           _sent(1, "Second point that trails off", 10, 20, term=""),   # no terminator
           _sent(2, "Third point ends here.", 20, 30)]
    node = ContentNode(node_id="t1", level="topic", title="X", start=0, end=30, sentence_range=(0, 3))
    pick = T.TopicPick(node, "teaching", 0.9, 0.8, "")
    monkeypatch.setattr(T, "llm_json", lambda *a, **k: T.WindowChoice(start_idx=0, end_idx=1))
    w = T.extract_best_window(pick, seq, {})
    assert seq[w.end_idx].ends_with_period          # walked back to idx0


def test_window_clamps_out_of_range(monkeypatch):
    monkeypatch.setattr(T, "llm_json", lambda *a, **k: T.WindowChoice(start_idx=99, end_idx=999))
    w = T.extract_best_window(_pick(1, 6, 10.0, 60.0), SENTS, {})
    assert 0 <= w.start_idx <= w.end_idx <= len(SENTS) - 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd clips && .venv/bin/python -m pytest backend/pipeline/assemble/tests/test_topics_window.py -q`
Expected: FAIL (`AttributeError: ... 'extract_best_window'`).

- [ ] **Step 3: Implement `extract_best_window` + helpers**

Append to `backend/pipeline/assemble/topics.py`:

```python
WINDOW_SYSTEM = (
    "You trim ONE topic to the single best self-contained clip. Choose inclusive sentence "
    "indices start_idx and end_idx from the list. The clip MUST: open on a sentence that frames "
    "the idea for a COLD viewer (never a dangling 'this/these/that/it' pointing outside the clip); "
    "close on a sentence that completes a thought; and carry the topic's core (definition+"
    "explanation, or mechanism+example), dropping recaps, meta-asides and tangents. Keep it at most "
    "{max_s} seconds (aim ~{target_s}s). The '+Ns' marker after each index is elapsed seconds; a "
    "trailing ' .' marks a sentence that ends on a terminator — prefer those as the close."
)


def _window_prompt(sentences: list[Sentence], lo: int, hi: int) -> str:
    t0 = sentences[lo].start
    out = []
    for i in range(lo, hi + 1):
        s = sentences[i]
        mark = " ." if s.ends_with_period else ""
        out.append(f"[{i}] (+{s.end - t0:.0f}s){mark} {s.text}")
    return "SENTENCES:\n" + "\n".join(out)


def _snap_end_to_terminator(sentences: list[Sentence], a: int, b: int, warnings: list[str]) -> int:
    """Walk the end back to the nearest terminator-ending sentence >= a."""
    j = b
    while j > a and not sentences[j].ends_with_period:
        j -= 1
    if not sentences[j].ends_with_period:           # none in [a, b]
        warnings.append("window_close_forced")
        return b
    if j != b:
        warnings.append("window_close_snapped")
    return j


def _fit_budget(sentences: list[Sentence], a: int, b: int, max_s: float, warnings: list[str]) -> int:
    """Truncate the end to the last terminator-ending sentence within max_s of the start."""
    if sentences[b].end - sentences[a].start <= max_s:
        return b
    j = b
    while j > a and sentences[j].end - sentences[a].start > max_s:
        j -= 1
    k = j                                            # prefer a terminator within budget
    while k > a and not sentences[k].ends_with_period:
        k -= 1
    warnings.append("window_truncated_to_budget")
    return k if sentences[k].ends_with_period and k > a else j


def extract_best_window(pick: TopicPick, sentences: list[Sentence],
                        settings: dict) -> Optional[Window]:
    """Pick the best <=CLIP_MAX_S self-contained window inside (and just before) a topic."""
    node = pick.node
    i0, i1 = node.sentence_range                     # half-open
    win = int(settings.get("boundary_window") or config.TOPIC_BOUNDARY_WINDOW)
    lo = max(0, i0 - win)                             # allow opening a little earlier
    hi = min(len(sentences) - 1, i1 - 1)
    if hi < lo:
        return None
    max_s = float(settings.get("clip_max_s") or config.CLIP_MAX_S)
    try:
        sys = WINDOW_SYSTEM.format(max_s=int(max_s), target_s=int(config.CLIP_TARGET_S))
        ch = llm_json(sys, _window_prompt(sentences, lo, hi), WindowChoice, temperature=0.1)
        a, b, title, why = int(ch.start_idx), int(ch.end_idx), ch.title, ch.why
    except Exception:
        a, b, title, why = i0, hi, node.title, ""    # fall back to the whole topic span

    a = min(max(a, lo), hi)                           # clamp into the shown range
    b = min(max(b, a), hi)
    warnings: list[str] = list(pick.warnings)
    b = _snap_end_to_terminator(sentences, a, b, warnings)
    b = _fit_budget(sentences, a, b, max_s, warnings)
    return Window(node.node_id, a, b, sentences[a].start, sentences[b].end,
                  title or node.title, pick.type, why, tuple(warnings))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd clips && .venv/bin/python -m pytest backend/pipeline/assemble/tests/test_topics_window.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline/assemble/topics.py backend/pipeline/assemble/tests/test_topics_window.py
git commit -m "feat(topics): extract_best_window — framing open, terminator close, budget fit"
```

---

### Task 5: `assemble_topic_clips` — the engine entry (specs + rejections + stats)

**Files:**
- Modify: `backend/pipeline/assemble/topics.py`
- Test: `backend/pipeline/assemble/tests/test_topics_assemble.py` (create)

**Interfaces:**
- Consumes: `select_topics`, `extract_best_window`, `Rejection(cand_id, title, role, stage, reason, start=, end=)`.
- Produces: `assemble_topic_clips(structure, topic, sentences, url, video_id, settings, adapter, progress=None, stats=None) -> tuple[list[dict], str, list[Rejection]]` — SAME shape as `assemble_clips`. Each spec dict has at least `start,end,cut_end,facet,reason,title,sentence_start_idx,sentence_end_idx,role,context_card,unit_ids,final_quality,warnings,ship_flagged,notes,sequence_index,prerequisite_clips`. Fills `stats["n_topics_total"|"n_topics_kept"|"n_topics_dropped"]`.

- [ ] **Step 1: Write the failing test**

```python
# backend/pipeline/assemble/tests/test_topics_assemble.py
import backend.pipeline.assemble.topics as T
from backend.pipeline.understand.models import ContentMap, ContentNode, Structure
from backend.pipeline.sentences import Sentence


def _sent(idx, text, start, end, term="."):
    return Sentence(idx, text, start, end, term, term in ".?!", idx, idx, 1.0, ())


def _node(nid, title, i0, i1, start, end):
    return ContentNode(node_id=nid, level="topic", title=title, summary=title,
                       start=start, end=end, sentence_range=(i0, i1), keywords=[])


NODES = [_node("t0", "Intro", 0, 1, 0.0, 10.0),
         _node("t1", "Reflex arc", 1, 3, 10.0, 30.0),
         _node("t2", "Pitch", 3, 5, 30.0, 50.0)]
SENTS = [_sent(0, "Welcome everyone.", 0, 10),
         _sent(1, "A reflex arc lets you react.", 10, 20),
         _sent(2, "Touch a hot stove and pull back.", 20, 30),
         _sent(3, "Frequency sets the pitch.", 30, 40),
         _sent(4, "High frequency is high pitch.", 40, 50)]


def _fake_select(structure, sentences, settings):
    keep = [T.TopicPick(NODES[1], "teaching", 0.9, 0.8, "mechanism"),
            T.TopicPick(NODES[2], "teaching", 0.7, 0.7, "definition")]
    drop = [T.TopicPick(NODES[0], "intro", 0.1, 0.2, "welcome")]
    return keep, drop


def _fake_window(pick, sentences, settings):
    i0, i1 = pick.node.sentence_range
    return T.Window(pick.node.node_id, i0, i1 - 1, sentences[i0].start, sentences[i1 - 1].end,
                    pick.node.title, pick.type, pick.why, ())


def test_assemble_builds_specs_drops_filler(monkeypatch):
    monkeypatch.setattr(T, "select_topics", _fake_select)
    monkeypatch.setattr(T, "extract_best_window", _fake_window)
    st = Structure(video_id="vid", content_map=ContentMap(nodes=NODES, engine="treeseg"))
    stats = {}
    specs, notes, rejections = T.assemble_topic_clips(
        st, "reflexes", SENTS, "http://x", "vid", {}, adapter=None, stats=stats)
    assert len(specs) == 2
    assert [s["title"] for s in specs] == ["Reflex arc", "Pitch"]     # chronological
    assert [s["sequence_index"] for s in specs] == [1, 2]
    for s in specs:                                                   # mandatory keys present
        assert isinstance(s["start"], float) and isinstance(s["end"], float)
        assert s["cut_end"] >= s["end"]
    assert [r.stage for r in rejections] == ["topic_select"]          # the dropped intro
    assert stats["n_topics_kept"] == 2 and stats["n_topics_total"] == 3


def test_assemble_empty_when_no_topics(monkeypatch):
    st = Structure(video_id="vid", content_map=ContentMap(nodes=[], engine="treeseg"))
    specs, notes, rejections = T.assemble_topic_clips(
        st, "x", SENTS, "u", "vid", {}, adapter=None, stats={})
    assert specs == [] and "segment" in notes.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd clips && .venv/bin/python -m pytest backend/pipeline/assemble/tests/test_topics_assemble.py -q`
Expected: FAIL (`AttributeError: ... 'assemble_topic_clips'`).

- [ ] **Step 3: Implement `assemble_topic_clips`**

Append to `backend/pipeline/assemble/topics.py`:

```python
def assemble_topic_clips(structure: Structure, topic: str, sentences: list[Sentence], url: str,
                         video_id: str, settings: dict, adapter,
                         progress=None, stats: Optional[dict] = None) -> tuple[list[dict], str, list[Rejection]]:
    """CLIP_ENGINE=topic entry. Mirrors assemble_clips' (specs, notes, rejections) contract."""
    stats = stats if stats is not None else {}

    def emit(frac: float, msg: str = "") -> None:
        if progress:
            progress(max(0.0, min(1.0, frac)), msg)

    rejections: list[Rejection] = []
    if not sentences:
        return [], "No transcript was available to clip.", rejections
    topics = structure.content_map.topics() or structure.content_map.chapters()
    if not topics:
        return [], "This video couldn't be segmented into topics.", rejections

    emit(0.05, "Selecting substantive topics…")
    kept, dropped = select_topics(structure, sentences, settings)
    stats["n_topics_total"] = len(topics)
    stats["n_topics_kept"] = len(kept)
    stats["n_topics_dropped"] = len(dropped)
    for p in dropped:
        rejections.append(Rejection(
            cand_id=p.node.node_id, title=p.node.title or "", role=p.type, stage="topic_select",
            reason=f"dropped as {p.type} (informativeness {p.informativeness:.2f})",
            start=p.node.start, end=p.node.end))
    if not kept:
        return [], "No substantive teaching topics were found in this video.", rejections

    emit(0.2, "Trimming to the best windows…")
    workers = max(1, min(config.UNDERSTAND_WORKERS, len(kept)))
    windows: list[Optional[Window]] = [None] * len(kept)
    if workers == 1:
        for i, p in enumerate(kept):
            windows[i] = extract_best_window(p, sentences, settings)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(extract_best_window, p, sentences, settings): i
                    for i, p in enumerate(kept)}
            for fut in as_completed(futs):
                try:
                    windows[futs[fut]] = fut.result()
                except Exception:
                    windows[futs[fut]] = None       # one bad window never kills the batch

    tail = float(settings.get("tail_pad_s") or config.DEFAULTS["tail_pad_s"])
    specs: list[dict] = []
    for w in windows:
        if w is None:
            continue
        specs.append({
            "start": round(w.start_s, 3),
            "end": round(w.end_s, 3),
            "cut_end": round(w.end_s + tail, 3),
            "facet": (w.facet or "other"),
            "reason": w.why,
            "title": w.title,
            "role": "",
            "context_card": "",
            "sentence_start_idx": w.start_idx,
            "sentence_end_idx": w.end_idx,
            "unit_ids": [],
            "final_quality": None,
            "warnings": w.warnings,
            "ship_flagged": False,
            "notes": [],
        })

    specs.sort(key=lambda s: s["start"])
    for i, s in enumerate(specs):
        s["sequence_index"] = i + 1
        s["prerequisite_clips"] = []

    emit(1.0, f"Assembled {len(specs)} clip(s)")
    notes = (f"{len(specs)} clip(s) from {len(kept)} topic(s)"
             + (f"; {len(dropped)} filler topic(s) dropped." if dropped else "."))
    return specs, notes, rejections
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd clips && .venv/bin/python -m pytest backend/pipeline/assemble/tests/test_topics_assemble.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline/assemble/topics.py backend/pipeline/assemble/tests/test_topics_assemble.py
git commit -m "feat(topics): assemble_topic_clips — specs + filler rejections + stats"
```

---

### Task 6: Wire the `CLIP_ENGINE` switch (orchestrator + CLI)

**Files:**
- Modify: `backend/orchestrator.py:314-319`
- Modify: `backend/cli.py` (the `assemble_clips(...)` call site, ~line 86)
- Test: `backend/tests/test_clip_engine_switch.py` (create)

**Interfaces:**
- Consumes: `config.CLIP_ENGINE`, `settings["clip_engine"]`, `assemble_topic_clips`, legacy `assemble_clips`.
- Produces: both entry points route to `assemble_topic_clips` when `engine == "topic"`, else `assemble_clips`, preserving the identical 3‑tuple.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_clip_engine_switch.py
from backend import config


def _resolve(settings):
    # mirrors the one-liner used at both call sites
    return str(settings.get("clip_engine") or config.CLIP_ENGINE).lower()


def test_engine_resolution():
    assert _resolve({}) == "topic"                       # inherits config default
    assert _resolve({"clip_engine": "unit"}) == "unit"   # explicit override wins
    assert _resolve({"clip_engine": None}) == config.CLIP_ENGINE


def test_orchestrator_imports_topic_engine():
    # the topic engine is importable where the orchestrator wires it
    from backend.pipeline.assemble.topics import assemble_topic_clips
    assert callable(assemble_topic_clips)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd clips && .venv/bin/python -m pytest backend/tests/test_clip_engine_switch.py -q`
Expected: PASS on `test_engine_resolution` but this test file also documents the wiring; run the FULL suite after Step 3 to confirm no regressions. (If `assemble_topic_clips` import fails, Tasks 2–5 are incomplete.)

- [ ] **Step 3: Add the import + switch in `orchestrator.py`**

At the top of `backend/orchestrator.py`, with the other assemble imports, add:

```python
from .pipeline.assemble.topics import assemble_topic_clips
```

Replace lines 314‑319 (`registry.publish(... "assembling" ...)` through the `assemble_clips` call) with:

```python
    registry.publish(job, ProgressEvent("assembling", 72.0, "Assembling self-contained clips…"))
    stats: dict = {}                           # machine-readable run signals (I1/W25-G)
    engine = str(settings.get("clip_engine") or config.CLIP_ENGINE).lower()
    _assemble = assemble_topic_clips if engine == "topic" else assemble_clips
    clips_spec, notes, rejections = await run(
        _assemble, structure, job.topic, sentences, job.url, video_id,
        settings, adapter, emit("assembling", 72, 90), stats,
    )
```

- [ ] **Step 4: Mirror the switch in `cli.py`**

In `backend/cli.py`, add near the top imports:

```python
from backend.pipeline.assemble.topics import assemble_topic_clips
```

Replace the `clips_spec, notes, rejections = assemble_clips(st, topic, sents, url, video_id, settings, adapter, _p("assemble"))` call (~line 86) with:

```python
    engine = str(settings.get("clip_engine") or config.CLIP_ENGINE).lower()
    _assemble = assemble_topic_clips if engine == "topic" else assemble_clips
    clips_spec, notes, rejections = _assemble(
        st, topic, sents, url, video_id, settings, adapter, _p("assemble"))
```

- [ ] **Step 5: Run the FULL suite to verify no regression**

Run: `cd clips && .venv/bin/python -m pytest backend -q`
Expected: PASS (previous 735 + the new topic tests; the legacy `unit`‑engine tests still pass because the switch defaults them only when `clip_engine="unit"` is set — the default is now `topic`, so any test that asserts *unit‑engine* clip content must set `settings["clip_engine"]="unit"`; see Step 6).

- [ ] **Step 6: Fix any legacy assemble tests that assert unit‑engine output**

If any existing test drives the full `run_pipeline`/`_run_full` path and asserts unit‑anchored clip content, add `settings["clip_engine"] = "unit"` (or `job.settings[...]`) to that test's setup so it keeps exercising the legacy engine. Do NOT change `assemble_clips` itself. Re‑run `pytest backend -q` until green.

- [ ] **Step 7: Commit**

```bash
git add backend/orchestrator.py backend/cli.py backend/tests/test_clip_engine_switch.py
git commit -m "feat(engine): route CLIP_ENGINE=topic to assemble_topic_clips (unit = revert switch)"
```

---

### Task 7: End‑to‑end integration test (synthetic structure)

**Files:**
- Test: `backend/pipeline/assemble/tests/test_topics_integration.py` (create)

**Interfaces:**
- Consumes: `assemble_topic_clips` with a real `Structure` + `Sentence` list and a *mocked* `llm_json` that returns deterministic selection + window choices.

- [ ] **Step 1: Write the failing test**

```python
# backend/pipeline/assemble/tests/test_topics_integration.py
"""End-to-end (mocked LLM): a 6-topic structure incl. intro/outro + one over-long topic.
Asserts the acceptance properties from the spec: <=CLIP_MAX_S, 0 filler, terminator ends,
chronological, mandatory spec keys."""
import backend.pipeline.assemble.topics as T
from backend import config
from backend.pipeline.understand.models import ContentMap, ContentNode, Structure
from backend.pipeline.sentences import Sentence


def _sent(idx, text, t):
    return Sentence(idx, text, float(t), float(t + 10), ".", True, idx, idx, 1.0, ())


# 60 sentences @10s each = 600s. Topics: intro, A, B(over-long 200s), C, promo-outro
SENTS = [_sent(i, f"Sentence number {i} makes a complete point.", i * 10) for i in range(60)]
NODES = [
    ContentNode(node_id="t0", level="topic", title="Intro",   summary="welcome", start=0,   end=30,  sentence_range=(0, 3)),
    ContentNode(node_id="t1", level="topic", title="Topic A", summary="concept", start=30,  end=120, sentence_range=(3, 12)),
    ContentNode(node_id="t2", level="topic", title="Topic B", summary="big",     start=120, end=320, sentence_range=(12, 32)),  # 200s
    ContentNode(node_id="t3", level="topic", title="Topic C", summary="concept", start=320, end=420, sentence_range=(32, 42)),
    ContentNode(node_id="t4", level="topic", title="Outro",   summary="subscribe", start=420, end=600, sentence_range=(42, 60)),
]

_TYPES = {"t0": "intro", "t1": "teaching", "t2": "teaching", "t3": "teaching", "t4": "outro"}
_SCORE = {"t0": 0.1, "t1": 0.9, "t2": 0.8, "t3": 0.7, "t4": 0.05}


def _fake_llm(system, user, schema, **kw):
    if schema is T.TopicSelection:
        # one judgment per id present in the prompt
        js = [T.TopicJudgment(node_id=nid, type=_TYPES[nid], informativeness=_SCORE[nid],
                              self_contained=0.7) for nid in _TYPES if f"[{nid}]" in user]
        return T.TopicSelection(topics=js)
    # WindowChoice: open at the shown lo, close ~5 sentences later (over-reaches for t2)
    first = user.split("[", 1)[1].split("]", 1)[0]
    lo = int(first)
    return T.WindowChoice(start_idx=lo, end_idx=lo + 18, title="win")  # deliberately long


def test_integration_properties(monkeypatch):
    monkeypatch.setattr(T, "llm_json", _fake_llm)
    st = Structure(video_id="vid", content_map=ContentMap(nodes=NODES, engine="treeseg"))
    specs, notes, rejections = T.assemble_topic_clips(
        st, "the subject", SENTS, "http://x", "vid", {}, adapter=None, stats={})

    assert 3 <= len(specs) <= config.TOPIC_MAX_CLIPS          # t1,t2,t3 kept
    titles = {r.title for r in rejections}
    assert "Intro" in titles and "Outro" in titles            # filler dropped
    for s in specs:
        assert s["end"] - s["start"] <= config.CLIP_MAX_S     # ceiling respected (t2 truncated)
        assert SENTS[s["sentence_end_idx"]].ends_with_period  # terminator close
        assert isinstance(s["start"], float) and isinstance(s["end"], float)
    starts = [s["start"] for s in specs]
    assert starts == sorted(starts)                           # chronological
    assert [s["sequence_index"] for s in specs] == list(range(1, len(specs) + 1))
```

- [ ] **Step 2: Run test to verify it fails, then passes**

Run: `cd clips && .venv/bin/python -m pytest backend/pipeline/assemble/tests/test_topics_integration.py -q`
Expected: PASS if Tasks 2–5 are correct. If `CLIP_MAX_S` is exceeded, the bug is in `_fit_budget` — fix there, not in the test.

- [ ] **Step 3: Commit**

```bash
git add backend/pipeline/assemble/tests/test_topics_integration.py
git commit -m "test(topics): e2e acceptance properties on a synthetic 5-topic structure"
```

---

### Task 8: Eval columns — `window_len`, `topic_selectivity`, `filler_leakage`

**Files:**
- Modify: `backend/eval/metrics.py` (add three metric fns)
- Modify: `backend/eval/run_eval.py` (surface the columns)
- Test: `backend/eval/tests/test_topic_metrics.py` (create)

**Interfaces:**
- Consumes: a shipped `clips` list (dicts with `start,end`) + the run `stats` dict (`n_topics_total`, `n_topics_kept`, `n_topics_dropped`).
- Produces: `window_len_stats(clips) -> dict{min,max,mean}`, `topic_selectivity(stats) -> float` (kept/total), `filler_leakage(clips, sentences, opens_mid_thought) -> float` (fraction of shipped clips whose opener still reads mid‑thought by the LLM/regex check).

- [ ] **Step 1: Write the failing test**

```python
# backend/eval/tests/test_topic_metrics.py
from backend.eval import metrics as M


def test_window_len_stats():
    clips = [{"start": 0.0, "end": 60.0}, {"start": 100.0, "end": 130.0}]
    s = M.window_len_stats(clips)
    assert s["min"] == 30.0 and s["max"] == 60.0 and s["mean"] == 45.0


def test_topic_selectivity():
    assert M.topic_selectivity({"n_topics_total": 20, "n_topics_kept": 8}) == 0.4
    assert M.topic_selectivity({}) == 0.0            # missing ⇒ 0, never divide-by-zero
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd clips && .venv/bin/python -m pytest backend/eval/tests/test_topic_metrics.py -q`
Expected: FAIL (`AttributeError: ... 'window_len_stats'`).

- [ ] **Step 3: Implement the metrics**

Add to `backend/eval/metrics.py`:

```python
def window_len_stats(clips: list[dict]) -> dict:
    durs = [float(c["end"]) - float(c["start"]) for c in clips if c.get("end") is not None]
    if not durs:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {"min": round(min(durs), 1), "max": round(max(durs), 1),
            "mean": round(sum(durs) / len(durs), 1)}


def topic_selectivity(stats: dict) -> float:
    total = int(stats.get("n_topics_total") or 0)
    kept = int(stats.get("n_topics_kept") or 0)
    return round(kept / total, 3) if total else 0.0
```

Then in `backend/eval/run_eval.py`, where per‑video result rows are assembled, add the two columns (follow the existing column‑append pattern in that file):

```python
        "window_len": window_len_stats(clips),
        "topic_selectivity": topic_selectivity(stats),
```

(Import them at the top: `from .metrics import window_len_stats, topic_selectivity`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd clips && .venv/bin/python -m pytest backend/eval/tests/test_topic_metrics.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/eval/metrics.py backend/eval/run_eval.py backend/eval/tests/test_topic_metrics.py
git commit -m "feat(eval): window_len + topic_selectivity columns for the topic engine"
```

---

### Task 9: Full‑suite green + live smoke on a real video

**Files:** none (verification)

- [ ] **Step 1: Full offline suite**

Run: `cd clips && .venv/bin/python -m pytest backend -q`
Expected: all green (prior 735 + new topic/config/metric tests).

- [ ] **Step 2: Restart the server on the new code**

Run: `cd clips && .venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000` (kill the old pid first).

- [ ] **Step 3: Clip a real video and inspect `clips.json`**

Use the CLI or the app on `4yvfd8aoUBc` (a fresh structure exists). Confirm the acceptance properties from the spec §9:
- ~8–12 clips, each duration ≤ `CLIP_MAX_S` (~75 s);
- **0 filler** — the intro (`t0`), the review/promo transition (`t12`), and the outro (`t23`) do NOT appear;
- every opener reads as clean framing (spot‑check the first sentence of each clip);
- every clip ends on a terminator.

Reuse the boundary‑audit script pattern from this session to print each clip's first/last sentence:
```bash
cd clips && .venv/bin/python - <<'PY'
import json
clips = json.load(open("output/4yvfd8aoUBc/clips.json"))
sents = json.load(open("work/4yvfd8aoUBc/punctuation.json"))["result"]["sentences"]
def near_s(t): return min(sents, key=lambda s: abs(s["start"]-t))
def near_e(t): return min(sents, key=lambda s: abs(s["end"]-t))
for c in clips:
    print(f"{c['n']:>2} {c['end']-c['start']:5.0f}s  open: {near_s(c['start'])['text'][:55]!r}")
    print(f"          close: {near_e(c['end'])['text'][-55:]!r}")
PY
```

- [ ] **Step 4: Decision gate**

If the properties hold, the topic engine is accepted as default. If a specific video regresses, set `CLIP_ENGINE=unit` to revert instantly and file the failing structure for a follow‑up. **Do not** delete the unit machinery until at least 2–3 real videos pass.

- [ ] **Step 5: Commit any tuning**

If `INFORMATIVENESS_MIN`, `CLIP_MAX_S`, or `TOPIC_MAX_CLIPS` need adjustment from the live run, change the `config.py` defaults and commit:
```bash
git add backend/config.py
git commit -m "tune(topics): calibrate thresholds from real-video verification"
```

---

## Later (NOT in this plan — separate, gated follow‑ups)

- **Advisory light judge** (`spec §5.3`): run `judge_clip` on each window as ADVISORY only (attach `final_quality` + warnings, never kill) — mirrors the `edge_probe` default‑OFF pattern. Add once the core engine is accepted.
- **LLM onset metric** replacing the regex `opening_onset` in eval (`spec §11`).
- **Delete the unit machinery** (`spec §9 step 3`): remove `candidates.py`, `closure.py`, `arcs.py`, `plan.py`, the refund loop and coverage quotas — only after ≥2–3 real videos pass on the topic engine and the `unit` revert path is confirmed no longer needed.

## Self‑Review

**Spec coverage:**
- §2 goal "best ≤60 s self‑contained window of a substantive teaching topic" → Tasks 3 (select) + 4 (window) + 5 (assemble).
- §2 "only substantive teaching topics, drop filler" → Task 3 + Task 5 rejections.
- §3 length "target ~58 s, max ~75 s, never end mid‑thought" → Task 4 `_fit_budget` + `_snap_end_to_terminator` + Task 1 config.
- §5.1 `select_topics` / `extract_best_window` → Tasks 3 / 4. §7 spec dict shape → Task 5 (matches `_build_embed_clips` `.get()` contract).
- §6 cut precision preserved → the orchestrator's existing `refine_clip_boundaries` runs unchanged after the switch (Task 6); specs carry `start/end/cut_end`.
- §8 error handling → Task 3 never‑zero, Task 4 clamp/forced‑close, Task 5 empty‑topics message.
- §9 rollout `CLIP_ENGINE` flag + verify + defer deletion → Tasks 1, 6, 9, "Later".
- §10 config keys → Task 1. §11 tests → Tasks 3,4,5,7,8. §12 tunables → Task 9 Step 5.

**Placeholder scan:** every code step is complete runnable code; no TBD/TODO. Verification commands have exact expected outcomes.

**Type consistency:** `select_topics → (list[TopicPick], list[TopicPick])` consumed by Task 5; `extract_best_window → Optional[Window]` consumed by Task 5; `Window` field names (`start_s/end_s/start_idx/end_idx/facet/why/warnings`) match their use in Task 5's spec‑dict build; `Sentence.idx/ends_with_period` used consistently; `Rejection(cand_id,title,role,stage,reason,start=,end=)` matches `integrity.py`. `assemble_topic_clips` signature is byte‑for‑byte the `assemble_clips` signature the orchestrator calls (Task 6).
