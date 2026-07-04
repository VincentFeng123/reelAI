# Discourse-Onset Clip Boundaries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Guarantee that every shipped clip begins at the discourse-onset of the thought it opens — never mid-thought ("and then…", "so the answer is…"), never at the answer — universally across genres.

**Architecture:** Add one DRY primitive (`pipeline/discourse.py`) that decides, from a sentence's text alone, whether it *opens mid-thought*. Use it in two places: (a) a new backward-extension START guard in `refine._snap_one` — the symmetric twin of the existing weak-END guard, fixing both pipeline paths at once; (b) a new `opening_onset_rate` eval metric for before/after proof. Then protect the onset from the repair-trim lattice and the closure card-demotion, add a judge gate that a card can't launder, and retune length/anchor-budget config for short, fewer clips.

**Tech Stack:** Python 3, pydantic, pytest. Offline-testable (no audio/whisper/LLM) except the judge-gate wording and the config/eval A/B.

## Global Constraints

- Guards are **preferences with a safe fallback**, never hard rejects: a clip is never dropped or left unplaceable by the onset guard (mirror the weak-END guard's contract). Only-weak openers ship flagged `weak_start_boundary`.
- The onset detector is **text-only and genre-independent** — it keys on discourse structure (continuation markers, dangling anaphora, question/framing cues), never on role labels or domain.
- Preserve existing behavior on already-good clips: `#6 "So let's start with KI. How can we name this compound?"`, `#7 "Now what about MgBr2?"`, `#11 "What is the charge on chlorine?"` MUST remain valid onsets (not flagged, not moved).
- `max_clip_duration_s` becomes a **soft** ceiling; **overflow is allowed** (never split/trim-middle/drop to hit it). Setup is **never** demoted to a context card.
- Test module paths are rooted at `backend.` (e.g. `from backend.pipeline.refine import _snap_one`). Run tests from the `clips/` directory: `cd clips && python -m pytest backend/...`.
- **VCS note:** `clips/` is not a git repo. Either run `git init` in `clips/` first, or treat every "Commit" step as a checkpoint (stage the same files, skip the commit). Steps below show the git command for when a repo exists.

---

### Task 1: Discourse-onset primitive (`pipeline/discourse.py`)

The DRY core. A pure, text-only function `opens_mid_thought(text) -> bool` shared by the snap guard (Task 3) and the metric (Task 2). Must resolve the "so" ambiguity (a leading continuation marker is fine when the sentence is self-contained framing or a question).

**Files:**
- Create: `backend/pipeline/discourse.py`
- Test: `backend/pipeline/tests/test_discourse_onset.py`

**Interfaces:**
- Produces: `opens_mid_thought(text: str) -> bool` — True when the sentence, used as a clip's first line, drops the viewer mid-thought (leading continuation marker without self-contained framing, dangling anaphor, context-dependent definite NP like "the answer", or mid-clause fragment). `is_onset(text: str) -> bool` = `not opens_mid_thought(text)`.

- [ ] **Step 1: Write the failing tests**

```python
# backend/pipeline/tests/test_discourse_onset.py
"""Discourse-onset primitive — text-only, offline. Decides whether a sentence used as a
clip's FIRST line drops the viewer mid-thought. A leading continuation marker is NOT weak
when the sentence is self-contained framing or a question (the 'so' disambiguation)."""
from __future__ import annotations

from backend.pipeline.discourse import opens_mid_thought, is_onset

# Real bad openers pulled from the shipped corpus — MUST be flagged weak.
def test_answer_first_is_weak():
    assert opens_mid_thought("So the answer is magnesium bromide.")

def test_and_then_continuation_is_weak():
    assert opens_mid_thought("And then mg which stands for magnesium,")

def test_however_continuation_is_weak():
    assert opens_mid_thought("However we do have a subscript next to O and it's a two.")

def test_dangling_anaphor_is_weak():
    assert opens_mid_thought("This is why the reaction proceeds so quickly.")
    assert opens_mid_thought("That gives us the final concentration.")

def test_context_dependent_np_is_weak():
    assert opens_mid_thought("The answer is fifteen newtons.")
    assert opens_mid_thought("The previous equation tells us the velocity.")

def test_mid_clause_fragment_is_weak():
    assert opens_mid_thought("writing oxygen we're going to write a two")  # lowercase mid-clause

# Real GOOD openers — MUST be treated as onsets (the critical false-positives).
def test_so_lets_start_is_onset():
    assert is_onset("So let's start with KI. How can we name this compound?")

def test_now_what_about_is_onset():
    assert is_onset("Now what about MgBr2?")

def test_bare_question_is_onset():
    assert is_onset("What is the charge on chlorine?")

def test_declarative_topic_sentence_is_onset():
    assert is_onset("Newton's second law relates force and acceleration.")

def test_hortative_framing_is_onset():
    assert is_onset("Let's consider a block sliding down a ramp.")
    assert is_onset("Suppose we have two moles of hydrogen.")

def test_empty_is_weak():
    assert opens_mid_thought("")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd clips && python -m pytest backend/pipeline/tests/test_discourse_onset.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.pipeline.discourse'`

- [ ] **Step 3: Write the implementation**

```python
# backend/pipeline/discourse.py
"""Discourse-onset detection (text-only, genre-independent).

Decides whether a sentence, used as a clip's FIRST line, drops a cold viewer mid-thought.
Grounded in: Decontextualization (Choi et al., TACL 2021) — the dominant context-dependence
is a dangling referring expression (~40%); and cue-phrase disambiguation (Hirschberg & Litman
1993) — a leading discourse marker ("so", "now") is a genuine onset when the sentence is
self-contained framing or a question, and a mid-thought continuation otherwise.

Shared by the snap START guard (refine.py) and the opening_onset_rate metric (eval/metrics.py).
"""
from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)

# Leading tokens that, without self-contained framing, signal continuation of a prior thought.
CONTINUATION_MARKERS: frozenset[str] = frozenset({
    "and", "so", "but", "because", "cuz", "cause", "therefore", "thus", "hence",
    "anyway", "anyways", "also", "plus", "or", "nor", "yet", "then", "well",
    "okay", "ok", "alright", "actually", "basically", "meanwhile", "however",
    "moreover", "furthermore", "additionally", "consequently", "again",
})

# Bare anaphora that, as the first word before a verb/aux, lack an in-clip antecedent.
ANAPHORS: frozenset[str] = frozenset({
    "this", "that", "these", "those", "it", "they", "them", "he", "she",
    "him", "her", "here", "there", "its", "their",
})

# Definite-NP heads that are context-dependent ("the answer", "the previous equation").
CONTEXT_DEP_HEADS: frozenset[str] = frozenset({
    "answer", "result", "value", "number", "equation", "formula", "expression",
    "problem", "reason", "difference", "ratio", "sum", "product", "solution",
    "previous", "next", "first", "second", "third", "latter", "former", "same",
    "above", "below", "point", "step", "one", "thing",
})

# Verb/aux tokens; a leading anaphor immediately followed by one is a dangling reference.
_AUX_VERB: frozenset[str] = frozenset({
    "is", "are", "was", "were", "be", "been", "'s", "gives", "shows", "means",
    "tells", "makes", "gets", "goes", "comes", "has", "have", "had", "will",
    "would", "can", "could", "should", "does", "do", "did", "equals", "becomes",
})

# Framing / segment-onset cues: their presence makes even a marker-led sentence an onset.
_FRAMING_PATTERNS = (
    "let's", "lets", "let us", "we're going to", "we are going to", "we will",
    "we'll", "i want to", "i'm going to", "consider", "suppose", "imagine",
    "here's", "here is", "take ", "let me", "start with", "starting with",
    "look at", "move on to", "moving on", "turn to", "next up", "first,",
    "to begin", "begin with", "picture ", "think about", "what about",
    "how about", "say we", "say you",
)

_INTERROGATIVE = frozenset({
    "what", "how", "why", "where", "when", "which", "who", "whose", "whom",
    "is", "are", "can", "could", "would", "should", "does", "do", "did", "will",
})


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text or "")


def _is_framing_or_question(text: str, words: list[str]) -> bool:
    """A leading marker is fine when the sentence stands on its own as new framing/a question."""
    low = (text or "").lower()
    if "?" in low:
        return True
    if any(pat in low for pat in _FRAMING_PATTERNS):
        return True
    # A question that lost its '?' in ASR: an interrogative appears within the first 3 words.
    return any(w in _INTERROGATIVE for w in words[:3])


def opens_mid_thought(text: str) -> bool:
    """True when this sentence, as a clip's first line, drops the viewer mid-thought."""
    words = _words(text)
    if not words:
        return True

    framing = _is_framing_or_question(text, words)
    if framing:
        return False                       # self-contained framing/question wins outright

    w0 = words[0].lower()
    w1 = words[1].lower() if len(words) > 1 else ""

    # 1) leading continuation marker without framing → mid-thought
    if w0 in CONTINUATION_MARKERS:
        return True
    # 2) dangling anaphor (pronoun immediately before a verb/aux) → unresolved reference
    if w0 in ANAPHORS and (w1 in _AUX_VERB or w1 == "s"):
        return True
    # 3) context-dependent definite NP: "the answer/previous/…"
    if w0 == "the" and w1 in CONTEXT_DEP_HEADS:
        return True
    # 4) mid-clause fragment: begins lowercase (post-punctuation-restoration sentences are
    #    capitalized at true onsets) or too few words to be a complete opener
    stripped = (text or "").lstrip()
    if stripped and stripped[0].islower():
        return True
    if len(words) < 3:
        return True
    return False


def is_onset(text: str) -> bool:
    return not opens_mid_thought(text)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd clips && python -m pytest backend/pipeline/tests/test_discourse_onset.py -v`
Expected: PASS (all 12+ tests)

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline/discourse.py backend/pipeline/tests/test_discourse_onset.py
git commit -m "feat: discourse-onset primitive (opens_mid_thought)"
```

---

### Task 2: `opening_onset_rate` metric + run_eval wiring

Build the measurement first so we can baseline before changing behavior.

**Files:**
- Modify: `backend/eval/metrics.py` (add function near `unresolved_reference_rate`, ~line 69)
- Modify: `backend/eval/run_eval.py` (add to the label-free report dict — see `comprehension`/`ends_on_period_rate` wiring)
- Test: `backend/eval/tests/test_opening_onset_metric.py`

**Interfaces:**
- Consumes: `opens_mid_thought` from Task 1; a `spec` dict carrying `sentence_start_idx`; the `sentences` list.
- Produces: `opening_onset_rate(specs, sentences) -> float` — fraction of clips whose first sentence is a discourse-onset.

- [ ] **Step 1: Write the failing test**

```python
# backend/eval/tests/test_opening_onset_metric.py
from __future__ import annotations

from backend.eval.metrics import opening_onset_rate
from backend.pipeline.sentences import Sentence


def _sent(idx, text):
    return Sentence(idx=idx, text=text, start=float(idx), end=float(idx) + 1.0,
                    terminator=".", ends_with_period=True, word_start_idx=idx,
                    word_end_idx=idx, align_confidence=1.0)


def test_opening_onset_rate_counts_only_good_openers():
    sentences = [
        _sent(0, "Now what about MgBr2?"),          # onset
        _sent(1, "So the answer is magnesium bromide."),  # weak
        _sent(2, "Newton's second law relates force and acceleration."),  # onset
    ]
    specs = [
        {"sentence_start_idx": 0},
        {"sentence_start_idx": 1},
        {"sentence_start_idx": 2},
    ]
    assert opening_onset_rate(specs, sentences) == 2 / 3


def test_opening_onset_rate_empty_is_zero():
    assert opening_onset_rate([], []) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd clips && python -m pytest backend/eval/tests/test_opening_onset_metric.py -v`
Expected: FAIL — `ImportError: cannot import name 'opening_onset_rate'`

- [ ] **Step 3: Add the metric to `backend/eval/metrics.py`**

Add this function immediately after `unresolved_reference_rate` (after ~line 82):

```python
def opening_onset_rate(specs, sentences) -> float:
    """Fraction of clips whose FIRST sentence is a discourse-onset (not mid-thought /
    not at the answer). Operationalizes 'a cold viewer isn't dropped in the middle'
    (PodReels' audience-confusion signal). The headline START-quality number."""
    from ..pipeline.discourse import opens_mid_thought
    if not specs:
        return 0.0
    good = 0
    for s in specs:
        i0 = s.get("sentence_start_idx", 0)
        i0 = max(0, min(i0, len(sentences) - 1)) if sentences else 0
        text = sentences[i0].text if sentences else ""
        good += 0 if opens_mid_thought(text) else 1
    return good / len(specs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd clips && python -m pytest backend/eval/tests/test_opening_onset_metric.py -v`
Expected: PASS

- [ ] **Step 5: Wire into `run_eval.py`**

In `backend/eval/run_eval.py`, find the label-free metrics block that sets `m[...]` (near `ends_on_period_rate` / `unresolved_reference_rate`) and add one line alongside them:

```python
    m["opening_onset_rate"] = _round_nan(metrics.opening_onset_rate(specs, sentences))
```

(Match the exact `_round_nan(...)`/`m[...]` style used by the neighboring metric lines. If the surrounding lines don't use `_round_nan`, mirror whatever wrapper they use.)

- [ ] **Step 6: Baseline measurement (record it)**

Run: `cd clips && python -m backend.eval.run_eval --freeze --runs 1`
Record the current `opening_onset_rate` per video and the mean into
`docs/superpowers/plans/2026-07-03-discourse-onset-clip-boundaries.md` under a new
"## Baseline (before)" section. This is the number Task 8 proves we moved.

- [ ] **Step 7: Commit**

```bash
git add backend/eval/metrics.py backend/eval/run_eval.py backend/eval/tests/test_opening_onset_metric.py
git commit -m "feat: opening_onset_rate metric + baseline"
```

---

### Task 3: Backward-extension START guard in `_snap_one`

The core fix. Symmetric to the existing weak-END guard: when the start sentence opens mid-thought, extend the start **backward** to the nearest onset, bounded to the anchor's topic node (never cross into an unrelated topic). Fixes both pipeline paths (full via `boundary_adapt.py:85`, fast via `refine.py:294`).

**Files:**
- Modify: `backend/pipeline/refine.py` (add `_is_weak_start` + `_STRONG_START_LOOKBACK`; wire into `_snap_one` after `si`/`ei` are clamped, before the min-duration loop at ~line 141)
- Test: `backend/pipeline/tests/test_onset_start_guard.py`

**Interfaces:**
- Consumes: `opens_mid_thought` (Task 1); `_snap_one`'s existing `cand["node_span"]` (optional `[node_start_s, node_end_s]`, already threaded by Part B; absent on the legacy path).
- Produces: `_is_weak_start(s: Sentence) -> bool`; `_snap_one` now emits `warnings` containing `weak_start_boundary` when only a weak start is reachable, and moves `sentence_start_idx` back to a strong onset when one exists in-node.

- [ ] **Step 1: Write the failing tests**

```python
# backend/pipeline/tests/test_onset_start_guard.py
"""Onset START guard — the symmetric twin of the weak-END guard. Offline (no audio/LLM).
A start that opens mid-thought extends BACKWARD to the nearest in-node onset; only-weak
starts still ship, flagged weak_start_boundary. Never drops a clip. Good onsets untouched."""
from __future__ import annotations

from backend.pipeline.refine import _is_weak_start, _snap_one
from backend.pipeline.sentences import Sentence


def _sent(idx, start, end, text, terminator="."):
    return Sentence(idx=idx, text=text, start=start, end=end, terminator=terminator,
                    ends_with_period=bool(terminator), word_start_idx=idx, word_end_idx=idx,
                    align_confidence=1.0)


_SNAP = dict(allow_qe=False, min_dur=1.0, tail_pad=0.05, max_dur=500.0)


def test_is_weak_start_matches_primitive():
    assert _is_weak_start(_sent(0, 0, 4, "So the answer is magnesium bromide."))
    assert not _is_weak_start(_sent(0, 0, 4, "Now what about MgBr2?"))


def test_weak_start_extends_back_to_onset():
    sents = [
        _sent(0, 0.0, 4.0, "Now what about MgBr2?"),                 # onset
        _sent(1, 4.1, 8.0, "Well mg is magnesium and br is bromine."),
        _sent(2, 8.1, 12.0, "So the answer is magnesium bromide."),  # weak — candidate start
    ]
    clip = _snap_one({"i_start": 2, "i_end": 2, "facet": "other"}, sents, **_SNAP)
    assert clip["sentence_start_idx"] == 0                           # moved back to the question
    assert "weak_start_boundary" not in clip["warnings"]


def test_backward_extension_bounded_by_node_span():
    sents = [
        _sent(0, 0.0, 4.0, "Earlier we discussed the periodic table."),  # DIFFERENT topic
        _sent(1, 4.1, 8.0, "So the answer is magnesium bromide."),       # weak start, node begins here
    ]
    # node_span starts at 4.1 → the guard must NOT cross back into sentence 0's topic.
    clip = _snap_one({"i_start": 1, "i_end": 1, "facet": "other", "node_span": [4.1, 8.0]},
                     sents, **_SNAP)
    assert clip["sentence_start_idx"] == 1
    assert "weak_start_boundary" in clip["warnings"]                 # only weak reachable → flagged


def test_good_onset_start_unchanged():
    sents = [
        _sent(0, 0.0, 4.0, "Newton's second law relates force and acceleration."),
        _sent(1, 4.1, 8.0, "The mass is the constant of proportionality."),
    ]
    clip = _snap_one({"i_start": 0, "i_end": 1, "facet": "other"}, sents, **_SNAP)
    assert clip["sentence_start_idx"] == 0
    assert "weak_start_boundary" not in clip["warnings"]


def test_only_weak_start_still_places_flagged():
    sents = [_sent(0, 0.0, 4.0, "So the answer is magnesium bromide.")]
    clip = _snap_one({"i_start": 0, "i_end": 0, "facet": "other"}, sents, **_SNAP)
    assert clip is not None                                          # never unplaceable
    assert clip["sentence_start_idx"] == 0
    assert "weak_start_boundary" in clip["warnings"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd clips && python -m pytest backend/pipeline/tests/test_onset_start_guard.py -v`
Expected: FAIL — `ImportError: cannot import name '_is_weak_start'`

- [ ] **Step 3: Add `_is_weak_start` and the constant to `backend/pipeline/refine.py`**

After `_is_weak_end` (line 38) add:

```python
_STRONG_START_LOOKBACK = 8          # sentences scanned BACK for a strong onset before flagging


def _is_weak_start(s: Sentence) -> bool:
    """A start sentence is WEAK when it opens the clip mid-thought (leading continuation
    marker without framing, dangling anaphor, context-dependent definite NP, or fragment).
    Symmetric to _is_weak_end; preferred against, never a hard reject."""
    from .discourse import opens_mid_thought
    return opens_mid_thought(s.text or "")
```

- [ ] **Step 4: Wire the guard into `_snap_one`**

In `backend/pipeline/refine.py::_snap_one`, immediately after the `si`/`ei` clamp block (after line 100, `if ei < si: ei = si`) and before `warnings: list[str] = []` is used for the end guard, insert the START guard. Replace the line `warnings: list[str] = []` (line 102) with:

```python
    warnings: list[str] = []
    # START guard (symmetric to the END guard below): if the start sentence opens mid-thought,
    # extend the start BACKWARD to the nearest onset, bounded to the anchor's topic node
    # (cand["node_span"]) so we never cross into an unrelated topic. Only-weak starts still
    # ship, flagged weak_start_boundary — a clip is NEVER dropped by this guard.
    if _is_weak_start(sentences[si]):
        node_lo = None
        ns = cand.get("node_span")
        if ns:
            node_lo = float(ns[0]) - 1e-6
        strong = None
        lo = max(0, si - _STRONG_START_LOOKBACK)
        for k in range(si - 1, lo - 1, -1):
            if node_lo is not None and sentences[k].start < node_lo:
                break                                   # do not cross the topic-node boundary
            if not _is_weak_start(sentences[k]):
                strong = k
                break
        if strong is not None:
            si = strong
        else:
            warnings.append("weak_start_boundary")      # only a weak start reachable — flagged
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd clips && python -m pytest backend/pipeline/tests/test_onset_start_guard.py backend/pipeline/tests/test_bnd1_boundary_guards.py -v`
Expected: PASS (new onset tests + existing BND1 end-guard tests still green — no regression)

- [ ] **Step 6: Commit**

```bash
git add backend/pipeline/refine.py backend/pipeline/tests/test_onset_start_guard.py
git commit -m "feat: backward-extension discourse-onset START guard in _snap_one"
```

---

### Task 4: Protect the onset from the repair-trim lattice + role hygiene

The repair loop trims units farthest from the anchor (`_trim_lattice`), which can advance the start past a leading setup/onset unit. Protect the temporally-earliest in-span unit. Also fix the misleading `NON_ANCHOR` "trimmable" comment by introducing a correct `EDGE_TRIMMABLE` set.

**Files:**
- Modify: `backend/roles.py` (correct the misleading `NON_ANCHOR` "safe to trim" comment)
- Modify: `backend/pipeline/assemble/validate.py::_protected_unit_ids` (line 715-729)
- Test: `backend/pipeline/assemble/tests/test_onset_protection.py`

**Interfaces:**
- Consumes: `Candidate.unit_ids`, `Candidate.anchor_id`, `units_by_id`, adapter (existing `_protected_unit_ids` signature).
- Produces: `_protected_unit_ids` now also protects the earliest in-span unit (the onset).

- [ ] **Step 1: Write the failing test**

```python
# backend/pipeline/assemble/tests/test_onset_protection.py
"""The repair-trim lattice must never trim the clip's leading (onset) unit — trimming it
would advance the start past the setup/problem-read. Offline."""
from __future__ import annotations

from backend.pipeline.assemble.types import Candidate
from backend.pipeline.assemble.validate import _protected_unit_ids
from backend.pipeline.understand.models import Unit
from backend.adapters import get_adapter


def _unit(uid, start, end, role):
    return Unit(unit_id=uid, start=start, end=end, sentence_range=(int(start), int(end)),
                role=role, summary="", transcript="")


def test_leading_onset_unit_is_protected_from_trim():
    units = [
        _unit("u0", 0, 4, "example_setup"),   # the problem-read (onset)
        _unit("u1", 4, 8, "worked_step"),
        _unit("u2", 8, 12, "result"),         # anchor (payoff)
    ]
    by_id = {u.unit_id: u for u in units}
    # Candidate required fields (types.py): cand_id, anchor_id, role, facet, title, reason,
    # unit_ids, referential, i_start, i_end, start, end. contract_role is optional.
    cand = Candidate(cand_id="c0", anchor_id="u2", role="result", facet="worked_example",
                     title="", reason="", unit_ids=["u0", "u1", "u2"], referential=[],
                     i_start=0, i_end=12, start=0.0, end=12.0, contract_role="result")
    adapter = get_adapter("generic")
    protected = _protected_unit_ids(cand, by_id, adapter)
    assert "u0" in protected      # the leading onset unit must never be trimmed away
    assert "u2" in protected      # the anchor stays protected (existing behavior)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd clips && python -m pytest backend/pipeline/assemble/tests/test_onset_protection.py -v`
Expected: FAIL — `assert "u0" in protected` (leading unit not yet protected)

- [ ] **Step 3: Correct the misleading `NON_ANCHOR` comment in `backend/roles.py`**

Replace the `NON_ANCHOR` comment (lines 40-41) with an accurate one — `NON_ANCHOR` is an
anchor-policy set, NOT a trim signal, and `setup` must be kept when it is the clip's onset:

```python
# Structural "connective tissue" — never a clip ANCHOR (this set is consumed only by the
# adapter's anchor policy). NOTE: it is NOT a leading-edge trim signal — a `setup` unit that
# forms the clip's onset (the problem-read / equation-introduction) must be KEPT, not trimmed.
# Onset protection lives in validate._protected_unit_ids.
```

(Leave the `frozenset({...})` membership unchanged.)

- [ ] **Step 4: Protect the leading unit in `_protected_unit_ids`**

In `backend/pipeline/assemble/validate.py::_protected_unit_ids` (line 715), before `return protected`, add the leading-onset protection:

```python
    # Protect the clip's LEADING (temporally earliest) in-span unit: trimming it would advance
    # the start past the setup/problem-read and re-open the clip mid-thought (the discourse-onset
    # invariant). The end guard already protects completeness; this protects the opening.
    in_span = [units_by_id[uid] for uid in cand.unit_ids if uid in units_by_id]
    if in_span:
        leading = min(in_span, key=lambda u: (u.start, u.sentence_range[0]))
        protected.add(leading.unit_id)
    return protected
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd clips && python -m pytest backend/pipeline/assemble/tests/test_onset_protection.py backend/pipeline/assemble/tests/test_repair_rework.py -v`
Expected: PASS (new protection test + existing repair tests still green)

- [ ] **Step 6: Commit**

```bash
git add backend/roles.py backend/pipeline/assemble/validate.py backend/pipeline/assemble/tests/test_onset_protection.py
git commit -m "feat: protect clip onset unit from repair-trim lattice"
```

---

### Task 5: Judge gate `opening_in_context` + close the card escape hatch

Add a position-aware verdict field the judge must satisfy, wire it into the completeness gate, and forbid a context card from satisfying it (a card can resolve a distant one-line prerequisite, never launder a mid-thought open).

**Files:**
- Modify: `backend/pipeline/assemble/validate.py` (`JudgeVerdict` ~line 43-61; `JUDGE_SYSTEM` ~line 74-119)
- Modify: `backend/adapters/base.py` (`CORE_VERDICT_FIELDS` line 145-146)
- Test: `backend/pipeline/assemble/tests/test_opening_gate.py`

**Interfaces:**
- Produces: `JudgeVerdict.opening_in_context: bool`; it is added to `CORE_VERDICT_FIELDS` so `is_complete` gates on it for every role.

- [ ] **Step 1: Write the failing test**

```python
# backend/pipeline/assemble/tests/test_opening_gate.py
"""opening_in_context is a required verdict field: a clip that opens mid-thought fails
is_complete even if everything else passes; a card cannot satisfy it."""
from __future__ import annotations

from backend.pipeline.assemble.validate import JudgeVerdict, is_complete
from backend.adapters import get_adapter


def test_opening_in_context_is_required_for_completeness():
    adapter = get_adapter("generic")
    v = JudgeVerdict(score_10=9, understandable=True, opening_in_context=False)
    assert not is_complete(v, "explanation", adapter, min_score=0.7)
    v_ok = JudgeVerdict(score_10=9, understandable=True, opening_in_context=True)
    assert is_complete(v_ok, "explanation", adapter, min_score=0.7)


def test_opening_in_context_in_core_fields():
    from backend.adapters.base import CORE_VERDICT_FIELDS
    assert "opening_in_context" in CORE_VERDICT_FIELDS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd clips && python -m pytest backend/pipeline/assemble/tests/test_opening_gate.py -v`
Expected: FAIL — `JudgeVerdict` has no `opening_in_context`; not in `CORE_VERDICT_FIELDS`.

- [ ] **Step 3: Add the verdict field**

In `backend/pipeline/assemble/validate.py::JudgeVerdict`, after `source_grounded: bool = True` (line 56) add:

```python
    opening_in_context: bool = True   # does the FIRST sentence stand on its own (not mid-thought
                                      # / not at the answer before the question is posed)?
```

- [ ] **Step 4: Add the judge instruction**

In `JUDGE_SYSTEM` (validate.py:74), add an evaluation bullet in the boolean list (after the
`source_grounded` bullet, line 105) and a sentence to step 1:

```python
    "- opening_in_context: does the FIRST sentence open the thought — introducing its own "
    "subject/problem/equation/question — rather than continuing one ('and then…', 'so the "
    "answer is…') or referring back ('this', 'the answer') to material shown before the clip? "
    "A CONTEXT CARD does NOT satisfy this — the spoken opening itself must stand on its own. "
    "If FALSE, add a failure_reason kind 'not_source_grounded' quoting the mid-thought opener.\n"
```

- [ ] **Step 5: Add to `CORE_VERDICT_FIELDS`**

In `backend/adapters/base.py` line 145-146, extend the tuple:

```python
CORE_VERDICT_FIELDS = ("topic_identifiable", "purpose_identifiable",
                       "all_references_resolved", "prerequisites_satisfied", "source_grounded",
                       "opening_in_context")
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd clips && python -m pytest backend/pipeline/assemble/tests/test_opening_gate.py backend/pipeline/assemble/tests/test_judge_rubric.py -v`
Expected: PASS (new gate tests + existing judge-rubric tests still green)

- [ ] **Step 7: Commit**

```bash
git add backend/pipeline/assemble/validate.py backend/adapters/base.py backend/pipeline/assemble/tests/test_opening_gate.py
git commit -m "feat: opening_in_context judge gate; card cannot satisfy it"
```

---

### Task 6: Closure keeps the onset inline (never a card) + arc overflow

Stop the two paths that push the reconstructed opener to a `referential` card: (a) the closure demote-to-referential for the onset unit; (b) the oversized-arc fallback. Per the locked decision, let the span **overflow** rather than card the opener.

**Files:**
- Modify: `backend/pipeline/assemble/closure.py::compute_closure` (the referential-demote branch, lines 132-134 and 154-162)
- Modify: `backend/pipeline/assemble/candidates.py` (`build_arc_candidate` oversized branch, lines 505-521)
- Test: `backend/pipeline/assemble/tests/test_onset_inline.py`

**Interfaces:**
- Consumes: `ClosureResult` (closure.py), `ArcCandidate.opener_ids` (understand/arcs.py).
- Produces: for a payoff/arc anchor, `opener_ids[0]` (and the contract-required "before" onset element) appear in `unit_ids`, never in `referential`.

- [ ] **Step 1: Write the failing test**

```python
# backend/pipeline/assemble/tests/test_onset_inline.py
"""The reconstructed opener (problem-read / equation-introduction) is inlined into the clip
span, never demoted to a referential card — even when that overflows the soft span budget.
Uses the real Graph([], units) + BaseAdapter() fixture from test_closure_runs.py."""
from __future__ import annotations

from backend.adapters.base import BaseAdapter
from backend.pipeline.assemble.closure import compute_closure, ClosureBudget
from backend.pipeline.assemble.graph import Graph
from backend.pipeline.understand.models import Unit


def _u(uid, role, start, end, i, node="c0.t1"):
    return Unit(unit_id=uid, start=start, end=end, sentence_range=(i, i), role=role,
                node_id=node, transcript=f"sentence {i}.", summary=uid)


def test_required_before_onset_is_inlined_even_past_span_budget():
    # result anchor whose GENERIC contract requires an example_setup BEFORE it. With a TIGHT
    # span budget the old code demotes the setup to a referential card; the onset must instead
    # be FORCE-inlined (overflow allowed).
    units = [
        _u("u0", "example_setup", 0.0, 4.0, 0),
        _u("u1", "worked_step", 4.0, 8.0, 1),
        _u("u2", "result", 8.0, 12.0, 2),   # anchor (payoff)
    ]
    by_id = {u.unit_id: u for u in units}
    res = compute_closure(by_id["u2"], Graph([], units), by_id, BaseAdapter(), units,
                          ClosureBudget(max_span_s=10.0))   # inlining u0 → 12s span > 10s budget
    assert "u0" in res.unit_ids                              # onset inlined, not carded
    assert "u0" not in [uid for uid, _ in res.referential]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd clips && python -m pytest backend/pipeline/assemble/tests/test_onset_inline.py -v`
Expected: FAIL — with `max_span_s=10.0`, inlining `u0` makes a 12s span, so the current code
demotes `u0` to `referential` (`assert "u0" in res.unit_ids` fails).

- [ ] **Step 3: Force-inline the onset in `compute_closure`**

In `backend/pipeline/assemble/closure.py`, the contract-required loop (lines 124-134) currently
demotes a required element to `referential` when it would exceed `budget.max_span_s`. Change it so
a required **"before"** onset element is always inlined (overflow allowed):

```python
    # 1. contract-required elements → inline. A required *before* onset (problem_statement /
    #    example_setup / practice_prompt / setup) is FORCE-inlined even past the span budget:
    #    the clip must open with it (discourse-onset invariant, overflow allowed). Other
    #    required elements keep the budget check.
    required_ids, recommended = _contract_needs(anchor, adapter, units, order, budget)
    onset_ids = _required_before_ids(anchor, adapter, units, order)   # new helper, see below
    for uid in sorted(set(required_ids), key=lambda x: order.get(x, 0)):
        u = units_by_id.get(uid)
        if not u or uid in inline:
            continue
        new_span = [min(span[0], u.start), max(span[1], u.end)]
        if uid in onset_ids or (new_span[1] - new_span[0]) <= budget.max_span_s:
            inline.add(uid)
            span = new_span
        else:
            referential.append((uid, "prerequisite"))
            truncated = True
```

Add the helper near `_contract_needs`:

```python
def _required_before_ids(anchor: Unit, adapter, units: list[Unit], order: dict[str, int]) -> set[str]:
    """Unit ids satisfying a REQUIRED, position='before' contract element of the anchor — the
    onset (problem-read / equation-introduction) that the clip must open with."""
    contract = adapter.contract_for(anchor.role)
    if not contract:
        return set()
    ai = order.get(anchor.unit_id, 0)
    ids: set[str] = set()
    for el in contract.elements:
        if el.necessity != "required" or el.position != "before":
            continue
        want = set(el.roles)
        cand = [u for u in units if order[u.unit_id] < ai and u.role in want]
        if cand:
            ids.add(cand[-1].unit_id)          # nearest preceding onset unit
    return ids
```

- [ ] **Step 4: Keep the opener inline for oversized arcs in `candidates.py`**

In `backend/pipeline/assemble/candidates.py::build_arc_candidate` (lines 505-521), the oversized-hull
branch pushes `arc.opener_ids` to `referential` + `truncated=True`. Change it to keep the opener
inline and let the span overflow. Replace the `if hull_s > max_span:` branch body with a version
that extends the closure span to include `opener_ids[0]` instead of carding it:

```python
    opener = units_by_id.get(arc.opener_ids[0]) if arc.opener_ids else None
    if hull_s > max_span:
        synth = (terminal if terminal.role == arc.terminal_role
                 else terminal.model_copy(update={"role": arc.terminal_role}))
        cand = build_candidate(synth, graph, adapter, units, units_by_id, sentences,
                               relevance, settings)
        if cand is None:
            return None
        # Overflow (locked decision): keep the opener IN the span, never card it. Extend the
        # candidate's unit set + i_start/start back to the opener rather than pushing it to a card.
        if opener is not None and opener.unit_id not in cand.unit_ids:
            new_ids = [opener.unit_id] + list(cand.unit_ids)
            i_start = min(cand.i_start, opener.sentence_range[0])
            cand = replace(cand, unit_ids=new_ids, i_start=i_start,
                           start=min(cand.start, float(sentences[opener.sentence_range[0]].start)))
        return replace(cand, cand_id=f"c_{arc.arc_id}", arc_id=arc.arc_id,
                       warnings=tuple(set(cand.warnings or ()) | set(clamp_warn)))
```

(Confirm `replace` is imported from `dataclasses` at the top of `candidates.py`; it is used
elsewhere in the file.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd clips && python -m pytest backend/pipeline/assemble/tests/test_onset_inline.py backend/pipeline/assemble/tests/test_closure_runs.py backend/pipeline/assemble/tests/test_practice_preservation.py -v`
Expected: PASS (new inline test + existing closure/practice tests still green)

- [ ] **Step 6: Commit**

```bash
git add backend/pipeline/assemble/closure.py backend/pipeline/assemble/candidates.py backend/pipeline/assemble/tests/test_onset_inline.py
git commit -m "feat: inline the discourse onset (never a card); arc overflow keeps opener"
```

---

### Task 7: Config — short target, soft ceiling, fewer clips

Retune to "short like Instagram, quality over quantity" without hard-severing context.

**Files:**
- Modify: `backend/config.py` (`DEFAULTS` lines 128-133; `MAX_SEGMENTS` line 118; `ANCHOR_MIN_PRIORITY` line 254)
- Test: `backend/pipeline/tests/test_config_targets.py`

**Interfaces:**
- Produces: `config.DEFAULTS["target_clip_duration_s"]`, a lower `max_clip_duration_s`, a lower default clip count.

- [ ] **Step 1: Write the failing test**

```python
# backend/pipeline/tests/test_config_targets.py
from backend import config


def test_short_instagram_targets():
    assert config.DEFAULTS["target_clip_duration_s"] == 45.0
    # soft ceiling replaces the old 240s (overflow still allowed by the assembler, but the
    # DEFAULT ceiling is Instagram-short)
    assert config.DEFAULTS["max_clip_duration_s"] <= 90.0
    assert config.DEFAULTS["min_clip_duration_s"] <= 20.0


def test_fewer_clips_by_default():
    assert config.MAX_SEGMENTS <= 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd clips && python -m pytest backend/pipeline/tests/test_config_targets.py -v`
Expected: FAIL — `target_clip_duration_s` missing; `max_clip_duration_s == 240.0`; `MAX_SEGMENTS == 12`.

- [ ] **Step 3: Edit `backend/config.py`**

Line 118: `MAX_SEGMENTS = 12` → `MAX_SEGMENTS = 8`

Lines 128-133 `DEFAULTS` — set the short targets (keep the `max_clips: None` inherit behavior):

```python
    "min_clip_duration_s": 15.0,     # a complete short thought can be brief
    "target_clip_duration_s": 45.0,  # Instagram-short aim (scoring target, not a cutter)
    "max_clip_duration_s": 90.0,     # SOFT ceiling; the assembler allows overflow for a
                                     # complete thought (never split/trim-middle/drop to hit it)
```

Line 254: `ANCHOR_MIN_PRIORITY = 40` → `ANCHOR_MIN_PRIORITY = 45` (drop the weakest anchors →
fewer, stronger clips).

> Note: `max_clip_duration_s` is now a soft ceiling. Task 6 already lets the opener overflow it;
> `refine._snap_one`'s max-duration cap (lines 162-176) still applies to the END only, which is
> correct — it trims trailing content, never the protected onset.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd clips && python -m pytest backend/pipeline/tests/test_config_targets.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/config.py backend/pipeline/tests/test_config_targets.py
git commit -m "feat: short-clip targets (45s target / 90s soft ceiling) + fewer clips"
```

---

### Task 8: End-to-end verification (before/after A/B)

Prove the invariant holds on the real corpus with the cached-structure replay harness.

**Files:**
- Modify: `docs/superpowers/plans/2026-07-03-discourse-onset-clip-boundaries.md` (append results)

- [ ] **Step 1: Run the full offline test suite**

Run: `cd clips && python -m pytest backend/pipeline backend/pipeline/assemble/tests backend/eval/tests -q`
Expected: PASS (no regressions across boundary, closure, judge, repair, metric suites).

- [ ] **Step 2: Run the frozen A/B on the corpus**

Run: `cd clips && python -m backend.eval.run_eval --freeze --runs 3`
Compare `opening_onset_rate` (mean ± std) against the Task 2 baseline. Also read
`clips/video`, `comprehension`, `context_complete_rate`, `ends_on_period_rate`.

- [ ] **Step 3: Acceptance checks**

- `opening_onset_rate` ≥ **0.95** (from the recorded baseline).
- No regression: `comprehension`, `context_complete_rate`, `ends_on_period_rate` within noise
  of baseline (use `--runs 3` std as the noise band).
- `clips/video` drops toward a handful (was 14.2 avg).
- Spot-check `-KfG8kH-r3Y`: clip #8 now opens at the question ("Now what about MgBr2?" style),
  not "So the answer is…"; #1/#2/#5/#12 open cleanly; #6/#7/#11 unchanged.

- [ ] **Step 4: Record results + commit**

Append a "## Results (after)" section to this plan with the before/after table, then:

```bash
git add docs/superpowers/plans/2026-07-03-discourse-onset-clip-boundaries.md
git commit -m "docs: discourse-onset before/after results"
```

---

## Self-review notes (author)

- **Spec coverage:** Fix #1 → Tasks 1+3; #2 → Task 6; #3 → Task 4; #4 → Task 5; #5 → Task 7;
  #6 → Task 2; #7 (punctuation-restoration, Mode A durability) is deferred per spec §7 and is
  intentionally **not** a task here (Task 1's mid-clause check is the interim mitigation).
- **Correction vs spec:** `NON_ANCHOR` is used only for anchor policy, not trimming; the operative
  onset-trim protection is in `validate._protected_unit_ids` (Task 4). The spec's `EDGE_TRIMMABLE`
  split is dropped as YAGNI (no filler-edge-trim task consumes it); only the misleading `NON_ANCHOR`
  comment is corrected.
- **Type consistency:** `opens_mid_thought(text)`/`is_onset(text)` (Task 1) are the exact names used
  in Tasks 2 and 3; `opening_in_context` field name is identical in Task 5's field, prompt, and
  `CORE_VERDICT_FIELDS`.
- **Fixture caveats flagged inline:** `Candidate` (Task 4) and the graph builder (Task 6) tell the
  implementer to read the real constructor/helper rather than invent one.
