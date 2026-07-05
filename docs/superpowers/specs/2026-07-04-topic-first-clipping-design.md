# Topic‑first clipping — design

**Date:** 2026-07-04
**Status:** Approved (brainstorm), pending implementation plan
**Supersedes (in spirit):** the unit‑anchored assembly path (`assemble/candidates.py`,
`closure.py`, `arcs.py`, `plan.py`, coverage quotas, refund loop). Keeps the
precise‑cutting (`2026-07-04-precise-clip-cutting-design.md`) and discourse work as reused
substrate, not as the primary boundary mechanism.

---

## 1. Problem

Three user‑reported defects, verified on **fresh latest‑code output** (`output/4yvfd8aoUBc`,
built 16:19 on 2026‑07‑04 from a `15:52` structure — i.e. not stale cache):

1. **No context / random starts.** Clips open on a reference to earlier material —
   *"These neurons are what create a reflex arc."*, *"These give an individual pain relief."*,
   *"In action potential."* — dropping a cold viewer mid‑thought. A clip should span **one
   whole topic, from its opening framing sentence to its concluding sentence.**
2. **Quantity over quality.** 19 clips for one ~68‑min video, **12 of 19 facet `"other"`**
   (a vague catch‑all); the very first clip is an intro (*"My name is Mr sin, and today
   we're going to review…"*). Want **informative teaching content only**, filler dropped.
3. **Cut precision.** Each clip should begin right after a `.`/`!`/`?` and end on a
   terminator. (Already satisfied — see §6 — restated as a hard requirement to preserve.)

### Root cause (the load‑bearing finding)

The shipped onset/cutting features **pass their own metrics while missing the goal**. The
pipeline's own rule reports `0/19` mid‑thought openers and `0/19` non‑terminal ends on the
failing output — because `discourse.py` is a regex/wordlist that only flags a pronoun
*immediately* followed by an aux/verb (`"It is…"`). Noun‑phrase references
(*"These neurons…"*, *"These drugs…"*) and fragments (*"In action potential."*) sail
through. Nothing checks the thing the user actually wants: **does this clip cover a whole,
self‑contained topic?** Clips are anchored to individual **units** (400 of them for this
video) and stitched with prerequisite closure — inherently sub‑topic and reference‑prone.

### The unlock

`structure.content_map` is already a hierarchical topic tree
(`video → 5 chapters → 24 topic nodes`). Each topic node has `title`, `summary`,
`keywords`, `sentence_range`, and sentence‑accurate `start`/`end`. Granularity ≈ **170 s
average**, right in the desired 120–180 s band. The whole‑topic clips the user wants
**already exist in the data** — `assemble/` just discards the tree and rebuilds from units.
Measured against the topic nodes:

- Topic‑node openers are clean framing 19/24 by the pipeline's own rule
  (*"All right now it's time to talk about the different structures…"*,
  *"Let's first talk about sound…"*,
  *"Speaking of pain we need to talk about the gate control theory…"*).
- The 5 "flags" are mostly the regex **over‑flagging good transitions**
  (*"So that's the brain, and now comes the time to talk about sleep"* is a great onset),
  and 2 of the 5 are filler (`t12`, `t23`) that selection drops anyway.
- ⇒ Boundary work is a **light semantic nudge on ~3 topics**, not a rebuild.

## 2. Goals / non‑goals

**Goals**
- One clip = one **complete topic**, cut from its true discourse start to its true end.
- Ship **only substantive teaching topics** (~6–10 per video); drop intros, outros,
  channel plugs, transitions, admin, throwaway asides.
- Preserve precise cutting: start right after a terminator, end on one, snapped to silence.
- Reuse the existing `content_map`, precise cutter, judge infra, context cards, frontend.

**Non‑goals**
- No internal trimming of a kept topic (user chose *whole topic intact*, not "trimmed core").
- No fresh LLM re‑chaptering (TreeSeg boundaries are good here; revisit only if they prove bad).
- Not chasing the eval judge's comprehension score (known to over‑flag; the human live‑test
  is the acceptance signal).

## 3. Decisions (from brainstorm)

| Axis | Decision |
|---|---|
| What is a clip | **Whole topic, start→end** (~90–180 s, ~6–10 per video) |
| Selection bar | **Only substantive teaching topics**; drop intro/outro/plug/transition/admin/aside |
| Length | **Soft target ~120–180 s**; a strong topic may run to **~4 min**; never cut mid‑arc |
| Approach | **A — topic‑node‑first**; replace unit‑anchored assembly, reuse the topic tree + cutter |
| `MAX_CLIPS` | target **~10** (config) |
| Over‑long ceiling | **~240 s** (~4 min); beyond it, split at a natural sub‑boundary or run long |

## 4. Architecture

`understand/` and `content_map` are unchanged. Only the **front of `assemble/`** changes.

```
OLD  units(400) → select_anchors → build_candidate(closure) → arcs/coverage/refund
                                                    → snap → judge/repair → sequence
NEW  content_map.topic_nodes(24) → select_topics → split_overlong
                                  → refine_topic_boundaries → snap → judge(light) → cards → sequence
```

The back half (`snap_candidates`, `refine_clip_boundaries`, `judge_clip`,
`generate_context_card`, `sequence_clips`) is reused; the front half is replaced.

## 5. Components

### 5.1 `assemble/topics.py` (new)

**`select_topics(content_map, sentences, adapter, settings) -> list[TopicPick]`**
- Input per topic node: `title`, `summary`, `keywords`, first & last sentence text.
- **One batched LLM call** over all topic nodes (so intro‑vs‑teaching is judged relative to
  the whole video). Structured output per node:
  `{node_id, type: teaching|intro|outro|transition|admin|promo|tangent,
    informativeness: 0..1, self_contained: 0..1, why: str}`.
- Keep `type == "teaching" AND informativeness >= INFORMATIVENESS_MIN`.
- Rank kept by `informativeness`; cap at `MAX_CLIPS`.
- **Never zero on a real teaching video:** if nothing clears, ship the top‑N by
  informativeness with a `low_confidence_selection` warning.
- Delivers **#2**.

**`refine_topic_boundaries(pick, sentences, adapter) -> (start_idx, end_idx)`**
- Per kept topic, LLM sees a window of `±BOUNDARY_WINDOW` (=3) sentences around the node's
  start and end. Picks the exact **first** sentence (best standalone framing of *this*
  topic) and **last** sentence (cleanest conclusion), returned as **sentence indices**
  (grounded — no free‑text timestamps).
- Catches *"These neurons…"* (extend start back to the framing sentence, or if the referent
  is not in‑window, flag for a context card) and trims recap/meta openers
  (*"Okay now I know this video is long…"* → skip forward to real content).
- **Clamp** any out‑of‑window index back into the node's own `sentence_range` (fallback).
- Delivers **#1**.

**`split_overlong(pick, units, sentences, adapter) -> list[TopicPick]`**
- If `duration > OVERLONG_CEILING_S`, find internal sub‑boundaries from the **units** whose
  `node_id` is this topic (natural concept‑cluster gaps) or one LLM split‑point call. Each
  part re‑checked for self‑containedness (its own `refine_topic_boundaries`).
- If no clean sub‑boundary exists, **do not split** (honor "never cut mid‑arc") — let it run.
- Example: `t7` "Brain Structures and Functions" (578 s, forebrain/midbrain/hindbrain) →
  splits at sub‑arc boundaries; each part self‑contained.

### 5.2 Reused unchanged
- **`boundary_adapt.snap_candidates`** + **`refine.refine_clip_boundaries`** — precise
  silence‑snap cut. Fed the chosen sentence span. Delivers **#3**, already correct.
- **`context_card.generate_context_card`** — only when a kept opener still references a
  prior not resolvable in‑clip (now rare, since topics open with framing).
- **`sequence.sequence_clips`** — chronological order.
- Download / transcribe / punctuation / `content_map` / Gemini judge infra / frontend.

### 5.3 `judge_clip` — role change
Stays as the **final gate** (self‑contained? teaches one idea? ends cleanly?) but no longer
drives unit‑closure repair. Preserve the invariant *no clip ships opening mid‑thought* —
now judged **semantically**, not by the regex.

## 6. Cut precision (#3) — preserved, not rebuilt
Verified on `output/4yvfd8aoUBc`: every clip ends on a terminator (`19/19`), and cuts land
in inter‑word silence (clip 2 starts at `135.248` = clip 1's exact end; clip 3 at `170.632`
vs prev end `170.485`). The new front‑end hands **sentence‑boundary** spans to the same
`snap_candidates → refine_clip_boundaries` pass. No change to the cutter; #3 stays satisfied.

## 7. Data flow / output shape
`content_map.nodes[level=topic]` → `select_topics` (keep ~6–10) → `split_overlong` →
`refine_topic_boundaries` (→ sentence span) → `snap_candidates` → `refine_clip_boundaries`
→ `judge_clip` (light) → `generate_context_card` (rare) → `sequence_clips` → `clips.json`.

`clips.json` gains a real **`title`** per clip (from the topic node) and a `topic_node_id`;
the `facet` catch‑all `"other"` is replaced by the topic `type`/title.

## 8. Error handling / degradation
- **Weak/empty topic tree** (`degraded`, or `< 2` topic nodes): fall back to **chapter‑level**
  nodes; if still empty, emit an honest "couldn't segment this video" rejection.
- **Selection clears nothing** → top‑N by informativeness + warning (never zero clips).
- **Boundary LLM out‑of‑window / malformed** → clamp to node `sentence_range`.
- **Split finds no clean sub‑boundary** → run long rather than cut mid‑arc.
- Hard invariant (semantic): **no shipped clip opens mid‑thought.**

## 9. Rollout (safety: 735 tests + live product)
1. Build the topic path behind **`CLIP_ENGINE=topic|unit`** (default **`topic`**).
2. Verify on real videos incl. `4yvfd8aoUBc`: expect ~8–12 clips, **0 filler**
   (`t0/t12/t23` dropped), all clean onsets, all terminator ends.
3. **Then** delete the unit/closure/arc/coverage/refund machinery
   (`candidates.py`, `closure.py`, `arcs.py`, `plan.py`, refund loop, coverage quotas) in a
   follow‑up commit — reaching the "delete it" end‑state without nuking a working product
   mid‑change.

## 10. Config (new / changed)
| Key | Default | Meaning |
|---|---|---|
| `CLIP_ENGINE` | `topic` | `topic` (new) or `unit` (legacy, temporary) |
| `MAX_CLIPS` | `10` | cap on kept topics |
| `INFORMATIVENESS_MIN` | `0.5` | selection threshold (tune on real videos) |
| `OVERLONG_CEILING_S` | `240` | split‑or‑run‑long threshold |
| `BOUNDARY_WINDOW` | `3` | sentences of context each side for boundary refine |

## 11. Testing (TDD)
- `select_topics`: mocked LLM over the real 24‑node map → keeps teaching topics, drops
  `t0` (intro) / `t12` (promo) / `t23` (outro); respects `MAX_CLIPS`; never returns zero.
- `refine_topic_boundaries`: *"These neurons"* window → picks the earlier framing sentence;
  *"Okay now I know this video is long"* window → skips forward; out‑of‑window → clamps.
- `split_overlong`: `t7` (578 s) → ≥2 self‑contained parts each `< OVERLONG_CEILING_S`;
  a 300 s topic with no clean sub‑boundary → left whole.
- Integration on `4yvfd8aoUBc`: ~8–12 clips, 0 filler, all clean onsets, all terminator ends.
- Eval: add `topic_completeness` + `filler_leakage` columns to `run_eval`; replace the regex
  `opening_onset` metric with an LLM‑judged onset check.
- Preserve full offline suite green; `CLIP_ENGINE=unit` keeps legacy tests passing until the
  §9 deletion.

## 12. Open tunables (user may veto)
- `MAX_CLIPS=10`, `OVERLONG_CEILING_S=240`, `INFORMATIVENESS_MIN=0.5` — all set from the
  brainstorm answers; expect to tune on 2–3 real videos during verification.
