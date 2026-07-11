# Topic‑first clipping — design

**Date:** 2026-07-04
**Status:** Approved (brainstorm), pending implementation plan
**Supersedes (in spirit):** the unit‑anchored assembly path (`assemble/candidates.py`,
`closure.py`, `arcs.py`, `plan.py`, coverage quotas, refund loop). Keeps the
precise‑cutting (`2026-07-04-precise-clip-cutting-design.md`) and discourse work as reused
substrate, not as the primary boundary mechanism.

> **Revision (2026‑07‑04, post‑approval):** a clip is **not** the whole topic. It is the
> **single best ~60 s self‑contained window** *within* a selected substantive topic
> (hard‑ish ~60 s ceiling, "max the quality" inside that budget). Within‑topic trimming is
> wanted; multi‑part splitting is out. This matches the real product (the ~45 s
> Instagram‑style feed). §2/§3/§4/§5/§10/§11 reflect this; the topic‑selection and
> reuse‑the‑cutter spine is unchanged.

---

## 1. Problem

Three user‑reported defects, verified on **fresh latest‑code output** (`output/4yvfd8aoUBc`,
built 16:19 on 2026‑07‑04 from a `15:52` structure — i.e. not stale cache):

1. **No context / random starts.** Clips open on a reference to earlier material —
   *"These neurons are what create a reflex arc."*, *"These give an individual pain relief."*,
   *"In action potential."* — dropping a cold viewer mid‑thought. A clip must **open on a
   self‑contained framing sentence and close on a concluded thought.**
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
through. Nothing checks the thing the user actually wants: **is this a self‑contained,
informative window of a real teaching topic?** Clips are anchored to individual **units**
(400 of them for this video) and stitched with prerequisite closure — inherently sub‑topic,
reference‑prone, and with no length discipline.

### The unlock

`structure.content_map` is already a hierarchical topic tree
(`video → 5 chapters → 24 topic nodes`). Each topic node has `title`, `summary`,
`keywords`, `sentence_range`, and sentence‑accurate `start`/`end`. This gives us **clean,
quality‑selectable topic units** to (a) drop filler and (b) source a self‑contained ~60 s
window from — instead of rebuilding from 400 units. Measured against the topic nodes:

- Topic‑node openers are clean framing 19/24 by the pipeline's own rule
  (*"All right now it's time to talk about the different structures…"*,
  *"Let's first talk about sound…"*,
  *"Speaking of pain we need to talk about the gate control theory…"*).
- The 5 "flags" are mostly the regex **over‑flagging good transitions**
  (*"So that's the brain, and now comes the time to talk about sleep"* is a great onset),
  and 2 of the 5 are filler (`t12`, `t23`) that selection drops anyway.
- ⇒ A topic's framing sentence is usually the natural window **start**; the work is choosing
  the best **≤60 s** span that also **closes** cleanly, semantically (not by regex).

## 2. Goals / non‑goals

**Goals**
- One clip = the **single best self‑contained ~60 s window** of a **substantive teaching
  topic** — opening on framing, closing on a concluded thought, packed with the topic's core.
- Ship **only substantive teaching topics** (~6–10 per video); drop intros, outros,
  channel plugs, transitions, admin, throwaway asides.
- Preserve precise cutting: start right after a terminator, end on one, snapped to silence.
- Reuse the existing `content_map`, precise cutter, judge infra, context cards, frontend.

**Non‑goals**
- No whole‑topic clips that blow past the ceiling (a 9.6‑min topic yields ONE ~60 s window,
  not a 9.6‑min clip and not N split parts).
- No fresh LLM re‑chaptering (TreeSeg boundaries are good here; revisit only if they prove bad).
- Not chasing the eval judge's comprehension score (known to over‑flag; the human live‑test
  is the acceptance signal).

## 3. Decisions (from brainstorm)

| Axis | Decision |
|---|---|
| What is a clip | **Best self‑contained ~60 s window** within a selected substantive topic |
| Selection bar | **Only substantive teaching topics**; drop intro/outro/plug/transition/admin/aside |
| Length | **Target ~55–60 s, hard‑ish max ~75 s** (allow finishing the closing sentence); pick the window that maximizes quality within budget; never end mid‑thought to hit a number |
| Approach | **A — topic‑node‑first**; select topics, extract best window, reuse the cutter |
| `MAX_CLIPS` | **safety ceiling ~40** — ship ALL substantive teaching topics (smoke-tuned 2026-07-04; 10 dropped real content) |
| One window per topic | **Yes** — no multi‑part splitting; a topic contributes at most one clip |

## 4. Architecture

`understand/` and `content_map` are unchanged. Only the **front of `assemble/`** changes.

```
OLD  units(400) → select_anchors → build_candidate(closure) → arcs/coverage/refund
                                                    → snap → judge/repair → sequence
NEW  content_map.topic_nodes(24) → select_topics → extract_best_window (≤~60s)
                                  → snap → judge(light) → cards(rare) → sequence
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

**`extract_best_window(pick, sentences, adapter, settings) -> Window`**
- For each kept topic, choose the contiguous sentence span that:
  1. **opens** on a self‑contained framing/onset sentence (semantic, catches
     *"These neurons…"* — if the topic's first sentence dangles, move the start to the real
     framing sentence, even if that trims a slow lead‑in),
  2. **closes** on a concluded thought (a sentence that resolves an idea, not a hanging clause),
  3. fits **`CLIP_MAX_S`** (~75 s hard‑ish), targeting **`CLIP_TARGET_S`** (~55–60 s),
  4. **maximizes informativeness** — the span carrying the topic's core payoff (the
     definition + its explanation / the mechanism + its example), dropping tangents & recaps.
- Implemented as **one LLM call per kept topic** over that topic's sentences (each tagged
  with index + cumulative duration), returning **`{start_idx, end_idx, title, why}`** as
  grounded sentence indices — no free‑text timestamps.
- **Clamp** any out‑of‑range / over‑budget index back inside the topic's `sentence_range`
  and truncate to the last sentence that fits `CLIP_MAX_S` while still ending on a terminator.
- Often the window **starts at the topic's framing sentence** (openers are already clean);
  the LLM's main job is the **close** and dropping internal filler to fit ~60 s.
- Delivers **#1** and the ~60 s ceiling.
- `Window = {topic_node_id, start_idx, end_idx, start_s, end_s, title, type, why, warnings}`.

### 5.2 Reused unchanged
- **`boundary_adapt.snap_candidates`** + **`refine.refine_clip_boundaries`** — precise
  silence‑snap cut. Fed the chosen sentence span. Delivers **#3**, already correct.
- **`context_card.generate_context_card`** — only when a kept opener still references a
  prior not resolvable in‑clip (now rare, since windows open with framing).
- **`sequence.sequence_clips`** — chronological order.
- Download / transcribe / punctuation / `content_map` / Gemini judge infra / frontend.

### 5.3 `judge_clip` — role change
Stays as the **final gate** (self‑contained? teaches one idea? opens on framing? closes
cleanly? ≤ `CLIP_MAX_S`?) but no longer drives unit‑closure repair. Preserve the invariant
*no clip ships opening mid‑thought* — now judged **semantically**, not by the regex.

## 6. Cut precision (#3) — preserved, not rebuilt
Verified on `output/4yvfd8aoUBc`: every clip ends on a terminator (`19/19`), and cuts land
in inter‑word silence (clip 2 starts at `135.248` = clip 1's exact end; clip 3 at `170.632`
vs prev end `170.485`). The new front‑end hands **sentence‑boundary** spans to the same
`snap_candidates → refine_clip_boundaries` pass. No change to the cutter; #3 stays satisfied.

## 7. Data flow / output shape
`content_map.nodes[level=topic]` → `select_topics` (keep ~6–10) → `extract_best_window`
(≤~60 s, → sentence span) → `snap_candidates` → `refine_clip_boundaries` → `judge_clip`
(light) → `generate_context_card` (rare) → `sequence_clips` → `clips.json`.

`clips.json` gains a real **`title`** per clip (from the window) and a `topic_node_id`; the
`facet` catch‑all `"other"` is replaced by the topic `type`/title. Durations are now
**~45–75 s** instead of 26–123 s.

## 8. Error handling / degradation
- **Weak/empty topic tree** (`degraded`, or `< 2` topic nodes): fall back to **chapter‑level**
  nodes; if still empty, emit an honest "couldn't segment this video" rejection.
- **Selection clears nothing** → top‑N by informativeness + warning (never zero clips).
- **Window LLM out‑of‑range / over‑budget / malformed** → clamp to topic `sentence_range`,
  truncate to the last terminator‑ending sentence within `CLIP_MAX_S`.
- **No sentence closes cleanly within budget** → take the longest terminator‑ending prefix
  that fits; flag `window_close_forced`.
- Hard invariant (semantic): **no shipped clip opens mid‑thought.**

## 9. Rollout (safety: 735 tests + live product)
1. Build the topic path behind **`CLIP_ENGINE=topic|unit`** (default **`topic`**).
2. Verify on real videos incl. `4yvfd8aoUBc`: expect ~8–12 clips of **~45–75 s**, **0 filler**
   (`t0/t12/t23` dropped), all clean onsets, all terminator ends.
3. **Then** delete the unit/closure/arc/coverage/refund machinery
   (`candidates.py`, `closure.py`, `arcs.py`, `plan.py`, refund loop, coverage quotas) in a
   follow‑up commit — reaching the "delete it" end‑state without nuking a working product
   mid‑change.

## 10. Config (new / changed)
| Key | Default | Meaning |
|---|---|---|
| `CLIP_ENGINE` | `topic` | `topic` (new) or `unit` (legacy, temporary) |
| `TOPIC_MAX_CLIPS` | `40` | safety ceiling only — ship all substantive topics (TreeSeg caps at 24) |
| `TOPIC_MODEL` | `gemini-3.1-pro-preview` | model for the selection+window calls (bulk authoring stays flash) |
| `INFORMATIVENESS_MIN` | `0.5` | selection threshold (tune on real videos) |
| `CLIP_TARGET_S` | `58` | target window length |
| `CLIP_MAX_S` | `75` | hard‑ish ceiling; a window may reach this to finish a sentence |
| `BOUNDARY_WINDOW` | `3` | sentences of context each side when judging the close |

## 11. Testing (TDD)
- `select_topics`: mocked LLM over the real 24‑node map → keeps teaching topics, drops
  `t0` (intro) / `t12` (promo) / `t23` (outro); respects `MAX_CLIPS`; never returns zero.
- `extract_best_window`:
  - a topic whose first sentence dangles (*"These neurons…"* case) → start moves to the
    real framing sentence;
  - a 9.6‑min topic (`t7`) → returns ONE window `≤ CLIP_MAX_S` that opens+closes cleanly;
  - a window whose natural close exceeds budget → truncated to the last terminator‑ending
    sentence within `CLIP_MAX_S`; `window_close_forced` flagged;
  - out‑of‑range LLM index → clamped to the topic `sentence_range`.
- Integration on `4yvfd8aoUBc`: ~8–12 clips, each **≤ `CLIP_MAX_S`**, 0 filler, all clean
  onsets, all terminator ends.
- Eval: add `window_len`, `topic_selectivity`, and `filler_leakage` columns to `run_eval`;
  replace the regex `opening_onset` metric with an LLM‑judged onset check.
- Preserve full offline suite green; `CLIP_ENGINE=unit` keeps legacy tests passing until the
  §9 deletion.

## 12. Open tunables (user may veto)
- `MAX_CLIPS=10`, `CLIP_TARGET_S=58`, `CLIP_MAX_S=75`, `INFORMATIVENESS_MIN=0.5` — set from
  the brainstorm answers; expect to tune on 2–3 real videos during verification.
