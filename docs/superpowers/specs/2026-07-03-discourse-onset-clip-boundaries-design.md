# Discourse-onset clip boundaries — every clip starts where its thought begins — design

**Date:** 2026-07-03 · **Status:** approved-pending-user-review · **Source:** user report
("the way videos are being clipped is dogshit… I want full context… never start on a random
sentence… for a problem, start with the person reading the question… extremely universal…
quality over quantity, short like Instagram"). Diagnosis from a 9-agent code read + a
corpus scan of `work/*/runs/*/shipped.json` + a 21-source deep-research pass.

---

## 1. The problem, quantified

Across the current shipped corpus (**142 clips, 10 videos**):

- **60% of clips open with a continuation/anaphor token** — `so` ×31, `now` ×22, `and` ×11,
  `well`, `that`, `however`, `okay`, `it`, `this`…
- **14.2 clips/video** (up to 22) — over-production, the opposite of quality-over-quantity.
- Real openers (video `-KfG8kH-r3Y`): #8 *"So the answer is magnesium bromide."* (**starts at
  the answer**), #1 *"And then mg which stands for magnesium,"*, #5 *"However we do have a
  subscript next to O…"* (212 s), #12 *"So if you were to type in… organic chemistry tutor in
  YouTube…"* (administrative aside). The **good** ones (#6 *"So let's start with KI. How can we
  name this compound?"*, #7 *"Now what about MgBr2?"*, #11 *"What is the charge on chlorine?"*)
  open at the question — proving the machinery works when setup sits right before the anchor.

### Root cause (mechanism)

The pipeline enforces *completeness of content* (are problem/steps/result present somewhere in
the span?) and *no dangling "this/that"* — but has **no invariant that a clip's first sentence
is a discourse-onset**. Concretely:

- **`refine.py::_snap_one`** finalizes the start as `start_t = sentences[si].start` with `si`
  taken **verbatim** from the candidate. It has a rich *end* guard (`_is_weak_end`,
  `_END_STOPWORDS`, `_STRONG_END_LOOKAHEAD`) and **zero symmetric start guard**. This one
  function feeds *both* pipeline paths (full path via `assemble/boundary_adapt.py:85`, fast
  path via `refine.py:294`).
- **`assemble/closure.py`** *can* pull setup in before a payoff anchor, but it is budget-capped
  (`CLOSURE_MAX_GAP_S=25`, `CLOSURE_MAX_SPAN_S=300`, `CLOSURE_MAX_EXTRA_UNITS=6`); when setup is
  too far it is **demoted to a `referential` context card** and the physical start stays on the
  answer (`closure.py:132-134,154-162`). That is the `truncated=True` on most long clips.
- **`candidates.py:507-521`** (oversized arc hull) pushes the reconstructed `opener_ids` to a
  card + `truncated=True`, keeping the terminal.
- **`roles.py:42-47`** — `SETUP` is in `NON_ANCHOR`, documented "safe to trim from a clip's
  **leading**/trailing edge." The pipeline is literally allowed to trim off the problem statement.
- **`roles.py::coerce_role`** defaults unknown labels to `EXPLANATION`, whose contract requires
  no "before" element → a mislabeled problem-read imposes no start constraint.
- **`assemble/validate.py`** judge checks presence, not *position*; card-as-repair
  (`validate.py:~976`) flips a failing verdict **without moving the start** — the card launders
  a broken open.

### The missing invariant

> **A clip's start is the discourse-onset of the thought it opens** — its first spoken sentence
> introduces its own subject/problem/equation/question and stands on its own, rather than
> *continuing* ("and then…", "so the answer is…") or *referring back* ("this", "the answer") to
> material cut away before it.

This is a **discourse property, not a genre property** → universal across lecture / problem /
interview / tutorial / podcast.

---

## 2. Research foundation (deep-research, adversarially verified)

Findings that shaped the design (21 sources; 3-vote verification, 2/3-refute kills):

- **Prior art validates the invariant.** *PodReels* (Wang et al., Adobe/Columbia, DIS 2024):
  "clear context" is one of four principles of a good short clip; audiences reported *feeling
  confused* (8/10) by teasers full of *"out-of-context words"* like "here"/"them" and *content
  "from the middle of the episode"* — our exact Modes B/C. Expert editors' manual repair is
  literally our fix: *"bring in neighboring segments… find surrounding statements, adjust the
  clip at the sentence level, and remove filler words."* Their validated target is short
  single-moment (~30 s, range 11–51 s); multi-moment was rejected as "too long, diluted."
  Automated tools' known failure is over-production (Clips.AI ≈80 results; Opus 15 "identical,
  generic" clips) — matching our 14–22/video.
- **Pure lexical onset detection is unreliable (REFUTED claims).** "A sentence opening with a
  connective signals continuation" and "onset detectable by head n-gram phrases" were both
  refuted 0-3. → a stopword list alone is wrong; e.g. good clip #6 opens with "So." Cue phrases
  have **structural (onset) vs sentential (continuation)** uses (Hirschberg & Litman 1993) that
  must be disambiguated by what *follows* the marker.
- **Taxonomy of "not stand-alone"** (*Decontextualization*, Choi et al., TACL 2021, Table 1):
  Pronoun/definite-NP swap **40.5%** (dangling referring expression — the dominant case, incl.
  "the answer"), Bridging/global-scoping ~20%, Name completion 11.5%, discourse-marker removal
  3.5% (far higher in speech). Only **~30% of sentences stand alone with no edits**; **~12% are
  INFEASIBLE** (narrative-embedded / "rely heavily on the preceding few sentences"); technical/
  math content is **hardest** ("requires background from preceding paragraphs") — i.e. our
  chemistry/physics lectures are worst-case. Most edits are recoverable by **coreference
  resolution**.
- **Onset detection needs the following context**, not just the sentence (cross-segment models).
  Confirms the guard must consider the anchor/arc a sentence leads into.
- **Repair = extend to include real surrounding media, not rewrite** (we keep authentic footage;
  PodReels + our embed model). Decontextualization tells us *what to detect*; PodReels tells us
  the media fix is *backward extension*.

---

## 3. Decisions (locked with user)

1. **Context always wins.** Every clip starts at its thought's discourse-onset. Shortness comes
   from trimming filler + choosing self-contained thoughts, never from amputating setup.
2. **Length target 30–60 s, soft ceiling ~90 s** (replaces `max_clip_duration_s = 240`).
   **Overflow allowed**: when a *complete* thought genuinely needs more, ship it longer — never
   split, trim-the-middle, or drop to hit the ceiling. Setup is **never** demoted to a card.
3. **Surgical + new invariant.** Keep the structure-first architecture; add the discourse-onset
   invariant at snapping + closure + judge, protect setup from trimming, add a metric, ship
   fewer/better clips.
4. Filler edge-trimming (`transition`/`administrative`/`irrelevant`) stays desirable (assumed
   yes). Backward reach for a strong onset is generous but bounded to the enclosing topic-node
   so we never cross into an unrelated topic (assumed yes).

---

## 4. Fixes

Ordered by leverage. #1 and #4 catch the bulk cheaply and universally; #2 reaches distant
setup; #3 stops setup being trimmed; #5–#6 deliver "short + fewer + provable"; #7 is durability.

### 1. Universal discourse-onset START guard — `backend/pipeline/refine.py`

The symmetric twin of the existing `_is_weak_end`, added to `_snap_one` so it fixes **both**
paths at once. A **3-tier detector** (research says lexical-only is insufficient):

- **Tier A — lexical prefilter (cheap, flags candidates).** First sentence opens with a
  continuation/discourse-marker token (`and, so, but, because, cuz, therefore, thus, hence,
  anyway, also, plus, which, then, now, well, okay, alright, "so the answer"…`) **or** a bare
  anaphor / context-dependent definite NP (`this, that, these, those, it, they, he, she, here,
  there, the answer, the result, the previous …`). Not-flagged ⇒ accept as-is.
- **Tier B — disambiguation (structural vs sentential; resolves the "so" ambiguity).** A flagged
  start is a **genuine onset** (accept) if the sentence is itself self-contained framing:
  it poses a question (`?`, "how/what/why…"), or is hortative/segment-initial framing ("let's
  start with", "let's look at", "consider", "suppose", "here's"), or introduces a *new* named
  entity/topic not carried from before. It is a **weak continuation** (extend) if it carries an
  **unresolved referring expression** — an anaphor / definite NP whose antecedent is not inside
  the clip. Use the raw material that already exists: `Unit.references` (labeled "this/that
  equation/as we saw" + `resolves_to`) and `refers_to`/`requires`/`answers` graph edges. If the
  opening unit references a unit *outside* the clip → weak.
- **Tier C — repair.** On weak start, extend the start **backward** to the nearest strong onset
  in the **same topic-node** (and, when the anchor is an arc, no later than `arc.opener_ids[0]`),
  preferring a sentence that (a) resolves the dangling reference and (b) is not itself weak.
  Mirror the end guard's bounded scan (`_STRONG_START_LOOKBACK`). If none is reachable in-node,
  ship flagged `weak_start_boundary` (parallel to `weak_end_boundary`) rather than fail.

Also treat a **mid-clause fragment** (Tier A: sentence begins lowercase mid-clause / < 3 words,
mirroring `_is_weak_end` / no clause-initial token) as weak → Mode A. And a first unit whose `node_id` differs from the
anchor's → weak → Mode D.

Kills Mode B (the 60%), catches A (clause-onset), D (same-node). Keys on discourse structure,
not role labels ⇒ universal, and robust on math/chem where labels are least reliable.

### 2. Opener is mandatory-inline, never a card — `assemble/closure.py` + `candidates.py`

Invert the budget priority for the onset:

- `closure.py`: the contract-required "before" onset element (problem_statement / example_setup /
  practice_prompt / setup) and any graph antecedent that the onset **directly refers to** is
  **force-inlined**; the demote-to-`referential` path (`:132-134,154-162`) is disabled *for the
  onset unit*. Distant *non-onset* prerequisites may still become cards.
- `candidates.py:507-521` (oversized arc): keep `opener_ids` **inline** with the terminal and
  let the span **overflow** (per decision 2) instead of carding the opener + `truncated=True`.
- Relax `CLOSURE_MAX_GAP_S` / `CLOSURE_MAX_SPAN_S` enough that the onset is always reachable
  (keep a sanity ceiling far above 90 s). The soft ceiling lives in scoring/target, not as a
  hard sever that cuts context.

### 3. Protect the onset from leading-edge trimming — `roles.py` + `assemble/validate.py`

Split `NON_ANCHOR`'s two conflated jobs into `NON_ANCHOR` (can't seed a clip) and a new
`EDGE_TRIMMABLE = {transition, administrative, irrelevant}` (filler safe to trim from an edge).
`setup`/`example_setup`/`practice_prompt` stay non-anchor but are **not** edge-trimmable. Every
leading-edge trim site (incl. `validate.py::_trim_lattice`, `refine.py::_trim_start_after`) uses
`EDGE_TRIMMABLE` and may never advance the start past the onset unit.

### 4. Judge gates on start-position + close the card escape hatch — `assemble/validate.py`

- Add verdict field **`opening_in_context`**: *"Does the FIRST sentence stand on its own, or open
  mid-thought / at the answer before the question is posed?"* with a required `evidence_quote`
  (the offending opening phrase) so it survives the asymmetric kill gate.
- Add it to `required_verdict_fields` so `is_complete` (`validate.py:~193`) actually gates on it.
- **Zero-LLM precheck** (don't wait on the flaky judge): first in-span unit role is a payoff
  (`result`/`solution`), **or** Tier-A/B flags a weak start ⇒ deterministically route to the
  onset-pull repair (extend backward) before/independent of the LLM verdict.
- **A context card can NEVER satisfy `opening_in_context`.** Card-as-repair may still resolve a
  genuinely distant one-line *prerequisite*, but can no longer launder a mid-thought open.

### 5. Fewer, better, shorter clips — `config.py` + anchor policy

- `max_clip_duration_s: 240 →` soft target 30–60 s / ceiling ~90 s with overflow (no hard sever).
  Add `target_clip_duration_s` used by scoring, not as a cutter.
- Tighten anchor selectivity + dedupe so a video ships a *handful* of strong clips, not 22
  (raise `ANCHOR_MIN_PRIORITY` / per-video anchor budget; strengthen near-duplicate dedupe).
- Auto-trim pure filler openers/enders (the "were there any questions… I gotta erase it",
  "type this into YouTube" cases) via `EDGE_TRIMMABLE` + a filler-phrase check.

### 6. Measure it — `backend/eval/metrics.py`

Add **`opening_onset_rate`**: fraction of shipped clips whose first sentence passes the onset
test (no unresolved referring expression, no bare continuation-DM, not payoff-first, clause-
initial, same-node). This is the missing headline start-quality number, operationalizing
PodReels' "cold viewer isn't confused." Optionally an LLM `cold_open_score` for spot audits.
Report alongside existing `unresolved_reference_rate` / `ends_on_period_rate` /
`context_complete_rate`.

### 7. Sentence-boundary fidelity (Mode A durability, fast-follow) — punctuation restoration

Tier-A's mid-clause check catches most fragment-starts; hardening punctuation-restoration so a
"sentence" never begins mid-clause is the durable lever. Flagged as follow-up, not a blocker.

---

## 5. Verification plan

- **Cheap, trustworthy A/B via `run_eval.py --freeze`** — replays only assembly over cached
  `work/<id>/structure.json` (no network/LLM), isolating this change. Run
  `--freeze --runs 3` on the 10-video corpus **before and after**; primary metric
  `opening_onset_rate` (measure the baseline first — the "60% open on a continuation token"
  scan implies a large fail rate; target **≥95%**), guardrails: no regression in
  `context_complete_rate`, `comprehension`, `ends_on_period_rate`; `clips/video` drops toward a
  handful.
- **Golden acceptance on the named failures:** #8 must open at *"Now what about MgBr2?"* (not
  "So the answer is…"); #1/#2/#5/#12 open cleanly; #6/#7/#11 stay unchanged (no regressions on
  already-good clips).
- **TDD:** unit tests for `_is_weak_start` (continuation tokens; the "So let's start with KI"
  false-positive MUST pass as a good onset; dangling anaphor; mid-clause; cross-node) and for the
  closure "opener never referential" invariant, written before implementation.

---

## 6. Sequencing

**1 + 4** (universal guard + judge gate — cheap, catch the 60%) → **3** (stop trimming setup) →
**2** (reach distant setup, overflow) → **5 + 6** (short/fewer + proof) → **7** (durability).
Each step is independently testable offline via `--freeze`.

## 7. Deferred / non-goals

- No rewrite of segment selection (surgical decision). No split/merge machinery (overflow chosen).
- Per-adapter contract tweaks are cleanup only, not the fix — the guard is genre-independent by
  design (avoids per-domain hacks that won't generalize to interview/podcast).
- Repo is not under git; spec is saved but not committed (no VCS available here).
