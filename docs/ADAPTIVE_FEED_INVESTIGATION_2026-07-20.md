# Adaptive Feed and Clip Pipeline Investigation

- Target: `https://studyreels.app`
- Date: 2026-07-20
- Scope: user input through generated clips, clip quality/boundaries/context, concept metadata, orchestration/progression, feedback and quiz adaptation, repetition, and subsequent batches
- Excluded: `practice`, `practice copy`, and all unrelated product areas

## End-to-end pipeline trace

1. `UploadPanel.onSubmit()` sends topic/text/file input through `uploadMaterial()` and opens `/feed` for the returned material ID.
2. `POST /api/material` normalizes the input, extracts concepts/objectives, embeds material chunks, persists the material/concepts, and initializes learner progress.
3. `/api/feed` and `POST /api/reels/generate` derive the learner level and concept-level feedback/quiz profile, build an adaptation-specific durable request key, and submit or attach to a bounded generation job.
4. `_run_leased_generation_job()` invokes `ReelService.generate_reels()`. Topic acquisition concepts are ordered from lowest mastery/strongest remediation to highest mastery.
5. Retrieval discovers YouTube sources, obtains metadata/transcripts, scores relevance, and sends transcript cues plus the exact learning request to `segment_clips_detailed()`.
6. Gemini returns grounded boundaries and per-clip semantic metadata. `_public_clips()` now exposes the narrow `facet` as `concept`; ingestion normalizes and persists a material-scoped clip concept, the verified transcript range, assessment, and provenance.
7. The lesson organizer receives each clip's narrow concept, summary/takeaways/transcript, dependency/chain metadata, and learner signal. It selects a coherent subset and orders it from prerequisites/orientation through explanation/application, preserving dependencies and source chronology.
8. The worker persists the organizer selection and emits the authoritative final event. Serving treats that release as an allow-list, then reranks its members for the current learner and excludes already seen reels.
9. The frontend reconciles cached rows with the authoritative feed by clip identity, so only backend-backed reel IDs remain feedback-eligible.
10. `Got it`/`Need help` update per-reel feedback and increment `feedback_revision`. Recall-check completion writes per-concept correct/wrong adjustments and increments the same revision. Both signals affect ranking, difficulty targeting, organizer inclusion, acquisition order, and the next generation request key.

## Production baseline (`studyreels.app`)

- Tested the existing Newton's second law session for material `fe3ddb61-d5ee-49a6-a5a5-8a2973672a5c` in the available anonymous browser session.
- Delivered order:
  1. Newton's second law and `F = ma` (27 seconds): relevant and complete, though it opens by referring back to the first law.
  2. Acceleration prerequisite (2:14): relevant and complete, but pedagogically belongs before the core `F = ma` explanation.
  3. Deriving the Newton unit (23 seconds): relevant setup, but it omits the derived result and therefore lacks the claimed payoff (AF-007).
  4. Mass-versus-acceleration relationship (12 seconds): relevant and complete.
  5. Net force (25 seconds): relevant and complete.
- Relevance: all five clips directly addressed the request; four were self-contained enough for the stated objective.
- Repetition: no duplicate or near-duplicate clip appeared in this five-reel sample.
- Progression: the ingredients are sensible, but the acceleration prerequisite arrived after the central law, and the incomplete unit clip interrupted the explanation-to-application flow.
- Adaptive controls: selecting `Need help` on a displayed reel twice returned `reel_id not found` (AF-008). This also prevented a trustworthy live recall-check/adaptive-next-batch comparison on that stale inventory.
- A separate stored mitosis session returned `material_id not found` for both generation and recall resume (AF-001).
- Fresh provider-backed search was unavailable to the anonymous session because production requires a verified account and search quota. Local integration tests therefore establish signal direction until the fixed build is deployed and exercised with a verified production session.

## Findings

### AF-001 — Stored reel session cannot resume when its material no longer exists

- Status: Fixed locally; production retest pending
- Severity: Blocks the affected adaptive-feed and recall-check flow
- Reproduction:
  1. Open `https://studyreels.app` in the existing anonymous production session.
  2. Select the stored “Compare mitosis and meiosis…” reel search.
- Evidence:
  - Browser warning: `Background reel generation failed ... ApiError: material_id not found`.
  - Browser warning: `Could not resume pending recall check: ApiError: material_id not found`.
  - The selected history item remains on the new-search view instead of restoring a playable reel batch.
- Expected: A stored session either restores a valid batch or cleanly expires without attempting adaptive generation/assessment against an invalid material.
- Root cause: The 404 path only displayed an error; it retained the material's cached reels, history, progress, generation eligibility, and assessment bootstrap state, which caused repeated requests and could re-persist the stale session.
- Fix: Expire only the missing material's local/remote history entry, feed snapshot, and progress; abort its request scope; clear its generation/assessment state and stale reels; and prevent the expired session from being persisted again.
- Regression test: Two focused frontend tests cover scoped storage cleanup/idempotence and prove generation plus assessment stop. Full feed suite: 69 passed; TypeScript check passed.
- Exact retest: Pending.

### AF-002 — Current selection-contract reels bypass concept-level adaptation

- Status: Fixed locally; production retest pending
- Severity: High — directly breaks thumb/quiz topic-frequency requirements
- Evidence:
  - `ReelService.ranked_feed()` computes learner `concept_adjustments` and concept coverage from manual feedback and `assessment_concept_outcomes`.
  - All current Gemini reels take the selection-contract branch.
  - That branch calls `_selection_contract_order(..., concept_adjustments=...)`, but `_selection_contract_order()` never reads `concept_adjustments` or the per-concept coverage/latest-remediation signal.
  - Its priority key is limited to difficulty stage, role/coverage, quality, topic relevance, source rank, and time.
- Expected: Helpful/correct signals lower the affected concept's future exposure; confusing/wrong signals raise it, while preserving prerequisites and source chronology.
- Root cause: The legacy adaptive scheduler contains concept coverage/remediation logic, but the newer selection-contract topological scheduler did not carry that logic forward. The parameter was added at the call boundary but never applied.
- Fix: The selection-contract topological scheduler now uses concept exposure, per-concept adjustment, and latest remediation. Helpful/correct concepts move later and target harder inventory; confusing/wrong concepts move earlier and prefer an easier alternate source, while prerequisites remain mandatory.
- Regression test: Focused tests prove helpful versus unseen ordering, confusing alternate-source remediation, and opposing correct/wrong quiz ordering.
- Exact retest: Pending.

### AF-003 — Gemini’s clip-specific topic label is discarded before persistence

- Status: Fixed locally; production retest pending
- Severity: High — feedback and quiz outcomes are attributed to the search concept instead of the clip’s actual taught concept
- Evidence:
  - Gemini segmentation returns a per-clip `title` (plus `facet`) alongside transcript-grounded clip boundaries.
  - `clip_engine.bridge.to_segment()` keeps only start, end, text, and score.
  - `IngestSegment` has no clip-concept field.
  - `ReelService._reel_attribution_to_dict()` overwrites attribution with the acquisition concept’s ID/title, so lesson ordering receives the broad/search concept rather than Gemini’s clip-specific topic.
- Expected: Gemini emits a normalized concept for each clip; it survives conversion and persistence; assessment and feedback attach to that concept; the organizer sees it with the clip content and learner signal.
- Root cause: The clip selector’s semantic label was never included in the ingestion/persistence contract, so downstream code can only reuse the concept that initiated search.
- Fix: Gemini's narrow `facet` is now returned as `concept`; ingestion normalizes it, reuses or deterministically creates a material-scoped concept, persists acquisition and clip-concept provenance, and returns the persisted concept through reel attribution. Topic continuation can acquire those persisted clip concepts. The generation request schema was advanced so pre-concept-metadata production jobs are not reused as current inventory.
- Regression test: Persistence tests cover Unicode normalization, material scoping, deterministic reuse, and provenance; generation tests prove the clip concept survives attribution and legacy generation jobs are rejected for cross-request reuse.

### AF-004 — Lesson organizer is contractually unable to omit a mastered topic

- Status: Fixed locally; production retest pending
- Severity: High — contradicts the requested feedback-aware include/exclude decision
- Evidence:
  - The lesson-order prompt says it may “only change the order” and must return every supplied reel ID exactly once.
  - Response validation rejects any subset.
  - Learner feedback/mastery is not included in `CLIPS_JSON`; only learner level and clip content are provided.
- Expected: The organizer receives clip-specific concepts/content plus learner signals and may omit clips for mastered concepts or prioritize/repeat clips for struggling concepts, subject to minimum batch and prerequisite safety.
- Root cause: Lesson ordering was intentionally built as an order-only stage before adaptive filtering was part of its responsibilities.
- Fix: The organizer now receives narrow concept/content plus numeric learner signals and may return a non-empty subset. Validation still rejects unknown/duplicate IDs, same-source chronology violations, missing prerequisites, or a non-prefix dependency chain; deterministic fallback keeps the complete valid batch.
- Regression test: Lesson-order tests cover mastered-concept omission, signal payloads, prerequisite closure, chain-prefix closure, cache behavior, and fallback. Full lesson-order suite passed.

### AF-005 — The authoritative release order overwrites adaptive reranking

- Status: Fixed locally; production retest pending
- Severity: High — feedback revision invalidates the ranking cache but still cannot change the delivered order
- Evidence:
  - `ranked_feed()` recomputes after `feedback_revision` changes.
  - `_ranked_request_reels()` then calls `_authoritative_release_reel_ids()` and reconstructs the result in the original final-event order.
  - The generation request key contains learner and named knowledge level, but no feedback/mastery revision, so the completed generation and its static final event are reused after thumbs or quiz answers.
- Expected: The authoritative release remains the safe inventory boundary, while current learner adaptation is allowed to select/order within that released inventory for later feed requests.
- Root cause: “Authoritative” release was implemented as both an allow-list and a permanently frozen presentation order; the latter masks the adaptive ranking stage.
- Fix: The authoritative final is now an allow-list rather than a permanent presentation order. Feed and generate endpoints share an adaptation fingerprint; completed, active, and cross-request job reuse requires the current schema and the same fingerprint, so an old organizer subset cannot survive changed feedback.
- Regression test: Release tests prove private reels stay hidden while current ranking determines released order; request-key/compatibility tests prove feedback changes invalidate both the adaptive batch and cross-request subset without duplicating an identical no-signal request.

### AF-006 — Quiz outcomes do not steer which concept a continuation acquires

- Status: Fixed locally; production retest pending
- Severity: High — wrong/right quiz concepts cannot directly alter future topic inventory
- Evidence:
  - Assessment completion persists per-concept adjustments in `assessment_concept_outcomes` and increments `feedback_revision`.
  - `_order_concepts()`, which chooses the single concept family for each bounded acquisition pass, reads only manual `reel_feedback`.
  - The worker always passes `acquisition_concept_offset=0`, so the first concept is decisive and assessment outcomes are absent from that choice.
- Expected: Correct answers deprioritize that concept for future acquisition; wrong answers prioritize it.
- Root cause: Assessment adaptation was added to serving/ranking context but not to the acquisition concept scheduler.
- Fix: `_order_concepts()` now combines manual mastery with persisted assessment adjustments, and topic bootstrap exposes only clip concepts proven by reel provenance. Wrong concepts therefore enter acquisition before correct concepts.
- Regression test: Focused acquisition test proves a wrong quiz concept is selected before a right concept; ranking test independently proves the same opposing effect on served clips.

### AF-007 — Production clip ends before completing the claimed unit derivation

- Status: Fixed locally; regenerated production retest pending
- Severity: Medium — the clip is relevant but omits the payoff needed for a complete explanation
- Reproduction:
  1. Open the existing Newton’s second law reel set for material `fe3ddb61-d5ee-49a6-a5a5-8a2973672a5c`.
  2. Advance to the reel summarized as “Explain how the Newton is derived as the SI unit of force.”
- Evidence:
  - The clip runs for 23 seconds and ends with: `...one meter per second squared for mass and acceleration`.
  - It never states the resulting `1 N = 1 kg·m/s²`, despite its summary/takeaway claiming a derivation of the Newton.
  - The adjacent next reel changes objective to how force scales with mass, so the omitted result is not supplied there.
- Expected: A derivation/explanation clip includes its resulting unit or is labeled as setup rather than a completed derivation.
- Root cause: Trusted Gemini conversion validated schema, grounding, and structural/grammatical edges, but not whether metadata claiming a calculation/derivation actually reached a spoken result. The grammatical substitution premise therefore passed.
- Fix: A narrow semantic-completeness guard rejects calculation/derivation/proof/solution claims when selected speech contains a substitution setup but no explicit grounded result after it. It runs in the trusted Flash, universal Pro, and generic conversion paths. Current request-schema invalidation prevents the persisted pre-fix batch from being reused for a fresh request.
- Regression test: The captured Newton transcript is rejected; an otherwise identical range extended through the spoken Newton result passes. Boundary/selector/calculus suites: 738 passed.

### AF-008 — Production feedback submission fails for a reel that is currently playable

- Status: Fixed locally; production retest pending
- Severity: High — blocks the thumb signal before any adaptive logic can apply it
- Reproduction:
  1. Open the existing Newton’s second law reel set for material `fe3ddb61-d5ee-49a6-a5a5-8a2973672a5c`.
  2. Advance to the 25-second net-force reel from “Newton’s Second Law of Motion: F = ma”.
  3. Select `Need help`.
- Evidence:
  - The reel remains playable in the production feed.
  - The UI immediately displays `reel_id not found` and offers `Retry`.
- Expected: The confusing signal is persisted for the exact displayed reel and the unseen feed tail is recomputed.
- Root cause: Restored snapshots were accepted by cached reel ID/URL, and page-one reconciliation permanently preserved every watched/current cached row even when the authoritative backend no longer contained it. The player could therefore display a deleted reel ID that feedback correctly rejected with 404.
- Fix: Strict restored-feed reconciliation drops unmatched cached prefix rows and replaces clip-identical rows with the authoritative reel ID. It remaps the active/resume index to the canonical clip; live generation settlement still preserves its locked prefix for playback continuity.
- Regression test: Frontend reconciliation test proves a vanished cached row is removed, a clip-identical current row adopts the server ID, and active/resume state follows it. Full feed suite and typecheck passed.
- Exact retest: Pending.

### AF-009 — Feedback API accepts contradictory helpful and confusing signals

- Status: Fixed locally; production retest pending
- Severity: Medium — a malformed client can simultaneously increase and decrease the same concept signal
- Evidence:
  - `FeedbackRequest` independently accepts `helpful=true` and `confusing=true`.
  - `record_feedback()` persists both values and downstream mastery/adaptation code aggregates both, producing an ambiguous learner signal.
- Expected: A reel response can be helpful, confusing, or neutral, but not helpful and confusing at the same time.
- Root cause: Field-level validation constrains types and rating range, but there is no request-level mutual-exclusion check.
- Fix: `FeedbackRequest` now rejects `helpful=true` with `confusing=true` before the endpoint can write either value.
- Regression test: `test_feedback_is_unique_per_learner_and_reel` now submits the contradictory pair, asserts HTTP 422, and confirms only the two valid learner rows exist. Passed locally.
- Exact retest: Pending.

### AF-010 — Persisted organizer subsets are rejected when reapplied

- Status: Fixed locally; production retest pending
- Severity: High — an organizer decision to omit a mastered/redundant topic can be lost on worker replay
- Evidence:
  - The lesson organizer now returns a valid non-empty subset.
  - `_apply_generation_lesson_order()` still requires the stored IDs to exactly equal the full valid inventory, and otherwise returns every reel.
- Expected: A persisted non-empty unique subset should be reapplied if every selected ID still exists, while unknown/duplicate IDs remain invalid.
- Root cause: The persistence/replay helper retained the former order-only exact-set contract after the organizer contract changed to selection plus ordering.
- Fix: Replay now accepts any non-empty stored subset whose unique IDs all exist in valid inventory; unknown IDs still fail safe to the full valid inventory.
- Regression test: Focused generation-job test covers reordered full inventory, one-item omission, and unknown-ID fallback.
- Exact retest: Pending.

### AF-011 — Invalid empty lesson-order metadata hides all valid reels

- Status: Fixed locally; production retest pending
- Severity: High — malformed/empty persisted organizer metadata can turn a valid batch into an empty terminal result
- Evidence:
  - `_stored_generation_lesson_order_ids()` intentionally returns `[]` as the invalid-metadata sentinel.
  - After subset support was added, `_apply_generation_lesson_order()` accepted `set([])` as a subset and returned no reels.
- Expected: A valid organizer selection is non-empty; invalid empty metadata must fail safe to the complete valid inventory.
- Root cause: The exact-set validator previously rejected an empty list by length mismatch; the new subset validator did not preserve that explicit non-empty condition.
- Fix: Replay now explicitly rejects the empty sentinel before subset validation and returns the complete valid inventory.
- Regression test: The persisted-order regression covers the empty sentinel alongside valid full/subset selections and unknown-ID fallback. Passed locally.
- Exact retest: Pending.

## Verification Matrix

| Requirement | Baseline | After fix | Evidence |
|---|---|---|---|
| User input to final clips | Traced | Locally verified | Ten-stage trace above; generation/worker API regressions pass |
| Clip relevance and educational value | 5/5 live Newton clips relevant; one summary overclaimed completion | Guarded locally | Live transcript review plus captured AF-007 regression |
| Clip boundaries and context | 4/5 acceptable; Newton-unit payoff missing | Fixed locally | Exact incomplete range rejected; grounded-result control accepted |
| Gemini per-clip concept metadata | Broad search topic on every live reel | Fixed locally | Gemini → persistence → attribution tests |
| Orchestrator filtering and ordering | Order-only; no learner signal | Fixed locally | Subset, signal, dependency, and replay tests |
| Thumbs-up reduces topic frequency | Live submission blocked by AF-008 | Fixed in scheduler locally | Helpful concept moves behind unseen concept; organizer may omit mastered repeat |
| Thumbs-down increases topic frequency | Live submission blocked by AF-008 | Fixed in scheduler locally | Confusing concept prioritized with easier alternate source |
| Correct quiz concepts appear less | Not observable on stale live batch | Fixed locally | Correct concept deprioritized in acquisition and ranking tests |
| Wrong quiz concepts appear more | Not observable on stale live batch | Fixed locally | Wrong concept prioritized in acquisition and remediation ranking tests |
| Difficulty and educational progression | Core F=ma preceded its acceleration prerequisite | Improved locally | Organizer prompt and validators enforce prerequisite/chain closure; remediation selects easier alternate source |
| Duplicate/near-duplicate control | No duplicates among the 5 live Newton reels | Preserved | Existing clip/source identity dedupe plus organizer anti-redundancy policy |
| Subsequent batch regeneration | Blocked by stale session and reusable static job | Fixed locally | Scoped expiry, adaptation fingerprint, current-schema reuse, and continuation tests |

## Local verification

- Adaptive/clip-focused backend suite: 412 passed, 14 subtests passed.
- Full active backend suite: 2,405 passed, 1 skipped, 37 subtests passed.
- Frontend suite: 179 passed.
- Feed-focused frontend suite: 69 passed.
- TypeScript check: passed.
- Production Next.js build: passed (outside the filesystem sandbox because Turbopack's worker tried to bind an internal port).
- `git diff --check`: passed after the final graph rebuild.
- The only excluded backend test file was `backend/tests/test_labels_api.py`; collection imports the unrelated standalone `backend/main.py` and the local environment lacks `sse_starlette`. The active runtime under review is `backend.app.main`, and every active-backend/adaptive test collected and passed.
