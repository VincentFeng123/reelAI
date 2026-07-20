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

### AF-012 — Feedback and quiz reranks leave the pre-signal generation stream active

- Status: Fixed locally; production retest pending
- Severity: High — clips selected under stale concept signals can re-enter the personalized unseen tail
- Evidence:
  - Both successful thumb submission and completed assessment call `rerankUnseenTail()`.
  - That callback fetches inventory under the new backend adaptation fingerprint, but calls `clearGenerationTracking()` and `renewActiveSearchScope()` only when the coarse `knowledge_level` changes.
  - A feedback or per-concept quiz outcome normally changes concept signals without changing `knowledge_level`, so an already-running old-fingerprint stream remains subscribed and its `onCandidate` callback can append stale candidates after the new tail is applied.
  - The same conditional controls `setCanRequestMore(true)`. If the old request was already exhausted, a same-level signal change removes its unseen tail but cannot request the replacement adaptive batch.
  - Unlike the normal page loader, the rerank path records only continuation tokens. If its current-fingerprint response reports a job already queued by another request, it neither records nor consumes that job.
- Expected: Every successful adaptive rerank invalidates local job/continuation state and aborts streams started under the previous concept-signal fingerprint before applying the new unseen tail.
- Root cause: Generation invalidation was coupled to level changes even though concept-signal changes independently define a new durable generation request.
- Fix: Every adaptive rerank now aborts the old search scope, clears old job/continuation tracking, reopens generation, preserves only the watched prefix, applies the current-fingerprint inventory, and attaches/consumes the current job even when that inventory is temporarily empty.
- Regression test: The compiled rerank callback proves the old scope is aborted, an empty queued response preserves the watched prefix, and the current job is remembered and consumed on the renewed scope. Full frontend suite passed locally.
- Exact retest: Pending.

### AF-013 — The server can release or continue a job after its adaptation fingerprint becomes stale

- Status: Fixed locally; production retest pending
- Severity: High — pre-feedback topic acquisition can survive a thumb or quiz update
- Evidence:
  - Adaptation fingerprints prevent lookup/reuse of an old completed or active job, but a worker already holding that job does not revalidate the fingerprint before authoritative release.
  - The organizer rereads current signals, so it may reorder or omit old inventory, but it cannot add a remediation concept that the pre-signal acquisition pass never retrieved.
  - Explicit continuation validation compares material, learner, content, level, mode, licensing, duration, relevance, and exclusions, but not `adaptation_fingerprint` or the current generation request schema.
  - For an accepted continuation with unseen source-chain reels, the worker intentionally sets the fresh provider budget to zero and drains that old inventory first.
  - The stale queued/running job still counts against the one-active-job-per-learner limit, so the first current-fingerprint replacement request can fail as “generation busy” before the worker reaches its release-time guard.
  - Worker setup attaches a private `result_generation_id` before retrieval. The status/replay serializers treated every terminal state as surfaceable, so a stale job cancelled before activation could expose that private inventory through generation status.
  - A final fingerprint check followed by separate activation/event/terminal commits still has a time-of-check/time-of-use window. Feedback or assessment can commit after the check and before the final event.
  - A request can compute an old fingerprint before feedback commits and insert its old-signal job afterward unless request snapshot/submission shares the same serialization point as signal writes.
- Expected: A worker must not release a batch when its learner concept-signal fingerprint has changed, and a continuation token is valid only under the same fingerprint that created it.
- Root cause: Fingerprint compatibility was enforced at job lookup/reuse boundaries but omitted from leased-worker release and explicit continuation validation.
- Fix: Feed/generate snapshot and submission, feedback, assessment completion, level changes, and final release now serialize on the learner-material progress row. The worker checks schema, concept fingerprint, and selected level before provider work, after acquisition, and inside the atomic activation/final/terminal transaction. Stale same-material jobs are cancelled before replacement capacity checks; stale continuations are rejected; failed/cancelled status and replay never expose private inventory.
- Regression test: Focused tests cover stale schema/fingerprint continuation rejection, pre-provider and post-acquisition cancellation, a signal change at the atomic release boundary, stale active replacement through both generate and feed, selected-level invalidation, cancelled-after-final replay sanitization, and current ranking of the released allow-list. The affected generation/job/billing/feedback/assessment suite passed locally.
- Exact retest: Pending.

### AF-014 — Fresh adaptive generation does not exclude content already watched in the session

- Status: Fixed locally; production retest pending
- Severity: Medium — an exact or near-duplicate can consume the personalized replacement slot and produce zero visible growth
- Evidence:
  - `adaptiveExcludeReelIdsRef` is sent to `/api/feed`, but the direct `/api/reels/generate` path does not send any watched reel, clip, or video exclusion.
  - Reel IDs are generation-scoped, so rediscovering the same source span can create a different UUID.
  - Frontend clip-key deduplication then discards that result only after provider work and batch selection have already spent the slot.
- Expected: A fresh post-feedback/post-quiz acquisition excludes sources already watched in that feed session so it cannot return the same or a near-identical clip as the adaptive replacement.
- Root cause: Adaptive session exclusions were implemented only for ranked-feed reads and were not translated into the generate endpoint's existing `exclude_video_ids` contract.
- Fix: Adaptive rerank derives canonical bare YouTube IDs from the watched prefix (including persisted `yt:` IDs and embed URLs), stores them with adaptive reel exclusions, and sends them through both feed reads and direct generation requests.
- Regression test: API tests prove normalized/deduplicated `exclude_video_ids`; feed tests prove watched source IDs are derived and included in post-signal generation.
- Exact retest: Pending.

### AF-015 — Recall “Continue learning” loses its owed move while the adaptive reel is queued

- Status: Fixed locally; production retest pending
- Severity: Medium — quiz adaptation appears not to take effect because the learner remains on the same reel
- Reproduction:
  1. Open recall from a forward swipe at the current tail.
  2. Complete the assessment so rerank retains only the watched prefix and attaches a queued current-fingerprint job.
  3. Select `Continue learning` before the first candidate arrives.
- Evidence:
  - `closeAssessmentAndContinue()` calls `commitOneReelMove(1)` when the assessment owes a forward advance.
  - At the temporary tail, `commitOneReelMove()` clears `pendingAutoplayAdvanceRef`, calls `maybeLoadMore()`, and returns.
  - The adopted job already holds the generation lock, so `maybeLoadMore()` does nothing; when the candidate arrives, no pending move remains for the existing append effect to consume.
- Expected: Continue either advances immediately to an available personalized reel or retains exactly one pending advance until the queued candidate is appended.
- Root cause: Assessment-close navigation reused a tail-move helper whose first action intentionally clears autoplay debt, even when no next reel exists yet.
- Fix: Assessment close advances immediately when a next reel exists; otherwise, when inventory is available or being generated, it retains exactly one pending autoplay move and asks the existing loader to continue.
- Regression test: Compiled callback coverage proves a queued adaptive tail retains one advancement debt until the candidate arrives.
- Exact retest: Pending.

### AF-016 — Keyboard navigation can open recall without recording the forward move

- Status: Fixed locally; production retest pending
- Severity: Medium — the keyboard path can complete quiz adaptation yet remain on the old reel
- Reproduction:
  1. Reach the current tail and press ArrowDown or PageDown when recall is due.
  2. Complete recall while the personalized replacement reel is still queued.
- Evidence:
  - The tail branch of `jumpOneReel(1)` calls `reportForwardScrollForReel()` but discards its gate request, leaving `advanceRequested=false`.
  - If a pending hidden assessment is opened at the tail, its close-debt flag is based only on whether a reel already exists, ignoring available/queued generation work.
- Expected: Every forward input path marks the recall gate as owing one move when the user intended to advance and a next reel is available or still being produced.
- Root cause: Wheel-tail handling records `advanceRequested`, but the shared keyboard/direct jump tail branch and hidden-assessment reopen path did not preserve the same intent.
- Fix: Keyboard/direct tail movement now records the gate's forward intent, and a hidden pending assessment treats available, requestable, or in-flight inventory as an owed next reel.
- Regression test: Compiled tail-jump and hidden-assessment callbacks prove the forward debt survives both paths.
- Exact retest: Pending.

### AF-017 — Final quiz adaptation can commit before its job cancellation on SQLite

- Status: Fixed locally; production retest pending
- Severity: High — a failed request can persist mastery while leaving the old generation active
- Evidence:
  - `get_conn(transactional=True)` uses SQLite's deferred transaction mode.
  - `AssessmentService.answer()` opens a savepoint; when no outer DML has begun, releasing that outermost savepoint commits the final attempt, outcomes, and feedback revision before the endpoint recomputes difficulty and cancels stale jobs.
  - An exception after `answer()` can therefore leave partial adaptive state committed.
- Expected: The final answer, per-concept outcomes, difficulty adjustment, and stale-job cancellation commit or roll back together.
- Root cause: The endpoint did not begin/lock the caller-owned SQLite transaction before entering the assessment service's nested atomic write.
- Fix: The answer endpoint resolves the learner-owned session and locks its learner-material progress row before calling the assessment service. That DML begins the real outer SQLite transaction and also serializes completed-session replays with concurrent feedback.
- Regression test: A real file-backed API test forces a failure after final assessment persistence and proves the final attempt/outcomes roll back while the first two answers remain. A two-connection SQLite test proves the adaptation lock blocks the competing transaction until commit.
- Exact retest: Pending.

### AF-018 — A selected-level change does not make an in-flight job stale

- Status: Fixed locally; production retest pending
- Severity: Medium — a beginner batch can publish or occupy the only learner slot after switching to advanced
- Evidence:
  - Worker staleness compared request schema and concept fingerprint, but the concept fingerprint intentionally excludes the named knowledge level.
  - The level PATCH changed `selected_level` without immediately cancelling active generation.
- Expected: A job may publish only under the selected level that created it, and a level change immediately frees the learner slot for the replacement difficulty.
- Root cause: Knowledge-level compatibility existed in lookup/request keys but not in the leased-worker stale guard or stale-active cancellation helper.
- Fix: Worker and active-job cancellation now compare stored and current selected levels; the level PATCH uses the same adaptation lock and cancels old-level work in its transaction.
- Regression test: Focused tests prove an old-level lease is cancelled before provider work and a level PATCH cancels its queued old-difficulty job.
- Exact retest: Pending.

### AF-019 — First-use adaptation locking can reset newly committed progress

- Status: Fixed locally; production retest pending
- Severity: High — concurrent first feedback/generation can erase level, mastery adjustment, and revision
- Evidence:
  - The initial adaptation lock called `learner_progress()` when no progress row existed.
  - Its generic upsert uses `ON CONFLICT DO UPDATE` for all progress fields.
  - Two first-use transactions can both observe no row; after the winner commits a signal, the waiting loser's conflict update can replace the winner's level, adjustment, reset timestamp, and revision with defaults.
- Expected: Creating the serialization row must never alter an existing learner-progress value, including a row committed while the creator was waiting.
- Root cause: A read-or-overwriting-upsert helper was reused for insert-only lock seeding.
- Fix: Adaptation locking now seeds with `INSERT ... ON CONFLICT DO NOTHING`, then acquires the no-op row update lock. The conflict path preserves every winner-owned value on SQLite and PostgreSQL.
- Regression test: The two-connection SQLite test now begins with no progress row; transaction one creates and changes it to advanced/0.2/revision 7 while transaction two waits, then proves transaction two acquires the lock without resetting any value.
- Exact retest: Pending.

### AF-020 — Fresh production generation immediately rejects its continuation requests

- Status: Fixed locally; production retest pending
- Severity: High — a newly created material can remain indefinitely on “Finding the first clips” with no baseline reels
- Production reproduction (2026-07-20):
  1. Signed in to `studyreels.app` and submitted a new slow-mode beginner material covering Newton's laws, balanced forces, F=ma, free-body diagrams, action–reaction pairs, worked problems, and misconceptions.
  2. The app created material `ed4192fd-fdb9-46f3-92b7-2a4144c3b65a` and navigated to its feed.
  3. No reel appeared; the feed remained on “Finding the first clips”.
- Evidence: The production browser emitted four same-second warnings from the feed's background generation path: `ApiError: The requested reel batch can no longer be continued.`
- Expected: A fresh material's current generation job is adopted and streamed, or a compatible replacement job is created; the client must not issue an already-invalid continuation that leaves the initial feed empty.
- Root cause: `material_content_fingerprint()` treated every material concept as input content. Gemini clip analysis deterministically persisted four new narrow clip concepts while the job ran, so the job changed its own content fingerprint. Its `partial` terminal token then failed the continuation validator's current-fingerprint check.
- Fix: Material-wide fingerprints now exclude only concepts whose IDs prove they were deterministically generated from clip analysis; explicitly requested clip concepts remain fingerprinted, and ordinary source/user concepts still invalidate inventory when changed. The request schema advances to `adaptive_clip_concepts_v2` so old fingerprint semantics cannot be reused.
- Regression test: A focused fingerprint test proves a generated clip facet is inert while a source concept changes the hash. A full generate API regression persists a clip facet during a material-wide partial batch, then proves its continuation queues successfully. The complete generation job/API files pass: 168 tests.
- Exact retest: Pending on the same production material sequence.

### AF-021 — A long single-line learning request is silently treated as pasted source text

- Status: Fixed locally; production retest pending
- Severity: Medium — detailed user requests seed fragmented concepts before clip analysis and weaken the adaptive curriculum
- Production evidence: The single-line Newton request used for AF-020 exceeded 80 characters. `resolveUnifiedComposerRoute()` therefore sent it as `raw_text` instead of `subject_tag`; production seeded phrase-fragment concepts such as “Teach Newton'S”, “Laws First”, “Principles Inertia”, and “Diagrams Action-Reaction”.
- Expected: A normal single-line “Ask ReelAI” request remains a topic prompt regardless of useful detail; pasted source text should require an unambiguous signal such as line breaks or a file.
- Root cause: The composer uses `prompt.length > 80` as a hidden source-mode switch.
- Fix: The unified composer now uses an unambiguous line-break signal for pasted source text. A non-URL single-line request remains a topic regardless of length; file and URL routing are unchanged.
- Regression test: The composer routing test proves a 240-character single-line request remains a topic while multiline text remains source input. All 9 UploadPanel tests pass.
- Exact retest: Pending with a detailed single-line topic prompt.

### AF-022 — A clean material's automatic continuation is rejected by Supadata

- Status: Fixed and verified on deployed revision `8d24b5d`
- Severity: High — the initial batch stops before covering the requested curriculum, and the quiz/feedback matrix cannot observe healthy later batches
- Production reproduction (2026-07-20, clean Postgres and Redis, deployed revision `4e0973f`):
  1. Submit the one-line beginner topic “Newton's laws: inertia and balanced forces, net force and F=ma, free-body diagrams, action-reaction pairs, then worked problems and common misconceptions.”
  2. Material `7e776465-5015-4234-baa1-7c0e94611f0b` correctly routes as `source_type=topic` and job `58e1f943-7378-46ce-a0ff-62c12458c1f5` publishes seven reels as a partial batch.
  3. The automatic continuation job `dd36b76b-4f3d-4b01-ba9c-5d508cd6977d` immediately fails.
- Evidence: The terminal code is `provider_request_rejected` with message `Supadata rejected the search request (400).` The active batch covers first-law/inertia, balanced forces, third-law pairs/misconceptions, and mass/acceleration intuition, but it has no explicit F=ma calculation, free-body-diagram instruction, or worked numerical problem.
- Expected: A valid continuation searches the uncovered concepts and appends a later batch, so broad but normal learning prompts can reach the requested curriculum without a provider-level rejection.
- Root cause: The search orchestration correctly carried both the logical query and provider cursor into the Supadata adapter, but the adapter's `page_token` branch discarded the query and sent only `nextPageToken`. Production now validates its documented required `query` field and returned `400 query: Required`; the existing unit test incorrectly required the broken token-only wire shape.
- Fix: Paginated Supadata calls now send the whitespace-normalized logical query together with `nextPageToken` while continuing to omit initial-page filters.
- Regression test: The adapter fake now rejects a missing query exactly like production and asserts the paginated request is exactly `{query, nextPageToken}`. All 15 Supadata adapter tests and five focused provider-cursor/consumed-page tests pass; the old implementation raises `ProviderRequestError` under this test.
- Exact retest: The deployed continuation now sends the required query and reaches Supadata pagination. Its different `400 Invalid or expired continuation token` response is not the missing-query defect and is tracked separately as AF-033.

### AF-023 — Degraded organizer fallback starts with an advanced misconception

- Status: Fixed locally; production retest pending
- Severity: High — a beginner feed reverses the requested prerequisite progression
- Production reproduction (2026-07-20): Open generation `e3cf97d7-6f6d-4dd8-9815-5dc9a04ec0bd` for fresh material `7e776465-5015-4234-baa1-7c0e94611f0b`.
- Evidence:
  - Seven concept-attributed reels are persisted. One `difficulty=0.35` reel is intentionally level-deferred for a beginner request, so the six-ID lesson/UI count is correct and is not part of this defect.
  - The organizer metadata says `degraded=true` and `fallback_reason=invalid_model_order`.
  - The visible order starts with `third law misconceptions`, then `action-reaction pairs`, and only afterward introduces `Newton's first law`, `concept of inertia`, and balanced forces. The `mass and acceleration relation` reel is omitted.
- Expected: Even when model output is invalid, deterministic fallback preserves all surface-eligible clips in a constraint-safe educational sequence—source chronology, prerequisites, and derivation-chain order—so an explanation precedes its dependent example or misconception.
- Root cause: Gemini returned schema-valid output that failed semantic order validation. The degraded `_fallback` then returned the raw ranked input unchanged even though that input itself ran the same source at `96.9s → 10.24s → 77.495s`, putting the third-law misconception before its introduction and first-law foundation.
- Fix: Degraded fallback now performs a stable topological order over same-source chronology, declared prerequisites, and chain positions, using original rank as the tie-breaker and retaining every eligible clip. Contradictory metadata cycles break deterministically without dropping inventory.
- Regression test: Updated prerequisite, chain, and chronology tests make the raw input intentionally invalid, and a production-shaped six-clip test proves the fallback retains every clip while restoring source chronology. All 18 lesson-ordering tests pass.
- Exact retest: Pending on a fresh beginner Newton curriculum.

### AF-024 — A visible reel is fully contained inside another visible reel

- Status: Fixed locally; production retest pending
- Severity: High — a near-duplicate consumes scarce curriculum inventory and repeats the same explanation instead of covering an unmet concept
- Production evidence (generation `e3cf97d7-6f6d-4dd8-9815-5dc9a04ec0bd`):
  - `ingest-d5af4e4fc4bd4331` (`concept of inertia`) uses YouTube source `LQyFshgm-hU` from `38.52–148.61` (110.09 seconds).
  - `ingest-c0ac05a369874f2a` (`balanced forces on a ball`) uses the same source from `38.52–120.25` (81.73 seconds).
  - The second span is fully contained in the first, and both IDs appear in the six-reel visible organizer order.
- Expected: Same-source spans with complete or substantial temporal containment are treated as near-duplicates before final selection; the stronger clip remains and the freed slot is used for missing curriculum coverage.
- Root cause: The trusted Gemini selector path intentionally bypassed the earlier candidate finalizer, while lesson validation checked IDs, chronology, prerequisites, and chains but had no temporal-overlap invariant. Both a valid organizer response and degraded raw-order fallback could therefore release substantially contained same-source spans.
- Fix: The release layer now removes a later organizer-selected same-source clip when it overlaps at least 80% of the shorter span. The organizer's first choice wins, removed checkpoints are filtered, and clips participating in a declared prerequisite or chain remain protected so deduplication cannot break lesson closure.
- Regression test: Focused tests reproduce the exact `38.52–148.61` / `38.52–120.25` production pair in degraded fallback, prove short-first organizer preference and checkpoint filtering, and preserve prerequisite/chain members. All 21 lesson-ordering tests pass.
- Exact retest: Pending on a fresh Newton batch with overlapping candidate spans.

### AF-025 — “Got it” suppresses only an exact concept ID, not the same concept family

- Status: Fixed locally; production retest pending
- Severity: High — thumbs-up appears to work at the storage layer while the next lesson repeats the same topic under newly generated labels
- Production reproduction (2026-07-20):
  1. On reel `ingest-15d951e27e4945f7`, submit “Got it” for concept `action-reaction pairs`.
  2. Production persists `helpful=1`, increments `feedback_revision` to 1, changes the adaptation fingerprint, and creates completed generation `12bb9d09-a112-43c7-8abd-e3e66d50e007` while excluding the watched source video.
  3. The new active organizer lesson nevertheless selects multiple action–reaction variants, including `identifying action-reaction pairs`, `gravitational action-reaction pairs`, and `action-reaction acceleration misconception`.
- Expected: Positive mastery reduces the future frequency of that semantic concept family, not merely one deterministic concept UUID; new clip labels for equivalent or narrower variants must inherit the relevant learner signal before organizer selection.
- Root cause: Gemini-created facets were correctly persisted as distinct normalized concept IDs, but acquisition ordering, feed ordering/remediation, organizer signals, and quiz adjustments all looked up learner state by exact `concept_id`. A signal on `action-reaction pairs` therefore never reached its narrower labels. The first conservative lexical patch handled those observed labels but could not safely infer true synonyms such as `Newton's first law` ↔ `law of inertia` or `Newton's second law` ↔ `F=ma`.
- Fix: The production Gemini selector now returns a required domain-qualified `concept_family` plus at most four exact-equivalence aliases for every selected clip. Ingestion persists that trusted metadata under versioned `concept_family_v1`; adaptive reads join only direct normalized intersections from trusted same-material reel metadata and reject bare or conflicting ordinals. Legacy rows retain the conservative teaching-role lexical fallback. Family-aware signals feed acquisition, remediation, feed ordering, and organizer payloads, where the organizer now sees the clip family, aliases, transcript content, and inherited learner signal. Active-job stale fingerprints intentionally remain exact-source keyed so Gemini adding a related target facet cannot invalidate its own job.
- Regression test: Focused tests cover both thumb directions across all observed action–reaction labels, correct/wrong quiz propagation into acquisition and easier remediation, first-law/inertia and second-law/F=ma equivalences, numbered-law and thermodynamics non-conflation, Gemini schema enforcement, public-output preservation, cache rejection of missing family metadata, durable ingestion provenance, organizer signal expansion, and active-job fingerprint stability. The final semantic-family matrix passes 73 tests; compilation and scoped `git diff --check` pass.
- Exact retest: Pending with two post-feedback batches on a fresh isolated material.

### AF-026 — The live recall checkpoint disappears without creating a quiz session

- Status: Fixed locally; production retest pending
- Severity: High — quiz answers cannot affect later reels when the checkpoint never becomes answerable
- Production reproduction (2026-07-20, clean Postgres and Redis, deployed revision `4e0973f`):
  1. Open beginner material `0de66b42-3765-45b6-a38f-5bbd0a5db336` and progress through all nine surfaced reels without submitting thumb feedback.
  2. Forward navigation briefly displays `Preparing recall check...` while each scroll is recorded.
  3. Even after the ninth distinct reel and a final tail swipe, no recall modal opens.
- Evidence: A read-only production query immediately after the reproduction finds zero `assessment_sessions` and zero `assessment_session_questions` for this material, so this is not merely a hidden or snoozed frontend session.
- Additional evidence: All nine distinct reels have durable `scrolled_at` rows. Their active version-2 lesson plan is `degraded=true` with `assessment_checkpoint_reel_ids=null`; no `/api/assessments/next` request is emitted because every scroll response remains not-ready.
- Expected: The checkpoint creates an assessment session, presents answerable concept-grounded questions, records each outcome, and updates subsequent generation inputs.
- Root cause: The degraded version-2 organizer plan is still treated as authoritative cadence control. With no valid checkpoint ID it returns `(cadence_target=0, organizer_plan_active=true)` forever, explicitly suppressing the three-scroll compatibility cadence even after the entire batch is traversed.
- Fix: Canonical degraded lesson plans now mark their own reels as not organizer-controlled for cadence, allowing the existing safe three-scroll numeric cadence. Those assignments still resolve the reels, so an older plan cannot reclaim them. Valid non-degraded empty checkpoint lists and corrupt/incomplete metadata remain conservative and do not invent quizzes.
- Regression test: The exact nine-reel/no-prepared-question regression proves the first two scrolls are not ready, scroll three and later are ready, and `/next` deterministically prepares questions and creates a three-question session. The assessment service/API files pass 59 tests.
- Exact retest: Pending on a fresh isolated quiz material.

### AF-027 — Existing selector-contract fixtures omit the new required concept-family fields

- Status: Fixed locally
- Severity: Release blocker — the runtime contract was correct, but a narrow test selection concealed stale fixtures in the full selector matrix
- Reproduction: Run the combined clip-generation matrix after making `family` and `aliases` mandatory in the compact Gemini schema. Forty-one selector-contract tests fail while converting their old helper payloads because those payloads still represent the pre-family response contract; complete backend passes subsequently expose two stale Pro budget fixtures, one stale ranked-cache version assertion, and the captionless YouTube end-to-end mock.
- Root cause: The first focused verification ran only the new schema assertion and semantic-family tests, not the complete selector and backend matrices. Shared/direct proposal fixtures still omitted `concept_family` / `concept_aliases`, and one cache test still pinned version 42.
- Fix: Updated only those shared/direct test fixtures and the captionless Gemini mock to emit valid, exact concept-family metadata and advanced the cache assertion to version 43; the production requirement remains strict and missing live fields are still rejected.
- Exact retest: The full selector-contract file passes all 637 tests; the final affected matrix passes all 1,103 tests; and the complete active backend passes 2,435 tests plus 37 subtests.

### AF-028 — Family propagation crashes compatibility databases without a `concepts` table

- Status: Fixed locally
- Severity: Medium — feed finalization can fail instead of preserving exact-ID adaptation in a partially migrated or deliberately minimal database
- Reproduction: Run the complete backend matrix. Five ranked-request regression cases build the established compatibility schema without `concepts`; `_learner_adaptation_context()` now queries that table unconditionally and raises `sqlite3.OperationalError` before returning the feed.
- Root cause: The new family expansion read had no equivalent of the existing missing-assessment-table compatibility guard.
- Fix: When SQLite specifically reports a missing `concepts` table, adaptation disables only semantic-family propagation for that call and preserves the established exact-ID path. All other database errors still raise.
- Regression test: The five existing ranked-request cases reproduce the compatibility path; all pass on focused retest and in the final complete-backend matrix.

### AF-029 — Ranked-feed shaping strips concept-family metadata before the organizer

- Status: Fixed locally; production retest pending
- Severity: Critical — stored family adaptation works, but the organizer cannot make the requested semantic include/exclude decision because it receives empty family fields
- Reproduction: Persist a trusted `concept_family_v1` reel, run it through the real `ranked_feed()` and `_public_generation_reel(..., preserve_lesson_order_metadata=True)` path, then inspect the lesson-order prompt. The reel's family and aliases are absent even though direct `_selection_metadata()` output contains them.
- Root cause: `ranked_feed()` removes every `_selection_*` key and selectively restores ordering metadata without restoring the two new family keys. The subsequent public-generation preservation allowlist also omits them. The first organizer regression bypassed both boundaries by calling `_selection_metadata()` directly.
- Fix: Ranked-feed response shaping now preserves the already validated internal concept family and alias list, and the lesson-order-only allowlist carries both fields through the public-generation boundary. They remain excluded from ordinary public reel output.
- Regression test: The organizer signal test now traverses persistence → real ranked feed → lesson-order-preserving generation shaping → organizer prompt and proves that `law of inertia`, its `Newton's first law` alias, and the inherited helpful signal all arrive together. Exact focused retest passes.

### AF-030 — An older authoritative plan suppresses cadence for later degraded-plan reels

- Status: Fixed locally; production retest pending
- Severity: High — a mixed feed history can still prevent quiz answers from ever influencing future reels
- Reproduction: Scroll one reel owned by an older valid organizer plan with an explicit empty checkpoint list, followed by three reels owned by a newer degraded plan with null checkpoints. All four scroll responses remain not-ready because the older reel makes the entire window organizer-controlled.
- Root cause: `_cadence_target()` treated any authoritative assignment as control over the complete mixed scroll window and ignored the explicit degraded assignments on later reels.
- Fix: Within a mixed organizer-controlled window, degraded-plan positions are counted separately. The third such reel activates the numeric fallback at its actual absolute position, while unrelated unassigned/legacy reels still cannot override an authoritative empty plan.
- Regression test: The exact one-authoritative-plus-three-degraded sequence now reaches a three-question quiz on the fourth scroll. The focused cadence matrix passes 7 tests and the assessment service/API suites pass all 60 tests.

### AF-031 — Rollout and ambiguity gaps in concept-family matching

- Status: Fixed locally; production retest pending
- Severity: Critical — adaptation can either fail to cross a trusted synonym boundary or leak across unrelated laws
- Independent review reproductions:
  1. A legacy concept titled `Newton's first law` does not join a new trusted `law of inertia` profile even when that profile explicitly aliases `Newton's first law`, because the matcher ignores the one available profile whenever the other side lacks metadata.
  2. Legacy lexical fallback accepts the ambiguous bare title `first law` as a two-token subset of both `Newton's first law` and `first law of thermodynamics`.
  3. Two individually valid Gemini profiles on reels sharing one broad facet/concept ID are unioned. First-law and second-law (or Newton and thermodynamics) identities can therefore form one aggregate bridge.
- Follow-up adversarial review: The first patch still allowed a one-sided trusted profile to fall back to lexical matching after an exact miss (`Newton's law` → `Newton's law of cooling`), intersection-only profile consensus discarded valid optional-alias evidence, and discarding an unsafe profile could not undo feedback already aggregated under the same facet-derived concept ID.
- Root cause: One-sided profile matching fell directly back to title containment; the bare-ordinal guard existed only in canonical identity creation; persisted profile evidence lacked an order-independent connected-consensus check; and durable clip concept IDs were still derived solely from `facet` rather than the trusted canonical family.
- Fix: If either side has trusted metadata, family matching now succeeds only on an exact normalized family/alias identity and otherwise fails closed; lexical fallback is limited to two legacy rows and rejects titles that cannot form a domain-qualified identity. Repeated profiles are combined only when all evidence sets form one connected overlap component, with aggregate ordinal validation. Most importantly, new trusted Gemini clips use a versioned family-derived deterministic concept identity while retaining the narrower facet in `clip_concept_raw` / `clip_concept_key` provenance. Identical facets with different families therefore receive different concept IDs before any feedback or quiz aggregation. Family persistence also carries its Gemini authority through the direct ingest path.
- Regression test: Focused cases prove trusted `law of inertia` → legacy `Newton's first law` rollout bridging, reject one-sided `Newton's law of cooling` and both bare-ordinal cross-domain matches, preserve connected optional-alias evidence, discard disconnected first-/second-law profiles sharing an old concept ID, and give identical `first law` facets separate Newton/thermodynamics concept IDs with isolated feedback adjustments. The focused persistence/adaptation/generation matrix passes all 79 tests.

### AF-032 — Family-derived concepts break continuations and split ordinal synonyms

- Status: Fixed locally; production retest pending
- Severity: Critical — a job can invalidate its own material fingerprint after its first persisted clip, while exact family synonyms can fragment feedback and quiz history
- Independent review reproductions:
  1. Persisting a new `clip-concept-family-v1` concept changes the material-wide content fingerprint because `_is_generated_clip_concept()` recognizes only the older facet-derived UUID namespace. A continuation then cannot match its root request key, resurrecting AF-020.
  2. `Newton's 1st law` and `Newton's first law` normalize to different family identities and deterministic UUIDs.
- Root cause: The generated-concept classifier and concept-family normalizer were not advanced with the new identity contract.
- Fix: Family-scoped concept rows now store the canonical family title, allowing the generated-concept classifier to recognize both legacy facet and family-v1 deterministic namespaces without querying reel state. Material-wide fingerprints ignore both forms, while concept-scoped fingerprints still include the selected generated concept. A shared family normalizer canonicalizes possessives and `1st`/`2nd`/`3rd`/`4th` spellings before persistence UUID generation, profile matching, and fingerprint classification.
- Regression test: The material fingerprint remains stable after creating both legacy and family-v1 clip concepts, the family concept still changes its concept-scoped fingerprint, the exact live continuation regression uses the family-v1 path, and `Newton's 1st law` shares both its semantic identity and UUID with `Newton's first law`. The focused fingerprint/persistence/adaptation set passes all 43 tests.

### AF-033 — The newest-schema live continuation is still rejected by Supadata

- Status: Fixed locally; production retest pending
- Severity: High — the first clean batch contains relevant foundations, but already-analyzed second-law and free-body clips remain unreachable when the automatic continuation fails
- Production reproduction (2026-07-20, after the exact GitHub/Railway/Vercel SHA was verified and scoped Postgres/Redis state was cleared):
  1. Submit the beginner topic “Newton's laws: begin with first-law inertia and balanced forces, then net force and F=ma, then free-body diagrams, then third-law action-reaction pairs, and finish with worked problems and common misconceptions.”
  2. Material `588811b2-3bdb-4479-802c-dbbfd41a6747` creates initial job `162ca842-1579-4b39-b04a-c8d2a9ec3467` with request schema `adaptive_clip_concepts_v2` and active generation `1b76a5bd-1dce-4a29-bdd8-87e92d10ec24`.
  3. Gemini family concepts persist successfully, and the organizer emits a non-degraded five-reel foundation beginning with first law, balanced forces, and inertia.
  4. Automatic continuation job `3eee03f6-ca80-420f-bba9-db5624dbd572` fails with `provider_request_rejected`: `Supadata rejected the search request (400).`
- Evidence: Ten relevant clips were persisted, including explicit `Newton's second law` and `free-body diagram` clips, but the active five-ID lesson omits both. The failed continuation creates an empty pending generation `cb2ddbbf-711f-4166-8117-f5eeea0a5e58`.
- Expected: One unusable opaque cursor cannot fail the whole discovery stage. The step retries through another independent, subject-grounded query branch and can expose uncovered requested concepts; permanent authentication, quota, and non-pagination request failures remain terminal.
- Root cause: Direct production-key probes proved Supadata's cursor itself was unusable for this long compound query: both the provider-issued escaped token and a once-decoded token returned the same `400 Invalid or expired continuation token`, while the documented `query + nextPageToken` wire contract was correct. The failure became fatal because intent validation reserved one of three slots for a literal broad query. Gemini had returned three collectively complete focused Newton queries, but no two could cover every named constraint beside that reserved literal, so validation discarded all three. The literal query was the only branch, and `_continue_provider_pages()` sent every cursor frontier in one call and propagated its first `400`.
- Fix: When Gemini supplies no valid broad query, validation may now use all three focused slots only if their union preserves every request constraint. Expansion retries the same Flash-Lite step once for a transient dispatch, schema, or intent-contract failure before the deterministic literal fallback. Provider pagination now processes cursor branches FIFO one at a time; only a `400` whose detail explicitly identifies an invalid continuation/page token exhausts that opaque branch and retries discovery through the next grounded query. Cancellation, budget, authentication, quota, rate-limit, transient, unrelated-400, and non-400 failures retain their existing bounded/fatal semantics. A rejected token is never replayed within the attempt and is never logged. The search cap is five provider attempts: three complementary initial queries, one rejected cursor, and one independent recovery branch.
- Regression test: The exact long Newton request accepts the three-query complete cover; an incomplete all-focused union still falls back literally; a first Flash failure succeeds on the bounded second attempt; permanent configuration is not retried; and a cold-cache, budgeted three-query cursor reproduction proves calls occur as `[initial, bad branch, good branch]`, returns fresh inventory on the fifth and final reservation, and preserves a generic warning. Unrelated `400 query is required`, non-400 rejection, empty pages, deeper cursor chains, and budget-open behavior remain covered.
- Exact retest: Pending after the fix is deployed, its exact GitHub/Railway/Vercel revision is verified, and production Postgres/Redis learning state is cleared again.

### AF-034 — A persisted organizer selection is rejected on feed reload

- Status: Fixed locally; production retest pending
- Severity: High — the first response can show the organizer's intended progression while a reload silently discards that same persisted lesson selection
- Production evidence: For material `588811b2-3bdb-4479-802c-dbbfd41a6747`, generation `1b76a5bd-1dce-4a29-bdd8-87e92d10ec24` persisted a non-degraded five-ID `lesson_order_v4` plan. Production then logged `Ignoring invalid lesson selection generation_id=1b76a5bd-... stored=5 valid=0` twice while serving the feed.
- Expected: A valid organizer plan remains valid across the initial response, polling, watched-reel filtering, and reload. Removing a watched reel projects the stored sequence onto the remaining clips rather than invalidating the sequence.
- Root cause: `_apply_generation_lesson_order()` required every persisted organizer ID to remain present after seen/exclusion filtering. Once the first reel was completed, the current reel set was a proper subset of the five-ID plan, so the helper rejected the entire plan and ranking exposed a third-law clip before the intended balanced-force and inertia foundations. Terminal `/api/feed` additionally skipped the helper when an authoritative release existed.
- Fix: The helper now projects a persisted order when current reels are a filtered subset, while retaining the existing selected-subset and invalid-mixed-set safeguards. For a terminal organizer chain, the feed projects filtered rows onto the chain-wide authoritative final-event sequence only when every contributing generation has lesson metadata exactly matching its own authoritative release. This preserves older unseen releases and reused source siblings across organizer continuation batches without letting one child plan freeze a legacy source batch; legacy or mismatched releases retain their existing allow-list/current-ranking behavior.
- Regression test: Stored `[a,b,c]` with current `[c,b]` now returns `[b,c]`; an empty set stays empty and a mixed unknown set remains unchanged. A production-shaped released feed shuffled as `[third,second,first]` with `first` seen returns `[second,third]`. A two-generation reused-sibling case proves source release `[a,b]`, child release `[c,d]`, and `a` seen returns `[b,c,d]` rather than dropping `c` or `b`. Additional controls prove a legacy source batch is not frozen by a child organizer, unknown stored IDs cannot authorize terminal projection, and a later authoritative empty release hides its raw inventory without disabling an earlier valid organizer order.
- Exact retest: Pending after deployment on the same fresh-material reload sequence.

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

- Original affected generation/job/billing/feedback/assessment backend suite: 231 passed.
- AF-020/AF-021 affected backend suite: 396 passed, 14 subtests passed.
- Full active backend suite after AF-020/AF-021: 2,420 passed, 1 skipped, 37 subtests passed.
- Final affected adaptive/selection/generation/assessment suite after AF-022 through AF-032: 1,103 passed.
- Final full active backend suite after AF-022 through AF-032: 2,435 passed, 1 skipped, 37 subtests passed.
- Final full active backend suite after AF-033 and AF-034: 2,450 passed, 1 skipped, 37 subtests passed.
- Independent atomic re-review: GO; all prior release blockers reproduced as fixed in five focused checks and direct probes.
- Frontend suite: 182 passed.
- Feed-focused frontend suite: 72 passed.
- TypeScript check: passed.
- Production Next.js build: passed.
- `git diff --check`: passed after the final graph rebuild.
- The only excluded backend test file was `backend/tests/test_labels_api.py`; collection imports the unrelated standalone `backend/main.py` and the local environment lacks `sse_starlette`. The active runtime under review is `backend.app.main`, and every active-backend/adaptive test collected and passed.
