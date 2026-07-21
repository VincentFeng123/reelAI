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

### AF-035 — Clean production fallback reverses the user's explicit concept sequence

- Status: Fixed locally; production retest pending
- Severity: High — every released clip is individually relevant, but the lesson begins at the requested fourth concept and then moves backward to its prerequisites
- Production reproduction (2026-07-20, after exact GitHub/Railway/Vercel revision verification and a fresh scoped Postgres/Redis clear):
  1. Submit “Newton's laws: begin with first-law inertia and balanced forces, then net force and F=ma, then free-body diagrams, then third-law action-reaction pairs, and finish with worked problems and common misconceptions.”
  2. Material `6d16c8bb-95ae-4804-890c-8d8cf49fc852` creates job `15584162-e4a5-4108-93e8-7bdfb9f3ec13` and generation `0253f774-2d2c-4bca-8821-afdd969209ec`.
  3. Open or reload the released feed.
- Evidence:
  - The persisted seven-ID lesson order is three `Newton's third law of motion` clips followed by `Newton's first law`, another first-law example, `net force`, and a final first-law recap.
  - Durable learner progress contains only the first third-law reel, so the browser did not silently consume or skip a prerequisite reel before displaying it.
  - The validated inventory contains 13 concept-attributed clips, including first law, net force, second law/F=ma, free-body diagram, and third law. This is an organizer/fallback defect rather than absence of all requested concepts.
  - Organizer metadata is `degraded=true`, `fallback_reason=invalid_model_order`, and `provider_called=true`; the one schema-valid but semantically invalid model result is immediately replaced by fallback without retrying the organizer step.
- Expected: The organizer uses the learning request as trusted subject/progression intent while treating it as untrusted for policy or output-format instructions, retries a recoverable schema/semantic ordering failure once, and keeps the explicitly requested concept sequence in any deterministic fallback when the supplied clips support it.
- Root cause: The organizer does receive the exact material text, but its system contract says to ignore any instruction in every `CLIPS_JSON` field including `topic`. Validation enforces IDs, checkpoints, source chronology, prerequisites, and chain positions, but no requested cross-concept sequence. When Gemini's first order fails that structural validation, `_fallback()` immediately topologically sorts only source/chain/prerequisite edges and uses discovery rank to break independent ties; the first discovered source happened to teach third law.
- Fix: Organizer contract `lesson_order_v5` now separates `LEARNING_REQUEST_JSON` from untrusted `CLIPS_JSON`: the learning text may supply only subject/scope/emphasis/relative concept order and cannot change policy, IDs, or output rules. A schema or semantic model-order failure retries the same organizer step once with a fresh reservation. If both responses remain structurally invalid, fallback uses the explicit requested concept positions and the last schema-parsed known-ID preference as stable topological priorities while still enforcing source chronology, prerequisites, chains, and overlap filtering. Transient provider/reservation failures retry once; permanent configuration, blocked output, and ordinary `400/401/402/403/404/422` failures do not loop.
- Regression test: A production-shaped third-law-first candidate batch fails model validation twice yet degrades to first law/inertia → net force → third law while retaining every safe clip and repairing same-source chronology. A first-invalid/second-valid case proves exactly two organizer calls; transient `503` succeeds on attempt two and permanent `400` stops after one. Prompt assertions prove the learner sequence is curriculum intent while injected policy/schema text remains powerless. All 26 lesson-ordering tests and 139 generation-job tests pass.
- Exact retest: Pending locally, then on a newly cleared production material after deployment verification.

### AF-036 — A balanced-force clip is mislabeled as a third-law action–reaction pair

- Status: Fixed locally; production retest pending
- Severity: Critical — the reel's generated concept and summary teach a common Newton's-third-law misconception as though it were correct
- Production evidence (material `6d16c8bb-95ae-4804-890c-8d8cf49fc852`, reel `ingest-5313d41c12ec4edd`):
  - The complete released transcript is: “As you sit in your chair right now, the force of gravity is pulling you down towards the center of the earth, but something called the normal force points straight up with the same magnitude, which is why you remain perfectly still.”
  - Gemini persisted concept/family `Newton's third law of motion`, summary/match reason “Identify the action-reaction pair of gravity and normal force while sitting,” and takeaway “Action-Reaction Forces in a Chair.”
  - Gravity and the chair's normal force both act on the seated person. They can balance, but they are not the equal-and-opposite forces on two interacting bodies required by Newton's third law. The selected span never states the claimed action–reaction relationship.
  - The selector nevertheless marked `directly_teaches_topic=true`, `factually_grounded=true`, and used “normal force points straight up with the same magnitude” as third-law evidence.
- Expected: Clip concept metadata, summary, and evidence describe only what the selected speech actually and correctly teaches. Equal magnitude/opposite direction alone cannot establish an action–reaction pair; if adjacent context corrects a misconception, the boundary must include that correction or the standalone clip must be omitted/reclassified as balanced forces.
- Root cause: The selector contract required a literal claim quote but never required that a Newton's-third-law quote establish reciprocal forces between two interacting objects on different bodies. The final Pro audit could repair word boundaries but inherited the selector's concept family, aliases, title, and evidence, so the model's unsupported semantic label survived both validation stages.
- Fix: The selector contract now explicitly distinguishes same-body balanced forces from two-body third-law pairs. The final Pro audit must independently return the corrected title, facet, family, aliases, directness, and constraint evidence, and those fields are validated against the selected speech before replacing the selector metadata. A malformed or semantically invalid audit response retries that audit step once; exhaustion safely retains the grounded selector result rather than partially applying an invalid repair.
- Regression test: A production-shaped chair/gravity/normal-force transcript is reclassified as balanced forces/static equilibrium, while a skater pushing a wall remains a valid two-object third-law pair. Contract tests also cover missing/duplicate audit IDs, bad semantic evidence, first-invalid/second-valid recovery, and twice-invalid exhaustion. The consolidated selector, audit, Gemini, routing, and budget matrix passes all 861 tests.
- Exact retest: Pending on a clean deployed feed against this transcript plus a valid two-object third-law control.

### AF-037 — Continuation releases a clip that fully contains an earlier released clip

- Status: Fixed locally; production retest pending
- Severity: High — scarce subsequent-batch inventory repeats already released teaching instead of covering F=ma, free-body diagrams, problems, or misconceptions
- Production evidence (material `6d16c8bb-95ae-4804-890c-8d8cf49fc852`):
  - Initial generation `0253f774-2d2c-4bca-8821-afdd969209ec` releases reel `ingest-2e6567bea1d94dec` from YouTube source `LQyFshgm-hU` at `38.5–60.4`.
  - Automatic continuation generation `88dc22f1-0677-4a94-a7cb-191ccf40cb96` releases reel `ingest-51075adacb8e452e` from the same source at `38.5–113.3`.
  - The continuation span contains 100% of the earlier span. The cumulative feed reports eight ready reels even though the eighth substantially repeats the earlier first-law example.
- Expected: Duplicate/overlap control applies across the complete released generation chain, including reused deferred inventory, so a continuation cannot append a same-source span that substantially contains an earlier released span.
- Root cause: Organizer overlap filtering is generation-local. The one-clip continuation organizer cannot compare its reused clip with source-generation releases, and authoritative chain assembly concatenates generation release IDs without a cumulative temporal-overlap check.
- Fix: Authoritative root-to-child release assembly now applies the existing 80%-of-shorter same-source overlap rule cumulatively. The first released span wins, watched filtering cannot resurrect its overlapping child, and persisted prerequisite/chain metadata protects genuine dependent lesson steps from deduplication.
- Regression test: The exact `38.5–60.4` source span followed by the `38.5–113.3` child releases only the first; a later non-overlapping same-source clip and an explicitly protected overlapping chain both remain. The complete generation-job file passes 139 tests.
- Exact retest: Pending with the exact nested spans and a non-overlapping same-source continuation control.

### AF-038 — Rejected cursor recovery has no fresh branch after expansion fallback

- Status: Fixed locally; production retest pending
- Severity: High — the code catches the provider failure but still exhausts the curriculum without making a recovery search
- Production evidence (continuation job `49551839-83b4-4641-99a6-abfc51cb5d0c`):
  - The bounded Gemini expansion step is invoked twice, proving its new retry executes.
  - Both attempts fail the focused-query contract and fall back to the single literal long Newton request.
  - Supadata rejects that request's continuation token with `400`; Railway logs “continuing another query branch,” but provider usage contains exactly one Supadata search call and no later branch.
  - The job ends `exhausted` with zero reels. Final visible coverage remains three third-law, four first-law, and one net-force reel, while F=ma, free-body-diagram, worked-problem, and misconception coverage is absent.
- Expected: A recoverable discovery-branch failure causes a bounded step-level retry/recovery even when query expansion produced only the literal fallback. The failed opaque token is never replayed, while permanent request/auth/quota failures remain terminal.
- Root cause: Invalid-cursor handling can advance only to an already-existing independent query. When expansion falls back literally, the queue is empty after isolating that cursor, so “continue another branch” performs no provider call and cannot recover.
- Fix: A rejected opaque cursor may now enqueue bounded fresh, deterministic component queries only when the twice-attempted Gemini expansion fell back literally. Each recovery query is anchored to the literal colon-delimited governing subject, strips only sequencing words, never replays the rejected token, and runs one at a time under the existing provider budget. AI-validated independent branches retain priority; unrelated `400`s still propagate.
- Regression test: The exact long Newton request with literal-only expansion, one consumed result, and one invalid cursor makes a fresh `Newton's laws first-law inertia` request without a token and returns unseen inventory. A companion assertion proves every extracted Newton component remains subject-grounded. All 78 practice-fast retrieval tests pass.
- Exact retest: Pending with a literal-only expansion, rejected cursor, and recoverable fresh inventory on the fallback branch.

### AF-039 — Recoverable clip-pipeline step failures are not retried consistently

- Status: Fixed locally; production retest pending
- Severity: Critical — a single transient provider/preflight/database failure can discard a relevant source, preserve a poorer boundary, or lose already-paid clip analysis
- Code reproductions:
  - The production Pro transcript selector passes `max_retries=0` while the Gemini SDK is also configured for one physical attempt, so a transient transport/408/429/5xx failure drops that source immediately.
  - The final Pro boundary audit also passes `max_retries=0` and fail-opens to the unaudited cut after one recoverable outage.
  - Gemini `countTokens` preflight makes one raw HTTP request; a transient preflight error prevents the selector dispatch.
  - Supadata search/transcript HTTP loops contain correct bounded retries, but each physical attempt consumes the shared operation budget. Normal slow mode starts three transcripts with a budget of three, so one source's documented retry can be blocked before its second request.
  - Per-clip Postgres persistence performs one whole transaction and has no retry for known transient connection/serialization/deadlock errors; the caller then skips the analyzed clip.
- Expected: Every recoverable external step retries itself once or within its existing bounded policy, with cancellation/deadline checks and accurate physical-attempt telemetry. Permanent validation, configuration, authentication, quota, and ordinary 4xx errors do not loop. A logical operation reserves its capacity once even when that same operation performs a physical retry.
- Root cause: Retry behavior was implemented independently per adapter, several active Pro call sites explicitly disabled it, Supadata charged physical attempts against logical-work capacity, and database mutations lacked a shared typed transient classifier plus replay-safe identities. Job `max_attempts=2` is lease recovery, not a substitute for retrying the failed provider or transaction step itself. Gemini retry telemetry also settled a logical reservation to only the successful final response, dropping the unknown billing exposure of earlier physical attempts.
- Fix: The production code now retries the exact recoverable step once (or the adapter's existing bounded poll policy) with cancellation/deadline checks: Gemini CountTokens, Pro selection, partial/whole structured selection, final semantic/boundary audit, lesson ordering, Supadata search/transcript requests and malformed responses, per-clip persistence, generation worker writes, request-entry material/generate/feed transactions, adaptive feedback/progress/scroll/quiz transactions, and terminal status writes. Logical provider capacity is reserved once while every physical attempt is telemetered; unknown earlier Gemini attempts retain their worst-case billing exposure. Database retries use fresh transactions, stable material/chunk/job/usage/feedback/assessment identities, and convergence checks so they do not rerun extraction, embeddings, retrieval, Gemini, quota reservations, organizer work, or committed adaptive mutations. Permanent auth/configuration/validation/integrity failures, ordinary non-retryable 4xx responses, valid empty results, cancellation, and expired deadlines remain single-shot.
- Regression test: Focused tests inject recoverable-first/permanent-first failures, malformed 2xx provider payloads, exhausted structured retries, definite PostgreSQL aborts, and lost commit acknowledgements. They prove recovery without duplicate clips, jobs, chunks, feedback revisions, quiz outcomes, provider usage, quota reservations, or model/retrieval calls. Final local results are `2549 passed, 1 skipped, 40 subtests passed` for the active backend collection and `654 passed` for the eligible pipeline collection.
- Exact retest: Local regression is complete; exact deployed-revision verification, production state clear, and the clean live adaptive-feed matrix remain pending.

### AF-040 — Legacy boundary-curation fixtures omit required concept families

- Status: Fixed locally
- Severity: Release blocker in verification — production concept enforcement is correct, but a separate pipeline collection cannot reach its boundary regressions
- Reproduction: Run `backend/pipeline/tests/test_gemini_segment_curation.py` after concept-family enforcement. Eleven tests fail while converting their shared legacy proposal because it explicitly supplies `concept_family=""`; the boundary logic under test never runs.
- Root cause: The shared `_topic()` fixture predates per-clip concept metadata. `_BoundaryTopic` retains an empty default only for non-live compatibility, but serializing that fixture turns the absent default into an explicitly invalid blank value when a stricter proposal model validates it.
- Fix: The shared test builders now emit a nonblank family derived from each fixture's facet and an explicit alias list; the forty-candidate output-budget fixture supplies its own domain-qualified family. Production validation is unchanged and still rejects missing or blank live concept metadata.
- Exact retest: Focused curation regressions pass, and the full eligible pipeline collection passes (`646 passed`). Two unrelated collections remain excluded because their optional runtime dependencies (`sse_starlette` and `faster_whisper`) are not installed.

### AF-041 — Required concept metadata can exhaust the compact selector output cap

- Status: Fixed locally
- Severity: High — an exhaustive valid selector response can truncate before returning all relevant clips
- Reproduction: Serialize the maximum forty realistic compact boundary candidates with the new required concept-family field. The response is approximately 5,349 tokens; adding the contract's existing 1,024-token hidden-reasoning/safety margin exceeds the old 6,000-token output cap.
- Root cause: The live compact schema gained required per-clip family metadata, but its provider output ceiling remained sized for the older payload.
- Fix: Raise only the compact boundary-selector ceiling from 6,000 to 6,400 tokens. The maximum candidate count, prompt, retry count, selection behavior, and separate Pro/audit ceilings are unchanged.
- Exact retest: The forty-candidate serialization/cap assertion passes, and the consolidated Gemini/selector/provider group passes (`933 passed, 3 subtests passed`).

### AF-042 — Retrying material persistence can publish the uploaded file twice

- Status: Fixed locally
- Severity: Medium — the database converges, but a transient failure after object publication can leave an orphaned duplicate upload
- Reproduction: Submit an idempotency-keyed text file and inject PostgreSQL `40001` on the first `materials` upsert. Storage publication occurs before that upsert; the fresh transaction then calls UUID-keyed `save_bytes()` a second time.
- Root cause: The retry closure kept `source_path` outside the transaction but did not check whether the first attempt had already populated it.
- Fix: Publish only while `source_path` is unset and reuse that exact path on the fresh-transaction retry. The idempotency row lock still fences ownership, and provider extraction/embeddings remain outside the retry.
- Exact retest: The file-upload fault injection passes inside the generation/assessment/material group (`185 passed`), proving two database attempts, one provider pass, one object publication, and one persisted material path.

### AF-043 — A schema-clean retry can erase a valid salvaged clip

- Status: Fixed locally
- Severity: Critical — retrying malformed Gemini output can reduce a relevant nonempty result to no clips
- Reproduction: First Pro selector response contains one malformed candidate and one valid sibling; the retry returns a schema-clean empty `topics` list. The old comparison scores fewer schema errors before valid-topic count, so `(0 errors, 0 clips)` replaces `(1 error, 1 clip)`.
- Root cause: Partial-response retry quality was ordered by schema cleanliness first, even though the retry exists to recover candidates and must never discard already validated inventory.
- Fix: Validate both attempts independently, merge surviving proposals by stable `candidate_id`, let a valid retry repair the same ID, append new retry-only candidates up to the existing forty-item cap, and retain first-attempt candidates omitted by a partial or empty retry. Incompatible request intent is never merged.
- Exact retest: The consolidated Gemini/selector/provider group passes (`933 passed, 3 subtests passed`), including valid recovery, same-ID repair, retry-only append, repeated-malformed exhaustion, and clean-empty retry controls.

### AF-044 — Empty or truncated Gemini success responses are not retried

- Status: Fixed locally
- Severity: High — a recoverable 2xx provider response can discard a source without rerunning the failed selection/audit step
- Reproduction: Gemini returns `finish_reason=MAX_TOKENS` or blank response text while the caller allows one retry. The client immediately raises a typed error after one physical request; only transport exceptions use the retry loop.
- Root cause: Response-contract validation occurred after the transport retry branch and raised directly instead of classifying malformed successful responses as recoverable.
- Fix: Empty and truncated responses now consume the caller's same bounded retry allowance, preserve physical-attempt/error telemetry, and respect cancellation and the shared deadline. Safety, recitation, and blocklist finishes remain permanent and single-shot.
- Exact retest: The consolidated Gemini/selector/provider group passes (`933 passed, 3 subtests passed`), including first-invalid/second-valid recovery, twice-invalid exhaustion, and permanent blocked-output controls.

### AF-045 — Timestamped Supadata cues with no speech text bypass malformed-response retry

- Status: Fixed locally
- Severity: High — a malformed 2xx cue list becomes a terminal no-transcript result instead of retrying the transcript step
- Reproduction: Supadata returns a cue with valid offset/duration but missing or whitespace-only `text`. Contract validation accepts it, normalization drops it, and the source ends as unavailable after one request.
- Root cause: The provider contract validated cue shape and timing but not the spoken-text field.
- Fix: A nonempty cue list now requires normalized nonblank text in every cue and uses the existing malformed-2xx retry path. A genuinely empty `content=[]` remains a valid no-speech result and is not retried.
- Exact retest: The consolidated Gemini/selector/provider group passes (`933 passed, 3 subtests passed`), including blank-cue recovery and the existing valid-empty no-speech control.

### AF-046 — A transient lease-check outage suppresses the database retry it guards

- Status: Fixed locally; production retest pending
- Severity: Critical — the worker abandons a recoverable transaction instead of making its promised fresh-connection attempt
- Reproduction: The worker transaction fails with typed transient PostgreSQL error, then `retry_should_stop` checks lease state through the same unavailable database and raises another transient error. The wrapper propagates the guard failure before attempt two.
- Expected: Local cancellation/stop state aborts immediately; a transient failure to refresh durable stop state must not consume the one bounded transaction retry.
- Root cause: The retry guard is itself an unchecked database call.
- Fix: The transaction wrapper now treats only a typed transient failure from the durable lease check as an unavailable observation and proceeds with its one remaining fresh-connection attempt. Local stop/cancellation and permanent guard failures still abort immediately.
- Exact retest: Generation/assessment/material focused regression group passes (`185 passed`), including transient-guard recovery, permanent-guard failure, and local-stop controls.

### AF-047 — An unrelated adaptation revision can make feedback retry drop this reel's signal

- Status: Fixed locally; production retest pending
- Severity: Critical — “Got it” or “Need help” can return success without applying the requested concept signal
- Reproduction: First feedback transaction rolls back after obtaining its attempted global revision; before retry, a different feedback or quiz mutation increments `feedback_revision`. Retry sees `current_revision >= attempted_revision` and skips this reel write even though the exact payload never committed.
- Root cause: Replay convergence uses a material-global revision rather than the exact reel/learner feedback identity.
- Fix: A retry always replays the exact reel/learner feedback payload through the existing idempotent `record_feedback` path; it no longer infers that this write committed from a material-global revision.
- Exact retest: Generation/assessment/material focused regression group passes (`185 passed`), including an unrelated intervening adaptation revision and exact duplicate replay.

### AF-048 — Gemini retry accounting can exceed the advertised hard job-cost ceiling

- Status: Fixed locally; production retest pending
- Severity: Critical — truthful post-call accounting can report more provider exposure than the admission guard permits
- Reproduction: Admit a roughly $0.90 selector reservation under a $1.00 job cap, then reconcile one earlier unknown physical attempt plus a successful final attempt. Committed exposure becomes roughly $1.80 because retry exposure was added only after dispatch.
- Root cause: One-attempt worst-case cost is admitted before the call, while physical retry exposure is retained only during reconciliation.
- Fix: Retry/failover-capable Gemini operations reserve the full bounded physical-attempt envelope before dispatch while consuming one logical selector slot. Settlement releases unused contingency on a healthy first attempt and retains worst-case exposure for earlier attempts whose token billing is unknown.
- Exact retest: The consolidated Gemini/selector/provider group passes (`933 passed, 3 subtests passed`), including hard-cap admission, retry settlement, and a five-selector healthy cohort that admits without waiting.

### AF-049 — Ambiguous setup commit can fail a generation before provider work begins

- Status: Fixed locally; production retest pending
- Severity: Critical — a lost PostgreSQL commit acknowledgement terminalizes a recoverable job as failed
- Reproduction: The worker setup transaction creates/attaches a generation and commit raises typed `08xxx`. Conservative unknown-commit handling refuses replay; the outer handler then attempts `generation_failed` terminalization regardless of whether setup committed.
- Root cause: Setup lacks a stable preallocated generation identity and convergence lookup, so it is treated unlike replay-safe API entry transactions.
- Fix: Worker setup preallocates one stable generation ID and attaches that same ID idempotently, allowing the setup transaction alone to converge after an ambiguous commit acknowledgement.
- Exact retest: Generation/assessment/material focused regression group passes (`185 passed`), including committed and rolled-back ambiguous-ack controls that assert one generation identity and no provider replay.

### AF-050 — Final audit accepts extra unknown candidate decisions

- Status: Fixed locally; production retest pending
- Severity: High — a response violating the exact-one-decision-per-candidate contract is treated as authoritative
- Reproduction: Audit returns one valid expected ID plus one valid unknown ID. Unknown items are ignored and validation compares only the resolved expected-ID set, so no contract retry occurs.
- Root cause: Contract validation does not require exact item count, exact ID set, and one occurrence of every expected ID simultaneously.
- Fix: Audit acceptance now requires an exact one-to-one candidate-ID set, exact item count, and one decision for every expected candidate. Any extra, missing, or duplicate ID gets one bounded contract retry; exhaustion retains the untouched selector plan.
- Exact retest: The consolidated Gemini/selector/provider group passes (`933 passed, 3 subtests passed`), including extra-ID recovery and twice-invalid exhaustion controls.

### AF-051 — Invalid or overbroad concept families fail only during persistence

- Status: Fixed locally; production retest pending
- Severity: Critical — paid selection/audit work can be discarded and numbered laws can be semantically merged
- Reproduction: Selector/audit accepts a blank-domain family such as `first law`, conflicting aliases such as first-law family plus second-law alias, or a broad `Newton's laws of motion` family for speech about exactly one numbered law. Persistence later rejects some forms, while broad forms can collapse distinct laws.
- Root cause: Gemini-boundary normalization checks nonblank syntax but does not apply persistence-compatible identity safety or same-ordinal cross-field consistency.
- Fix: Selector and audit contracts now validate persistence-compatible, domain-qualified families and aliases before conversion, and require numbered-law ordinals to agree with the candidate's grounded semantic fields. A genuinely multi-law unit may retain a broad family; a single-law unit may not.
- Exact retest: The consolidated Gemini/selector/provider group passes (`933 passed, 3 subtests passed`), including blank-domain, conflicting-ordinal, single-law broad-family, valid atomic-family, valid multi-law, and persistence controls.

### AF-052 — Trusted live Pro selection bypasses request-intent validation

- Status: Fixed locally; production retest pending
- Severity: Critical — schema-valid clips can answer a different or incomplete learning request
- Reproduction: Live Pro returns an `exact_request` different from user input or omits required request constraints. The trusted compact path proceeds directly to audit/report without invoking the existing intent-contract validator.
- Root cause: `_validated_intent_constraints()` is wired for other conversion paths but not the live trusted Pro selector boundary.
- Fix: The trusted live Pro path validates `exact_request`, grounded request coverage, joint/comparison structure, and retry-intent compatibility before audit. An invalid contract gets one bounded retry and fails closed to no candidates after repeated invalid intent.
- Exact retest: The consolidated Gemini/selector/provider group passes (`933 passed, 3 subtests passed`), including wrong-request recovery, incomplete joint intent, incompatible retry intent, and twice-invalid fail-closed controls.

### AF-053 — A factual-correction guard is phrased as Newton-specific production policy

- Status: Fixed locally; production retest pending
- Severity: High — the chair-force regression is covered, but a one-domain prompt patch does not protect analogous semantic-role errors in biology, chemistry, mathematics, law, or other subjects
- Reproduction: Inspect the new selector/audit instructions and fallback token normalization. They explicitly name Newton, force pairs, and `motion`/`newton` tokens instead of expressing the underlying invariant: every defining relation, participant, direction, role, and domain identity must be entailed by the returned speech.
- Root cause: The first correction encoded the observed physics example rather than the domain-independent semantic-entailment rule that the example violated.
- Fix and exact retest: Production prompts now require every defining relation, participant, direction, role, and domain identity to be entailed by the returned speech. Newton remains only a regression example; biology, chemistry, mathematics, software, and law controls pass in the selector contract.

### AF-054 — Numbered-concept isolation stops at fourth

- Status: Fixed locally; production retest pending
- Severity: Critical — selector validation, persistence admission, and adaptive normalization can merge a single higher- or lower-ordinal law into a broad family
- Reproduction: A clip titled `Kepler's fifth law` with family `Kepler's laws` or `Asimov's zeroth law` with family `Asimov's laws of robotics` bypasses the first-through-fourth-only checks. Equivalent fixed maps exist in selection, persistence, adaptation, and ordering. Common variants also fragment: `5th`/`fifth`, `Type II`/`Type 2`, `Phase II`/`Phase 2`, and `law number two`/`second law`.
- Root cause: Ordinal normalization was implemented as a four-entry case table instead of a bounded domain-independent ordinal parser shared in behavior across all three layers.
- Fix and exact retest: One shared strong-grammar ordinal normalizer now handles zeroth, arbitrary numeric/word/compound ordinals, and context-bound cardinal/Roman labels across selector, persistence, adaptation, and ordering. Fifth/zeroth/21st/101st, Type/Phase/World-War forms, rate/count controls, and genuine multi-concept families pass.

### AF-055 — Lesson ordering retries ordinary permanent 4xx responses

- Status: Fixed locally; production retest pending
- Severity: High — a permanent provider rejection consumes a second request and can lengthen a failed generation step without any chance of recovery
- Reproduction: `_ordering_failure_is_retryable()` returns true for status 409 or 418 because it excludes only a short hand-picked 4xx set; ordering telemetry applies the same classification.
- Root cause: The allowlist was expressed as selected permanent codes instead of the universal retry policy: statusless transport failures, 408/429, and 5xx only.
- Fix and exact retest: Ordering now retries only statusless transport failures, 408/429, and 5xx responses. Permanent 409/418 errors stay single-dispatch, transient controls recover within the bounded retry, and the healthy ordering path remains one call.

### AF-056 — Equivalent selector retries become incompatible when Gemini renames constraint IDs

- Status: Fixed locally; production retest pending
- Severity: Critical — a valid retry can repair or add clips for the same request yet be discarded because an arbitrary model-local identifier changed
- Reproduction: First and second selector attempts return the same exact request, constraint kind, grounded source phrase, and requirement, but call the constraint `subject` and `topic` respectively. Both contracts validate independently; their signatures differ only by ID, so retry inventory is not merged.
- Root cause: Retry compatibility treats model-authored constraint IDs as semantic identity even though the prompt requires uniqueness, not a deterministic naming scheme.
- Fix and exact retest: Retry compatibility now compares canonical constraint content, constructs an unambiguous retry-to-first ID map, and remaps evidence before inventory merge. Equivalent renamed retries recover; genuinely changed or ambiguous intent remains rejected.

### AF-057 — Lesson ordering retries permanently blocked Gemini finishes

- Status: Fixed locally; production retest pending
- Severity: High — safety, recitation, or blocklist output consumes a second provider request even though retry is forbidden and cannot make the same step valid
- Reproduction: The ordering adapter records the finish reason but classifies blank blocked output as `GeminiEmptyResponseError`; the general response-contract retry path dispatches again.
- Root cause: The hand-rolled ordering call does not convert permanent blocked finish reasons to `GeminiBlockedResponseError` before empty/truncated-response handling.
- Fix and exact retest: Ordering classifies shared blocked finish reasons before empty/truncated handling. SAFETY, RECITATION, and BLOCKLIST controls remain single-dispatch, while genuinely empty or truncated responses retain one bounded recovery attempt.

### AF-058 — Cursor recovery loses the subject for colonless natural requests

- Status: Fixed locally; production retest pending
- Severity: Critical — fresh recovery branches can search generic facet phrases in the wrong domain
- Reproduction: `Teach cellular respiration, begin with glycolysis, then the Krebs cycle, then oxidative phosphorylation and ATP yield` produces recovery branches such as `the Krebs cycle`, `oxidative phosphorylation`, and `ATP yield` without the `cellular respiration` anchor. Existing controls cover only a `subject: ordered facets` prompt shape.
- Root cause: Subject extraction recognizes only text before a colon; comma-separated and natural sequence phrasing are split directly into independent queries.
- Fix and exact retest: Recovery derives a bounded subject anchor from natural teaching/sequence syntax and carries it into every facet branch without a domain keyword table. Biology, software, law, colon-delimited, and simple-query controls pass.

### AF-059 — Retry-cost reservations serialize healthy selector-to-audit flow

- Status: Fixed locally; production latency retest pending
- Severity: Critical — the new accounting can delay a healthy audit behind unrelated in-flight selectors before any failure occurs, violating the no-streaming-regression requirement
- Reproduction: Three Slow selectors reserve their two-attempt envelopes concurrently. After the first settles at ordinary usage, its audit's two-attempt envelope cannot fit under the $1.50 job cap until peer selectors settle, so the audit waits despite a completely healthy selector and provider.
- Root cause: Reserving every operation's full retry envelope independently at dispatch protects the hard cap but ignores the interleaved selector→audit dependency graph; downstream healthy work competes with contingency held by unrelated upstream calls.
- Fix and exact retest: Provider admission reserves one physical attempt at a time and admits retry/failover attempts only when they are actually needed, retaining unknown exposure while releasing healthy first-attempt capacity. The interleaved three-selector→first-audit test, one-attempt release, retry hard-cap, and accounting suites pass; live timing remains gated against the 115.312-second baseline.

### AF-060 — Concept normalization erases meaning-bearing attached symbols

- Status: Fixed locally; production retest pending
- Severity: Critical — feedback or quiz mastery for one programming language can change the frequency/difficulty of another
- Reproduction: Fresh persistence identities for `C memory management`, `C++ memory management`, and `C# memory management` collide; `A* search algorithm` collides with `A search algorithm`; and charged species such as `Cl−`/`Cl` or `e−`/`e` collide. Selector/order tokenization drops the same meaningful attached symbols.
- Root cause: Concept token regexes preserve letters and apostrophes but discard language-, algorithm-, notation-, and charge-defining `+`, `#`/`♯`, `*`, and terminal `-`/`−` before deterministic persistence and adaptive-family matching.
- Fix and exact retest: The shared semantic tokenizer preserves meaning-bearing attached suffixes, normalizes safe Unicode sharp/minus/star variants, and still treats prose hyphens/punctuation as separators. C/C++/C#/C♯, A*/A, and Cl−/Cl remain distinct and aligned across search, selector, persistence, serving, and ordering.

### AF-061 — Adaptive frontend rerank silently stops after one transient fetch failure

- Status: Fixed locally; production retest pending
- Severity: Critical — accepted thumbs or quiz feedback may not change the visible/subsequent reel inventory even though the signal persisted
- Reproduction: Make the rerank feed request return one transient 503. The callback catches the failure to `null`, performs one request, preserves the stale unseen tail, resolves without a UI error, and never retries. Feedback and completed-quiz flows both depend on this callback.
- Root cause: The adaptive rerank caller has no bounded response-aware retry, while the shared fetch layer does not retry feed requests.
- Fix and exact retest: Adaptive rerank retries the identical request once only for transport, 408, 429, or 5xx failures; aborts and permanent 4xx responses remain single-shot. Frontend controls prove recovery, a two-attempt ceiling, and one healthy request with no added delay.

### AF-062 — Ordinal normalization mistakes rate units such as “per second” for numbered concepts

- Status: Fixed locally; production retest pending
- Severity: Critical — valid scientific concepts can be rejected or split because an ordinary time unit is interpreted as an ordinal identifier
- Reproduction: A valid `radioactive decay law` proposal whose evidence says `probability per second` is flagged as a second-law mismatch. The proximity rule also turns ordinary counts such as `Asimov's 3 laws`, `top 10 principles`, and `systems of 3 equations` into third/tenth numbered concepts.
- Root cause: Numbered-concept detection uses broad token proximity instead of local numbered-label grammar, so rate units and `value → plural kind` counts masquerade as identifiers.
- Fix and exact retest: The shared parser recognizes only explicit ordinals and strong local numbered-label grammar, not rate units, later quantities, or value-before-plural counts. Second Law/law 2/2nd Amendment/Type II plus zeroth, fifth, compound, list, and rate-law controls pass across every consumer.

### AF-063 — Concept normalization collapses operator concepts

- Status: Fixed locally; production retest pending
- Severity: Critical — thumbs/quiz signals for distinct programming or mathematical operators can share one concept identity
- Reproduction: `JavaScript && operator`, `JavaScript || operator`, and `JavaScript ?? operator` normalize identically; so do `C bitwise &`/`C bitwise |` and `Swift String?`/`Swift String`.
- Root cause: Concept lexers preserve selected terminal symbols but discard standalone operator tokens and attached nullability/operator markers before selector validation, deterministic persistence, and adaptive matching.
- Fix and exact retest: One shared tokenizer preserves bounded operator runs and attached notation while query prose still drops sentence punctuation. `&&`/`||`/`??`, `&`/`|`, `String?`/`String`, arithmetic/comparison, and ordinary-punctuation controls remain distinct or equivalent as intended across all layers.

### AF-064 — Search-plan and feed query cache keys erase symbol identity

- Status: Fixed locally; production retest pending
- Severity: Critical — a cached acquisition plan for one concept can be reused for a distinct symbol-bearing concept before clip selection begins
- Reproduction: `A*`/`A`, `C*-algebra`/`C algebra`, and `Cl-`/`Cl` currently produce identical search-plan and reel-service query keys, while equivalent `C♯`/`C#` spellings split into different keys.
- Root cause: Search-plan and feed-query cache normalization use separate alphanumeric-only tokenizers instead of the concept identity contract used downstream.
- Fix and exact retest: Search-plan and feed cache identities now use the shared semantic-token normalizer without rewriting raw provider queries. Meaning-bearing symbols remain distinct and safe Unicode/ASCII notation variants converge in cache-key regressions.

### AF-065 — Unicode star notation collapses into the plain concept

- Status: Fixed locally; production retest pending
- Severity: Critical — mathematically equivalent Unicode notation fragments while distinct plain concepts inherit its feedback/mastery
- Reproduction: `C∗-algebra` and `C⋆-algebra` collapse to `C algebra`, and `A∗ search` collapses to `A search`, even though ASCII `C*`/`A*` remain distinct after the partial symbol fix.
- Root cause: NFKC does not fold U+2217 or U+22C6 to ASCII `*`, and the mirrored tokenizers do not translate them explicitly.
- Fix and exact retest: Common sharp, minus, and star glyphs are translated once in the shared contract. Unicode and ASCII spellings converge, remain distinct from symbol-free concepts, and pass across identity, persistence, cache, and adaptive consumers.

### AF-066 — Roman normalization rejects valid higher numeral symbols after losing case evidence

- Status: Fixed locally; production retest pending
- Severity: High — common named concepts such as `Super Bowl LVIII` and `Chapter XL` fragment from numeric equivalents
- Reproduction: The current Roman parser accepts only I/V/X characters to avoid interpreting ordinary lowercase words such as `div` and `mix`; consequently canonical uppercase `LVIII` and `XL` are never normalized.
- Root cause: Callers case-fold tokens before Roman parsing, discarding the uppercase-form evidence that distinguishes conventional Roman labels from ordinary words and product letters.
- Revised finding: Global uppercase-Roman conversion is unsafe: ordinary acronyms such as `Washington DC`, `CI pipeline`, `IV therapy`, and `MI treatment` look like valid numerals. Case alone cannot prove semantic identity.
- Fix and exact retest: Roman/Arabic conversion now requires strong local numbered-label grammar; ambiguous bare acronyms and numerals remain lexical. Type I/II, World War II, Chapter XL, Type-C/100, Model-X/10, acronym, and mismatch controls pass without alias inference.

### AF-067 — Non-ordinal numeric identities can merge across concepts

- Status: Fixed locally under the canonical-only AI contract; production retest pending
- Severity: Critical — adaptive signals can cross-contaminate named entities such as Apollo missions, software versions, formulas, or route numbers
- Reproduction: Selector validation accepts family `Apollo 11 mission` with alias `Apollo 13 mission`; broad families also erase `Windows 11`, `Formula 1`, and `Highway 101` identifiers.
- Root cause: Family safety compares only ordinal identifiers, so standalone numeric identifiers outside a finite numbered-kind vocabulary are neither retained nor compared.
- Revised finding: Repetition in title/facet does not prove identity: valid worked units such as `x=5`, a derivative at `x=2`, and rolling a `6` repeat the result or input while correctly belonging to a broader concept family.
- Fix and exact retest: Canonical AI labels retain raw numeric runs in their stable identity, aliases are empty, and deterministic code no longer guesses broad/narrow semantic equivalence. Apollo 11/13, Windows 10/11, Python 3.11/3.12, Highway 101/102, Formula 1/2, worked-value, probability, pH, level, and acronym controls pass.

### AF-068 — Semantic operator identity changes with whitespace and drops common math symbols

- Status: Fixed locally; production retest pending
- Severity: Critical — equivalent mathematical requests fragment while distinct operators collapse before search, selection, persistence, ordering, and adaptation
- Reproduction: `x+y` and `x + y` produce different keys; `x-y`, `x/y`, and `x y` can collapse; attached `*` and `->` also vary with spacing; Unicode relations/set operators such as `≥`, `∩`, and `∪` disappear.
- Root cause: The shared lexer still models selected symbols as word suffixes and treats hyphen/slash as unconditional prose separators instead of scanning bounded operators with operand context.
- Fix and exact retest: Operators are tokenized independently of spacing with safe Unicode equivalence and conservative operand context for hyphen/slash. Spaced/unspaced parity, cross-operator distinction, prose compounds, and every downstream consumer pass.

### AF-069 — Inferred numeric lists and lowercase Roman words create false concept IDs

- Status: Fixed locally; production retest pending
- Severity: Critical — worked-example quantities and ordinary words can be persisted as ordinal families, while adaptive signals split or contaminate unrelated concepts
- Reproduction: `solve 1 or 2 equations` and `top 1 and 2 principles` become first/second identifiers; lowercase `chapter mix` becomes Roman 1009; standalone inference also treats `Python 3.12` as identifiers 3 and 12.
- Root cause: Prefix-list and standalone-number heuristics infer semantic identity from surface numbers, and Roman parsing ignores the strong uppercase-form evidence needed before numbered-label grammar.
- Fix and exact retest: Numeric-before-plural and standalone-ID inference were removed; raw numbers remain lexical and Roman conversion requires strong label grammar. Explicit ordinals, kind→number/list, Type II, Chapter XL, World War II, count/value, and numeric distinction controls pass.

### AF-070 — Case folding and compatibility normalization merge scientific identities

- Status: Fixed locally; production retest pending
- Severity: Critical — searches, persisted concepts, and learner signals can cross between chemically or mathematically distinct notation
- Reproduction: `Co`/`CO`, `ℂ`/`C`, `ℝ`/`R`, and `ℤ`/`Z` collapse; `OH•` loses its radical marker; unary `-5` collapses into `5`.
- Root cause: Blanket NFKC plus case folding erases compatibility symbols and structural case, while the semantic operator set omits radical/unary notation.
- Fix and exact retest: Normalization now uses NFC plus explicit safe glyph translations, preserves structural formula/notation case, and retains radical/unary operators while prose remains case-insensitive. Co/CO, ℂ/C, ℝ/R, ℤ/Z, OH•/OH, negative/positive, title-case, cache, family, and adaptation controls pass.

### AF-071 — The production reset is scoped but its surviving state was not made explicit

- Status: Fixed and release-isolation postconditions verified on production
- Severity: Critical to test validity — an apparently “clean” live matrix can inherit learner history, generation inventory, cache entries, or quota usage
- Production reproduction (read-only, 2026-07-20): The linked production topology contains persistent Postgres, Redis, and `reelAI` service volumes plus object storage. Authoritative Postgres still contains the `Asplarity` account and five sessions, as well as 2 materials, 13 reels, learner progress, generation jobs/events/usage, provider/search/LLM/ranked-feed caches, quota reservations, and daily usage. Redis database 0 is empty. `Asplarity` has no billing-subscription row and is therefore not yet the requested Pro test account.
- Root cause: Earlier notes accurately called the operation a *scoped* clear, but neither the preserved-table manifest nor zero-count postcondition was recorded. Authentication is not coming from a mystery Redis store: the API joins the surviving `community_sessions` and `community_accounts` rows in the same Postgres. Subsequent production probes then repopulated adaptive/feed tables.
- Expected fix and retest: After the final SHA is identical on GitHub, Railway, and Vercel, execute one explicit reset manifest that preserves only the required test identity/authentication state, clears every in-scope learning/feed/generation/cache/quota table, empties Redis, assigns `Asplarity` Pro, and proves the intended zero/nonzero counts before creating the first fresh material. Record the manifest and sanitized connection fingerprints so a second database or stale deployment cannot be mistaken for the test target.
- Exact production retest (2026-07-21): GitHub `main`, Railway deployment `53965fc2-e7f8-4e17-820a-6a481f8a444b`, and Vercel Production deployment `5538891448` all identify SHA `f04d91de7f5a0d109e0a3c59f87ae99a44d58e18`; all three deployment checks succeeded and the Railway health endpoint reported one live generation worker with Supadata and Gemini configured. The guarded reset targeted sanitized Postgres fingerprint `8cf0b72d1a0af5d503edab65813c09f7`, after a readable custom-format backup at `/tmp/studyreels-reset-backup.W8fo9t/production-before-adaptive-reset.dump` (SHA-256 `0fe55ae747ef2fd37c124957cc2dc1dd5d3e983dd909db243ee3af118b7cd0c4`). It preserved exactly one verified Asplarity account, eight existing sessions, and one active Production Pro subscription; every enumerated adaptive/material/reel/generation/provider/cache/quota/assessment/feed-snapshot table reached zero rows, including the cascaded request frontier/mining state. Redis `FLUSHDB` returned `OK` and `DBSIZE` returned zero. The separate `/data` volume was deliberately not wiped: production uses Postgres as the authoritative database, while `/data` holds the verification key and potentially unreferenced uploads; erasing it would break preserved identity verification without making any Postgres-backed adaptive record cleaner.

### AF-072 — A syntactically valid concept family can describe a wholly unrelated subject

- Status: Fixed locally under the canonical-only AI contract; live semantic retest pending
- Severity: Critical — feedback and quiz outcomes can be persisted under an unrelated family and then steer future reels toward or away from the wrong topic
- Reproduction: A photosynthesis clip with grounded title/facet/evidence is accepted when Gemini returns the otherwise well-formed family `quantum mechanics`; existing checks validate family shape and ordinal consistency but never require lexical grounding in the clip.
- Root cause: Family validation treats domain qualification as a syntactic property rather than an evidence-backed identity contract, and selector/persistence/serving have separate generic-token rules.
- Fix and exact retest: The independent high-thinking Pro audit is the semantic authority and must return an exact transcript-anchored evidence quote with its canonical family; aliases are prohibited. Shared code requires the evidence to exist but deliberately does not re-decide photosynthesis/quantum or synonym meaning with word overlap. Selector→ingestion→serving contract tests pass; cross-domain Gemini semantic accuracy remains an explicit live matrix item.

### AF-073 — Topic expansion retries permanent client rejections

- Status: Fixed locally; production retest pending
- Severity: High — a permanent 409, 410, 418, or 422 consumes a duplicate Gemini dispatch and delays the search pipeline without a possible recovery
- Reproduction: `_practice_fast_failure_is_retryable()` returns true for every status except a small 400–404 set, so permanent client failures outside that set are retried.
- Root cause: Expansion uses a blacklist of selected permanent statuses instead of the shared bounded policy: statusless transport/local contract failures, 408/429, and 5xx only, with an explicit status taking precedence over stale retryable flags.
- Fix and exact retest: Expansion now uses the universal retry classifier: 409/410/418/422 and blocked/configuration failures are single-dispatch; transport, 408/429, 5xx, and recoverable schema failures get at most one retry. The healthy path is one dispatch with no wait; all 108 retrieval/expansion tests pass.

### AF-074 — Concept-family safety differs across selector, ingestion, and serving

- Status: Fixed locally under the canonical-only AI contract; production retest pending
- Severity: Critical — an alternate or stale authoritative path can persist a family the selector rejects, after which feedback can aggregate through a broad or conflicting identity
- Reproduction: Selector rejects bare plural generics such as `laws`, `concepts`, and `equations`, while ingestion's singular-only generic set and ReelService's ad-hoc stemming accept some of them. A broad family with no identifier can also accept a narrower alias carrying an identifier (`Newton's laws` + `Newton's first law`, `Apollo missions` + `Apollo 11 mission`, `Python typing` + `Python 3 typing`).
- Root cause: The three layers independently implement domain/generic and identifier-consistency checks, and ordinal/numeric conflicts are enforced only when the broad side already exposes an identifier.
- Fix and exact retest: Selector, ingestion, persistence, ReelService, and adaptation now share one canonical identity helper and v2 trust contract; singular/plural generic-only labels reject identically and aliases are always empty. Physics, spaceflight, software, law, and generic matrices pass across all boundaries.

### AF-075 — Compound magnitude ordinals collapse to their trailing unit ordinal

- Status: Fixed locally; production retest pending
- Severity: Critical — distinct named concepts can merge or a correct family can be rejected before persistence
- Reproduction: `One Hundred First Airborne Division` reports identifier `{1}` and can match `First Airborne Division`, while the correct `101st Airborne Division` reports `{101}`; `hundredth` is not recognized.
- Root cause: The ordinal parser recognizes only the terminal `first` in a number-word magnitude phrase instead of either parsing the bounded compound or treating unresolved compounds as opaque.
- Fix and exact retest: The shared parser resolves bounded English magnitude ordinals as one value and fails unresolved phrases closed rather than emitting a trailing unit. 101st/first/hundredth, ordinary count, and rate-unit controls pass and round-trip through stored canonical keys.

### AF-076 — Terminal punctuation has different identity semantics across layers

- Status: Fixed locally by AF-097; production retest pending
- Severity: Critical — a search/cache identity can differ from the persisted/adaptive identity for the same ordinary concept, while notation-bearing concepts may be collapsed accidentally
- Reproduction: Search-plan normalization strips the question mark from `Bayes Theorem?`, while persistence and ReelService can retain it; conversely `Swift String?` requires an explicit notation signal to remain distinct from `Swift String`.
- Root cause: Consumers choose different `preserve_terminal_suffix` behavior without separating ordinary sentence punctuation from an explicitly trusted notation-bearing label.
- Fix and exact retest: Query/prose keys strip sentence punctuation, while the trusted AI canonical-family boundary preserves terminal notation exactly and carries it through persistence, serving, ordering, and adaptation. Ordinary questions/exclamations remain query-equivalent; Swift `String?`, factorial `n!`, and detached TypeScript `!` remain distinct in the AF-097 matrix.

### AF-077 — Broad families can discard protected notation and named identifiers

- Status: Fixed locally under the canonical-only AI contract; live semantic retest pending
- Severity: Critical — mastery for a narrow programming, scientific, or named-entity concept can be stored under and propagated through a broader, non-equivalent family
- Reproduction: Selector validation accepts broad families for clips explicitly about `C++`, `String?`, `Python 3.12`, `Apollo 11`, `HLA Class II`, or `Factor V`; the family can omit the exact notation/version/label even when those qualifiers are grounded in the clip.
- Root cause: The family contract compares only a partial ordinal set and has no shared protected-qualifier signature covering operators/attached notation, strong numbered labels, and pairwise named numeric identifiers.
- Fix and exact retest: Deterministic qualifier/signature inference was removed because it could both erase valid broad concepts and reject worked quantities. Pro must choose the exact canonical family for the audited semantic unit; code preserves that full label, prohibits aliases, and fails closed across differing canonical identities. Cross-domain exact/broad-family and protected-distinction tests pass locally; actual Gemini choices remain a live matrix requirement.

### AF-078 — Public clip shaping compatibility-folds mathematical identity symbols

- Status: Fixed locally; production retest pending
- Severity: Critical — a correctly selected `ℂ`, `ℝ`, or `ℤ` concept becomes ordinary `C`, `R`, or `Z` before persistence and adaptive attribution
- Reproduction: `_public_clips()` normalizes Gemini `facet` and `concept_family` with NFKC, so `ℂ vector space` is emitted as `C vector space` even though the shared identity tokenizer preserves the distinction.
- Root cause: Two public-output normalization lines retained blanket compatibility normalization after the shared tokenizer moved to NFC plus explicit safe glyph translations.
- Fix: Public concept/family shaping now uses NFC, and family/alias sanitization follows the shared identity contract without case-folding structural formulas.
- Exact retest: A selector→public-output regression preserves `ℂ vector space` / `ℂ vector spaces`; the cross-layer operator/case/math matrix passes.

### AF-079 — Search coverage compares equivalent numbered notation literally

- Status: Fixed locally; production retest pending
- Severity: High — recovery queries or bootstrap inventory can be rejected as missing the subject when only ordinal notation differs
- Reproduction: Search coverage treats `Fifth Amendment` differently from `5th Amendment` and `Phase II` differently from `Phase 2`, while persistence already canonicalizes them.
- Root cause: `_search_coverage_tokens()` consumes the generic semantic stream rather than the strong-grammar concept-identifier stream.
- Fix: Coverage now canonicalizes explicit/strong-label ordinals before stopword filtering and stemming, while preserving structural case and operator/set distinctions.
- Exact retest: Fifth/5th and Phase II/2 are equal; `∩`/`∪` and `Co`/`CO` remain distinct; the full search-plan suite passes.

### AF-080 — ReelService legacy fallback drops protected numeric qualifiers

- Status: Fixed locally; production retest pending
- Severity: Critical — even safe persisted metadata can be bypassed by a legacy lexical path that merges `Apollo missions` with `Apollo 11 missions` or `Python typing` with `Python 3 typing`
- Reproduction: After bare-number ordinal inference was removed, `_same_concept_family()` sees the broad token set as a subset of the narrow set and returns true.
- Root cause: The legacy matcher checks only ordinal identifiers before lexical containment and does not consult the shared notation/numeric/case qualifier signature.
- Fix: Legacy fallback now rejects unequal protected signatures before containment. A one-sided ordinal synonym remains possible only when the qualified form adds grounded domain language (for example action–reaction pairs ↔ Newton's third-law action–reaction pairs), while a qualifier-only extension fails closed.
- Exact retest: Apollo/Python broad-narrow pairs reject; existing action–reaction and trusted-profile synonym propagation still pass in the adaptive curriculum suite.

### AF-081 — Full family validation accepts metadata with no clip evidence

- Status: Fixed locally; production retest pending
- Severity: Critical — an alternate authoritative path can stamp any syntactically valid family onto a clip when all grounding fields are absent
- Reproduction: `validate_concept_family_contract("quantum mechanics", [], title="", facet="", objective="", evidence="")` succeeds because lexical grounding is checked only when the aggregate evidence set is nonempty.
- Root cause: Empty evidence is treated as “nothing contradicted the family” instead of failure to prove the identity.
- Fix: The full selector/ingestion contract now fails closed when no grounding evidence exists; the label-only reader remains limited to checking already versioned metadata integrity.
- Exact retest: Empty evidence and photosynthesis/quantum both reject; a grounded canonical family or grounded exact alias succeeds.

### AF-082 — Decimal quantities protect only their trailing digits

- Status: Fixed locally; production retest pending
- Severity: High — a valid boundary repair can be discarded because an ordinary measured quantity is misread as a named concept identifier
- Reproduction: Evidence `one pound-force is about 4.45 newtons` tokenizes the decimal into `4`, `45`; `4` is excluded by the quantity-context word `about`, but `45` is retained and conflicts with the `force units` family.
- Root cause: Quantity-context exclusion is applied independently to each numeric token and is not propagated through the contiguous numeric components of one decimal.
- Fix: Once a contextual numeric component is classified as a value/count, immediately adjacent numeric components inherit that classification. Named forms such as Python 3.12 remain protected because their first component is not context-excluded.
- Exact retest: The exact pounds-force boundary extension succeeds; Python 3.12 broad-family rejection remains green.

### AF-083 — Spoken formula names do not cover symbolic exact aliases

- Status: Fixed locally; production retest pending
- Severity: High — a valid exact alias such as `F=ma` can make a second-law clip fail family validation when the speech says “F equals ma”
- Reproduction: The family carries ordinal 2 and alias signature carries `=`, while title/objective/evidence carry the words `F equals ma`; exact signature equality rejects the audit result and retains an incomplete original boundary.
- Root cause: Protected-symbol validation compares notation literally without using the already grounded non-generic alias terms.
- Fix: A different-category exact alias signature is considered semantically covered only when all of that alias's non-generic terms are grounded in the clip. A broad family still cannot acquire any protected alias, and conflicting same-category identifiers remain rejected.
- Exact retest: Second law ↔ `F=ma` succeeds for both symbolic and spoken equality; Apollo 11/13, first/second law, Co/CO, and C++/C# conflicts remain rejected.

### AF-084 — Canonically duplicate aliases are handled differently by selector and ingestion

- Status: Fixed locally; production retest pending
- Severity: High — metadata accepted by one authoritative boundary can disappear at another, changing the adaptive identity after paid selection
- Reproduction: A family and alias that normalize to the same identity (for example fifth vs 5th spelling of the same complete label) is rejected by selector payload deduplication but accepted by ingestion/shared label validation.
- Root cause: Alias duplicate sets were initialized differently across layers.
- Fix: Selector, shared validation, and ingestion all seed alias deduplication with the canonical family key. Notation spellings already converge through normalization and no longer need a redundant alias.
- Exact retest: Identical/duplicate aliases reject consistently, while a distinct exact descriptive alias with the same protected ordinal succeeds.

### AF-085 — Healthy Pro audits retry when an exact alias is absent from the audit paraphrase

- Status: Fixed locally under the canonical-only AI contract; production latency retest pending
- Severity: Critical — a valid healthy-path audit makes a second paid provider dispatch, directly increasing reel-generation latency
- Reproduction: The selector emits the validated family `Newton's second law of motion` with exact alias `F=ma`; the Pro audit keeps that family and alias but paraphrases its title, facet, objective, and evidence without repeating the symbolic alias. Audit-only family validation rejects the otherwise unchanged identity, marks the whole audit contract invalid, and retries once. Three healthy-path tests observe one extra audit call.
- Root cause: Deterministic code was trying to prove the semantic equivalence of model aliases from lexical overlap. That cannot be reliable across formulas, synonyms, domains, and audit paraphrases, and it added a false retry to the healthy path.
- Fix: The independent high-thinking Pro audit is now the sole semantic authority for one canonical family. `concept_aliases` is a reserved compatibility field that must be empty and is never a mastery edge. Code retains only schema bounds, stable identity normalization, and independently anchored evidence presence; it does not require a synonym or formula to be repeated.
- Exact retest: Passed locally. The exact F=ma → `Newton's second law of motion` audit makes one dispatch, strips aliases, and records no contract retry; the selector equivalent is also one dispatch. Existing recoverable-failure tests retain one bounded retry.

### AF-086 — Selector grounding contaminates changed-family audit repairs

- Status: Fixed locally under the canonical-only AI contract; production retest pending
- Severity: Critical — a correct Pro reclassification can be discarded and retried, retaining the selector's wrong concept and boundary metadata
- Reproduction: A selector misclassifies balanced same-body forces as Newton's third law; the grounded audit correctly changes the family to static equilibrium. AF-085's unconditional selector/audit field union carries the old third-law ordinal into validation and rejects the new family. The same union also carries incidental structural evidence such as `SI` into an unchanged `force units` audit and rejects a valid payoff extension. Conversely, a blanket unchanged-identity shortcut can move the repaired range to a neighboring kinetic-energy lesson while retaining the stale second-law/`F=ma` family, because no audit semantic grounding is checked.
- Root cause: Selector fields were reused as semantic evidence for a later audit decision, so stale model metadata could either veto a correct reclassification or bless a neighboring lesson.
- Fix: The Pro audit must derive and return the canonical family for its repaired semantic unit. Audit validation uses that audited unit's independently anchored exact evidence; selector semantic fields are not unioned into it. No lexical rule re-decides whether a synonym, formula, or broad/narrow label has the same meaning.
- Exact retest: Passed locally in the full 650-test production selector contract, including balanced-forces reclassification, Work–Energy/stale-family handling, force-unit payoff extension, and canonical synonym controls.

### AF-087 — Ordinary acronyms and worked quantities become concept identifiers

- Status: Fixed locally under the canonical-only AI contract; production retest pending
- Severity: Critical — healthy clips across science, math, engineering, and software can fail family validation and trigger paid retries
- Reproduction: `force units` with evidence mentioning `SI`, `photosynthesis` with `ATP`, or `cell respiration` with `NADH` acquires an unrelated structural-token requirement—even when the acronym appears in the title. Likewise, `Newton's second law` with `the force is 10 newtons`, `photosynthesis produces 2 ATP`, `Python lists have 3 elements`, `negligence has 4 elements`, or `Apollo had 11 crewed missions` treats worked values/counts as missing family identifiers. Direct noun-value phrasing such as `Force 10 newtons` is also misread as a named ID. Discourse ordering (`First explain...`, `the second stage...`) becomes a numbered concept, while objective-only notation such as `C++` can be lost. Named identities such as Apollo 11, Python 3.12, Windows 11, Highway 101, Newton's first law, HLA Class II, Factor V, and a second-order rate law must remain protected.
- Root cause: Deterministic validation promoted incidental prose acronyms, quantities, ordinals, operators, and case into required semantic identity, then tried to infer broad/narrow meaning from adjacency rules.
- Fix: Delete prose-derived semantic qualifier comparison. Gemini/Pro chooses the exact canonical family; code normalizes that label without interpreting incidental prose. Numeric version spelling (`Python version 3.12` versus `Python 3.12`) converges mechanically while the numeric run remains in the identity so 3.11 stays distinct.
- Exact retest: Passed locally across SI/ATP/NADH, worked values/counts, discourse ordinals, Python versions, Apollo/Windows/Highway numbers, numbered laws, HLA/Factor labels, formulas/operators, C/C++/C#, Co/CO, and ℂ/C.

### AF-088 — Trusted exact-equivalence profiles depend on family/alias orientation

- Status: Fixed locally by the versioned canonical-only contract; live convergence retest pending
- Severity: Critical — feedback or quiz mastery can fail to reach a semantically identical concept when Gemini reverses the canonical family and exact alias
- Reproduction: One trusted reel stores family `Newton's first law` with alias `law of inertia`; another stores the same exact pair as family `law of inertia` with alias `Newton's first law`. The reader rejects the second profile, so its concept receives no helpful signal and the organizer sees no adaptation.
- Root cause: Alias-based equivalence made mastery identity depend on which synonym Gemini placed in the `family` field and which it placed in `aliases`.
- Fix: Contract `concept_family_v2` has exactly one AI-selected canonical family and no aliases. Selector output shaping, ingestion, persistence, serving, and adaptive profile readers all force or require `concept_aliases == []`; v1 alias profiles are not trusted by the v2 reader. The selector/audit prompt must choose the same formal canonical name across synonym wording.
- Exact retest: Passed locally for prompt/schema enforcement, one-call F=ma canonicalization, v2 persistence/adaptive merging, and protected first/second-law separation. Actual Gemini convergence for input-wording variants remains a required live test.

### AF-089 — Ungrounded aliases can join unrelated adaptive concepts

- Status: Fixed locally by the versioned canonical-only contract; production retest pending
- Severity: Critical — thumbs and quiz mastery can propagate between unrelated topics and directly distort future reel frequency/difficulty
- Reproduction: Empty-signature pairs such as `photosynthesis` ↔ `cellular respiration` or `kinematics` ↔ `dynamics` pass when only the family is grounded. Protected-looking pairs can also escape through incomplete grammar, including hepatitis versus hepatitis C, blood type A versus B, or World wars versus World War II. A generic disjoint-word exception intended for `law of inertia` ↔ Newton's first law also accepts force units ↔ Apollo 11 and photosynthesis ↔ Newton's first law.
- Root cause: The application treated model-proposed synonyms as authoritative graph edges even though lexical rules cannot prove exact equivalence.
- Fix: Alias edges are removed from the adaptive contract. Public selector output, ingestion metadata, selection metadata, and v2 profile reads all force or require an empty alias list, so an unrelated proposed alias cannot join two mastery histories. Only the audited canonical family is persisted and supplied to the orchestrator with the clip's content and learner signals.
- Exact retest: Passed locally: selector/public output and ingestion strip aliases, segment cache rejects nonempty aliases, v2 profile reader ignores a malicious alias row, organizer receives `[]`, and unrelated cross-domain identities remain separate.

### AF-090 — Legacy lexical fallback can still join unrelated mastery histories

- Status: Fixed locally; production adaptive retest pending
- Severity: Critical — when trusted v2 family metadata is absent or rejected, thumbs and quiz mastery can propagate between unrelated concepts.
- Reproduction: The legacy title matcher treats `blood type A` and `blood type B` as one family because uppercase `A` is discarded as an article/noise token and the remaining token sets satisfy subset matching.
- Root cause: `_concept_family_ids()` falls back to lexical broad/narrow similarity when neither concept has an AI-audited canonical family. Token similarity cannot prove semantic equivalence and bypasses the new AI-authority boundary.
- Fix: Without a trusted v2 canonical family, cross-concept matching now fails closed. Exact concept IDs still retain their own history; trusted canonical keys can still connect equivalent concepts, but the lexical heuristic can no longer join feedback or quiz state.
- Exact retest: Passed locally. Blood types A/B and law-of-inertia/Newton-first titles remain self-only without a v2 profile; identical trusted v2 families propagate thumbs/quiz signals; disjoint numbered families remain separate.

### AF-091 — Ranked-feed cache can replay alias-era adaptive ordering

- Status: Fixed locally; production cache retest pending
- Severity: Critical — a deployment can serve a cached ranking computed with v1 alias propagation even after the canonical-only code is live.
- Reproduction: The ranked-feed cache version remains `43`, the same namespace used before `concept_family_v2`, so an otherwise valid cache row is not invalidated by the adaptive identity contract change.
- Root cause: The concept-family contract version changed at persistence/profile boundaries but was not included directly in the ranked-feed cache namespace.
- Fix: Bump the ranked-feed cache version to `44`. This invalidates only old derived rankings; it adds no healthy-path provider call or sleep, and a fresh ranking is cached normally.
- Exact retest: Passed locally for cache-version assertions and stale-version rejection; the healthy v44 cache behavior remains covered by the ranked-feed cache suite.

### AF-092 — Public Gemini evidence is read from the wrong persistence field

- Status: Fixed locally; production selector-to-ingestion retest pending
- Severity: Critical — every otherwise valid AI family can be discarded before persistence, leaving the orchestrator and adaptive scheduler without canonical concept metadata.
- Reproduction: Public selector clips carry their grounded claim as `topic_evidence_quote` (and diagnostic `model_claim_quote`), while `_concept_family_search_context()` reads only `claim_quote`; a valid authoritative Work–Energy clip therefore returns an empty family context.
- Root cause: The family contract was wired to the internal audit proposal field name rather than the public clip field produced by `_plan_to_report()`.
- Fix: Persistence accepts the internal `claim_quote` when present and otherwise uses the public grounded `topic_evidence_quote`. It requires the audited evidence to exist and does not use diagnostic or selector-only prose; semantic family meaning remains the Pro audit's responsibility.
- Exact retest: Passed locally through `_public_clips()` → `_concept_family_search_context()` → authoritative ingestion → persisted v2 context; the generation-job organizer test receives the canonical family, empty aliases, full clip content, and learner signal.

### AF-093 — Untrusted title ordinals override an exact AI family match

- Status: Fixed locally; production adaptive retest pending
- Severity: Critical — equivalent broad concepts can split into separate mastery histories even when both carry the same trusted v2 canonical family.
- Reproduction: Titles `Newton's first and second laws` and `Newton's laws comparison`, both profiled as `Newton's laws of motion`, fail to connect because title ordinal checks run before profile intersection; the same occurs for World War I/II comparison versus World Wars overview.
- Root cause: Legacy title heuristics have higher precedence than two trusted canonical AI identities.
- Fix: When both concepts have v2 profiles, exact canonical profile intersection is authoritative and evaluated first. Disjoint profiles remain separate; title-based migration checks are limited to the one-profile compatibility case.
- Exact retest: Passed locally for Newton multi-law and World Wars broad-family profiles, plus disjoint first/second-law and Apollo 11/13 identities.

### AF-094 — Persistence uses a different canonical key than adaptation

- Status: Fixed locally; production persistence retest pending
- Severity: Critical — equivalent AI canonical labels can create separate concepts, splitting frequency, feedback, and quiz mastery before the adaptive reader runs.
- Reproduction: Shared identity normalizes `Python version 3.12 typing` and `Python 3.12 typing` to `python 3 12 typing`, but persistence previously produced `python version third twelfth typing` for the first spelling and therefore a different deterministic concept UUID.
- Root cause: `normalize_clip_concept_family()` duplicated an older ordinal normalizer instead of using the shared canonical identity function.
- Fix: Persistence now delegates to `concept_family_identity_key()` while preserving the readable title and existing UUID namespace. Obsolete ingestion ordinal helpers created by the superseded heuristic were removed.
- Exact retest: Passed locally: both 3.12 spellings share one deterministic UUID, 3.11 is distinct, generation-job recognition remains aligned, and the adaptive profile matrix passes.

### AF-095 — Canonical ordinal keys are not idempotent

- Status: Fixed locally; production persistence retest pending
- Severity: Critical — protected numbered concepts can lose their ordinal when a stored canonical key is read again, weakening separation of feedback and quiz histories.
- Reproduction: `Twenty-first Amendment` and `21st Amendment` normalize to `ordinal_21 amendment`, but normalizing that stored key again yields `ordinal 21 amendment`; profile ordinal extraction then misses 21. The same occurs for 101st and other non-simple ordinals.
- Root cause: The generic raw-label tokenizer was incorrectly reused for values that were already canonical profile keys, so it split the internal `ordinal_<n>` sentinel at its underscore.
- Fix: Adaptive profile readers parse already-canonical whitespace-delimited keys directly, while raw/model labels continue through the normal tokenizer. This preserves 21st/101st without reserving or rewriting a literal software identifier such as `ordinal_21`.
- Exact retest: Passed locally for 21st/101st profile extraction, literal `ordinal_21` separation, persistence/read parity, and distinct numbered-family profiles.

### AF-096 — Possessives break explicit curriculum fallback matching

- Status: Fixed locally; production lesson-order retest pending
- Severity: High — when both organizer attempts are invalid, an explicit request such as `Newton's first law` may not match the canonical family and the fallback can start at a later concept.
- Reproduction: `_sequence_tokens("Newton's first law")` produced `newton' first law`, while the non-possessive formal spelling produced `newton first law`.
- Root cause: Fallback sequence tokenization applied plural stemming before the shared possessive normalization used by concept identities.
- Fix: Strip straight or normalized curly possessive suffixes before plural stemming. Canonical families remain alias-free; the progression fixture uses request-grounded viewer-facing facets alongside the formal canonical family.
- Exact retest: Passed locally: possessive and ordinal variants normalize, the twice-invalid organizer fallback follows the requested concept progression, same-source chronology remains intact, and the organizer payload contains canonical family plus learner signal with aliases empty.

### AF-097 — Terminal concept notation is erased or rejected

- Status: Fixed locally; production retest pending
- Severity: Critical — valid programming and mathematical concepts can merge into one mastery history, while healthy Gemini audits can be rejected and retried
- Reproduction: `Swift String?` normalizes to the same canonical key as `Swift String` and fails label validation as `family_ambiguous_terminal_suffix`. `factorial n!` normalizes to the same key as `factorial n`; a detached terminal `!` in `TypeScript non-null assertion !` is also discarded.
- Root cause: The shared lexical layer always strips a terminal `!`, conditionally strips terminal `?`, and the family validator rejects the one preservation mode instead of retaining the AI-selected notation. That converts safe structural identity into a prose-punctuation guess.
- Fix: Query/prose keys continue ignoring sentence punctuation, while canonical concept-family identity explicitly preserves attached or detached terminal notation. The AI-selected label therefore fails closed into a distinct identity instead of being merged or rejected. No language- or topic-specific synonym rule was added.
- Exact retest: Passed locally. Swift `String?`/`String`, factorial `n!`/`n`, and detached TypeScript `!` remain distinct through shared identity, deterministic concept UUID persistence, adaptive profiles, and the 652-test production selector contract. The focused identity/adaptive/persistence/ordering matrix passed 94 tests plus 3 subtests.

### AF-098 — Live retrieval cache contract test retained the previous version

- Status: Fixed locally; production cache retest pending
- Severity: Medium — the complete backend release gate fails even though the live retrieval implementation correctly invalidates the older expansion prompt cache
- Reproduction: `test_practice_fast_expansion_requests_focused_sources` expects expansion cache version `7`, while the production retrieval function uses version `8` after its prompt/intent contract changed.
- Root cause: The test expectation was not advanced with the production cache namespace. Despite the historical `practice_fast` name, `ingestion/pipeline.py` enables this exact retrieval path for generated reels, so this is part of the in-scope input-to-clips pipeline rather than the excluded practice feature.
- Fix: Update only the stale assertion to the production version; do not change cache behavior, practice UI, or unrelated practice code.
- Exact retest: Passed locally: the production retrieval/expansion group passed all 108 tests with the version-8 assertion, and the complete backend suite passed 4,038 tests plus 40 subtests with 1 skip.

### AF-099 — Fresh cross-domain production searches terminate without persisted reels

- Status: Root causes confirmed; universal fixes implemented locally; exact-SHA production retest pending
- Severity: Critical — a clean signed-in Pro search can finish without showing any educational clip, preventing relevance, boundary, progression, quiz, and feedback adaptation from functioning
- Production reproduction (exact deployed SHA `9af1d72ec4f63307571317b5c8c42a57fa9fe53f`, clean Postgres/Redis): Physics material `89b3deef…` and initial law jobs emitted terminal empty batches despite successful YouTube discovery, transcript retrieval, and first Gemini Pro selector calls. Biology `6f3a9669…`, math `407be62e…`, and software `db9cdd14…` exhausted after roughly 8–10 seconds with the UI's `No matching YouTube videos` message. Direct Supadata probes disproved source scarcity: the focused law query returned 20 videos while the conversational literal returned only one.
- Root cause A — retrieval: Gemini Flash-Lite returned valid concise three-query plans twice with HTTP 200, but `_validated_ai_queries()` rejected them. Biology's focused branches omitted only a format constraint already retained by its broad branch; math omitted discourse words such as `smooth`, `progression`, `start`, and `then`; software omitted `teach` and `then`. The deterministic whole-prompt token-union and complete-N-slot facet-cover gates therefore discarded good AI retrieval plans and sent only the long conversational sentence to Supadata.
- Root cause B — clipping retries: Slow jobs admit three logical source selectors. Three sources selected concurrently and each first Pro call succeeded, but schema/intent-contract retry calls were incorrectly counted as new logical source selectors. With all three source slots already claimed, every retry was denied before dispatch as `ProviderBudgetExceededError`; the retained invalid/empty plans had no candidate for the required Pro audit. The quota represented physical retry attempts as new sources instead of charging only their separately bounded cost.
- Root cause C — representation continuity: The expansion output already contained a corrected intent, but ingestion sent the original conversational literal back into Gemini Pro. Retrieval and clipping therefore did not share the same compact AI interpretation, forcing the selector to re-parse filler and sequencing language independently.
- Universal fix: Expansion cache v11 makes the existing Flash-Lite call return one <=220-character standalone, intent-preserving learning summary and searches from that summary; no additional model call is added. Validation retains anchored source constraints, known IDs, and all subject IDs, but no longer uses local token heuristics to veto otherwise valid subject-grounded AI branches or synthesize the long literal when focused branches exist. The original request remains persisted as `topic_terms`/`_literal_topic`, while the same AI summary is now passed into Gemini Pro. Selector and audit schema/contract retries reuse their source's logical quota and still acquire a separately cost-bounded physical dispatch ticket.
- Focused retest: The captured biology/math/software production plans now dispatch three short Supadata queries after one Gemini call, preserve the original literal for traceability, and pass 142 expansion/search/provider regressions. A live provider smoke returned the cellular-respiration summary and three focused queries on its first call. The summary-to-Pro handoff test passes. The independent changed-path regression gate passes 928 tests, including selector/audit retry accounting and production-shaped diagnostics.
- Exact production retest: Pending the same five-domain clean-state sequence, complete regressions, identical-SHA deployment verification, and time-to-first-reel comparison.

### AF-100 — Non-Gemini correction was mistaken for the compressed intent

- Status: Fixed locally; exact-SHA production retest pending
- Severity: High — a deterministic or legacy discovery result can narrow a short ambiguous learning request before Gemini Pro, producing clips for the surrounding subject instead of the requested leaf concept
- Reproduction: The complete backend gate passed discovery topic `Atp in cellular respiration mitochondria`, but a deterministic `corrected` value of `cellular respiration` was sent to clipping. Both material-generation variants failed because the ATP leaf identity disappeared.
- Root cause: The first AF-099 summary handoff treated every discovery `corrected` field as the new Gemini intent summary. Older deterministic/fallback paths use that field for retrieval normalization and do not guarantee an intent-preserving compression contract.
- Universal fix: Trust `corrected` as the Gemini Pro selection intent only when discovery records `provider_used == gemini`; deterministic and literal-fallback paths retain the authoritative topic. The original request remains `_literal_topic` in all paths.
- Exact retest: The focused Gemini-summary and non-Gemini-preservation tests pass; the original acronym/sibling-context regressions and complete backend gate are rerun below.

### AF-101 — Compressed summary was not structurally bound to its anchored intent

- Status: Fixed locally; exact-SHA production retest pending
- Severity: Critical — a malformed but schema-valid expansion could return grounded subject constraints and search queries while placing an unrelated concept in `corrected`, after which Gemini Pro would clip against the unrelated summary
- Reproduction: A chain-rule expansion can declare anchored chain-rule constraints/queries while returning `corrected = quotient rule`; the current validator accepts the search plan because it validates constraints and query IDs but never binds the downstream summary to those IDs.
- Root cause: The initial compressed-summary schema described `corrected` as intent-preserving but did not require an explicit structural coverage claim for that field. Query coverage IDs therefore protected retrieval branches without protecting the separate summary handed to clipping.
- Universal fix: The same expansion response must return the exact set of anchored intent-constraint IDs preserved by its concise summary. Deterministic validation checks only exact known-ID set equality; it does not re-decide semantics with lexical matching and adds no provider call. A missing, duplicate, incomplete, or unknown summary binding rejects that response and uses the existing bounded retry.
- Exact retest: Passed locally: incomplete, unknown, and duplicate summary-constraint bindings are rejected; the complete expansion/retrieval group passes 115 tests and the independent changed-path gate passes 928 tests. The final real-provider matrix accepted all five domain summaries and focused query plans on their first call.

### AF-102 — Real Gemini expansion can lengthen the request and exhaust its retry

- Status: Fixed locally; exact-SHA production retest pending
- Severity: Critical — after both bounded expansion attempts fail their response contract, retrieval falls back to the long conversational literal that produced zero biology, math, and software results in AF-099
- Reproduction: The first real v10 Newton provider response exceeded the schema's 240-character `corrected` limit. The actual two-attempt production entry point then returned `provider_used=literal_fallback` with the original 211-character request as its only query. Later direct samples landed at 240 and 236 characters, showing that schema-only length metadata did not reliably make Gemini compress the request.
- Root cause: The system prompt required a concise summary but stated no concrete length boundary in natural language. Gemini could expand the request up to or beyond the schema edge; the retry repeated the same underspecified instruction.
- Universal fix: The prompt now requires a compact summary of at most 200 characters without dropping any anchored constraint, while the response schema allows a 220-character safety margin. The cache advances to v11. The existing two-attempt retry remains unchanged, so the healthy path still makes one Flash-Lite call and adds no sleep or network round trip.
- Exact retest: Passed locally and against the real provider. The expansion/retrieval group passes 115 tests. Physics, biology, math, software, and law each returned `provider_used=gemini` on the first call, with compact accepted summaries of 197, 175, 162, 168, and 202 characters respectively and three focused queries each; none used the literal fallback.

### AF-103 — Fresh curriculum collapses onto the first requested facet

- Status: Fixed locally; exact clean-production retest pending
- Severity: Critical — the app now returns reels, but repeated continuation batches can spend the entire visible curriculum on one early concept instead of progressing through the user's requested lesson
- Production reproduction (exact SHA `7d0402c903203f6cc34de388d0070a20f4d63c63`, clean Postgres/Redis): Fresh beginner physics material `3b465ad4-9d68-4c5c-8b53-d50c7fc00aa4` requested first-law inertia/balanced forces, then F=ma, free-body diagrams, third-law action-reaction, worked problems, and misconceptions. Four final events released nine unique reels. Eight are attributed to Newton's first law or inertia and one to Newton's second law; none covers free-body diagrams, the third law, or a worked problem. The initial three are relevant and context-complete, but subsequent batches repeat the same first facet rather than advancing.
- Expected: After enough inventory for nine reels, the organizer and acquisition loop should cover distinct requested concepts in the stated prerequisite order before adding same-family repetition, unless no qualifying source exists. Direct YouTube/Supadata evidence already establishes that source scarcity is not an acceptable explanation.
- Root cause: Two independent upstream gates removed the missing facets before the organizer. First, the three-source analysis prefix prioritized channel diversity before unseen Gemini query-family coverage; the initial prefix therefore repeated overview/inertia branches while the first free-body-diagram source sat at rank four. Second, Gemini Pro returned request constraints for otherwise valid sources, but `_validated_intent_constraints` rejected the entire selector result unless the stemmed union of those constraint labels repeated every token in the compact summary. That lexical check treated discourse/level words such as `explain`, `beginner`, `start`, `move`, and `ending` as semantic omissions, triggered a redundant retry, and could cache zero clips before the independent Pro audit saw them. The organizer received no free-body-diagram, third-law, or worked-problem clip and did not omit one.
- Universal fix: For an all-AI retrieval pool, keep the strongest source and fill scarce analysis slots with an unseen query family before channel-only repetition; literal-anchor flows retain their prior ordering. Remove only the whole-summary lexical token-union veto. Exact-request equality, unique and grounded constraint phrases, relationship/comparison/path structure, concept-family validation, transcript-grounded evidence and edges, and the final independent Gemini Pro audit remain unchanged. The same compact Gemini summary continues to drive both search expansion and Pro selection.
- Focused regression: The exact production-shaped rank pool now moves a free-body-diagram family into the first three analysis slots. Contract tests cover physics, biology, math, software, and law summaries whose AI constraints preserve every teaching facet without mechanically repeating sequencing words; exact-request mismatch and joint-relationship negative controls still reject. Focused retrieval/contract tests pass locally.
- Exact retest: Pending full regression gates, identical-SHA deployment, clean Postgres/Redis, the exact physics reproduction, the five-domain matrix, and adaptive thumb/quiz sequence.

### AF-104 — Failed or unattempted sources are permanently consumed

- Status: Fixed locally; exact clean-production retry retest pending
- Severity: Critical — a transient transcript/Pro/deadline failure can permanently remove the strongest source for a missing curriculum facet from every continuation
- Production reproduction: The physics chain persisted `kKKM8Y-u7ds`, `aeqh3bFvmXU`, and `MaabUHLIIXA` as consumed even though their Pro step failed or yielded no processed result; `ruBfXIVSYZ8` was also consumed despite never reaching Pro and having no transcript artifact. Later continuation search excluded all four.
- Root cause: Generation usage serialized every `retrieved_video_ids` entry as `consumed_video_ids` before distinguishing completed source analysis from discovered-but-unattempted, timed-out, or provider-failed work. Completion was also recorded before clip persistence, and a twice-invalid Gemini selector contract returned `error=None`, so both persistence failures and operational selector failures were misclassified as terminal successes. Failed sources had no durable attempt count, allowing a deterministic failure to occupy a scarce source slot forever if consumption was simply removed.
- Universal fix: Track discovered, actually scheduled, and terminally completed sources separately. A source becomes attempted only after executor submission succeeds. A valid semantic zero-clip response completes immediately; a clip-bearing response completes only after persistence returns. Provider, timeout, selector-contract, cancellation, and persistence failures remain retryable. Each terminal job stores exact failed-source attempts; a source receives one later acquisition retry and is retired after two failed attempts so it cannot monopolize future batches. Invalid Gemini contract responses retry inside the selector; a failed transport retry or a second invalid response now surfaces an operational error, while a genuinely valid semantic zero remains successful.
- Focused retest: Deep and bootstrap scheduled/unattempted, valid-empty, success, timeout, provider-failure, and persistence-failure states pass. The integrated continuation chain retries the same failed source once, persists exact attempt counts, then excludes it on the third batch while completed sources remain consumed. The full selector-contract suite passes 660 tests, and the material/service accounting suites pass 135 tests plus four subtests.
- Exact retest: Pending the clean five-domain production run with a forced/recovered provider failure and durable continuation inspection.

### AF-105 — Organizer sees a pre-truncated page, not the candidate reservoir

- Status: Fixed locally; exact clean-production progression/adaptation retest pending
- Severity: High — feedback-aware selection and curriculum progression cannot choose a better three from available alternatives, and progression can reset across continuation pages
- Production reproduction: Concept positions released as `3,3,4`, then `3,3,3`, then `3,7,3`; after F=ma, later pages regressed to first-law/inertia. Organizer input equaled each already-reduced three-reel page, so it had no access to lower-ranked diverse or mastery-aware candidates.
- Root cause: `_generation_job_reels` applied the request limit before `order_lesson_batch`; ordering was persisted per generation/page rather than over a bounded larger reservoir with prior-chain coverage. The cached-sibling shortcut could therefore release any unseen sibling before fresh acquisition, regardless of whether its trusted concept family had already saturated the lesson.
- Universal fix: The original two-times reservoir widened the page but still excluded valid persisted candidates. AF-110 supersedes that intermediate limit: the organizer now receives every current-contract, boundary-verified, unseen candidate persisted by the unchanged bounded source pass, plus trusted Gemini concept metadata, current thumb/quiz learner signal, and authoritative prior-release concept coverage. Prior coverage groups only by trusted canonical family or exact concept ID; delivery count is explicitly not treated as mastery. Acquisition count, source count, Gemini calls, Groq work, and acoustic verification remain at their original requested-shortfall caps.
- Focused retest: A continuation with three prior releases exposes six private verified candidates to one organizer call, performs zero retrieval calls, supplies exact prior coverage, and can release five of the six. AF-110 additionally proves that all 120 candidates from a three-source slow pass, plus the maximum eight unseen parent-chain candidates that can trigger that pass, reach the same organizer call. The same organizer decisions survive raw final events, sanitized replay, and later job polling. Feedback-aware selection can choose lower-ranked confusing concepts in one healthy Gemini call; dynamic cardinalities, over-limit retry, dependency-safe fallback, and fixed prompt-budget guards all pass.
- Exact retest: Pending clean production progression plus thumbs-up/down and correct/wrong quiz continuations.

### AF-106 — Frontend fetch size was also the organizer's maximum output size

- Status: Fixed locally; production cardinality/replay retest pending
- Severity: High — a normal continuation requested three reels, so Gemini could never include four or more already-verified clips even when that was the best coherent adaptive lesson; after lifting that cap, terminal polling still silently truncated the persisted decision back to three.
- Reproduction: Supply six verified continuation candidates with `num_reels=3`. The organizer receives `release_limit=3`. When a test organizer selects five, the raw worker final contains five but `_generation_job_status_payload()` and sanitized replay return only the first three.
- Root cause: `requested_count` served two unrelated roles: provider acquisition/stream-fetch sizing and editorial subset cardinality. `_generation_job_reels()` independently sliced every persisted lesson order back to the original request count.
- Universal fix: Keep the request count only for acquisition sizing. Set the organizer's safe maximum to the number of verified candidates actually supplied, and treat the persisted organizer order as the authoritative release subset on all read/replay paths. This permits any non-empty subset without allowing unknown clips or adding provider work.
- Exact retest: Local integration releases five of six candidates from a three-reel acquisition request and returns the identical five through final replay and polling. Dynamic 1/4/6 organizer-cardinality tests pass. Production replay/poll parity remains pending.

### AF-107 — A late invalid selector contract was consumed as a semantic no-match

- Status: Fixed locally; exact clean-production retry retest pending
- Severity: Critical — when less than five seconds remained in the shared source deadline, an invalid Gemini Pro request contract could not enter its bounded repair branch and silently returned `error=None`; ingestion then marked the source terminally analyzed and excluded it from later acquisition.
- Reproduction: Return a Pro plan whose `exact_request` mismatches `photosynthesis`, with no safe topics, and give the selector 4.9 seconds of remaining deadline. Before the fix it made one call and returned zero clips with `classification=invalid`, `intent_contract_request_mismatch`, and no operational error.
- Root cause: Operational failure conversion existed only inside the branch that had at least five seconds available for a second selector call. The no-retry branch appended telemetry and fell through to ordinary empty-report conversion even though its empty result came from a rejected contract rather than a valid semantic decision.
- Universal fix: If a Pro contract is rejected, leaves no safe topic, and the bounded repair cannot fit before the deadline, emit a retryable `GeminiSelectorContractError` with explicit insufficient-deadline telemetry. Valid semantic zero-topic responses and partially salvageable contract responses retain their existing behavior; no healthy-path call or wait was added.
- Focused retest: The exact 4.9-second reproduction now makes one call, returns an operational error, and remains eligible for the durable later-source retry. Failed-retry, twice-invalid, valid-empty, and ordinary healthy controls pass (4 focused tests).

### AF-108 — Failed and exhausted batches strand durable source retries

- Status: Fixed locally; exact clean-production recovery retest pending
- Severity: Critical — AF-104 could record a source failure correctly but never use that record when every source in a batch failed, so the same strongest source could be retried in unrelated fresh generations without reaching its retirement bound
- Reproduction: A job with one scheduled source and `failed_source_attempts = {video: 1}` terminalizes as `failed` when no other source makes progress. The generation endpoint rejects that job as a continuation, the feed's compatible-job lookup omits it, and a same-request reload creates an unlinked generation. An `exhausted` job is accepted as a continuation token but immediately returns terminal empty even when it contains the first of two allowed source attempts. The partial-job continuation regression therefore passed while the all-source-failure state machine remained broken.
- Root cause: Durable attempt accounting was attached to generation ancestry, but only `completed`, `partial`, and `exhausted` jobs could establish that ancestry; failed jobs were invisible to compatible recovery, and exhausted handling ran before checking whether an unconsumed failed source still had its one bounded retry.
- Universal fix: Recovery now derives exact retryable source IDs from the whole generation chain: the source must be failed, unconsumed, and below the two-attempt bound. Failed batches with an open provider cursor may also create the next linked acquisition so retired sources are excluded and fresh candidates can be tried. Explicit continuations, same-request generation retries, and feed reload/autofill all attach the new generation to that terminal source chain. A failed chain with closed-cursor source attempts already at the bound returns `source_retry_exhausted` instead of silently resetting its counts. A normal semantic-zero result is consumed; therefore it has no retryable failed ID, and an exhausted semantic-zero batch remains terminal rather than looping.
- Focused retest: An implicit failed-job retry inherits the first generation, accumulates the same source to exactly two failures, and refuses a third unlinked retry. An exhausted job with one failed source accepts one explicit continuation; when that retry returns a valid semantic zero, the source becomes consumed, the next continuation returns the existing exhausted result, and the job count stays fixed. A feed reload after an all-source failed batch queues one linked retry. The complete generation-job suite passes 164 tests.
- Exact retest: Pending an exact-SHA production run that forces one selector/source failure, verifies the linked recovery job and durable attempt count, then confirms either recovered clips or bounded retirement without a semantic-zero loop.

### AF-109 — Candidate-schema exhaustion was also cached as a semantic no-match

- Status: Fixed locally; exact clean-production retry retest pending
- Severity: Critical — if Gemini returned candidates but every candidate failed the clip schema, the selector could return `error=None` and ingestion would permanently consume that source even though Gemini had never made a valid semantic decision that no relevant clip existed.
- Reproduction: Return a request-valid Pro plan with no retained topics and `schema_rejection_reasons=["candidate_1:invalid_claim_quote"]`. With 4.9 seconds remaining, the selector made one call and returned a successful empty result; with enough time, two identically rejected responses still returned a successful empty result.
- Root cause: The operational-empty guards added for AF-107 recognized only request/intent contract rejection. Candidate-schema rejection used the same bounded repair path but was omitted both when the deadline could not fit that repair and when the second response was also wholly rejected.
- Universal fix: Treat either schema or contract rejection as a retryable operational failure when it leaves zero safe topics and the single bounded repair is unavailable or exhausted. Preserve distinct `GeminiSelectorSchemaError` and `GeminiSelectorContractError` telemetry. A valid semantic zero-topic response remains successful, and a response with any safe retained topic remains salvageable; the healthy path gains no provider call or delay.
- Focused retest: Short-deadline schema rejection now makes one call and returns a retryable schema error; two rejected responses make exactly two calls and return the same operational error. The valid semantic-empty control remains a one-call success. Contract short-deadline, failed-retry, and twice-invalid controls also pass (6 focused tests).

### AF-110 — The organizer still sees only two request-sized pages

- Status: Fixed locally; exact clean-production cardinality/latency retest pending
- Severity: Critical — valid persisted clips outside a six-item window can never be considered for progression or feedback adaptation, even though no additional retrieval or clipping work is needed
- Reproduction: Submit a fresh slow job with `num_reels=3`. The unchanged three-source pass can persist up to 40 current-contract clips per source, but `_generation_job_reels()` supplies only `2 × num_reels = 6` to the lesson organizer. A seventh or later boundary-verified unseen clip is therefore omitted before Gemini can compare its concept and learner signal. Simply raising that read to 120 with the full object schema creates a second defect: a max-field prompt measured 427,425 characters (roughly 107k tokens), which risks violating the no-streaming-latency requirement.
- Root cause: The first organizer fix coupled editorial visibility to an arbitrary request multiplier rather than the already-paid selector/source contract. A cross-request top-up added a second bound error: up to eight unseen parent reels can coexist with the 120 clips persisted by a fresh slow pass, while reusable-inventory counting reapplied the parent's stored editorial subset and could mistake 120 raw eligible clips with eight previously selected clips for an eight-clip reservoir. Ranked-feed response shaping also discarded selector candidate, chain, position, and prerequisite metadata before the organizer, and the next request finalizer stripped every remaining `_selection_*` field. The lesson prompt repeated long JSON keys and raw internal IDs for every clip, had no total input budget, and retained a 1,024-token response ceiling designed for tiny batches. Existing ranked-cache rows and `adaptive_clip_concepts_v2` completed jobs could preserve those old semantics after deployment.
- Universal fix: Derive the private organizer read ceiling from existing hard work limits plus the only parent headroom that can trigger acquisition. Gemini can persist at most 40 clips per analyzed source, with two fast or three slow sources; raw verified source-chain inventory is now counted independently of any stored editorial subset up to the nine-reel startup target, so a fresh pass runs only when at most eight parent candidates remain. The resulting organizer limits are 88 fast and 128 slow under the organizer's 200-ID schema. Every current-contract, boundary-verified, unseen parent/fresh candidate reaches one organizer call; `num_reels`, `max_new_reels`, source budgets, YouTube/Supadata work, Gemini selector calls, Groq work, acoustic verification, retries, and the ten-second organizer timeout are unchanged. Selector candidate, chain, finite nonnegative position, prerequisite, family, and quality fields survive only through an explicit internal organizer allowlist; normal final/feed payloads still expose no private selector fields. Ranked-feed cache version 45 invalidates rows missing those fields, and request schema `adaptive_clip_concepts_v3` prevents reuse of pre-fix completed jobs. Small batches keep their full object payload byte-for-byte. Only an over-64,000-character user prompt switches to versioned compact rows: exact reel IDs remain for output, internal candidate/chain/prerequisite/source/concept identities become deterministic per-prompt integer references that preserve equality and dependency semantics, prior concept coverage shares the same concept aliases, and every clip retains times, a fair concept/family prefix, learner signal, difficulty/relevance, plus equal non-empty summary/takeaway/transcript excerpts (missing enrichment fields reuse another available semantic field rather than becoming blank). The lesson prompt/cache advance to v7/v6. The legal 128-ID order plus 128 checkpoints is 10,041 ASCII bytes and measures 8,593–8,761 tokens over 100 deterministic UUID samples with the Gemini tokenizer, so the 10,240-token response ceiling remains sufficient; actual generated length, not the ceiling, controls latency, and no call or sleep is added.
- Focused retest: A slow cross-request top-up starts with eight eligible unseen parent clips, invokes acquisition once with `num_reels=9`, unchanged `max_new_reels=1` and `max_generation_videos=3`, persists 120 fresh current acoustically verified Gemini candidates from exactly three analyzed sources, and gives all 128 clips with intact candidate/chain/position/prerequisite/family/quality metadata to one organizer call. Its final public event contains none of those private fields. The inverse reproduction starts with 120 raw eligible parent clips but a stored eight-clip editorial subset: raw counting reaches the nine-reel startup threshold, performs zero acquisition, and still gives all 120 parent candidates to the organizer. An adversarial 128-candidate prompt with maximum escaped fields, forty prior-coverage rows, and sixteen maximum-length prerequisite IDs per clip stays below 64,000 characters, retains all exact reel IDs and relationship aliases, omits raw internal IDs, and gives every clip non-empty concept, family, summary, takeaway, and transcript content. A separate legal 128-ID/checkpoint response succeeds in one dispatch with no retry sleep; the small-batch prompt remains byte-identical. Stale request-schema and ranked-cache compatibility guards also pass; the combined focused set passes 29 checks.
- Exact retest: Pending identical-SHA deployment, clean Postgres/Redis, five-domain generation, production prompt/usage telemetry, and time-to-first-reel comparison against the recorded baseline.

### AF-111 — Bootstrap capacity deferral was counted as source failure

- Status: Fixed locally; exact clean-production accounting retest pending
- Severity: High — when three sources completed selection but a one-reel bootstrap cap accepted only the strongest source, the other two successful sources were written as failed attempts. Repeating that normal capacity outcome could retire useful sources without any provider, selector, or persistence failure.
- Reproduction: Submit three bootstrap sources with one valid clip each and `max_reels=1`. Before the fix, all three IDs were attempted, only the persisted winner was completed, and the two clip-bearing capacity deferrals fell into `attempted - completed`, producing false `failed_source_attempts`.
- Root cause: AF-104 correctly delayed completion of clip-bearing sources until persistence returned, but its accounting had only completed and failed terminal states. Bootstrap intentionally stops offering completed source results to persistence after its inventory cap fills, so a successful capacity deferral was indistinguishable from a persistence exception.
- Universal fix: Track capacity-deferred source IDs explicitly from ingestion through the worker usage payload. Only successfully completed clip-bearing results that were never offered to persistence because the bootstrap cap was already full enter that set. Exclude those IDs from failed-attempt accounting while leaving them unconsumed and eligible for a later batch. A source whose persistence call actually raises remains attempted, neither completed nor deferred, and therefore keeps its bounded retry attempt.
- Focused retest: The exact cap-one/three-submitted reproduction now records `attempted=3`, `completed=1`, `capacity_deferred=2`, and zero failures. Deep/bootstrap persistence exceptions remain attempted failures with no capacity deferral. Service pass-through and durable worker usage tests pass (5 tests plus 2 subtests).

### AF-112 — A secondary selector regression still required the removed lexical veto

- Status: Fixed locally; exact-SHA production contract retest pending
- Severity: Medium — the secondary pipeline gate failed even though the production selector now correctly accepts AI-grounded constraints without mechanically repeating every sequencing or teaching word from the compact request
- Reproduction: The routing fixture labeled a stemmed token-union gap as `intent_contract_incomplete_request_coverage` and expected a second Gemini Pro call. The same collection also expected a twice-invalid exact-request mismatch to return an ordinary empty result instead of the operational error required by AF-107/AF-109.
- Root cause: The test encoded the superseded local token-union heuristic and pre-retry failure classification. It therefore contradicted the universal AF-103 contract while still exercising the live production selector path.
- Universal fix: Change only the obsolete expectation. A lexical token-union gap with otherwise grounded, structurally valid Gemini constraints is accepted in one healthy call. A true `exact_request` mismatch still uses the single bounded repair and, when repeated, returns `GeminiSelectorContractError` so durable source retry accounting can act on it.
- Exact retest: The four focused recovered/unrecovered cases pass, and the complete secondary pipeline collection passes 1,404 tests. No production validator or provider budget was changed for this test correction.

### AF-113 — A retryable provider outage terminalizes the durable job after its first attempt

- Status: Fixed locally; exact clean-production retry retest pending
- Severity: Critical — a temporary Gemini outage can return zero reels even though both the provider telemetry and terminal job error explicitly declare the failure retryable
- Production reproduction (exact SHA `a45fcd531cf41ca1ec2e7cdc5bca8b2ad5848d4d`, clean Postgres/Redis): Beginner slow-mode physics material `09d50059-36b4-4179-95fa-eef8899172e9` created durable job `ff9c0996-8fa2-4d2a-975e-ce847ef221f6`. Query expansion and three Supadata searches/transcripts succeeded. Gemini Pro selection then returned retryable 503/504 transport failures for two sources and a non-retryable 499 cancellation for the third. The terminal payload records `provider_transient`, `retryable=true`, three usable transcripts, an open provider cursor, and three failed-source attempts. Nevertheless the job changed directly to `failed` at `attempt_count=1` with `max_attempts=2`, emitted only one terminal event, and persisted zero reels. Recovery occurred only because the feed submitted a different linked job after observing that premature terminal state; the durable worker itself performed no retry.
- Expected: A retryable terminal generation failure must requeue the same replay-safe durable job until its bounded job-attempt ceiling is reached. Per the live request, the universal ceiling will be three attempts; healthy jobs remain one attempt and existing per-provider retries remain bounded independently.
- Root cause: The worker caught every typed provider error and unconditionally called the terminal transition without reading `error.retryable`; `max_attempts` therefore applied only to expired/crashed leases. A fresh `GenerationContext` also replaced the prior attempt's source ledger, so simply requeuing would forget earlier consumed and failed sources. Finally, an all-source parallel batch raised the first completed provider error, allowing completion order to classify a mixed permanent/transient outage as permanent.
- Universal fix: Use three total durable attempts. A lease-fenced, deadline- and cancellation-aware transition now returns only typed retryable provider failures to the same queued job while preserving its attached generation, execution deadline, cumulative graceful-requeue telemetry, consumed/deferred/failed source identities, and retry diagnostics; the per-call provider table remains the authoritative hard-lease-loss audit ledger. Requeue emits no terminal event, does not settle the search reservation, and converges after an unknown commit acknowledgement; attempt three retains the original provider error as the single terminal outcome. A durable not-before timestamp honors bounded provider `Retry-After`, and the worker waits only until that timestamp rather than either hammering the provider or falling into its 15-minute recovery poll. Authentication, configuration, quota, model, budget, and other job-global non-retryable provider failures remain terminal on attempt one. Passive feed refreshes report that terminal job without creating linked work, independent of whether any video reached analysis; a later explicit generate action may deliberately create one fresh root job after credentials, quota, or configuration recover, and repeated explicit calls converge on that active job. Parallel source aggregation gives those global errors precedence, otherwise propagating a retryable representative whenever any sibling failed retryably. Each explicit failed source analysis increments the existing two-attempt AF-108 bound even inside one durable job, so attempt three retires a twice-failed video and searches fresh YouTube sources; an unknown crash/lease recovery does not invent a source failure. Cumulative diagnostic counters survive graceful attempts, while `provider_cursor_open` remains a last-attempt work-availability gauge so an old cursor cannot silently bypass the retry ceiling. A provider response that arrives after lease expiry yields and wakes immediate recovery instead of escaping into the long poll. Schema, migrations, status defaults, and active jobs consistently use the three-attempt ceiling.
- Focused retest: Retryable failures on attempts one and two followed by success on attempt three preserve one job, one generation, one reel, cumulative provider/source history, and exactly `final -> terminal`; three retryable failures emit one terminal event and permit no fourth lease; ancestral plus same-job source failures retire at exactly two; a non-retryable quota control terminates on attempt one; an explicit later request creates one fresh deduplicated recovery root while passive feed reloads create none, including with an empty source ledger; committed-requeue/lost-ack replay converges; Retry-After prevents an early lease; lease-expiry races wake recovery; repository fencing preserves quota/event semantics; and mixed quota, rate-limit, transient, and source-local errors have completion-order-independent precedence. Focused generation-job, worker, and pipeline collections pass. Full-suite and exact-SHA production retests remain pending.

### AF-114 — One semantic-empty source masks retryable failures from every sibling source

- Status: Fixed locally; exact clean-production mixed-source retest pending
- Severity: Critical — a batch can return zero reels as a successful semantic no-match even though most discovered YouTube sources failed transiently and remain unanalyzed
- Reproduction: Analyze three sources concurrently. Let one Gemini selector validly return no clip for that individual video while the other two selectors return retryable provider failures. The pipeline stores the empty source in `completed_results`, so its total-outage check is false and the worker receives an ordinary empty list instead of the retryable error.
- Root cause: Failure propagation keyed on whether any source analysis completed, not whether the batch produced any persisted reel. A valid empty result for one video therefore suppressed operational failures from unrelated videos.
- Universal fix: When a batch produces no reels, propagate its provider failures. Job-global configuration/authentication/quota/model/budget errors take precedence; otherwise any retryable sibling makes the batch retryable. Preserve the existing partial-success behavior whenever at least one reel was actually persisted.
- Focused retest: Valid-empty plus two transient siblings now raises the retryable error; global quota plus transient raises quota in either completion order; source-local request rejection plus transient remains retryable in either order; and a valid persisted sibling still completes despite other source failures (7 focused checks pass).

### AF-115 — Durable retries reset the Gemini job-wide cost ceiling

- Status: Fixed locally; exact clean-production retry/cost retest pending
- Severity: High — a three-attempt durable job can otherwise admit up to three independent copies of the configured Gemini spend ceiling
- Reproduction: Persist a retryable attempt whose budget snapshot contains committed or still-in-flight Gemini exposure, lease the same job again, and construct its new `GenerationContext`. The second context begins at zero exposure even though telemetry later merges the two attempts for reporting.
- Root cause: Durable retry persistence was added above the pre-existing in-memory `GenerationBudget`; every worker invocation created a fresh budget and hydrated source/usage ledgers but did not hydrate the job-level Gemini admission state.
- Universal fix: Restore prior job-level Gemini committed, unknown, and unresolved in-flight exposure into the next attempt. On lease/process recovery, attempt two also reconstructs known and billing-unknown exposure from already-committed per-call provider rows and takes the maximum of that ledger and the job snapshot, so overlapping telemetry is never double-counted. Keep selector/audit call slots, acquisition passes, and ordinary search/transcript reservations attempt-local so the retry can actually recover after all first-attempt source slots fail, as they did in AF-113. Lifetime reservation diagnostics remain cumulative without triangular summary merging. The healthy first attempt performs no added ledger query, provider call, sleep, or database write.
- Focused retest: Three first-attempt selector slots reopen on retry while prior cost stays charged; unresolved exposure blocks a request above the remaining ceiling; a reclaimed lease with stale `{}` job usage reconstructs `$0.92` from its durable provider row and rejects a crossing dispatch; three graceful attempts retain monotonic exposure and matching cumulative reservation diagnostics; attempt one never invokes ledger recovery; and a failed retry-only ledger read never starts a heartbeat or prevents later lease recovery. Runtime, worker, and reporting checks pass.
- Explicit boundary: A process death after dispatch but before any provider-usage row can be committed is not observable after restart; similarly, a reclaimed attempt can hydrate before an old in-flight worker commits its late response. Eliminating those timing windows would require a pre-dispatch database write on every healthy Gemini call, which conflicts with the no-stream-latency requirement. Already-committed provider rows and snapshot-only records are fingerprint-unioned without double counting and remain fail-closed. After a hard loss, that union is authoritative for admission; `usage_json` call/stage aggregates remain best-effort because reconstructing them is not required for safety.

### AF-116 — The ingestion rate limiter bypasses durable provider retry handling

- Status: Fixed locally; exact clean-production rate-limit retest pending
- Severity: High — local platform saturation is explicitly recoverable and carries `Retry-After`, but currently terminalizes as an untyped `generation_failed`
- Reproduction: Make `ReelService.generate_reels()` raise `ingestion.errors.RateLimitedError`. `_run_leased_generation_job()` does not catch that type in its provider branch, so the generic exception handler marks the job failed instead of requeuing it.
- Root cause: The worker durable-retry branch recognizes only clip-engine `ProviderError` even though the ingestion pipeline has a separate typed rate-limit exception used by the same generation step.
- Universal fix: Normalize ingestion saturation to the existing retryable provider-rate-limit contract at the worker boundary, preserve its bounded `Retry-After`, and reuse the same lease-fenced requeue path. Rate limiting is job-global for terminal/feed policy but remains retryable inside the same three-attempt job. A non-retryable quota/auth/configuration failure still wins over a simultaneous rate limit; otherwise the longest bounded rate-limit delay wins over source-local transient errors.
- Focused retest: The worker stores `provider_rate_limited`, queues the same job in `retrying`, preserves the five-second not-before timestamp and diagnostic detail, emits no terminal event, rejects an early lease, and leases attempt two at the timestamp. Quota-versus-rate-limit and rate-limit-versus-transient controls pass in both source completion orders; passive feed reload remains pinned after the three-attempt ceiling.

### AF-117 — A clip can be released as self-contained while ending on a dangling connective

- Status: Reproduced on the exact production SHA; universal fix implemented locally; exact-SHA production retest pending
- Severity: High — the reel boundary can cut a teaching sentence before its conclusion even though the selector and persisted boundary contract label the clip self-contained and context-aligned
- Production reproduction (SHA `f04d91de7f5a0d109e0a3c59f87ae99a44d58e18`, clean adaptive Postgres/Redis): Physics material `0f9ac995-cde3-4673-aa13-3b2cf0008aaf` recovered through durable attempt two and released six relevant reels. Reel `ingest-e9c82c0738ba40a4` from video `Ee6CHn0MRKE` spans `1.68–81.6`, is marked `self_contained=true` and `boundary_status=context_aligned`, but its exact final transcript words are `...inversely proportional to the mass of the object so`. Boundary diagnostics report `partial_edge_fallback:selected_cue_range_unavailable`, confirming that the released end is a fallback edge rather than a completed explanatory thought.
- Expected: A clip may end only after the selected explanation completes; a model assertion of self-containment must not override direct transcript evidence that the chosen end still opens a continuation.
- Root cause: The selector and independent Pro audit already require a complete final clause, but the deterministic AI-output invariant did not classify a bare terminal connective `so` as incomplete. Both `_cue_has_explicit_dangling_end()` and `_terminal_content_is_explicitly_incomplete()` returned false, so the converter trusted Gemini's `self_contained=true`. The downstream selected-cue fallback retained the incomplete edge but did not create it.
- Universal fix: Treat terminal connective `so` as an unfinished result-clause lead-in while exempting complete proforms such as `if so`, `do so`, and `think so`. Reuse the existing bounded transcript completion path; if no same-unit grounded completion is available before a section reset, lexical reset, or completion bound, reject the candidate instead of shipping it with an unresolved diagnostic. This adds no provider call, sleep, or healthy-path latency.
- Focused retest: The exact three-cue Newton shape extends only through `the acceleration doubles.` and excludes the next topic. Complete proforms—including `if/do/think so`, declarative `this/that/it is so`, and `why this is so`—stay at their original edge, while `We have the premise, so.` remains unfinished. Ungroundable section-gap, lexical-reset, and bounded-completion cases reject with `proposal_N:unresolved_dangling_end`. The final selector/curation matrix passes 949 tests.
- Boundary-quality control: The same batch's 222.72-second friction problem is not a defect. Timed-caption review confirms one coherent two-part objective—setup, net-force calculation, sign/direction qualification, acceleration calculation, and consistency check—starting after the prior example and ending before a new velocity scenario. A generic duration cap would truncate required context, so no code change was made for that clip.

### AF-118 — Cross-domain durable retries can exhaust the cost ceiling on repeated selector-contract rejection and still return zero reels

- Status: Reproduced on the exact production SHA; universal fix implemented locally and retested with production provider credentials; exact new-SHA production retest pending
- Severity: Critical — a normal biology learning request has abundant discovered YouTube sources and complete transcripts, yet every clip is discarded before persistence and the user receives no reel after the bounded recovery ceiling
- Production reproduction (SHA `f04d91de7f5a0d109e0a3c59f87ae99a44d58e18`, clean adaptive Postgres/Redis): Material `c1cfa597-1654-49ff-b261-8bef1b201402` asks a beginner explanation of ATP production across glycolysis, the Krebs cycle, and the electron transport chain, including oxygen and ATP-yield comparison. Job `3a8e38d6-3489-45ff-ba79-3f1e3214ef84` discovered and transcribed three videos on attempt one. Attempts one and two each ended in retryable `GeminiSelectorContractError` after Pro returned HTTP-200 responses and consumed the contract-repair dispatch; attempt three searched fresh sources but the restored job-wide Gemini exposure correctly rejected further selection before dispatch. The single durable job reached exactly `attempt_count=3/max_attempts=3`, terminalized once with `provider_budget_exceeded`, and persisted zero reels.
- Expected: Valid transcript-grounded biology teaching clips must survive the universal selector contract. Durable retries should recover transient model/schema failures, but repeating a deterministic validator/model mismatch must not spend the entire job budget without exposing or repairing the actual rejected contract condition.
- Confirmed non-root-causes: YouTube scarcity, transcript scarcity, missing retry execution, job identity loss, and cost-reset bugs are ruled out. Search/transcript calls succeeded, the same job consumed all three attempts, and the third attempt honored accumulated Gemini exposure.
- Exact diagnostic reproduction: Replaying the same request and production source `2f7YwCtHcgk` through the old selector produced `intent_contract_incomplete_joint_structure` on its first HTTP-200 response. With the local selector contract and production Railway provider credentials, the exact source returns eight valid clips with no rejection reasons. A later optional Pro call reached a real 504 deadline, but the already-valid clips were retained rather than converted to a zero-reel result.
- Root cause: `_validated_intent_constraints()` treats any request containing `compare` as malformed unless Gemini assigns at least one constraint the arbitrary `RELATIONSHIP` enum. Gemini can validly represent the exact intent as subject `ATP production in cellular respiration`, task `Explain`, separately grounded scopes for glycolysis/Krebs/ETC, outcome `why oxygen matters`, and task/outcome `compare ATP yield`. All meaning and source grounding are preserved, but the absence of the redundant enum empties every returned topic and the identical repair prompt can make the same classification again.
- Universal fix: Accept a grounded multi-facet decomposition whose final comparison is encoded as `TASK`/`OUTCOME` only when the exact request itself has an explicit colon-led list with at least two separators, at least three grounded scope constraints, a grounded subject, and comparison wording in that task/outcome. Pure or unstructured comparisons still require `RELATIONSHIP`; an unambiguous binary comparison can use the existing local repair outside live Pro, while live Pro fails closed and retries. Exact-request equality, unique grounded source phrases, transcript evidence, concept-family validation, boundary checks, and the independent Pro audit remain unchanged. No provider call, sleep, or healthy-path work was added.
- Adversarial retest: A hostile `Explain how precision compares with recall` plan labeled as `SUBJECT + TASK + three SCOPE + OUTCOME` now fails `intent_contract_incomplete_joint_structure`; biology, math, software, and law list-shaped controls remain accepted. Exact schema/contract rejection codes now survive `SegmentResult`, provider usage, and durable job rejection summaries without storing prompt or transcript text. The focused relationship set passes 21 tests and is included in the 949-test selector/curation matrix.

### AF-119 — Broad concept-family identity prevents topic-specific thumb and quiz adaptation

- Status: Reproduced on the exact production SHA; universal concept-contract fix implemented locally; exact new-SHA production adaptation retest pending
- Severity: Critical — feedback for one taught subtopic changes every reel under the umbrella law, so thumbs and quiz outcomes cannot produce the requested concept-specific frequency behavior
- Production reproduction: The first six Newton reels have distinct Gemini facets—equation/units, core concept, proportionality, net force, frictionless problem, and friction problem—but five persist the same sole adaptive identity `Newton's second law of motion`. Pressing `Need help` on the proportionality reel `ingest-453bc906c9a3490f` correctly persisted `confusing=1`, raised the material feedback revision, and triggered generation `d4f6e9eb-3867-4e2a-8d38-66ef2743e2fb`. That generation completed in one attempt, but its five Newton-law facets all inherited the same confusing concept signal; the organizer had no way to increase proportionality without also increasing acceleration definition, equation, force proportionality, mass proportionality, and generic calculations.
- Root cause: Gemini already returns a narrow `facet`, but the selector/audit prompt defines the broader `concept_family` as the sole adaptive identity and explicitly normalizes F=ma material to the umbrella Newton law. Persistence deterministically keys `concept_id` from that family. The narrow trusted `clip_concept_raw` is persisted in search context but not surfaced to the organizer.
- Universal fix: Contract `concept_family_v3` asks Gemini and its independent audit for the smallest reusable taught subtopic, never a broad law, field, system, or course umbrella. Genuine paraphrases reuse one ID; units, net force, and distinct problem/application subtopics stay separate. The trusted narrow clip concept now survives the internal organizer allow-list with summary, takeaways, transcript excerpt, family, difficulty, and learner signal; public output still strips private selection fields. The selector/inventory contract advances to `quality_silence_v39`, Pro audit prompt to `pro_candidate_audit_v8`, generation request schema to `adaptive_clip_concepts_v4`, and ranked cache to v46 so broad pre-fix inventory cannot be replayed and telemetry identifies the new audit contract.
- Focused retest: Six production-shaped physics clips yield five adaptive IDs with two proportionality paraphrases sharing one; helpful feedback affects only that ID, and a wrong friction outcome affects only friction. Cross-domain identity, persistence, acquisition, ranking, organizer-signal, and prior-coverage controls pass without a local semantic merge heuristic.
- Compatibility-gate addendum: Bumping the trusted profile reader from `concept_family_v2` to `concept_family_v3` did not by itself isolate old mastery. Acquisition reel counts, exact-ID thumb aggregates, quiz outcomes, global level drift, latest-remediation selection, automatic level promotion, organizer learner signals, and prior released coverage still joined through the unchanged deterministic `concept_id`; an old broad v2 row could therefore bias a new narrow v3 feed even though neither the v3 family reader nor the v39 inventory would surface that row. A shared contract-version constant and provenance gate now exclude only rows with an explicit non-current Gemini family contract at every adaptive read boundary. Current v3 rows remain active, and unversioned, malformed, or non-Gemini legacy rows retain their prior exact-ID behavior because their provenance does not prove incompatibility. Global-drift reads use newest-first pages of 24 rows and stop after 96 rows per signal source, so compatibility filtering adds no provider work and cannot materialize or scan an unbounded history. Focused controls verify v2 rejection, v3 acceptance, unversioned/malformed compatibility, compatible events behind more than twelve stale events, the all-stale scan cap, acquisition counts/order, thumb and quiz adjustments, latest remediation, global drift, automatic promotion, organizer signal payloads, and prior-coverage filtering.

### AF-120 — Organizer retry repeats the same Lite model and degraded fallback cannot remove semantic duplicates

- Status: Reproduced on the exact production SHA; universal AI failover fix implemented locally; exact new-SHA production progression retest pending
- Severity: High — a transient or capacity failure persists a deterministic order that cannot compare cross-source meaning or infer a concept-to-application progression
- Production reproduction: The initial Newton generation records `fallback_reason=provider_call_failed`. Both organizer attempts used `gemini-2.5-flash-lite`; the retry therefore repeated the same capacity. The deterministic fallback correctly preserves source chronology and explicit dependencies but kept differently worded cross-source proportionality restatements and cannot infer the intended concept → explanation → net-force application → worked-problem progression.
- Root cause: Organizer resilience is model-local. On dual Lite failure the fallback only has timestamp/dependency structure; upstream local duplicate handling is lexical and cannot prove semantic equivalence across sources. Adding another token/timestamp heuristic would repeat the same class of defect.
- Universal fix: Preserve the one-call healthy path on `gemini-2.5-flash-lite`, but use the existing second attempt with `gemini-2.5-flash`; versioned Lite and Flash aliases resolve to the same opposite-capacity pairing. The `lesson_order_v8` prompt asks the AI to omit cross-source semantic restatements, retain multiple examples only when each adds a new reasoning/application step, and order concept before explanation before application/worked example. Permanent provider errors fail fast; dual-provider failure keeps the dependency-safe deterministic fallback. The organizer receives 18 compact, positionally aligned fields without truncating candidate meaning. Its step ceiling remains two while the durable generation ceiling remains three, so retries do not multiply and the healthy path gains no call.
- Focused retest: Healthy ordering makes exactly one Lite call; retryable Lite failure recovers through exactly one Flash call; permanent errors make one call; dual failure degrades once; versioned aliases choose the opposite model; compact reconstruction preserves every field; and an alternate-AI response can omit a semantic restatement and order concept before application.

### AF-121 — Bare slash notation is misclassified as an explicit comparison

- Status: Found during final cross-domain audit; universal fix implemented locally; exact new-SHA production software-domain retest pending
- Severity: Critical — ordinary technical requests can be rejected by the same selector-contract path that produced the biology zero-reel failure
- Reproduction: Valid `SUBJECT` contracts for `Explain HTTP/2`, `Teach TCP/IP basics`, `Explain input/output processing`, and `C/C++ memory management` all returned `intent_contract_incomplete_joint_structure`. The pure-binary parser correctly rejected these as comparisons, but three later request checks and the selector topic instruction independently treated every `/` as comparison syntax.
- Root cause: Bare punctuation was used as local semantic authority. A slash commonly denotes versions, protocols, paired I/O terminology, or language names and cannot by itself prove that the user requested one clip comparing two endpoints.
- Universal fix: Remove bare `/` from lexical comparison, conjunctive, and topic-prompt rules. A slash request becomes a joint comparison only when Gemini supplies a grounded `RELATIONSHIP` constraint plus at least two grounded `SUBJECT`/`OUTCOME` endpoints; downstream transcript evidence must still state the actual comparison. This is model-structured intent, not a technical-notation allowlist, and adds no provider call, retry, sleep, or healthy-path latency.
- Focused retest: All four technical notation requests retain their ordinary subject contracts. A `precision/recall` request with a grounded full-phrase `RELATIONSHIP` and two grounded subjects remains joint and valid. The relationship/adversarial set passes 26 tests.

### AF-122 — Independent Pro audit cannot reject academically incorrect but on-topic teaching

- Status: Reproduced on exact production SHA `1fe87e8f47bce8b08af6d2ff24e245d799dc776e`; universal fix implemented locally; exact new-SHA production retest pending
- Severity: Critical — a relevant, well-cut clip can still teach false or conflated concepts and lead an otherwise coherent lesson
- Production reproduction: Clean physics continuation job `9ceedd26-1ef9-4c5d-8893-aca6f35a7c70` released a 98.391-second opening clip that says `F=ma` is “written wrong,” folds inertia and Newton's third-law action/reaction into the second-law equation, and uses an unrelated punching example. Five later released clips give a clean force proportionality → mass proportionality → equation → unit → worked calculation sequence, so the misleading opening is unnecessary as well as educationally harmful.
- Root cause: The selector task says to keep academically sound teaching, but the compact `factually_grounded` field is defined only as whether the supplied transcript supports the claim. The independent high-thinking audit has only `keep`, `reject_unrelated`, and `reject_filler_dominated`, and its prompt explicitly forbids rejection for factual uncertainty. It therefore cannot correct a transcript-grounded factual error or concept conflation that is on topic.
- Universal fix: The existing independent high-thinking audit now has a distinct `reject_factually_incorrect` decision. It is limited to a definitive, central, explicitly spoken false claim or conflation whose short evidence quote contains the error. The auditor must salvage an academically sound related unit first and must fail open for ambiguity, approximation, disputed/incomplete claims, misconceptions being posed or corrected, missing visuals, possible transcription error, or its own uncertainty. This remains an AI subject-matter judgment in the existing audit call; no local fact lexicon, provider call, sleep, or healthy-path delay was added. Prompt telemetry advances to `pro_candidate_audit_v9`.
- Focused retest: An explicit Newton-law conflation is rejectable, while factual uncertainty remains `keep`; unrelated/filler behavior, salvage, call count, and output budget remain intact. Selector/audit, output-budget, persistence, cache, ordering, and generation integration are included in the 1,015-test affected matrix.

### AF-123 — Broad family metadata still overrides Gemini's clip-specific concept identity

- Status: Reproduced on exact production SHA `1fe87e8f47bce8b08af6d2ff24e245d799dc776e`; universal fix implemented locally; exact new-SHA production retest pending
- Severity: Critical — thumbs and quiz outcomes for one narrow subtopic can still alter unrelated clips under the same umbrella family
- Production reproduction: The clean physics batch persisted distinct Gemini clip concepts `Force, mass, and acceleration relationship` and `Mass and required force`, but both reels received the same adaptive concept ID/title `Newton's second law of motion`. The v3 provenance is present, so this is current behavior rather than stale inventory.
- Root cause: Ingestion correctly receives Gemini's narrow `concept`, then passes the broader `concept_family` as `semantic_identity` to `ensure_clip_concept()`. That optional argument deterministically replaces the concept title and UUID with the family identity. The trusted narrow value survives only as `clip_concept_raw`, so organizer prose can see it but every persisted thumb/quiz learner signal still joins through the broad family ID.
- Universal fix: Persistence now uses the normalized Gemini clip concept as the material-scoped adaptive UUID/title. Case/spacing aliases of that same narrow concept reuse one identity, while distinct facets inside the same audited family remain distinct. The family stays as versioned provenance and organizer context and no longer replaces the adaptive identity. No semantic word matcher or provider call was added.
- Focused retest: Two case/spacing variants of `Net Force—Acceleration` reuse one concept, while `Mass and required force` receives a different concept despite sharing the same family. Persistence/provenance integrity passes with 22 tests and 3 subtests as part of the combined affected matrix.

### AF-124 — A continuation can stream a whole-source clip that contains all previously released clips

- Status: Reproduced on exact production SHA `1fe87e8f47bce8b08af6d2ff24e245d799dc776e`; universal fix implemented locally; exact new-SHA production event retest pending
- Severity: High — the raw authoritative continuation can briefly expose a several-minute near-duplicate before later feed reconciliation removes it
- Production reproduction: Biology job `fd826b8e-cae4-49c2-a684-9300d7b8c90b` first released four same-source clips for glycolysis, Krebs, ETC, and ATP yield over `186.472–443.052`. Its cached-sibling continuation `8abb682a-b7e8-40a3-b9d8-013c3e6695ed` then emitted a single `104.876–443.052` whole-process clip from the same video, fully containing all four prior spans.
- Root cause: Same-source temporal overlap filtering runs inside one organizer response and again when a later `/api/feed` read reconstructs the full authoritative generation chain. The worker's raw continuation `final` event is validated only against that continuation's current candidate set, so it cannot compare the new subset with spans already released by its source generation before streaming the event.
- Universal fix: Before lesson metadata and the authoritative `final` event are persisted, the continuation subset is reconciled against the complete prior authoritative release chain with the same source/time geometry and dependency protections used by stable feed reads. The filter remains geometric for clips without the new contract, but a substantially overlapping clip is retained when it carries a grounded requested-obligation key not covered by the earlier overlapping spans; this prevents deduplication from deleting the only teaching for a requested facet. No provider work was added.
- Focused retest: A containing continuation with no novel obligation is removed before the event; a same-source overlapping clip with a novel grounded facet survives; checkpoint, chain, and prerequisite behavior remains valid. The complete generation-job collection passes 182 tests.

### AF-125 — Need-help acquisition finds the exact concept but organizer omits it

- Status: Reproduced on exact production SHA `1fe87e8f47bce8b08af6d2ff24e245d799dc776e`; universal fix implemented locally; exact new-SHA thumb/quiz retest pending
- Severity: Critical — thumbs-down can regenerate the requested concept successfully yet make that concept disappear from the released adaptive batch
- Production reproduction: `Need help` on reel `ingest-704fc50f227e473d` persisted `confusing=1` for exact concept ID `electron transport chain` and advanced `feedback_revision` to 1. Adaptive job `2fab33ba-1642-462b-8ff6-1844cf998065` correctly acquired all seven candidates under `electron transport chain`, including new-source reel `ingest-8e0ac8a6ba7941d0` with the same exact concept ID. The organizer released five lower-difficulty generic/glycolysis/oxygen/yield clips and omitted that sole ETC remediation candidate.
- Root cause: Feedback steers acquisition and is supplied to the organizer, but the subset contract treats the signal as advisory. No postcondition requires an available exact confusing/wrong concept to survive the organizer subset, so beginner difficulty preference can override the core frequency guarantee.
- Additional production evidence: A deliberately wrong `remoteness (law)` quiz persisted adjustment `-0.12` and triggered job `9d35b995-b92f-47a5-a4a4-fa688d97bf6f`. The job recovered on durable attempt 2 and released a narrower proximate-cause/remoteness lesson plus duty, breach, causation, and damages material. This proves the quiz signal and three-attempt recovery path operate, but the pre-fix broad identity/organizer contract still did not guarantee the exact concept witness.
- Universal fix: Exact post-reset thumb and quiz deficits are retained separately from family-propagated acquisition signals. After the existing organizer call, one unified deterministic postcondition makes the strongest available exact remediation concept and every available grounded request obligation mandatory together, including prerequisite and chain closure and the release limit. It prefers the easiest exact candidate, never substitutes an unrelated easier topic, and lets the organizer's remaining choices fill unused slots. Helpful/correct concepts remain omittable. No provider call or wait was added.
- Focused retest: The organizer cannot omit an available easiest exact ETC remediation clip or its prerequisite; exact remediation and a separate requested law facet coexist in one capacity-limited release; cached and fallback organizer paths run the same postcondition.

### AF-126 — Independent batches can place a recap before an explicitly requested method is taught

- Status: Reproduced on exact production SHA `1fe87e8f47bce8b08af6d2ff24e245d799dc776e`; universal fix implemented locally; exact new-SHA five-domain progression retest pending
- Severity: High — individually good clips can form an incomplete or backwards lesson when the first batch omits an available requested concept and a later refinement appends it after a recap
- Production reproduction: Clean math generation `3805d4c4-04da-4238-ad89-ae679324188b` had ten valid candidates and released five clips ordered as two factoring examples → a longer factoring/method-choice explanation → quadratic-formula example → recap of all three methods. It omitted the already-available `completing the square method` candidate. Refinement generation `0ca5735f-89a3-465b-8bbd-e831d33fff81` later appended a methods comparison and two completing-the-square lessons, so the stable lesson teaches that requested method only after the first batch's final recap.
- Root cause: The organizer selects and orders each generation independently. Its semantic prompt encourages coverage and progression but neither the authoritative subset validator nor the cross-generation reconciliation enforces available explicit-intent coverage or prevents a recap/summary clip from preceding later teaching of a missing requested concept.
- Universal fix: Each Gemini-selected clip now carries a versioned obligation object built from its grounded constraint enum, exact request phrase and character offset, complete requirement, and exact transcript evidence. An opaque key is derived from that exact case-preserving source span, validated through segment cache and persistence, and supplied with the clip's content to the existing organizer. The organizer receives the union of available obligations and prior released keys; its response also marks the first terminal recap/whole-lesson summary. One postcondition fills omitted available obligations before that recap, and authoritative cross-generation reconstruction stable-partitions later teaching before earlier recap suffixes. Prior-covered facets are not forced again, one clip may cover multiple facets, unavailable facets are never invented, and clip count remains bounded only by existing candidate/release limits.
- Focused retest: When Gemini returns `intro → recap` but a completing-square candidate is available, the release becomes `intro → completing square → recap`; prior coverage suppresses forced repetition; one comparison clip satisfies both time and space obligations; cache/fallback and cross-generation marker paths agree. The 128-candidate/16-obligation prompt remains below 64,000 characters and uses the same one organizer call. Prompt/cache versions advance to `lesson_order_v9`/7.

### AF-127 — A valid multi-facet comparison request spends durable retries on a malformed joint selector contract

- Status: Reproduced on exact production SHA `1fe87e8f47bce8b08af6d2ff24e245d799dc776e`; all three durable attempts observed; universal fix implemented locally; exact new-SHA software/law retest pending
- Severity: Critical — an ordinary educational request with abundant YouTube sources can spend multiple paid attempts and substantially delay or impoverish the first released lesson
- Production reproduction: Software prompt `Explain sorting algorithms for a beginner: bubble sort, merge sort, and quicksort, then compare their time and space tradeoffs.` created job `bc6d731a-b30a-435d-8885-ede7bcc6aa06`. Attempt 1 found three videos and two usable transcripts, but both Pro selector sources exhausted their same-call contract retries with `intent_contract_incomplete_joint_structure`; the durable worker correctly requeued it. Attempt 2 repeated the same contract failure and requeued. Attempt 3 recovered and emitted a partial result after 218.705 seconds, proving the three-attempt mechanism itself works, but the released subset contained only two merge-sort clips and still omitted bubble sort, quicksort, and the requested tradeoff comparison while background refinement began.
- Root cause: The live intent validator inferred semantic grouping from enum positions and punctuation. That made valid atomic relationships containing commas fail, let some bundled or missing-final-relation plans pass, and locally rewrote certain binary comparisons. No punctuation rule can universally distinguish a list from a compound clause, and enum roles are semantic labels rather than fixed slots.
- Universal fix: Gemini now returns an explicit bounded `joint_structures` graph containing member constraint IDs and the relation constraint ID. The transport envelope remains stable, while Gemini may choose honest `SUBJECT`/`SCOPE`/`TASK` member roles and `RELATIONSHIP`/`TASK`/`OUTCOME` relation roles. Local code only validates known unique IDs, exact grounded spans, non-overlapping member topology, and coverage of the requested comparison/transition; it performs no punctuation-based semantic grouping or binary repair. Bundled members, overlaps, swallowed members, and missing comparison clauses still fail. Set-level coverage is handled by AF-126. Primary selector telemetry advances to `pro_boundary_v21`.
- Error-classification fix: HTTP-200 schema/contract rejections are now retryable `provider_response_invalid` with bounded internal reason codes and the truthful message that Gemini responded but validation failed. Genuine 429/503 responses remain `provider_transient`/temporarily unavailable. Durable generation still performs at most three total attempts.
- Focused retest: Valid biology, math, software, and law multi-facet plans pass with `SUBJECT`/`SCOPE` and `RELATIONSHIP`/`TASK`/`OUTCOME` variation, including natural no-colon lists. Fake bundled/overlapping/missing-relation plans fail. 429/503 controls preserve outage classification; all affected tests pass in the 1,015-test matrix.

### AF-128 — Naming requested concepts in an agenda is incorrectly counted as teaching all of them

- Status: Reproduced on exact production SHA `1fe87e8f47bce8b08af6d2ff24e245d799dc776e`; universal AI-contract fix implemented locally; exact new-SHA law/software retest pending
- Severity: Critical — the selector and organizer can mark an incomplete lesson as fully covering the request and stop prioritizing the missing teaching
- Production reproduction: Law prompt `Explain negligence for a beginner: duty, breach, causation, and damages, then compare contributory and comparative negligence.` released reel `ingest-450356a265614cd9`. The clip says only `The elements are duty, breach, causation, and damage`, then teaches negligence defenses. Its persisted `intent_evidence` reuses that single agenda sentence for the four separate duty/breach/causation/damages constraints, reports `intent_coverage=1.0`, and labels the clip `primary`, although none of the four elements is explained. The first batch consequently released the defenses clip plus an unrequested assumption-of-risk clip while omitting substantive teaching of all four requested elements.
- Root cause: Gemini's selector/audit evidence contract checks that evidence quotes are literal and attached to grounded constraint IDs, but it does not clearly distinguish substantive fulfillment of an `explain`/`teach` obligation from a title, list, agenda, or passing mention. The organizer trusts the resulting coverage value, so an enumeration can falsely satisfy multiple learning objectives.
- Universal fix: Both the selector schema/prompt and the independent Pro-audit schema/prompt now state that a title, agenda, outline, list, transition, or passing mention cannot fulfill a teach/explain constraint. Each evidence quote must substantively teach that exact item's meaning, mechanism, rule, relationship, or application; one enumeration sentence cannot prove its members were taught. This stays an AI semantic decision grounded in exact transcript quotes. No local teaching-verb heuristic, domain lexicon, provider call, or sleep was added.
- Focused retest: Sorting and negligence agenda/list examples are present in both AI contracts and cannot be represented as substantive per-item evidence under the stated schema. The exact production retest must verify Gemini follows that judgment on real sources; local code deliberately does not replace it with keyword matching.

### AF-129 — Phrase-only obligation IDs can merge distinct request spans

- Status: Fixed locally; exact new-SHA five-domain retest pending
- Severity: High — repeated words or case-sensitive identifiers can make one taught span incorrectly satisfy another requested facet
- Root cause: The obligation key used only the normalized source phrase. Repeated identical text at different positions and case-sensitive pairs such as `C` versus `c` therefore had no stable, distinct identity; Gemini casing or punctuation variation could also produce a noncanonical key for the same literal request span. The first positioned-key repair still used compatibility normalization, which collapsed mathematical `ℂ` into plain `C` in both the key and persisted obligation text.
- Universal fix: Gemini still chooses the semantic constraint, but trusted conversion resolves its phrase to the exact case-preserving slice of the original request and persists `source_start`. The key hashes both NFC-preserved exact text and offset, while normalization rejects missing or inconsistent position metadata without collapsing compatibility symbols.
- Focused retest: Repeated phrases receive different keys; `C`/`c` and `ℂ`/`C` remain distinct; `Merge Sort`/`merge-sort` response variation canonicalizes to the literal user span. The compatibility-symbol obligation preserves source and requirement text through normalization round-trip. Intent/cache/ingestion pass 39 tests plus 3 subtests, and the 69-test lesson-ordering suite passes.

### AF-130 — Greedy mandatory selection can miss a feasible complete lesson

- Status: Fixed locally; exact new-SHA progression retest pending
- Severity: High — an available requested facet or exact confusing/wrong concept can be omitted even though all mandatory coverage fits the release limit
- Reproduction: With a two-reel limit, candidate `x={a,b,c,d}`, `y={a,b,e}`, and `z={c,d,f}`, greedy selection chooses `x+y` and misses `f`; `y+z` covers all six. A separate tie chose a harder standalone remediation over an easier remediation plus its required prerequisite.
- Root cause: Mandatory obligations used greedy set cover, while exact remediation was reconciled separately and dependency-closure size outranked remediation difficulty.
- Universal fix: One bounded exact semantic search jointly optimizes at most sixteen grounded obligation bits plus the strongest exact remediation bit. It preserves dependency and chain closure, maximizes grounded coverage, prefers the easiest exact remediation, then minimizes the complete released dependency closure before using root count as a tie-break. A joint witness therefore displaces redundant siblings, while a necessary prerequisite does not make an easier remediation lose.
- Focused retest: The nongreedy six-facet counterexample selects `y+z`; exact remediation plus a separate obligation coexist; a joint witness wins over redundant exact/obligation clips; and the easier prerequisite-closed remediation wins over the harder standalone clip. Full lesson-ordering suite: 58 passed.

### AF-131 — Continuation history performs database work proportional to chain length

- Status: Fixed locally; exact new-SHA latency retest pending
- Severity: High — long adaptive sessions can delay authoritative output even though provider work is unchanged
- Reproduction: A 40-generation continuation chain issued 198 `SELECT` statements during one pre-final overlap-filter pass.
- Root cause: History traversal repeatedly fetched each generation, its jobs, events, lesson metadata, and reels, then repeated the ancestry walk during authoritative reconstruction.
- Universal fix: A cycle-safe recursive CTE snapshots all anchored ancestry, one ranked bulk query loads authoritative terminal events while preserving the existing 20-job/500-event bounds, and one reel query supplies overlap data. Filtering and teaching-before-recap reconstruction run in memory from that snapshot.
- Focused retest: Two- and forty-generation chains both require at most three history reads and return the same authoritative IDs and overlap decisions. No provider call, sleep, or healthy-path request was added.

### AF-132 — Prior obligation history silently ignores reels after the first 200 IDs

- Status: Fixed locally; exact new-SHA continuation retest pending
- Severity: Medium — a facet taught in an older/larger released chain can become mandatory again and create avoidable repetition
- Root cause: Prior coverage sorted all delivered reel IDs and sliced the first 200 before reading their obligation metadata. The bound was arbitrary and semantic, not a database transport limit.
- Universal fix: Read every authoritative delivered ID in bounded 400-ID query chunks and union all trusted obligation keys. This removes semantic truncation while retaining bounded query parameter sizes.
- Focused retest: A 205-reel history whose only witness is the final lexical ID remains recognized as already taught; narrow prior coverage still does not propagate through broad concept families.

### AF-133 — Exhaustive mandatory-selection search can delay final reel release

- Status: Fixed locally; exact new-SHA latency retest pending
- Severity: Critical — the correctness postcondition runs after the organizer and before the authoritative final event, so combinatorial search directly increases time-to-first-reel
- Reproduction: With 128 legal candidates and sixteen grounded obligations, 128 singleton witnesses (eight per obligation) exceeded eight seconds at release limits five and nine. A 4-of-16 overlap matrix also exceeded eight seconds, and shared-prerequisite candidates took 3.14 seconds at limit five and exceeded eight seconds at limit nine.
- Root cause: The exact branch search memoized the complete selected-reel set and explored every alternative witness combination even after equivalent coverage states and a complete solution existed. Its theoretical obligation bound did not bound the number of candidate combinations.
- Universal fix: The optimizer now compresses candidates with identical semantic, dependency-overlap, and capacity effects. Dependency-free batches use an exact mask DP; dependency-bearing batches use minimum-release-slot complete-cover search plus a dependency-signature DP for exact maximal partial coverage. Both are bounded by the seventeen semantic bits and release capacity, preserve shared prerequisite unions, and stop as soon as a certified optimum is reached. No provider call, heuristic domain vocabulary, or wait was added.
- Focused retest: The original eight adversarial maximum-cardinality cases now complete in 0.030–0.367 seconds. The dense 128-candidate/16-obligation matrix proves all sixteen facets fit in seven clips and completes below the one-second release guard; the five-clip control correctly proves thirteen is maximal. Duplicate-witness, impossible-singleton, and shared-prerequisite matrices remain maximal and bounded.

### AF-134 — Minimum-root coverage can waste release slots on prerequisites

- Status: Reproduced locally during independent universal audit; universal fix implemented and focused regression passed; exact new-SHA progression retest pending
- Severity: High — a complete request can consume an avoidable extra reel and displace the organizer's introductory or worked-example clip
- Reproduction: With two requested facets and a three-reel release limit, one joint witness requires two prerequisite clips while two standalone witnesses need no prerequisites. The optimizer chooses the one-root joint witness's three-reel closure and drops the organizer-selected intro, although the two-clip standalone cover leaves room for it.
- Root cause: Complete-cover search minimizes the number of candidate roots before the number of actual released reel IDs. Dependency closure makes root count an invalid proxy for feed capacity.
- Universal fix: Complete-cover search now iterates actual released-slot capacity before candidate-root count. The shared objective maximizes mandatory coverage, preserves the easiest exact remediation priority required by confusing/wrong feedback, then minimizes the cardinality of the entire dependency-closed release set; root count only breaks a tie after equal educational difficulty and slot use.
- Focused retest: The two standalone witnesses now preserve the organizer-selected intro, while the existing easier exact-remediation-plus-prerequisite case still beats a harder standalone remediation. The full lesson-ordering suite passes: 65 tests in 1.99 seconds, including all three maximum-contract latency guards.

### AF-135 — Coverage-mask deduplication can discard an easier exact remediation

- Status: Fixed locally; exhaustive property and focused regressions passed; exact new-SHA adaptation retest pending
- Severity: High — thumbs-down or wrong-quiz remediation can select a harder explanation even though an easier exact-concept clip fits without reducing requested-facet coverage
- Reproduction: With a two-reel limit, one harder exact clip covers all four obligations while an easier exact clip covers a subset. The true optimum keeps both clips, retaining full coverage with the easier exact witness, but the dependency-free optimizer returns only the harder clip.
- Root cause: The mask DP globally marks an obligation-coverage mask as seen. A later state reaching the same mask with better exact-remediation difficulty is discarded even though its selected identities affect the educational objective.
- Universal fix: Exact remediation is modeled as its own thresholded semantic bit instead of being collapsed into ordinary obligation coverage. The dependency-free solver first proves maximum requested-facet coverage, then binary-searches the easiest exact difficulty that preserves that maximum. A harder clip can still supply its grounded facets without falsely satisfying the exact remediation bit, so an easier exact witness is retained when capacity permits.
- Focused retest: The harder-full/easier-partial counterexample now retains both different-source clips and the easier remediation; same-source overlap is still removed only when the retained witness preserves the mandatory semantic mask. The 69-test lesson-ordering suite and 12,000 randomized exhaustive semantic comparisons pass.

### AF-136 — Root-depth pruning can retain a worse dependency-bearing remediation state

- Status: Fixed locally; exhaustive dependency-DAG audit passed; exact new-SHA adaptation retest pending
- Severity: High — an avoidable prerequisite/dependent clip can consume a feed slot even when a smaller, easier exact-remediation set has identical maximal requested-facet coverage
- Reproduction: An easy exact clip plus a joint A/B witness reaches the same semantic/dependency-capacity signature as a wrapper-root closure containing the harder joint witness. The earlier one-root state prunes the later two-root state solely by depth, and the final release contains the wrapper as an unnecessary third clip.
- Root cause: Dependency-state memoization treats fewer selected roots as dominance even though exact-remediation difficulty and total dependency-closed reel cardinality rank before root count.
- Universal fix: Dependency states retain their exact dependency identities and nondependency capacity, while equivalent states are compared by the same educational dominance order used at final selection: released slots, roots, organizer-preserved exact witness, organizer retention, then stable reel order. Plain candidates are folded through a bounded semantic-mask DP after the dependency frontier, preventing root-depth history from overriding a better remediation set.
- Focused retest: The easy-exact plus joint-witness counterexample displaces the unnecessary wrapper. A 12,000-case brute-force audit, including full and partial dependency DAGs and 7,829 cases where a semantic root can add no new nondependency reel, found zero primary or organizer-semantic mismatches.

### AF-137 — Linear exact-difficulty feasibility search can violate release latency

- Status: Fixed locally; repeated sub-second guards passed; exact new-SHA latency retest pending
- Severity: Critical — mandatory postselection runs synchronously before the authoritative final event and can add seconds without any provider or database delay
- Reproduction: A legal 128-candidate, sixteen-obligation, nine-reel matrix with 34 exact candidates carrying distinct prerequisite subsets took 6.037 CPU seconds. Only the standalone exact candidate plus eight pair witnesses can complete the lesson.
- Root cause: Complete-cover selection reruns the set-cover search once for every infeasible exact-difficulty threshold, making runtime scale linearly with the number of exact alternatives.
- Universal fix: Complete-cover search branches on exact remediation first in increasing difficulty and proves the easiest feasible witness in one exhaustive pass. Selected-slot and root limits use cached monotonic searches, a safe remaining-capacity bound prunes impossible branches, and a cardinality-lower-bound probe skips the dense semantic frontier only when feasibility at that proven minimum is established. No provider call, wait, or healthy-path retry was added.
- Focused retest: The 128-candidate exact/dependency matrix repeatedly releases the standalone easiest feasible exact clip plus all sixteen facets in nine reels below the one-CPU-second guard. The no-exact and unavailable-facet guards also pass repeatedly; the full 69-test lesson-ordering suite passes.

### AF-138 — An unavailable exact-remediation closure can crash mandatory selection

- Status: Fixed locally; unavailable-exact regression and property audit passed; exact new-SHA adaptation retest pending
- Severity: Critical — a valid batch can fail before release when the only exact confusing/wrong-concept clip needs more prerequisite slots than the current release limit
- Reproduction: With a one-reel limit, the exact remediation requires one prerequisite and is therefore ineligible, while a standalone obligation witness remains eligible. The dependency-free solver indexes an empty exact-difficulty list and raises `IndexError`.
- Root cause: Exact availability is derived from all candidate reels, but optimizer difficulty thresholds are derived only from capacity-eligible closures; the empty intersection was not represented.
- Universal fix: The required exact bit and difficulty thresholds are derived only from capacity-eligible options. If no exact closure fits, exact remediation is marked unavailable for this release while the optimizer still maximizes every available grounded obligation; no family-level or lexical substitute is fabricated.
- Focused retest: The one-reel counterexample now releases its available obligation witness without an exception. The exhaustive audit covered 556 unavailable-exact cases and 1,376 unavailable obligation bits with no objective mismatch.

### AF-139 — Incomplete dependency-bearing coverage can exhaust the semantic state space

- Status: Fixed locally; sustained-suite and isolated latency guards passed; exact new-SHA latency retest pending
- Severity: Critical — one unavailable requested facet can add many seconds synchronously before the authoritative final event
- Reproduction: A 53-candidate batch with sixteen obligations, one deliberately ineligible ten-reel facet closure, one dependency-bearing witness, and forty 4-of-15 witnesses took 6.436 CPU seconds at a nine-reel release limit; a 73-candidate variant took 17.178 seconds.
- Root cause: Once complete cover is impossible, the dependency-state DP exhaustively explores through every root depth to prove maximal partial coverage and tie quality after the earlier unsafe early stop was removed.
- Universal fix: When complete cover is impossible, dependency-bearing options are explored only across semantic coverage, dependency identity, and nondependency capacity; dependency-free options are then folded through a 0/1 mask DP. Exact difficulty uses cached monotonic solves, and equivalent masks retain the educationally dominant representative without enumerating root histories. The semantic full-cover frontier remains a correctness fallback only when the cheaper cardinality-bound feasibility proof fails.
- Focused retest: The 53-candidate unavailable-facet matrix proves fourteen available facets are maximal and remains below the one-CPU-second guard under isolated and sustained-suite runs. The dependency/full-cover and dense dependency-free controls pass, and 12,000 exhaustive small cases show zero primary or organizer-semantic mismatches.

### AF-140 — First feasible complete cover can lose organizer-retention ties

- Status: Fixed locally; focused ties, expanded semantic oracle, and latency guards passed; exact new-SHA progression retest pending
- Severity: Medium — two equally complete, equally easy, equally sized lessons can choose a non-organizer clip instead of the organizer's intended clip
- Reproduction: At a two-reel limit, one exact `0.2` clip covers facets 0–3. Facet 4 can be completed either by an organizer-selected nonexact clip or a non-organizer exact clip. Both covers tie through exact presence, all five facets, difficulty, released slots, and roots, but the search returns the non-organizer pair.
- Root cause: Complete-cover feasibility returns the first solution at the proven difficulty/slot/root limits. Per-option branch ordering cannot guarantee the lexicographic organizer-retention and stable-ID tie of the combined selected set.
- Universal fix: After exact difficulty, released-slot count, and root count are proven, bounded feasibility passes first require an organizer-selected exact witness when one can participate in a complete cover, then search descending organizer-retention counts. Constrained states retain organizer identities so memoization cannot collapse a better organizer set. The ordinary dense path stops immediately when its first solution already reaches the provable organizer upper bound; no exhaustive terminal-history walk is added.
- Focused retest: Both the two-reel organizer-count tie and the dependency-bearing organizer-exact tie now retain the organizer's intended exact/companion clips. The 12,000-case oracle found zero primary or organizer-semantic mismatches, and repeated maximum-contract CPU guards pass at 0.56, 0.44, and 0.08 seconds. Exact deployed progression remains pending.

### AF-141 — First-hit partial mask DP can lose organizer-retention ties

- Status: Fixed locally; exact regression, lesson-ordering suite, and exhaustive semantic audit passed; exact new-SHA progression retest pending
- Severity: Medium — when not every requested facet fits, an equally complete partial lesson can discard two organizer-selected clips for a non-organizer alternative
- Reproduction: With three facets and a two-reel limit, two exact/nonexact pairs each cover the maximal two facets at the same exact difficulty and size. The first-hit DP returns a pair retaining one organizer clip although the other pair retains both.
- Root cause: The dependency-free mask DP keeps the first witness for each next mask and immediately returns the first state reaching the coverage upper bound. Neither step compares equal-primary-rank states using organizer retention and stable reel order.
- Universal fix: Each next-mask state now retains the full-rank dominant representative. A depth is evaluated completely, and only then does the solver return the best qualifying state; the earliest qualifying depth still proves minimum released slots and roots without reopening later-depth histories.
- Focused retest: The exact three-facet/two-reel organizer tie now retains both organizer choices. The 69-test lesson-ordering suite, 12,000-case semantic oracle, and all maximum-contract latency guards pass; exact deployed progression remains pending.

### AF-142 — Option compression can discard a globally preferable organizer exact clip

- Status: Fixed locally; exact compression regression, expanded semantic oracle, and latency guards passed; exact new-SHA progression retest pending
- Severity: Medium — the final complete lesson can use a non-organizer exact clip even though an organizer-selected exact clip preserves the same global remediation difficulty, coverage, and release capacity
- Reproduction: One `0.1` exact clip supplies facet 0. Two singleton exact clips supply facets 1–2: non-organizer difficulty `0.1` and organizer difficulty `0.2`. Once the first clip fixes global exact difficulty at `0.1`, either second clip ties on every primary objective, but pre-search compression keeps only the standalone-easier non-organizer option.
- Root cause: Option compression keys semantic mask, dependency identity, and capacity, then chooses by standalone selection quality. It omits organizer identity, whose contribution depends on which other clips are already selected and therefore cannot be safely reduced to standalone difficulty.
- Universal fix: The compression signature now includes the complete organizer-selected identity bitset in each closure. Semantically/capacity-equivalent non-organizer alternatives remain compressed, while distinct organizer contributions survive for the bounded organizer-exact/count feasibility passes.
- Focused retest: The two-reel exact reproducer now selects the `0.1` foundation plus the `0.2` organizer exact witness, not the semantically identical non-organizer alternative. The 12,000-case oracle found zero primary or organizer-semantic mismatches, and all three maximum-contract latency guards pass. Forty-seven index-only ties selected educationally and organizer-equivalent IDs and are non-semantic; deployed progression remains pending.

### AF-143 — Empty-topic joint intent can reference an unknown constraint and crash validation

- Status: Fixed locally; focused schema/retry and full selector-contract suite passed; exact new-SHA generation retest pending
- Severity: High — an otherwise schema-valid Gemini selector response can raise `KeyError` instead of producing a typed contract rejection that the selector step can retry
- Reproduction: For unfiltered URL/source generation (`topic=""`), return a request intent whose joint structure names a member or relation ID absent from its constraints. Nonempty topics reject this semantically, but the empty-topic path builds joint signatures by direct dictionary indexing and crashes.
- Root cause: `_RequestIntent` validates unique constraint and joint IDs but not referential integrity. The live empty-topic path intentionally skips request-grounding validation, leaving the unknown reference unguarded at signature construction.
- Universal fix: `_RequestIntent` model validation now enforces every joint member and relation ID against the same response's declared constraint-ID set. Invalid Gemini output therefore becomes the same typed schema rejection on filtered and unfiltered paths, preserving the selector's existing bounded retry behavior instead of reaching signature construction.
- Focused retest: Unknown member and unknown relation responses both fail model validation; the valid unfiltered synthetic-request control and repeated-invalid-schema retry control pass. The complete production selector-contract suite passes: 711 tests in 4.13 seconds.

### AF-144 — Local relationship keywords can override Gemini's semantic intent graph

- Status: Fixed locally; cross-domain selector regressions passed; exact new-SHA retest pending
- Severity: High — honest single concepts containing words such as `comparison` or `contrast` can be rejected repeatedly even though Gemini correctly returns one subject and no joint relationship
- Reproduction: `Explain the comparison principle in elliptic PDEs` and `Explain contrast dye in MRI` each return a grounded single SUBJECT with `joint_structures=[]`. Local request regexes classify both as multi-member comparisons and return `intent_contract_incomplete_joint_structure`.
- Root cause: The new explicit Gemini relationship graph remains gated by legacy prose classifiers that decide whether a graph must exist and whether a supplied relation span contains a locally recognized comparison marker. This duplicates semantic interpretation outside the AI contract.
- Universal fix: Local validation now checks only referential integrity, allowed relation roles, selected source-occurrence topology, and exact grounded evidence IDs for structures Gemini supplies. Whether the request contains a comparison, transition, or ordered relationship remains Gemini's semantic decision under the selector and independent-auditor prompts; no local relationship keyword decides that contract.
- Focused retest: Comparison-principle and contrast-dye single-concept controls pass without a graph. Biology, economics, AP Statistics, math, software, and law graph controls cover incident relationship nouns, multi-sentence/pronominal comparisons, adjacent definitions, invalid references/topology, and bounded retry behavior. Three stale direct-conversion regressions were reconciled with the AI-authority contract: graph fixtures now supply every AI-declared evidence ID, one deliberately reused AI-tagged quote is no longer locally reclassified, and plural/singular semantics no longer call the removed matcher. Live boundary regressions pass 20 tests, and the complete production selector-contract suite passes 731 tests in 3.74 seconds.

### AF-145 — Reordered repeated phrases can swap persisted obligation identity

- Status: Fixed locally; strict schema, dispatch-retry, and order-invariance regressions passed; exact new-SHA retest pending
- Severity: Critical — equivalent Gemini intent contracts can assign different durable concept-obligation keys to the same semantic requirement, breaking prior coverage and future thumb/quiz adaptation
- Reproduction: For `Teach C, then test C.`, constraints `[teach, test]` assign the first `C` span to teach and the second to test. Returning the identical constraints in `[test, teach]` order swaps those spans and therefore swaps the generated obligation keys.
- Root cause: Repeated exact source phrases were disambiguated by claiming the first unclaimed request span in constraint-list order, although the Gemini prompt declares constraint order irrelevant. The first fix added occurrence identity but left three universal holes: the semantic duplicate guard still case-folded `C` and `c`, Pydantic coerced booleans/floats/strings into integer occurrence indexes, and exact-request equality case-folded the user's copied request so a model could swap `C` and `c` before binding the occurrences.
- Universal fix: Every Gemini constraint now declares a strict nonnegative integer `source_occurrence`. Trusted conversion first compares the copied request with the shared semantic-token identity, which folds ordinary prose case but preserves structural case and compatibility math symbols, then resolves the numbered, mechanically grounded phrase occurrence to the exact case-preserving request slice and character offset before semantic deduplication; signature and persistence identity include that canonical position. Array order cannot assign spans, coercive types trigger the existing structured-response retry, and no local requirement-prose heuristic is used. Primary selector telemetry advances to `pro_boundary_v22`.
- Focused retest: Forward and reversed arrays produce identical live selector signatures and exact span/key mappings, including a leading multibyte emoji. Missing, repeated, out-of-range, boolean, float, and string occurrence claims fail closed; swapped structural-case and compatibility-symbol-collapsed requests are rejected before they can reverse the user's semantics, while harmless prose capitalization remains a healthy one-dispatch contract. A malformed first response dispatches exactly one immediate repair and recovers on its second dispatch; the healthy control dispatches once, and both paths prove there is no sleep. `Compare C and c.` retains two positioned members. The complete production selector-contract suite passes: 731 tests in 3.74 seconds.

### AF-146 — An exact reel ID can be re-released by a continuation batch

- Status: Fixed locally; exact-ID and full generation-job regressions passed; exact new-SHA retest pending
- Severity: High — the strongest possible duplicate can survive into a subsequent batch even though the parent reel was already authoritatively released
- Reproduction: An authoritative parent release contains `same-reel`; a child continuation returns another reel dictionary with `reel_id=same-reel`. The continuation temporal-overlap filter returns the child instead of removing it.
- Root cause: The overlap input de-duplicates `prior_reel_ids + current_ids` before filtering, then converts retained IDs to a set. The retained parent ID is therefore indistinguishable from permission to return the child dictionary with the same ID.
- Universal fix: Remove exact parent IDs from the current child-ID mapping before interval overlap evaluation. Preserve the old behavior for malformed reel dictionaries without IDs, and return an empty child batch when every valid child ID was already released.
- Focused retest: The exact parent/child ID regression, containing-span regression, bounded continuation-history query controls, organizer-prefilter capacity case, and all-overlap exhaustion case pass. The complete generation-job API suite passes: 189 tests in 2.51 seconds.

### AF-147 — Continuation overlap filtering occurs after organizer selection

- Status: Fixed locally; exact new-SHA full-suite and live continuation retests pending
- Severity: Critical — an already covered child clip can consume organizer capacity or satisfy a mandatory remediation/intent obligation, then disappear after ordering while a novel eligible alternative remains unselected
- Reproduction: A parent releases clip A. The child candidate pool contains same-source overlapping sibling B followed by novel alternative C. A capacity-limited organizer selects B; the post-order overlap filter deletes B and releases nothing, although C was available.
- Root cause: Fresh lesson ordering received the unfiltered continuation candidate pool. Temporal overlap removal was a release postcondition only, so Gemini and the mandatory-selection optimizer made decisions over candidates that could never surface.
- Universal fix: Filter prior-release exact IDs and temporal overlaps before computing the lesson-order prompt and mandatory postconditions. The organizer's output is then defensively intersected with that one surfaceable candidate snapshot without repeating database queries; stored-order replay retains its independent temporal filter.
- Focused retest: The tight parent-A/overlapping-B/novel-C regression now shows only C reaches the organizer and C is released. The larger bounded candidate-window, exact-ID duplicate, and containing-span controls pass together (4 tests in 2.77 seconds).

### AF-148 — The bounded organizer history can omit the only adaptive signal

- Status: Fixed locally; full optimizer and prompt-budget regressions passed; exact new-SHA and live thumb/quiz retests pending
- Severity: Critical — a thumbs-down or quiz-wrong concept can disappear from Gemini's organizer context solely because its stable concept key sorts after forty unrelated prior concepts
- Reproduction: Supply forty valid zero-signal prior concepts followed by a signaled prior concept whose family/title semantically matches a current candidate under a new concept ID. The old loop stops before computing its learner signal, so neither the adaptive signal nor its semantic bridge reaches the organizer.
- Root cause: `prior_concept_coverage` applies its forty-row cap in caller order before learner-signal lookup or candidate relevance. Main's coverage order is stable identity order, not adaptive importance.
- Universal fix: Normalize the complete bounded history first, then deterministically prioritize nonzero thumb/quiz signals, current candidate exact IDs, highest delivery frequency, and stable identity before retaining the existing forty-row/token cap. This changes no provider count or prompt-size ceiling.
- Focused retest: The 42-row regression retains the signaled old-ID semantic bridge first and the current exact-ID row second while remaining capped at forty. The full lesson-ordering suite passes: 69 tests in 1.77 seconds, including exhaustive semantic and latency guards.

### AF-149 — Case-sensitive one-letter concepts share adaptive history

- Status: Fixed locally; full shared-token/query/cache, exact new-SHA, and live C/c adaptation retests pending
- Severity: Critical — feedback or quiz mastery for one symbolic concept can suppress or amplify a distinct concept that differs only by meaningful case
- Reproduction: `normalize_clip_concept("C")` and `normalize_clip_concept("c")` produced the same normalized key. `ensure_clip_concept` therefore reused one concept ID/title, and thumbs or quiz outcomes leaked between the uppercase and lowercase identifiers even though positioned intent obligations distinguished them. The initial ASCII-only repair still merged Unicode mathematical pairs such as `Δ/δ`, while the selector's older case-folded semantic dedup rejected `Compare C and c.` before either concept could persist.
- Root cause: The shared structural-case tokenizer treated every one-letter alphabetic token as case-insensitive, and the selector independently repeated the same case-folded assumption before canonical request positions were available. That assumption is safe for prose articles/pronouns but false for identifiers, variables, matrix notation, genetics, programming symbols, and cased non-ASCII mathematical alphabets.
- Universal fix: The shared structural-case token rule now preserves any standalone single uppercase alphabetic Unicode token except ordinary one-letter prose stop words (`A` and `I`). The selector validates and canonicalizes each exact request occurrence before duplicate checking and includes the case-preserving text plus character offset in identity. Lowercase identifiers and ordinary title capitalization retain their prior behavior; attached language suffixes such as `C++` and `C#` retain their established normalization.
- Focused retest: C/c, X/x, and Δ/δ remain distinct; A/a and I/i remain equivalent; existing notation and suffix controls pass. `ensure_clip_concept` persists distinct C/c and Δ/δ IDs. Helpful-uppercase, confusing-lowercase, and wrong-quiz-lowercase signals remain isolated with adjustments `+0.04`, `-0.06`, then `-0.18` only for the lowercase concept. The shared token/adaptation matrix passes 37 tests in 0.68 seconds, and the upstream selector cases pass inside the 731-test contract suite.

### AF-150 — An all-overlap continuation can persist and activate an empty order

- Status: Fixed locally; full generation-job, exact new-SHA, and live continuation retests pending
- Severity: Critical — a continuation whose entire apparent inventory was already covered can persist `ordered_reel_ids=[]`, activate an empty generation, and prevent later inventory from receiving fresh adaptive ordering
- Reproduction: The outer release branch is entered because database inventory exists, but the new pre-organizer overlap filter removes every child candidate. The old flow still calls `order_lesson_batch([])`, writes empty v2 lesson metadata, and sets `activate_generation=True`.
- Root cause: Branch eligibility was computed before the surfaceability postcondition and was not reevaluated after overlap filtering.
- Universal fix: Materialize and prefilter the candidate pool before entering the ordering/release branch. An empty pool now uses the existing partial/exhausted path, performs no organizer call, writes no lesson order, and never activates the child; stored-order replay behavior is unchanged.
- Focused retest: The all-overlap worker regression ends exhausted with an empty authoritative payload, a failed non-activated child generation, no lesson-order metadata, and zero organizer/activation calls. It passes with the capacity, larger-window, and exact-ID continuation controls (4 tests in 1.84 seconds).

### Release constraint — healthy-path reel streaming latency

- Requirement: These fixes must not increase time-to-first-ready or subsequent reel streaming cadence on successful requests.
- Guardrail: Healthy provider/database paths remain one physical attempt with no added sleeps; retries run only after a recoverable failure inside the background worker, and already-ready reels remain visible while later work retries.
- Pre-deployment baseline: On deployed revision `2774a76`, clean Newton job `15584162-e4a5-4108-93e8-7bdfb9f3ec13` was created at `19:11:49.668260Z` and emitted its seven-reel final event at `19:13:44.980606Z`: 115.312 seconds from job creation (115.254 seconds from worker start). This revision emits no earlier partial reel event for that job.
- Verification: Pending the identical Newton request on the exact new deployed revision, comparing job creation/start to first reel-bearing event and terminal batch cadence.

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
- Scoped production/test/log `git diff --check`: passed after the final graph rebuild. The generated, unstaged `graphify-out/GRAPH_REPORT.md` still contains generator-emitted trailing spaces and is excluded from the release diff.
- The only excluded backend test file was `backend/tests/test_labels_api.py`; collection imports the unrelated standalone `backend/main.py` and the local environment lacks `sse_starlette`. The active runtime under review is `backend.app.main`, and every active-backend/adaptive test collected and passed.
- Canonical-only Gemini/Pro selector contract after AF-085 through AF-096: 652 passed in 3.84 seconds.
- Canonical identity, persistence, adaptation, segment-cache, and organizer regression matrix: 372 passed, 3 subtests passed in 4.24 seconds.
- Current complete backend collection: 4,037 passed, 1 skipped, 40 subtests passed; the sole failure is the explicitly out-of-scope practice-fast test's stale cache-version assertion (`7` versus current `8`). No production or non-practice failure remains.
- Current non-practice backend collection: 3,956 passed, 1 skipped, 40 subtests passed in 51.00 seconds.
- Healthy-path latency guards: Gemini dispatch performs one physical attempt with no retry sleep, and the exact F=ma Pro audit performs one audit dispatch with no contract retry (2 focused tests passed).
- Current frontend suite: 189 passed; the adaptive refresh regression proves a successful request remains one call while only transient failures receive one identical retry.
- Current TypeScript check: passed.
- Current production Next.js build: passed.
- Final complete backend suite after AF-097/AF-098: 4,038 passed, 1 skipped, 40 subtests passed in 46.66 seconds; no failure remains.
- Final AF-099 through AF-102 changed-path gate: 928 passed; independent review GO with no domain-specific branch, new healthy-path call, or new sleep.
- Final expansion/retrieval contract group: 115 passed; the real v11 provider matrix passed physics, biology, math, software, and law on the first call.
- Final complete backend suite after AF-102: 4,047 passed, 1 skipped, 40 subtests passed in 52.17 seconds.
- Final healthy-path latency guards: 4 passed; expansion, Gemini dispatch, and selector success paths remain one physical attempt with no retry wait.
- Current frontend suite after the backend-only AF-099 through AF-102 patch: 191 passed; TypeScript check and production Next.js build passed.
- Final AF-103 through AF-112 application/API collection: 2,677 passed, 1 skipped, 44 subtests passed in 25.42 seconds.
- Final secondary pipeline/adapters/evaluation collection: 1,404 passed in 6.83 seconds.
- Final explicit five-domain/healthy-path release guard: 12 passed in 1.82 seconds, covering physics, biology, math, software, and law intent constraints; one-call F=ma audit; one-dispatch/no-sleep 128-candidate organizer output; genuine raw-reservoir no-top-up; and unchanged one-pass 128-candidate worker acquisition.
- Independent AF-110 prompt/persistence audit: GO. The adversarial 128-candidate plus 40-prior-concept prompt is 63,835 characters, every semantic excerpt is non-empty, raw internal IDs are absent, the legal response stays below the 10,240-token ceiling, and the eight-test independent stack passes. Worst-case local prompt construction averaged about 28 ms; live Gemini duration remains part of the exact-SHA production TTFR gate.
- Final frontend suite: 191 passed. TypeScript check and production Next.js build passed.
- Final AF-113 through AF-116 affected collections: provider runtime 52 passed; durable repository 43 passed; generation worker/API 176 passed; mixed-source pipeline 159 passed. The independent 30-test retry matrix and four healthy-path one-call/no-sleep guards pass.
- Final complete backend suite after AF-113 through AF-116: 2,701 passed, 1 skipped, 44 subtests passed. Frontend: 195 passed; TypeScript check and production Next.js build passed.
- Final complete backend suite after AF-117 through AF-121: 4,172 passed, 1 skipped, 44 subtests passed in 31.25 seconds. An initial run exposed five test-only SQLite fixtures missing `reels.search_context_json`; adding that production-schema column to the shared fixture made the exact five failures pass before the clean full rerun.
- Final frontend suite: 195 passed. TypeScript check and production Next.js build passed.
- Final healthy-path latency guards: 4 passed in 0.76 seconds. AI query expansion, organizer success, selector success, and Pro audit success each remain one physical call with no retry wait; organizer failover remains confined to its existing second attempt, while the durable job ceiling is three total attempts.
- Independent final diff/adversarial audit: GO. Its last 108 focused checks cover fake multi-facet enums, four-domain list contracts, technical slash notation, grounded slash comparison, terminal-`so` positives/negatives, v3 stale-signal gates, compact organizer alignment, distinct Lite/Flash failover, and one-call healthy behavior.
- Final AF-122 through AF-142 adaptive/selection matrix: 1,166 passed, 3 subtests passed in 12.62 seconds.
- Final lesson-ordering suite: 69 passed. The three repeated maximum-contract CPU guards measured 0.56, 0.44, and 0.08 seconds, all below one second.
- Independent final optimizer audit: GO on 12,000 exhaustive randomized cases with zero primary or organizer-semantic mismatches. Forty-seven index-only ties were equivalent on exact presence, requested coverage, difficulty, released slots, roots, organizer-exact retention, and organizer count.
- Final complete backend suite: 2,865 passed, 1 skipped, 44 subtests passed in 44.74 seconds.
- Final frontend suite: 208 passed. TypeScript check and production Next.js build passed.
- Final exact-index AF-143 through AF-150 changed-path gate: 1,202 passed, 3 subtests passed in 9.48 seconds before the two stale cross-collection contract tests were reconciled; all 20 live-boundary, 26 expansion, and 12 AP Statistics focused regressions then passed.
- Final exact-index complete backend suite after AF-143 through AF-150: 2,831 passed, 1 skipped, 44 subtests passed in 30.91 seconds. Independent cached-diff audit: GO; production code includes the exact-ID and pre-organizer continuation filters, healthy success paths add no dispatch or sleep, and no Goals, billing, account, auth, or frontend change is present in the release index.
- Final exact-index frontend verification: 184 passed; TypeScript check passed; the production Next.js build passed after replacing the isolated test tree's external dependency symlink with a real dependency copy (the first Turbopack attempt rejected only that test harness symlink).
