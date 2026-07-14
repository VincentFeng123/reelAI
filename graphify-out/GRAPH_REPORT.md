# Graph Report - .  (2026-07-14)

## Corpus Check
- 388 files · ~1,063,079 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 6626 nodes · 17980 edges · 205 communities detected
- Extraction: 65% EXTRACTED · 35% INFERRED · 0% AMBIGUOUS · INFERRED: 6220 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## God Nodes (most connected - your core abstractions)
1. `Sentence` - 206 edges
2. `ReelService` - 198 edges
3. `Unit` - 180 edges
4. `GenerationContext` - 159 edges
5. `CancellationError` - 139 edges
6. `YouTubeService` - 132 edges
7. `DatabaseIntegrityError` - 130 edges
8. `ProviderError` - 127 edges
9. `MediumRegressionTests` - 104 edges
10. `Structure` - 101 edges

## Surprising Connections (you probably didn't know these)
- `Return [{text, start(sec), end(sec)}] ordered by time.` --uses--> `PipelineError`  [INFERRED]
  backend/supadata_client.py → backend/app/clip_engine/clipper/errors.py
- `_BoundaryRepairCandidate` --uses--> `Sentence`  [INFERRED]
  backend/pipeline/gemini_segment.py → backend/pipeline/sentences.py
- `Gemini 3.5 single-pass prompt: policy/examples, context, task last.` --uses--> `Sentence`  [INFERRED]
  backend/pipeline/gemini_segment.py → backend/pipeline/sentences.py
- `Render only the neighboring cue windows needed to repair dirty edges.` --uses--> `Sentence`  [INFERRED]
  backend/pipeline/gemini_segment.py → backend/pipeline/sentences.py
- `Pre-router production prompt used by ``production_pro_v0``.` --uses--> `Sentence`  [INFERRED]
  backend/pipeline/gemini_segment.py → backend/pipeline/sentences.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.01
Nodes (489): _arc_verify_enabled(), ArcCandidate, ArcCheckLLM, ArcVerifyLLM, _concepts(), detect_arcs(), _pair_practice_prompts(), Deterministic instructional-arc detection (Wave 2 P3a).  Scans the time-ordered (+481 more)

### Community 1 - "Community 1"
Cohesion: 0.04
Nodes (364): AssessmentCancelledError, AssessmentService, _BackfillPlan, Adaptive recall-check persistence, readiness, and session lifecycle., Promote one named level after sustained, broad mastery evidence.          Manual, Validate and persist one private answer-bearing question for a reel., Choose a stable 2-5 reel cadence, then adapt it to current evidence., Record one distinct forward navigation without changing watch analytics. (+356 more)

### Community 2 - "Community 2"
Cohesion: 0.01
Nodes (209): _await_with_probe(), is_cancelled(), raise_if_cancelled(), Small async-to-sync bridge for actively cancellable provider requests.  The publ, Run an async request from synchronous pipeline code.      Normal generation work, run_cancellable(), sleep_with_probe(), wait_with_probe() (+201 more)

### Community 3 - "Community 3"
Cohesion: 0.02
Nodes (52): BaseSettings, get_settings(), Settings, _difficulty(), _get_punct_pipeline(), _has_selection_contract(), _importance_ranker_enabled(), _is_short_leaf_topic() (+44 more)

### Community 4 - "Community 4"
Cohesion: 0.03
Nodes (218): _activate_generation(), _adaptive_min_relevance_floor(), answer_assessment(), _auto_verify_community_account_if_allowed(), _bare_video_id(), _build_allowed_origins(), _build_generation_head_id(), can_generate_reels() (+210 more)

### Community 5 - "Community 5"
Cohesion: 0.02
Nodes (159): _artifact_path(), _chunk_path(), load_artifact(), load_chunk(), On-disk caching for punctuation.  Two levels, both under ``work/<video_id>/``: -, save_artifact(), save_chunk(), transcript_fingerprint() (+151 more)

### Community 6 - "Community 6"
Cohesion: 0.02
Nodes (149): _energy_min_snap(), _ensure_audio(), _gap_after(), _gap_before(), _pick_end(), _pick_start(), Precise boundary refinement with targeted Whisper.  Supadata gives fast but coar, Absolute time of the lowest-RMS ``frame_ms`` frame within ``[a, b]`` — the quiet (+141 more)

### Community 7 - "Community 7"
Cohesion: 0.02
Nodes (163): _jsonable(), Run-artifact persistence (W25-G): every assembled run leaves an auditable trail., Best-effort JSON projection: dataclass → asdict, pydantic → model_dump,     dict, Persist one assembled run's plan/arcs/shipped/ledger under     ``<work_dir>/<vid, write_run_artifacts(), best_match(), find_human_clip(), gold_chapters() (+155 more)

### Community 8 - "Community 8"
Cohesion: 0.03
Nodes (161): _apply_enrichment(), _Assessment, _AssessmentDraft, _authoritative_pro(), _boundary_prompts(), _boundary_repair_prompts(), _BoundaryPlan, _BoundaryRepairCandidate (+153 more)

### Community 9 - "Community 9"
Cohesion: 0.03
Nodes (52): _normalise_cue(), Shared transcript quality validation.  This module is intentionally dependency-f, Check whether a transcript adequately covers a video.      Parameters     ------, Result of :func:`validate_transcript`., Convert any supported cue format to ``(start, end, text)``.      Returns ``None`, TranscriptQuality, validate_transcript(), _backoff_delay() (+44 more)

### Community 10 - "Community 10"
Cohesion: 0.03
Nodes (80): _assessment(), Strict shipping contract for the guarded Gemini educational selector., _run(), _segs(), _selection_task_tail(), test_adjacent_facets_inside_one_coarse_cue_remain_distinct(), test_ambiguous_projected_edge_quote_is_rejected(), test_apostrophe_typography_is_grounded_and_source_spelling_is_preserved() (+72 more)

### Community 11 - "Community 11"
Cohesion: 0.04
Nodes (105): FakeAdapter, AnchorAdapter, _built_budget(), _run_repair(), test_closure_budget_clamped_to_ship_cap(), test_closure_budget_smaller_closure_span_wins(), test_repair_expansion_capped_at_ship_cap(), test_repair_expansion_smaller_closure_span_wins() (+97 more)

### Community 12 - "Community 12"
Cohesion: 0.03
Nodes (94): extract_concepts(), _extract_concepts_via_llm(), Extract higher-quality concepts via Gemini (falling back to Groq)., _summary_for_terms(), _adapt_query_for_postgres(), _connect_postgres_with_retry(), _database_url(), _db_path() (+86 more)

### Community 13 - "Community 13"
Cohesion: 0.04
Nodes (71): _anchor(), _run_gate(), _sent(), test_card5_no_introducer_no_rescue_kill_stands(), test_card5_no_suppress_when_subject_absent(), test_card5_seeds_from_introducer_when_not_in_referential(), test_card5_suppress_when_first_sentence_names_anchor_concept(), test_card5_suppress_when_first_sentence_names_topic() (+63 more)

### Community 14 - "Community 14"
Cohesion: 0.02
Nodes (2): MediumRegressionTests, _validated_query_plan()

### Community 15 - "Community 15"
Cohesion: 0.03
Nodes (82): crop_clip(), crop_clip_local(), crop_highlights(), crop_highlights_local(), _cut_subclip(), _ratio(), Local clipping: ffmpeg subclip + OpenCV face-aware vertical crop.  Two stages pe, Submit one autocrop job and return the URL of the rendered short. (+74 more)

### Community 16 - "Community 16"
Cohesion: 0.07
Nodes (84): answerAssessmentQuestion(), ApiError, apiUrl(), askStudyChat(), buildApiError(), buildGenerateReelsRequestBody(), cancelGenerationJob(), changeCommunityPassword() (+76 more)

### Community 17 - "Community 17"
Cohesion: 0.04
Nodes (44): block_emb(), make_sents(), n sentences, `sec` seconds each; a `gap`-second pause BEFORE each index in gap_a, Unit vectors; block j points along axis j%dim → maximal between-block scatter at, _stub_stages(), test_none_content_map_builds_internally(), test_precompute_vs_internal_structures_identical(), test_precomputed_content_map_skips_internal_build() (+36 more)

### Community 18 - "Community 18"
Cohesion: 0.05
Nodes (67): embed_url(), Canonical YouTube embed-URL helper — the single source of truth shared by the se, YouTube embed URL that plays only [start, end] (whole seconds: floor start / cei, assemble_accepted(), _clip_range(), corrupt_antecedent_removal(), corrupt_chop_end(), corrupt_chop_start() (+59 more)

### Community 19 - "Community 19"
Cohesion: 0.08
Nodes (52): _discovery(), _one_cue_selector_result(), _pipeline(), _pipeline_with_semantic(), _plan(), _quality_clip(), test_acoustic_boundary_plan_fails_closed_for_missing_cue_ids(), test_acoustic_failure_is_diagnostic_only_and_never_persisted_or_emitted() (+44 more)

### Community 20 - "Community 20"
Cohesion: 0.05
Nodes (71): aggregate_usage(), calls_cost_usd(), _exception_row(), _first(), _hybrid_row(), _identity(), _is_number(), load_pricing_snapshot() (+63 more)

### Community 21 - "Community 21"
Cohesion: 0.04
Nodes (32): _prepared(), A source/caption edge is not proof of an in-cue semantic handoff., test_background_noise_above_required_threshold_is_unavailable(), test_backward_search_stitches_short_quiet_fragments_across_window_seam(), test_caption_handoff_does_not_scan_an_entire_inter_caption_gap(), test_caption_handoff_never_moves_cuts_inside_required_speech(), test_caption_handoff_never_skips_sound_before_a_late_start_quiet_run(), test_caption_handoff_observation_accepts_only_the_straddling_quiet_run() (+24 more)

### Community 22 - "Community 22"
Cohesion: 0.07
Nodes (43): _clip(), _private_clip(), _report(), _result(), _segments(), test_all_authoritative_pro_calls_use_selected_baseline(), test_boundary_profile_rejects_bad_model_quote_from_cited_cue(), test_cancelled_worker_never_publishes_late_boundary_or_done_progress() (+35 more)

### Community 23 - "Community 23"
Cohesion: 0.1
Nodes (42): _conn(), _insert_generation_reel(), _patch_request_context(), _set_reel_boundary_state(), _terminal_job_for_generation(), test_authoritative_job_inventory_drops_candidates_absent_from_final_rank(), test_completed_job_status_and_replay_drop_currently_invalid_candidate(), test_cross_level_reservoir_rejects_unverified_or_implicit_surface_rows() (+34 more)

### Community 24 - "Community 24"
Cohesion: 0.08
Nodes (38): _audio_bitrate(), _audio_entry(), AudioPreparationResult, _canonical_watch_url(), _decode_window(), _EdgeSearchResult, _end_search_windows(), _ffmpeg_headers() (+30 more)

### Community 25 - "Community 25"
Cohesion: 0.06
Nodes (40): get_ingest_logger(), log_event(), new_trace_id(), Structured logging for the ingestion pipeline.  Every log line emitted from inge, Injects the current trace_id into every log record., Return a logger with the trace_id filter attached.      We install the filter id, Generate a fresh trace id. Callers store it via `set_trace_id()`., Set the current trace id for subsequent log calls in this task.      If `trace_i (+32 more)

### Community 26 - "Community 26"
Cohesion: 0.06
Nodes (26): buildCommunitySetInformationParagraphs(), createDraftReelRow(), detectYouTubeDurationWithIframeApi(), draftRowsFromReels(), extractYouTubeVideoId(), formatCommunityPlatformSummary(), formatCompact(), formatLastEditedLabel() (+18 more)

### Community 27 - "Community 27"
Cohesion: 0.13
Nodes (42): RuntimeError, _call_v3(), _ConstrainedSchema, _enum_value(), _FakeClient, _FakeModels, _FakeResponse, _HTTPError (+34 more)

### Community 28 - "Community 28"
Cohesion: 0.08
Nodes (14): _blend_engine_out(), _build_engine_out(), _clip_side_effect(), _difficulty_engine_out(), DifficultyPersistenceTests, EmbedUrlCeilTests, _fractional_engine_out(), IngestTopicProgressTests (+6 more)

### Community 29 - "Community 29"
Cohesion: 0.09
Nodes (43): _by_id(), test_arc_ids_sequential(), test_bus_problem_slice_from_audit_yields_two_arcs(), test_closer_before_steps_does_not_close_the_arc(), test_closer_between_steps_does_not_truncate_the_arc(), test_closer_exactly_at_the_gap_bound_is_accepted(), test_closer_terminal_accepted_only_after_steps(), test_closer_without_steps_is_not_a_terminal() (+35 more)

### Community 30 - "Community 30"
Cohesion: 0.06
Nodes (9): test_boundary_plan_rejects_a_quote_removed_with_a_forward_setup(), test_candidates_are_validated_independently_without_model_repair(), test_clean_fast_path_never_dispatches_boundary_repair(), test_dirty_edges_use_only_the_one_low_thinking_selector_call(), test_no_boundary_repair_is_attempted_after_selector_validation(), test_plan_rejects_live_biology_cue_despite_later_framing_sentence(), test_plan_rejects_long_unpunctuated_biology_cue_with_late_framing(), _topic() (+1 more)

### Community 31 - "Community 31"
Cohesion: 0.08
Nodes (37): A keyframe-anchored span of the video (from ffmpeg scene cuts ∪ a uniform grid)., Scene, VisualEvent, available(), merge_into_events(), ocr_keyframes(), OcrBlock, Optional local OCR of keyframes (default OFF — ``OCR_ENGINE="none"``).  Gemini-v (+29 more)

### Community 32 - "Community 32"
Cohesion: 0.08
Nodes (14): ClipEngineGenerateReelsTests, _discover_result(), _five_minute_engine_out(), LevelAwareFeedTests, _multi_clip_engine_out(), _quality_v2_engine_out(), Tests for the clip-engine-routed ReelService.generate_reels (Task T4).  The lega, Serve-time level scoring: matched clips first, off-level kept at the     back, a (+6 more)

### Community 33 - "Community 33"
Cohesion: 0.12
Nodes (39): _collapse_fixture(), _det(), _passing_judge(), W25-F replace bar: a ship-flagged newcomer (hard core failed, phantom evidence), _rel(), _same_model(), _structure(), _superset_fixture() (+31 more)

### Community 34 - "Community 34"
Cohesion: 0.14
Nodes (30): _assessment(), _convert(), _plan(), _segs(), test_bad_line_indices_are_rejected_instead_of_clamped(), test_boundary_repair_without_explicit_clip_limit_does_not_compare_none(), test_chemistry_fast_path_closes_two_cues_of_missing_setup(), test_clip_limit_uses_overall_quality_before_stable_chronology() (+22 more)

### Community 35 - "Community 35"
Cohesion: 0.13
Nodes (35): BaseAdapter, _by_id(), _mk_cand(), role_units(), _same_model(), test_anchor_role_still_reported_in_spec_payload(), test_calculation_answer_satisfies_solution_contract(), test_calculation_as_final_binds_result_not_procedure() (+27 more)

### Community 36 - "Community 36"
Cohesion: 0.12
Nodes (25): _mk_row(), _sent(), _sents(), _spec(), _structure(), test_antecedent_removal_removes_source_unit_sentences(), test_antecedent_removal_skips_when_no_references(), test_antecedent_removal_skips_when_removal_would_empty_clip() (+17 more)

### Community 37 - "Community 37"
Cohesion: 0.17
Nodes (32): AnchorAdapter, _count_judge(), _flagged_at_repair_then(), _mk_candidate(), _one_unit_structure(), _prereq_judge(), _prereq_settings(), _prereq_structure() (+24 more)

### Community 38 - "Community 38"
Cohesion: 0.12
Nodes (32): _dd(), _hard_core_partial(), _mk_cand(), _node_units(), _over_inclusion(), _pass(), _script(), _snap() (+24 more)

### Community 39 - "Community 39"
Cohesion: 0.06
Nodes (3): _FakeGeminiClient, test_failed_expansion_dispatch_is_recorded_once(), test_successful_expansion_dispatch_is_not_double_recorded()

### Community 40 - "Community 40"
Cohesion: 0.05
Nodes (1): Discourse-onset primitive — text-only, offline. Decides whether a sentence used

### Community 41 - "Community 41"
Cohesion: 0.14
Nodes (1): Do not turn a transient all-provider fallback into a 24-hour result.

### Community 42 - "Community 42"
Cohesion: 0.11
Nodes (33): _proposal(), test_articleless_look_at_visual_noun_remains_visual_dependent(), test_bare_look_at_this_remains_visual_dependent(), test_boundary_quote_reanchoring_never_discards_substantive_context(), test_boundary_quote_reanchoring_remains_exact_unique_and_in_range(), test_carolingian_visual_dependent_span_is_rejected(), test_chain_rule_query_keeps_related_prerequisite_and_worked_paraphrase(), test_comparison_query_keeps_each_substantive_side_as_its_own_facet() (+25 more)

### Community 43 - "Community 43"
Cohesion: 0.14
Nodes (30): _complete(), _seed_completed_accuracy(), _seed_promotion_outcome(), _seed_reel(), test_answer_reveals_key_then_applies_concept_outcomes_once(), test_attempt_insert_race_reloads_the_winning_choice(), test_auto_promotion_does_not_require_three_separate_sessions(), test_auto_promotion_is_blocked_by_recent_negative_outcome() (+22 more)

### Community 44 - "Community 44"
Cohesion: 0.06
Nodes (4): _make_clip(), Pure unit tests for backend.app.clip_engine.bridge — no DB, no network., test_pick_best_clip_fallback_when_none_in_bounds(), test_pick_best_clip_prefers_in_bounds()

### Community 45 - "Community 45"
Cohesion: 0.09
Nodes (22): Exception, APIStatusError, BadRequestError, chat(), get_client(), parse_groq_duration(), RateBudget, RateLimitError (+14 more)

### Community 46 - "Community 46"
Cohesion: 0.13
Nodes (29): _acquire_generation_submit_lock(), append_event(), _atomic_write(), build_request_key(), expire_stale_queued_job(), _fail_unclaimable_job(), find_active_job(), get_job() (+21 more)

### Community 47 - "Community 47"
Cohesion: 0.11
Nodes (18): _atomic_write(), _available_questions(), _cadence_session_id(), _cadence_target(), _check_cancelled(), _completed_rows(), _ensure_learner_progress(), _information_units() (+10 more)

### Community 48 - "Community 48"
Cohesion: 0.11
Nodes (30): _cache_age_seconds(), deterministic_expand(), expand_query(), expand_query_practice_fast(), _expansion_cache_key(), free_expand(), _key(), literal_fallback() (+22 more)

### Community 49 - "Community 49"
Cohesion: 0.09
Nodes (7): CommunityAuthSecurityTests, CommunityChangeEmailTests, CommunityDeleteAccountTests, HostedVerificationDeliveryTests, Insert an unverified account + session directly into the DB., VerificationDisabledModeTests, _VerificationTestBase

### Community 50 - "Community 50"
Cohesion: 0.06
Nodes (5): AnchorExtraction, BroadQueries, MediumAndNone, NarrowQueries, Golden table tests for ``backend/app/services/query_intent.py``.  The classifier

### Community 51 - "Community 51"
Cohesion: 0.12
Nodes (20): _fr(), _mk_candidate(), _run_gate(), test_accept_path_records_stats_and_no_flag(), test_confirm_kill_claim_number_mapping_and_missing_default_false(), test_confirm_kill_outage_confirms_nothing_and_marks_outage(), test_confirm_kill_prompt_and_containment(), test_confirm_outage_ships_flagged_with_outage_marker() (+12 more)

### Community 52 - "Community 52"
Cohesion: 0.14
Nodes (21): _memory_conn(), _RecordingCursor, _RecordingPostgresConnection, _submit(), test_active_capacity_coalesces_identical_requests_before_rejecting_new_work(), test_active_capacity_limits_each_learner_without_consuming_remaining_global_slots(), test_append_event_rolls_back_sequence_when_event_insert_fails(), test_cancellation_is_idempotent_and_revokes_running_lease() (+13 more)

### Community 53 - "Community 53"
Cohesion: 0.07
Nodes (8): IntroDetection, OutroDetection, PenaltyTable, Unit tests for ``backend/app/services/structural_classifier.py``.  Cases are gro, The hard cases — these must NOT be flagged as structural., RecapAndTransition, SponsorDetection, SubstantiveContent

### Community 54 - "Community 54"
Cohesion: 0.11
Nodes (19): _ai_json(), _conn(), _manual_plan(), test_calculus_basics_timestamped_window_corpus_is_fail_closed(), test_fast_and_slow_plans_use_one_bounded_pass(), test_final_gate_uses_exact_timestamped_cues_for_native_or_auto_transcripts(), test_intro_to_python_fallback_signature_keeps_the_domain_anchor(), test_long_literal_fallback_preserves_identity_and_bounds_provider_query() (+11 more)

### Community 55 - "Community 55"
Cohesion: 0.13
Nodes (3): Difficulty-stage and value ordering for versioned clipping selections., _selection_item(), SelectionContractOrderingTests

### Community 56 - "Community 56"
Cohesion: 0.11
Nodes (16): buildHistoryInfoSections(), formatHistoryInfoAccuracy(), formatHistoryInfoBoolean(), formatHistoryInfoBooleanQuery(), formatHistoryInfoDate(), formatHistoryInfoReturnTab(), formatHistoryInfoSeconds(), formatHistoryInfoStrictness() (+8 more)

### Community 57 - "Community 57"
Cohesion: 0.11
Nodes (26): align_edge_anchor(), EdgeAnchor, fetch_json3_words(), _fetch_payload(), _is_original_asr_alias(), Json3CaptionTrack, _language_matches(), LexicalWord (+18 more)

### Community 58 - "Community 58"
Cohesion: 0.11
Nodes (27): adjudication_requirements(), _blind_assessment(), _blind_clip(), build_blind_review_bundle(), build_blind_whole_video_bundle(), decode_whole_video_review_records(), _index_generation_rows(), _manifest_id() (+19 more)

### Community 59 - "Community 59"
Cohesion: 0.07
Nodes (1): Unit tests for the trustworthy-eval harness (freeze + average-N-runs + variance)

### Community 60 - "Community 60"
Cohesion: 0.09
Nodes (10): _cols(), _measure_fixture(), _rej(), _structure(), test_forward_requires_edges_counts_only_forward_requires(), test_measure_inventory_recall_is_gold_gated(), test_phantom_quotable_rate_reads_rejections_and_kind_variants(), test_wave2_columns_w25g_none_vs_zero_semantics() (+2 more)

### Community 61 - "Community 61"
Cohesion: 0.11
Nodes (26): assess_educational_quality(), assess_topic_relevance(), ClipEvaluation, create_material(), evaluate_clip(), extract_video_id(), generate_reels(), _innertube_fetch_transcript() (+18 more)

### Community 62 - "Community 62"
Cohesion: 0.09
Nodes (6): _sents(), test_assign_segments_accepts_yt_dlp_keys(), test_assign_segments_greatest_overlap(), test_assign_segments_no_overlap_falls_back_to_nearest_midpoint(), test_assign_segments_tie_breaks_to_earlier_segment(), test_measure_emits_chapter_gold_metrics_only_with_gold_chapters()

### Community 63 - "Community 63"
Cohesion: 0.1
Nodes (8): test_alignment_tolerates_provider_drift_away_from_verified_edge(), test_direct_impersonated_fetch_explicitly_disables_environment_proxy(), test_end_anchor_is_first_excluded_suffix_onset(), test_fetches_one_url_with_supplied_transport_bounds(), test_impersonated_track_uses_curl_transport_with_same_bounds(), test_json3_parser_uses_explicit_offsets_and_proven_zero_onsets(), test_start_anchor_uses_unique_unicode_normalized_quote_and_prefix(), _word()

### Community 64 - "Community 64"
Cohesion: 0.23
Nodes (24): _age_seconds(), _ai_term_rejection(), _append_unique(), build_search_query_plan(), _cache_key(), _clean(), _lexically_coherent(), normalize_query() (+16 more)

### Community 65 - "Community 65"
Cohesion: 0.16
Nodes (3): AdaptiveCurriculumTests, _item(), Focused adaptive curriculum and learner-feedback contract tests.

### Community 66 - "Community 66"
Cohesion: 0.13
Nodes (16): Af(), Bf(), Da(), Df(), ed(), F(), Gf(), If() (+8 more)

### Community 67 - "Community 67"
Cohesion: 0.2
Nodes (22): clampNumber(), defaultClipDurationBounds(), dispatchSettingsUpdated(), enforceClipDurationGap(), hasLegacySettingsSnapshot(), normalizeSettingsAccountId(), normalizeStudyReelsSettings(), parseScopedStudyReelsSettingsSnapshot() (+14 more)

### Community 68 - "Community 68"
Cohesion: 0.15
Nodes (16): _FakeClient, _FakeModels, _FakeResp, VID2 edge-probe (Tier-1 video judge) tests. Fully OFFLINE: the video-judge SDK c, _spec(), test_both_edges_flagged(), test_build_embed_clips_omits_edge_keys_when_probe_off(), test_build_embed_clips_surfaces_edge_booleans_when_present() (+8 more)

### Community 69 - "Community 69"
Cohesion: 0.16
Nodes (17): _gate(), _healthy_rollback_metrics(), _rows(), test_all_promotion_thresholds_pass_on_qualifying_rows(), test_cluster_bootstrap_uses_pair_means_not_individual_clips(), test_each_monitoring_trigger_requires_pro_only_rollback(), test_each_threshold_can_block_promotion(), test_inclusive_thresholds_pass_at_their_boundaries() (+9 more)

### Community 70 - "Community 70"
Cohesion: 0.19
Nodes (14): _judge(), E1a labeling exporter — pure manifest construction, stratum tagging, per-stratum, _rej(), _spec(), test_accepted_entry_fields(), test_build_manifest_and_write(), test_build_manifest_applies_limit_across_videos(), test_collect_video_missing_cache_is_none() (+6 more)

### Community 71 - "Community 71"
Cohesion: 0.16
Nodes (14): _cols(), _structure(), test_chapter_coverage_fraction_by_sentence_containment(), test_chapter_coverage_nan_without_topics_zero_without_clips(), test_chapter_coverage_uses_unit_node_id_when_present(), test_topic_span_coverage_nan_without_timing_zero_without_clips(), test_topic_span_coverage_sliver_reads_low_where_chapter_coverage_reads_full(), test_topic_span_coverage_unions_overlaps_and_clips_to_node() (+6 more)

### Community 72 - "Community 72"
Cohesion: 0.17
Nodes (18): _manifest(), _profile_result(), test_bad_hash_blocks_execution_before_output(), test_empty_or_missing_telemetry_never_prices_as_zero_complete_usage(), test_executable_runner_emits_375_rows_and_never_calls_hybrid(), test_hybrid_preserves_green_split_enrichment_fallback_telemetry(), test_hybrid_synthesis_failure_still_emits_all_fifth_rows(), test_hybrid_uses_same_repeat_flash_or_corrected_pro_without_extra_call() (+10 more)

### Community 73 - "Community 73"
Cohesion: 0.1
Nodes (7): _FakeYoutubeService, IngestionUrlTests, NormalizeClipWindowTests, Tests for the reel ingestion pipeline (`backend/app/ingestion/`).  Retired tests, Search tests can make multiple calls; bump per-platform limits to avoid noise., Running the same search twice must land both batches under the same material_id., Drop-in stand-in for YouTubeService that returns canned transcript cues.

### Community 74 - "Community 74"
Cohesion: 0.16
Nodes (15): _intro_to_python_plan(), _plan(), _run_discover(), test_discover_excludes_and_limits(), test_discover_threads_level_to_rank(), test_excluded_consensus_does_not_stop_expansion(), test_fast_context_limits_initial_expansion_to_three_queries(), test_intro_to_python_searches_literal_before_hd_ai_terms() (+7 more)

### Community 75 - "Community 75"
Cohesion: 0.22
Nodes (18): _generation_rows(), _items(), _record(), _scores(), test_blind_bundle_separates_hidden_identity_and_supplies_four_second_context(), test_bundle_is_deterministic_for_seed_and_caps_context_at_video_edges(), test_clip_resolution_uses_explicit_adjudicator_and_emits_one_grounded_record(), test_one_point_disagreement_does_not_require_adjudication() (+10 more)

### Community 76 - "Community 76"
Cohesion: 0.12
Nodes (4): detectMobilePhoneDevice(), detectTouchLikeDevice(), onResize(), update()

### Community 77 - "Community 77"
Cohesion: 0.13
Nodes (5): _artifact(), test_transcript_cache_rejects_tombstoned_video(), test_transcript_cache_version_invalidates_coarser_cue_artifacts(), test_transcript_validation_accepts_auto_mode_artifact(), test_transcript_validation_rejects_nonfinite_and_nonmonotonic_cues()

### Community 78 - "Community 78"
Cohesion: 0.16
Nodes (10): _entry(), _human(), E1d judge-calibration math — kappa hand-checked, bootstrap CI shape, the 0.5s jo, _row(), test_join_matches_within_half_second_both_endpoints(), test_join_skips_unanswered_labels(), test_kappa_pairs_come_from_joined_rows(), test_per_kind_human_only_kind_with_no_kills() (+2 more)

### Community 79 - "Community 79"
Cohesion: 0.2
Nodes (17): _canonical_reference_term(), _explicit_reference_has_antecedent(), first_lexical_character_index(), _has_unresolved_opening_back_reference(), _has_unresolved_question_reference(), _is_framing_or_question(), is_onset(), opens_mid_thought() (+9 more)

### Community 80 - "Community 80"
Cohesion: 0.21
Nodes (14): Per-clip difficulty: parsed, normalized, carried — NEVER gates a clip., _run(), _segments(), _segs(), test_difficulty_carried_on_clip(), test_difficulty_defaults_to_half_when_omitted(), test_difficulty_is_carried(), test_difficulty_is_required() (+6 more)

### Community 81 - "Community 81"
Cohesion: 0.23
Nodes (2): CommunitySetOwnershipTests, _owner_hash()

### Community 82 - "Community 82"
Cohesion: 0.16
Nodes (6): _PostgresConnection, _PostgresCursor, Portable, dependency-safe source takedown tests., Exercise the psycopg branches of the shared DB helpers over SQLite., _seed_takedown_graph(), test_takedown_cleans_assessment_dependencies_portably()

### Community 83 - "Community 83"
Cohesion: 0.12
Nodes (6): ClipEngineFeedTests, Tests for Task 12: ingest_feed routed through clip engine (YouTube-only).  TDD f, resolve_feed_urls extracts watch URLs from yt_dlp entries., resolve_feed_urls truncates to max_items., resolve_feed_urls swallows exceptions and returns []., ResolveFeedUrlsUnitTests

### Community 84 - "Community 84"
Cohesion: 0.12
Nodes (0):

### Community 85 - "Community 85"
Cohesion: 0.14
Nodes (3): _clear_text_provider_env(), test_availability_ignores_credentials_for_disabled_providers(), test_gemini_builder_accepts_rotated_key_without_api_blackout()

### Community 86 - "Community 86"
Cohesion: 0.12
Nodes (2): KnowledgeLevelApiTests, API contract: create-material level field, PATCH level, feed fields. FastAPI Tes

### Community 87 - "Community 87"
Cohesion: 0.22
Nodes (3): DirectAdapterMediaTailTests, _media_tail_engine_out(), Regression coverage for quiet handoffs beyond the final caption timestamp.

### Community 88 - "Community 88"
Cohesion: 0.17
Nodes (9): buildCanvasFont(), clamp(), createProgram(), createShader(), fillTrackedText(), measureTitleBounds(), measureTrackedText(), parseCssPixelValue() (+1 more)

### Community 89 - "Community 89"
Cohesion: 0.2
Nodes (14): attach_prerequisites(), is_severed_pair(), link_severed_pairs(), _pair_opener_in_later(), Chronological sequencing + prerequisite hints (spec §7C) + severed-pair linking, W25-F pair-scoped replacement for the blanket 'later has no opener roles' test —, One instructional event cut in two: the EARLIER clip has opener roles     (examp, P4b, run AFTER sequencing. Pass 1 attempts merges: a severed pair whose combined (+6 more)

### Community 90 - "Community 90"
Cohesion: 0.2
Nodes (2): PersistenceIntegrityTests, _seed_identity()

### Community 91 - "Community 91"
Cohesion: 0.29
Nodes (2): LevelAutoAdjustTests, Learner-scoped global difficulty adjustment semantics.

### Community 92 - "Community 92"
Cohesion: 0.3
Nodes (12): _clip(), _key(), test_segment_cache_accepts_complete_clips_longer_than_180_seconds(), test_segment_cache_accepts_more_than_sixteen_distinct_candidates(), test_segment_cache_keeps_distinct_facets_inside_one_coarse_cue(), test_segment_cache_key_tracks_transcript_topic_and_policy(), test_segment_cache_preserves_difficulty_order_not_chronology(), test_segment_cache_revalidates_public_clip_contract() (+4 more)

### Community 93 - "Community 93"
Cohesion: 0.23
Nodes (9): _rej(), _spec(), test_integrity_columns_shapes_and_nan_convention(), test_kill_counts_split_by_confirmation_and_stage(), test_phantom_rate_counts_specs_and_rejections(), test_phantom_rate_nan_when_no_reasons_recorded(), test_phantom_rate_specs_only_and_rejections_only(), test_phantom_rate_tolerates_missing_keys() (+1 more)

### Community 94 - "Community 94"
Cohesion: 0.26
Nodes (13): Silence-aware start/end placement (Tasks 5-6). Offline: energy_fn=None → pure ga, _s(), test_end_cuts_into_gap_never_into_next_word(), test_end_hybrid_beyond_budget_keeps_tight(), test_end_hybrid_nudges_to_next_gap_within_budget(), test_end_last_sentence_gap_unmeasurable_grows(), test_end_no_valid_end_grows(), test_end_small_gap_uses_midpoint() (+5 more)

### Community 95 - "Community 95"
Cohesion: 0.19
Nodes (6): GenerationIdPersistenceTests, Tests for generation_id threading through the ingest persistence layer (Task T1), Reel persisted with generation_id='gen-x' is:         - stored in the DB with th, Reel persisted with generation_id=None (or omitted) stores NULL         and is f, The same (material_id, video_id, t_start, t_end) tuple under two         differe, Base kwargs for upsert_reel_row — override specific fields as needed.

### Community 96 - "Community 96"
Cohesion: 0.28
Nodes (11): _key(), _make_reversed_windows(), _make_windows(), Order-invariance of the parallel boundary REFINE (latency lever).  The per-clip, spec[round(s0,3)] = (idx, target_start, target_end) — the deterministic per-clip, A plain (no coordination) `_whisper_window`: recover the clip from the window st, Same deterministic sentences, but forces STRICTLY REVERSED completion: clip idx, _run() (+3 more)

### Community 97 - "Community 97"
Cohesion: 0.17
Nodes (9): Tests for CLIP_ENGINE routing (Task 6).  Covers three things: 1. test_engine_res, clip_engine='unit' → legacy unit engine., _resolve_assemble_fn is imported at the orchestrator's call site., _resolve_assemble_fn is imported and used at the cli's call site., _resolve(), test_cli_uses_resolve_assemble_fn(), test_engine_resolution(), test_orchestrator_uses_resolve_assemble_fn() (+1 more)

### Community 98 - "Community 98"
Cohesion: 0.15
Nodes (4): ClipEngineSearchTests, Tests for Task 11: ingest_search routed through Supadata + clip engine (YouTube-, Caller passes platforms=["yt","ig","tt"]; result must have platforms==["yt"], When discover() returns a truthy "warning" (e.g. out-of-credits) and no videos,

### Community 99 - "Community 99"
Cohesion: 0.15
Nodes (2): Covers 3 invariants:     1. Same match_count: boosted edu video outranks penalis, test_educational_ranking_and_bounds()

### Community 100 - "Community 100"
Cohesion: 0.24
Nodes (7): BND1 — free text-only boundary guards. Offline (no audio, no whisper, no LLM)., _sent(), test_only_conjunction_ends_available_still_places(), test_only_weak_end_still_places_with_warning(), test_real_terminators_never_flagged_weak_when_clause_complete(), test_strong_end_preferred_over_weak_conjunction_end(), test_two_word_end_preferred_against_when_alternative_exists()

### Community 101 - "Community 101"
Cohesion: 0.24
Nodes (8): Window extension + refine orchestration (Task 7). Fully offline: _whisper_window, fn(win_start, win_end) -> list[Sentence]; adapts to the (sents, wav=None) contra, _stub_window(), test_end_accepts_far_period_found_via_growth(), test_end_exhaustion_flags_and_ships(), test_end_grows_until_period_found(), test_end_respects_max_clip_end(), test_start_grows_backward_to_see_prev()

### Community 102 - "Community 102"
Cohesion: 0.36
Nodes (11): _channel_name(), _duration(), _edu_score(), _first_nonblank(), _integer(), _level_score(), merge_and_rank(), merge_and_rank_practice_fast() (+3 more)

### Community 103 - "Community 103"
Cohesion: 0.27
Nodes (6): ABC, get_storage(), LocalStorage, S3Storage, _safe_filename(), Storage

### Community 104 - "Community 104"
Cohesion: 0.21
Nodes (2): KnowledgeLevelMigrationTests, Columns for the knowledge-level feature exist after init and are idempotent.

### Community 105 - "Community 105"
Cohesion: 0.18
Nodes (2): NormalizeTests, TargetTests

### Community 106 - "Community 106"
Cohesion: 0.24
Nodes (2): ClipEngineIngestUrlTests, _fake_engine_out()

### Community 107 - "Community 107"
Cohesion: 0.29
Nodes (8): _post(), E1c labels endpoints — POST /api/labels merge-not-clobber + GET resume.  FastAPI, test_bad_video_ids_rejected(), test_post_corrupt_golden_file_is_409_and_untouched(), test_post_creates_golden_file_and_get_resumes(), test_post_empty_note_keeps_existing_note(), test_post_preserves_existing_gold_keys(), test_post_upserts_by_span_within_tolerance_and_appends_new()

### Community 108 - "Community 108"
Cohesion: 0.22
Nodes (4): ClipEngineTopicCutTests, _fake_engine_out_two_clips(), Tests for the clip-engine-routed ingest_topic_cut (Task 10).  Strategy: mirror t, Returns a transcript + 2 clips.     Clip 0 (30-90s): talks about the "chain rule

### Community 109 - "Community 109"
Cohesion: 0.22
Nodes (3): normalizeSignupEmailForComparison(), onSendVerificationEmail(), onVerifyAccount()

### Community 110 - "Community 110"
Cohesion: 0.18
Nodes (0):

### Community 111 - "Community 111"
Cohesion: 0.22
Nodes (2): FastAPI assessment contract, privacy, resume, and learner isolation., TestAssessmentApi

### Community 112 - "Community 112"
Cohesion: 0.2
Nodes (0):

### Community 113 - "Community 113"
Cohesion: 0.39
Nodes (8): _pick(), _sent(), test_fit_budget_no_terminator_in_budgeted_span(), test_fit_budget_terminator_only_at_window_start(), test_window_clamps_out_of_range(), test_window_moves_start_off_dangling_opener(), test_window_snaps_end_to_terminator(), test_window_truncates_to_budget()

### Community 114 - "Community 114"
Cohesion: 0.22
Nodes (2): test_real_treeseg_inclusive_content_map_reaches_topic_assembly(), test_serial_extract_best_window_failure_isolated()

### Community 115 - "Community 115"
Cohesion: 0.36
Nodes (6): _structure(), test_select_keeps_teaching_drops_filler(), test_select_never_zero_on_llm_failure(), test_select_respects_max_clips(), test_target_topic_does_not_fallback_to_off_topic_clips(), test_target_topic_filters_selection_and_reaches_prompt()

### Community 116 - "Community 116"
Cohesion: 0.56
Nodes (8): _closure(), _qp_c0t1(), test_after_run_breaks_at_topic_node_boundary(), test_after_run_is_bounded_by_max_extra_units(), test_after_run_spans_the_full_definition_block(), test_before_run_inlines_the_whole_problem_statement(), test_before_run_respects_span_budget_and_flags_truncation(), _u()

### Community 117 - "Community 117"
Cohesion: 0.36
Nodes (7): Precise-boundary resilience + direction-safe Whisper picks. Offline (no audio, n, _sent(), test_pick_end_never_moves_earlier_than_window_floor(), test_pick_end_normal_path_unchanged(), test_pick_start_keep_first_at_video_start(), test_pick_start_never_moves_later_than_window_ceiling(), test_pick_start_normal_path_unchanged()

### Community 118 - "Community 118"
Cohesion: 0.22
Nodes (2): RankedExclusionNormalizationTests, Regression test for Finding #1: client pagination exclusion across the video_id

### Community 119 - "Community 119"
Cohesion: 0.39
Nodes (6): comprehension() excludes error verdicts; judge_error_rate accounting. Offline., _Sent, _specs(), test_comprehension_all_errors(), test_comprehension_excludes_error_verdicts(), test_judge_failures_carries_error_flag()

### Community 120 - "Community 120"
Cohesion: 0.46
Nodes (6): Energy-minimum snap (Task 3). Synthesizes a tone+silence wav — no whisper, no ff, test_absolute_offset_respected(), test_snaps_to_silent_frame(), test_subframe_interval_returns_none(), _tone_then_silence(), _write_wav()

### Community 121 - "Community 121"
Cohesion: 0.36
Nodes (6): _FakeModel, _install_fake(), Refine-model Whisper singleton (Task 2). WhisperModel is stubbed — no real model, test_refine_reuses_full_singleton_when_models_match(), test_refine_singleton_builds_refine_model(), test_refine_singleton_uses_refine_workers()

### Community 122 - "Community 122"
Cohesion: 0.43
Nodes (7): Onset START guard — the symmetric twin of the weak-END guard. Offline (no audio/, _sent(), test_backward_extension_bounded_by_node_span(), test_good_onset_start_unchanged(), test_is_weak_start_matches_primitive(), test_only_weak_start_still_places_flagged(), test_weak_start_extends_back_to_onset()

### Community 123 - "Community 123"
Cohesion: 0.32
Nodes (7): difficulty_matches_knowledge_level(), effective_level_target(), normalize_knowledge_level(), Knowledge-level semantics: level names, difficulty-scale mapping, and the effect, Lowercased/stripped level name; absent -> 'beginner'; unknown -> ValueError., The difficulty the feed should aim at RIGHT NOW for this material., Match selector difficulty bins without overlap at level boundaries.

### Community 124 - "Community 124"
Cohesion: 0.29
Nodes (1): CommunityAuthDbMigrationTests

### Community 125 - "Community 125"
Cohesion: 0.29
Nodes (3): ClipEngineContractTests, _fake_engine_out(), HTTP-layer contract smoke test for POST /api/ingest/url (Task 13).  Asserts that

### Community 126 - "Community 126"
Cohesion: 0.36
Nodes (1): CommunityHistorySyncTests

### Community 127 - "Community 127"
Cohesion: 0.32
Nodes (3): _run_clean_process(), test_embedding_request_cannot_lazy_load_torch(), test_main_import_uses_no_torch_embedding_backend()

### Community 128 - "Community 128"
Cohesion: 0.25
Nodes (1): CommunityReelDurationSecurityTests

### Community 129 - "Community 129"
Cohesion: 0.36
Nodes (1): CommunitySettingsSyncTests

### Community 130 - "Community 130"
Cohesion: 0.46
Nodes (5): historyScopeStorageKey(), normalizeHistoryAccountId(), readScopedHistorySnapshot(), seedGuestHistoryScopeFromActiveHistory(), writeScopedHistorySnapshot()

### Community 131 - "Community 131"
Cohesion: 0.62
Nodes (6): _closure(), test_non_onset_required_overflowing_the_soft_budget_is_carded(), test_onset_beyond_the_hard_cap_is_carded_not_inlined(), test_onset_overflowing_into_the_soft_hard_window_is_inlined(), test_required_before_onset_is_inlined_even_past_span_budget(), _u()

### Community 132 - "Community 132"
Cohesion: 0.48
Nodes (6): Per-clip orientation cards (feed display). Offline — llm_json monkeypatched, zer, test_extractive_fallback_on_llm_error(), test_grounded_llm_cards_assigned_per_clip(), test_never_blank_and_word_capped(), test_ungrounded_llm_card_rejected(), _units()

### Community 133 - "Community 133"
Cohesion: 0.29
Nodes (0):

### Community 134 - "Community 134"
Cohesion: 0.43
Nodes (5): clip(), is_valid_timestamped_supadata_transcript(), pro_boundary_fallback(), _transcribe(), _wire_segment_runtime()

### Community 135 - "Community 135"
Cohesion: 0.33
Nodes (6): classify_passage(), label_penalty(), Structural role classifier for transcript passages.  Labels a passage as INTRO /, Return the structural role of ``text`` with a position-aware prior.      Pass th, Confidence-scaled penalty for ``label`` — caller subtracts from score., StructuralLabel

### Community 136 - "Community 136"
Cohesion: 0.38
Nodes (3): _reload_segment_config(), test_legacy_pro_override_wins_then_explicit_pro_model_is_fallback(), test_segment_router_rejects_invalid_values_and_clamps_percent()

### Community 137 - "Community 137"
Cohesion: 0.33
Nodes (4): IngestionTopicCutTests, _make_long_transcript_cues(), Integration tests for `POST /api/ingest/topic-cut` and the corresponding `Ingest, 20 cues × 30 seconds = 10 minutes of fake speech, two clearly distinct topics.

### Community 138 - "Community 138"
Cohesion: 0.48
Nodes (2): _build_request(), RateLimitClientIpResolutionTests

### Community 139 - "Community 139"
Cohesion: 0.48
Nodes (5): _cancel_shortly(), test_active_async_request_is_cancelled_within_three_hundred_ms(), test_chat_http_request_receives_cancellation(), test_gemini_active_request_receives_cancellation(), test_supadata_active_socket_receives_cancellation_and_does_not_retry()

### Community 140 - "Community 140"
Cohesion: 0.29
Nodes (0):

### Community 141 - "Community 141"
Cohesion: 0.43
Nodes (5): applySearchFeedSettingsToParams(), buildSearchFeedQuery(), parseQueryBoolean(), parseQueryNumber(), readSearchFeedQuerySettings()

### Community 142 - "Community 142"
Cohesion: 0.33
Nodes (0):

### Community 143 - "Community 143"
Cohesion: 0.33
Nodes (0):

### Community 144 - "Community 144"
Cohesion: 0.33
Nodes (1): Precise-cutting config surface (Task 1). Pure constants — no audio/whisper.

### Community 145 - "Community 145"
Cohesion: 0.47
Nodes (4): _columns(), Assessment/reel-content schema parity for SQLite and PostgreSQL., test_existing_sqlite_reels_and_history_are_migrated_idempotently(), test_fresh_sqlite_schema_contains_assessment_tables_and_private_keys()

### Community 146 - "Community 146"
Cohesion: 0.33
Nodes (1): FE1 (embed bleed fix) + FE2 (quality payload) — the serving embed path.  FE1 roo

### Community 147 - "Community 147"
Cohesion: 0.33
Nodes (0):

### Community 148 - "Community 148"
Cohesion: 0.33
Nodes (0):

### Community 149 - "Community 149"
Cohesion: 0.33
Nodes (1): VID4 eval columns for the edge probe (advisory rates). Offline, pure functions.

### Community 150 - "Community 150"
Cohesion: 0.4
Nodes (1): refine_clip_boundaries wiring (Task 8). Offline: audio + _refine_one stubbed.

### Community 151 - "Community 151"
Cohesion: 0.8
Nodes (4): _connection(), _run_until_cancelled(), test_deep_expansion_forwards_cancellation_to_shared_query_plan(), test_topic_expansion_cancels_active_http_and_skips_later_calls_and_cache()

### Community 152 - "Community 152"
Cohesion: 0.5
Nodes (4): _collect_referenced_names(), Completeness guard: every config.<CONSTANT> referenced in the vendored gemini-pa, Every config.NAME reference in the gemini-path modules must be on the shim., test_shim_exposes_all_referenced_constants()

### Community 153 - "Community 153"
Cohesion: 0.5
Nodes (0):

### Community 154 - "Community 154"
Cohesion: 0.5
Nodes (0):

### Community 155 - "Community 155"
Cohesion: 0.5
Nodes (0):

### Community 156 - "Community 156"
Cohesion: 0.5
Nodes (3): Email service using Resend for transactional email delivery., Send a welcome email to a newly registered user.      Errors are caught and logg, send_welcome_email()

### Community 157 - "Community 157"
Cohesion: 0.67
Nodes (2): _cue(), QueryFocusedSnippetTests

### Community 158 - "Community 158"
Cohesion: 0.67
Nodes (2): _sent(), test_opening_onset_rate_counts_only_good_openers()

### Community 159 - "Community 159"
Cohesion: 0.5
Nodes (0):

### Community 160 - "Community 160"
Cohesion: 0.67
Nodes (2): createInitialState(), LoadingFlappyMiniGame()

### Community 161 - "Community 161"
Cohesion: 0.67
Nodes (0):

### Community 162 - "Community 162"
Cohesion: 0.67
Nodes (1): pysbd's Segmenter is not thread-safe: segment() stashes the input on self.origin

### Community 163 - "Community 163"
Cohesion: 0.67
Nodes (0):

### Community 164 - "Community 164"
Cohesion: 0.67
Nodes (0):

### Community 165 - "Community 165"
Cohesion: 0.67
Nodes (0):

### Community 166 - "Community 166"
Cohesion: 0.67
Nodes (0):

### Community 167 - "Community 167"
Cohesion: 0.67
Nodes (0):

### Community 168 - "Community 168"
Cohesion: 0.67
Nodes (0):

### Community 169 - "Community 169"
Cohesion: 1.0
Nodes (0):

### Community 170 - "Community 170"
Cohesion: 1.0
Nodes (0):

### Community 171 - "Community 171"
Cohesion: 1.0
Nodes (0):

### Community 172 - "Community 172"
Cohesion: 1.0
Nodes (0):

### Community 173 - "Community 173"
Cohesion: 1.0
Nodes (0):

### Community 174 - "Community 174"
Cohesion: 1.0
Nodes (0):

### Community 175 - "Community 175"
Cohesion: 1.0
Nodes (0):

### Community 176 - "Community 176"
Cohesion: 1.0
Nodes (0):

### Community 177 - "Community 177"
Cohesion: 1.0
Nodes (0):

### Community 178 - "Community 178"
Cohesion: 1.0
Nodes (0):

### Community 179 - "Community 179"
Cohesion: 1.0
Nodes (0):

### Community 180 - "Community 180"
Cohesion: 1.0
Nodes (0):

### Community 181 - "Community 181"
Cohesion: 1.0
Nodes (0):

### Community 182 - "Community 182"
Cohesion: 1.0
Nodes (0):

### Community 183 - "Community 183"
Cohesion: 1.0
Nodes (0):

### Community 184 - "Community 184"
Cohesion: 1.0
Nodes (0):

### Community 185 - "Community 185"
Cohesion: 1.0
Nodes (0):

### Community 186 - "Community 186"
Cohesion: 1.0
Nodes (0):

### Community 187 - "Community 187"
Cohesion: 1.0
Nodes (0):

### Community 188 - "Community 188"
Cohesion: 1.0
Nodes (0):

### Community 189 - "Community 189"
Cohesion: 1.0
Nodes (0):

### Community 190 - "Community 190"
Cohesion: 1.0
Nodes (0):

### Community 191 - "Community 191"
Cohesion: 1.0
Nodes (0):

### Community 192 - "Community 192"
Cohesion: 1.0
Nodes (0):

### Community 193 - "Community 193"
Cohesion: 1.0
Nodes (0):

### Community 194 - "Community 194"
Cohesion: 1.0
Nodes (0):

### Community 195 - "Community 195"
Cohesion: 1.0
Nodes (0):

### Community 196 - "Community 196"
Cohesion: 1.0
Nodes (0):

### Community 197 - "Community 197"
Cohesion: 1.0
Nodes (0):

### Community 198 - "Community 198"
Cohesion: 1.0
Nodes (0):

### Community 199 - "Community 199"
Cohesion: 1.0
Nodes (0):

### Community 200 - "Community 200"
Cohesion: 1.0
Nodes (0):

### Community 201 - "Community 201"
Cohesion: 1.0
Nodes (0):

### Community 202 - "Community 202"
Cohesion: 1.0
Nodes (0):

### Community 203 - "Community 203"
Cohesion: 1.0
Nodes (0):

### Community 204 - "Community 204"
Cohesion: 1.0
Nodes (0):

## Knowledge Gaps
- **391 isolated node(s):** `Submit one autocrop job and return the URL of the rendered short.`, `MuAPI result shapes vary by endpoint — try common keys.`, `Find the most viral-worthy highlights in a transcript.  Logic ported from ViralV`, `Default LLM backend: MuAPI gpt-5-mini.`, `gpt-5-4 sometimes wraps JSON in markdown fences — strip and parse.` (+386 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 169`** (2 nodes): `ProcessingStepper.tsx`, `ProcessingStepper()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 170`** (2 nodes): `ResultsGrid.tsx`, `ResultsGrid()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 171`** (2 nodes): `DownloadAll()`, `DownloadAll.tsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 172`** (2 nodes): `VolumetricGlowBackground.tsx`, `VolumetricGlowBackground()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 173`** (2 nodes): `useJobStream.ts`, `useJobStream()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 174`** (2 nodes): `test_topics_types.py`, `test_schemas_and_dataclasses_exist()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 175`** (2 nodes): `test_youtube_transport.py`, `test_youtube_impersonation_target_is_supported()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 176`** (2 nodes): `test_provider_cache_db.py`, `test_sqlite_provider_cache_round_trip_and_tombstone_filter()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 177`** (2 nodes): `test_singleflight.py`, `test_singleflight_coalesces_identical_concurrent_work()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 178`** (2 nodes): `RootLayout()`, `layout.tsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 179`** (2 nodes): `Loading()`, `loading.tsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 180`** (2 nodes): `FullscreenLoadingScreen()`, `FullscreenLoadingScreen.tsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 181`** (2 nodes): `ViewportModalPortal.tsx`, `ViewportModalPortal()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 182`** (2 nodes): `youtubeIframeApi.ts`, `loadYouTubeIframeApi()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 183`** (1 nodes): `next-env.d.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 184`** (1 nodes): `tailwind.config.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 185`** (1 nodes): `postcss.config.js`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 186`** (1 nodes): `next.config.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 187`** (1 nodes): `tailwind.config.js`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 188`** (1 nodes): `vite.config.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 189`** (1 nodes): `postcss.config.js`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 190`** (1 nodes): `main.tsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 191`** (1 nodes): `types.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 192`** (1 nodes): `SettingsDrawer.tsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 193`** (1 nodes): `InputCard.tsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 194`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 195`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 196`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 197`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 198`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 199`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 200`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 201`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 202`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 203`** (1 nodes): `GenerationProgress.tsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 204`** (1 nodes): `types.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ReelService` connect `Community 3` to `Community 65`, `Community 1`, `Community 14`, `Community 55`, `Community 91`?**
  _High betweenness centrality (0.071) - this node is a cross-community bridge._
- **Why does `Sentence` connect `Community 0` to `Community 96`, `Community 1`, `Community 100`, `Community 5`, `Community 6`, `Community 101`, `Community 8`, `Community 17`, `Community 117`, `Community 122`, `Community 94`?**
  _High betweenness centrality (0.068) - this node is a cross-community bridge._
- **Why does `IngestTranscriptCue` connect `Community 1` to `Community 25`, `Community 90`, `Community 3`, `Community 157`?**
  _High betweenness centrality (0.051) - this node is a cross-community bridge._
- **Are the 200 inferred relationships involving `Sentence` (e.g. with `Precise boundary refinement with targeted Whisper.  Supadata gives fast but coar` and `Absolute time of the lowest-RMS ``frame_ms`` frame within ``[a, b]`` — the quiet`) actually correct?**
  _`Sentence` has 200 INFERRED edges - model-reasoned connections that need verification._
- **Are the 81 inferred relationships involving `ReelService` (e.g. with `Core rate limit check against a pre-built key. Raises HTTPException on breach.` and `Preferred rate-limit bucket key for a request. Uses the owner key hash when`) actually correct?**
  _`ReelService` has 81 INFERRED edges - model-reasoned connections that need verification._
- **Are the 178 inferred relationships involving `Unit` (e.g. with `PlanItemLLM` and `ExtractionPlanLLM`) actually correct?**
  _`Unit` has 178 INFERRED edges - model-reasoned connections that need verification._
- **Are the 141 inferred relationships involving `GenerationContext` (e.g. with `Core rate limit check against a pre-built key. Raises HTTPException on breach.` and `Preferred rate-limit bucket key for a request. Uses the owner key hash when`) actually correct?**
  _`GenerationContext` has 141 INFERRED edges - model-reasoned connections that need verification._