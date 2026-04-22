# Graph Report - .  (2026-04-21)

## Corpus Check
- 119 files · ~738,119 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 3047 nodes · 9532 edges · 76 communities detected
- Extraction: 56% EXTRACTED · 44% INFERRED · 0% AMBIGUOUS · INFERRED: 4198 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 59|Community 59]]
- [[_COMMUNITY_Community 60|Community 60]]
- [[_COMMUNITY_Community 61|Community 61]]
- [[_COMMUNITY_Community 62|Community 62]]
- [[_COMMUNITY_Community 63|Community 63]]
- [[_COMMUNITY_Community 64|Community 64]]
- [[_COMMUNITY_Community 65|Community 65]]
- [[_COMMUNITY_Community 66|Community 66]]
- [[_COMMUNITY_Community 67|Community 67]]
- [[_COMMUNITY_Community 68|Community 68]]
- [[_COMMUNITY_Community 69|Community 69]]
- [[_COMMUNITY_Community 70|Community 70]]
- [[_COMMUNITY_Community 71|Community 71]]
- [[_COMMUNITY_Community 72|Community 72]]
- [[_COMMUNITY_Community 73|Community 73]]
- [[_COMMUNITY_Community 74|Community 74]]
- [[_COMMUNITY_Community 75|Community 75]]

## God Nodes (most connected - your core abstractions)
1. `ReelService` - 384 edges
2. `IngestTranscriptCue` - 265 edges
3. `YouTubeService` - 253 edges
4. `EmbeddingService` - 236 edges
5. `TranscriptQuality` - 195 edges
6. `IngestTranscriptWord` - 171 edges
7. `MediumRegressionTests` - 161 edges
8. `TranscriptCue` - 149 edges
9. `DatabaseIntegrityError` - 139 edges
10. `SegmentMatch` - 112 edges

## Surprising Connections (you probably didn't know these)
- `DatabaseIntegrityError` --uses--> `Central LLM router: Gemini primary, Groq fallback, Cerebras final fallback.  Rep`  [INFERRED]
  backend/app/db.py → backend/app/services/llm_router.py
- `DatabaseIntegrityError` --uses--> `Return the ``google.genai`` module when a key is available.      The new SDK bui`  [INFERRED]
  backend/app/db.py → backend/app/services/llm_router.py
- `DatabaseIntegrityError` --uses--> `Return an Ollama config dict when ``OLLAMA_BASE_URL`` is set.      Ollama expose`  [INFERRED]
  backend/app/db.py → backend/app/services/llm_router.py
- `DatabaseIntegrityError` --uses--> `Call an Ollama server via its OpenAI-compatible chat endpoint.`  [INFERRED]
  backend/app/db.py → backend/app/services/llm_router.py
- `DatabaseIntegrityError` --uses--> `Cerebras Llama 3.3 70B call — OpenAI-compatible, same shape as Groq.`  [INFERRED]
  backend/app/db.py → backend/app/services/llm_router.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.01
Nodes (382): AuditCase, build_transcript(), check_contract(), inspect_clip(), Inspection, main(), Audit the punctuation contract: every clip returned by pick_clip_heuristic and s, Reconstruct the exact words inside [t_start, t_end] from word     timings, plus (+374 more)

### Community 1 - "Community 1"
Cohesion: 0.02
Nodes (122): fetch_transcript(), REAL end-to-end simulation — hits YouTube via the Data API, fetches actual trans, Fetch transcript — prefer manual (punctuated) over auto (unpunctuated).      Ret, Real YouTube Data API v3 search — returns video metadata., run(), _separator(), yt_search(), BoundaryStressTests (+114 more)

### Community 2 - "Community 2"
Cohesion: 0.02
Nodes (28): Exception, ReelService, BoundaryPaddingTests, ContinuationInvariantTests, make_transcript(), Invariant tests for reel cutting — verify the contract the user cares about:  1., Topic reels end at a complete sentence when transcript allows., When a topic exceeds max_len, consecutive reels continue seamlessly. (+20 more)

### Community 3 - "Community 3"
Cohesion: 0.03
Nodes (179): DatabaseIntegrityError, _anchor_density(), _bi_encoder_relevance(), _cluster_id_for_segment(), _cross_encoder_relevance(), get_cross_encoder(), rank_segments(), RankedSegment (+171 more)

### Community 4 - "Community 4"
Cohesion: 0.02
Nodes (231): Email service using Resend for transactional email delivery., Send a welcome email to a newly registered user.      Errors are caught and logg, send_welcome_email(), _activate_generation(), _adaptive_min_relevance_floor(), admin_diagnose_topic(), admin_run_simulation(), _advance_refinement_state() (+223 more)

### Community 5 - "Community 5"
Cohesion: 0.08
Nodes (179): AdapterResult, BaseAdapter, The canonical output of any adapter's `resolve()` call.      `video_path` is the, Interface every platform adapter must implement., BaseAdapter, BaseModel, DownloadError, IngestError (+171 more)

### Community 6 - "Community 6"
Cohesion: 0.02
Nodes (1): MediumRegressionTests

### Community 7 - "Community 7"
Cohesion: 0.03
Nodes (75): Temp workspace lifecycle + orphan sweeper.  Everything the ingestion pipeline do, Create a fresh temp directory, yield its Path, and delete it on exit.      Uses, Delete any `reelai-ingest-*` directories older than `max_age_sec` from the syste, sweep_orphans(), TempWorkspace(), _emit(), get_ingest_logger(), instrumented() (+67 more)

### Community 8 - "Community 8"
Cohesion: 0.04
Nodes (32): ABC, BaseAdapter ABC and the AdapterResult dataclass.  Every platform adapter returns, BaseSettings, extract_concepts(), _extract_concepts_via_llm(), Extract higher-quality concepts via Gemini (falling back to Groq)., _summary_for_terms(), get_settings() (+24 more)

### Community 9 - "Community 9"
Cohesion: 0.08
Nodes (67): ApiError, apiUrl(), askStudyChat(), buildApiError(), buildGenerateReelsRequestBody(), changeCommunityPassword(), changeCommunityVerificationEmail(), checkReelsCanGenerate() (+59 more)

### Community 10 - "Community 10"
Cohesion: 0.05
Nodes (64): _adapt_query_for_postgres(), _connect_postgres_with_retry(), _database_url(), _db_path(), ensure_db_initialized(), _ensure_reels_generation_index_postgres(), _ensure_reels_generation_index_sqlite(), execute_modify() (+56 more)

### Community 11 - "Community 11"
Cohesion: 0.06
Nodes (44): _cache_key(), _call_whisper(), clip_audio_refine_conditional(), _download_clip_audio(), _ensure_clip_audio_cache_dir(), _ensure_full_video_audio(), _faster_whisper_words(), _lock_for_video() (+36 more)

### Community 12 - "Community 12"
Cohesion: 0.06
Nodes (34): _FakeCue, _FakeWord, Unit tests for the LLM-direct clip cutting path:   * `clip_boundary.snap_llm_bou, Lightweight stand-in for IngestTranscriptWord., Lightweight stand-in for IngestTranscriptCue (just the attrs     refine_boundari, Boundary-level filler trimming was removed to preserve the strict     begin/end-, Trailing fillers are NOT trimmed at the boundary — the clip must     still end o, The picker must cover the substantive explanation (sentence 3 at     t=15-22). S (+26 more)

### Community 13 - "Community 13"
Cohesion: 0.06
Nodes (26): buildCommunitySetInformationParagraphs(), createDraftReelRow(), detectYouTubeDurationWithIframeApi(), draftRowsFromReels(), extractYouTubeVideoId(), formatCommunityPlatformSummary(), formatCompact(), formatLastEditedLabel() (+18 more)

### Community 14 - "Community 14"
Cohesion: 0.07
Nodes (9): _fake_info_dict_instagram(), _fake_info_dict_youtube(), _FakeAdapter, _FakeSearchAdapter, _FakeYoutubeService, IngestionUrlTests, NormalizeClipWindowTests, _patch_ffmpeg_and_ffprobe() (+1 more)

### Community 15 - "Community 15"
Cohesion: 0.1
Nodes (20): buildCommunityFeedReel(), buildCommunityFeedReelId(), buildCommunityPreviewReel(), clamp(), compactFeedSessionSnapshot(), compactStoredReel(), compactStoredText(), getCommunityPlatformLabel() (+12 more)

### Community 16 - "Community 16"
Cohesion: 0.1
Nodes (29): _build_clip_system_prompt(), _build_rerank_system_prompt(), ClipPick, _heuristic_rerank_fallback(), _llm_timestamp_strict_validation(), _parse_clip_pick_json(), _parse_rerank_response(), pick_clip_llm() (+21 more)

### Community 17 - "Community 17"
Cohesion: 0.09
Nodes (7): CommunityAuthSecurityTests, CommunityChangeEmailTests, CommunityDeleteAccountTests, HostedVerificationDeliveryTests, Insert an unverified account + session directly into the DB., VerificationDisabledModeTests, _VerificationTestBase

### Community 18 - "Community 18"
Cohesion: 0.06
Nodes (5): AnchorExtraction, BroadQueries, MediumAndNone, NarrowQueries, Golden table tests for ``backend/app/services/query_intent.py``.  The classifier

### Community 19 - "Community 19"
Cohesion: 0.07
Nodes (8): IntroDetection, OutroDetection, PenaltyTable, Unit tests for ``backend/app/services/structural_classifier.py``.  Cases are gro, The hard cases — these must NOT be flagged as structural., RecapAndTransition, SponsorDetection, SubstantiveContent

### Community 20 - "Community 20"
Cohesion: 0.11
Nodes (26): assess_educational_quality(), assess_topic_relevance(), ClipEvaluation, create_material(), evaluate_clip(), extract_video_id(), generate_reels(), _innertube_fetch_transcript() (+18 more)

### Community 21 - "Community 21"
Cohesion: 0.09
Nodes (7): GenerateClipCandidatesTests, HookPatternTests, PayoffBonusTests, Phase 2 + 3 unit tests for `backend/app/services/clip_boundary.py`.  Covered:, ScoreWindowBreakdownTests, SelfContainmentTests, _sent()

### Community 22 - "Community 22"
Cohesion: 0.12
Nodes (7): KillSwitchTests, Phase 4 tests for `backend/app/ingestion/whisperx_transcribe.py`.  Covered:   *, Re-import so env-var flips take effect for a single test case., _reload_whisperx_module(), VersionPinTests, WhisperxAlignWithMockTests, WhisperxWordsForAudioTests

### Community 23 - "Community 23"
Cohesion: 0.19
Nodes (23): clampNumber(), defaultClipDurationBounds(), dispatchSettingsUpdated(), enforceClipDurationGap(), hasLegacySettingsSnapshot(), normalizeSettingsAccountId(), normalizeStudyReelsSettings(), parseScopedStudyReelsSettingsSnapshot() (+15 more)

### Community 24 - "Community 24"
Cohesion: 0.14
Nodes (16): buildHistoryInfoSections(), formatHistoryInfoBoolean(), formatHistoryInfoBooleanQuery(), formatHistoryInfoClipRange(), formatHistoryInfoDate(), formatHistoryInfoReturnTab(), formatHistoryInfoSeconds(), formatHistoryInfoStrictness() (+8 more)

### Community 25 - "Community 25"
Cohesion: 0.13
Nodes (8): CentralityFallback, IdfTableExposure, InstructionalDensityOrdering, Unit tests for ``backend/app/services/segment_features.py``.  We test in three l, A discourse-rich segment should score above a filler segment., _seg(), StructuralLabelIntegration, TokenAndAnchorMatching

### Community 26 - "Community 26"
Cohesion: 0.32
Nodes (20): case_all_caps_shouting(), case_all_filler(), case_bracket_artifacts(), case_empty_transcript(), case_filler_heavy(), case_mixed_languages(), case_no_punctuation(), case_normal_english() (+12 more)

### Community 27 - "Community 27"
Cohesion: 0.17
Nodes (18): active_topics(), _deliver_threadsafe(), _evict_idle_topics_locked(), _now_ms(), publish(), In-process pub/sub bus for refinement-job progress events (Phase D.1).  Backend, Publish an event to all subscribers of `job_id` and append it to the replay buff, Mark a job's stream complete. Subscribers still attached will receive the option (+10 more)

### Community 28 - "Community 28"
Cohesion: 0.2
Nodes (5): FeedChainIntegrityTests, MergeRequestReelListsChainIntegrityTests, Chain-integrity tests for the feed layer.  Two integrity properties are verified, Cross-generation merge integrity.      ``_merge_request_reel_lists`` is called b, _reel()

### Community 29 - "Community 29"
Cohesion: 0.13
Nodes (5): EndToEndFallbackTests, HeuristicFallbackTests, _mk_cand(), ParseRerankResponseTests, PromptRenderingTests

### Community 30 - "Community 30"
Cohesion: 0.23
Nodes (2): CommunitySetOwnershipTests, _owner_hash()

### Community 31 - "Community 31"
Cohesion: 0.16
Nodes (9): BreakerState, CircuitBreaker, _ClientState, Client-level circuit breaker (Phase B.3).  Used to gate yt-dlp `fallback_clients, Dump all tracked clients' state — for /admin/health observability., Return True if work should be attempted on `client` right now.         A client, Multi-client circuit breaker keyed on a string client identifier., Enum (+1 more)

### Community 32 - "Community 32"
Cohesion: 0.17
Nodes (9): buildCanvasFont(), clamp(), createProgram(), createShader(), fillTrackedText(), measureTitleBounds(), measureTrackedText(), parseCssPixelValue() (+1 more)

### Community 33 - "Community 33"
Cohesion: 0.18
Nodes (6): IngestionTopicCutTests, _make_long_transcript_cues(), _make_mock_openai_client(), Integration tests for `POST /api/ingest/topic-cut` and the corresponding `Ingest, 20 cues × 30 seconds = 10 minutes of fake speech, two clearly distinct topics., Build a chat-completions mock that returns `segments_payload` as JSON.

### Community 34 - "Community 34"
Cohesion: 0.17
Nodes (4): detectMobilePhoneDevice(), detectTouchLikeDevice(), onResize(), update()

### Community 35 - "Community 35"
Cohesion: 0.22
Nodes (3): normalizeSignupEmailForComparison(), onSendVerificationEmail(), onVerifyAccount()

### Community 36 - "Community 36"
Cohesion: 0.33
Nodes (3): _fake_conn(), _FakeYouTubeService, MassClipAuditFetchTranscriptTests

### Community 37 - "Community 37"
Cohesion: 0.22
Nodes (0): 

### Community 38 - "Community 38"
Cohesion: 0.29
Nodes (1): CommunityAuthDbMigrationTests

### Community 39 - "Community 39"
Cohesion: 0.36
Nodes (1): CommunitySettingsSyncTests

### Community 40 - "Community 40"
Cohesion: 0.46
Nodes (5): historyScopeStorageKey(), normalizeHistoryAccountId(), readScopedHistorySnapshot(), seedGuestHistoryScopeFromActiveHistory(), writeScopedHistorySnapshot()

### Community 41 - "Community 41"
Cohesion: 0.38
Nodes (1): CommunityHistorySyncTests

### Community 42 - "Community 42"
Cohesion: 0.29
Nodes (1): CommunityReelDurationSecurityTests

### Community 43 - "Community 43"
Cohesion: 0.48
Nodes (2): _build_request(), RateLimitClientIpResolutionTests

### Community 44 - "Community 44"
Cohesion: 0.43
Nodes (5): applySearchFeedSettingsToParams(), buildSearchFeedQuery(), parseQueryBoolean(), parseQueryNumber(), readSearchFeedQuerySettings()

### Community 45 - "Community 45"
Cohesion: 0.4
Nodes (0): 

### Community 46 - "Community 46"
Cohesion: 0.5
Nodes (0): 

### Community 47 - "Community 47"
Cohesion: 0.67
Nodes (2): createInitialState(), LoadingFlappyMiniGame()

### Community 48 - "Community 48"
Cohesion: 0.67
Nodes (0): 

### Community 49 - "Community 49"
Cohesion: 0.67
Nodes (0): 

### Community 50 - "Community 50"
Cohesion: 1.0
Nodes (0): 

### Community 51 - "Community 51"
Cohesion: 1.0
Nodes (0): 

### Community 52 - "Community 52"
Cohesion: 1.0
Nodes (0): 

### Community 53 - "Community 53"
Cohesion: 1.0
Nodes (0): 

### Community 54 - "Community 54"
Cohesion: 1.0
Nodes (0): 

### Community 55 - "Community 55"
Cohesion: 1.0
Nodes (0): 

### Community 56 - "Community 56"
Cohesion: 1.0
Nodes (0): 

### Community 57 - "Community 57"
Cohesion: 1.0
Nodes (0): 

### Community 58 - "Community 58"
Cohesion: 1.0
Nodes (0): 

### Community 59 - "Community 59"
Cohesion: 1.0
Nodes (0): 

### Community 60 - "Community 60"
Cohesion: 1.0
Nodes (0): 

### Community 61 - "Community 61"
Cohesion: 1.0
Nodes (0): 

### Community 62 - "Community 62"
Cohesion: 1.0
Nodes (1): Return True iff this adapter can handle the given URL.

### Community 63 - "Community 63"
Cohesion: 1.0
Nodes (1): Return the platform code ('yt' / 'ig' / 'tt') for a supported URL.

### Community 64 - "Community 64"
Cohesion: 1.0
Nodes (1): Download the media referenced by `url` into `workspace` and return an AdapterRes

### Community 65 - "Community 65"
Cohesion: 1.0
Nodes (1): Resolve a feed-like URL (profile / hashtag / playlist) to a list of individual r

### Community 66 - "Community 66"
Cohesion: 1.0
Nodes (0): 

### Community 67 - "Community 67"
Cohesion: 1.0
Nodes (0): 

### Community 68 - "Community 68"
Cohesion: 1.0
Nodes (0): 

### Community 69 - "Community 69"
Cohesion: 1.0
Nodes (0): 

### Community 70 - "Community 70"
Cohesion: 1.0
Nodes (0): 

### Community 71 - "Community 71"
Cohesion: 1.0
Nodes (0): 

### Community 72 - "Community 72"
Cohesion: 1.0
Nodes (0): 

### Community 73 - "Community 73"
Cohesion: 1.0
Nodes (0): 

### Community 74 - "Community 74"
Cohesion: 1.0
Nodes (0): 

### Community 75 - "Community 75"
Cohesion: 1.0
Nodes (0): 

## Knowledge Gaps
- **88 isolated node(s):** `True when the error text indicates PG is still booting or unreachable.      Matc`, `Connect to Postgres, retrying transient startup errors.      Returns an open psy`, `Return a reusable PostgreSQL connection, or create a fresh one.`, `Return a connection to the pool (or close it if the pool is full).`, `Plain INSERT — raises DatabaseIntegrityError on unique constraint violation.` (+83 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 50`** (2 nodes): `VolumetricGlowBackground.tsx`, `VolumetricGlowBackground()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (2 nodes): `run_mass_youtube_test.py`, `run_test()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (2 nodes): `RootLayout()`, `layout.tsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 53`** (2 nodes): `Loading()`, `loading.tsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 54`** (2 nodes): `FullscreenLoadingScreen()`, `FullscreenLoadingScreen.tsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 55`** (2 nodes): `ViewportModalPortal.tsx`, `ViewportModalPortal()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 56`** (2 nodes): `youtubeIframeApi.ts`, `loadYouTubeIframeApi()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 57`** (1 nodes): `next-env.d.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 58`** (1 nodes): `tailwind.config.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 59`** (1 nodes): `postcss.config.js`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 60`** (1 nodes): `next.config.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 61`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 62`** (1 nodes): `Return True iff this adapter can handle the given URL.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 63`** (1 nodes): `Return the platform code ('yt' / 'ig' / 'tt') for a supported URL.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 64`** (1 nodes): `Download the media referenced by `url` into `workspace` and return an AdapterRes`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 65`** (1 nodes): `Resolve a feed-like URL (profile / hashtag / playlist) to a list of individual r`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 66`** (1 nodes): `quick_boundary_check.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 67`** (1 nodes): `material.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 68`** (1 nodes): `index.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 69`** (1 nodes): `health.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 70`** (1 nodes): `feed.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 71`** (1 nodes): `chat.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 72`** (1 nodes): `[...path].py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 73`** (1 nodes): `generate.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 74`** (1 nodes): `feedback.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 75`** (1 nodes): `types.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ReelService` connect `Community 2` to `Community 0`, `Community 1`, `Community 3`, `Community 5`, `Community 6`, `Community 28`?**
  _High betweenness centrality (0.205) - this node is a cross-community bridge._
- **Why does `IngestTranscriptCue` connect `Community 0` to `Community 3`, `Community 5`, `Community 7`, `Community 11`, `Community 14`, `Community 22`, `Community 26`?**
  _High betweenness centrality (0.103) - this node is a cross-community bridge._
- **Why does `YouTubeService` connect `Community 1` to `Community 0`, `Community 2`, `Community 3`, `Community 5`, `Community 6`, `Community 11`?**
  _High betweenness centrality (0.099) - this node is a cross-community bridge._
- **Are the 177 inferred relationships involving `ReelService` (e.g. with `ProbeResult` and `Core rate limit check against a pre-built key. Raises HTTPException on breach.`) actually correct?**
  _`ReelService` has 177 INFERRED edges - model-reasoned connections that need verification._
- **Are the 262 inferred relationships involving `IngestTranscriptCue` (e.g. with `ReelOut` and `_PlatformRateLimiter`) actually correct?**
  _`IngestTranscriptCue` has 262 INFERRED edges - model-reasoned connections that need verification._
- **Are the 140 inferred relationships involving `YouTubeService` (e.g. with `ProbeResult` and `Core rate limit check against a pre-built key. Raises HTTPException on breach.`) actually correct?**
  _`YouTubeService` has 140 INFERRED edges - model-reasoned connections that need verification._
- **Are the 227 inferred relationships involving `EmbeddingService` (e.g. with `ProbeResult` and `Core rate limit check against a pre-built key. Raises HTTPException on breach.`) actually correct?**
  _`EmbeddingService` has 227 INFERRED edges - model-reasoned connections that need verification._