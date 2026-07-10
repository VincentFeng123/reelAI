"""Single source of truth: paths, model names, free-tier limits, defaults.

All tunables live here so they're easy to update as Groq's free tier changes.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent          # /Users/.../clips

# override=True so the project's .env is authoritative over any stale vars the
# process may have inherited from the parent shell / supervisor environment.
load_dotenv(BASE_DIR / ".env", override=True)
WORK_DIR = BASE_DIR / "work"
OUTPUT_DIR = BASE_DIR / "output"
STATIC_DIR = Path(__file__).resolve().parent / "static"    # built React app
WORK_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Providers ──────────────────────────────────────────────────────────────
# Transcription: "supadata" (fast API) | "faster_whisper" (local) | "groq" (cloud Whisper)
TRANSCRIBER = os.environ.get("TRANSCRIBER", "supadata")
# Selection LLM: "gemini" (Google) | "groq" (OpenAI-compatible)
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")

# ── Supadata (transcript API) ────────────────────────────────────────────────
SUPADATA_API_KEY = os.environ.get("SUPADATA_API_KEY", "")
SUPADATA_BASE = os.environ.get("SUPADATA_BASE", "https://api.supadata.ai/v1")
# Small chunks → finer clip boundaries (captions are often unpunctuated).
SUPADATA_CHUNK_SIZE = int(os.environ.get("SUPADATA_CHUNK_SIZE", "180"))

# ── Playback / export ────────────────────────────────────────────────────────
# Main flow yields YouTube-embed clips (full quality, no download). Export cuts a
# downloadable .mp4 of a single clip on demand at this resolution.
EXPORT_RESOLUTION = int(os.environ.get("EXPORT_RESOLUTION", "1080"))

# ── Groq (used only when a provider above is set to "groq") ──────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
STT_MODEL = "whisper-large-v3-turbo"            # Groq Whisper (word timestamps)
LLM_PRIMARY = "openai/gpt-oss-120b"             # Groq strict json_schema
LLM_FALLBACK = "llama-3.3-70b-versatile"        # Groq json_object + retry
GROQ_MAX_FILE_MB = 25                            # free-tier upload ceiling
AUDIO_CHUNK_OVERLAP_S = 5                         # overlap when splitting audio > 25 MB

# ── Gemini (Google AI Studio; free tier) ────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
# Authoring model for the NEW topic-first calls ONLY (select_topics / extract_best_window) —
# the quality-critical redesign. Verified working via generate_json (generate_content +
# response_schema + thinking_budget=0). Kept OFF the high-volume authoring path (punctuation /
# units / content map stay on GEMINI_MODEL) so the Pro-preview latency/RPM cost lands only on
# the ~1 + N_kept selection/window calls per video. Empty / == GEMINI_MODEL disables the split.
TOPIC_MODEL = os.environ.get("TOPIC_MODEL", "gemini-3.1-pro-preview")

# ── Gemini-segment clip engine (default) ────────────────────────────────────
# A single Gemini pass reads the timestamped supadata transcript and returns substantive
# topic clips {title,start,end} directly — NO punctuation / structure understanding / whisper
# refine / multimodal. Uses the Pro model for real comprehension (section-level topics); boundaries are
# fine-snapped onto supadata's interpolated per-word times when SEGMENT_FINE_SNAP is on.
SEGMENT_MODEL = os.environ.get("SEGMENT_MODEL", TOPIC_MODEL)
SEGMENT_FINE_SNAP = os.environ.get("SEGMENT_FINE_SNAP", "1") not in ("0", "false", "")
SEGMENT_MIN_CLIP_S = 1.0                                             # fixed validity guard
SEGMENT_MAX_CLIP_S = 180.0                                           # reject; never hard-cut
SEGMENT_INFORMATIVENESS_MIN = float(os.environ.get("SEGMENT_INFORMATIVENESS_MIN", "0.6"))
SEGMENT_TOPIC_RELEVANCE_MIN = float(os.environ.get("SEGMENT_TOPIC_RELEVANCE_MIN", "0.6"))
SEGMENT_MAX_CLIPS = int(os.environ.get("SEGMENT_MAX_CLIPS", "40"))        # safety ceiling
# The whole-transcript plan (many topics × title+quotes) is a large JSON, and a thinking/pro
# model spends output budget on thinking too — so this cap must be well above the 8192 default
# or the plan JSON truncates mid-string.
SEGMENT_MAX_OUTPUT_TOKENS = int(os.environ.get("SEGMENT_MAX_OUTPUT_TOKENS", "24576"))

# ── faster-whisper (local transcription) ────────────────────────────────────
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")   # tiny|base|small|medium|large-v3
WHISPER_COMPUTE = os.environ.get("WHISPER_COMPUTE", "int8")  # int8 is fast on CPU
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")     # faster-whisper: cpu (no MPS)

# Precise-boundary REFINE uses a dedicated (usually larger) Whisper model than full
# transcription: the window is small so cost is modest, and word timestamps are more precise.
# Default "medium"; set REFINE_WHISPER_MODEL="" to fall back to WHISPER_MODEL. (~1.5 GB one-time
# download on first CPU/int8 run.)
REFINE_WHISPER_MODEL = os.environ.get("REFINE_WHISPER_MODEL", "medium") or WHISPER_MODEL
# VAD on the refine window pads speech segments (speech_pad_ms), giving usable silence margin.
REFINE_VAD = os.environ.get("REFINE_VAD", "1") not in ("0", "false", "")

# ── Precise boundary refinement ──────────────────────────────────────────────
# After the LLM picks rough ranges (from coarse Supadata captions), run Whisper on
# a small window around each boundary to snap start→sentence start, end→period.
PRECISE_BOUNDARIES = os.environ.get("PRECISE_BOUNDARIES", "1") not in ("0", "false", "")
BOUNDARY_PAD_S = float(os.environ.get("BOUNDARY_PAD_S", "10"))   # window half-width / max snap drift
# When no period-terminated sentence end with a usable trailing gap is in the window, the refine
# pass GROWS the window (pad→2·pad→4·pad…) and re-transcribes, up to this forward/backward reach.
MAX_BOUNDARY_SEARCH_S = float(os.environ.get("MAX_BOUNDARY_SEARCH_S", "45"))
# A word-gap must be at least this wide to count as a clean cut site (the cut lands inside it).
SILENCE_MIN_GAP_S = float(os.environ.get("SILENCE_MIN_GAP_S", "0.12"))
# HYBRID end policy (handoff §8): when the chosen complete-sentence end has no trailing pause,
# advance the clip END to the next period-terminated sentence WITH a gap only within this budget
# (~one sentence). Beyond it, best-available tight cut at the original end.
END_EXTEND_MAX_S = float(os.environ.get("END_EXTEND_MAX_S", "8"))
# Latency lever: the per-clip whisper edge-window passes in refine_clip_boundaries are
# independent, so they run concurrently over a thread pool sharing the singleton model.
# CPU-bound (local whisper), so cap near physical cores — over-subscribing thrashes.
# REFINE_WORKERS=1 restores the exact serial behavior (revert switch). Output-identical:
# each clip's window(s) are computed in isolation; only the loop order changes.
REFINE_WORKERS = int(os.environ.get("REFINE_WORKERS", "4"))

# ── Local models (relevance / dedup) ───────────────────────────────────────
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L6-v2"
BI_ENCODER = "all-MiniLM-L6-v2"
TORCH_DEVICE = os.environ.get("TORCH_DEVICE", "cpu")   # "cpu" | "mps"

# ── Download ───────────────────────────────────────────────────────────────
MAX_RESOLUTION = 720
VIDEO_FORMAT = (
    f"bestvideo[height<={MAX_RESOLUTION}]+bestaudio/best[height<={MAX_RESOLUTION}]"
)
AUDIO_FORMAT = "bestaudio/best"
TARGET_AUDIO_SR = 16000     # 16 kHz
TARGET_AUDIO_CH = 1         # mono
AUDIO_BITRATE_K = 48        # 48 kbps mono AAC ≈ < 22 MB / hour → under the 25 MB cap

# ── Free-tier rate limits (approximate, configurable) ──────────────────────
RPM_LIMIT = 30
TPM_LIMIT = 12_000
RPD_LIMIT = 1_000
TPD_LIMIT = 100_000
TPM_SAFETY = 0.85

# ── Token budgeting ────────────────────────────────────────────────────────
MAX_PROMPT_TOKENS = 8_000        # rendered transcript above this → Stage 2 prefilter
CHUNK_TOKEN_BUDGET = 6_000       # max tokens of chunks sent to the LLM
EXPECTED_OUTPUT_TOKENS = 2_500
SCAFFOLD_TOKENS = 1_000
CHARS_PER_TOKEN = 4

# ── Stage 2 chunking / cross-encoder ───────────────────────────────────────
WINDOW = 5
OVERLAP = 2
STRIDE = WINDOW - OVERLAP         # 3
CE_SCORE_FLOOR_ABS = 0.0          # ms-marco logit; > 0 ≈ sigmoid > 0.5 (relevant)
CE_SCORE_GAP = 4.0                # keep chunks within this logit gap of the top score
CE_TOP_N_CAP = 60                 # hard cap on chunks sent to the LLM

# ── Quote anchoring ────────────────────────────────────────────────────────
MIN_QUOTE_SCORE = 80              # rapidfuzz partial_ratio 0..100
QUOTE_TIME_WINDOW = 30.0          # sec; free locality budget
QUOTE_TIME_PENALTY = 0.5          # score points per sec beyond the window
QUOTE_WORDS = 8

# ── MMR de-dup + dynamic count ─────────────────────────────────────────────
MMR_LAMBDA = 0.6
REL_FLOOR = 0.28                  # topic cosine (vs enriched topic prototype); below = off-topic, stop
REDUNDANCY_SIM = 0.82            # cosine between two clips above which the later is a near-dup, skip
MAX_SEGMENTS = 8
OVERLAP_IOU_DROP = 0.5            # sentence-range IoU above which the dup is dropped

# ── Per-job settings (overridable via the request body) ────────────────────
DEFAULTS: dict = {
    "max_resolution": MAX_RESOLUTION,
    "allow_question_exclaim_ends": False,   # vestigial: sentences ending in '?'/'!' are now valid
                                            # clip ends too (see Sentence.ends_with_period)
    "mmr_lambda": MMR_LAMBDA,               # 1.0 = pure relevance, 0 = pure diversity
    "tail_pad_s": 0.15,   # trailing cushion — the cut lands in the gap AFTER the last word
    "lead_pad_s": 0.06,   # leading cushion — the cut lands in the gap BEFORE the first word
    "min_clip_duration_s": 15.0,     # a complete short thought can be brief
    "target_clip_duration_s": 45.0,  # Instagram-short AIM (scoring target, not a cutter)
    "max_clip_duration_s": 180.0,    # HARD ship cap / overflow ceiling (was 240). NOT a soft
                                     # 90 cut: the onset overflows the SOFT closure budget
                                     # (CLOSURE_MAX_SPAN_S=120, set in the Task 6 fix) up to
                                     # this hard cap. Must stay > CLOSURE_MAX_SPAN_S.
    # None → inherit: the ship cap becomes max(MAX_SEGMENTS, per-video anchor budget) so the
    # final truncation never undercuts the content-scaled budget (Q1a). A NUMBER here (or in
    # the request body) is an EXPLICIT user dial and is respected exactly — even when smaller.
    "max_clips": None,
    "transcription_mode": "groq",           # "groq" | "offline" (offline disabled in MVP)
    # ── structure-first per-job overrides (constants defined below) ──
    "analysis_profile": os.environ.get("ANALYSIS_PROFILE", "full"),  # "full" | "fast"
    "multimodal": None,                     # None → inherit config.MULTIMODAL
    "content_map_engine": None,             # None → inherit config.CONTENT_MAP_ENGINE
    "output_mode": os.environ.get("OUTPUT_MODE", "embed"),           # "embed" | "cut"
    "domain_override": None,                # force an adapter key, skip detection
    "anchor_selector": None,                # None → inherit config.ANCHOR_SELECTOR
    "arc_verify": None,                     # None → inherit config.ARC_VERIFY
    # None → content-scaled per-video budget (candidates.compute_anchor_budget):
    # clamp(ceil(n_anchor_eligible/4) + 4·[density=high], [MAX_ANCHORS, MAX_ANCHORS_CEIL]).
    # A NUMBER is an explicit user dial and wins outright (no scaling).
    "max_anchors": None,
    "refund_rounds": None,                  # None → inherit config.REFUND_ROUNDS (Q1e)
    "closure_max_span_s": 120.0,            # SOFT closure budget for NON-onset context; sits
                                            # BELOW the hard ship cap (max_clip_duration_s=180)
                                            # so the onset-overflow window (120, 180] is LIVE.
                                            # (This DEFAULTS value — not config.CLOSURE_MAX_SPAN_S —
                                            # is the one that reaches build_candidate in prod.)
    "min_comprehension_score": 0.70,
    "quality_floor": None,                  # None → inherit config.QUALITY_FLOOR (W25-G)
    "diarization": False,
    "edge_probe": None,                     # None → inherit config.EDGE_PROBE_ENABLED (VID2, default OFF)
    "clip_engine": None,                    # None → inherit config.CLIP_ENGINE (default "gemini")
}

# ── ffmpeg cutting ─────────────────────────────────────────────────────────
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "/opt/homebrew/bin/ffmpeg")
CRF = 20
PRESET = "veryfast"
AUDIO_BITRATE = "160k"            # output clip audio bitrate
CUT_CONCURRENCY = 3              # 2-3 parallel ffmpeg processes
# Cut encoder: h264_videotoolbox = Apple-Silicon HARDWARE H.264 — ~2.5× faster than libx264 and
# still frame-accurate (the `-ss`-before-`-i` + re-encode trims to the exact ms), so exact
# sentence/punctuation boundaries stay cheap. Falls back to libx264 per-clip if it errors.
# "libx264" restores the pure-software path. Bitrate (VT is rate-controlled, not CRF).
CUT_ENCODER = os.environ.get("CUT_ENCODER", "h264_videotoolbox")   # "h264_videotoolbox" | "libx264"
CUT_VIDEO_BITRATE = os.environ.get("CUT_VIDEO_BITRATE", "3000k")

# ── Server ─────────────────────────────────────────────────────────────────
HOST = "0.0.0.0"                 # bind 0.0.0.0 so phones on the LAN can reach it
PORT = 8000
DEV_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
]
SSE_HEARTBEAT_S = 15

# ── Backoff ────────────────────────────────────────────────────────────────
BACKOFF_MAX_RETRIES = 6
BACKOFF_BASE = 1.0
BACKOFF_CAP = 60.0
FALLBACK_RETRIES = 2

# ── Structure-first pipeline (understand-the-whole-video, then clip) ─────────
# analysis_profile: "full" runs the structure-first pipeline (units → roles →
# dependency graph → context-closure → clip-only-judge). "fast" is the legacy
# single-pass select_segments path (transcript-only, no download) and is also the
# universal graceful-degrade target when the full path can't run.
ANALYSIS_PROFILE = os.environ.get("ANALYSIS_PROFILE", "full")   # "full" | "fast"
# Multimodal perception (Phase 2+). Phase 1 is transcript-only regardless of this.
MULTIMODAL = os.environ.get("MULTIMODAL", "1") not in ("0", "false", "")
OUTPUT_MODE = os.environ.get("OUTPUT_MODE", "embed")           # "embed" | "cut"
# Cache the topic-independent Structure per video_id so re-clipping a video for a
# new topic skips ingest/perception/structuring and runs only the topic half.
STRUCTURE_CACHE = os.environ.get("STRUCTURE_CACHE", "1") not in ("0", "false", "")
FEED_DEFAULT_PROFILE = os.environ.get("FEED_DEFAULT_PROFILE", "fast")

# ── Topic-first clip engine (CLIP_ENGINE=topic) ─────────────────────────────
# "topic" and "unit" are explicit heavier experiments. "topic" selects substantive
# teaching topics from the content_map, then ships ONE
# best <=CLIP_MAX_S self-contained window per topic. "unit": legacy unit-anchored
# assemble_clips (revert switch). See docs/superpowers/specs/2026-07-04-topic-first-clipping-design.md
CLIP_ENGINE = os.environ.get("CLIP_ENGINE", "gemini")           # "gemini" | "topic" | "unit"
# SAFETY ceiling, not a curation dial: ship ALL substantive teaching topics (the type +
# TOPIC_INFORMATIVENESS_MIN filter is the only real gate). TreeSeg caps the topic tree at
# TREESEG_MAX_TOPICS=24, so this only binds on a runaway non-TreeSeg content map.
TOPIC_MAX_CLIPS = int(os.environ.get("TOPIC_MAX_CLIPS", "40"))
CLIP_TARGET_S = float(os.environ.get("CLIP_TARGET_S", "58"))    # window length aim
CLIP_MAX_S = float(os.environ.get("CLIP_MAX_S", "75"))          # hard-ish ceiling (finish the sentence)
TOPIC_INFORMATIVENESS_MIN = float(os.environ.get("TOPIC_INFORMATIVENESS_MIN", "0.5"))
TOPIC_BOUNDARY_WINDOW = int(os.environ.get("TOPIC_BOUNDARY_WINDOW", "3"))  # sentences of slack each side

# ── Content-type detection ──────────────────────────────────────────────────
DETECT_SAMPLE_SEGMENTS = 25
DETECT_TAIL_SEGMENTS = 8
DETECT_SAMPLE_CHARS = 3500

# ── Units / structure ───────────────────────────────────────────────────────
# Version stamp for the understanding-stage prompts (content map / unit extraction /
# dependency edges). Bump on ANY understanding-prompt change: build_structure stamps it
# into Structure.prompt_version and load_structure treats a mismatch as stale — the
# schema-compatible-but-old-prompts cache state finally has a version (SCHEMA_VERSION
# only guards the persisted SHAPE, never the prompts that produced the content).
UNDERSTANDING_PROMPT_VERSION = "understand-v1"
UNIT_TARGET_SEC = 45.0        # target atomic-unit length hint for the LLM
UNIT_MIN_SEC = 8.0
MAX_UNITS = 400               # safety cap on units for very long videos
CONTENT_MAP_MAX_SENTS_PER_CALL = 220   # chunk long videos for the structuring pass
# Latency lever: extract_units fires one LLM call PER topic and they are mutually
# independent, so they run concurrently over a thread pool; the deterministic
# post-pass (unit-id numbering, per-topic cursor clamp, MAX_UNITS truncation) stays
# serial in topic order, so the built Structure is output-neutral to worker count.
# UNDERSTAND_WORKERS=1 restores exact serial building (revert switch). Also caps the
# treeseg-labeling / llm-fallback chunk loops.
UNDERSTAND_WORKERS = int(os.environ.get("UNDERSTAND_WORKERS", "8"))
EMBED_SIM_THRESHOLD = 0.55    # adjacent-sentence cosine minima → topic boundary
# 3-level content map (chapter → topic → subtopic). Chapters group topics; subtopics split
# a topic at internal pause/discourse seams (embedding/scene signals are a Phase-2 upgrade).
CHAPTER_MAX_TOPICS = 5        # group at most this many topics into one chapter
CHAPTER_GAP_S = 8.0           # an inter-topic pause larger than this starts a new chapter
SUBTOPIC_MIN_SENTS = 6        # only split topics at least this long into subtopics
SUBTOPIC_MAX = 4             # cap subtopics per topic
# Dependency graph: bridge-concept prerequisite edges (EDM 2018) — a concept introduced earlier
# that reappears in a later unit signals that unit requires the earlier one, catching prerequisites
# the per-unit concepts_required missed (attacks the prerequisite-gap failure mode).
BRIDGE_PREREQ_EDGES = os.environ.get("BRIDGE_PREREQ_EDGES", "1") not in ("0", "false", "")
BRIDGE_MAX_PER_UNIT = 6      # cap bridge prerequisite edges added per unit (precision guard)

# ── Content-map engine ─────────────────────────────────────────────────────
# Content-map engine: "treeseg" = deterministic embedding-based divisive segmentation
# (arXiv:2407.12028) with a cheap LLM labeling pass; "llm" = the legacy per-chunk LLM
# boundary pass (also the graceful-degrade fallback when treeseg fails).
CONTENT_MAP_ENGINE = os.environ.get("CONTENT_MAP_ENGINE", "treeseg")   # "treeseg" | "llm"
TREESEG_TARGET_TOPIC_SEC = float(os.environ.get("TREESEG_TARGET_TOPIC_SEC", "120"))
TREESEG_MIN_TOPICS = 2
TREESEG_MAX_TOPICS = 24
TREESEG_MIN_TOPIC_SENTS = 3
TREESEG_COHERENCE_FLOOR = float(os.environ.get("TREESEG_COHERENCE_FLOOR", "0.0"))  # 0 = target-K driven
TREESEG_PAUSE_PRIOR = float(os.environ.get("TREESEG_PAUSE_PRIOR", "0.15"))
TREESEG_LABEL_BATCH = 12

# ── Anchors / candidate assembly ────────────────────────────────────────────
# Q1a content-scaled anchor budget: MAX_ANCHORS is the FLOOR (the old constant cap), the
# per-video budget is clamp(ceil(n_anchor_eligible / 4) + 4·[density == "high"],
# [MAX_ANCHORS, MAX_ANCHORS_CEIL]) — see candidates.compute_anchor_budget. An explicit
# settings["max_anchors"] bypasses the scaling entirely (user dial).
MAX_ANCHORS = 12             # floor of the per-video anchor budget (was the hard cap)
MAX_ANCHORS_CEIL = int(os.environ.get("MAX_ANCHORS_CEIL", "32"))   # budget ceiling
# Q1e refund loop (CCQGen pattern): after snap/dedupe, up to this many extra selection
# rounds re-run over the anchor-eligible units no surviving spec covers, refilling the
# budget dedupe collapsed. 0 disables refunds.
REFUND_ROUNDS = int(os.environ.get("REFUND_ROUNDS", "2"))
ANCHOR_MIN_PRIORITY = 45     # ignore anchor roles below this priority
ANCHOR_REL_FLOOR = 0.25      # topic cosine floor for an anchor to count as on-topic
# Anchor selector (Wave 2 P3). "plan" (default): ONE authoring-model call proposes WHAT to
# extract from the video's ACTUAL inventory (content map + unit table + detected arcs), with
# the adapter's anchor-priority table supplied as a PRIOR rather than a hard rule; the
# deterministic layer then enforces topic coverage, saturation quotas, arc retention, and
# the MAX_ANCHORS cap. "priority": the legacy flat priority sort, byte-equivalent to the
# pre-plan selector (the eval A/B lever). Plan-call failure or empty/garbage output
# auto-falls back to the priority sort ("plan-fallback") — stderr-logged + noted, never a
# crash (same policy as CONTENT_MAP_ENGINE).
ANCHOR_SELECTOR = os.environ.get("ANCHOR_SELECTOR", "plan")     # "plan" | "priority"
# W25-C role saturation is PER (role, home-topic), replacing the old video-global
# PLAN_ROLE_CAP=4: on qP 46 eligible claims competed for 4 video-wide slots, starving the
# claim-dense projectile zone (gold items 21-22) — saturation, not budget, was binding.
# Applies to BOTH enforce_plan and the legacy priority selector; the per-topic quota
# ceil(cap/n_topics)+1 is unchanged. Unmapped units (node_id "") share one pseudo-topic.
PLAN_ROLE_CAP_PER_TOPIC = 2  # no role holds more than this many anchors within one topic node
# W25-C real coverage floor: a topic node only counts COVERED when the selected entries'
# units span at least this fraction of the node's timed span — one 3.3s anchor no longer
# marks an 81s topic covered (the items-3/4 sliver class). Untimed nodes (end <= start)
# keep the legacy any-anchor rule.
MIN_NODE_COVERAGE = float(os.environ.get("MIN_NODE_COVERAGE", "0.5"))
# Verify detected instructional arcs with ONE batched LLM call per video (MathNet pattern:
# the model returns only the unit ids confirming each arc's problem/steps/answer; ids
# outside the arc are rule-discarded; rejected arcs are dropped). LLM failure degrades to
# unverified-but-kept with a note.
ARC_VERIFY = os.environ.get("ARC_VERIFY", "1") not in ("0", "false", "")
# W25-D(d) arc substance floor: ZERO-step arcs (concept-paired practice prompt→solution)
# whose member units sum below this many seconds are dropped AT DETECTION (arcs.py). On
# qP 8 of 11 detected arcs were 2-15s Socratic micro-pairs (two inside the sponsor
# promo) and micro-arcs are triple-protected downstream — saturation-exempt
# (candidates.py), dropped last in dedupe, min-duration snap-padded to 20s (refine.py) —
# so no later stage can filter them. Arcs with real steps are exempt from this floor.
MIN_ARC_SUBSTANCE_S = float(os.environ.get("MIN_ARC_SUBSTANCE_S", "12.0"))
# W25-D review-fix locality bound: the W25-D grammar broadening (closer terminals +
# neutral transparency) let the scan CRAWL — on kinematics it manufactured a phantom
# 302s "arc" (u0006 temperature prompt riding a 73.5s hop to an unrelated example's
# setup, then a 69s hop to a far physical_interpretation closer). A unit may only join
# via the two NEW paths (closer-as-terminal; pre-step opener accumulation) when it
# starts within this many seconds of the last member's end. Calibrated on the cached
# qP+kinematics structures: real joins on these paths max out at 13.1s while the
# phantom hops are 69.0/73.5s. Steps and TRUE result/solution terminals are deliberately
# NOT bounded (old-grammar semantics — kinematics' real arc_3 crawls 59s to its result).
MAX_ARC_MEMBER_GAP_S = float(os.environ.get("MAX_ARC_MEMBER_GAP_S", "30.0"))
CLOSURE_MAX_EXTRA_UNITS = 6  # context-closure growth budget (units)
CLOSURE_MAX_GAP_S = 25.0     # a context unit farther than this is referential, not inlined
CLOSURE_MAX_SPAN_S = 120.0   # SOFT closure budget for NON-onset context; must sit BELOW the
                             # hard ship cap (DEFAULTS["max_clip_duration_s"]=240) so the
                             # onset-overflow window (soft, hard] = (120, 240] is non-empty.

# ── Clip-only judge / repair (also the eval comprehension scorer) ───────────
JUDGE_ENABLED = os.environ.get("JUDGE_ENABLED", "1") not in ("0", "false", "")
JUDGE_MAX_REPAIR = 3
JUDGE_MIN_SCORE = 0.70       # comprehension threshold (headline metric cutoff)
JUDGE_WORKERS = int(os.environ.get("JUDGE_WORKERS", "4"))  # per-candidate judge/repair concurrency
# ^ Kept at 4 (NOT raised): a concurrency probe of the judge model showed the paid tier SOFT-
# THROTTLES per-call latency under load (mean 1.33s@4 → 1.61s@8 → 2.01s@12, no 429s), and the
# repair loop is per-candidate SEQUENTIAL (each verdict picks the next grow/trim move), so
# inflating per-call latency hurts those chains — raising workers to 8 measured NET SLOWER on
# assemble (a cold run went 71s→135s; mostly run-variance, but 8 gave no benefit). Verdicts are
# temp-0 and (unit_ids, text_hash)-keyed, so worker count never changes WHICH clips ship or the
# kill authority (unverified_kill stays 0) — purely a throughput knob. The real assemble-latency
# lever is batch-judging (deferred: kill-authority risk). PUNCT_WORKERS is decoupled below.
# Judging on a DIFFERENT model than the authoring passes avoids self-preference bias: an LLM judge
# favors its own generations (G-Eval EMNLP'23; Panickssery NeurIPS'24), inflating comprehension.
# Opt-in (requires a working second provider): set JUDGE_PROVIDER="groq" (needs a valid GROQ_API_KEY)
# or "gemini". Default "same" judges with the authoring model — no cross-model benefit, but never
# routes to an unconfigured provider. judge_clip falls back to the authoring model if the cross
# model errors, so a bad key degrades to biased-but-functional rather than failing the job.
JUDGE_PROVIDER = os.environ.get("JUDGE_PROVIDER", "same")
# ...or judge with a DIFFERENT Gemini model than the authoring GEMINI_MODEL (same working key, no
# second provider needed). A different model favors its own generations less — a partial but
# immediately-usable self-preference mitigation. Empty / == GEMINI_MODEL disables it.
# flash-lite keeps throughput high so the mitigation stays active under free-tier limits; set
# JUDGE_MODEL="gemini-2.5-pro" for a stronger (but rate-limited) judge.
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemini-2.5-flash-lite")
# W25-G: the final_quality ship floor (assemble step 5's quality filter), previously an
# inline 0.45 literal in assemble/__init__. Per-job override via settings["quality_floor"]
# (DEFAULTS key None → inherit this constant; an explicit number — even 0.0 — wins).
QUALITY_FLOOR = float(os.environ.get("QUALITY_FLOOR", "0.45"))
CONTEXT_CARD_MAX_WORDS = 40

# ── Multimodal perception (Phase 2/3; unused in Phase 1) ────────────────────
SCENE_THRESHOLD = 0.30
# Scene-cut keyframes come from a FULL-DECODE ffmpeg pass — the single slowest part of perception
# (~40-90s on a 20-min video). SCENE_DETECTION=0 drops it and keeps only the uniform grid + dHash
# dedup: much faster, but a visual shown only BETWEEN grid samples (e.g. a fast slide flip) can be
# missed. A/B on real content before flipping the default (user is evaluating this trade-off).
SCENE_DETECTION = os.environ.get("SCENE_DETECTION", "1") not in ("0", "false", "")
KEYFRAME_GRID_S = 15.0
KEYFRAME_MIN_GAP_S = 4.0
KEYFRAME_MAX = 400
DHASH_HAMMING_DROP = 4
OCR_ENGINE = os.environ.get("OCR_ENGINE", "none")             # none|gemini|easyocr|pytesseract
VISION_ENGINE = os.environ.get("VISION_ENGINE", "gemini_keyframes")
VISION_BATCH = 8
# Latency lever: keyframe caption batches are independent Gemini calls, run concurrently
# over a thread pool; captioned events are re-sorted by scene index before event_id
# assignment so OUTPUT is order-stable. VISION_WORKERS=1 restores serial captioning.
VISION_WORKERS = int(os.environ.get("VISION_WORKERS", "8"))
HF_TOKEN = os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGINGFACE_TOKEN", "")
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
DIARIZATION_ENABLED = os.environ.get("DIARIZATION", "0") not in ("0", "false", "")

# ── Video judge (Wave 4 item 21; ADVISORY-only, Gemini-only — Groq has no video) ──
# Tier 1 EDGE PROBE (VID2): after boundaries are final, cut the first/last ~N seconds of each
# SHIPPED clip locally (cut.build_cmd) and ask a video-capable Gemini (inline mp4 bytes, LOW
# media resolution) whether the audio starts/ends cleanly — catching the F7 mid-sentence-audio
# blind spot. It only adds WARNINGS + a tiny final_quality dock; it NEVER kills a clip and
# NEVER creates a Rejection. Default OFF (mirrors DIARIZATION_ENABLED): with it off, behavior is
# byte-identical. Per-job override via settings["edge_probe"] (DEFAULTS key None → inherit).
EDGE_PROBE_ENABLED = os.environ.get("EDGE_PROBE", "0") not in ("0", "false", "")
EDGE_PROBE_SECONDS = float(os.environ.get("EDGE_PROBE_SECONDS", "8"))   # head/tail probe length
# Tier 2 RENDER AUDIT (VID3): reserved — NOT implemented yet (Files-upload + per-clip offsets).
VIDEO_JUDGE_ENABLED = os.environ.get("VIDEO_JUDGE", "0") not in ("0", "false", "")
# The Gemini model both tiers judge video with (Groq cannot serve video regardless of
# JUDGE_PROVIDER). flash-lite keeps cost/throughput sane under free-tier limits.
VIDEO_JUDGE_MODEL = os.environ.get("VIDEO_JUDGE_MODEL", "gemini-2.5-flash-lite")
VIDEO_MEDIA_RESOLUTION = os.environ.get("VIDEO_MEDIA_RESOLUTION", "low")   # low|medium|high

# ── Punctuation restoration (raw timed words → readable sentences) ───────────
# Runs between transcription and sentence segmentation. It turns unpunctuated ASR captions into
# real sentences (preserving every word/timestamp) so downstream segmentation gets linguistic
# boundaries instead of time-windows. Fires only when the transcript looks under-punctuated
# (Supadata captions / sparse periods) unless PUNCTUATION_FORCE is set — Whisper output that is
# already punctuated is left alone to save LLM credits.
PUNCTUATION_ENABLED = os.environ.get("PUNCTUATION", "1") not in ("0", "false", "")
PUNCT_FORCE = os.environ.get("PUNCTUATION_FORCE", "0") not in ("0", "false", "")
PUNCTUATION_PROVIDER = os.environ.get("PUNCTUATION_PROVIDER", "") or LLM_PROVIDER
PUNCTUATION_MODEL = os.environ.get("PUNCTUATION_MODEL", "")     # "" → provider default model
PUNCT_TARGET_WORDS = int(os.environ.get("PUNCT_TARGET_WORDS", "500"))
PUNCT_OVERLAP_WORDS = int(os.environ.get("PUNCT_OVERLAP_WORDS", "60"))
PUNCT_MAX_WORDS = int(os.environ.get("PUNCT_MAX_WORDS", "700"))
PUNCT_MIN_WORDS = int(os.environ.get("PUNCT_MIN_WORDS", "300"))
PUNCT_PAUSE_SENTENCE_MS = int(os.environ.get("PUNCT_PAUSE_SENTENCE_MS", "700"))
PUNCT_PAUSE_COMMA_MS = int(os.environ.get("PUNCT_PAUSE_COMMA_MS", "250"))
PUNCT_MAX_RETRIES = int(os.environ.get("PUNCT_MAX_RETRIES", "2"))
# Decoupled from JUDGE_WORKERS and kept at 8: punctuation chunk calls are INDEPENDENT (no
# sequential chain), so firing all ~6-8 chunks in a single wave beats 2 waves even with the mild
# per-call latency inflation under concurrency (measured ~12.6s→8.7s). Few calls → low 429 risk.
PUNCT_WORKERS = int(os.environ.get("PUNCT_WORKERS", "8"))
PUNCT_SPEAKER_AWARE = os.environ.get("PUNCT_SPEAKER_AWARE", "1") not in ("0", "false", "")
