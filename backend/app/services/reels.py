import difflib
import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import threading
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Callable, Literal

import numpy as np

from ..config import get_settings
from ..db import DatabaseIntegrityError, dumps_json, execute_modify, fetch_all, fetch_one, now_iso, upsert
from . import llm_router
from .concepts import build_takeaways
from ..ingestion.errors import RateLimitedError as _IngestRateLimitedError
from ..ingestion.segment import normalize_clip_window as _normalize_clip_window_fn
from ..clip_engine import expand as _clip_engine_expand
from ..clip_engine.config import SEGMENT_MAX_CLIP_S as _SEGMENT_MAX_CLIP_S
from ..clip_engine.metadata import extract_video_id as _extract_embed_video_id
from .provider_registry import ProviderRegistry
from .segmenter import (
    SegmentMatch,
    TranscriptChunk,
    chunk_transcript,
    lexical_overlap_score,
    normalize_terms,
    select_segments,
)
from .topic_expansion import TopicExpansionService
from .structural_classifier import classify_passage
from .knowledge_level import effective_level_target

# Serving-side ceiling: legacy whole-video slabs (persisted before the engine's
# SEGMENT_MAX_CLIP_S curation gate existed) must not surface in ranked feeds.
_SERVING_MAX_CLIP_SEC = float(_SEGMENT_MAX_CLIP_S) + 15.0


def clip_spans_duplicate(
    a_start: float, a_end: float, b_start: float, b_end: float,
    threshold: float = 0.6,
) -> bool:
    """True when two clips of the SAME video cover substantially the same
    footage: overlap ≥ ``threshold`` of the shorter clip. Catches the
    near-duplicate spans that exact-key dedup misses (fine-snap jitter across
    concurrent generations)."""
    overlap = min(a_end, b_end) - max(a_start, b_start)
    if overlap <= 0:
        return False
    shorter = max(0.1, min(a_end - a_start, b_end - b_start))
    return overlap / shorter >= threshold

logger = logging.getLogger(__name__)


def _importance_ranker_enabled() -> bool:
    """Read ``REELS_IMPORTANCE_RANKER_ENABLED`` env flag (default on)."""
    raw = os.environ.get("REELS_IMPORTANCE_RANKER_ENABLED", "true")
    return raw.strip().lower() not in {"0", "false", "no", "off"}


_IMPORTANCE_TARGET_N = {"narrow": 2, "broad": 5, "medium": 3, "none": 3}
_STRUCTURAL_HARD_SKIP_LABELS = {"intro", "recap", "sponsor", "outro"}
_STRUCTURAL_EDGE_TRIM_LABELS = _STRUCTURAL_HARD_SKIP_LABELS | {"transition"}


# Playback padding applied to the ends of every refined reel window.
# YouTube's iframe/embed player seeks to the nearest keyframe AFTER the
# requested start timestamp (~100-300 ms jitter), which clips the first
# audible word. Ending precisely on a cue boundary can also truncate the
# trailing consonant. A small symmetric pre-roll/post-roll buffer
# reliably eliminates both without bleeding in noticeable extra content.
# Applied by ``_refine_clip_window_from_transcript`` for single-reel paths
# and by ``_split_into_consecutive_windows`` to the first/last reel of a
# chain only (middle reels chain exactly so consecutive-reel playback
# doesn't double-play the padded region).
REEL_PAD_START_SEC = 0.3
REEL_PAD_END_SEC = 0.3
STRICT_WORD_BOUNDARY_PRE_ROLL_SEC = 0.08
STRICT_WORD_BOUNDARY_POST_ROLL_SEC = 0.08

# Latency/cost guardrail for the clip-engine material->reels path: bound the
# number of paid discover / run.clip (Gemini) calls per generation so a single
# request can't fan out into an unbounded number of engine invocations.
MATERIAL_MAX_VIDEOS_PER_CONCEPT = 3   # discover cap per concept
MATERIAL_GEN_MAX_VIDEOS = 12          # hard ceiling on run.clip calls per generation


# A2: ratio-based auto-punctuation gate. Restoration fires on transcripts
# whose cue-terminal-punct ratio falls below this threshold. Set to `0.0`
# to disable restoration entirely, `1.0` to force it on every transcript.
# Default 0.25 catches both fully-bare auto-captions AND semi-punctuated
# transcripts that still have enough unpunctuated cues to trip the picker
# into ending mid-sentence.
def _read_punct_restore_threshold() -> float:
    raw = os.environ.get("PUNCT_RESTORE_THRESHOLD")
    if raw is None or not raw.strip():
        return 0.25
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return 0.25
    return max(0.0, min(1.0, v))


_PUNCT_RESTORE_THRESHOLD = _read_punct_restore_threshold()


class GenerationCancelledError(Exception):
    """Raised when the caller abandons reel generation mid-request."""


@dataclass
class QueryCandidate:
    text: str
    strategy: str
    confidence: float
    source_terms: list[str] = field(default_factory=list)
    weight: float = 1.0
    stage: str = "broad"
    source_surface: str = "youtube_api"
    family_key: str = ""
    source_family: str = ""
    anchor_mode: str = ""
    seed_video_id: str = ""
    seed_channel_id: str = ""


@dataclass
class RetrievalStagePlan:
    name: str
    queries: list[QueryCandidate]
    budget: int
    min_good_results: int
    max_budget: int | None = None


@dataclass
class TranscriptPrefetchTask:
    video_ids: tuple[str, ...]
    executor: ThreadPoolExecutor | None
    future_by_video_id: dict[str, Any] = field(default_factory=dict)
    cached_transcripts: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


@dataclass(frozen=True)
class PlannedQuery:
    text: str
    strategy: str
    stage: str
    confidence: float
    source_terms: tuple[str, ...] = ()
    weight: float = 1.0
    source_surface: str = "youtube_html"
    disambiguator: str | None = None
    normalization_key: str = ""
    cluster_key: str = ""
    rationale: str = ""
    family_key: str = ""
    source_family: str = ""
    anchor_mode: str = ""
    seed_video_id: str = ""
    seed_channel_id: str = ""


@dataclass(frozen=True)
class ConceptIntentPlan:
    strategy: str
    suffix: str
    rationale: str


@dataclass(frozen=True)
class ConceptSelectionDecision:
    concept_id: str
    concept_text: str
    concept_rank: int
    selected: bool
    reason: str


@dataclass(frozen=True)
class ConceptQueryPlan:
    concept_id: str
    concept_text: str
    concept_rank: int
    reason_selected: str
    literal_query: PlannedQuery
    intent_query: PlannedQuery
    selected_intent_strategy: str
    disambiguator: str | None
    normalization_key: str
    recovery_queries: tuple[PlannedQuery, ...] = ()
    expansion_queries: tuple[PlannedQuery, ...] = ()


@dataclass(frozen=True)
class QueryPlanningResult:
    selected_concepts: tuple[ConceptQueryPlan, ...]
    skipped_concepts: tuple[ConceptSelectionDecision, ...]
    global_queries: tuple[PlannedQuery, ...]
    total_selected_concepts: int
    total_first_pass_queries: int
    total_recovery_queries_allowed: int
    query_budget_exhausted: bool


RetrievalProfile = Literal["bootstrap", "deep"]


class _ChainBufferingEmitter:
    """Buffer reels belonging to a continuation chain until the chain ends.

    Continuation chains share a ``cluster_group_id``. Reels for an in-progress
    chain are staged in arrival order; once a different chain (or a
    chain-less reel) is observed, the previously-buffered chain flushes
    atomically so the NDJSON stream never interleaves reels inside a chain.
    ``drop_chain`` flushes the partial prefix when a chain is truncated
    mid-way; ``flush_all`` is the final safety net.
    """

    def __init__(self, downstream) -> None:
        self._downstream = downstream
        self._buffers: dict[str, list[dict[str, Any]]] = {}

    def emit(self, reel: dict[str, Any], *, chain_id: str) -> None:
        if self._downstream is None:
            return
        if not chain_id:
            self._flush_others(active_chain_id=None)
            self._forward(reel)
            return
        self._flush_others(active_chain_id=chain_id)
        self._buffers.setdefault(chain_id, []).append(reel)

    def drop_chain(self, chain_id: str) -> None:
        if not chain_id:
            return
        self._flush_chain(chain_id)

    def flush_all(self) -> None:
        for chain_id in list(self._buffers.keys()):
            self._flush_chain(chain_id)

    def _flush_others(self, *, active_chain_id: str | None) -> None:
        for chain_id in list(self._buffers.keys()):
            if chain_id != active_chain_id:
                self._flush_chain(chain_id)

    def _flush_chain(self, chain_id: str) -> None:
        buffered = self._buffers.pop(chain_id, [])
        for reel in buffered:
            self._forward(reel)

    def _forward(self, reel: dict[str, Any]) -> None:
        try:
            self._downstream(reel)
        except Exception:
            logger.exception(
                "chain_buffering downstream failed for reel_id=%s",
                str(reel.get("reel_id") or ""),
            )


class ReelService:
    FRONTIER_ROOT_EXACT = "root_exact"
    FRONTIER_ROOT_COMPANION = "root_companion"
    FRONTIER_ANCHORED_ADJACENT = "anchored_adjacent"
    FRONTIER_RECOVERY_GRAPH = "recovery_graph"
    MINING_STATE_UNMINED = "unmined"
    MINING_STATE_PARTIALLY_MINED = "partially_mined"
    MINING_STATE_HIGH_YIELD = "high_yield"
    MINING_STATE_LOW_YIELD = "low_yield"
    MINING_STATE_EXHAUSTED = "exhausted"
    VALID_VIDEO_POOL_MODES = {"short-first", "balanced", "long-form"}
    VALID_VIDEO_DURATION_PREFS = {"any", "short", "medium", "long"}
    DEFAULT_TARGET_CLIP_DURATION_SEC = 55
    MIN_TARGET_CLIP_DURATION_SEC = 15
    MAX_TARGET_CLIP_DURATION_SEC = 180
    MIN_TARGET_CLIP_DURATION_RANGE_GAP_SEC = 15
    DEFAULT_RETRIEVAL_PROFILE: RetrievalProfile = "bootstrap"
    BOOTSTRAP_CONCEPT_LIMIT = 4
    BOOTSTRAP_PRIMARY_QUERY_COUNT = 3
    BOOTSTRAP_RECOVERY_QUERY_COUNT = 2
    BOOTSTRAP_TRANSCRIPT_CANDIDATES = 6
    BOOTSTRAP_TRANSCRIPT_CANDIDATES_SERVERLESS = 4
    BOOTSTRAP_WEAK_POOL_MIN_KEPT = 3
    BOOTSTRAP_WEAK_POOL_MIN_TOP_SCORE = 0.24
    BOOTSTRAP_WEAK_POOL_MIN_UNIQUE_CHANNELS = 2
    BOOTSTRAP_WEAK_POOL_MIN_UNIQUE_STRATEGIES = 2
    GENERIC_CONTEXT_TERMS = {
        "basics",
        "basic",
        "beginner",
        "beginners",
        "concept",
        "concepts",
        "course",
        "crash",
        "definition",
        "definitions",
        "example",
        "examples",
        "explained",
        "explainer",
        "explanation",
        "fundamental",
        "fundamentals",
        "guide",
        "introduction",
        "intro",
        "learn",
        "learning",
        "lesson",
        "lessons",
        "overview",
        "practice",
        "problem",
        "problems",
        "proof",
        "review",
        "shorts",
        "study",
        "tutorial",
        "video",
        "walkthrough",
    }
    EDUCATIONAL_CUE_TERMS = {
        "class",
        "course",
        "derivation",
        "example",
        "examples",
        "explained",
        "explanation",
        "guide",
        "lesson",
        "lecture",
        "math",
        "practice",
        "problem",
        "problems",
        "proof",
        "review",
        "study",
        "theorem",
        "tutorial",
        "walkthrough",
    }
    STRONG_EDUCATIONAL_CUE_TERMS = {
        "analysis",
        "applications",
        "case",
        "derivation",
        "example",
        "examples",
        "experiment",
        "history",
        "lecture",
        "methods",
        "proof",
        "research",
        "theorem",
        "tutorial",
        "walkthrough",
    }
    LEXICON_CONFLICT_TOKENS = {
        "dictionary",
        "hindi",
        "meaning",
        "pronounce",
        "pronunciation",
        "sentence",
        "translate",
        "translation",
        "vocabulary",
        "word",
        "words",
    }
    LEXICON_CONFLICT_PHRASES: dict[str, float] = {
        "english to hindi": 0.42,
        "how to pronounce": 0.38,
        "how to say": 0.36,
        "ka matlab": 0.4,
        "meaning in": 0.42,
        "meaning of": 0.3,
        "pronunciation of": 0.34,
        "vocabulary lesson": 0.34,
        "word meaning": 0.34,
    }
    OFF_TOPIC_PHRASES: dict[str, float] = {
        "dental calculus": 0.34,
        "motivational speech": 0.24,
        "inspirational speech": 0.22,
        "inspirational video": 0.16,
        "official audio": 0.16,
        "official song": 0.2,
        "success motivation": 0.15,
        "teeth cleaning": 0.24,
        "tooth calculus": 0.32,
        "whatsapp status": 0.2,
        "motivation status": 0.2,
        "quotes shorts": 0.16,
        "music video": 0.2,
        "lyrical video": 0.12,
        "reaction video": 0.1,
    }
    OFF_TOPIC_TOKENS = {
        "affirmation",
        "compilation",
        "dental",
        "dentist",
        "edit",
        "grindset",
        "lyrics",
        "manifest",
        "meme",
        "motivation",
        "motivational",
        "music",
        "oral",
        "podcast",
        "plaque",
        "prank",
        "quotes",
        "reaction",
        "remix",
        "sigma",
        "song",
        "songs",
        "status",
        "tartar",
        "teeth",
        "tooth",
        "vlog",
    }
    MATH_TOPIC_TOKENS = {
        "algebra",
        "calculus",
        "derivative",
        "derivatives",
        "equation",
        "equations",
        "function",
        "functions",
        "geometry",
        "integral",
        "integrals",
        "limit",
        "limits",
        "math",
        "mathematics",
        "polynomial",
        "polynomials",
        "theorem",
        "theorems",
        "trigonometry",
    }
    DENTAL_CONFLICT_TOKENS = {
        "cleaning",
        "dental",
        "dentist",
        "hygiene",
        "oral",
        "plaque",
        "tartar",
        "teeth",
        "tooth",
    }
    ENTERTAINMENT_CONFLICT_TOKENS = {
        "audio",
        "dance",
        "karaoke",
        "lyric",
        "lyrics",
        "music",
        "official",
        "remix",
        "song",
        "songs",
        "vevo",
    }
    # Channels that use academic-sounding names but produce entertainment.
    # Checked against lowercased channel_title.
    KNOWN_ENTERTAINMENT_CHANNELS = {
        "the game theorists",
        "game theory",
        "matpat",
        "film theory",
        "food theory",
        "style theory",
        "the film theorists",
        "the food theorists",
        "the style theorists",
        "vsauce2",
        "vsauce3",
        "bright side",
        "5-minute crafts",
        "watchmojo",
        "watchmojo.com",
        "screen rant",
        "looper",
    }
    # Title patterns that indicate gaming/entertainment, not education.
    ENTERTAINMENT_TITLE_PATTERNS = (
        "doomed to die",
        "is actually",
        "isn't what you think",
        "you won't believe",
        "secret ending",
        "easter egg",
        "fan theory",
        "creepypasta",
        "horror game",
        "top 10",
        "top 5",
        "ranking every",
    )
    # Title patterns that indicate film/TV/entertainment media clips rather than
    # educational material. Critical for ambiguous queries like "calculus" where
    # movie scenes (e.g. "Mean Girls — Calculus Scene") can otherwise sneak in.
    # Matching is case-insensitive substring on `channel + " " + title`.
    MOVIE_TITLE_PATTERNS = (
        "movie clip",
        "movie scene",
        "film clip",
        "film scene",
        "official trailer",
        "official clip",
        "deleted scene",
        "full scene",
        "movie moment",
        "cinemacon",
        "- scene",
        " scene -",
        " scene (",
        "blu-ray",
        "behind the scenes",
        " (hd)",
        " (1080p)",
        " (4k)",
        "tv show",
        "sitcom",
    )
    AMBIGUOUS_CONCEPT_TOKENS = {
        "atom",
        "atoms",
        "bonds",
        "calculus",
        "cell",
        "cells",
        "current",
        "derivatives",
        "energy",
        "evolution",
        "field",
        "force",
        "function",
        "gravity",
        "group",
        "integrals",
        "java",
        "javascript",
        "limits",
        "loop",
        "loops",
        "matrix",
        "momentum",
        "power",
        "python",
        "relativity",
        "resistance",
        "revolution",
        "ring",
        "roots",
        "stress",
        "vector",
        "wave",
    }
    PROGRAMMING_TOKENS = {
        "algorithm",
        "algorithms",
        "api",
        "array",
        "arrays",
        "backend",
        "binary",
        "bug",
        "bugs",
        "code",
        "coding",
        "computer",
        "css",
        "data",
        "database",
        "frontend",
        "function",
        "functions",
        "javascript",
        "leetcode",
        "loop",
        "loops",
        "node",
        "object",
        "objects",
        "programming",
        "python",
        "react",
        "recursion",
        "sorting",
        "sql",
        "stack",
        "tree",
        "trees",
        "typescript",
    }
    PROGRAMMING_LANGUAGE_HINTS = {
        "c#",
        "c++",
        "go",
        "java",
        "javascript",
        "kotlin",
        "python",
        "react",
        "ruby",
        "rust",
        "sql",
        "swift",
        "typescript",
    }
    MATH_PHYSICS_TOKENS = {
        "algebra",
        "calculus",
        "derivative",
        "derivatives",
        "equation",
        "equations",
        "formula",
        "formulas",
        "function",
        "functions",
        "geometry",
        "integral",
        "integrals",
        "kinematics",
        "limit",
        "limits",
        "math",
        "mathematics",
        "matrix",
        "physics",
        "probability",
        "proof",
        "theorem",
        "vector",
        "vectors",
    }
    PROCESS_VISUAL_TOKENS = {
        "anatomy",
        "biology",
        "cell",
        "cells",
        "chemistry",
        "cycle",
        "division",
        "dna",
        "ecosystem",
        "enzyme",
        "enzymes",
        "metabolism",
        "mitosis",
        "molecule",
        "molecules",
        "pathway",
        "photosynthesis",
        "process",
        "reaction",
        "reactions",
        "respiration",
        "system",
        "systems",
    }
    HISTORY_HUMANITIES_TOKENS = {
        "anthropology",
        "civics",
        "culture",
        "economics",
        "government",
        "history",
        "humanities",
        "law",
        "literature",
        "philosophy",
        "politics",
        "psychology",
        "religion",
        "revolution",
        "society",
        "sociology",
        "war",
        "wars",
    }
    PROBLEM_SOLVING_TOKENS = {
        "application",
        "applications",
        "derive",
        "derivation",
        "derivations",
        "example",
        "examples",
        "exercise",
        "exercises",
        "practice",
        "problem",
        "problems",
        "proof",
        "solve",
        "solved",
        "solution",
        "solutions",
        "worked",
    }
    INTENT_SUFFIX_PRIORITY: dict[str, tuple[str, ...]] = {
        "programming": ("tutorial", "demo", "explained"),
        "problem_solving": ("worked example", "tutorial", "explained"),
        "process_visual": ("animation", "explained", "tutorial"),
        "history_humanities": ("explained", "documentary", "lecture"),
        "general": ("explained", "tutorial", "lecture"),
        "long_form_general": ("lecture", "tutorial", "explained"),
    }
    QUERY_STRATEGY_PRIOR: dict[str, float] = {
        "literal": 0.78,
        "explained": 0.82,
        "worked_example": 0.84,
        "tutorial": 0.83,
        "lecture": 0.79,
        "animation": 0.86,
        "demo": 0.8,
        "documentary": 0.76,
        "paraphrase": 0.72,
        "scene": 0.84,
        "object": 0.8,
        "action": 0.79,
        "broll": 0.87,
        "news_doc": 0.73,
        "tutorial_demo": 0.75,
        "recovery_adjacent": 0.64,
        "broadened_parent": 0.72,
    }
    SOURCE_SURFACE_PRIOR: dict[str, float] = {
        "youtube_api": 1.0,
        "youtube_html": 0.94,
        "youtube_related": 0.88,
        "youtube_channel": 0.8,
        "duckduckgo_site": 0.87,
        "bing_site": 0.85,
        "duckduckgo_quoted": 0.9,
        "bing_quoted": 0.88,
        "local_cache": 0.68,
    }
    CHANNEL_QUALITY_BONUS: dict[str, float] = {
        "news": 0.06,
        "education": 0.06,
        "known_educational": 0.12,
        "tutorial": 0.04,
        "stock_footage": 0.08,
        "podcast": -0.03,
        "low_quality_compilation": -0.11,
        "entertainment_media": -0.14,
    }
    # ----------------------------------------------------------------------- #
    # Scoring rubric — documents the weights used in _score_video_candidate.
    # Not used programmatically (yet); exists for transparency and auditability.
    # The discovery_feature_score weights sum to ~1.28 (normalised by 1.15+stage).
    # final_score = 0.74 * discovery + 0.26 * clipability
    # ----------------------------------------------------------------------- #
    SCORING_RUBRIC: dict[str, tuple[float, str]] = {
        # Discovery sub-features (weight, description)
        "semantic_title":              (0.26, "Embedding similarity: title vs concept terms"),
        "semantic_description":        (0.18, "Embedding similarity: description vs concept terms"),
        "specific_concept_anchor":     (0.13, "Anchor-term presence in metadata"),
        "query_alignment":             (0.12, "How well title/desc match the query text"),
        "strategy_prior":              (0.10, "Historical keep-rate for the search strategy"),
        "channel_quality":             (0.10, "Channel tier + feedback + engagement"),
        "educational_intent":          (0.10, "Structural edu signals (6-signal composite)"),
        "root_topic_alignment":        (0.09, "Root-topic term overlap"),
        "duration_fit":                (0.08, "Duration matches preferred range"),
        "visual_intent_match":         (0.05, "Visual scene spec keyword overlap"),
        "freshness_fit":               (0.04, "Publication recency"),
        "engagement_fit":              (0.03, "View-count signal"),
        # Top-level composite
        "clipability":                 (0.26, "Metadata-based clip potential (in final_score)"),
        "discovery":                   (0.74, "Aggregate discovery score (in final_score)"),
        # Post-hoc modifiers (applied as penalties/gates, not additive weights)
        "lexicon_noise":               (-0.22, "Penalty for off-topic/lexicon-conflict tokens"),
        "source_prior":                (1.0,  "Multiplicative factor by search source"),
        "transcript_coverage":         (0.0,  "Soft penalty when coverage < 90% (applied post-discovery)"),
        "llm_relevance":               (0.0,  "Pre-filter gate: drops clearly irrelevant candidates"),
    }
    KNOWN_EDUCATIONAL_CHANNELS: set[str] = {
        # Math
        "3blue1brown",
        "blackpenredpen",
        "dr. trefor bazett",
        "eddie woo",
        "mathologer",
        "mindyourdecisions",
        "nancypi",
        "numberphile",
        "patrickjmt",
        "pbs infinite series",
        "professor leonard",
        "richard e. borcherds",
        "stand-up maths",
        "think twice",
        "tibees",
        "tipping point math",
        "vihart",
        "zach star",
        # Physics
        "david butler",
        "drphysicsa",
        "fermilab",
        "for the allure of physics",
        "minutephysics",
        "physics explained",
        "physics girl",
        "physics videos by eugene",
        "physicshigh",
        "scienceclic english",
        "sixty symbols",
        "the science asylum",
        "viascience",
        "xylyxylyX",
        # Chemistry
        "chemical force",
        "chemistry in a nutshell",
        "david sherrill",
        "explosions&fire",
        "extractions&ire",
        "nilered",
        "nilered shorts",
        "nileblue",
        "nurdrage",
        "organic chemistry tutor",
        "periodic videos",
        "reactivechem",
        "rhodanide",
        "the organic chemistry tutor",
        "tmp chem",
        "tom's lab",
        # Biology
        "amoeba sisters",
        "animal fact files",
        "animalogic",
        "antscanada",
        "anthill art",
        "atlas pro",
        "basin79",
        "bbc earth",
        "ben g thomas",
        "cornell lab of ornithology",
        "crime pays but botany doesn't",
        "deep look",
        "evnautilus",
        "geologyhub",
        "history of the earth",
        "insecthausTV",
        "journey to the microcosmos",
        "kate tectonics",
        "made in the wild",
        "mbari",
        "microbehunter",
        "moth light media",
        "natural world facts",
        "nature on pbs",
        "nick zentner",
        "north 02",
        "pbs eons",
        "planet fungi",
        "plants insider",
        "quaoar power",
        "stated clearly",
        "stefan milo",
        "tierzoo",
        "andrew millison",
        "climate town",
        # Anatomy / Medicine
        "armando hasudungan",
        "chubbyemu",
        "institute of human anatomy",
        "ninja nerd",
        "sam webster",
        # General Science
        "alpha phoenix",
        "arxiv insights",
        "backstagescience",
        "be smart",
        "branch education",
        "domain of science",
        "kyle hill",
        "minute earth",
        "nighthawkinlight",
        "nottinghamscience",
        "real science",
        "sci show",
        "scishow",
        "scishow space",
        "science channel",
        "science marshall",
        "sciencephile the ai",
        "smarter every day",
        "smartereveryday",
        "steve mould",
        "the action lab",
        "up and atom",
        "veritasium",
        "verge science",
        # Space
        "anton petrov",
        "astrum",
        "chandra x-ray observatory",
        "cool worlds",
        "deep sky videos",
        "dr. becky",
        "event horizon",
        "everyday astronaut",
        "history of the universe",
        "isaac arthur",
        "jared owen",
        "john michael godier",
        "launch pad astronomy",
        "pbs space time",
        "primal space",
        "sabine hossenfelder",
        "scott manley",
        "sea",
        "the vintage space",
        # Engineering
        "agentjayz",
        "dejmian xyz simulations",
        "engineerguy",
        "engineering explained",
        "found and explained",
        "kyle.engineers",
        "lesics",
        "practical engineering",
        "real engineering",
        "the b1m",
        "the engineering mindset",
        # Electronics
        "afrotechmods",
        "ali hajimiri",
        "bigclivedotcom",
        "curiousmarc",
        "diodegonewild",
        "eevblog",
        "edisontechcenter",
        "electroboom",
        "esperantanaso",
        "fesz electronics",
        "greatscott!",
        "how to mechatronics",
        "jeri ellsworth",
        "mikeselectricstuff",
        "moritz klein",
        "mr carlson's lab",
        "simply electronics",
        "styropyro",
        "technology connections",
        "tesla500",
        "w2aew",
        # Computer Science
        "ben eater",
        "code bullet",
        "codeparade",
        "computerphile",
        "cs dojo",
        "deepmind",
        "fireship",
        "javidx9",
        "liveoverflow",
        "neso academy",
        "sebastian lague",
        "sentdex",
        "the coding train",
        "two minute papers",
        "welch labs",
        # Coding
        "bro code",
        "corey schafer",
        "freecodecamp",
        "javascript mastery",
        "kevin powell",
        "programming with mosh",
        "scrimba",
        "the net ninja",
        "web dev simplified",
        # Lectures & Courses
        "ak lectures",
        "bozeman science",
        "crash course",
        "harvard physics",
        "jeff hanson",
        "khan academy",
        "lumen learning",
        "mit opencourseware",
        "professor dave explains",
        "yalecourses",
        # General Explanation
        "art of the problem",
        "atomic frontier",
        "cgp grey",
        "facts in motion",
        "half as interesting",
        "joe scott",
        "koranos",
        "kurzgesagt",
        "lemmino",
        "melodysheep",
        "new mind",
        "reallifelore",
        "ted-ed",
        "tom scott",
        "vox",
        "vsauce",
        "vsauce2",
        "wendover productions",
        # Science Experiments & Building
        "applied science",
        "cody's lab",
        "huygens optics",
        "mark rober",
        "rctestflight",
        "sam zeloof",
        "stuffmadehere",
        "tech ingredients",
        "thebackyardscientist",
        "thought emporium",
        "tom stanton",
        # History
        "coneofcarc",
        "dan davis history",
        "drachinifel",
        "extra credits",
        "fall of civilizations",
        "fire of learning",
        "forgotten weapons",
        "historia civilis",
        "history buffs",
        "historymarche",
        "invicta",
        "jay foreman",
        "kings and generals",
        "knowledgia",
        "lindybeige",
        "military history visualized",
        "oversimplified",
        "sam o'nella academy",
        "tasting history with max miller",
        "the great war",
        "the history guy",
        "the operations room",
        "timeghost history",
        "toldinstone",
        "weird history",
        # Documentaries
        "dw documentary",
        "frontline pbs",
        # Nature & Environment
        "national geographic",
        "patagonia",
        # Workshop & Making
        "clickspring",
        "simone giertz",
        "this old tony",
        # Other Educational
        "bright side of mathematics",
        "brilliant",
        "captain disillusion",
        "company man",
        "jbstatistics",
        "not just bikes",
        "simply explained",
        "the royal institution",
        # Music Theory & Education
        "12tone",
        "8-bit music theory",
        "adam neely",
        "david bennett piano",
        "david bruce composer",
        "sideways",
        "tantacrul",
    }
    RISKY_SEARCH_SOURCES = {
        "bing_quoted",
        "bing_site",
        "duckduckgo_quoted",
        "duckduckgo_site",
        "local_cache",
    }
    STRICT_TOPIC_ROOT_ANCHOR_SOURCES = {
        "youtube_related",
        "youtube_channel",
        "bing_quoted",
        "bing_site",
        "duckduckgo_quoted",
        "duckduckgo_site",
        "local_cache",
    }
    QUERY_ALIGNMENT_NOISE_TOKENS = {
        "archive",
        "bite",
        "class",
        "complete",
        "concise",
        "course",
        "crash",
        "deep",
        "dive",
        "example",
        "examples",
        "explained",
        "explanation",
        "footage",
        "full",
        "fundamentals",
        "guide",
        "lesson",
        "lecture",
        "practice",
        "problem",
        "problems",
        "quick",
        "seconds",
        "short",
        "site",
        "tutorial",
        "video",
        "videos",
        "walk",
        "walkthrough",
        "watch",
        "worked",
        "youtube",
    }
    QUERY_EXPANSION_STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "core",
        "for",
        "from",
        "how",
        "ideas",
        "in",
        "into",
        "intuition",
        "is",
        "it",
        "learn",
        "of",
        "on",
        "or",
        "step",
        "steps",
        "study",
        "subtopic",
        "the",
        "their",
        "this",
        "to",
        "terminology",
        "using",
        "with",
        "within",
    }
    CONTROLLED_QUERY_SYNONYMS: dict[str, tuple[str, ...]] = {
        "big o": ("time complexity", "space complexity", "asymptotic analysis"),
        "big o notation": ("time complexity", "space complexity", "asymptotic analysis"),
        "binary search tree": ("BST", "tree insertion", "tree traversal"),
        "binary search trees": ("BST", "tree insertion", "tree traversal"),
        "breadth first search": ("BFS", "level order traversal", "graph traversal"),
        "cell signaling": ("signal transduction", "cell communication"),
        "chain rule": ("composite function derivative", "differentiating composite functions"),
        "depth first search": ("DFS", "graph traversal", "tree traversal"),
        "derivative": ("differentiation", "rate of change"),
        "derivatives": ("differentiation", "rate of change"),
        "integral": ("antiderivative", "integration"),
        "integrals": ("antiderivatives", "integration"),
        "joins": ("inner join", "left join", "outer join"),
        "mitosis": ("cell division", "cell cycle"),
        "photosynthesis": ("light dependent reactions", "calvin cycle"),
        "product rule": ("derivative of a product", "product differentiation"),
        "recursion": ("recursive function", "recursive call stack"),
        "signal transduction": ("cell signaling", "cell communication"),
        "sql join": ("inner join", "left join", "outer join"),
        "sql joins": ("inner join", "left join", "outer join"),
    }
    BROAD_TOPIC_SUBTOPICS: dict[str, tuple[str, ...]] = {
        "calculus": (
            "limits",
            "continuity",
            "derivatives",
            "chain rule",
            "product rule",
            "implicit differentiation",
            "integrals",
            "u substitution",
            "optimization",
            "fundamental theorem of calculus",
        ),
        "algebra": (
            "linear equations",
            "quadratic equations",
            "factoring",
            "systems of equations",
            "polynomials",
        ),
        "computer science": (
            "data structures",
            "algorithms",
            "recursion",
            "time complexity",
            "dynamic programming",
        ),
        "biology": (
            "cell signaling",
            "photosynthesis",
            "cell cycle",
            "genetics",
            "homeostasis",
        ),
        "economics": (
            "supply and demand",
            "opportunity cost",
            "elasticity",
            "market equilibrium",
            "macroeconomics",
            "microeconomics",
        ),
        "psychology": (
            "cognitive psychology",
            "behavioral psychology",
            "developmental psychology",
            "social psychology",
            "abnormal psychology",
            "conditioning",
        ),
        "statistics": (
            "descriptive statistics",
            "probability distributions",
            "hypothesis testing",
            "confidence intervals",
            "regression",
            "sampling",
        ),
    }
    CLIPABILITY_PENALTY_TOKENS = {
        "live stream",
        "podcast",
        "full episode",
        "compilation",
        "reaction",
    }
    # ---- retrieval & transcript parallelism ----
    # Bumped from 6 to 10. The biggest single wall-clock cost in reel
    # generation is fetching transcripts from YouTube — each call takes a
    # few seconds and is almost entirely spent waiting on the network.
    # With 10 workers instead of 6 we can fetch 60% more transcripts in
    # the same wall-clock time, and the underlying HTTP session pool
    # (`SESSION_POOL_SIZE = 48`) is now large enough to support it.
    #
    # Bigger numbers (20+) were tested but started hitting intermittent
    # 429 rate-limit responses from YouTube — 10 is the sweet spot we
    # landed on.
    QUERY_RETRIEVAL_WORKERS_FAST = 10
    QUERY_RETRIEVAL_WORKERS_SLOW = 10
    TRANSCRIPT_FETCH_WORKERS_FAST = 10
    TRANSCRIPT_FETCH_WORKERS_SLOW = 10
    # Bump whenever the cached row shape changes so stale entries are invalidated.
    # v4: video_id retained on response rows (was stripped in v3).
    # v5: reel rows now originate from _persist_ingest path (T4 clip-engine swap).
    RANKED_FEED_CACHE_VERSION = 7
    REFILL_STAGE_EXACT_ROOT = 0
    REFILL_STAGE_ROOT_COMPANION = 1
    REFILL_STAGE_MULTI_CLIP_STRICT = 2
    REFILL_STAGE_ANCHORED_ADJACENT = 3
    REFILL_STAGE_MULTI_CLIP_EXPANDED = 4
    REFILL_STAGE_RECOVERY_GRAPH = 5
    MAX_REFILL_STAGE = REFILL_STAGE_RECOVERY_GRAPH

    def __init__(self, embedding_service, youtube_service, ingestion_pipeline=None) -> None:
        settings = get_settings()
        self.embedding_service = embedding_service
        self.youtube_service = youtube_service
        self.ingestion_pipeline = ingestion_pipeline
        self.chat_model = settings.gemini_model
        self.retrieval_engine_v2_enabled = bool(settings.retrieval_engine_v2_enabled)
        self.retrieval_tier2_enabled = bool(settings.retrieval_tier2_enabled)
        self.retrieval_debug_logging = bool(settings.retrieval_debug_logging)
        self._min_relevance_threshold = 0.0
        self.serverless_mode = bool(
            os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME") or os.getenv("K_SERVICE")
        )
        self._strategy_history_cache: dict[str, float] = {}
        self._strategy_history_cache_lock = threading.Lock()
        self._strategy_history_cache_max_size = 10_000
        self.llm_available = llm_router.gemini_or_groq_available()
        self.topic_expansion_service = TopicExpansionService()
        self._provider_registry = ProviderRegistry()
        # Cache of video_id → provider so bare-id transcript fetches route
        # correctly once a non-YouTube row has been seen. Populated by
        # `_register_video_provider` (called from search wiring in
        # youtube.py:_search_external_fallbacks) and consulted by
        # `_get_transcript` when the caller only has a video_id string.
        self._provider_by_video_id: dict[str, str] = {}

    def _reel_scope_where(
        self,
        *,
        material_id: str,
        generation_id: str | None,
        alias: str | None = None,
    ) -> tuple[str, tuple[Any, ...]]:
        prefix = f"{alias}." if alias else ""
        if generation_id:
            return f"{prefix}generation_id = ?", (generation_id,)
        return f"{prefix}material_id = ? AND COALESCE({prefix}generation_id, '') = ''", (material_id,)

    def _ranked_feed_cache_key(
        self,
        *,
        material_id: str,
        generation_id: str | None,
        fast_mode: bool,
        subject_tag: str | None = None,
        strict_topic_only: bool = False,
        level_target: float = 0.5,
    ) -> str:
        # Include subject_tag + strict_topic_only in the key because ranked_feed's
        # relevance gates (`_passes_relevance_gate`, strict topic filter, hard-
        # blocked check) consult them and their output is baked into the cached
        # rows. Without this the cache would return stale pre-filtered data when
        # the caller's context (topic/source) changed.
        # level_target is included so a level change invalidates the cached order.
        payload = {
            "material_id": material_id,
            "generation_id": generation_id or "",
            "fast_mode": bool(fast_mode),
            "subject_tag": (subject_tag or "").strip().lower(),
            "strict_topic_only": bool(strict_topic_only),
            "level_target": round(float(level_target), 3),
            "version": self.RANKED_FEED_CACHE_VERSION,
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _ranked_feed_source_fingerprint(
        self,
        conn,
        *,
        material_id: str,
        generation_id: str | None,
        fast_mode: bool,
        subject_tag: str | None,
        reel_where: str,
        reel_params: tuple[Any, ...],
    ) -> str:
        relevant_reels_cte = f"""
            WITH relevant_reels AS (
                SELECT r.id, r.video_id, r.created_at
                FROM reels r
                WHERE {reel_where}
                  AND (r.t_end - r.t_start) >= 1
                  AND (r.t_end - r.t_start) <= {_SERVING_MAX_CLIP_SEC}
            )
        """
        reel_stats = fetch_one(
            conn,
            relevant_reels_cte
            + """
            SELECT
                COUNT(*) AS reel_count,
                COALESCE(MAX(relevant_reels.created_at), '') AS reel_updated_at,
                COALESCE(MAX(v.created_at), '') AS video_updated_at,
                COALESCE(COUNT(f.reel_id), 0) AS feedback_count,
                COALESCE(MAX(f.created_at), '') AS feedback_updated_at
            FROM relevant_reels
            LEFT JOIN videos v ON v.id = relevant_reels.video_id
            LEFT JOIN reel_feedback f ON f.reel_id = relevant_reels.id
            """,
            reel_params,
        ) or {}
        transcript_stats = fetch_one(
            conn,
            relevant_reels_cte
            + """
            SELECT
                COUNT(*) AS transcript_count,
                COALESCE(MAX(tc.created_at), '') AS transcript_updated_at
            FROM transcript_cache tc
            WHERE tc.video_id IN (SELECT DISTINCT video_id FROM relevant_reels)
            """,
            reel_params,
        ) or {}
        concept_stats = fetch_one(
            conn,
            """
            SELECT
                COUNT(*) AS concept_count,
                COALESCE(MAX(created_at), '') AS concept_updated_at
            FROM concepts
            WHERE material_id = ?
            """,
            (material_id,),
        ) or {}
        summary_mode = "fallback" if fast_mode or not self.llm_available else f"ai:{self.chat_model}"
        payload = {
            "version": self.RANKED_FEED_CACHE_VERSION,
            "material_id": material_id,
            "generation_id": generation_id or "",
            "fast_mode": bool(fast_mode),
            "summary_mode": summary_mode,
            "subject_tag": subject_tag or "",
            "reel_count": int(reel_stats.get("reel_count") or 0),
            "reel_updated_at": str(reel_stats.get("reel_updated_at") or ""),
            "video_updated_at": str(reel_stats.get("video_updated_at") or ""),
            "feedback_count": int(reel_stats.get("feedback_count") or 0),
            "feedback_updated_at": str(reel_stats.get("feedback_updated_at") or ""),
            "transcript_count": int(transcript_stats.get("transcript_count") or 0),
            "transcript_updated_at": str(transcript_stats.get("transcript_updated_at") or ""),
            "concept_count": int(concept_stats.get("concept_count") or 0),
            "concept_updated_at": str(concept_stats.get("concept_updated_at") or ""),
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _load_ranked_feed_cache(
        self,
        conn,
        *,
        material_id: str,
        generation_id: str | None,
        fast_mode: bool,
        source_fingerprint: str,
        subject_tag: str | None = None,
        strict_topic_only: bool = False,
        level_target: float = 0.5,
    ) -> list[dict[str, Any]] | None:
        cache_key = self._ranked_feed_cache_key(
            material_id=material_id,
            generation_id=generation_id,
            fast_mode=fast_mode,
            subject_tag=subject_tag,
            strict_topic_only=strict_topic_only,
            level_target=level_target,
        )
        cached = fetch_one(
            conn,
            "SELECT source_fingerprint, response_json FROM ranked_feed_cache WHERE cache_key = ?",
            (cache_key,),
        )
        if not cached:
            return None
        if str(cached.get("source_fingerprint") or "") != source_fingerprint:
            return None
        try:
            payload = json.loads(str(cached.get("response_json") or "[]"))
        except (TypeError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, list) else None

    def _store_ranked_feed_cache(
        self,
        conn,
        *,
        material_id: str,
        generation_id: str | None,
        fast_mode: bool,
        source_fingerprint: str,
        reels: list[dict[str, Any]],
        subject_tag: str | None = None,
        strict_topic_only: bool = False,
        level_target: float = 0.5,
    ) -> None:
        timestamp = now_iso()
        upsert(
            conn,
            "ranked_feed_cache",
            {
                "cache_key": self._ranked_feed_cache_key(
                    material_id=material_id,
                    generation_id=generation_id,
                    fast_mode=fast_mode,
                    subject_tag=subject_tag,
                    strict_topic_only=strict_topic_only,
                    level_target=level_target,
                ),
                "material_id": material_id,
                "generation_id": generation_id or "",
                "fast_mode": 1 if fast_mode else 0,
                "source_fingerprint": source_fingerprint,
                "response_json": dumps_json(reels),
                "created_at": timestamp,
                "updated_at": timestamp,
            },
            pk="cache_key",
        )

    def generate_reels(
        self,
        conn,
        material_id: str,
        concept_id: str | None,
        num_reels: int,
        creative_commons_only: bool,
        exclude_video_ids: list[str] | None = None,
        fast_mode: bool = False,
        video_pool_mode: str = "short-first",
        preferred_video_duration: str = "any",
        target_clip_duration_sec: int = DEFAULT_TARGET_CLIP_DURATION_SEC,
        target_clip_duration_min_sec: int | None = None,
        target_clip_duration_max_sec: int | None = None,
        dry_run: bool = False,
        retrieval_profile: RetrievalProfile = DEFAULT_RETRIEVAL_PROFILE,
        generation_id: str | None = None,
        exclude_generation_ids: list[str] | None = None,
        min_relevance_threshold: float = 0.0,
        page_hint: int = 1,
        recovery_stage: int = 0,
        on_reel_created: Callable[[dict[str, Any]], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        multi_platform_search: bool = False,
    ) -> list[dict[str, Any]]:
        def raise_if_cancelled() -> None:
            if should_cancel is None:
                return
            try:
                if should_cancel():
                    raise GenerationCancelledError("Generation cancelled.")
            except GenerationCancelledError:
                raise
            except Exception:
                return

        raise_if_cancelled()
        chain_emitter = _ChainBufferingEmitter(on_reel_created) if on_reel_created is not None else None
        self._min_relevance_threshold = max(0.0, min(1.0, float(min_relevance_threshold)))
        safe_page_hint = max(1, int(page_hint or 1))
        safe_recovery_stage = max(0, int(recovery_stage or 0))
        allow_adjacent_recovery = self._allow_adjacent_recovery(
            page_hint=safe_page_hint,
            recovery_stage=safe_recovery_stage,
        )
        safe_retrieval_profile = self._normalize_retrieval_profile(retrieval_profile)
        params: tuple[Any, ...] = (material_id,)
        concept_where = "WHERE material_id = ?"
        if concept_id:
            concept_where += " AND id = ?"
            params = (material_id, concept_id)

        concepts = fetch_all(
            conn,
            (
                "SELECT id, material_id, title, keywords_json, summary, embedding_json, created_at "
                f"FROM concepts {concept_where} ORDER BY created_at ASC"
            ),
            params,
        )
        material = fetch_one(
            conn,
            "SELECT subject_tag, source_type, knowledge_level, level_adjustment FROM materials WHERE id = ?",
            (material_id,),
        )
        material_knowledge_level = str((material or {}).get("knowledge_level") or "beginner")
        subject_tag = str((material or {}).get("subject_tag") or "").strip() or None
        strict_topic_only = str((material or {}).get("source_type") or "").strip().lower() == "topic"
        if strict_topic_only and subject_tag:
            # Spellcheck ONCE so the wiki concept expansion and every downstream
            # search seed off a clean topic ('pychology' → junk concepts like
            # 'Contemporary Jewry' poisoned whole generations). Persisted so the
            # read side (ranked_feed anchors, cache fingerprints, concept sync)
            # sees the same subject the reels were generated for.
            _corrected = self._corrected_subject_tag(subject_tag)
            if _corrected != subject_tag:
                execute_modify(
                    conn,
                    "UPDATE materials SET subject_tag = ? WHERE id = ?",
                    (_corrected, material_id),
                )
                subject_tag = _corrected
        if not concepts and strict_topic_only and subject_tag:
            concepts = self._build_topic_only_concepts_from_expansion(
                conn,
                material_id=material_id,
                subject_tag=subject_tag,
            )
        if not concepts:
            return []
        strict_topic_expansion: dict[str, Any] | None = None
        root_topic_terms: list[str] = []
        if strict_topic_only and subject_tag:
            strict_topic_expansion = self.topic_expansion_service.expand_topic(
                conn,
                topic=subject_tag,
                max_subtopics=10,
                max_aliases=8,
                max_related_terms=8,
            )
            concepts = self._sync_topic_expansion_concepts(
                conn,
                material_id=material_id,
                concepts=concepts,
                subject_tag=subject_tag,
                expansion=strict_topic_expansion,
            )
            root_topic_terms = self._topic_root_anchor_terms(
                subject_tag=subject_tag,
                expansion=strict_topic_expansion,
                concepts=concepts,
            )
            if safe_retrieval_profile == "bootstrap":
                concepts = self._bootstrap_topic_retrieval_concepts(
                    conn=conn,
                    concepts=concepts,
                    subject_tag=subject_tag,
                    material_id=material_id,
                )
        concepts = self._order_concepts(conn, material_id, concepts)
        safe_video_pool_mode = self._normalize_video_pool_mode(video_pool_mode)
        safe_video_duration_pref = self._normalize_preferred_video_duration(preferred_video_duration)
        clip_min_len, clip_max_len, safe_target_clip_duration = self._resolve_clip_duration_bounds(
            target_clip_duration_sec=target_clip_duration_sec,
            target_clip_duration_min_sec=target_clip_duration_min_sec,
            target_clip_duration_max_sec=target_clip_duration_max_sec,
        )

        existing_reels_where, existing_reels_params = self._reel_scope_where(
            material_id=material_id,
            generation_id=generation_id,
        )
        if generation_id:
            existing_clip_rows = fetch_all(
                conn,
                f"SELECT video_id, t_start, t_end, transcript_snippet FROM reels WHERE {existing_reels_where}",
                existing_reels_params,
            )
            existing_video_rows = fetch_all(
                conn,
                f"SELECT video_id, COUNT(*) AS reel_count FROM reels WHERE {existing_reels_where} GROUP BY video_id",
                existing_reels_params,
            )
        else:
            existing_clip_rows = fetch_all(
                conn,
                """
                SELECT video_id, t_start, t_end, transcript_snippet
                FROM reels
                WHERE material_id = ?
                  AND COALESCE(generation_id, '') = ''
                """,
                (material_id,),
            )
            existing_video_rows = fetch_all(
                conn,
                """
                SELECT video_id, COUNT(*) AS reel_count
                FROM reels
                WHERE material_id = ?
                  AND COALESCE(generation_id, '') = ''
                GROUP BY video_id
                """,
                (material_id,),
            )
        existing_clip_keys = {
            self._clip_key(
                str(r.get("video_id") or ""),
                float(r.get("t_start") or 0),
                float(r.get("t_end") or 0),
            )
            for r in existing_clip_rows
            if r.get("video_id")
        }
        accepted_clip_contexts_by_video: dict[str, list[dict[str, Any]]] = {}
        for row in existing_clip_rows:
            video_id = str(row.get("video_id") or "").strip()
            if not video_id:
                continue
            context = self._build_clip_context(
                text=str(row.get("transcript_snippet") or ""),
                clip_duration_sec=float(row.get("t_end") or 0.0) - float(row.get("t_start") or 0.0),
            )
            accepted_clip_contexts_by_video.setdefault(video_id, []).append(context)
        existing_video_counts = {
            str(r["video_id"]): int(r["reel_count"] or 0)
            for r in existing_video_rows
            if r.get("video_id")
        }
        generated_video_counts: dict[str, int] = {}
        generated_clip_keys: set[str] = set()
        default_max_segments_per_video = self._request_page_segment_cap(
            video_duration_sec=0,
            fast_mode=fast_mode,
            page_hint=safe_page_hint,
        )
        excluded_generation_ids = [
            candidate_id
            for candidate_id in (exclude_generation_ids or [])
            if candidate_id and candidate_id != generation_id
        ]
        if excluded_generation_ids:
            placeholders = ", ".join(["?"] * len(excluded_generation_ids))
            excluded_clip_rows = fetch_all(
                conn,
                f"SELECT video_id, t_start, t_end, transcript_snippet FROM reels WHERE generation_id IN ({placeholders})",
                tuple(excluded_generation_ids),
            )
            excluded_video_rows = fetch_all(
                conn,
                f"SELECT video_id, COUNT(*) AS reel_count FROM reels WHERE generation_id IN ({placeholders}) GROUP BY video_id",
                tuple(excluded_generation_ids),
            )
            for row in excluded_clip_rows:
                video_id = str(row.get("video_id") or "")
                if not video_id:
                    continue
                existing_clip_keys.add(
                    self._clip_key(
                        video_id,
                        float(row.get("t_start") or 0),
                        float(row.get("t_end") or 0),
                    )
                )
                context = self._build_clip_context(
                    text=str(row.get("transcript_snippet") or ""),
                    clip_duration_sec=float(row.get("t_end") or 0.0) - float(row.get("t_start") or 0.0),
                )
                accepted_clip_contexts_by_video.setdefault(video_id, []).append(context)
            for row in excluded_video_rows:
                video_id = str(row.get("video_id") or "")
                if not video_id:
                    continue
                existing_video_counts[video_id] = existing_video_counts.get(video_id, 0) + int(
                    row.get("reel_count") or 0
                )
        generated: list[dict[str, Any]] = []
        if strict_topic_only and subject_tag:
            if safe_retrieval_profile == "deep":
                topic_expansion = self._deep_topic_expansion(
                    conn,
                    material_id=material_id,
                    subject_tag=subject_tag,
                    generation_id=generation_id,
                )
                concepts = self._sync_topic_expansion_concepts(
                    conn,
                    material_id=material_id,
                    concepts=concepts,
                    subject_tag=subject_tag,
                    expansion=topic_expansion,
                )
                root_topic_terms = self._topic_root_anchor_terms(
                    subject_tag=subject_tag,
                    expansion=topic_expansion,
                    concepts=concepts,
                )
        accumulated_exclusions: list[str] = list(exclude_video_ids or [])
        # Finding #3: exclude videos we've already clipped for this generation and
        # videos from the excluded prior generations (exclude_generation_ids) so
        # refinement/extension doesn't re-pay the engine to re-discover + re-clip the
        # same videos. `existing_video_counts` aggregates both scans; rows carry
        # `yt:`-prefixed ids while discover matches BARE ids, so normalize.
        _already_excluded = set(accumulated_exclusions)
        for _prior_video_id in existing_video_counts:
            _bare = str(_prior_video_id or "").strip().split(":", 1)[-1]
            if _bare and _bare not in _already_excluded:
                accumulated_exclusions.append(_bare)
                _already_excluded.add(_bare)
        # The user's actual topic (root concept) always gets the first tranche of
        # the video budget: _order_concepts sorts by (mastery, reel_count, ...) so
        # once the root has reels it sinks behind fresh wiki subtopics, starving
        # the real topic on refinement pages.
        if strict_topic_only and subject_tag:
            _subject_key = self._normalize_query_key(subject_tag)
            _root_idx = next(
                (
                    i
                    for i, c in enumerate(concepts)
                    if self._normalize_query_key(str(c.get("title") or "")) == _subject_key
                ),
                -1,
            )
            if _root_idx > 0:
                concepts.insert(0, concepts.pop(_root_idx))
        videos_processed = 0
        total_concepts = len(concepts)

        for idx, concept in enumerate(concepts):
            raise_if_cancelled()
            # RAW-PRACTICE: no `len(generated) >= num_reels` break — persistence
            # of engine clips is no longer truncated by num_reels; only the COST
            # guardrail below (MATERIAL_GEN_MAX_VIDEOS) bounds the paid work.
            if videos_processed >= MATERIAL_GEN_MAX_VIDEOS:
                break
            topic = self._concept_topic_query(concept)
            if not topic:
                continue
            video_budget = min(
                MATERIAL_MAX_VIDEOS_PER_CONCEPT,
                MATERIAL_GEN_MAX_VIDEOS - videos_processed,
            )
            if video_budget <= 0:
                break

            def _stream(reel_obj, _concept=concept, _idx=idx):
                if chain_emitter is not None:
                    chain_emitter.emit(
                        self._reel_attribution_to_dict(reel_obj, _concept, _idx, total_concepts),
                        chain_id="",
                    )

            try:
                reels, resolved_ids = self.ingestion_pipeline.ingest_topic(
                    topic=topic,
                    material_id=material_id,
                    concept_id=concept["id"],
                    generation_id=generation_id,
                    exclude_video_ids=accumulated_exclusions,
                    target_clip_duration_sec=safe_target_clip_duration,
                    target_clip_duration_min_sec=clip_min_len,
                    target_clip_duration_max_sec=clip_max_len,
                    language="en",
                    knowledge_level=material_knowledge_level,
                    max_videos=video_budget,
                    max_reels=None,
                    on_reel_created=(None if dry_run else _stream),
                    dry_run=dry_run,
                )
            except _IngestRateLimitedError:
                # Process-wide rate limit tripped: subsequent concepts would hit it
                # too. Stop and surface it so the endpoint maps it to HTTP 429.
                raise
            except Exception:
                # One concept's engine failure must not abort the whole generation —
                # log it and move on to the next concept (Finding #4a).
                logger.exception(
                    "generate_reels: ingest_topic failed for concept=%s; skipping to next concept",
                    concept.get("id"),
                )
                continue
            accumulated_exclusions.extend(resolved_ids)
            videos_processed += len(resolved_ids)

            if dry_run:
                # Discover-only viability probe: one minimal preview per resolved
                # video id (zero DB writes). T6 finalizes can-generate parity.
                for vid in resolved_ids:
                    generated.append(
                        {
                            "reel_id": f"dry-run-{vid}",
                            "video_id": vid,
                            "material_id": material_id,
                            "concept_id": concept["id"],
                            "concept_title": concept["title"],
                            "score": 0.0,
                        }
                    )
            else:
                for reel_obj in reels:
                    generated.append(
                        self._reel_attribution_to_dict(reel_obj, concept, idx, total_concepts)
                    )
        if chain_emitter is not None:
            try:
                chain_emitter.flush_all()
            except Exception:
                logger.exception("chain_emitter.flush_all failed at end of generate_reels")
        return self._finalize_generated_reels(
            generated=generated,
            num_reels=num_reels,
            preferred_video_duration=safe_video_duration_pref,
        )

    def _normalize_video_pool_mode(self, value: str | None) -> str:
        if value in self.VALID_VIDEO_POOL_MODES:
            return str(value)
        return "short-first"

    def _normalize_preferred_video_duration(self, value: str | None) -> str:
        if value in self.VALID_VIDEO_DURATION_PREFS:
            return str(value)
        return "any"

    def _normalize_retrieval_profile(self, value: str | None) -> RetrievalProfile:
        if value == "deep":
            return "deep"
        return self.DEFAULT_RETRIEVAL_PROFILE

    def _normalize_target_clip_duration(self, value: int | float | None) -> int:
        if value is None:
            return self.DEFAULT_TARGET_CLIP_DURATION_SEC
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return self.DEFAULT_TARGET_CLIP_DURATION_SEC
        return max(self.MIN_TARGET_CLIP_DURATION_SEC, min(self.MAX_TARGET_CLIP_DURATION_SEC, parsed))

    def _duration_plan(self, video_pool_mode: str, preferred_video_duration: str) -> tuple[str | None, ...]:
        if preferred_video_duration == "short":
            return ("short", "medium", "long", None)
        if preferred_video_duration == "medium":
            return ("medium", "long", "short", None)
        if preferred_video_duration == "long":
            return ("long", "medium", "short", None)
        if video_pool_mode == "long-form":
            return ("long", "medium", "short", None)
        if video_pool_mode == "balanced":
            return ("short", "long", "medium", None)
        return ("short", "long", "medium", None)


    def _bootstrap_pool_is_weak(
        self,
        stage_candidates: list[dict[str, Any]],
        *,
        max_generation_target: int,
    ) -> bool:
        if not stage_candidates:
            return True

        min_kept = max(2, min(self.BOOTSTRAP_WEAK_POOL_MIN_KEPT, max_generation_target + 1))
        if len(stage_candidates) < min_kept:
            return True

        top_score = max(
            float((candidate.get("ranking") or {}).get("final_score") or 0.0)
            for candidate in stage_candidates
        )
        if top_score < self.BOOTSTRAP_WEAK_POOL_MIN_TOP_SCORE:
            return True

        unique_channels = {
            str((candidate.get("video") or {}).get("channel_title") or "").strip().lower()
            for candidate in stage_candidates
            if str((candidate.get("video") or {}).get("channel_title") or "").strip()
        }
        if len(unique_channels) < min(self.BOOTSTRAP_WEAK_POOL_MIN_UNIQUE_CHANNELS, len(stage_candidates)):
            return True

        unique_strategies = {
            str(query_candidate.strategy or "").strip().lower()
            for candidate in stage_candidates
            for query_candidate in [candidate.get("query_candidate")]
            if isinstance(query_candidate, QueryCandidate) and str(query_candidate.strategy or "").strip()
        }
        if len(unique_strategies) < min(self.BOOTSTRAP_WEAK_POOL_MIN_UNIQUE_STRATEGIES, len(stage_candidates)):
            return True

        return False





    def _video_duration_bucket(self, duration_sec: int | float | None) -> str | None:
        try:
            parsed = int(duration_sec or 0)
        except (TypeError, ValueError):
            return None
        if parsed <= 0:
            return None
        if parsed < 4 * 60:
            return "short"
        if parsed <= 20 * 60:
            return "medium"
        return "long"

    def _max_segments_allowed_for_video(
        self,
        *,
        video_duration_sec: int | float | None,
        fast_mode: bool,
    ) -> int:
        try:
            parsed = int(video_duration_sec or 0)
        except (TypeError, ValueError):
            parsed = 0
        if parsed <= 0:
            return 2 if fast_mode else 3
        if parsed < 4 * 60:
            return 1
        if parsed <= 20 * 60:
            return 2 if fast_mode else 3
        if parsed <= 45 * 60:
            return 4 if fast_mode else 6
        if parsed <= 90 * 60:
            return 6 if fast_mode else 9
        return 8 if fast_mode else 12

    def _topic_breadth_class(self, subject_tag: str | None, *, conn: Any = None) -> str:
        cleaned = self._clean_query_text(subject_tag or "")
        subject_key = self._normalize_query_key(cleaned)
        if not subject_key:
            return "default"
        if subject_key in {
            self._normalize_query_key(topic)
            for topic in (
                *getattr(self.topic_expansion_service, "STATIC_TOPIC_SUBTOPICS", {}).keys(),
                *self.BROAD_TOPIC_SUBTOPICS.keys(),
            )
        }:
            rules_class = "curated_broad"
        else:
            likely_language = False
            try:
                likely_language = bool(self.topic_expansion_service._looks_like_language_topic(cleaned))
            except Exception:
                likely_language = False
            try:
                opaque_topic = bool(
                    self.topic_expansion_service._is_opaque_single_token_topic(
                        cleaned,
                        canonical_topic=cleaned,
                        likely_language=likely_language,
                    )
                )
            except Exception:
                opaque_topic = False
            rules_class = "opaque_niche" if opaque_topic else "default"
        return self._merge_breadth_class_with_llm(
            cleaned,
            subject_key=subject_key,
            rules_class=rules_class,
            conn=conn,
        )

    def _merge_breadth_class_with_llm(
        self,
        cleaned: str,
        *,
        subject_key: str,
        rules_class: str,
        conn: Any,
    ) -> str:
        if conn is None or not cleaned:
            return rules_class
        try:
            raw = llm_router.chat_completion(
                conn=conn,
                cache_key=f"breadth_class:v1:{subject_key}",
                system=(
                    "Classify a search-topic's breadth for a video retrieval system. "
                    "Return JSON {\"class\": \"curated_broad\"|\"opaque_niche\"|\"default\", "
                    "\"confidence\": 0.0-1.0}. curated_broad = common broad topic (e.g. "
                    "'machine learning', 'cooking'); opaque_niche = obscure/single-term niche "
                    "topic (e.g. 'paleography', 'triboelectric'); default = neither."
                ),
                user=cleaned,
                temperature=0.0,
                json_mode=True,
            )
        except Exception:
            logger.exception("breadth_class LLM call failed for %s", subject_key)
            return rules_class
        if not raw:
            return rules_class
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            return rules_class
        llm_class = str(payload.get("class") or "").strip().lower()
        try:
            confidence = float(payload.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if llm_class not in {"curated_broad", "opaque_niche", "default"}:
            return rules_class
        if rules_class == "curated_broad" and llm_class != "curated_broad" and confidence >= 0.75:
            return "default"
        if rules_class == "default" and llm_class == "curated_broad" and confidence >= 0.7:
            return "curated_broad"
        if rules_class == "default" and llm_class == "opaque_niche" and confidence >= 0.7:
            return "opaque_niche"
        if rules_class == "opaque_niche" and llm_class != "opaque_niche" and confidence >= 0.75:
            return "default"
        return rules_class

    def _topic_novelty_profile(
        self,
        *,
        subject_tag: str | None,
        retrieval_profile: RetrievalProfile,
        fast_mode: bool,
        conn: Any = None,
    ) -> dict[str, float]:
        breadth_class = self._topic_breadth_class(subject_tag, conn=conn)
        technical_tokens = set(
            str(token or "").strip().lower()
            for token in (
                *getattr(self.topic_expansion_service, "QUANTITATIVE_TOKENS", ()),
                *getattr(self.topic_expansion_service, "SCIENCE_TOKENS", ()),
            )
        )
        subject_tokens = set(re.findall(r"[a-z0-9\+#]+", str(subject_tag or "").lower()))
        if breadth_class == "curated_broad":
            cross_video = 0.9
            same_video = 0.86
        elif breadth_class == "opaque_niche" or subject_tokens.intersection(technical_tokens):
            cross_video = 0.95
            same_video = 0.91
        else:
            cross_video = 0.92
            same_video = 0.88
        if retrieval_profile == "deep" and not fast_mode:
            cross_video = min(0.98, cross_video + 0.01)
            same_video = min(0.96, same_video + 0.01)
        return {
            "cross_video_similarity": float(cross_video),
            "same_video_similarity": float(same_video),
        }

    def _recovery_stage_allows_related(self, *, page_hint: int, recovery_stage: int) -> bool:
        return max(1, int(page_hint or 1)) >= 3 and int(recovery_stage or 0) >= self.REFILL_STAGE_ANCHORED_ADJACENT

    def _recovery_stage_allows_channel(self, *, page_hint: int, recovery_stage: int) -> bool:
        return max(1, int(page_hint or 1)) >= 3 and int(recovery_stage or 0) >= self.REFILL_STAGE_ANCHORED_ADJACENT

    def _recovery_stage_allows_adjacent(self, *, page_hint: int, recovery_stage: int) -> bool:
        return max(1, int(page_hint or 1)) >= 4 and int(recovery_stage or 0) >= self.REFILL_STAGE_RECOVERY_GRAPH

    def _recovery_stage_allows_source_surface(
        self,
        *,
        source_surface: str,
        page_hint: int,
        recovery_stage: int,
    ) -> bool:
        surface = str(source_surface or "").strip().lower()
        if surface in {"", "youtube_api", "youtube_html"}:
            return True
        if surface == "youtube_related":
            return self._recovery_stage_allows_related(page_hint=page_hint, recovery_stage=recovery_stage)
        if surface == "youtube_channel":
            return self._recovery_stage_allows_channel(page_hint=page_hint, recovery_stage=recovery_stage)
        if surface in {"local_cache", "duckduckgo_site", "bing_site", "duckduckgo_quoted", "bing_quoted"}:
            return self._recovery_stage_allows_adjacent(page_hint=page_hint, recovery_stage=recovery_stage)
        return True

    def _video_segment_cap(
        self,
        *,
        video_duration_sec: int | float | None,
        fast_mode: bool,
        default_cap: int,
        page_hint: int = 1,
    ) -> int:
        request_cap = self._request_page_segment_cap(
            video_duration_sec=video_duration_sec,
            fast_mode=fast_mode,
            page_hint=page_hint,
        )
        duration_cap = self._max_segments_allowed_for_video(
            video_duration_sec=video_duration_sec,
            fast_mode=fast_mode,
        )
        return max(1, min(duration_cap, max(max(1, int(default_cap or 1)), request_cap)))

    def _request_page_segment_cap(
        self,
        *,
        video_duration_sec: int | float | None,
        fast_mode: bool,
        page_hint: int,
    ) -> int:
        safe_page = max(1, int(page_hint or 1))
        try:
            duration_value = int(video_duration_sec or 0)
        except (TypeError, ValueError):
            duration_value = 0
        if safe_page <= 2:
            return 2 if safe_page == 1 else 3
        if safe_page == 3:
            return 5
        if duration_value > 20 * 60:
            return 10
        return 8

    def _allow_adjacent_recovery(self, *, page_hint: int, recovery_stage: int) -> bool:
        return self._recovery_stage_allows_adjacent(page_hint=page_hint, recovery_stage=recovery_stage)

    def _generation_result_score(self, reel: dict[str, Any]) -> float:
        relevance = reel.get("relevance_score")
        if isinstance(relevance, (int, float)):
            return float(relevance)
        score = reel.get("score")
        if isinstance(score, (int, float)):
            return float(score)
        return 0.0





    def _generation_target_cap(
        self,
        num_reels: int,
        preferred_video_duration: str,
        fast_mode: bool,
    ) -> int:
        if preferred_video_duration != "any" or num_reels <= 1:
            return num_reels
        extra = max(2, min(8, num_reels // 2 + (1 if fast_mode else 2)))
        return num_reels + extra





    def _reel_attribution_to_dict(
        self,
        reel_obj: Any,
        concept: dict[str, Any],
        idx: int,
        total_concepts: int,
    ) -> dict[str, Any]:
        """Convert a clip-engine ``ReelOutWithAttribution`` into the plain dict
        shape ``_create_reel`` returns, so ``_finalize_generated_reels`` /
        ``_group_by_video`` and the streaming emitter behave identically to the
        legacy path. ``video_id`` is derived from the embed ``video_url``
        because ``ReelOut`` carries no explicit video_id field."""
        video_url = str(getattr(reel_obj, "video_url", "") or "")
        video_id = _extract_embed_video_id(video_url) or ""
        captions = [
            cue.model_dump() if hasattr(cue, "model_dump") else cue
            for cue in (getattr(reel_obj, "captions", None) or [])
        ]
        t_start = float(getattr(reel_obj, "t_start", 0.0) or 0.0)
        t_end = float(getattr(reel_obj, "t_end", 0.0) or 0.0)
        return {
            "reel_id": reel_obj.reel_id,
            "material_id": reel_obj.material_id,
            "concept_id": concept["id"],
            "concept_title": concept["title"],
            "video_id": video_id,
            "video_title": getattr(reel_obj, "video_title", "") or "",
            "video_description": getattr(reel_obj, "video_description", "") or "",
            "channel_name": getattr(reel_obj, "channel_name", "") or "",
            "ai_summary": getattr(reel_obj, "ai_summary", "") or "",
            "video_url": video_url,
            "t_start": t_start,
            "t_end": t_end,
            "transcript_snippet": getattr(reel_obj, "transcript_snippet", "") or "",
            "takeaways": list(getattr(reel_obj, "takeaways", None) or []),
            "captions": captions,
            "score": float(getattr(reel_obj, "score", 0.0) or 0.0),
            "relevance_score": getattr(reel_obj, "relevance_score", None),
            "discovery_score": float(getattr(reel_obj, "discovery_score", None) or 0.0),
            "clipability_score": float(getattr(reel_obj, "clipability_score", None) or 0.0),
            "query_strategy": getattr(reel_obj, "query_strategy", "") or "",
            "retrieval_stage": getattr(reel_obj, "retrieval_stage", "") or "",
            "source_surface": getattr(reel_obj, "source_surface", "") or "",
            "matched_terms": list(getattr(reel_obj, "matched_terms", None) or []),
            "relevance_reason": getattr(reel_obj, "relevance_reason", "") or "",
            "video_duration_sec": getattr(reel_obj, "video_duration_sec", None),
            "clip_duration_sec": float(
                getattr(reel_obj, "clip_duration_sec", None) or max(0.0, t_end - t_start)
            ),
        }

    def _finalize_generated_reels(
        self,
        generated: list[dict[str, Any]],
        num_reels: int,
        preferred_video_duration: str,
    ) -> list[dict[str, Any]]:
        if not generated or num_reels <= 0:
            return []
        if preferred_video_duration != "any" or num_reels <= 1:
            return self._group_by_video(generated[:num_reels])

        ranked = sorted(generated, key=self._generation_result_score, reverse=True)
        short_candidates = [reel for reel in ranked if self._video_duration_bucket(reel.get("video_duration_sec")) == "short"]
        long_candidates = [reel for reel in ranked if self._video_duration_bucket(reel.get("video_duration_sec")) == "long"]

        selected: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for candidate in [short_candidates[0] if short_candidates else None, long_candidates[0] if long_candidates else None]:
            if not candidate:
                continue
            reel_id = str(candidate.get("reel_id") or "")
            if reel_id and reel_id in seen_ids:
                continue
            selected.append(candidate)
            if reel_id:
                seen_ids.add(reel_id)

        for candidate in ranked:
            if len(selected) >= num_reels:
                break
            reel_id = str(candidate.get("reel_id") or "")
            if reel_id and reel_id in seen_ids:
                continue
            selected.append(candidate)
            if reel_id:
                seen_ids.add(reel_id)

        selected = selected[:num_reels]
        return self._group_by_video(selected)

    def _group_by_video(self, reels: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Reorder a flat list so all clips from the same video appear
        contiguously. Video order follows the best clip in each group
        (highest generation score first); within a video the clips play
        in chronological order (ascending t_start). Preserves the input
        list's membership — only the order changes."""
        if len(reels) <= 1:
            return list(reels)
        groups: dict[str, list[dict[str, Any]]] = {}
        order: list[str] = []
        for reel in reels:
            video_id = str(reel.get("video_id") or "").strip()
            if video_id not in groups:
                groups[video_id] = []
                order.append(video_id)
            groups[video_id].append(reel)

        def _group_best_score(video_id: str) -> float:
            return max(
                (self._generation_result_score(r) for r in groups[video_id]),
                default=0.0,
            )

        order.sort(key=_group_best_score, reverse=True)
        result: list[dict[str, Any]] = []
        for video_id in order:
            group = groups[video_id]
            group.sort(key=lambda r: float(r.get("t_start") or 0.0))
            result.extend(group)
        return result

    def _parse_keywords_json(self, value: Any) -> list[str]:
        raw = str(value or "[]")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = []
        if not isinstance(parsed, list):
            return []
        clean: list[str] = []
        for item in parsed:
            token = str(item or "").strip()
            if not token:
                continue
            clean.append(token)
            if len(clean) >= 12:
                break
        return clean

    def _clip_length_bounds(self, target_clip_duration_sec: int) -> tuple[int, int]:
        safe_target = self._normalize_target_clip_duration(target_clip_duration_sec)
        min_len = max(self.MIN_TARGET_CLIP_DURATION_SEC, int(round(safe_target * 0.35)))
        max_len = max(min_len + self.MIN_TARGET_CLIP_DURATION_RANGE_GAP_SEC, safe_target)
        return min_len, max_len

    def _resolve_clip_duration_bounds(
        self,
        target_clip_duration_sec: int,
        target_clip_duration_min_sec: int | None,
        target_clip_duration_max_sec: int | None,
    ) -> tuple[int, int, int]:
        safe_target = self._normalize_target_clip_duration(target_clip_duration_sec)
        default_min, default_max = self._clip_length_bounds(safe_target)

        if target_clip_duration_min_sec is None and target_clip_duration_max_sec is None:
            return default_min, default_max, safe_target

        safe_min = default_min if target_clip_duration_min_sec is None else self._normalize_target_clip_duration(
            target_clip_duration_min_sec
        )
        safe_max = default_max if target_clip_duration_max_sec is None else self._normalize_target_clip_duration(
            target_clip_duration_max_sec
        )
        if safe_min > safe_max:
            safe_min, safe_max = safe_max, safe_min
        if safe_max - safe_min < self.MIN_TARGET_CLIP_DURATION_RANGE_GAP_SEC:
            expanded_max = min(self.MAX_TARGET_CLIP_DURATION_SEC, safe_min + self.MIN_TARGET_CLIP_DURATION_RANGE_GAP_SEC)
            if expanded_max - safe_min >= self.MIN_TARGET_CLIP_DURATION_RANGE_GAP_SEC:
                safe_max = expanded_max
            else:
                safe_min = max(self.MIN_TARGET_CLIP_DURATION_SEC, safe_max - self.MIN_TARGET_CLIP_DURATION_RANGE_GAP_SEC)
        safe_target = max(safe_min, min(safe_max, safe_target))
        return safe_min, safe_max, safe_target

    def _parse_embedding_vector(self, raw_value: object) -> np.ndarray | None:
        if not raw_value:
            return None
        try:
            vector = np.array(json.loads(str(raw_value)), dtype=np.float32)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if vector.ndim != 1 or vector.size != self.embedding_service.dim:
            return None
        return vector






    def _clean_query_text(self, value: str) -> str:
        return " ".join(str(value or "").split()).strip()

    def _normalize_query_key(self, value: str) -> str:
        cleaned = self._clean_query_text(value).lower()
        tokens = re.findall(r"[a-z0-9\+#]+", cleaned)
        return " ".join(tokens)

    def _strategy_from_suffix(self, suffix: str) -> str:
        mapping = {
            "animation": "animation",
            "demo": "demo",
            "documentary": "documentary",
            "explained": "explained",
            "lecture": "lecture",
            "tutorial": "tutorial",
            "worked example": "worked_example",
        }
        return mapping.get(self._clean_query_text(suffix).lower(), "explained")

    def _source_family_from_strategy(self, strategy: str, query_text: str = "") -> str:
        normalized_strategy = self._clean_query_text(strategy).lower().replace(" ", "_")
        normalized_query = self._clean_query_text(query_text).lower()
        if normalized_strategy in {"lecture", "conference_talk"} or "lecture" in normalized_query:
            return "lecture"
        if normalized_strategy == "documentary" or "documentary" in normalized_query:
            return "documentary"
        if normalized_strategy in {"tutorial", "worked_example", "demo"}:
            return "tutorial"
        if normalized_strategy in {"explained", "animation", "broadened_parent"}:
            return "explainer"
        if "conference" in normalized_query or "symposium" in normalized_query:
            return "conference"
        if "podcast" in normalized_query or "episode" in normalized_query:
            return "podcast"
        if "course" in normalized_query or "lesson" in normalized_query:
            return "course"
        if "interview" in normalized_query:
            return "interview"
        if "field" in normalized_query or "footage" in normalized_query:
            return "field_footage"
        return "other"

    def _frontier_family_for_query(self, *, stage: str, strategy: str, source_surface: str) -> str:
        normalized_stage = str(stage or "").strip().lower()
        normalized_strategy = self._clean_query_text(strategy).lower().replace(" ", "_")
        normalized_surface = str(source_surface or "").strip().lower()
        if normalized_stage == "high_precision" and normalized_strategy == "literal":
            return self.FRONTIER_ROOT_EXACT
        if normalized_stage == "recovery" and normalized_surface in {
            "youtube_related",
            "youtube_channel",
            "local_cache",
            "duckduckgo_site",
            "duckduckgo_quoted",
            "bing_site",
            "bing_quoted",
        }:
            return self.FRONTIER_RECOVERY_GRAPH
        if normalized_stage == "recovery":
            return self.FRONTIER_ANCHORED_ADJACENT
        return self.FRONTIER_ROOT_COMPANION

    def _frontier_anchor_mode(self, family: str) -> str:
        if family == self.FRONTIER_ROOT_EXACT:
            return "root_exact"
        if family == self.FRONTIER_ROOT_COMPANION:
            return "root_companion"
        if family == self.FRONTIER_ANCHORED_ADJACENT:
            return "anchored_adjacent"
        return "recovery_graph"

    def _frontier_family_key(self, family: str, query_text: str) -> str:
        normalized_query = self._normalize_query_key(query_text)
        return f"{family}:{normalized_query}" if normalized_query else family

    def _allowed_frontier_families(self, *, page_hint: int, recovery_stage: int) -> set[str]:
        allowed = {self.FRONTIER_ROOT_EXACT}
        if max(1, int(page_hint or 1)) <= 2 or int(recovery_stage or 0) >= self.REFILL_STAGE_ROOT_COMPANION:
            allowed.add(self.FRONTIER_ROOT_COMPANION)
        if max(1, int(page_hint or 1)) >= 3 and int(recovery_stage or 0) >= self.REFILL_STAGE_ANCHORED_ADJACENT:
            allowed.add(self.FRONTIER_ANCHORED_ADJACENT)
        if max(1, int(page_hint or 1)) >= 4 and int(recovery_stage or 0) >= self.REFILL_STAGE_RECOVERY_GRAPH:
            allowed.add(self.FRONTIER_RECOVERY_GRAPH)
        return allowed

    def _request_frontier_entry_id(self, *, material_id: str, request_key: str, family_key: str) -> str:
        payload = f"{material_id}|{request_key}|{family_key}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _load_request_frontier_entries(
        self,
        conn,
        *,
        material_id: str,
        request_key: str,
    ) -> dict[str, dict[str, Any]]:
        if not material_id or not request_key:
            return {}
        rows = fetch_all(
            conn,
            """
            SELECT *
            FROM request_frontier_entries
            WHERE material_id = ?
              AND request_key = ?
            """,
            (material_id, request_key),
        )
        entries: dict[str, dict[str, Any]] = {}
        for row in rows:
            family_key = str(row.get("family_key") or "").strip()
            if family_key:
                entries[family_key] = dict(row)
        return entries

    def _upsert_request_frontier_entry(
        self,
        conn,
        *,
        material_id: str,
        request_key: str,
        family_key: str,
        stage: str,
        query_text: str,
        source_family: str,
        anchor_mode: str,
        seed_video_id: str = "",
        seed_channel_id: str = "",
        stat_deltas: dict[str, float | int] | None = None,
        exhausted: bool | None = None,
    ) -> dict[str, Any]:
        existing = fetch_one(
            conn,
            """
            SELECT *
            FROM request_frontier_entries
            WHERE material_id = ?
              AND request_key = ?
              AND family_key = ?
            """,
            (material_id, request_key, family_key),
        )
        now = now_iso()
        row = {
            "id": str((existing or {}).get("id") or self._request_frontier_entry_id(
                material_id=material_id,
                request_key=request_key,
                family_key=family_key,
            )),
            "material_id": material_id,
            "request_key": request_key,
            "family_key": family_key,
            "stage": stage,
            "query_text": query_text,
            "source_family": source_family,
            "seed_video_id": seed_video_id,
            "seed_channel_id": seed_channel_id,
            "anchor_mode": anchor_mode,
            "runs": int((existing or {}).get("runs") or 0),
            "new_good_videos": int((existing or {}).get("new_good_videos") or 0),
            "new_accepted_reels": int((existing or {}).get("new_accepted_reels") or 0),
            "new_visible_reels": int((existing or {}).get("new_visible_reels") or 0),
            "duplicate_rate": float((existing or {}).get("duplicate_rate") or 0.0),
            "off_topic_rate": float((existing or {}).get("off_topic_rate") or 0.0),
            "last_run_at": (existing or {}).get("last_run_at"),
            "cooldown_until": (existing or {}).get("cooldown_until"),
            "exhausted": 1 if bool((existing or {}).get("exhausted")) else 0,
            "created_at": str((existing or {}).get("created_at") or now),
            "updated_at": now,
        }
        deltas = stat_deltas or {}
        if deltas:
            row["runs"] = int(row["runs"]) + int(deltas.get("runs") or 0)
            row["new_good_videos"] = int(row["new_good_videos"]) + int(deltas.get("new_good_videos") or 0)
            row["new_accepted_reels"] = int(row["new_accepted_reels"]) + int(deltas.get("new_accepted_reels") or 0)
            row["new_visible_reels"] = int(row["new_visible_reels"]) + int(deltas.get("new_visible_reels") or 0)
            if "duplicate_rate" in deltas:
                row["duplicate_rate"] = float(deltas.get("duplicate_rate") or 0.0)
            if "off_topic_rate" in deltas:
                row["off_topic_rate"] = float(deltas.get("off_topic_rate") or 0.0)
            if int(deltas.get("runs") or 0) > 0:
                row["last_run_at"] = now
        if exhausted is not None:
            row["exhausted"] = 1 if exhausted else 0
        if (
            int(row["runs"]) >= 3
            and int(row["new_visible_reels"]) <= 0
            and (
                float(row["duplicate_rate"]) >= 0.7
                or float(row["off_topic_rate"]) >= 0.4
            )
        ):
            row["exhausted"] = 1
        upsert(conn, "request_frontier_entries", row)
        return row

    def _apply_frontier_scheduler(
        self,
        conn,
        *,
        material_id: str,
        request_key: str | None,
        query_candidates: list[QueryCandidate],
        page_hint: int,
        recovery_stage: int,
    ) -> list[QueryCandidate]:
        if not request_key:
            return query_candidates
        frontier_entries = self._load_request_frontier_entries(
            conn,
            material_id=material_id,
            request_key=request_key,
        )
        allowed_families = self._allowed_frontier_families(
            page_hint=page_hint,
            recovery_stage=recovery_stage,
        )
        family_priority = {
            self.FRONTIER_ROOT_EXACT: 0,
            self.FRONTIER_ROOT_COMPANION: 1,
            self.FRONTIER_ANCHORED_ADJACENT: 2,
            self.FRONTIER_RECOVERY_GRAPH: 3,
        }

        scheduled: list[QueryCandidate] = []
        for candidate in query_candidates:
            family_key = candidate.family_key or self._frontier_family_key(
                self._frontier_family_for_query(
                    stage=candidate.stage,
                    strategy=candidate.strategy,
                    source_surface=candidate.source_surface,
                ),
                candidate.text,
            )
            family = family_key.split(":", 1)[0]
            if family not in allowed_families:
                continue
            entry = frontier_entries.get(family_key) or {}
            if bool(int(entry.get("exhausted") or 0)):
                continue
            scheduled.append(
                QueryCandidate(
                    text=candidate.text,
                    strategy=candidate.strategy,
                    confidence=candidate.confidence,
                    source_terms=list(candidate.source_terms),
                    weight=candidate.weight,
                    stage=candidate.stage,
                    source_surface=candidate.source_surface,
                    family_key=family_key,
                    source_family=candidate.source_family or self._source_family_from_strategy(candidate.strategy, candidate.text),
                    anchor_mode=candidate.anchor_mode or self._frontier_anchor_mode(family),
                    seed_video_id=candidate.seed_video_id,
                    seed_channel_id=candidate.seed_channel_id,
                )
            )

        def sort_key(candidate: QueryCandidate) -> tuple[Any, ...]:
            family_key = candidate.family_key
            family = family_key.split(":", 1)[0] if family_key else self.FRONTIER_ROOT_COMPANION
            entry = frontier_entries.get(family_key) or {}
            runs = int(entry.get("runs") or 0)
            visible_yield = float(entry.get("new_visible_reels") or 0.0) / max(1, runs)
            good_video_yield = float(entry.get("new_good_videos") or 0.0) / max(1, runs)
            duplicate_rate = float(entry.get("duplicate_rate") or 0.0)
            off_topic_rate = float(entry.get("off_topic_rate") or 0.0)
            return (
                family_priority.get(family, 9),
                0 if runs <= 0 else 1,
                -visible_yield,
                -good_video_yield,
                duplicate_rate,
                off_topic_rate,
                -float(candidate.confidence * candidate.weight),
            )

        scheduled.sort(key=sort_key)
        return scheduled


    def _build_planned_query(
        self,
        *,
        text: str,
        strategy: str,
        stage: str,
        confidence: float,
        source_terms: list[str],
        concept_title: str,
        weight: float = 1.0,
        source_surface: str = "youtube_html",
        disambiguator: str | None = None,
        rationale: str = "",
        family: str | None = None,
        source_family: str | None = None,
        anchor_mode: str | None = None,
        seed_video_id: str = "",
        seed_channel_id: str = "",
    ) -> PlannedQuery:
        cleaned = self._clean_query_text(text)
        normalization_key = self._normalize_query_key(cleaned)
        concept_key = self._normalize_query_key(concept_title)
        cluster_key = f"{stage}:{strategy}:{concept_key}"
        if stage == "recovery":
            cluster_key = f"recovery:{concept_key}:{normalization_key}"
        elif stage == "broad":
            cluster_key = f"broad:{strategy}:{concept_key}:{normalization_key}"
        resolved_family = family or self._frontier_family_for_query(
            stage=stage,
            strategy=strategy,
            source_surface=source_surface,
        )
        return PlannedQuery(
            text=cleaned,
            strategy=self._clean_query_text(strategy).lower().replace(" ", "_"),
            stage=stage,
            confidence=max(0.0, min(1.0, confidence)),
            source_terms=tuple(self._clean_query_text(term) for term in source_terms if self._clean_query_text(term))[:8],
            weight=max(0.05, float(weight)),
            source_surface=source_surface or "youtube_html",
            disambiguator=self._clean_query_text(disambiguator or "") or None,
            normalization_key=normalization_key,
            cluster_key=cluster_key,
            rationale=self._clean_query_text(rationale),
            family_key=self._frontier_family_key(resolved_family, cleaned),
            source_family=source_family or self._source_family_from_strategy(strategy, cleaned),
            anchor_mode=anchor_mode or self._frontier_anchor_mode(resolved_family),
            seed_video_id=str(seed_video_id or "").strip(),
            seed_channel_id=str(seed_channel_id or "").strip(),
        )

    def _select_concepts(
        self,
        concepts: list[dict[str, Any]],
        *,
        retrieval_profile: RetrievalProfile,
        request_need: int,
        fast_mode: bool,
        targeted_concept_id: str | None,
        subject_tag: str | None = None,
        conn: Any = None,
    ) -> tuple[tuple[dict[str, Any], ...], tuple[ConceptSelectionDecision, ...], bool]:
        if not concepts:
            return (), (), False

        if targeted_concept_id:
            concept_limit = 1
            budget_reason = "targeted concept request"
        elif self.serverless_mode:
            concept_limit = min(2, len(concepts))
            budget_reason = "serverless concept budget"
        elif retrieval_profile == "bootstrap":
            bootstrap_cap = self.BOOTSTRAP_CONCEPT_LIMIT
            if self._topic_breadth_class(subject_tag, conn=conn) == "curated_broad":
                bootstrap_cap = max(bootstrap_cap, 6)
            concept_limit = max(1, min(bootstrap_cap, request_need + (1 if request_need <= 2 else 0)))
            budget_reason = f"bootstrap concept budget {concept_limit} for request need {request_need}"
        else:
            deep_cap = 4 if fast_mode else 8
            concept_limit = max(2, min(deep_cap, request_need + (1 if fast_mode else 3)))
            budget_reason = f"deep concept budget {concept_limit} for request need {request_need}"

        selected: list[dict[str, Any]] = []
        decisions: list[ConceptSelectionDecision] = []
        for rank, concept in enumerate(concepts, start=1):
            concept_id = str(concept.get("id") or "")
            concept_title = self._clean_query_text(str(concept.get("title") or ""))
            is_selected = rank <= concept_limit
            reason = (
                f"ordered concept rank {rank}; selected within {budget_reason}"
                if is_selected
                else f"ordered concept rank {rank}; skipped because {budget_reason} was exhausted"
            )
            decisions.append(
                ConceptSelectionDecision(
                    concept_id=concept_id,
                    concept_text=concept_title,
                    concept_rank=rank,
                    selected=is_selected,
                    reason=reason,
                )
            )
            if is_selected:
                selected.append(concept)

        return tuple(selected), tuple(decisions), len(concepts) > concept_limit

    def _extract_programming_language_hint(self, terms: list[str]) -> str | None:
        for raw in terms:
            cleaned = self._clean_query_text(raw)
            lowered = cleaned.lower()
            if lowered in self.PROGRAMMING_LANGUAGE_HINTS:
                return cleaned
            for token in re.findall(r"[A-Za-z0-9\+#]+", lowered):
                if token in self.PROGRAMMING_LANGUAGE_HINTS:
                    return token.upper() if token == "sql" else token
        return None

    def _choose_disambiguator(
        self,
        *,
        title: str,
        keywords: list[str],
        context_terms: list[str],
        subject_tag: str | None,
    ) -> str | None:
        title_key = self._normalize_query_key(title)
        title_tokens = normalize_terms([title])
        language_hint = self._extract_programming_language_hint([*keywords, *context_terms, str(subject_tag or "")])
        subject = self._clean_query_text(subject_tag or "")
        subject_key = self._normalize_query_key(subject)
        opaque_root_topic = bool(
            subject_key
            and subject_key == title_key
            and self.topic_expansion_service._is_opaque_single_token_topic(
                subject,
                canonical_topic=subject,
                likely_language=self.topic_expansion_service._looks_like_language_topic(subject),
            )
        )

        # Exact multi-word subject roots should keep their literal phrase intact.
        # Over-disambiguating these produces malformed queries like
        # "python Python Programming" or "d day World War Ii".
        if subject_key and subject_key == title_key:
            if len(title_tokens) >= 2:
                return None
            if (
                not opaque_root_topic
                and language_hint is None
                and not title_tokens.intersection(self.AMBIGUOUS_CONCEPT_TOKENS)
            ):
                return None

        candidates: list[tuple[float, str]] = []
        if language_hint:
            candidates.append((3.6, language_hint))
        if subject and self._normalize_query_key(subject) != title_key:
            candidates.append((3.1, subject))
        for term in [*keywords[:4], *context_terms[:4]]:
            cleaned = self._clean_query_text(term)
            if not cleaned:
                continue
            normalized = self._normalize_query_key(cleaned)
            if not normalized or normalized == title_key:
                continue
            if len(title_tokens) > 1 and set(normalized.split()).issubset(title_tokens):
                continue
            score = 2.2 if " " in cleaned else 1.7
            if normalized in self.PROGRAMMING_LANGUAGE_HINTS:
                score += 1.0
            if len(title_tokens) == 1:
                if title_key in normalized:
                    score -= 0.7
                elif self.topic_expansion_service._allows_unanchored_opaque_search_term(cleaned):
                    score += 0.6
            candidates.append((score, cleaned))

        if not candidates:
            return None

        needs_disambiguation = (
            len(title_tokens) <= 3
            or bool(title_tokens.intersection(self.AMBIGUOUS_CONCEPT_TOKENS))
            or language_hint is not None
        )
        if not needs_disambiguation:
            return None

        candidates.sort(key=lambda item: (-item[0], len(item[1]), item[1].lower()))
        return candidates[0][1]

    def _classify_concept_intent(
        self,
        *,
        title: str,
        keywords: list[str],
        summary: str,
        subject_tag: str | None,
        context_terms: list[str],
        video_pool_mode: str,
        fast_mode: bool = True,
    ) -> ConceptIntentPlan:
        # Fix H: Use LLM for intent classification in slow mode
        if not fast_mode and not self.serverless_mode:
            llm_plan = self._classify_intent_via_llm(
                title=title, keywords=keywords, summary=summary,
                subject_tag=subject_tag, video_pool_mode=video_pool_mode,
            )
            if llm_plan is not None:
                return llm_plan

        tokens = normalize_terms([title, summary, subject_tag or "", *keywords, *context_terms])
        has_problem_solving = bool(tokens.intersection(self.PROBLEM_SOLVING_TOKENS))
        has_programming = bool(tokens.intersection(self.PROGRAMMING_TOKENS))
        has_process_visual = bool(tokens.intersection(self.PROCESS_VISUAL_TOKENS))
        has_history = bool(tokens.intersection(self.HISTORY_HUMANITIES_TOKENS))
        has_math_physics = bool(tokens.intersection(self.MATH_PHYSICS_TOKENS))

        if has_programming:
            family = "programming"
            suffixes = ("tutorial", "demo", "explained") if video_pool_mode != "long-form" else ("tutorial", "lecture", "demo")
            rationale = "programming concepts retrieve best with tutorial-led phrasing"
        elif has_problem_solving or has_math_physics:
            family = "problem_solving"
            suffixes = (
                ("worked example", "tutorial", "lecture")
                if video_pool_mode != "long-form"
                else ("tutorial", "worked example", "lecture")
            )
            rationale = "math and problem-solving concepts favor worked examples or tutorial walkthroughs"
        elif has_process_visual:
            family = "process_visual"
            suffixes = (
                ("animation", "explained", "tutorial")
                if video_pool_mode != "long-form"
                else ("explained", "animation", "lecture")
            )
            rationale = "process-heavy science concepts benefit from animation or explanation phrasing"
        elif has_history:
            family = "history_humanities"
            suffixes = (
                ("explained", "documentary", "lecture")
                if video_pool_mode != "long-form"
                else ("documentary", "lecture", "explained")
            )
            rationale = "history and humanities concepts need explanatory or documentary framing"
        else:
            family = "long_form_general" if video_pool_mode == "long-form" else "general"
            suffixes = self.INTENT_SUFFIX_PRIORITY[family]
            rationale = "default to a single explanatory query when domain cues are limited"

        suffix = suffixes[0]
        return ConceptIntentPlan(
            strategy=self._strategy_from_suffix(suffix),
            suffix=suffix,
            rationale=f"{family}: {rationale}",
        )

    def _classify_intent_via_llm(
        self,
        *,
        title: str,
        keywords: list[str],
        summary: str,
        subject_tag: str | None,
        video_pool_mode: str,
    ) -> ConceptIntentPlan | None:
        """Fix H: LLM-powered intent classification for slow mode."""
        cache_key = f"llm_intent:{title}|{subject_tag or ''}|{video_pool_mode}"
        with self._strategy_history_cache_lock:
            cached = self._strategy_history_cache.get(cache_key)
        if cached is not None:
            return cached if cached != "_none_" else None
        if not self.llm_available:
            return None
        try:
            valid_types = ["tutorial", "explained", "lecture", "worked example", "animation", "documentary", "demo"]
            prompt = (
                f"Given the study concept '{title}' (keywords: {', '.join(keywords[:5])}, "
                f"subject: {subject_tag or 'general'}, mode: {video_pool_mode}), "
                f"what type of YouTube video would best teach this? "
                f"Choose exactly one: {', '.join(valid_types)}. "
                f"Reply with just the type name."
            )
            answer = (llm_router.chat_completion(
                system="You pick the best YouTube content format for a study concept.",
                user=prompt,
                temperature=0.2,
                max_tokens=20,
            ) or "").strip().lower()
            for vt in valid_types:
                if vt in answer:
                    result = ConceptIntentPlan(
                        strategy=self._strategy_from_suffix(vt),
                        suffix=vt,
                        rationale=f"llm_intent: LLM selected '{vt}' for concept '{title}'",
                    )
                    with self._strategy_history_cache_lock:
                        self._evict_strategy_cache_if_full()
                        self._strategy_history_cache[cache_key] = result
                    return result
        except Exception:
            pass
        with self._strategy_history_cache_lock:
            self._evict_strategy_cache_if_full()
            self._strategy_history_cache[cache_key] = "_none_"
        return None

    def _evict_strategy_cache_if_full(self) -> None:
        """Evict oldest half of cache entries when max size is reached. Must be called under lock."""
        if len(self._strategy_history_cache) >= self._strategy_history_cache_max_size:
            keys = list(self._strategy_history_cache.keys())
            for key in keys[: len(keys) // 2]:
                del self._strategy_history_cache[key]

    def _build_literal_query(
        self,
        *,
        title: str,
        keywords: list[str],
        disambiguator: str | None,
        subject_tag: str | None = None,
    ) -> str:
        clean_title = self._clean_query_text(title)
        if not clean_title:
            return ""
        parts: list[str] = []
        clean_disambiguator = self._clean_query_text(disambiguator or "")
        clean_title_tokens = set(self._normalize_query_key(clean_title).split())
        clean_disambiguator_tokens = set(self._normalize_query_key(clean_disambiguator).split())
        if (
            clean_disambiguator
            and self._normalize_query_key(clean_disambiguator) != self._normalize_query_key(clean_title)
            and not (
                len(clean_title_tokens) > 1
                and clean_disambiguator_tokens
                and clean_disambiguator_tokens.issubset(clean_title_tokens)
            )
        ):
            parts.append(clean_disambiguator)
        parts.append(clean_title)
        exact_subject_root = (
            self._normalize_query_key(clean_title)
            and self._normalize_query_key(clean_title) == self._normalize_query_key(subject_tag or "")
        )
        if not clean_disambiguator and len(normalize_terms([clean_title])) <= 1 and not exact_subject_root:
            for term in keywords[:2]:
                cleaned = self._clean_query_text(term)
                if not cleaned:
                    continue
                if self._normalize_query_key(cleaned) == self._normalize_query_key(clean_title):
                    continue
                parts.append(cleaned)
                break
        return self._clean_query_text(" ".join(parts))

    # Process-wide correction cache: can-generate probes and refinements re-run
    # generate_reels for the same subject many times; correct each subject once.
    _SUBJECT_CORRECTION_CACHE: dict[str, str] = {}

    def _corrected_subject_tag(self, subject_tag: str) -> str:
        """Spellcheck a topic-material subject via the clip engine's expansion
        pass (Gemini flash-lite when keyed; the keyless fallback returns it
        unchanged, so this never breaks offline paths).

        Only conservative SPELLING fixes are accepted ('pychology' →
        'psychology'); field-qualifying rewrites ('jaguar' → 'jaguar animal
        biology') are rejected — discover()'s expansion applies those per-query
        without renaming the user's topic."""
        cache_key = subject_tag.strip().lower()
        cached = self._SUBJECT_CORRECTION_CACHE.get(cache_key)
        if cached is not None:
            return cached
        try:
            corrected = str(
                _clip_engine_expand.expand_query(subject_tag, 1).get("corrected") or ""
            ).strip()
        except Exception:
            # Transient failure: fall back WITHOUT caching, so a later
            # generation can still pick up the real correction.
            logger.exception("subject correction failed; using raw subject %r", subject_tag)
            return subject_tag
        result = subject_tag
        if corrected and corrected.lower() != cache_key:
            same_word_count = len(corrected.split()) == len(subject_tag.split())
            similarity = difflib.SequenceMatcher(
                None, cache_key, corrected.lower()
            ).ratio()
            if same_word_count and similarity >= 0.75:
                result = corrected
        if len(self._SUBJECT_CORRECTION_CACHE) < 512:
            self._SUBJECT_CORRECTION_CACHE[cache_key] = result
        return result

    def _concept_topic_query(self, concept_row: dict) -> str:
        """Return a search-engine topic string for a concept row: its clean title.

        No wiki-keyword append — the clip engine's `discover()` expansion
        (spellcheck + field inference + educational phrasings) owns
        disambiguation now, and appended `keywords_json` terms polluted the
        seed query with unrelated wiki noise.
        """
        title = str(concept_row.get("title") or "")
        return self._clean_query_text(title)

    def _build_intent_query(
        self,
        *,
        literal_query: str,
        intent_plan: ConceptIntentPlan,
    ) -> str:
        base = self._clean_query_text(literal_query)
        suffix = self._clean_query_text(intent_plan.suffix)
        if not base:
            return ""
        if suffix and self._normalize_query_key(base).endswith(self._normalize_query_key(suffix)):
            return base
        return self._clean_query_text(f"{base} {suffix}")

    def _dedupe_queries(
        self,
        queries: list[PlannedQuery],
        *,
        limit: int | None = None,
    ) -> list[PlannedQuery]:
        ordered = sorted(
            queries,
            key=lambda item: (item.stage != "high_precision", -(item.confidence * item.weight), len(item.text)),
        )
        seen_normalized: set[str] = set()
        seen_clusters: set[str] = set()
        deduped: list[PlannedQuery] = []
        for item in ordered:
            if not item.text:
                continue
            normalization_key = item.normalization_key or self._normalize_query_key(item.text)
            cluster_key = item.cluster_key or normalization_key
            if normalization_key in seen_normalized or cluster_key in seen_clusters:
                continue
            seen_normalized.add(normalization_key)
            seen_clusters.add(cluster_key)
            deduped.append(item)
            if limit and len(deduped) >= limit:
                break
        return deduped

    def _deep_query_expansion_limit(self, *, fast_mode: bool, request_need: int) -> int:
        safe_need = max(1, int(request_need))
        if self.serverless_mode:
            return min(3, 1 + safe_need // 6)
        if fast_mode:
            return min(4, 2 + safe_need // 6)
        # Fix U: Slow mode gets more expansions for deeper coverage
        return min(8, 4 + safe_need // 3)

    def _extract_summary_focus_terms(self, summary: str, *, max_terms: int = 8) -> list[str]:
        cleaned = self._clean_query_text(summary)
        if not cleaned:
            return []

        tokens = re.findall(r"[A-Za-z0-9\+#]+", cleaned)
        if len(tokens) < 2:
            return []

        phrases: list[str] = []
        seen: set[str] = set()
        for size in (3, 2):
            for start in range(0, len(tokens) - size + 1):
                window = tokens[start : start + size]
                lowered = [token.lower() for token in window]
                if lowered[0] in self.QUERY_EXPANSION_STOPWORDS or lowered[-1] in self.QUERY_EXPANSION_STOPWORDS:
                    continue
                if any(token in self.GENERIC_CONTEXT_TERMS or token in self.QUERY_EXPANSION_STOPWORDS for token in lowered):
                    continue
                if sum(1 for token in lowered if len(token) >= 4 or token in self.PROGRAMMING_LANGUAGE_HINTS) < size:
                    continue
                phrase = self._clean_query_text(" ".join(window))
                normalized = self._normalize_query_key(phrase)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                phrases.append(phrase)
                if len(phrases) >= max_terms:
                    return phrases
        return phrases

    def _expand_synonyms_via_llm(self, terms: list[str]) -> list[str] | None:
        """Fix K: Use LLM to generate richer query synonyms in slow mode."""
        if not self.llm_available:
            return None
        try:
            cleaned_terms = [t.strip() for t in terms if t.strip()][:5]
            if not cleaned_terms:
                return None
            prompt = (
                f"For these study terms: {', '.join(cleaned_terms)}\n"
                f"Generate 4-6 alternative search phrases that would find educational YouTube videos "
                f"teaching the same concepts. Return one phrase per line, no numbering."
            )
            content = (llm_router.chat_completion(
                system="You generate alternative educational search phrases.",
                user=prompt,
                temperature=0.4,
                max_tokens=200,
            ) or "").strip()
            synonyms = [line.strip() for line in content.split("\n") if line.strip() and len(line.strip()) > 2]
            return synonyms[:8] if synonyms else None
        except Exception:
            return None

    def _broaden_concept_queries_via_llm(
        self,
        *,
        concept_title: str,
        keywords: list[str],
        summary: str,
        subject_tag: str | None,
    ) -> list[str] | None:
        """Generate broader parent topics for niche/specific concepts that lack hardcoded expansions."""
        if not self.llm_available:
            return None
        try:
            subject_hint = f" within {subject_tag}" if subject_tag else ""
            prompt = (
                f"Study concept: '{concept_title}'{subject_hint}\n"
                f"Keywords: {', '.join(keywords[:5]) or 'none'}\n\n"
                f"This concept is specific. List 5-7 broader educational topics that "
                f"a YouTube video might cover which would include '{concept_title}' "
                f"as a section, subtopic, or worked example.\n"
                f"Focus on real video titles that exist on YouTube.\n"
                f"Also include 2-3 closely related or prerequisite concepts.\n"
                f"One topic per line, no numbering or explanation."
            )
            content = llm_router.chat_completion(
                system="You expand study concepts into real YouTube topics.",
                user=prompt,
                temperature=0.3,
                max_tokens=250,
            )
            if not content:
                return None
            terms = [line.strip() for line in content.split("\n") if line.strip() and len(line.strip()) > 2]
            return terms[:8] if terms else None
        except Exception:
            return None

    def _expand_controlled_synonyms(self, terms: list[str], fast_mode: bool = True) -> list[str]:
        # Fix K: Use LLM for synonym expansion in slow mode
        if not fast_mode and not self.serverless_mode:
            llm_synonyms = self._expand_synonyms_via_llm(terms)
            if llm_synonyms:
                return llm_synonyms

        expansions: list[str] = []
        seen: set[str] = set()
        for raw in terms:
            value = self._clean_query_text(str(raw or ""))
            normalized_value = self._normalize_query_key(value)
            if not normalized_value:
                continue

            candidates: list[str] = []
            for key, mapped in self.CONTROLLED_QUERY_SYNONYMS.items():
                normalized_key = self._normalize_query_key(key)
                if normalized_key == normalized_value or normalized_key in normalized_value or normalized_value in normalized_key:
                    candidates.extend(mapped)

            if len(normalized_value.split()) == 1 and len(normalized_value) >= 5:
                if normalized_value.endswith("s"):
                    candidates.append(normalized_value[:-1])
                else:
                    candidates.append(f"{normalized_value}s")

            for candidate in candidates:
                cleaned = self._clean_query_text(candidate)
                normalized = self._normalize_query_key(cleaned)
                if not cleaned or not normalized or normalized == normalized_value or normalized in seen:
                    continue
                seen.add(normalized)
                expansions.append(cleaned)
        return expansions[:32]

    def _broad_topic_subtopic_terms(
        self,
        *,
        concept_title: str,
        keywords: list[str],
        summary: str,
        request_need: int,
    ) -> list[str]:
        normalized_title = self._normalize_query_key(concept_title)
        if not normalized_title:
            return []
        normalized_keywords = {
            self._normalize_query_key(keyword)
            for keyword in keywords
            if self._normalize_query_key(keyword)
        }
        normalized_summary = self._normalize_query_key(summary)

        candidates: tuple[str, ...] = ()
        for raw_topic, topic_terms in self.BROAD_TOPIC_SUBTOPICS.items():
            normalized_topic = self._normalize_query_key(raw_topic)
            if not normalized_topic:
                continue
            if (
                normalized_title == normalized_topic
                or normalized_topic in normalized_title
                or normalized_topic in normalized_summary
                or any(normalized_topic == keyword or normalized_topic in keyword for keyword in normalized_keywords)
            ):
                candidates = topic_terms
                break

        if not candidates:
            return []

        deduped: list[str] = []
        seen: set[str] = set()
        cap = 4 if request_need <= 4 else 6 if request_need <= 8 else 8
        for term in candidates:
            cleaned = self._clean_query_text(term)
            normalized = self._normalize_query_key(cleaned)
            if not cleaned or not normalized or normalized == normalized_title or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(cleaned)
            if len(deduped) >= cap:
                break
        return deduped

    def _related_query_terms(
        self,
        *,
        concept_title: str,
        keywords: list[str],
        summary: str,
        context_terms: list[str],
        fast_mode: bool,
        request_need: int,
        subject_tag: str | None = None,
    ) -> list[str]:
        title_key = self._normalize_query_key(concept_title)
        title_tokens = set(title_key.split())
        scored_terms: dict[str, tuple[float, str]] = {}

        def add_term(raw_value: str, score: float) -> None:
            cleaned = self._clean_query_text(raw_value)
            normalized = self._normalize_query_key(cleaned)
            if not cleaned or not normalized or normalized == title_key:
                return
            if normalized in self.GENERIC_CONTEXT_TERMS:
                return
            tokens = set(normalized.split())
            if not tokens or tokens.issubset(title_tokens):
                return
            if len(tokens) == 1 and len(next(iter(tokens))) < 4 and normalized not in self.PROGRAMMING_LANGUAGE_HINTS:
                return
            existing = scored_terms.get(normalized)
            if existing is None or score > existing[0]:
                scored_terms[normalized] = (score, cleaned)

        for index, term in enumerate(
            self._broad_topic_subtopic_terms(
                concept_title=concept_title,
                keywords=keywords,
                summary=summary,
                request_need=request_need,
            )
        ):
            add_term(term, 4.9 - index * 0.18)

        for index, term in enumerate(keywords[:8]):
            cleaned = self._clean_query_text(term)
            if not cleaned:
                continue
            base_score = 4.4 - index * 0.22
            if " " in cleaned:
                base_score += 0.35
            add_term(cleaned, base_score)

        summary_focus_terms = self._extract_summary_focus_terms(summary)
        for index, term in enumerate(summary_focus_terms[:8]):
            add_term(term, 3.4 - index * 0.18)

        for index, term in enumerate(self._expand_controlled_synonyms([concept_title, *keywords], fast_mode=fast_mode)[:10]):
            alias_score = 3.15 - index * 0.16
            if len(self._normalize_query_key(term).split()) == 1 and len(term) <= 5:
                alias_score += 0.3
            add_term(term, alias_score)

        if request_need >= 5:
            for index, term in enumerate(context_terms[:4]):
                add_term(term, 2.2 - index * 0.12)

        if len(scored_terms) < 3 and not self.serverless_mode:
            broadened = self._broaden_concept_queries_via_llm(
                concept_title=concept_title,
                keywords=keywords,
                summary=summary,
                subject_tag=subject_tag,
            )
            if broadened:
                for index, term in enumerate(broadened[:6]):
                    add_term(term, 3.8 - index * 0.2)

        ranked = sorted(scored_terms.values(), key=lambda item: (-item[0], len(item[1]), item[1].lower()))
        limit = self._deep_query_expansion_limit(fast_mode=fast_mode, request_need=request_need)
        return [term for _score, term in ranked[: max(1, limit)]]

    def _bootstrap_standalone_alias_term(self, *, concept_title: str, keywords: list[str]) -> str | None:
        title_tokens = set(self._normalize_query_key(concept_title).split())
        if not title_tokens:
            return None
        for raw_term in keywords[:8]:
            cleaned = self._clean_query_text(raw_term)
            normalized = self._normalize_query_key(cleaned)
            tokens = set(normalized.split())
            if not cleaned or not normalized or not tokens:
                continue
            if tokens.intersection(title_tokens):
                continue
            if len(tokens) > 2:
                continue
            if normalized in self.GENERIC_CONTEXT_TERMS:
                continue
            if len(normalized) < 4:
                continue
            return cleaned
        return None

    def _maybe_expand_queries(
        self,
        *,
        concept_title: str,
        keywords: list[str],
        summary: str,
        context_terms: list[str],
        literal_query: str,
        intent_plan: ConceptIntentPlan,
        retrieval_profile: RetrievalProfile,
        fast_mode: bool,
        subject_tag: str | None,
        disambiguator: str | None,
        request_need: int,
        allow_bootstrap_subtopic_expansion: bool,
    ) -> tuple[PlannedQuery, ...]:
        if retrieval_profile == "bootstrap":
            bootstrap_expansions: list[PlannedQuery] = []
            broad_topic_seed = self._normalize_query_key(concept_title) in {
                self._normalize_query_key(topic) for topic in self.BROAD_TOPIC_SUBTOPICS
            }
            prefer_subtopic_bootstrap = self._is_vague_concept(
                title=concept_title,
                keywords=keywords,
                summary=summary,
            ) or broad_topic_seed
            if prefer_subtopic_bootstrap and allow_bootstrap_subtopic_expansion:
                related_terms = self._related_query_terms(
                    concept_title=concept_title,
                    keywords=keywords,
                    summary=summary,
                    context_terms=context_terms,
                    fast_mode=fast_mode,
                    request_need=max(3, request_need),
                    subject_tag=subject_tag,
                )
                for index, related_term in enumerate(related_terms[:1]):
                    bootstrap_expansions.append(
                        self._build_planned_query(
                            text=f"{literal_query} {related_term}",
                            strategy="literal",
                            stage="broad",
                            confidence=max(0.8, 0.88 - index * 0.04),
                            source_terms=[concept_title, related_term],
                            concept_title=concept_title,
                            weight=max(0.78, 0.9 - index * 0.05),
                            rationale="bootstrap uses one anchored subtopic or synonym expansion before deeper refinement",
                        )
                    )
            elif subject_tag:
                standalone_alias = self._bootstrap_standalone_alias_term(
                    concept_title=concept_title,
                    keywords=keywords,
                )
                if standalone_alias:
                    bootstrap_expansions.append(
                        self._build_planned_query(
                            text=f"{standalone_alias} {intent_plan.suffix}",
                            strategy=intent_plan.strategy,
                            stage="high_precision",
                            confidence=0.84,
                            source_terms=[concept_title, standalone_alias, intent_plan.suffix],
                            concept_title=concept_title,
                            weight=0.84,
                            rationale="bootstrap uses one standalone alias query when the topic name itself is too niche",
                        )
                    )

            alternate_suffixes = ("explained", "tutorial")
            if intent_plan.strategy == "tutorial":
                alternate_suffixes = ("explained",)
            elif intent_plan.strategy == "explained":
                alternate_suffixes = ("tutorial",)
            for suffix in alternate_suffixes[:1]:
                strategy = self._strategy_from_suffix(suffix)
                if strategy == intent_plan.strategy:
                    continue
                bootstrap_expansions.append(
                    self._build_planned_query(
                        text=f"{literal_query} {suffix}",
                        strategy=strategy,
                        stage="broad",
                        confidence=0.76,
                        source_terms=[concept_title, suffix],
                        concept_title=concept_title,
                        weight=0.78,
                        rationale="bootstrap allows one alternate pedagogical angle for broader coverage",
                    )
                )

            # Niche concept broadening: if the concept is NOT vague and NOT a broad topic,
            # use LLM to find a broader parent topic and add it as an hp-stage query
            # (broad_budget=0 in bootstrap, so only hp-stage queries execute).
            if not prefer_subtopic_bootstrap and not self.serverless_mode:
                broadened = self._broaden_concept_queries_via_llm(
                    concept_title=concept_title,
                    keywords=keywords,
                    summary=summary,
                    subject_tag=subject_tag,
                )
                if broadened:
                    best_parent = broadened[0]
                    bootstrap_expansions.append(
                        self._build_planned_query(
                            text=best_parent,
                            strategy="broadened_parent",
                            stage="high_precision",
                            confidence=0.82,
                            source_terms=[concept_title, best_parent],
                            concept_title=concept_title,
                            weight=0.80,
                            rationale="niche concept broadened via LLM to a parent topic likely to have YouTube coverage",
                        )
                    )

            return tuple(self._dedupe_queries(bootstrap_expansions, limit=2))

        expansions: list[PlannedQuery] = []
        literal_key = self._normalize_query_key(literal_query)
        literal_tokens = set(literal_key.split())
        related_terms = self._related_query_terms(
            concept_title=concept_title,
            keywords=keywords,
            summary=summary,
            context_terms=context_terms,
            fast_mode=fast_mode,
            request_need=request_need,
            subject_tag=subject_tag,
        )
        for index, related_term in enumerate(related_terms):
            if set(self._normalize_query_key(related_term).split()).issubset(literal_tokens):
                continue
            expansions.append(
                self._build_planned_query(
                    text=f"{literal_query} {related_term}",
                    strategy="literal",
                    stage="broad",
                    confidence=max(0.72, 0.9 - index * 0.04),
                    source_terms=[concept_title, related_term],
                    concept_title=concept_title,
                    weight=max(0.66, 0.92 - index * 0.05),
                    rationale="deep profile broadens with anchored keyword and subtopic terms from concept metadata",
                )
            )

        alias_prefix_parts: list[str] = []
        for part in [disambiguator or "", subject_tag or ""]:
            cleaned = self._clean_query_text(part)
            if not cleaned or cleaned in alias_prefix_parts:
                continue
            alias_prefix_parts.append(cleaned)
        alias_prefix = self._clean_query_text(" ".join(alias_prefix_parts))
        alias_terms = self._expand_controlled_synonyms([concept_title, *keywords], fast_mode=fast_mode)
        for index, alias_term in enumerate(alias_terms[:2]):
            alias_key = self._normalize_query_key(alias_term)
            if not alias_key or set(alias_key.split()).issubset(literal_tokens):
                continue
            alias_base = self._clean_query_text(" ".join(part for part in [alias_prefix, alias_term] if part))
            if not alias_base:
                continue
            expansions.append(
                self._build_planned_query(
                    text=f"{alias_base} {intent_plan.suffix}",
                    strategy=intent_plan.strategy,
                    stage="broad",
                    confidence=max(0.7, 0.82 - index * 0.05),
                    source_terms=[concept_title, alias_term, intent_plan.suffix],
                    concept_title=concept_title,
                    weight=max(0.62, 0.8 - index * 0.06),
                    disambiguator=disambiguator,
                    rationale="deep profile uses controlled aliases only after exact concept queries are exhausted",
                )
            )

        alternate_suffixes = self.INTENT_SUFFIX_PRIORITY.get("general", ())
        if intent_plan.strategy == "tutorial":
            alternate_suffixes = ("explained", "lecture")
        elif intent_plan.strategy == "worked_example":
            alternate_suffixes = ("tutorial", "explained")
        elif intent_plan.strategy == "animation":
            alternate_suffixes = ("explained", "tutorial")
        elif intent_plan.strategy == "documentary":
            alternate_suffixes = ("explained", "lecture")
        elif intent_plan.strategy == "lecture":
            alternate_suffixes = ("tutorial", "explained")

        for suffix in alternate_suffixes:
            strategy = self._strategy_from_suffix(suffix)
            if strategy == intent_plan.strategy:
                continue
            expansions.append(
                self._build_planned_query(
                    text=f"{literal_query} {suffix}",
                    strategy=strategy,
                    stage="broad",
                    confidence=0.8 if not fast_mode else 0.76,
                    source_terms=[concept_title, suffix],
                    concept_title=concept_title,
                    weight=0.82,
                    rationale="deep profile allows one alternate pedagogical angle after the primary intent query",
                )
            )
            if len(expansions) >= self._deep_query_expansion_limit(fast_mode=fast_mode, request_need=request_need):
                break

        capped = self._deep_query_expansion_limit(fast_mode=fast_mode, request_need=request_need)
        return tuple(self._dedupe_queries(expansions, limit=capped))

    def _plan_recovery_queries(
        self,
        *,
        concept_title: str,
        keywords: list[str],
        summary: str,
        subject_tag: str | None,
        context_terms: list[str],
        retrieval_profile: RetrievalProfile,
    ) -> list[PlannedQuery]:
        disambiguator = self._choose_disambiguator(
            title=concept_title,
            keywords=keywords,
            context_terms=context_terms,
            subject_tag=subject_tag,
        )
        recovery_terms = self._decompose_concept_for_recovery(concept_title, summary, keywords)
        queries: list[PlannedQuery] = []
        limit = self.BOOTSTRAP_RECOVERY_QUERY_COUNT if retrieval_profile == "bootstrap" else 2
        for recovery_term in recovery_terms:
            cleaned_term = self._clean_query_text(recovery_term)
            if not cleaned_term:
                continue
            base = self._clean_query_text(" ".join(part for part in [disambiguator or "", cleaned_term] if part))
            if not base:
                continue
            if self._normalize_query_key(base) == self._normalize_query_key(concept_title):
                base = self._clean_query_text(f"{base} explained")
            queries.append(
                self._build_planned_query(
                    text=base,
                    strategy="recovery_adjacent",
                    stage="recovery",
                    confidence=0.64,
                    source_terms=[concept_title, cleaned_term],
                    concept_title=concept_title,
                    weight=0.68,
                    disambiguator=disambiguator,
                    rationale="recovery broadens carefully with one adjacent term only after a weak primary pool",
                )
            )
            if len(queries) >= limit:
                break
        return self._dedupe_queries(queries, limit=limit)


    def _plan_query_set_for_concepts(
        self,
        *,
        concepts: list[dict[str, Any]],
        subject_tag: str | None,
        material_context_terms: list[str],
        retrieval_profile: RetrievalProfile,
        fast_mode: bool,
        video_pool_mode: str,
        preferred_video_duration: str,
        request_need: int,
        targeted_concept_id: str | None,
        allow_bootstrap_subtopic_expansion: bool = True,
        conn: Any = None,
    ) -> QueryPlanningResult:
        # preferred_video_duration is now used below for pool-mode query hints.
        selected_concepts, selection_decisions, exhausted = self._select_concepts(
            concepts,
            retrieval_profile=retrieval_profile,
            request_need=max(1, request_need),
            fast_mode=fast_mode,
            targeted_concept_id=targeted_concept_id,
            subject_tag=subject_tag,
            conn=conn,
        )
        selected_ids = {str(concept.get("id") or "") for concept in selected_concepts}
        selected_plans: list[ConceptQueryPlan] = []
        global_queries: list[PlannedQuery] = []

        for decision in selection_decisions:
            if not decision.selected:
                continue
            concept = next((item for item in selected_concepts if str(item.get("id") or "") == decision.concept_id), None)
            if concept is None:
                continue
            concept_title = self._clean_query_text(str(concept.get("title") or ""))
            concept_keywords = self._parse_keywords_json(concept.get("keywords_json"))
            concept_summary = str(concept.get("summary") or "")
            concept_terms = [concept_title, *concept_keywords, concept_summary]
            context_terms = self._context_terms_for_concept(concept_terms, material_context_terms)
            disambiguator = self._choose_disambiguator(
                title=concept_title,
                keywords=concept_keywords,
                context_terms=context_terms,
                subject_tag=subject_tag,
            )
            intent_plan = self._classify_concept_intent(
                title=concept_title,
                keywords=concept_keywords,
                summary=concept_summary,
                subject_tag=subject_tag,
                context_terms=context_terms,
                video_pool_mode=video_pool_mode,
                fast_mode=fast_mode,
            )
            literal_query_text = self._build_literal_query(
                title=concept_title,
                keywords=concept_keywords,
                disambiguator=disambiguator,
                subject_tag=subject_tag,
            )
            literal_query = self._build_planned_query(
                text=literal_query_text,
                strategy="literal",
                stage="high_precision",
                confidence=0.96,
                source_terms=[concept_title, disambiguator or ""],
                concept_title=concept_title,
                weight=1.0,
                disambiguator=disambiguator,
                rationale="literal query anchors the exact concept with at most one high-value disambiguator",
            )
            intent_query_text = self._build_intent_query(
                literal_query=literal_query.text,
                intent_plan=intent_plan,
            )
            # Fix S: videoPoolMode influences query phrasing
            if video_pool_mode == "long-form" and "lecture" not in intent_query_text.lower():
                intent_query_text = self._clean_query_text(f"{intent_query_text} full lecture")
            elif video_pool_mode == "short-first" and preferred_video_duration == "short":
                intent_query_text = self._clean_query_text(f"{intent_query_text} shorts")
            intent_query = self._build_planned_query(
                text=intent_query_text,
                strategy=intent_plan.strategy,
                stage="high_precision",
                confidence=0.9,
                source_terms=[concept_title, intent_plan.suffix, disambiguator or ""],
                concept_title=concept_title,
                weight=0.96,
                disambiguator=disambiguator,
                rationale=intent_plan.rationale,
            )
            expansion_queries = self._maybe_expand_queries(
                concept_title=concept_title,
                keywords=concept_keywords,
                summary=concept_summary,
                context_terms=context_terms,
                literal_query=literal_query.text,
                intent_plan=intent_plan,
                retrieval_profile=retrieval_profile,
                fast_mode=fast_mode,
                subject_tag=subject_tag,
                disambiguator=disambiguator,
                request_need=request_need,
                allow_bootstrap_subtopic_expansion=allow_bootstrap_subtopic_expansion,
            )
            recovery_queries = self._plan_recovery_queries(
                concept_title=concept_title,
                keywords=concept_keywords,
                summary=concept_summary,
                subject_tag=subject_tag,
                context_terms=context_terms,
                retrieval_profile=retrieval_profile,
            )

            first_pass_queries = self._dedupe_queries(
                [literal_query, intent_query, *expansion_queries],
                limit=self.BOOTSTRAP_PRIMARY_QUERY_COUNT if retrieval_profile == "bootstrap" else None,
            )
            literal_query = first_pass_queries[0]
            intent_query = first_pass_queries[1] if len(first_pass_queries) > 1 else intent_query
            expansion_queries = tuple(query for query in first_pass_queries[2:])

            concept_plan = ConceptQueryPlan(
                concept_id=decision.concept_id,
                concept_text=concept_title,
                concept_rank=decision.concept_rank,
                reason_selected=decision.reason,
                literal_query=literal_query,
                intent_query=intent_query,
                selected_intent_strategy=intent_plan.strategy,
                disambiguator=disambiguator,
                normalization_key=self._normalize_query_key(concept_title),
                recovery_queries=tuple(recovery_queries),
                expansion_queries=expansion_queries,
            )
            selected_plans.append(concept_plan)
            global_queries.extend(first_pass_queries)

        skipped = tuple(decision for decision in selection_decisions if not decision.selected and decision.concept_id not in selected_ids)
        deduped_global = tuple(self._dedupe_queries(global_queries))
        return QueryPlanningResult(
            selected_concepts=tuple(selected_plans),
            skipped_concepts=skipped,
            global_queries=deduped_global,
            total_selected_concepts=len(selected_plans),
            total_first_pass_queries=len(deduped_global),
            total_recovery_queries_allowed=sum(len(plan.recovery_queries) for plan in selected_plans),
            query_budget_exhausted=exhausted,
        )

    def _build_query_candidates(
        self,
        title: str,
        keywords: list[str],
        summary: str,
        subject_tag: str | None,
        context_terms: list[str],
        visual_spec: dict[str, list[str]],
        fast_mode: bool,
        retrieval_profile: RetrievalProfile,
    ) -> list[QueryCandidate]:
        del visual_spec  # Query planning is now concept- and settings-driven rather than visual-template-driven.
        clean_title = self._clean_query_text(title)
        if not clean_title:
            return []

        concept = {
            "id": "single-concept",
            "title": clean_title,
            "keywords_json": dumps_json([self._clean_query_text(item) for item in keywords if self._clean_query_text(item)]),
            "summary": summary,
        }
        plan = self._plan_query_set_for_concepts(
            concepts=[concept],
            subject_tag=subject_tag,
            material_context_terms=[self._clean_query_text(item) for item in context_terms if self._clean_query_text(item)],
            retrieval_profile=retrieval_profile,
            fast_mode=fast_mode,
            video_pool_mode="short-first",
            preferred_video_duration="any",
            request_need=1,
            targeted_concept_id="single-concept",
        )
        if not plan.selected_concepts:
            return []
        concept_plan = plan.selected_concepts[0]
        queries = [concept_plan.literal_query, concept_plan.intent_query, *concept_plan.expansion_queries]
        return [
            QueryCandidate(
                text=query.text,
                strategy=query.strategy,
                confidence=query.confidence,
                source_terms=list(query.source_terms),
                weight=query.weight,
                stage=query.stage,
                source_surface=query.source_surface,
                family_key=query.family_key,
                source_family=query.source_family,
                anchor_mode=query.anchor_mode,
                seed_video_id=query.seed_video_id,
                seed_channel_id=query.seed_channel_id,
            )
            for query in queries
        ]

    def _build_bootstrap_primary_query(
        self,
        *,
        title: str,
        keywords: list[str],
        context_terms: list[str],
        subject_tag: str | None,
    ) -> str:
        return self._build_literal_query(
            title=title,
            keywords=keywords or context_terms,
            disambiguator=self._choose_disambiguator(
                title=title,
                keywords=keywords,
                context_terms=context_terms,
                subject_tag=subject_tag,
            ),
            subject_tag=subject_tag,
        )

    def _build_retrieval_stage_plan(
        self,
        query_candidates: list[QueryCandidate],
        fast_mode: bool,
        retrieval_profile: RetrievalProfile,
        request_need: int,
        subject_tag: str | None = None,
        page_hint: int = 1,
        recovery_stage: int = 0,
        conn: Any = None,
    ) -> list[RetrievalStagePlan]:
        stage_map: dict[str, list[QueryCandidate]] = {"high_precision": [], "broad": [], "recovery": []}
        allowed_families = self._allowed_frontier_families(
            page_hint=page_hint,
            recovery_stage=recovery_stage,
        )
        for item in query_candidates:
            family = item.family_key.split(":", 1)[0] if item.family_key else self._frontier_family_for_query(
                stage=item.stage,
                strategy=item.strategy,
                source_surface=item.source_surface,
            )
            if retrieval_profile == "bootstrap" and family != self.FRONTIER_ROOT_EXACT:
                continue
            if retrieval_profile != "bootstrap" and family not in allowed_families:
                continue
            stage = item.stage if item.stage in stage_map else "broad"
            stage_map[stage].append(item)

        if retrieval_profile == "bootstrap":
            high_precision_budget = max(1, min(len(stage_map["high_precision"]), 2 if fast_mode else 3))
            high_precision_min = min(2, high_precision_budget)
            broad_budget = 0
            broad_max = 0
            broad_min = 0
            recovery_budget = 0
            recovery_min = 0
        elif self.serverless_mode:
            high_precision_budget = 1 if int(recovery_stage or 0) <= self.REFILL_STAGE_EXACT_ROOT else 2
            high_precision_min = 2
            if int(recovery_stage or 0) < self.REFILL_STAGE_ROOT_COMPANION:
                broad_budget = 0
                broad_max = 0
            elif int(recovery_stage or 0) < self.REFILL_STAGE_ANCHORED_ADJACENT:
                broad_budget = min(len(stage_map["broad"]), 1)
                broad_max = broad_budget
            else:
                broad_budget = min(len(stage_map["broad"]), 2)
                broad_max = min(len(stage_map["broad"]), 3)
            broad_min = 1 if broad_budget > 0 else 0
            recovery_budget = (
                min(len(stage_map["recovery"]), 1 if int(recovery_stage or 0) >= self.REFILL_STAGE_RECOVERY_GRAPH else 0)
                if self._recovery_stage_allows_related(page_hint=page_hint, recovery_stage=recovery_stage)
                else 0
            )
            recovery_min = 0
        else:
            if int(recovery_stage or 0) <= self.REFILL_STAGE_ROOT_COMPANION:
                high_precision_budget = 1 if fast_mode else 2
                high_precision_min = 1 if fast_mode else 2
            else:
                high_precision_budget = 2 if fast_mode else 3
                high_precision_min = 2 if fast_mode else 3
            breadth_class = self._topic_breadth_class(subject_tag, conn=conn)
            if breadth_class == "opaque_niche":
                slow_base_broad_budget = 2
                slow_max_broad_budget = 4
            elif breadth_class == "curated_broad":
                slow_base_broad_budget = 6
                slow_max_broad_budget = 8
            else:
                slow_base_broad_budget = 4
                slow_max_broad_budget = 6
            if fast_mode:
                base_broad_budget = max(1, (slow_base_broad_budget + 1) // 2)
                max_broad_budget = max(base_broad_budget, (slow_max_broad_budget + 1) // 2)
            else:
                base_broad_budget = slow_base_broad_budget
                max_broad_budget = slow_max_broad_budget
            if int(recovery_stage or 0) < self.REFILL_STAGE_ROOT_COMPANION:
                broad_budget = 0
                broad_max = 0
            elif int(recovery_stage or 0) < self.REFILL_STAGE_ANCHORED_ADJACENT:
                broad_budget = min(len(stage_map["broad"]), max(1, base_broad_budget // 2))
                broad_max = min(len(stage_map["broad"]), max(1, max_broad_budget // 2))
            else:
                broad_budget = min(len(stage_map["broad"]), base_broad_budget)
                broad_max = min(len(stage_map["broad"]), max_broad_budget)
            if fast_mode:
                broad_min = 1 if broad_budget > 0 else 0
                recovery_budget = (
                    min(len(stage_map["recovery"]), 1 if int(recovery_stage or 0) >= self.REFILL_STAGE_ANCHORED_ADJACENT else 0)
                    if self._recovery_stage_allows_related(page_hint=page_hint, recovery_stage=recovery_stage)
                    else 0
                )
                recovery_min = 1 if recovery_budget > 0 else 0
            else:
                broad_min = min(2, broad_budget)
                recovery_budget = (
                    min(
                        len(stage_map["recovery"]),
                        1 if int(recovery_stage or 0) < self.REFILL_STAGE_RECOVERY_GRAPH else 2 + max(0, int(request_need) - 6) // 6,
                    )
                    if self._recovery_stage_allows_related(page_hint=page_hint, recovery_stage=recovery_stage)
                    else 0
                )
                recovery_min = min(1, recovery_budget)

        plans = [
            RetrievalStagePlan(
                name="high_precision",
                queries=sorted(stage_map["high_precision"], key=lambda q: -(q.confidence * q.weight)),
                budget=high_precision_budget,
                min_good_results=high_precision_min,
            ),
            RetrievalStagePlan(
                name="broad",
                queries=sorted(stage_map["broad"], key=lambda q: -(q.confidence * q.weight)),
                budget=broad_budget,
                min_good_results=broad_min,
                max_budget=broad_max,
            ),
            RetrievalStagePlan(
                name="recovery",
                queries=sorted(stage_map["recovery"], key=lambda q: -(q.confidence * q.weight)),
                budget=recovery_budget,
                min_good_results=recovery_min,
            ),
        ]
        return [plan for plan in plans if plan.queries and plan.budget > 0]


    def _graph_profile_for_stage(
        self,
        *,
        fast_mode: bool,
        retrieval_profile: RetrievalProfile,
        page_hint: int,
        recovery_stage: int,
        subject_tag: str | None = None,
        conn: Any = None,
    ) -> Literal["off", "light", "deep"]:
        if retrieval_profile == "bootstrap":
            if subject_tag and self._topic_breadth_class(subject_tag, conn=conn) == "opaque_niche":
                return "light"
            return "off"
        if not self._recovery_stage_allows_related(page_hint=page_hint, recovery_stage=recovery_stage):
            return "off"
        if int(recovery_stage or 0) < self.REFILL_STAGE_RECOVERY_GRAPH:
            return "light"
        if self.serverless_mode:
            return "light"
        return "light" if fast_mode else "deep"



    def _get_transcript(self, video: Any) -> list[dict[str, Any]]:
        """Route transcript fetch by provider. Returns a list of
        `{"start": float, "duration": float, "text": str}` cues in
        YouTube-compatible format (empty list on any failure).

        Accepts:
          * a dict video_row (reads `provider` + `video_id`/`id`)
          * a `(video_id, provider)` tuple
          * a bare video_id string (provider inferred from
            `_provider_by_video_id`, else 'youtube')
        """
        provider_name = "youtube"
        video_id = ""
        if isinstance(video, dict):
            video_id = str(video.get("video_id") or video.get("id") or "").strip()
            provider_name = str(video.get("provider") or "youtube").strip().lower() or "youtube"
        elif isinstance(video, tuple) and len(video) == 2:
            raw_id, raw_provider = video
            video_id = str(raw_id or "").strip()
            provider_name = str(raw_provider or "youtube").strip().lower() or "youtube"
        else:
            video_id = str(video or "").strip()
            provider_name = self._provider_by_video_id.get(video_id, "youtube")
        if not video_id:
            return []
        if provider_name == "youtube":
            try:
                return list(self.youtube_service.get_transcript(None, video_id) or [])
            except Exception:
                return []
        try:
            transcript = self._provider_registry.fetch_transcript(provider_name, video_id)
        except Exception:
            return []
        if transcript is None:
            return []
        return [
            {
                "start": float(cue.start),
                "duration": max(0.0, float(cue.end) - float(cue.start)),
                "text": str(cue.text or ""),
            }
            for cue in transcript.cues
        ]


    def _transcript_prefetch_worker_count(self, *, fast_mode: bool, requested_count: int) -> int:
        workers_limit = self.TRANSCRIPT_FETCH_WORKERS_FAST if fast_mode else self.TRANSCRIPT_FETCH_WORKERS_SLOW
        if self.serverless_mode:
            workers_limit = min(workers_limit, 2)
        return max(1, min(workers_limit, max(1, int(requested_count))))

    def _launch_transcript_prefetch_task(
        self,
        video_ids: list[str],
        *,
        fast_mode: bool,
        cached_transcripts: dict[str, list[dict[str, Any]]] | None = None,
    ) -> TranscriptPrefetchTask | None:
        ordered_ids = [
            str(video_id).strip()
            for video_id in dict.fromkeys(video_ids)
            if str(video_id).strip()
        ]
        if not ordered_ids:
            return None
        executor = ThreadPoolExecutor(
            max_workers=self._transcript_prefetch_worker_count(
                fast_mode=fast_mode,
                requested_count=len(ordered_ids),
            )
        )
        future_by_video_id = {
            video_id: executor.submit(self._get_transcript, video_id)
            for video_id in ordered_ids
        }
        return TranscriptPrefetchTask(
            video_ids=tuple(ordered_ids),
            executor=executor,
            future_by_video_id=future_by_video_id,
            cached_transcripts=dict(cached_transcripts or {}),
        )

    def _submit_transcript_prefetch_ids(
        self,
        *,
        prefetch_task: TranscriptPrefetchTask,
        video_ids: list[str],
        fast_mode: bool,
    ) -> TranscriptPrefetchTask:
        ordered_ids = [
            str(video_id).strip()
            for video_id in dict.fromkeys(video_ids)
            if str(video_id).strip()
        ]
        if not ordered_ids:
            return prefetch_task
        pending_ids = [
            video_id
            for video_id in ordered_ids
            if video_id not in prefetch_task.cached_transcripts and video_id not in prefetch_task.future_by_video_id
        ]
        if not pending_ids:
            return prefetch_task
        if prefetch_task.executor is None:
            prefetch_task.executor = ThreadPoolExecutor(
                max_workers=self._transcript_prefetch_worker_count(
                    fast_mode=fast_mode,
                    requested_count=len(pending_ids),
                )
            )
        for video_id in pending_ids:
            prefetch_task.future_by_video_id[video_id] = prefetch_task.executor.submit(
                self._get_transcript,
                video_id,
            )
        prefetch_task.video_ids = tuple(
            dict.fromkeys([*prefetch_task.video_ids, *pending_ids])
        )
        return prefetch_task

    def _shutdown_transcript_prefetch_task(
        self,
        prefetch_task: TranscriptPrefetchTask | None,
        *,
        wait: bool,
        cancel_futures: bool = False,
    ) -> None:
        if prefetch_task is None:
            return
        executor = prefetch_task.executor
        prefetch_task.executor = None
        if cancel_futures:
            for future in list(prefetch_task.future_by_video_id.values()):
                if future is not None and not future.done():
                    future.cancel()
        if executor is not None:
            try:
                executor.shutdown(wait=wait, cancel_futures=cancel_futures)
            except TypeError:
                executor.shutdown(wait=wait)
        prefetch_task.future_by_video_id = {}
        prefetch_task.video_ids = ()

    def _resolve_transcript_prefetch(
        self,
        *,
        prefetch_task: TranscriptPrefetchTask,
        video_id: str,
        timeout: float | None,
        fast_mode: bool,
    ) -> dict[str, list[dict[str, Any]]]:
        clean_video_id = str(video_id).strip()
        if not clean_video_id:
            return {}
        if clean_video_id in prefetch_task.cached_transcripts:
            return {
                clean_video_id: list(prefetch_task.cached_transcripts.get(clean_video_id) or []),
            }
        future = prefetch_task.future_by_video_id.get(clean_video_id)
        transcript: list[dict[str, Any]] = []
        try:
            if future is not None:
                transcript = list(future.result(timeout=timeout) or [])
            else:
                transcript = list(self._get_transcript(clean_video_id) or [])
        except FutureTimeoutError:
            logger.warning(
                "Transcript prefetch timed out after %.1fs for video=%s",
                float(timeout or 0.0),
                clean_video_id,
            )
            try:
                transcript = list(self._get_transcript(clean_video_id) or [])
            except Exception:
                transcript = []
        except Exception:
            logger.exception("Transcript prefetch failed for video=%s", clean_video_id)
        prefetch_task.cached_transcripts[clean_video_id] = transcript
        prefetch_task.future_by_video_id.pop(clean_video_id, None)
        prefetch_task.video_ids = tuple(
            video
            for video in prefetch_task.video_ids
            if video != clean_video_id and video not in prefetch_task.cached_transcripts
        )
        return {clean_video_id: transcript}

    def _seed_transcript_prefetch_ids(
        self,
        *,
        candidates: list[dict[str, Any]],
        transcript_budget: int,
        clip_min_len: int,
        clip_max_len: int,
    ) -> list[str]:
        if transcript_budget <= 0 or not candidates:
            return []
        ranked = sorted(
            candidates,
            key=lambda row: float((row.get("ranking") or {}).get("final_score") or 0.0),
            reverse=True,
        )
        selected_ids: list[str] = []
        seen_ids: set[str] = set()
        for candidate in ranked:
            video_id = str(candidate.get("video_id") or "").strip()
            if not video_id or video_id in seen_ids:
                continue
            video_duration_val = int(candidate.get("video_duration") or 0)
            use_full_short_clip = self._should_use_full_short_clip(
                prefer_short_query=self._video_duration_bucket(video_duration_val) == "short",
                video_duration_sec=video_duration_val,
                clip_min_len=clip_min_len,
                clip_max_len=clip_max_len,
            )
            if use_full_short_clip:
                continue
            seen_ids.add(video_id)
            selected_ids.append(video_id)
            if len(selected_ids) >= transcript_budget:
                break
        return selected_ids

    def _maybe_launch_transcript_prefetch(
        self,
        *,
        prefetch_task: TranscriptPrefetchTask | None,
        stage_candidates: list[dict[str, Any]],
        transcript_budget: int,
        clip_min_len: int,
        clip_max_len: int,
        fast_mode: bool,
    ) -> TranscriptPrefetchTask | None:
        if not stage_candidates or transcript_budget <= 0:
            return prefetch_task
        seed_ids = self._seed_transcript_prefetch_ids(
            candidates=stage_candidates,
            transcript_budget=transcript_budget,
            clip_min_len=clip_min_len,
            clip_max_len=clip_max_len,
        )
        min_seed_count = 1 if transcript_budget <= 1 else min(3, transcript_budget)
        if len(seed_ids) < min_seed_count:
            return prefetch_task
        if prefetch_task is None:
            return self._launch_transcript_prefetch_task(seed_ids, fast_mode=fast_mode)
        return self._submit_transcript_prefetch_ids(
            prefetch_task=prefetch_task,
            video_ids=seed_ids,
            fast_mode=fast_mode,
        )


    def _transcript_for_candidate(
        self,
        *,
        prefetch_task: TranscriptPrefetchTask | None,
        transcript_cache: dict[str, list[dict[str, Any]]],
        video_id: str,
        fast_mode: bool,
    ) -> list[dict[str, Any]]:
        clean_video_id = str(video_id).strip()
        if not clean_video_id:
            return []
        cached = transcript_cache.get(clean_video_id)
        if cached is not None:
            return list(cached)
        if prefetch_task is not None:
            transcript = list(
                self._resolve_transcript_prefetch(
                    prefetch_task=prefetch_task,
                    video_id=clean_video_id,
                    timeout=30.0,
                    fast_mode=fast_mode,
                ).get(clean_video_id)
                or []
            )
        else:
            try:
                transcript = list(self._get_transcript(clean_video_id) or [])
            except Exception:
                transcript = []
        transcript_cache[clean_video_id] = transcript
        return transcript




    def _dense_transcript_windows(
        self,
        conn,
        *,
        transcript: list[dict[str, Any]],
        concept_terms: list[str],
        root_topic_terms: list[str],
        target_clip_duration_sec: int,
        prior_contexts: list[dict[str, Any]],
        fast_mode: bool,
        max_windows: int,
    ) -> list[SegmentMatch]:
        entries = [entry for entry in transcript if str(entry.get("text") or "").strip()]
        if not entries:
            return []
        safe_target = max(20, int(target_clip_duration_sec or DEFAULT_TARGET_CLIP_DURATION_SEC))
        window_len = max(75, 2 * safe_target)
        stride = max(20, safe_target // 2)
        first_start = max(0, int(float(entries[0].get("start") or 0.0)))
        last_entry = entries[-1]
        total_end = int(float(last_entry.get("start") or 0.0) + float(last_entry.get("duration") or 0.0))
        if total_end <= first_start:
            return []

        anchor_terms = root_topic_terms or concept_terms[:4]
        cue_pattern = re.compile(
            r"\b(explains?|because|therefore|for example|for instance|means|refers to|"
            r"defined as|in practice|lets|let's|suppose|imagine|proof|derive|application)\b"
        )
        candidates: list[SegmentMatch] = []
        for window_start in range(first_start, max(first_start + 1, total_end), stride):
            window_end = min(total_end, window_start + window_len)
            if window_end - window_start < 35:
                continue
            window_entries = [
                entry
                for entry in entries
                if float(entry.get("start") or 0.0) < window_end
                and (float(entry.get("start") or 0.0) + float(entry.get("duration") or 0.0)) > window_start
            ]
            if not window_entries:
                continue
            text = " ".join(str(entry.get("text") or "").strip() for entry in window_entries).strip()
            if len(text) < 120:
                continue
            concept_support = lexical_overlap_score(text, concept_terms)
            root_support = lexical_overlap_score(text, anchor_terms)
            if root_support <= 0.0 and concept_support < 0.08:
                continue
            clip_duration = float(max(0, window_end - window_start))
            clip_context = self._build_clip_context(text=text, clip_duration_sec=clip_duration)
            clipability = self._clip_self_containment_score(text=text, clip_duration_sec=clip_duration)
            cue_hits = len(cue_pattern.findall(text.lower()))
            teaching_density = min(1.0, 0.18 + 0.12 * cue_hits + 0.01 * len(normalize_terms([text])))
            novelty_penalty = 0.0
            for prior in prior_contexts[:8]:
                prior_text = str(prior.get("text") or "").strip()
                if not prior_text:
                    continue
                lexical = lexical_overlap_score(text, [prior_text])
                similarity = lexical
                if lexical >= 0.32:
                    similarity = max(
                        similarity,
                        self._text_pair_similarity(
                            conn,
                            left_text=text,
                            right_text=prior_text,
                            fast_mode=fast_mode,
                        ),
                    )
                novelty_penalty = max(novelty_penalty, similarity)
            duration_fit = max(0.0, 1.0 - abs(clip_duration - safe_target) / max(30.0, float(safe_target * 2)))
            score = (
                0.28 * root_support
                + 0.26 * concept_support
                + 0.16 * teaching_density
                + 0.16 * clipability
                + 0.08 * duration_fit
                + 0.06 * float(clip_context.get("function_confidence") or 0.0)
                - 0.14 * novelty_penalty
            )
            candidates.append(
                SegmentMatch(
                    chunk_index=len(candidates),
                    t_start=float(window_start),
                    t_end=float(window_end),
                    text=text[:1200],
                    score=float(score),
                )
            )

        candidates.sort(key=lambda item: item.score, reverse=True)
        selected: list[SegmentMatch] = []
        for candidate in candidates:
            duplicate = False
            for prior in selected:
                lexical = lexical_overlap_score(candidate.text, [prior.text])
                if lexical >= 0.7:
                    duplicate = True
                    break
                if lexical >= 0.35:
                    similarity = self._text_pair_similarity(
                        conn,
                        left_text=candidate.text,
                        right_text=prior.text,
                        fast_mode=fast_mode,
                    )
                    if similarity >= 0.82:
                        duplicate = True
                        break
            if duplicate:
                continue
            selected.append(candidate)
            if len(selected) >= max(1, int(max_windows or 1)):
                break
        return selected


    def _stage_duration_plan(
        self,
        stage_name: str,
        preferred_video_duration: str,
        video_pool_mode: str,
        fast_mode: bool,
        retrieval_profile: RetrievalProfile,
    ) -> tuple[str | None, ...]:
        if retrieval_profile == "bootstrap":
            if preferred_video_duration in {"short", "medium", "long"}:
                return (preferred_video_duration, None)
            return (None,)
        if stage_name == "high_precision":
            if preferred_video_duration in {"short", "medium", "long"}:
                return (preferred_video_duration,)
            return ("short", None)
        if stage_name == "broad":
            if fast_mode:
                if preferred_video_duration in {"short", "medium", "long"}:
                    return (preferred_video_duration, None)
                if video_pool_mode == "long-form":
                    return ("long", None)
                if video_pool_mode == "balanced":
                    return ("short", "long")
                return ("short", None)
            return self._duration_plan(video_pool_mode, preferred_video_duration)
        if fast_mode:
            if preferred_video_duration in {"short", "medium", "long"}:
                return (preferred_video_duration, None)
            return (None, "short")
        return (None, "short", "medium", "long")

    def _score_video_candidate(
        self,
        conn,
        video: dict[str, Any],
        query_candidate: QueryCandidate,
        concept_terms: list[str],
        context_terms: list[str],
        concept_embedding: np.ndarray | None,
        subject_tag: str | None,
        visual_spec: dict[str, list[str]],
        preferred_video_duration: str,
        stage_name: str,
        require_context: bool,
        fast_mode: bool,
        quick_signals: dict[str, Any] | None = None,
        root_topic_terms: list[str] | None = None,
        strict_topic_only: bool = False,
    ) -> dict[str, Any]:
        title = str(video.get("title") or "")
        description = str(video.get("description") or "")
        if self._is_hard_blocked_low_value_video(
            title=title,
            description=description,
            channel_title=str(video.get("channel_title") or ""),
            subject_tag=subject_tag,
        ):
            return {
                "passes": False,
                "final_score": 0.0,
                "discovery_score": 0.0,
                "clipability_score": 0.0,
                "text_relevance": {"passes": False, "score": 0.0, "off_topic_penalty": 1.0},
                "features": {
                    "semantic_title": 0.0,
                    "semantic_description": 0.0,
                    "strategy_prior": 0.0,
                    "duration_fit": 0.0,
                    "freshness_fit": 0.0,
                    "channel_quality": 0.0,
                    "engagement_fit": 0.0,
                    "educational_intent": 0.0,
                    "visual_intent_match": 0.0,
                    "query_alignment": 0.0,
                    "query_alignment_hits": [],
                    "root_topic_alignment": 0.0,
                    "root_topic_alignment_hits": [],
                    "specific_concept_anchor": 0.0,
                    "specific_concept_anchor_hits": [],
                    "specific_concept_anchor_required": False,
                    "lexicon_noise": 1.0,
                    "source_prior": 0.0,
                },
            }
        metadata_text = str((quick_signals or {}).get("metadata_text") or self._video_metadata_text(video))
        text_relevance = self._score_text_relevance(
            conn,
            text=metadata_text,
            concept_terms=concept_terms,
            context_terms=context_terms,
            concept_embedding=concept_embedding,
            subject_tag=subject_tag,
        )

        semantic_title = self._semantic_similarity(
            conn,
            text=title,
            concept_terms=concept_terms,
            concept_embedding=concept_embedding,
        )
        skip_semantic_description = bool((quick_signals or {}).get("skip_semantic_description"))
        semantic_description = 0.0
        if description and not skip_semantic_description:
            semantic_description = self._semantic_similarity(
                conn,
                text=description,
                concept_terms=concept_terms,
                concept_embedding=concept_embedding,
            )
        strategy_prior = float(self.QUERY_STRATEGY_PRIOR.get(query_candidate.strategy, 0.66))
        strategy_prior *= self._learned_strategy_factor(conn, query_candidate.strategy)
        stage_prior = 0.1 if stage_name == "high_precision" else (0.05 if stage_name == "broad" else 0.0)
        duration_fit = self._duration_fit_score(
            duration_sec=int(video.get("duration_sec") or 0),
            preferred_video_duration=preferred_video_duration,
        )
        freshness_fit = self._freshness_fit_score(video.get("published_at"))
        channel_quality = self._channel_quality_score(video, conn=conn)
        if quick_signals is not None and "educational_intent" in quick_signals:
            educational_intent = float(quick_signals.get("educational_intent") or 0.0)
        else:
            educational_intent = self._educational_intent_score(video)
        cached_query_alignment = (quick_signals or {}).get("query_alignment")
        if isinstance(cached_query_alignment, dict):
            query_alignment = dict(cached_query_alignment)
        else:
            query_alignment = self._query_alignment_score(
                metadata_text,
                query_candidate=query_candidate,
                subject_tag=subject_tag,
            )
        cached_root_alignment = (quick_signals or {}).get("root_topic_alignment")
        if isinstance(cached_root_alignment, dict):
            root_topic_alignment = dict(cached_root_alignment)
        else:
            root_topic_alignment = self._root_topic_alignment_score(
                metadata_text,
                subject_tag=subject_tag,
                root_topic_terms=root_topic_terms,
            )
        cached_specific_anchor = (quick_signals or {}).get("specific_concept_anchor")
        if isinstance(cached_specific_anchor, dict):
            specific_concept_anchor = dict(cached_specific_anchor)
        else:
            specific_concept_anchor = self._specific_concept_anchor_score(
                metadata_text,
                query_candidate=query_candidate,
            )
        visual_intent_match = lexical_overlap_score(
            f"{title} {description}",
            [
                *visual_spec.get("environment", []),
                *visual_spec.get("objects", []),
                *visual_spec.get("actions", []),
            ],
        )
        source_prior = float(self.SOURCE_SURFACE_PRIOR.get(str(video.get("search_source") or ""), 0.82))
        clipability = self._score_clipability_from_metadata(video, strategy=query_candidate.strategy)
        engagement_fit = self._engagement_fit_score(video.get("view_count"))
        lexicon_noise = self._lexicon_noise_score(video, subject_tag=subject_tag)
        discovery_feature_score = (
            0.26 * semantic_title
            + 0.18 * semantic_description
            + 0.10 * strategy_prior
            + 0.08 * duration_fit
            + 0.04 * freshness_fit
            + 0.10 * channel_quality
            + 0.03 * engagement_fit
            + 0.05 * visual_intent_match
            + 0.10 * educational_intent
            + 0.12 * float(query_alignment.get("score") or 0.0)
            + 0.09 * float(root_topic_alignment.get("score") or 0.0)
            + 0.13 * float(specific_concept_anchor.get("score") or 0.0)
        )
        discovery_total_weight = 1.15 + stage_prior
        discovery = (discovery_feature_score + stage_prior) / max(1e-6, discovery_total_weight)
        discovery *= source_prior
        discovery -= 0.22 * lexicon_noise
        discovery = float(max(0.0, min(1.0, discovery)))

        text_pass = self._passes_relevance_gate(
            relevance=text_relevance,
            require_context=require_context,
            fast_mode=fast_mode,
        )
        strong_text_support = self._has_strong_topic_support(text_relevance, fast_mode=fast_mode)
        low_quality_channel = self._infer_channel_tier(
            channel=str(video.get("channel_title") or "").lower(),
            title=title.lower(),
        ) == "low_quality_compilation"
        risky_source = str(video.get("search_source") or "") in self.RISKY_SEARCH_SOURCES
        root_anchor_required = (
            strict_topic_only
            and str(video.get("search_source") or "") in self.STRICT_TOPIC_ROOT_ANCHOR_SOURCES
        )
        query_alignment_score = float(query_alignment.get("score") or 0.0)
        root_topic_alignment_score = float(root_topic_alignment.get("score") or 0.0)
        specific_concept_anchor_score = float(specific_concept_anchor.get("score") or 0.0)
        specific_concept_anchor_required = bool(specific_concept_anchor.get("required"))
        duration_sec = max(0, int(video.get("duration_sec") or 0))
        short_form_topic_support = (
            0 < duration_sec <= 3 * 60
            and (
                strong_text_support
                or query_alignment_score >= (0.07 if fast_mode else 0.05)
                or root_topic_alignment_score >= (0.1 if fast_mode else 0.08)
                or specific_concept_anchor_score >= (0.14 if fast_mode else 0.1)
            )
        )
        precision_guard = strong_text_support or query_alignment_score >= (0.08 if fast_mode else 0.06)
        if risky_source and not strong_text_support:
            precision_guard = precision_guard and query_alignment_score >= (0.1 if fast_mode else 0.08)
        if low_quality_channel and query_alignment_score < 0.12:
            precision_guard = precision_guard and (educational_intent >= 0.56 or short_form_topic_support)
        if root_anchor_required and root_topic_alignment_score < (0.12 if fast_mode else 0.1):
            precision_guard = False
        if (
            strict_topic_only
            and specific_concept_anchor_required
            and specific_concept_anchor_score < (0.16 if fast_mode else 0.12)
            and root_topic_alignment_score < (0.12 if fast_mode else 0.1)
        ):
            precision_guard = False
        if (
            strict_topic_only
            and lexicon_noise >= 0.26
            and not self._looks_like_language_subject(subject_tag)
            and educational_intent < 0.78
            and specific_concept_anchor_score < 0.18
        ):
            precision_guard = False
        min_discovery = 0.14 if fast_mode else 0.16
        if short_form_topic_support:
            min_discovery = max(0.12 if fast_mode else 0.14, min_discovery - 0.02)
        # Raise the floor when user has set a higher relevance threshold
        if self._min_relevance_threshold > 0.0:
            min_discovery = max(min_discovery, min_discovery + 0.06 * self._min_relevance_threshold)
        passes = text_pass and discovery >= min_discovery and precision_guard

        final_score = 0.74 * discovery + 0.26 * clipability
        text_relevance_result = dict(text_relevance)
        text_relevance_result["passes"] = passes
        return {
            "passes": passes,
            "final_score": float(final_score),
            "discovery_score": discovery,
            "clipability_score": float(clipability),
            "text_relevance": text_relevance_result,
            "features": {
                "semantic_title": float(semantic_title),
                "semantic_description": float(semantic_description),
                "strategy_prior": strategy_prior,
                "duration_fit": float(duration_fit),
                "freshness_fit": float(freshness_fit),
                "channel_quality": float(channel_quality),
                "engagement_fit": float(engagement_fit),
                "educational_intent": float(educational_intent),
                "visual_intent_match": float(visual_intent_match),
                "query_alignment": float(query_alignment_score),
                "query_alignment_hits": list(query_alignment.get("hits") or []),
                "root_topic_alignment": float(root_topic_alignment_score),
                "root_topic_alignment_hits": list(root_topic_alignment.get("hits") or []),
                "specific_concept_anchor": float(specific_concept_anchor_score),
                "specific_concept_anchor_hits": list(specific_concept_anchor.get("hits") or []),
                "specific_concept_anchor_required": bool(specific_concept_anchor_required),
                "lexicon_noise": float(lexicon_noise),
                "source_prior": float(source_prior),
            },
        }

    def _quick_candidate_metadata_gate(
        self,
        *,
        video: dict[str, Any],
        query_candidate: QueryCandidate,
        concept_terms: list[str],
        context_terms: list[str],
        subject_tag: str | None,
        strict_topic_only: bool,
        require_context: bool,
        fast_mode: bool,
        root_topic_terms: list[str] | None = None,
    ) -> dict[str, Any]:
        metadata_text = self._video_metadata_text(video)
        if not metadata_text.strip():
            return {"passes": False, "metadata_text": metadata_text}
        if self._is_hard_blocked_low_value_video(
            title=str(video.get("title") or ""),
            description=str(video.get("description") or ""),
            channel_title=str(video.get("channel_title") or ""),
            subject_tag=subject_tag,
        ):
            return {"passes": False, "metadata_text": metadata_text, "lexicon_noise": 1.0}

        query_alignment = self._query_alignment_score(
            metadata_text,
            query_candidate=query_candidate,
            subject_tag=subject_tag,
        )
        root_topic_alignment = self._root_topic_alignment_score(
            metadata_text,
            subject_tag=subject_tag,
            root_topic_terms=root_topic_terms,
        )
        specific_concept_anchor = self._specific_concept_anchor_score(
            metadata_text,
            query_candidate=query_candidate,
        )
        query_alignment_score = float(query_alignment.get("score") or 0.0)
        root_topic_alignment_score = float(root_topic_alignment.get("score") or 0.0)
        specific_concept_anchor_score = float(specific_concept_anchor.get("score") or 0.0)
        specific_concept_anchor_required = bool(specific_concept_anchor.get("required"))
        concept_hits = self._extract_matched_terms(metadata_text, concept_terms, limit=6)
        context_hits = self._extract_matched_terms(metadata_text, context_terms, limit=4) if context_terms else []
        concept_overlap = lexical_overlap_score(metadata_text, concept_terms)
        context_overlap = lexical_overlap_score(metadata_text, context_terms) if context_terms else 0.0
        subject_overlap = lexical_overlap_score(metadata_text, [subject_tag]) if subject_tag else 0.0
        educational_intent = self._educational_intent_score(video)
        lexicon_noise = self._lexicon_noise_score(video, subject_tag=subject_tag)
        low_quality_channel = self._infer_channel_tier(
            channel=str(video.get("channel_title") or "").lower(),
            title=str(video.get("title") or "").lower(),
        ) == "low_quality_compilation"
        risky_source = str(video.get("search_source") or "") in self.RISKY_SEARCH_SOURCES
        root_anchor_required = (
            strict_topic_only
            and str(video.get("search_source") or "") in self.STRICT_TOPIC_ROOT_ANCHOR_SOURCES
        )

        lowered = f" {metadata_text.lower()} "
        phrase_hits = 0
        for raw_term in concept_terms:
            normalized = " ".join(str(raw_term or "").lower().split()).strip()
            if " " in normalized and f" {normalized} " in lowered:
                phrase_hits += 1
                if phrase_hits >= 2:
                    break

        strong_query = query_alignment_score >= (0.1 if fast_mode else 0.08)
        strong_concept_support = (
            len(concept_hits) >= 2
            or concept_overlap >= (0.08 if fast_mode else 0.06)
            or phrase_hits > 0
        )
        duration_sec = max(0, int(video.get("duration_sec") or 0))
        short_form_topic_support = (
            0 < duration_sec <= 3 * 60
            and (
                strong_query
                or root_topic_alignment_score >= (0.11 if fast_mode else 0.09)
                or specific_concept_anchor_score >= (0.14 if fast_mode else 0.1)
                or phrase_hits > 0
                or len(concept_hits) >= 1
            )
        )
        context_supported = (
            not require_context
            or bool(context_hits)
            or context_overlap >= (0.04 if fast_mode else 0.03)
            or strong_query
        )
        educational_support = educational_intent >= (0.62 if fast_mode else 0.58)
        passes = (
            (
                strong_query
                or strong_concept_support
                or short_form_topic_support
                or (bool(concept_hits) and educational_support)
            )
            and context_supported
        )
        if risky_source and not strong_query and len(concept_hits) < 2:
            passes = False
        if (
            low_quality_channel
            and not strong_query
            and not educational_support
            and not short_form_topic_support
            and len(concept_hits) < 2
        ):
            passes = False
        if strict_topic_only and subject_tag:
            direct_topic_support = (
                query_alignment_score >= (0.1 if fast_mode else 0.08)
                or root_topic_alignment_score >= (0.12 if fast_mode else 0.1)
                or specific_concept_anchor_score >= (0.16 if fast_mode else 0.12)
                or subject_overlap >= 0.04
                or concept_overlap >= (0.08 if fast_mode else 0.06)
                or len(concept_hits) >= 2
                or phrase_hits > 0
                or short_form_topic_support
            )
            if not direct_topic_support:
                passes = False
            if root_anchor_required and root_topic_alignment_score < (0.12 if fast_mode else 0.1):
                passes = False
            if (
                specific_concept_anchor_required
                and specific_concept_anchor_score < (0.16 if fast_mode else 0.12)
                and root_topic_alignment_score < (0.12 if fast_mode else 0.1)
            ):
                passes = False
            if (
                lexicon_noise >= 0.26
                and not self._looks_like_language_subject(subject_tag)
                and educational_intent < 0.78
                and specific_concept_anchor_score < 0.18
            ):
                passes = False

        return {
            "passes": passes,
            "metadata_text": metadata_text,
            "query_alignment": query_alignment,
            "root_topic_alignment": root_topic_alignment,
            "specific_concept_anchor": specific_concept_anchor,
            "educational_intent": educational_intent,
            "lexicon_noise": lexicon_noise,
            "skip_semantic_description": fast_mode and (strong_query or len(concept_hits) >= 2 or phrase_hits > 0),
        }

    def _educational_intent_score(self, video: dict[str, Any], transcript: list[dict[str, Any]] | None = None) -> float:
        """Score educational quality 0.0-1.0 using metadata + optional transcript signals.

        Six weighted signals:
          1. Keyword presence (0.15) — existing edu/entertainment token matching
          2. Title structure  (0.20) — regex patterns for structured educational content
          3. Description depth (0.15) — length + timestamp markers + link presence
          4. Duration band     (0.10) — 5-30 min educational sweet spot
          5. Channel tier      (0.20) — _infer_channel_tier mapped to 0-1
          6. Transcript vocab  (0.20) — unique word ratio (optional, when available)
        """
        title = str(video.get("title") or "")
        description = str(video.get("description") or "")
        channel = str(video.get("channel_title") or "")
        duration = int(video.get("duration_sec") or 0)

        score = 0.0

        # Signal 1: Keyword presence (0.15)
        metadata_tokens = normalize_terms([title, description, channel])
        edu_hits = len(metadata_tokens.intersection(self.EDUCATIONAL_CUE_TERMS))
        ent_hits = len(metadata_tokens.intersection(self.ENTERTAINMENT_CONFLICT_TOKENS))
        keyword_score = 0.5 + 0.1 * min(3, edu_hits) - 0.15 * min(3, ent_hits)
        score += 0.15 * max(0.0, min(1.0, keyword_score))

        # Signal 2: Title structure patterns (0.20)
        structure_score = 0.3
        title_lower = title.lower()
        if re.search(r'\b(?:part|chapter|lecture|episode|module|unit)\s+\d', title_lower):
            structure_score = 0.9
        if re.search(r'\b(?:explained|introduction to|overview of|fundamentals of)\b', title_lower):
            structure_score = max(structure_score, 0.8)
        if re.search(r'\b(?:how to|step.by.step|beginner|complete guide)\b', title_lower):
            structure_score = max(structure_score, 0.7)
        if re.search(r'\b(?:worked example|problem set|practice)\b', title_lower):
            structure_score = max(structure_score, 0.75)
        score += 0.20 * structure_score

        # Signal 3: Description depth (0.15)
        desc_len = len(description)
        has_timestamps = bool(re.search(r'\d{1,2}:\d{2}', description))
        has_links = 'http' in description.lower()
        desc_score = min(1.0, desc_len / 800)
        if has_timestamps:
            desc_score = min(1.0, desc_score + 0.2)
        if has_links:
            desc_score = min(1.0, desc_score + 0.1)
        score += 0.15 * desc_score

        # Signal 4: Duration appropriateness (0.10)
        if 300 <= duration <= 1800:
            dur_score = 0.9
        elif 180 <= duration <= 3600:
            dur_score = 0.6
        elif duration > 3600:
            dur_score = 0.5
        elif duration > 0:
            dur_score = 0.2
        else:
            dur_score = 0.4  # unknown duration
        score += 0.10 * dur_score

        # Signal 5: Channel tier (0.20)
        tier = self._infer_channel_tier(channel.lower(), title_lower)
        tier_scores = {
            "known_educational": 1.0, "education": 0.85, "tutorial": 0.7,
            "news": 0.6, "podcast": 0.4, "stock_footage": 0.3,
            "low_quality_compilation": 0.1,
            "entertainment_media": 0.05,
        }
        score += 0.20 * tier_scores.get(tier, 0.5)

        # Signal 6: Transcript vocabulary richness (0.20, optional)
        if transcript and len(transcript) > 10:
            words = " ".join(str(c.get("text", "")) for c in transcript[:200]).split()
            if len(words) > 20:
                unique_ratio = len(set(w.lower() for w in words)) / len(words)
                vocab_score = min(1.0, unique_ratio / 0.5)
                score += 0.20 * vocab_score
            else:
                score = score / 0.80  # redistribute weight
        else:
            score = score / 0.80  # redistribute weight

        return float(max(0.0, min(1.0, score)))

    def _looks_like_language_subject(self, subject_tag: str | None) -> bool:
        cleaned = self._clean_query_text(subject_tag or "")
        if not cleaned:
            return False
        try:
            return bool(self.topic_expansion_service._looks_like_language_topic(cleaned))
        except Exception:
            return False

    def _lexicon_noise_score(self, video: dict[str, Any], *, subject_tag: str | None) -> float:
        if self._looks_like_language_subject(subject_tag):
            return 0.0
        metadata_text = self._video_metadata_text(video)
        lowered = f" {metadata_text.lower()} "
        metadata_tokens = normalize_terms(
            [
                str(video.get("title") or ""),
                str(video.get("description") or ""),
                str(video.get("channel_title") or ""),
            ]
        )
        score = 0.0
        score += 0.08 * min(4, len(metadata_tokens.intersection(self.LEXICON_CONFLICT_TOKENS)))
        for phrase, penalty in self.LEXICON_CONFLICT_PHRASES.items():
            if f" {phrase} " in lowered:
                score += penalty
        score -= 0.06 * min(2, len(metadata_tokens.intersection(self.STRONG_EDUCATIONAL_CUE_TERMS)))
        return float(max(0.0, min(1.0, score)))

    def _is_hard_blocked_low_value_video(
        self,
        *,
        title: str,
        description: str,
        channel_title: str,
        subject_tag: str | None,
    ) -> bool:
        if self._looks_like_language_subject(subject_tag):
            return False

        lowered = f" {self._clean_query_text(' '.join([title, description, channel_title])).lower()} "
        normalized_subject = self._clean_query_text(subject_tag or "").lower()
        metadata_tokens = normalize_terms([title, description, channel_title])

        # Block known entertainment channels that mimic academic names.
        channel_lower = channel_title.strip().lower()
        if channel_lower in self.KNOWN_ENTERTAINMENT_CHANNELS:
            return True
        # Block entertainment title patterns.
        title_lower = title.strip().lower()
        if any(p in title_lower for p in self.ENTERTAINMENT_TITLE_PATTERNS):
            return True
        if " provided to youtube " in lowered:
            return True
        if normalized_subject == "calculus" and " lambda calculus " in lowered:
            return True
        if normalized_subject:
            if re.search(
                rf"\b{re.escape(normalized_subject)}\b.*\b(meaning|pronunciation|definition|dictionary)\b",
                lowered,
            ) or re.search(
                rf"\b(meaning|pronunciation|definition|dictionary)\b.*\b{re.escape(normalized_subject)}\b",
                lowered,
            ):
                return True
            if re.search(
                rf"\bwhat does\b.*\b{re.escape(normalized_subject)}\b.*\bmean\b",
                lowered,
            ):
                return True
        if any(f" {phrase} " in lowered for phrase in self.LEXICON_CONFLICT_PHRASES):
            return True
        if len(metadata_tokens.intersection(self.LEXICON_CONFLICT_TOKENS)) >= 2:
            return True
        if len(metadata_tokens.intersection(self.ENTERTAINMENT_CONFLICT_TOKENS)) >= 2:
            return True
        if any(
            token in lowered
            for token in (
                " official audio ",
                " karaoke ",
                " lyrics ",
                " lyric video ",
                " remix ",
                " music video ",
            )
        ):
            return True
        return False

    def _specific_concept_anchor_terms(self, query_candidate: QueryCandidate) -> list[str]:
        seen: set[str] = set()
        terms: list[str] = []
        raw_candidates = [*query_candidate.source_terms, query_candidate.text]
        for raw_term in raw_candidates:
            cleaned = self._clean_query_text(raw_term)
            normalized = self._normalize_query_key(cleaned)
            if not cleaned or not normalized or normalized in seen:
                continue
            tokens = [
                token
                for token in normalized.split()
                if token not in self.GENERIC_CONTEXT_TERMS and token not in self.QUERY_ALIGNMENT_NOISE_TOKENS
            ]
            if len(tokens) < 2:
                continue
            if all(token in self.EDUCATIONAL_CUE_TERMS for token in tokens):
                continue
            seen.add(normalized)
            terms.append(" ".join(tokens))
            if len(terms) >= 4:
                break
        return terms

    def _specific_concept_anchor_score(
        self,
        text: str,
        *,
        query_candidate: QueryCandidate,
    ) -> dict[str, Any]:
        cleaned = self._clean_query_text(text)
        anchor_terms = self._specific_concept_anchor_terms(query_candidate)
        if not cleaned or not anchor_terms:
            return {"score": 0.0, "hits": [], "terms": anchor_terms, "required": bool(anchor_terms)}

        normalized_text = self._normalize_query_key(cleaned)
        lowered = f" {normalized_text} "
        text_tokens = set(normalized_text.split())
        hits: list[str] = []
        best_score = 0.0
        for term in anchor_terms:
            normalized = self._normalize_query_key(term)
            tokens = normalized.split()
            if not tokens:
                continue
            overlap = len(text_tokens.intersection(tokens))
            coverage = overlap / max(1, len(tokens))
            hit_score = 0.0
            if f" {normalized} " in lowered or set(tokens).issubset(text_tokens):
                hit_score = 0.34 if len(tokens) <= 2 else 0.3
            elif len(tokens) >= 3 and coverage >= 0.67:
                hit_score = 0.18
            if hit_score > 0.0:
                hits.append(term)
            best_score = max(best_score, hit_score)
        return {
            "score": float(max(0.0, min(1.0, best_score))),
            "hits": hits[:4],
            "terms": anchor_terms,
            "required": bool(anchor_terms),
        }

    def _query_alignment_terms(self, query_candidate: QueryCandidate, subject_tag: str | None) -> list[str]:
        seen: set[str] = set()
        terms: list[str] = []
        for raw_term in [*query_candidate.source_terms, subject_tag or ""]:
            normalized = " ".join(str(raw_term or "").split()).strip()
            key = normalized.lower()
            if not normalized or key in seen:
                continue
            seen.add(key)
            terms.append(normalized)

        noisy_tokens = self.GENERIC_CONTEXT_TERMS.union(self.QUERY_ALIGNMENT_NOISE_TOKENS)
        for token in normalize_terms([query_candidate.text, subject_tag or ""]):
            if len(token) < 4 or token in noisy_tokens or token in seen:
                continue
            seen.add(token)
            terms.append(token)
            if len(terms) >= 8:
                break
        return terms[:8]

    def _query_alignment_score(
        self,
        text: str,
        *,
        query_candidate: QueryCandidate,
        subject_tag: str | None,
    ) -> dict[str, Any]:
        cleaned = " ".join(str(text or "").split()).strip()
        if not cleaned:
            return {"score": 0.0, "hits": [], "terms": []}

        alignment_terms = self._query_alignment_terms(query_candidate, subject_tag)
        if not alignment_terms:
            return {"score": 0.0, "hits": [], "terms": []}

        hits = self._extract_matched_terms(cleaned, alignment_terms, limit=6)
        overlap = lexical_overlap_score(cleaned, alignment_terms)
        lowered = f" {cleaned.lower()} "
        phrase_terms = [term for term in alignment_terms if " " in str(term).strip()]
        phrase_hits = 0
        for term in phrase_terms:
            normalized = " ".join(str(term or "").lower().split()).strip()
            if f" {normalized} " in lowered:
                phrase_hits += 1

        score = overlap + 0.04 * min(4, len(hits)) + 0.06 * min(2, phrase_hits)
        if phrase_terms and phrase_hits == 0 and len(hits) <= 1:
            score *= 0.3
        elif len(alignment_terms) >= 3 and len(hits) == 1:
            score *= 0.5
        return {
            "score": float(max(0.0, min(1.0, score))),
            "hits": hits,
            "terms": alignment_terms,
        }

    def _root_topic_alignment_terms(
        self,
        *,
        subject_tag: str | None,
        root_topic_terms: list[str] | None,
    ) -> list[str]:
        seen: set[str] = set()
        terms: list[str] = []
        for raw_term in [subject_tag or "", *(root_topic_terms or [])]:
            cleaned = self._clean_query_text(raw_term)
            normalized = self._normalize_query_key(cleaned)
            if not cleaned or not normalized or normalized in seen:
                continue
            seen.add(normalized)
            terms.append(cleaned)
            if len(terms) >= 8:
                break
        return terms

    def _root_topic_alignment_score(
        self,
        text: str,
        *,
        subject_tag: str | None,
        root_topic_terms: list[str] | None,
    ) -> dict[str, Any]:
        cleaned = self._clean_query_text(text)
        if not cleaned:
            return {"score": 0.0, "hits": [], "terms": []}

        alignment_terms = self._root_topic_alignment_terms(
            subject_tag=subject_tag,
            root_topic_terms=root_topic_terms,
        )
        if not alignment_terms:
            return {"score": 0.0, "hits": [], "terms": []}

        normalized_text = self._normalize_query_key(cleaned)
        lowered = f" {normalized_text} "
        text_tokens = set(normalized_text.split())
        hits: list[str] = []
        seen: set[str] = set()
        phrase_hits = 0
        single_hits = 0
        for term in alignment_terms:
            normalized = self._normalize_query_key(term)
            if not normalized:
                continue
            tokens = normalized.split()
            if len(tokens) > 1:
                if f" {normalized} " in lowered or set(tokens).issubset(text_tokens):
                    if normalized not in seen:
                        seen.add(normalized)
                        hits.append(normalized)
                    phrase_hits += 1
            elif normalized in text_tokens:
                if normalized not in seen:
                    seen.add(normalized)
                    hits.append(normalized)
                single_hits += 1

        score = 0.14 * min(2, single_hits) + 0.2 * min(2, phrase_hits)
        return {
            "score": float(max(0.0, min(1.0, score))),
            "hits": hits,
            "terms": alignment_terms,
        }

    def _has_strong_topic_support(self, relevance: dict[str, Any], *, fast_mode: bool) -> bool:
        concept_hits = len(relevance.get("concept_hits") or [])
        context_hits = len(relevance.get("context_hits") or [])
        concept_overlap = float(relevance.get("concept_overlap") or 0.0)
        context_overlap = float(relevance.get("context_overlap") or 0.0)
        subject_overlap = float(relevance.get("subject_overlap") or 0.0)
        embedding_sim = float(relevance.get("embedding_sim") or 0.0)
        score = float(relevance.get("score") or -1.0)

        if concept_hits >= 2 or context_hits >= 2:
            return True
        if concept_overlap >= 0.12 or context_overlap >= 0.08 or subject_overlap >= 0.08:
            return True
        if concept_hits >= 1 and (context_overlap >= 0.04 or subject_overlap >= 0.04):
            return True
        return embedding_sim >= (0.36 if fast_mode else 0.32) and score >= (0.12 if fast_mode else 0.1)

    def _engagement_fit_score(self, view_count: Any) -> float:
        try:
            views = int(view_count or 0)
        except (TypeError, ValueError):
            views = 0
        if views <= 0:
            return 0.55
        if views >= 5_000_000:
            return 0.95
        if views >= 1_000_000:
            return 0.9
        if views >= 300_000:
            return 0.84
        if views >= 100_000:
            return 0.78
        if views >= 30_000:
            return 0.7
        if views >= 10_000:
            return 0.64
        return 0.58

    def _semantic_similarity(
        self,
        conn,
        text: str,
        concept_terms: list[str],
        concept_embedding: np.ndarray | None,
    ) -> float:
        cleaned = " ".join(str(text or "").split()).strip()
        if not cleaned:
            return 0.0
        lexical = lexical_overlap_score(cleaned, concept_terms)
        if concept_embedding is None:
            return float(lexical)
        try:
            text_embedding = self.embedding_service.embed_texts(conn, [cleaned])[0]
            semantic = float(text_embedding @ concept_embedding.astype(np.float32))
        except Exception:
            semantic = 0.0
        return float(max(0.0, min(1.0, 0.82 * max(0.0, semantic) + 0.18 * lexical)))

    def _text_pair_similarity(
        self,
        conn,
        *,
        left_text: str,
        right_text: str,
        fast_mode: bool,
    ) -> float:
        left_clean = " ".join(str(left_text or "").split()).strip()
        right_clean = " ".join(str(right_text or "").split()).strip()
        if not left_clean or not right_clean:
            return 0.0
        lexical = lexical_overlap_score(left_clean, [right_clean])
        if fast_mode:
            return float(max(0.0, min(1.0, lexical)))
        try:
            embeddings = self.embedding_service.embed_texts(conn, [left_clean, right_clean])
            if len(embeddings) >= 2:
                semantic = float(embeddings[0] @ embeddings[1].astype(np.float32))
            else:
                semantic = 0.0
        except Exception:
            semantic = 0.0
        return float(max(0.0, min(1.0, 0.72 * max(0.0, semantic) + 0.28 * lexical)))

    def _classify_educational_function(self, text: str) -> tuple[str, float]:
        cleaned = " ".join(str(text or "").split()).strip().lower()
        if not cleaned:
            return ("definition", 0.0)
        cues = {
            "definition": ("is defined as", "means", "refers to", "definition", "what is"),
            "intuition": ("intuition", "think of", "imagine", "picture this", "conceptually"),
            "worked_example": ("for example", "example", "solve", "worked example", "let's do"),
            "derivation": ("derive", "derivation", "proof", "follows from", "therefore"),
            "application": ("application", "used in", "real world", "in practice", "applied to"),
            "misconception": ("common mistake", "misconception", "don't confuse", "pitfall", "wrong"),
            "history": ("history", "historically", "origin", "discovered", "first described"),
        }
        best_label = "definition"
        best_hits = 0
        for label, phrases in cues.items():
            hits = sum(1 for phrase in phrases if phrase in cleaned)
            if hits > best_hits:
                best_hits = hits
                best_label = label
        confidence = min(0.98, 0.48 + 0.18 * best_hits)
        if best_hits <= 0 and any(token in cleaned for token in {"example", "demo", "walkthrough"}):
            best_label = "worked_example"
            confidence = 0.7
        return (best_label, float(max(0.0, confidence)))

    def _build_clip_context(self, *, text: str, clip_duration_sec: float) -> dict[str, Any]:
        function_label, function_confidence = self._classify_educational_function(text)
        return {
            "text": " ".join(str(text or "").split()).strip(),
            "function_label": function_label,
            "function_confidence": float(function_confidence),
            "clip_duration_sec": float(max(0.0, clip_duration_sec)),
        }


    def _clip_self_containment_score(
        self,
        *,
        text: str,
        clip_duration_sec: float,
    ) -> float:
        cleaned = " ".join(str(text or "").split()).strip()
        tokens = normalize_terms([cleaned])
        if not tokens:
            return 0.0
        score = 0.18
        score += min(0.32, 0.02 * len(tokens))
        if clip_duration_sec >= 18:
            score += 0.18
        if clip_duration_sec >= 28:
            score += 0.08
        if re.search(r"[.!?][\"')\]]?$", cleaned):
            score += 0.14
        if len(re.findall(r"[A-Za-z]", cleaned)) >= 40:
            score += 0.1
        return float(max(0.0, min(1.0, score)))


    def _passes_same_video_clip_novelty(
        self,
        conn,
        *,
        clip_context: dict[str, Any],
        prior_contexts: list[dict[str, Any]],
        subject_tag: str | None,
        retrieval_profile: RetrievalProfile,
        fast_mode: bool,
    ) -> bool:
        if not prior_contexts:
            return True
        thresholds = self._topic_novelty_profile(
            subject_tag=subject_tag,
            retrieval_profile=retrieval_profile,
            fast_mode=fast_mode,
            conn=conn,
        )
        same_video_threshold = float(thresholds.get("same_video_similarity") or 0.88)
        current_text = str(clip_context.get("text") or "")
        current_label = str(clip_context.get("function_label") or "")
        current_confidence = float(clip_context.get("function_confidence") or 0.0)
        for prior in prior_contexts:
            similarity = self._text_pair_similarity(
                conn,
                left_text=current_text,
                right_text=str(prior.get("text") or ""),
                fast_mode=fast_mode,
            )
            if similarity <= 0.0:
                continue
            semantic_distance = max(0.0, 1.0 - similarity)
            prior_label = str(prior.get("function_label") or "")
            if semantic_distance >= 0.12:
                continue
            if (
                current_label
                and current_label != prior_label
                and current_confidence >= 0.85
                and semantic_distance >= 0.08
            ):
                continue
            if similarity >= same_video_threshold:
                return False
        return True


    def _duration_fit_score(self, duration_sec: int, preferred_video_duration: str) -> float:
        if duration_sec <= 0:
            return 0.55
        bucket = self._video_duration_bucket(duration_sec)
        if preferred_video_duration in {"short", "medium", "long"}:
            return 0.95 if bucket == preferred_video_duration else 0.35
        if bucket == "short":
            return 0.88
        if bucket == "medium":
            return 0.72
        return 0.5

    def _freshness_fit_score(self, published_at: Any) -> float:
        published = str(published_at or "").strip()
        if not published:
            return 0.58
        try:
            year = int(published[:4])
        except (TypeError, ValueError):
            return 0.58
        if year >= 2024:
            return 0.84
        if year >= 2020:
            return 0.72
        if year >= 2015:
            return 0.6
        return 0.46

    def _channel_quality_score(self, video: dict[str, Any], conn=None) -> float:
        title = str(video.get("title") or "").lower()
        channel = str(video.get("channel_title") or "").lower()
        bucket = self._infer_channel_tier(channel=channel, title=title)
        bonus = float(self.CHANNEL_QUALITY_BONUS.get(bucket, 0.0))
        base = float(max(0.0, min(1.0, 0.6 + bonus)))
        # Engagement signal: high views/minute suggests popular educational content.
        # view_count is available from HTML renderer data (no API key needed).
        view_count = int(video.get("view_count") or 0)
        duration_sec = int(video.get("duration_sec") or 0)
        if view_count > 0 and duration_sec > 60:
            views_per_min = view_count / max(1, duration_sec / 60)
            if views_per_min > 10000:
                base = min(1.0, base + 0.05)
            elif views_per_min < 10:
                base = max(0.0, base - 0.03)
        # Fix T: Boost/penalize channels based on user feedback
        if conn is not None and channel:
            feedback_factor = self._feedback_channel_factor(conn, channel)
            base = float(max(0.0, min(1.0, base * feedback_factor)))
        return base

    def _feedback_channel_factor(self, conn, channel: str) -> float:
        """Return a multiplicative factor based on aggregated user feedback for this channel."""
        with self._strategy_history_cache_lock:
            cached = self._strategy_history_cache.get(f"channel_fb:{channel}")
        if cached is not None:
            return cached

        try:
            row = fetch_one(
                conn,
                """
                SELECT
                    SUM(f.helpful) AS total_helpful,
                    SUM(f.confusing) AS total_confusing,
                    SUM(f.saved) AS total_saved,
                    COUNT(*) AS total_feedback
                FROM reel_feedback f
                JOIN reels r ON r.id = f.reel_id
                JOIN videos v ON v.id = r.video_id
                WHERE LOWER(v.channel_title) = ?
                """,
                (channel,),
            )
        except Exception:
            row = None

        if not row or not int(row.get("total_feedback") or 0):
            factor = 1.0
        else:
            helpful = int(row.get("total_helpful") or 0)
            saved = int(row.get("total_saved") or 0)
            confusing = int(row.get("total_confusing") or 0)
            # Positive signals boost, negative signals penalize
            factor = 1.0 + 0.08 * min(5, helpful + saved) - 0.12 * min(3, confusing)
            factor = max(0.7, min(1.3, factor))

        with self._strategy_history_cache_lock:
            self._strategy_history_cache[f"channel_fb:{channel}"] = factor
        return factor

    def _learned_strategy_factor(self, conn, strategy: str) -> float:
        key = str(strategy or "").strip().lower() or "literal"
        with self._strategy_history_cache_lock:
            cached = self._strategy_history_cache.get(key)
        if cached is not None:
            return cached

        try:
            row = fetch_one(
                conn,
                """
                SELECT
                    AVG(CASE WHEN result_count > 0 THEN CAST(kept_count AS REAL) / result_count ELSE 0 END) AS kept_ratio
                FROM (
                    SELECT result_count, kept_count
                    FROM retrieval_queries
                    WHERE strategy = ?
                    ORDER BY created_at DESC
                    LIMIT 240
                ) q
                """,
                (key,),
            )
        except Exception:
            row = None

        kept_ratio = float((row or {}).get("kept_ratio") or 0.0)
        # Map historical keep ratio into a multiplicative factor (widened range for more impact).
        factor = max(0.75, min(1.35, 0.85 + 0.65 * kept_ratio))
        with self._strategy_history_cache_lock:
            existing = self._strategy_history_cache.get(key)
            if existing is not None:
                return existing
            self._strategy_history_cache[key] = factor
            return factor

    def _infer_channel_tier(self, channel: str, title: str) -> str:
        if channel.strip() in self.KNOWN_EDUCATIONAL_CHANNELS:
            return "known_educational"
        # Block known entertainment channels that use academic-sounding names.
        if channel.strip().lower() in self.KNOWN_ENTERTAINMENT_CHANNELS:
            return "low_quality_compilation"
        hay_lower = f"{channel} {title}".lower()
        title_lower = title.lower().strip()
        # Film/TV entertainment patterns — must be checked before education so a
        # title like "Calculus Scene - Mean Girls" doesn't classify as
        # educational on the "scene"-adjacent text.
        if any(token in hay_lower for token in self.MOVIE_TITLE_PATTERNS):
            return "entertainment_media"
        # Movie-clip titles commonly end in "Scene", "Scenes", "- HD", etc.
        # Regex catches patterns that pure substring matching misses
        # (e.g. "Mean Girls - Calculus Scene").
        if re.search(
            r"\b(scene|scenes)\s*(?:\([^\)]*\))?\s*[\-\|\[]*\s*(?:hd|4k|1080p|720p)?\s*$",
            title_lower,
        ):
            return "entertainment_media"
        # "(Movie) Scene", "(Film) Scene" — parenthetical or bracketed film tags.
        if re.search(r"\(\s*(movie|film|tv|show)\b", title_lower):
            return "entertainment_media"
        # From here on, substring checks compare against a LOWERCASED haystack
        # (hay_lower) so titles like "Calculus Basics Explained Simply" still
        # match the "explained" token even when capitalised.
        if any(token in hay_lower for token in ["vevo", "official audio", "lyrics", "karaoke", "remix"]):
            return "low_quality_compilation"
        if any(token in hay_lower for token in ["compilation", "best moments", "reaction", "clips", "tiktok", "meme"]):
            return "low_quality_compilation"
        if any(token in hay_lower for token in ["news", "times", "bbc", "cnn", "reuters", "al jazeera", "pbs"]):
            return "news"
        # Use word-boundary-aware matching for "academy" to avoid false positives
        # like "Academy Awards" or gaming channel names containing "academy".
        if any(token in hay_lower for token in ["course", "university", "opencourseware", "explained"]):
            return "education"
        if re.search(r'\b(?:academy|education|science)\b', hay_lower) and not any(
            noise in hay_lower for noise in ["gaming", "awards", "beauty", "sport", "fitness"]
        ):
            return "education"
        if any(token in hay_lower for token in ["stock", "footage", "cinematic"]):
            return "stock_footage"
        if any(token in hay_lower for token in ["tutorial", "how to", "walkthrough", "demo", "masterclass"]):
            return "tutorial"
        if "podcast" in hay_lower:
            return "podcast"
        return "tutorial"


    def _score_clipability_from_metadata(self, video: dict[str, Any], strategy: str) -> float:
        duration_sec = int(video.get("duration_sec") or 0)
        title = str(video.get("title") or "").lower()
        description = str(video.get("description") or "").lower()

        if duration_sec <= 0:
            duration_score = 0.58
        elif duration_sec <= 75:
            duration_score = 0.9
        elif duration_sec <= 6 * 60:
            duration_score = 0.82
        elif duration_sec <= 18 * 60:
            duration_score = 0.7
        else:
            duration_score = 0.5

        penalty = 0.0
        metadata_text = f"{title} {description}"
        for token in self.CLIPABILITY_PENALTY_TOKENS:
            if token in metadata_text:
                penalty += 0.07
        strategy_boost = 0.08 if strategy in {"scene", "action", "broll"} else 0.0
        return float(max(0.0, min(1.0, duration_score + strategy_boost - penalty)))








    def _decompose_concept_for_recovery(self, title: str, summary: str, keywords: list[str]) -> list[str]:
        lowered = f"{title} {summary}".lower()
        decomposed = []
        if any(token in lowered for token in {"anxiety", "isolation", "burnout", "stress", "hope", "failure"}):
            decomposed.extend(
                [
                    "person worried at desk bills",
                    "close up hands typing late night",
                    "office at night tired expression",
                    "stock market red screen",
                    "empty wallet close up",
                ]
            )
        decomposed.extend(
            self._broad_topic_subtopic_terms(
                concept_title=title,
                keywords=keywords,
                summary=summary,
                request_need=8,
            )[:6]
        )
        decomposed.extend(keywords[:4])
        decomposed.append(title)
        deduped: list[str] = []
        seen: set[str] = set()
        for item in decomposed:
            clean = " ".join(str(item or "").split()).strip().lower()
            if not clean or clean in seen:
                continue
            seen.add(clean)
            deduped.append(clean)
        return deduped[:20]

    def _recover_candidates_from_local_corpus(
        self,
        conn,
        material_id: str,
        concept_terms: list[str],
        context_terms: list[str],
        concept_embedding: np.ndarray | None,
        subject_tag: str | None,
        visual_spec: dict[str, list[str]],
        preferred_video_duration: str,
        fast_mode: bool,
        strict_topic_only: bool,
        existing_video_counts: dict[str, int],
        generated_video_counts: dict[str, int],
        max_segments_per_video: int,
        concept_title: str,
        page_hint: int = 1,
        root_topic_terms: list[str] | None = None,
        bootstrap_fallback: bool = False,
    ) -> list[dict[str, Any]]:
        # Fix Q: Search larger corpus; use FAISS pre-filter when embeddings available.
        # Prefer material-local history first, then fall back to global reel-backed videos
        # so common topics can bootstrap from the existing corpus when live search is thin.
        corpus_limit = 640 if fast_mode else 1200
        rows = fetch_all(
            conn,
            """
            SELECT DISTINCT
                v.id,
                v.title,
                v.channel_title,
                v.description,
                v.duration_sec,
                COALESCE(v.view_count, 0) AS view_count,
                v.is_creative_commons,
                v.created_at
            FROM videos v
            JOIN reels r ON r.video_id = v.id
            WHERE r.material_id = ?
            ORDER BY v.created_at DESC
            LIMIT ?
            """,
            (material_id, corpus_limit),
        )
        if not rows:
            rows = fetch_all(
                conn,
                """
                SELECT DISTINCT
                    v.id,
                    v.title,
                    v.channel_title,
                    v.description,
                    v.duration_sec,
                    COALESCE(v.view_count, 0) AS view_count,
                    v.is_creative_commons,
                    v.created_at
                FROM videos v
                WHERE EXISTS (
                    SELECT 1
                    FROM reels r
                    WHERE r.video_id = v.id
                )
                ORDER BY v.created_at DESC
                LIMIT ?
                """,
                (corpus_limit,),
            )
        if not rows:
            return []

        # Fix Q: Pre-rank with FAISS when embeddings available to avoid scoring the entire corpus
        if concept_embedding is not None and len(rows) > 100:
            try:
                from .vector_search import top_k_cosine
                metadata_texts = [
                    f"{str(r.get('title') or '')} {str(r.get('description') or '')[:200]}"
                    for r in rows
                ]
                metadata_embeddings = self.embedding_service.embed_texts(conn, metadata_texts)
                faiss_top = min(120 if fast_mode else 240, len(rows))
                ranked_indices = top_k_cosine(concept_embedding, metadata_embeddings, top_k=faiss_top)
                rows = [rows[idx] for idx, _score in ranked_indices]
            except Exception:
                pass  # Fall back to scoring all rows

        query_candidate = QueryCandidate(
            text=f"local cache {concept_title}".strip(),
            strategy="literal" if bootstrap_fallback else "recovery_adjacent",
            confidence=0.76 if bootstrap_fallback else 0.45,
            source_terms=[concept_title],
            weight=0.76 if bootstrap_fallback else 0.55,
            stage="high_precision" if bootstrap_fallback else "recovery",
            source_surface="local_bootstrap" if bootstrap_fallback else "local_cache",
        )

        candidates: list[dict[str, Any]] = []
        for row in rows:
            video_id = str(row.get("id") or "").strip()
            if not video_id:
                continue
            video_duration = int(row.get("duration_sec") or 0)
            video_segment_cap = self._video_segment_cap(
                video_duration_sec=video_duration,
                fast_mode=fast_mode,
                default_cap=max_segments_per_video,
                page_hint=page_hint,
            )
            existing_for_video = existing_video_counts.get(video_id, 0)
            generated_for_video = generated_video_counts.get(video_id, 0)
            if existing_for_video + generated_for_video >= video_segment_cap:
                continue
            video = {
                "id": video_id,
                "title": str(row.get("title") or ""),
                "channel_title": str(row.get("channel_title") or ""),
                "description": str(row.get("description") or ""),
                "duration_sec": video_duration,
                "view_count": int(row.get("view_count") or 0),
                "is_creative_commons": bool(row.get("is_creative_commons")),
                "published_at": "",
                "search_source": "local_bootstrap" if bootstrap_fallback else "local_cache",
                "query_strategy": "literal" if bootstrap_fallback else "recovery_adjacent",
                "query_stage": "high_precision" if bootstrap_fallback else "recovery",
                "search_query": "local_bootstrap" if bootstrap_fallback else "local_cache",
            }
            quick_signals = self._quick_candidate_metadata_gate(
                video=video,
                query_candidate=query_candidate,
                concept_terms=concept_terms,
                context_terms=context_terms,
                subject_tag=subject_tag,
                root_topic_terms=root_topic_terms,
                strict_topic_only=strict_topic_only,
                require_context=False,
                fast_mode=fast_mode,
            )
            if not bool(quick_signals.get("passes")):
                continue
            ranking = self._score_video_candidate(
                conn,
                video=video,
                query_candidate=query_candidate,
                concept_terms=concept_terms,
                context_terms=context_terms,
                concept_embedding=concept_embedding,
                subject_tag=subject_tag,
                root_topic_terms=root_topic_terms,
                strict_topic_only=strict_topic_only,
                visual_spec=visual_spec,
                preferred_video_duration=preferred_video_duration,
                stage_name="recovery",
                require_context=False,
                fast_mode=fast_mode,
                quick_signals=quick_signals,
            )
            discovery = float(ranking.get("discovery_score") or 0.0)
            text_relevance = dict(ranking.get("text_relevance") or {})
            query_alignment = float(((ranking.get("features") or {}).get("query_alignment") or 0.0))
            if not self._has_strong_topic_support(text_relevance, fast_mode=fast_mode):
                continue
            if discovery < 0.12 and query_alignment < 0.08:
                continue
            if strict_topic_only and not self._passes_selection_topic_guard(
                video=video,
                ranking=ranking,
                segment_relevance=text_relevance,
                transcript_ranking=None,
                has_transcript=False,
                fast_mode=fast_mode,
                strict_topic_only=True,
                subject_tag=subject_tag,
                root_topic_terms=root_topic_terms,
            ):
                continue
            candidates.append(
                {
                    "video": video,
                    "video_id": video_id,
                    "video_duration": int(video.get("duration_sec") or 0),
                    "video_relevance": ranking.get("text_relevance") or {},
                    "ranking": ranking,
                    "query_candidate": query_candidate,
                    "stage": "recovery",
                }
            )

        candidates.sort(key=lambda row: float((row.get("ranking") or {}).get("final_score") or 0.0), reverse=True)
        return candidates[: (18 if fast_mode else 36)]


    def _build_material_context_terms(
        self,
        concepts: list[dict[str, Any]],
        subject_tag: str | None,
        max_terms: int = 24,
    ) -> list[str]:
        scores: dict[str, float] = {}
        order: dict[str, int] = {}

        def add_terms(value: str, weight: float) -> None:
            for token in re.findall(r"[A-Za-z][A-Za-z0-9\-']*", value.lower()):
                if len(token) < 4:
                    continue
                if token in self.GENERIC_CONTEXT_TERMS:
                    continue
                if token not in order:
                    order[token] = len(order)
                scores[token] = scores.get(token, 0.0) + weight

        for concept in concepts:
            add_terms(str(concept.get("title") or ""), 2.0)
            add_terms(str(concept.get("summary") or ""), 0.7)
            keywords_json = str(concept.get("keywords_json") or "[]")
            try:
                keywords = json.loads(keywords_json)
            except json.JSONDecodeError:
                keywords = []
            if not isinstance(keywords, list):
                keywords = []
            for kw in keywords[:8]:
                add_terms(str(kw), 1.8 if " " in str(kw) else 1.2)

        if subject_tag:
            add_terms(subject_tag, 2.2)

        ranked = sorted(scores.items(), key=lambda item: (-item[1], order[item[0]]))
        return [term for term, _ in ranked[:max_terms]]

    def _bootstrap_topic_retrieval_concepts(
        self,
        *,
        conn,
        concepts: list[dict[str, Any]],
        subject_tag: str,
        material_id: str,
    ) -> list[dict[str, Any]]:
        subject_key = self._normalize_query_key(subject_tag)
        root_concept = next(
            (
                dict(concept)
                for concept in concepts
                if self._normalize_query_key(str(concept.get("title") or "")) == subject_key
            ),
            None,
        )
        if root_concept is None:
            first = dict(concepts[0]) if concepts else {}
            root_concept = {
                **first,
                "id": str(first.get("id") or uuid.uuid4()),
                "material_id": str(first.get("material_id") or material_id),
                "title": self._title_case_phrase(subject_tag),
                "summary": f"Core ideas, terminology, and intuition for {self._title_case_phrase(subject_tag)}.",
                "created_at": str(first.get("created_at") or now_iso()),
            }
        root_concept["keywords_json"] = dumps_json(
            self._bootstrap_topic_keywords(root_concept, subject_tag=subject_tag, conn=conn)
        )
        return [root_concept]

    def _bootstrap_topic_keywords(self, concept: dict[str, Any], *, subject_tag: str, conn=None) -> list[str]:
        expansion = self.topic_expansion_service.expand_topic(
            conn,
            topic=subject_tag,
            max_subtopics=4,
            max_aliases=4,
            max_related_terms=4,
        )
        search_terms = self.topic_expansion_service.build_topic_search_terms(
            topic=subject_tag,
            expansion=expansion,
            limit=6,
        )
        canonical_topic = self._clean_query_text(str(expansion.get("canonical_topic") or ""))
        companion_terms = [
            self._clean_query_text(str(term or ""))
            for term in (expansion.get("related_terms") or [])
            if self.topic_expansion_service._allows_unanchored_opaque_search_term(str(term or ""))
            and not self.topic_expansion_service._is_topic_anchor_candidate(
                topic=subject_tag,
                canonical_topic=canonical_topic,
                candidate=str(term or ""),
            )
        ]
        candidates = [
            self._clean_query_text(subject_tag),
            self._clean_query_text(str(concept.get("title") or "")),
            *companion_terms,
            *[self._clean_query_text(term) for term in search_terms],
        ]
        deduped: list[str] = []
        seen: set[str] = set()
        for value in candidates:
            normalized = self._normalize_query_key(value)
            if not value or not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(value.lower())
        return deduped or [self._clean_query_text(subject_tag).lower()]

    def _deep_topic_expansion(
        self,
        conn,
        *,
        material_id: str,
        subject_tag: str,
        generation_id: str | None,
    ) -> dict[str, Any]:
        base_expansion = self.topic_expansion_service.expand_topic(
            conn,
            topic=subject_tag,
            max_subtopics=12,
            max_aliases=8,
            max_related_terms=8,
        )
        observed_examples = self._topic_expansion_observed_examples(
            conn,
            material_id=material_id,
            generation_id=generation_id,
            limit=4,
        )
        ai_expansion = self._expand_topic_via_llm(
            conn,
            subject_tag=subject_tag,
            base_expansion=base_expansion,
            observed_examples=observed_examples,
        )
        return self._merge_topic_expansions(base_expansion, ai_expansion, subject_tag=subject_tag)

    def _topic_expansion_observed_examples(
        self,
        conn,
        *,
        material_id: str,
        generation_id: str | None,
        limit: int,
    ) -> list[dict[str, str]]:
        params: list[Any] = [material_id]
        generation_clause = ""
        if generation_id:
            generation_clause = " AND COALESCE(r.generation_id, '') = ?"
            params.append(str(generation_id))
        rows = fetch_all(
            conn,
            f"""
            SELECT
                COALESCE(v.title, '') AS title,
                COALESCE(r.transcript_snippet, '') AS snippet,
                MIN(COALESCE(r.created_at, '')) AS first_seen_at
            FROM reels r
            JOIN videos v ON v.id = r.video_id
            WHERE r.material_id = ?
              {generation_clause}
            GROUP BY COALESCE(v.title, ''), COALESCE(r.transcript_snippet, '')
            ORDER BY first_seen_at ASC, title ASC, snippet ASC
            LIMIT ?
            """,
            tuple([*params, int(limit)]),
        )
        examples: list[dict[str, str]] = []
        for row in rows:
            title = self._clean_query_text(str(row.get("title") or ""))
            snippet = self._clean_query_text(str(row.get("snippet") or ""))
            if not title and not snippet:
                continue
            examples.append({"title": title, "snippet": snippet})
        return examples

    def _expand_topic_via_llm(
        self,
        conn,
        *,
        subject_tag: str,
        base_expansion: dict[str, Any],
        observed_examples: list[dict[str, str]],
    ) -> dict[str, Any]:
        injected_client = getattr(self, "openai_client", None)
        if not self.llm_available and injected_client is None:
            return {}

        observed_block = "\n".join(
            f"- title: {example.get('title') or 'n/a'} | snippet: {example.get('snippet') or 'n/a'}"
            for example in observed_examples[:4]
        )
        base_terms = self._topic_expansion_terms(
            expansion=base_expansion,
            subject_tag=subject_tag,
            limit=12,
        )
        cache_payload = "|".join(
            [
                "topic_deep_expand_v1",
                self.chat_model,
                self._normalize_query_key(subject_tag),
                *[self._normalize_query_key(term) for term in base_terms[:8]],
                *[
                    self._normalize_query_key(f"{example.get('title') or ''} {example.get('snippet') or ''}")
                    for example in observed_examples[:4]
                ],
            ]
        )
        cache_key = f"topic_deep_expand:{hashlib.sha256(cache_payload.encode('utf-8')).hexdigest()}"
        cached = fetch_one(conn, "SELECT response_json FROM llm_cache WHERE cache_key = ?", (cache_key,))
        if cached:
            try:
                payload = json.loads(str(cached.get("response_json") or "{}"))
                if isinstance(payload, dict):
                    return payload
            except json.JSONDecodeError:
                pass

        system_prompt = (
            "You expand educational study topics into strict on-topic YouTube search terms. "
            "Return strict JSON only with keys aliases, subtopics, related_terms. "
            "Keep every term educational, concrete, and tightly anchored to the topic. "
            "Do not output generic filler like basics, overview, foundations, worked examples, or problem solving."
        )
        user_prompt = (
            f"Topic: {subject_tag}\n"
            f"Existing non-AI expansion terms: {', '.join(base_terms[:12]) or 'none'}\n"
            f"Observed starter reels:\n{observed_block or '- none yet'}\n"
            "Return JSON with:\n"
            "- aliases: 0-6 alternate names or near-synonyms for the same study topic\n"
            "- subtopics: 4-10 concrete subtopics that would find more educational videos\n"
            "- related_terms: 0-6 tightly related educational search phrases\n"
            "Rules:\n"
            "- stay strictly within the topic\n"
            "- prefer academically useful terms\n"
            "- avoid generic pedagogy words alone\n"
            "- no numbering, prose, or markdown"
        )
        try:
            if injected_client is not None:
                response = injected_client.chat.completions.create(
                    model=self.chat_model,
                    temperature=0.2,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                raw = response.choices[0].message.content or "{}"
            else:
                raw = llm_router.chat_completion(
                    system=system_prompt,
                    user=user_prompt,
                    temperature=0.2,
                    json_mode=True,
                ) or "{}"
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                return {}
            sanitized = self._sanitize_topic_expansion_payload(payload, subject_tag=subject_tag)
            upsert(
                conn,
                "llm_cache",
                {
                    "cache_key": cache_key,
                    "response_json": dumps_json(sanitized),
                    "created_at": now_iso(),
                },
                pk="cache_key",
            )
            return sanitized
        except Exception:
            return {}

    def _sanitize_topic_expansion_payload(self, payload: dict[str, Any], *, subject_tag: str) -> dict[str, Any]:
        return {
            "canonical_topic": self._clean_query_text(str(payload.get("canonical_topic") or subject_tag)) or subject_tag,
            "aliases": self._sanitize_topic_term_list(payload.get("aliases"), subject_tag=subject_tag, limit=8),
            "subtopics": self._sanitize_topic_term_list(payload.get("subtopics"), subject_tag=subject_tag, limit=12),
            "related_terms": self._sanitize_topic_term_list(payload.get("related_terms"), subject_tag=subject_tag, limit=8),
        }

    def _sanitize_topic_term_list(self, value: Any, *, subject_tag: str, limit: int) -> list[str]:
        if not isinstance(value, list):
            return []
        subject_key = self._normalize_query_key(subject_tag)
        generic = {
            "basics",
            "foundations",
            "introduction",
            "overview",
            "problem solving",
            "worked examples",
        }
        cleaned_terms: list[str] = []
        seen: set[str] = set()
        for item in value:
            cleaned = self._clean_query_text(str(item or ""))
            normalized = self._normalize_query_key(cleaned)
            if (
                not cleaned
                or not normalized
                or normalized == subject_key
                or normalized in seen
                or normalized in generic
            ):
                continue
            seen.add(normalized)
            cleaned_terms.append(cleaned)
            if len(cleaned_terms) >= limit:
                break
        return cleaned_terms

    def _merge_topic_expansions(
        self,
        base_expansion: dict[str, Any],
        ai_expansion: dict[str, Any],
        *,
        subject_tag: str,
    ) -> dict[str, Any]:
        merged = {
            "canonical_topic": str(ai_expansion.get("canonical_topic") or base_expansion.get("canonical_topic") or subject_tag),
            "aliases": [],
            "subtopics": [],
            "related_terms": [],
        }
        for key, limit in (("aliases", 8), ("subtopics", 12), ("related_terms", 8)):
            seen: set[str] = set()
            values: list[str] = []
            for source in (ai_expansion, base_expansion):
                for raw in (source.get(key) or []):
                    cleaned = self._clean_query_text(str(raw or ""))
                    normalized = self._normalize_query_key(cleaned)
                    if not cleaned or not normalized or normalized in seen:
                        continue
                    seen.add(normalized)
                    values.append(cleaned)
                    if len(values) >= limit:
                        break
                if len(values) >= limit:
                    break
            merged[key] = values
        return merged

    def _topic_root_anchor_terms(
        self,
        *,
        subject_tag: str,
        expansion: dict[str, Any] | None,
        concepts: list[dict[str, Any]] | None = None,
        limit: int = 8,
    ) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()

        def add_term(raw_value: str) -> None:
            cleaned = self._clean_query_text(str(raw_value or ""))
            normalized = self._normalize_query_key(cleaned)
            if not cleaned or not normalized or normalized in seen:
                return
            seen.add(normalized)
            ordered.append(cleaned)

        add_term(subject_tag)
        if expansion:
            add_term(str(expansion.get("canonical_topic") or ""))
            for raw_term in expansion.get("aliases") or []:
                add_term(str(raw_term or ""))
                if len(ordered) >= limit:
                    return ordered[:limit]

        for concept in concepts or []:
            title = self._clean_query_text(str((concept or {}).get("title") or ""))
            if not title:
                continue
            if (
                self._term_has_root_topic_anchor(title, root_topic_terms=ordered)
                and not self._is_low_signal_topic_concept(title, subject_tag=subject_tag)
            ):
                add_term(title)
            if len(ordered) >= limit:
                break
        return ordered[:limit]

    def _term_has_root_topic_anchor(self, value: str, *, root_topic_terms: list[str] | None) -> bool:
        normalized_value = self._normalize_query_key(value)
        if not normalized_value:
            return False
        value_tokens = set(normalized_value.split())
        lowered = f" {normalized_value} "
        for raw_term in root_topic_terms or []:
            normalized_term = self._normalize_query_key(raw_term)
            if not normalized_term:
                continue
            if normalized_value == normalized_term:
                return True
            if f" {normalized_term} " in lowered:
                return True
            term_tokens = normalized_term.split()
            if len(term_tokens) > 1 and set(term_tokens).issubset(value_tokens):
                return True
        return False

    def _build_topic_only_concepts_from_expansion(
        self,
        conn,
        *,
        material_id: str,
        subject_tag: str,
    ) -> list[dict[str, Any]]:
        expansion = self.topic_expansion_service.expand_topic(
            conn,
            topic=subject_tag,
            max_subtopics=10,
            max_aliases=8,
            max_related_terms=8,
        )
        root_topic_terms = self._topic_root_anchor_terms(subject_tag=subject_tag, expansion=expansion)
        expansion_terms = self._topic_expansion_terms(
            expansion=expansion,
            subject_tag=subject_tag,
            limit=8,
        )
        expansion_terms = [
            term for term in expansion_terms if self._term_has_root_topic_anchor(term, root_topic_terms=root_topic_terms)
        ]

        subject_title = self._title_case_phrase(subject_tag)
        created_concepts: list[dict[str, Any]] = []
        root_id = str(uuid.uuid4())
        root_keywords = self._topic_concept_keywords(
            subject_tag=subject_tag,
            primary_term=subject_tag,
            expansion_terms=expansion_terms,
        )
        root_row = {
            "id": root_id,
            "material_id": material_id,
            "title": subject_title,
            "keywords_json": dumps_json(root_keywords[:8]),
            "summary": f"Core ideas, terminology, and intuition for {subject_title}.",
            "embedding_json": None,
            "created_at": now_iso(),
        }
        upsert(conn, "concepts", root_row)
        created_concepts.append(root_row)

        for term in expansion_terms[:5]:
            concept_row = {
                "id": str(uuid.uuid4()),
                "material_id": material_id,
                "title": self._title_case_phrase(term),
                "keywords_json": dumps_json(
                    self._topic_concept_keywords(
                        subject_tag=subject_tag,
                        primary_term=term,
                        expansion_terms=expansion_terms,
                    )[:8]
                ),
                "summary": f"Key subtopic within {subject_title}: {term}.",
                "embedding_json": None,
                "created_at": now_iso(),
            }
            upsert(conn, "concepts", concept_row)
            created_concepts.append(concept_row)
        return created_concepts

    def _sync_topic_expansion_concepts(
        self,
        conn,
        *,
        material_id: str,
        concepts: list[dict[str, Any]],
        subject_tag: str,
        expansion: dict[str, Any],
    ) -> list[dict[str, Any]]:
        working = [dict(concept) for concept in concepts]
        root_topic_terms = self._topic_root_anchor_terms(
            subject_tag=subject_tag,
            expansion=expansion,
            concepts=working,
        )
        expansion_terms = self._topic_expansion_terms(
            expansion=expansion,
            subject_tag=subject_tag,
            limit=max(8, len(working) + 2),
        )
        keyword_expansion_terms = list(expansion_terms)
        expansion_terms = [
            term for term in expansion_terms if self._term_has_root_topic_anchor(term, root_topic_terms=root_topic_terms)
        ]

        subject_key = self._normalize_query_key(subject_tag)
        represented: set[str] = set()
        for concept in working:
            normalized = self._normalize_query_key(str(concept.get("title") or ""))
            if normalized:
                represented.add(normalized)
        root_index = next(
            (
                index
                for index, concept in enumerate(working)
                if self._normalize_query_key(str(concept.get("title") or "")) == subject_key
            ),
            -1,
        )
        subject_title = self._title_case_phrase(subject_tag)

        if root_index < 0:
            root_concept = dict(working[0]) if working else {"id": str(uuid.uuid4()), "created_at": now_iso()}
            root_concept["title"] = subject_title
            root_concept["keywords_json"] = dumps_json(
                self._topic_concept_keywords(
                    subject_tag=subject_tag,
                    primary_term=subject_tag,
                    expansion_terms=keyword_expansion_terms,
                )[:8]
            )
            root_concept["summary"] = f"Core ideas, terminology, and intuition for {subject_title}."
            root_concept["embedding_json"] = None
            root_concept["material_id"] = str(root_concept.get("material_id") or material_id)
            upsert(conn, "concepts", root_concept)
            if working:
                working[0] = root_concept
                root_index = 0
            else:
                working.append(root_concept)
                root_index = 0
            represented.add(subject_key)

        if root_index >= 0 and working:
            root_concept = dict(working[root_index])
            root_keywords = self._topic_concept_keywords(
                subject_tag=subject_tag,
                primary_term=subject_tag,
                expansion_terms=keyword_expansion_terms,
            )
            root_concept["keywords_json"] = dumps_json(root_keywords[:8])
            root_summary = str(root_concept.get("summary") or "").strip()
            if not root_summary or self._is_low_signal_topic_concept(root_summary, subject_tag=subject_tag):
                root_concept["summary"] = f"Core ideas, terminology, and intuition for {self._title_case_phrase(subject_tag)}."
            root_concept["material_id"] = str(root_concept.get("material_id") or material_id)
            root_concept["title"] = subject_title
            root_concept["embedding_json"] = None
            upsert(conn, "concepts", root_concept)
            working[root_index] = root_concept

        curated_terms = self._curated_topic_subtopic_terms(subject_tag)
        if curated_terms and root_index >= 0 and working:
            desired_terms = curated_terms[: max(1, min(7, len(curated_terms)))]
            curated_working = [dict(working[root_index])]
            exact_match_by_key = {
                self._normalize_query_key(str(concept.get("title") or "")): dict(concept)
                for index, concept in enumerate(working)
                if index != root_index and self._normalize_query_key(str(concept.get("title") or ""))
            }
            reusable_pool = [
                dict(concept)
                for index, concept in enumerate(working)
                if index != root_index
            ]
            used_ids: set[str] = set()

            for term in desired_terms:
                term_key = self._normalize_query_key(term)
                concept = dict(exact_match_by_key.get(term_key) or {})
                if concept and str(concept.get("id") or "") in used_ids:
                    concept = {}
                if not concept:
                    while reusable_pool:
                        candidate = dict(reusable_pool.pop(0))
                        candidate_id = str(candidate.get("id") or "")
                        if candidate_id and candidate_id in used_ids:
                            continue
                        concept = candidate
                        break
                if not concept:
                    concept = {"id": str(uuid.uuid4()), "created_at": now_iso()}
                concept["title"] = self._title_case_phrase(term)
                concept["keywords_json"] = dumps_json(
                    self._topic_concept_keywords(
                        subject_tag=subject_tag,
                        primary_term=term,
                        expansion_terms=desired_terms,
                    )[:8]
                )
                concept["summary"] = f"Key subtopic within {subject_title}: {term}."
                concept["embedding_json"] = None
                concept["material_id"] = str(concept.get("material_id") or material_id)
                upsert(conn, "concepts", concept)
                curated_working.append(concept)
                concept_id = str(concept.get("id") or "")
                if concept_id:
                    used_ids.add(concept_id)
            return curated_working

        replacement_terms = [term for term in expansion_terms if self._normalize_query_key(term) not in represented]
        filler_indexes = [
            index
            for index, concept in enumerate(working)
            if index != root_index
            and (
                self._is_low_signal_topic_concept(str(concept.get("title") or ""), subject_tag=subject_tag)
                or not self._term_has_root_topic_anchor(
                    str(concept.get("title") or ""),
                    root_topic_terms=root_topic_terms,
                )
            )
        ]

        for index in filler_indexes:
            if not replacement_terms:
                break
            term = replacement_terms.pop(0)
            concept = dict(working[index])
            concept["title"] = self._title_case_phrase(term)
            concept["keywords_json"] = dumps_json(
                self._topic_concept_keywords(
                    subject_tag=subject_tag,
                    primary_term=term,
                    expansion_terms=keyword_expansion_terms,
                )[:8]
            )
            concept["summary"] = f"Key subtopic within {subject_title}: {term}."
            concept["embedding_json"] = None
            concept["material_id"] = str(concept.get("material_id") or material_id)
            upsert(conn, "concepts", concept)
            working[index] = concept
            represented.add(self._normalize_query_key(term))

        while replacement_terms and len(working) < max(4, min(8, len(expansion_terms) + 1)):
            term = replacement_terms.pop(0)
            concept = {
                "id": str(uuid.uuid4()),
                "material_id": material_id,
                "title": self._title_case_phrase(term),
                "keywords_json": dumps_json(
                    self._topic_concept_keywords(
                        subject_tag=subject_tag,
                        primary_term=term,
                        expansion_terms=keyword_expansion_terms,
                    )[:8]
                ),
                "summary": f"Key subtopic within {subject_title}: {term}.",
                "embedding_json": None,
                "created_at": now_iso(),
            }
            upsert(conn, "concepts", concept)
            working.append(concept)
            represented.add(self._normalize_query_key(term))

        filtered_working = [
            concept
            for index, concept in enumerate(working)
            if index == root_index
            or self._term_has_root_topic_anchor(
                str(concept.get("title") or ""),
                root_topic_terms=root_topic_terms,
            )
            and not self._is_low_signal_topic_concept(
                str(concept.get("title") or ""),
                subject_tag=subject_tag,
            )
        ]
        if filtered_working:
            return filtered_working
        return working

    def _topic_expansion_terms(
        self,
        *,
        expansion: dict[str, Any],
        subject_tag: str,
        limit: int,
    ) -> list[str]:
        subject_key = self._normalize_query_key(subject_tag)
        ordered: list[str] = []
        seen: set[str] = set()

        def add_term(raw_value: str) -> bool:
            cleaned = self._clean_query_text(str(raw_value or ""))
            normalized = self._normalize_query_key(cleaned)
            allow_companion = self.topic_expansion_service._allows_unanchored_opaque_search_term(cleaned)
            if (
                not cleaned
                or not normalized
                or normalized == subject_key
                or normalized in seen
            ):
                return False
            if not allow_companion and self._is_low_signal_topic_concept(cleaned, subject_tag=subject_tag):
                return False
            seen.add(normalized)
            ordered.append(cleaned)
            return True

        for raw_value in self._curated_topic_subtopic_terms(subject_tag):
            add_term(raw_value)
            if len(ordered) >= limit:
                return ordered[:limit]

        candidate_terms = self.topic_expansion_service.build_topic_search_terms(
            topic=subject_tag,
            expansion=expansion,
            limit=max(limit + 2, 8),
        )
        for raw_value in candidate_terms:
            if not add_term(str(raw_value or "")):
                continue
            if len(ordered) >= limit:
                return ordered
        return ordered

    def _curated_topic_subtopic_terms(self, subject_tag: str) -> list[str]:
        subject_key = self._normalize_query_key(subject_tag)
        if not subject_key:
            return []

        ordered: list[str] = []
        seen: set[str] = set()

        def add_term(raw_value: str) -> None:
            cleaned = self._clean_query_text(str(raw_value or ""))
            normalized = self._normalize_query_key(cleaned)
            if not cleaned or not normalized or normalized == subject_key or normalized in seen:
                return
            seen.add(normalized)
            ordered.append(cleaned)

        for mapping in (
            getattr(self.topic_expansion_service, "STATIC_TOPIC_SUBTOPICS", {}),
            self.BROAD_TOPIC_SUBTOPICS,
        ):
            for raw_topic, topic_terms in mapping.items():
                if self._normalize_query_key(raw_topic) != subject_key:
                    continue
                for term in topic_terms:
                    add_term(str(term or ""))
                break

        for term in self._expand_controlled_synonyms([subject_tag], fast_mode=True):
            if len(self._normalize_query_key(term).split()) < 2:
                continue
            add_term(term)
        return ordered[:10]

    def _topic_concept_keywords(
        self,
        *,
        subject_tag: str,
        primary_term: str,
        expansion_terms: list[str],
    ) -> list[str]:
        candidates = [
            self._clean_query_text(primary_term),
            self._clean_query_text(subject_tag),
            *[self._clean_query_text(term) for term in expansion_terms[:5]],
            self._clean_query_text(f"{primary_term} explained"),
        ]
        deduped: list[str] = []
        seen: set[str] = set()
        for value in candidates:
            normalized = self._normalize_query_key(value)
            if not value or not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(value.lower())
        return deduped

    def _is_low_signal_topic_concept(self, value: str, *, subject_tag: str) -> bool:
        normalized = self._normalize_query_key(value)
        subject_key = self._normalize_query_key(subject_tag)
        if not normalized:
            return True
        opaque_topic = self.topic_expansion_service._is_opaque_single_token_topic(
            subject_tag,
            canonical_topic=subject_tag,
            likely_language=self.topic_expansion_service._looks_like_language_topic(subject_tag),
        )
        generic_phrases = {
            "topic",
            f"topic {subject_key}",
            f"{subject_key} foundations",
            f"{subject_key} worked examples",
            f"{subject_key} problem solving",
            f"introduction to {subject_key}",
            f"overview of {subject_key}",
            f"core concepts in {subject_key}",
            f"key terms in {subject_key}",
            f"major theories in {subject_key}",
            f"research methods in {subject_key}",
            f"classic studies in {subject_key}",
            f"worked examples for {subject_key}",
            f"applications of {subject_key}",
            f"{subject_key} from greek",
            f"{subject_key} also known as",
        }
        if normalized in generic_phrases:
            return True
        if " also known" in normalized:
            return True
        if " contained in " in normalized:
            return True
        if normalized.startswith("about "):
            return True
        if " from greek" in normalized:
            return True
        if subject_key and subject_key in normalized:
            tokens = normalized.split()
            if not normalized.startswith(subject_key) and len(tokens) > len(subject_key.split()) + 1:
                return True
            if opaque_topic and normalized.startswith(subject_key):
                suffix_tokens = tokens[len(subject_key.split()) :]
                if len(suffix_tokens) > 2:
                    return True
        if opaque_topic and subject_key and subject_key not in normalized:
            return True
        return normalized.startswith("topic ")

    def _title_case_phrase(self, value: str) -> str:
        parts = [segment for segment in self._clean_query_text(value).split(" ") if segment]
        return " ".join(part[:1].upper() + part[1:] for part in parts)

    def _context_terms_for_concept(self, concept_terms: list[str], material_context_terms: list[str]) -> list[str]:
        concept_tokens = normalize_terms(concept_terms)
        filtered = [term for term in material_context_terms if term not in concept_tokens]
        return filtered[:10]

    def _video_metadata_text(self, video: dict[str, Any]) -> str:
        pieces = [
            str(video.get("title") or "").strip(),
            str(video.get("description") or "").strip(),
            str(video.get("channel_title") or "").strip(),
        ]
        return " ".join(part for part in pieces if part).strip()

    def _score_text_relevance(
        self,
        conn,
        text: str,
        concept_terms: list[str],
        context_terms: list[str],
        concept_embedding: np.ndarray | None,
        subject_tag: str | None,
    ) -> dict[str, Any]:
        cleaned = " ".join(str(text or "").split()).strip()
        if not cleaned:
            return {
                "score": -1.0,
                "embedding_sim": 0.0,
                "concept_overlap": 0.0,
                "context_overlap": 0.0,
                "subject_overlap": 0.0,
                "concept_hits": [],
                "context_hits": [],
                "matched_terms": [],
                "off_topic_penalty": 0.0,
                "passes": False,
            }

        concept_hits = self._extract_matched_terms(cleaned, concept_terms, limit=6)
        context_hits = self._extract_matched_terms(cleaned, context_terms, limit=6)
        concept_overlap = lexical_overlap_score(cleaned, concept_terms)
        context_overlap = lexical_overlap_score(cleaned, context_terms) if context_terms else 0.0
        subject_overlap = lexical_overlap_score(cleaned, [subject_tag]) if subject_tag else 0.0

        embedding_sim = 0.0
        if concept_embedding is not None:
            text_embedding = self.embedding_service.embed_texts(conn, [cleaned])[0]
            embedding_sim = float(text_embedding @ concept_embedding.astype(np.float32))

        allowed_terms = [*concept_terms, *context_terms]
        if subject_tag:
            allowed_terms.append(subject_tag)
        off_topic_penalty = self._off_topic_penalty(
            cleaned,
            allowed_terms=allowed_terms,
            concept_terms=concept_terms,
            subject_tag=subject_tag,
        )

        score = (
            0.52 * max(0.0, embedding_sim)
            + 0.25 * concept_overlap
            + 0.17 * context_overlap
            + 0.08 * subject_overlap
            + 0.02 * min(4, len(concept_hits) + len(context_hits))
            - off_topic_penalty
        )
        score = float(max(-1.0, min(1.2, score)))

        merged_terms: list[str] = []
        for term in [*concept_hits, *context_hits]:
            if term not in merged_terms:
                merged_terms.append(term)

        return {
            "score": score,
            "embedding_sim": float(embedding_sim),
            "concept_overlap": float(concept_overlap),
            "context_overlap": float(context_overlap),
            "subject_overlap": float(subject_overlap),
            "concept_hits": concept_hits,
            "context_hits": context_hits,
            "matched_terms": merged_terms[:8],
            "off_topic_penalty": float(off_topic_penalty),
            "passes": False,
        }

    def _passes_relevance_gate(
        self,
        relevance: dict[str, Any],
        require_context: bool,
        fast_mode: bool,
    ) -> bool:
        concept_signal = float(relevance.get("concept_overlap") or 0.0) >= 0.05
        if not concept_signal:
            concept_signal = float(relevance.get("embedding_sim") or 0.0) >= (0.22 if fast_mode else 0.2)

        context_signal = True
        if require_context:
            context_signal = float(relevance.get("context_overlap") or 0.0) >= 0.04 or bool(relevance.get("context_hits"))

        off_topic_penalty = float(relevance.get("off_topic_penalty") or 0.0)
        if off_topic_penalty >= 0.24 and float(relevance.get("context_overlap") or 0.0) < 0.12:
            return False

        sparse_topic_signal = (
            len(relevance.get("concept_hits") or []) <= 1
            and float(relevance.get("context_overlap") or 0.0) < 0.04
            and float(relevance.get("subject_overlap") or 0.0) < 0.04
        )
        if sparse_topic_signal and off_topic_penalty > 0.0:
            min_sparse_embedding = 0.32 if fast_mode else 0.28
            min_sparse_score = 0.14 if fast_mode else 0.12
            if float(relevance.get("embedding_sim") or 0.0) < min_sparse_embedding:
                return False
            if float(relevance.get("score") or -1.0) < min_sparse_score:
                return False

        min_score = 0.1 if fast_mode else 0.08
        return concept_signal and context_signal and float(relevance.get("score") or -1.0) >= min_score

    def _passes_selection_topic_guard(
        self,
        *,
        video: dict[str, Any],
        ranking: dict[str, Any],
        segment_relevance: dict[str, Any],
        transcript_ranking: dict[str, Any] | None,
        has_transcript: bool,
        fast_mode: bool,
        strict_topic_only: bool,
        subject_tag: str | None = None,
        root_topic_terms: list[str] | None = None,
    ) -> bool:
        if self._is_hard_blocked_low_value_video(
            title=str(video.get("title") or ""),
            description=str(video.get("description") or ""),
            channel_title=str(video.get("channel_title") or ""),
            subject_tag=subject_tag,
        ):
            return False
        video_relevance = dict(ranking.get("text_relevance") or {})
        features = dict(ranking.get("features") or {})
        query_alignment = float(features.get("query_alignment") or 0.0)
        query_hits = list(features.get("query_alignment_hits") or [])
        root_topic_alignment = float(features.get("root_topic_alignment") or 0.0)
        root_topic_hits = list(features.get("root_topic_alignment_hits") or [])
        specific_concept_anchor = float(features.get("specific_concept_anchor") or 0.0)
        specific_concept_anchor_required = bool(features.get("specific_concept_anchor_required"))
        lexicon_noise = float(features.get("lexicon_noise") or 0.0)
        risky_source = str(video.get("search_source") or "") in self.RISKY_SEARCH_SOURCES
        root_anchor_required = (
            strict_topic_only
            and str(video.get("search_source") or "") in self.STRICT_TOPIC_ROOT_ANCHOR_SOURCES
        )
        low_quality_channel = self._infer_channel_tier(
            channel=str(video.get("channel_title") or "").lower(),
            title=str(video.get("title") or "").lower(),
        ) == "low_quality_compilation"
        transcript_passes = bool((transcript_ranking or {}).get("passes"))
        transcript_topic_score = float((transcript_ranking or {}).get("topic_score") or 0.0)
        strong_video = self._has_strong_topic_support(video_relevance, fast_mode=fast_mode)
        strong_segment = self._has_strong_topic_support(segment_relevance, fast_mode=fast_mode)
        if strict_topic_only:
            subject_support = max(
                float(video_relevance.get("subject_overlap") or 0.0),
                float(segment_relevance.get("subject_overlap") or 0.0),
            )
            direct_topic_support = (
                query_alignment >= (0.1 if fast_mode else 0.08)
                or root_topic_alignment >= (0.12 if fast_mode else 0.1)
                or specific_concept_anchor >= (0.16 if fast_mode else 0.12)
                or (query_hits and query_alignment >= (0.08 if fast_mode else 0.06))
                or subject_support >= 0.04
            )
            if not direct_topic_support and not transcript_passes:
                return False
            if float(video_relevance.get("off_topic_penalty") or 0.0) >= 0.12 and not transcript_passes:
                return False
            if root_anchor_required and root_topic_alignment < (0.12 if fast_mode else 0.1):
                return False
            if (
                specific_concept_anchor_required
                and specific_concept_anchor < (0.16 if fast_mode else 0.12)
                and root_topic_alignment < (0.12 if fast_mode else 0.1)
                and not transcript_passes
            ):
                return False
            if (
                lexicon_noise >= 0.26
                and not self._looks_like_language_subject(subject_tag)
                and not transcript_passes
            ):
                return False

        if has_transcript:
            if not strong_segment:
                return False
            if not (strong_video or transcript_passes or transcript_topic_score >= (0.16 if fast_mode else 0.13)):
                return False
            if risky_source and not transcript_passes and query_alignment < 0.12:
                return False
        else:
            if not strong_video:
                return False
            if float(video_relevance.get("score") or -1.0) < (0.18 if fast_mode else 0.15):
                return False
            if query_alignment < (0.08 if fast_mode else 0.06):
                return False

        if float(video_relevance.get("off_topic_penalty") or 0.0) >= 0.18 and query_alignment < 0.12 and not transcript_passes:
            return False
        if low_quality_channel and query_alignment < 0.12 and not transcript_passes:
            return False
        if risky_source and not query_hits and not transcript_passes:
            return False
        if root_anchor_required and not root_topic_hits:
            return False
        return True


    def _merge_relevance_context(self, video_relevance: dict[str, Any], segment_relevance: dict[str, Any]) -> dict[str, Any]:
        merged_terms: list[str] = []
        for term in [*segment_relevance.get("matched_terms", []), *video_relevance.get("matched_terms", [])]:
            term_clean = str(term).strip()
            if not term_clean or term_clean in merged_terms:
                continue
            merged_terms.append(term_clean)

        context_hits: list[str] = []
        for term in [*segment_relevance.get("context_hits", []), *video_relevance.get("context_hits", [])]:
            term_clean = str(term).strip()
            if not term_clean or term_clean in context_hits:
                continue
            context_hits.append(term_clean)

        score = 0.44 * float(video_relevance.get("score") or 0.0) + 0.56 * float(segment_relevance.get("score") or 0.0)
        reason_parts: list[str] = []
        if merged_terms:
            reason_parts.append(f"Matched terms: {', '.join(merged_terms[:4])}")
        if context_hits:
            reason_parts.append(f"Material context: {', '.join(context_hits[:3])}")
        if not reason_parts:
            reason_parts.append("Matched semantically to the uploaded material")

        return {
            "score": float(score),
            "matched_terms": merged_terms[:8],
            "reason": ". ".join(reason_parts) + ".",
            "passes": bool(video_relevance.get("passes", True)) and bool(segment_relevance.get("passes", True)),
        }

    def _extract_matched_terms(self, text: str, terms: list[str], limit: int = 8) -> list[str]:
        text_lower = f" {text.lower()} "
        text_tokens = normalize_terms([text])
        hits: list[str] = []
        seen: set[str] = set()

        for raw_term in terms:
            term = " ".join(str(raw_term or "").lower().split()).strip()
            if not term:
                continue
            if " " in term and f" {term} " in text_lower:
                if term not in seen:
                    seen.add(term)
                    hits.append(term)
                if len(hits) >= limit:
                    break
                continue
            for token in normalize_terms([term]):
                if token in text_tokens and token not in seen:
                    seen.add(token)
                    hits.append(token)
                    if len(hits) >= limit:
                        break
            if len(hits) >= limit:
                break
        return hits

    def _off_topic_penalty(
        self,
        text: str,
        allowed_terms: list[str],
        concept_terms: list[str] | None = None,
        subject_tag: str | None = None,
    ) -> float:
        lowered = " ".join(text.lower().split())
        allowed_tokens = normalize_terms(allowed_terms)
        text_tokens = normalize_terms([lowered])
        penalty = 0.0
        concept_tokens = normalize_terms(concept_terms or [])
        subject_tokens = normalize_terms([subject_tag]) if subject_tag else set()
        topic_tokens = allowed_tokens.union(concept_tokens).union(subject_tokens)

        for phrase, weight in self.OFF_TOPIC_PHRASES.items():
            phrase_tokens = normalize_terms([phrase])
            if phrase in lowered and phrase_tokens.isdisjoint(allowed_tokens):
                penalty += weight

        for token in self.OFF_TOPIC_TOKENS:
            if token not in text_tokens:
                continue
            if token in allowed_tokens:
                continue
            if any(token.startswith(allowed) or allowed.startswith(token) for allowed in allowed_tokens):
                continue
            penalty += 0.035

        if topic_tokens.intersection(self.MATH_TOPIC_TOKENS):
            if text_tokens.intersection(self.DENTAL_CONFLICT_TOKENS):
                penalty += 0.34
            if (
                text_tokens.intersection(self.ENTERTAINMENT_CONFLICT_TOKENS)
                and text_tokens.isdisjoint(self.EDUCATIONAL_CUE_TERMS)
            ):
                penalty += 0.18
        elif (
            text_tokens.intersection(self.ENTERTAINMENT_CONFLICT_TOKENS)
            and text_tokens.isdisjoint(self.EDUCATIONAL_CUE_TERMS)
        ):
            penalty += 0.08

        return min(0.42, penalty)

    def _clip_key(self, video_id: str, t_start: float, t_end: float) -> str:
        start = Decimal(str(float(t_start))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        end = Decimal(str(float(t_end))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        return f"{video_id}:{start}:{end}"

    def _order_concepts(self, conn, material_id: str, concepts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        concept_counts = {
            row["concept_id"]: int(row["reel_count"])
            for row in fetch_all(
                conn,
                "SELECT concept_id, COUNT(*) AS reel_count FROM reels WHERE material_id = ? GROUP BY concept_id",
                (material_id,),
            )
        }
        concept_feedback = {
            row["concept_id"]: {
                "helpful": float(row["helpful_votes"]),
                "confusing": float(row["confusing_votes"]),
                "avg_rating": float(row["avg_rating"] or 3.0),
            }
            for row in fetch_all(
                conn,
                """
                SELECT
                    r.concept_id,
                    COALESCE(SUM(f.helpful), 0) AS helpful_votes,
                    COALESCE(SUM(f.confusing), 0) AS confusing_votes,
                    COALESCE(AVG(f.rating), 3.0) AS avg_rating
                FROM reels r
                LEFT JOIN reel_feedback f ON f.reel_id = r.id
                WHERE r.material_id = ?
                GROUP BY r.concept_id
                """,
                (material_id,),
            )
        }

        def concept_key(concept: dict[str, Any]) -> tuple[float, int, str]:
            feedback = concept_feedback.get(concept["id"], {"helpful": 0.0, "confusing": 0.0, "avg_rating": 3.0})
            mastery = self._concept_mastery(
                helpful=feedback["helpful"],
                confusing=feedback["confusing"],
                avg_rating=feedback["avg_rating"],
            )
            reel_count = concept_counts.get(concept["id"], 0)
            created = concept.get("created_at") or ""
            return (mastery, reel_count, created)

        return sorted(concepts, key=concept_key)

    def _concept_mastery(self, helpful: float, confusing: float, avg_rating: float) -> float:
        return 0.25 * helpful - 0.35 * confusing + 0.15 * (avg_rating - 3.0)

    LEVEL_FEEDBACK_WINDOW = 20
    LEVEL_FEEDBACK_MIN_ROWS = 5

    def update_level_adjustment(self, conn, material_id: str) -> float:
        """Recompute the material's level drift from its most recent feedback.

        signal = 0.25*helpful_rate - 0.35*confusing_rate + 0.15*(avg_rating-3)/2
        (same shape as _concept_mastery), clamped to ±ADJUSTMENT_BOUND.
        Fewer than LEVEL_FEEDBACK_MIN_ROWS rows -> 0 (cold-start gate)."""
        from .knowledge_level import ADJUSTMENT_BOUND

        rows = fetch_all(
            conn,
            """
            SELECT f.helpful, f.confusing, f.rating
            FROM reel_feedback f
            JOIN reels r ON r.id = f.reel_id
            WHERE r.material_id = ?
            ORDER BY f.created_at DESC
            LIMIT ?
            """,
            (material_id, self.LEVEL_FEEDBACK_WINDOW),
        )
        if len(rows) < self.LEVEL_FEEDBACK_MIN_ROWS:
            adjustment = 0.0
        else:
            n = len(rows)
            helpful_rate = sum(1 for r in rows if int(r["helpful"] or 0) > 0) / n
            confusing_rate = sum(1 for r in rows if int(r["confusing"] or 0) > 0) / n
            ratings = [float(r["rating"]) for r in rows if r["rating"] is not None]
            avg_rating = (sum(ratings) / len(ratings)) if ratings else 3.0
            signal = 0.25 * helpful_rate - 0.35 * confusing_rate + 0.15 * (avg_rating - 3.0) / 2.0
            adjustment = max(-ADJUSTMENT_BOUND, min(ADJUSTMENT_BOUND, signal))
        execute_modify(
            conn,
            "UPDATE materials SET level_adjustment = ? WHERE id = ?",
            (adjustment, material_id),
        )
        return adjustment






    def _is_vague_concept(self, title: str, keywords: list[str], summary: str) -> bool:
        title_terms = normalize_terms([title])
        all_terms = normalize_terms([title, *keywords, summary])
        generic_terms = {
            "basics",
            "basic",
            "introduction",
            "intro",
            "overview",
            "beginner",
            "beginners",
            "tutorial",
            "guide",
            "fundamentals",
            "learn",
            "learning",
            "concept",
            "concepts",
        }
        if title_terms and title_terms.issubset(generic_terms):
            return True
        if len(title_terms) <= 1 and len(all_terms) <= 7:
            return True
        if len(summary.strip()) < 50 and len(keywords) < 3:
            return True
        return False





    def _build_query_focused_transcript_snippet(
        self,
        transcript: list[dict[str, Any]] | None,
        *,
        clip_start: float,
        clip_end: float,
        query_text: str | None = None,
        fallback_text: str | None = None,
        max_chars: int = 700,
    ) -> str:
        if transcript:
            try:
                from ..ingestion.models import IngestTranscriptCue
                from ..ingestion.segment import snippet_for_window
            except Exception:
                snippet_for_window = None  # type: ignore[assignment]
                IngestTranscriptCue = None  # type: ignore[assignment]
            if snippet_for_window is not None and IngestTranscriptCue is not None:
                cues: list[Any] = []
                for entry in transcript:
                    text = str(entry.get("text") or "").replace("\n", " ").strip()
                    if not text:
                        continue
                    try:
                        start = float(entry.get("start") or 0.0)
                    except (TypeError, ValueError):
                        continue
                    duration_value = entry.get("duration")
                    try:
                        duration = float(duration_value) if duration_value is not None else 0.0
                    except (TypeError, ValueError):
                        duration = 0.0
                    if duration <= 0:
                        duration = 1.5
                    cues.append(
                        IngestTranscriptCue(
                            start=start,
                            end=start + duration,
                            text=text,
                            words=[],
                            word_source="legacy",
                        )
                    )
                if cues:
                    try:
                        snippet = snippet_for_window(
                            cues,
                            float(clip_start),
                            float(clip_end),
                            max_chars=max_chars,
                            focus_query=query_text,
                        )
                    except Exception:
                        snippet = ""
                    if snippet:
                        return snippet

        fallback = str(fallback_text or "").strip()
        if len(fallback) > max_chars:
            fallback = fallback[:max_chars].rstrip() + "…"
        return fallback

    def _sentence_spans_in_clip(
        self,
        transcript: list[dict[str, Any]],
        *,
        clip_start: float,
        clip_end: float,
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for entry in transcript:
            text = str(entry.get("text") or "").replace("\n", " ").strip()
            if not text:
                continue
            try:
                start = float(entry.get("start") or 0.0)
            except (TypeError, ValueError):
                continue
            duration_value = entry.get("duration")
            try:
                duration = float(duration_value) if duration_value is not None else 0.0
            except (TypeError, ValueError):
                duration = 0.0
            if duration <= 0:
                duration = 1.5
            end = start + duration
            if end <= clip_start or start >= clip_end:
                continue
            entries.append({"start": start, "end": end, "text": text})

        if not entries:
            return []

        punct_ratio = self._terminal_punct_ratio(entries)
        if punct_ratio < _PUNCT_RESTORE_THRESHOLD:
            entries = self._auto_punctuate_entries(entries)

        use_pause_boundaries = not self._transcript_has_terminal_punct(entries)
        pause_breaks: set[int] = set()
        if use_pause_boundaries:
            pause_breaks = self._cue_pause_breaks(entries, min_pause_sec=0.6)
            if not pause_breaks:
                pause_breaks = self._cue_pause_breaks(entries, min_pause_sec=0.3)
            if not pause_breaks:
                pause_breaks = set(range(1, len(entries)))

        spans: list[dict[str, Any]] = []
        sentence_start: float | None = None
        sentence_end: float | None = None
        pieces: list[str] = []
        for idx, entry in enumerate(entries):
            start = max(float(clip_start), float(entry["start"]))
            end = min(float(clip_end), float(entry["end"]))
            if end <= start:
                continue
            text = str(entry["text"]).strip()
            if not text:
                continue
            if sentence_start is None:
                sentence_start = start
            sentence_end = end
            pieces.append(text)
            ends_sentence = (
                ((idx + 1) in pause_breaks)
                if use_pause_boundaries
                else self._is_sentence_end(text)
            )
            if ends_sentence:
                spans.append(
                    {
                        "t_start": sentence_start,
                        "t_end": sentence_end,
                        "text": " ".join(pieces).strip(),
                    }
                )
                sentence_start = None
                sentence_end = None
                pieces = []

        if sentence_start is not None and sentence_end is not None and pieces:
            spans.append(
                {
                    "t_start": sentence_start,
                    "t_end": sentence_end,
                    "text": " ".join(pieces).strip(),
                }
            )
        return spans

    def _trim_structural_edges_from_clip(
        self,
        transcript: list[dict[str, Any]],
        *,
        clip_start: float,
        clip_end: float,
        video_duration_sec: int,
        min_len: int,
    ) -> tuple[float, float] | None:
        spans = self._sentence_spans_in_clip(
            transcript,
            clip_start=float(clip_start),
            clip_end=float(clip_end),
        )
        if not spans:
            return (float(clip_start), float(clip_end))

        video_duration = float(video_duration_sec) if video_duration_sec > 0 else None
        start_idx = 0
        end_idx = len(spans) - 1

        while start_idx <= end_idx:
            label = classify_passage(
                str(spans[start_idx]["text"] or ""),
                t_start=float(spans[start_idx]["t_start"] or 0.0),
                video_duration=video_duration,
            )
            if label.name not in _STRUCTURAL_EDGE_TRIM_LABELS:
                break
            start_idx += 1

        while end_idx >= start_idx:
            label = classify_passage(
                str(spans[end_idx]["text"] or ""),
                t_start=float(spans[end_idx]["t_start"] or 0.0),
                video_duration=video_duration,
            )
            if label.name not in _STRUCTURAL_EDGE_TRIM_LABELS:
                break
            end_idx -= 1

        if start_idx > end_idx:
            logger.info(
                "structural_trim dropped clip [%.2f, %.2f]: all %d sentence spans matched trim labels",
                float(clip_start),
                float(clip_end),
                len(spans),
            )
            return None

        start_trimmed = start_idx > 0
        end_trimmed = end_idx < (len(spans) - 1)
        new_start = max(0.0, float(clip_start))
        new_end = float(clip_end)
        if start_trimmed:
            new_start = max(0.0, float(spans[start_idx]["t_start"] or clip_start))
        if end_trimmed:
            new_end = float(spans[end_idx]["t_end"] or clip_end)
        if video_duration_sec > 0:
            new_end = min(new_end, float(video_duration_sec))
        if new_end - new_start < float(min_len):
            logger.info(
                "structural_trim dropped clip [%.2f, %.2f]: post-trim window %.2f-%.2f shorter than min_len=%d",
                float(clip_start),
                float(clip_end),
                new_start,
                new_end,
                int(min_len),
            )
            return None
        return (round(new_start, 2), round(new_end, 2))


    def _create_reel(
        self,
        conn,
        material_id: str,
        concept: dict[str, Any],
        video: dict[str, Any],
        segment: SegmentMatch,
        clip_window: tuple[int, int] | None = None,
        transcript: list[dict[str, Any]] | None = None,
        relevance_context: dict[str, Any] | None = None,
        fast_mode: bool = False,
        target_clip_duration_sec: int = DEFAULT_TARGET_CLIP_DURATION_SEC,
        target_clip_duration_min_sec: int | None = None,
        target_clip_duration_max_sec: int | None = None,
        generation_id: str | None = None,
        finalized_clip_window: bool = False,
    ) -> dict[str, Any] | None:
        reel_id = str(uuid.uuid4())
        clip_min_len, clip_max_len, _ = self._resolve_clip_duration_bounds(
            target_clip_duration_sec=target_clip_duration_sec,
            target_clip_duration_min_sec=target_clip_duration_min_sec,
            target_clip_duration_max_sec=target_clip_duration_max_sec,
        )
        normalized_clip_window: tuple[float, float] | None
        if finalized_clip_window and clip_window is not None:
            try:
                raw_start = float(clip_window[0])
                raw_end = float(clip_window[1])
            except (TypeError, ValueError):
                return None
            video_duration = float(video.get("duration_sec") or 0)
            start_f = max(0.0, raw_start)
            end_f = raw_end
            if video_duration > 0:
                end_f = min(video_duration, end_f)
            duration = end_f - start_f
            if duration <= 0:
                return None
            if duration < max(1.0, float(clip_min_len) - 0.5):
                return None
            if clip_max_len > 0 and duration > float(clip_max_len) + 8.0:
                return None
            normalized_clip_window = (round(start_f, 3), round(end_f, 3))
        elif clip_window is None:
            normalized_clip_window = self._normalize_clip_window(
                segment.t_start,
                segment.t_end,
                int(video.get("duration_sec") or 0),
                min_len=clip_min_len,
                max_len=clip_max_len,
            )
        else:
            normalized_clip_window = self._normalize_clip_window(
                clip_window[0],
                clip_window[1],
                int(video.get("duration_sec") or 0),
                min_len=clip_min_len,
                max_len=clip_max_len,
            )
        if not normalized_clip_window:
            return None
        start_sec, end_sec = normalized_clip_window
        video_duration_sec = int(video.get("duration_sec") or 0)
        if transcript and not finalized_clip_window:
            trimmed_window = self._trim_structural_edges_from_clip(
                transcript,
                clip_start=float(start_sec),
                clip_end=float(end_sec),
                video_duration_sec=video_duration_sec,
                min_len=clip_min_len,
            )
            if trimmed_window is None:
                return None
            start_sec, end_sec = trimmed_window
        query_text = str((relevance_context or {}).get("query_text") or "").strip()
        transcript_snippet = self._build_query_focused_transcript_snippet(
            transcript,
            clip_start=float(start_sec),
            clip_end=float(end_sec),
            query_text=query_text or None,
            fallback_text=segment.text,
        )
        video_id = video["id"]
        provider = str(video.get("provider") or "youtube").strip().lower() or "youtube"
        if provider == "youtube":
            url = (
                f"https://www.youtube.com/embed/{video_id}?start={start_sec}&end={end_sec}"
                f"&playlist={video_id}&autoplay=1&mute=1&playsinline=1"
                "&loop=1&controls=1&modestbranding=1&iv_load_policy=3&rel=0"
            )
        else:
            # Non-YouTube providers (Dailymotion / Vimeo / Bilibili / TikTok /
            # Twitch) bring their own embed URL from ProviderCandidate →
            # _candidate_to_row → videos row. The client's detectVideoProvider
            # keys off this URL's host to pick the right renderer (YouTube
            # iframe API vs plain <iframe>). Without this branch every reel
            # would get a YouTube embed with a non-YouTube video_id and fail
            # to load entirely — the reason the feed was empty.
            url = str(video.get("playback_url") or video.get("video_url") or "").strip()
        takeaways = build_takeaways(concept, transcript_snippet or segment.text)
        captions = self._build_caption_cues(
            transcript=transcript or [],
            clip_start=float(start_sec),
            clip_end=float(end_sec),
            fallback_text=transcript_snippet or segment.text,
        )
        video_title = str(video.get("title") or "").strip()
        video_description = self._clean_video_description(str(video.get("description") or ""))
        ai_summary = self._brief_ai_summary(
            conn,
            video_id=video_id,
            concept_title=str(concept.get("title") or ""),
            video_title=video_title,
            video_description=video_description,
            transcript_snippet=transcript_snippet,
            takeaways=takeaways,
            fast_mode=fast_mode,
        )
        relevance_score = float((relevance_context or {}).get("score") or segment.score)
        matched_terms = [
            str(term).strip()
            for term in (relevance_context or {}).get("matched_terms", [])
            if str(term).strip()
        ][:8]
        relevance_reason = str((relevance_context or {}).get("reason") or "").strip()

        try:
            upsert(
                conn,
                "reels",
                {
                    "id": reel_id,
                    "generation_id": generation_id,
                    "material_id": material_id,
                    "concept_id": concept["id"],
                    "video_id": video_id,
                    "video_url": url,
                    "t_start": float(start_sec),
                    "t_end": float(end_sec),
                    "transcript_snippet": transcript_snippet,
                    "takeaways_json": dumps_json(takeaways),
                    "base_score": float(segment.score),
                    "created_at": now_iso(),
                },
            )
        except (sqlite3.IntegrityError, DatabaseIntegrityError):
            # DB-level uniqueness guard: skip duplicates safely if concurrent generation races.
            return None

        return {
            "reel_id": reel_id,
            "material_id": material_id,
            "concept_id": concept["id"],
            "concept_title": concept["title"],
            "video_title": video_title,
            "video_description": video_description,
            "channel_name": str(video.get("channel_title") or "").strip(),
            "ai_summary": ai_summary,
            "video_url": url,
            "t_start": float(start_sec),
            "t_end": float(end_sec),
            "transcript_snippet": transcript_snippet,
            "takeaways": takeaways,
            "captions": captions,
            "score": float(segment.score),
            "relevance_score": relevance_score,
            "discovery_score": float((relevance_context or {}).get("discovery_score") or 0.0),
            "clipability_score": float((relevance_context or {}).get("clipability_score") or 0.0),
            "query_strategy": str((relevance_context or {}).get("query_strategy") or ""),
            "retrieval_stage": str((relevance_context or {}).get("retrieval_stage") or ""),
            "source_surface": str((relevance_context or {}).get("source_surface") or ""),
            "matched_terms": matched_terms,
            "relevance_reason": relevance_reason,
            "video_duration_sec": video_duration_sec,
            "clip_duration_sec": float(max(0.0, end_sec - start_sec)),
        }

    def _clean_video_description(self, description: str, max_chars: int = 7000) -> str:
        cleaned = description.strip()
        if not cleaned:
            return ""
        cleaned = re.sub(r"\r\n?", "\n", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned[:max_chars]

    def _fallback_ai_summary(
        self,
        *,
        concept_title: str,
        video_title: str,
        video_description: str,
        transcript_snippet: str,
        takeaways: list[str],
    ) -> str:
        takeaway_text = " ".join(t.strip() for t in takeaways if t.strip())
        candidates = [transcript_snippet.strip(), takeaway_text.strip(), video_description.strip()]
        source = next((c for c in candidates if c), "")
        if not source:
            return f"Brief overview of {concept_title or video_title or 'this reel'}."

        compact = " ".join(source.split())
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", compact) if s.strip()]
        if not sentences:
            summary = compact
        elif len(sentences) == 1:
            summary = sentences[0]
        else:
            summary = f"{sentences[0]} {sentences[1]}"
        summary = summary[:320].strip()
        if summary and summary[-1] not in ".!?":
            summary = f"{summary}."
        return summary

    def _brief_ai_summary(
        self,
        conn,
        *,
        video_id: str,
        concept_title: str,
        video_title: str,
        video_description: str,
        transcript_snippet: str,
        takeaways: list[str],
        fast_mode: bool = False,
    ) -> str:
        fallback = self._fallback_ai_summary(
            concept_title=concept_title,
            video_title=video_title,
            video_description=video_description,
            transcript_snippet=transcript_snippet,
            takeaways=takeaways,
        )
        if fast_mode:
            return fallback

        cache_payload = "|".join(
            [
                self.chat_model,
                video_id,
                concept_title[:120],
                video_title[:200],
                video_description[:1200],
                transcript_snippet[:1200],
                " ".join(takeaways[:4]),
            ]
        )
        cache_key = f"reel_ai_summary:{hashlib.sha256(cache_payload.encode('utf-8')).hexdigest()}"
        cached = fetch_one(conn, "SELECT response_json FROM llm_cache WHERE cache_key = ?", (cache_key,))
        if cached:
            try:
                payload = json.loads(cached["response_json"])
                cached_summary = str(payload.get("summary") or "").strip()
                if cached_summary:
                    return cached_summary
            except (TypeError, json.JSONDecodeError):
                pass

        summary = fallback
        if self.llm_available:
            prompt = (
                "Write a brief study summary of this video clip in 1-2 sentences.\n"
                "Keep it concrete and under 220 characters.\n"
                "Do not add markdown or bullet points.\n\n"
                f"Concept: {concept_title or 'General topic'}\n"
                f"Video title: {video_title or 'Unknown'}\n"
                f"Video description: {video_description[:1500] or 'N/A'}\n"
                f"Clip transcript: {transcript_snippet[:1500] or 'N/A'}\n"
                f"Takeaways: {'; '.join(takeaways[:4]) or 'N/A'}"
            )
            try:
                generated = llm_router.chat_completion(
                    system="You write concise educational summaries.",
                    user=prompt,
                    temperature=0.2,
                ) or ""
                generated = generated.strip()
                if generated:
                    compact = " ".join(generated.split())
                    summary = compact[:320].strip() or fallback
            except Exception:
                summary = fallback

        upsert(
            conn,
            "llm_cache",
            {
                "cache_key": cache_key,
                "response_json": dumps_json({"summary": summary}),
                "created_at": now_iso(),
            },
            pk="cache_key",
        )
        return summary

    def _normalize_clip_window(
        self,
        t_start: float,
        t_end: float,
        video_duration_sec: int,
        min_len: int = 15,
        max_len: int = 60,
        allow_exceed_max: bool = False,
        allow_below_min: bool = False,
    ) -> tuple[float, float] | None:
        return _normalize_clip_window_fn(
            t_start,
            t_end,
            video_duration_sec,
            min_len=min_len,
            max_len=max_len,
            allow_exceed_max=allow_exceed_max,
            allow_below_min=allow_below_min,
        )

    def _should_use_full_short_clip(
        self,
        prefer_short_query: bool,
        video_duration_sec: int,
        clip_min_len: int,
        clip_max_len: int,
    ) -> bool:
        if not prefer_short_query:
            return False
        if video_duration_sec <= 0:
            return False
        if clip_max_len > 0 and video_duration_sec > clip_max_len:
            return False
        if clip_min_len > 0 and video_duration_sec < max(8, int(round(clip_min_len * 0.55))):
            return False
        # Shorts are currently up to ~3 minutes; keep full playback for these.
        return video_duration_sec <= 185




    def _is_sentence_end(self, text: str) -> bool:
        cleaned = text.strip()
        if not cleaned:
            return False
        return bool(re.search(r"[.!?…][\"'\)\]]*$", cleaned))

    # Lazy-loaded punctuation restoration model (loaded once on first use).
    # Warmed eagerly at app startup via `warm_punct_pipeline` so the first
    # user-facing search doesn't pay the 3-5s model-load cost.
    _punct_pipeline = None

    @classmethod
    def _get_punct_pipeline(cls):
        """Return the punctuation-restoration NER pipeline, loading on first call."""
        if cls._punct_pipeline is None:
            try:
                from transformers import pipeline as hf_pipeline, AutoModelForTokenClassification, AutoTokenizer
                model_name = "oliverguhr/fullstop-punctuation-multilang-large"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForTokenClassification.from_pretrained(model_name)
                cls._punct_pipeline = hf_pipeline(
                    "ner", model=model, tokenizer=tokenizer,
                    aggregation_strategy="simple",
                )
            except Exception:
                logger.warning("Failed to load punctuation model; auto-captions will use pause-based boundaries")
                cls._punct_pipeline = False  # sentinel — don't retry
        return cls._punct_pipeline if cls._punct_pipeline is not False else None

    @classmethod
    def warm_punct_pipeline(cls) -> bool:
        """Eagerly load the punctuation-restoration model at app startup so
        the first user request doesn't pay the model-load cost. Returns True
        on success, False on load failure (a single failure flips the
        sentinel so subsequent ingests skip quietly)."""
        pipe = cls._get_punct_pipeline()
        return pipe is not None

    @classmethod
    def punct_pipeline_loaded(cls) -> bool:
        """True iff the punctuation pipeline is live (loaded and non-sentinel)."""
        return cls._punct_pipeline not in (None, False)

    def _auto_punctuate_entries(
        self,
        entries: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Add punctuation to unpunctuated transcript entries in-place.

        Concatenates all cue text, runs a BERT-based punctuation-restoration
        model, then maps the restored punctuation back to each cue. This
        converts YouTube auto-captions like ``"so lets talk about"`` into
        ``"So lets talk about."`` so that downstream sentence-boundary
        detection works correctly.

        Falls back to the original entries if the model is unavailable.
        """
        pipe = self._get_punct_pipeline()
        if pipe is None:
            return entries

        # Concatenate cue texts with a single space, tracking char→cue mapping
        parts: list[str] = []
        cue_char_ranges: list[tuple[int, int]] = []  # (start_char, end_char) per cue
        offset = 0
        for entry in entries:
            text = str(entry.get("text") or "").strip()
            if not text:
                cue_char_ranges.append((offset, offset))
                continue
            if offset > 0:
                offset += 1  # space separator
            start_c = offset
            offset += len(text)
            cue_char_ranges.append((start_c, offset))
            parts.append(text)

        full_text = " ".join(parts)
        if not full_text.strip():
            return entries

        try:
            results = pipe(full_text)
        except Exception:
            logger.debug("Punctuation model inference failed; skipping auto-punctuate")
            return entries

        # Build a set of character positions where punctuation should be inserted
        # result: {char_position: punct_char}
        punct_insertions: dict[int, str] = {}
        for r in results:
            label = r.get("entity_group", "0")
            if label in (".", "?", "!"):
                punct_insertions[int(r["end"])] = label
            elif label == ",":
                punct_insertions[int(r["end"])] = ","

        # Map insertions back to cues: for each cue, find punct that falls
        # within or right after its character range
        new_entries = []
        for i, entry in enumerate(entries):
            text = str(entry.get("text") or "").strip()
            if not text:
                new_entries.append(entry)
                continue
            c_start, c_end = cue_char_ranges[i]
            # Find the last punctuation mark at or near this cue's end
            best_punct = None
            for pos, p in punct_insertions.items():
                if c_start <= pos <= c_end + 1:
                    best_punct = (pos, p)
            if best_punct is not None:
                # Append punct to end of cue text if not already there
                if text and text[-1] not in ".!?,;:":
                    text = text + best_punct[1]
            # Capitalize first letter if this cue follows a sentence-ending cue
            if i > 0 and new_entries:
                prev_t = str(new_entries[-1].get("text") or "").strip()
                if prev_t and prev_t[-1] in ".!?":
                    text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
            elif i == 0 and text:
                text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
            new_entries.append({**entry, "text": text})

        return new_entries

    def _terminal_punct_ratio(self, entries: list[dict[str, Any]]) -> float:
        """Fraction of the first 120 cues that end with terminal punctuation.
        Used by both the pause-boundary decision (via
        ``_transcript_has_terminal_punct``) and the A2 auto-restoration
        gate at ``_PUNCT_RESTORE_THRESHOLD``."""
        sample = entries[:120]
        if not sample:
            return 0.0
        n_punct = 0
        for entry in sample:
            if self._is_sentence_end(str(entry.get("text") or "")):
                n_punct += 1
        return n_punct / len(sample)

    def _transcript_has_terminal_punct(self, entries: list[dict[str, Any]]) -> bool:
        """True when the transcript appears genuinely punctuated.

        YouTube auto-generated captions typically ship without punctuation,
        so ``_is_sentence_end`` never fires for them. Some generated-caption
        videos include a handful of incidental periods (chapter timestamps,
        URLs, decimals) — we guard against those producing a false positive
        by requiring at least ~15% of the sampled cues to end with terminal
        punctuation. Under that threshold we treat the transcript as
        unpunctuated and fall back to cue-boundary alignment.
        """
        return self._terminal_punct_ratio(entries) >= 0.15

    def _cue_pause_breaks(
        self,
        entries: list[dict[str, Any]],
        *,
        min_pause_sec: float = 0.6,
    ) -> set[int]:
        """Indices ``i`` such that cues[i-1] → cues[i] has a gap ≥ min_pause_sec.

        Used as a sentence-boundary proxy for unpunctuated auto-captions.
        A longer gap between captions almost always coincides with a
        speaker's sentence break in practice; this is what YouTube's own
        caption segmenter uses as a heuristic too.

        Accepts both transcript formats: the raw ``{start, duration, text}``
        shape from youtube-transcript-api AND the refiner's internal
        ``{start, end, text}`` shape.
        """
        def _end(entry: dict[str, Any]) -> float:
            # Prefer explicit `end` key when present; otherwise compute from
            # start + duration. Falls back to start when neither is given.
            try:
                if "end" in entry and entry["end"] is not None:
                    return float(entry["end"])
                start = float(entry.get("start") or 0.0)
                dur = float(entry.get("duration") or 0.0)
                return start + dur
            except (TypeError, ValueError):
                return 0.0

        breaks: set[int] = set()
        for i in range(1, len(entries)):
            try:
                prev_end = _end(entries[i - 1])
                cur_start = float(entries[i].get("start") or 0.0)
            except (TypeError, ValueError):
                continue
            if cur_start - prev_end >= min_pause_sec:
                breaks.add(i)
        return breaks

    def _build_caption_cues(
        self,
        transcript: list[dict[str, Any]],
        clip_start: float,
        clip_end: float,
        fallback_text: str | None = None,
    ) -> list[dict[str, Any]]:
        if clip_end <= clip_start:
            return []

        clip_len = max(0.2, float(clip_end - clip_start))
        cues: list[dict[str, Any]] = []

        for entry in transcript:
            text = str(entry.get("text") or "").replace("\n", " ").strip()
            if not text:
                continue

            try:
                entry_start = float(entry.get("start") or 0.0)
            except (TypeError, ValueError):
                continue

            duration_value = entry.get("duration")
            try:
                entry_duration = float(duration_value) if duration_value is not None else 0.0
            except (TypeError, ValueError):
                entry_duration = 0.0

            if entry_duration <= 0:
                entry_duration = 1.8

            entry_end = entry_start + entry_duration
            if entry_end <= clip_start or entry_start >= clip_end:
                continue

            cue_start = max(0.0, max(entry_start, clip_start) - clip_start)
            cue_end = min(clip_len, min(entry_end, clip_end) - clip_start)
            if cue_end - cue_start < 0.16:
                cue_end = min(clip_len, cue_start + 0.9)

            payload = {
                "start": round(float(cue_start), 2),
                "end": round(float(min(clip_len, max(cue_end, cue_start + 0.16))), 2),
                "text": text[:220],
            }

            if cues and payload["text"] == cues[-1]["text"] and payload["start"] - cues[-1]["end"] <= 0.2:
                cues[-1]["end"] = payload["end"]
            else:
                cues.append(payload)

            if len(cues) >= 140:
                break

        if not cues and fallback_text and fallback_text.strip():
            # Truncate to 240 chars but snap to a sentence or word boundary
            # so the cue text doesn't end mid-word.
            raw = fallback_text.strip()
            if len(raw) <= 240:
                final_text = raw
            else:
                window = raw[:240]
                # Prefer the last sentence end within the window.
                sent_match = list(re.finditer(r"[.!?…][\"'\)\]]*", window))
                if sent_match:
                    final_text = window[: sent_match[-1].end()].rstrip()
                else:
                    # Fall back to last word boundary.
                    space_idx = window.rfind(" ")
                    final_text = (window[:space_idx] if space_idx > 100 else window).rstrip()
                    # Append ellipsis to indicate truncation.
                    if not final_text.endswith(("…", "...")):
                        final_text = final_text.rstrip(",;:") + "…"
            cues.append(
                {
                    "start": 0.0,
                    "end": round(clip_len, 2),
                    "text": final_text,
                }
            )
        return cues

    def record_feedback(
        self,
        conn,
        reel_id: str,
        helpful: bool,
        confusing: bool,
        rating: int | None,
        saved: bool,
    ) -> None:
        existing = fetch_one(conn, "SELECT id FROM reel_feedback WHERE reel_id = ?", (reel_id,))
        upsert(
            conn,
            "reel_feedback",
            {
                "id": str((existing or {}).get("id") or uuid.uuid4()),
                "reel_id": reel_id,
                "helpful": 1 if helpful else 0,
                "confusing": 1 if confusing else 0,
                "rating": rating,
                "saved": 1 if saved else 0,
                "created_at": now_iso(),
            },
            pk="reel_id",
        )
        try:
            reel_row = fetch_one(conn, "SELECT material_id FROM reels WHERE id = ?", (reel_id,))
            if reel_row and reel_row.get("material_id"):
                self.update_level_adjustment(conn, str(reel_row["material_id"]))
        except Exception:
            logger.exception("level adjustment recompute failed for reel %s", reel_id)

    def ranked_feed(
        self,
        conn,
        material_id: str,
        fast_mode: bool = False,
        generation_id: str | None = None,
        page_hint: int = 1,
    ) -> list[dict[str, Any]]:
        material = fetch_one(conn, "SELECT subject_tag, source_type, knowledge_level, level_adjustment FROM materials WHERE id = ?", (material_id,))
        subject_tag = str((material or {}).get("subject_tag") or "").strip() or None
        strict_topic_only = str((material or {}).get("source_type") or "").strip().lower() == "topic"
        level_target = effective_level_target(
            (material or {}).get("knowledge_level"),
            (material or {}).get("level_adjustment"),
        )
        self._last_effective_level_target = level_target
        self._last_knowledge_level = str((material or {}).get("knowledge_level") or "beginner")

        reel_where, reel_params = self._reel_scope_where(
            material_id=material_id,
            generation_id=generation_id,
            alias="r",
        )
        source_fingerprint = self._ranked_feed_source_fingerprint(
            conn,
            material_id=material_id,
            generation_id=generation_id,
            fast_mode=fast_mode,
            subject_tag=subject_tag,
            reel_where=reel_where,
            reel_params=reel_params,
        )
        cached = self._load_ranked_feed_cache(
            conn,
            material_id=material_id,
            generation_id=generation_id,
            fast_mode=fast_mode,
            source_fingerprint=source_fingerprint,
            subject_tag=subject_tag,
            strict_topic_only=strict_topic_only,
            level_target=level_target,
        )
        if cached is not None:
            return cached

        reel_rows = fetch_all(
            conn,
            f"""
            SELECT
                r.id AS reel_id,
                r.concept_id,
                r.video_id,
                c.title AS concept_title,
                c.keywords_json AS concept_keywords_json,
                c.summary AS concept_summary,
                c.embedding_json AS concept_embedding_json,
                v.title AS video_title,
                COALESCE(v.channel_title, '') AS video_channel_title,
                COALESCE(v.description, '') AS video_description,
                COALESCE(v.duration_sec, 0) AS video_duration_sec,
                r.video_url,
                r.t_start,
                r.t_end,
                r.transcript_snippet,
                r.takeaways_json,
                r.base_score,
                r.difficulty,
                COALESCE(SUM(f.helpful), 0) AS helpful_votes,
                COALESCE(SUM(f.confusing), 0) AS confusing_votes,
                COALESCE(AVG(f.rating), 3.0) AS avg_rating,
                COALESCE(SUM(f.saved), 0) AS saves,
                r.created_at
            FROM reels r
            JOIN concepts c ON c.id = r.concept_id
            JOIN videos v ON v.id = r.video_id
            LEFT JOIN reel_feedback f ON f.reel_id = r.id
            WHERE {reel_where}
              AND (r.t_end - r.t_start) >= 1
              AND (r.t_end - r.t_start) <= {_SERVING_MAX_CLIP_SEC}
            GROUP BY
                r.id,
                r.concept_id,
                r.video_id,
                c.title,
                c.keywords_json,
                c.summary,
                c.embedding_json,
                v.title,
                v.channel_title,
                v.description,
                v.duration_sec,
                r.video_url,
                r.t_start,
                r.t_end,
                r.transcript_snippet,
                r.takeaways_json,
                r.base_score,
                r.difficulty,
                r.created_at
            """,
            reel_params,
        )

        if self.youtube_service:
            enrichable_video_ids = sorted(
                {
                    str(row.get("video_id") or "").strip()
                    for row in reel_rows
                    if str(row.get("video_id") or "").strip()
                    and len(str(row.get("video_description") or "").strip()) < 240
                }
            )
            if enrichable_video_ids:
                details_by_id = self.youtube_service.video_details(enrichable_video_ids)
                for row in reel_rows:
                    video_id = str(row.get("video_id") or "").strip()
                    if not video_id:
                        continue
                    detail = details_by_id.get(video_id) or {}
                    detail_description = self._clean_video_description(str(detail.get("description") or ""))
                    current_description = self._clean_video_description(str(row.get("video_description") or ""))
                    if len(detail_description) <= len(current_description):
                        continue
                    # In-memory only: patch the row we're about to rank so this
                    # request sees the richer metadata, but do NOT persist back
                    # to the `videos` table from a read path. Persistence was
                    # causing concurrent-read-path races (two requests each
                    # hitting the YouTube API + upserting) and could also
                    # regress fields like `is_creative_commons` when `detail`
                    # was partially populated. Write-back belongs in the
                    # ingestion / refinement pipelines, not here.
                    row["video_title"] = str(detail.get("title") or row.get("video_title") or "").strip()
                    row["video_channel_title"] = str(detail.get("channel_title") or row.get("video_channel_title") or "").strip()
                    row["video_description"] = detail_description
                    row["video_duration_sec"] = int(detail.get("duration_sec") or row.get("video_duration_sec") or 0)

        concept_signal_totals: dict[str, list[float]] = {}
        for row in reel_rows:
            concept_id = str(row.get("concept_id") or "")
            if not concept_id:
                continue
            totals = concept_signal_totals.setdefault(concept_id, [0.0, 0.0])
            totals[0] += float(row.get("helpful_votes") or 0.0)
            totals[1] += float(row.get("confusing_votes") or 0.0)
        concept_signal = {
            concept_id: (totals[0], totals[1])
            for concept_id, totals in concept_signal_totals.items()
        }

        concept_rows = fetch_all(
            conn,
            "SELECT id, title, keywords_json, summary, embedding_json FROM concepts WHERE material_id = ? ORDER BY created_at ASC, title ASC",
            (material_id,),
        )
        concept_order = {row["id"]: idx + 1 for idx, row in enumerate(concept_rows)}
        concept_by_id = {row["id"]: row for row in concept_rows}
        total_concepts = len(concept_rows)
        material_context_terms = self._build_material_context_terms(concepts=concept_rows, subject_tag=subject_tag)
        video_ids = sorted({str(row.get("video_id") or "") for row in reel_rows if str(row.get("video_id") or "").strip()})
        retrieval_candidate_by_video: dict[str, dict[str, Any]] = {}
        if video_ids:
            placeholders = ", ".join(["?"] * len(video_ids))
            candidate_rows = fetch_all(
                conn,
                f"""
                SELECT
                    rc.video_id,
                    rc.strategy,
                    rc.stage,
                    rc.source_surface,
                    rc.discovery_score,
                    rc.clipability_score,
                    rc.final_score,
                    rc.feature_json,
                    rc.created_at
                FROM retrieval_candidates rc
                JOIN retrieval_runs rr ON rr.id = rc.run_id
                WHERE rr.material_id = ?
                  AND rc.video_id IN ({placeholders})
                ORDER BY rc.created_at DESC, rc.position ASC
                """,
                tuple([material_id, *video_ids]),
            )
            for candidate_row in candidate_rows:
                video_id = str(candidate_row.get("video_id") or "").strip()
                if not video_id or video_id in retrieval_candidate_by_video:
                    continue
                retrieval_candidate_by_video[video_id] = dict(candidate_row)

        scored: list[dict[str, Any]] = []
        for row in reel_rows:
            concept_helpful, concept_confusing = concept_signal.get(row["concept_id"], (0.0, 0.0))
            try:
                takeaways = json.loads(row["takeaways_json"])
            except (TypeError, json.JSONDecodeError):
                takeaways = []
            if not isinstance(takeaways, list):
                takeaways = []
            video_title = str(row.get("video_title") or "").strip()
            video_description = self._clean_video_description(str(row.get("video_description") or ""))
            transcript_snippet = str(row.get("transcript_snippet") or "")
            concept_row = concept_by_id.get(row["concept_id"], row)
            try:
                concept_keywords = json.loads(str(concept_row.get("concept_keywords_json") or concept_row.get("keywords_json") or "[]"))
            except json.JSONDecodeError:
                concept_keywords = []
            if not isinstance(concept_keywords, list):
                concept_keywords = []
            concept_title = str(concept_row.get("concept_title") or concept_row.get("title") or row.get("concept_title") or "").strip()
            concept_summary = str(concept_row.get("concept_summary") or concept_row.get("summary") or "").strip()
            concept_terms = [concept_title, *[str(k) for k in concept_keywords[:8]], concept_summary]
            context_terms = self._context_terms_for_concept(concept_terms, material_context_terms)
            video_id = str(row.get("video_id") or "").strip()
            retrieval_candidate = retrieval_candidate_by_video.get(video_id, {})

            concept_embedding: np.ndarray | None = None
            if not fast_mode:
                embedding_json = str(
                    concept_row.get("concept_embedding_json")
                    or concept_row.get("embedding_json")
                    or row.get("concept_embedding_json")
                    or ""
                )
                if embedding_json:
                    concept_embedding = self._parse_embedding_vector(embedding_json)

            relevance = self._score_text_relevance(
                conn,
                text=" ".join([video_title, video_description, transcript_snippet]).strip(),
                concept_terms=concept_terms,
                context_terms=context_terms,
                concept_embedding=concept_embedding,
                subject_tag=subject_tag,
            )
            if self._is_hard_blocked_low_value_video(
                title=video_title,
                description=video_description,
                channel_title=str(row.get("video_channel_title") or ""),
                subject_tag=subject_tag,
            ):
                continue
            relevance["passes"] = self._passes_relevance_gate(
                relevance=relevance,
                require_context=bool(context_terms),
                fast_mode=fast_mode,
            )
            if strict_topic_only:
                direct_topic_support = (
                    len(relevance.get("concept_hits") or []) >= 1
                    or float(relevance.get("concept_overlap") or 0.0) >= (0.08 if fast_mode else 0.06)
                    or float(relevance.get("subject_overlap") or 0.0) >= 0.04
                )
                if not direct_topic_support:
                    continue
                if float(relevance.get("off_topic_penalty") or 0.0) >= 0.12 and float(relevance.get("subject_overlap") or 0.0) < 0.04:
                    continue
            if not relevance["passes"] and (
                float(relevance.get("off_topic_penalty") or 0.0) >= 0.24
                or float(relevance.get("score") or -1.0) < 0.02
            ):
                # Hide strongly off-topic clips that can still exist from older generations.
                continue
            relevance_context = self._merge_relevance_context(relevance, relevance)

            ai_summary = self._brief_ai_summary(
                conn,
                video_id=str(row.get("video_id") or ""),
                concept_title=concept_title,
                video_title=video_title,
                video_description=video_description,
                transcript_snippet=transcript_snippet,
                takeaways=takeaways,
                fast_mode=fast_mode,
            )
            safe_page_hint = max(1, int(page_hint))
            _diff = 0.5 if row.get("difficulty") is None else float(row["difficulty"])
            score = (
                float(row["base_score"])
                + 0.18 * float(row["helpful_votes"])
                - 0.22 * float(row["confusing_votes"])
                + 0.06 * (float(row["avg_rating"] or 3.0) - 3.0)
                + 0.05 * float(row["saves"])
                + 0.04 * concept_helpful
                - 0.06 * concept_confusing
                + 0.22 * float(relevance_context.get("score") or 0.0)
                - 0.12 * float(relevance.get("off_topic_penalty") or 0.0)
                + 0.04 * float(self.SOURCE_SURFACE_PRIOR.get(str(retrieval_candidate.get("source_surface") or ""), 0.82))
                + 0.12 * (1.0 - 2.0 * abs(_diff - level_target))
                + 0.05 * (1.0 - _diff) * max(0.0, 1.0 - (safe_page_hint - 1) / 2.0)
            )
            scored.append(
                {
                    "reel_id": row["reel_id"],
                    "material_id": material_id,
                    "video_id": video_id,
                    "concept_id": row["concept_id"],
                    "concept_title": concept_title,
                    "video_title": video_title,
                    "video_description": video_description,
                    "channel_name": str(row.get("video_channel_title") or "").strip(),
                    "ai_summary": ai_summary,
                    "video_url": row["video_url"],
                    "t_start": float(row["t_start"]),
                    "t_end": float(row["t_end"]),
                    "transcript_snippet": transcript_snippet,
                    "takeaways": takeaways,
                    "score": score,
                    "relevance_score": float(relevance_context.get("score") or 0.0),
                    "discovery_score": float(retrieval_candidate.get("discovery_score") or relevance_context.get("score") or 0.0),
                    "clipability_score": float(
                        retrieval_candidate.get("clipability_score")
                        or self._score_clipability_from_metadata({"duration_sec": row.get("video_duration_sec")}, strategy="literal")
                    ),
                    "query_strategy": str(retrieval_candidate.get("strategy") or ""),
                    "retrieval_stage": str(retrieval_candidate.get("stage") or ""),
                    "source_surface": str(retrieval_candidate.get("source_surface") or ""),
                    "matched_terms": relevance_context.get("matched_terms", []),
                    "relevance_reason": str(relevance_context.get("reason") or ""),
                    "concept_position": concept_order.get(row["concept_id"]),
                    "total_concepts": total_concepts,
                    "video_duration_sec": int(row.get("video_duration_sec") or 0),
                    "clip_duration_sec": round(max(0.0, float(row["t_end"]) - float(row["t_start"])), 2),
                    "difficulty": (None if row.get("difficulty") is None else float(row["difficulty"])),
                    "created_at": row["created_at"],
                }
            )

        scored.sort(key=lambda x: (x["score"], x["created_at"]), reverse=True)
        deduped: list[dict[str, Any]] = []
        seen_reel_ids: set[str] = set()
        seen_clip_keys: set[str] = set()
        kept_spans_by_video: dict[str, list[tuple[float, float]]] = {}
        for item in scored:
            reel_id = str(item.get("reel_id") or "")
            if reel_id and reel_id in seen_reel_ids:
                continue
            video_id = str(item.get("video_id") or "")
            if not video_id:
                continue
            t_start = float(item.get("t_start") or 0)
            t_end = float(item.get("t_end") or 0)
            clip_key = self._clip_key(video_id, t_start, t_end)
            if clip_key in seen_clip_keys:
                continue
            # Score-descending order → the best of a near-duplicate cluster wins.
            if any(
                clip_spans_duplicate(t_start, t_end, k0, k1)
                for k0, k1 in kept_spans_by_video.get(video_id, ())
            ):
                continue
            if reel_id:
                seen_reel_ids.add(reel_id)
            seen_clip_keys.add(clip_key)
            kept_spans_by_video.setdefault(video_id, []).append((t_start, t_end))
            deduped.append(dict(item))

        # --- Same-video grouping ---
        # Without this, score-based sorting can interleave reels from
        # different YouTube videos, so a multi-reel arc from one clip gets
        # interrupted by a reel from another video. Users experience this
        # as "videos from the same clip are interrupted by other videos."
        # Fix: after score-sort + dedup, group reels by ``video_id`` and
        # emit each group consecutively. Within a group, sort by
        # ``t_start`` so the clip's narrative plays in chronological order
        # (part 1 → part 2 → part 3). Groups themselves are ordered by
        # their best (max) reel score, preserving overall ranking.
        if deduped:
            grouped_by_video: dict[str, list[dict[str, Any]]] = {}
            video_first_seen_order: list[str] = []
            video_best_score: dict[str, float] = {}
            video_best_created_at: dict[str, str] = {}
            for item in deduped:
                vid = str(item.get("video_id") or "")
                if vid not in grouped_by_video:
                    grouped_by_video[vid] = []
                    video_first_seen_order.append(vid)
                grouped_by_video[vid].append(item)
                s = float(item.get("score") or 0.0)
                if s > video_best_score.get(vid, float("-inf")):
                    video_best_score[vid] = s
                    video_best_created_at[vid] = str(item.get("created_at") or "")
            # Sort video groups by their best reel's score (desc), then by
            # that reel's created_at (desc) for tiebreaks — mirrors the
            # original per-reel sort key so ordering remains deterministic.
            video_first_seen_order.sort(
                key=lambda v: (
                    video_best_score.get(v, 0.0),
                    video_best_created_at.get(v, ""),
                ),
                reverse=True,
            )
            regrouped: list[dict[str, Any]] = []
            for vid in video_first_seen_order:
                group = sorted(
                    grouped_by_video[vid],
                    key=lambda x: (
                        float(x.get("t_start") or 0.0),
                        float(x.get("t_end") or 0.0),
                    ),
                )
                regrouped.extend(group)
            deduped = regrouped

        deduped_video_ids = sorted({str(item.get("video_id") or "") for item in deduped if item.get("video_id")})
        transcript_by_video: dict[str, list[dict[str, Any]]] = {}
        if deduped_video_ids:
            placeholders = ", ".join(["?"] * len(deduped_video_ids))
            transcript_rows = fetch_all(
                conn,
                f"SELECT video_id, transcript_json FROM transcript_cache WHERE video_id IN ({placeholders})",
                tuple(deduped_video_ids),
            )
            for trow in transcript_rows:
                try:
                    transcript_by_video[str(trow["video_id"])] = json.loads(trow["transcript_json"])
                except (TypeError, json.JSONDecodeError):
                    transcript_by_video[str(trow["video_id"])] = []

        response_rows: list[dict[str, Any]] = []
        for item in deduped:
            clean_item = dict(item)
            # Keep video_id on the response row so downstream filters (notably
            # main._ranked_request_reels's exclude_video_ids filter) can match
            # on it. Stripping it here silently defeated client pagination.
            video_id = str(clean_item.get("video_id") or "")
            clean_item["video_id"] = video_id
            clean_item["captions"] = self._build_caption_cues(
                transcript=transcript_by_video.get(video_id, []),
                clip_start=float(clean_item.get("t_start") or 0.0),
                clip_end=float(clean_item.get("t_end") or 0.0),
                fallback_text=str(clean_item.get("transcript_snippet") or ""),
            )
            response_rows.append(clean_item)

        self._store_ranked_feed_cache(
            conn,
            material_id=material_id,
            generation_id=generation_id,
            fast_mode=fast_mode,
            source_fingerprint=source_fingerprint,
            reels=response_rows,
            subject_tag=subject_tag,
            strict_topic_only=strict_topic_only,
            level_target=level_target,
        )
        return response_rows
