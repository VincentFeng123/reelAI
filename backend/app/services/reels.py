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
from ..db import (
    LEGACY_LEARNER_ID,
    DatabaseIntegrityError,
    dumps_json,
    execute_modify,
    fetch_all,
    fetch_one,
    now_iso,
    upsert,
)
from . import llm_router
from .concepts import build_takeaways
from ..ingestion.errors import RateLimitedError as _IngestRateLimitedError
from ..ingestion.segment import normalize_clip_window as _normalize_clip_window_fn
from ..clip_engine.cancellation import raise_if_cancelled as _raise_if_clip_cancelled
from ..clip_engine.errors import CancellationError as _ClipEngineCancellationError
from ..clip_engine.errors import ProviderError as _ClipEngineProviderError
from ..clip_engine.metadata import extract_video_id as _extract_embed_video_id
from ..clip_engine.metadata import normalize_youtube_video_id
from ..clip_engine.provider_cache import validate_transcript_payload
from ..clip_engine.provider_runtime import GenerationContext
from ..clip_engine.silence import (
    persisted_boundary_is_usable,
    persisted_boundary_is_verified,
)
from .segmenter import (
    SegmentMatch,
    TranscriptChunk,
    chunk_transcript,
    lexical_overlap_score,
    normalize_terms,
    select_segments,
)
from .search_query_plan import build_search_query_plan
from .topic_expansion import TopicExpansionService
from .structural_classifier import classify_passage
from .knowledge_level import (
    difficulty_matches_knowledge_level,
    effective_level_target,
)

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

# One small Supadata batch can still yield many clips per source while keeping
# first-result latency bounded.
MATERIAL_MAX_VIDEOS_PER_CONCEPT = 5
MATERIAL_GEN_MAX_VIDEOS = 12
MATERIAL_REEL_INVENTORY_LIMIT = 300


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

    def emit(
        self,
        reel: dict[str, Any],
        *,
        chain_id: str,
        prerequisite_safe: bool = False,
    ) -> None:
        if self._downstream is None:
            return
        if prerequisite_safe:
            self._flush_others(active_chain_id=None)
            self._forward(reel)
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
    VALID_VIDEO_DURATION_PREFS = {"any", "short", "medium", "long"}
    DEFAULT_TARGET_CLIP_DURATION_SEC = 55
    MIN_TARGET_CLIP_DURATION_SEC = 15
    MAX_TARGET_CLIP_DURATION_SEC = 180
    MIN_TARGET_CLIP_DURATION_RANGE_GAP_SEC = 15
    DEFAULT_RETRIEVAL_PROFILE: RetrievalProfile = "deep"
    BOOTSTRAP_CONCEPT_LIMIT = 4
    BOOTSTRAP_PRIMARY_QUERY_COUNT = 3
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
    # Bump whenever the cached row shape changes so stale entries are invalidated.
    # v4: video_id retained on response rows (was stripped in v3).
    # v5: reel rows now originate from _persist_ingest path (T4 clip-engine swap).
    # v13: full caption text and the current transcript-semantic surfaceability
    # gates must be recomputed instead of replaying pre-gate cached rows.
    # v14: discard rows accepted only by the retired -24 dBFS adaptive verifier.
    # v17: retain public source identity and authoritative selector relevance.
    # v18: bind captions to an immutable selection-time cue snapshot.
    # v19: require the current acoustic-boundary inventory contract.
    # v20: reject openings with unresolved explicit backward references.
    # v22: require exact cross-cue boundary grounding from the current selector.
    # v23: prefer the requested difficulty bin, with a nearest valid-bin fallback.
    # v24: expose current selector relevance consistently in cached feed rows.
    # v25: accept validated transcript-context boundaries in current inventory.
    # v26: rank exact-request fulfillment within each difficulty stage.
    # v30: invalidate inventory produced before coherent rolling-caption edges.
    # v32: invalidate inventory produced before structural evidence normalization.
    # v33: invalidate inventory produced before split-caption prompt recovery.
    # v34: invalidate inventory produced before early structural-onset recovery.
    # v35: require the v28 selector and request compatibility contract.
    # v36: require the v29 sentence-tail selection contract.
    # v37: require the v30 direct-URL selection contract.
    RANKED_FEED_CACHE_VERSION = 37
    RANKED_FEED_CACHE_CONTRACT_VERSION = "quality_silence_v30"
    DIFFICULTY_FALLBACK_CONTRACTS = frozenset({
        "quality_silence_v3",
        "quality_silence_v4",
        "quality_silence_v5",
        "quality_silence_v6",
        "quality_silence_v7",
        "quality_silence_v8",
        "quality_silence_v9",
        "quality_silence_v10",
        "quality_silence_v11",
        "quality_silence_v12",
        "quality_silence_v13",
        "quality_silence_v14",
        "quality_silence_v15",
        "quality_silence_v16",
        "quality_silence_v17",
        "quality_silence_v18",
        "quality_silence_v19",
        "quality_silence_v20",
        "quality_silence_v21",
        "quality_silence_v22",
        "quality_silence_v23",
        "quality_silence_v24",
        "quality_silence_v25",
        "quality_silence_v26",
        "quality_silence_v27",
        "quality_silence_v28",
        "quality_silence_v29",
        "quality_silence_v30",
    })
    CONCEPT_ADJUSTMENT_BOUND = 0.25
    GOT_IT_CONCEPT_STEP = 0.04
    NEED_HELP_CONCEPT_STEP = 0.06
    GLOBAL_FEEDBACK_WINDOW = 12
    GLOBAL_FEEDBACK_MIN_ROWS = 3
    def __init__(self, embedding_service, youtube_service=None, ingestion_pipeline=None) -> None:
        settings = get_settings()
        self.embedding_service = embedding_service
        # Retained as a compatibility argument for older tests/callers. Reel
        # retrieval and ranking never call Google's YouTube API.
        del youtube_service
        self.ingestion_pipeline = ingestion_pipeline
        self.chat_model = settings.gemini_model
        self.retrieval_engine_v2_enabled = bool(settings.retrieval_engine_v2_enabled)
        self.retrieval_tier2_enabled = bool(settings.retrieval_tier2_enabled)
        self.retrieval_debug_logging = bool(settings.retrieval_debug_logging)
        self._generation_state = threading.local()
        self.serverless_mode = bool(
            os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME") or os.getenv("K_SERVICE")
        )
        self._strategy_history_cache: dict[str, float] = {}
        self._strategy_history_cache_lock = threading.Lock()
        self._strategy_history_cache_max_size = 10_000
        self.llm_available = llm_router.gemini_or_groq_available()
        self.topic_expansion_service = TopicExpansionService()
        # Kept only for compatibility with the legacy YouTube planner. The active
        # clipping pipeline is YouTube-only and never routes to another provider.
        self._provider_by_video_id: dict[str, str] = {}

    def learner_progress(self, conn, material_id: str, learner_id: str) -> dict[str, Any]:
        """Return learner-scoped progress, lazily seeded from the material default."""
        learner_id = str(learner_id or LEGACY_LEARNER_ID)
        existing = fetch_one(
            conn,
            "SELECT * FROM learner_material_progress WHERE learner_id = ? AND material_id = ?",
            (learner_id, material_id),
        )
        if existing:
            return existing
        material = fetch_one(
            conn,
            "SELECT knowledge_level FROM materials WHERE id = ?",
            (material_id,),
        )
        if not material:
            raise ValueError(f"unknown material_id: {material_id}")
        level = str(material.get("knowledge_level") or "beginner")
        timestamp = now_iso()
        upsert(
            conn,
            "learner_material_progress",
            {
                "learner_id": learner_id,
                "material_id": material_id,
                "selected_level": level,
                "global_adjustment": 0.0,
                # Legacy rows remain aggregate-only, but legacy service callers
                # still see their historical feedback in compatibility tests.
                "difficulty_reset_at": "" if learner_id == LEGACY_LEARNER_ID else timestamp,
                "feedback_revision": 0,
                "updated_at": timestamp,
            },
            pk=["learner_id", "material_id"],
        )
        return fetch_one(
            conn,
            "SELECT * FROM learner_material_progress WHERE learner_id = ? AND material_id = ?",
            (learner_id, material_id),
        ) or {}

    def set_learner_level(
        self, conn, material_id: str, learner_id: str, selected_level: str,
    ) -> dict[str, Any]:
        progress = self.learner_progress(conn, material_id, learner_id)
        timestamp = now_iso()
        execute_modify(
            conn,
            """
            UPDATE learner_material_progress
            SET selected_level = ?, global_adjustment = 0.0,
                difficulty_reset_at = ?, feedback_revision = feedback_revision + 1,
                updated_at = ?
            WHERE learner_id = ? AND material_id = ?
            """,
            (selected_level, timestamp, timestamp, learner_id, material_id),
        )
        return {**progress, "selected_level": selected_level, "global_adjustment": 0.0,
                "difficulty_reset_at": timestamp,
                "feedback_revision": int(progress.get("feedback_revision") or 0) + 1,
                "updated_at": timestamp}

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
        learner_id: str = LEGACY_LEARNER_ID,
        feedback_revision: int = 0,
        page_hint: int = 1,
        exclusions_fingerprint: str = "",
        content_fingerprint: str = "",
        require_verified_boundaries: bool = False,
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
            "learner_id": str(learner_id),
            "feedback_revision": int(feedback_revision),
            "page_hint": max(1, int(page_hint)),
            "exclusions_fingerprint": str(exclusions_fingerprint),
            "content_fingerprint": str(content_fingerprint),
            "require_verified_boundaries": bool(require_verified_boundaries),
            "version": self.RANKED_FEED_CACHE_VERSION,
            "selection_contract_version": self.RANKED_FEED_CACHE_CONTRACT_VERSION,
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
        learner_id: str,
    ) -> str:
        relevant_reels_cte = f"""
            WITH relevant_reels AS (
                SELECT r.id, r.video_id, r.created_at
                FROM reels r
                WHERE {reel_where}
                  AND r.t_start >= 0
                  AND r.t_end > r.t_start
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
                COALESCE(MAX(COALESCE(NULLIF(f.updated_at, ''), f.created_at)), '') AS feedback_updated_at
            FROM relevant_reels
            LEFT JOIN videos v ON v.id = relevant_reels.video_id
            LEFT JOIN reel_feedback f
              ON f.reel_id = relevant_reels.id
             AND f.learner_id = ?
            """,
            (*reel_params, learner_id),
        ) or {}
        reel_state_rows = fetch_all(
            conn,
            f"""
            SELECT
                r.id,
                r.t_start,
                r.t_end,
                r.transcript_snippet,
                r.informativeness,
                r.base_score,
                r.difficulty,
                r.selected_cue_ids_json,
                r.search_context_json
            FROM reels r
            WHERE {reel_where}
              AND r.t_start >= 0
              AND r.t_end > r.t_start
            ORDER BY r.id ASC
            """,
            reel_params,
        )
        reel_state_fingerprint = hashlib.sha256(
            json.dumps(
                [dict(row) for row in reel_state_rows],
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        transcript_stats = fetch_one(
            conn,
            relevant_reels_cte
            + """
            SELECT
                COUNT(*) AS transcript_count,
                COALESCE(MAX(tc.created_at), '') AS transcript_updated_at
            FROM transcript_artifacts tc
            WHERE tc.video_id IN (
                SELECT DISTINCT CASE
                    WHEN video_id LIKE 'yt:%' THEN SUBSTR(video_id, 4)
                    ELSE video_id
                END
                FROM relevant_reels
            )
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
            "selection_contract_version": self.RANKED_FEED_CACHE_CONTRACT_VERSION,
            "material_id": material_id,
            "generation_id": generation_id or "",
            "fast_mode": bool(fast_mode),
            "summary_mode": summary_mode,
            "subject_tag": subject_tag or "",
            "reel_count": int(reel_stats.get("reel_count") or 0),
            "reel_updated_at": str(reel_stats.get("reel_updated_at") or ""),
            "reel_state_fingerprint": reel_state_fingerprint,
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
        learner_id: str = LEGACY_LEARNER_ID,
        feedback_revision: int = 0,
        page_hint: int = 1,
        exclusions_fingerprint: str = "",
        content_fingerprint: str = "",
        require_verified_boundaries: bool = False,
    ) -> list[dict[str, Any]] | None:
        cache_key = self._ranked_feed_cache_key(
            material_id=material_id,
            generation_id=generation_id,
            fast_mode=fast_mode,
            subject_tag=subject_tag,
            strict_topic_only=strict_topic_only,
            level_target=level_target,
            learner_id=learner_id,
            feedback_revision=feedback_revision,
            page_hint=page_hint,
            exclusions_fingerprint=exclusions_fingerprint,
            content_fingerprint=content_fingerprint,
            require_verified_boundaries=require_verified_boundaries,
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
        learner_id: str = LEGACY_LEARNER_ID,
        feedback_revision: int = 0,
        page_hint: int = 1,
        exclusions_fingerprint: str = "",
        content_fingerprint: str = "",
        require_verified_boundaries: bool = False,
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
                    learner_id=learner_id,
                    feedback_revision=feedback_revision,
                    page_hint=page_hint,
                    exclusions_fingerprint=exclusions_fingerprint,
                    content_fingerprint=content_fingerprint,
                    require_verified_boundaries=require_verified_boundaries,
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
        on_reel_created: Callable[[dict[str, Any]], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        knowledge_level_override: str | None = None,
        learner_id: str = LEGACY_LEARNER_ID,
        generation_context: GenerationContext | None = None,
        max_generation_videos: int | None = None,
        acquisition_concept_offset: int = 0,
        max_new_reels: int | None = None,
        analyzed_video_ids: set[str] | None = None,
        retrieved_video_ids: set[str] | None = None,
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

        def run_pre_ingestion(operation: Callable[[], Any]) -> Any:
            try:
                return operation()
            except _ClipEngineCancellationError as exc:
                raise GenerationCancelledError("Generation cancelled.") from exc

        raise_if_cancelled()
        retrieval_profile = self._normalize_retrieval_profile(retrieval_profile)
        new_reel_limit = (
            None if max_new_reels is None else max(0, int(max_new_reels))
        )
        if new_reel_limit == 0:
            return []
        analyzed_ids = analyzed_video_ids if analyzed_video_ids is not None else set()
        chain_emitter = _ChainBufferingEmitter(on_reel_created) if on_reel_created is not None else None
        self._generation_state.min_relevance_threshold = max(
            0.0, min(1.0, float(min_relevance_threshold))
        )
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
        stored_knowledge_level = str((material or {}).get("knowledge_level") or "beginner")
        requested_knowledge_level = str(knowledge_level_override or "").strip().lower()
        material_knowledge_level = (
            requested_knowledge_level
            if requested_knowledge_level in {"beginner", "intermediate", "advanced"}
            else stored_knowledge_level
        )
        subject_tag = str((material or {}).get("subject_tag") or "").strip() or None
        literal_subject_tag = self._clean_query_text(subject_tag or "") or None
        strict_topic_only = str((material or {}).get("source_type") or "").strip().lower() == "topic"
        strict_topic_expansion: dict[str, Any] | None = None
        if strict_topic_only and subject_tag:
            # The practice retrieval path owns query expansion. Keep material
            # generation anchored to the user's literal topic so a second AI
            # planner cannot delay or redirect the request before video search.
            strict_topic_expansion = {
                "canonical_topic": subject_tag,
                "aliases": [],
                "subtopics": [],
                "related_terms": [],
            }
        if not concepts and strict_topic_only and subject_tag:
            concepts = run_pre_ingestion(
                lambda: self._build_topic_only_concepts_from_expansion(
                    conn,
                    material_id=material_id,
                    subject_tag=subject_tag,
                    expansion=strict_topic_expansion,
                    should_cancel=should_cancel,
                )
            )
        if not concepts:
            return []
        root_topic_terms: list[str] = []
        if strict_topic_only and subject_tag:
            concepts = run_pre_ingestion(
                lambda: self._bootstrap_topic_retrieval_concepts(
                    conn=conn,
                    concepts=concepts,
                    subject_tag=subject_tag,
                    material_id=material_id,
                    expansion=strict_topic_expansion,
                    should_cancel=should_cancel,
                )
            )
            if not concepts:
                return []
            root_topic_terms = [subject_tag]
        material_parent_concepts = concepts
        if not subject_tag and concept_id:
            material_parent_concepts = fetch_all(
                conn,
                """
                SELECT id, material_id, title, keywords_json, summary, embedding_json, created_at
                FROM concepts
                WHERE material_id = ?
                ORDER BY created_at ASC
                """,
                (material_id,),
            )
        material_parent_topic = subject_tag
        if not material_parent_topic:
            material_parent_topic = " ".join(
                self._build_material_context_terms(
                    concepts=material_parent_concepts,
                    subject_tag=None,
                    max_terms=3,
                )
            ) or None
        concepts = self._order_concepts(conn, material_id, concepts, learner_id)
        if generation_context is not None and concepts:
            # One concept/query family per durable acquisition pass keeps the
            # fast (3-search) and slow (6-search) initial budgets truthful.
            concept_index = max(0, int(acquisition_concept_offset)) % len(concepts)
            concepts = [concepts[concept_index]]
        safe_video_duration_pref = self._normalize_preferred_video_duration(preferred_video_duration)
        # Deprecated request fields remain in the public signature, but clip
        # completeness and silence—not duration—define the selected range.
        clip_min_len, clip_max_len, safe_target_clip_duration = (0, 0, 0)

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
        generation_video_limit = max(
            1,
            min(
                MATERIAL_GEN_MAX_VIDEOS,
                int(max_generation_videos or MATERIAL_GEN_MAX_VIDEOS),
            ),
        )
        total_concepts = len(concepts)

        for idx, concept in enumerate(concepts):
            raise_if_cancelled()
            if new_reel_limit is not None and len(generated) >= new_reel_limit:
                break
            # Persistence is never truncated by the requested response page;
            # only the material-wide inventory ceiling and provider budgets bound it.
            material_reel_count = int(
                (
                    fetch_one(
                        conn,
                        "SELECT COUNT(*) AS reel_count FROM reels WHERE material_id = ?",
                        (material_id,),
                    )
                    or {}
                ).get("reel_count")
                or 0
            )
            remaining_reel_capacity = max(
                0,
                MATERIAL_REEL_INVENTORY_LIMIT - material_reel_count,
            )
            if remaining_reel_capacity <= 0:
                break
            if videos_processed >= generation_video_limit:
                break
            topic = self._concept_topic_query(
                concept,
                parent_topic=material_parent_topic,
            )
            if not topic:
                continue
            video_budget = min(
                MATERIAL_MAX_VIDEOS_PER_CONCEPT,
                generation_video_limit - videos_processed,
            )
            if video_budget <= 0:
                break
            surface_reel_capacity = (
                new_reel_limit - len(generated)
                if new_reel_limit is not None
                else (
                    num_reels - len(generated)
                    if generation_context is not None
                    else max(3, num_reels - len(generated)) + 2
                )
            )
            ingest_reel_cap = min(
                remaining_reel_capacity,
                max(0, surface_reel_capacity),
            )
            if ingest_reel_cap <= 0:
                break

            def _stream(reel_obj, _concept=concept, _idx=idx):
                if chain_emitter is not None:
                    streamed = self._reel_attribution_to_dict(
                        reel_obj,
                        _concept,
                        _idx,
                        total_concepts,
                    )
                    chain_emitter.emit(
                        streamed,
                        chain_id=str(streamed.get("chain_id") or ""),
                        prerequisite_safe=bool(
                            streamed.get("selection_contract_version")
                        ),
                    )

            try:
                analyzed_before = len(analyzed_ids)
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
                    max_reels=ingest_reel_cap,
                    max_persisted_reels=remaining_reel_capacity,
                    on_reel_created=(None if dry_run else _stream),
                    dry_run=dry_run,
                    should_cancel=should_cancel,
                    creative_commons_only=creative_commons_only,
                    preferred_video_duration=safe_video_duration_pref,
                    generation_context=generation_context,
                    literal_topic=(
                        literal_subject_tag
                        if literal_subject_tag and concept_id is None
                        else topic
                    ),
                    retrieval_profile=retrieval_profile,
                    analyzed_video_ids=analyzed_ids,
                    retrieved_video_ids=retrieved_video_ids,
                )
            except _ClipEngineCancellationError as exc:
                raise GenerationCancelledError("Generation cancelled.") from exc
            except _IngestRateLimitedError:
                # Process-wide rate limit tripped: subsequent concepts would hit it
                # too. Stop and surface it so the endpoint maps it to HTTP 429.
                raise
            except _ClipEngineProviderError:
                existing_progress = bool(generated)
                if not existing_progress and generation_id:
                    existing_progress = fetch_one(
                        conn,
                        "SELECT id FROM reels WHERE generation_id = ? LIMIT 1",
                        (generation_id,),
                    ) is not None
                if not existing_progress:
                    raise
                logger.warning(
                    "generate_reels: provider failed after partial progress; "
                    "keeping generation=%s",
                    generation_id,
                )
                break
            except Exception:
                # One concept's engine failure must not abort the whole generation —
                # log it and move on to the next concept (Finding #4a).
                logger.exception(
                    "generate_reels: ingest_topic failed for concept=%s; skipping to next concept",
                    concept.get("id"),
                )
                continue
            accumulated_exclusions.extend(resolved_ids)
            if retrieved_video_ids is not None:
                retrieved_video_ids.update(
                    str(video_id or "").strip().split(":", 1)[-1]
                    for video_id in resolved_ids
                    if str(video_id or "").strip()
                )
            analyzed_delta = max(0, len(analyzed_ids) - analyzed_before)
            videos_processed += analyzed_delta or len(resolved_ids)

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

    def _normalize_preferred_video_duration(self, value: str | None) -> str:
        if value in self.VALID_VIDEO_DURATION_PREFS:
            return str(value)
        return "any"

    def _normalize_retrieval_profile(self, value: str | None) -> RetrievalProfile:
        if value == "bootstrap":
            return "bootstrap"
        return self.DEFAULT_RETRIEVAL_PROFILE

    def _normalize_target_clip_duration(self, value: int | float | None) -> int:
        if value is None:
            return self.DEFAULT_TARGET_CLIP_DURATION_SEC
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return self.DEFAULT_TARGET_CLIP_DURATION_SEC
        return max(self.MIN_TARGET_CLIP_DURATION_SEC, min(self.MAX_TARGET_CLIP_DURATION_SEC, parsed))

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

    def _topic_breadth_class(
        self,
        subject_tag: str | None,
        *,
        conn: Any = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> str:
        _raise_if_clip_cancelled(should_cancel)
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
            should_cancel=should_cancel,
        )

    def _merge_breadth_class_with_llm(
        self,
        cleaned: str,
        *,
        subject_key: str,
        rules_class: str,
        conn: Any,
        should_cancel: Callable[[], bool] | None = None,
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
                should_cancel=should_cancel,
            )
            _raise_if_clip_cancelled(should_cancel)
        except _ClipEngineCancellationError:
            raise
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
        raw_informativeness = getattr(reel_obj, "informativeness", None)
        raw_difficulty = getattr(reel_obj, "difficulty", None)
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
            "match_reason": getattr(reel_obj, "match_reason", "") or "",
            "informativeness": float(
                0.6 if raw_informativeness is None else raw_informativeness
            ),
            "video_url": video_url,
            "t_start": t_start,
            "t_end": t_end,
            "transcript_snippet": getattr(reel_obj, "transcript_snippet", "") or "",
            "takeaways": list(getattr(reel_obj, "takeaways", None) or []),
            "captions": captions,
            "score": float(getattr(reel_obj, "score", 0.0) or 0.0),
            "relevance_score": getattr(reel_obj, "relevance_score", None),
            "topic_relevance": getattr(
                reel_obj,
                "selection_topic_relevance",
                getattr(reel_obj, "topic_relevance", None),
            ),
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
            "difficulty": float(0.5 if raw_difficulty is None else raw_difficulty),
            "selection_contract_version": (
                getattr(reel_obj, "selection_contract_version", None)
            ),
            "boundary_confidence": getattr(
                reel_obj, "boundary_confidence", None
            ),
            "is_standalone": bool(getattr(reel_obj, "is_standalone", True)),
            "chain_id": str(getattr(reel_obj, "chain_id", "") or ""),
            "chain_position": float(
                getattr(reel_obj, "chain_position", 0.0) or 0.0
            ),
            "selection_candidate_id": str(
                getattr(reel_obj, "selection_candidate_id", "") or ""
            ),
            "prerequisite_ids": list(
                getattr(reel_obj, "prerequisite_ids", None) or []
            ),
            "_selection_quality_floor": getattr(
                reel_obj, "selection_quality_floor", None
            ),
            "_selection_quality_mean": getattr(
                reel_obj, "selection_quality_mean", None
            ),
            "_selection_topic_relevance": getattr(
                reel_obj, "selection_topic_relevance", None
            ),
            "_selection_source_rank": int(
                getattr(reel_obj, "selection_source_rank", 0) or 0
            ),
            "_selection_intent_role": str(
                getattr(reel_obj, "selection_intent_role", "primary") or "primary"
            ).strip().lower(),
            "_selection_intent_coverage": self._selection_number(
                getattr(reel_obj, "selection_intent_coverage", 1.0), 1.0
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
        if all(
            str(reel.get("selection_contract_version") or "").strip()
            in {
                "quality_silence_v3",
                "quality_silence_v4",
                "quality_silence_v5",
                "quality_silence_v6",
                "quality_silence_v7",
                "quality_silence_v8",
                "quality_silence_v9",
                "quality_silence_v10",
                "quality_silence_v11",
                "quality_silence_v12",
                "quality_silence_v13",
                "quality_silence_v14",
                "quality_silence_v15",
                "quality_silence_v16",
                "quality_silence_v17",
                "quality_silence_v18",
                "quality_silence_v19",
                "quality_silence_v20",
                "quality_silence_v21",
                "quality_silence_v22",
                "quality_silence_v23",
                "quality_silence_v24",
                "quality_silence_v25",
                "quality_silence_v26",
                "quality_silence_v27",
                "quality_silence_v28",
                "quality_silence_v29",
                "quality_silence_v30",
            }
            for reel in generated
        ):
            ordered = sorted(
                enumerate(generated),
                key=lambda pair: self._selection_contract_sort_key(
                    pair[1], input_order=pair[0]
                ),
            )
            return [
                {
                    key: value
                    for key, value in reel.items()
                    if not key.startswith("_selection_")
                }
                for _index, reel in ordered[:num_reels]
            ]
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
        """Interleave sources while preserving chronological order per source."""
        if len(reels) <= 1:
            return list(reels)
        groups: dict[str, list[dict[str, Any]]] = {}
        for reel in reels:
            video_id = str(reel.get("video_id") or "").strip()
            groups.setdefault(video_id, []).append(reel)
        for group in groups.values():
            group.sort(key=lambda row: float(row.get("t_start") or 0.0))
        result: list[dict[str, Any]] = []
        last_video = ""
        last_concept = ""
        while groups:
            heads = [(video_id, group[0]) for video_id, group in groups.items() if group]
            candidates = [pair for pair in heads if pair[0] != last_video] or heads
            different_concept = [
                pair for pair in candidates
                if str(pair[1].get("concept_id") or "") != last_concept
            ]
            candidates = different_concept or candidates
            video_id, chosen = max(
                candidates, key=lambda pair: self._generation_result_score(pair[1])
            )
            groups[video_id].pop(0)
            if not groups[video_id]:
                groups.pop(video_id)
            result.append(chosen)
            last_video = video_id
            last_concept = str(chosen.get("concept_id") or "")
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

    def _semantic_embeddings_available(self) -> bool:
        available = getattr(self.embedding_service, "semantic_available", False)
        return available if isinstance(available, bool) else False






    def _clean_query_text(self, value: str) -> str:
        return " ".join(str(value or "").split()).strip()

    def _normalize_query_key(self, value: str) -> str:
        cleaned = self._clean_query_text(value).lower()
        tokens = re.findall(r"[a-z0-9\+#]+", cleaned)
        return " ".join(tokens)

    def _concept_topic_query(
        self,
        concept_row: dict,
        *,
        parent_topic: str | None = None,
    ) -> str:
        """Return a search-engine topic string for a concept row: its clean title.

        Topic-material concepts and search terms come from the shared cached AI
        query plan; document concepts retain their source-grounded titles. Short
        leaf names keep the parent material topic because title-cased acronyms such
        as ``Atp`` are otherwise ambiguous outside their source material.
        """
        title = self._clean_query_text(str(concept_row.get("title") or ""))
        topic = title
        search_terms = concept_row.get("_search_terms")
        if isinstance(search_terms, list):
            for raw_term in search_terms:
                term = self._clean_query_text(str(raw_term or ""))
                if term:
                    topic = term
                    break
        parent = self._clean_query_text(str(parent_topic or ""))
        if not topic or not parent:
            return topic
        topic_tokens = self._normalize_query_key(topic).split()
        parent_tokens = self._normalize_query_key(parent).split()
        if not topic_tokens or not parent_tokens or set(parent_tokens).issubset(topic_tokens):
            return topic
        if self._is_short_leaf_topic(topic):
            return f"{topic} in {parent}"
        return topic

    @staticmethod
    def _is_short_leaf_topic(value: str) -> bool:
        tokens = re.findall(r"[A-Za-z0-9+#]+", str(value or ""))
        if len(tokens) != 1:
            return False
        compact = re.sub(r"[^A-Za-z0-9]", "", tokens[0])
        if not compact or len(compact) > 5:
            return False
        letters = [character for character in compact if character.isalpha()]
        if len(letters) < 2:
            return False
        # Preserve explicit acronym casing (ATP, DNA, H2O, mRNA). Concept titles
        # are title-cased before persistence, so also recognize compact three-
        # letter forms such as ``Atp`` without treating ordinary short words such
        # as Cell, Atom, Force, Logic, or Rome as ambiguous acronyms.
        if sum(character.isupper() for character in letters) >= 2:
            return True
        return (
            len(letters) <= 3
            and compact[0].isupper()
            and all(character.islower() for character in letters[1:])
        )

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

    def _expand_synonyms_via_llm(
        self,
        terms: list[str],
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[str] | None:
        """Fix K: Use LLM to generate richer query synonyms in slow mode."""
        _raise_if_clip_cancelled(should_cancel)
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
                should_cancel=should_cancel,
            ) or "").strip()
            _raise_if_clip_cancelled(should_cancel)
            synonyms = [line.strip() for line in content.split("\n") if line.strip() and len(line.strip()) > 2]
            return synonyms[:8] if synonyms else None
        except _ClipEngineCancellationError:
            raise
        except Exception:
            return None

    def _broaden_concept_queries_via_llm(
        self,
        *,
        concept_title: str,
        keywords: list[str],
        summary: str,
        subject_tag: str | None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[str] | None:
        """Generate broader parent topics for niche/specific concepts that lack hardcoded expansions."""
        _raise_if_clip_cancelled(should_cancel)
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
                should_cancel=should_cancel,
            )
            _raise_if_clip_cancelled(should_cancel)
            if not content:
                return None
            terms = [line.strip() for line in content.split("\n") if line.strip() and len(line.strip()) > 2]
            return terms[:8] if terms else None
        except _ClipEngineCancellationError:
            raise
        except Exception:
            return None

    def _expand_controlled_synonyms(
        self,
        terms: list[str],
        fast_mode: bool = True,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[str]:
        _raise_if_clip_cancelled(should_cancel)
        # Fix K: Use LLM for synonym expansion in slow mode
        if not fast_mode and not self.serverless_mode:
            llm_synonyms = self._expand_synonyms_via_llm(
                terms, should_cancel=should_cancel
            )
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
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[str]:
        _raise_if_clip_cancelled(should_cancel)
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

        for index, term in enumerate(
            self._expand_controlled_synonyms(
                [concept_title, *keywords],
                fast_mode=fast_mode,
                should_cancel=should_cancel,
            )[:10]
        ):
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
                should_cancel=should_cancel,
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
        min_relevance_threshold = float(
            getattr(self._generation_state, "min_relevance_threshold", 0.0)
        )
        if min_relevance_threshold > 0.0:
            min_discovery = max(
                min_discovery,
                min_discovery + 0.06 * min_relevance_threshold,
            )
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
        if concept_embedding is None or not self._semantic_embeddings_available():
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
        if fast_mode or not self._semantic_embeddings_available():
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

    def _topic_expansion_from_query_plan(
        self,
        conn,
        *,
        subject_tag: str,
        should_cancel: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        literal = self._clean_query_text(subject_tag)
        fallback = {
            "canonical_topic": literal,
            "aliases": [],
            "subtopics": [],
            "related_terms": [],
        }
        if not literal:
            return fallback
        try:
            plan = build_search_query_plan(
                conn,
                literal_query=literal,
                should_cancel=should_cancel,
            )
        except _ClipEngineCancellationError:
            raise
        except Exception as exc:
            logger.warning(
                "topic query plan unavailable during reel generation error_type=%s",
                type(exc).__name__,
            )
            return fallback
        _raise_if_clip_cancelled(should_cancel)
        if plan.ai_status != "validated":
            return fallback
        return plan.as_topic_expansion()

    def _bootstrap_topic_retrieval_concepts(
        self,
        *,
        conn,
        concepts: list[dict[str, Any]],
        subject_tag: str,
        material_id: str,
        expansion: dict[str, Any] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[dict[str, Any]]:
        _raise_if_clip_cancelled(should_cancel)
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
            self._bootstrap_topic_keywords(
                root_concept,
                subject_tag=subject_tag,
                conn=conn,
                expansion=expansion,
                should_cancel=should_cancel,
            )
        )
        return [root_concept]

    def _bootstrap_topic_keywords(
        self,
        concept: dict[str, Any],
        *,
        subject_tag: str,
        conn=None,
        expansion: dict[str, Any] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[str]:
        expansion = expansion or self._topic_expansion_from_query_plan(
            conn,
            subject_tag=subject_tag,
            should_cancel=should_cancel,
        )
        _raise_if_clip_cancelled(should_cancel)
        candidates = [
            self._clean_query_text(subject_tag),
            self._clean_query_text(str(concept.get("title") or "")),
            self._clean_query_text(str(expansion.get("canonical_topic") or "")),
            *[
                self._clean_query_text(str(term or ""))
                for key in ("aliases", "subtopics", "related_terms")
                for term in (expansion.get(key) or [])
            ],
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
        should_cancel: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        _raise_if_clip_cancelled(should_cancel)
        return self._topic_expansion_from_query_plan(
            conn,
            subject_tag=subject_tag,
            should_cancel=should_cancel,
        )

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
        expansion: dict[str, Any] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[dict[str, Any]]:
        _raise_if_clip_cancelled(should_cancel)
        expansion = expansion or self._topic_expansion_from_query_plan(
            conn,
            subject_tag=subject_tag,
            should_cancel=should_cancel,
        )
        expansion_terms = self._topic_expansion_terms(
            expansion=expansion,
            subject_tag=subject_tag,
            limit=8,
        )

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
        _raise_if_clip_cancelled(should_cancel)
        upsert(conn, "concepts", root_row)
        created_concepts.append(root_row)

        for term in expansion_terms[:5]:
            _raise_if_clip_cancelled(should_cancel)
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
            _raise_if_clip_cancelled(should_cancel)
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
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[dict[str, Any]]:
        """Attach expansion context without rewriting learner concept records.

        Earlier versions recycled concept IDs by changing their titles to whichever
        static/LLM expansion happened to win that run. Besides corrupting progress,
        that made a concept mean different things across generations. Database rows
        stay untouched, while the generation working set is limited to the literal
        topic and terms in the current validated AI plan.
        """
        _raise_if_clip_cancelled(should_cancel)
        canonical = self._clean_query_text(
            str(expansion.get("canonical_topic") or subject_tag)
        ) or self._clean_query_text(subject_tag)
        subject_key = self._normalize_query_key(subject_tag)
        allowed_concept_keys = {
            self._normalize_query_key(value)
            for value in [
                subject_tag,
                canonical,
                *[
                    str(term or "")
                    for key in ("aliases", "subtopics", "related_terms")
                    for term in (expansion.get(key) or [])
                ],
            ]
            if self._normalize_query_key(value)
        }
        working = [
            dict(concept)
            for concept in concepts
            if self._normalize_query_key(str(concept.get("title") or ""))
            in allowed_concept_keys
        ]
        if not working:
            return working

        root_index = next(
            (
                index
                for index, concept in enumerate(working)
                if self._normalize_query_key(str(concept.get("title") or ""))
                == subject_key
            ),
            0,
        )

        typed_terms: list[tuple[str, str]] = [(canonical, "corrected_topic")]
        for key, kind in (
            ("aliases", "alias"),
            ("subtopics", "expansion"),
            ("related_terms", "related"),
        ):
            for raw_term in expansion.get(key) or []:
                typed_terms.append((self._clean_query_text(str(raw_term or "")), kind))

        seen_global: set[tuple[str, str]] = set()
        normalized_terms: list[tuple[str, str]] = []
        for term, kind in typed_terms:
            normalized = self._normalize_query_key(term)
            identity = (normalized, kind)
            if not term or not normalized or identity in seen_global:
                continue
            seen_global.add(identity)
            normalized_terms.append((term, kind))

        try:
            execute_modify(
                conn,
                "DELETE FROM concept_search_terms WHERE material_id = ?",
                (material_id,),
            )
        except Exception as exc:
            # Hand-built legacy test schemas may omit this rollout table. Real
            # databases are migrated by init_db before generation begins.
            if "concept_search_terms" not in str(exc).lower():
                raise

        for index, concept in enumerate(working):
            _raise_if_clip_cancelled(should_cancel)
            concept_id = str(concept.get("id") or "").strip()
            title = self._clean_query_text(str(concept.get("title") or ""))
            if not concept_id or not title:
                continue

            contextual = title
            title_tokens = set(self._normalize_query_key(title).split())
            context_tokens = set(self._normalize_query_key(canonical).split())
            if canonical and not context_tokens.issubset(title_tokens):
                contextual = self._clean_query_text(f"{title} {canonical}")

            persisted_terms: list[tuple[str, str]] = [(contextual, "material_context")]
            if index == root_index:
                persisted_terms.extend(normalized_terms)

            seen_concept_terms: set[str] = set()
            concept["_search_terms"] = []
            for term, kind in persisted_terms:
                normalized = self._normalize_query_key(term)
                if not term or not normalized or normalized in seen_concept_terms:
                    continue
                seen_concept_terms.add(normalized)
                concept["_search_terms"].append(term)
                row_id = hashlib.sha256(
                    f"{concept_id}|{kind}|{normalized}".encode("utf-8")
                ).hexdigest()
                try:
                    upsert(
                        conn,
                        "concept_search_terms",
                        {
                            "id": row_id,
                            "concept_id": concept_id,
                            "material_id": material_id,
                            "term": term,
                            "term_kind": kind,
                            "created_at": now_iso(),
                        },
                    )
                except Exception as exc:
                    if "concept_search_terms" not in str(exc).lower():
                        raise

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
            if (
                not cleaned
                or not normalized
                or normalized == subject_key
                or normalized in seen
            ):
                return False
            seen.add(normalized)
            ordered.append(cleaned)
            return True

        for key in ("subtopics", "aliases", "related_terms"):
            for raw_value in expansion.get(key) or []:
                if add_term(str(raw_value or "")) and len(ordered) >= limit:
                    return ordered
        add_term(str(expansion.get("canonical_topic") or ""))
        return ordered[:limit]

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
        if concept_embedding is not None and self._semantic_embeddings_available():
            try:
                text_embedding = self.embedding_service.embed_texts(conn, [cleaned])[0]
                embedding_sim = float(text_embedding @ concept_embedding.astype(np.float32))
            except Exception:
                embedding_sim = 0.0

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
        start = Decimal(str(float(t_start))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        end = Decimal(str(float(t_end))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        return f"{video_id}:{start}:{end}"

    def _order_concepts(
        self,
        conn,
        material_id: str,
        concepts: list[dict[str, Any]],
        learner_id: str = LEGACY_LEARNER_ID,
    ) -> list[dict[str, Any]]:
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
                LEFT JOIN reel_feedback f
                  ON f.reel_id = r.id
                 AND f.learner_id = ?
                WHERE r.material_id = ?
                GROUP BY r.concept_id
                """,
                (str(learner_id or LEGACY_LEARNER_ID), material_id),
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

    def update_level_adjustment(
        self, conn, material_id: str, learner_id: str = LEGACY_LEARNER_ID,
    ) -> float:
        """Recompute learner-global drift from the latest 12 mastery responses."""
        from .knowledge_level import ADJUSTMENT_BOUND

        progress = self.learner_progress(conn, material_id, learner_id)
        reset_at = str(progress.get("difficulty_reset_at") or "")
        feedback_rows = fetch_all(
            conn,
            """
            SELECT f.helpful, f.confusing, r.concept_id,
                   f.mastery_updated_at AS event_at
            FROM reel_feedback f
            JOIN reels r ON r.id = f.reel_id
            WHERE f.learner_id = ?
              AND r.material_id = ?
              AND f.mastery_updated_at IS NOT NULL
              AND f.mastery_updated_at > ?
              AND (f.helpful <> 0 OR f.confusing <> 0)
            ORDER BY f.mastery_updated_at DESC
            LIMIT ?
            """,
            (learner_id, material_id, reset_at, self.GLOBAL_FEEDBACK_WINDOW),
        )
        try:
            assessment_rows = fetch_all(
                conn,
                """
                SELECT concept_id, adjustment, created_at AS event_at
                FROM assessment_concept_outcomes
                WHERE learner_id = ?
                  AND material_id = ?
                  AND created_at > ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (learner_id, material_id, reset_at, self.GLOBAL_FEEDBACK_WINDOW),
            )
        except sqlite3.OperationalError as exc:
            if "no such table: assessment_concept_outcomes" not in str(exc):
                raise
            assessment_rows = []
        rows = [
            {
                **row,
                "signal": (
                    self.GOT_IT_CONCEPT_STEP
                    if int(row.get("helpful") or 0) > 0
                    else 0.0
                )
                - (
                    self.NEED_HELP_CONCEPT_STEP
                    if int(row.get("confusing") or 0) > 0
                    else 0.0
                ),
                "source": "manual",
            }
            for row in feedback_rows
        ]
        for row in assessment_rows:
            adjustment = float(row.get("adjustment") or 0.0)
            rows.append(
                {
                    **row,
                    "signal": adjustment,
                    "source": "assessment",
                }
            )
        rows.sort(key=lambda row: str(row.get("event_at") or ""), reverse=True)
        rows = rows[: self.GLOBAL_FEEDBACK_WINDOW]
        assessment_present = bool(assessment_rows)
        manual_rows = [row for row in rows if row.get("source") == "manual"]
        manual_concepts = {
            str(row.get("concept_id") or "") for row in manual_rows
        }
        if (
            not assessment_present
            and (
                len(manual_rows) < self.GLOBAL_FEEDBACK_MIN_ROWS
                or len(manual_concepts) < 2
            )
        ):
            adjustment = 0.0
        else:
            signal = sum(float(row.get("signal") or 0.0) for row in rows)
            adjustment = max(-ADJUSTMENT_BOUND, min(ADJUSTMENT_BOUND, signal))
        execute_modify(
            conn,
            """
            UPDATE learner_material_progress
            SET global_adjustment = ?, updated_at = ?
            WHERE learner_id = ? AND material_id = ?
            """,
            (adjustment, now_iso(), learner_id, material_id),
        )
        return adjustment

    def _learner_adaptation_context(
        self, conn, material_id: str, learner_id: str,
    ) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[str, Any] | None, float]:
        progress = self.learner_progress(conn, material_id, learner_id)
        reset_at = str(progress.get("difficulty_reset_at") or "")
        rows = fetch_all(
            conn,
            """
            SELECT f.reel_id, f.helpful, f.confusing, f.mastery_updated_at,
                   r.concept_id, r.video_id, r.difficulty
            FROM reel_feedback f
            JOIN reels r ON r.id = f.reel_id
            WHERE f.learner_id = ? AND r.material_id = ?
            """,
            (learner_id, material_id),
        )
        try:
            assessment_rows = fetch_all(
                conn,
                """
                SELECT session_id, concept_id, adjustment, source_reel_id AS reel_id,
                       source_video_id AS video_id, source_difficulty AS difficulty,
                       created_at AS mastery_updated_at
                FROM assessment_concept_outcomes
                WHERE learner_id = ? AND material_id = ?
                """,
                (learner_id, material_id),
            )
        except sqlite3.OperationalError as exc:
            if "no such table: assessment_concept_outcomes" not in str(exc):
                raise
            assessment_rows = []
        coverage: dict[str, dict[str, float]] = {}
        post_reset: dict[str, list[float]] = {}
        assessment_adjustments: dict[str, float] = {}
        mastery_rows: list[dict[str, Any]] = []
        latest_manual_at = ""
        post_reset_assessment_rows: list[dict[str, Any]] = []
        for row in rows:
            concept_id = str(row.get("concept_id") or "")
            if not concept_id:
                continue
            bucket = coverage.setdefault(concept_id, {"helpful": 0.0, "confusing": 0.0})
            bucket["helpful"] += 1.0 if int(row.get("helpful") or 0) > 0 else 0.0
            bucket["confusing"] += 1.0 if int(row.get("confusing") or 0) > 0 else 0.0
            mastery_at = str(row.get("mastery_updated_at") or "")
            if not mastery_at or mastery_at <= reset_at:
                continue
            values = post_reset.setdefault(concept_id, [0.0, 0.0])
            values[0] += 1.0 if int(row.get("helpful") or 0) > 0 else 0.0
            values[1] += 1.0 if int(row.get("confusing") or 0) > 0 else 0.0
            if int(row.get("helpful") or 0) > 0 or int(row.get("confusing") or 0) > 0:
                mastery_rows.append({**row, "mastery_source": "manual"})
                latest_manual_at = max(latest_manual_at, mastery_at)
        for row in assessment_rows:
            concept_id = str(row.get("concept_id") or "")
            if not concept_id:
                continue
            adjustment = float(row.get("adjustment") or 0.0)
            bucket = coverage.setdefault(concept_id, {"helpful": 0.0, "confusing": 0.0})
            if adjustment > 0:
                bucket["helpful"] += 1.0
            elif adjustment < 0:
                bucket["confusing"] += 1.0
            mastery_at = str(row.get("mastery_updated_at") or "")
            if not mastery_at or mastery_at <= reset_at:
                continue
            assessment_adjustments[concept_id] = (
                assessment_adjustments.get(concept_id, 0.0) + adjustment
            )
            mastery_row = {
                **row,
                "helpful": 1 if adjustment > 0 else 0,
                "confusing": 1 if adjustment < 0 else 0,
                "mastery_source": "assessment",
            }
            mastery_rows.append(mastery_row)
            post_reset_assessment_rows.append(mastery_row)
        adjustments: dict[str, float] = {}
        for concept_id in set(post_reset) | set(assessment_adjustments):
            values = post_reset.get(concept_id, [0.0, 0.0])
            combined = (
                self.GOT_IT_CONCEPT_STEP * values[0]
                - self.NEED_HELP_CONCEPT_STEP * values[1]
                + assessment_adjustments.get(concept_id, 0.0)
            )
            adjustments[concept_id] = max(
                -self.CONCEPT_ADJUSTMENT_BOUND,
                min(self.CONCEPT_ADJUSTMENT_BOUND, combined),
            )
        latest = max(
            mastery_rows,
            key=lambda row: (
                str(row.get("mastery_updated_at") or ""),
                int(row.get("confusing") or 0),
                int(row.get("helpful") or 0),
                str(row.get("concept_id") or ""),
            ),
            default=None,
        )
        assessment_remediations: list[dict[str, Any]] = []
        if post_reset_assessment_rows:
            newest_assessment = max(
                post_reset_assessment_rows,
                key=lambda row: (
                    str(row.get("mastery_updated_at") or ""),
                    str(row.get("session_id") or ""),
                ),
            )
            newest_assessment_at = str(newest_assessment.get("mastery_updated_at") or "")
            newest_session_id = str(newest_assessment.get("session_id") or "")
            if newest_assessment_at >= latest_manual_at:
                assessment_remediations = sorted(
                    (
                        row
                        for row in post_reset_assessment_rows
                        if str(row.get("session_id") or "") == newest_session_id
                        and int(row.get("confusing") or 0) > 0
                    ),
                    key=lambda row: str(row.get("concept_id") or ""),
                )
        if latest is not None and assessment_remediations:
            latest = {**latest, "assessment_remediations": assessment_remediations}
        level_target = effective_level_target(
            progress.get("selected_level"), progress.get("global_adjustment")
        )
        return coverage, adjustments, latest, level_target

    @staticmethod
    def _difficulty(item: dict[str, Any]) -> float:
        value = item.get("difficulty")
        if value is None:
            return 0.5
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.5

    @classmethod
    def _selection_difficulty_stage(cls, item: dict[str, Any]) -> int:
        """Return the selector's non-overlapping beginner/intermediate/advanced bin."""
        difficulty = cls._difficulty(item)
        if difficulty < 0.34:
            return 0
        if difficulty < 0.67:
            return 1
        return 2

    @classmethod
    def select_difficulty_inventory(
        cls,
        items: list[dict[str, Any]],
        knowledge_level: str | None,
    ) -> list[dict[str, Any]]:
        """Keep the requested difficulty bin, or the nearest valid bin if empty."""
        staged = [
            item
            for item in items
            if str(
                item.get("_selection_contract_version")
                or item.get("selection_contract_version")
                or ""
            ).strip()
            in cls.DIFFICULTY_FALLBACK_CONTRACTS
        ]
        if not staged:
            return list(items)
        target_stage = {
            "beginner": 0,
            "intermediate": 1,
            "advanced": 2,
        }.get(str(knowledge_level or "").strip().lower(), 0)
        available_stages = {
            cls._selection_difficulty_stage(item) for item in staged
        }
        chosen_stage = min(
            available_stages,
            key=lambda stage: (abs(stage - target_stage), stage),
        )
        return [
            item
            for item in items
            if str(
                item.get("_selection_contract_version")
                or item.get("selection_contract_version")
                or ""
            ).strip()
            not in cls.DIFFICULTY_FALLBACK_CONTRACTS
            or cls._selection_difficulty_stage(item) == chosen_stage
        ]

    @staticmethod
    def _selection_number(value: Any, default: float = 0.0) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = float(default)
        return max(0.0, min(1.0, parsed))

    @classmethod
    def _selection_contract_sort_key(
        cls,
        item: dict[str, Any],
        *,
        input_order: int = 0,
    ) -> tuple[int, int, float, float, float, float, int, float, int]:
        """Rank value within a difficulty stage, with deterministic fallbacks."""
        compatibility_score = cls._selection_number(
            item.get("_selection_content_score"), 0.0
        )
        raw_informativeness = item.get("_selection_informativeness")
        if raw_informativeness is None:
            raw_informativeness = item.get("informativeness")
        raw_topic_relevance = item.get("_selection_topic_relevance")
        if raw_topic_relevance is None:
            raw_topic_relevance = item.get("topic_relevance")
        raw_importance = item.get("_selection_educational_importance")
        raw_quality_mean = item.get("_selection_quality_mean")
        if (
            raw_importance is None
            and raw_quality_mean is not None
            and raw_informativeness is not None
            and raw_topic_relevance is not None
        ):
            raw_importance = (
                3.0 * cls._selection_number(raw_quality_mean)
                - cls._selection_number(raw_informativeness)
                - cls._selection_number(raw_topic_relevance)
            )
        selector_values = [
            cls._selection_number(value)
            for value in (
                raw_informativeness,
                raw_topic_relevance,
                raw_importance,
            )
            if value is not None
        ]
        if len(selector_values) == 3:
            quality_floor = min(selector_values)
            quality_mean = sum(selector_values) / len(selector_values)
        else:
            quality_floor = cls._selection_number(
                item.get("_selection_quality_floor"), compatibility_score
            )
            quality_mean = cls._selection_number(
                item.get("_selection_quality_mean"), quality_floor
            )
        topic_relevance = cls._selection_number(
            raw_topic_relevance, compatibility_score
        )
        try:
            source_rank = max(0, int(item.get("_selection_source_rank") or 0))
        except (TypeError, ValueError):
            source_rank = 0
        intent_role = str(
            item.get("_selection_intent_role")
            or item.get("intent_role")
            or "primary"
        ).strip().lower()
        intent_coverage = cls._selection_number(
            item.get("_selection_intent_coverage", item.get("intent_coverage")),
            1.0,
        )
        return (
            cls._selection_difficulty_stage(item),
            0 if intent_role == "primary" else 1,
            -intent_coverage,
            -quality_floor,
            -quality_mean,
            -topic_relevance,
            source_rank,
            float(item.get("t_start") or 0.0),
            int(input_order),
        )

    @classmethod
    def _selection_metadata(
        cls,
        value: Any,
        *,
        t_start: object | None = None,
        t_end: object | None = None,
    ) -> dict[str, Any]:
        """Read the versioned selection contract stored with a persisted reel."""
        if isinstance(value, dict):
            parsed = dict(value)
        else:
            try:
                parsed = json.loads(str(value or "{}"))
            except (TypeError, json.JSONDecodeError):
                return {}
        if not isinstance(parsed, dict):
            return {}
        nested = parsed.get("selection_metadata") or parsed.get("selection")
        if isinstance(nested, dict):
            parsed = {**parsed, **nested}
        version = str(parsed.get("selection_contract_version") or "").strip()
        if not version or version.lower() in {"0", "legacy", "none"}:
            # Acoustic eligibility is an operational serving guard, not a
            # selector-version feature. Preserve it for legacy cached selector
            # rows so an unavailable deferred clip can never leak into feed.
            operational: dict[str, Any] = {}
            if "surface_eligible" in parsed:
                surface_eligible = parsed.get("surface_eligible")
                if isinstance(surface_eligible, str):
                    surface_eligible = surface_eligible.strip().lower() in {
                        "1", "true", "yes", "on",
                    }
                operational["_selection_surface_eligible"] = bool(surface_eligible)
            operational["_selection_boundary_status"] = str(
                parsed.get("boundary_status") or ""
            ).strip().lower()
            operational["_selection_acoustic_verified"] = (
                persisted_boundary_is_verified(parsed)
            )
            operational["_selection_boundary_usable"] = (
                persisted_boundary_is_usable(
                    parsed, t_start=t_start, t_end=t_end
                )
            )
            operational["_selection_surface_reason"] = str(
                parsed.get("surface_reason") or ""
            ).strip().lower()
            return operational

        prerequisites = parsed.get("prerequisite_ids")
        if prerequisites is None:
            prerequisites = parsed.get("prerequisite_reel_ids")
        if prerequisites is None:
            prerequisites = parsed.get("prerequisite_candidate_ids")
        if not isinstance(prerequisites, list):
            prerequisites = []

        standalone = parsed.get("is_standalone", parsed.get("standalone", False))
        if isinstance(standalone, str):
            standalone = standalone.strip().lower() in {"1", "true", "yes", "on"}

        def selection_bool(key: str, default: bool) -> bool:
            value = parsed.get(key, default)
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)

        metadata: dict[str, Any] = {
            "_selection_contract_version": version,
            "_selection_boundary_confidence": cls._selection_number(
                parsed.get("boundary_confidence"), 0.0
            ),
            "_selection_self_contained": selection_bool(
                "self_contained", False
            ),
            "_selection_is_standalone": bool(standalone),
            "_selection_chain_id": str(parsed.get("chain_id") or "").strip(),
            "_selection_prerequisite_ids": [
                str(item).strip() for item in prerequisites if str(item).strip()
            ],
            "_selection_candidate_id": str(
                parsed.get("selection_candidate_id") or parsed.get("candidate_id") or ""
            ).strip(),
            "_selection_intent_role": str(
                parsed.get("intent_role") or "primary"
            ).strip().lower(),
            "_selection_intent_coverage": cls._selection_number(
                parsed.get("intent_coverage"), 1.0
            ),
        }
        for source_key, metadata_key in (
            ("quality_floor", "_selection_quality_floor"),
            ("quality_mean", "_selection_quality_mean"),
            ("informativeness", "_selection_informativeness"),
            ("topic_relevance", "_selection_topic_relevance"),
            ("educational_importance", "_selection_educational_importance"),
        ):
            if parsed.get(source_key) is not None:
                metadata[metadata_key] = cls._selection_number(parsed.get(source_key))
        if "surface_eligible" in parsed:
            surface_eligible = parsed.get("surface_eligible")
            if isinstance(surface_eligible, str):
                surface_eligible = surface_eligible.strip().lower() in {
                    "1", "true", "yes", "on",
                }
            metadata["_selection_surface_eligible"] = bool(surface_eligible)
        metadata["_selection_boundary_status"] = str(
            parsed.get("boundary_status") or ""
        ).strip().lower()
        metadata["_selection_acoustic_verified"] = persisted_boundary_is_verified(
            parsed
        )
        metadata["_selection_boundary_usable"] = persisted_boundary_is_usable(
            parsed, t_start=t_start, t_end=t_end
        )
        metadata["_selection_surface_reason"] = str(
            parsed.get("surface_reason") or ""
        ).strip().lower()
        metadata["_selection_directly_teaches_topic"] = selection_bool(
            "directly_teaches_topic",
            version not in {
                "quality_silence_v2",
                "quality_silence_v3",
                "quality_silence_v4",
                "quality_silence_v5",
                "quality_silence_v6",
                "quality_silence_v7",
                "quality_silence_v8",
                "quality_silence_v9",
                "quality_silence_v10",
                "quality_silence_v11",
                "quality_silence_v12",
                "quality_silence_v13",
                "quality_silence_v14",
                "quality_silence_v15",
                "quality_silence_v16",
                "quality_silence_v17",
                "quality_silence_v18",
                "quality_silence_v19",
                "quality_silence_v20",
                "quality_silence_v21",
                "quality_silence_v22",
                "quality_silence_v23",
                "quality_silence_v24",
                "quality_silence_v25",
                "quality_silence_v26",
                "quality_silence_v27",
                "quality_silence_v28",
                "quality_silence_v29",
                "quality_silence_v30",
            },
        )
        metadata["_selection_substantive"] = selection_bool(
            "substantive",
            version not in {
                "quality_silence_v2",
                "quality_silence_v3",
                "quality_silence_v4",
                "quality_silence_v5",
                "quality_silence_v6",
                "quality_silence_v7",
                "quality_silence_v8",
                "quality_silence_v9",
                "quality_silence_v10",
                "quality_silence_v11",
                "quality_silence_v12",
                "quality_silence_v13",
                "quality_silence_v14",
                "quality_silence_v15",
                "quality_silence_v16",
                "quality_silence_v17",
                "quality_silence_v18",
                "quality_silence_v19",
                "quality_silence_v20",
                "quality_silence_v21",
                "quality_silence_v22",
                "quality_silence_v23",
                "quality_silence_v24",
                "quality_silence_v25",
                "quality_silence_v26",
                "quality_silence_v27",
                "quality_silence_v28",
                "quality_silence_v29",
                "quality_silence_v30",
            },
        )
        metadata["_selection_factually_grounded"] = selection_bool(
            "factually_grounded", False
        )
        metadata["_selection_topic_evidence_quote"] = str(
            parsed.get("topic_evidence_quote") or ""
        ).strip()
        metadata["_selection_uncertainty"] = str(
            parsed.get("uncertainty") or "low"
        ).strip().lower()
        metadata["_selection_deferred_level"] = selection_bool("deferred_level", False)
        metadata["_selection_speech_corridor_verified"] = selection_bool(
            "speech_corridor_verified", False
        )
        metadata["_selection_transcript_artifact_key"] = str(
            parsed.get("transcript_artifact_key") or ""
        ).strip()
        raw_caption_cues = parsed.get("selection_caption_cues")
        metadata["_selection_caption_cues"] = (
            [dict(cue) for cue in raw_caption_cues if isinstance(cue, dict)]
            if isinstance(raw_caption_cues, list)
            else []
        )
        try:
            metadata["_selection_source_rank"] = max(
                0, int(parsed.get("source_rank") or 0)
            )
        except (TypeError, ValueError):
            metadata["_selection_source_rank"] = 0
        try:
            metadata["_selection_chain_position"] = float(
                parsed.get("chain_position") or 0.0
            )
        except (TypeError, ValueError):
            metadata["_selection_chain_position"] = 0.0

        if parsed.get("content_score") is not None:
            metadata["_selection_content_score"] = cls._selection_number(
                parsed.get("content_score")
            )
        else:
            topic_relevance = cls._selection_number(parsed.get("topic_relevance"), 0.0)
            importance = cls._selection_number(
                parsed.get("educational_importance", parsed.get("importance")), 0.0
            )
            informativeness = cls._selection_number(parsed.get("informativeness"), 0.0)
            metadata["_selection_content_score"] = (
                0.45 * topic_relevance
                + 0.35 * importance
                + 0.20 * informativeness
            )
        return metadata

    @staticmethod
    def _has_selection_contract(item: dict[str, Any]) -> bool:
        return bool(str(item.get("_selection_contract_version") or "").strip())

    def _selection_contract_order(
        self,
        items: list[dict[str, Any]],
        *,
        level_target: float,
        concept_adjustments: dict[str, float],
        previous_video_id: str,
    ) -> list[dict[str, Any]]:
        """Priority topological sort for confidence-gated selection rows."""
        nodes: dict[str, dict[str, Any]] = {}
        aliases: dict[str, str] = {}
        for index, raw in enumerate(items):
            item = dict(raw)
            reel_id = str(item.get("reel_id") or "").strip()
            node_id = reel_id or f"selection-node-{index}"
            item["_selection_node_id"] = node_id
            item["_selection_input_order"] = index
            item["score"] = self._selection_number(
                item.get("_selection_quality_floor"),
                self._selection_number(item.get("_selection_content_score"), 0.0),
            )
            nodes[node_id] = item
            aliases[node_id] = node_id
            candidate_id = str(item.get("_selection_candidate_id") or "").strip()
            if candidate_id:
                aliases[candidate_id] = node_id

        dependencies: dict[str, set[str]] = {node_id: set() for node_id in nodes}
        for node_id, item in nodes.items():
            raw_prerequisites = item.get("_selection_prerequisite_ids") or []
            for prerequisite in raw_prerequisites:
                prerequisite_id = str(prerequisite or "").strip()
                if not prerequisite_id:
                    continue
                dependencies[node_id].add(
                    aliases.get(prerequisite_id, f"missing:{prerequisite_id}")
                )

        chains: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
        for node_id, item in nodes.items():
            chain_id = str(item.get("_selection_chain_id") or "").strip()
            if chain_id:
                chains[chain_id].append((node_id, item))
        for chain in chains.values():
            chain.sort(
                key=lambda pair: (
                    float(pair[1].get("_selection_chain_position") or 0.0),
                    float(pair[1].get("t_start") or 0.0),
                    pair[0],
                )
            )
            for previous, current in zip(chain, chain[1:]):
                dependencies[current[0]].add(previous[0])

        remaining = set(nodes)
        satisfied: set[str] = set()
        ordered: list[dict[str, Any]] = []
        last_video = str(previous_video_id or "")
        while remaining:
            eligible = [
                node_id
                for node_id in remaining
                if dependencies[node_id].issubset(satisfied)
            ]
            if not eligible:
                break
            if not ordered:
                eligible = [
                    node_id
                    for node_id in eligible
                    if bool(nodes[node_id].get("_selection_is_standalone"))
                    and self._selection_number(
                        nodes[node_id].get("_selection_boundary_confidence"), 0.0
                    ) >= 0.80
                ]
                if not eligible:
                    return []

            chosen_id = min(
                eligible,
                key=lambda node_id: self._selection_contract_sort_key(
                    nodes[node_id],
                    input_order=int(
                        nodes[node_id].get("_selection_input_order") or 0
                    ),
                ),
            )
            chosen = nodes[chosen_id]
            ordered.append(chosen)
            remaining.remove(chosen_id)
            satisfied.add(chosen_id)
            last_video = str(chosen.get("video_id") or "")
        return ordered

    def adaptive_curriculum_order(
        self,
        conn,
        material_id: str,
        learner_id: str,
        items: list[dict[str, Any]],
        previous_video_id: str = "",
    ) -> list[dict[str, Any]]:
        """Interleave source/concept queues while preserving each source's chronology."""
        if not items:
            return []
        uses_selection_contract = any(
            self._has_selection_contract(item) for item in items
        )
        if not uses_selection_contract and len(items) <= 1:
            return list(items)
        try:
            coverage, adjustments, latest, level_target = self._learner_adaptation_context(
                conn, material_id, learner_id
            )
        except ValueError:
            return list(items)
        if uses_selection_contract:
            versioned_items: list[dict[str, Any]] = []
            for raw in items:
                if self._has_selection_contract(raw):
                    versioned_items.append(raw)
                    continue
                legacy = dict(raw)
                video_id = str(legacy.get("video_id") or "unknown")
                legacy.update(
                    _selection_contract_version="legacy-bridge",
                    _selection_content_score=self._selection_number(
                        legacy.get("score"), 0.0
                    ),
                    _selection_boundary_confidence=0.80,
                    _selection_is_standalone=True,
                    _selection_chain_id=f"legacy-source:{video_id}",
                    _selection_chain_position=float(
                        legacy.get("t_start") or 0.0
                    ),
                    _selection_prerequisite_ids=[],
                )
                versioned_items.append(legacy)
            return self._selection_contract_order(
                versioned_items,
                level_target=level_target,
                concept_adjustments=adjustments,
                previous_video_id=previous_video_id,
            )
        queues: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for raw in items:
            item = dict(raw)
            queues[str(item.get("video_id") or item.get("video_url") or item.get("reel_id") or "")].append(item)
        for queue in queues.values():
            queue.sort(key=lambda row: (float(row.get("t_start") or 0.0), float(row.get("t_end") or 0.0)))

        ordered: list[dict[str, Any]] = []
        last_video = str(previous_video_id or "")
        last_concept = ""
        latest_concept = str((latest or {}).get("concept_id") or "")
        latest_video = str((latest or {}).get("video_id") or "")

        heads = lambda: [(video_id, queue[0]) for video_id, queue in queues.items() if queue]

        remediation_signals = list((latest or {}).get("assessment_remediations") or [])
        if not remediation_signals and latest and int(latest.get("confusing") or 0) > 0:
            remediation_signals = [latest]
        for remediation_signal in remediation_signals:
            remediation_concept = str(remediation_signal.get("concept_id") or "")
            remediation_video = str(remediation_signal.get("video_id") or "")
            current_difficulty = self._difficulty(remediation_signal)
            remediation = [
                (video_id, row)
                for video_id, row in heads()
                if str(row.get("concept_id") or "") == remediation_concept
                and self._difficulty(row) < current_difficulty
            ]
            alternate = [pair for pair in remediation if pair[0] != remediation_video]
            boundary_safe_alternate = [pair for pair in alternate if pair[0] != last_video]
            boundary_safe_fallback = [pair for pair in remediation if pair[0] != last_video]
            pool = boundary_safe_alternate or boundary_safe_fallback or alternate or remediation
            if pool:
                concept_target = max(0.0, min(1.0, level_target + adjustments.get(remediation_concept, 0.0)))
                video_id, row = min(
                    pool,
                    key=lambda pair: (
                        abs(self._difficulty(pair[1]) - concept_target),
                        -float(pair[1].get("score") or 0.0),
                    ),
                )
                queues[video_id].pop(0)
                if not queues[video_id]:
                    queues.pop(video_id, None)
                ordered.append(row)
                last_video = video_id
                last_concept = remediation_concept

        first_after_helpful = bool(latest and int(latest.get("helpful") or 0) > 0)
        while queues:
            candidates = heads()
            if len({video_id for video_id, _ in candidates}) > 1:
                other_source = [pair for pair in candidates if pair[0] != last_video]
                if other_source:
                    candidates = other_source
            if first_after_helpful and any(
                str(row.get("concept_id") or "") != latest_concept for _, row in candidates
            ):
                candidates = [
                    pair for pair in candidates
                    if str(pair[1].get("concept_id") or "") != latest_concept
                ]
            elif any(str(row.get("concept_id") or "") != last_concept for _, row in candidates):
                candidates = [
                    pair for pair in candidates
                    if str(pair[1].get("concept_id") or "") != last_concept
                ]

            def priority(pair: tuple[str, dict[str, Any]]) -> tuple[float, str]:
                row = pair[1]
                concept_id = str(row.get("concept_id") or "")
                signal = coverage.get(concept_id, {})
                coverage_shift = (
                    -0.12 * float(signal.get("helpful") or 0.0)
                    + 0.10 * float(signal.get("confusing") or 0.0)
                )
                concept_target = max(0.0, min(1.0, level_target + adjustments.get(concept_id, 0.0)))
                difficulty_fit = 1.0 - abs(self._difficulty(row) - concept_target)
                return (
                    float(row.get("score") or 0.0) + coverage_shift + 0.20 * difficulty_fit,
                    str(row.get("created_at") or ""),
                )

            video_id, chosen = max(candidates, key=priority)
            queues[video_id].pop(0)
            if not queues[video_id]:
                queues.pop(video_id, None)
            ordered.append(chosen)
            last_video = video_id
            last_concept = str(chosen.get("concept_id") or "")
            first_after_helpful = False
        return ordered






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
        ai_summary = self._fallback_ai_summary(
            concept_title=str(concept.get("title") or ""),
            video_title=video_title,
            video_description=video_description,
            transcript_snippet=transcript_snippet,
            takeaways=takeaways,
        )
        relevance_score = float((relevance_context or {}).get("score") or segment.score)
        matched_terms = [
            str(term).strip()
            for term in (relevance_context or {}).get("matched_terms", [])
            if str(term).strip()
        ][:8]
        relevance_reason = str((relevance_context or {}).get("reason") or "").strip()
        match_reason = relevance_reason or (
            f"This clip directly develops {str(concept.get('title') or 'the selected topic')[:180]} "
            "using the source transcript."
        )
        informativeness = 0.6

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
                    "ai_summary": ai_summary,
                    "match_reason": match_reason,
                    "informativeness": informativeness,
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
            "match_reason": match_reason,
            "informativeness": informativeness,
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
        selected_cue_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if clip_end <= clip_start:
            return []

        clip_len = max(0.2, float(clip_end - clip_start))
        cues: list[dict[str, Any]] = []
        selected_cue_id_set = {
            str(cue_id) for cue_id in (selected_cue_ids or []) if str(cue_id).strip()
        }

        for index, entry in enumerate(transcript):
            cue_id = str(entry.get("cue_id") or f"cue-{index}")
            if selected_cue_id_set and cue_id not in selected_cue_id_set:
                continue
            text = str(entry.get("text") or "").replace("\n", " ").strip()
            if not text:
                continue

            try:
                entry_start = float(entry.get("start") or 0.0)
            except (TypeError, ValueError):
                continue

            explicit_end = entry.get("end")
            try:
                entry_end = float(explicit_end) if explicit_end is not None else None
            except (TypeError, ValueError):
                entry_end = None

            duration_value = entry.get("duration")
            try:
                entry_duration = float(duration_value) if duration_value is not None else 0.0
            except (TypeError, ValueError):
                entry_duration = 0.0

            if entry_end is None or entry_end <= entry_start:
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
                "text": text,
            }

            if cues and payload["text"] == cues[-1]["text"] and payload["start"] - cues[-1]["end"] <= 0.2:
                cues[-1]["end"] = payload["end"]
            else:
                cues.append(payload)

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
        learner_id: str = LEGACY_LEARNER_ID,
    ) -> int:
        reel_row = fetch_one(
            conn, "SELECT material_id FROM reels WHERE id = ?", (reel_id,)
        )
        if not reel_row:
            raise ValueError(f"unknown reel_id: {reel_id}")
        material_id = str(reel_row["material_id"])
        progress = self.learner_progress(conn, material_id, learner_id)
        existing = fetch_one(
            conn,
            "SELECT * FROM reel_feedback WHERE learner_id = ? AND reel_id = ?",
            (learner_id, reel_id),
        )
        helpful_value = 1 if helpful else 0
        confusing_value = 1 if confusing else 0
        mastery_changed = (
            existing is None
            and (helpful_value > 0 or confusing_value > 0)
        ) or (
            existing is not None
            and (
                int(existing.get("helpful") or 0) != helpful_value
                or int(existing.get("confusing") or 0) != confusing_value
            )
        )
        timestamp = now_iso()
        upsert(
            conn,
            "reel_feedback",
            {
                "id": str((existing or {}).get("id") or uuid.uuid4()),
                "learner_id": learner_id,
                "reel_id": reel_id,
                "helpful": helpful_value,
                "confusing": confusing_value,
                "rating": rating,
                "saved": 1 if saved else 0,
                "mastery_updated_at": (
                    timestamp if mastery_changed else (existing or {}).get("mastery_updated_at")
                ),
                "updated_at": timestamp,
                "created_at": str((existing or {}).get("created_at") or timestamp),
            },
            pk=["learner_id", "reel_id"],
        )
        revision = int(progress.get("feedback_revision") or 0) + 1
        execute_modify(
            conn,
            """
            UPDATE learner_material_progress
            SET feedback_revision = ?, updated_at = ?
            WHERE learner_id = ? AND material_id = ?
            """,
            (revision, timestamp, learner_id, material_id),
        )
        if mastery_changed:
            self.update_level_adjustment(conn, material_id, learner_id)
        return revision

    def ranked_feed(
        self,
        conn,
        material_id: str,
        fast_mode: bool = False,
        generation_id: str | None = None,
        page_hint: int = 1,
        learner_id: str = LEGACY_LEARNER_ID,
        exclusions_fingerprint: str = "",
        content_fingerprint: str = "",
        require_verified_boundaries: bool = False,
    ) -> list[dict[str, Any]]:
        material = fetch_one(conn, "SELECT subject_tag, source_type FROM materials WHERE id = ?", (material_id,))
        subject_tag = str((material or {}).get("subject_tag") or "").strip() or None
        strict_topic_only = str((material or {}).get("source_type") or "").strip().lower() == "topic"
        progress = self.learner_progress(conn, material_id, learner_id)
        level_target = effective_level_target(
            progress.get("selected_level"),
            progress.get("global_adjustment"),
        )
        feedback_revision = int(progress.get("feedback_revision") or 0)

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
            learner_id=learner_id,
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
            learner_id=learner_id,
            feedback_revision=feedback_revision,
            page_hint=page_hint,
            exclusions_fingerprint=exclusions_fingerprint,
            content_fingerprint=content_fingerprint,
            require_verified_boundaries=require_verified_boundaries,
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
                r.ai_summary,
                r.match_reason,
                r.informativeness,
                r.base_score,
                r.difficulty,
                r.model_used,
                r.quality_degraded,
                r.selected_cue_ids_json,
                r.search_context_json,
                COALESCE(SUM(f.helpful), 0) AS helpful_votes,
                COALESCE(SUM(f.confusing), 0) AS confusing_votes,
                COALESCE(AVG(f.rating), 3.0) AS avg_rating,
                COALESCE(SUM(f.saved), 0) AS saves,
                r.created_at
            FROM reels r
            JOIN concepts c ON c.id = r.concept_id
            JOIN videos v ON v.id = r.video_id
            LEFT JOIN reel_feedback f
              ON f.reel_id = r.id
             AND f.learner_id = ?
            WHERE {reel_where}
              AND r.t_start >= 0
              AND r.t_end > r.t_start
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
                r.ai_summary,
                r.match_reason,
                r.informativeness,
                r.base_score,
                r.difficulty,
                r.model_used,
                r.quality_degraded,
                r.selected_cue_ids_json,
                r.search_context_json,
                r.created_at
            """,
            (learner_id, *reel_params),
        )

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
        learner_coverage, concept_adjustments, _, _ = self._learner_adaptation_context(
            conn, material_id, learner_id
        )

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
            try:
                selected_cue_ids = json.loads(str(row.get("selected_cue_ids_json") or "[]"))
            except (TypeError, json.JSONDecodeError):
                selected_cue_ids = []
            if not isinstance(selected_cue_ids, list):
                selected_cue_ids = []
            selected_cue_ids = [str(cue_id) for cue_id in selected_cue_ids if str(cue_id).strip()]
            selection_metadata = self._selection_metadata(
                row.get("search_context_json"),
                t_start=row.get("t_start"),
                t_end=row.get("t_end"),
            )
            selection_version = str(
                selection_metadata.get("_selection_contract_version") or ""
            ).strip()
            surface_eligible = selection_metadata.get(
                "_selection_surface_eligible", True
            )
            surface_reason = str(
                selection_metadata.get("_selection_surface_reason") or ""
            )
            legacy_difficulty_matches_level = (
                abs(self._difficulty(row) - level_target) <= 0.35
            )
            difficulty_matches_level = (
                difficulty_matches_knowledge_level(
                    self._difficulty(row),
                    str(progress.get("selected_level") or "beginner"),
                )
                if selection_version in {
                    "quality_silence_v3",
                    "quality_silence_v4",
                    "quality_silence_v5",
                    "quality_silence_v6",
                    "quality_silence_v7",
                    "quality_silence_v8",
                    "quality_silence_v9",
                    "quality_silence_v10",
                    "quality_silence_v11",
                    "quality_silence_v12",
                    "quality_silence_v13",
                    "quality_silence_v14",
                    "quality_silence_v15",
                    "quality_silence_v16",
                    "quality_silence_v17",
                    "quality_silence_v18",
                    "quality_silence_v19",
                    "quality_silence_v20",
                    "quality_silence_v21",
                    "quality_silence_v22",
                    "quality_silence_v23",
                    "quality_silence_v24",
                    "quality_silence_v25",
                    "quality_silence_v26",
                    "quality_silence_v27",
                    "quality_silence_v28",
                    "quality_silence_v29",
                    "quality_silence_v30",
                }
                else legacy_difficulty_matches_level
            )
            deferred_level_candidate = (
                surface_reason == "level_mismatch"
                and (
                    difficulty_matches_level
                    or selection_version in self.DIFFICULTY_FALLBACK_CONTRACTS
                )
            )
            prerequisite_may_become_ready = (
                surface_reason == "prerequisite_not_surfaceable"
                and difficulty_matches_level
            )
            if surface_eligible is False and not (
                deferred_level_candidate or prerequisite_may_become_ready
            ):
                continue
            boundary_status = selection_metadata.get(
                "_selection_boundary_status"
            )
            if require_verified_boundaries:
                if (
                    boundary_status not in {"verified", "context_aligned"}
                    or selection_metadata.get("_selection_boundary_usable") is not True
                    or (
                        selection_version
                        in {
                            "quality_silence_v5",
                            "quality_silence_v6",
                            "quality_silence_v7",
                            "quality_silence_v8",
                            "quality_silence_v9",
                            "quality_silence_v10",
                            "quality_silence_v11",
                            "quality_silence_v12",
                            "quality_silence_v13",
                            "quality_silence_v14",
                            "quality_silence_v15",
                            "quality_silence_v16",
                            "quality_silence_v17",
                            "quality_silence_v18",
                            "quality_silence_v19",
                            "quality_silence_v20",
                            "quality_silence_v21",
                            "quality_silence_v22",
                            "quality_silence_v23",
                            "quality_silence_v24",
                            "quality_silence_v25",
                            "quality_silence_v26",
                            "quality_silence_v27",
                            "quality_silence_v28",
                            "quality_silence_v29",
                            "quality_silence_v30",
                        }
                        and selection_metadata.get(
                            "_selection_speech_corridor_verified"
                        ) is not True
                    )
                ):
                    continue
            elif boundary_status == "unavailable":
                continue
            if selection_metadata:
                if selection_version in {
                    "quality_silence_v2",
                    "quality_silence_v3",
                    "quality_silence_v4",
                    "quality_silence_v5",
                    "quality_silence_v6",
                    "quality_silence_v7",
                    "quality_silence_v8",
                    "quality_silence_v9",
                    "quality_silence_v10",
                    "quality_silence_v11",
                    "quality_silence_v12",
                    "quality_silence_v13",
                    "quality_silence_v14",
                    "quality_silence_v15",
                    "quality_silence_v16",
                    "quality_silence_v17",
                    "quality_silence_v18",
                    "quality_silence_v19",
                    "quality_silence_v20",
                    "quality_silence_v21",
                    "quality_silence_v22",
                    "quality_silence_v23",
                    "quality_silence_v24",
                    "quality_silence_v25",
                    "quality_silence_v26",
                    "quality_silence_v27",
                    "quality_silence_v28",
                    "quality_silence_v29",
                    "quality_silence_v30",
                } and (
                    (
                        min(
                            self._selection_number(
                                selection_metadata.get("_selection_informativeness"), 0.0
                            ),
                            self._selection_number(
                                selection_metadata.get("_selection_topic_relevance"), 0.0
                            ),
                            self._selection_number(
                                selection_metadata.get("_selection_educational_importance"), 0.0
                            ),
                        ) < 0.75
                        if selection_version
                        in {
                            "quality_silence_v2",
                            "quality_silence_v6",
                            "quality_silence_v7",
                            "quality_silence_v8",
                            "quality_silence_v9",
                            "quality_silence_v10",
                            "quality_silence_v11",
                            "quality_silence_v12",
                            "quality_silence_v13",
                            "quality_silence_v14",
                            "quality_silence_v15",
                            "quality_silence_v16",
                            "quality_silence_v17",
                            "quality_silence_v18",
                            "quality_silence_v19",
                            "quality_silence_v20",
                            "quality_silence_v21",
                            "quality_silence_v22",
                            "quality_silence_v23",
                            "quality_silence_v24",
                            "quality_silence_v25",
                            "quality_silence_v26",
                            "quality_silence_v27",
                            "quality_silence_v28",
                            "quality_silence_v29",
                            "quality_silence_v30",
                        }
                        else self._selection_number(
                            selection_metadata.get("_selection_topic_relevance"), 0.0
                        ) < 0.75
                    )
                    or not selection_metadata.get("_selection_self_contained")
                    or not selection_metadata.get("_selection_is_standalone")
                    or not selection_metadata.get("_selection_topic_evidence_quote")
                ):
                    continue
                if (
                    not selection_metadata.get("_selection_directly_teaches_topic", True)
                    or not selection_metadata.get("_selection_substantive", True)
                    or (
                        selection_version
                        and not selection_metadata.get(
                            "_selection_factually_grounded", False
                        )
                    )
                ):
                    continue
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
            if not strict_topic_only and self._is_hard_blocked_low_value_video(
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
            if (
                not strict_topic_only
                and context_terms
                and self._is_short_leaf_topic(concept_title)
                and not relevance["passes"]
            ):
                # Short leaf concepts such as ATP are valid inside their material,
                # but ambiguous in isolation. Cached rows must still demonstrate
                # the parent material context before they can surface.
                continue
            if not strict_topic_only and not relevance["passes"] and (
                float(relevance.get("off_topic_penalty") or 0.0) >= 0.24
                or float(relevance.get("score") or -1.0) < 0.02
            ):
                # Hide strongly off-topic clips that can still exist from older generations.
                continue
            # Topic materials already passed practice's transcript-grounded clip
            # validation. ReelAI only reorders those clips for relevance and the
            # learner level; it must not collapse the result set with a second
            # lexical hard gate.
            relevance_context = self._merge_relevance_context(relevance, relevance)

            ai_summary = str(row.get("ai_summary") or "").strip()
            if not ai_summary:
                ai_summary = self._fallback_ai_summary(
                    concept_title=concept_title,
                    video_title=video_title,
                    video_description=video_description,
                    transcript_snippet=transcript_snippet,
                    takeaways=takeaways,
                )
            match_reason = str(row.get("match_reason") or "").strip()
            if not match_reason:
                matched_idea = concept_title or (takeaways[0] if takeaways else "") or "this topic"
                match_reason = f"This clip directly develops {matched_idea[:180]} using the source transcript."
            try:
                informativeness = float(row.get("informativeness"))
            except (TypeError, ValueError):
                informativeness = 0.6
            informativeness = max(0.0, min(1.0, informativeness))
            safe_page_hint = max(1, int(page_hint))
            _diff = self._difficulty(row)
            concept_target = max(
                0.0,
                min(1.0, level_target + concept_adjustments.get(str(row["concept_id"]), 0.0)),
            )
            learner_signal = learner_coverage.get(str(row["concept_id"]), {})
            if self._has_selection_contract(selection_metadata):
                score = self._selection_number(
                    selection_metadata.get("_selection_quality_floor"),
                    self._selection_number(
                        selection_metadata.get("_selection_content_score"), 0.0
                    ),
                )
            else:
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
                    + 0.12 * (1.0 - 2.0 * abs(_diff - concept_target))
                    + 0.05 * (1.0 - _diff) * max(0.0, 1.0 - (safe_page_hint - 1) / 2.0)
                    - 0.04 * float(learner_signal.get("helpful") or 0.0)
                    + 0.04 * float(learner_signal.get("confusing") or 0.0)
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
                    "match_reason": match_reason,
                    "informativeness": informativeness,
                    "video_url": row["video_url"],
                    "t_start": float(row["t_start"]),
                    "t_end": float(row["t_end"]),
                    "transcript_snippet": transcript_snippet,
                    "takeaways": takeaways,
                    "score": score,
                    "relevance_score": (
                        self._selection_number(
                            selection_metadata.get("_selection_topic_relevance"),
                            0.0,
                        )
                        if selection_version
                        == self.RANKED_FEED_CACHE_CONTRACT_VERSION
                        else float(relevance_context.get("score") or 0.0)
                    ),
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
                    "difficulty": self._difficulty(row),
                    "model_used": str(row.get("model_used") or "").strip() or None,
                    "quality_degraded": bool(row.get("quality_degraded")),
                    "selected_cue_ids": selected_cue_ids,
                    "created_at": row["created_at"],
                    **selection_metadata,
                }
            )

        scored = self.select_difficulty_inventory(
            scored,
            str(progress.get("selected_level") or "beginner"),
        )
        scored.sort(key=lambda x: (x["score"], x["created_at"]), reverse=True)
        deduped: list[dict[str, Any]] = []
        seen_reel_ids: set[str] = set()
        seen_clip_keys: set[str] = set()
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
            if reel_id:
                seen_reel_ids.add(reel_id)
            seen_clip_keys.add(clip_key)
            deduped.append(dict(item))

        deduped = self.adaptive_curriculum_order(conn, material_id, learner_id, deduped)

        deduped_video_ids = sorted({str(item.get("video_id") or "") for item in deduped if item.get("video_id")})
        legacy_transcript_by_video: dict[str, list[dict[str, Any]]] = {}
        transcript_by_artifact_key: dict[str, list[dict[str, Any]]] = {}
        if deduped_video_ids:
            bare_to_response_ids: dict[str, list[str]] = defaultdict(list)
            for response_video_id in deduped_video_ids:
                bare_id = normalize_youtube_video_id(response_video_id)
                if bare_id:
                    bare_to_response_ids[bare_id].append(response_video_id)
            bare_video_ids = sorted(bare_to_response_ids)
            transcript_rows = []
            if bare_video_ids:
                placeholders = ", ".join(["?"] * len(bare_video_ids))
                transcript_rows = fetch_all(
                    conn,
                    "SELECT cache_key, video_id, artifact_json FROM transcript_artifacts "
                    f"WHERE video_id IN ({placeholders}) "
                    "ORDER BY created_at DESC",
                    tuple(bare_video_ids),
                )
            for trow in transcript_rows:
                try:
                    payload = json.loads(str(trow.get("artifact_json") or "{}"))
                except (TypeError, json.JSONDecodeError):
                    continue
                artifact = validate_transcript_payload(payload)
                if artifact is None:
                    continue
                cache_key = str(trow.get("cache_key") or "").strip()
                if cache_key and cache_key == artifact.artifact_key:
                    transcript_by_artifact_key[cache_key] = artifact.segments
                for response_video_id in bare_to_response_ids.get(artifact.video_id, []):
                    legacy_transcript_by_video.setdefault(
                        response_video_id, artifact.segments
                    )

        response_rows: list[dict[str, Any]] = []
        for item in deduped:
            clean_item = dict(item)
            selection_ordered = self._has_selection_contract(clean_item)
            selection_contract_version = str(
                clean_item.get("_selection_contract_version") or ""
            ).strip()
            selection_quality_floor = self._selection_number(
                clean_item.get("_selection_quality_floor"), 0.0
            )
            selection_quality_mean = self._selection_number(
                clean_item.get("_selection_quality_mean"), 0.0
            )
            selection_topic_relevance = self._selection_number(
                clean_item.get("_selection_topic_relevance"), 0.0
            )
            selection_informativeness = clean_item.get(
                "_selection_informativeness"
            )
            if selection_informativeness is not None:
                selection_informativeness = self._selection_number(
                    selection_informativeness
                )
            selection_educational_importance = clean_item.get(
                "_selection_educational_importance"
            )
            if selection_educational_importance is not None:
                selection_educational_importance = self._selection_number(
                    selection_educational_importance
                )
            selection_source_rank = int(
                clean_item.get("_selection_source_rank") or 0
            )
            transcript_artifact_key = str(
                clean_item.get("_selection_transcript_artifact_key") or ""
            ).strip()
            selection_caption_cues = list(
                clean_item.get("_selection_caption_cues") or []
            )
            for internal_key in [
                key for key in clean_item if key.startswith("_selection_")
            ]:
                clean_item.pop(internal_key, None)
            # Request shaping preserves relative order, so main can avoid
            # accidentally applying the legacy chronological scheduler again.
            clean_item["_selection_ordered"] = selection_ordered
            if selection_contract_version:
                clean_item["selection_contract_version"] = (
                    selection_contract_version
                )
                clean_item["topic_relevance"] = selection_topic_relevance
                clean_item["_selection_quality_floor"] = selection_quality_floor
                clean_item["_selection_quality_mean"] = selection_quality_mean
                clean_item["_selection_topic_relevance"] = selection_topic_relevance
                if selection_informativeness is not None:
                    clean_item["_selection_informativeness"] = (
                        selection_informativeness
                    )
                if selection_educational_importance is not None:
                    clean_item["_selection_educational_importance"] = (
                        selection_educational_importance
                    )
                clean_item["_selection_source_rank"] = selection_source_rank
            # Keep video_id on the response row so downstream filters (notably
            # main._ranked_request_reels's exclude_video_ids filter) can match
            # on it. Stripping it here silently defeated client pagination.
            video_id = str(clean_item.get("video_id") or "")
            clean_item["video_id"] = video_id
            if selection_caption_cues:
                caption_transcript = selection_caption_cues
            elif selection_contract_version in {
                "quality_silence_v5",
                "quality_silence_v6",
                "quality_silence_v7",
                "quality_silence_v8",
                "quality_silence_v9",
                "quality_silence_v10",
                "quality_silence_v11",
                "quality_silence_v12",
                "quality_silence_v13",
                "quality_silence_v14",
                "quality_silence_v15",
                "quality_silence_v16",
                "quality_silence_v17",
                "quality_silence_v18",
                "quality_silence_v19",
                "quality_silence_v20",
                "quality_silence_v21",
                "quality_silence_v22",
                "quality_silence_v23",
                "quality_silence_v24",
                "quality_silence_v25",
                "quality_silence_v26",
                "quality_silence_v27",
                "quality_silence_v28",
                "quality_silence_v29",
                "quality_silence_v30",
            }:
                # V5+ captions must be immutable selection-time evidence. A
                # provider artifact key identifies a retrieval profile and may
                # be overwritten by a later same-profile refresh.
                caption_transcript = []
            elif transcript_artifact_key:
                caption_transcript = transcript_by_artifact_key.get(
                    transcript_artifact_key, []
                )
            else:
                caption_transcript = legacy_transcript_by_video.get(video_id, [])
            clean_item["captions"] = self._build_caption_cues(
                transcript=caption_transcript,
                clip_start=float(clean_item.get("t_start") or 0.0),
                clip_end=float(clean_item.get("t_end") or 0.0),
                fallback_text=(
                    ""
                    if selection_contract_version
                    in {
                        "quality_silence_v5",
                        "quality_silence_v6",
                        "quality_silence_v7",
                        "quality_silence_v8",
                        "quality_silence_v9",
                        "quality_silence_v10",
                        "quality_silence_v11",
                        "quality_silence_v12",
                        "quality_silence_v13",
                        "quality_silence_v14",
                        "quality_silence_v15",
                        "quality_silence_v16",
                        "quality_silence_v17",
                        "quality_silence_v18",
                        "quality_silence_v19",
                        "quality_silence_v20",
                        "quality_silence_v21",
                        "quality_silence_v22",
                        "quality_silence_v23",
                        "quality_silence_v24",
                        "quality_silence_v25",
                        "quality_silence_v26",
                        "quality_silence_v27",
                        "quality_silence_v28",
                        "quality_silence_v29",
                        "quality_silence_v30",
                    }
                    or transcript_artifact_key
                    else str(clean_item.get("transcript_snippet") or "")
                ),
                selected_cue_ids=list(clean_item.get("selected_cue_ids") or []),
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
            learner_id=learner_id,
            feedback_revision=feedback_revision,
            page_hint=page_hint,
            exclusions_fingerprint=exclusions_fingerprint,
            content_fingerprint=content_fingerprint,
            require_verified_boundaries=require_verified_boundaries,
        )
        return response_rows
