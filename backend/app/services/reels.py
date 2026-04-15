import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError, as_completed
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Callable, Literal

import numpy as np

from ..config import get_settings
from ..db import DatabaseIntegrityError, dumps_json, fetch_all, fetch_one, now_iso, upsert
from .concepts import build_takeaways
from .openai_client import build_openai_client
from .segmenter import (
    SegmentMatch,
    TranscriptChunk,
    chunk_transcript,
    lexical_overlap_score,
    normalize_terms,
    select_segments,
)
from .topic_expansion import TopicExpansionService
from .transcript_validation import TranscriptQuality, validate_transcript

logger = logging.getLogger(__name__)


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
        "java",
        "javascript",
        "loop",
        "loops",
        "python",
        "stress",
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
    RANKED_FEED_CACHE_VERSION = 4
    REFILL_STAGE_EXACT_ROOT = 0
    REFILL_STAGE_ROOT_COMPANION = 1
    REFILL_STAGE_MULTI_CLIP_STRICT = 2
    REFILL_STAGE_ANCHORED_ADJACENT = 3
    REFILL_STAGE_MULTI_CLIP_EXPANDED = 4
    REFILL_STAGE_RECOVERY_GRAPH = 5
    MAX_REFILL_STAGE = REFILL_STAGE_RECOVERY_GRAPH

    def __init__(self, embedding_service, youtube_service) -> None:
        settings = get_settings()
        self.embedding_service = embedding_service
        self.youtube_service = youtube_service
        self.chat_model = settings.openai_chat_model
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
        allow_openai_serverless = os.getenv("ALLOW_OPENAI_IN_SERVERLESS") == "1"
        can_use_openai = (
            bool(settings.openai_enabled)
            and bool(settings.openai_api_key)
            and (not self.serverless_mode or allow_openai_serverless)
        )
        self.openai_client = build_openai_client(
            api_key=settings.openai_api_key,
            timeout=8.0,
            enabled=can_use_openai,
        )
        self.topic_expansion_service = TopicExpansionService()

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
    ) -> str:
        # Include subject_tag + strict_topic_only in the key because ranked_feed's
        # relevance gates (`_passes_relevance_gate`, strict topic filter, hard-
        # blocked check) consult them and their output is baked into the cached
        # rows. Without this the cache would return stale pre-filtered data when
        # the caller's context (topic/source) changed.
        payload = {
            "material_id": material_id,
            "generation_id": generation_id or "",
            "fast_mode": bool(fast_mode),
            "subject_tag": (subject_tag or "").strip().lower(),
            "strict_topic_only": bool(strict_topic_only),
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
        summary_mode = "fallback" if fast_mode or not self.openai_client else f"ai:{self.chat_model}"
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
    ) -> list[dict[str, Any]] | None:
        cache_key = self._ranked_feed_cache_key(
            material_id=material_id,
            generation_id=generation_id,
            fast_mode=fast_mode,
            subject_tag=subject_tag,
            strict_topic_only=strict_topic_only,
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
        self._min_relevance_threshold = max(0.0, min(1.0, float(min_relevance_threshold)))
        safe_page_hint = max(1, int(page_hint or 1))
        safe_recovery_stage = max(0, int(recovery_stage or 0))
        allow_adjacent_recovery = self._allow_adjacent_recovery(
            page_hint=safe_page_hint,
            recovery_stage=safe_recovery_stage,
        )
        safe_retrieval_profile = self._normalize_retrieval_profile(retrieval_profile)
        request_key = ""
        if generation_id:
            generation_row = fetch_one(
                conn,
                "SELECT request_key FROM reel_generations WHERE id = ?",
                (generation_id,),
            )
            request_key = str((generation_row or {}).get("request_key") or "").strip()
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
        material = fetch_one(conn, "SELECT subject_tag, source_type FROM materials WHERE id = ?", (material_id,))
        subject_tag = str((material or {}).get("subject_tag") or "").strip() or None
        strict_topic_only = str((material or {}).get("source_type") or "").strip().lower() == "topic"
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
        max_generation_target = self._generation_target_cap(
            num_reels=num_reels,
            preferred_video_duration=safe_video_duration_pref,
            fast_mode=fast_mode,
        )
        if self.serverless_mode:
            max_generation_target = min(max_generation_target, max(3, num_reels))
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
        material_context_terms = self._build_material_context_terms(concepts=concepts, subject_tag=subject_tag)
        query_plan = self._plan_query_set_for_concepts(
            concepts=concepts,
            subject_tag=subject_tag,
            material_context_terms=material_context_terms,
            retrieval_profile=safe_retrieval_profile,
            fast_mode=fast_mode,
            video_pool_mode=safe_video_pool_mode,
            preferred_video_duration=safe_video_duration_pref,
            request_need=max(1, num_reels),
            targeted_concept_id=concept_id,
            allow_bootstrap_subtopic_expansion=not (
                strict_topic_only
                and safe_retrieval_profile == "bootstrap"
                and self._topic_breadth_class(subject_tag) != "curated_broad"
            ),
        )
        selected_concept_ids = {plan.concept_id for plan in query_plan.selected_concepts}
        selected_concept_by_id = {
            str(concept.get("id") or ""): concept
            for concept in concepts
            if str(concept.get("id") or "") in selected_concept_ids
        }
        # Fix A: Pre-compute all concept embeddings in batch before the main loop
        precomputed_embeddings: dict[str, np.ndarray | None] = {}
        if not fast_mode:
            embed_texts_for_concepts: list[str] = []
            embed_concept_ids: list[str] = []
            for cp in query_plan.selected_concepts:
                c = selected_concept_by_id.get(cp.concept_id)
                if c is None:
                    continue
                embed_concept_ids.append(cp.concept_id)
                embed_texts_for_concepts.append(
                    f"{str(c.get('title') or '')} {str(c.get('summary') or '')}"
                )
            if embed_texts_for_concepts:
                try:
                    batch_embeddings = self.embedding_service.embed_texts(conn, embed_texts_for_concepts)
                    for i, cid in enumerate(embed_concept_ids):
                        precomputed_embeddings[cid] = batch_embeddings[i]
                except Exception as exc:
                    logger.warning("Batch concept embedding failed: %s", exc)

        material_seen_video_ids: set[str] = set()
        for excluded_video_id in exclude_video_ids or []:
            clean_video_id = str(excluded_video_id or "").strip()
            if clean_video_id:
                material_seen_video_ids.add(clean_video_id)

        for concept_plan in query_plan.selected_concepts:
            raise_if_cancelled()
            concept = selected_concept_by_id.get(concept_plan.concept_id)
            if concept is None:
                continue
            if self._should_finalize_generation(
                generated=generated,
                num_reels=num_reels,
                preferred_video_duration=safe_video_duration_pref,
                max_generation_target=max_generation_target,
                fast_mode=fast_mode,
            ):
                break

            concept_keywords = self._parse_keywords_json(concept.get("keywords_json"))
            concept_summary = str(concept.get("summary") or "")
            concept_embedding: np.ndarray | None = precomputed_embeddings.get(concept_plan.concept_id)
            if concept_embedding is None and not fast_mode:
                try:
                    concept_embedding = self._get_concept_embedding(conn, concept)
                except Exception:
                    concept_embedding = None

            concept_terms = [concept["title"], *concept_keywords, concept_summary]
            context_terms = self._context_terms_for_concept(concept_terms, material_context_terms)
            vague_topic = self._is_vague_concept(
                title=concept["title"],
                keywords=concept_keywords,
                summary=concept_summary,
            )
            visual_spec = self._derive_visual_scene_spec(
                title=concept["title"],
                keywords=concept_keywords,
                summary=concept_summary,
            )
            planned_queries = [concept_plan.literal_query, concept_plan.intent_query, *concept_plan.expansion_queries]
            if safe_retrieval_profile == "deep":
                planned_queries.extend(concept_plan.recovery_queries)
                if safe_recovery_stage >= self.REFILL_STAGE_RECOVERY_GRAPH and request_key:
                    planned_queries.extend(
                        self._plan_source_graph_queries(
                            conn,
                            material_id=material_id,
                            request_key=request_key,
                            current_generation_id=generation_id,
                            concept_title=str(concept.get("title") or ""),
                            subject_tag=subject_tag,
                        )
                    )
            if self.retrieval_engine_v2_enabled:
                query_candidates = [
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
                    for query in planned_queries
                ]
            else:
                query_candidates = [
                    QueryCandidate(
                        text=query,
                        strategy="literal",
                        confidence=0.7,
                        source_terms=[concept["title"]],
                        stage="high_precision" if index < self.BOOTSTRAP_PRIMARY_QUERY_COUNT else "broad",
                        source_surface="youtube_html",
                        family_key=self._frontier_family_key(
                            self.FRONTIER_ROOT_EXACT if index == 0 else self.FRONTIER_ROOT_COMPANION,
                            query,
                        ),
                        source_family="other",
                        anchor_mode="root_exact" if index == 0 else "root_companion",
                    )
                    for index, query in enumerate(
                        self._build_query_variants(
                            concept["title"],
                            concept_keywords,
                            subject_tag,
                            context_terms=context_terms,
                        )
                    )
                ]
            query_candidates = self._apply_frontier_scheduler(
                conn,
                material_id=material_id,
                request_key=request_key or None,
                query_candidates=query_candidates,
                page_hint=safe_page_hint,
                recovery_stage=safe_recovery_stage,
            )
            retrieval_stages = self._build_retrieval_stage_plan(
                query_candidates=query_candidates,
                fast_mode=fast_mode,
                retrieval_profile=safe_retrieval_profile,
                request_need=max(1, num_reels),
                subject_tag=subject_tag,
                page_hint=safe_page_hint,
                recovery_stage=safe_recovery_stage,
            )
            retrieval_run = self._init_retrieval_debug_run(
                material_id=material_id,
                concept_id=str(concept.get("id") or ""),
                concept_title=str(concept.get("title") or ""),
            )

            seen_video_ids: set[str] = set(material_seen_video_ids)
            stage_candidates: list[dict[str, Any]] = []
            all_query_reports: list[dict[str, Any]] = []
            accepted_reels_by_family: dict[str, int] = defaultdict(int)
            retrieval_metrics = {
                "raw_discovered_videos": 0,
                "filtered_relevant_videos": 0,
                "clip_candidate_videos": 0,
                "final_unique_reels": 0,
                "dropped_duplicate": 0,
                "dropped_segment_cap": 0,
                "dropped_duration": 0,
                "dropped_metadata_gate": 0,
                "dropped_ranking": 0,
                "low_relevance": 0,
                "low_transcript_purity": 0,
                "low_novelty": 0,
                "low_self_containment": 0,
            }
            max_results_for_query = self._search_results_budget(
                fast_mode=fast_mode,
                generated_count=len(generated),
                max_generation_target=max_generation_target,
                retrieval_profile=safe_retrieval_profile,
                recovery_stage=safe_recovery_stage,
            )
            if safe_retrieval_profile == "bootstrap":
                transcript_budget = self._bootstrap_transcript_budget(
                    generated_count=len(generated),
                    max_generation_target=max_generation_target,
                )
            else:
                transcript_budget = min(
                    20,
                    self._transcript_expansion_budget(
                        fast_mode=fast_mode,
                        generated_count=len(generated),
                        max_generation_target=max_generation_target,
                        recovery_stage=safe_recovery_stage,
                    ),
                )
            mined_stage_candidates: list[dict[str, Any]] = []
            if safe_retrieval_profile == "deep" and request_key and safe_recovery_stage >= self.REFILL_STAGE_MULTI_CLIP_STRICT:
                mined_stage_candidates = self._mine_existing_request_chain_videos(
                    conn,
                    material_id=material_id,
                    request_key=request_key,
                    generation_id=generation_id,
                    concept=concept,
                    concept_terms=concept_terms,
                    context_terms=context_terms,
                    concept_embedding=concept_embedding,
                    subject_tag=subject_tag,
                    root_topic_terms=root_topic_terms,
                    visual_spec=visual_spec,
                    preferred_video_duration=safe_video_duration_pref,
                    fast_mode=fast_mode,
                    strict_topic_only=strict_topic_only,
                    existing_video_counts=existing_video_counts,
                    generated_video_counts=generated_video_counts,
                    accepted_clip_contexts_by_video=accepted_clip_contexts_by_video,
                    target_clip_duration_sec=safe_target_clip_duration,
                    page_hint=safe_page_hint,
                    recovery_stage=safe_recovery_stage,
                    default_max_segments_per_video=default_max_segments_per_video,
                )
                if mined_stage_candidates:
                    stage_candidates.extend(mined_stage_candidates)
                    retrieval_metrics["clip_candidate_videos"] = max(
                        int(retrieval_metrics.get("clip_candidate_videos") or 0),
                        len(mined_stage_candidates),
                    )
                    for mined_candidate in mined_stage_candidates:
                        clean_video_id = str(mined_candidate.get("video_id") or "").strip()
                        if clean_video_id:
                            seen_video_ids.add(clean_video_id)
            transcript_prefetch_task: TranscriptPrefetchTask | None = None
            primary_stages = [stage for stage in retrieval_stages if stage.name == "high_precision"]
            expansion_stages = [stage for stage in retrieval_stages if stage.name != "high_precision"]
            pass_groups = [] if mined_stage_candidates else [primary_stages, expansion_stages]

            for pass_index, stage_group in enumerate(pass_groups):
                raise_if_cancelled()
                if not stage_group:
                    continue
                if pass_index > 0 and self._fast_pass_is_sufficient(
                    stage_candidates,
                    fast_mode=fast_mode,
                    max_generation_target=max_generation_target,
                    retrieval_profile=safe_retrieval_profile,
                ):
                    break

                for stage in stage_group:
                    raise_if_cancelled()
                    if self._should_finalize_generation(
                        generated=generated,
                        num_reels=num_reels,
                        preferred_video_duration=safe_video_duration_pref,
                        max_generation_target=max_generation_target,
                        fast_mode=fast_mode,
                    ):
                        break

                    good_results = 0
                    stage_duration_plan = self._stage_duration_plan(
                        stage_name=stage.name,
                        preferred_video_duration=safe_video_duration_pref,
                        video_pool_mode=safe_video_pool_mode,
                        fast_mode=fast_mode,
                        retrieval_profile=safe_retrieval_profile,
                    )
                    strict_duration = stage.name == "high_precision" and safe_video_duration_pref in {
                        "short",
                        "medium",
                        "long",
                    }
                    allow_unknown_duration = stage.name != "high_precision"
                    stage_budget = int(stage.budget or 0)
                    if (
                        stage.name == "broad"
                        and stage.max_budget is not None
                        and len(stage_candidates) < max(1, int(num_reels))
                    ):
                        stage_budget = min(int(stage.max_budget or stage_budget), len(stage.queries))
                    stage_queries = stage.queries[: max(0, stage_budget)]
                    stage_query_reports = [
                        {
                            "query": query_candidate.text,
                            "strategy": query_candidate.strategy,
                            "stage": stage.name,
                            "source_terms": query_candidate.source_terms,
                            "weight": float(query_candidate.weight),
                            "surface": query_candidate.source_surface,
                            "family_key": query_candidate.family_key,
                            "source_family": query_candidate.source_family,
                            "anchor_mode": query_candidate.anchor_mode,
                            "seed_video_id": query_candidate.seed_video_id,
                            "seed_channel_id": query_candidate.seed_channel_id,
                            "results": 0,
                            "kept": 0,
                            "dropped_duplicate": 0,
                            "dropped_segment_cap": 0,
                            "dropped_duration": 0,
                            "dropped_metadata_gate": 0,
                            "dropped_ranking": 0,
                        }
                        for query_candidate in stage_queries
                    ]
                    search_jobs = self._stage_search_jobs_parallel(
                        stage_name=stage.name,
                        stage_queries=stage_queries,
                        stage_duration_plan=stage_duration_plan,
                        max_results_for_query=max_results_for_query,
                        creative_commons_only=creative_commons_only,
                        fast_mode=fast_mode,
                        retrieval_profile=safe_retrieval_profile,
                        page_hint=safe_page_hint,
                        recovery_stage=safe_recovery_stage,
                        allow_external_fallbacks=allow_adjacent_recovery and stage.name == "recovery",
                        variant_limit=(
                            (4 if self._topic_breadth_class(subject_tag) == "opaque_niche" else 3)
                            if safe_retrieval_profile == "bootstrap"
                            else None
                        ),
                        subject_tag=subject_tag,
                    )
                    raise_if_cancelled()

                    # Fix E: Pre-filter videos and batch-embed their metadata for efficiency
                    scorable_videos: list[tuple[int, QueryCandidate, dict[str, Any], dict[str, Any]]] = []
                    for query_idx, _duration_idx, query_candidate, _duration, videos in search_jobs:
                        raise_if_cancelled()
                        if query_idx >= len(stage_query_reports):
                            continue
                        query_report = stage_query_reports[query_idx]
                        query_report["results"] += len(videos)
                        retrieval_metrics["raw_discovered_videos"] += len(videos)

                        for video in videos:
                            if not self._recovery_stage_allows_source_surface(
                                source_surface=str(video.get("search_source") or query_candidate.source_surface or ""),
                                page_hint=safe_page_hint,
                                recovery_stage=safe_recovery_stage,
                            ):
                                continue
                            video_id = str(video.get("id") or "").strip()
                            if not video_id or video_id in seen_video_ids:
                                query_report["dropped_duplicate"] += 1
                                retrieval_metrics["dropped_duplicate"] += 1
                                continue
                            seen_video_ids.add(video_id)
                            video_duration_val = int(video.get("duration_sec") or 0)
                            existing_for_video = existing_video_counts.get(video_id, 0)
                            generated_for_video = generated_video_counts.get(video_id, 0)
                            video_segment_cap = self._video_segment_cap(
                                video_duration_sec=video_duration_val,
                                fast_mode=fast_mode,
                                default_cap=default_max_segments_per_video,
                                page_hint=safe_page_hint,
                            )
                            if existing_for_video + generated_for_video >= video_segment_cap:
                                query_report["dropped_segment_cap"] += 1
                                retrieval_metrics["dropped_segment_cap"] += 1
                                continue
                            if strict_duration and not self._video_matches_preferred_duration(
                                video_duration_sec=video_duration_val,
                                preferred_video_duration=safe_video_duration_pref,
                                allow_unknown_duration=allow_unknown_duration,
                            ):
                                query_report["dropped_duration"] += 1
                                retrieval_metrics["dropped_duration"] += 1
                                continue
                            quick_signals = self._quick_candidate_metadata_gate(
                                video=video,
                                query_candidate=query_candidate,
                                concept_terms=concept_terms,
                                context_terms=context_terms,
                                subject_tag=subject_tag,
                                root_topic_terms=root_topic_terms,
                                strict_topic_only=strict_topic_only,
                                require_context=bool(context_terms) and vague_topic,
                                fast_mode=fast_mode,
                            )
                            if not bool(quick_signals.get("passes")):
                                query_report["dropped_metadata_gate"] += 1
                                retrieval_metrics["dropped_metadata_gate"] += 1
                                continue
                            scorable_videos.append((query_idx, query_candidate, video, quick_signals))

                        # Batch-embed all candidate metadata texts in one call to warm the cache
                        if concept_embedding is not None and scorable_videos:
                            batch_texts = []
                            for _, _, sv, quick_signals in scorable_videos:
                                batch_texts.append(str(quick_signals.get("metadata_text") or self._video_metadata_text(sv)))
                                batch_texts.append(str(sv.get("title") or "").strip())
                                if not bool(quick_signals.get("skip_semantic_description")):
                                    batch_texts.append(str(sv.get("description") or "").strip())
                            try:
                                self.embedding_service.embed_texts(conn, batch_texts)
                            except Exception:
                                pass

                        for query_idx, query_candidate, video, quick_signals in scorable_videos:
                            raise_if_cancelled()
                            video_id = str(video.get("id") or "").strip()
                            video_duration = int(video.get("duration_sec") or 0)

                            if query_idx < len(stage_query_reports):
                                query_report = stage_query_reports[query_idx]
                            else:
                                query_report = {"kept": 0}

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
                                preferred_video_duration=safe_video_duration_pref,
                                stage_name=stage.name,
                                require_context=bool(context_terms) and vague_topic,
                                fast_mode=fast_mode,
                                quick_signals=quick_signals,
                            )
                            if not bool(ranking.get("passes", False)):
                                query_report["dropped_ranking"] += 1
                                retrieval_metrics["dropped_ranking"] += 1
                                continue
                            if not dry_run:
                                self._upsert_video(conn, video)
                            query_report["kept"] += 1
                            retrieval_metrics["filtered_relevant_videos"] += 1
                            if float(ranking.get("discovery_score") or 0.0) >= 0.2:
                                good_results += 1

                            stage_candidates.append(
                                {
                                    "video": video,
                                    "video_id": video_id,
                                    "video_duration": video_duration,
                                    "video_relevance": ranking["text_relevance"],
                                    "ranking": ranking,
                                    "query_candidate": query_candidate,
                                    "stage": stage.name,
                                }
                            )
                        transcript_prefetch_task = self._maybe_launch_transcript_prefetch(
                            prefetch_task=transcript_prefetch_task,
                            stage_candidates=stage_candidates,
                            transcript_budget=transcript_budget,
                            clip_min_len=clip_min_len,
                            clip_max_len=clip_max_len,
                            fast_mode=fast_mode,
                        )

                        all_query_reports.extend(stage_query_reports)
                        if good_results >= stage.min_good_results and len(stage_candidates) >= stage.min_good_results:
                            break

                    if pass_index == 0 and self._fast_pass_is_sufficient(
                        stage_candidates,
                        fast_mode=fast_mode,
                        max_generation_target=max_generation_target,
                        retrieval_profile=safe_retrieval_profile,
                    ):
                        break

                raise_if_cancelled()
                if (
                    safe_retrieval_profile == "bootstrap"
                    and self._bootstrap_pool_is_weak(
                        stage_candidates,
                        max_generation_target=max_generation_target,
                    )
                ):
                    recovery_query = concept_plan.recovery_queries[0] if concept_plan.recovery_queries else None
                    if recovery_query is not None:
                        recovery_candidate = QueryCandidate(
                            text=recovery_query.text,
                            strategy=recovery_query.strategy,
                            confidence=recovery_query.confidence,
                            source_terms=list(recovery_query.source_terms),
                            weight=recovery_query.weight,
                            stage="recovery",
                            source_surface=recovery_query.source_surface,
                            family_key=recovery_query.family_key,
                            source_family=recovery_query.source_family,
                            anchor_mode=recovery_query.anchor_mode,
                            seed_video_id=recovery_query.seed_video_id,
                            seed_channel_id=recovery_query.seed_channel_id,
                        )
                        recovery_reports = [
                            {
                                "query": recovery_candidate.text,
                                "strategy": recovery_candidate.strategy,
                                "stage": "recovery",
                                "source_terms": recovery_candidate.source_terms,
                                "weight": float(recovery_candidate.weight),
                                "surface": recovery_candidate.source_surface,
                                "family_key": recovery_candidate.family_key,
                                "source_family": recovery_candidate.source_family,
                                "anchor_mode": recovery_candidate.anchor_mode,
                                "seed_video_id": recovery_candidate.seed_video_id,
                                "seed_channel_id": recovery_candidate.seed_channel_id,
                                "results": 0,
                                "kept": 0,
                                "dropped_duplicate": 0,
                                "dropped_segment_cap": 0,
                                "dropped_duration": 0,
                                "dropped_metadata_gate": 0,
                                "dropped_ranking": 0,
                            }
                        ]
                        recovery_jobs = self._stage_search_jobs_parallel(
                            stage_name="recovery",
                            stage_queries=[recovery_candidate],
                            stage_duration_plan=self._stage_duration_plan(
                                stage_name="recovery",
                                preferred_video_duration=safe_video_duration_pref,
                                video_pool_mode=safe_video_pool_mode,
                                fast_mode=fast_mode,
                                retrieval_profile=safe_retrieval_profile,
                            ),
                            max_results_for_query=max_results_for_query,
                            creative_commons_only=creative_commons_only,
                            fast_mode=fast_mode,
                            retrieval_profile=safe_retrieval_profile,
                            page_hint=safe_page_hint,
                            recovery_stage=safe_recovery_stage,
                            allow_external_fallbacks=False,
                            variant_limit=1,
                            subject_tag=subject_tag,
                        )
                        for query_idx, _duration_idx, query_candidate, _duration, videos in recovery_jobs:
                            if query_idx >= len(recovery_reports):
                                continue
                            query_report = recovery_reports[query_idx]
                            query_report["results"] += len(videos)
                            retrieval_metrics["raw_discovered_videos"] += len(videos)

                            for video in videos:
                                if not self._recovery_stage_allows_source_surface(
                                    source_surface=str(video.get("search_source") or query_candidate.source_surface or ""),
                                    page_hint=safe_page_hint,
                                    recovery_stage=safe_recovery_stage,
                                ):
                                    continue
                                video_id = str(video.get("id") or "").strip()
                                if not video_id or video_id in seen_video_ids:
                                    query_report["dropped_duplicate"] += 1
                                    retrieval_metrics["dropped_duplicate"] += 1
                                    continue
                                seen_video_ids.add(video_id)
                                video_duration = int(video.get("duration_sec") or 0)
                                existing_for_video = existing_video_counts.get(video_id, 0)
                                generated_for_video = generated_video_counts.get(video_id, 0)
                                video_segment_cap = self._video_segment_cap(
                                    video_duration_sec=video_duration,
                                    fast_mode=fast_mode,
                                    default_cap=default_max_segments_per_video,
                                    page_hint=safe_page_hint,
                                )
                                if existing_for_video + generated_for_video >= video_segment_cap:
                                    query_report["dropped_segment_cap"] += 1
                                    retrieval_metrics["dropped_segment_cap"] += 1
                                    continue
                                quick_signals = self._quick_candidate_metadata_gate(
                                    video=video,
                                    query_candidate=query_candidate,
                                    concept_terms=concept_terms,
                                    context_terms=context_terms,
                                    subject_tag=subject_tag,
                                    root_topic_terms=root_topic_terms,
                                    strict_topic_only=strict_topic_only,
                                    require_context=bool(context_terms) and vague_topic,
                                    fast_mode=fast_mode,
                                )
                                if not bool(quick_signals.get("passes")):
                                    query_report["dropped_metadata_gate"] += 1
                                    retrieval_metrics["dropped_metadata_gate"] += 1
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
                                    preferred_video_duration=safe_video_duration_pref,
                                    stage_name="recovery",
                                    require_context=bool(context_terms) and vague_topic,
                                    fast_mode=fast_mode,
                                    quick_signals=quick_signals,
                                )
                                if not bool(ranking.get("passes", False)):
                                    query_report["dropped_ranking"] += 1
                                    retrieval_metrics["dropped_ranking"] += 1
                                    continue
                                if not dry_run:
                                    self._upsert_video(conn, video)
                                query_report["kept"] += 1
                                retrieval_metrics["filtered_relevant_videos"] += 1
                                stage_candidates.append(
                                    {
                                        "video": video,
                                        "video_id": video_id,
                                        "video_duration": video_duration,
                                        "video_relevance": ranking["text_relevance"],
                                        "ranking": ranking,
                                        "query_candidate": query_candidate,
                                        "stage": "recovery",
                                    }
                                )
                        transcript_prefetch_task = self._maybe_launch_transcript_prefetch(
                            prefetch_task=transcript_prefetch_task,
                            stage_candidates=stage_candidates,
                            transcript_budget=transcript_budget,
                            clip_min_len=clip_min_len,
                            clip_max_len=clip_max_len,
                            fast_mode=fast_mode,
                        )
                        all_query_reports.extend(recovery_reports)

                second_recovery_query = (
                    concept_plan.recovery_queries[1]
                    if len(concept_plan.recovery_queries) > 1
                    else None
                )
                if (
                    second_recovery_query is not None
                    and self._bootstrap_pool_is_weak(
                        stage_candidates,
                        max_generation_target=max_generation_target,
                    )
                ):
                    fallback_candidate = QueryCandidate(
                        text=second_recovery_query.text,
                        strategy=second_recovery_query.strategy,
                        confidence=second_recovery_query.confidence,
                        source_terms=list(second_recovery_query.source_terms),
                        weight=second_recovery_query.weight,
                        stage="recovery",
                        source_surface=second_recovery_query.source_surface,
                        family_key=second_recovery_query.family_key,
                        source_family=second_recovery_query.source_family,
                        anchor_mode=second_recovery_query.anchor_mode,
                        seed_video_id=second_recovery_query.seed_video_id,
                        seed_channel_id=second_recovery_query.seed_channel_id,
                    )
                    fallback_reports = [
                        {
                            "query": fallback_candidate.text,
                            "strategy": fallback_candidate.strategy,
                            "stage": "recovery",
                            "source_terms": fallback_candidate.source_terms,
                            "weight": float(fallback_candidate.weight),
                            "surface": fallback_candidate.source_surface,
                            "family_key": fallback_candidate.family_key,
                            "source_family": fallback_candidate.source_family,
                            "anchor_mode": fallback_candidate.anchor_mode,
                            "seed_video_id": fallback_candidate.seed_video_id,
                            "seed_channel_id": fallback_candidate.seed_channel_id,
                            "results": 0,
                            "kept": 0,
                            "dropped_duplicate": 0,
                            "dropped_segment_cap": 0,
                            "dropped_duration": 0,
                            "dropped_metadata_gate": 0,
                            "dropped_ranking": 0,
                        }
                    ]
                    fallback_jobs = self._stage_search_jobs_parallel(
                        stage_name="recovery",
                        stage_queries=[fallback_candidate],
                        stage_duration_plan=self._stage_duration_plan(
                            stage_name="recovery",
                            preferred_video_duration=safe_video_duration_pref,
                            video_pool_mode=safe_video_pool_mode,
                            fast_mode=fast_mode,
                            retrieval_profile=safe_retrieval_profile,
                        ),
                        max_results_for_query=max_results_for_query,
                        creative_commons_only=creative_commons_only,
                        fast_mode=fast_mode,
                        retrieval_profile=safe_retrieval_profile,
                        page_hint=safe_page_hint,
                        recovery_stage=safe_recovery_stage,
                        allow_external_fallbacks=True,
                        variant_limit=1,
                        subject_tag=subject_tag,
                    )
                    for query_idx, _duration_idx, query_candidate, _duration, videos in fallback_jobs:
                        if query_idx >= len(fallback_reports):
                            continue
                        query_report = fallback_reports[query_idx]
                        query_report["results"] += len(videos)
                        retrieval_metrics["raw_discovered_videos"] += len(videos)

                        for video in videos:
                            if not self._recovery_stage_allows_source_surface(
                                source_surface=str(video.get("search_source") or query_candidate.source_surface or ""),
                                page_hint=safe_page_hint,
                                recovery_stage=safe_recovery_stage,
                            ):
                                continue
                            video_id = str(video.get("id") or "").strip()
                            if not video_id or video_id in seen_video_ids:
                                query_report["dropped_duplicate"] += 1
                                retrieval_metrics["dropped_duplicate"] += 1
                                continue
                            seen_video_ids.add(video_id)
                            video_duration = int(video.get("duration_sec") or 0)
                            existing_for_video = existing_video_counts.get(video_id, 0)
                            generated_for_video = generated_video_counts.get(video_id, 0)
                            video_segment_cap = self._video_segment_cap(
                                video_duration_sec=video_duration,
                                fast_mode=fast_mode,
                                default_cap=default_max_segments_per_video,
                                page_hint=safe_page_hint,
                            )
                            if existing_for_video + generated_for_video >= video_segment_cap:
                                query_report["dropped_segment_cap"] += 1
                                retrieval_metrics["dropped_segment_cap"] += 1
                                continue
                            quick_signals = self._quick_candidate_metadata_gate(
                                video=video,
                                query_candidate=query_candidate,
                                concept_terms=concept_terms,
                                context_terms=context_terms,
                                subject_tag=subject_tag,
                                root_topic_terms=root_topic_terms,
                                strict_topic_only=strict_topic_only,
                                require_context=bool(context_terms) and vague_topic,
                                fast_mode=fast_mode,
                            )
                            if not bool(quick_signals.get("passes")):
                                query_report["dropped_metadata_gate"] += 1
                                retrieval_metrics["dropped_metadata_gate"] += 1
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
                                preferred_video_duration=safe_video_duration_pref,
                                stage_name="recovery",
                                require_context=bool(context_terms) and vague_topic,
                                fast_mode=fast_mode,
                                quick_signals=quick_signals,
                            )
                            if not bool(ranking.get("passes", False)):
                                query_report["dropped_ranking"] += 1
                                retrieval_metrics["dropped_ranking"] += 1
                                continue
                            if not dry_run:
                                self._upsert_video(conn, video)
                            query_report["kept"] += 1
                            retrieval_metrics["filtered_relevant_videos"] += 1
                            stage_candidates.append(
                                {
                                    "video": video,
                                    "video_id": video_id,
                                    "video_duration": video_duration,
                                    "video_relevance": ranking["text_relevance"],
                                    "ranking": ranking,
                                    "query_candidate": query_candidate,
                                    "stage": "recovery",
                                }
                            )
                    transcript_prefetch_task = self._maybe_launch_transcript_prefetch(
                        prefetch_task=transcript_prefetch_task,
                        stage_candidates=stage_candidates,
                        transcript_budget=transcript_budget,
                        clip_min_len=clip_min_len,
                        clip_max_len=clip_max_len,
                        fast_mode=fast_mode,
                    )
                    all_query_reports.extend(fallback_reports)

            bootstrap_local_recovery = (
                safe_retrieval_profile == "bootstrap"
                and self._topic_breadth_class(subject_tag) == "curated_broad"
            )
            if not stage_candidates and (allow_adjacent_recovery or bootstrap_local_recovery):
                local_candidates = self._recover_candidates_from_local_corpus(
                    conn,
                    material_id=material_id,
                    concept_terms=concept_terms,
                    context_terms=context_terms,
                    concept_embedding=concept_embedding,
                    subject_tag=subject_tag,
                    root_topic_terms=root_topic_terms,
                    visual_spec=visual_spec,
                    preferred_video_duration=safe_video_duration_pref,
                    fast_mode=fast_mode,
                    strict_topic_only=strict_topic_only,
                    existing_video_counts=existing_video_counts,
                    generated_video_counts=generated_video_counts,
                    max_segments_per_video=default_max_segments_per_video,
                    concept_title=str(concept.get("title") or ""),
                    page_hint=safe_page_hint,
                    bootstrap_fallback=bootstrap_local_recovery,
                )
                if local_candidates:
                    stage_candidates = local_candidates
                    retrieval_metrics["raw_discovered_videos"] += len(local_candidates)
                    retrieval_metrics["filtered_relevant_videos"] += len(local_candidates)
                    all_query_reports.append(
                        {
                            "query": f"local_cache:{concept.get('title')}",
                            "strategy": "literal" if bootstrap_local_recovery else "recovery_adjacent",
                            "stage": "high_precision" if bootstrap_local_recovery else "recovery",
                            "source_terms": concept_terms[:4],
                            "weight": 0.76 if bootstrap_local_recovery else 0.5,
                            "surface": "local_bootstrap" if bootstrap_local_recovery else "local_cache",
                            "family_key": self._frontier_family_key(
                                self.FRONTIER_ROOT_EXACT if bootstrap_local_recovery else self.FRONTIER_RECOVERY_GRAPH,
                                f"local_cache:{concept.get('title')}",
                            ),
                            "source_family": "other",
                            "anchor_mode": "root_exact" if bootstrap_local_recovery else "recovery_graph",
                            "seed_video_id": "",
                            "seed_channel_id": "",
                            "results": len(local_candidates),
                            "kept": len(local_candidates),
                            "dropped_duplicate": 0,
                            "dropped_segment_cap": 0,
                            "dropped_duration": 0,
                            "dropped_metadata_gate": 0,
                            "dropped_ranking": 0,
                        }
                    )
                    transcript_prefetch_task = self._maybe_launch_transcript_prefetch(
                        prefetch_task=transcript_prefetch_task,
                        stage_candidates=stage_candidates,
                        transcript_budget=transcript_budget,
                        clip_min_len=clip_min_len,
                        clip_max_len=clip_max_len,
                        fast_mode=fast_mode,
                    )

            if not stage_candidates:
                self._shutdown_transcript_prefetch_task(
                    transcript_prefetch_task,
                    wait=False,
                    cancel_futures=True,
                )
                transcript_prefetch_task = None
                self._persist_retrieval_debug_run(
                    conn,
                    run=retrieval_run,
                    query_reports=all_query_reports,
                    candidate_rows=[],
                    selected=None,
                    failure_reason=self._exhaustion_reason_from_metrics(
                        metrics=retrieval_metrics,
                        selected=None,
                    ),
                    dry_run=dry_run,
                    metrics=retrieval_metrics,
                )
                if request_key:
                    self._persist_request_frontier_reports(
                        conn,
                        material_id=material_id,
                        request_key=request_key,
                        query_reports=all_query_reports,
                        accepted_reels_by_family=accepted_reels_by_family,
                    )
                continue

            if self.retrieval_tier2_enabled:
                stage_candidates = self._collapse_near_duplicate_candidates(stage_candidates)
            else:
                stage_candidates = sorted(
                    stage_candidates,
                    key=lambda row: float((row.get("ranking") or {}).get("final_score") or 0.0),
                    reverse=True,
                )
            if self.retrieval_tier2_enabled:
                ranked_candidates = self._diversify_video_candidates(
                    stage_candidates,
                    top_k=max(12, transcript_budget * 2),
                )
            else:
                ranked_candidates = stage_candidates[: max(12, transcript_budget * 2)]

            # Hard drop for ambiguous concepts. Runs in BOTH fast and non-fast
            # modes. When the subject/concept contains a known homonym
            # (e.g. "calculus", "cell", "python"), any candidate classified as
            # entertainment_media or low_quality_compilation is almost
            # certainly the wrong sense of the word and should be dropped
            # outright rather than scored against a soft gate.
            concept_ambig_tokens = normalize_terms([
                str(subject_tag or ""),
                str(concept.get("title") or ""),
            ])
            ambiguous_concept = bool(concept_ambig_tokens & self.AMBIGUOUS_CONCEPT_TOKENS)
            if ambiguous_concept and ranked_candidates:
                pre_count = len(ranked_candidates)
                filtered: list[dict[str, Any]] = []
                for cand in ranked_candidates:
                    video_row = cand.get("video") or {}
                    tier = self._infer_channel_tier(
                        channel=str(video_row.get("channel_title") or "").lower(),
                        title=str(video_row.get("title") or "").lower(),
                    )
                    if tier in {"entertainment_media", "low_quality_compilation"}:
                        if self.retrieval_debug_logging:
                            logger.info(
                                "reels.loop drop_candidate concept=%s video=%s tier=%s reason=ambiguous_concept_entertainment",
                                str(concept.get("id") or ""),
                                str(cand.get("video_id") or ""),
                                tier,
                            )
                        continue
                    filtered.append(cand)
                dropped = pre_count - len(filtered)
                if dropped:
                    retrieval_metrics["ambiguous_concept_dropped"] = int(
                        retrieval_metrics.get("ambiguous_concept_dropped", 0)
                    ) + dropped
                ranked_candidates = filtered

            # LLM relevance pre-filter: drops clearly irrelevant candidates before
            # the expensive transcript-fetch step. Only runs in non-fast mode.
            if not fast_mode and self.openai_client and ranked_candidates:
                pre_filter_count = len(ranked_candidates)
                ranked_candidates = self._llm_relevance_prefilter(
                    conn,
                    candidates=ranked_candidates,
                    concept_text=str(concept.get("title") or ""),
                    subject_tag=subject_tag,
                    context_terms=context_terms,
                )
                retrieval_metrics["llm_prefilter_dropped"] = int(
                    retrieval_metrics.get("llm_prefilter_dropped", 0)
                ) + (pre_filter_count - len(ranked_candidates))

            transcript_candidates = ranked_candidates[:transcript_budget]
            retrieval_metrics["clip_candidate_videos"] = max(
                int(retrieval_metrics.get("clip_candidate_videos") or 0),
                len(transcript_candidates),
            )
            transcript_prefetch_ids = self._seed_transcript_prefetch_ids(
                candidates=transcript_candidates,
                transcript_budget=transcript_budget,
                clip_min_len=clip_min_len,
                clip_max_len=clip_max_len,
            )

            selected_outcome: dict[str, Any] | None = None
            candidate_records: list[dict[str, Any]] = []
            for candidate in ranked_candidates[: max(20, transcript_budget * 2)]:
                ranking = dict(candidate.get("ranking") or {})
                candidate_records.append(
                    {
                        "video_id": str(candidate.get("video_id") or ""),
                        "video_title": str((candidate.get("video") or {}).get("title") or ""),
                        "channel_title": str((candidate.get("video") or {}).get("channel_title") or ""),
                        "strategy": str((candidate.get("query_candidate") or QueryCandidate("", "", 0.0)).strategy or ""),
                        "stage": str(candidate.get("stage") or ""),
                        "query": str((candidate.get("query_candidate") or QueryCandidate("", "", 0.0)).text or ""),
                        "final_score": float(ranking.get("final_score") or 0.0),
                        "discovery_score": float(ranking.get("discovery_score") or 0.0),
                        "clipability_score": float(ranking.get("clipability_score") or 0.0),
                        "source_surface": str((candidate.get("video") or {}).get("search_source") or ""),
                        "discovery_path": str((candidate.get("video") or {}).get("discovery_path") or ""),
                        "crawl_depth": int((candidate.get("video") or {}).get("crawl_depth") or 0),
                        "seed_video_id": str((candidate.get("video") or {}).get("seed_video_id") or ""),
                        "seed_channel_id": str((candidate.get("video") or {}).get("seed_channel_id") or ""),
                        "features": ranking.get("features") or {},
                    }
                )

            transcript_cache = self._collect_transcript_prefetch(
                prefetch_task=transcript_prefetch_task,
                requested_video_ids=transcript_prefetch_ids,
                fast_mode=fast_mode,
            )

            for candidate in transcript_candidates:
                raise_if_cancelled()
                video = candidate["video"]
                video_id = str(candidate["video_id"])
                video_duration = int(candidate["video_duration"])
                ranking = dict(candidate.get("ranking") or {})
                query_candidate = candidate.get("query_candidate")
                if not isinstance(query_candidate, QueryCandidate):
                    query_candidate = QueryCandidate(text="", strategy="literal", confidence=0.5)

                existing_for_video = existing_video_counts.get(video_id, 0)
                generated_for_video = generated_video_counts.get(video_id, 0)
                video_segment_cap = self._video_segment_cap(
                    video_duration_sec=video_duration,
                    fast_mode=fast_mode,
                    default_cap=default_max_segments_per_video,
                    page_hint=safe_page_hint,
                )
                if existing_for_video + generated_for_video >= video_segment_cap:
                    continue
                remaining_segment_capacity = max(
                    1,
                    video_segment_cap - (existing_for_video + generated_for_video),
                )

                use_full_short_clip = self._should_use_full_short_clip(
                    prefer_short_query=self._video_duration_bucket(video_duration) == "short",
                    video_duration_sec=video_duration,
                    clip_min_len=clip_min_len,
                    clip_max_len=clip_max_len,
                )
                mined_transcript = candidate.get("mined_transcript")
                transcript = (
                    list(mined_transcript)
                    if isinstance(mined_transcript, list)
                    else (
                        []
                        if use_full_short_clip
                        else self._transcript_for_candidate(
                            prefetch_task=transcript_prefetch_task,
                            transcript_cache=transcript_cache,
                            video_id=video_id,
                            fast_mode=fast_mode,
                        )
                    )
                )
                transcript_ranking: dict[str, Any] | None = None
                transcript_quality: TranscriptQuality | None = None
                if transcript:
                    transcript_ranking = self._score_transcript_alignment(
                        conn,
                        transcript=transcript,
                        concept_terms=concept_terms,
                        context_terms=context_terms,
                        concept_embedding=concept_embedding,
                        subject_tag=subject_tag,
                        visual_spec=visual_spec,
                        require_context=bool(context_terms) and vague_topic,
                        fast_mode=fast_mode,
                    )
                    ranking["discovery_score"] = min(
                        1.0,
                        float(ranking.get("discovery_score") or 0.0) + 0.12 * float(transcript_ranking.get("concept_match") or 0.0),
                    )
                    ranking["clipability_score"] = min(
                        1.0,
                        float(ranking.get("clipability_score") or 0.0)
                        + 0.18 * float(transcript_ranking.get("clipability_signal") or 0.0),
                    )
                    # Soft penalty for poor transcript coverage.
                    transcript_quality = validate_transcript(
                        transcript, float(video_duration) if video_duration else None,
                    )
                    if not transcript_quality.is_adequate and transcript_quality.coverage_ratio < 0.90:
                        coverage_penalty = max(0.0, 0.90 - transcript_quality.coverage_ratio)
                        ranking["discovery_score"] = max(
                            0.0,
                            float(ranking.get("discovery_score") or 0.0) * (1.0 - coverage_penalty),
                        )
                    ranking["transcript_coverage"] = transcript_quality.coverage_ratio

                segments: list[SegmentMatch] = []
                mined_segments = candidate.get("mined_segments")
                if isinstance(mined_segments, list) and mined_segments:
                    segments = list(mined_segments)
                elif transcript:
                    target_segment_budget = min(10, max(4, remaining_segment_capacity + 2))

                    # ---- Topic-boundary cutting for long-form videos ---- #
                    # For non-Short videos, use topic_cut.py for precise boundary
                    # detection: YouTube Chapters → Gemini/Groq LLM → sentence-
                    # transformer → Jaccard heuristic. Each clip starts where the
                    # creator introduces a topic and ends when they transition away.
                    # Falls back to mention-clustering if topic_cut returns nothing.
                    if not use_full_short_clip:
                        segments = self._topic_boundary_segments_for_concept(
                            transcript=transcript,
                            video_id=video_id,
                            video_duration_sec=video_duration,
                            clip_min_len=clip_min_len,
                            clip_max_len=clip_max_len,
                            max_segments=target_segment_budget,
                            concept_terms=concept_terms,
                            concept_title=str(concept.get("title") or ""),
                            info_dict=video.get("info_dict"),
                        )
                        # Fallback to keyword-mention clustering when topic-boundary
                        # detection returns nothing (Short, no transcript, etc.)
                        if not segments:
                            segments = self._topic_cut_segments_for_concept(
                                transcript=transcript,
                                video_id=video_id,
                                video_duration_sec=video_duration,
                                clip_min_len=clip_min_len,
                                clip_max_len=clip_max_len,
                                max_segments=target_segment_budget,
                                concept_terms=concept_terms,
                            )

                    # Legacy embedding-based path: runs when topic_cut returned
                    # nothing (transcript too short, classification mismatch) AND
                    # for Shorts that have a mined transcript. This guarantees we
                    # never silently emit zero reels for a video with a transcript.
                    if not segments:
                        if fast_mode:
                            segments = self._fast_segments_from_transcript(
                                transcript=transcript,
                                concept_terms=concept_terms,
                                max_segments=min(6, max(2, remaining_segment_capacity + 1)),
                            )
                        else:
                            chunks, chunk_embeddings = self._load_or_create_transcript_chunks(conn, video_id, transcript)
                            if chunks and len(chunk_embeddings) > 0:
                                if concept_embedding is not None:
                                    semantic_segments = select_segments(
                                        concept_embedding,
                                        chunk_embeddings,
                                        chunks,
                                        concept_terms=concept_terms,
                                        top_k=max(4, min(target_segment_budget, 6 if vague_topic else 5)),
                                    )
                                    short_segments = self._split_video_into_short_segments(
                                        concept_embedding,
                                        chunk_embeddings,
                                        chunks,
                                        concept_terms=concept_terms,
                                        max_segments=target_segment_budget,
                                    )
                                    segments = self._merge_unique_segments(
                                        [*semantic_segments, *short_segments],
                                        max_items=target_segment_budget,
                                    )
                                else:
                                    segments = self._fast_segments_from_transcript(
                                        transcript=transcript,
                                        concept_terms=concept_terms,
                                        max_segments=min(8, max(3, remaining_segment_capacity + 1)),
                                    )
                        if not segments:
                            segments = self._fallback_segments_from_transcript(transcript)
                else:
                    metadata_segment = self._fallback_segment_from_video_metadata(
                        video,
                        concept_terms,
                        target_clip_duration_sec=safe_target_clip_duration,
                    )
                    if metadata_segment:
                        segments = [metadata_segment]
                    elif not use_full_short_clip:
                        continue

                if not segments:
                    continue

                segment_candidates = self._rank_segments_by_relevance(
                    conn,
                    segments=segments,
                    concept_terms=concept_terms,
                    context_terms=context_terms,
                    concept_embedding=concept_embedding,
                    subject_tag=subject_tag,
                    require_context=bool(context_terms) and vague_topic,
                    fast_mode=fast_mode,
                )
                if not segment_candidates:
                    continue

                # Chain-state for topic_cut sub-segments. There are TWO
                # chaining regimes:
                #
                #   1. Sub-parts of a single oversized cluster share a
                #      ``cluster_group_id`` (assigned by
                #      `_topic_cut_segments_for_concept`). These must chain
                #      exactly — no gap, no overlap.
                #   2. Neighbouring clusters that were bridged at their gap
                #      midpoint by `_topic_cut_segments_for_concept` have
                #      DIFFERENT cluster_group_ids but their t_start/t_end
                #      still line up. We detect this at the main-loop level
                #      by tracking the last refined end per video, and when
                #      the next segment's raw t_start is within a small
                #      tolerance, we chain across the cluster boundary too.
                cluster_chain_last_end: dict[str, float] = {}
                topic_cut_last_refined_end: float | None = None
                TOPIC_CUT_BRIDGE_TOLERANCE_SEC = 2.0

                # Process segments in TEMPORAL order so consecutive clusters
                # in the video are refined in the same order they appear.
                # Sub-parts of the same cluster naturally sort in
                # cluster_sub_index order because their t_start values are
                # strictly increasing.
                segment_candidates = sorted(
                    segment_candidates,
                    key=lambda item: (
                        float(item[0].t_start),
                        int(getattr(item[0], "cluster_sub_index", 0)),
                    ),
                )

                for segment, segment_relevance in segment_candidates:
                    raise_if_cancelled()
                    # Compute one or more consecutive clip windows for this segment.
                    # If the segment exceeds max_len, _split_into_consecutive_windows
                    # divides it into multiple sentence-aligned reels so that
                    # long-form topic content isn't dropped.
                    if use_full_short_clip:
                        single_win = self._full_short_clip_window(
                            video_duration,
                            transcript=transcript if transcript else None,
                        )
                        clip_windows: list[tuple[float, float]] = [single_win] if single_win else []
                    elif getattr(segment, "source", "legacy") == "topic_cut":
                        # Topic-cut segments were produced by topic_cut.py,
                        # which already picks topic-aligned boundaries (LLM /
                        # chapters / Jaccard) and snaps t_end to cue gaps.
                        # However, cue boundaries are NOT sentence boundaries;
                        # we still must guarantee the reel ends on a complete
                        # sentence when possible. Two branches:
                        #
                        #   - seg_span > clip_max_len + 16 → split into
                        #     consecutive sentence-aligned windows so the topic
                        #     carries across multiple reels seamlessly and
                        #     each honors the user's max_len.
                        #   - seg_span ≤ clip_max_len + 16 → run the same
                        #     transcript refiner with a generous max_len so the
                        #     end lands on a sentence terminator WITHOUT
                        #     truncating the topic below the user's max. Pass
                        #     min_start=segment.t_start so refinement can only
                        #     move the start FORWARD (never before the topic
                        #     introduction).
                        seg_span = max(0.0, float(segment.t_end) - float(segment.t_start))
                        # Chaining: if this segment is a sub-part of an oversized
                        # topic cluster, force continuity with the previous
                        # sub-part's refined end so the reels play back as one
                        # continuous topic discussion.
                        chain_id = str(getattr(segment, "cluster_group_id", "") or "")
                        chain_prev_end = cluster_chain_last_end.get(chain_id) if chain_id else None
                        # When chained within the same cluster, the next
                        # sub-part starts at exactly the previous sub-part's
                        # refined end — no gap, no overlap.
                        if chain_prev_end is not None:
                            effective_start = float(chain_prev_end)
                        elif (
                            topic_cut_last_refined_end is not None
                            and abs(float(segment.t_start) - topic_cut_last_refined_end)
                            <= TOPIC_CUT_BRIDGE_TOLERANCE_SEC
                        ):
                            # Bridged adjacent clusters (different
                            # cluster_group_id but touching timestamps). Chain
                            # across the boundary so there's no overlap.
                            effective_start = float(topic_cut_last_refined_end)
                        else:
                            effective_start = float(segment.t_start)
                        if transcript and seg_span > float(clip_max_len) + 16.0:
                            clip_windows = self._split_into_consecutive_windows(
                                transcript=transcript,
                                segment_start=effective_start,
                                segment_end=segment.t_end,
                                video_duration_sec=video_duration,
                                min_len=clip_min_len,
                                max_len=clip_max_len,
                            )
                        elif transcript:
                            # Give the refiner enough headroom that it won't
                            # truncate a legitimate topic span shorter than
                            # the user's max_len; also ensure min_len doesn't
                            # pull the start backward through the topic intro.
                            refiner_max = int(max(seg_span + 16.0, float(clip_max_len)))
                            refiner_min = max(1, min(int(clip_min_len), int(max(1.0, seg_span * 0.6))))
                            single_win = self._refine_clip_window_from_transcript(
                                transcript=transcript,
                                proposed_start=effective_start,
                                proposed_end=segment.t_end,
                                video_duration_sec=video_duration,
                                min_len=refiner_min,
                                max_len=refiner_max,
                                min_start=effective_start,
                            )
                            if not single_win:
                                single_win = self._normalize_clip_window(
                                    effective_start,
                                    segment.t_end,
                                    video_duration,
                                    min_len=clip_min_len,
                                    max_len=clip_max_len,
                                    allow_exceed_max=True,
                                )
                            clip_windows = [single_win] if single_win else []
                        else:
                            single_win = self._normalize_clip_window(
                                effective_start,
                                segment.t_end,
                                video_duration,
                                min_len=clip_min_len,
                                max_len=clip_max_len,
                                allow_exceed_max=True,
                            )
                            clip_windows = [single_win] if single_win else []
                        # Record the last refined end for this cluster chain so
                        # subsequent sub-parts can snap to it. Also record the
                        # video-wide last refined end for cross-cluster bridging.
                        if clip_windows:
                            if chain_id:
                                cluster_chain_last_end[chain_id] = float(clip_windows[-1][1])
                            topic_cut_last_refined_end = float(clip_windows[-1][1])
                    elif transcript:
                        # Preserve the old behavior for short/normal segments so
                        # working topics aren't perturbed. Only invoke the split
                        # path when the segment is long enough that a single
                        # max_len reel would noticeably truncate it.
                        seg_span = max(0.0, float(segment.t_end) - float(segment.t_start))
                        if seg_span > float(clip_max_len) + 16.0:
                            clip_windows = self._split_into_consecutive_windows(
                                transcript=transcript,
                                segment_start=segment.t_start,
                                segment_end=segment.t_end,
                                video_duration_sec=video_duration,
                                min_len=clip_min_len,
                                max_len=clip_max_len,
                            )
                        else:
                            single_win = self._refine_clip_window_from_transcript(
                                transcript=transcript,
                                proposed_start=segment.t_start,
                                proposed_end=segment.t_end,
                                video_duration_sec=video_duration,
                                min_len=clip_min_len,
                                max_len=clip_max_len,
                            )
                            clip_windows = [single_win] if single_win else []
                    else:
                        single_win = self._normalize_clip_window(
                            segment.t_start,
                            segment.t_end,
                            video_duration,
                            min_len=clip_min_len,
                            max_len=clip_max_len,
                        )
                        clip_windows = [single_win] if single_win else []
                    if self.retrieval_debug_logging:
                        seg_span_dbg = max(0.0, float(segment.t_end) - float(segment.t_start))
                        if use_full_short_clip:
                            branch_dbg = "full_short"
                        elif getattr(segment, "source", "legacy") == "topic_cut":
                            if transcript and seg_span_dbg > float(clip_max_len) + 16.0:
                                branch_dbg = "topic_cut_split"
                            else:
                                branch_dbg = "topic_cut"
                        elif transcript and seg_span_dbg > float(clip_max_len) + 16.0:
                            branch_dbg = "split"
                        elif transcript:
                            branch_dbg = "refine_single"
                        else:
                            branch_dbg = "normalize_only"
                        logger.info(
                            "reels.loop seg concept=%s video=%s src=%s span=%.1f branch=%s windows=%d",
                            str(concept.get("id") or ""),
                            video_id,
                            getattr(segment, "source", "legacy"),
                            seg_span_dbg,
                            branch_dbg,
                            len(clip_windows),
                        )
                    if not clip_windows:
                        if self.retrieval_debug_logging:
                            logger.info(
                                "reels.loop drop_segment concept=%s video=%s reason=empty_clip_windows",
                                str(concept.get("id") or ""),
                                video_id,
                            )
                        continue

                    hit_video_cap = False
                    current_segment_contexts: list[dict[str, Any]] = []
                    for clip_window in clip_windows:
                        if not clip_window:
                            if self.retrieval_debug_logging:
                                logger.info(
                                    "reels.loop drop concept=%s video=%s reason=window_none",
                                    str(concept.get("id") or ""),
                                    video_id,
                                )
                            continue
                        start_sec, end_sec = clip_window
                        clip_key = self._clip_key(video_id, start_sec, end_sec)
                        if clip_key in existing_clip_keys or clip_key in generated_clip_keys:
                            if self.retrieval_debug_logging:
                                logger.info(
                                    "reels.loop drop concept=%s video=%s win=(%.2f,%.2f) reason=clip_key_dup",
                                    str(concept.get("id") or ""),
                                    video_id,
                                    float(start_sec),
                                    float(end_sec),
                                )
                            continue

                        relevance_context = self._merge_relevance_context(
                            candidate.get("video_relevance") or {},
                            segment_relevance,
                        )
                        relevance_context["query_strategy"] = query_candidate.strategy
                        relevance_context["retrieval_stage"] = str(candidate.get("stage") or "")
                        relevance_context["discovery_score"] = float(ranking.get("discovery_score") or 0.0)
                        relevance_context["clipability_score"] = float(ranking.get("clipability_score") or 0.0)
                        relevance_context["source_surface"] = str(video.get("search_source") or "")
                        relevance_context["score"] = (
                            0.58 * float(relevance_context.get("score") or 0.0)
                            + 0.28 * float(ranking.get("discovery_score") or 0.0)
                            + 0.14 * float(ranking.get("clipability_score") or 0.0)
                        )
                        if not bool(relevance_context.get("passes", True)):
                            retrieval_metrics["low_relevance"] += 1
                            if self.retrieval_debug_logging:
                                logger.info(
                                    "reels.loop drop concept=%s video=%s win=(%.2f,%.2f) reason=low_relevance_pre_guard",
                                    str(concept.get("id") or ""),
                                    video_id,
                                    float(start_sec),
                                    float(end_sec),
                                )
                            continue
                        if not self._passes_selection_topic_guard(
                            video=video,
                            ranking=ranking,
                            segment_relevance=segment_relevance,
                            transcript_ranking=transcript_ranking,
                            has_transcript=bool(transcript),
                            fast_mode=fast_mode,
                            strict_topic_only=strict_topic_only,
                            subject_tag=subject_tag,
                            root_topic_terms=root_topic_terms,
                        ):
                            retrieval_metrics["low_transcript_purity"] += 1
                            if self.retrieval_debug_logging:
                                logger.info(
                                    "reels.loop drop concept=%s video=%s win=(%.2f,%.2f) reason=low_transcript_purity",
                                    str(concept.get("id") or ""),
                                    video_id,
                                    float(start_sec),
                                    float(end_sec),
                                )
                            continue

                        # Build clip_context from a per-window transcript slice so
                        # novelty / self-containment scoring reflects what actually
                        # plays in this window, not the full segment text. For
                        # multi-window splits this also prevents window 2+ from
                        # being rejected as near-duplicates of window 1 (same text).
                        window_text = self._slice_transcript_text_in_window(
                            transcript,
                            float(start_sec),
                            float(end_sec),
                            fallback_text=segment.text,
                        )
                        clip_context = self._build_clip_context(
                            text=window_text,
                            clip_duration_sec=float(max(0, end_sec - start_sec)),
                        )
                        # Exclude this segment's already-accepted windows from the
                        # novelty comparison — they intentionally cover consecutive
                        # portions of the same topic, so near-duplicate text is
                        # expected and should not block later windows.
                        current_context_ids = {id(x) for x in current_segment_contexts}
                        prior_contexts_for_novelty = [
                            c
                            for c in accepted_clip_contexts_by_video.get(video_id, [])
                            if id(c) not in current_context_ids
                        ]
                        quality_floor_passes, quality_floor_reason = self._passes_clip_quality_floor(
                            conn,
                            relevance_context=relevance_context,
                            clip_context=clip_context,
                            prior_contexts=prior_contexts_for_novelty,
                            subject_tag=subject_tag,
                            retrieval_profile=safe_retrieval_profile,
                            fast_mode=fast_mode,
                            page_hint=safe_page_hint,
                            transcript_quality=transcript_quality,
                        )
                        if not quality_floor_passes:
                            retrieval_metrics[quality_floor_reason] = int(retrieval_metrics.get(quality_floor_reason) or 0) + 1
                            if self.retrieval_debug_logging:
                                logger.info(
                                    "reels.loop drop concept=%s video=%s win=(%.2f,%.2f) reason=%s",
                                    str(concept.get("id") or ""),
                                    video_id,
                                    float(start_sec),
                                    float(end_sec),
                                    quality_floor_reason,
                                )
                            continue

                        accepted_count_for_video = existing_for_video + generated_video_counts.get(video_id, 0)
                        if accepted_count_for_video >= 9:
                            relevance_context["score"] = max(0.0, float(relevance_context.get("score") or 0.0) - 0.10)
                        elif accepted_count_for_video >= 6:
                            relevance_context["score"] = max(0.0, float(relevance_context.get("score") or 0.0) - 0.06)
                        elif accepted_count_for_video >= 3:
                            relevance_context["score"] = max(0.0, float(relevance_context.get("score") or 0.0) - 0.03)
                        if float(relevance_context.get("score") or 0.0) < self._quality_floor_min_relevance(page_hint=safe_page_hint):
                            retrieval_metrics["low_relevance"] += 1
                            if self.retrieval_debug_logging:
                                logger.info(
                                    "reels.loop drop concept=%s video=%s win=(%.2f,%.2f) reason=low_relevance_post_penalty",
                                    str(concept.get("id") or ""),
                                    video_id,
                                    float(start_sec),
                                    float(end_sec),
                                )
                            continue

                        if dry_run:
                            preview = self._build_dry_run_reel_preview(
                                concept=concept,
                                video=video,
                                segment=segment,
                                clip_window=clip_window,
                                relevance_context=relevance_context,
                            )
                            generated.append(preview)
                            if on_reel_created is not None:
                                try:
                                    on_reel_created(preview)
                                except Exception:
                                    logger.exception("on_reel_created callback failed for dry-run reel")
                        else:
                            try:
                                reel = self._create_reel(
                                    conn,
                                    material_id=material_id,
                                    concept=concept,
                                    video=video,
                                    segment=segment,
                                    clip_window=clip_window,
                                    transcript=transcript,
                                    relevance_context=relevance_context,
                                    fast_mode=fast_mode,
                                    target_clip_duration_sec=safe_target_clip_duration,
                                    generation_id=generation_id,
                                )
                            except Exception:
                                # Surface previously-silent exceptions so a broken
                                # persistence path doesn't look like an empty topic.
                                logger.exception(
                                    "reels.loop create_reel failed concept=%s video=%s win=(%.2f,%.2f)",
                                    str(concept.get("id") or ""),
                                    video_id,
                                    float(start_sec),
                                    float(end_sec),
                                )
                                continue
                            if not reel:
                                if self.retrieval_debug_logging:
                                    logger.info(
                                        "reels.loop drop concept=%s video=%s win=(%.2f,%.2f) reason=create_reel_none",
                                        str(concept.get("id") or ""),
                                        video_id,
                                        float(start_sec),
                                        float(end_sec),
                                    )
                                continue
                            generated.append(reel)
                            if on_reel_created is not None:
                                try:
                                    on_reel_created(reel)
                                except Exception:
                                    logger.exception(
                                        "on_reel_created callback failed for reel_id=%s",
                                        str(reel.get("reel_id") or ""),
                                    )

                        generated_clip_keys.add(clip_key)
                        generated_video_counts[video_id] = generated_video_counts.get(video_id, 0) + 1
                        accepted_clip_contexts_by_video.setdefault(video_id, []).append(clip_context)
                        current_segment_contexts.append(clip_context)
                        mining_request_key = str(candidate.get("mining_request_key") or "").strip()
                        mining_video_id = str(candidate.get("mining_video_id") or "").strip()
                        if mining_request_key and mining_video_id:
                            updated_video_count = existing_for_video + generated_video_counts.get(video_id, 0)
                            mining_cap = max(1, int(candidate.get("mining_segment_cap") or video_segment_cap))
                            mining_exhausted = updated_video_count >= mining_cap
                            next_mining_state = (
                                self.MINING_STATE_EXHAUSTED
                                if mining_exhausted
                                else self.MINING_STATE_HIGH_YIELD
                                if updated_video_count >= 3
                                else self.MINING_STATE_PARTIALLY_MINED
                            )
                            self._upsert_request_video_mining_state(
                                conn,
                                material_id=material_id,
                                request_key=mining_request_key,
                                video_id=mining_video_id,
                                mining_state=next_mining_state,
                                accepted_clip_delta=1,
                                exhausted=mining_exhausted,
                            )
                        family_key = str(query_candidate.family_key or "").strip()
                        if family_key:
                            accepted_reels_by_family[family_key] += 1
                        selected_outcome = {
                            "video_id": video_id,
                            "reasons": [
                                f"query_strategy:{query_candidate.strategy}",
                                f"stage:{candidate.get('stage')}",
                                f"discovery:{round(float(ranking.get('discovery_score') or 0.0), 3)}",
                                f"clipability:{round(float(ranking.get('clipability_score') or 0.0), 3)}",
                                f"function:{clip_context.get('function_label')}",
                            ],
                            "clip_window": {"t_start": start_sec, "t_end": end_sec},
                        }
                        retrieval_metrics["final_unique_reels"] = int(
                            retrieval_metrics.get("final_unique_reels") or 0
                        ) + 1

                        if self._should_finalize_generation(
                            generated=generated,
                            num_reels=num_reels,
                            preferred_video_duration=safe_video_duration_pref,
                            max_generation_target=max_generation_target,
                            fast_mode=fast_mode,
                        ):
                            self._persist_retrieval_debug_run(
                                conn,
                                run=retrieval_run,
                                query_reports=all_query_reports,
                                candidate_rows=candidate_records,
                                selected=selected_outcome,
                                failure_reason="",
                                dry_run=dry_run,
                                metrics=retrieval_metrics,
                            )
                            if request_key:
                                self._persist_request_frontier_reports(
                                    conn,
                                    material_id=material_id,
                                    request_key=request_key,
                                    query_reports=all_query_reports,
                                    accepted_reels_by_family=accepted_reels_by_family,
                                )
                            self._shutdown_transcript_prefetch_task(
                                transcript_prefetch_task,
                                wait=False,
                                cancel_futures=True,
                            )
                            transcript_prefetch_task = None
                            return self._finalize_generated_reels(
                                generated=generated,
                                num_reels=num_reels,
                                preferred_video_duration=safe_video_duration_pref,
                            )
                        if existing_for_video + generated_video_counts.get(video_id, 0) >= video_segment_cap:
                            hit_video_cap = True
                            break

                    if hit_video_cap:
                        break

            self._persist_retrieval_debug_run(
                conn,
                run=retrieval_run,
                query_reports=all_query_reports,
                candidate_rows=candidate_records,
                selected=selected_outcome,
                failure_reason=self._exhaustion_reason_from_metrics(
                    metrics=retrieval_metrics,
                    selected=selected_outcome,
                ),
                dry_run=dry_run,
                metrics=retrieval_metrics,
            )
            if request_key:
                self._persist_request_frontier_reports(
                    conn,
                    material_id=material_id,
                    request_key=request_key,
                    query_reports=all_query_reports,
                    accepted_reels_by_family=accepted_reels_by_family,
                )
            self._shutdown_transcript_prefetch_task(
                transcript_prefetch_task,
                wait=False,
                cancel_futures=True,
            )
            material_seen_video_ids.update(seen_video_ids)
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

    def _bootstrap_transcript_budget(
        self,
        *,
        generated_count: int,
        max_generation_target: int,
    ) -> int:
        remaining = max(1, max_generation_target - max(0, generated_count))
        cap = (
            self.BOOTSTRAP_TRANSCRIPT_CANDIDATES_SERVERLESS
            if self.serverless_mode
            else self.BOOTSTRAP_TRANSCRIPT_CANDIDATES
        )
        return max(1, min(cap, remaining))

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

    def _bootstrap_recovery_query(
        self,
        *,
        title: str,
        summary: str,
        keywords: list[str],
        subject_tag: str | None,
    ) -> str:
        recovery_query = self._plan_recovery_queries(
            concept_title=title,
            keywords=keywords,
            summary=summary,
            subject_tag=subject_tag,
            context_terms=[],
            retrieval_profile="bootstrap",
        )
        if recovery_query:
            return recovery_query[0].text
        fallback = " ".join(part for part in [subject_tag or "", title, "explained"] if part).strip()
        return fallback

    def _select_bootstrap_external_fallback_query(
        self,
        retrieval_stages: list[RetrievalStagePlan],
    ) -> QueryCandidate | None:
        for stage_name in ("recovery", "high_precision"):
            stage = next((item for item in retrieval_stages if item.name == stage_name), None)
            if stage and stage.queries:
                return stage.queries[0]
        return None

    def _search_passes(self, video_pool_mode: str, preferred_video_duration: str) -> list[dict[str, Any]]:
        if preferred_video_duration in {"short", "medium", "long"}:
            return [
                {
                    "duration_plan": (preferred_video_duration,),
                    "strict_duration": True,
                    "allow_unknown_duration": False,
                },
                {
                    "duration_plan": self._duration_plan(video_pool_mode, preferred_video_duration),
                    "strict_duration": False,
                    "allow_unknown_duration": True,
                },
            ]
        return [
            {
                "duration_plan": self._duration_plan(video_pool_mode, preferred_video_duration),
                "strict_duration": False,
                "allow_unknown_duration": True,
            }
        ]

    def _video_matches_preferred_duration(
        self,
        video_duration_sec: int,
        preferred_video_duration: str,
        allow_unknown_duration: bool,
    ) -> bool:
        if preferred_video_duration not in {"short", "medium", "long"}:
            return True
        if video_duration_sec <= 0:
            return allow_unknown_duration
        if preferred_video_duration == "short":
            return video_duration_sec < 4 * 60
        if preferred_video_duration == "medium":
            return 4 * 60 <= video_duration_sec <= 20 * 60
        return video_duration_sec > 20 * 60

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

    def _topic_breadth_class(self, subject_tag: str | None) -> str:
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
            return "curated_broad"
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
        if opaque_topic:
            return "opaque_niche"
        return "default"

    def _topic_novelty_profile(
        self,
        *,
        subject_tag: str | None,
        retrieval_profile: RetrievalProfile,
        fast_mode: bool,
    ) -> dict[str, float]:
        breadth_class = self._topic_breadth_class(subject_tag)
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

    def _has_short_and_long_mix(self, generated: list[dict[str, Any]]) -> bool:
        has_short = False
        has_long = False
        for reel in generated:
            bucket = self._video_duration_bucket(reel.get("video_duration_sec"))
            if bucket == "short":
                has_short = True
            elif bucket == "long":
                has_long = True
            if has_short and has_long:
                return True
        return False

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

    def _search_results_budget(
        self,
        *,
        fast_mode: bool,
        generated_count: int,
        max_generation_target: int,
        retrieval_profile: RetrievalProfile,
        recovery_stage: int = 0,
    ) -> int:
        remaining = max(1, max_generation_target - max(0, generated_count))
        if retrieval_profile == "bootstrap":
            if self.serverless_mode:
                return max(4, min(6, 3 + remaining))
            if fast_mode:
                return max(5, min(7, 4 + remaining))
            return max(8, min(14, 6 + remaining * 2))
        if int(recovery_stage or 0) <= self.REFILL_STAGE_ROOT_COMPANION:
            return max(6, min(18 if not fast_mode else 10, 4 + remaining * (2 if not fast_mode else 1)))
        if self.serverless_mode:
            return max(6, min(10, 4 + remaining))
        if fast_mode:
            return max(6, min(12, 5 + remaining))
        return max(18, min(42, 14 + remaining * 4))

    def _transcript_expansion_budget(
        self,
        *,
        fast_mode: bool,
        generated_count: int,
        max_generation_target: int,
        recovery_stage: int = 0,
    ) -> int:
        remaining = max(1, max_generation_target - max(0, generated_count))
        if self.serverless_mode:
            return max(1, min(3, remaining))
        if fast_mode:
            base_budget = max(2, min(4, remaining))
            if int(recovery_stage or 0) >= self.REFILL_STAGE_MULTI_CLIP_STRICT:
                return max(base_budget, min(8, remaining * 2))
            return base_budget
        base_budget = max(6, min(16, remaining * 3))
        if int(recovery_stage or 0) >= self.REFILL_STAGE_MULTI_CLIP_STRICT:
            return max(base_budget, min(24, remaining * 4))
        return base_budget

    def _should_finalize_generation(
        self,
        generated: list[dict[str, Any]],
        num_reels: int,
        preferred_video_duration: str,
        max_generation_target: int,
        fast_mode: bool,
    ) -> bool:
        if len(generated) < num_reels:
            return False
        if preferred_video_duration != "any" or num_reels <= 1:
            return True
        if self._has_short_and_long_mix(generated):
            return True
        if self._strong_generation_pool_ready(generated, fast_mode=fast_mode, target_count=num_reels):
            return True
        return len(generated) >= max_generation_target

    def _strong_generation_pool_ready(
        self,
        generated: list[dict[str, Any]],
        *,
        fast_mode: bool,
        target_count: int,
    ) -> bool:
        if not generated:
            return False
        ranked = sorted(generated, key=self._generation_result_score, reverse=True)[: max(1, target_count)]
        if len(ranked) < target_count:
            return False
        min_relevance = 0.32 if fast_mode else 0.28
        min_discovery = 0.24 if fast_mode else 0.2
        strong = [
            reel
            for reel in ranked
            if float(reel.get("relevance_score") or 0.0) >= min_relevance
            and float(reel.get("discovery_score") or 0.0) >= min_discovery
        ]
        if len(strong) < target_count:
            return False
        distinct_concepts = {
            str(reel.get("concept_id") or "").strip()
            for reel in ranked
            if str(reel.get("concept_id") or "").strip()
        }
        return len(distinct_concepts) >= min(2, target_count)

    def _finalize_generated_reels(
        self,
        generated: list[dict[str, Any]],
        num_reels: int,
        preferred_video_duration: str,
    ) -> list[dict[str, Any]]:
        if not generated or num_reels <= 0:
            return []
        if preferred_video_duration != "any" or num_reels <= 1:
            return generated[:num_reels]

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
        selected.sort(key=self._generation_result_score, reverse=True)
        return selected

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

    def _get_concept_embedding(self, conn, concept: dict[str, Any]) -> np.ndarray:
        parsed_embedding = self._parse_embedding_vector(concept.get("embedding_json"))
        if parsed_embedding is not None:
            return parsed_embedding

        parsed_keywords = self._parse_keywords_json(concept.get("keywords_json"))
        concept_text = (
            f"{concept['title']}. "
            f"Keywords: {' '.join(parsed_keywords)}. "
            f"Summary: {concept['summary']}"
        )
        embedding = self.embedding_service.embed_texts(conn, [concept_text])[0]
        existing = fetch_one(
            conn,
            "SELECT material_id, created_at FROM concepts WHERE id = ?",
            (concept["id"],),
        )
        if not existing:
            raise ValueError(f"Concept not found: {concept['id']}")
        if self.embedding_service.should_persist_replacement(concept.get("embedding_json")):
            upsert(
                conn,
                "concepts",
                {
                    "id": concept["id"],
                    "material_id": existing["material_id"],
                    "title": concept["title"],
                    "keywords_json": concept["keywords_json"],
                    "summary": concept["summary"],
                    "embedding_json": dumps_json(embedding.tolist()),
                    "created_at": existing["created_at"],
                },
            )
        return embedding

    def _init_retrieval_debug_run(self, material_id: str, concept_id: str, concept_title: str) -> dict[str, Any]:
        return {
            "run_id": str(uuid.uuid4()),
            "material_id": material_id,
            "concept_id": concept_id,
            "concept_title": concept_title,
        }

    def _exhaustion_reason_from_metrics(
        self,
        *,
        metrics: dict[str, Any],
        selected: dict[str, Any] | None = None,
    ) -> str:
        if selected:
            return ""
        raw_discovered = int(metrics.get("raw_discovered_videos") or 0)
        filtered_relevant = int(metrics.get("filtered_relevant_videos") or 0)
        clip_candidates = int(metrics.get("clip_candidate_videos") or 0)
        dropped_segment_cap = int(metrics.get("dropped_segment_cap") or 0)
        quality_floor_drops = sum(
            int(metrics.get(key) or 0)
            for key in ("low_relevance", "low_transcript_purity", "low_novelty", "low_self_containment")
        )

        if raw_discovered <= 0:
            return "search_recall_exhausted"
        if filtered_relevant <= 0 or quality_floor_drops > 0:
            return "quality_blocked"
        if dropped_segment_cap > 0 and clip_candidates <= 0:
            return "quota_or_cap_exhausted"
        if clip_candidates <= 0:
            return "clip_yield_exhausted"
        if dropped_segment_cap > 0:
            return "quota_or_cap_exhausted"
        return "clip_yield_exhausted"

    def _persist_retrieval_debug_run(
        self,
        conn,
        run: dict[str, Any],
        query_reports: list[dict[str, Any]],
        candidate_rows: list[dict[str, Any]],
        selected: dict[str, Any] | None,
        failure_reason: str,
        dry_run: bool,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        if not self.retrieval_debug_logging:
            return
        run_id = str(run.get("run_id") or "").strip()
        if not run_id:
            return

        upsert(
            conn,
            "retrieval_runs",
            {
                "id": run_id,
                "material_id": str(run.get("material_id") or ""),
                "concept_id": str(run.get("concept_id") or ""),
                "concept_title": str(run.get("concept_title") or ""),
                "selected_video_id": str((selected or {}).get("video_id") or ""),
                "failure_reason": failure_reason.strip(),
                "debug_json": dumps_json(
                    {
                        "selected": selected,
                        "query_count": len(query_reports),
                        "candidate_count": len(candidate_rows),
                        "dry_run": bool(dry_run),
                        "metrics": metrics or {},
                    }
                ),
                "created_at": now_iso(),
            },
        )
        selected_video_id = str((selected or {}).get("video_id") or "").strip()
        log_payload = {
            "run_id": run_id,
            "material_id": str(run.get("material_id") or ""),
            "concept_id": str(run.get("concept_id") or ""),
            "concept_title": str(run.get("concept_title") or ""),
            "selected_video_id": selected_video_id or None,
            "failure_reason": failure_reason.strip() or None,
            "query_count": len(query_reports),
            "candidate_count": len(candidate_rows),
            "dry_run": bool(dry_run),
            "metrics": metrics or {},
        }
        if failure_reason.strip():
            logger.warning("Retrieval run summary: %s", json.dumps(log_payload, sort_keys=True))
        else:
            logger.info("Retrieval run summary: %s", json.dumps(log_payload, sort_keys=True))

        for idx, row in enumerate(query_reports[:240]):
            upsert(
                conn,
                "retrieval_queries",
                {
                    "id": str(uuid.uuid4()),
                    "run_id": run_id,
                    "query_text": str(row.get("query") or ""),
                    "strategy": str(row.get("strategy") or ""),
                    "stage": str(row.get("stage") or ""),
                    "source_surface": str(row.get("surface") or ""),
                    "source_terms_json": dumps_json(row.get("source_terms") or []),
                    "weight": float(row.get("weight") or 0.0),
                    "result_count": int(row.get("results") or 0),
                    "kept_count": int(row.get("kept") or 0),
                    "position": idx,
                    "created_at": now_iso(),
                },
            )

        for idx, row in enumerate(candidate_rows[:320]):
            upsert(
                conn,
                "retrieval_candidates",
                {
                    "id": str(uuid.uuid4()),
                    "run_id": run_id,
                    "video_id": str(row.get("video_id") or ""),
                    "video_title": str(row.get("video_title") or ""),
                    "channel_title": str(row.get("channel_title") or ""),
                    "strategy": str(row.get("strategy") or ""),
                    "stage": str(row.get("stage") or ""),
                    "query_text": str(row.get("query") or ""),
                    "source_surface": str(row.get("source_surface") or ""),
                    "discovery_score": float(row.get("discovery_score") or 0.0),
                    "clipability_score": float(row.get("clipability_score") or 0.0),
                    "final_score": float(row.get("final_score") or 0.0),
                    "feature_json": dumps_json(
                        {
                            **dict(row.get("features") or {}),
                            "discovery_path": str(row.get("discovery_path") or ""),
                            "crawl_depth": int(row.get("crawl_depth") or 0),
                            "seed_video_id": str(row.get("seed_video_id") or ""),
                            "seed_channel_id": str(row.get("seed_channel_id") or ""),
                        }
                    ),
                    "position": idx,
                    "created_at": now_iso(),
                },
            )

        if selected:
            selected_video_id = str(selected.get("video_id") or "")
            can_write_outcome = not dry_run
            if not can_write_outcome and selected_video_id:
                can_write_outcome = bool(fetch_one(conn, "SELECT id FROM videos WHERE id = ?", (selected_video_id,)))
            if not can_write_outcome:
                return
            clip_window = selected.get("clip_window") or {}
            upsert(
                conn,
                "retrieval_outcomes",
                {
                    "id": str(uuid.uuid4()),
                    "run_id": run_id,
                    "video_id": selected_video_id,
                    "t_start": float(clip_window.get("t_start") or 0.0),
                    "t_end": float(clip_window.get("t_end") or 0.0),
                    "reason_json": dumps_json(selected.get("reasons") or []),
                    "created_at": now_iso(),
                },
            )

    def _derive_visual_scene_spec(self, title: str, keywords: list[str], summary: str) -> dict[str, list[str]]:
        raw = " ".join([title, summary, *keywords]).strip().lower()
        noun_like = [t for t in normalize_terms([raw]) if len(t) >= 4][:18]
        actions: list[str] = []
        for token in noun_like:
            if token.endswith("ing") or token in {"show", "build", "explain", "demo", "review"}:
                actions.append(token)
        if not actions:
            actions = ["showing", "demonstration"]

        return {
            "environment": noun_like[:5],
            "subjects": noun_like[5:10],
            "objects": noun_like[10:16],
            "actions": actions[:5],
            "camera": ["close up", "b-roll", "cinematic"],
            "mood": ["clean", "focused"],
        }

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

    def _persist_request_frontier_reports(
        self,
        conn,
        *,
        material_id: str,
        request_key: str,
        query_reports: list[dict[str, Any]],
        accepted_reels_by_family: dict[str, int] | None = None,
        visible_reels_by_family: dict[str, int] | None = None,
    ) -> None:
        if not material_id or not request_key:
            return
        accepted_counts = accepted_reels_by_family or {}
        visible_counts = visible_reels_by_family or {}
        aggregated: dict[str, dict[str, Any]] = {}
        for report in query_reports:
            family_key = str(report.get("family_key") or "").strip()
            if not family_key:
                continue
            stats = aggregated.setdefault(
                family_key,
                {
                    "stage": str(report.get("stage") or ""),
                    "query": str(report.get("query") or ""),
                    "source_family": str(report.get("source_family") or ""),
                    "anchor_mode": str(report.get("anchor_mode") or ""),
                    "seed_video_id": str(report.get("seed_video_id") or ""),
                    "seed_channel_id": str(report.get("seed_channel_id") or ""),
                    "runs": 0,
                    "results": 0,
                    "kept": 0,
                    "dropped_duplicate": 0,
                    "dropped_off_topic": 0,
                },
            )
            stats["runs"] += 1
            stats["results"] += int(report.get("results") or 0)
            stats["kept"] += int(report.get("kept") or 0)
            stats["dropped_duplicate"] += int(report.get("dropped_duplicate") or 0)
            stats["dropped_off_topic"] += int(report.get("dropped_metadata_gate") or 0) + int(report.get("dropped_ranking") or 0)
        for family_key, stats in aggregated.items():
            total_results = max(0, int(stats["results"]))
            duplicate_rate = float(stats["dropped_duplicate"]) / max(1, total_results)
            off_topic_rate = float(stats["dropped_off_topic"]) / max(1, total_results)
            self._upsert_request_frontier_entry(
                conn,
                material_id=material_id,
                request_key=request_key,
                family_key=family_key,
                stage=str(stats["stage"] or ""),
                query_text=str(stats["query"] or ""),
                source_family=str(stats["source_family"] or ""),
                anchor_mode=str(stats["anchor_mode"] or ""),
                seed_video_id=str(stats["seed_video_id"] or ""),
                seed_channel_id=str(stats["seed_channel_id"] or ""),
                stat_deltas={
                    "runs": int(stats["runs"]),
                    "new_good_videos": int(stats["kept"]),
                    "new_accepted_reels": int(accepted_counts.get(family_key) or 0),
                    "new_visible_reels": int(visible_counts.get(family_key) or 0),
                    "duplicate_rate": duplicate_rate,
                    "off_topic_rate": off_topic_rate,
                },
            )

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
            if self._topic_breadth_class(subject_tag) == "curated_broad":
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
        settings = get_settings()
        if not (settings.openai_enabled and settings.openai_api_key):
            return None
        try:
            client = build_openai_client(api_key=settings.openai_api_key, timeout=8.0, enabled=True)
            if client is None:
                return None
            valid_types = ["tutorial", "explained", "lecture", "worked example", "animation", "documentary", "demo"]
            prompt = (
                f"Given the study concept '{title}' (keywords: {', '.join(keywords[:5])}, "
                f"subject: {subject_tag or 'general'}, mode: {video_pool_mode}), "
                f"what type of YouTube video would best teach this? "
                f"Choose exactly one: {', '.join(valid_types)}. "
                f"Reply with just the type name."
            )
            response = client.chat.completions.create(
                model=settings.openai_chat_model or "gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=20,
            )
            answer = (response.choices[0].message.content or "").strip().lower()
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
        settings = get_settings()
        if not (settings.openai_enabled and settings.openai_api_key):
            return None
        try:
            client = build_openai_client(api_key=settings.openai_api_key, timeout=8.0, enabled=True)
            if client is None:
                return None
            cleaned_terms = [t.strip() for t in terms if t.strip()][:5]
            if not cleaned_terms:
                return None
            prompt = (
                f"For these study terms: {', '.join(cleaned_terms)}\n"
                f"Generate 4-6 alternative search phrases that would find educational YouTube videos "
                f"teaching the same concepts. Return one phrase per line, no numbering."
            )
            response = client.chat.completions.create(
                model=settings.openai_chat_model or "gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=200,
            )
            content = (response.choices[0].message.content or "").strip()
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
        if self.openai_client is None:
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
            response = self.openai_client.chat.completions.create(
                model=self.chat_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=250,
            )
            content = (response.choices[0].message.content or "").strip()
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

    def _plan_source_graph_queries(
        self,
        conn,
        *,
        material_id: str,
        request_key: str,
        current_generation_id: str | None,
        concept_title: str,
        subject_tag: str | None,
        limit: int = 2,
    ) -> list[PlannedQuery]:
        clean_request_key = str(request_key or "").strip()
        if not clean_request_key:
            return []
        rows = fetch_all(
            conn,
            """
            SELECT
                v.channel_title,
                v.title
            FROM reels r
            JOIN reel_generations g ON g.id = r.generation_id
            JOIN videos v ON v.id = r.video_id
            WHERE g.material_id = ?
              AND g.request_key = ?
              AND (? = '' OR g.id <> ?)
            ORDER BY r.created_at DESC
            LIMIT 24
            """,
            (material_id, clean_request_key, str(current_generation_id or ""), str(current_generation_id or "")),
        )
        if not rows:
            return []
        queries: list[PlannedQuery] = []
        seen_channels: set[str] = set()
        base_topic = self._clean_query_text(subject_tag or concept_title)
        for row in rows:
            channel_title = self._clean_query_text(str(row.get("channel_title") or ""))
            if not channel_title:
                continue
            normalized_channel = self._normalize_query_key(channel_title)
            if not normalized_channel or normalized_channel in seen_channels:
                continue
            seen_channels.add(normalized_channel)
            query_text = self._clean_query_text(" ".join(part for part in [channel_title, base_topic] if part))
            if not query_text:
                continue
            queries.append(
                self._build_planned_query(
                    text=query_text,
                    strategy="channel_graph",
                    stage="recovery",
                    confidence=0.68,
                    source_terms=[channel_title, base_topic],
                    concept_title=concept_title,
                    weight=0.72,
                    source_surface="youtube_channel",
                    rationale="recovery expands from channels that already produced accepted request-chain clips",
                    family=self.FRONTIER_RECOVERY_GRAPH,
                    source_family=self._source_family_from_strategy("channel_graph", query_text),
                    anchor_mode="recovery_graph",
                    seed_channel_id=channel_title,
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
    ) -> QueryPlanningResult:
        # preferred_video_duration is now used below for pool-mode query hints.
        selected_concepts, selection_decisions, exhausted = self._select_concepts(
            concepts,
            retrieval_profile=retrieval_profile,
            request_need=max(1, request_need),
            fast_mode=fast_mode,
            targeted_concept_id=targeted_concept_id,
            subject_tag=subject_tag,
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
            breadth_class = self._topic_breadth_class(subject_tag)
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

    def _fast_pass_is_sufficient(
        self,
        stage_candidates: list[dict[str, Any]],
        *,
        fast_mode: bool,
        max_generation_target: int,
        retrieval_profile: RetrievalProfile,
    ) -> bool:
        if retrieval_profile == "bootstrap":
            if len(stage_candidates) < max(4, min(6, max_generation_target + 2)):
                return False
            return not self._bootstrap_pool_is_weak(
                stage_candidates,
                max_generation_target=max_generation_target,
            )
        if not stage_candidates:
            return False
        if fast_mode:
            target = max(6, min(14, max_generation_target + 2))
        else:
            target = max(8, min(40, max_generation_target * 3))
        if len(stage_candidates) < target:
            return False
        strong = sum(
            1
            for candidate in stage_candidates
            if float((candidate.get("ranking") or {}).get("discovery_score") or 0.0) >= 0.2
        )
        return strong >= (max(2, target // 4) if fast_mode else max(4, target // 3))

    def _graph_profile_for_stage(
        self,
        *,
        fast_mode: bool,
        retrieval_profile: RetrievalProfile,
        page_hint: int,
        recovery_stage: int,
        subject_tag: str | None = None,
    ) -> Literal["off", "light", "deep"]:
        if retrieval_profile == "bootstrap":
            if subject_tag and self._topic_breadth_class(subject_tag) == "opaque_niche":
                return "light"
            return "off"
        if not self._recovery_stage_allows_related(page_hint=page_hint, recovery_stage=recovery_stage):
            return "off"
        if int(recovery_stage or 0) < self.REFILL_STAGE_RECOVERY_GRAPH:
            return "light"
        if self.serverless_mode:
            return "light"
        return "light" if fast_mode else "deep"

    def _stage_search_jobs_parallel(
        self,
        *,
        stage_name: str,
        stage_queries: list[QueryCandidate],
        stage_duration_plan: tuple[str | None, ...],
        max_results_for_query: int,
        creative_commons_only: bool,
        fast_mode: bool,
        retrieval_profile: RetrievalProfile,
        page_hint: int,
        recovery_stage: int,
        allow_external_fallbacks: bool = True,
        variant_limit: int | None = None,
        subject_tag: str | None = None,
    ) -> list[tuple[int, int, QueryCandidate, str | None, list[dict[str, Any]]]]:
        jobs: list[tuple[int, int, QueryCandidate, str | None]] = []
        has_any_duration = None in stage_duration_plan
        for query_idx, query_candidate in enumerate(stage_queries):
            for duration_idx, duration in enumerate(stage_duration_plan):
                if has_any_duration and duration is not None:
                    continue
                jobs.append((query_idx, duration_idx, query_candidate, duration))
        if not jobs:
            return []

        workers_limit = self.QUERY_RETRIEVAL_WORKERS_FAST if fast_mode else self.QUERY_RETRIEVAL_WORKERS_SLOW
        if self.serverless_mode:
            workers_limit = min(workers_limit, 2)
        workers = max(1, min(workers_limit, len(jobs)))
        if workers == 1:
            output: list[tuple[int, int, QueryCandidate, str | None, list[dict[str, Any]]]] = []
            for query_idx, duration_idx, query_candidate, duration in jobs:
                videos = self.youtube_service.search_videos(
                    None,
                    query=query_candidate.text,
                    max_results=max_results_for_query,
                    creative_commons_only=creative_commons_only,
                    video_duration=duration,
                    retrieval_strategy=query_candidate.strategy,
                    retrieval_stage=stage_name,
                    source_surface=query_candidate.source_surface,
                    retrieval_profile=retrieval_profile,
                    allow_external_fallbacks=allow_external_fallbacks,
                    variant_limit=variant_limit,
                    graph_profile=self._graph_profile_for_stage(
                        fast_mode=fast_mode,
                        retrieval_profile=retrieval_profile,
                        page_hint=page_hint,
                        recovery_stage=recovery_stage,
                        subject_tag=subject_tag,
                    ),
                    root_terms=list(query_candidate.source_terms),
                )
                output.append((query_idx, duration_idx, query_candidate, duration, videos))
            return output

        output: list[tuple[int, int, QueryCandidate, str | None, list[dict[str, Any]]]] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(
                    self.youtube_service.search_videos,
                    None,
                    query_candidate.text,
                    max_results_for_query,
                    creative_commons_only,
                    duration,
                    query_candidate.strategy,
                    stage_name,
                    query_candidate.source_surface,
                    retrieval_profile,
                    allow_external_fallbacks,
                    variant_limit,
                    self._graph_profile_for_stage(
                        fast_mode=fast_mode,
                        retrieval_profile=retrieval_profile,
                        page_hint=page_hint,
                        recovery_stage=recovery_stage,
                        subject_tag=subject_tag,
                    ),
                    list(query_candidate.source_terms),
                ): (query_idx, duration_idx, query_candidate, duration)
                for query_idx, duration_idx, query_candidate, duration in jobs
            }
            for future in as_completed(future_map):
                query_idx, duration_idx, query_candidate, duration = future_map[future]
                try:
                    videos = future.result()
                except Exception:
                    videos = []
                output.append((query_idx, duration_idx, query_candidate, duration, videos))

        output.sort(key=lambda row: (row[0], row[1]))
        return output

    def _prefetch_transcripts_parallel(
        self,
        video_ids: list[str],
        *,
        fast_mode: bool,
    ) -> dict[str, list[dict[str, Any]]]:
        unique_ids = [str(video_id).strip() for video_id in dict.fromkeys(video_ids) if str(video_id).strip()]
        if not unique_ids:
            return {}

        workers_limit = self.TRANSCRIPT_FETCH_WORKERS_FAST if fast_mode else self.TRANSCRIPT_FETCH_WORKERS_SLOW
        if self.serverless_mode:
            workers_limit = min(workers_limit, 2)
        workers = max(1, min(workers_limit, len(unique_ids)))
        if workers == 1:
            return {video_id: self.youtube_service.get_transcript(None, video_id) for video_id in unique_ids}

        transcripts: dict[str, list[dict[str, Any]]] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(self.youtube_service.get_transcript, None, video_id): video_id for video_id in unique_ids
            }
            for future in as_completed(future_map):
                video_id = future_map[future]
                try:
                    transcript = future.result()
                except Exception:
                    transcript = []
                transcripts[video_id] = transcript
        return transcripts

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
            video_id: executor.submit(self.youtube_service.get_transcript, None, video_id)
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
                self.youtube_service.get_transcript,
                None,
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
                transcript = list(self.youtube_service.get_transcript(None, clean_video_id) or [])
        except FutureTimeoutError:
            logger.warning(
                "Transcript prefetch timed out after %.1fs for video=%s",
                float(timeout or 0.0),
                clean_video_id,
            )
            try:
                transcript = list(self.youtube_service.get_transcript(None, clean_video_id) or [])
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

    def _collect_transcript_prefetch(
        self,
        *,
        prefetch_task: TranscriptPrefetchTask | None,
        requested_video_ids: list[str],
        fast_mode: bool,
    ) -> dict[str, list[dict[str, Any]]]:
        transcripts: dict[str, list[dict[str, Any]]] = {}
        if prefetch_task is not None:
            transcripts.update(dict(prefetch_task.cached_transcripts))
            for video_id in requested_video_ids:
                clean_video_id = str(video_id).strip()
                if not clean_video_id or clean_video_id in transcripts:
                    continue
                future = prefetch_task.future_by_video_id.get(clean_video_id)
                if future is None:
                    continue
                if future.done():
                    transcripts.update(
                        self._resolve_transcript_prefetch(
                            prefetch_task=prefetch_task,
                            video_id=clean_video_id,
                            timeout=0.0,
                            fast_mode=fast_mode,
                        )
                    )
        prefetched_ids = set(transcripts)
        missing_ids = [
            str(video_id).strip()
            for video_id in requested_video_ids
            if (
                str(video_id).strip()
                and str(video_id).strip() not in prefetched_ids
                and (
                    prefetch_task is None
                    or str(video_id).strip() not in prefetch_task.future_by_video_id
                )
            )
        ]
        if missing_ids:
            try:
                transcripts.update(self._prefetch_transcripts_parallel(missing_ids, fast_mode=fast_mode))
            except Exception:
                pass
        return transcripts

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
                transcript = list(self.youtube_service.get_transcript(None, clean_video_id) or [])
            except Exception:
                transcript = []
        transcript_cache[clean_video_id] = transcript
        return transcript

    def _request_video_mining_state_id(self, *, material_id: str, request_key: str, video_id: str) -> str:
        payload = f"{material_id}|{request_key}|{video_id}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _fetch_request_video_mining_state(
        self,
        conn,
        *,
        material_id: str,
        request_key: str,
        video_id: str,
    ) -> dict[str, Any] | None:
        if not material_id or not request_key or not video_id:
            return None
        return fetch_one(
            conn,
            """
            SELECT *
            FROM request_video_mining_state
            WHERE material_id = ?
              AND request_key = ?
              AND video_id = ?
            """,
            (material_id, request_key, video_id),
        )

    def _upsert_request_video_mining_state(
        self,
        conn,
        *,
        material_id: str,
        request_key: str,
        video_id: str,
        mining_state: str | None = None,
        quality_tier: str | None = None,
        transcript_fetched: bool | None = None,
        windows_scanned_delta: int = 0,
        clusters_mined_delta: int = 0,
        accepted_clip_delta: int = 0,
        visible_clip_delta: int = 0,
        remaining_spans: list[dict[str, Any]] | None = None,
        exhausted: bool | None = None,
    ) -> dict[str, Any]:
        existing = self._fetch_request_video_mining_state(
            conn,
            material_id=material_id,
            request_key=request_key,
            video_id=video_id,
        )
        now = now_iso()
        row = {
            "id": str((existing or {}).get("id") or self._request_video_mining_state_id(
                material_id=material_id,
                request_key=request_key,
                video_id=video_id,
            )),
            "material_id": material_id,
            "request_key": request_key,
            "video_id": video_id,
            "mining_state": str((existing or {}).get("mining_state") or self.MINING_STATE_UNMINED),
            "quality_tier": str((existing or {}).get("quality_tier") or ""),
            "transcript_fetched": 1 if bool((existing or {}).get("transcript_fetched")) else 0,
            "windows_scanned": int((existing or {}).get("windows_scanned") or 0),
            "clusters_mined": int((existing or {}).get("clusters_mined") or 0),
            "accepted_clip_count": int((existing or {}).get("accepted_clip_count") or 0),
            "visible_clip_count": int((existing or {}).get("visible_clip_count") or 0),
            "remaining_spans_json": str((existing or {}).get("remaining_spans_json") or "[]"),
            "last_mined_at": str((existing or {}).get("last_mined_at") or "") or None,
            "exhausted": 1 if bool((existing or {}).get("exhausted")) else 0,
            "created_at": str((existing or {}).get("created_at") or now),
            "updated_at": now,
        }
        if mining_state is not None:
            row["mining_state"] = str(mining_state or self.MINING_STATE_UNMINED)
        if quality_tier is not None:
            row["quality_tier"] = str(quality_tier or "")
        if transcript_fetched is not None:
            row["transcript_fetched"] = 1 if transcript_fetched else 0
        row["windows_scanned"] = int(row["windows_scanned"]) + max(0, int(windows_scanned_delta or 0))
        row["clusters_mined"] = int(row["clusters_mined"]) + max(0, int(clusters_mined_delta or 0))
        row["accepted_clip_count"] = int(row["accepted_clip_count"]) + max(0, int(accepted_clip_delta or 0))
        row["visible_clip_count"] = int(row["visible_clip_count"]) + max(0, int(visible_clip_delta or 0))
        if remaining_spans is not None:
            row["remaining_spans_json"] = dumps_json(remaining_spans[:24])
        if exhausted is not None:
            row["exhausted"] = 1 if exhausted else 0
        if row["mining_state"] == self.MINING_STATE_EXHAUSTED:
            row["exhausted"] = 1
        row["last_mined_at"] = now
        upsert(conn, "request_video_mining_state", row)
        return row

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

    def _mine_existing_request_chain_videos(
        self,
        conn,
        *,
        material_id: str,
        request_key: str,
        generation_id: str | None,
        concept: dict[str, Any],
        concept_terms: list[str],
        context_terms: list[str],
        concept_embedding: np.ndarray | None,
        subject_tag: str | None,
        root_topic_terms: list[str],
        visual_spec: dict[str, list[str]],
        preferred_video_duration: str,
        fast_mode: bool,
        strict_topic_only: bool,
        existing_video_counts: dict[str, int],
        generated_video_counts: dict[str, int],
        accepted_clip_contexts_by_video: dict[str, list[dict[str, Any]]],
        target_clip_duration_sec: int,
        page_hint: int,
        recovery_stage: int,
        default_max_segments_per_video: int,
    ) -> list[dict[str, Any]]:
        if not material_id or not request_key or recovery_stage < self.REFILL_STAGE_MULTI_CLIP_STRICT:
            return []

        max_videos = 3 if recovery_stage < self.REFILL_STAGE_MULTI_CLIP_EXPANDED else 5
        candidate_rows = fetch_all(
            conn,
            """
            SELECT
                v.*,
                COUNT(r.id) AS request_reel_count,
                MAX(r.created_at) AS last_reel_created_at
            FROM reels r
            JOIN reel_generations g ON g.id = r.generation_id
            JOIN videos v ON v.id = r.video_id
            WHERE g.material_id = ?
              AND g.request_key = ?
              AND COALESCE(v.duration_sec, 0) >= 480
            GROUP BY v.id
            ORDER BY COUNT(r.id) DESC, MAX(r.created_at) DESC
            LIMIT ?
            """,
            (material_id, request_key, max_videos * 3),
        )
        if not candidate_rows:
            return []

        state_priority = {
            self.MINING_STATE_HIGH_YIELD: 0,
            self.MINING_STATE_PARTIALLY_MINED: 1,
            self.MINING_STATE_UNMINED: 2,
            self.MINING_STATE_LOW_YIELD: 3,
            self.MINING_STATE_EXHAUSTED: 4,
        }
        ranked_rows: list[tuple[tuple[Any, ...], dict[str, Any], dict[str, Any] | None]] = []
        for row in candidate_rows:
            video_id = str(row.get("id") or "").strip()
            if not video_id:
                continue
            state_row = self._fetch_request_video_mining_state(
                conn,
                material_id=material_id,
                request_key=request_key,
                video_id=video_id,
            )
            mining_state = str((state_row or {}).get("mining_state") or self.MINING_STATE_UNMINED)
            if mining_state == self.MINING_STATE_EXHAUSTED or bool(int((state_row or {}).get("exhausted") or 0)):
                continue
            if mining_state == self.MINING_STATE_LOW_YIELD and recovery_stage < self.REFILL_STAGE_MULTI_CLIP_EXPANDED:
                continue
            ranked_rows.append(
                (
                    (
                        state_priority.get(mining_state, 9),
                        -int(row.get("request_reel_count") or 0),
                        str(row.get("last_reel_created_at") or ""),
                    ),
                    dict(row),
                    state_row,
                )
            )
        ranked_rows.sort(key=lambda item: item[0])

        mined_candidates: list[dict[str, Any]] = []
        for _priority, row, state_row in ranked_rows[:max_videos]:
            video = dict(row)
            video_id = str(video.get("id") or "").strip()
            video_duration = int(video.get("duration_sec") or 0)
            video_segment_cap = self._video_segment_cap(
                video_duration_sec=video_duration,
                fast_mode=fast_mode,
                default_cap=default_max_segments_per_video,
                page_hint=page_hint,
            )
            existing_for_video = existing_video_counts.get(video_id, 0)
            generated_for_video = generated_video_counts.get(video_id, 0)
            accepted_count = existing_for_video + generated_for_video
            if accepted_count >= video_segment_cap:
                self._upsert_request_video_mining_state(
                    conn,
                    material_id=material_id,
                    request_key=request_key,
                    video_id=video_id,
                    mining_state=self.MINING_STATE_EXHAUSTED,
                    exhausted=True,
                )
                continue

            try:
                transcript = list(self.youtube_service.get_transcript(None, video_id) or [])
            except Exception:
                transcript = []
            if not transcript:
                prior_low_yield = str((state_row or {}).get("mining_state") or "") == self.MINING_STATE_LOW_YIELD
                self._upsert_request_video_mining_state(
                    conn,
                    material_id=material_id,
                    request_key=request_key,
                    video_id=video_id,
                    mining_state=self.MINING_STATE_EXHAUSTED if prior_low_yield else self.MINING_STATE_LOW_YIELD,
                    transcript_fetched=False,
                    exhausted=prior_low_yield,
                )
                continue

            remaining_capacity = max(1, video_segment_cap - accepted_count)
            dense_segments = self._dense_transcript_windows(
                conn,
                transcript=transcript,
                concept_terms=concept_terms,
                root_topic_terms=root_topic_terms,
                target_clip_duration_sec=target_clip_duration_sec,
                prior_contexts=accepted_clip_contexts_by_video.get(video_id, []),
                fast_mode=fast_mode,
                max_windows=min(18, max(4, remaining_capacity * 3)),
            )
            if not dense_segments:
                prior_low_yield = str((state_row or {}).get("mining_state") or "") == self.MINING_STATE_LOW_YIELD
                self._upsert_request_video_mining_state(
                    conn,
                    material_id=material_id,
                    request_key=request_key,
                    video_id=video_id,
                    mining_state=self.MINING_STATE_EXHAUSTED if prior_low_yield else self.MINING_STATE_LOW_YIELD,
                    transcript_fetched=True,
                    windows_scanned_delta=1,
                    exhausted=prior_low_yield,
                )
                continue

            stage_name = "broad" if recovery_stage < self.REFILL_STAGE_MULTI_CLIP_EXPANDED else "recovery"
            query_candidate = QueryCandidate(
                text=str(video.get("title") or concept.get("title") or video_id),
                strategy="video_mining",
                confidence=0.84,
                source_terms=concept_terms[:6],
                weight=0.94,
                stage=stage_name,
                source_surface=str(video.get("search_source") or "youtube_html"),
            )
            ranking = self._score_video_candidate(
                conn,
                video=video,
                query_candidate=query_candidate,
                concept_terms=concept_terms,
                context_terms=context_terms,
                concept_embedding=concept_embedding,
                subject_tag=subject_tag,
                visual_spec=visual_spec,
                preferred_video_duration=preferred_video_duration,
                stage_name=stage_name,
                require_context=False,
                fast_mode=fast_mode,
                root_topic_terms=root_topic_terms,
                strict_topic_only=strict_topic_only,
            )
            if not bool(ranking.get("passes", False)):
                self._upsert_request_video_mining_state(
                    conn,
                    material_id=material_id,
                    request_key=request_key,
                    video_id=video_id,
                    mining_state=self.MINING_STATE_LOW_YIELD,
                    transcript_fetched=True,
                    windows_scanned_delta=len(dense_segments),
                )
                continue

            quality_score = float(ranking.get("final_score") or 0.0)
            quality_tier = "high" if quality_score >= 0.7 else "medium" if quality_score >= 0.55 else "low"
            next_state = (
                self.MINING_STATE_HIGH_YIELD
                if len(dense_segments) >= max(3, remaining_capacity)
                else self.MINING_STATE_PARTIALLY_MINED
            )
            remaining_spans = [
                {
                    "t_start": float(segment.t_start),
                    "t_end": float(segment.t_end),
                }
                for segment in dense_segments[:12]
            ]
            self._upsert_request_video_mining_state(
                conn,
                material_id=material_id,
                request_key=request_key,
                video_id=video_id,
                mining_state=next_state,
                quality_tier=quality_tier,
                transcript_fetched=True,
                windows_scanned_delta=len(dense_segments),
                clusters_mined_delta=len(dense_segments),
                remaining_spans=remaining_spans,
            )
            mined_candidates.append(
                {
                    "video": video,
                    "video_id": video_id,
                    "video_duration": video_duration,
                    "video_relevance": ranking["text_relevance"],
                    "ranking": ranking,
                    "query_candidate": query_candidate,
                    "stage": stage_name,
                    "mined_transcript": transcript,
                    "mined_segments": dense_segments,
                    "mining_video_id": video_id,
                    "mining_request_key": request_key,
                    "mining_segment_cap": video_segment_cap,
                }
            )
        return mined_candidates

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

    def _slice_transcript_text_in_window(
        self,
        transcript: list[dict[str, Any]] | None,
        clip_start: float,
        clip_end: float,
        fallback_text: str,
    ) -> str:
        """Concatenate transcript cue text whose midpoint falls inside [clip_start, clip_end].

        Used to build a window-specific clip_context when a long topic segment
        is split into multiple consecutive reels, so novelty/self-containment
        scoring gets the text actually playing in this window rather than the
        full segment. Falls back to ``fallback_text`` (usually ``segment.text``)
        when no transcript is available or no cues land inside the window.
        """
        if not transcript:
            return fallback_text
        parts: list[str] = []
        for entry in transcript:
            try:
                start = float(entry.get("start") or 0.0)
                duration = float(entry.get("duration") or 0.0)
            except (TypeError, ValueError):
                continue
            mid = start + max(duration, 0.01) * 0.5
            if mid < float(clip_start) - 0.5:
                continue
            if mid > float(clip_end) + 0.5:
                break
            text_piece = str(entry.get("text") or "").replace("\n", " ").strip()
            if text_piece:
                parts.append(text_piece)
        joined = " ".join(parts).strip()
        return joined or fallback_text

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

    def _quality_floor_min_relevance(self, *, page_hint: int) -> float:
        requested = max(0.0, float(self._min_relevance_threshold or 0.0))
        safe_page = max(1, int(page_hint or 1))
        if safe_page <= 2:
            return max(requested, 0.3)
        if safe_page <= 5:
            return max(requested - 0.04, 0.24)
        return max(requested - 0.08, 0.2)

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

    def _passes_clip_quality_floor(
        self,
        conn,
        *,
        relevance_context: dict[str, Any],
        clip_context: dict[str, Any],
        prior_contexts: list[dict[str, Any]],
        subject_tag: str | None,
        retrieval_profile: RetrievalProfile,
        fast_mode: bool,
        page_hint: int,
        transcript_quality: TranscriptQuality | None = None,
    ) -> tuple[bool, str]:
        if float(relevance_context.get("score") or 0.0) < self._quality_floor_min_relevance(page_hint=page_hint):
            return (False, "low_relevance")
        # Reject clips when transcript coverage is very poor.
        if transcript_quality is not None and transcript_quality.coverage_ratio < 0.50:
            return (False, "low_transcript_coverage")
        if transcript_quality is not None and transcript_quality.largest_gap_sec > 20.0:
            clip_start = float(clip_context.get("clip_start_sec") or 0.0)
            clip_end = clip_start + float(clip_context.get("clip_duration_sec") or 0.0)
            if clip_start > 0 and clip_end > clip_start and transcript_quality.largest_gap_sec > (clip_end - clip_start) * 0.5:
                return (False, "low_transcript_coverage")
        self_containment = self._clip_self_containment_score(
            text=str(clip_context.get("text") or ""),
            clip_duration_sec=float(clip_context.get("clip_duration_sec") or 0.0),
        )
        if self_containment < (0.4 if fast_mode else 0.44):
            return (False, "low_self_containment")
        if not self._passes_same_video_clip_novelty(
            conn,
            clip_context=clip_context,
            prior_contexts=prior_contexts,
            subject_tag=subject_tag,
            retrieval_profile=retrieval_profile,
            fast_mode=fast_mode,
        ):
            return (False, "low_novelty")
        return (True, "")

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

    def _llm_relevance_prefilter(
        self,
        conn,
        *,
        candidates: list[dict[str, Any]],
        concept_text: str,
        subject_tag: str | None,
        context_terms: list[str],
    ) -> list[dict[str, Any]]:
        """Filter candidates using an LLM. Drops clearly irrelevant videos.

        Only runs when ``self.openai_client`` is available.  On any failure,
        all candidates pass through (graceful degradation).  Results are cached
        in ``llm_cache`` to avoid repeated API calls for the same candidate set.

        For concepts in ``AMBIGUOUS_CONCEPT_TOKENS`` (e.g. "calculus", which
        could refer to math or dental deposits or a movie title), the prompt
        includes explicit disambiguation examples and the post-filter is
        tightened — candidates omitted by the LLM are treated as rejections,
        not passes, and the confidence floor is raised.
        """
        if not self.openai_client or not candidates:
            return candidates

        # Ambiguity detection keys off both the subject_tag and the concept
        # text — we want strictness to apply whenever ANY token in the concept
        # is in the ambiguous set, not just the subject.
        concept_tokens = normalize_terms([str(subject_tag or ""), concept_text])
        ambiguous_query = bool(concept_tokens & self.AMBIGUOUS_CONCEPT_TOKENS)
        ambiguous_tokens = sorted(concept_tokens & self.AMBIGUOUS_CONCEPT_TOKENS)

        BATCH_SIZE = 8
        results: list[dict[str, Any]] = []
        # Cache-version bump ensures we don't reuse cached ratings produced by
        # the older, looser prompt for ambiguous concepts.
        prefilter_version = "v2"

        for batch_start in range(0, len(candidates), BATCH_SIZE):
            batch = candidates[batch_start : batch_start + BATCH_SIZE]
            video_list = "\n".join(
                f"{i + 1}. Title: {str((c.get('video') or {}).get('title', 'N/A'))[:120]} | "
                f"Channel: {str((c.get('video') or {}).get('channel_title', 'N/A'))[:60]} | "
                f"Duration: {int((c.get('video') or {}).get('duration_sec', 0))}s"
                for i, c in enumerate(batch)
            )

            ambiguity_note = ""
            if ambiguous_query:
                ambiguity_note = (
                    "\n\nCRITICAL DISAMBIGUATION: the subject terms "
                    f"{ambiguous_tokens} are known homonyms. "
                    "Reject any match that is NOT the educational sense of the term. "
                    "Examples to reject: movie/TV scenes that merely mention the word, "
                    "music videos, song lyrics with the word, sports teams / mascots, "
                    "unrelated branding, dental-hygiene content when the query is math "
                    "'calculus', biology 'cell' when the query is prison/storage cell, "
                    "'python' the snake when the query is the language, etc. "
                    "When in doubt about a generic-looking title, err toward REJECT "
                    "(set relevant=false, confidence≥0.6). Only accept titles that "
                    "clearly signal educational treatment of the concept."
                )

            user_prompt = (
                f"Target concept: {concept_text}\n"
                f"Subject area: {subject_tag or 'General'}\n"
                f"Context: {', '.join(context_terms[:10])}\n\n"
                "Rate each video's educational relevance to the target concept.\n"
                'Return JSON: {"ratings": [{"index": 1, "relevant": true, "confidence": 0.8, "reason": "brief"}]}\n'
                "You MUST return one rating per video, in the same order as listed.\n\n"
                "Rules:\n"
                "- relevant = the video likely contains substantive educational content about the concept\n"
                "- Music, vlogs, reactions, compilations = not relevant\n"
                "- Movie clips, film scenes, TV scenes, trailers = not relevant\n"
                "- Tangential matches (e.g., dental calculus vs math calculus) = not relevant\n"
                "- Videos that teach, explain, demonstrate the concept = relevant"
                f"{ambiguity_note}\n\n"
                f"Videos:\n{video_list}"
            )

            cache_payload = f"llm_prefilter_{prefilter_version}|{self.chat_model}|{user_prompt}"
            cache_key = f"llm_prefilter:{hashlib.sha256(cache_payload.encode('utf-8')).hexdigest()[:40]}"

            try:
                cached = fetch_one(conn, "SELECT response_json FROM llm_cache WHERE cache_key = ?", (cache_key,))
                if cached and cached.get("response_json"):
                    ratings = json.loads(cached["response_json"])
                else:
                    response = self.openai_client.chat.completions.create(
                        model=self.chat_model,
                        temperature=0.0,
                        response_format={"type": "json_object"},
                        messages=[
                            {
                                "role": "system",
                                "content": "You are an educational content classifier. Always respond with valid JSON.",
                            },
                            {"role": "user", "content": user_prompt},
                        ],
                    )
                    raw = response.choices[0].message.content or "{}"
                    ratings = json.loads(raw)
                    upsert(
                        conn,
                        "llm_cache",
                        {
                            "cache_key": cache_key,
                            "response_json": dumps_json(ratings),
                            "created_at": now_iso(),
                        },
                        pk="cache_key",
                    )

                # Initialize each candidate with a default; for ambiguous
                # queries the default is REJECT (not pass) so that a partial
                # LLM response can't accidentally whitelist unrated videos.
                default_relevant = not ambiguous_query
                default_confidence = 0.5 if not ambiguous_query else 0.0
                for c in batch:
                    c["llm_relevant"] = default_relevant
                    c["llm_relevance"] = default_confidence
                    c["llm_rated"] = False

                for rating in ratings.get("ratings", []):
                    idx = int(rating.get("index", 0)) - 1
                    if 0 <= idx < len(batch):
                        batch[idx]["llm_relevance"] = float(rating.get("confidence", 0.5))
                        batch[idx]["llm_relevant"] = bool(rating.get("relevant", True))
                        batch[idx]["llm_rated"] = True
                results.extend(batch)
            except Exception:
                logger.debug("LLM pre-filter failed for batch; passing all candidates through", exc_info=True)
                for c in batch:
                    # Fail-open only for non-ambiguous queries. Ambiguous ones
                    # stay in the result so downstream transcript-relevance
                    # scoring can still reject them, but we mark them as
                    # unrated so downstream gates know to be stricter.
                    c["llm_relevance"] = 0.5
                    c["llm_relevant"] = True
                    c["llm_rated"] = False
                results.extend(batch)

        if ambiguous_query:
            # Strict path: only keep candidates the LLM explicitly marked
            # relevant with confidence ≥ 0.6. Unrated / omitted candidates are
            # dropped.
            return [
                c for c in results
                if bool(c.get("llm_relevant"))
                and float(c.get("llm_relevance") or 0.0) >= 0.6
            ]
        # Default path: only drop candidates the LLM is confident are NOT relevant.
        return [
            c for c in results
            if c.get("llm_relevant", True) or float(c.get("llm_relevance", 0.5)) > 0.7
        ]

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

    def _score_transcript_alignment(
        self,
        conn,
        transcript: list[dict[str, Any]],
        concept_terms: list[str],
        context_terms: list[str],
        concept_embedding: np.ndarray | None,
        subject_tag: str | None,
        visual_spec: dict[str, list[str]],
        require_context: bool,
        fast_mode: bool,
    ) -> dict[str, Any]:
        if not transcript:
            return {"concept_match": 0.0, "clipability_signal": 0.0, "topic_score": 0.0, "passes": False}

        excerpt_parts: list[str] = []
        first_window = []
        for entry in transcript[:120]:
            text = str(entry.get("text") or "").replace("\n", " ").strip()
            if not text:
                continue
            excerpt_parts.append(text)
            if float(entry.get("start") or 0.0) <= 70:
                first_window.append(text)
            if len(excerpt_parts) >= 80:
                break
        excerpt = " ".join(excerpt_parts).strip()
        early_excerpt = " ".join(first_window).strip()

        concept_match = self._semantic_similarity(
            conn,
            text=excerpt,
            concept_terms=concept_terms,
            concept_embedding=concept_embedding,
        )
        transcript_relevance = self._score_text_relevance(
            conn,
            text=excerpt,
            concept_terms=concept_terms,
            context_terms=context_terms,
            concept_embedding=concept_embedding,
            subject_tag=subject_tag,
        )
        transcript_passes = self._passes_relevance_gate(
            relevance=transcript_relevance,
            require_context=require_context,
            fast_mode=fast_mode,
        )
        early_signal = lexical_overlap_score(early_excerpt, concept_terms) if early_excerpt else 0.0
        visual_match = lexical_overlap_score(
            excerpt,
            [
                *visual_spec.get("environment", []),
                *visual_spec.get("objects", []),
                *visual_spec.get("actions", []),
            ],
        )
        dense_segments = 0
        for entry in transcript[:120]:
            text = str(entry.get("text") or "").strip()
            if len(text.split()) >= 4:
                dense_segments += 1
        density = min(1.0, dense_segments / max(1, min(60, len(transcript))))
        clipability_signal = 0.5 * density + 0.25 * early_signal + 0.25 * visual_match
        return {
            "concept_match": float(max(0.0, min(1.0, concept_match))),
            "clipability_signal": float(max(0.0, min(1.0, clipability_signal))),
            "topic_score": float(max(0.0, transcript_relevance.get("score") or 0.0)),
            "passes": bool(transcript_passes),
        }

    def _collapse_near_duplicate_candidates(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not candidates:
            return []
        ordered = sorted(
            candidates,
            key=lambda row: float((row.get("ranking") or {}).get("final_score") or 0.0),
            reverse=True,
        )
        kept: list[dict[str, Any]] = []
        for row in ordered:
            duplicate = False
            for prev in kept:
                if self._video_candidates_near_duplicate(row, prev):
                    duplicate = True
                    break
            if duplicate:
                continue
            kept.append(row)
        return kept

    def _video_candidates_near_duplicate(self, left: dict[str, Any], right: dict[str, Any]) -> bool:
        lv = left.get("video") or {}
        rv = right.get("video") or {}
        lt = self._normalize_title_for_similarity(str(lv.get("title") or ""))
        rt = self._normalize_title_for_similarity(str(rv.get("title") or ""))
        if not lt or not rt:
            return False
        title_sim = self._token_jaccard(lt, rt)
        ld = int(lv.get("duration_sec") or 0)
        rd = int(rv.get("duration_sec") or 0)
        close_duration = ld > 0 and rd > 0 and abs(ld - rd) <= 5
        same_channel = str(lv.get("channel_title") or "").strip().lower() == str(rv.get("channel_title") or "").strip().lower()
        return title_sim >= 0.86 or (title_sim >= 0.72 and close_duration and same_channel)

    def _normalize_title_for_similarity(self, text: str) -> set[str]:
        tokens = normalize_terms([text])
        return {t for t in tokens if t not in {"video", "official", "footage", "clip", "shorts"}}

    def _token_jaccard(self, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        overlap = len(left.intersection(right))
        union = len(left.union(right))
        return float(overlap / max(1, union))

    def _diversify_video_candidates(self, candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not candidates:
            return []
        remaining = sorted(
            candidates,
            key=lambda row: float((row.get("ranking") or {}).get("final_score") or 0.0),
            reverse=True,
        )
        selected: list[dict[str, Any]] = []
        while remaining and len(selected) < top_k:
            best_idx = 0
            best_score = -1.0
            for idx, cand in enumerate(remaining):
                relevance = float((cand.get("ranking") or {}).get("final_score") or 0.0)
                diversity_penalty = 0.0
                for prev in selected:
                    diversity_penalty = max(diversity_penalty, self._candidate_similarity(cand, prev))
                    if str((cand.get("video") or {}).get("channel_title") or "").strip().lower() == str(
                        (prev.get("video") or {}).get("channel_title") or ""
                    ).strip().lower():
                        diversity_penalty = max(diversity_penalty, 0.24)
                mmr = 0.78 * relevance - 0.22 * diversity_penalty
                if mmr > best_score:
                    best_score = mmr
                    best_idx = idx
            selected.append(remaining.pop(best_idx))
        return selected

    def _candidate_similarity(self, left: dict[str, Any], right: dict[str, Any]) -> float:
        lv = left.get("video") or {}
        rv = right.get("video") or {}
        title_sim = self._token_jaccard(
            self._normalize_title_for_similarity(str(lv.get("title") or "")),
            self._normalize_title_for_similarity(str(rv.get("title") or "")),
        )
        strategy_left = str((left.get("query_candidate") or QueryCandidate("", "literal", 0.5)).strategy or "")
        strategy_right = str((right.get("query_candidate") or QueryCandidate("", "literal", 0.5)).strategy or "")
        strategy_bonus = 0.08 if strategy_left == strategy_right else 0.0
        return float(max(0.0, min(1.0, title_sim + strategy_bonus)))

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

    def _build_query_variants(
        self,
        title: str,
        keywords: list[str],
        subject_tag: str | None,
        context_terms: list[str] | None = None,
    ) -> list[str]:
        concept = {
            "id": "legacy-query-builder",
            "title": self._clean_query_text(title),
            "keywords_json": dumps_json([self._clean_query_text(item) for item in keywords if self._clean_query_text(item)]),
            "summary": "",
        }
        plan = self._plan_query_set_for_concepts(
            concepts=[concept],
            subject_tag=subject_tag,
            material_context_terms=[self._clean_query_text(term) for term in (context_terms or []) if self._clean_query_text(term)],
            retrieval_profile="deep",
            fast_mode=False,
            video_pool_mode="short-first",
            preferred_video_duration="any",
            request_need=1,
            targeted_concept_id="legacy-query-builder",
        )
        if not plan.selected_concepts:
            return []
        concept_plan = plan.selected_concepts[0]
        queries = [
            concept_plan.literal_query.text,
            concept_plan.intent_query.text,
            *(query.text for query in concept_plan.expansion_queries),
        ]
        seen: set[str] = set()
        deduped: list[str] = []
        for query in queries:
            normalized = self._normalize_query_key(query)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(query)
        return deduped

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
        if self.openai_client is None:
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
            response = self.openai_client.chat.completions.create(
                model=self.chat_model,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            payload = json.loads(response.choices[0].message.content or "{}")
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

    def _rank_segments_by_relevance(
        self,
        conn,
        segments: list[SegmentMatch],
        concept_terms: list[str],
        context_terms: list[str],
        concept_embedding: np.ndarray | None,
        subject_tag: str | None,
        require_context: bool,
        fast_mode: bool,
    ) -> list[tuple[SegmentMatch, dict[str, Any]]]:
        ranked: list[tuple[SegmentMatch, dict[str, Any]]] = []
        for segment in segments:
            relevance = self._score_text_relevance(
                conn,
                text=segment.text,
                concept_terms=concept_terms,
                context_terms=context_terms,
                concept_embedding=concept_embedding,
                subject_tag=subject_tag,
            )
            passes = self._passes_relevance_gate(
                relevance=relevance,
                require_context=require_context,
                fast_mode=fast_mode,
            )
            relevance["passes"] = passes
            if not passes:
                continue

            combined_score = 0.65 * float(segment.score) + 0.35 * float(relevance["score"])
            ranked.append(
                (
                    SegmentMatch(
                        chunk_index=segment.chunk_index,
                        t_start=segment.t_start,
                        t_end=segment.t_end,
                        text=segment.text,
                        score=combined_score,
                    ),
                    relevance,
                )
            )

        ranked.sort(key=lambda row: row[0].score, reverse=True)
        return ranked

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

    def _upsert_video(self, conn, video: dict[str, Any]) -> None:
        upsert(
            conn,
            "videos",
            {
                "id": video["id"],
                "title": video["title"],
                "channel_title": video.get("channel_title", ""),
                "description": video.get("description", ""),
                "duration_sec": int(video.get("duration_sec") or 0),
                "view_count": int(video.get("view_count") or 0),
                "is_creative_commons": 1 if video.get("is_creative_commons") else 0,
                "created_at": now_iso(),
            },
        )

    def _load_or_create_transcript_chunks(
        self,
        conn,
        video_id: str,
        transcript: list[dict[str, Any]],
    ) -> tuple[list[TranscriptChunk], np.ndarray]:
        rows = fetch_all(
            conn,
            "SELECT id, chunk_index, t_start, t_end, text, embedding_json, created_at FROM transcript_chunks WHERE video_id = ? ORDER BY chunk_index ASC",
            (video_id,),
        )
        if rows:
            chunks = [
                TranscriptChunk(
                    chunk_index=int(r["chunk_index"]),
                    t_start=float(r["t_start"]),
                    t_end=float(r["t_end"]),
                    text=r["text"],
                )
                for r in rows
            ]
            parsed_embeddings = [self._parse_embedding_vector(r.get("embedding_json")) for r in rows]
            if all(embedding is not None for embedding in parsed_embeddings):
                embeddings = np.array(parsed_embeddings, dtype=np.float32)
                return chunks, embeddings

            texts = [chunk.text for chunk in chunks]
            embeddings = self.embedding_service.embed_texts(conn, texts)
            for row, chunk, emb in zip(rows, chunks, embeddings):
                if not self.embedding_service.should_persist_replacement(row.get("embedding_json")):
                    continue
                upsert(
                    conn,
                    "transcript_chunks",
                    {
                        "id": str(row["id"]),
                        "video_id": video_id,
                        "chunk_index": chunk.chunk_index,
                        "t_start": chunk.t_start,
                        "t_end": chunk.t_end,
                        "text": chunk.text,
                        "embedding_json": dumps_json(emb.tolist()),
                        "created_at": str(row.get("created_at") or now_iso()),
                    },
                )
            return chunks, embeddings

        chunks = chunk_transcript(transcript)
        if not chunks:
            return [], np.empty((0, self.embedding_service.dim), dtype=np.float32)

        texts = [c.text for c in chunks]
        embeddings = self.embedding_service.embed_texts(conn, texts)

        for chunk, emb in zip(chunks, embeddings):
            upsert(
                conn,
                "transcript_chunks",
                {
                    "id": str(uuid.uuid4()),
                    "video_id": video_id,
                    "chunk_index": chunk.chunk_index,
                    "t_start": chunk.t_start,
                    "t_end": chunk.t_end,
                    "text": chunk.text,
                    "embedding_json": dumps_json(emb.tolist()),
                    "created_at": now_iso(),
                },
            )

        return chunks, embeddings

    def _fast_segments_from_transcript(
        self,
        transcript: list[dict[str, Any]],
        concept_terms: list[str],
        max_segments: int = 2,
    ) -> list[SegmentMatch]:
        chunks = chunk_transcript(transcript)
        if not chunks:
            return self._fallback_segments_from_transcript(transcript)

        scored: list[SegmentMatch] = []
        for chunk in chunks:
            lexical = lexical_overlap_score(chunk.text, concept_terms)
            scored.append(
                SegmentMatch(
                    chunk_index=chunk.chunk_index,
                    t_start=chunk.t_start,
                    t_end=chunk.t_end,
                    text=chunk.text,
                    score=0.05 + 0.35 * lexical,
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        selected: list[SegmentMatch] = []
        for candidate in scored:
            overlap = False
            for prev in selected:
                latest_start = max(candidate.t_start, prev.t_start)
                earliest_end = min(candidate.t_end, prev.t_end)
                if earliest_end - latest_start > 6:
                    overlap = True
                    break
            if overlap:
                continue
            selected.append(candidate)
            if len(selected) >= max_segments:
                break

        if not selected:
            return self._fallback_segments_from_transcript(transcript)
        return selected

    def _topic_cut_segments_for_concept(
        self,
        *,
        transcript: list[dict[str, Any]],
        video_id: str,
        video_duration_sec: int,
        clip_min_len: int,
        clip_max_len: int,
        max_segments: int,
        concept_terms: list[str] | None = None,
    ) -> list[SegmentMatch]:
        """
        Cut a long-form video at the moments where the user's concept is
        actually being discussed, not where it's name-dropped in the intro.

        Strategy: **concept-mention clustering** with intro-skip.

        1. Skip the first INTRO_BUFFER_SEC of every video. The opening of a
           YouTube video almost always contains "today we'll talk about X"
           where X is the concept name — that's a name-drop, not a topic.
           Anchoring on it produces clips that are nothing but the welcome
           sequence, which was the bug the user reported.

        2. Walk the transcript and mark every cue that mentions any concept
           term (using `normalize_terms` from segmenter.py so plurals/stems
           still match — same semantics as the rest of the relevance ranker).

        3. **Cluster** consecutive mentions whose gap is ≤ MAX_GAP_SEC. A
           cluster is a contiguous run of mentions that belong to the same
           discussion. Isolated mentions (gap > MAX_GAP_SEC from neighbors)
           don't form a cluster — they're passing references, not topic
           content. This is the key insight that fixes the "clip is just
           the intro" bug: the intro contains ONE mention of the concept,
           which can never form a cluster of two.

        4. Drop clusters with fewer than MIN_CLUSTER_MENTIONS mentions —
           this is the hard relevance guarantee that every clip is
           substantively about the user's search.

        5. For each surviving cluster:
             * `t_start` = the start of the cue containing the FIRST mention
               in the cluster, with up to LEAD_IN_CUES of lead-in to catch
               introductory phrases like "Now let's talk about X". Lead-in
               cues that fall inside INTRO_BUFFER_SEC are excluded.
             * `t_end` = the end of the cue containing the LAST mention in
               the cluster, padded with TAIL_BUFFER_SEC for the speaker to
               finish their sentence.
             * Apply duration guardrails: if shorter than `clip_min_len`,
               expand the END outward (capped at the next cluster's start).
               If longer than `clip_max_len`, split into equal sub-parts.

        6. Sort clusters by mention count (densest first) and take the top
           `max_segments`.

        Returns [] when:
          * concept_terms is None or empty (caller should fall back)
          * the transcript is empty
          * no cluster meets the density threshold
        The caller handles the empty case by falling through to the legacy
        embedding-based `select_segments` path so we never silently emit
        zero reels for a video with a transcript.

        For YouTube Shorts the caller should NOT invoke this method — the
        existing `use_full_short_clip` path handles those by emitting the
        full video.
        """
        # Tunables — kept as locals so callers (and tests) can patch them
        # without touching module state.
        INTRO_BUFFER_SEC = 5.0
        MAX_GAP_SEC = 60.0
        MIN_CLUSTER_MENTIONS = 2
        LEAD_IN_CUES = 0
        TAIL_BUFFER_SEC = 1.0
        SENTENCE_END_SEARCH_SEC = 8.0  # look up to 8s beyond last mention for a sentence end
        # When multiple clusters from the same video survive, the user wants
        # them to play back-to-back without temporal gaps. We bridge adjacent
        # clusters whose gap is ≤ MERGE_GAP_SEC by extending each window's
        # boundary to the gap midpoint. Beyond this distance, the clusters
        # are too far apart to be perceived as the same continuous discussion
        # and we leave them disjoint.
        MERGE_GAP_SEC = 120.0

        if not concept_terms or not transcript:
            return []

        concept_token_set = normalize_terms(concept_terms)
        if not concept_token_set:
            return []

        # Compact representation of the transcript for the clustering loop.
        # We materialize parallel arrays so we can index by position cheaply.
        cue_starts: list[float] = []
        cue_ends: list[float] = []
        cue_texts: list[str] = []
        mention_idxs: list[int] = []

        for entry in transcript:
            try:
                start = float(entry.get("start") or 0.0)
                duration = float(entry.get("duration") or 0.0)
            except (TypeError, ValueError):
                continue
            text = str(entry.get("text") or "").replace("\n", " ").strip()
            if not text:
                continue
            cue_idx = len(cue_starts)
            cue_starts.append(start)
            cue_ends.append(start + max(duration, 0.01))
            cue_texts.append(text)

            # Skip mentions in the video intro entirely. The intro mention is
            # almost always a name-drop ("today we'll talk about X") and is
            # NOT where the substantive discussion happens.
            if start < INTRO_BUFFER_SEC:
                continue

            cue_token_set = normalize_terms([text])
            if concept_token_set & cue_token_set:
                mention_idxs.append(cue_idx)

        if len(mention_idxs) < MIN_CLUSTER_MENTIONS:
            return []

        # Cluster consecutive mentions: a cluster ends when the gap between
        # two mentions exceeds MAX_GAP_SEC. The intuition is that if the
        # creator hasn't mentioned the concept for a full minute, they've
        # moved to a different topic.
        clusters: list[list[int]] = []
        current: list[int] = [mention_idxs[0]]
        for prev, curr in zip(mention_idxs, mention_idxs[1:]):
            gap = cue_starts[curr] - cue_ends[prev]
            if gap <= MAX_GAP_SEC:
                current.append(curr)
            else:
                if len(current) >= MIN_CLUSTER_MENTIONS:
                    clusters.append(current)
                current = [curr]
        if len(current) >= MIN_CLUSTER_MENTIONS:
            clusters.append(current)

        if not clusters:
            return []

        # Step 1: rank clusters by density (mention count) and take the
        # `max_segments` most substantive ones. The user might have asked
        # for fewer reels than we have clusters; density is the right
        # selection criterion because it reflects how much of the video
        # is actually about the topic.
        clusters.sort(key=lambda c: (-len(c), c[0]))
        selected = clusters[:max_segments]

        # Step 2: compute the initial (t_start, t_end) for each selected
        # cluster. t_start is the first mention with up to LEAD_IN_CUES of
        # lead-in (capped by the intro buffer); t_end is the last mention
        # plus TAIL_BUFFER_SEC for the speaker to finish their sentence.
        # We do NOT yet bridge to neighbors — that's step 3.
        windows: list[dict[str, Any]] = []
        for cluster in selected:
            first_mention = cluster[0]
            last_mention = cluster[-1]

            # Snap t_start to the nearest sentence boundary before the
            # first mention (look back up to 8s for a sentence end in the
            # preceding cue — start right after it).
            raw_start = cue_starts[first_mention]
            t_start = max(0.0, raw_start - 1.0)  # default: 1s lead-in
            for back in range(first_mention - 1, max(-1, first_mention - 10), -1):
                if back < 0:
                    t_start = 0.0  # start of video is a clean boundary
                    break
                if cue_starts[first_mention] - cue_ends[back] > 8.0:
                    break
                cue_text = cue_texts[back].strip()
                if cue_text and re.search(r"[.!?…][\"'\)\]]*$", cue_text):
                    t_start = cue_ends[back]  # start right after the sentence end
                    break

            # Snap t_end to the nearest sentence boundary after the last
            # mention (look forward up to 8s for a sentence-ending cue).
            raw_end = cue_ends[last_mention] + TAIL_BUFFER_SEC
            t_end = raw_end
            best_sent_end = None
            best_sent_cost = float("inf")
            for fwd in range(last_mention, min(len(cue_texts), last_mention + 12)):
                if cue_starts[fwd] > raw_end + SENTENCE_END_SEARCH_SEC:
                    break
                cue_text = cue_texts[fwd].strip()
                if cue_text and re.search(r"[.!?…][\"'\)\]]*$", cue_text):
                    cost = abs(cue_ends[fwd] - raw_end)
                    if cost < best_sent_cost:
                        best_sent_cost = cost
                        best_sent_end = cue_ends[fwd]
            if best_sent_end is not None:
                t_end = best_sent_end

            if video_duration_sec and video_duration_sec > 0:
                t_end = min(t_end, float(video_duration_sec))

            windows.append({
                "cluster": cluster,
                "t_start": t_start,
                "t_end": t_end,
                "natural_start": t_start,  # remember the unmerged values for fallbacks
                "natural_end": t_end,
            })

        # Step 3: sort by time so we can walk neighbors. The user wants the
        # clips to come out in narrative order — first discussion first,
        # later discussion later — so they appear "side by side" in the feed
        # in the same order the YouTuber said them.
        windows.sort(key=lambda w: w["t_start"])

        # Step 4: bridge adjacent clusters by extending each pair to meet at
        # the midpoint of the gap between them, but ONLY when the gap is
        # ≤ MERGE_GAP_SEC. Larger gaps probably contain a different topic
        # and we don't want to drag the user through it just to make playback
        # contiguous. The result: when iOS or web plays the clips back to
        # back, source-time progresses continuously across the cluster
        # boundary instead of jumping forward.
        for i in range(len(windows) - 1):
            a = windows[i]
            b = windows[i + 1]
            gap = b["t_start"] - a["t_end"]
            if gap <= 0:
                # Already touching or overlapping (the LEAD_IN_CUES walk
                # might have made them adjacent). Ensure no overlap by
                # taking the average.
                shared = (a["t_end"] + b["t_start"]) / 2.0
                a["t_end"] = shared
                b["t_start"] = shared
                continue
            if gap > MERGE_GAP_SEC:
                continue
            midpoint = (a["t_end"] + b["t_start"]) / 2.0
            a["t_end"] = midpoint
            b["t_start"] = midpoint

        # Step 5: per-window guardrails — minimum length, optional split for
        # too-long windows. Now in time order, with bridged boundaries.
        segments: list[SegmentMatch] = []
        next_chunk_index = 0
        for win_idx, win in enumerate(windows):
            t_start = float(win["t_start"])
            t_end = float(win["t_end"])
            cluster = win["cluster"]

            # Min-length guardrail: if the window is too short (the cluster
            # was tiny and there was no neighbor to bridge to), expand the
            # END outward, never the START — the user explicitly wants the
            # clip to begin at the topic introduction.
            duration = t_end - t_start
            if duration < clip_min_len:
                deficit = clip_min_len - duration
                new_end = t_end + deficit
                if video_duration_sec and video_duration_sec > 0:
                    new_end = min(new_end, float(video_duration_sec))
                # Don't run into the next bridged window.
                if win_idx + 1 < len(windows):
                    next_start = float(windows[win_idx + 1]["t_start"])
                    if new_end > next_start - 0.1:
                        new_end = next_start - 0.1
                t_end = max(t_end, new_end)
                duration = t_end - t_start
                if duration < clip_min_len:
                    # Couldn't reach min length — drop rather than emit a
                    # too-short clip.
                    continue

            # Max-length guardrail: split too-long windows into equal parts.
            # Each sub-part is tagged with a shared ``cluster_group`` id so the
            # main candidate loop can refine them as a continuous chain
            # (forcing each sub-part's refined t_start to the previous
            # sub-part's refined t_end). Without this chaining, independent
            # refinement of each sub-part can introduce gaps or overlaps at
            # the sub-segment boundaries.
            sub_windows: list[tuple[float, float]] = []
            if duration > clip_max_len:
                import math
                num_parts = int(math.ceil(duration / clip_max_len))
                part_dur = duration / num_parts
                if part_dur < clip_min_len:
                    sub_windows = [(t_start, t_end)]
                else:
                    for i in range(num_parts):
                        a = t_start + i * part_dur
                        b = t_end if i == num_parts - 1 else (t_start + (i + 1) * part_dur)
                        sub_windows.append((a, b))
            else:
                sub_windows = [(t_start, t_end)]

            multi_part = len(sub_windows) > 1
            cluster_group_id = f"tcut-{video_id}-{win_idx}" if multi_part else ""

            for sub_idx, (a, b) in enumerate(sub_windows):
                text_parts: list[str] = []
                for i in range(len(cue_starts)):
                    if cue_starts[i] < a - 0.01:
                        continue
                    if cue_starts[i] > b + 0.01:
                        break
                    text_parts.append(cue_texts[i])
                joined_text = " ".join(text_parts).strip()
                if not joined_text:
                    continue

                # Density-weighted score: clusters with more mentions rank
                # higher. The downstream concept-embedding ranker still
                # adjusts this, so the absolute number doesn't matter much
                # — only the ordering within this video.
                density_score = 0.4 + min(0.5, 0.05 * len(cluster))

                match = SegmentMatch(
                    chunk_index=next_chunk_index,
                    t_start=float(a),
                    t_end=float(b),
                    text=joined_text,
                    score=density_score,
                )
                # Chaining metadata — read by the main candidate loop so
                # consecutive refined windows are contiguous.
                if multi_part:
                    match.cluster_group_id = cluster_group_id
                    match.cluster_sub_index = sub_idx
                    match.cluster_is_last = sub_idx == len(sub_windows) - 1
                segments.append(match)
                next_chunk_index += 1

        return segments

    def _topic_boundary_segments_for_concept(
        self,
        *,
        transcript: list[dict[str, Any]],
        video_id: str,
        video_duration_sec: int,
        clip_min_len: int,
        clip_max_len: int,
        max_segments: int,
        concept_terms: list[str] | None = None,
        concept_title: str = "",
        info_dict: dict[str, Any] | None = None,
    ) -> list[SegmentMatch]:
        """
        Use the proper topic-boundary detection from `topic_cut.py` to find
        segments where the creator INTRODUCES and TRANSITIONS AWAY from topics,
        then filter by concept relevance.

        This replaces `_topic_cut_segments_for_concept` as the primary path:
        - Chapters → Gemini/Groq LLM → sentence-transformer → Jaccard heuristic
        - Semantic + Jaccard relevance filtering against the user's concept
        - Sentence-boundary snapping for precise cuts

        Falls back to [] when:
        - No transcript available
        - Video is a Short
        - topic_cut produces no usable segments
        The caller falls through to `_topic_cut_segments_for_concept` when empty.
        """
        if not transcript or not concept_terms:
            return []

        from .topic_cut import (
            TranscriptCue as TopicCutCue,
            cut_video_into_topic_reels,
        )

        # Convert reels.py transcript format → topic_cut.TranscriptCue
        tc_cues: list[TopicCutCue] = []
        for entry in transcript:
            try:
                start = float(entry.get("start") or 0.0)
                duration = float(entry.get("duration") or 0.0)
                text = str(entry.get("text") or "").replace("\n", " ").strip()
            except (TypeError, ValueError):
                continue
            if not text:
                continue
            tc_cues.append(TopicCutCue(start=start, duration=max(duration, 0.01), text=text))

        if not tc_cues:
            return []

        # Build a query from concept_title + concept_terms for relevance filtering.
        query = concept_title or " ".join(concept_terms[:5])

        try:
            classification, topic_reels = cut_video_into_topic_reels(
                video_id,
                query=query,
                duration_sec=float(video_duration_sec) if video_duration_sec else None,
                use_llm=True,
                refine_boundaries=True,
                transcript=tc_cues,
                info_dict=info_dict,
                min_reel_sec=max(clip_min_len, 15),
                max_reel_sec=max(clip_max_len, 60),
            )
        except Exception:
            logger.exception(
                "topic_cut.cut_video_into_topic_reels failed for video %s",
                video_id,
            )
            return []

        if not topic_reels:
            return []

        # Convert TopicReel → SegmentMatch with source="topic_cut"
        segments: list[SegmentMatch] = []
        for i, tr in enumerate(topic_reels):
            # Build text from cues within this reel's time range.
            text_parts: list[str] = []
            for entry in transcript:
                try:
                    start = float(entry.get("start") or 0.0)
                except (TypeError, ValueError):
                    continue
                if start < tr.t_start - 0.01:
                    continue
                if start > tr.t_end + 0.01:
                    break
                text_parts.append(str(entry.get("text") or "").strip())
            joined_text = " ".join(text_parts).strip()

            segments.append(
                SegmentMatch(
                    chunk_index=i,
                    t_start=float(tr.t_start),
                    t_end=float(tr.t_end),
                    text=joined_text or tr.label,
                    score=1.0,
                    source="topic_cut",
                )
            )

        return segments[:max_segments]

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

    def _split_video_into_short_segments(
        self,
        concept_embedding: np.ndarray,
        chunk_embeddings: np.ndarray,
        chunks: list[TranscriptChunk],
        concept_terms: list[str],
        max_segments: int = 8,
    ) -> list[SegmentMatch]:
        if len(chunks) == 0 or len(chunk_embeddings) == 0:
            return []

        concept_vec = concept_embedding.astype(np.float32)
        sim_scores = (chunk_embeddings @ concept_vec).astype(np.float32)
        candidates: list[SegmentMatch] = []

        for idx, chunk in enumerate(chunks):
            text = chunk.text
            lexical = lexical_overlap_score(text, concept_terms)
            score = float(sim_scores[idx]) + 0.12 * lexical
            candidates.append(
                SegmentMatch(
                    chunk_index=chunk.chunk_index,
                    t_start=chunk.t_start,
                    t_end=chunk.t_end,
                    text=text,
                    score=score,
                )
            )

            if idx + 1 < len(chunks):
                nxt = chunks[idx + 1]
                start = chunk.t_start
                end = nxt.t_end
                length = end - start
                if 15 <= length <= 60:
                    pair_text = f"{chunk.text} {nxt.text}".strip()
                    pair_lexical = lexical_overlap_score(pair_text, concept_terms)
                    pair_score = float((sim_scores[idx] + sim_scores[idx + 1]) / 2.0) + 0.14 * pair_lexical + 0.02
                    candidates.append(
                        SegmentMatch(
                            chunk_index=chunk.chunk_index,
                            t_start=start,
                            t_end=end,
                            text=pair_text,
                            score=pair_score,
                        )
                    )

        candidates.sort(key=lambda item: item.score, reverse=True)

        selected: list[SegmentMatch] = []
        for cand in candidates:
            length = cand.t_end - cand.t_start
            if length < 15 or length > 60:
                continue
            overlap = False
            for prev in selected:
                latest_start = max(cand.t_start, prev.t_start)
                earliest_end = min(cand.t_end, prev.t_end)
                if earliest_end - latest_start > 6:
                    overlap = True
                    break
            if overlap:
                continue
            selected.append(cand)
            if len(selected) >= max_segments:
                break

        return selected

    def _merge_unique_segments(self, segments: list[SegmentMatch], max_items: int) -> list[SegmentMatch]:
        if not segments:
            return []
        deduped: list[SegmentMatch] = []
        for seg in sorted(segments, key=lambda item: item.score, reverse=True):
            is_dup = False
            for prev in deduped:
                if abs(seg.t_start - prev.t_start) <= 3 and abs(seg.t_end - prev.t_end) <= 3:
                    is_dup = True
                    break
            if is_dup:
                continue
            deduped.append(seg)
            if len(deduped) >= max_items:
                break
        return deduped

    def _fallback_segments_from_transcript(self, transcript: list[dict[str, Any]]) -> list[SegmentMatch]:
        if not transcript:
            return []

        entries = [entry for entry in transcript if (entry.get("text") or "").strip()]
        if not entries:
            return []

        start = float(entries[0].get("start") or 0.0)
        last = entries[-1]
        end = float(last.get("start") or start) + float(last.get("duration") or 0.0)
        if end <= start:
            return []

        snippet = " ".join(str(entry.get("text") or "").strip() for entry in entries[:40]).strip()
        if not snippet:
            return []

        return [
            SegmentMatch(
                chunk_index=0,
                t_start=start,
                t_end=end,
                text=snippet[:900],
                score=0.02,
            )
        ]

    def _fallback_segment_from_video_metadata(
        self,
        video: dict[str, Any],
        concept_terms: list[str],
        target_clip_duration_sec: int,
    ) -> SegmentMatch | None:
        title = str(video.get("title") or "").strip()
        description = str(video.get("description") or "").strip()
        metadata_text = " ".join(part for part in [title, description] if part).strip()
        if not metadata_text:
            return None

        duration_sec = int(video.get("duration_sec") or 0)
        desired_clip_len = self._normalize_target_clip_duration(target_clip_duration_sec)

        if duration_sec > 0:
            clip_len = min(desired_clip_len, max(15, duration_sec))
            if duration_sec <= clip_len:
                start = 0.0
            else:
                # Skip very early intro content when we must fall back to metadata-only clipping.
                start = max(0.0, min(float(duration_sec - clip_len), float(duration_sec) * 0.18))
            end = min(float(duration_sec), start + float(clip_len))
        else:
            start = 0.0
            end = float(desired_clip_len)

        if end - start < 15:
            end = start + 15.0

        lexical = lexical_overlap_score(metadata_text, concept_terms)
        return SegmentMatch(
            chunk_index=0,
            t_start=float(start),
            t_end=float(end),
            text=metadata_text[:900],
            score=0.04 + 0.08 * lexical,
        )

    def _build_dry_run_reel_preview(
        self,
        concept: dict[str, Any],
        video: dict[str, Any],
        segment: SegmentMatch,
        clip_window: tuple[int, int],
        relevance_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        start_sec, end_sec = clip_window
        video_id = str(video.get("id") or "").strip()
        video_url = f"https://www.youtube.com/embed/{video_id}?start={start_sec}&end={end_sec}" if video_id else ""
        matched_terms = [
            str(term).strip()
            for term in (relevance_context or {}).get("matched_terms", [])
            if str(term).strip()
        ][:8]
        return {
            "reel_id": f"dry-run-{video_id}-{start_sec}-{end_sec}",
            "concept_id": str(concept.get("id") or ""),
            "concept_title": str(concept.get("title") or ""),
            "video_title": str(video.get("title") or "").strip(),
            "video_description": self._clean_video_description(str(video.get("description") or "")),
            "channel_name": str(video.get("channel_title") or "").strip(),
            "ai_summary": "",
            "video_url": video_url,
            "t_start": float(start_sec),
            "t_end": float(end_sec),
            "transcript_snippet": str(segment.text or "")[:700],
            "takeaways": [],
            "captions": [],
            "score": float(segment.score),
            "relevance_score": float((relevance_context or {}).get("score") or segment.score),
            "discovery_score": float((relevance_context or {}).get("discovery_score") or 0.0),
            "clipability_score": float((relevance_context or {}).get("clipability_score") or 0.0),
            "query_strategy": str((relevance_context or {}).get("query_strategy") or ""),
            "retrieval_stage": str((relevance_context or {}).get("retrieval_stage") or ""),
            "source_surface": str((relevance_context or {}).get("source_surface") or ""),
            "matched_terms": matched_terms,
            "relevance_reason": str((relevance_context or {}).get("reason") or "").strip(),
            "video_duration_sec": int(video.get("duration_sec") or 0),
            "clip_duration_sec": float(max(0.0, end_sec - start_sec)),
        }

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
    ) -> dict[str, Any] | None:
        reel_id = str(uuid.uuid4())
        clip_min_len, clip_max_len, _ = self._resolve_clip_duration_bounds(
            target_clip_duration_sec=target_clip_duration_sec,
            target_clip_duration_min_sec=target_clip_duration_min_sec,
            target_clip_duration_max_sec=target_clip_duration_max_sec,
        )
        normalized_clip_window: tuple[int, int] | None
        if clip_window is None:
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
        video_id = video["id"]
        url = (
            f"https://www.youtube.com/embed/{video_id}?start={start_sec}&end={end_sec}"
            f"&playlist={video_id}&autoplay=1&mute=1&playsinline=1"
            "&loop=1&controls=1&modestbranding=1&iv_load_policy=3&rel=0"
        )
        takeaways = build_takeaways(concept, segment.text)
        captions = self._build_caption_cues(
            transcript=transcript or [],
            clip_start=float(start_sec),
            clip_end=float(end_sec),
            fallback_text=segment.text,
        )
        video_title = str(video.get("title") or "").strip()
        video_description = self._clean_video_description(str(video.get("description") or ""))
        ai_summary = self._brief_ai_summary(
            conn,
            video_id=video_id,
            concept_title=str(concept.get("title") or ""),
            video_title=video_title,
            video_description=video_description,
            transcript_snippet=segment.text[:700],
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
                    "transcript_snippet": segment.text[:700],
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
            "transcript_snippet": segment.text[:700],
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
            "video_duration_sec": int(video.get("duration_sec") or 0),
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
        if self.openai_client:
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
                response = self.openai_client.chat.completions.create(
                    model=self.chat_model,
                    temperature=0.2,
                    messages=[
                        {
                            "role": "system",
                            "content": "You write concise educational summaries.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                )
                generated = (response.choices[0].message.content or "").strip()
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
        if min_len < 1:
            min_len = 1

        start_sec = max(0.0, round(float(t_start), 2))
        end_sec = max(start_sec + 1.0, round(float(t_end), 2))

        if max_len > 0 and not allow_exceed_max and end_sec - start_sec > max_len:
            end_sec = start_sec + max_len
        if not allow_below_min and end_sec - start_sec < min_len:
            end_sec = start_sec + min_len

        if video_duration_sec > 0:
            if not allow_below_min and video_duration_sec < min_len:
                return None
            if end_sec > video_duration_sec:
                end_sec = video_duration_sec
            if not allow_below_min and end_sec - start_sec < min_len:
                start_sec = max(0, end_sec - min_len)
            if max_len > 0 and not allow_exceed_max and end_sec - start_sec > max_len:
                end_sec = start_sec + max_len
            if end_sec > video_duration_sec:
                end_sec = video_duration_sec

        if end_sec <= start_sec:
            return None
        if not allow_below_min and end_sec - start_sec < min_len:
            return None
        if max_len > 0 and not allow_exceed_max and end_sec - start_sec > max_len:
            return None
        return (start_sec, end_sec)

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

    def _full_short_clip_window(
        self,
        video_duration_sec: int,
        transcript: list[dict[str, Any]] | None = None,
    ) -> tuple[int, int] | None:
        """Return a window covering the full Short.

        When a transcript is supplied, the start is snapped to the first
        sentence beginning (or stays at 0 if the video opens on a sentence
        start anyway). This prevents a Short whose first caption runs
        "...and that's why calculus is useful!" from producing a reel that
        begins mid-thought. Only the start is adjusted — the end stays at
        the video duration so the full Short plays out.
        """
        t_start = 0.0
        if transcript:
            # Decide whether the transcript has real punctuation. If not,
            # use a pause-based proxy for sentence boundaries.
            use_pause_proxy = not self._transcript_has_terminal_punct(transcript)
            # Look at the first few cues. If cue[0] already starts a sentence
            # (or we have no sentence terminator before cue[1]), keep t_start
            # at 0. Otherwise skip ahead to the first cue whose preceding
            # cue ended on terminal punctuation.
            parsed: list[tuple[float, float, str]] = []
            for entry in transcript[:12]:
                try:
                    start = float(entry.get("start") or 0.0)
                    dur = float(entry.get("duration") or 0.0)
                except (TypeError, ValueError):
                    continue
                text = str(entry.get("text") or "").replace("\n", " ").strip()
                if not text:
                    continue
                parsed.append((start, dur, text))
            # Only shift forward if the earliest terminator is within the
            # first ~4 seconds AND leaves at least 8s of content afterwards.
            # This guards against Shorts with a single long monologue where
            # snapping forward would clip out most of the video.
            remaining_floor = max(8.0, float(video_duration_sec) * 0.4)
            for i in range(1, len(parsed)):
                prev_start, prev_dur, prev_text = parsed[i - 1]
                cur_start, _, _ = parsed[i]
                if cur_start > 4.0:
                    break
                # Boundary test:
                #   - punctuated: prev cue ends with .!?…
                #   - unpunctuated: gap between prev_end and cur_start ≥ 0.6s
                prev_end = prev_start + prev_dur
                is_boundary = (
                    (cur_start - prev_end) >= 0.6
                    if use_pause_proxy
                    else self._is_sentence_end(prev_text)
                )
                if is_boundary:
                    if float(video_duration_sec) - cur_start >= remaining_floor:
                        t_start = cur_start
                    break
        return self._normalize_clip_window(
            t_start=t_start,
            t_end=float(video_duration_sec),
            video_duration_sec=video_duration_sec,
            min_len=1,
            max_len=0,
            allow_exceed_max=True,
            allow_below_min=True,
        )

    def _split_into_consecutive_windows(
        self,
        transcript: list[dict[str, Any]] | None,
        segment_start: float,
        segment_end: float,
        video_duration_sec: int,
        min_len: int,
        max_len: int,
        *,
        max_splits: int = 5,
        skip_refinement: bool = False,
    ) -> list[tuple[float, float]]:
        """Split a long segment into consecutive sentence-aligned reels.

        If the segment fits in one reel (duration <= max_len + small extension),
        returns a single window.  Otherwise produces up to *max_splits*
        consecutive windows that together cover the full segment.  The last
        window may exceed *max_len* by more than the normal +8s allowance to
        ensure the segment ends on a clean sentence boundary.

        When *skip_refinement* is True (e.g. for topic_cut segments that are
        already snapped), splits use plain normalization rather than transcript
        refinement.
        """
        seg_dur = max(0.0, float(segment_end) - float(segment_start))
        # Single window path: segment fits or no transcript to refine with.
        single_threshold = float(max_len) + 8.0
        if seg_dur <= single_threshold or not transcript or skip_refinement:
            if skip_refinement or not transcript:
                win = self._normalize_clip_window(
                    segment_start, segment_end, video_duration_sec,
                    min_len=min_len, max_len=max_len,
                    allow_exceed_max=skip_refinement,
                )
            else:
                win = self._refine_clip_window_from_transcript(
                    transcript=transcript,
                    proposed_start=segment_start,
                    proposed_end=segment_end,
                    video_duration_sec=video_duration_sec,
                    min_len=min_len,
                    max_len=max_len,
                )
            return [win] if win else []

        # Multi-window path: segment exceeds max_len, split into consecutive reels.
        windows: list[tuple[float, float]] = []
        current_start = float(segment_start)
        seg_end = float(segment_end)

        for split_idx in range(max_splits):
            remaining = seg_end - current_start
            if remaining < float(min_len):
                # Less than min_len left — extend the previous window to absorb it.
                if windows:
                    prev_s, prev_e = windows[-1]
                    windows[-1] = (prev_s, min(seg_end, float(video_duration_sec) if video_duration_sec > 0 else seg_end))
                break

            is_last_split = (split_idx == max_splits - 1) or (remaining <= max_len + 16)

            if is_last_split:
                # Final reel: try to capture all remaining content; allow generous
                # extension to land on a sentence boundary even if it exceeds max_len.
                proposed_end = seg_end
                effective_max = int(max(remaining + 16, max_len))
                # Final reel is lenient — prefer a sentence end but accept a
                # clean cue boundary rather than dropping the remainder.
                win = self._refine_clip_window_from_transcript(
                    transcript=transcript,
                    proposed_start=current_start,
                    proposed_end=proposed_end,
                    video_duration_sec=video_duration_sec,
                    min_len=min_len,
                    max_len=effective_max,
                    min_start=current_start,
                )
            else:
                proposed_end = current_start + float(max_len)
                effective_max = max_len
                # Non-last split: REQUIRE the end to be a real sentence
                # terminator so the continuation in the next window can pick
                # up mid-thought without the prior reel having truncated
                # mid-sentence. If strict mode can't land one (sparse
                # transcript, no terminal punct in range), fall back to
                # lenient mode rather than emitting zero reels.
                win = self._refine_clip_window_from_transcript(
                    transcript=transcript,
                    proposed_start=current_start,
                    proposed_end=proposed_end,
                    video_duration_sec=video_duration_sec,
                    min_len=min_len,
                    max_len=effective_max,
                    min_start=current_start,
                    require_sentence_end=True,
                )
                if not win:
                    win = self._refine_clip_window_from_transcript(
                        transcript=transcript,
                        proposed_start=current_start,
                        proposed_end=proposed_end,
                        video_duration_sec=video_duration_sec,
                        min_len=min_len,
                        max_len=effective_max,
                        min_start=current_start,
                    )
            if not win:
                break
            s, e = win
            # Continuity invariant: the next window must start at or after
            # `current_start` (= previous window's end). Passing `min_start`
            # to the refiner usually enforces this, but re-clamp here as a
            # final safety net — _normalize_clip_window may round the refined
            # start, and zero-duration edge cases can still slip through.
            if s < current_start:
                s = current_start
            # Guard against zero-progress (e.g. refiner snapped end back past
            # current_start, or end <= start after the clamp).
            if e <= current_start + 1.0:
                break

            windows.append((s, e))
            current_start = float(e)

            if e >= seg_end - 2.0:
                break  # reached segment end

        if not windows:
            # Fallback: at least produce one window the normal way.
            win = self._refine_clip_window_from_transcript(
                transcript=transcript,
                proposed_start=segment_start,
                proposed_end=segment_end,
                video_duration_sec=video_duration_sec,
                min_len=min_len,
                max_len=max_len,
            )
            if win:
                windows.append(win)
        if not windows:
            # Final safety net: transcript refinement failed in both the
            # per-split loop and the single-window fallback. Produce a plain
            # normalized window so long segments never silently drop to zero
            # reels (mirrors the else-branch behavior in the main loop).
            win = self._normalize_clip_window(
                segment_start,
                segment_end,
                video_duration_sec,
                min_len=min_len,
                max_len=max_len,
            )
            if win:
                windows.append(win)
        return windows

    def _refine_clip_window_from_transcript(
        self,
        transcript: list[dict[str, Any]],
        proposed_start: float,
        proposed_end: float,
        video_duration_sec: int,
        min_len: int = 15,
        max_len: int = 60,
        min_start: float | None = None,
        require_sentence_end: bool = False,
    ) -> tuple[int, int] | None:
        """Snap ``(proposed_start, proposed_end)`` to sentence boundaries.

        Parameters of interest:
          * ``min_start`` — lower bound the refined start must NOT cross. When
            set (e.g. by ``_split_into_consecutive_windows`` to the previous
            window's end), the backward search is clamped so consecutive
            reels never overlap. Defaults to None (no clamp).
          * ``require_sentence_end`` — when True, refuse to return a window
            whose end is mid-sentence. Used to hard-fail non-last splits so
            the caller can pick a different boundary rather than silently
            truncating.
        """
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
            entries.append(
                {
                    "start": start,
                    "end": start + duration,
                    "text": text,
                }
            )

        if not entries:
            return self._normalize_clip_window(
                proposed_start,
                proposed_end,
                video_duration_sec,
                min_len=min_len,
                max_len=max_len,
            )

        # Decide whether the transcript has real punctuation. If not (most
        # YouTube auto-captions), fall back to a pause-based sentence proxy:
        # treat cue-index ``i`` as a sentence boundary when the gap from
        # cues[i-1] to cues[i] is ≥ 0.6s. This mirrors what speakers do — a
        # brief inhale between thoughts — and empirically lines up with
        # sentence breaks on unpunctuated captions.
        use_pause_boundaries = not self._transcript_has_terminal_punct(entries)
        pause_breaks = (
            self._cue_pause_breaks(entries, min_pause_sec=0.6)
            if use_pause_boundaries
            else set()
        )

        def _is_boundary_before(idx: int) -> bool:
            """Index `idx` starts a new sentence if the previous cue ended
            one, or (for unpunctuated transcripts) if there's a pause before
            it. Idx 0 is always a boundary (start of transcript)."""
            if idx <= 0:
                return True
            if use_pause_boundaries:
                return idx in pause_breaks
            return self._is_sentence_end(str(entries[idx - 1]["text"]))

        def _ends_sentence(idx: int) -> bool:
            """Cue `idx` ends a sentence if its text has terminal punct, or
            (for unpunctuated transcripts) if the NEXT cue starts after a
            pause (i.e. idx+1 is a pause break).

            Note: we deliberately do NOT treat the last cue as an automatic
            sentence end — the video may legitimately cut mid-sentence, and
            strict callers rely on this to reject bad endings.
            """
            if use_pause_boundaries:
                return (idx + 1) in pause_breaks
            return self._is_sentence_end(str(entries[idx]["text"]))

        desired_start = max(0.0, float(proposed_start))
        desired_end = max(desired_start + 1.0, float(proposed_end))
        # Clamp the back-search floor so we never return a window that begins
        # before a caller-supplied lower bound (e.g. the previous consecutive
        # window's end).
        back_search_floor = max(0.0, desired_start - 8.0)
        if min_start is not None:
            back_search_floor = max(back_search_floor, float(min_start))

        start_idx = 0
        for i, item in enumerate(entries):
            if float(item["end"]) >= desired_start:
                start_idx = i
                break

        # Search up to 8s before desired_start for a sentence boundary.
        # Pick the closest sentence-start to desired_start (minimize drift).
        refined_start_idx = start_idx
        best_start_cost = float("inf")
        for i in range(start_idx, -1, -1):
            if float(entries[i]["start"]) < back_search_floor:
                break
            if _is_boundary_before(i):
                cost = abs(float(entries[i]["start"]) - desired_start)
                if cost < best_start_cost:
                    best_start_cost = cost
                    refined_start_idx = i
        # Also search a few seconds forward for a sentence start — the
        # speaker may finish their previous sentence just after desired_start.
        search_ceil = min(len(entries) - 1, start_idx + 4)
        for i in range(start_idx + 1, search_ceil + 1):
            if float(entries[i]["start"]) > desired_start + 4.0:
                break
            if _is_boundary_before(i):
                cost = abs(float(entries[i]["start"]) - desired_start)
                if cost < best_start_cost:
                    best_start_cost = cost
                    refined_start_idx = i

        refined_start = float(entries[refined_start_idx]["start"])
        # Enforce min_start after picking a candidate — if the closest sentence
        # anchor landed before min_start (possible if min_start > desired_start
        # or the floor calculation was overridden elsewhere), advance to the
        # first cue whose start is at or after min_start so the refined start
        # still aligns to a cue boundary (not to a mid-cue second).
        if min_start is not None and refined_start < float(min_start):
            advanced = False
            for j in range(refined_start_idx, len(entries)):
                if float(entries[j]["start"]) >= float(min_start) - 0.01:
                    refined_start_idx = j
                    refined_start = float(entries[j]["start"])
                    advanced = True
                    break
            if not advanced:
                # Fell off the end of the transcript — fall back to the bare
                # min_start timestamp (better than a stale pre-clamp value).
                refined_start = float(min_start)
        min_end = refined_start + float(min_len)
        max_end = refined_start + float(max_len)

        best_sentence_end: float | None = None
        best_sentence_cost = float("inf")
        best_any_end: float | None = None
        best_any_cost = float("inf")

        # Allow sentence boundaries up to 8s past max_end — it's better to
        # have a slightly longer clip that ends cleanly than a shorter one
        # that cuts mid-sentence.  Non-sentence ends are still capped at
        # max_end so we only exceed the target for a clean ending.
        SENTENCE_SEARCH_BEYOND_SEC = 8.0

        for i in range(refined_start_idx, len(entries)):
            item_end = float(entries[i]["end"])
            if item_end < min_end:
                continue
            if item_end > max_end + SENTENCE_SEARCH_BEYOND_SEC:
                break
            # Accept any cue end within the normal window for fallback.
            if item_end <= max_end:
                any_cost = abs(item_end - desired_end)
                if any_cost < best_any_cost:
                    best_any_cost = any_cost
                    best_any_end = item_end
            # Accept sentence ends in the extended window (up to +8s). For
            # unpunctuated transcripts we use the pause-based proxy — a cue
            # "ends a sentence" if the next cue starts after a pause.
            if _ends_sentence(i):
                # Prefer sentence ends closer to desired_end, but penalize
                # going past max_end so we don't pick a sentence 8s late when
                # one exists at the target.
                overshoot = max(0.0, item_end - max_end)
                sent_cost = abs(item_end - desired_end) + 1.5 * overshoot
                if sent_cost < best_sentence_cost:
                    best_sentence_cost = sent_cost
                    best_sentence_end = item_end

        # When the caller requires a clean sentence ending (non-last split
        # during consecutive-window generation), refuse to fall back to a
        # mid-sentence cue end. Returning None tells the caller to stop
        # splitting and either let the last-split branch handle the rest or
        # fall through to single-window normalization.
        if require_sentence_end and best_sentence_end is None:
            return None

        refined_end = best_sentence_end if best_sentence_end is not None else best_any_end
        if refined_end is None:
            refined_end = min(max_end, max(min_end, desired_end))

        return self._normalize_clip_window(
            refined_start,
            refined_end,
            video_duration_sec,
            min_len=min_len,
            max_len=max_len,
        )

    def _is_sentence_end(self, text: str) -> bool:
        cleaned = text.strip()
        if not cleaned:
            return False
        return bool(re.search(r"[.!?…][\"'\)\]]*$", cleaned))

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
        sample = entries[:120]
        if not sample:
            return False
        n_punct = 0
        for entry in sample:
            if self._is_sentence_end(str(entry.get("text") or "")):
                n_punct += 1
        return (n_punct / len(sample)) >= 0.15

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

    def ranked_feed(
        self,
        conn,
        material_id: str,
        fast_mode: bool = False,
        generation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        material = fetch_one(conn, "SELECT subject_tag, source_type FROM materials WHERE id = ?", (material_id,))
        subject_tag = str((material or {}).get("subject_tag") or "").strip() or None
        strict_topic_only = str((material or {}).get("source_type") or "").strip().lower() == "topic"

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
                    "created_at": row["created_at"],
                }
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
            clip_key = self._clip_key(video_id, float(item.get("t_start") or 0), float(item.get("t_end") or 0))
            if clip_key in seen_clip_keys:
                continue
            if reel_id:
                seen_reel_ids.add(reel_id)
            seen_clip_keys.add(clip_key)
            deduped.append(dict(item))

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
        )
        return response_rows
