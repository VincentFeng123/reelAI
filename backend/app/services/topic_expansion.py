import hashlib
import html
import json
import logging
import os
import re
import unicodedata
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import httpx

from ..clip_engine.cancellation import raise_if_cancelled, run_cancellable
from ..clip_engine.errors import CancellationError
from ..db import dumps_json, fetch_one, now_iso, upsert
from .text_utils import normalize_whitespace

logger = logging.getLogger(__name__)


class TopicExpansionService:
    CACHE_VERSION = 9
    CACHE_TTL_SEC = 24 * 60 * 60
    WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
    WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
    DATAMUSE_API_URL = "https://api.datamuse.com/words"
    DEFAULT_TIMEOUT_SEC = 3.0
    SERVERLESS_TIMEOUT_SEC = 1.75
    WIKIPEDIA_RESULT_LIMIT = 6
    WIKIPEDIA_LINK_LIMIT = 40
    WIKIDATA_RESULT_LIMIT = 6
    DATAMUSE_RESULT_LIMIT = 10
    LANGUAGE_TOPICS = {
        "arabic",
        "chinese",
        "english",
        "french",
        "german",
        "greek",
        "hindi",
        "italian",
        "japanese",
        "korean",
        "latin",
        "mandarin",
        "portuguese",
        "russian",
        "spanish",
    }
    LANGUAGE_SUBTOPICS = (
        "grammar",
        "verb conjugation",
        "vocabulary",
        "pronunciation",
        "common phrases",
        "listening practice",
        "speaking practice",
        "reading practice",
        "tenses",
        "subjunctive mood",
    )
    STATIC_TOPIC_SUBTOPICS: dict[str, tuple[str, ...]] = {
        "accounting": (
            "financial accounting",
            "managerial accounting",
            "balance sheet",
            "income statement",
            "cash flow statement",
            "journal entries",
            "debits and credits",
        ),
        "anatomy": (
            "skeletal system",
            "muscular system",
            "nervous system",
            "cardiovascular system",
            "respiratory system",
            "digestive system",
        ),
        "chemistry": (
            "atomic structure",
            "chemical bonding",
            "stoichiometry",
            "equilibrium",
            "acids and bases",
            "thermodynamics",
        ),
        "calculus": (
            "limits",
            "derivatives",
            "integrals",
            "chain rule",
            "fundamental theorem of calculus",
            "optimization",
        ),
        "economics": (
            "supply and demand",
            "opportunity cost",
            "market equilibrium",
            "elasticity",
            "macroeconomics",
            "microeconomics",
            "fiscal policy",
            "monetary policy",
        ),
        "english": (
            "grammar",
            "vocabulary",
            "reading comprehension",
            "writing skills",
            "literary analysis",
            "essay structure",
        ),
        "finance": (
            "time value of money",
            "discounted cash flow",
            "risk and return",
            "capital budgeting",
            "financial statements",
            "portfolio theory",
        ),
        "geometry": (
            "triangles",
            "circles",
            "angles",
            "proofs",
            "coordinate geometry",
            "area and volume",
        ),
        "history": (
            "historical context",
            "causes and effects",
            "key events",
            "major figures",
            "primary sources",
            "historical analysis",
        ),
        "machine learning": (
            "supervised learning",
            "unsupervised learning",
            "regression",
            "classification",
            "neural networks",
            "model evaluation",
        ),
        "marketing": (
            "market segmentation",
            "consumer behavior",
            "branding",
            "positioning",
            "digital marketing",
            "marketing funnel",
        ),
        "physics": (
            "kinematics",
            "forces",
            "energy",
            "momentum",
            "electricity",
            "waves",
        ),
        "photosynthesis": (
            "light dependent reactions",
            "calvin cycle",
            "chloroplast",
            "carbon fixation",
            "atp and nadph",
            "stomata",
        ),
        "physiology": (
            "homeostasis",
            "nervous system",
            "endocrine system",
            "cardiovascular physiology",
            "respiratory physiology",
            "renal physiology",
        ),
        "psychology": (
            "cognitive psychology",
            "behavioral psychology",
            "developmental psychology",
            "social psychology",
            "abnormal psychology",
            "personality psychology",
            "research methods",
            "conditioning",
        ),
        "sociology": (
            "socialization",
            "culture",
            "social stratification",
            "institutions",
            "deviance",
            "inequality",
        ),
        "statistics": (
            "descriptive statistics",
            "probability distributions",
            "hypothesis testing",
            "confidence intervals",
            "regression",
            "sampling",
        ),
        "trigonometry": (
            "unit circle",
            "sine and cosine",
            "trigonometric identities",
            "graphs of trig functions",
            "law of sines",
            "law of cosines",
        ),
        "python programming": (
            "variables",
            "loops",
            "functions",
            "lists and dictionaries",
            "classes and objects",
            "file handling",
        ),
        "world war ii": (
            "causes of world war ii",
            "european theater",
            "pacific theater",
            "d day",
            "axis and allies",
            "holocaust",
        ),
        "algebra": (
            "linear equations",
            "quadratic equations",
            "polynomials",
            "factoring",
            "inequalities",
            "systems of equations",
        ),
        "art history": (
            "renaissance art",
            "baroque art",
            "impressionism",
            "modern art",
            "ancient art",
            "art movements",
        ),
        "biology": (
            "cell biology",
            "genetics",
            "evolution",
            "ecology",
            "molecular biology",
            "human biology",
        ),
        "computer science": (
            "algorithms",
            "data structures",
            "operating systems",
            "computer networks",
            "databases",
            "software engineering",
        ),
        "engineering": (
            "statics",
            "dynamics",
            "thermodynamics",
            "fluid mechanics",
            "materials science",
            "circuit analysis",
        ),
        "environmental science": (
            "climate change",
            "ecosystems",
            "biodiversity",
            "pollution",
            "renewable energy",
            "conservation",
        ),
        "music theory": (
            "scales and modes",
            "chord progressions",
            "rhythm and meter",
            "harmony",
            "counterpoint",
            "music notation",
        ),
        "nursing": (
            "pharmacology",
            "patient assessment",
            "pathophysiology",
            "medical surgical nursing",
            "fundamentals of nursing",
            "clinical skills",
        ),
        "organic chemistry": (
            "functional groups",
            "reaction mechanisms",
            "stereochemistry",
            "substitution reactions",
            "elimination reactions",
            "spectroscopy",
        ),
        "philosophy": (
            "epistemology",
            "ethics",
            "logic",
            "metaphysics",
            "political philosophy",
            "philosophy of mind",
        ),
        "political science": (
            "political theory",
            "comparative politics",
            "international relations",
            "public policy",
            "constitutional law",
            "political institutions",
        ),
    }
    DETERMINISTIC_ALIAS_TERMS: dict[str, tuple[str, ...]] = {
        "apiology": ("melittology",),
        "melittology": ("apiology",),
        "odonatology": ("odonata",),
    }
    DETERMINISTIC_COMPANION_TERMS: dict[str, tuple[str, ...]] = {
        "apiology": ("bee", "bees", "honey bee", "pollinator"),
        "melittology": ("bee", "bees", "honey bee", "pollinator"),
        "myrmecology": ("ant", "ants", "formicidae"),
        "bryology": ("moss", "mosses", "liverwort", "liverworts", "hornwort", "hornworts"),
        "odonatology": ("dragonfly", "dragonflies", "damselfly", "damselflies"),
    }
    EDUCATIONAL_CUE_TERMS = {
        "alphabet",
        "analysis",
        "basics",
        "conjugation",
        "continuity",
        "derivative",
        "derivatives",
        "equation",
        "equations",
        "example",
        "examples",
        "foundations",
        "fundamentals",
        "grammar",
        "integral",
        "integrals",
        "intro",
        "introduction",
        "lesson",
        "lessons",
        "limit",
        "limits",
        "listening",
        "macroeconomics",
        "microeconomics",
        "orthography",
        "phonology",
        "practice",
        "pronunciation",
        "proof",
        "reading",
        "speaking",
        "subjunctive",
        "syntax",
        "tense",
        "tenses",
        "theorem",
        "tutorial",
        "verb",
        "verbs",
        "vocabulary",
    }
    NEGATIVE_PREFIXES = (
        "comparison of",
        "culture of",
        "demographics of",
        "economy of",
        "glossary of",
        "geography of",
        "history of",
        "index of",
        "list of",
        "outline of",
        "timeline of",
    )
    NEGATIVE_TOKENS = {
        "academy",
        "association",
        "award",
        "awards",
        "band",
        "city",
        "country",
        "countries",
        "demographic",
        "demographics",
        "disambiguation",
        "festival",
        "film",
        "movie",
        "musician",
        "organization",
        "people",
        "population",
        "region",
        "school",
        "society",
        "song",
        "university",
    }
    PHRASE_STOPWORDS = {
        "a",
        "an",
        "and",
        "at",
        "by",
        "for",
        "from",
        "in",
        "into",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
    GENERIC_ACADEMIC_TERMS = {
        "applications",
        "basics",
        "concepts",
        "examples",
        "foundations",
        "fundamentals",
        "introduction",
        "overview",
    }
    COMPANION_STOPWORDS = {
        "area",
        "aspect",
        "basics",
        "behavior",
        "behaviour",
        "branch",
        "classification",
        "concept",
        "concepts",
        "discipline",
        "field",
        "forms",
        "history",
        "introduction",
        "method",
        "methods",
        "overview",
        "practice",
        "process",
        "processes",
        "research",
        "science",
        "species",
        "study",
        "system",
        "systems",
        "theory",
        "topic",
    }
    HUMANITIES_TOKENS = {
        "anthropology",
        "economics",
        "ethics",
        "history",
        "law",
        "literature",
        "philosophy",
        "politics",
        "psychology",
        "religion",
        "sociology",
    }
    QUANTITATIVE_TOKENS = {
        "algebra",
        "calculus",
        "derivative",
        "derivatives",
        "determinant",
        "eigenvalue",
        "eigenvalues",
        "eigenvector",
        "eigenvectors",
        "geometry",
        "gradient",
        "hessian",
        "jacobian",
        "linear",
        "matrix",
        "matrices",
        "mathematics",
        "multivariable",
        "partial",
        "probability",
        "statistics",
        "tensor",
        "tensors",
        "trigonometry",
        "vector",
        "vectors",
    }
    SCIENCE_TOKENS = {
        "anatomy",
        "astronomy",
        "biology",
        "chemistry",
        "ecology",
        "geology",
        "neuroscience",
        "physics",
        "physiology",
        "science",
    }
    TOPIC_SHAPE_CUE_TERMS = {
        "analysis",
        "behavior",
        "conditioning",
        "development",
        "dynamics",
        "grammar",
        "mechanics",
        "method",
        "methods",
        "process",
        "processes",
        "pronunciation",
        "research",
        "structure",
        "system",
        "systems",
        "theories",
        "theory",
    }

    def __init__(self) -> None:
        self.serverless_mode = bool(
            os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME") or os.getenv("K_SERVICE")
        )
        self.request_timeout_sec = self.SERVERLESS_TIMEOUT_SEC if self.serverless_mode else self.DEFAULT_TIMEOUT_SEC
        self._request_headers = {"User-Agent": "StudyReels/1.0 topic-expansion"}

    def expand_topic(
        self,
        conn,
        *,
        topic: str,
        max_subtopics: int = 10,
        max_aliases: int = 6,
        max_related_terms: int = 6,
        should_cancel: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        raise_if_cancelled(should_cancel)
        clean_topic = normalize_whitespace(topic or "").strip()
        if not clean_topic:
            return {"canonical_topic": "", "aliases": [], "subtopics": [], "related_terms": []}

        cache_key = self._cache_key(
            topic=clean_topic,
            max_subtopics=max_subtopics,
            max_aliases=max_aliases,
            max_related_terms=max_related_terms,
        )
        if conn is not None:
            cached = fetch_one(
                conn,
                "SELECT response_json, created_at FROM llm_cache WHERE cache_key = ?",
                (cache_key,),
            )
            if cached and self._cache_row_is_fresh(cached):
                try:
                    payload = json.loads(str(cached.get("response_json") or "{}"))
                    if isinstance(payload, dict):
                        raise_if_cancelled(should_cancel)
                        return payload
                except json.JSONDecodeError:
                    pass

        payload = self._expand_topic_uncached(
            topic=clean_topic,
            max_subtopics=max_subtopics,
            max_aliases=max_aliases,
            max_related_terms=max_related_terms,
            should_cancel=should_cancel,
        )
        raise_if_cancelled(should_cancel)
        if conn is not None and self._expansion_is_cacheable(clean_topic, payload):
            upsert(
                conn,
                "llm_cache",
                {
                    "cache_key": cache_key,
                    "response_json": dumps_json(payload),
                    "created_at": now_iso(),
                },
                pk="cache_key",
            )
        return payload

    def _cache_row_is_fresh(self, row: dict[str, Any]) -> bool:
        raw_created_at = str(row.get("created_at") or "").strip()
        if not raw_created_at:
            return False
        try:
            created_at = datetime.fromisoformat(raw_created_at.replace("Z", "+00:00"))
        except ValueError:
            return False
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - created_at).total_seconds() <= self.CACHE_TTL_SEC

    def _expansion_is_cacheable(self, topic: str, payload: dict[str, Any]) -> bool:
        """Do not turn a transient all-provider fallback into a 24-hour result."""
        canonical = self._normalize_key(str(payload.get("canonical_topic") or ""))
        topic_key = self._normalize_key(topic)
        if canonical and canonical != topic_key:
            return True
        if payload.get("aliases") or payload.get("related_terms"):
            return True
        if self._static_topic_subtopics(topic):
            return True
        generic = {
            self._normalize_key(term)
            for term in self._generic_family_subtopics(
                topic,
                likely_language=self._looks_like_language_topic(topic),
            )
        }
        subtopics = {
            self._normalize_key(str(term or ""))
            for term in payload.get("subtopics") or []
            if self._normalize_key(str(term or ""))
        }
        return bool(subtopics - generic)

    def _expand_topic_uncached(
        self,
        *,
        topic: str,
        max_subtopics: int,
        max_aliases: int,
        max_related_terms: int,
        should_cancel: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        raise_if_cancelled(should_cancel)
        clean_topic = normalize_whitespace(topic).strip()
        normalized_topic = self._normalize_key(clean_topic)
        if not normalized_topic:
            return {"canonical_topic": clean_topic, "aliases": [], "subtopics": [], "related_terms": []}

        likely_language = self._looks_like_language_topic(clean_topic)
        wikipedia_queries = [clean_topic]
        if likely_language and "language" not in normalized_topic:
            wikipedia_queries.insert(0, f"{clean_topic} language")

        wikipedia_results: list[dict[str, str]] = []
        for query in wikipedia_queries[:2]:
            raise_if_cancelled(should_cancel)
            wikipedia_results.extend(
                self._search_wikipedia_results(query, should_cancel=should_cancel)
            )
        search_titles = [str(item.get("title") or "") for item in wikipedia_results if str(item.get("title") or "")]
        wikidata_entities = self._search_wikidata_entities(
            clean_topic, should_cancel=should_cancel
        )

        canonical_topic = clean_topic
        if search_titles:
            canonical_topic = self._best_wikipedia_title(
                topic=clean_topic,
                titles=search_titles,
                likely_language=likely_language,
            ) or clean_topic
        elif wikidata_entities:
            canonical_topic = str((wikidata_entities[0] or {}).get("label") or clean_topic).strip() or clean_topic

        if canonical_topic and canonical_topic != clean_topic:
            likely_language = likely_language or self._looks_like_language_topic(canonical_topic)

        candidate_titles = [title for title in search_titles if title and title != canonical_topic]
        if len(candidate_titles) < max(4, max_subtopics // 2) and canonical_topic:
            candidate_titles.extend(
                self._fetch_wikipedia_links(
                    canonical_topic, should_cancel=should_cancel
                )
            )
        outline_titles = self._outline_titles(topic=clean_topic, canonical_topic=canonical_topic, titles=search_titles)
        weighted_candidates: list[tuple[str, float]] = []
        opaque_topic = self._is_opaque_single_token_topic(
            clean_topic,
            canonical_topic=canonical_topic,
            likely_language=likely_language,
        )

        # Use a local timeout to avoid permanently inflating the instance value.
        effective_timeout = self.request_timeout_sec * 1.5 if opaque_topic else self.request_timeout_sec

        aliases = self._collect_aliases(
            topic=clean_topic,
            canonical_topic=canonical_topic,
            titles=[
                *search_titles,
                *[str((item or {}).get("label") or "") for item in wikidata_entities],
                *[
                    alias
                    for item in wikidata_entities
                    for alias in ((item or {}).get("aliases") or [])
                ],
            ],
            max_aliases=max_aliases,
        )
        aliases = self._merge_term_lists(
            self._deterministic_alias_terms(topic=clean_topic, canonical_topic=canonical_topic),
            aliases,
            limit=max_aliases,
        )
        static_subtopics = self._static_topic_subtopics(clean_topic)
        generic_subtopics = self._generic_family_subtopics(clean_topic, likely_language=likely_language)
        weighted_candidates.extend((term, 4.4) for term in static_subtopics)
        weighted_candidates.extend((term, 2.0) for term in generic_subtopics)
        weighted_candidates.extend((title, 2.8) for title in candidate_titles)
        for outline_title in outline_titles:
            raise_if_cancelled(should_cancel)
            weighted_candidates.extend(
                (title, 4.0)
                for title in self._fetch_wikipedia_links(
                    outline_title, should_cancel=should_cancel
                )
            )
        for result in wikipedia_results:
            title = str(result.get("title") or "").strip()
            snippet = str(result.get("snippet") or "").strip()
            if title:
                weighted_candidates.append((title, 2.6))
            for phrase in self._extract_candidate_phrases(snippet, topic=clean_topic):
                weighted_candidates.append((phrase, 2.3))
        for entity in wikidata_entities:
            description = str((entity or {}).get("description") or "").strip()
            label = str((entity or {}).get("label") or "").strip()
            aliases_for_entity = [str(item).strip() for item in ((entity or {}).get("aliases") or []) if str(item).strip()]
            if label:
                weighted_candidates.append((label, 2.4))
            for alias in aliases_for_entity:
                weighted_candidates.append((alias, 2.1))
            for phrase in self._extract_candidate_phrases(description, topic=clean_topic):
                weighted_candidates.append((phrase, 1.9))
        companion_terms = self._collect_companion_terms(
            topic=clean_topic,
            canonical_topic=canonical_topic,
            wikipedia_results=wikipedia_results,
            wikidata_entities=wikidata_entities,
            max_terms=max(2, min(4, max_related_terms)),
        )
        companion_terms = self._merge_term_lists(
            self._deterministic_companion_terms(topic=clean_topic, canonical_topic=canonical_topic),
            companion_terms,
            limit=max_related_terms,
        )
        raw_datamuse_terms = self._fetch_datamuse_related_terms(
            clean_topic, should_cancel=should_cancel
        )
        if opaque_topic:
            datamuse_terms = [
                term for term in raw_datamuse_terms
                if self._is_topic_anchor_candidate(topic=clean_topic, canonical_topic=canonical_topic, candidate=term)
            ]
        else:
            datamuse_terms = raw_datamuse_terms
        weighted_candidates.extend((term, 1.6) for term in datamuse_terms)
        subtopics = self._collect_subtopics(
            topic=clean_topic,
            canonical_topic=canonical_topic,
            candidate_terms=weighted_candidates,
            likely_language=likely_language,
            max_subtopics=max_subtopics,
        )
        related_terms = self._collect_related_terms(
            topic=clean_topic,
            canonical_topic=canonical_topic,
            aliases=[*aliases, *datamuse_terms],
            companion_terms=companion_terms,
            subtopics=subtopics,
            likely_language=likely_language,
            max_related_terms=max_related_terms,
        )

        raise_if_cancelled(should_cancel)
        return {
            "canonical_topic": canonical_topic,
            "aliases": aliases,
            "subtopics": subtopics,
            "related_terms": related_terms,
        }

    def _merge_term_lists(self, *term_lists: list[str] | tuple[str, ...], limit: int) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for values in term_lists:
            for raw_value in values or []:
                cleaned = normalize_whitespace(str(raw_value or "")).strip()
                normalized = self._normalize_key(cleaned)
                if not cleaned or not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                ordered.append(cleaned)
                if len(ordered) >= limit:
                    return ordered[:limit]
        return ordered[:limit]

    def _deterministic_alias_terms(self, *, topic: str, canonical_topic: str) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for raw in [topic, canonical_topic]:
            normalized = self._normalize_key(raw)
            if not normalized:
                continue
            for alias in self.DETERMINISTIC_ALIAS_TERMS.get(normalized, ()):
                alias_clean = normalize_whitespace(alias).strip()
                alias_key = self._normalize_key(alias_clean)
                if not alias_clean or not alias_key or alias_key in seen:
                    continue
                seen.add(alias_key)
                ordered.append(alias_clean)
        return ordered

    def _deterministic_companion_terms(self, *, topic: str, canonical_topic: str) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for raw in [topic, canonical_topic]:
            normalized = self._normalize_key(raw)
            if not normalized:
                continue
            for term in self.DETERMINISTIC_COMPANION_TERMS.get(normalized, ()):
                clean_term = normalize_whitespace(term).strip()
                normalized_term = self._normalize_key(clean_term)
                if not clean_term or not normalized_term or normalized_term in seen:
                    continue
                seen.add(normalized_term)
                ordered.append(clean_term)
        return ordered

    def build_topic_search_terms(
        self,
        *,
        topic: str,
        expansion: dict[str, Any],
        limit: int = 8,
    ) -> list[str]:
        clean_topic = normalize_whitespace(topic or "").strip()
        canonical_topic = normalize_whitespace(str(expansion.get("canonical_topic") or "")).strip()
        likely_language = self._looks_like_language_topic(canonical_topic or clean_topic)
        opaque_topic = self._is_opaque_single_token_topic(
            clean_topic,
            canonical_topic=canonical_topic,
            likely_language=likely_language,
        )

        terms: list[str] = []
        seen: set[str] = set()

        def add_term(raw_value: Any, *, require_anchor: bool, allow_unanchored: bool) -> None:
            cleaned = normalize_whitespace(str(raw_value or "")).strip()
            normalized = self._normalize_key(cleaned)
            if not cleaned or not normalized or normalized in seen:
                return
            if self._is_low_signal_topic_term(topic=clean_topic, canonical_topic=canonical_topic, candidate=cleaned):
                return
            if require_anchor and opaque_topic and not (allow_unanchored and self._allows_unanchored_opaque_search_term(cleaned)) and not self._is_topic_anchor_candidate(
                topic=clean_topic,
                canonical_topic=canonical_topic,
                candidate=cleaned,
            ):
                return
            seen.add(normalized)
            terms.append(cleaned)

        add_term(clean_topic, require_anchor=False, allow_unanchored=False)
        add_term(canonical_topic, require_anchor=False, allow_unanchored=False)
        for key in ("aliases", "related_terms", "subtopics"):
            require_anchor = opaque_topic or key == "subtopics"
            allow_unanchored = opaque_topic and key == "related_terms"
            for raw_value in (expansion.get(key) or []):
                add_term(raw_value, require_anchor=require_anchor, allow_unanchored=allow_unanchored)
                if len(terms) >= limit:
                    return terms[:limit]
        return terms[:limit]

    def _cache_key(
        self,
        *,
        topic: str,
        max_subtopics: int,
        max_aliases: int,
        max_related_terms: int,
    ) -> str:
        payload = "|".join(
            [
                str(self.CACHE_VERSION),
                self._normalize_key(topic),
                str(max_subtopics),
                str(max_aliases),
                str(max_related_terms),
            ]
        )
        return f"topic_expansion:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"

    def _search_wikipedia_results(
        self,
        query: str,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[dict[str, str]]:
        payload = self._request_json(
            self.WIKIPEDIA_API_URL,
            {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": query,
                "srlimit": self.WIKIPEDIA_RESULT_LIMIT,
                "utf8": 1,
                "origin": "*",
            },
            should_cancel=should_cancel,
        )
        items = (((payload or {}).get("query") or {}).get("search") or [])
        results: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in items:
            title = self._clean_wikipedia_title(str((item or {}).get("title") or ""))
            snippet = self._clean_snippet_text(str((item or {}).get("snippet") or ""))
            normalized = self._normalize_key(title)
            if not title or not normalized or normalized in seen:
                continue
            seen.add(normalized)
            results.append({"title": title, "snippet": snippet})
        return results

    def _search_wikidata_entities(
        self,
        query: str,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[dict[str, Any]]:
        payload = self._request_json(
            self.WIKIDATA_API_URL,
            {
                "action": "wbsearchentities",
                "format": "json",
                "language": "en",
                "type": "item",
                "search": query,
                "limit": self.WIKIDATA_RESULT_LIMIT,
                "origin": "*",
            },
            should_cancel=should_cancel,
        )
        entities = payload.get("search") if isinstance(payload, dict) else []
        results: list[dict[str, Any]] = []
        for item in entities or []:
            if not isinstance(item, dict):
                continue
            label = normalize_whitespace(str(item.get("label") or "")).strip()
            description = normalize_whitespace(str(item.get("description") or "")).strip()
            aliases_in = item.get("aliases")
            aliases = [normalize_whitespace(str(alias)).strip() for alias in aliases_in or [] if normalize_whitespace(str(alias)).strip()]
            if not label:
                continue
            results.append({"label": label, "description": description, "aliases": aliases[:8]})
        return results

    def _fetch_datamuse_related_terms(
        self,
        topic: str,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[str]:
        try:
            async def request() -> Any:
                async with httpx.AsyncClient(
                    headers=self._request_headers,
                    timeout=self.request_timeout_sec,
                ) as client:
                    response = await client.get(
                        self.DATAMUSE_API_URL,
                        params={"ml": topic, "max": self.DATAMUSE_RESULT_LIMIT},
                    )
                    response.raise_for_status()
                    return response.json()

            payload = run_cancellable(request, should_cancel)
        except CancellationError:
            raise
        except (httpx.HTTPError, ValueError) as exc:
            logger.debug("topic expansion datamuse request failed for %s: %s", topic, exc)
            return []

        terms: list[str] = []
        seen: set[str] = set()
        for item in payload or []:
            if not isinstance(item, dict):
                continue
            value = normalize_whitespace(str(item.get("word") or "")).strip()
            normalized = self._normalize_key(value)
            if not value or not normalized or normalized in seen:
                continue
            if len(normalized.split()) > 4:
                continue
            if normalized in self.GENERIC_ACADEMIC_TERMS:
                continue
            seen.add(normalized)
            terms.append(value)
        return terms

    def _fetch_wikipedia_links(
        self,
        title: str,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[str]:
        clean_title = self._clean_wikipedia_title(title)
        if not clean_title:
            return []

        payload = self._request_json(
            self.WIKIPEDIA_API_URL,
            {
                "action": "query",
                "format": "json",
                "prop": "links",
                "titles": clean_title,
                "pllimit": self.WIKIPEDIA_LINK_LIMIT,
                "plnamespace": 0,
                "redirects": 1,
                "origin": "*",
            },
            should_cancel=should_cancel,
        )
        pages = (((payload or {}).get("query") or {}).get("pages") or {})
        results: list[str] = []
        seen: set[str] = set()
        for page in pages.values():
            for item in ((page or {}).get("links") or []):
                link_title = self._clean_wikipedia_title(str((item or {}).get("title") or ""))
                normalized = self._normalize_key(link_title)
                if not link_title or not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                results.append(link_title)
        return results

    def _best_wikipedia_title(self, *, topic: str, titles: list[str], likely_language: bool) -> str | None:
        best_title: str | None = None
        best_score = float("-inf")
        topic_key = self._normalize_key(topic)
        topic_tokens = set(topic_key.split())

        for raw_title in titles:
            title = self._clean_wikipedia_title(raw_title)
            normalized = self._normalize_key(title)
            if not title or not normalized:
                continue
            score = 0.0
            title_tokens = set(normalized.split())
            if normalized == topic_key:
                score += 5.0
            if topic_tokens and topic_tokens.issubset(title_tokens):
                score += 2.5
            if likely_language and "language" in title_tokens:
                score += 3.5
            if any(normalized.startswith(prefix) for prefix in self.NEGATIVE_PREFIXES):
                score -= 4.0
            if "(" in title and ")" in title:
                score -= 0.8
            if score > best_score:
                best_score = score
                best_title = title
        return best_title

    def _collect_aliases(
        self,
        *,
        topic: str,
        canonical_topic: str,
        titles: list[str],
        max_aliases: int,
    ) -> list[str]:
        topic_key = self._normalize_key(topic)
        opaque_topic = self._is_opaque_single_token_topic(
            topic,
            canonical_topic=canonical_topic,
            likely_language=self._looks_like_language_topic(canonical_topic or topic),
        )
        aliases: list[str] = []
        if canonical_topic and self._normalize_key(canonical_topic) != topic_key:
            aliases.append(canonical_topic)
        for title in titles:
            clean_title = self._clean_wikipedia_title(title)
            normalized = self._normalize_key(clean_title)
            if not clean_title or not normalized or normalized == topic_key:
                continue
            if any(normalized.startswith(prefix) for prefix in self.NEGATIVE_PREFIXES):
                continue
            if self._is_low_signal_topic_term(topic=topic, canonical_topic=canonical_topic, candidate=clean_title):
                continue
            if len(normalized.split()) > 4:
                continue
            if opaque_topic and not self._is_topic_anchor_candidate(
                topic=topic,
                canonical_topic=canonical_topic,
                candidate=clean_title,
            ):
                continue
            if clean_title not in aliases:
                aliases.append(clean_title)
            if len(aliases) >= max_aliases:
                break
        return aliases[:max_aliases]

    def _static_topic_subtopics(self, topic: str) -> list[str]:
        normalized_topic = self._normalize_key(topic)
        if not normalized_topic:
            return []
        for raw_topic, subtopics in self.STATIC_TOPIC_SUBTOPICS.items():
            normalized_raw = self._normalize_key(raw_topic)
            if (
                normalized_topic == normalized_raw
                or self._contains_token_sequence(normalized_topic, normalized_raw)
                or self._contains_token_sequence(normalized_raw, normalized_topic)
            ):
                return [normalize_whitespace(item).strip() for item in subtopics if normalize_whitespace(item).strip()]
        return []

    @staticmethod
    def _contains_token_sequence(haystack: str, needle: str) -> bool:
        haystack_tokens = haystack.split()
        needle_tokens = needle.split()
        if not haystack_tokens or not needle_tokens or len(needle_tokens) > len(haystack_tokens):
            return False
        width = len(needle_tokens)
        return any(
            haystack_tokens[index : index + width] == needle_tokens
            for index in range(len(haystack_tokens) - width + 1)
        )

    def _collect_subtopics(
        self,
        *,
        topic: str,
        canonical_topic: str,
        candidate_terms: list[tuple[str, float]],
        likely_language: bool,
        max_subtopics: int,
    ) -> list[str]:
        topic_key = self._normalize_key(topic)
        canonical_key = self._normalize_key(canonical_topic)
        opaque_topic = self._is_opaque_single_token_topic(
            topic,
            canonical_topic=canonical_topic,
            likely_language=likely_language,
        )
        scored_terms: dict[str, tuple[float, str]] = {}

        if likely_language:
            for index, term in enumerate(self.LANGUAGE_SUBTOPICS):
                normalized = self._normalize_key(term)
                scored_terms[normalized] = (5.0 - index * 0.15, term)

        for raw_title, source_weight in candidate_terms:
            clean_title = self._clean_wikipedia_title(raw_title)
            simplified = self._simplify_candidate_term(
                topic=topic,
                canonical_topic=canonical_topic,
                candidate=clean_title,
                likely_language=likely_language,
            )
            normalized = self._normalize_key(simplified)
            if not simplified or not normalized or normalized in {topic_key, canonical_key}:
                continue
            if self._is_low_signal_topic_term(topic=topic, canonical_topic=canonical_topic, candidate=simplified):
                continue
            if opaque_topic and not self._is_topic_anchor_candidate(
                topic=topic,
                canonical_topic=canonical_topic,
                candidate=simplified,
            ):
                continue
            score = self._score_candidate_term(
                topic=topic,
                canonical_topic=canonical_topic,
                candidate=simplified,
                likely_language=likely_language,
            )
            score += float(source_weight)
            if score < (2.4 if likely_language else 2.0):
                continue
            existing = scored_terms.get(normalized)
            if existing is None or score > existing[0]:
                scored_terms[normalized] = (score, simplified)

        ranked = sorted(scored_terms.values(), key=lambda item: (-item[0], len(item[1]), item[1].lower()))
        return [term for _score, term in ranked[:max_subtopics]]

    def _collect_related_terms(
        self,
        *,
        topic: str,
        canonical_topic: str,
        aliases: list[str],
        companion_terms: list[str],
        subtopics: list[str],
        likely_language: bool,
        max_related_terms: int,
    ) -> list[str]:
        topic_key = self._normalize_key(topic)
        related: list[str] = []

        if canonical_topic and self._normalize_key(canonical_topic) not in {topic_key, ""}:
            related.append(canonical_topic)

        for term in companion_terms:
            clean_term = normalize_whitespace(term).strip()
            normalized_term = self._normalize_key(clean_term)
            if not clean_term or not normalized_term or normalized_term == topic_key:
                continue
            if clean_term not in related:
                related.append(clean_term)

        for alias in aliases:
            clean_alias = normalize_whitespace(alias).strip()
            normalized_alias = self._normalize_key(clean_alias)
            if not clean_alias or not normalized_alias or normalized_alias == topic_key:
                continue
            if clean_alias not in related:
                related.append(clean_alias)

        if likely_language:
            for term in ("conversation practice", "listening comprehension", "verb tenses"):
                if term not in related:
                    related.append(term)

        for term in subtopics:
            normalized_term = self._normalize_key(term)
            if not normalized_term or normalized_term == topic_key:
                continue
            if term not in related:
                related.append(term)
            if len(related) >= max_related_terms:
                break
        return related[:max_related_terms]

    def _collect_companion_terms(
        self,
        *,
        topic: str,
        canonical_topic: str,
        wikipedia_results: list[dict[str, str]],
        wikidata_entities: list[dict[str, Any]],
        max_terms: int,
    ) -> list[str]:
        likely_language = self._looks_like_language_topic(canonical_topic or topic)
        if not self._is_opaque_single_token_topic(topic, canonical_topic=canonical_topic, likely_language=likely_language):
            return []

        scored_terms: dict[str, tuple[float, str]] = {}

        def add_term(raw_value: str, score: float) -> None:
            cleaned = self._clean_companion_term(raw_value, topic=topic, canonical_topic=canonical_topic)
            normalized = self._normalize_key(cleaned)
            if not cleaned or not normalized:
                return
            existing = scored_terms.get(normalized)
            if existing is None or score > existing[0]:
                scored_terms[normalized] = (score, cleaned)

        for result in wikipedia_results:
            title = str(result.get("title") or "").strip()
            if title and not self._is_topic_anchor_candidate(
                topic=topic,
                canonical_topic=canonical_topic,
                candidate=title,
            ):
                continue
            snippet = str(result.get("snippet") or "").strip()
            for term in self._extract_companion_terms_from_text(
                snippet,
                topic=topic,
                canonical_topic=canonical_topic,
            ):
                add_term(term, 3.6)

        for entity in wikidata_entities:
            label = str((entity or {}).get("label") or "").strip()
            if label and not self._is_topic_anchor_candidate(
                topic=topic,
                canonical_topic=canonical_topic,
                candidate=label,
            ):
                continue
            description = str((entity or {}).get("description") or "").strip()
            for term in self._extract_companion_terms_from_text(
                description,
                topic=topic,
                canonical_topic=canonical_topic,
            ):
                add_term(term, 3.2)

        ranked = sorted(scored_terms.values(), key=lambda item: (-item[0], len(item[1]), item[1].lower()))
        return [term for _score, term in ranked[:max_terms]]

    def _extract_companion_terms_from_text(
        self,
        text: str,
        *,
        topic: str,
        canonical_topic: str,
    ) -> list[str]:
        cleaned = self._clean_snippet_text(text)
        if not cleaned:
            return []

        patterns = (
            r"(?:scientific\s+)?study of ([A-Za-z][A-Za-z\s\-]{2,48})",
            r"focused on ([A-Za-z][A-Za-z\s\-]{2,48})",
            r"concerned with ([A-Za-z][A-Za-z\s\-]{2,48})",
            r"dealing with ([A-Za-z][A-Za-z\s\-]{2,48})",
        )
        companions: list[str] = []
        seen: set[str] = set()
        for pattern in patterns:
            for match in re.finditer(pattern, cleaned, flags=re.IGNORECASE):
                candidate = self._clean_companion_term(
                    match.group(1),
                    topic=topic,
                    canonical_topic=canonical_topic,
                )
                normalized = self._normalize_key(candidate)
                if not candidate or not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                companions.append(candidate)
                if len(companions) >= 6:
                    return companions
        return companions

    def _clean_companion_term(self, value: str, *, topic: str, canonical_topic: str) -> str:
        cleaned = self._clean_snippet_text(value)
        if not cleaned:
            return ""
        cleaned = re.split(r"[.;,:/]|(?:\s+(?:and|or|that|which|while)\s+)", cleaned, maxsplit=1)[0]
        cleaned = re.sub(r"^(?:a|an|the)\s+", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = normalize_whitespace(cleaned)
        normalized = self._normalize_key(cleaned)
        if not normalized:
            return ""
        if self._is_topic_anchor_candidate(topic=topic, canonical_topic=canonical_topic, candidate=cleaned):
            return ""
        tokens = normalized.split()
        if not (1 <= len(tokens) <= 3):
            return ""
        if any(token in self.PHRASE_STOPWORDS or token in self.NEGATIVE_TOKENS for token in tokens):
            return ""
        if any(token in self.GENERIC_ACADEMIC_TERMS or token in self.COMPANION_STOPWORDS for token in tokens):
            return ""
        if len("".join(tokens)) < 4:
            return ""
        return cleaned

    def _allows_unanchored_opaque_search_term(self, value: str) -> bool:
        normalized = self._normalize_key(value)
        if not normalized:
            return False
        tokens = normalized.split()
        if not (1 <= len(tokens) <= 3):
            return False
        if any(token in self.NEGATIVE_TOKENS or token in self.COMPANION_STOPWORDS for token in tokens):
            return False
        if any(token in self.GENERIC_ACADEMIC_TERMS for token in tokens):
            return False
        if any(token in self.EDUCATIONAL_CUE_TERMS for token in tokens):
            return False
        return True

    def _outline_titles(self, *, topic: str, canonical_topic: str, titles: list[str]) -> list[str]:
        clean_titles = [self._clean_wikipedia_title(item) for item in titles if self._clean_wikipedia_title(item)]
        seen: set[str] = set()
        outlined: list[str] = []
        for title in clean_titles:
            lowered = title.lower()
            if lowered.startswith("outline of ") or lowered.startswith("glossary of "):
                key = self._normalize_key(title)
                if key and key not in seen:
                    seen.add(key)
                    outlined.append(title)
        for candidate in (
            f"Outline of {canonical_topic or topic}",
            f"Glossary of {canonical_topic or topic}",
        ):
            clean_candidate = self._clean_wikipedia_title(candidate)
            key = self._normalize_key(clean_candidate)
            if clean_candidate and key and key not in seen:
                seen.add(key)
                outlined.append(clean_candidate)
        return outlined[:3]

    def _generic_family_subtopics(self, topic: str, *, likely_language: bool) -> list[str]:
        clean_topic = normalize_whitespace(topic).strip()
        if not clean_topic:
            return []
        normalized = self._normalize_key(clean_topic)
        tokens = set(normalized.split())
        if likely_language:
            return list(self.LANGUAGE_SUBTOPICS)
        if self._is_opaque_single_token_topic(clean_topic, canonical_topic=clean_topic, likely_language=likely_language):
            return []
        if tokens.intersection(self.HUMANITIES_TOKENS) or normalized.endswith("ology"):
            return [
                f"major theories in {clean_topic}",
                f"research methods in {clean_topic}",
                f"classic studies in {clean_topic}",
                f"applications of {clean_topic}",
                f"core concepts in {clean_topic}",
            ]
        if tokens.intersection(self.QUANTITATIVE_TOKENS):
            return [
                f"{clean_topic} worked examples",
                f"{clean_topic} problem solving",
                f"{clean_topic} formulas and rules",
                f"{clean_topic} applications",
                f"core concepts in {clean_topic}",
            ]
        if tokens.intersection(self.SCIENCE_TOKENS):
            return [
                f"foundations of {clean_topic}",
                f"research methods in {clean_topic}",
                f"key systems in {clean_topic}",
                f"processes in {clean_topic}",
                f"applications of {clean_topic}",
            ]
        return [
            f"core concepts in {clean_topic}",
            f"key terms in {clean_topic}",
            f"worked examples for {clean_topic}",
            f"applications of {clean_topic}",
            f"introduction to {clean_topic}",
        ]

    def _extract_candidate_phrases(self, text: str, *, topic: str) -> list[str]:
        cleaned = self._clean_snippet_text(text)
        if not cleaned:
            return []
        tokens = re.findall(r"[A-Za-z][A-Za-z\-']*", cleaned)
        phrases: list[str] = []
        seen: set[str] = set()
        topic_key = self._normalize_key(topic)
        for size in (4, 3, 2):
            for index in range(0, len(tokens) - size + 1):
                window = tokens[index : index + size]
                lowered = [token.lower() for token in window]
                if lowered[0] in self.PHRASE_STOPWORDS or lowered[-1] in self.PHRASE_STOPWORDS:
                    continue
                if lowered[0] in self.GENERIC_ACADEMIC_TERMS or lowered[-1] in self.GENERIC_ACADEMIC_TERMS:
                    continue
                if sum(1 for token in lowered if token in self.PHRASE_STOPWORDS) > 1:
                    continue
                if all(token in self.PHRASE_STOPWORDS for token in lowered):
                    continue
                phrase = normalize_whitespace(" ".join(window))
                normalized = self._normalize_key(phrase)
                if not normalized or normalized == topic_key or normalized in seen:
                    continue
                if len(normalized.split()) > 4:
                    continue
                if normalized in self.GENERIC_ACADEMIC_TERMS:
                    continue
                if any(normalized.startswith(prefix) for prefix in self.NEGATIVE_PREFIXES):
                    continue
                seen.add(normalized)
                phrases.append(phrase)
                if len(phrases) >= 12:
                    return phrases
        return phrases

    def _simplify_candidate_term(
        self,
        *,
        topic: str,
        canonical_topic: str,
        candidate: str,
        likely_language: bool,
    ) -> str:
        cleaned = self._clean_wikipedia_title(candidate)
        if not cleaned:
            return ""

        topic_variants = [
            self._normalize_key(topic),
            self._normalize_key(canonical_topic),
        ]
        for variant in topic_variants:
            if not variant:
                continue
            prefix = f"{variant} "
            normalized = self._normalize_key(cleaned)
            if likely_language and normalized.startswith(prefix):
                remainder = cleaned[len(variant) + 1 :].strip()
                remainder = re.sub(r"^[\-\–:]+", "", remainder).strip()
                if remainder:
                    return remainder
        return cleaned

    def _score_candidate_term(
        self,
        *,
        topic: str,
        canonical_topic: str,
        candidate: str,
        likely_language: bool,
    ) -> float:
        normalized = self._normalize_key(candidate)
        if not normalized:
            return float("-inf")
        if self._is_low_signal_topic_term(topic=topic, canonical_topic=canonical_topic, candidate=candidate):
            return float("-inf")
        if any(normalized.startswith(prefix) for prefix in self.NEGATIVE_PREFIXES):
            return float("-inf")
        tokens = set(normalized.split())
        if tokens.intersection(self.NEGATIVE_TOKENS):
            return float("-inf")
        if any(char.isdigit() for char in candidate):
            return float("-inf")

        score = 0.0
        word_count = len(normalized.split())
        if 1 <= word_count <= 4:
            score += 1.6
        elif word_count <= 6:
            score += 0.6
        else:
            score -= 1.0
        if any(term in normalized for term in self.EDUCATIONAL_CUE_TERMS):
            score += 1.8
        if tokens.intersection(self.TOPIC_SHAPE_CUE_TERMS):
            score += 0.8
        if any(char in candidate for char in ":,;/"):
            score -= 0.9
        if normalized in self.GENERIC_ACADEMIC_TERMS:
            score -= 0.8

        topic_tokens = set(self._normalize_key(topic).split())
        canonical_tokens = set(self._normalize_key(canonical_topic).split())
        if topic_tokens and topic_tokens.intersection(tokens):
            score += 0.9
        if canonical_tokens and canonical_tokens.intersection(tokens):
            score += 0.4

        if likely_language:
            if any(
                cue in normalized
                for cue in (
                    "alphabet",
                    "conjugation",
                    "grammar",
                    "listening",
                    "orthography",
                    "phonology",
                    "phrases",
                    "pronunciation",
                    "reading",
                    "speaking",
                    "subjunctive",
                    "syntax",
                    "tense",
                    "vocabulary",
                )
            ):
                score += 2.4
            elif not topic_tokens.intersection(tokens):
                score -= 1.5
        return score

    def _is_opaque_single_token_topic(
        self,
        topic: str,
        *,
        canonical_topic: str,
        likely_language: bool,
    ) -> bool:
        if likely_language:
            return False
        normalized_topic = self._normalize_key(topic)
        if not normalized_topic:
            return False
        topic_tokens = normalized_topic.split()
        if len(topic_tokens) != 1:
            return False
        token = topic_tokens[0]
        if len(token) < 7:
            return False
        if token in self.LANGUAGE_TOPICS:
            return False
        if token in self.STATIC_TOPIC_SUBTOPICS:
            return False
        if token in self.HUMANITIES_TOKENS or token in self.QUANTITATIVE_TOKENS or token in self.SCIENCE_TOKENS:
            return False
        canonical_tokens = self._normalize_key(canonical_topic).split()
        if len(canonical_tokens) > 1:
            return True
        return True

    def _is_topic_anchor_candidate(
        self,
        *,
        topic: str,
        canonical_topic: str,
        candidate: str,
    ) -> bool:
        normalized_candidate = self._normalize_key(candidate)
        if not normalized_candidate:
            return False
        candidate_tokens = set(normalized_candidate.split())
        for anchor in (topic, canonical_topic):
            normalized_anchor = self._normalize_key(anchor)
            if not normalized_anchor:
                continue
            anchor_tokens = set(normalized_anchor.split())
            if normalized_anchor in normalized_candidate or candidate_tokens.intersection(anchor_tokens):
                return True
        return False

    def _is_low_signal_topic_term(
        self,
        *,
        topic: str,
        canonical_topic: str,
        candidate: str,
    ) -> bool:
        normalized = self._normalize_key(candidate)
        if not normalized:
            return True
        opaque_topic = self._is_opaque_single_token_topic(
            topic,
            canonical_topic=canonical_topic,
            likely_language=self._looks_like_language_topic(canonical_topic or topic),
        )
        anchors = {
            self._normalize_key(topic),
            self._normalize_key(canonical_topic),
        }
        anchors.discard("")
        generic_prefixes = (
            "applications of",
            "classic studies in",
            "core concepts in",
            "introduction to",
            "key terms in",
            "major theories in",
            "research methods in",
            "worked examples for",
        )
        for anchor in anchors:
            if normalized in {f"{prefix} {anchor}" for prefix in generic_prefixes}:
                return True
        if anchors and any(anchor in normalized for anchor in anchors):
            if " also known" in normalized:
                return True
            if " contained in " in normalized:
                return True
            if normalized.startswith("about "):
                return True
            if " from greek" in normalized:
                return True
            if " scientific study" in normalized:
                return True
            if opaque_topic:
                tokens = normalized.split()
                for anchor in anchors:
                    if not anchor or anchor not in normalized:
                        continue
                    if not normalized.startswith(anchor) and len(tokens) > len(anchor.split()) + 1:
                        return True
                    if normalized.startswith(anchor):
                        suffix_tokens = tokens[len(anchor.split()) :]
                        if len(suffix_tokens) > 2:
                            return True
                        if any(len(token) <= 2 for token in suffix_tokens):
                            return True
        return False

    def _looks_like_language_topic(self, value: str) -> bool:
        normalized = self._normalize_key(value)
        if not normalized:
            return False
        if normalized in self.LANGUAGE_TOPICS:
            return True
        return normalized.endswith(" language")

    def _clean_wikipedia_title(self, value: str) -> str:
        cleaned = normalize_whitespace(value or "").strip()
        cleaned = re.sub(r"\s*\([^)]*\)\s*$", "", cleaned).strip()
        cleaned = cleaned.replace("_", " ")
        return normalize_whitespace(cleaned)

    def _clean_snippet_text(self, value: str) -> str:
        text = html.unescape(str(value or ""))
        text = re.sub(r"<[^>]+>", " ", text)
        return normalize_whitespace(text)

    def _normalize_key(self, value: str) -> str:
        cleaned = unicodedata.normalize("NFKC", normalize_whitespace(value or ""))
        cleaned = cleaned.strip().casefold().replace("_", " ")
        cleaned = re.sub(r"[^\w\+# ]+", " ", cleaned, flags=re.UNICODE)
        return re.sub(r"\s+", " ", cleaned).strip()

    def _request_json(
        self,
        url: str,
        params: dict[str, Any],
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        try:
            async def request() -> Any:
                async with httpx.AsyncClient(
                    headers=self._request_headers,
                    timeout=self.request_timeout_sec,
                ) as client:
                    response = await client.get(url, params=params)
                    response.raise_for_status()
                    return response.json()

            payload = run_cancellable(request, should_cancel)
            return payload if isinstance(payload, dict) else {}
        except CancellationError:
            raise
        except (httpx.HTTPError, ValueError) as exc:
            logger.debug("topic expansion request failed for %s: %s", url, exc)
            return {}
