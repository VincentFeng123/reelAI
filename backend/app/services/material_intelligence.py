import hashlib
import json
import os
import uuid
from typing import Any

from ..config import get_settings
from ..db import dumps_json, fetch_one, now_iso, upsert
from .concepts import extract_concepts, extract_learning_objectives
from .openai_client import build_openai_client
from .text_utils import normalize_whitespace
from .topic_expansion import TopicExpansionService


class MaterialIntelligenceService:
    TOPIC_SUBTOPICS: dict[str, tuple[str, ...]] = {
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
        "biology": (
            "cell signaling",
            "photosynthesis",
            "cell cycle",
            "genetics",
            "evolution",
        ),
        "chemistry": (
            "stoichiometry",
            "chemical bonding",
            "equilibrium",
            "acids and bases",
            "thermodynamics",
        ),
        "physics": (
            "kinematics",
            "forces",
            "energy",
            "momentum",
            "electricity",
        ),
        "computer science": (
            "data structures",
            "algorithms",
            "recursion",
            "time complexity",
            "dynamic programming",
        ),
        "machine learning": (
            "supervised learning",
            "unsupervised learning",
            "regression",
            "classification",
            "neural networks",
            "model evaluation",
        ),
        "linear algebra": (
            "vectors",
            "matrices",
            "linear transformations",
            "eigenvalues",
            "eigenvectors",
        ),
        "photosynthesis": (
            "light dependent reactions",
            "calvin cycle",
            "chloroplast",
            "carbon fixation",
            "atp and nadph",
            "stomata",
        ),
        "probability": (
            "conditional probability",
            "random variables",
            "expected value",
            "distributions",
            "bayes theorem",
        ),
        "python programming": (
            "variables",
            "loops",
            "functions",
            "lists and dictionaries",
            "classes and objects",
            "file handling",
        ),
        "statistics": (
            "descriptive statistics",
            "probability distributions",
            "hypothesis testing",
            "confidence intervals",
            "regression",
        ),
        "world war ii": (
            "causes of world war ii",
            "european theater",
            "pacific theater",
            "d day",
            "axis and allies",
            "holocaust",
        ),
    }

    def __init__(self) -> None:
        settings = get_settings()
        self.model = settings.openai_chat_model
        serverless_mode = bool(os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME") or os.getenv("K_SERVICE"))
        allow_openai_serverless = os.getenv("ALLOW_OPENAI_IN_SERVERLESS") == "1"
        can_use_openai = (
            bool(settings.openai_enabled)
            and bool(settings.openai_api_key)
            and (not serverless_mode or allow_openai_serverless)
        )
        self.client = build_openai_client(
            api_key=settings.openai_api_key,
            timeout=8.0,
            enabled=can_use_openai,
        )
        self.topic_expansion_service = TopicExpansionService()

    def extract_concepts_and_objectives(
        self,
        conn,
        text: str,
        subject_tag: str | None = None,
        max_concepts: int = 12,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        prefer_topic_seed = self._is_topic_only_material(text=text, subject_tag=subject_tag)
        opaque_topic_seed = False
        clean_subject = normalize_whitespace(subject_tag or "").strip()
        if prefer_topic_seed and clean_subject:
            expansion = self.topic_expansion_service.expand_topic(
                conn,
                topic=clean_subject,
                max_subtopics=max(8, max_concepts + 2),
                max_aliases=6,
                max_related_terms=6,
            )
            opaque_topic_seed = self.topic_expansion_service._is_opaque_single_token_topic(
                clean_subject,
                canonical_topic=str(expansion.get("canonical_topic") or clean_subject),
                likely_language=self.topic_expansion_service._looks_like_language_topic(
                    str(expansion.get("canonical_topic") or clean_subject)
                ),
            )
        topic_seed_concepts, topic_seed_objectives = self._topic_seed_material(
            conn=conn,
            text=text,
            subject_tag=subject_tag,
            max_concepts=max_concepts,
        )
        heuristic_concepts = extract_concepts(text, max_concepts=max_concepts)
        if prefer_topic_seed and topic_seed_concepts:
            heuristic_concepts = self._filter_topic_only_heuristic_concepts(
                heuristic_concepts,
                subject_tag=subject_tag,
            )
        heuristic_objectives = extract_learning_objectives(text, limit=6)
        base_concepts = self._merge_concepts(topic_seed_concepts, heuristic_concepts, max_concepts=max_concepts)
        base_objectives = self._merge_objectives(topic_seed_objectives, heuristic_objectives, limit=6)

        if prefer_topic_seed and opaque_topic_seed:
            return base_concepts, base_objectives

        if not self.client:
            return base_concepts, base_objectives

        payload = self._cached_or_generate(conn, text, subject_tag, max_concepts=max_concepts)
        llm_concepts = self._sanitize_concepts(payload.get("concepts"), max_concepts=max_concepts)
        llm_objectives = self._sanitize_objectives(payload.get("objectives"), limit=6)
        priority_focus = self._sanitize_priority_focus(payload, text)

        if priority_focus and not prefer_topic_seed:
            llm_concepts = [priority_focus, *llm_concepts]

        if prefer_topic_seed:
            merged_concepts = self._merge_concepts(base_concepts, llm_concepts, max_concepts=max_concepts)
            merged_objectives = self._merge_objectives(base_objectives, llm_objectives, limit=6)
        else:
            merged_concepts = self._merge_concepts(llm_concepts, base_concepts, max_concepts=max_concepts)
            merged_objectives = self._merge_objectives(llm_objectives, base_objectives, limit=6)
        return merged_concepts, merged_objectives

    def _topic_seed_material(
        self,
        *,
        conn,
        text: str,
        subject_tag: str | None,
        max_concepts: int,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        clean_subject = normalize_whitespace(subject_tag or "").strip()
        if not clean_subject:
            return [], []

        normalized_subject = clean_subject.lower()
        if not self._is_topic_only_material(text=text, subject_tag=subject_tag):
            return [], []

        expansion = self.topic_expansion_service.expand_topic(
            conn,
            topic=clean_subject,
            max_subtopics=max(8, max_concepts + 2),
            max_aliases=6,
            max_related_terms=6,
        )
        search_terms = self.topic_expansion_service.build_topic_search_terms(
            topic=clean_subject,
            expansion=expansion,
            limit=8,
        )
        canonical_topic = str(expansion.get("canonical_topic") or "").strip().lower()
        opaque_topic = self.topic_expansion_service._is_opaque_single_token_topic(
            clean_subject,
            canonical_topic=str(expansion.get("canonical_topic") or clean_subject),
            likely_language=self.topic_expansion_service._looks_like_language_topic(
                str(expansion.get("canonical_topic") or clean_subject)
            ),
        )
        search_subtopics = [
            term
            for term in search_terms
            if term.strip().lower() not in {normalized_subject, canonical_topic}
            and len(normalize_whitespace(term).split()) <= 2
            and (
                not opaque_topic
                or self.topic_expansion_service._is_topic_anchor_candidate(
                    topic=clean_subject,
                    canonical_topic=str(expansion.get("canonical_topic") or clean_subject),
                    candidate=term,
                )
            )
        ]
        expansion_subtopics = [str(item).strip() for item in (expansion.get("subtopics") or []) if str(item).strip()]
        prefer_search_subtopics = bool(search_subtopics) and (
            not expansion_subtopics
            or not any(len(normalize_whitespace(term).split()) <= 2 for term in expansion_subtopics[:4])
        )
        subtopics = self._dedupe_topic_terms(
            [
                *self._topic_subtopics(clean_subject),
                *(search_subtopics if prefer_search_subtopics else expansion_subtopics),
            ]
        )
        root_keywords = self._dedupe_topic_terms(
            [
                normalized_subject,
                *[term.lower() for term in search_terms],
                *[str(item).strip().lower() for item in subtopics],
            ]
        )
        title_subject = self._title_case_topic(clean_subject)
        concepts: list[dict[str, Any]] = [
            {
                "id": str(uuid.uuid4()),
                "title": title_subject,
                "keywords": root_keywords[:8] or [normalized_subject],
                "summary": self._topic_root_summary(
                    subject=title_subject,
                    canonical_topic=str(expansion.get("canonical_topic") or "").strip(),
                ),
            }
        ]

        if subtopics:
            for term in subtopics[: max(0, max_concepts - 1)]:
                related_keywords = self._dedupe_topic_terms(
                    [
                        term.lower(),
                        normalized_subject,
                        *root_keywords[:4],
                        f"{term.lower()} explained",
                    ]
                )
                concepts.append(
                    {
                        "id": str(uuid.uuid4()),
                        "title": self._title_case_topic(term),
                        "keywords": related_keywords[:8],
                        "summary": f"Key subtopic within {title_subject}: {term}.",
                    }
                )
        else:
            generic_titles = (
                f"{title_subject} Foundations",
                f"{title_subject} Worked Examples",
                f"{title_subject} Problem Solving",
            )
            for term in generic_titles[: max(0, max_concepts - 1)]:
                concepts.append(
                    {
                        "id": str(uuid.uuid4()),
                        "title": term,
                        "keywords": [normalized_subject, term.lower(), f"{normalized_subject} tutorial"][:8],
                        "summary": f"Foundational study path for {title_subject}.",
                    }
                )

        objectives = [
            f"Understand the core definitions and intuition behind {title_subject}.",
            f"Solve representative problems in {title_subject}.",
        ]
        objectives.extend(
            f"Explain how {term} fits into {title_subject}."
            for term in subtopics[:4]
        )
        return concepts[:max_concepts], objectives[:6]

    def _topic_subtopics(self, subject_tag: str) -> list[str]:
        normalized_subject = normalize_whitespace(subject_tag).strip().lower()
        if not normalized_subject:
            return []
        for topic, subtopics in self.TOPIC_SUBTOPICS.items():
            normalized_topic = topic.lower()
            if normalized_subject == normalized_topic or normalized_topic in normalized_subject or normalized_subject in normalized_topic:
                return [str(item).strip() for item in subtopics if str(item).strip()]
        return []

    def _title_case_topic(self, value: str) -> str:
        parts = [segment for segment in normalize_whitespace(value).split(" ") if segment]
        return " ".join(part[:1].upper() + part[1:] for part in parts)

    def _topic_root_summary(self, *, subject: str, canonical_topic: str) -> str:
        clean_canonical = normalize_whitespace(canonical_topic or "").strip()
        if clean_canonical and clean_canonical.lower() != subject.lower():
            return f"Core ideas, terminology, and intuition for {subject} ({clean_canonical})."
        return f"Core ideas, terminology, and intuition for {subject}."

    def _dedupe_topic_terms(self, values: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for raw in values:
            cleaned = normalize_whitespace(raw or "").strip()
            key = cleaned.lower()
            if not cleaned or key in seen:
                continue
            seen.add(key)
            deduped.append(cleaned)
        return deduped

    def _is_topic_only_material(self, *, text: str, subject_tag: str | None) -> bool:
        clean_subject = normalize_whitespace(subject_tag or "").strip().lower()
        if not clean_subject:
            return False
        normalized_text = normalize_whitespace(text).strip().lower()
        return normalized_text in {
            clean_subject,
            f"topic: {clean_subject}",
            f"topic {clean_subject}",
        }

    def _filter_topic_only_heuristic_concepts(
        self,
        concepts: list[dict[str, Any]],
        *,
        subject_tag: str | None,
    ) -> list[dict[str, Any]]:
        clean_subject = normalize_whitespace(subject_tag or "").strip().lower()
        if not clean_subject:
            return concepts

        filtered: list[dict[str, Any]] = []
        blocked_titles = {
            "topic",
            clean_subject,
            f"topic {clean_subject}",
            f"topic: {clean_subject}",
        }
        blocked_prefixes = (
            "topic ",
            "topic:",
        )
        for concept in concepts:
            title = normalize_whitespace(str(concept.get("title") or "")).strip()
            title_key = title.lower()
            if not title:
                continue
            if title_key in blocked_titles:
                continue
            if any(title_key.startswith(prefix) for prefix in blocked_prefixes):
                continue
            filtered.append(concept)
        return filtered

    def chat_assistant(
        self,
        message: str,
        topic: str | None = None,
        text: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        question = normalize_whitespace(message or "").strip()
        if not question:
            return "Ask me a study question and I will help."

        topic_clean = normalize_whitespace(topic or "").strip()
        text_clean = normalize_whitespace(text or "").strip()
        context_excerpt = self._build_excerpt(text_clean, max_chars=7000) if text_clean else ""

        if not self.client:
            if topic_clean:
                return (
                    f"Based on '{topic_clean}', focus on core definitions first, then one worked example. "
                    "Send your question again after adding more source text for a better answer."
                )
            return "Add a topic or source text, then ask a specific question for better guidance."

        system_prompt = (
            "You are a concise study coach. Answer directly, use plain language, and stay grounded in the provided context. "
            "If context is missing, state assumptions briefly. Keep responses under 120 words."
        )
        context_block = (
            f"Topic: {topic_clean or 'not provided'}\n"
            f"Source context:\n{context_excerpt or 'not provided'}"
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": context_block},
        ]
        for item in (history or [])[-8:]:
            role = str(item.get("role") or "").strip().lower()
            content = normalize_whitespace(str(item.get("content") or "")).strip()
            if role not in {"user", "assistant"} or not content:
                continue
            messages.append({"role": role, "content": content[:1800]})
        messages.append({"role": "user", "content": question[:2000]})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.25,
                messages=messages,
            )
            answer = (response.choices[0].message.content or "").strip()
            if answer:
                return answer[:1200]
        except Exception:
            pass

        if topic_clean:
            return f"I could not reach the model right now. For '{topic_clean}', start with one key definition and one example."
        return "I could not reach the model right now. Try again in a moment."

    def _cached_or_generate(
        self,
        conn,
        text: str,
        subject_tag: str | None,
        max_concepts: int,
    ) -> dict[str, Any]:
        cache_key = self._cache_key(text=text, subject_tag=subject_tag, max_concepts=max_concepts)
        cached = fetch_one(conn, "SELECT response_json FROM llm_cache WHERE cache_key = ?", (cache_key,))
        if cached:
            try:
                return json.loads(cached["response_json"])
            except json.JSONDecodeError:
                pass

        payload = self._generate_payload(text=text, subject_tag=subject_tag, max_concepts=max_concepts)
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

    def _generate_payload(self, text: str, subject_tag: str | None, max_concepts: int) -> dict[str, Any]:
        if not self.client:
            return {}

        cleaned = normalize_whitespace(text)
        excerpt = self._build_excerpt(cleaned, max_chars=14000)
        subject = subject_tag.strip() if subject_tag else ""

        system_prompt = (
            "You are an expert curriculum designer. Extract high-quality learning concepts from source material. "
            "Return strict JSON only with keys: concepts, objectives, priority_summary, priority_keywords."
        )
        user_prompt = (
            f"Subject hint: {subject or 'none'}\n"
            f"Max concepts: {max_concepts}\n"
            "Requirements:\n"
            "- concepts: array of objects {title, keywords, summary}\n"
            "- Each concept title should be concise and distinct.\n"
            "- keywords should be 3 to 8 specific terms/phrases.\n"
            "- summary should be 1-2 sentences and <= 240 characters.\n"
            "- objectives: array of concise learning outcomes.\n"
            "- priority_summary: 1 concise paragraph with the most important part to learn first.\n"
            "- priority_keywords: 4-8 terms useful for retrieving the most relevant explanations.\n"
            "Material:\n"
            f"{excerpt}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content or "{}"
            return json.loads(content)
        except Exception:
            return {}

    def _sanitize_concepts(self, value: Any, max_concepts: int) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        concepts: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            keywords_in = item.get("keywords")
            if isinstance(keywords_in, list):
                keywords = [str(k).strip().lower() for k in keywords_in if str(k).strip()]
            else:
                keywords = []
            if not keywords:
                keywords = [w.lower() for w in title.split()[:4] if w]

            summary = str(item.get("summary") or "").strip()
            if not summary:
                summary = f"Core idea: {title}"
            summary = summary[:240]

            concepts.append(
                {
                    "id": str(uuid.uuid4()),
                    "title": title[:80],
                    "keywords": keywords[:8],
                    "summary": summary,
                }
            )
            if len(concepts) >= max_concepts:
                break
        return concepts

    def _sanitize_objectives(self, value: Any, limit: int) -> list[str]:
        if not isinstance(value, list):
            return []
        objectives: list[str] = []
        for item in value:
            text = str(item).strip()
            if not text:
                continue
            objectives.append(text[:180])
            if len(objectives) >= limit:
                break
        return objectives

    def _merge_concepts(
        self,
        llm_concepts: list[dict[str, Any]],
        heuristic_concepts: list[dict[str, Any]],
        max_concepts: int,
    ) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen_titles: set[str] = set()

        for concept in llm_concepts + heuristic_concepts:
            title = str(concept.get("title") or "").strip()
            if not title:
                continue
            key = title.lower()
            if key in seen_titles:
                continue
            seen_titles.add(key)
            merged.append(concept)
            if len(merged) >= max_concepts:
                break
        return merged

    def _merge_objectives(self, llm_objectives: list[str], heuristic_objectives: list[str], limit: int) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for objective in llm_objectives + heuristic_objectives:
            cleaned = objective.strip()
            key = cleaned.lower()
            if not cleaned or key in seen:
                continue
            seen.add(key)
            merged.append(cleaned)
            if len(merged) >= limit:
                break
        return merged

    def _sanitize_priority_focus(self, payload: dict[str, Any], text: str) -> dict[str, Any] | None:
        if len(text) < 3500:
            return None

        summary = str(payload.get("priority_summary") or "").strip()
        if not summary:
            return None
        summary = summary[:240]

        keywords_in = payload.get("priority_keywords")
        keywords: list[str] = []
        if isinstance(keywords_in, list):
            keywords = [str(k).strip().lower() for k in keywords_in if str(k).strip()]
        if not keywords:
            keywords = [w.lower() for w in summary.split()[:6] if w]

        return {
            "id": str(uuid.uuid4()),
            "title": "Priority Focus",
            "keywords": keywords[:8],
            "summary": summary,
        }

    def _build_excerpt(self, cleaned: str, max_chars: int) -> str:
        if len(cleaned) <= max_chars:
            return cleaned

        head = cleaned[: int(max_chars * 0.45)]
        if " " in head:
            head = head.rsplit(" ", 1)[0]
        mid_start = max(0, len(cleaned) // 2 - int(max_chars * 0.1))
        mid_end = min(len(cleaned), mid_start + int(max_chars * 0.2))
        middle = cleaned[mid_start:mid_end].strip()
        if mid_start > 0 and mid_start < len(cleaned) and not cleaned[mid_start - 1].isspace() and " " in middle:
            middle = middle.split(" ", 1)[1]
        if mid_end < len(cleaned) and mid_end > 0 and not cleaned[mid_end - 1].isspace() and " " in middle:
            middle = middle.rsplit(" ", 1)[0]
        tail = cleaned[-int(max_chars * 0.35) :]
        if " " in tail:
            tail = tail.split(" ", 1)[1]
        return normalize_whitespace(f"{head}\n\n{middle}\n\n{tail}")[:max_chars]

    def _cache_key(self, text: str, subject_tag: str | None, max_concepts: int) -> str:
        normalized_text = normalize_whitespace(text)
        normalized_subject = normalize_whitespace(subject_tag or "").strip()
        text_hash = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
        payload = f"{self.model}|{max_concepts}|{normalized_subject}|{text_hash}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
