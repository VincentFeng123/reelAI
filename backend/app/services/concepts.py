import json
import logging
import os
import uuid

from .text_utils import headings_from_text, keyword_candidates, split_sentences

logger = logging.getLogger(__name__)


def extract_learning_objectives(text: str, limit: int = 5) -> list[str]:
    objectives: list[str] = []
    for sentence in split_sentences(text):
        lower = sentence.lower()
        if "objective" in lower or "goal" in lower or "you will learn" in lower:
            objectives.append(sentence)
        if len(objectives) >= limit:
            break
    return objectives


def _extract_concepts_via_llm(text: str, max_concepts: int = 12) -> list[dict] | None:
    """Fix G: Use LLM to extract higher-quality concepts when OpenAI is available."""
    from ..config import get_settings
    settings = get_settings()
    serverless_mode = bool(os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME") or os.getenv("K_SERVICE"))
    allow_openai_serverless = os.getenv("ALLOW_OPENAI_IN_SERVERLESS") == "1"
    if not (settings.openai_enabled and settings.openai_api_key and (not serverless_mode or allow_openai_serverless)):
        return None

    from .openai_client import build_openai_client
    client = build_openai_client(api_key=settings.openai_api_key, timeout=15.0, enabled=True)
    if client is None:
        return None

    truncated = text[:6000]
    prompt = f"""Extract the {max_concepts} most important study concepts from this text.
For each concept, provide:
- title: A concise concept name (2-5 words)
- keywords: 3-5 related search terms that would find good YouTube educational videos
- summary: A one-sentence description of what this concept covers

Return a JSON array of objects with keys: title, keywords (array of strings), summary.
Only return the JSON array, no other text.

Text:
{truncated}"""

    try:
        response = client.chat.completions.create(
            model=settings.openai_chat_model or "gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1500,
        )
        content = (response.choices[0].message.content or "").strip()
        # Strip markdown code fences if present
        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
            if content.endswith("```"):
                content = content[:-3]
        parsed = json.loads(content)
        if not isinstance(parsed, list):
            return None
        concepts: list[dict] = []
        for item in parsed[:max_concepts]:
            if not isinstance(item, dict) or not item.get("title"):
                continue
            keywords = item.get("keywords", [])
            if not isinstance(keywords, list):
                keywords = []
            concepts.append({
                "id": str(uuid.uuid4()),
                "title": str(item["title"]),
                "keywords": [str(k) for k in keywords[:5]],
                "summary": str(item.get("summary", ""))[:240],
            })
        return concepts if concepts else None
    except Exception as exc:
        logger.warning("LLM concept extraction failed; falling back to keyword extraction: %s", exc)
        return None


def extract_concepts(text: str, max_concepts: int = 12) -> list[dict]:
    # Fix G: Try LLM-powered extraction first for better accuracy
    llm_concepts = _extract_concepts_via_llm(text, max_concepts=max_concepts)
    if llm_concepts:
        return llm_concepts[:max_concepts]

    # Fallback to keyword-based extraction
    headings = headings_from_text(text, max_headings=max_concepts)
    keywords = keyword_candidates(text, limit=40)
    sentences = split_sentences(text)

    concepts: list[dict] = []

    for heading in headings:
        related = [k for k in keywords if any(tok in heading.lower() for tok in k.split())][:5]
        summary = _summary_for_terms(sentences, related or heading.split())
        concepts.append(
            {
                "id": str(uuid.uuid4()),
                "title": heading,
                "keywords": related[:5] or heading.lower().split()[:4],
                "summary": summary,
            }
        )

    i = 0
    while len(concepts) < max_concepts and i < len(keywords):
        term = keywords[i]
        title = term.title()
        if any(c["title"].lower() == title.lower() for c in concepts):
            i += 1
            continue
        related = [k for k in keywords if k != term and (term.split()[0] in k or any(t in term for t in k.split()))][:4]
        summary = _summary_for_terms(sentences, [term])
        concepts.append(
            {
                "id": str(uuid.uuid4()),
                "title": title,
                "keywords": [term] + related,
                "summary": summary,
            }
        )
        i += 1

    if not concepts:
        fallback = "Core Topic"
        concepts.append(
            {
                "id": str(uuid.uuid4()),
                "title": fallback,
                "keywords": [w for w in text.split()[:5] if w],
                "summary": (sentences[0] if sentences else text[:200]).strip(),
            }
        )

    return concepts[:max_concepts]


def _summary_for_terms(sentences: list[str], terms: list[str]) -> str:
    terms_lower = [t.lower() for t in terms if t]
    if not sentences:
        return ""
    for sentence in sentences:
        lower = sentence.lower()
        if any(term in lower for term in terms_lower):
            return sentence[:240]
    return sentences[0][:240]


def build_takeaways(concept: dict, snippet: str, limit: int = 3) -> list[str]:
    bullets: list[str] = []
    snippet_sentences = split_sentences(snippet)

    if snippet_sentences:
        bullets.extend(s[:120] for s in snippet_sentences[:limit])

    if len(bullets) < limit:
        for kw in concept.get("keywords", []):
            bullet = f"Focus on: {kw}"
            if bullet not in bullets:
                bullets.append(bullet)
            if len(bullets) >= limit:
                break

    return bullets[:limit]
