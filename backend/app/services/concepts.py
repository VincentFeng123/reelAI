import uuid

from .text_utils import headings_from_text, keyword_candidates, split_sentences


def extract_learning_objectives(text: str, limit: int = 5) -> list[str]:
    objectives: list[str] = []
    for sentence in split_sentences(text):
        lower = sentence.lower()
        if "objective" in lower or "goal" in lower or "you will learn" in lower:
            objectives.append(sentence)
        if len(objectives) >= limit:
            break
    return objectives


def extract_concepts(text: str, max_concepts: int = 12) -> list[dict]:
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
