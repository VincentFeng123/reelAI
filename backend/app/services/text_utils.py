import re
from collections import Counter

STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "with",
    "would",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    text = text.replace("\n", " ")
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, target_words: int = 170, overlap_words: int = 30) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    idx = 0
    while idx < len(words):
        end = min(idx + target_words, len(words))
        chunk = " ".join(words[idx:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        idx = max(0, end - overlap_words)
    return chunks


def headings_from_text(text: str, max_headings: int = 8) -> list[str]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    candidates: list[str] = []
    for line in lines:
        if len(line.split()) > 10:
            continue
        if line.endswith(":"):
            candidates.append(line.rstrip(":"))
            continue
        if line.istitle() or line.isupper():
            candidates.append(line.title())
    seen = set()
    uniq = []
    for c in candidates:
        if c.lower() in seen:
            continue
        seen.add(c.lower())
        uniq.append(c)
        if len(uniq) >= max_headings:
            break
    return uniq


def keyword_candidates(text: str, limit: int = 30) -> list[str]:
    tokens = [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z\-']+", text)]
    filtered = [t for t in tokens if t not in STOPWORDS and len(t) > 2]

    unigram_counts = Counter(filtered)
    bigram_counts: Counter[str] = Counter()

    for i in range(len(filtered) - 1):
        a, b = filtered[i], filtered[i + 1]
        if a in STOPWORDS or b in STOPWORDS:
            continue
        bigram_counts[f"{a} {b}"] += 1

    top_terms: list[str] = []
    for phrase, count in bigram_counts.most_common(limit):
        if count < 2:
            break
        top_terms.append(phrase)

    for word, count in unigram_counts.most_common(limit):
        if count < 2:
            break
        top_terms.append(word)

    seen = set()
    deduped = []
    for term in top_terms:
        if term in seen:
            continue
        seen.add(term)
        deduped.append(term)
        if len(deduped) >= limit:
            break

    # Short user inputs often have no repeated phrases; add top bigrams even at count 1.
    if len(deduped) < min(6, limit):
        for phrase, _ in bigram_counts.most_common(limit):
            if phrase in seen:
                continue
            seen.add(phrase)
            deduped.append(phrase)
            if len(deduped) >= limit:
                break

    # Final short-input fallback: add top tokens by frequency.
    if len(deduped) < min(6, limit):
        for word, _ in unigram_counts.most_common(limit):
            if word in seen:
                continue
            seen.add(word)
            deduped.append(word)
            if len(deduped) >= limit:
                break
    return deduped
