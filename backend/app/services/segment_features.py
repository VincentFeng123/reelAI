"""Per-segment feature extraction for the importance ranker.

For a list of ``SegmentMatch`` from one video, compute:

  * TF-IDF top terms + per-video IDF table
  * Discourse marker hits (causal/definitional/contrastive/etc.)
  * Hearst pattern hits ("X is a Y", "we call this X", ...)
  * NER anchors via spaCy (with regex fallback)
  * Equation regex anchors
  * Lexical density (content / total words)
  * Instructional density (weighted blend, see formula)
  * Structural label + penalty (delegates to ``structural_classifier``)
  * Centrality (TextRank on cosine graph, with edge-discount + structural-cap)

Returns a ``VideoFeatureBundle`` carrying the per-segment list AND a per-video
IDF dict that ``topic_concentration`` reuses.

Lazy-loads spaCy ``en_core_web_lg`` and NetworkX once per process. Falls back
gracefully if either is unavailable.
"""

from __future__ import annotations

import logging
import math
import re
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .segmenter import SegmentMatch
from .structural_classifier import classify_passage, label_penalty

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Lexicons
# --------------------------------------------------------------------------- #


_DISCOURSE_MARKER_PHRASES: tuple[str, ...] = (
    # Causal
    "because", "since", "as a result", "therefore", "thus", "hence",
    "consequently", "so that", "due to", "owing to", "that's why",
    "this is why", "the reason is", "leads to", "results in", "causes",
    # Definitional
    "is defined as", "refers to", "means that", "in other words",
    "that is", "namely", "specifically", "by definition", "we call this",
    "what we call", "known as", "also called",
    # Contrastive
    "however", "but", "although", "whereas", "on the other hand",
    "in contrast", "instead", "rather than", "unlike",
    # Sequential
    "first", "second", "next", "then", "finally", "after that",
    "before that", "the next step", "step one", "step two",
    # Elaborative
    "for example", "for instance", "such as", "like when", "consider",
    "take the case of", "suppose",
    # Summative
    "in summary", "to summarize", "overall", "the key point",
    "the takeaway", "the main idea", "remember that", "the important thing",
    # Pedagogical
    "notice that", "observe that", "you can see", "as you can see",
    "this shows", "this tells us", "this means", "what this means",
    "here's why", "the trick is",
)

_DISCOURSE_MARKER_RE: re.Pattern[str] = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p in _DISCOURSE_MARKER_PHRASES) + r")\b",
    re.IGNORECASE,
)

_HEARST_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("x_is_a_y", re.compile(
        r"\b([A-Z][a-zA-Z\- ]{1,40}?)\s+is\s+(?:a|an|the)\s+([a-z][a-zA-Z\- ]{2,40})",
    )),
    ("x_refers_to_y", re.compile(
        r"\b([a-zA-Z\- ]{1,40})\s+refers?\s+to\s+([a-zA-Z\- ]{2,60})",
        re.IGNORECASE,
    )),
    ("x_defined_as_y", re.compile(
        r"\b([a-zA-Z\- ]{1,40})\s+is\s+defined\s+as\s+([a-zA-Z\- ]{2,80})",
        re.IGNORECASE,
    )),
    ("we_call_this_x", re.compile(
        r"\bwe\s+call\s+(?:this|it|these)\s+([a-zA-Z\- ]{1,40})",
        re.IGNORECASE,
    )),
    ("known_as_x", re.compile(
        r"\b(?:known|referred\s+to)\s+as\s+([a-zA-Z\- ]{1,40})",
        re.IGNORECASE,
    )),
    ("term_x_means", re.compile(
        r"\bthe\s+(?:term|word|concept)\s+([a-zA-Z\- ]{1,40})\s+(?:means|refers)",
        re.IGNORECASE,
    )),
    ("x_means_y", re.compile(
        r"\b([a-zA-Z\- ]{1,40})\s+means\s+([a-zA-Z\- ]{2,80})",
        re.IGNORECASE,
    )),
    ("by_x_we_mean", re.compile(
        r"\bby\s+([a-zA-Z\- ]{1,40})\s+we\s+mean",
        re.IGNORECASE,
    )),
    ("in_other_words", re.compile(
        r"\bin\s+other\s+words[,]?\s+([a-zA-Z\- ]{5,80})",
        re.IGNORECASE,
    )),
    ("that_is", re.compile(
        r"\bthat\s+is\s*[,]?\s+([a-zA-Z\- ]{5,80})",
        re.IGNORECASE,
    )),
    ("x_also_called_y", re.compile(
        r"\b([a-zA-Z\- ]{1,40})\s+is\s+also\s+called\s+([a-zA-Z\- ]{2,40})",
        re.IGNORECASE,
    )),
    ("x_occurs_when_y", re.compile(
        r"\b([a-zA-Z\- ]{1,40})\s+occurs?\s+when\s+([a-zA-Z\- ]{5,80})",
        re.IGNORECASE,
    )),
    ("formula_for_x_is_y", re.compile(
        r"\bthe\s+formula\s+for\s+([a-zA-Z\- ]{1,40})\s+is\s+([^.]{2,80})",
        re.IGNORECASE,
    )),
    ("acronym_stands_for", re.compile(
        r"\b([A-Z]{2,5})\s+stands\s+for\s+([A-Z][a-zA-Z\- ]{2,80})",
    )),
)

# Hearst false-positive guard: any X-is-a-Y match whose Y head is a generic
# noun ("problem", "thing") is dropped.
_HEARST_GENERIC_NOUNS: frozenset[str] = frozenset({
    "problem", "thing", "way", "reason", "lot", "kind", "type", "sort",
    "bunch", "idea", "point", "fact", "case", "matter", "concept",
})

_EQUATION_RE: re.Pattern[str] = re.compile(
    r"\b([A-Za-zα-ω][A-Za-z_α-ω0-9]{0,3})\s*=\s*"
    r"([A-Za-zα-ω0-9·\*\(\)/\^\+\-\s]{1,30}[A-Za-zα-ω0-9\)])"
)

_PROPER_NOUN_RE: re.Pattern[str] = re.compile(
    r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3}"
)

_YEAR_RE: re.Pattern[str] = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")

_CONTENT_WORD_RE: re.Pattern[str] = re.compile(r"[A-Za-z][A-Za-z0-9\-']*")

_FILLER_TOKENS: frozenset[str] = frozenset({
    "um", "uh", "er", "ah", "hmm", "okay", "ok", "yeah", "right",
    "alright", "just", "really", "actually", "basically", "literally",
    "kind", "sort", "like", "well", "now", "so",
})


# --------------------------------------------------------------------------- #
# Lazy spaCy + NetworkX loaders
# --------------------------------------------------------------------------- #


_NLP_LOCK = threading.Lock()
_NLP: Any | None = None
_NLP_TRIED = False
_NLP_MODEL_NAME = "en_core_web_lg"


def _get_nlp() -> Any | None:
    """Lazy-load spaCy ``en_core_web_lg``. Returns None if unavailable."""
    global _NLP, _NLP_TRIED
    if _NLP_TRIED:
        return _NLP
    with _NLP_LOCK:
        if _NLP_TRIED:
            return _NLP
        _NLP_TRIED = True
        try:
            import spacy  # type: ignore
        except Exception:
            logger.info("spaCy not installed; using regex NER fallback")
            return None
        try:
            _NLP = spacy.load(_NLP_MODEL_NAME, disable=["parser"])
        except Exception:
            logger.info("spaCy %s not installed; using regex NER fallback", _NLP_MODEL_NAME)
            _NLP = None
        return _NLP


def _get_networkx() -> Any | None:
    try:
        import networkx as nx  # type: ignore
        return nx
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Anchor-matching helpers (lemma tokens, not substring)
# --------------------------------------------------------------------------- #


def _suffix_strip(tok: str) -> str:
    """Light suffix stripping. Handles -ing / -ies / -ed / -es / -s and a
    trailing -e on tokens of length > 4 so that ``torque`` / ``torques`` /
    ``torquing`` all collapse to the same stem ``torqu``.

    Length guards keep short common tokens (``use``, ``see``, ``ride``) intact.
    """
    out = tok
    for suf in ("ing", "ies", "ed", "es", "s"):
        if len(out) > len(suf) + 2 and out.endswith(suf):
            out = out[: -len(suf)]
            if suf == "ies":
                out += "y"
            break
    if len(out) > 4 and out.endswith("e"):
        out = out[:-1]
    return out


def tokens_for_match(text: str, nlp: Any | None = None) -> list[str]:
    """Lemmatized lowercase content tokens used for anchor matching.

    With spaCy: ``tok.lemma_`` filtered to alpha non-stop tokens.
    Without spaCy: regex tokenize + lightweight suffix stripping.
    """
    if nlp is not None:
        try:
            return [
                tok.lemma_.lower()
                for tok in nlp(text)
                if tok.is_alpha and not tok.is_stop
            ]
        except Exception:
            pass
    raw = _CONTENT_WORD_RE.findall(text)
    return [_suffix_strip(t.lower()) for t in raw if len(t) > 1]


def match_anchors_in_text(
    text: str,
    anchors: Iterable[str],
    nlp: Any | None = None,
) -> dict[str, int]:
    """Return ``{anchor: count_in_text}`` using lemma-token n-gram match."""
    text_tokens = tokens_for_match(text, nlp)
    if not text_tokens:
        return {}
    counts: dict[str, int] = {}
    n_text = len(text_tokens)
    for anchor in anchors:
        a_tokens = tokens_for_match(anchor, nlp)
        if not a_tokens:
            continue
        n = len(a_tokens)
        if n > n_text:
            continue
        hits = sum(
            1
            for i in range(n_text - n + 1)
            if text_tokens[i : i + n] == a_tokens
        )
        if hits:
            counts[anchor] = hits
    return counts


# --------------------------------------------------------------------------- #
# Per-segment helpers
# --------------------------------------------------------------------------- #


def _word_count(text: str) -> int:
    return sum(1 for _ in _CONTENT_WORD_RE.finditer(text or ""))


def _content_words(text: str) -> list[str]:
    """Lowercase content tokens minus fillers — analogue of clip_boundary._extract_content_words."""
    toks = [m.group(0).lower() for m in _CONTENT_WORD_RE.finditer(text or "")]
    return [t for t in toks if t not in _FILLER_TOKENS and len(t) > 1]


def _lexical_density(text: str) -> float:
    total = _word_count(text)
    if not total:
        return 0.0
    return min(1.0, len(_content_words(text)) / float(total))


def _discourse_marker_count(text: str) -> int:
    return sum(1 for _ in _DISCOURSE_MARKER_RE.finditer(text or ""))


def _hearst_hits(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for name, pat in _HEARST_PATTERNS:
        for m in pat.finditer(text or ""):
            span = m.group(0).strip()
            if name == "x_is_a_y":
                # Drop generic-Y false positives.
                y_head = m.group(2).strip().split()[0].lower() if m.group(2) else ""
                if y_head in _HEARST_GENERIC_NOUNS:
                    continue
            out.append((name, span))
    return out


def _ner_anchors(text: str, nlp: Any | None) -> dict[str, list[str]]:
    """Return ``{label: [span_text, ...]}`` covering PERSON/GPE/ORG/DATE/EVENT/EQUATION."""
    out: dict[str, list[str]] = {}
    if nlp is not None:
        try:
            doc = nlp(text)
            wanted = {"PERSON", "GPE", "ORG", "DATE", "EVENT", "NORP", "LOC", "WORK_OF_ART", "LAW"}
            for ent in doc.ents:
                if ent.label_ in wanted:
                    out.setdefault(ent.label_, []).append(ent.text.strip())
        except Exception:
            logger.debug("spaCy NER raised; falling back to regex", exc_info=True)
    if not out:
        for m in _PROPER_NOUN_RE.finditer(text):
            out.setdefault("PROPN", []).append(m.group(0))
        for m in _YEAR_RE.finditer(text):
            out.setdefault("DATE", []).append(m.group(0))
    eq_hits = [m.group(0) for m in _EQUATION_RE.finditer(text)]
    if eq_hits:
        out["EQUATION"] = eq_hits
    return out


def _ner_anchor_total(ner: dict[str, list[str]]) -> int:
    return sum(len(v) for v in ner.values())


def _normalize(x: float, p95: float) -> float:
    if p95 <= 0:
        return 0.0
    return min(x / p95, 1.0)


def _instructional_density(
    *,
    discourse_density: float,
    hearst_count: int,
    lexical_density: float,
    ner_count: int,
) -> float:
    return (
        0.40 * _normalize(discourse_density, p95=8.0)
        + 0.30 * _normalize(float(hearst_count), p95=3.0)
        + 0.20 * lexical_density
        + 0.10 * _normalize(float(ner_count), p95=5.0)
    )


# --------------------------------------------------------------------------- #
# TF-IDF (per video)
# --------------------------------------------------------------------------- #


def _compute_tfidf(segments: list[SegmentMatch]) -> tuple[list[list[tuple[str, float]]], dict[str, float]]:
    """Return (per-segment top-5 terms, per-video IDF dict).

    Tries sklearn's TfidfVectorizer; falls back to a simple stdlib implementation
    so the module remains importable without sklearn.
    """
    texts = [(seg.text or "").strip() for seg in segments]
    n = len(texts)
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    except Exception:
        return _tfidf_fallback(texts)
    if n == 0:
        return [], {}
    try:
        vec = TfidfVectorizer(
            min_df=1,
            max_df=0.8 if n >= 5 else 1.0,
            ngram_range=(1, 2),
            stop_words="english",
        )
        matrix = vec.fit_transform(texts)
    except Exception:
        return _tfidf_fallback(texts)

    vocab = vec.vocabulary_  # term -> col
    idf_arr = vec.idf_
    idf = {term: float(idf_arr[col]) for term, col in vocab.items()}

    per_seg: list[list[tuple[str, float]]] = []
    inv_vocab = {col: term for term, col in vocab.items()}
    for row_idx in range(n):
        row = matrix.getrow(row_idx)
        coo = row.tocoo()
        pairs = sorted(
            ((float(v), int(c)) for v, c in zip(coo.data, coo.col)),
            reverse=True,
        )[:5]
        per_seg.append([(inv_vocab[c], v) for v, c in pairs])
    return per_seg, idf


def _tfidf_fallback(
    texts: list[str],
) -> tuple[list[list[tuple[str, float]]], dict[str, float]]:
    """Stdlib-only TF-IDF using lowercased content tokens."""
    n = len(texts)
    if n == 0:
        return [], {}
    docs_tokens = [_content_words(t) for t in texts]
    df: dict[str, int] = {}
    for toks in docs_tokens:
        for term in set(toks):
            df[term] = df.get(term, 0) + 1
    idf: dict[str, float] = {
        term: math.log((n + 1) / (1 + cnt)) + 1.0 for term, cnt in df.items()
    }
    per_seg: list[list[tuple[str, float]]] = []
    for toks in docs_tokens:
        if not toks:
            per_seg.append([])
            continue
        tf: dict[str, int] = {}
        for tok in toks:
            tf[tok] = tf.get(tok, 0) + 1
        scored = [
            (term, tf[term] / float(len(toks)) * idf.get(term, 1.0))
            for term in tf
        ]
        scored.sort(key=lambda p: p[1], reverse=True)
        per_seg.append(scored[:5])
    return per_seg, idf


# --------------------------------------------------------------------------- #
# Centrality (TextRank-ish PageRank on cosine graph)
# --------------------------------------------------------------------------- #


def _compute_centrality(
    segments: list[SegmentMatch],
    structural_penalties: list[float],
    embedder: Any | None,
    *,
    video_duration: float,
    conn: Any,
) -> list[float]:
    """Return per-segment centrality in [0, 1].

    Edge-discount: segments in first 10% / last 10% get centrality=0 (intros
    and outros use repeated framing language and would over-rank).

    Structural-penalty cap: multiply by ``max(0, 1 - 2 * penalty)`` so segments
    flagged as sponsor/intro/outro can't ride centrality back into contention.

    Falls back to uniform 0.5 when ``n_segments <= 6`` (graph too small to
    distinguish anything).
    """
    n = len(segments)
    if n == 0:
        return []
    if n <= 6:
        return [0.5] * n

    keep_mask: list[bool] = []
    for seg in segments:
        if video_duration > 0:
            frac = max(0.0, min(1.0, seg.t_start / video_duration))
            keep_mask.append(0.10 <= frac <= 0.90)
        else:
            keep_mask.append(True)

    interior_indices = [i for i, keep in enumerate(keep_mask) if keep]
    if len(interior_indices) < 4:
        return [0.5 if keep else 0.0 for keep in keep_mask]

    if embedder is None:
        return [0.5 if keep else 0.0 for keep in keep_mask]
    try:
        import numpy as np
        vectors = embedder.embed_texts(conn, [segments[i].text for i in interior_indices])
    except Exception:
        logger.debug("centrality: embedder failed; falling back to uniform", exc_info=True)
        return [0.5 if keep else 0.0 for keep in keep_mask]

    if vectors is None or len(vectors) != len(interior_indices):
        return [0.5 if keep else 0.0 for keep in keep_mask]

    nx = _get_networkx()
    sims = vectors @ vectors.T
    threshold = 0.35

    if nx is not None:
        try:
            graph = nx.Graph()
            for local_i in range(len(interior_indices)):
                graph.add_node(local_i)
            for i in range(len(interior_indices)):
                for j in range(i + 1, len(interior_indices)):
                    w = float(sims[i, j])
                    if w >= threshold:
                        graph.add_edge(i, j, weight=w)
            ranks = nx.pagerank(graph, weight="weight") if graph.number_of_edges() else {i: 1.0 / len(interior_indices) for i in graph.nodes()}
        except Exception:
            ranks = _power_iteration_pagerank(sims, threshold)
    else:
        ranks = _power_iteration_pagerank(sims, threshold)

    raw = [ranks.get(i, 0.0) for i in range(len(interior_indices))]
    if not raw:
        return [0.0] * n
    max_r = max(raw)
    norm = [r / max_r for r in raw] if max_r > 0 else [0.0] * len(raw)

    out: list[float] = [0.0] * n
    for local_i, global_i in enumerate(interior_indices):
        cap = max(0.0, 1.0 - 2.0 * structural_penalties[global_i])
        out[global_i] = norm[local_i] * cap
    return out


def _power_iteration_pagerank(
    sims: Any,
    threshold: float,
    *,
    damping: float = 0.85,
    iterations: int = 30,
    tol: float = 1e-4,
) -> dict[int, float]:
    """Simple power-iteration PageRank on a cosine graph (no nx dep)."""
    import numpy as np
    n = sims.shape[0]
    adj = (sims >= threshold).astype("float32")
    np.fill_diagonal(adj, 0.0)
    deg = adj.sum(axis=1)
    deg[deg == 0] = 1.0
    transition = adj / deg[:, None]
    rank = np.full(n, 1.0 / n, dtype="float32")
    teleport = (1.0 - damping) / n
    for _ in range(iterations):
        new = teleport + damping * (transition.T @ rank)
        if float(np.abs(new - rank).sum()) < tol:
            rank = new
            break
        rank = new
    return {i: float(rank[i]) for i in range(n)}


# --------------------------------------------------------------------------- #
# Public dataclasses + entry point
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SegmentFeatures:
    chunk_index: int
    tfidf_top: tuple[tuple[str, float], ...]
    discourse_marker_count: int
    discourse_marker_density: float
    hearst_hits: tuple[tuple[str, str], ...]
    ner_anchors: dict[str, tuple[str, ...]]
    instructional_density: float
    structural_label: str
    structural_penalty: float
    centrality: float
    word_count: int


@dataclass(frozen=True)
class VideoFeatureBundle:
    segments: tuple[SegmentFeatures, ...]
    idf: dict[str, float]


def extract_features(
    segments: list[SegmentMatch],
    embedder: Any | None,
    *,
    nlp: Any | None = None,
    video_duration: float,
    conn: Any = None,
) -> VideoFeatureBundle:
    """Compute per-segment features for one video's segments.

    ``embedder`` should expose ``embed_texts(conn, list[str]) -> np.ndarray``.
    Pass ``None`` to skip centrality (uniform 0.5).

    ``nlp`` should be a spaCy ``Language`` or ``None`` to use regex fallback.
    """
    if not segments:
        return VideoFeatureBundle(segments=(), idf={})

    if nlp is None:
        nlp = _get_nlp()

    tfidf_per_seg, idf = _compute_tfidf(segments)

    structural_penalties: list[float] = []
    structural_labels: list[str] = []
    for seg in segments:
        label = classify_passage(
            seg.text or "",
            t_start=float(seg.t_start),
            video_duration=float(video_duration) if video_duration else None,
        )
        structural_labels.append(label.name)
        structural_penalties.append(label_penalty(label))

    centralities = _compute_centrality(
        segments,
        structural_penalties,
        embedder,
        video_duration=float(video_duration) if video_duration else 0.0,
        conn=conn,
    )

    out: list[SegmentFeatures] = []
    for i, seg in enumerate(segments):
        text = seg.text or ""
        wc = _word_count(text)
        marker_count = _discourse_marker_count(text)
        density = (marker_count / max(wc, 1)) * 100.0
        hearst = _hearst_hits(text)
        ner = _ner_anchors(text, nlp)
        instr = _instructional_density(
            discourse_density=density,
            hearst_count=len(hearst),
            lexical_density=_lexical_density(text),
            ner_count=_ner_anchor_total(ner),
        )
        out.append(
            SegmentFeatures(
                chunk_index=int(seg.chunk_index),
                tfidf_top=tuple(tfidf_per_seg[i]) if i < len(tfidf_per_seg) else (),
                discourse_marker_count=marker_count,
                discourse_marker_density=density,
                hearst_hits=tuple(hearst),
                ner_anchors={k: tuple(v) for k, v in ner.items()},
                instructional_density=instr,
                structural_label=structural_labels[i],
                structural_penalty=structural_penalties[i],
                centrality=centralities[i] if i < len(centralities) else 0.0,
                word_count=wc,
            )
        )
    return VideoFeatureBundle(segments=tuple(out), idf=idf)


__all__ = [
    "SegmentFeatures",
    "VideoFeatureBundle",
    "extract_features",
    "match_anchors_in_text",
    "tokens_for_match",
]
