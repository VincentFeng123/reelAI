"""Subject-agnostic importance ranker.

Combines per-segment features from ``segment_features.VideoFeatureBundle`` with
``query_intent.QueryIntent`` to produce ``RankedSegment``s ordered by importance,
then ``select_clips`` returns the intent-appropriate count using
cluster-diversified greedy selection.

Importance formula:

    importance = wc · centrality
               + wd · instructional_density
               + wa · anchor_density
               + wq · query_relevance
               + wt · topic_concentration
               - ws · structural_penalty

Weights vary by intent; see ``_WEIGHTS_BY_INTENT``.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any

from .query_intent import QueryIntent
from .segment_features import (
    SegmentFeatures,
    VideoFeatureBundle,
    match_anchors_in_text,
    tokens_for_match,
)
from .segmenter import SegmentMatch
from .structural_classifier import classify_passage, label_penalty

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Weights and thresholds
# --------------------------------------------------------------------------- #


_WEIGHTS_BY_INTENT: dict[str, dict[str, float]] = {
    "none":   {"wc": 0.30, "wd": 0.50, "wa": 0.00, "wq": 0.00, "wt": 0.00, "ws": 0.40},
    "broad":  {"wc": 0.20, "wd": 0.40, "wa": 0.25, "wq": 0.15, "wt": 0.00, "ws": 0.40},
    "medium": {"wc": 0.15, "wd": 0.35, "wa": 0.20, "wq": 0.25, "wt": 0.05, "ws": 0.40},
    "narrow": {"wc": 0.05, "wd": 0.20, "wa": 0.25, "wq": 0.30, "wt": 0.20, "ws": 0.40},
}

_TARGET_N_BY_INTENT: dict[str, int] = {
    "narrow": 2,
    "broad": 5,
    "medium": 3,
    "none": 3,
}

_HARD_SKIP_STRUCTURAL_LABELS = {"sponsor", "outro", "intro", "recap"}

_TEMPORAL_OVERLAP_THRESHOLD = 0.3

# Cross-encoder gate tuning
_NARROW_RERANK_TOP_K = 50
_AMBIGUOUS_GAP_THRESHOLD = 0.05

_EQUATION_RE: re.Pattern[str] = re.compile(
    r"[A-Za-zα-ω][A-Za-z_α-ω0-9]{0,3}\s*=\s*[A-Za-zα-ω0-9·\*\(\)/\^\+\-]"
)
_PROPER_NOUN_MULTI_RE: re.Pattern[str] = re.compile(
    r"\b[A-Z][a-zA-Z]+(?:\s+(?:of|the|de|and|von|la)\s+[A-Z][a-zA-Z]+|"
    r"\s+[A-Z][a-zA-Z]+|-[A-Z][a-zA-Z]+)+"
)
_SCHOOL_SUBJECTS: frozenset[str] = frozenset({
    "physics", "biology", "chemistry", "history", "spanish", "english",
    "calculus", "algebra", "geometry", "statistics", "psychology",
    "sociology", "economics", "government", "literature", "geography",
    "anatomy", "civics", "trigonometry", "precalculus",
})


# --------------------------------------------------------------------------- #
# Cross-encoder lazy loader
# --------------------------------------------------------------------------- #


_CROSS_ENCODER_LOCK = threading.Lock()
_CROSS_ENCODER: Any | None = None
_CROSS_ENCODER_TRIED = False
_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L12-v2"


def get_cross_encoder() -> Any | None:
    """Lazy-load the cross-encoder. Returns None if unavailable."""
    global _CROSS_ENCODER, _CROSS_ENCODER_TRIED
    if os.getenv("REELS_CROSS_ENCODER_ENABLED", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return None
    if _CROSS_ENCODER_TRIED:
        return _CROSS_ENCODER
    if os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
        _CROSS_ENCODER_TRIED = True
        return None
    with _CROSS_ENCODER_LOCK:
        if _CROSS_ENCODER_TRIED:
            return _CROSS_ENCODER
        _CROSS_ENCODER_TRIED = True
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except Exception:
            logger.info("sentence-transformers not installed; cross-encoder disabled")
            return None
        try:
            _CROSS_ENCODER = CrossEncoder(_CROSS_ENCODER_MODEL, device="cpu")
        except Exception:
            logger.info("could not load %s; cross-encoder disabled", _CROSS_ENCODER_MODEL)
            _CROSS_ENCODER = None
        return _CROSS_ENCODER


# --------------------------------------------------------------------------- #
# Public dataclass
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RankedSegment:
    match: SegmentMatch
    importance: float
    components: dict[str, float]
    cluster_id: str
    features: SegmentFeatures


# --------------------------------------------------------------------------- #
# Component scorers
# --------------------------------------------------------------------------- #


def _anchor_density(
    seg: SegmentMatch,
    feats: SegmentFeatures,
    anchors: tuple[str, ...],
    nlp: Any | None,
) -> float:
    if not anchors:
        return 0.0
    counts = match_anchors_in_text(seg.text or "", anchors, nlp)
    total = sum(counts.values())
    if not total:
        return 0.0
    per_100 = total / max(feats.word_count / 100.0, 1.0)
    return min(1.0, per_100)


def _bi_encoder_relevance(
    seg_texts: list[str],
    query: str,
    embedder: Any | None,
    conn: Any,
) -> list[float]:
    """Return per-segment cosine vs the query (in [-1, 1] clipped to [0, 1])."""
    if embedder is None or not query.strip():
        return [0.0] * len(seg_texts)
    try:
        import numpy as np
        seg_vecs = embedder.embed_texts(conn, seg_texts)
        q_vec = embedder.embed_texts(conn, [query])[0]
    except Exception:
        logger.debug("bi-encoder relevance: embedder failed", exc_info=True)
        return [0.0] * len(seg_texts)
    if seg_vecs is None or len(seg_vecs) != len(seg_texts):
        return [0.0] * len(seg_texts)
    sims = (seg_vecs @ q_vec).tolist()
    return [max(0.0, min(1.0, float(s))) for s in sims]


def _should_rerank(intent: QueryIntent, bi_scores: list[float]) -> bool:
    """Cross-encoder gate. Fires on any of:

    1. Equation in query.
    2. ≤3 content tokens AND no school-subject token.
    3. Multi-word capitalized proper noun.
    4. Top bi-encoder scores ambiguous (top - 5th < 0.05).
    """
    raw = intent.raw_query or ""
    if _EQUATION_RE.search(raw):
        return True
    if _PROPER_NOUN_MULTI_RE.search(raw):
        return True
    tokens = [a.lower() for a in intent.anchors]
    if 1 <= len(tokens) <= 3 and not any(t in _SCHOOL_SUBJECTS for t in tokens):
        return True
    if len(bi_scores) >= 5:
        sorted_scores = sorted(bi_scores, reverse=True)
        if (sorted_scores[0] - sorted_scores[4]) < _AMBIGUOUS_GAP_THRESHOLD:
            return True
    return False


def _cross_encoder_relevance(
    seg_texts: list[str],
    query: str,
    cross_encoder: Any,
    bi_scores: list[float],
) -> list[float]:
    """Rerank top-K candidates with the cross-encoder; blend 0.4·bi + 0.6·ce."""
    if not seg_texts or cross_encoder is None:
        return bi_scores
    n = len(seg_texts)
    indexed = sorted(range(n), key=lambda i: bi_scores[i], reverse=True)
    top_k = indexed[: min(_NARROW_RERANK_TOP_K, n)]
    pairs = [(query, seg_texts[i]) for i in top_k]
    try:
        ce_raw = cross_encoder.predict(pairs)
    except Exception:
        logger.debug("cross-encoder predict failed; falling back to bi", exc_info=True)
        return bi_scores
    if hasattr(ce_raw, "tolist"):
        ce_raw = ce_raw.tolist()
    ce_norm = _sigmoid_list([float(x) for x in ce_raw])
    out = list(bi_scores)
    for local_i, global_i in enumerate(top_k):
        out[global_i] = 0.4 * bi_scores[global_i] + 0.6 * ce_norm[local_i]
    return out


def _sigmoid_list(xs: list[float]) -> list[float]:
    import math
    return [1.0 / (1.0 + math.exp(-x)) for x in xs]


def _topic_concentration(
    seg: SegmentMatch,
    feats: SegmentFeatures,
    anchors: tuple[str, ...],
    idf: dict[str, float],
    nlp: Any | None,
) -> float:
    """IDF-weighted anchor coverage × discourse density factor.

    A casual mention of a low-IDF anchor (the word "force" everywhere in a
    physics-of-forces video) does NOT count — the per-sentence anchor weight
    must clear half the max IDF before it qualifies as a mention. Coverage is
    the fraction of segment time within ±15s windows of qualifying mentions.
    """
    if not anchors or feats.word_count == 0:
        return 0.0
    text = seg.text or ""
    if not text.strip():
        return 0.0

    seg_dur = max(0.001, float(seg.t_end) - float(seg.t_start))
    anchor_weight = {a: idf.get(a.lower(), 1.0) for a in anchors}
    max_w = max(anchor_weight.values(), default=1.0) or 1.0
    threshold = 0.5 * max_w

    sentences = _split_sentences_in_segment(text, seg.t_start, seg.t_end)
    if not sentences:
        return 0.0

    mention_times: list[float] = []
    for s_text, s_start in sentences:
        hits = match_anchors_in_text(s_text, anchors, nlp)
        weight = sum(anchor_weight.get(a, 1.0) * count for a, count in hits.items())
        if weight >= threshold:
            mention_times.append(s_start)
    if not mention_times:
        return 0.0

    intervals = _union_clipped(
        [(t - 15.0, t + 15.0) for t in mention_times],
        float(seg.t_start),
        float(seg.t_end),
    )
    coverage = sum(end - start for start, end in intervals) / seg_dur
    coverage = max(0.0, min(1.0, coverage))

    marker_density = feats.discourse_marker_density
    discourse_factor = min(marker_density / 5.0, 1.0)
    return coverage * (0.5 + 0.5 * discourse_factor)


def _split_sentences_in_segment(
    text: str,
    t_start: float,
    t_end: float,
) -> list[tuple[str, float]]:
    """Cheap regex split — assigns each sentence a proportional t_start."""
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return []
    span = max(0.001, float(t_end) - float(t_start))
    step = span / float(len(parts))
    return [(p, float(t_start) + i * step) for i, p in enumerate(parts)]


def _union_clipped(
    intervals: list[tuple[float, float]],
    lo: float,
    hi: float,
) -> list[tuple[float, float]]:
    clipped = [
        (max(lo, s), min(hi, e))
        for s, e in intervals
        if min(hi, e) > max(lo, s)
    ]
    if not clipped:
        return []
    clipped.sort()
    out: list[tuple[float, float]] = [clipped[0]]
    for s, e in clipped[1:]:
        last_s, last_e = out[-1]
        if s <= last_e:
            out[-1] = (last_s, max(last_e, e))
        else:
            out.append((s, e))
    return out


# --------------------------------------------------------------------------- #
# Cluster id (top-1 TF-IDF tier — anchor-family clustering deferred)
# --------------------------------------------------------------------------- #


def _cluster_id_for_segment(feats: SegmentFeatures) -> str:
    """Top-1 TF-IDF term (third-tier fallback in the plan).

    Anchor-family clustering is left for a follow-up — top-1 already gives
    sane diversification and anchor-family adds a second embedder pass.
    """
    if not feats.tfidf_top:
        return f"_chunk_{feats.chunk_index}"
    return feats.tfidf_top[0][0]


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #


def rank_segments(
    segments: list[SegmentMatch],
    bundle: VideoFeatureBundle,
    intent: QueryIntent,
    *,
    embedder: Any | None = None,
    cross_encoder: Any | None = None,
    conn: Any = None,
    nlp: Any | None = None,
) -> list[RankedSegment]:
    """Score and order segments by importance under the given intent.

    Returns sorted by importance descending. Selection (target_n, MMR-style
    diversification) is in ``select_clips``.
    """
    if not segments:
        return []
    feats_by_chunk = {f.chunk_index: f for f in bundle.segments}
    weights = _WEIGHTS_BY_INTENT.get(intent.type, _WEIGHTS_BY_INTENT["medium"])

    seg_texts = [(seg.text or "") for seg in segments]
    if intent.type != "none" and intent.raw_query:
        bi_scores = _bi_encoder_relevance(seg_texts, intent.raw_query, embedder, conn)
        if intent.type == "narrow" and _should_rerank(intent, bi_scores):
            ce = cross_encoder if cross_encoder is not None else get_cross_encoder()
            if ce is not None:
                bi_scores = _cross_encoder_relevance(seg_texts, intent.raw_query, ce, bi_scores)
    else:
        bi_scores = [0.0] * len(segments)

    ranked: list[RankedSegment] = []
    for i, seg in enumerate(segments):
        feats = feats_by_chunk.get(seg.chunk_index)
        if feats is None:
            continue
        anchor_dens = _anchor_density(seg, feats, intent.anchors, nlp)
        topic_conc = (
            _topic_concentration(seg, feats, intent.anchors, bundle.idf, nlp)
            if intent.type == "narrow"
            else 0.0
        )
        components = {
            "centrality": float(feats.centrality),
            "instructional_density": float(feats.instructional_density),
            "anchor_density": float(anchor_dens),
            "query_relevance": float(bi_scores[i]),
            "topic_concentration": float(topic_conc),
            "structural_penalty": float(feats.structural_penalty),
        }
        importance = (
            weights["wc"] * components["centrality"]
            + weights["wd"] * components["instructional_density"]
            + weights["wa"] * components["anchor_density"]
            + weights["wq"] * components["query_relevance"]
            + weights["wt"] * components["topic_concentration"]
            - weights["ws"] * components["structural_penalty"]
        )
        importance = max(0.0, min(1.0, importance))
        ranked.append(
            RankedSegment(
                match=seg,
                importance=importance,
                components=components,
                cluster_id=_cluster_id_for_segment(feats),
                features=feats,
            )
        )
    ranked.sort(key=lambda r: r.importance, reverse=True)
    return ranked


def _temporal_overlap_ratio(a: SegmentMatch, b: SegmentMatch) -> float:
    """Overlap as a fraction of the shorter span."""
    lo = max(float(a.t_start), float(b.t_start))
    hi = min(float(a.t_end), float(b.t_end))
    overlap = max(0.0, hi - lo)
    if overlap <= 0:
        return 0.0
    short = max(0.001, min(float(a.t_end) - float(a.t_start), float(b.t_end) - float(b.t_start)))
    return overlap / short


def select_clips(
    ranked: list[RankedSegment],
    intent: QueryIntent,
    *,
    target_n: int | None = None,
) -> list[RankedSegment]:
    """Cluster-diversified greedy selection.

    Pass 1: prefer one segment per distinct cluster id.
    Pass 2: if under-budget, relax cluster-uniqueness, keep temporal suppression.
    Floor: at least 1 (narrow) or 3 (broad/medium/none) when the pool allows.

    Sponsor / outro / intro / recap segments are excluded entirely.
    """
    if not ranked:
        return []

    n = target_n if target_n is not None else _TARGET_N_BY_INTENT.get(intent.type, 3)
    if intent.type == "narrow":
        n = min(2, n)
    elif intent.type == "broad":
        n = max(3, min(5, n))

    pool = [
        r for r in ranked
        if r.features.structural_label not in _HARD_SKIP_STRUCTURAL_LABELS
    ]
    if not pool:
        pool = list(ranked)

    selected: list[RankedSegment] = []
    selected_clusters: set[str] = set()

    for cand in pool:
        if len(selected) >= n:
            break
        if cand.cluster_id in selected_clusters:
            continue
        if any(_temporal_overlap_ratio(cand.match, s.match) > _TEMPORAL_OVERLAP_THRESHOLD
               for s in selected):
            continue
        selected.append(cand)
        selected_clusters.add(cand.cluster_id)

    if len(selected) < n:
        seen_ids = {id(r) for r in selected}
        for cand in pool:
            if len(selected) >= n:
                break
            if id(cand) in seen_ids:
                continue
            if any(_temporal_overlap_ratio(cand.match, s.match) > _TEMPORAL_OVERLAP_THRESHOLD
                   for s in selected):
                continue
            selected.append(cand)
            seen_ids.add(id(cand))

    floor = 1 if intent.type == "narrow" else 3
    if len(selected) < floor:
        seen_ids = {id(r) for r in selected}
        for cand in pool:
            if id(cand) in seen_ids:
                continue
            selected.append(cand)
            seen_ids.add(id(cand))
            if len(selected) >= floor:
                break

    selected.sort(key=lambda r: r.match.t_start)
    return selected


__all__ = [
    "RankedSegment",
    "rank_segments",
    "select_clips",
    "get_cross_encoder",
]
