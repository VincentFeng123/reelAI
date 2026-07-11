"""Stage: hierarchical content map (spec §2).

Builds a 3-level hierarchy (chapter → topic → subtopic). Topics — the unit-bearing
partition — come either from deterministic embedding-based segmentation (engine="treeseg")
or from an LLM structuring pass over the sentence-indexed transcript (engine="llm").
The treeseg path bisects sentence embeddings (Ward-style scatter gain) and labels the
resulting fixed segments with a cheap LLM pass; the legacy LLM path chunks long videos
and asks the model to return sentence-index ranges. Treeseg failure falls back to the LLM
engine (engine="llm-fallback") — never crashes the job.
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from typing import Callable, Optional

from pydantic import BaseModel, Field

from ... import config
from ..select import render_sentences
from ..sentences import Sentence
from .models import ContentMap, ContentNode
from .segment import chunk_sentences, discourse_hits, gap_before
from .treeseg import boundary_priors, chapter_cut, divisive_segments, embed_sentences

ProgressCb = Optional[Callable[[float, str], None]]


class TopicLLM(BaseModel):
    title: str = ""
    summary: str = ""
    sentence_start: int = 0
    sentence_end: int = 0
    keywords: list[str] = Field(default_factory=list)


class ContentMapLLM(BaseModel):
    topics: list[TopicLLM] = Field(default_factory=list)


CM_SYSTEM = (
    "You are a content architect. Given a timestamped, index-numbered transcript of a video "
    "of unknown genre, divide it into a sequence of contiguous TOPICS — the major segments a "
    "viewer would recognize as 'this part is about X'. Each topic is one coherent subject. "
    "Return each topic's first and last sentence INDEX (from the [index] tags), a short specific "
    "title, a one-line summary, and a few keywords. Topics must be in order, non-overlapping, and "
    "together cover every sentence. Prefer several focused topics over one giant one; do not split "
    "a single explanation or worked example across topics. Output only the structured result."
)


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


def _normalize_topics(raw: list[tuple[int, int, TopicLLM]], n: int) -> list[tuple[int, int, TopicLLM]]:
    """Sort, clamp, and repair to a gapless non-overlapping partition of [0, n-1]."""
    raw = sorted((r for r in raw if r[1] >= r[0]), key=lambda r: r[0])
    if not raw:
        return [(0, n - 1, TopicLLM(title="Full video", sentence_start=0, sentence_end=n - 1))]
    out: list[tuple[int, int, TopicLLM]] = []
    cursor = 0
    for s0, s1, t in raw:
        s0 = max(s0, cursor)
        s1 = max(s1, s0)
        if s0 > n - 1:
            break
        s1 = min(s1, n - 1)
        out.append((s0, s1, t))
        cursor = s1 + 1
        if cursor > n - 1:
            break
    if not out:
        out = [(0, n - 1, raw[0][2])]
    # extend first→0 and last→n-1 so coverage is total
    f0, f1, ft = out[0]
    out[0] = (0, f1, ft)
    l0, l1, lt = out[-1]
    out[-1] = (l0, n - 1, lt)
    return out


def _group_chapters(topics: list[tuple[int, int, TopicLLM]], sentences: list[Sentence]) -> list[tuple[int, int]]:
    """Group consecutive topics into chapters, breaking at large inter-topic pauses or a size
    cap. Returns inclusive ``(first_topic_idx, last_topic_idx)`` ranges."""
    if not topics:
        return []
    groups: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(topics)):
        gap = float(sentences[topics[i][0]].start) - float(sentences[topics[i - 1][1]].end)
        if gap >= config.CHAPTER_GAP_S or (i - start) >= config.CHAPTER_MAX_TOPICS:
            groups.append((start, i - 1))
            start = i
    groups.append((start, len(topics) - 1))
    return groups


def _split_subtopics(s0: int, s1: int, sentences: list[Sentence]) -> list[tuple[int, int]]:
    """Split a topic's sentence range into finer subtopic spans at the strongest internal
    pause/discourse seams. Returns a single span (no subdivision) for short topics or when no
    internal seam stands out."""
    if (s1 - s0 + 1) < config.SUBTOPIC_MIN_SENTS:
        return [(s0, s1)]
    hits = discourse_hits(sentences)
    seams = [(gap_before(sentences, i) + (2.0 if i in hits else 0.0), i) for i in range(s0 + 1, s1 + 1)]
    seams = [(score, i) for score, i in seams if score > 0]
    if not seams:
        return [(s0, s1)]
    seams.sort(reverse=True)
    max_cuts = max(1, min(config.SUBTOPIC_MAX - 1, (s1 - s0 + 1) // config.SUBTOPIC_MIN_SENTS))
    cuts = sorted(i for _, i in seams[:max_cuts])
    spans, prev = [], s0
    for c in cuts:
        if c - prev >= 2:                              # avoid 1-sentence slivers
            spans.append((prev, c - 1))
            prev = c
    spans.append((prev, s1))
    return spans


def _build_content_map_llm(sentences: list[Sentence], settings: dict,
                            progress: ProgressCb = None) -> ContentMap:
    from ...llm import llm_json
    n = len(sentences)

    chunks = chunk_sentences(sentences)
    raw: list[tuple[int, int, TopicLLM]] = []
    for ci, (a, b) in enumerate(chunks):
        rendered = render_sentences(sentences[a:b + 1])
        user = (f"TRANSCRIPT (each line: [index] (start-end seconds) text), indices {a}–{b}:\n"
                f"{rendered}\n\nDivide indices {a}–{b} into contiguous topics.")
        try:
            res = llm_json(CM_SYSTEM, user, ContentMapLLM, temperature=0.1)
        except Exception:
            res = ContentMapLLM(topics=[TopicLLM(title="Segment", sentence_start=a, sentence_end=b)])
        for t in res.topics:
            s0 = _clamp(t.sentence_start, a, b)
            s1 = _clamp(t.sentence_end, a, b)
            raw.append((min(s0, s1), max(s0, s1), t))
        if progress:
            progress((ci + 1) / len(chunks), f"Mapping topics {ci + 1}/{len(chunks)}")

    topics = _normalize_topics(raw, n)
    chapters = _group_chapters(topics, sentences)

    video = ContentNode(node_id="video", level="video", title="",
                        start=sentences[0].start, end=sentences[-1].end, sentence_range=(0, n - 1))
    nodes: list[ContentNode] = [video]
    ti_global = 0
    for ci, (t0, t1) in enumerate(chapters):
        cid = f"c{ci}"
        cs0, cs1 = topics[t0][0], topics[t1][1]
        chapter = ContentNode(
            node_id=cid, level="chapter", parent_id="video",
            title=topics[t0][2].title or f"Chapter {ci + 1}",
            start=float(sentences[cs0].start), end=float(sentences[cs1].end), sentence_range=(cs0, cs1),
        )
        video.children_ids.append(cid)
        nodes.append(chapter)
        for ti in range(t0, t1 + 1):
            s0, s1, t = topics[ti]
            tid = f"{cid}.t{ti_global}"
            topic = ContentNode(
                node_id=tid, level="topic", parent_id=cid,
                title=t.title or f"Topic {ti_global + 1}", summary=t.summary,
                start=float(sentences[s0].start), end=float(sentences[s1].end),
                sentence_range=(s0, s1), keywords=t.keywords,
            )
            chapter.children_ids.append(tid)
            nodes.append(topic)
            subs = _split_subtopics(s0, s1, sentences)
            if len(subs) >= 2:                          # only annotate genuinely subdivided topics
                for si, (ss0, ss1) in enumerate(subs):
                    sid = f"{tid}.s{si}"
                    topic.children_ids.append(sid)
                    nodes.append(ContentNode(
                        node_id=sid, level="subtopic", parent_id=tid, title=f"{topic.title} · part {si + 1}",
                        start=float(sentences[ss0].start), end=float(sentences[ss1].end),
                        sentence_range=(ss0, ss1),
                    ))
            ti_global += 1
    return ContentMap(root_id="video", nodes=nodes)


# ── Treeseg labeling models + helpers ────────────────────────────────────────

class SegLabelLLM(BaseModel):
    index: int = 0
    title: str = ""
    summary: str = ""
    keywords: list[str] = Field(default_factory=list)


class SegLabelsLLM(BaseModel):
    labels: list[SegLabelLLM] = Field(default_factory=list)


LABEL_SYSTEM = (
    "You are labeling ALREADY-SEGMENTED topics of a video transcript. For each numbered "
    "segment excerpt, return its index with a short specific title, a one-line summary, and "
    "a few keywords. Do NOT merge, split, re-order, or invent segments. "
    "Output only the structured result."
)

_STOP = frozenset(
    "the a an and or of to in on for with is are was were be this that it its as at by "
    "we you i so now what when where which have has had will would can could about".split())


def _excerpt(sentences: list[Sentence], s0: int, s1: int, cap: int = 420) -> str:
    """First / middle / last sentences of the segment, bounded — enough to label, cheap to send."""
    idxs = sorted({max(s0, min(s1, i)) for i in (s0, s0 + 1, (s0 + s1) // 2, s1 - 1, s1)})
    return " … ".join(p for p in ((sentences[i].text or "").strip() for i in idxs) if p)[:cap]


def _fallback_title(sentences: list[Sentence], s0: int, s1: int, ti: int) -> str:
    """Deterministic keyword title when the label LLM fails: top content words of the segment."""
    text = " ".join((sentences[i].text or "") for i in range(s0, s1 + 1)).lower()
    words = [w for w in re.findall(r"[a-zA-Z][a-zA-Z'-]{3,}", text) if w not in _STOP]
    common = [w for w, _ in Counter(words).most_common(3)]
    return " ".join(common).title() or f"Topic {ti + 1}"


def _label_segments(segments: list[tuple[int, int]], sentences: list[Sentence]) -> list[TopicLLM]:
    """One cheap LLM pass per TREESEG_LABEL_BATCH segments. Labels align by index and are
    clamped to the fixed partition — they can never re-segment. Any failure degrades to
    deterministic keyword titles; the partition is untouched."""
    from ...llm import llm_json
    out = [TopicLLM(title="", sentence_start=s0, sentence_end=s1) for s0, s1 in segments]
    for base in range(0, len(segments), config.TREESEG_LABEL_BATCH):
        batch = segments[base: base + config.TREESEG_LABEL_BATCH]
        listing = "\n".join(
            f"[{base + j}] ({sentences[s0].start:.0f}-{sentences[s1].end:.0f}s) "
            f"{_excerpt(sentences, s0, s1)}"
            for j, (s0, s1) in enumerate(batch))
        try:
            res = llm_json(LABEL_SYSTEM,
                           f"SEGMENT EXCERPTS:\n{listing}\n\n"
                           f"Label every segment [{base}]–[{base + len(batch) - 1}].",
                           SegLabelsLLM, temperature=0.1)
        except Exception:
            res = SegLabelsLLM()
        for lab in res.labels:
            i = int(lab.index)
            if base <= i < base + len(batch) and lab.title.strip():
                out[i].title = lab.title.strip()
                out[i].summary = lab.summary.strip()
                out[i].keywords = [k for k in lab.keywords if k]
    for ti, ((s0, s1), t) in enumerate(zip(segments, out)):
        if not t.title:
            t.title = _fallback_title(sentences, s0, s1, ti)
    return out


# ── Treeseg assembly + builder ────────────────────────────────────────────────

def _assemble_treeseg(sentences: list[Sentence], segments: list[tuple[int, int]],
                      labels: list[TopicLLM], chapters: list[tuple[int, int]]) -> ContentMap:
    n = len(sentences)
    video = ContentNode(node_id="video", level="video", title="",
                        start=sentences[0].start, end=sentences[-1].end, sentence_range=(0, n - 1))
    nodes: list[ContentNode] = [video]
    for ci, (t0, t1) in enumerate(chapters):
        cid = f"c{ci}"
        cs0, cs1 = segments[t0][0], segments[t1][1]
        chapter = ContentNode(
            node_id=cid, level="chapter", parent_id="video",
            title=labels[t0].title or f"Chapter {ci + 1}",
            start=float(sentences[cs0].start), end=float(sentences[cs1].end),
            sentence_range=(cs0, cs1))
        video.children_ids.append(cid)
        nodes.append(chapter)
        for ti in range(t0, t1 + 1):
            s0, s1 = segments[ti]
            t = labels[ti]
            tid = f"{cid}.t{ti}"
            topic = ContentNode(
                node_id=tid, level="topic", parent_id=cid,
                title=t.title or f"Topic {ti + 1}", summary=t.summary,
                start=float(sentences[s0].start), end=float(sentences[s1].end),
                sentence_range=(s0, s1), keywords=t.keywords)
            chapter.children_ids.append(tid)
            nodes.append(topic)
    return ContentMap(root_id="video", nodes=nodes, engine="treeseg")


def _build_content_map_treeseg(sentences: list[Sentence], settings: dict,
                               progress: ProgressCb = None) -> ContentMap:
    n = len(sentences)
    duration = float(sentences[-1].end) - float(sentences[0].start)
    k = int(round(duration / config.TREESEG_TARGET_TOPIC_SEC))
    k = max(config.TREESEG_MIN_TOPICS, min(config.TREESEG_MAX_TOPICS, k))
    if n < 2 * config.TREESEG_MIN_TOPIC_SENTS or k <= 1:
        segments, split_order = [(0, n - 1)], []          # too small to segment — one topic
    else:
        emb = embed_sentences(sentences)                   # raises → caller falls back to legacy
        priors = boundary_priors(sentences, config.TREESEG_PAUSE_PRIOR)
        segments, split_order = divisive_segments(
            emb, target_k=k, min_size=config.TREESEG_MIN_TOPIC_SENTS,
            coherence_floor=config.TREESEG_COHERENCE_FLOOR, priors=priors)
    if progress:
        progress(0.6, f"Segmented into {len(segments)} topics")
    labels = _label_segments(segments, sentences)
    chapters = chapter_cut(split_order, segments, config.CHAPTER_MAX_TOPICS)
    if progress:
        progress(1.0, f"Labeled {len(segments)} topics")
    return _assemble_treeseg(sentences, segments, labels, chapters)


# ── Public dispatcher (signature unchanged) ───────────────────────────────────

def build_content_map(sentences: list[Sentence], settings: dict,
                      progress: ProgressCb = None) -> ContentMap:
    """Topic partition + labels. Engine "treeseg" (default): deterministic embedding
    boundaries + LLM labels; "llm": the legacy per-chunk LLM pass. Treeseg failure
    degrades to the legacy engine (engine="llm-fallback") — never crashes the job."""
    if not sentences:
        return ContentMap(nodes=[ContentNode(node_id="video", level="video")])
    engine = str(settings.get("content_map_engine") or config.CONTENT_MAP_ENGINE)
    if engine == "treeseg":
        try:
            return _build_content_map_treeseg(sentences, settings, progress)
        except Exception as e:
            print(f"[content_map] treeseg failed ({e!r}); falling back to llm engine", file=sys.stderr)
            cm = _build_content_map_llm(sentences, settings, progress)
            cm.engine = "llm-fallback"
            return cm
    cm = _build_content_map_llm(sentences, settings, progress)
    cm.engine = "llm"
    return cm
