"""Topic-first clip assembly (CLIP_ENGINE=topic).

One clip per SELECTED substantive teaching topic: a batched LLM selection judge drops
filler (intro/outro/transition/promo/tangent), then per kept topic an LLM picks the best
self-contained <=CLIP_MAX_S window (opens on framing, closes on a terminator). The chosen
sentence spans become clip spec dicts fed to the SAME precise cutter the unit engine uses.
Spec: docs/superpowers/specs/2026-07-04-topic-first-clipping-design.md
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel, Field

from ... import config
from ...llm import llm_json
from ..sentences import Sentence
from ..understand.models import ContentNode, Structure
from .integrity import Rejection


# ── LLM schemas ──────────────────────────────────────────────────────────────
class TopicJudgment(BaseModel):
    node_id: str
    type: str = "teaching"          # teaching|intro|outro|transition|admin|promo|tangent
    informativeness: float = 0.0    # 0..1, standalone value
    self_contained: float = 0.0     # 0..1
    why: str = ""


class TopicSelection(BaseModel):
    topics: list[TopicJudgment] = Field(default_factory=list)


class WindowChoice(BaseModel):
    start_idx: int
    end_idx: int
    title: str = ""
    why: str = ""


# ── internal data ────────────────────────────────────────────────────────────
@dataclass
class TopicPick:
    node: ContentNode
    type: str
    informativeness: float
    self_contained: float
    why: str
    warnings: tuple[str, ...] = ()


@dataclass
class Window:
    node_id: str
    start_idx: int
    end_idx: int
    start_s: float
    end_s: float
    title: str
    facet: str
    why: str
    warnings: tuple[str, ...] = ()


# ── selection judge ───────────────────────────────────────────────────────────
_TEACHING = "teaching"

SELECT_SYSTEM = (
    "You curate short, self-contained TEACHING clips from a video's topic outline. For EACH "
    "topic decide its type and how informative it is ON ITS OWN, judged RELATIVE to the whole "
    "video. type is one of: teaching (explains a concept / mechanism / definition-with-"
    "explanation / worked idea / self-contained argument or story), intro, outro, transition, "
    "admin, promo (subscribe / patreon / 'grab the packet'), tangent. informativeness and "
    "self_contained are 0..1. A topic that only welcomes, thanks, recaps, or promotes is NOT "
    "teaching. Return exactly one entry per topic id, using the ids given."
)


def _topic_prompt(topics: list[ContentNode], sentences: list[Sentence]) -> str:
    lines = []
    n = len(sentences)
    for node in topics:
        i0, i1 = node.sentence_range
        first = sentences[i0].text if 0 <= i0 < n else ""
        last = sentences[i1 - 1].text if 0 < i1 <= n else ""
        kw = ", ".join(node.keywords[:8])
        lines.append(
            f"[{node.node_id}] title={node.title!r}"
            + (f" keywords=[{kw}]" if kw else "")
            + (f"\n    summary: {node.summary}" if node.summary else "")
            + f"\n    opens: {first!r}\n    closes: {last!r}"
        )
    return "TOPICS:\n" + "\n".join(lines)


def select_topics(structure: Structure, sentences: list[Sentence],
                  settings: dict) -> tuple[list[TopicPick], list[TopicPick]]:
    """Keep substantive teaching topics; drop filler. Returns (kept, dropped).

    kept is chronological (by node.start) and capped at max_clips; dropped is every
    non-kept topic (for the rejection ledger). Never returns an empty kept list when
    the video has topics (LLM failure ⇒ neutral-teaching fallback)."""
    topics = structure.content_map.topics() or structure.content_map.chapters()
    if not topics:
        return [], []
    thr = float(settings.get("informativeness_min") or config.TOPIC_INFORMATIVENESS_MIN)
    cap = int(settings.get("max_clips") or config.TOPIC_MAX_CLIPS)
    by_id = {node.node_id: node for node in topics}
    try:
        sel = llm_json(SELECT_SYSTEM, _topic_prompt(topics, sentences),
                       TopicSelection, temperature=0.1, model=config.TOPIC_MODEL)
        judged = {j.node_id: j for j in sel.topics if j.node_id in by_id}
    except Exception:
        judged = {}

    picks: list[TopicPick] = []
    for nid, node in by_id.items():
        j = judged.get(nid)
        if j is None:                       # unknown ⇒ neutral teaching (never silently lost)
            picks.append(TopicPick(node, _TEACHING, 0.5, 0.5, ""))
        else:
            picks.append(TopicPick(node, (j.type or _TEACHING).strip().lower(),
                                   float(j.informativeness), float(j.self_contained), j.why))

    kept = [p for p in picks if p.type == _TEACHING and p.informativeness >= thr]
    if not kept:                            # never zero on a real teaching video
        kept = sorted(picks, key=lambda p: p.informativeness, reverse=True)[:max(1, min(cap, len(picks)))]
        kept = [TopicPick(p.node, p.type, p.informativeness, p.self_contained, p.why,
                          ("low_confidence_selection",)) for p in kept]

    kept.sort(key=lambda p: p.informativeness, reverse=True)
    kept = kept[:cap]
    kept.sort(key=lambda p: p.node.start)   # chronological for downstream
    kept_ids = {p.node.node_id for p in kept}
    dropped = [p for p in picks if p.node.node_id not in kept_ids]
    return kept, dropped
