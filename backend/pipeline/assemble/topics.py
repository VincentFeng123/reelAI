"""Topic-first clip assembly (CLIP_ENGINE=topic).

One clip per SELECTED substantive teaching topic: a batched LLM selection judge drops
filler (intro/outro/transition/promo/tangent), then per kept topic an LLM picks the best
self-contained <=CLIP_MAX_S window (opens on framing, closes on a terminator). The chosen
sentence spans become clip spec dicts fed to the SAME precise cutter the unit engine uses.
Spec: docs/superpowers/specs/2026-07-04-topic-first-clipping-design.md
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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

    # A node the LLM didn't score gets a neutral placeholder — but on a TOTAL failure (no
    # usable judgments at all, e.g. a TOPIC_MODEL outage) it must NOT clear the keep
    # threshold, or every filler topic ships as confident teaching. Score it at 0.0 so `kept`
    # comes back empty → the low_confidence_selection fallback ships a FLAGGED top-N instead.
    neutral = 0.5 if judged else 0.0

    picks: list[TopicPick] = []
    for nid, node in by_id.items():
        j = judged.get(nid)
        if j is None:                       # unknown ⇒ neutral teaching (never silently lost)
            picks.append(TopicPick(node, _TEACHING, neutral, neutral, ""))
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


# ── window extraction ─────────────────────────────────────────────────────────
WINDOW_SYSTEM = (
    "You trim ONE topic to the single best self-contained clip. Choose inclusive sentence "
    "indices start_idx and end_idx from the list. The clip MUST: open on a sentence that frames "
    "the idea for a COLD viewer (never a dangling 'this/these/that/it' pointing outside the clip); "
    "close on a sentence that completes a thought; and carry the topic's core (definition+"
    "explanation, or mechanism+example), dropping recaps, meta-asides and tangents. Keep it at most "
    "{max_s} seconds (aim ~{target_s}s). The '+Ns' marker after each index is elapsed seconds; a "
    "trailing ' .' marks a sentence that ends on a terminator — prefer those as the close."
)


def _window_prompt(sentences: list[Sentence], lo: int, hi: int) -> str:
    t0 = sentences[lo].start
    out = []
    for i in range(lo, hi + 1):
        s = sentences[i]
        mark = " ." if s.ends_with_period else ""
        out.append(f"[{i}] (+{s.end - t0:.0f}s){mark} {s.text}")
    return "SENTENCES:\n" + "\n".join(out)


def _snap_end_to_terminator(sentences: list[Sentence], a: int, b: int, warnings: list[str]) -> int:
    """Walk the end back to the nearest terminator-ending sentence >= a."""
    j = b
    while j > a and not sentences[j].ends_with_period:
        j -= 1
    if not sentences[j].ends_with_period:           # none in [a, b]
        warnings.append("window_close_forced")
        return b
    if j != b:
        warnings.append("window_close_snapped")
    return j


def _fit_budget(sentences: list[Sentence], a: int, b: int, max_s: float, warnings: list[str]) -> int:
    """Truncate the end to the last terminator-ending sentence within max_s of the start."""
    if sentences[b].end - sentences[a].start <= max_s:
        return b
    j = b
    while j > a and sentences[j].end - sentences[a].start > max_s:
        j -= 1
    warnings.append("window_truncated_to_budget")
    # Prefer a terminator-ending sentence within budget; accept k == a (a terminal
    # single-sentence window beats a longer non-terminal one — spec #3 ends-on-terminator).
    k = j
    while k > a and not sentences[k].ends_with_period:
        k -= 1
    if sentences[k].ends_with_period:
        return k
    warnings.append("window_close_forced")   # no terminator anywhere in the budgeted span
    return j


def extract_best_window(pick: TopicPick, sentences: list[Sentence],
                        settings: dict) -> Optional[Window]:
    """Pick the best <=CLIP_MAX_S self-contained window inside (and just before) a topic."""
    node = pick.node
    i0, i1 = node.sentence_range                     # half-open
    win = int(settings.get("boundary_window") or config.TOPIC_BOUNDARY_WINDOW)
    lo = max(0, i0 - win)                             # allow opening a little earlier
    hi = min(len(sentences) - 1, i1 - 1)
    if hi < lo:
        return None
    max_s = float(settings.get("clip_max_s") or config.CLIP_MAX_S)
    try:
        sys = WINDOW_SYSTEM.format(max_s=int(max_s), target_s=int(config.CLIP_TARGET_S))
        ch = llm_json(sys, _window_prompt(sentences, lo, hi), WindowChoice,
                      temperature=0.1, model=config.TOPIC_MODEL)
        a, b, title, why = int(ch.start_idx), int(ch.end_idx), ch.title, ch.why
    except Exception:
        a, b, title, why = i0, hi, node.title, ""    # fall back to the whole topic span

    a = min(max(a, lo), hi)                           # clamp into the shown range
    b = min(max(b, a), hi)
    warnings: list[str] = list(pick.warnings)
    b = _snap_end_to_terminator(sentences, a, b, warnings)
    b = _fit_budget(sentences, a, b, max_s, warnings)
    return Window(node.node_id, a, b, sentences[a].start, sentences[b].end,
                  title or node.title, pick.type, why, tuple(warnings))


def assemble_topic_clips(structure: Structure, topic: str, sentences: list[Sentence], url: str,
                         video_id: str, settings: dict, adapter,
                         progress=None, stats: Optional[dict] = None) -> tuple[list[dict], str, list[Rejection]]:
    """CLIP_ENGINE=topic entry. Mirrors assemble_clips' (specs, notes, rejections) contract."""
    stats = stats if stats is not None else {}

    def emit(frac: float, msg: str = "") -> None:
        if progress:
            progress(max(0.0, min(1.0, frac)), msg)

    rejections: list[Rejection] = []
    if not sentences:
        return [], "No transcript was available to clip.", rejections
    topics = structure.content_map.topics() or structure.content_map.chapters()
    if not topics:
        return [], "This video couldn't be segmented into topics.", rejections

    emit(0.05, "Selecting substantive topics…")
    kept, dropped = select_topics(structure, sentences, settings)
    stats["n_topics_total"] = len(topics)
    stats["n_topics_kept"] = len(kept)
    stats["n_topics_dropped"] = len(dropped)
    for p in dropped:
        rejections.append(Rejection(
            cand_id=p.node.node_id, title=p.node.title or "", role=p.type, stage="topic_select",
            reason=f"dropped as {p.type} (informativeness {p.informativeness:.2f})",
            start=p.node.start, end=p.node.end))
    if not kept:
        return [], "No substantive teaching topics were found in this video.", rejections

    emit(0.2, "Trimming to the best windows…")
    workers = max(1, min(config.UNDERSTAND_WORKERS, len(kept)))
    windows: list[Optional[Window]] = [None] * len(kept)
    if workers == 1:
        for i, p in enumerate(kept):
            try:
                windows[i] = extract_best_window(p, sentences, settings)
            except Exception:
                windows[i] = None       # one bad window never kills the batch (matches parallel path)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(extract_best_window, p, sentences, settings): i
                    for i, p in enumerate(kept)}
            for fut in as_completed(futs):
                try:
                    windows[futs[fut]] = fut.result()
                except Exception:
                    windows[futs[fut]] = None       # one bad window never kills the batch

    tail = float(settings.get("tail_pad_s") or config.DEFAULTS["tail_pad_s"])
    specs: list[dict] = []
    for w in windows:
        if w is None:
            continue
        specs.append({
            "start": float(round(w.start_s, 3)),
            "end": float(round(w.end_s, 3)),
            "cut_end": float(round(w.end_s + tail, 3)),
            "facet": (w.facet or "other"),
            "reason": w.why,
            "title": w.title,
            "role": "",
            "context_card": "",
            "sentence_start_idx": w.start_idx,
            "sentence_end_idx": w.end_idx,
            "unit_ids": [],
            "final_quality": None,
            "warnings": w.warnings,
            "ship_flagged": False,
            "notes": [],
        })

    specs.sort(key=lambda s: s["start"])
    for i, s in enumerate(specs):
        s["sequence_index"] = i + 1
        s["prerequisite_clips"] = []

    emit(1.0, f"Assembled {len(specs)} clip(s)")
    notes = (f"{len(specs)} clip(s) from {len(kept)} topic(s)"
             + (f"; {len(dropped)} filler topic(s) dropped." if dropped else "."))
    return specs, notes, rejections
