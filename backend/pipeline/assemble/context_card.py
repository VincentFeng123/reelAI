"""Source-grounded context cards (spec §7B).

A ≤2-sentence intro generated ONLY from the anchor + the far "referential" prerequisite
units that closure chose not to inline. Every card sentence must cite a unit and pass a
fuzzy-entailment check against that unit's own words; anything unverifiable is dropped,
and an empty card is omitted rather than hallucinated.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from ... import config
from ..select import _normalize
from ..understand.models import Unit


class CardSentence(BaseModel):
    text: str = ""
    source_unit_id: str = ""


class ContextCardDraft(BaseModel):
    sentences: list[CardSentence] = Field(default_factory=list)


_CARD_SYSTEM = (
    "You write a one- or two-sentence 'earlier in the video' preface that makes a clip easier to "
    "follow. Use ONLY the provided unit summaries; never state anything not in them. Cite each "
    "sentence's source_unit_id. Keep it under {max_words} words total. If the units add nothing "
    "useful, return no sentences. Output only the structured result."
)


def _grounded(text: str, unit: Unit) -> bool:
    from rapidfuzz import fuzz
    hay = _normalize(" ".join([unit.summary] + unit.concepts_introduced + unit.claims))
    needle = _normalize(text)
    if not needle:
        return False
    return fuzz.partial_ratio(needle, hay) >= 55.0


# ── per-clip orientation cards (spec §7B, feed display) ──────────────────────
# The referential preface above only fires when a clip depends on DISTANT earlier material, so a
# self-contained clip gets no card and a prerequisite-dense video gets the SAME foundational card
# on every clip. The feed needs a card per clip that says what THAT clip is about — synthesized
# from the clip's OWN units (grounded by construction), in ONE batched call for all clips (cheaper
# and faster than the old N sequential per-clip referential calls), with a zero-LLM extractive
# fallback so a clip is never left blank.
class _OrientCard(BaseModel):
    clip_index: int = -1
    text: str = ""


class _OrientCards(BaseModel):
    cards: list[_OrientCard] = Field(default_factory=list)


_ORIENT_SYSTEM = (
    "You caption clips pulled from a longer video for a scrollable feed. For EACH clip, write ONE "
    "plain sentence (max {max_words} words) telling a viewer what that clip is about so it stands "
    "on its own. Use ONLY that clip's provided unit summaries — never add a fact that isn't there. "
    "Do not begin with 'This clip', 'This video', or 'The video'. Return exactly one entry per "
    "clip_index. Output only the structured result."
)


def _clip_units(spec: dict, units_by_id: dict[str, Unit]) -> list[Unit]:
    ids = list(spec.get("unit_ids") or [])
    if not ids and spec.get("anchor_id"):
        ids = [spec["anchor_id"]]
    return [units_by_id[u] for u in ids if u in units_by_id]


def _extractive_card(units: list[Unit]) -> str:
    """Grounded-by-construction floor: the first non-empty unit summary (usually the anchor)."""
    for u in units:
        if (u.summary or "").strip():
            words = u.summary.strip().split()
            return " ".join(words[:config.CONTEXT_CARD_MAX_WORDS])
    return ""


def generate_orientation_cards(specs: list[dict], units_by_id: dict[str, Unit], adapter,
                               topic: str) -> list[str]:
    """One grounded orientation card per spec (same order). Extractive baseline + one batched LLM
    polish kept only where it stays grounded to that clip's own units; never fabricates."""
    from rapidfuzz import fuzz

    per_units = [_clip_units(s, units_by_id) for s in specs]
    cards = [_extractive_card(us) for us in per_units]            # grounded floor + fallback

    rows, hay = [], {}
    for i, (s, us) in enumerate(zip(specs, per_units)):
        if not us:
            continue
        sums = " | ".join((u.summary or "").strip() for u in us if (u.summary or "").strip())
        if not sums:
            continue
        rows.append(f"clip_index={i} title={s.get('title', '')!r}\n  units: {sums}")
        hay[i] = _normalize(" ".join(
            " ".join([u.summary or ""] + list(u.concepts_introduced) + list(u.claims)) for u in us))
    if not rows:
        return cards

    from ...llm import llm_json
    system = _ORIENT_SYSTEM.format(max_words=config.CONTEXT_CARD_MAX_WORDS)
    user = (f"TOPIC: {topic or '(general)'}\n\nCLIPS:\n" + "\n".join(rows)
            + "\n\nWrite one caption per clip_index.")
    try:                                                         # any error/malformed response →
        draft = llm_json(system, user, _OrientCards, temperature=0.2)   # keep the extractive floor
        for c in draft.cards:
            i, t = c.clip_index, (c.text or "").strip()
            if not (0 <= i < len(specs)) or not t or i not in hay:
                continue
            if fuzz.partial_ratio(_normalize(t), hay[i]) >= 55.0:   # entailed by the clip's own units
                cards[i] = " ".join(t.split()[:config.CONTEXT_CARD_MAX_WORDS])
    except Exception:
        pass
    return cards


def generate_context_card(spec: dict, units_by_id: dict[str, Unit], adapter, topic: str) -> str:
    in_clip = set(spec.get("unit_ids", []))
    ref_pairs = [(uid, rel) for uid, rel in spec.get("referential", [])
                 if uid in units_by_id and uid not in in_clip]
    ref_units = [units_by_id[uid] for uid, _rel in ref_pairs]
    if not ref_units:
        return ""
    anchor = units_by_id.get(spec.get("anchor_id", ""))
    allowed = {u.unit_id: u for u in (([anchor] if anchor else []) + ref_units)}
    rows = "\n".join(f"{u.unit_id}: {u.summary}" for u in allowed.values())
    from ...llm import llm_json
    system = _CARD_SYSTEM.format(max_words=config.CONTEXT_CARD_MAX_WORDS)
    user = f"TOPIC: {topic or '(general)'}\n\nUNITS (earlier context for the clip):\n{rows}\n\nWrite the preface."
    card = ""
    try:
        draft = llm_json(system, user, ContextCardDraft, temperature=0.2)
        kept = [cs.text.strip() for cs in draft.sentences
                if cs.source_unit_id in allowed and _grounded(cs.text, allowed[cs.source_unit_id])]
        card = " ".join(t for t in kept if t)
    except Exception:
        card = ""
    if not card:
        # extractive fallback: verbatim referential-unit summary — grounded by construction
        for u in ref_units:
            if (u.summary or "").strip():
                card = u.summary.strip()
                break
    words = card.split()
    if len(words) > config.CONTEXT_CARD_MAX_WORDS:     # hard word budget (prompt asks for it too)
        card = " ".join(words[:config.CONTEXT_CARD_MAX_WORDS])
    return card
