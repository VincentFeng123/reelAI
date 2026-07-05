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
