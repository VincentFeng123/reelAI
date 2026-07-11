"""Offline test fixtures: fake providers + word/target helpers.

Everything here runs without a network or API key. A ``TargetProvider`` derives the *correct* sparse
edits by parsing an expected readable string, so tests can assert the pipeline reconstructs an exact
target given a well-behaved model — without invoking a real LLM.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from backend.pipeline.punctuation.types import ChunkEdits, TokenEdit

TERMINALS = {".", "?", "!"}


# ── word builders ────────────────────────────────────────────────────────────
def make_words(text: str, *, start: float = 0.0, dur: float = 0.30, gap: float = 0.02,
               speakers: Optional[list[str]] = None, uniform: bool = False):
    """Build ``[{word,start,end,speaker?}]`` from a plain token string.

    ``uniform=True`` mimics Supadata (evenly spaced, no real pauses)."""
    toks = text.split()
    words = []
    t = start
    for i, tok in enumerate(toks):
        s = t
        e = s + dur
        w = {"word": tok, "start": round(s, 3), "end": round(e, 3)}
        if speakers:
            w["speaker"] = speakers[i]
        words.append(w)
        t = e + (0.0 if uniform else gap)
    return words


def strip_readable(readable: str) -> str:
    """Lowercase and drop punctuation/whitespace collapse — for token-preservation checks."""
    return re.sub(r"\s+", " ", re.sub(r"[.,?!;:\"'“”‘’]", "", readable)).strip().lower()


# ── prompt parsing ───────────────────────────────────────────────────────────
def parse_prompt_tokens(user: str) -> list[dict]:
    line = user.split("TOKENS:\n", 1)[1].splitlines()[0]
    return json.loads(line)


# ── target alignment ─────────────────────────────────────────────────────────
def _align_target(target: str) -> list[tuple[bool, str]]:
    """Per-token (capitalize, punctuationAfter) derived from the expected readable string."""
    out: list[tuple[bool, str]] = []
    for piece in target.split():
        m = re.match(r"^([\"'“”‘’]*)(.*?)([.,?!;:]*)$", piece)
        body, punct = m.group(2), m.group(3)
        cap = bool(body) and body[0].isupper()
        out.append((cap, punct[0] if punct else ""))
    return out


class TargetProvider:
    """Returns the sparse edits that reconstruct ``target`` (aligned to global token indices)."""

    def __init__(self, target: str):
        self.aligned = _align_target(target)

    def infer(self, system: str, user: str, *, est_tokens: int = 0) -> ChunkEdits:
        toks = parse_prompt_tokens(user)
        edits: list[TokenEdit] = []
        for tok in toks:
            gi = int(tok["id"][1:])
            cap, p = self.aligned[gi]
            prev_p = self.aligned[gi - 1][1] if gi > 0 else "."
            ss = (gi == 0) or (prev_p in TERMINALS)
            se = p in TERMINALS
            if cap or p or ss or se:
                edits.append(TokenEdit(id=tok["id"], p=p, cap=cap, ss=ss, se=se, conf=0.97))
        return ChunkEdits(edits=edits)


class RaisingProvider:
    """Always raises — simulates a transport failure / non-JSON output."""

    def infer(self, system: str, user: str, *, est_tokens: int = 0) -> ChunkEdits:
        raise RuntimeError("simulated provider failure")


class UnknownIdProvider:
    """Returns an edit for an id that was never provided — a hard validation failure every time."""

    def infer(self, system: str, user: str, *, est_tokens: int = 0) -> ChunkEdits:
        return ChunkEdits(edits=[TokenEdit(id="w999999", p=".", se=True)])


class FlakyProvider:
    """Fails (bad output) for the first ``bad`` calls, then delegates to a good provider."""

    def __init__(self, good, bad: int = 1, mode: str = "unknown"):
        self.good = good
        self.bad = bad
        self.mode = mode
        self.calls = 0

    def infer(self, system: str, user: str, *, est_tokens: int = 0) -> ChunkEdits:
        self.calls += 1
        if self.calls <= self.bad:
            if self.mode == "raise":
                raise RuntimeError("flaky failure")
            return ChunkEdits(edits=[TokenEdit(id="w999999", p=".", se=True)])
        return self.good.infer(system, user, est_tokens=est_tokens)
