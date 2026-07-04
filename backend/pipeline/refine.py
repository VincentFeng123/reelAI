"""Snap AI-selected segments to sentence boundaries: start at a sentence start,
end exactly on a period (`.`), enforce a minimum duration, add a tiny tail-pad to
the cut point, then remove overlaps so no two clips share footage.

Candidates arrive already anchored to sentence indices (i_start/i_end) by select.py;
this module guarantees the boundary invariants and non-overlap.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

from .. import config
from .sentences import Sentence

ProgressCb = Optional[Callable[[float, str], None]]
NEAR_DUP_EPS = 0.4

# ── BND1 free text-only end guards ───────────────────────────────────────────
# Conjunctions/prepositions/articles that, as a clip's FINAL word, signal a mid-clause cut.
# A weak end is PREFERRED AGAINST when a nearby real-clause end exists, never rejected.
_END_STOPWORDS = frozenset({
    "and", "or", "but", "so", "nor", "yet", "for",
    "to", "of", "in", "on", "at", "by", "with", "from", "into", "onto", "upon",
    "as", "than", "that", "the", "a", "an", "if", "because", "while", "about",
})
_STRONG_END_LOOKAHEAD = 3           # sentences scanned for a STRONG end before accepting a weak one
_WORD_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)


def _is_weak_end(s: Sentence) -> bool:
    """BND1: an end sentence is WEAK when it has <3 words or ends on a conjunction/
    preposition/article — i.e. a mid-clause cut. Weak ends are preferred against, never
    rejected: a clip whose ONLY available end is weak still ships (flagged weak_end_boundary)."""
    words = _WORD_RE.findall(s.text or "")
    if len(words) < 3:
        return True
    return words[-1].lower() in _END_STOPWORDS


def _end_acceptor(sentences: list[Sentence], allow_qe: bool) -> Callable[[Sentence], bool]:
    """BND1 SAFE FALLBACK. Return an end-acceptability predicate: a real terminator is always
    acceptable; when NO sentence carries a real terminator (a fully-unpunctuated caption
    video) chunk edges are the ONLY cut points, so 'chunk_boundary' sentences become
    acceptable ends too and the snapper NEVER leaves such a clip unplaceable. When real
    terminators DO exist they are strictly preferred (a bare chunk edge is not accepted)."""
    has_real_end = any(s.is_valid_end(allow_qe) for s in sentences)

    def end_ok(s: Sentence) -> bool:
        if s.is_valid_end(allow_qe):
            return True
        return (not has_real_end) and ("chunk_boundary" in (s.warnings or ()))
    return end_ok


# ── boundary helpers ─────────────────────────────────────────────────────────
def _valid_end_at_or_after(sentences: list[Sentence], i: int, allow_qe: bool,
                           end_ok: Optional[Callable[[Sentence], bool]] = None) -> Optional[int]:
    ok = end_ok if end_ok is not None else (lambda s: s.is_valid_end(allow_qe))
    for j in range(max(i, 0), len(sentences)):
        if ok(sentences[j]):
            return j
    return None


def _last_valid_end(sentences: list[Sentence], allow_qe: bool,
                    end_ok: Optional[Callable[[Sentence], bool]] = None) -> Optional[int]:
    ok = end_ok if end_ok is not None else (lambda s: s.is_valid_end(allow_qe))
    for j in range(len(sentences) - 1, -1, -1):
        if ok(sentences[j]):
            return j
    return None


def _sentence_containing_time(sentences: list[Sentence], t: float) -> int:
    for i, s in enumerate(sentences):
        if s.start <= t <= s.end:
            return i
    # nearest start at/after t, else last
    for i, s in enumerate(sentences):
        if s.start >= t:
            return i
    return len(sentences) - 1


def _snap_one(cand: dict, sentences: list[Sentence], allow_qe: bool,
              min_dur: float, tail_pad: float, max_dur: float = 0.0) -> Optional[dict]:
    n = len(sentences)
    if n == 0:
        return None
    si = cand.get("i_start")
    ei = cand.get("i_end")
    if si is None or si < 0 or si >= n:
        si = _sentence_containing_time(sentences, float(cand.get("start", 0.0)))
    if ei is None or ei < 0 or ei >= n:
        ei = _sentence_containing_time(sentences, float(cand.get("end", sentences[si].end)))
    si = max(0, min(si, n - 1))
    ei = max(0, min(ei, n - 1))
    if ei < si:
        ei = si

    warnings: list[str] = []
    # BND1: a real terminator beats a fabricated chunk edge (end_ok); on a fully-unpunctuated
    # caption video chunk edges are the only cut points and the snapper falls back to them so
    # placement never fails.
    end_ok = _end_acceptor(sentences, allow_qe)
    # snap end → nearest acceptable end at/after ei (extend outward), PREFERRING a STRONG end
    # (≥3 words, not ending on a conjunction/preposition) over a weak one within a bounded
    # look-ahead. If only a weak end is reachable it is STILL used, flagged weak_end_boundary —
    # a clip is NEVER left unplaceable and no clip is ever rejected by this guard.
    j = _valid_end_at_or_after(sentences, ei, allow_qe, end_ok)
    if j is None:
        last = _last_valid_end(sentences, allow_qe, end_ok)
        if last is None or last < si:
            warnings.append("no_period_terminated_end")
        else:
            j = last
            warnings.append("no_period_after_used_last_period")
    elif _is_weak_end(sentences[j]):
        strong = None
        for k in range(j, min(n, j + 1 + _STRONG_END_LOOKAHEAD)):
            if end_ok(sentences[k]) and not _is_weak_end(sentences[k]) \
                    and (not max_dur or sentences[k].end - sentences[si].start <= max_dur):
                strong = k                       # a nearby real-clause end — prefer it
                break
        if strong is not None:
            j = strong
        else:
            warnings.append("weak_end_boundary")   # only a weak end available — ship it flagged
    ei = max(j if j is not None else ei, si)

    # enforce minimum duration by extending outward to full period sentences. Forward is
    # the default; W25-E: when BOTH directions are available and the forward boundary
    # would leave the anchor's topic node (cand["node_span"], threaded by Part B's
    # snap_candidates; legacy fast path never sets it) while an earlier sentence start
    # stays inside it, extend the START backward instead — 2s prompts stop swallowing
    # the next event's units (their answers) by default.
    node_span = cand.get("node_span")
    guard = 0
    extended = False
    while sentences[ei].end - sentences[si].start < min_dur and guard < n:
        nxt = _valid_end_at_or_after(sentences, ei + 1, allow_qe, end_ok)
        fwd_ok = nxt is not None and nxt != ei
        go_back = False
        if node_span and fwd_ok and si > 0:
            fwd_inside = sentences[nxt].end <= float(node_span[1]) + 1e-6
            back_inside = sentences[si - 1].start >= float(node_span[0]) - 1e-6
            go_back = (not fwd_inside) and back_inside
        if go_back:
            si -= 1
        elif fwd_ok:
            ei = nxt
        else:
            warnings.append("min_duration_unreachable")
            break
        extended = True
        guard += 1
    if extended:
        warnings.append("extended_for_min_duration")   # content beyond the judged span

    # cap maximum duration: pull the end IN to the last boundary within the cap
    if max_dur and (sentences[ei].end - sentences[si].start) > max_dur:
        cap_time = sentences[si].start + max_dur
        best = None
        for j in range(si, ei + 1):
            if end_ok(sentences[j]) and sentences[j].end <= cap_time:
                best = j
        if best is not None and sentences[best].end - sentences[si].start >= min_dur:
            ei = best
            warnings.append("capped_max_duration")
        else:
            fv = _valid_end_at_or_after(sentences, si, allow_qe, end_ok)
            if fv is not None and fv <= ei:
                ei = fv
                warnings.append("capped_max_duration")

    start_t = sentences[si].start
    end_t = sentences[ei].end
    if end_t <= start_t:
        return None
    cap = sentences[ei + 1].start if ei + 1 < n else end_t + tail_pad
    cut_end = min(end_t + tail_pad, cap)

    return {
        "start": round(start_t, 3),
        "end": round(end_t, 3),
        "cut_end": round(cut_end, 3),
        "facet": cand.get("facet", "other"),
        "reason": cand.get("reason", ""),
        "score": float(cand.get("rel", cand.get("score", 0.0))),
        "sentence_start_idx": si,
        "sentence_end_idx": ei,
        "warnings": tuple(warnings),
    }


# ── overlap removal ──────────────────────────────────────────────────────────
def _keep_key(c: dict) -> tuple:
    """P4a dedupe tie-break for overlap/containment losers (Wave 2 §16), in order:
    (i) a spec passing ALL hard-core judge gates (topic/purpose/grounded/references —
    stored by Part B as ``hard_gates_ok``) beats one that does not: a judge-inflated
    score on a mid-clause fragment must never outrank a complete arc that passed the
    gates; (ii) greater contract-required-element coverage (``contract_coverage``,
    choose_contract's satisfied/required scoring over the spec's bound contract);
    (iii) higher final_quality. Legacy fast-path clips carry neither Part-B field, so
    (i)/(ii) are neutral and the comparison reduces to the pre-P4 score tie-break."""
    return (1 if c.get("hard_gates_ok") else 0,
            float(c.get("contract_coverage") or 0.0),
            float(c.get("final_quality", c.get("score", 0.0))))


def _better(a: dict, b: dict) -> dict:
    ka, kb = _keep_key(a), _keep_key(b)
    if ka != kb:
        return a if ka > kb else b
    return a if (a["end"] - a["start"]) >= (b["end"] - b["start"]) else b


def _merge(a: dict, b: dict) -> dict:
    lo, hi = (a, b) if a["start"] <= b["start"] else (b, a)
    end = max(a["end"], b["end"])
    return {
        **_better(a, b),
        "start": lo["start"],
        "end": end,
        "cut_end": max(a["cut_end"], b["cut_end"]),
        "sentence_start_idx": min(a["sentence_start_idx"], b["sentence_start_idx"]),
        "sentence_end_idx": max(a["sentence_end_idx"], b["sentence_end_idx"]),
        "warnings": tuple(set(a["warnings"]) | set(b["warnings"])),
    }


def _trim_start_after(c: dict, t: float, sentences: list[Sentence], min_dur: float) -> Optional[dict]:
    """Move c's start to the first sentence starting at/after t (a real boundary)."""
    for i in range(c["sentence_start_idx"], c["sentence_end_idx"] + 1):
        if sentences[i].start >= t - 1e-6:
            new_start = sentences[i].start
            if c["end"] - new_start >= min_dur:
                d = dict(c)
                d["start"] = round(new_start, 3)
                d["sentence_start_idx"] = i
                d["warnings"] = tuple(set(c.get("warnings") or ()) | {"trimmed_start"})
                return d
            return None
    return None


def _dedupe(clips: list[dict], sentences: list[Sentence], min_dur: float) -> list[dict]:
    clips = sorted(clips, key=lambda c: (c["start"], -c["end"]))
    kept: list[dict] = []
    for c in clips:
        if not kept:
            kept.append(c)
            continue
        k = kept[-1]
        if c["start"] >= k["end"]:                      # disjoint
            kept.append(c)
            continue
        if c["start"] >= k["start"] and c["end"] <= k["end"]:       # c contained in k
            kept[-1] = _better(k, c)
            continue
        if c["start"] <= k["start"] and c["end"] >= k["end"]:       # k contained in c
            kept[-1] = _better(c, k)
            continue
        if abs(c["start"] - k["start"]) <= NEAR_DUP_EPS and abs(c["end"] - k["end"]) <= NEAR_DUP_EPS:
            kept[-1] = _better(k, c)
            continue
        if c["facet"] == k["facet"]:                    # same facet → merge to union
            kept[-1] = _merge(k, c)
            continue
        # cross-facet overlap → try trimming c to start after k (sentence boundary)
        trimmed = _trim_start_after(c, k["end"], sentences, min_dur)
        if trimmed is not None:
            kept.append(trimmed)
        else:
            kept[-1] = _better(k, c)
    return kept


# ── public entry point ───────────────────────────────────────────────────────
def refine_and_snap(candidates: list[dict], sentences: list[Sentence], settings: dict,
                    progress: ProgressCb = None) -> list[dict]:
    allow_qe = bool(settings.get("allow_question_exclaim_ends", False))
    min_dur = float(settings.get("min_clip_duration_s", config.DEFAULTS["min_clip_duration_s"]))
    max_dur = float(settings.get("max_clip_duration_s", config.DEFAULTS["max_clip_duration_s"]))
    tail_pad = float(settings.get("tail_pad_s", config.DEFAULTS["tail_pad_s"]))
    # DEFAULTS["max_clips"] is None since Q1a (None → inherit the anchor-budget cap in the
    # structure-first path); the legacy fast path has no budget, so None means MAX_SEGMENTS.
    max_clips = int(settings.get("max_clips") or config.MAX_SEGMENTS)

    snapped: list[dict] = []
    total = max(1, len(candidates))
    for i, cand in enumerate(candidates):
        clip = _snap_one(cand, sentences, allow_qe, min_dur, tail_pad, max_dur)
        if clip is not None:
            snapped.append(clip)
        if progress:
            progress((i + 1) / total, f"Refining cut {i + 1}/{len(candidates)}")

    final = _dedupe(snapped, sentences, min_dur)
    final.sort(key=lambda c: c["start"])
    return final[:max_clips]
