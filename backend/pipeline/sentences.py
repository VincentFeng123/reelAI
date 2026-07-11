"""Build a sentence index with precise word-level start/end times and a reliable
`ends_with_period` flag (True for any sentence terminator — '.', '?' or '!') — the
foundation for landing cuts on a sentence boundary.

Punctuation comes from Whisper's `segments` text; precise times come from `words`.
We split the punctuated document into sentences (pysbd + abbreviation guards), then
align each sentence to the word list (monotonic two-pointer + difflib resync).
"""
from __future__ import annotations

import difflib
import re
import threading
from dataclasses import dataclass, field
from typing import Optional

import pysbd

# ── Data model ───────────────────────────────────────────────────────────────
@dataclass
class Sentence:
    idx: int
    text: str
    start: float
    end: float
    terminator: str          # '.', '?', '!', or ''
    ends_with_period: bool    # True for ANY sentence terminator ('.', '?', '!') — historical name
    word_start_idx: int      # index into the original transcript words list
    word_end_idx: int
    align_confidence: float
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def is_valid_end(self, allow_qe: bool = False) -> bool:
        # `ends_with_period` already covers '.', '?' and '!', so any real terminator is a valid clip
        # end. `allow_qe` is retained for call-site compatibility (now redundant).
        return self.ends_with_period or (allow_qe and self.terminator in ("?", "!"))


# ── Abbreviation / false-period guards ───────────────────────────────────────
ABBREVIATIONS = {
    # titles
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "mt", "rev", "hon", "gen",
    "capt", "sgt", "lt", "col", "fr",
    # latin / reference
    "vs", "etc", "e.g", "i.e", "eg", "ie", "cf", "al", "et", "viz", "ca", "approx",
    "circa", "esp", "ibid", "op",
    # academic / technical
    "fig", "figs", "eq", "eqs", "no", "nos", "vol", "vols", "pp", "p", "ch", "sec",
    "secs", "ref", "refs", "tbl", "tab", "def", "thm", "lem", "cor", "app", "appx",
    "ex", "resp", "incl", "min", "max", "approx",
    # orgs
    "inc", "ltd", "co", "corp", "dept", "univ", "assn", "bros", "mfg",
    # time / countries / degrees (also caught by the initials rule)
    "a.m", "p.m", "am", "pm", "ph.d", "m.d", "b.s", "m.s", "b.a", "m.a",
    "u.s", "u.k", "u.n", "e.u",
}

NON_SPEECH = {
    "[music]", "[applause]", "[laughter]", "(music)", "(applause)", "(laughter)",
    "[silence]", "[inaudible]", "[noise]", "(inaudible)",
}

_INITIAL_CHAIN = re.compile(r"([a-z]\.)+[a-z]$")
_CLOSERS = " \t\n)\"'”’"


def _final_token(core: str) -> str:
    parts = core.split()
    return parts[-1] if parts else core


def _final_period_guarded(core: str) -> bool:
    """`core` ends with '.'. True if that period is part of an abbreviation/initial."""
    tok = _final_token(core)
    t = tok.lower().rstrip(".")
    if not t:
        return False
    if t in ABBREVIATIONS:
        return True
    # NOTE: we deliberately do NOT treat a lone single letter ("J.", "v.") as an
    # abbreviation — that would merge physics/math sentences ending in a variable
    # (e.g. "the momentum is mv.", "the velocity v.") which are exactly the clip
    # boundaries we must preserve. Dotted multi-initial chains are unambiguous.
    if _INITIAL_CHAIN.fullmatch(t):          # "J.R.R."
        return True
    return False


def classify_terminator(text: str) -> str:
    """Return the sentence terminator: '.', '?', '!', or '' (none/guarded/ellipsis)."""
    core = text.rstrip(_CLOSERS)
    if not core:
        return ""
    if core.endswith("...") or core.endswith("…"):
        return ""
    last = core[-1]
    if last == ".":
        return "" if _final_period_guarded(core) else "."
    if last in ("?", "!"):
        return last
    return ""


def _ends_in_guarded_period(text: str) -> bool:
    core = text.rstrip(_CLOSERS)
    return (
        core.endswith(".")
        and not core.endswith("...")
        and not core.endswith("…")
        and _final_period_guarded(core)
    )


# ── Segmentation ─────────────────────────────────────────────────────────────
# pysbd's Segmenter is NOT thread-safe: segment() stashes the input on self.original_text and
# re-reads it inside sentences_with_char_spans(), so a single shared instance corrupts
# concurrent callers (a thread's sentences get searched against another thread's text and are
# dropped/misaligned). The boundary REFINE pass fans out per-clip Whisper windows over a thread
# pool, each calling segment_sentences(), so the segmenter is thread-LOCAL: one deterministic
# instance per thread, built lazily once. This is output-IDENTICAL to a serial shared instance —
# segmentation is a pure function of the input text and the instance keeps no output-affecting
# state between calls; it only removes the cross-thread race.
_segmenter_tls = threading.local()


def _get_segmenter() -> pysbd.Segmenter:
    seg: Optional[pysbd.Segmenter] = getattr(_segmenter_tls, "seg", None)
    if seg is None:
        seg = pysbd.Segmenter(language="en", clean=False)
        _segmenter_tls.seg = seg
    return seg


def build_document(segments: list[dict]) -> str:
    parts: list[str] = []
    for seg in segments:
        t = (seg.get("text") or "").strip()
        if not t or t.lower() in NON_SPEECH:
            continue
        parts.append(t)
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def segment_sentences(doc: str) -> list[tuple[str, str]]:
    """Return [(sentence_text, terminator), ...]."""
    if not doc:
        return []
    raw = _get_segmenter().segment(doc)
    merged: list[str] = []
    for s in raw:
        s = s.strip()
        if not s:
            continue
        if merged and _ends_in_guarded_period(merged[-1]):
            merged[-1] = (merged[-1] + " " + s).strip()
        else:
            merged.append(s)
    return [(s, classify_terminator(s)) for s in merged]


# ── Alignment ────────────────────────────────────────────────────────────────
_APOS = str.maketrans({"’": "'", "‘": "'"})


def _norm(s: str) -> str:
    s = s.translate(_APOS).lower().replace("'", "")
    return re.sub(r"[^a-z0-9]", "", s)


def _resolve_prefix(toks: list[str], ti: int, W: list[tuple], local: int) -> tuple[int, int]:
    """Handle Whisper merge/split. Return (advance_sentence_tokens, advance_words)."""
    n_t, n_w = len(toks), len(W)
    # Whisper merged: one word == several sentence tokens (W[local] == toks[ti]+toks[ti+1]+…)
    if W[local][0].startswith(toks[ti]) and W[local][0] != toks[ti]:
        acc, k = toks[ti], 1
        while ti + k < n_t and len(acc) < len(W[local][0]) and W[local][0].startswith(acc):
            acc += toks[ti + k]
            k += 1
        if acc == W[local][0]:
            return k, 1
    # Whisper split: one sentence token == several words (toks[ti] == W[local]+W[local+1]+…)
    if toks[ti].startswith(W[local][0]) and toks[ti] != W[local][0]:
        acc, j = W[local][0], 1
        while local + j < n_w and len(acc) < len(toks[ti]) and toks[ti].startswith(acc):
            acc += W[local + j][0]
            j += 1
        if acc == toks[ti]:
            return 1, j
    return 0, 0


def _resync(toks: list[str], W: list[tuple], wstart: int, window: int = 60) -> tuple:
    win = W[wstart : wstart + window]
    sm = difflib.SequenceMatcher(a=toks, b=[w[0] for w in win], autojunk=False)
    blocks = [bl for bl in sm.get_matching_blocks() if bl.size > 0]
    if not blocks:
        return None, None, 0, wstart
    first_w = wstart + blocks[0].b
    last_blk = blocks[-1]
    last_w = wstart + last_blk.b + last_blk.size - 1
    matched = sum(bl.size for bl in blocks)
    return first_w, last_w, matched, last_w + 1


def _align(sent_tokens: list[list[str]], W: list[tuple], drift_limit: int = 4) -> list[tuple]:
    results: list[tuple] = []
    wcur = 0
    n = len(W)
    for toks in sent_tokens:
        if not toks:
            results.append((None, None, 0.0))
            continue
        first_w = last_w = None
        matched = 0
        miss = 0
        local = wcur
        ti = 0
        while ti < len(toks) and local < n:
            st, wt = toks[ti], W[local][0]
            if st == wt:
                if first_w is None:
                    first_w = local
                last_w = local
                matched += 1
                miss = 0
                ti += 1
                local += 1
            else:
                adv_s, adv_w = _resolve_prefix(toks, ti, W, local)
                if adv_w > 0:
                    if first_w is None:
                        first_w = local
                    last_w = local + adv_w - 1
                    matched += 1
                    miss = 0
                    ti += adv_s
                    local += adv_w
                elif local + 1 < n and st == W[local + 1][0]:
                    local += 1            # word-side insertion
                elif ti + 1 < len(toks) and toks[ti + 1] == wt:
                    ti += 1               # sentence-side insertion
                else:
                    ti += 1
                    miss += 1
            if miss > drift_limit:
                fw, lw, m, nxt = _resync(toks, W, wcur)
                if fw is not None:
                    first_w, last_w, matched = fw, lw, m
                    local = nxt
                break
        conf = matched / max(1, len(toks))
        if first_w is None:
            results.append((None, None, 0.0))
        else:
            results.append((first_w, last_w, conf))
            wcur = last_w + 1
    return results


def _interpolate(sentences: list[Sentence], total_end: float) -> None:
    """Fill start/end for any unmatched sentences from their neighbours."""
    n = len(sentences)
    for i, s in enumerate(sentences):
        if s.start is not None and s.end is not None:
            continue
        prev_end = next((sentences[j].end for j in range(i - 1, -1, -1)
                         if sentences[j].end is not None), 0.0)
        nxt_start = next((sentences[j].start for j in range(i + 1, n)
                          if sentences[j].start is not None), total_end)
        s.start = prev_end
        s.end = max(prev_end, nxt_start)


# ── Public entry point ───────────────────────────────────────────────────────
def build_sentence_index(transcript: dict) -> list[Sentence]:
    words = transcript.get("words") or []
    segments = transcript.get("segments") or []

    doc = build_document(segments)
    sent_texts = segment_sentences(doc)

    W: list[tuple] = []
    for i, w in enumerate(words):
        t = _norm(w.get("word", ""))
        if t:
            W.append((t, i, float(w.get("start", 0.0)), float(w.get("end", 0.0))))

    total_end = float(words[-1].get("end", 0.0)) if words else 0.0
    sent_tokens = [[_norm(p) for p in text.split() if _norm(p)] for text, _ in sent_texts]
    aligns = _align(sent_tokens, W)

    sentences: list[Sentence] = []
    for idx, ((text, term), (fw, lw, conf)) in enumerate(zip(sent_texts, aligns)):
        warnings: list[str] = []
        if fw is None:
            start = end = None
            ws = we = -1
            warnings.append("unmatched")
        else:
            start, end = W[fw][2], W[lw][3]
            ws, we = W[fw][1], W[lw][1]
            if conf < 0.6:
                warnings.append("low_confidence_align")
        sentences.append(
            Sentence(idx, text, start, end, term, term in (".", "?", "!"), ws, we, conf, tuple(warnings))
        )

    _interpolate(sentences, total_end)
    return sentences


def window_sentences(transcript: dict, target_sec: float = 12.0) -> list[Sentence]:
    """Re-window a transcript into ~target_sec units from word times.

    Used when caption punctuation is too sparse to give usable sentence
    boundaries (run-on auto-captions). Each window is a valid clip boundary.
    """
    words = transcript.get("words") or []
    if not words:
        return sentences_from_chunks(transcript.get("chunks", []))
    units: list[tuple[float, float, list[str]]] = []
    start: Optional[float] = None
    toks: list[str] = []
    for w in words:
        if start is None:
            start = float(w.get("start", 0.0))
        toks.append((w.get("word") or "").strip())
        if float(w.get("end", 0.0)) - start >= target_sec and toks:
            units.append((start, float(w.get("end", 0.0)), toks))
            start, toks = None, []
    if toks and start is not None:
        units.append((start, float(words[-1].get("end", start)), toks))

    out: list[Sentence] = []
    for i, (s, e, tk) in enumerate(units):
        text = " ".join(t for t in tk if t).strip()
        if not text:
            continue
        out.append(
            Sentence(
                idx=len(out), text=text, start=s, end=e,
                terminator=".", ends_with_period=True,
                word_start_idx=-1, word_end_idx=-1, align_confidence=1.0,
                warnings=("windowed",),
            )
        )
    return out


def sentences_from_punctuation(result, words) -> list[Sentence]:
    """Convert a ``PunctuationResult`` into the legacy ``Sentence`` model, exactly.

    Times come straight from the restored spans (which were computed from the raw word timestamps),
    token indices from the immutable ``w<index>`` ids, and the terminator from the last token's
    ``punctuationAfter`` — re-checked through :func:`_final_period_guarded` so an LLM period after
    an abbreviation ("Dr.", "e.g.") is not treated as a clip-boundary period."""
    by_id = {pw.id: pw for pw in result.words}
    out: list[Sentence] = []
    for ts in result.sentences:
        if not ts.tokenIds:
            continue
        i = int(ts.tokenIds[0][1:])
        j = int(ts.tokenIds[-1][1:])
        last = by_id.get(ts.tokenIds[-1])
        mark = last.punctuationAfter if last else ""
        term = mark if mark in (".", "?", "!") else ""
        if term == "." and _final_period_guarded(ts.text.rstrip(_CLOSERS)):
            term = ""
        out.append(
            Sentence(
                idx=len(out),
                text=ts.text,
                start=float(ts.start),
                end=float(ts.end),
                terminator=term,
                ends_with_period=bool(term),   # True for '.', '?' or '!' (guarded '.' → '' → False)
                word_start_idx=i,
                word_end_idx=j,
                align_confidence=float(ts.confidence) if ts.confidence is not None else 1.0,
                warnings=("punctuated",) + (("degraded",) if result.status == "degraded" else ()),
            )
        )
    return out


def sentences_from_chunks(chunks: list[dict]) -> list[Sentence]:
    """Fallback: treat each transcript chunk as a sentence unit (chunk edges are
    valid clip boundaries). Used when caption transcripts lack real punctuation."""
    out: list[Sentence] = []
    for i, ch in enumerate(chunks):
        text = (ch.get("text") or "").strip()
        if not text:
            continue
        term = classify_terminator(text)
        out.append(
            Sentence(
                idx=len(out),
                text=text,
                start=float(ch.get("start", 0.0)),
                end=float(ch.get("end", 0.0)),
                # BND1: distinguish a REAL terminator from a fabricated chunk edge. A chunk
                # whose text truly ends in .?! is a STRONG end (ends_with_period=True → a valid
                # clip end); a bare chunk is a WEAK boundary — kept USABLE via the
                # 'chunk_boundary' warning (the snapper falls back to chunk edges when NO real
                # terminator exists, so placement never fails) but is_valid_end() is False so a
                # real terminator nearby is preferred. (Previously every chunk was stamped
                # ends_with_period=True — a fabricated terminator that made period-snapping a
                # no-op: clips ended mid-clause even though ends_on_period_rate read 1.0.)
                terminator=term,
                ends_with_period=bool(term),
                word_start_idx=-1,
                word_end_idx=-1,
                align_confidence=1.0,
                warnings=("chunk_boundary",),
            )
        )
    return out
