"""
Audit the punctuation contract: every clip returned by pick_clip_heuristic
and snap_llm_boundary should BEGIN on a sentence boundary (first word of
a complete thought, right after a period of the previous sentence or the
video start) and END on a terminal-punct boundary (last word is
immediately followed by ``.``, ``!``, ``?``, or ``…``).

This harness generates synthetic transcripts with a KNOWN punctuation
structure, runs the picker, reconstructs the clipped text from word-level
timestamps, and checks both edges against the contract.

Run:
    ./backend/.venv/bin/python -m backend.tests.audit_punctuation_contract

For each case the harness prints:
    * the clip's start and end timestamps
    * the character right BEFORE the clip (what came before t_start)
    * the first word of the clip
    * the last word of the clip
    * the character right AFTER the clip
    * verdict: PASS / FAIL-START / FAIL-END / FAIL-BOTH / FAIL-NONE
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("VERCEL", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.ingestion.models import IngestTranscriptCue, IngestTranscriptWord  # noqa: E402
from backend.app.services.clip_boundary import (  # noqa: E402
    HeuristicPickResult,
    pick_clip_heuristic,
    snap_llm_boundary,
)
from backend.app.services.sentences import split_sentences  # noqa: E402


_TERMINAL_PUNCT = {".", "!", "?", "…"}
# Filler tokens our trimmer removes — boundaries shifted by filler trim
# are still contractually valid because the trimmed words sat within a
# single sentence, so begin/end-on-punctuation is preserved at the
# sentence level even if the clip starts a few words in.
_TRIM_TOLERANCE_WORDS = 5


# --------------------------------------------------------------------------- #
# Synthetic transcript builder — we know exactly where every word starts/ends
# --------------------------------------------------------------------------- #


def build_transcript(
    sentences: list[str],
    *,
    start: float = 0.0,
    word_dur: float = 0.25,
    word_gap: float = 0.02,
    sentence_gap: float = 0.4,
) -> list[IngestTranscriptCue]:
    """Build a list of cues (one per sentence) with known word-level
    timestamps. Each word gets `word_dur` seconds and a `word_gap` pad;
    each sentence is followed by a longer `sentence_gap` pause.
    """
    cues: list[IngestTranscriptCue] = []
    t = start
    for sentence in sentences:
        words_txt = sentence.split()
        if not words_txt:
            continue
        cue_start = t
        words: list[IngestTranscriptWord] = []
        for w in words_txt:
            ws = t
            we = t + word_dur
            words.append(IngestTranscriptWord(start=ws, end=we, text=w))
            t = we + word_gap
        cue_end = t
        cues.append(IngestTranscriptCue(
            start=cue_start,
            end=cue_end,
            text=sentence,
            words=words,
            word_source="whisper",
        ))
        t += sentence_gap  # between-sentence pause
    return cues


# --------------------------------------------------------------------------- #
# Contract verification
# --------------------------------------------------------------------------- #


@dataclass
class Inspection:
    clip_start: float
    clip_end: float
    clip_first_word: str
    clip_last_word: str
    prev_word: str  # word immediately before clip (empty if clip at t=0)
    next_word: str  # word immediately after clip (empty if clip at end)
    clip_text: str


def inspect_clip(
    cues: list[IngestTranscriptCue],
    t_start: float,
    t_end: float,
) -> Inspection | None:
    """Reconstruct the exact words inside [t_start, t_end] from word
    timings, plus the immediately-adjacent words outside the clip.
    """
    flat: list[tuple[float, float, str]] = []
    for cue in cues:
        for w in cue.words:
            flat.append((float(w.start), float(w.end), str(w.text)))
    flat.sort(key=lambda x: x[0])
    if not flat:
        return None

    # A word is "inside the clip" if its midpoint lies in [t_start, t_end].
    inside_idx: list[int] = []
    for i, (ws, we, _) in enumerate(flat):
        mid = 0.5 * (ws + we)
        if t_start - 0.05 <= mid <= t_end + 0.05:
            inside_idx.append(i)
    if not inside_idx:
        return None

    first_i = inside_idx[0]
    last_i = inside_idx[-1]
    prev_word = flat[first_i - 1][2] if first_i > 0 else ""
    next_word = flat[last_i + 1][2] if last_i + 1 < len(flat) else ""
    clip_words = [flat[i][2] for i in inside_idx]
    return Inspection(
        clip_start=t_start,
        clip_end=t_end,
        clip_first_word=clip_words[0],
        clip_last_word=clip_words[-1],
        prev_word=prev_word,
        next_word=next_word,
        clip_text=" ".join(clip_words),
    )


def check_contract(insp: Inspection) -> tuple[bool, bool, str]:
    """Check whether the clip begins and ends on punctuation boundaries.

    Returns ``(starts_on_punct, ends_on_punct, note)``.

    A clip BEGINS on punctuation when one of:
      * There is no preceding word (clip at video start); OR
      * The preceding word ends with ``.!?…`` (previous sentence finished).

    A clip ENDS on punctuation when:
      * The clip's last word ends with ``.!?…`` (this sentence finished).

    Filler trimming can shift the clip start a few words INTO a sentence
    (trimming "Um, so" off the top). Those clips technically start mid-
    sentence in char terms, but the underlying sentence still began on
    punctuation. We accept that — the sentence contract holds, just the
    edge word changed. The note column flags this case for visibility.
    """
    notes: list[str] = []

    # START check
    starts_on_punct = False
    if not insp.prev_word:
        starts_on_punct = True
        notes.append("start=video-edge")
    elif insp.prev_word and insp.prev_word[-1] in _TERMINAL_PUNCT:
        starts_on_punct = True
    else:
        # Filler-trim tolerance: previous word might be a non-terminal
        # filler that we trimmed, and the real sentence start is further
        # back. This is a soft pass — the sentence-level contract still
        # holds.
        notes.append("start=mid-sentence-after-trim")

    # END check
    ends_on_punct = bool(
        insp.clip_last_word and insp.clip_last_word[-1] in _TERMINAL_PUNCT
    )
    if not ends_on_punct and insp.next_word == "":
        # Clip reaches end of video — allow this as a soft pass too.
        ends_on_punct = True
        notes.append("end=video-edge")

    return starts_on_punct, ends_on_punct, ",".join(notes)


# --------------------------------------------------------------------------- #
# Test cases — each builds a transcript with KNOWN punctuation structure
# --------------------------------------------------------------------------- #


@dataclass
class AuditCase:
    name: str
    sentences: list[str]
    query: str
    # Also run through snap_llm_boundary with a simulated LLM pick:
    llm_raw_t_start: float | None = None
    llm_raw_t_end: float | None = None


CASES: list[AuditCase] = [
    AuditCase(
        "basic_three_topics",
        [
            "Welcome to the channel everyone.",
            "Today we are discussing calculus fundamentals.",
            "Gradient descent is an optimization algorithm used in machine learning.",
            "It walks down the loss surface by following the negative gradient direction.",
            "The learning rate parameter controls how large each step is.",
            "Too large and you overshoot the minimum point of the loss.",
            "Too small and convergence is painfully slow to reach.",
            "In practice we default to Adam or RMSprop for training.",
            "Thanks for watching this tutorial video.",
        ],
        "gradient descent",
    ),
    AuditCase(
        "all_periods_simple",
        [
            "The cat sat on the mat.",
            "The dog barked at the cat.",
            "Then both animals fell asleep in the sun.",
            "A bird flew overhead and landed nearby.",
            "The cat opened one eye then closed it again.",
            "Hours passed with nobody moving from their spot.",
            "Eventually the sun set behind the distant hills.",
        ],
        "the cat",
    ),
    AuditCase(
        "with_questions_and_exclaims",
        [
            "What is gradient descent anyway?",
            "I am so glad you asked that question!",
            "It is the workhorse of modern machine learning optimization.",
            "You compute gradients and step downhill on the loss surface.",
            "The learning rate controls how big your steps are.",
            "Why does this matter so much in practice?",
            "Because without it training simply would not converge!",
            "That is the entire point of the algorithm really.",
        ],
        "gradient descent",
    ),
    AuditCase(
        "fillers_at_sentence_start",
        [
            "Um, so welcome back to the channel everyone.",
            "Uh, today we are going to talk about gradient descent in ML.",
            "Like, basically, gradient descent is an optimization method.",
            "You know, it walks down the loss surface iteratively.",
            "So actually, the learning rate controls the step size.",
            "I mean, too large and you overshoot the minimum.",
            "Well, too small and you converge way too slowly.",
            "Okay so, in practice we use Adam or RMSprop.",
        ],
        "gradient descent",
    ),
    AuditCase(
        "fillers_at_sentence_end",
        [
            "Welcome back to the channel everyone.",
            "Today we discuss gradient descent, you know.",
            "Gradient descent is an optimization method, basically.",
            "It walks down the loss surface, kind of.",
            "The learning rate controls the step size, sort of.",
            "Too large and you overshoot, you know.",
            "Too small and convergence is slow, I mean.",
            "In practice we use Adam, actually.",
        ],
        "gradient descent",
    ),
    AuditCase(
        "simulated_llm_pick_clean",
        [
            "Welcome back everyone today.",
            "We are covering gradient descent in depth now.",
            "Gradient descent is an optimization algorithm.",
            "It steps along the negative gradient direction.",
            "The learning rate scales the step size taken.",
            "Too large causes divergence away from minimum.",
            "Too small causes painfully slow convergence behavior.",
            "In practice Adam or RMSprop are the preferred choice.",
            "Thanks for watching the tutorial.",
        ],
        "gradient descent",
        # Simulate LLM picking t_start/t_end that land cleanly.
        llm_raw_t_start=2.0,
        llm_raw_t_end=20.0,
    ),
    AuditCase(
        "simulated_llm_pick_mid_sentence",
        [
            "Welcome back everyone today.",
            "We are covering gradient descent in depth now.",
            "Gradient descent is an optimization algorithm.",
            "It steps along the negative gradient direction.",
            "The learning rate scales the step size taken.",
            "Too large causes divergence away from minimum.",
            "Too small causes painfully slow convergence behavior.",
            "In practice Adam or RMSprop are the preferred choice.",
            "Thanks for watching the tutorial.",
        ],
        "gradient descent",
        # Simulate LLM picking mid-sentence timestamps (common LLM error).
        llm_raw_t_start=3.3,
        llm_raw_t_end=18.7,
    ),
    AuditCase(
        "long_sentences_sparse_punct",
        [
            ("An extremely long sentence with many clauses that goes on "
             "describing how gradient descent works by computing the "
             "derivative of a loss function and then stepping in the "
             "negative direction of that derivative which is why it "
             "is called gradient descent in the first place."),
            ("Another extremely long sentence that explains how the "
             "learning rate affects convergence because if the rate is "
             "too large the algorithm overshoots and diverges and if "
             "the rate is too small the algorithm converges way too "
             "slowly to be useful in practice for any real model."),
            ("A shorter wrap up sentence."),
        ],
        "gradient descent",
    ),
    AuditCase(
        "only_questions",
        [
            "What is gradient descent?",
            "Why does it work?",
            "How do we tune the learning rate?",
            "When does it fail to converge?",
            "Why is Adam so popular in practice?",
            "What happens if we pick the wrong rate?",
            "How do we know when we have found the minimum?",
            "Can we use second-order methods instead sometimes?",
        ],
        "gradient descent",
    ),
    AuditCase(
        "ellipsis_and_mixed",
        [
            "So let us begin…",
            "Gradient descent is the main optimization algorithm.",
            "But wait, there is more to consider!",
            "What about the learning rate?",
            "It controls how large each step is.",
            "Too large and things go wrong…",
            "Too small and you wait forever.",
            "In practice adaptive methods are best.",
        ],
        "gradient descent",
    ),
]


def run_case(c: AuditCase) -> list[tuple[str, bool, bool, str]]:
    """Run a case through the heuristic picker AND (optionally) snap_llm_boundary.
    Returns list of (path_label, starts_on_punct, ends_on_punct, note) tuples."""
    cues = build_transcript(c.sentences)
    sentences = split_sentences(cues)
    results: list[tuple[str, bool, bool, str]] = []

    # --- Heuristic picker path ---
    video_duration = cues[-1].end if cues else 0.0
    heur = pick_clip_heuristic(
        query=c.query,
        cues=cues,
        sentences=sentences,
        silence_ranges=None,
        user_min_sec=15.0,
        user_max_sec=60.0,
        user_target_sec=25.0,
        video_duration_sec=video_duration,
        embed_func=None,
    )
    if heur is not None:
        insp = inspect_clip(cues, heur.t_start, heur.t_end)
        if insp is None:
            results.append(("heuristic", False, False, "no-insp"))
        else:
            sop, eop, note = check_contract(insp)
            detail = (
                f"t={heur.t_start:.2f}-{heur.t_end:.2f} "
                f"[prev={insp.prev_word!r:<12}] "
                f"{insp.clip_first_word!r}…{insp.clip_last_word!r} "
                f"[next={insp.next_word!r:<12}] {note}"
            )
            results.append(("heuristic", sop, eop, detail))
    else:
        results.append(("heuristic", False, False, "picker returned None"))

    # --- Snap path (simulated LLM pick) ---
    if c.llm_raw_t_start is not None and c.llm_raw_t_end is not None:
        snap = snap_llm_boundary(
            raw_t_start=c.llm_raw_t_start,
            raw_t_end=c.llm_raw_t_end,
            sentences=sentences,
            silence_ranges=None,
            min_sec=15.0,
            max_sec=60.0,
            ingest_cues=cues,
        )
        if not snap.snapped:
            results.append(("snap", False, False, f"rejected: {snap.reason}"))
        else:
            insp = inspect_clip(cues, snap.t_start, snap.t_end)
            if insp is None:
                results.append(("snap", False, False, "no-insp"))
            else:
                sop, eop, note = check_contract(insp)
                detail = (
                    f"raw={c.llm_raw_t_start:.2f}-{c.llm_raw_t_end:.2f} "
                    f"→ snap={snap.t_start:.2f}-{snap.t_end:.2f} "
                    f"[prev={insp.prev_word!r:<12}] "
                    f"{insp.clip_first_word!r}…{insp.clip_last_word!r} "
                    f"[next={insp.next_word!r:<12}] {note}"
                )
                results.append(("snap", sop, eop, detail))
    return results


def main() -> None:
    total = 0
    passed = 0
    print(f"{'case':<38} {'path':<10} start  end  detail")
    print("-" * 140)
    for c in CASES:
        results = run_case(c)
        for (path_label, sop, eop, detail) in results:
            total += 1
            if sop and eop:
                passed += 1
            start_mark = "✓" if sop else "✗"
            end_mark = "✓" if eop else "✗"
            status_flag = "  " if (sop and eop) else "! "
            print(f"{status_flag}{c.name:<36} {path_label:<10}  {start_mark}    {end_mark}   {detail}")
    print("-" * 140)
    print(f"PASS: {passed}/{total}  ({100.0 * passed / max(1, total):.0f}%)")


if __name__ == "__main__":
    main()
