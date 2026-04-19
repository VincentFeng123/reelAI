"""
Stress-test the heuristic clip picker across the full range of transcript
shapes we see in the wild. NOT a pytest test — this is a dev harness you
run manually to inspect behavior. Use pytest for regression guards after
this reveals what's broken.

Run:
    ./backend/.venv/bin/python -m backend.tests.stress_heuristic_transcripts

Cases include:
    * well-punctuated English
    * no punctuation at all (YouTube auto-captions)
    * only commas (no terminal punct)
    * run-on single mega-sentence
    * filler-heavy
    * bracket artifacts ([Music], [Applause])
    * speaker labels ("Alice: ..." "Bob: ...")
    * all-caps shouting
    * Unicode + emoji + math
    * very short cues (<1s each)
    * very long cues (>20s each)
    * mixed languages
    * empty transcript
    * single-word clip
    * all-filler clip
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Force VERCEL=1 to bypass heavy sentence-transformers load in case the
# user runs this on a machine without the model. Tests still exercise the
# lexical / TF-IDF path and report behavior; switch to VERCEL=0 for the
# semantic leg when MiniLM is available.
os.environ.setdefault("VERCEL", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.ingestion.models import IngestTranscriptCue, IngestTranscriptWord  # noqa: E402
from backend.app.services.clip_boundary import pick_clip_heuristic  # noqa: E402
from backend.app.services.sentences import split_sentences  # noqa: E402


@dataclass
class StressCase:
    name: str
    cues: list[IngestTranscriptCue]
    query: str
    expected: str  # "clip" or "none" or "any"


def _mk_cue(start: float, text: str, duration: float = 3.0) -> IngestTranscriptCue:
    """Build a cue with proportional word-level timestamps."""
    end = start + duration
    words_txt = text.split() if text.strip() else []
    n = max(1, len(words_txt))
    dt = duration / n
    words = [
        IngestTranscriptWord(
            start=start + i * dt,
            end=start + (i + 1) * dt - 0.02,
            text=w,
        )
        for i, w in enumerate(words_txt)
    ]
    return IngestTranscriptCue(
        start=start,
        end=end,
        text=text,
        words=words,
        word_source="whisper" if words else "legacy",
    )


# --------------------------------------------------------------------------- #
# Case generators
# --------------------------------------------------------------------------- #


def case_normal_english() -> StressCase:
    """Well-punctuated. The baseline."""
    cues = [
        _mk_cue(0.0, "Hey everyone, welcome back to the channel.", 3.5),
        _mk_cue(3.5, "Today we're talking about gradient descent in machine learning.", 4.5),
        _mk_cue(8.0, "Gradient descent is the workhorse of modern optimization.", 5.0),
        _mk_cue(13.0, "It walks down the loss surface by following the negative gradient.", 6.0),
        _mk_cue(19.0, "The step size is controlled by a hyperparameter called the learning rate.", 6.5),
        _mk_cue(25.5, "Too large and you overshoot the minimum and diverge.", 5.0),
        _mk_cue(30.5, "Too small and convergence is painfully slow.", 4.5),
        _mk_cue(35.0, "In practice most practitioners use adaptive methods like Adam.", 5.0),
        _mk_cue(40.0, "Thanks for watching, don't forget to like and subscribe.", 4.0),
    ]
    return StressCase("normal_english", cues, "gradient descent", "clip")


def case_no_punctuation() -> StressCase:
    """YouTube auto-captions: no punctuation at all."""
    cues = [
        _mk_cue(0.0, "hey everyone welcome back to the channel", 3.5),
        _mk_cue(3.5, "today we're talking about gradient descent in machine learning", 4.5),
        _mk_cue(8.0, "gradient descent is the workhorse of modern optimization", 5.0),
        _mk_cue(13.0, "it walks down the loss surface by following the negative gradient", 6.0),
        _mk_cue(19.0, "the step size is controlled by a hyperparameter called the learning rate", 6.5),
        _mk_cue(25.5, "too large and you overshoot the minimum and diverge", 5.0),
        _mk_cue(30.5, "too small and convergence is painfully slow", 4.5),
        _mk_cue(35.0, "in practice most practitioners use adaptive methods like adam", 5.0),
    ]
    return StressCase("no_punctuation", cues, "gradient descent", "any")


def case_only_commas() -> StressCase:
    """Comma-spliced, no terminal punctuation."""
    cues = [
        _mk_cue(0.0, "Hey everyone, welcome back, today we're going deep,", 3.5),
        _mk_cue(3.5, "specifically about gradient descent, optimization, machine learning,", 4.5),
        _mk_cue(8.0, "gradient descent is the workhorse, it's everywhere,", 5.0),
        _mk_cue(13.0, "walks down loss surfaces, follows negative gradient,", 6.0),
        _mk_cue(19.0, "learning rate controls step size,", 5.0),
        _mk_cue(24.0, "too large means overshooting, divergence, chaos,", 5.0),
        _mk_cue(29.0, "too small means slow convergence, wasted compute,", 5.0),
    ]
    return StressCase("only_commas", cues, "gradient descent", "any")


def case_single_mega_sentence() -> StressCase:
    """One giant run-on sentence — real problem with some lecture transcripts."""
    text = (
        "So gradient descent is this algorithm where you take a loss function "
        "and compute its gradient with respect to your parameters and then you "
        "subtract a fraction of that gradient from your parameters and that "
        "fraction is controlled by the learning rate and if the learning rate "
        "is too large you overshoot the minimum and your loss actually goes up "
        "and if the learning rate is too small you take tiny steps and "
        "convergence takes forever which is why in practice we use adaptive "
        "methods like Adam or RMSprop that dynamically tune the effective "
        "learning rate per parameter based on the history of gradients seen."
    )
    cues = [_mk_cue(0.0, text, 50.0)]
    return StressCase("single_mega_sentence", cues, "gradient descent", "any")


def case_filler_heavy() -> StressCase:
    """Casual conversational speech — lots of fillers."""
    cues = [
        _mk_cue(0.0, "So um, like, hey everyone.", 3.0),
        _mk_cue(3.0, "Uh, today we're, you know, gonna talk about, um, gradient descent.", 5.0),
        _mk_cue(8.0, "Like, basically, it's an optimization algorithm, right?", 4.0),
        _mk_cue(12.0, "So um, you know, it walks down the loss surface, kind of.", 5.0),
        _mk_cue(17.0, "Um, the learning rate, like, controls the step size, you know.", 5.0),
        _mk_cue(22.0, "I mean, actually, you can use adaptive methods like Adam, right?", 5.0),
        _mk_cue(27.0, "So yeah, um, that's, like, the basic idea, you know?", 5.0),
        _mk_cue(32.0, "Alright so, thanks for watching.", 3.0),
    ]
    return StressCase("filler_heavy", cues, "gradient descent", "clip")


def case_bracket_artifacts() -> StressCase:
    """Transcripts with [Music], [Applause], (inaudible) tags."""
    cues = [
        _mk_cue(0.0, "[Music]", 2.0),
        _mk_cue(2.0, "Hey everyone.", 1.5),
        _mk_cue(3.5, "Today we're talking about gradient descent.", 3.5),
        _mk_cue(7.0, "[Applause]", 2.0),
        _mk_cue(9.0, "Gradient descent walks down the loss surface.", 4.5),
        _mk_cue(13.5, "(inaudible) the learning rate controls step size.", 5.0),
        _mk_cue(18.5, "Too large and you overshoot.", 3.0),
        _mk_cue(21.5, "Too small and convergence is slow.", 4.0),
        _mk_cue(25.5, "In practice we use Adam or RMSprop.", 4.5),
    ]
    return StressCase("bracket_artifacts", cues, "gradient descent", "clip")


def case_speaker_labels() -> StressCase:
    """Transcripts with 'Alice:' / 'Bob:' style labels."""
    cues = [
        _mk_cue(0.0, "Alice: Hey everyone, welcome to the show.", 3.5),
        _mk_cue(3.5, "Bob: Today we're covering gradient descent in ML.", 4.0),
        _mk_cue(7.5, "Alice: So gradient descent is the optimization workhorse.", 5.0),
        _mk_cue(12.5, "Bob: Right, it walks down the loss surface iteratively.", 5.0),
        _mk_cue(17.5, "Alice: The learning rate controls how big each step is.", 5.0),
        _mk_cue(22.5, "Bob: And too large means divergence, too small means slow.", 5.5),
        _mk_cue(28.0, "Alice: In practice we default to Adam or RMSprop.", 5.0),
        _mk_cue(33.0, "Bob: Thanks for watching, see you next episode.", 4.0),
    ]
    return StressCase("speaker_labels", cues, "gradient descent", "clip")


def case_all_caps_shouting() -> StressCase:
    """ALL CAPS — tests that casing normalization works."""
    cues = [
        _mk_cue(0.0, "HEY EVERYONE, WELCOME BACK!", 3.0),
        _mk_cue(3.0, "TODAY WE'RE TALKING ABOUT GRADIENT DESCENT.", 4.0),
        _mk_cue(7.0, "GRADIENT DESCENT IS THE OPTIMIZATION WORKHORSE.", 4.5),
        _mk_cue(11.5, "IT WALKS DOWN THE LOSS SURFACE!", 3.5),
        _mk_cue(15.0, "THE LEARNING RATE CONTROLS THE STEP SIZE.", 4.5),
        _mk_cue(19.5, "TOO LARGE AND YOU DIVERGE.", 3.0),
        _mk_cue(22.5, "TOO SMALL AND IT'S PAINFULLY SLOW.", 4.0),
        _mk_cue(26.5, "WE USE ADAPTIVE METHODS LIKE ADAM.", 4.0),
    ]
    return StressCase("all_caps_shouting", cues, "gradient descent", "clip")


def case_unicode_math() -> StressCase:
    """Transcripts with Unicode symbols, math notation."""
    cues = [
        _mk_cue(0.0, "Welcome back! Today: gradient descent 📉.", 3.5),
        _mk_cue(3.5, "The update rule is θ = θ − α∇L(θ).", 4.5),
        _mk_cue(8.0, "We compute ∇L with respect to each parameter.", 4.5),
        _mk_cue(12.5, "Then we step in the direction −∇L scaled by α.", 5.0),
        _mk_cue(17.5, "The parameter α ∈ ℝ⁺ is the learning rate.", 5.0),
        _mk_cue(22.5, "If α is too large, θ diverges — 💥.", 4.0),
        _mk_cue(26.5, "If α is too small, convergence is O(n²) instead of O(n).", 5.5),
        _mk_cue(32.0, "In practice: Adam, AdaGrad, RMSprop all default to α=1e-3.", 5.0),
    ]
    return StressCase("unicode_math", cues, "gradient descent", "clip")


def case_very_short_cues() -> StressCase:
    """Cues of <1s each — rapid-fire exchange."""
    cues = [
        _mk_cue(0.0, "Hey.", 0.5),
        _mk_cue(0.5, "Welcome.", 0.6),
        _mk_cue(1.1, "Today.", 0.5),
        _mk_cue(1.6, "Gradient.", 0.7),
        _mk_cue(2.3, "Descent.", 0.8),
        _mk_cue(3.1, "Is.", 0.4),
        _mk_cue(3.5, "An.", 0.3),
        _mk_cue(3.8, "Optimization.", 0.9),
        _mk_cue(4.7, "Algorithm.", 0.8),
        _mk_cue(5.5, "It.", 0.3),
        _mk_cue(5.8, "Walks.", 0.5),
        _mk_cue(6.3, "Down.", 0.5),
        _mk_cue(6.8, "The.", 0.3),
        _mk_cue(7.1, "Loss.", 0.6),
        _mk_cue(7.7, "Surface.", 0.8),
        _mk_cue(8.5, "Iteratively.", 1.0),
        _mk_cue(9.5, "Following.", 0.9),
        _mk_cue(10.4, "The.", 0.3),
        _mk_cue(10.7, "Negative.", 0.8),
        _mk_cue(11.5, "Gradient.", 0.9),
        _mk_cue(12.4, "The.", 0.3),
        _mk_cue(12.7, "Learning.", 0.7),
        _mk_cue(13.4, "Rate.", 0.5),
        _mk_cue(13.9, "Controls.", 0.8),
        _mk_cue(14.7, "The.", 0.3),
        _mk_cue(15.0, "Step.", 0.5),
        _mk_cue(15.5, "Size.", 0.6),
        _mk_cue(16.1, "Critically.", 1.0),
    ]
    return StressCase("very_short_cues", cues, "gradient descent", "any")


def case_very_long_cues() -> StressCase:
    """Single cues of 20s+ — uncommon but happens with bad segmentation."""
    cues = [
        _mk_cue(0.0,
                "Hey everyone welcome back to the channel where we dive into "
                "machine learning algorithms and today we're going to be "
                "spending this whole session talking about one particular "
                "optimization algorithm that shows up absolutely everywhere "
                "in modern ML which is gradient descent because every loss "
                "function minimization in practice is some variant of this.",
                25.0),
        _mk_cue(25.0,
                "So gradient descent is the workhorse we walk down the loss "
                "surface by following the negative gradient and the step "
                "size is controlled by the learning rate and in practice "
                "we use adaptive methods like Adam or RMSprop that tune "
                "the learning rate dynamically per parameter.",
                25.0),
    ]
    return StressCase("very_long_cues", cues, "gradient descent", "any")


def case_mixed_languages() -> StressCase:
    """English cues with occasional non-English terms — code-switching."""
    cues = [
        _mk_cue(0.0, "Hey everyone, welcome back.", 2.5),
        _mk_cue(2.5, "Today we cover gradient descent, or as we say in Spanish, descenso de gradiente.", 5.5),
        _mk_cue(8.0, "The Japanese term is 勾配降下法 and is widely used there.", 5.0),
        _mk_cue(13.0, "In French it's la descente de gradient.", 4.5),
        _mk_cue(17.5, "Regardless of language, the math is the same: θ_new = θ − α∇L(θ).", 6.0),
        _mk_cue(23.5, "The learning rate α governs the step size.", 4.5),
        _mk_cue(28.0, "Too large and you diverge.", 3.0),
        _mk_cue(31.0, "Too small and convergence lags.", 3.5),
    ]
    return StressCase("mixed_languages", cues, "gradient descent", "clip")


def case_empty_transcript() -> StressCase:
    """No cues at all."""
    return StressCase("empty_transcript", [], "gradient descent", "none")


def case_single_short_cue() -> StressCase:
    """One cue, way below min_sec."""
    cues = [_mk_cue(0.0, "Hey.", 1.0)]
    return StressCase("single_short_cue", cues, "gradient descent", "none")


def case_all_filler() -> StressCase:
    """Every word is a filler — trimmer should give up gracefully."""
    cues = [
        _mk_cue(0.0, "Um, uh, so, like.", 2.0),
        _mk_cue(2.0, "You know, basically, actually.", 2.5),
        _mk_cue(4.5, "I mean, sort of, kind of.", 2.5),
        _mk_cue(7.0, "So um, you know, like, yeah.", 3.0),
        _mk_cue(10.0, "Alright so, um, okay.", 2.5),
        _mk_cue(12.5, "Well, you know, um, basically.", 3.0),
        _mk_cue(15.5, "I mean, um, sort of, like.", 3.0),
        _mk_cue(18.5, "Actually, you know, um.", 2.5),
    ]
    return StressCase("all_filler", cues, "anything", "any")


def case_off_topic_transcript() -> StressCase:
    """Nothing in the transcript matches the query."""
    cues = [
        _mk_cue(0.0, "Hey everyone, welcome back.", 3.0),
        _mk_cue(3.0, "Today we're cooking pasta carbonara.", 4.0),
        _mk_cue(7.0, "First render the guanciale in a cold pan until crisp.", 5.5),
        _mk_cue(12.5, "Whisk egg yolks with pecorino romano cheese.", 4.5),
        _mk_cue(17.0, "Cook spaghetti al dente in heavily salted water.", 5.0),
        _mk_cue(22.0, "Reserve a cup of pasta water before draining.", 4.5),
        _mk_cue(26.5, "Toss pasta with guanciale off heat, then add egg mixture.", 5.5),
        _mk_cue(32.0, "Stir vigorously with pasta water until silky.", 4.5),
        _mk_cue(36.5, "Serve immediately with extra cheese and black pepper.", 5.0),
    ]
    # Query is for something nowhere near the transcript content.
    return StressCase("off_topic_transcript", cues, "gradient descent", "any")


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def run_case(c: StressCase) -> tuple[str, str]:
    """Run heuristic picker + report. Returns (status, detail)."""
    try:
        sentences = split_sentences(c.cues)
    except Exception as exc:
        return "CRASH@split_sentences", f"{type(exc).__name__}: {exc}"

    try:
        result = pick_clip_heuristic(
            query=c.query,
            cues=c.cues if c.cues else None,
            sentences=sentences,
            silence_ranges=None,
            user_min_sec=15.0,
            user_max_sec=60.0,
            user_target_sec=30.0,
            video_duration_sec=c.cues[-1].end if c.cues else 0.0,
            embed_func=None,  # lexical-only path
        )
    except Exception as exc:
        return "CRASH@pick_heuristic", f"{type(exc).__name__}: {exc}"

    if result is None:
        return "none", f"({len(sentences)} sentences, {len(c.cues)} cues)"

    dur = result.t_end - result.t_start
    is_cue_fallback = "cue-fallback" in (result.signal_summary or "")
    status = "clip-cuefallback" if is_cue_fallback else "clip"
    reason_bits = [
        f"t={result.t_start:.2f}-{result.t_end:.2f}",
        f"dur={dur:.1f}s",
        f"score={result.score:.3f}",
        f"start_sent={result.start_sentence.text[:45]!r}",
    ]
    # Validate contract: clip duration must be inside [15, 60].
    if not (15.0 <= dur <= 60.0):
        status = "OUT_OF_BOUNDS"
    # End sentence must have terminal punct UNLESS we're in cue-fallback
    # mode (transcripts without any terminal punctuation).
    if not is_cue_fallback and not result.end_sentence.terminal_punct:
        status = "NO_TERMINAL_ON_END"
    return status, " ".join(reason_bits)


def main() -> None:
    cases = [
        case_normal_english(),
        case_no_punctuation(),
        case_only_commas(),
        case_single_mega_sentence(),
        case_filler_heavy(),
        case_bracket_artifacts(),
        case_speaker_labels(),
        case_all_caps_shouting(),
        case_unicode_math(),
        case_very_short_cues(),
        case_very_long_cues(),
        case_mixed_languages(),
        case_empty_transcript(),
        case_single_short_cue(),
        case_all_filler(),
        case_off_topic_transcript(),
    ]
    print(f"{'case':<26} {'status':<20} detail")
    print("-" * 100)
    ok_statuses = {"clip", "clip-cuefallback", "none"}
    for c in cases:
        status, detail = run_case(c)
        marker = " " if status in ok_statuses else "!"
        print(f"{marker} {c.name:<24} {status:<20} {detail}")


if __name__ == "__main__":
    main()
