"""
Simulate a user search end-to-end: fetch a real YouTube transcript, run it
through the full clip-picker pipeline (topic_cut → LLM or heuristic →
snap → word-level refinement), and print each result clip with its text.

Uses youtube_transcript_api for the transcript fetch (no proxy / no
curl_cffi needed — direct library call). No LLM keys required; the run
will fall through to the heuristic picker if no keys are set.

Run:
    ./backend/.venv/bin/python -m backend.tests.simulate_user_search

Or override the video / query:
    VIDEO_ID=IHZwWFHWa-w QUERY="gradient descent" ./backend/.venv/bin/python -m backend.tests.simulate_user_search
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.ingestion.models import IngestTranscriptCue, IngestTranscriptWord  # noqa: E402
from backend.app.services.topic_cut import (  # noqa: E402
    TranscriptCue,
    cut_video_into_topic_reels,
)


# Defaults (3Blue1Brown "Gradient descent, how neural networks learn").
# Well-punctuated human captions, ~21 min, classic test fixture.
DEFAULT_VIDEO_ID = "IHZwWFHWa-w"
DEFAULT_QUERY = "gradient descent"

# User-settings defaults matching SettingsView.default.
USER_MIN_SEC = 20.0
USER_MAX_SEC = 55.0
USER_TARGET_SEC = 55.0


# --------------------------------------------------------------------------- #
# Transcript fetch
# --------------------------------------------------------------------------- #


def fetch_transcript(video_id: str) -> list[dict]:
    """Fetch a YouTube transcript via youtube_transcript_api. Returns the
    raw cue list as dicts [{"text": str, "start": float, "duration": float}].
    """
    from youtube_transcript_api import YouTubeTranscriptApi

    api = YouTubeTranscriptApi()
    fetched = api.fetch(video_id, languages=["en", "en-US", "en-GB"])
    # youtube_transcript_api 1.x returns a FetchedTranscript with .snippets
    return [
        {"text": s.text, "start": float(s.start), "duration": float(s.duration)}
        for s in fetched.snippets
    ]


# --------------------------------------------------------------------------- #
# Cue adaptation — build ingest_cues_for_precision with proportional word
# timestamps so the word-level refiner has data to work with. Real Whisper
# cues from the ingestion pipeline would give word-level timings; for this
# simulation we synthesize them from cue text + duration.
# --------------------------------------------------------------------------- #


def build_ingest_cues(raw: list[dict]) -> list[IngestTranscriptCue]:
    cues: list[IngestTranscriptCue] = []
    for entry in raw:
        start = float(entry["start"])
        duration = max(0.01, float(entry.get("duration") or 0.0))
        end = start + duration
        text = str(entry.get("text") or "").replace("\n", " ").strip()
        words_txt = text.split()
        n = max(1, len(words_txt))
        dt = duration / n
        words = [
            IngestTranscriptWord(
                start=start + i * dt,
                end=start + (i + 1) * dt - max(0.01, dt * 0.05),
                text=w,
            )
            for i, w in enumerate(words_txt)
        ]
        cues.append(IngestTranscriptCue(
            start=start,
            end=end,
            text=text,
            words=words,
            word_source="proportional",
        ))
    return cues


def build_transcript_cues(raw: list[dict]) -> list[TranscriptCue]:
    out: list[TranscriptCue] = []
    for entry in raw:
        text = str(entry.get("text") or "").replace("\n", " ").strip()
        if not text:
            continue
        out.append(TranscriptCue(
            start=float(entry["start"]),
            duration=float(entry.get("duration") or 0.0),
            text=text,
        ))
    return out


# --------------------------------------------------------------------------- #
# Reconstruct the clipped text from (t_start, t_end)
# --------------------------------------------------------------------------- #


def extract_clip_text(cues: list[IngestTranscriptCue], t_start: float, t_end: float) -> tuple[str, str, str]:
    """Return (prev_word_context, clip_text, next_word_context). Each
    context is ~3 words outside the clip for visual continuity."""
    # Flatten to (start, end, text, is_in_clip)
    flat: list[tuple[float, float, str, bool]] = []
    for cue in cues:
        if not cue.words:
            mid = 0.5 * (cue.start + cue.end)
            in_clip = t_start <= mid <= t_end
            flat.append((cue.start, cue.end, cue.text, in_clip))
            continue
        for w in cue.words:
            mid = 0.5 * (float(w.start) + float(w.end))
            in_clip = t_start <= mid <= t_end
            flat.append((float(w.start), float(w.end), w.text, in_clip))
    flat.sort(key=lambda x: x[0])

    first_in = next((i for i, tup in enumerate(flat) if tup[3]), None)
    last_in = None
    for i in range(len(flat) - 1, -1, -1):
        if flat[i][3]:
            last_in = i
            break
    if first_in is None or last_in is None:
        return "", "", ""

    prev_ctx = " ".join(t for _, _, t, _ in flat[max(0, first_in - 3):first_in])
    clip_text = " ".join(t for _, _, t, _ in flat[first_in:last_in + 1])
    next_ctx = " ".join(t for _, _, t, _ in flat[last_in + 1:last_in + 4])
    return prev_ctx, clip_text, next_ctx


def check_punct_contract(
    cues: list[IngestTranscriptCue], t_start: float, t_end: float,
) -> tuple[bool, bool, str]:
    prev_ctx, clip_text, next_ctx = extract_clip_text(cues, t_start, t_end)
    terminal = {".", "!", "?", "…"}
    if not clip_text:
        return False, False, "no-clip-words"
    prev_word = prev_ctx.split()[-1] if prev_ctx else ""
    last_word = clip_text.split()[-1]
    starts_on_punct = (
        not prev_word or (prev_word and prev_word[-1] in terminal)
    )
    ends_on_punct = last_word[-1] in terminal if last_word else False
    return starts_on_punct, ends_on_punct, f"prev={prev_word!r} last={last_word!r}"


# --------------------------------------------------------------------------- #
# Answer 1: does this clip talk about the central core idea (the query)?
# --------------------------------------------------------------------------- #


def central_core_score(clip_text: str, query: str) -> tuple[float, int, int]:
    """Return (coverage_ratio, sentence_hits, total_sentences).

    coverage_ratio = fraction of sentences in the clip that mention at
    least one word of the query. Very rough proxy for "directly about
    the query" — good enough to flag the 'book recommendation' case
    that mentions 'learning' but isn't about gradient descent.
    """
    import re as _re
    query_words = {w.lower() for w in _re.findall(r"[A-Za-z]+", query) if len(w) >= 3}
    if not query_words:
        return 0.0, 0, 0
    # Split clip into sentences (naive split on terminal punct).
    sentences = [s.strip() for s in _re.split(r"(?<=[.!?…])\s+", clip_text) if s.strip()]
    hits = 0
    for s in sentences:
        words = {w.lower() for w in _re.findall(r"[A-Za-z]+", s)}
        if query_words & words:
            hits += 1
    total = len(sentences) or 1
    return hits / total, hits, total


# --------------------------------------------------------------------------- #
# Answer 2: is the clip self-contained? i.e., does it start with a
# back-reference connector that assumes prior context the viewer won't have?
# --------------------------------------------------------------------------- #


# First-word (or first-phrase) indicators that the sentence is continuing
# a thread from a previous sentence the viewer didn't hear. Sourced from
# style-guide lists of "transitional" / "referential" openers.
_BACK_REFERENCE_OPENERS: frozenset[str] = frozenset({
    "basically", "naturally", "therefore", "thus", "however", "also",
    "moreover", "additionally", "furthermore", "so", "then", "next",
    "but", "and", "or", "yet", "still", "anyway", "anyways",
    "meanwhile", "likewise", "similarly", "conversely", "alternatively",
    "otherwise", "instead", "rather", "whereas",
})
_BACK_REFERENCE_BIGRAMS: frozenset[tuple[str, str]] = frozenset({
    ("to", "take"),     # "To take a simpler example..."
    ("for", "example"), ("for", "instance"),
    ("in", "other"),     # "In other words..."
    ("that", "is"),      ("that's", "why"), ("thats", "why"),
    ("but", "better"),
    ("in", "contrast"),
    ("in", "summary"),   ("to", "summarize"),
    ("which", "is"),     ("which", "means"),
    ("so", "that"),
})


def self_contained_check(clip_text: str) -> tuple[bool, str]:
    """True when the clip's opening doesn't depend on prior context."""
    import re as _re
    words = _re.findall(r"[A-Za-z']+", clip_text.lower())
    if len(words) < 2:
        return True, "too-short-to-judge"
    w0, w1 = words[0], words[1]
    if (w0, w1) in _BACK_REFERENCE_BIGRAMS:
        return False, f"bigram opener {(w0, w1)!r}"
    if w0 in _BACK_REFERENCE_OPENERS:
        return False, f"back-ref opener {w0!r}"
    return True, "ok"


# --------------------------------------------------------------------------- #
# Answer 3: precision check — does t_start sit exactly between the previous
# sentence's last word and the clip's first word? does t_end sit exactly at
# the clip's last word's offset?
# --------------------------------------------------------------------------- #


def precision_report(
    cues: list[IngestTranscriptCue], t_start: float, t_end: float,
) -> str:
    """Report the exact millisecond math at both boundaries."""
    flat: list[tuple[float, float, str]] = []
    for cue in cues:
        for w in cue.words:
            flat.append((float(w.start), float(w.end), str(w.text)))
    flat.sort(key=lambda x: x[0])

    # Find first word inside the clip.
    first_i = None
    for i, (ws, we, _) in enumerate(flat):
        mid = 0.5 * (ws + we)
        if t_start - 0.05 <= mid <= t_end + 0.05:
            first_i = i
            break
    last_i = None
    for i in range(len(flat) - 1, -1, -1):
        ws, we, _ = flat[i]
        mid = 0.5 * (ws + we)
        if t_start - 0.05 <= mid <= t_end + 0.05:
            last_i = i
            break
    if first_i is None or last_i is None:
        return "no words in clip"

    lines: list[str] = []

    # Start boundary
    first_word_start = flat[first_i][0]
    if first_i > 0:
        prev_word_text = flat[first_i - 1][2]
        prev_word_end = flat[first_i - 1][1]
        gap = first_word_start - prev_word_end
        expected_midpoint = prev_word_end + 0.5 * gap
        ms_from_midpoint = (t_start - expected_midpoint) * 1000.0
        lines.append(
            f"  START: prev={prev_word_text!r} ends at {prev_word_end:.3f}s, "
            f"clip-first={flat[first_i][2]!r} starts at {first_word_start:.3f}s "
            f"(gap {gap * 1000:.0f}ms); expected snap midpoint {expected_midpoint:.3f}s, "
            f"actual t_start={t_start:.3f}s "
            f"(Δ {ms_from_midpoint:+.1f}ms from midpoint)"
        )
    else:
        lines.append(f"  START: clip at video edge, t_start={t_start:.3f}s")

    # End boundary
    last_word_text = flat[last_i][2]
    last_word_end = flat[last_i][1]
    if last_i + 1 < len(flat):
        next_word_start = flat[last_i + 1][0]
        gap = next_word_start - last_word_end
        expected_midpoint = last_word_end + 0.5 * gap
        ms_from_midpoint = (t_end - expected_midpoint) * 1000.0
        lines.append(
            f"  END:   clip-last={last_word_text!r} ends at {last_word_end:.3f}s, "
            f"next={flat[last_i + 1][2]!r} starts at {next_word_start:.3f}s "
            f"(gap {gap * 1000:.0f}ms); expected snap midpoint {expected_midpoint:.3f}s, "
            f"actual t_end={t_end:.3f}s "
            f"(Δ {ms_from_midpoint:+.1f}ms from midpoint)"
        )
    else:
        lines.append(f"  END:   clip at video edge, t_end={t_end:.3f}s")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    video_id = os.environ.get("VIDEO_ID", DEFAULT_VIDEO_ID)
    query = os.environ.get("QUERY", DEFAULT_QUERY)

    print(f"=== User search simulation ===")
    print(f"  query:    {query!r}")
    print(f"  video_id: {video_id} (https://www.youtube.com/watch?v={video_id})")
    print(f"  settings: min={USER_MIN_SEC}s max={USER_MAX_SEC}s target={USER_TARGET_SEC}s")
    print()

    print("--- fetching transcript via youtube_transcript_api ---")
    try:
        raw = fetch_transcript(video_id)
    except Exception as exc:
        print(f"FETCH FAILED: {type(exc).__name__}: {exc}")
        print("This usually means YouTube is IP-blocking the transcript API from")
        print("this machine (proxy/residential IP, not cloud datacenter).")
        return 1
    print(f"  {len(raw)} cues, {raw[-1]['start'] + raw[-1]['duration']:.1f}s video")
    print()

    tc_cues = build_transcript_cues(raw)
    ingest_cues = build_ingest_cues(raw)

    print("--- running cut_video_into_topic_reels ---")
    print("  (will try Gemini → Groq → Cerebras for topic segmentation,")
    print("   then strong heuristic if all 3 unavailable)")
    print()

    # NOTE: intentionally pass ingest_cues_for_precision=None to mirror
    # exactly what the production search path does. reels.py does NOT
    # provide real Whisper word timings when generating clips from search
    # — those only come from the ingestion pipeline. The Phase A fix
    # inside cut_video_into_topic_reels synthesizes proportional ingest
    # cues from the transcript automatically, so the inner pickers still
    # run. We keep the locally-built `ingest_cues` only for the post-hoc
    # precision audit below (reconstruct clip text from word timings).
    classification, topic_reels = cut_video_into_topic_reels(
        video_id,
        query=query,
        duration_sec=raw[-1]["start"] + raw[-1]["duration"],
        use_llm=True,
        refine_boundaries=True,
        transcript=tc_cues,
        info_dict=None,  # no chapters available
        ingest_cues_for_precision=None,
        silence_ranges=None,
        user_min_sec=USER_MIN_SEC,
        user_max_sec=USER_MAX_SEC,
        user_target_sec=USER_TARGET_SEC,
    )

    print(f"=== Classification ===")
    print(f"  is_short: {classification.is_short}")
    print(f"  duration: {classification.duration_sec:.1f}s")
    print(f"  reason:   {classification.reason}")
    print()

    if not topic_reels:
        print("No reels produced. End of simulation.")
        return 0

    print(f"=== {len(topic_reels)} reel(s) ===\n")
    for i, r in enumerate(topic_reels, 1):
        prev_ctx, clip_text, next_ctx = extract_clip_text(ingest_cues, r.t_start, r.t_end)
        sop, eop, contract_note = check_punct_contract(ingest_cues, r.t_start, r.t_end)
        start_mark = "✓" if sop else "✗"
        end_mark = "✓" if eop else "✗"
        # Core-idea coverage
        core_ratio, core_hits, core_total = central_core_score(clip_text, query)
        core_mark = "✓" if core_ratio >= 0.35 else ("~" if core_ratio >= 0.15 else "✗")
        # Self-contained check
        self_contained, sc_note = self_contained_check(clip_text)
        sc_mark = "✓" if self_contained else "✗"
        # Precision report
        prec = precision_report(ingest_cues, r.t_start, r.t_end)
        # Show first 120 chars of text + last 80
        head = clip_text[:120]
        tail = clip_text[-80:] if len(clip_text) > 120 else ""
        print(f"[{i}] t={r.t_start:.3f}-{r.t_end:.3f} ({r.duration_sec:.1f}s)")
        print(f"    label:     {r.label}")
        print(f"    quality:   {r.boundary_quality}")
        print(f"    PUNCT CONTRACT:    start={start_mark}  end={end_mark}  ({contract_note})")
        print(f"    CORE IDEA:         {core_mark}  {core_hits}/{core_total} sentences mention query terms ({100 * core_ratio:.0f}%)")
        print(f"    SELF-CONTAINED:    {sc_mark}  ({sc_note})")
        print(f"    PRECISION:")
        print(prec)
        print(f"    context before:    ...{prev_ctx!r}")
        print(f"    clip head:         {head!r}...")
        if tail:
            print(f"    clip tail:         ...{tail!r}")
        print(f"    context after:     {next_ctx!r}...")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
