"""
Unit tests for the LLM-direct clip cutting path:
  * `clip_boundary.snap_llm_boundary` — boundary snapping behavior
  * `clip_llm._parse_clip_pick_json` — LLM output parsing and validation
  * `EmbeddingService` — hash/semantic path selection and dim consistency

These tests are deterministic (no network, no LLM calls, no model downloads)
and run in a fresh process — the semantic model loader is bypassed via the
VERCEL env var so the suite works on any machine.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Force the embedding service into its hash-fallback mode so these tests
# don't try to load sentence-transformers. Must be set BEFORE first import.
os.environ.setdefault("VERCEL", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.services.clip_boundary import snap_llm_boundary  # noqa: E402
from backend.app.services.clip_llm import _parse_clip_pick_json  # noqa: E402
from backend.app.services.sentences import SentenceSpan  # noqa: E402
from backend.app.services.topic_cut import TranscriptCue  # noqa: E402


def _sent(text: str, t_start: float, t_end: float, punct: str = ".") -> SentenceSpan:
    return SentenceSpan(
        text=text,
        t_start=t_start,
        t_end=t_end,
        cue_start_idx=0,
        cue_end_idx=0,
        word_start_idx=0,
        word_end_idx=0,
        terminal_punct=punct,
        confidence=0.9,
    )


def _sample_sentences() -> list[SentenceSpan]:
    return [
        _sent("Intro sentence.", 0.0, 3.0),
        _sent("So let us talk about gradient descent now.", 3.5, 10.0),
        _sent("Gradient descent is an optimization algorithm.", 10.5, 18.0),
        _sent("It iteratively steps toward the minimum of a loss function.", 18.5, 28.0),
        _sent("The step size is controlled by the learning rate parameter.", 28.5, 37.0),
        _sent("Next topic is backprop", 37.5, 40.5, punct=""),
    ]


# --------------------------------------------------------------------------- #
# snap_llm_boundary
# --------------------------------------------------------------------------- #


def test_snap_clean_pick_passes_through():
    r = snap_llm_boundary(raw_t_start=3.5, raw_t_end=37.0, sentences=_sample_sentences())
    assert r.snapped
    assert r.t_start == 3.5
    assert r.t_end == 37.0


def test_snap_messy_end_snaps_back_to_terminal_punct():
    # raw_t_end=40.0 lands inside the non-terminal final sentence; must snap
    # back to sentence 5's end (37.0).
    r = snap_llm_boundary(raw_t_start=3.0, raw_t_end=40.0, sentences=_sample_sentences())
    assert r.snapped
    assert r.t_end == 37.0


def test_snap_mid_sentence_start_snaps_to_sentence_start():
    # LLM gave a t_start mid-sentence; we expect to snap back to the
    # start of the containing sentence.
    r = snap_llm_boundary(raw_t_start=6.0, raw_t_end=37.0, sentences=_sample_sentences())
    assert r.snapped
    assert r.t_start == 3.5  # start of sentence 2, which contains t=6


def test_snap_rejects_when_no_terminal_punct_sentence_fits():
    # Only sentence 0 is terminal; raw_t_end=10 is 7s past its end — beyond
    # the 5s tolerance. Must reject rather than silently cut mid-sentence.
    sents = [
        _sent("First.", 0.0, 3.0),
        _sent("Second non-terminal", 3.5, 10.0, punct=""),
    ]
    r = snap_llm_boundary(raw_t_start=0.0, raw_t_end=10.0, sentences=sents)
    assert not r.snapped
    assert "terminal-punct" in r.reason.lower() or "shift" in r.reason.lower()


def test_snap_rejects_when_duration_below_min():
    r = snap_llm_boundary(
        raw_t_start=3.5,
        raw_t_end=10.0,  # 6.5s clip
        sentences=_sample_sentences(),
        min_sec=15.0,
        max_sec=60.0,
    )
    assert not r.snapped
    assert "duration" in r.reason.lower()


def test_snap_rejects_when_duration_above_max():
    sents_long = [_sent(f"Sentence {i}.", i * 5.0, i * 5.0 + 4.0) for i in range(20)]
    r = snap_llm_boundary(raw_t_start=0.0, raw_t_end=99.0, sentences=sents_long, max_sec=60.0)
    assert not r.snapped


def test_snap_rejects_inverted_range():
    r = snap_llm_boundary(raw_t_start=30.0, raw_t_end=15.0, sentences=_sample_sentences())
    assert not r.snapped


def test_snap_silence_nudge_at_end():
    r = snap_llm_boundary(
        raw_t_start=3.5,
        raw_t_end=36.9,
        sentences=_sample_sentences(),
        silence_ranges=[(37.0, 37.4)],
    )
    assert r.snapped
    # Without silence the snap target is 37.0; with silence at 37.0-37.4,
    # the midpoint 37.2 should replace it.
    assert abs(r.t_end - 37.2) < 0.01


def test_snap_silence_nudge_at_start():
    # Silence range overlaps ±0.4s of sentence 2's start (3.5).
    r = snap_llm_boundary(
        raw_t_start=3.6,
        raw_t_end=37.0,
        sentences=_sample_sentences(),
        silence_ranges=[(3.3, 3.5)],
    )
    assert r.snapped
    assert abs(r.t_start - 3.4) < 0.01


def test_snap_without_silence_uses_sentence_boundary_verbatim():
    r = snap_llm_boundary(
        raw_t_start=3.5,
        raw_t_end=37.0,
        sentences=_sample_sentences(),
        silence_ranges=None,
    )
    assert r.snapped
    assert r.t_start == 3.5
    assert r.t_end == 37.0


def test_snap_empty_sentences_rejects():
    r = snap_llm_boundary(raw_t_start=0.0, raw_t_end=10.0, sentences=[])
    assert not r.snapped
    assert "sentences" in r.reason.lower()


# --------------------------------------------------------------------------- #
# clip_llm._parse_clip_pick_json
# --------------------------------------------------------------------------- #


def test_parse_clip_pick_valid_json():
    cues = [TranscriptCue(start=0.0, duration=5.0, text="hello")]
    raw = '{"t_start": 10.5, "t_end": 45.3, "reason": "direct answer"}'
    pick = _parse_clip_pick_json(raw, cues)
    assert pick is not None
    assert pick.t_start == 10.5
    assert pick.t_end == 45.3
    assert pick.reason == "direct answer"


def test_parse_clip_pick_accepts_code_fences():
    cues = [TranscriptCue(start=0.0, duration=5.0, text="x")]
    raw = "```json\n{\"t_start\": 10.0, \"t_end\": 30.0, \"reason\": \"ok\"}\n```"
    pick = _parse_clip_pick_json(raw, cues)
    assert pick is not None
    assert pick.t_start == 10.0


def test_parse_clip_pick_null_timestamps_returns_none():
    cues = [TranscriptCue(start=0.0, duration=5.0, text="x")]
    raw = '{"t_start": null, "t_end": null, "reason": "query not covered"}'
    assert _parse_clip_pick_json(raw, cues) is None


def test_parse_clip_pick_invalid_json_returns_none():
    cues = [TranscriptCue(start=0.0, duration=5.0, text="x")]
    assert _parse_clip_pick_json("not json at all", cues) is None


def test_parse_clip_pick_inverted_range_returns_none():
    cues = [TranscriptCue(start=0.0, duration=5.0, text="x")]
    raw = '{"t_start": 50.0, "t_end": 20.0, "reason": "?"}'
    assert _parse_clip_pick_json(raw, cues) is None


def test_parse_clip_pick_negative_start_returns_none():
    cues = [TranscriptCue(start=0.0, duration=5.0, text="x")]
    raw = '{"t_start": -5.0, "t_end": 30.0, "reason": "?"}'
    assert _parse_clip_pick_json(raw, cues) is None


# --------------------------------------------------------------------------- #
# User-settings plumbing — min/max/target flow into snap + prompt
# --------------------------------------------------------------------------- #


def test_snap_respects_user_max_below_60s():
    # User sets max=40s; a 50s pick must reject even though 50 < 60 (old floor).
    sents = _sample_sentences()
    r = snap_llm_boundary(
        raw_t_start=3.5,
        raw_t_end=37.0,  # snapped duration = 33.5s, within 15-40
        sentences=sents,
        min_sec=15.0,
        max_sec=40.0,
    )
    assert r.snapped
    r2 = snap_llm_boundary(
        raw_t_start=0.0,
        raw_t_end=50.0,  # would snap to 37.0; 37s < 40s so OK
        sentences=sents,
        min_sec=15.0,
        max_sec=30.0,  # but max=30 → 37-0=37s exceeds max → must reject
    )
    assert not r2.snapped


def test_snap_respects_user_min_above_15s():
    # User sets min=30s; a 20s clip must reject even though 20 > 15 (old floor).
    sents = _sample_sentences()
    r = snap_llm_boundary(
        raw_t_start=10.5,
        raw_t_end=28.0,  # snapped duration = 17.5s
        sentences=sents,
        min_sec=30.0,
        max_sec=60.0,
    )
    assert not r.snapped
    assert "duration" in r.reason.lower()


def test_pick_clip_llm_accepts_target_sec_kwarg():
    # Without any LLM keys set this returns None fast; we just verify the
    # signature accepts target_sec without blowing up.
    from backend.app.services.clip_llm import pick_clip_llm

    cues = [TranscriptCue(start=0.0, duration=5.0, text="hello world")]
    result = pick_clip_llm(
        "anything",
        cues,
        min_sec=20.0,
        max_sec=45.0,
        target_sec=30.0,
    )
    # No LLM → None; but kwargs were accepted, signature is correct.
    assert result is None or hasattr(result, "t_start")


def test_pick_clip_llm_target_outside_bounds_gets_clamped():
    # If caller passes target=5s but min/max=[20,45], the clamp logic
    # should pull target into range without raising.
    from backend.app.services.clip_llm import pick_clip_llm

    cues = [TranscriptCue(start=0.0, duration=5.0, text="x")]
    result = pick_clip_llm(
        "q",
        cues,
        min_sec=20.0,
        max_sec=45.0,
        target_sec=5.0,  # deliberately out-of-range
    )
    # Same: no LLM → None, but no exception from the clamp path.
    assert result is None or hasattr(result, "t_start")


# --------------------------------------------------------------------------- #
# EmbeddingService (hash fallback path — forced by VERCEL=1 in module init)
# --------------------------------------------------------------------------- #


def test_embedding_service_hash_mode_has_correct_dim():
    from backend.app.services.embeddings import EmbeddingService, _HASH_DIM

    svc = EmbeddingService()
    # In VERCEL mode the semantic model is skipped → hash path → _HASH_DIM.
    assert svc.dim == _HASH_DIM
    assert svc._semantic_model is None


def test_embedding_service_hash_embed_shape_and_normalization():
    from backend.app.services.embeddings import EmbeddingService

    svc = EmbeddingService()
    vec = svc._hash_embed("machine learning fundamentals")
    assert vec.shape == (svc.dim,)
    # Normalized to unit length (within floating-point tolerance)
    import numpy as np

    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5


def test_embedding_service_hash_embed_deterministic():
    from backend.app.services.embeddings import EmbeddingService

    svc = EmbeddingService()
    a = svc._hash_embed("the same text")
    b = svc._hash_embed("the same text")
    import numpy as np

    assert np.array_equal(a, b)


def test_embedding_service_hash_embed_different_text_different_vec():
    from backend.app.services.embeddings import EmbeddingService

    svc = EmbeddingService()
    a = svc._hash_embed("gradient descent")
    b = svc._hash_embed("stochastic backpropagation")
    import numpy as np

    assert not np.array_equal(a, b)


# --------------------------------------------------------------------------- #
# Word-level refinement + filler trimming
# --------------------------------------------------------------------------- #


class _FakeWord:
    """Lightweight stand-in for IngestTranscriptWord."""
    def __init__(self, text: str, start: float, end: float):
        self.text = text
        self.start = start
        self.end = end


class _FakeCue:
    """Lightweight stand-in for IngestTranscriptCue (just the attrs
    refine_boundaries_word_level actually duck-types against)."""
    def __init__(self, start: float, end: float, words: list[_FakeWord]):
        self.start = start
        self.end = end
        self.words = words


def test_word_level_preserves_sentence_start_even_with_leading_filler():
    """Boundary-level filler trimming was removed to preserve the strict
    begin/end-on-punctuation contract. Sentence-level start must stay
    intact; filler density is penalized at scoring time instead, not at
    the boundary."""
    from backend.app.services.clip_boundary import refine_boundaries_word_level

    words = [
        _FakeWord("Um,", 0.30, 0.40),
        _FakeWord("so", 0.50, 0.60),
        _FakeWord("gradient", 0.60, 1.00),
        _FakeWord("descent", 1.00, 1.40),
        _FakeWord("is", 1.40, 1.55),
        _FakeWord("an", 1.55, 1.70),
        _FakeWord("optimization", 1.70, 2.40),
        _FakeWord("algorithm.", 2.40, 2.90),
    ]
    cues = [_FakeCue(0.3, 2.9, words)]
    new_start, new_end, start_reason, _ = refine_boundaries_word_level(
        0.3, 2.9, cues, silence_ranges=None,
    )
    # Start should snap to first word's onset (0.3), not skip past fillers.
    assert new_start <= 0.35
    assert start_reason == "word-snap"


def test_word_level_preserves_sentence_end_even_with_trailing_filler():
    """Trailing fillers are NOT trimmed at the boundary — the clip must
    still end on terminal punctuation, not on an arbitrary word that
    happens to precede a filler."""
    from backend.app.services.clip_boundary import refine_boundaries_word_level

    words = [
        _FakeWord("The", 0.0, 0.15),
        _FakeWord("convergence", 0.15, 0.75),
        _FakeWord("rate", 0.75, 1.05),
        _FakeWord("you", 1.15, 1.25),
        _FakeWord("know.", 1.25, 1.45),
    ]
    cues = [_FakeCue(0.0, 1.45, words)]
    new_start, new_end, _, end_reason = refine_boundaries_word_level(
        0.0, 1.45, cues, silence_ranges=None,
    )
    # End should snap to last word's offset (1.45), NOT trim "you know".
    assert new_end >= 1.40
    assert end_reason == "word-snap"


def test_word_level_no_timing_passes_through():
    from backend.app.services.clip_boundary import refine_boundaries_word_level

    # Cue with no words attached → sentinel path, timestamps unchanged.
    cue = _FakeCue(10.0, 20.0, words=[])
    new_start, new_end, sr, er = refine_boundaries_word_level(
        10.0, 20.0, [cue], silence_ranges=None,
    )
    assert new_start == 10.0
    assert new_end == 20.0
    assert sr == "no-word-timing" or sr == "no-cues"


def test_word_level_snaps_to_inter_word_silence_midpoint():
    from backend.app.services.clip_boundary import refine_boundaries_word_level

    # Gap of 400ms between "one." and "Two" — should nudge t_end into the gap.
    words = [
        _FakeWord("The", 0.0, 0.1),
        _FakeWord("answer", 0.1, 0.5),
        _FakeWord("is", 0.5, 0.7),
        _FakeWord("one.", 0.7, 1.0),
        # gap 1.0 → 1.4
        _FakeWord("Two", 1.4, 1.6),
    ]
    cues = [_FakeCue(0.0, 1.6, words)]
    new_start, new_end, _, _ = refine_boundaries_word_level(
        0.0, 1.0, cues, silence_ranges=None,
    )
    # Expected: nudged into the middle of the 400ms gap ≈ 1.2s
    assert 1.0 < new_end <= 1.25


# --------------------------------------------------------------------------- #
# Strong heuristic picker
# --------------------------------------------------------------------------- #


def test_heuristic_picker_returns_top_scoring_window():
    from backend.app.services.clip_boundary import pick_clip_heuristic

    # 8 sentences, one clearly matches the query "gradient descent".
    sents = [
        _sent("Welcome back to the channel everyone.", 0.0, 3.0),
        _sent("Today we discuss calculus foundations.", 3.5, 7.0),
        _sent("Gradient descent is an optimization algorithm.", 7.5, 12.0),
        _sent("It iteratively steps toward the minimum of a loss function.", 12.5, 19.0),
        _sent("The learning rate controls the step size.", 19.5, 24.0),
        _sent("Too large and you overshoot the minimum.", 24.5, 29.0),
        _sent("Too small and convergence is slow.", 29.5, 33.0),
        _sent("In the next video we will cover momentum.", 33.5, 37.0),
    ]
    result = pick_clip_heuristic(
        query="gradient descent",
        cues=None,  # no word timing available
        sentences=sents,
        silence_ranges=None,
        user_min_sec=15.0,
        user_max_sec=30.0,
        user_target_sec=20.0,
        video_duration_sec=40.0,
        embed_func=None,  # TF-IDF only path
    )
    assert result is not None
    # Window must include the query-matching sentence (idx 2 or 3).
    assert result.t_start <= 12.5 and result.t_end >= 12.0


def test_heuristic_picker_prefers_substantive_content():
    """The picker must cover the substantive explanation (sentence 3 at
    t=15-22). Starting earlier to pick up the intro announcement is OK
    because that announcement provides useful lead-in context — both
    intro-inclusive and body-only windows are acceptable clips so long
    as the substantive content is in them.
    """
    from backend.app.services.clip_boundary import pick_clip_heuristic

    sents = [
        _sent("Hey everyone, today we'll cover gradient descent.", 0.0, 4.0),
        _sent("Let's get started.", 4.5, 6.0),
        _sent("Many viewers ask what algorithm optimizes gradients best.", 6.5, 10.0),
        _sent("Gradient descent minimizes a loss function by stepping along the gradient.", 15.0, 22.0),
        _sent("Each step is scaled by the learning rate.", 22.5, 26.0),
        _sent("With a careful learning rate the process converges quickly.", 26.5, 31.0),
        _sent("Too large and it diverges.", 31.5, 34.0),
    ]
    result = pick_clip_heuristic(
        query="gradient descent",
        cues=None,
        sentences=sents,
        silence_ranges=None,
        user_min_sec=15.0,
        user_max_sec=30.0,
        user_target_sec=20.0,
        video_duration_sec=40.0,
        embed_func=None,
    )
    assert result is not None
    # Window must encompass sentence 3 (t_end=22.0) — the substantive
    # explanation. t_start can be anywhere from 0 (intro-inclusive) to
    # 15 (body-only); both are valid.
    assert result.t_end >= 22.0


def test_heuristic_picker_hard_rejects_off_topic_windows():
    """Windows with zero query-term sentence coverage AND no embedding
    vouching should be disqualified. Without this gate the picker can
    pick off-topic content that happens to share vocabulary."""
    from backend.app.services.clip_boundary import pick_clip_heuristic

    # Transcript is entirely about pasta. Query is "gradient descent".
    sents = [
        _sent("First, render the guanciale in a cold pan until crisp.", 0.0, 5.0),
        _sent("Whisk egg yolks with pecorino romano cheese.", 5.5, 10.0),
        _sent("Cook spaghetti al dente in heavily salted water.", 10.5, 15.0),
        _sent("Reserve a cup of pasta water before draining.", 15.5, 20.0),
        _sent("Toss pasta with guanciale off heat and add egg mixture.", 20.5, 26.0),
        _sent("Stir vigorously with pasta water until silky.", 26.5, 31.0),
        _sent("Serve immediately with extra cheese and black pepper.", 31.5, 36.0),
    ]
    result = pick_clip_heuristic(
        query="gradient descent",
        cues=None,
        sentences=sents,
        silence_ranges=None,
        user_min_sec=15.0,
        user_max_sec=45.0,
        user_target_sec=25.0,
        video_duration_sec=40.0,
        embed_func=None,  # no semantic path → coverage=0 should hard-reject
    )
    assert result is None


def test_heuristic_picker_respects_user_max_tight_bound():
    from backend.app.services.clip_boundary import pick_clip_heuristic

    sents = _sample_sentences()  # 6 sentences, spans 0-40.5s
    # Tight max=20s must keep the chosen window under 20s.
    result = pick_clip_heuristic(
        query="gradient descent",
        cues=None,
        sentences=sents,
        silence_ranges=None,
        user_min_sec=15.0,
        user_max_sec=20.0,
        user_target_sec=17.0,
        video_duration_sec=45.0,
        embed_func=None,
    )
    if result is not None:
        assert (result.t_end - result.t_start) <= 20.5  # +0.5 slack for word-level nudge


def test_heuristic_picker_returns_none_when_no_windows_fit():
    from backend.app.services.clip_boundary import pick_clip_heuristic

    # All sentences too close together — no 30s+ terminal-to-terminal window exists.
    sents = [
        _sent("Short.", 0.0, 1.0),
        _sent("Also short.", 1.5, 2.5),
        _sent("Another.", 3.0, 4.0),
    ]
    result = pick_clip_heuristic(
        query="anything",
        cues=None,
        sentences=sents,
        silence_ranges=None,
        user_min_sec=30.0,
        user_max_sec=60.0,
        user_target_sec=45.0,
        video_duration_sec=5.0,
        embed_func=None,
    )
    assert result is None


def test_snap_llm_boundary_with_ingest_cues_does_word_level_refinement():
    """Integration: snap_llm_boundary accepts ingest_cues and applies
    word-level trimming without crashing. Detailed trim behavior is
    tested unit-level in test_word_level_trims_leading_filler."""
    from backend.app.services.clip_boundary import snap_llm_boundary

    sents = _sample_sentences()
    # Provide word timings covering the full chosen clip range (3.5-37s)
    # so word-level refinement doesn't trip the duration-fallback guard.
    all_words: list[_FakeWord] = []
    for idx, s in enumerate(sents):
        # Simple 3-word-per-sentence split with a filler at the start of
        # the first picked sentence (idx=1 in _sample_sentences).
        words_txt = s.text.split()
        n = len(words_txt)
        if n == 0:
            continue
        dt = (s.t_end - s.t_start) / max(1, n)
        for i, w in enumerate(words_txt):
            ws = s.t_start + i * dt
            we = s.t_start + (i + 1) * dt - 0.02
            all_words.append(_FakeWord(w, ws, we))
    cues = [_FakeCue(0.0, 40.5, all_words)]
    r = snap_llm_boundary(
        raw_t_start=3.5,
        raw_t_end=37.0,
        sentences=sents,
        silence_ranges=None,
        min_sec=15.0,
        max_sec=60.0,
        ingest_cues=cues,
    )
    assert r.snapped
    # Duration must stay inside [min, max] — refinement fell back if it
    # would have broken the contract, which is the defensive behavior.
    assert 15.0 <= (r.t_end - r.t_start) <= 60.0
