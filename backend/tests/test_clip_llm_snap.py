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
