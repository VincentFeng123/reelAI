"""BND1 — free text-only boundary guards. Offline (no audio, no whisper, no LLM).

Two guards, both PREFERENCES with a SAFE FALLBACK (never a hard reject that could empty the
candidate set):
 (1) sentences_from_chunks no longer fabricates a terminator on every caption chunk — a real
     .?! makes a STRONG end (is_valid_end True), a bare chunk stays a usable-but-WEAK edge.
     On a fully-unpunctuated video the snapper FALLS BACK to chunk edges so a clip is never
     unplaceable; when real terminators exist they are strictly preferred.
 (2) _snap_one prefers an end whose final sentence has >=3 words and does NOT end on a
     conjunction/preposition; only-weak ends still ship, flagged weak_end_boundary. No clip
     is ever dropped by these guards.
"""
from __future__ import annotations

from backend.pipeline.refine import _is_weak_end, _snap_one, refine_and_snap
from backend.pipeline.sentences import Sentence, sentences_from_chunks


def _sent(idx: int, start: float, end: float, text: str, terminator: str = "",
          warnings: tuple = ()) -> Sentence:
    return Sentence(idx=idx, text=text, start=start, end=end, terminator=terminator,
                    ends_with_period=bool(terminator), word_start_idx=idx, word_end_idx=idx,
                    align_confidence=1.0, warnings=warnings)


_SNAP = dict(allow_qe=False, min_dur=1.0, tail_pad=0.05, max_dur=500.0)


# ── (1) sentences_from_chunks: honest terminators, no fabrication ──────────────
def test_chunk_terminators_reflect_real_punctuation():
    chunks = [
        {"start": 0.0, "end": 5.0, "text": "This one really ends here."},    # real '.'
        {"start": 5.0, "end": 10.0, "text": "but this one just trails off"},  # bare edge
        {"start": 10.0, "end": 15.0, "text": "does it stop?"},               # real '?'
    ]
    sents = sentences_from_chunks(chunks)
    assert [s.terminator for s in sents] == [".", "", "?"]
    assert [s.ends_with_period for s in sents] == [True, False, True]        # no fabricated '.'
    assert all("chunk_boundary" in s.warnings for s in sents)                # still usable edges
    assert sents[0].is_valid_end() and not sents[1].is_valid_end() and sents[2].is_valid_end()


# ── (1) safe fallback: an all-unpunctuated caption video STILL ships clips ─────
def test_unpunctuated_chunks_still_ship_via_fallback():
    chunks = [{"start": i * 10.0, "end": i * 10.0 + 9.5,
               "text": f"this is caption chunk number {i} running on with several words"}
              for i in range(4)]
    sents = sentences_from_chunks(chunks)
    assert all(not s.ends_with_period for s in sents)          # zero real terminators anywhere
    clip = _snap_one({"i_start": 0, "i_end": 2, "facet": "other"}, sents, **_SNAP)
    assert clip is not None                                    # placeable via chunk fallback
    assert clip["sentence_end_idx"] == 2                       # ends on the requested chunk edge
    assert "no_period_terminated_end" not in clip["warnings"]  # fallback, NOT a hard boundary fail


# ── (2) a real-terminator end is preferred over a fabricated chunk edge ────────
def test_real_terminator_preferred_over_chunk_edge():
    chunks = [
        {"start": 0.0, "end": 5.0, "text": "Hello there everyone."},                 # real '.'
        {"start": 5.0, "end": 10.0, "text": "this is a bare caption with no period"},  # bare edge
        {"start": 10.0, "end": 15.0, "text": "Another complete sentence here."},     # real '.'
    ]
    sents = sentences_from_chunks(chunks)
    assert [s.ends_with_period for s in sents] == [True, False, True]
    # a candidate whose end lands on the BARE chunk snaps OUT to the next real terminator —
    # the fabricated chunk edge is not accepted while real terminators exist.
    clip = _snap_one({"i_start": 0, "i_end": 1, "facet": "other"}, sents, **_SNAP)
    assert clip["sentence_end_idx"] == 2


# ── (2) a >=3-word non-conjunction end beats a weak/conjunction end when both exist ──
def test_strong_end_preferred_over_weak_conjunction_end():
    sents = [
        _sent(0, 0.0, 4.0, "We compute the force and.", terminator="."),      # ends on 'and' → weak
        _sent(1, 4.1, 8.0, "The result is fifteen newtons.", terminator="."),  # full clause → strong
    ]
    assert _is_weak_end(sents[0]) and not _is_weak_end(sents[1])
    clip = _snap_one({"i_start": 0, "i_end": 0, "facet": "other"}, sents, **_SNAP)
    assert clip["sentence_end_idx"] == 1                       # preferred the strong clause end
    assert "weak_end_boundary" not in clip["warnings"]


def test_two_word_end_preferred_against_when_alternative_exists():
    sents = [
        _sent(0, 0.0, 4.0, "It is.", terminator="."),                          # 2 words → weak
        _sent(1, 4.1, 8.0, "Acceleration stays perfectly constant.", terminator="."),  # strong
    ]
    clip = _snap_one({"i_start": 0, "i_end": 0, "facet": "other"}, sents, **_SNAP)
    assert clip["sentence_end_idx"] == 1
    assert "weak_end_boundary" not in clip["warnings"]


# ── (2) the ONLY-weak-end case still places the clip (flagged), never rejected ─
def test_only_weak_end_still_places_with_warning():
    sents = [_sent(0, 0.0, 4.0, "Momentum is.", terminator=".")]              # 2 words → weak, only end
    clip = _snap_one({"i_start": 0, "i_end": 0, "facet": "other"}, sents, **_SNAP)
    assert clip is not None                                    # never unplaceable
    assert clip["sentence_end_idx"] == 0
    assert "weak_end_boundary" in clip["warnings"]             # surfaced, not gated


def test_only_conjunction_ends_available_still_places():
    sents = [
        _sent(0, 0.0, 4.0, "We add the forces and.", terminator="."),          # ends on 'and'
        _sent(1, 4.1, 8.0, "then we divide by.", terminator="."),              # ends on 'by'
    ]
    clip = _snap_one({"i_start": 0, "i_end": 0, "facet": "other"}, sents, **_SNAP)
    assert clip is not None and clip["sentence_end_idx"] == 0
    assert "weak_end_boundary" in clip["warnings"]             # no strong end nearby → flagged, kept


# ── the guards NEVER drop a clip (no Rejection path here; refine keeps them all) ──
def test_guards_never_drop_a_clip():
    chunks = [{"start": i * 10.0, "end": i * 10.0 + 9.0,
               "text": f"caption chunk {i} running on without any terminal punctuation here"}
              for i in range(6)]
    sents = sentences_from_chunks(chunks)
    cands = [{"i_start": i, "i_end": i, "start": sents[i].start, "end": sents[i].end,
              "facet": f"f{i}"} for i in range(6)]
    out = refine_and_snap(cands, sents, {"min_clip_duration_s": 1.0,
                                         "max_clip_duration_s": 500.0, "tail_pad_s": 0.05,
                                         "max_clips": 20})
    assert len(out) == 6                                       # every disjoint clip survives


def test_real_terminators_never_flagged_weak_when_clause_complete():
    # a normal punctuated video: complete clauses end cleanly, no weak flag, span unchanged.
    sents = [
        _sent(0, 0.0, 4.0, "Newton's second law relates force and acceleration.", terminator="."),
        _sent(1, 4.1, 8.0, "The mass is the constant of proportionality.", terminator="."),
    ]
    clip = _snap_one({"i_start": 0, "i_end": 1, "facet": "other"}, sents, **_SNAP)
    assert clip["sentence_end_idx"] == 1
    assert "weak_end_boundary" not in clip["warnings"]
    assert "no_period_terminated_end" not in clip["warnings"]
