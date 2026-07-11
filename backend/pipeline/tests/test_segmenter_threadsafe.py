"""pysbd's Segmenter is not thread-safe: segment() stashes the input on self.original_text
and re-reads it, so a single shared instance corrupts concurrent callers. The boundary REFINE
pass (refine_clip_boundaries, REFINE_WORKERS>1) fans out per-clip Whisper windows that each
call segment_sentences(); _get_segmenter() is therefore thread-LOCAL.

This test drives the REAL segment_sentences() concurrently (the lever's own order test
monkeypatches _whisper_window and never touches this path). It asserts every concurrent
result equals the serial ground truth — it FAILS if the segmenter is shared across threads
(reproduces the misalign/drop race) and PASSES with the thread-local instance.
"""
from concurrent.futures import ThreadPoolExecutor

from backend.pipeline.sentences import segment_sentences

# Distinct multi-sentence docs of varying length/content: if two threads race on one shared
# Segmenter.original_text, one thread searches its sentences against the other's text and
# drops/misaligns them, so its result diverges from the serial segmentation.
_DOCS = [
    "The derivative measures the rate of change. It is the slope of the tangent line. "
    "We compute it as a limit.",
    "Newton and Leibniz both invented calculus. Their notations differ. Today we use both. "
    "The power rule is fundamental to differentiation.",
    "Consider a function f of x. Its integral accumulates area under the curve. "
    "The fundamental theorem links the two operations.",
    "Velocity is the derivative of position. Acceleration is the derivative of velocity. "
    "These are kinematic quantities. Units matter throughout.",
    "A limit describes behavior near a point. It need not equal the value at that point. "
    "Continuity requires that it does.",
    "The chain rule handles composition of functions. The product rule handles multiplication. "
    "The quotient rule follows directly from it.",
]


def test_segment_sentences_is_thread_safe_and_matches_serial():
    truth = {d: segment_sentences(d) for d in _DOCS}     # serial ground truth
    work = _DOCS * 30                                     # 180 calls → reliable contention

    def one(d):
        return d, segment_sentences(d)

    with ThreadPoolExecutor(max_workers=8) as pool:
        for d, res in pool.map(one, work):
            assert res == truth[d], f"concurrent segmentation diverged for {d[:40]!r}"
