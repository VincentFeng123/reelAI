# backend/pipeline/assemble/tests/test_topics_window.py
import backend.pipeline.assemble.topics as T
from backend.pipeline.understand.models import ContentNode
from backend.pipeline.sentences import Sentence


def _sent(idx, text, start, end, term="."):
    return Sentence(idx, text, start, end, term, term in ".?!", idx, idx, 1.0, ())


def _pick(i0, i1, start, end, title="Topic"):
    node = ContentNode(node_id="t1", level="topic", title=title, summary=title,
                       start=start, end=end, sentence_range=(i0, i1), keywords=[])
    return T.TopicPick(node, "teaching", 0.9, 0.8, "")


# 6 sentences, ~10s each. Topic node covers [1,6): first sentence (idx1) dangles.
SENTS = [
    _sent(0, "A reflex arc lets you react without thinking.", 0, 10),
    _sent(1, "These neurons carry the signal.", 10, 20),       # dangling opener
    _sent(2, "The pathway runs sensory to motor.", 20, 30),
    _sent(3, "Touch a hot stove and your hand pulls back.", 30, 40),
    _sent(4, "That is the reflex arc at work.", 40, 50),
    _sent(5, "Anyway, moving on to something else.", 50, 60),
]


def test_window_moves_start_off_dangling_opener(monkeypatch):
    # LLM chooses to open at the framing sentence (idx0), close at idx4
    monkeypatch.setattr(T, "llm_json", lambda *a, **k: T.WindowChoice(start_idx=0, end_idx=4, title="Reflex arc"))
    w = T.extract_best_window(_pick(1, 6, 10.0, 60.0), SENTS, {})
    assert w.start_idx == 0 and w.end_idx == 4
    assert w.start_s == 0.0 and w.end_s == 50.0


def test_window_truncates_to_budget(monkeypatch):
    # LLM over-reaches (0..5 = 60s); CLIP_MAX_S small ⇒ walk end back to a terminator within budget
    monkeypatch.setattr(T, "llm_json", lambda *a, **k: T.WindowChoice(start_idx=0, end_idx=5))
    w = T.extract_best_window(_pick(0, 6, 0.0, 60.0), SENTS, {"clip_max_s": 35.0})
    assert w.end_s - w.start_s <= 35.0
    assert SENTS[w.end_idx].ends_with_period
    assert "window_truncated_to_budget" in w.warnings


def test_window_snaps_end_to_terminator(monkeypatch):
    seq = [_sent(0, "First point.", 0, 10),
           _sent(1, "Second point that trails off", 10, 20, term=""),   # no terminator
           _sent(2, "Third point ends here.", 20, 30)]
    node = ContentNode(node_id="t1", level="topic", title="X", start=0, end=30, sentence_range=(0, 3))
    pick = T.TopicPick(node, "teaching", 0.9, 0.8, "")
    monkeypatch.setattr(T, "llm_json", lambda *a, **k: T.WindowChoice(start_idx=0, end_idx=1))
    w = T.extract_best_window(pick, seq, {})
    assert seq[w.end_idx].ends_with_period          # walked back to idx0


def test_window_clamps_out_of_range(monkeypatch):
    monkeypatch.setattr(T, "llm_json", lambda *a, **k: T.WindowChoice(start_idx=99, end_idx=999))
    w = T.extract_best_window(_pick(1, 6, 10.0, 60.0), SENTS, {})
    assert 0 <= w.start_idx <= w.end_idx <= len(SENTS) - 1
