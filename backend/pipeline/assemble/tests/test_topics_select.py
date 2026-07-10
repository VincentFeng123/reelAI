# backend/pipeline/assemble/tests/test_topics_select.py
import backend.pipeline.assemble.topics as T
from backend.pipeline.understand.models import ContentMap, ContentNode, Structure
from backend.pipeline.sentences import Sentence


def _sent(idx, text, start, end, term="."):
    return Sentence(idx, text, start, end, term, term in (".", "?", "!"), idx, idx, 1.0, ())


def _structure(nodes):
    return Structure(video_id="vid", content_map=ContentMap(nodes=nodes, engine="treeseg"))


def _node(nid, title, i0, i1, start, end):
    return ContentNode(node_id=nid, level="topic", title=title, summary=title,
                       start=start, end=end, sentence_range=(i0, i1), keywords=[])


# 4 topics: intro (drop), reflex arc (keep), sponsor promo (drop), pitch (keep)
NODES = [
    _node("t0", "Intro", 0, 0, 0.0, 30.0),
    _node("t1", "Reflex arc", 1, 1, 30.0, 120.0),
    _node("t2", "Subscribe promo", 2, 2, 120.0, 140.0),
    _node("t3", "Pitch and frequency", 3, 3, 140.0, 220.0),
]
SENTS = [_sent(0, "Welcome to the channel.", 0, 30),
         _sent(1, "A reflex arc lets you react without thinking.", 30, 120),
         _sent(2, "Smash subscribe and hit the bell.", 120, 140),
         _sent(3, "Frequency determines the pitch of a sound.", 140, 220)]

JUDGMENTS = T.TopicSelection(topics=[
    T.TopicJudgment(node_id="t0", type="intro", informativeness=0.1, self_contained=0.2,
                    topic_relevance=0.1),
    T.TopicJudgment(node_id="t1", type="teaching", informativeness=0.9, self_contained=0.8,
                    topic_relevance=0.9),
    T.TopicJudgment(node_id="t2", type="promo", informativeness=0.0, self_contained=0.1,
                    topic_relevance=0.1),
    T.TopicJudgment(node_id="t3", type="teaching", informativeness=0.7, self_contained=0.7,
                    topic_relevance=0.2),
])


def test_select_keeps_teaching_drops_filler(monkeypatch):
    monkeypatch.setattr(T, "llm_json", lambda *a, **k: JUDGMENTS)
    kept, dropped = T.select_topics(_structure(NODES), SENTS, {})
    assert [p.node.node_id for p in kept] == ["t1", "t3"]     # chronological
    assert {p.node.node_id for p in dropped} == {"t0", "t2"}


def test_select_respects_max_clips(monkeypatch):
    monkeypatch.setattr(T, "llm_json", lambda *a, **k: JUDGMENTS)
    kept, _ = T.select_topics(_structure(NODES), SENTS, {"max_clips": 1})
    assert [p.node.node_id for p in kept] == ["t1"]           # keep the single MOST informative (t1=0.9 > t3=0.7)


def test_select_never_zero_on_llm_failure(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("llm down")
    monkeypatch.setattr(T, "llm_json", boom)
    kept, _ = T.select_topics(_structure(NODES), SENTS, {})
    assert len(kept) >= 1                                     # unknown ⇒ neutral-teaching, never empty
    assert all("low_confidence_selection" in p.warnings for p in kept)  # outage routes through flagged fallback


def test_target_topic_filters_selection_and_reaches_prompt(monkeypatch):
    seen = {}

    def fake_llm(system, user, schema, **kwargs):
        seen["user"] = user
        return JUDGMENTS

    monkeypatch.setattr(T, "llm_json", fake_llm)
    kept, dropped = T.select_topics(_structure(NODES), SENTS, {}, topic="reflex arcs")
    assert [p.node.node_id for p in kept] == ["t1"]
    assert "TARGET TOPIC: reflex arcs" in seen["user"]
    assert {p.node.node_id for p in dropped} == {"t0", "t2", "t3"}


def test_target_topic_does_not_fallback_to_off_topic_clips(monkeypatch):
    off_topic = T.TopicSelection(topics=[
        T.TopicJudgment(node_id=n.node_id, type="teaching", informativeness=0.9,
                        self_contained=0.9, topic_relevance=0.1)
        for n in NODES
    ])
    monkeypatch.setattr(T, "llm_json", lambda *args, **kwargs: off_topic)
    kept, _ = T.select_topics(_structure(NODES), SENTS, {}, topic="quantum chromodynamics")
    assert kept == []
