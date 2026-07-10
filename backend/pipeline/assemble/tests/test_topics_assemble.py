import backend.pipeline.assemble.topics as T
from backend import config
from backend.pipeline.understand.content_map import _assemble_treeseg, TopicLLM
from backend.pipeline.understand.models import ContentMap, ContentNode, Structure
from backend.pipeline.sentences import Sentence


def _sent(idx, text, start, end, term="."):
    return Sentence(idx, text, start, end, term, term in (".", "?", "!"), idx, idx, 1.0, ())


def _node(nid, title, i0, i1, start, end):
    return ContentNode(node_id=nid, level="topic", title=title, summary=title,
                       start=start, end=end, sentence_range=(i0, i1), keywords=[])


NODES = [_node("t0", "Intro", 0, 0, 0.0, 10.0),
         _node("t1", "Reflex arc", 1, 2, 10.0, 30.0),
         _node("t2", "Pitch", 3, 4, 30.0, 50.0)]
SENTS = [_sent(0, "Welcome everyone.", 0, 10),
         _sent(1, "A reflex arc lets you react.", 10, 20),
         _sent(2, "Touch a hot stove and pull back.", 20, 30),
         _sent(3, "Frequency sets the pitch.", 30, 40),
         _sent(4, "High frequency is high pitch.", 40, 50)]


def _fake_select(structure, sentences, settings, topic=""):
    keep = [T.TopicPick(NODES[1], "teaching", 0.9, 0.8, "mechanism"),
            T.TopicPick(NODES[2], "teaching", 0.7, 0.7, "definition")]
    drop = [T.TopicPick(NODES[0], "intro", 0.1, 0.2, "welcome")]
    return keep, drop


def _fake_window(pick, sentences, settings):
    i0, i1 = pick.node.sentence_range
    return T.Window(pick.node.node_id, i0, i1, sentences[i0].start, sentences[i1].end,
                    pick.node.title, pick.type, pick.why, ())


def test_assemble_builds_specs_drops_filler(monkeypatch):
    monkeypatch.setattr(T, "select_topics", _fake_select)
    monkeypatch.setattr(T, "extract_best_window", _fake_window)
    st = Structure(video_id="vid", content_map=ContentMap(nodes=NODES, engine="treeseg"))
    stats = {}
    specs, notes, rejections = T.assemble_topic_clips(
        st, "reflexes", SENTS, "http://x", "vid", {}, adapter=None, stats=stats)
    assert len(specs) == 2
    assert [s["title"] for s in specs] == ["Reflex arc", "Pitch"]     # chronological
    assert [s["sequence_index"] for s in specs] == [1, 2]
    for s in specs:                                                   # mandatory keys present
        assert isinstance(s["start"], float) and isinstance(s["end"], float)
        assert s["cut_end"] >= s["end"]
    assert [r.stage for r in rejections] == ["topic_select"]          # the dropped intro
    assert stats["n_topics_kept"] == 2 and stats["n_topics_total"] == 3
    assert stats["n_topics_dropped"] == 1


def test_assemble_empty_when_no_topics(monkeypatch):
    st = Structure(video_id="vid", content_map=ContentMap(nodes=[], engine="treeseg"))
    specs, notes, rejections = T.assemble_topic_clips(
        st, "x", SENTS, "u", "vid", {}, adapter=None, stats={})
    assert specs == [] and "segment" in notes.lower()


def test_serial_extract_best_window_failure_isolated(monkeypatch):
    """Verify that extract_best_window raising in serial path (workers==1) doesn't crash."""
    call_count = [0]

    def _extract_raises_on_first(pick, sentences, settings):
        """Raises for first pick (Reflex arc), returns valid Window for others."""
        call_count[0] += 1
        if call_count[0] == 1:  # First call: Reflex arc node
            raise ValueError("Simulated extraction failure")
        # Second call: Pitch node — return a valid window
        i0, i1 = pick.node.sentence_range
        return T.Window(pick.node.node_id, i0, i1, sentences[i0].start, sentences[i1].end,
                        pick.node.title, pick.type, pick.why, ())

    monkeypatch.setattr(config, "UNDERSTAND_WORKERS", 1)  # Force serial path
    monkeypatch.setattr(T, "select_topics", _fake_select)
    monkeypatch.setattr(T, "extract_best_window", _extract_raises_on_first)

    st = Structure(video_id="vid", content_map=ContentMap(nodes=NODES, engine="treeseg"))
    specs, notes, rejections = T.assemble_topic_clips(
        st, "test", SENTS, "http://x", "vid", {}, adapter=None, stats={})

    # Should NOT raise, should return 1 spec (Pitch), and skip Reflex arc
    assert len(specs) == 1
    assert specs[0]["title"] == "Pitch"


def test_real_treeseg_inclusive_content_map_reaches_topic_assembly(monkeypatch):
    """TreeSeg emits inclusive ranges; assembly must include each topic's closing sentence."""
    import re

    sents = SENTS[:4]
    labels = [TopicLLM(title="First"), TopicLLM(title="Second")]
    content_map = _assemble_treeseg(sents, [(0, 1), (2, 3)], labels, [(0, 1)])
    assert [node.sentence_range for node in content_map.topics()] == [(0, 1), (2, 3)]
    seen = {}

    def fake_llm(system, user, schema, **kwargs):
        if schema is T.TopicSelection:
            seen["selection_prompt"] = user
            return T.TopicSelection(topics=[
                T.TopicJudgment(node_id=node.node_id, type="teaching", informativeness=0.9,
                                self_contained=0.9, topic_relevance=0.9)
                for node in content_map.topics()
            ])
        indices = [int(value) for value in re.findall(r"^\[(\d+)\]", user, re.MULTILINE)]
        return T.WindowChoice(start_idx=min(indices), end_idx=max(indices), title="window")

    monkeypatch.setattr(T, "llm_json", fake_llm)
    monkeypatch.setattr(config, "TOPIC_BOUNDARY_WINDOW", 0)
    specs, _, _ = T.assemble_topic_clips(
        Structure(video_id="vid", content_map=content_map),
        "mechanics", sents, "http://x", "vid", {}, adapter=None, stats={},
    )

    assert [(spec["sentence_start_idx"], spec["sentence_end_idx"]) for spec in specs] == [(0, 1), (2, 3)]
    assert [(spec["start"], spec["end"]) for spec in specs] == [(0.0, 20.0), (20.0, 40.0)]
    assert SENTS[1].text in seen["selection_prompt"]
    assert SENTS[3].text in seen["selection_prompt"]
    assert "TARGET TOPIC: mechanics" in seen["selection_prompt"]
