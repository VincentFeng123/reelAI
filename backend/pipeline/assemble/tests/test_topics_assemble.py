import backend.pipeline.assemble.topics as T
from backend.pipeline.understand.models import ContentMap, ContentNode, Structure
from backend.pipeline.sentences import Sentence


def _sent(idx, text, start, end, term="."):
    return Sentence(idx, text, start, end, term, term in ".?!", idx, idx, 1.0, ())


def _node(nid, title, i0, i1, start, end):
    return ContentNode(node_id=nid, level="topic", title=title, summary=title,
                       start=start, end=end, sentence_range=(i0, i1), keywords=[])


NODES = [_node("t0", "Intro", 0, 1, 0.0, 10.0),
         _node("t1", "Reflex arc", 1, 3, 10.0, 30.0),
         _node("t2", "Pitch", 3, 5, 30.0, 50.0)]
SENTS = [_sent(0, "Welcome everyone.", 0, 10),
         _sent(1, "A reflex arc lets you react.", 10, 20),
         _sent(2, "Touch a hot stove and pull back.", 20, 30),
         _sent(3, "Frequency sets the pitch.", 30, 40),
         _sent(4, "High frequency is high pitch.", 40, 50)]


def _fake_select(structure, sentences, settings):
    keep = [T.TopicPick(NODES[1], "teaching", 0.9, 0.8, "mechanism"),
            T.TopicPick(NODES[2], "teaching", 0.7, 0.7, "definition")]
    drop = [T.TopicPick(NODES[0], "intro", 0.1, 0.2, "welcome")]
    return keep, drop


def _fake_window(pick, sentences, settings):
    i0, i1 = pick.node.sentence_range
    return T.Window(pick.node.node_id, i0, i1 - 1, sentences[i0].start, sentences[i1 - 1].end,
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


def test_assemble_empty_when_no_topics(monkeypatch):
    st = Structure(video_id="vid", content_map=ContentMap(nodes=[], engine="treeseg"))
    specs, notes, rejections = T.assemble_topic_clips(
        st, "x", SENTS, "u", "vid", {}, adapter=None, stats={})
    assert specs == [] and "segment" in notes.lower()
