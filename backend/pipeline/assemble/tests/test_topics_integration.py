# backend/pipeline/assemble/tests/test_topics_integration.py
"""End-to-end (mocked LLM): a 6-topic structure incl. intro/outro + one over-long topic.
Asserts the acceptance properties from the spec: <=CLIP_MAX_S, 0 filler, terminator ends,
chronological, mandatory spec keys."""
import backend.pipeline.assemble.topics as T
from backend import config
from backend.pipeline.understand.models import ContentMap, ContentNode, Structure
from backend.pipeline.sentences import Sentence


def _sent(idx, text, t):
    return Sentence(idx, text, float(t), float(t + 10), ".", True, idx, idx, 1.0, ())


# 60 sentences @10s each = 600s. Topics: intro, A, B(over-long 200s), C, promo-outro
SENTS = [_sent(i, f"Sentence number {i} makes a complete point.", i * 10) for i in range(60)]
NODES = [
    ContentNode(node_id="t0", level="topic", title="Intro",   summary="welcome", start=0,   end=30,  sentence_range=(0, 2)),
    ContentNode(node_id="t1", level="topic", title="Topic A", summary="concept", start=30,  end=120, sentence_range=(3, 11)),
    ContentNode(node_id="t2", level="topic", title="Topic B", summary="big",     start=120, end=320, sentence_range=(12, 31)),  # 200s
    ContentNode(node_id="t3", level="topic", title="Topic C", summary="concept", start=320, end=420, sentence_range=(32, 41)),
    ContentNode(node_id="t4", level="topic", title="Outro",   summary="subscribe", start=420, end=600, sentence_range=(42, 59)),
]

_TYPES = {"t0": "intro", "t1": "teaching", "t2": "teaching", "t3": "teaching", "t4": "outro"}
_SCORE = {"t0": 0.1, "t1": 0.9, "t2": 0.8, "t3": 0.7, "t4": 0.05}


def _fake_llm(system, user, schema, **kw):
    if schema is T.TopicSelection:
        # one judgment per id present in the prompt
        js = [T.TopicJudgment(node_id=nid, type=_TYPES[nid], informativeness=_SCORE[nid],
                              self_contained=0.7, topic_relevance=0.9)
              for nid in _TYPES if f"[{nid}]" in user]
        return T.TopicSelection(topics=js)
    # WindowChoice: open at the shown lo, close ~5 sentences later (over-reaches for t2)
    first = user.split("[", 1)[1].split("]", 1)[0]
    lo = int(first)
    return T.WindowChoice(start_idx=lo, end_idx=lo + 18, title="win")  # deliberately long


def test_integration_properties(monkeypatch):
    monkeypatch.setattr(T, "llm_json", _fake_llm)
    st = Structure(video_id="vid", content_map=ContentMap(nodes=NODES, engine="treeseg"))
    specs, notes, rejections = T.assemble_topic_clips(
        st, "the subject", SENTS, "http://x", "vid", {}, adapter=None, stats={})

    assert 3 <= len(specs) <= config.TOPIC_MAX_CLIPS          # t1,t2,t3 kept
    titles = {r.title for r in rejections}
    assert "Intro" in titles and "Outro" in titles            # filler dropped
    for s in specs:
        assert s["end"] - s["start"] <= config.CLIP_MAX_S     # ceiling respected (t2 truncated)
        assert SENTS[s["sentence_end_idx"]].ends_with_period  # terminator close
        assert isinstance(s["start"], float) and isinstance(s["end"], float)
    starts = [s["start"] for s in specs]
    assert starts == sorted(starts)                           # chronological
    assert [s["sequence_index"] for s in specs] == list(range(1, len(specs) + 1))
