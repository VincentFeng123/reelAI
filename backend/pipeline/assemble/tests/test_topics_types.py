# backend/pipeline/assemble/tests/test_topics_types.py
from backend.pipeline.assemble import topics as T

def test_schemas_and_dataclasses_exist():
    sel = T.TopicSelection(topics=[T.TopicJudgment(node_id="c0.t1", type="teaching",
                                                   informativeness=0.9, self_contained=0.8, why="core")])
    assert sel.topics[0].node_id == "c0.t1"
    ch = T.WindowChoice(start_idx=3, end_idx=9, title="Reflex arc", why="mechanism+example")
    assert (ch.start_idx, ch.end_idx) == (3, 9)
    w = T.Window(node_id="c0.t1", start_idx=3, end_idx=9, start_s=10.0, end_s=65.0,
                 title="Reflex arc", facet="teaching", why="", warnings=())
    assert w.end_s - w.start_s == 55.0
