from backend.eval import metrics as M


def test_window_len_stats():
    clips = [{"start": 0.0, "end": 60.0}, {"start": 100.0, "end": 130.0}]
    s = M.window_len_stats(clips)
    assert s["min"] == 30.0 and s["max"] == 60.0 and s["mean"] == 45.0


def test_topic_selectivity():
    assert M.topic_selectivity({"n_topics_total": 20, "n_topics_kept": 8}) == 0.4
    assert M.topic_selectivity({}) == 0.0            # missing ⇒ 0, never divide-by-zero
