from backend.app.clip_engine.rank import merge_and_rank


def test_ranks_by_match_count_then_views():
    per_query = [
        {"query": "a", "videos": [{"id": "x", "title": "X", "viewCount": 100},
                                  {"id": "y", "title": "Y", "viewCount": 999}]},
        {"query": "b", "videos": [{"id": "x", "title": "X", "viewCount": 100}]},
    ]
    ranked = merge_and_rank(per_query)
    assert ranked[0]["id"] == "x"            # match_count 2 beats y's 1
    assert ranked[0]["match_count"] == 2
    assert ranked[0]["url"] == "https://www.youtube.com/watch?v=x"
    assert sorted(ranked[0]["matched_queries"]) == ["a", "b"]


def test_skips_videos_without_id():
    ranked = merge_and_rank([{"query": "a", "videos": [{"title": "no id"}]}])
    assert ranked == []
