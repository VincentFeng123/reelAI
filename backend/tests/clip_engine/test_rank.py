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


def test_educational_ranking_and_bounds():
    """
    Covers 3 invariants:
    1. Same match_count: boosted edu video outranks penalised entertainment video.
    2. edu_score is stored on each video and bounded to [-3.0, +3.0].
    3. Cross-match-count: a penalised 2-match video still beats a boosted 1-match video.
    """
    # Invariant 1 + 2: same match_count, educational beats entertainment
    per_query_same = [{"query": "q", "videos": [
        {"id": "edu", "title": "Photosynthesis explained — MIT lecture", "viewCount": 1000},
        {"id": "ent", "title": "Photosynthesis funny moments compilation", "viewCount": 1000},
    ]}]
    ranked_same = merge_and_rank(per_query_same)
    assert ranked_same[0]["id"] == "edu", "boosted video should rank first at same match_count"
    assert ranked_same[0]["edu_score"] > 0
    assert ranked_same[1]["edu_score"] < 0
    # edu_score bounded
    for v in ranked_same:
        assert -3.0 <= v["edu_score"] <= 3.0

    # Boundary: many boost hits still capped at +3.0
    heavy = {"id": "h", "title": (
        "lecture explained tutorial course fundamentals basics introduction intro to "
        "how things works professor university documentary crash course khan academy"
    ), "viewCount": 0}
    ranked_heavy = merge_and_rank([{"query": "q", "videos": [heavy]}])
    assert ranked_heavy[0]["edu_score"] <= 3.0

    # Invariant 3: penalised 2-match beats boosted 1-match (sort key unchanged)
    per_query_cross = [
        {"query": "q1", "videos": [
            {"id": "two_match", "title": "X reaction compilation funny meme", "viewCount": 100},
        ]},
        {"query": "q2", "videos": [
            {"id": "two_match", "title": "X reaction compilation funny meme", "viewCount": 100},
            {"id": "one_match", "title": "X explained lecture course tutorial", "viewCount": 100},
        ]},
    ]
    ranked_cross = merge_and_rank(per_query_cross)
    assert ranked_cross[0]["id"] == "two_match", "match_count wins over edu_score"
    assert ranked_cross[0]["match_count"] == 2
    assert ranked_cross[1]["match_count"] == 1


def test_level_score_bands():
    from backend.app.clip_engine.rank import _level_score
    intro = {"title": "Introduction to Physics 101", "channel": ""}
    grad = {"title": "Graduate Physics Seminar", "channel": ""}
    assert _level_score(intro, "beginner") > 0
    assert _level_score(grad, "beginner") < 0
    assert _level_score(grad, "advanced") > 0
    assert _level_score(intro, "advanced") < 0
    assert _level_score(intro, None) == 0.0
    assert _level_score(intro, "intermediate") == 0.0


def test_level_score_clamped():
    from backend.app.clip_engine.rank import _level_score
    stacked = {"title": "intro introduction basics beginner 101 crash course", "channel": ""}
    assert -2.0 <= _level_score(stacked, "beginner") <= 2.0
    # Lower bound: a title stacked with advanced terms scored against "beginner"
    # accumulates opposite-band misses and must clamp at -2.0.
    stacked_adv = {"title": "advanced graduate seminar research proofs lecture 101", "channel": ""}
    assert _level_score(stacked_adv, "beginner") >= -2.0


def test_merge_and_rank_level_reorders_within_match_band():
    from backend.app.clip_engine.rank import merge_and_rank
    per_query = [{"videos": [
        {"id": "adv", "title": "Graduate Physics Seminar", "viewCount": 100},
        {"id": "beg", "title": "Physics for Beginners", "viewCount": 100},
    ]}]
    ranked = merge_and_rank(per_query, level="beginner")
    assert [v["id"] for v in ranked][0] == "beg"
    ranked_none = merge_and_rank(per_query)  # level omitted -> original behavior
    assert len(ranked_none) == 2
    # Without a level, "adv" wins deterministically: equal match_count/viewCount
    # and both edu_score 0, but "adv" sits at list index 0 so its best_rank
    # gives rank_score 1.0 (vs 0.5) -> score ~14.04 vs ~13.04.
    assert ranked_none[0]["id"] == "adv"


def test_normalizes_prefixed_ids_and_merges_nonblank_metadata() -> None:
    video_id = "dQw4w9WgXcQ"
    per_query = [
        {"query": "first", "videos": [{
            "id": f"yt:{video_id}", "title": "", "channel": {"id": "UC1"},
            "viewCount": "40", "duration": "12.5",
        }]},
        {"query": "second", "videos": [{
            "url": f"https://youtu.be/{video_id}", "title": "Useful lecture",
            "description": "Description", "channelTitle": "Teacher",
            "channelUrl": "https://youtube.com/channel/UC1",
            "publishedAt": "2026-07-10", "thumbnailUrl": "thumb",
            "view_count": 42,
        }]},
    ]
    [result] = merge_and_rank(per_query)
    assert result["id"] == video_id
    assert result["url"] == f"https://www.youtube.com/watch?v={video_id}"
    assert result["title"] == "Useful lecture"
    assert result["description"] == "Description"
    assert result["channel"] == "Teacher"
    assert result["channel_id"] == "UC1"
    assert result["channel_url"].endswith("/UC1")
    assert result["duration"] == 12.5
    assert result["view_count"] == 42
    assert result["published_at"] == "2026-07-10"


def test_duplicate_video_within_one_query_counts_once() -> None:
    ranked = merge_and_rank([{
        "query": "same",
        "videos": [{"id": "x"}, {"id": "x", "title": "duplicate"}],
    }])
    assert ranked[0]["match_count"] == 1


def test_duplicate_literal_requests_share_one_consensus_family() -> None:
    [result] = merge_and_rank([
        {
            "query": "Intro to Python",
            "query_family": "python",
            "query_trust": "literal",
            "hd_preferred": False,
            "videos": [{"id": "python"}],
        },
        {
            "query": "Intro to Python",
            "query_family": "python",
            "query_trust": "literal",
            "hd_preferred": True,
            "videos": [{"id": "python"}],
        },
    ])

    assert result["match_count"] == 1
    assert result["trusted_match_count"] == 1
    assert result["matched_queries"] == ["Intro to Python"]
    assert result["hd_match"] is True


def test_hd_is_only_a_tiebreak_after_topic_signals() -> None:
    ranked = merge_and_rank([
        {
            "query": "Intro to Python",
            "query_family": "python",
            "query_trust": "literal",
            "videos": [{"id": "literal-sd", "viewCount": 10}],
        },
        {
            "query": "programming",
            "query_family": "programming",
            "query_trust": "ai",
            "hd_preferred": True,
            "videos": [{"id": "ai-hd", "viewCount": 10_000_000}],
        },
    ])
    assert ranked[0]["id"] == "literal-sd"

    tied = merge_and_rank([
        {
            "query": "Intro to Python",
            "query_family": "python",
            "query_trust": "literal",
            "videos": [{"id": "sd", "viewCount": 10}],
        },
        {
            "query": "Intro to Python",
            "query_family": "python",
            "query_trust": "literal",
            "filters_applied": {"features": ["hd"]},
            "videos": [{"id": "hd", "viewCount": 10}],
        },
    ])
    assert tied[0]["id"] == "hd"
