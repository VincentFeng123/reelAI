from __future__ import annotations

import json

import pytest

from backend.eval import segment_review as R


def _transcript():
    return {
        "duration": 20.0,
        "words": [
            {"word": f"w{i}", "start": float(i), "end": float(i + 1)}
            for i in range(20)
        ],
        "segments": [{"text": "fallback", "start": 0.0, "end": 20.0}],
    }


def _items():
    base = {
        "dataset_version": "segment-v1", "postprocess_version": "strict-v1",
        "pair_id": "pair-1", "video_id": "video-1", "topic": "vectors",
        "start": 5.0, "end": 15.0, "repeat": 1,
        "summary": "A grounded summary.", "takeaways": ["one", "two"],
        "assessment": {"prompt": "Question?", "options": ["a", "b", "c", "d"],
                       "model": "nested-secret", "confidence": 0.99, "cost_usd": 2.0},
        "model": "must-not-leak", "classification": "green", "cost_usd": 9.99,
    }
    return [
        {**base, "profile": "flash_single_v1", "clip_id": "flash-clip"},
        {**base, "profile": "corrected_pro_v1", "clip_id": "pro-clip", "start": 2.0},
    ]


def _scores(value=4, overrides=None):
    scores = {dimension: value for dimension in R.REVIEW_DIMENSIONS}
    scores.update(overrides or {})
    return scores


def _record(item_id, reviewer_id, scores, role="reviewer"):
    return {"review_item_id": item_id, "reviewer_id": reviewer_id, "role": role,
            "scores": scores, "human_usable": True, "assessment_correct": True,
            "severe_hallucination": False}


def test_blind_bundle_separates_hidden_identity_and_supplies_four_second_context():
    public, hidden = R.build_blind_review_bundle(
        _items(), {"pair-1": _transcript()}, seed=7, context_seconds=4.0)
    assert public["manifest_id"] == hidden["manifest_id"]
    assert tuple(public["rubric"]["dimensions"]) == R.REVIEW_DIMENSIONS
    assert len(public["items"]) == len(hidden["items"]) == 2
    serialized = json.dumps(public)
    for secret in ("flash_single_v1", "corrected_pro_v1", "flash-clip", "pro-clip",
                   "must-not-leak", "nested-secret", "confidence", "cost_usd", "green", "9.99"):
        assert secret not in serialized
    assert '"profile"' not in serialized and '"repeat"' not in serialized
    assert {item["profile"] for item in hidden["items"]} == {
        "flash_single_v1", "corrected_pro_v1"}

    by_start = {item["start"]: item for item in public["items"]}
    item = by_start[5.0]
    assert (item["opening_boundary_context"]["start"],
            item["opening_boundary_context"]["end"]) == (1.0, 9.0)
    assert (item["closing_boundary_context"]["start"],
            item["closing_boundary_context"]["end"]) == (11.0, 19.0)
    assert item["clip_transcript"] == " ".join(f"w{i}" for i in range(5, 15))


def test_bundle_is_deterministic_for_seed_and_caps_context_at_video_edges():
    first = R.build_blind_review_bundle(_items(), {"video-1": _transcript()}, seed=3)
    second = R.build_blind_review_bundle(_items(), {"video-1": _transcript()}, seed=3)
    assert first == second
    start_two = next(item for item in first[0]["items"] if item["start"] == 2.0)
    assert start_two["opening_boundary_context"]["start"] == 0.0


def test_two_independent_reviewers_and_greater_than_one_point_adjudication():
    item = "review-item-0001"
    records = [
        _record(item, "reviewer-a", _scores(4)),
        _record(item, "reviewer-b", _scores(4, {"opening_completeness": 2})),
    ]
    assert any("requires adjudication" in error for error in
               R.validate_review_records(records, expected_item_ids=[item]))
    needs = R.adjudication_requirements(records, expected_item_ids=[item])
    assert needs == [{"review_item_id": item,
                      "dimensions": ["opening_completeness"],
                      "judgments": [],
                      "adjudicator_present": False, "resolved": False}]

    records.append(_record(item, "reviewer-c", _scores(3), role="adjudicator"))
    assert R.validate_review_records(records, expected_item_ids=[item]) == []
    assert R.adjudication_requirements(records)[0]["resolved"] is True


def test_review_records_fail_closed_on_missing_explicit_judgments_or_second_reviewer():
    record = _record("item", "reviewer-a", _scores())
    del record["human_usable"]
    errors = R.validate_review_records([record], expected_item_ids=["item"])
    assert any("human_usable must be an explicit boolean" in error for error in errors)
    assert any("requires 2" in error for error in errors)


def test_one_point_disagreement_does_not_require_adjudication():
    records = [_record("item", "a", _scores(3)), _record("item", "b", _scores(4))]
    assert R.adjudication_requirements(records) == []


def test_clip_resolution_uses_explicit_adjudicator_and_emits_one_grounded_record():
    public, hidden = R.build_blind_review_bundle(
        [_items()[0]], {"pair-1": _transcript()})
    item_id = public["items"][0]["review_item_id"]
    records = [
        _record(item_id, "reviewer-a", _scores(5)),
        _record(item_id, "reviewer-b", _scores(5, {"opening_completeness": 2})),
        _record(item_id, "reviewer-c", _scores(4), role="adjudicator"),
    ]
    records[1]["human_usable"] = False
    records[2]["human_usable"] = True
    resolved = R.resolve_review_records(records, hidden)
    assert len(resolved) == 1
    assert resolved[0]["scores"]["opening_completeness"] == 4.0
    assert resolved[0]["human_usable"] is True
    assert resolved[0]["adjudicated_dimensions"] == ["opening_completeness"]
    assert resolved[0]["adjudicated_judgments"] == ["human_usable"]


def test_public_and_hidden_files_cannot_alias(tmp_path):
    public, hidden = R.build_blind_review_bundle(_items(), {"pair-1": _transcript()})
    public_path, hidden_path = R.write_review_bundle(
        public, hidden, public_path=tmp_path / "public.json", hidden_path=tmp_path / "hidden.json")
    assert json.loads(public_path.read_text())["blinded"] is True
    assert "profile" in hidden_path.read_text()


def _generation_rows(pair_count=6, repeats=2):
    candidate, pro = [], []
    for pair_index in range(pair_count):
        for repeat in range(1, repeats + 1):
            common = {
                "dataset_version": "segment-v1", "postprocess_version": "strict-v1",
                "pair_id": f"pair-{pair_index}", "repeat": repeat,
                "video_id": f"video-{pair_index}", "topic": f"topic {pair_index}",
                "model": "must-not-leak-model", "cost_usd": 123.45,
            }
            candidate.append({
                **common, "profile": "simulated_hybrid_v1",
                "clips": [{
                    "start": 1.0, "end": 5.0, "title": "Candidate lesson",
                    "summary": "Candidate summary", "takeaways": ["candidate fact"],
                    "assessment": {"prompt": "Candidate question?", "correct_index": 1,
                                   "model": "nested-secret"},
                    "classification": "green", "cost_usd": 5.0,
                }],
            })
            pro.append({
                **common, "profile": "corrected_pro_v1",
                "clips": [{
                    "start": 2.0, "end": 6.0, "title": "Pro lesson",
                    "summary": "Pro summary", "takeaways": ["pro fact"],
                    "assessment": None, "model": "nested-secret",
                }],
            })
    return candidate, pro


def test_whole_video_bundle_blinds_profiles_cost_and_orientation():
    candidate, pro = _generation_rows()
    public, hidden = R.build_blind_whole_video_bundle(candidate, pro, seed=7)
    assert public["manifest_id"] == hidden["manifest_id"]
    assert public["choice_options"] == ["a", "tied", "b"]
    assert len(public["comparisons"]) == len(hidden["comparisons"]) == 12
    serialized = json.dumps(public)
    for secret in ("simulated_hybrid_v1", "corrected_pro_v1", "must-not-leak-model",
                   "nested-secret", "123.45", '"cost_usd"', '"classification"'):
        assert secret not in serialized
    assert '"pair_id"' not in serialized and '"repeat"' not in serialized
    assert {item["candidate_side"] for item in hidden["comparisons"]} == {"a", "b"}
    assert {item["candidate_profile"] for item in hidden["comparisons"]} \
        == {"simulated_hybrid_v1"}
    # Only the whitelisted assessment fields can cross the blind boundary.
    assert "nested-secret" not in serialized


def test_whole_video_bundle_is_seeded_and_preserves_empty_clip_sets():
    candidate, pro = _generation_rows(pair_count=2, repeats=1)
    candidate[0]["clips"] = []
    first = R.build_blind_whole_video_bundle(candidate, pro, seed=3)
    second = R.build_blind_whole_video_bundle(candidate, pro, seed=3)
    different = R.build_blind_whole_video_bundle(candidate, pro, seed=4)
    assert first == second
    assert first != different
    mapping = {item["comparison_id"]: item for item in first[1]["comparisons"]}
    empty_id = next(item_id for item_id, item in mapping.items() if item["pair_id"] == "pair-0")
    public = next(item for item in first[0]["comparisons"]
                  if item["comparison_id"] == empty_id)
    side = mapping[empty_id]["candidate_side"]
    assert public[f"set_{side}"]["clips"] == []


def test_whole_video_bundle_rejects_missing_or_mismatched_pro_trial():
    candidate, pro = _generation_rows(pair_count=2, repeats=1)
    with pytest.raises(ValueError, match="keys must match"):
        R.build_blind_whole_video_bundle(candidate, pro[:-1])
    pro[0]["topic"] = "different"
    with pytest.raises(ValueError, match="disagree on topic"):
        R.build_blind_whole_video_bundle(candidate, pro)


def test_two_independent_whole_video_choices_validate_and_decode_candidate_relative():
    candidate, pro = _generation_rows(pair_count=2, repeats=1)
    public, hidden = R.build_blind_whole_video_bundle(candidate, pro, seed=9)
    records = []
    mapping = {item["comparison_id"]: item for item in hidden["comparisons"]}
    for comparison in public["comparisons"]:
        comparison_id = comparison["comparison_id"]
        candidate_side = mapping[comparison_id]["candidate_side"]
        records.extend([
            {"comparison_id": comparison_id, "reviewer_id": "reviewer-1",
             "role": "reviewer", "choice": candidate_side},
            {"comparison_id": comparison_id, "reviewer_id": "reviewer-2",
             "role": "reviewer", "choice": candidate_side},
        ])
    expected = [item["comparison_id"] for item in public["comparisons"]]
    assert R.validate_whole_video_review_records(
        records, expected_comparison_ids=expected) == []
    decoded = R.decode_whole_video_review_records(records, hidden)
    for comparison_id in expected:
        outcomes = [row["whole_video_comparison"] for row in decoded
                    if row["comparison_id"] == comparison_id]
        assert outcomes == ["better"]
    assert all(row["candidate_profile"] == "simulated_hybrid_v1" for row in decoded)
    assert all(row["selected_pro_profile"] == "corrected_pro_v1" for row in decoded)


def test_tied_choice_decodes_to_tied_for_either_orientation():
    candidate, pro = _generation_rows(pair_count=1, repeats=1)
    _public, hidden = R.build_blind_whole_video_bundle(candidate, pro, seed=1)
    comparison_id = hidden["comparisons"][0]["comparison_id"]
    records = [
        {"comparison_id": comparison_id, "reviewer_id": "a", "choice": "tied"},
        {"comparison_id": comparison_id, "reviewer_id": "b", "choice": "tied"},
    ]
    for record in records:
        record["role"] = "reviewer"
    assert {row["whole_video_comparison"]
            for row in R.decode_whole_video_review_records(records, hidden)} == {"tied"}


def test_whole_video_disagreement_requires_one_independent_adjudicator():
    candidate, pro = _generation_rows(pair_count=1, repeats=1)
    _public, hidden = R.build_blind_whole_video_bundle(candidate, pro, seed=2)
    mapping = hidden["comparisons"][0]
    comparison_id = mapping["comparison_id"]
    records = [
        {"comparison_id": comparison_id, "reviewer_id": "a", "role": "reviewer",
         "choice": "a"},
        {"comparison_id": comparison_id, "reviewer_id": "b", "role": "reviewer",
         "choice": "b"},
    ]
    assert any("requires adjudication" in error for error in
               R.validate_whole_video_review_records(records, expected_comparison_ids=[comparison_id]))
    with pytest.raises(ValueError, match="requires adjudication"):
        R.decode_whole_video_review_records(records, hidden)
    records.append({"comparison_id": comparison_id, "reviewer_id": "c",
                    "role": "adjudicator", "choice": mapping["candidate_side"]})
    resolved = R.resolve_whole_video_review_records(records, hidden)
    assert len(resolved) == 1
    assert resolved[0]["whole_video_comparison"] == "better"
    assert resolved[0]["adjudicated"] is True


def test_whole_video_choice_validation_rejects_duplicates_bad_choice_and_missing_reviewer():
    records = [
        {"comparison_id": "comparison", "reviewer_id": "same", "role": "reviewer",
         "choice": "candidate"},
        {"comparison_id": "comparison", "reviewer_id": "same", "role": "reviewer",
         "choice": "a"},
    ]
    errors = R.validate_whole_video_review_records(
        records, expected_comparison_ids=["comparison"])
    assert any("choice must be a|tied|b" in error for error in errors)
    assert any("duplicate whole-video review" in error for error in errors)
    assert any("requires 2" in error for error in errors)
