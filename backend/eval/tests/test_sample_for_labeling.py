"""E1a labeling exporter — pure manifest construction, stratum tagging, per-stratum limits.

OFFLINE: no LLM anywhere. The drive-path products (specs, rejections, sentences) are
injected as fixtures and judge fields come from a stub judge_fn; the one collect_video test
monkeypatches load_probe_inputs/assemble_clips so the whole wiring runs with zero llm_json
calls.
"""
from __future__ import annotations

import json

import pytest

import backend.eval.sample_for_labeling as S
from backend.pipeline.assemble.integrity import Rejection
from backend.pipeline.assemble.tests.conftest import mini_sents


def _spec(i0, i1, sents, role="explanation", card="", **kw):
    return {"start": sents[i0].start, "end": sents[i1].end,
            "sentence_start_idx": i0, "sentence_end_idx": i1,
            "role": role, "context_card": card, "cand_id": f"c{i0}", "title": "t",
            "warnings": tuple(kw.pop("warnings", ())), **kw}


def _judge(score=0.9, understandable=True, kinds=(), error=False):
    def fn(text, role, card):
        return {"score": score, "understandable": understandable,
                "failure_kinds": list(kinds), "error": error}
    return fn


def _rej(stage="repair", start=0.0, end=9.9, kinds=("missing_reasoning",), score=0.3):
    return Rejection(cand_id="r", title="rt", role="explanation", stage=stage,
                     reason="budget exhausted", score=score,
                     failure_kinds=list(kinds), start=start, end=end)


# ── accepted-spec entries ─────────────────────────────────────────────────────
def test_accepted_entry_fields():
    sents = mini_sents(6)
    entries = S.build_entries("vid", "My Title", [_spec(0, 2, sents, card="the card")],
                              [], sents, _judge(0.9, True))
    assert len(entries) == 1
    e = entries[0]
    assert e["video_id"] == "vid"
    assert e["video_title"] == "My Title"
    assert e["start"] == 0.0 and e["end"] == pytest.approx(29.9)
    assert e["text"] == "sentence 0. sentence 1. sentence 2."
    assert e["context_card"] == "the card"
    assert e["status"] == "shipped"
    assert e["stratum"] == "random"
    # judge fields live under a dedicated key so the UI can hide them (bias hygiene)
    assert e["judge"]["score"] == 0.9
    assert e["judge"]["understandable"] is True
    assert e["judge"]["stage"] == "accept"
    # embed url plays exactly the span (floor start, ceil end)
    assert e["embed_url"] == "https://www.youtube.com/embed/vid?start=0&end=30&rel=0"


def test_embed_url_rounding():
    assert S.embed_url("v", 12.3, 45.2) == "https://www.youtube.com/embed/v?start=12&end=46&rel=0"
    assert S.embed_url("v", -1.0, 0.2) == "https://www.youtube.com/embed/v?start=0&end=1&rel=0"


def test_status_shipped_flagged():
    sents = mini_sents(4)
    for spec in (_spec(0, 1, sents, ship_flagged=True),
                 _spec(0, 1, sents, warnings=("unverified_judge_concerns",))):
        e = S.build_entries("v", "", [spec], [], sents, _judge())[0]
        assert e["status"] == "shipped_flagged"


# ── stratum tagging ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("score,stratum", [
    (0.55, "band_4_7"), (0.4, "band_4_7"), (0.7, "band_4_7"),   # inclusive band edges
    (0.39, "random"), (0.71, "random"), (0.9, "random"), (None, "random"),
])
def test_stratum_band_on_accepted(score, stratum):
    assert S.stratum_for("shipped", score, []) == stratum
    assert S.stratum_for("shipped_flagged", score, []) == stratum


def test_stratum_kill_uses_first_kind():
    assert S.stratum_for("rejected", 0.5, ["missing_result", "other"]) == "kill:missing_result"
    assert S.stratum_for("rejected", 0.5, []) == "kill:other"          # no kinds recorded


# ── rejection entries ─────────────────────────────────────────────────────────
def test_only_judge_stage_rejections_become_entries():
    sents = mini_sents(4)
    rejections = [_rej(stage=st) for st in
                  ("repair", "post_merge_judge", "post_snap_judge",     # judge stages: kept
                   "snap", "dedupe", "quality_floor", "max_clips")]     # mechanical: excluded
    entries = S.build_entries("v", "", [], rejections, sents, _judge())
    assert len(entries) == 3
    assert all(e["status"] == "rejected" for e in entries)
    assert {e["judge"]["stage"] for e in entries} == set(S.JUDGE_REJECTION_STAGES)


def test_rejection_entry_fields_and_text_by_time_overlap():
    sents = mini_sents(4)                     # sentence i covers [10i, 10i+9.9]
    e = S.build_entries("v", "", [], [_rej(start=10.0, end=29.9, score=0.3,
                                           kinds=("missing_result",))],
                        sents, _judge())[0]
    assert e["text"] == "sentence 1. sentence 2."       # recovered by time overlap
    assert e["stratum"] == "kill:missing_result"
    assert e["judge"]["understandable"] is False
    assert e["judge"]["score"] == 0.3
    assert e["judge"]["reason"] == "budget exhausted"
    assert e["context_card"] == ""


# ── determinism + limits ──────────────────────────────────────────────────────
def test_deterministic_ordering_regardless_of_input_order():
    sents = mini_sents(6)
    specs = [_spec(0, 1, sents), _spec(3, 4, sents)]
    rejections = [_rej(start=20.0, end=29.9), _rej(start=50.0, end=59.9)]
    a = S.build_entries("v", "", specs, rejections, sents, _judge())
    b = S.build_entries("v", "", list(reversed(specs)), list(reversed(rejections)),
                        sents, _judge())
    assert a == b
    assert [e["start"] for e in a] == sorted(e["start"] for e in a)


def test_entry_ids_are_unique_even_for_identical_spans():
    sents = mini_sents(4)
    # two judge-stage kills of the SAME span (e.g. repair + post_snap in different runs)
    rejections = [_rej(stage="repair"), _rej(stage="post_snap_judge")]
    entries = S.build_entries("v", "", [], rejections, sents, _judge())
    ids = [e["id"] for e in entries]
    assert len(ids) == len(set(ids)) == 2


def test_apply_stratum_limit_caps_each_stratum_in_order():
    entries = [{"stratum": "random", "id": i} for i in range(3)] \
        + [{"stratum": "band_4_7", "id": 10}] \
        + [{"stratum": "kill:other", "id": i} for i in (20, 21)]
    out = S.apply_stratum_limit(entries, 1)
    assert [e["id"] for e in out] == [0, 10, 20]         # first of each stratum kept
    assert S.apply_stratum_limit(entries, None) == entries


# ── vocabulary + manifest payload ─────────────────────────────────────────────
def test_kind_options_are_validate_vocab_plus_boundary_kinds():
    # validate.FailureReason's documented kind vocabulary, verbatim
    assert S.JUDGE_KIND_VOCAB == (
        "unresolved_reference", "missing_prerequisite", "missing_visual",
        "missing_problem_statement", "missing_reasoning", "missing_result",
        "not_source_grounded", "off_topic", "other")
    assert S.HUMAN_KIND_OPTIONS == S.JUDGE_KIND_VOCAB + (
        "starts_mid_thought", "ends_unresolved", "boundary_garbage")


def test_build_manifest_and_write(tmp_path):
    sents = mini_sents(4)
    entries = S.build_entries("v", "T", [_spec(0, 1, sents)], [_rej()], sents, _judge(0.5))
    payload = S.build_manifest([("v", "T", entries)], limit=None, stamp="STAMP")
    assert payload["generated_at"] == "STAMP"
    assert payload["videos"] == [{"video_id": "v", "title": "T"}]
    assert payload["n_entries"] == 2
    assert payload["strata"] == {"band_4_7": 1, "kill:missing_reasoning": 1}
    assert payload["kind_options"] == list(S.HUMAN_KIND_OPTIONS)
    path = S.write_manifest(payload, tmp_path / "labeling" / "manifest.json")
    assert json.loads(path.read_text()) == payload


def test_build_manifest_applies_limit_across_videos():
    sents = mini_sents(4)
    e1 = S.build_entries("v1", "", [_spec(0, 1, sents)], [], sents, _judge(0.9))
    e2 = S.build_entries("v2", "", [_spec(0, 1, sents)], [], sents, _judge(0.9))
    payload = S.build_manifest([("v1", "", e1), ("v2", "", e2)], limit=1, stamp="")
    assert payload["n_entries"] == 1                     # both entries share stratum 'random'
    assert payload["entries"][0]["video_id"] == "v1"     # first video wins deterministically


# ── CLI parsing ───────────────────────────────────────────────────────────────
def test_parse_args_requires_collect():
    with pytest.raises(SystemExit):
        S.parse_args([])
    a = S.parse_args(["--collect", "v1", "v2", "--limit", "5"])
    assert a.collect == ["v1", "v2"]
    assert a.limit == 5
    assert a.out == S.DEFAULT_OUT


def test_parse_args_rejects_bad_limit():
    with pytest.raises(SystemExit):
        S.parse_args(["--collect", "v", "--limit", "0"])


# ── collect_video wiring (drive-path products injected; zero LLM) ─────────────
def test_collect_video_offline(monkeypatch):
    sents = mini_sents(6)
    specs = [_spec(0, 2, sents)]
    rejections = [_rej(start=30.0, end=49.9, kinds=("off_topic",))]
    ctx = {"structure": object(), "sents": sents, "topic": "t",
           "settings": {}, "adapter": object()}
    monkeypatch.setattr(S, "load_probe_inputs", lambda vid: ctx)
    monkeypatch.setattr(S, "assemble_clips",
                        lambda *args, **kw: (specs, "notes", rejections))
    monkeypatch.setattr(S, "video_title", lambda vid: "Cached Title")

    title, entries = S.collect_video("vid", judge_fn=_judge(0.55))
    assert title == "Cached Title"
    assert [e["status"] for e in entries] == ["shipped", "rejected"]
    assert entries[0]["stratum"] == "band_4_7"
    assert entries[1]["stratum"] == "kill:off_topic"
    assert all(e["video_title"] == "Cached Title" for e in entries)


def test_collect_video_missing_cache_is_none(monkeypatch):
    monkeypatch.setattr(S, "load_probe_inputs", lambda vid: None)
    assert S.collect_video("vid", judge_fn=_judge()) is None
