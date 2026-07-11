"""E1c labels endpoints — POST /api/labels merge-not-clobber + GET resume.

FastAPI TestClient, fully offline: GOLDEN_DIR is monkeypatched to tmp_path so no real
golden file is ever touched, and no pipeline/LLM code runs (the endpoints are pure I/O
over eval/golden JSON files).
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import backend.eval.golden as G
from backend.main import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "GOLDEN_DIR", tmp_path)   # load_golden + merge read it at call time
    return TestClient(app)


def _post(client, video_id="vid_1", labels=None, note=""):
    return client.post("/api/labels", json={
        "video_id": video_id,
        "labels": labels if labels is not None else [],
        "video_note": note,
    })


LABEL = {"start": 10.0, "end": 20.0, "understandable": False,
         "failure_kinds": ["missing_result", "ends_unresolved"],
         "needed_first": "that this is about projectile motion"}


# ── create + resume ───────────────────────────────────────────────────────────
def test_post_creates_golden_file_and_get_resumes(client, tmp_path):
    r = _post(client, labels=[LABEL], note="clip the worked example")
    assert r.status_code == 200
    assert r.json() == {"ok": True, "video_id": "vid_1", "n_clips": 1}

    saved = json.loads((tmp_path / "vid_1.json").read_text())
    assert saved["video_id"] == "vid_1"
    clip = saved["human"]["clips"][0]
    assert clip["understandable"] is False
    assert clip["failure_kinds"] == ["missing_result", "ends_unresolved"]
    assert clip["needed_first"] == "that this is about projectile motion"
    assert clip["labeled_at"]                              # server stamped it

    r = client.get("/api/labels/vid_1")                    # resume path
    assert r.status_code == 200
    body = r.json()
    assert body["video_note"] == "clip the worked example"
    assert len(body["clips"]) == 1 and body["clips"][0]["start"] == 10.0


def test_get_unlabeled_video_returns_empty_block(client):
    assert client.get("/api/labels/nothing_here").json() == {"clips": [], "video_note": ""}


# ── merge, never clobber ──────────────────────────────────────────────────────
def test_post_preserves_existing_gold_keys(client, tmp_path):
    gold = {"video_id": "vid_1", "topics": ["kinematics"],
            "units": [{"start": 0, "end": 5, "role": "definition"}],
            "chapters": [{"start_time": 0, "end_time": 60, "title": "intro"}],
            "chapters_provenance": "creator"}
    (tmp_path / "vid_1.json").write_text(json.dumps(gold))

    assert _post(client, labels=[LABEL]).status_code == 200
    saved = json.loads((tmp_path / "vid_1.json").read_text())
    for key in ("topics", "units", "chapters", "chapters_provenance"):
        assert saved[key] == gold[key]                     # hand-authored keys untouched
    assert len(saved["human"]["clips"]) == 1


def test_post_upserts_by_span_within_tolerance_and_appends_new(client, tmp_path):
    _post(client, labels=[LABEL])
    # relabel the same clip (0.2s/0.3s off — inside the 0.5s tolerance) → replaces
    _post(client, labels=[{"start": 10.2, "end": 20.3, "understandable": True,
                           "failure_kinds": [], "needed_first": ""}])
    clips = json.loads((tmp_path / "vid_1.json").read_text())["human"]["clips"]
    assert len(clips) == 1
    assert clips[0]["understandable"] is True
    # a different span appends (and clips stay time-sorted)
    _post(client, labels=[{"start": 3.0, "end": 8.0, "understandable": False,
                           "failure_kinds": ["starts_mid_thought"], "needed_first": "x"}])
    clips = json.loads((tmp_path / "vid_1.json").read_text())["human"]["clips"]
    assert [c["start"] for c in clips] == [3.0, 10.2]


def test_post_empty_note_keeps_existing_note(client):
    _post(client, note="first note")
    _post(client, note="")                                 # untouched note field on re-save
    assert client.get("/api/labels/vid_1").json()["video_note"] == "first note"
    _post(client, note="second note")                      # a real new note replaces
    assert client.get("/api/labels/vid_1").json()["video_note"] == "second note"


def test_post_corrupt_golden_file_is_409_and_untouched(client, tmp_path):
    (tmp_path / "vid_1.json").write_text("{not json!!")
    r = _post(client, labels=[LABEL])
    assert r.status_code == 409
    assert (tmp_path / "vid_1.json").read_text() == "{not json!!"   # never clobbered


# ── validation ────────────────────────────────────────────────────────────────
def test_bad_video_ids_rejected(client, tmp_path):
    assert _post(client, video_id="../evil").status_code == 400
    assert _post(client, video_id="a/b").status_code == 400
    assert _post(client, video_id="").status_code == 400
    assert client.get("/api/labels/bad*id").status_code == 400
    assert list(tmp_path.iterdir()) == []                  # nothing was written
