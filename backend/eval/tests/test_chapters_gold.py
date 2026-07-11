"""D-chapters-gold: creator-YouTube-chapter segmentation gold (YTSeg protocol).

Hand-computed Pk / WindowDiff on tiny synthetic cases (incl. the degenerate all-one-segment
and boundary-off-by-one cases), greatest-overlap sentence→chapter assignment with tie
handling, the clip straddle metric with its 5s tolerance, and the merge-not-clobber behavior
of the golden chapter writer. Offline — no network, no yt_dlp calls: fixture dicts only.
"""
from __future__ import annotations

import json
import math
import types

import pytest

import backend.eval.make_golden as MG
import backend.eval.metrics as metrics
import backend.eval.run_eval as R
from backend.eval.golden import gold_chapters


class _Sent:
    def __init__(self, start, end):
        self.start, self.end = float(start), float(end)
        self.text, self.ends_with_period = "s.", True


def _sents(spans):
    return [_Sent(s, e) for s, e in spans]


# ── Pk / WindowDiff, hand-computed ────────────────────────────────────────────
def test_pk_windowdiff_perfect_hypothesis_is_zero():
    ref = [0, 0, 0, 1, 1, 1]
    assert metrics.pk(ref, list(ref), k=2) == 0.0
    assert metrics.windowdiff(ref, list(ref), k=2) == 0.0


def test_pk_missed_boundary_hand_computed():
    # ref boundary between idx 2 and 3; hyp is one giant segment (misses it).
    ref = [0, 0, 0, 1, 1, 1]
    hyp = [0] * 6
    # k=2 probes (i,i+2): ref same-seg = [T,F,F,T]; hyp all T → disagree at i=1,2 → 2/4.
    assert metrics.pk(ref, hyp, k=2) == pytest.approx(0.5)
    # windows with the ref boundary (after idx 2) inside: i=1 and i=2 → 2/4.
    assert metrics.windowdiff(ref, hyp, k=2) == pytest.approx(0.5)


def test_pk_windowdiff_boundary_off_by_one_hand_computed():
    ref = [0, 0, 0, 1, 1, 1]        # boundary after idx 2
    hyp = [0, 0, 1, 1, 1, 1]        # boundary after idx 1 (off by one)
    # pk k=2: ref same-seg=[T,F,F,T]; hyp same-seg=[F,F,T,T] → disagree at i=0 and i=2 → 2/4.
    assert metrics.pk(ref, hyp, k=2) == pytest.approx(0.5)
    # wd k=2: ref boundary counts per window [0,1,1,0]; hyp [1,1,0,0] → differ at i=0,2 → 2/4.
    assert metrics.windowdiff(ref, hyp, k=2) == pytest.approx(0.5)


def test_degenerate_all_one_segment_reference():
    ref = [0] * 8
    assert metrics.pk(ref, [0] * 8, k=2) == 0.0
    assert metrics.windowdiff(ref, [0] * 8, k=2) == 0.0
    hyp = [0, 0, 0, 0, 1, 1, 1, 1]                       # one false boundary after idx 3
    # pk k=2: only probes (2,4) and (3,5) straddle the false boundary → 2/6.
    assert metrics.pk(ref, hyp, k=2) == pytest.approx(2 / 6)
    # wd k=2: hyp boundary inside windows i=2 and i=3 → 2/6.
    assert metrics.windowdiff(ref, hyp, k=2) == pytest.approx(2 / 6)


def test_pk_windowdiff_nan_conventions():
    assert math.isnan(metrics.pk([0, 1], [0], k=2))                  # length mismatch
    assert math.isnan(metrics.pk([], [], k=2))                       # empty
    assert math.isnan(metrics.pk([0, 0], [0, 0], k=2))               # n <= k
    assert math.isnan(metrics.windowdiff([0, 1], [0], k=2))
    assert math.isnan(metrics.windowdiff([], [], k=2))
    assert math.isnan(metrics.windowdiff([0, 0], [0, 0], k=2))


def test_window_size_is_half_mean_true_segment_length():
    assert metrics.window_size([0, 0, 0, 1, 1, 1]) == 2              # mean 3 → round(1.5)=2
    assert metrics.window_size([0] * 12 + [1] * 12) == 6             # mean 12 → 6
    assert metrics.window_size([0] * 10) == 5                        # one segment: mean 10 → 5
    assert metrics.window_size([0, 1, 2, 3]) == 2                    # mean 1 → floor of 2
    assert metrics.window_size([]) == 2


def test_pk_default_k_comes_from_reference():
    ref = [0, 0, 0, 1, 1, 1]                                          # window_size → 2
    hyp = [0] * 6
    assert metrics.pk(ref, hyp) == metrics.pk(ref, hyp, k=2)
    assert metrics.windowdiff(ref, hyp) == metrics.windowdiff(ref, hyp, k=2)


# ── sentence → chapter assignment (YTSeg greatest-overlap rule) ───────────────
def test_assign_segments_greatest_overlap():
    chapters = [{"start": 0, "end": 10}, {"start": 10, "end": 20}]
    # (8,13): 2s in ch0 vs 3s in ch1 → ch1 wins on greatest overlap.
    assert metrics.assign_segments(_sents([(0, 4), (8, 13), (15, 20)]), chapters) == [0, 1, 1]


def test_assign_segments_tie_breaks_to_earlier_segment():
    chapters = [{"start": 0, "end": 5}, {"start": 5, "end": 10}]
    assert metrics.assign_segments(_sents([(0, 10)]), chapters) == [0]   # 5s vs 5s → earlier


def test_assign_segments_no_overlap_falls_back_to_nearest_midpoint():
    chapters = [{"start": 0, "end": 5}, {"start": 10, "end": 15}]
    assert metrics.assign_segments(_sents([(6, 7)]), chapters) == [0]      # mid 6.5 → nearer ch0
    assert metrics.assign_segments(_sents([(8.5, 9.5)]), chapters) == [1]  # mid 9.0 → nearer ch1
    assert metrics.assign_segments([], chapters) == []
    assert metrics.assign_segments(_sents([(0, 1)]), []) == []


def test_assign_segments_accepts_yt_dlp_keys():
    chapters = [{"start_time": 0, "end_time": 10}, {"start_time": 10, "end_time": 20}]
    assert metrics.assign_segments(_sents([(1, 2), (15, 16)]), chapters) == [0, 1]


# ── clip straddle rate (5s tolerance on BOTH sides) ───────────────────────────
def test_clip_straddle_rate_requires_more_than_5s_on_both_sides():
    chapters = [{"start_time": 0, "end_time": 100, "title": "a"},
                {"start_time": 100, "end_time": 200, "title": "b"}]
    specs = [
        {"start": 90.0, "end": 110.0},    # 10s each side → straddles
        {"start": 96.0, "end": 110.0},    # only 4s before the boundary → no
        {"start": 90.0, "end": 105.0},    # exactly 5s after → no (strictly MORE than 5s)
        {"start": 10.0, "end": 50.0},     # doesn't cross → no
    ]
    assert metrics.clip_straddle_rate(specs, chapters) == pytest.approx(0.25)


def test_clip_straddle_rate_nan_and_single_chapter():
    chapters = [{"start_time": 0, "end_time": 100}]
    # a single chapter has no internal boundary → 0.0, a legitimate (perfect) value
    assert metrics.clip_straddle_rate([{"start": 10, "end": 90}], chapters) == 0.0
    assert math.isnan(metrics.clip_straddle_rate([], chapters))              # no clips → NaN
    assert math.isnan(metrics.clip_straddle_rate([{"start": 0, "end": 10}], []))  # no gold → NaN


# ── golden loading tolerance ──────────────────────────────────────────────────
def test_gold_chapters_tolerant_and_normalized():
    assert gold_chapters(None) == []
    assert gold_chapters({}) == []
    assert gold_chapters({"chapters": None}) == []
    gold = {"chapters": [
        {"start_time": 60, "end_time": 120, "title": "b"},
        {"start_time": 0, "end_time": 60, "title": "a"},
        {"start_time": 5, "end_time": 5},                  # empty span dropped
        {"start_time": None, "end_time": 10},              # malformed dropped
        "junk",                                            # non-dict dropped
    ]}
    chs = gold_chapters(gold)
    assert [c["title"] for c in chs] == ["a", "b"]         # time-sorted
    assert chs[0] == {"start": 0.0, "end": 60.0, "title": "a"}


def test_gold_chapters_explicit_none_falls_back_to_start_end_keys():
    # a chapter with start_time explicitly null but a valid 'start' must still parse
    gold = {"chapters": [{"start_time": None, "start": 3, "end_time": None, "end": 9,
                          "title": "x"}]}
    assert gold_chapters(gold) == [{"start": 3.0, "end": 9.0, "title": "x"}]


# ── golden writer: merge, never clobber ───────────────────────────────────────
def test_merge_chapters_creates_new_golden(tmp_path):
    chapters = [{"start_time": 0.0, "end_time": 60.0, "title": "intro"}]
    p = MG.merge_chapters_into_golden("vidA", chapters, golden_dir=tmp_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["video_id"] == "vidA"
    assert data["chapters"] == chapters
    assert data["chapters_provenance"] == "creator"


def test_merge_chapters_preserves_existing_keys(tmp_path):
    existing = {"video_id": "vidB", "topics": ["derivatives"],
                "reference_concepts": ["limit"],
                "units": [{"start": 0, "end": 1, "role": "hook"}],
                "anchors": [{"anchor_role": "result", "start": 0, "end": 1}]}
    (tmp_path / "vidB.json").write_text(json.dumps(existing), encoding="utf-8")
    chapters = [{"start_time": 0.0, "end_time": 30.0, "title": "c1"}]
    MG.merge_chapters_into_golden("vidB", chapters, golden_dir=tmp_path)
    data = json.loads((tmp_path / "vidB.json").read_text(encoding="utf-8"))
    assert data["units"] == existing["units"]              # hand labels untouched
    assert data["anchors"] == existing["anchors"]
    assert data["topics"] == existing["topics"]
    assert data["reference_concepts"] == existing["reference_concepts"]
    assert data["video_id"] == "vidB"
    assert data["chapters"] == chapters
    assert data["chapters_provenance"] == "creator"


def test_merge_chapters_refreshes_only_the_chapter_keys(tmp_path):
    (tmp_path / "vidC.json").write_text(json.dumps(
        {"video_id": "vidC", "reference_concepts": ["x"],
         "chapters": [{"start_time": 0, "end_time": 5, "title": "old"}],
         "chapters_provenance": "creator"}), encoding="utf-8")
    new = [{"start_time": 0.0, "end_time": 9.0, "title": "new"}]
    MG.merge_chapters_into_golden("vidC", new, golden_dir=tmp_path)
    data = json.loads((tmp_path / "vidC.json").read_text(encoding="utf-8"))
    assert data["chapters"] == new                         # re-import refreshes chapters…
    assert data["reference_concepts"] == ["x"]             # …but nothing else


def test_merge_chapters_never_clobbers_corrupt_golden(tmp_path):
    # a hand-authored golden with a stray syntax error must NOT lose its labels on re-import
    (tmp_path / "vidD.json").write_text('{"units": [oops', encoding="utf-8")
    with pytest.raises(ValueError):
        MG.merge_chapters_into_golden(
            "vidD", [{"start_time": 0.0, "end_time": 5.0, "title": "t"}], golden_dir=tmp_path)
    assert (tmp_path / "vidD.json").read_text(encoding="utf-8") == '{"units": [oops'


def test_merge_chapters_rejects_non_object_golden(tmp_path):
    (tmp_path / "vidE.json").write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError):
        MG.merge_chapters_into_golden(
            "vidE", [{"start_time": 0.0, "end_time": 5.0, "title": "t"}], golden_dir=tmp_path)
    assert (tmp_path / "vidE.json").read_text(encoding="utf-8") == "[1, 2]"


# ── --chapters CLI mode (fetch mocked; never touches yt_dlp/network) ──────────
def test_run_chapters_mode_no_creator_chapters_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(MG, "fetch_chapters", lambda vid: [])
    MG.run_chapters_mode(["vidX"], golden_dir=tmp_path)
    assert "no creator chapters" in capsys.readouterr().out
    assert not (tmp_path / "vidX.json").exists()


def test_run_chapters_mode_writes_and_reports(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(MG, "fetch_chapters",
                        lambda vid: [{"start_time": 0.0, "end_time": 10.0, "title": "t"}])
    MG.run_chapters_mode(["vidY"], golden_dir=tmp_path)
    assert "merged 1 creator chapters" in capsys.readouterr().out
    data = json.loads((tmp_path / "vidY.json").read_text(encoding="utf-8"))
    assert data["chapters_provenance"] == "creator"


def test_run_chapters_mode_corrupt_golden_skipped_not_fatal(tmp_path, monkeypatch, capsys):
    (tmp_path / "vidF.json").write_text("{corrupt", encoding="utf-8")
    monkeypatch.setattr(MG, "fetch_chapters",
                        lambda vid: [{"start_time": 0.0, "end_time": 10.0, "title": "t"}])
    MG.run_chapters_mode(["vidF", "vidG"], golden_dir=tmp_path)
    out = capsys.readouterr().out
    assert "not merging" in out
    assert (tmp_path / "vidF.json").read_text(encoding="utf-8") == "{corrupt"   # untouched
    data = json.loads((tmp_path / "vidG.json").read_text(encoding="utf-8"))     # batch continued
    assert data["chapters_provenance"] == "creator"


def test_run_chapters_mode_fetch_failure_is_reported_not_fatal(tmp_path, monkeypatch, capsys):
    def boom(vid):
        raise RuntimeError("network down")
    monkeypatch.setattr(MG, "fetch_chapters", boom)
    MG.run_chapters_mode(["vidZ", "vidZ2"], golden_dir=tmp_path)   # second id still attempted
    out = capsys.readouterr().out
    assert out.count("chapter fetch failed") == 2
    assert not (tmp_path / "vidZ.json").exists()


# ── run_eval gating: chapter-gold metrics appear iff the gold has chapters ────
def test_measure_emits_chapter_gold_metrics_only_with_gold_chapters():
    from backend.pipeline.understand.models import ContentMap, ContentNode, Structure
    nodes = [ContentNode(node_id="t1", level="topic", start=0.0, end=10.0),
             ContentNode(node_id="t2", level="topic", start=10.0, end=20.0),
             ContentNode(node_id="c1", level="chapter", start=0.0, end=20.0)]
    st = Structure(video_id="v", content_map=ContentMap(nodes=nodes))
    det = types.SimpleNamespace(domain="lecture")
    sents = _sents([(i, i + 1) for i in range(20)])        # 20 sentences, 1s each
    gold = {"chapters": [{"start_time": 0, "end_time": 10, "title": "a"},
                         {"start_time": 10, "end_time": 20, "title": "b"}]}

    m = R._measure(st, [], sents, None, det, "", gold, {}, verbose=False)
    # ref = 10+10 sentences → k=5; predicted topics match gold exactly → 0.0
    assert m["pk_topics"] == 0.0
    assert m["windowdiff_topics"] == 0.0
    # single predicted chapter vs two gold segments: probes i=5..9 disagree → 5/15
    assert m["pk_chapters"] == pytest.approx(round(5 / 15, 3))
    assert m["windowdiff_chapters"] == pytest.approx(round(5 / 15, 3))
    assert m["clip_straddle_rate"] is None                 # no shipped clips → NaN → null

    m2 = R._measure(st, [], sents, None, det, "", {}, {}, verbose=False)
    for key in ("pk_topics", "windowdiff_topics", "pk_chapters", "windowdiff_chapters",
                "clip_straddle_rate"):
        assert key not in m2                               # absent gold → absent keys
