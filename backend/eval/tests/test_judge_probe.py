"""Corruption-based judge probe: generators are pure/deterministic, the antecedent skip
rule holds, splice donor selection is deterministic, aggregation math is exact, and
--dry-run makes ZERO llm_json calls. All offline — no live judging anywhere."""
from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

import backend.eval.judge_probe as JP
from backend import config
from backend.pipeline.assemble.validate import JudgeVerdict
from backend.pipeline.sentences import Sentence
from backend.pipeline.understand.models import (
    ContentMap, ContentNode, Reference, Structure, Unit, save_structure,
)


# ── fixtures (10 sentences × 10s; two content-map topics; internal reference) ──
def _sent(i: int) -> Sentence:
    return Sentence(idx=i, text=f"sentence {i}.", start=i * 10.0, end=i * 10.0 + 9.9,
                    terminator=".", ends_with_period=True, word_start_idx=i, word_end_idx=i,
                    align_confidence=1.0)


def _sents(n: int = 10) -> list[Sentence]:
    return [_sent(i) for i in range(n)]


def _unit(uid: str, s0: int, s1: int, role: str = "explanation", node: str = "ch1.t1",
          refs=()) -> Unit:
    return Unit(unit_id=uid, start=s0 * 10.0, end=s1 * 10.0 + 9.9, sentence_range=(s0, s1),
                node_id=node, role=role, summary=f"summary of {uid}", transcript="",
                references=list(refs))


def _structure(units=None) -> Structure:
    nodes = [
        ContentNode(node_id="ch1", level="chapter", start=0.0, end=99.9, sentence_range=(0, 9)),
        ContentNode(node_id="ch1.t1", level="topic", start=0.0, end=49.9,
                    sentence_range=(0, 4), parent_id="ch1"),
        ContentNode(node_id="ch1.t2", level="topic", start=50.0, end=99.9,
                    sentence_range=(5, 9), parent_id="ch1"),
    ]
    if units is None:
        units = [
            _unit("u0", 0, 1, role="definition"),
            _unit("u1", 2, 3, role="claim",
                  refs=[Reference(text="this definition", resolves_to="momentum",
                                  source_unit="u0")]),
            _unit("u2", 5, 6, role="explanation", node="ch1.t2"),
            _unit("u3", 8, 9, role="claim", node="ch1.t2"),
        ]
    return Structure(video_id="vidprobe", title="t", duration=100.0,
                     content_map=ContentMap(nodes=nodes), units=units)


def _spec(**over) -> dict:
    s = {"cand_id": "c_u1", "anchor_id": "u1", "role": "claim",
         "unit_ids": ["u0", "u1"], "referential": [],
         "sentence_start_idx": 0, "sentence_end_idx": 3,
         "start": 0.0, "end": 39.9, "context_card": ""}
    s.update(over)
    return s


# ── chop_start / chop_end ─────────────────────────────────────────────────────
def test_chop_start_removes_first_sentence():
    c = JP.corrupt_chop_start(_spec(), _sents())
    assert c.kind == "chop_start"
    assert c.text == "sentence 1. sentence 2. sentence 3."
    assert c.relevant_gates == JP.RELEVANT_GATES["chop_start"]


def test_chop_end_removes_last_sentence():
    c = JP.corrupt_chop_end(_spec(), _sents())
    assert c.kind == "chop_end"
    assert c.text == "sentence 0. sentence 1. sentence 2."
    assert c.relevant_gates == JP.RELEVANT_GATES["chop_end"]


def test_chops_skip_single_sentence_clip():
    one = _spec(sentence_start_idx=2, sentence_end_idx=2)
    assert JP.corrupt_chop_start(one, _sents()) is None
    assert JP.corrupt_chop_end(one, _sents()) is None


# ── antecedent_removal ────────────────────────────────────────────────────────
def test_antecedent_removal_removes_source_unit_sentences():
    st = _structure()
    c = JP.corrupt_antecedent_removal(_spec(), _sents(), st.units_by_id())
    assert c.kind == "antecedent_removal"
    assert c.text == "sentence 2. sentence 3."          # u0's sentences 0-1 removed
    assert "u0" in c.detail and "this definition" in c.detail
    assert c.relevant_gates == JP.RELEVANT_GATES["antecedent_removal"]


def test_antecedent_removal_skips_when_source_unit_not_in_clip():
    st = _structure()
    # the referring unit u1 is in-clip but its source u0 is not → genuinely no variant
    c = JP.corrupt_antecedent_removal(
        _spec(unit_ids=["u1"], sentence_start_idx=2, sentence_end_idx=3),
        _sents(), st.units_by_id())
    assert c is None


def test_antecedent_removal_skips_when_no_references():
    st = _structure()
    c = JP.corrupt_antecedent_removal(
        _spec(unit_ids=["u0", "u2"]), _sents(), st.units_by_id())
    assert c is None


def test_antecedent_removal_skips_when_removal_would_empty_clip():
    st = _structure()
    # clip span covers ONLY the source unit's sentences → removal leaves nothing
    c = JP.corrupt_antecedent_removal(
        _spec(sentence_start_idx=0, sentence_end_idx=1), _sents(), st.units_by_id())
    assert c is None


# ── offtopic_splice ───────────────────────────────────────────────────────────
def test_offtopic_splice_uses_temporally_farthest_other_topic_unit():
    st = _structure()
    c = JP.corrupt_offtopic_splice(_spec(), _sents(), st)
    assert c.kind == "offtopic_splice"
    # clip 0-3 (topic ch1.t1); donors u2 (mid 60) and u3 (mid ~90) → u3 is farthest;
    # its first 2 sentences (8,9) land at the clip middle (position 2).
    assert c.text == ("sentence 0. sentence 1. sentence 8. sentence 9. "
                      "sentence 2. sentence 3.")
    assert "u3" in c.detail and "ch1.t2" in c.detail
    assert c.relevant_gates == JP.RELEVANT_GATES["offtopic_splice"]


def test_offtopic_splice_is_deterministic():
    st = _structure()
    a = JP.corrupt_offtopic_splice(_spec(), _sents(), st)
    b = JP.corrupt_offtopic_splice(_spec(), _sents(), st)
    assert a == b


def test_offtopic_splice_caps_donor_to_two_sentences():
    units = [
        _unit("u0", 0, 1, role="definition"),
        _unit("u1", 2, 3, role="claim"),
        _unit("u2", 5, 9, role="explanation", node="ch1.t2"),   # 5-sentence donor
    ]
    c = JP.corrupt_offtopic_splice(_spec(), _sents(), _structure(units))
    assert c.text == ("sentence 0. sentence 1. sentence 5. sentence 6. "
                      "sentence 2. sentence 3.")                # only 5-6 spliced


def test_offtopic_splice_skips_when_no_other_topic():
    units = [
        _unit("u0", 0, 1, role="definition"),
        _unit("u1", 2, 3, role="claim"),
        _unit("u2", 4, 4, role="explanation"),                  # same topic as the clip
    ]
    assert JP.corrupt_offtopic_splice(_spec(), _sents(), _structure(units)) is None


def test_offtopic_splice_skips_when_clip_topic_unknown():
    st = _structure()
    c = JP.corrupt_offtopic_splice(_spec(unit_ids=["nope"]), _sents(), st)
    assert c is None


# ── corruptions_for (kind coverage + skip propagation) ────────────────────────
def test_corruptions_for_yields_all_applicable_kinds():
    st = _structure()
    kinds = [c.kind for c in JP.corruptions_for(_spec(), _sents(), st)]
    assert kinds == ["chop_start", "chop_end", "antecedent_removal", "offtopic_splice"]


def test_corruptions_for_omits_antecedent_when_inapplicable():
    st = _structure()
    kinds = [c.kind for c in JP.corruptions_for(_spec(unit_ids=["u0", "u2"]), _sents(), st)]
    assert "antecedent_removal" not in kinds
    assert "chop_start" in kinds and "chop_end" in kinds


# ── detection rule ────────────────────────────────────────────────────────────
def test_is_flagged_on_relevant_gate_false():
    v = JudgeVerdict(score_10=9, source_grounded=False)
    assert JP.is_flagged(v, ("source_grounded",), 0.7) is True
    assert JP.is_flagged(v, ("topic_identifiable",), 0.7) is False   # gate not relevant


def test_is_flagged_on_score_below_threshold():
    v = JudgeVerdict(score_10=5)                                     # 0.5 < 0.7, gates all true
    assert JP.is_flagged(v, ("topic_identifiable",), 0.7) is True


def test_is_flagged_false_on_clean_pass():
    assert JP.is_flagged(JudgeVerdict(score_10=9), JP.GATE_UNION, 0.7) is False


# ── judge_rows wiring (judge_clip monkeypatched — never live) ─────────────────
def test_judge_rows_judges_original_plus_each_variant(monkeypatch):
    calls = []

    def fake_judge(text, role, adapter, visual_summary="", topic="", context_card=""):
        calls.append(text)
        return JudgeVerdict(score_10=9, understandable=True)

    monkeypatch.setattr(JP, "judge_clip", fake_judge)
    st = _structure()
    rows = JP.judge_rows("vidprobe", [_spec()], _sents(), st, None, "topic", 0.7)
    assert [r["kind"] for r in rows] == ["original", "chop_start", "chop_end",
                                         "antecedent_removal", "offtopic_splice"]
    assert len(calls) == 5                                    # 1 original + 4 variants
    assert calls[0] == "sentence 0. sentence 1. sentence 2. sentence 3."
    assert rows[0]["flagged"] is False                        # perfect verdict passes
    assert rows[0]["relevant_gates"] == list(JP.GATE_UNION)
    assert all(r["video_id"] == "vidprobe" and r["cand_id"] == "c_u1" for r in rows)


def test_judge_rows_error_verdicts_flag_none(monkeypatch):
    monkeypatch.setattr(JP, "judge_clip", lambda *a, **kw: JudgeVerdict(error=True))
    rows = JP.judge_rows("v", [_spec()], _sents(), _structure(), None, "t", 0.7)
    assert all(r["error"] is True and r["flagged"] is None for r in rows)


# ── aggregation math ──────────────────────────────────────────────────────────
def _mk_row(kind, flagged, score=0.9, error=False, gates=None, relevant=()):
    g = {x: True for x in JP.GATE_UNION}
    g.update(gates or {})
    return {"kind": kind, "error": error, "flagged": None if error else flagged,
            "score": score, "gates": g, "relevant_gates": list(relevant)}


def test_summarize_probe_tpr_tnr_and_per_gate():
    rows = [
        _mk_row("original", False, score=0.9),
        _mk_row("original", True, score=0.5, gates={"source_grounded": False}),
        _mk_row("chop_start", True, score=0.4, gates={"source_grounded": False},
                relevant=JP.RELEVANT_GATES["chop_start"]),
        _mk_row("chop_start", False, score=0.8,
                relevant=JP.RELEVANT_GATES["chop_start"]),
        _mk_row("antecedent_removal", True, error=True,          # outage: excluded from rates
                relevant=JP.RELEVANT_GATES["antecedent_removal"]),
    ]
    s = JP.summarize_probe(rows, 0.7)
    assert s["n_originals"] == 2 and s["n_originals_passed"] == 1
    assert s["tnr"] == pytest.approx(0.5)
    assert s["per_corruption"]["chop_start"]["n"] == 2
    assert s["per_corruption"]["chop_start"]["tpr"] == pytest.approx(0.5)
    assert s["per_corruption"]["chop_start"]["mean_score"] == pytest.approx(0.6)
    assert s["per_corruption"]["antecedent_removal"]["n"] == 0   # error row excluded
    assert s["per_corruption"]["antecedent_removal"]["tpr"] is None
    assert s["n_variant_errors"] == 1
    pg = s["per_gate"]["source_grounded"]
    assert pg["original_true_rate"] == pytest.approx(0.5)
    assert pg["corrupted_false_rate"] == pytest.approx(0.5)      # 1 of 2 relevant variants
    assert pg["n_corrupted_relevant"] == 2


def test_summarize_probe_empty_rows_yield_none_rates():
    s = JP.summarize_probe([], 0.7)
    assert s["tnr"] is None and s["n_originals"] == 0
    assert all(v["tpr"] is None for v in s["per_corruption"].values())


# ── results file (stamp supplied explicitly — no datetime in tests) ───────────
def test_write_results_creates_stamped_file(tmp_path):
    p = JP.write_results({"summary": {"tnr": 1.0}}, "TESTSTAMP", out_dir=tmp_path / "pr")
    assert p.name == "probe_TESTSTAMP.json"
    assert json.loads(p.read_text())["summary"]["tnr"] == 1.0


# ── CLI parsing (argparse: strict flags, ids required unless --all) ───────────
def test_parse_probe_args_zero_ids_errors(capsys):
    # the regression that bit the gate agent: a bare invocation must ERROR, never fall
    # through to a full live probe over every cached video.
    with pytest.raises(SystemExit) as e:
        JP.parse_probe_args([])
    assert e.value.code == 2
    assert "--all" in capsys.readouterr().err   # the error tells the user about the opt-in


def test_parse_probe_args_full():
    a = JP.parse_probe_args(["vid1", "vid2", "--limit", "3", "--dry-run"])
    assert a.video_ids == ["vid1", "vid2"]
    assert a.limit == 3
    assert a.dry_run is True
    assert a.all is False


def test_parse_probe_args_defaults_with_one_id():
    a = JP.parse_probe_args(["vid1"])
    assert a.video_ids == ["vid1"]
    assert a.limit is None and a.dry_run is False and a.all is False


def test_parse_probe_args_all_flag_allows_zero_ids():
    a = JP.parse_probe_args(["--all", "--dry-run"])
    assert a.all is True and a.video_ids == [] and a.dry_run is True


def test_parse_probe_args_rejects_ids_with_all():
    with pytest.raises(SystemExit):
        JP.parse_probe_args(["vid1", "--all"])


def test_parse_probe_args_unknown_flag_errors(capsys):
    with pytest.raises(SystemExit) as e:
        JP.parse_probe_args(["vid1", "--frobnicate"])
    assert e.value.code == 2
    assert "frobnicate" in capsys.readouterr().err


def test_parse_probe_args_invalid_limit_errors():
    for bad in (["vid", "--limit", "abc"],       # non-integer
                ["vid", "--limit", "0"],         # < 1
                ["vid", "--limit"]):             # missing value
        with pytest.raises(SystemExit):
            JP.parse_probe_args(bad)


def test_parse_probe_args_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as e:
        JP.parse_probe_args(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "--all" in out and "--limit" in out and "--dry-run" in out


def test_main_zero_ids_never_probes(tmp_path, monkeypatch):
    """main([]) with cached videos present must exit at parse time with ZERO llm_json calls
    — not run a full live probe."""
    import backend.llm as llm_mod

    monkeypatch.setattr(config, "WORK_DIR", tmp_path)
    vdir = tmp_path / "vidprobe"
    vdir.mkdir()
    chunks = [{"text": f"sentence {i}.", "start": i * 10.0, "end": i * 10.0 + 9.9}
              for i in range(10)]
    (vdir / "transcript.json").write_text(
        json.dumps({"source": "supadata", "title": "t", "chunks": chunks}), encoding="utf-8")
    save_structure(_structure())                # a cached video IS available — still no probe

    mock = Mock(side_effect=AssertionError("llm_json must never be called"))
    monkeypatch.setattr(llm_mod, "llm_json", mock)

    with pytest.raises(SystemExit):
        JP.main([])
    assert mock.call_count == 0


# ── --dry-run end-to-end: cached inputs → printed variants, ZERO llm_json calls ─
def test_dry_run_makes_zero_llm_json_calls(tmp_path, monkeypatch, capsys):
    import backend.llm as llm_mod

    monkeypatch.setattr(config, "WORK_DIR", tmp_path)
    vdir = tmp_path / "vidprobe"
    vdir.mkdir()
    chunks = [{"text": f"sentence {i}.", "start": i * 10.0, "end": i * 10.0 + 9.9}
              for i in range(10)]
    (vdir / "transcript.json").write_text(
        json.dumps({"source": "supadata", "title": "t", "chunks": chunks}), encoding="utf-8")
    save_structure(_structure())                # writes to the patched WORK_DIR

    mock = Mock(side_effect=AssertionError("llm_json must never be called in --dry-run"))
    monkeypatch.setattr(llm_mod, "llm_json", mock)

    JP.main(["vidprobe", "--dry-run"])

    assert mock.call_count == 0
    out = capsys.readouterr().out
    assert "chop_start" in out and "chop_end" in out and "offtopic_splice" in out


def test_dry_run_skips_video_without_cached_structure(tmp_path, monkeypatch, capsys):
    import backend.llm as llm_mod

    monkeypatch.setattr(config, "WORK_DIR", tmp_path)
    vdir = tmp_path / "nostructure"
    vdir.mkdir()
    (vdir / "transcript.json").write_text(json.dumps({"source": "supadata", "chunks": []}),
                                          encoding="utf-8")
    mock = Mock(side_effect=AssertionError("no llm calls on a skip"))
    monkeypatch.setattr(llm_mod, "llm_json", mock)

    JP.main(["nostructure", "--dry-run"])

    assert mock.call_count == 0
    assert "[skip] nostructure" in capsys.readouterr().out
