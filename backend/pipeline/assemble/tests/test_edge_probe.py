"""VID2 edge-probe (Tier-1 video judge) tests. Fully OFFLINE: the video-judge SDK call and
the ffmpeg cut are monkeypatched — ZERO network, ZERO live LLM, ZERO real ffmpeg.

Covers: the orchestrator-gated OFF path (no probe when disabled), the advisory warning
mapping (False → warning, clean → no warning, and NEVER a Rejection / kill), fail-soft on an
LLM error, and that the SDK enablers exist + call generate_content as expected.
"""
from __future__ import annotations

import backend.config as config
import backend.gemini_client as gc
import backend.pipeline.assemble.edge_probe as ep
from backend.orchestrator import _edge_probe_enabled


def _spec(**kw) -> dict:
    # final_quality/boundary_score are the honest component-derived baseline
    # (scoring.quality(0.8, 0.9, 1.0, 0.5) == 0.84) so a post-probe dock is measurable.
    base = {"start": 100.0, "end": 160.0, "warnings": (),
            "completeness_score": 0.8, "grounding_score": 0.9, "priority": 0.5,
            "boundary_score": 1.0, "final_quality": 0.84}
    base.update(kw)
    return base


def _verdict(starts=True, ends=True) -> ep.EdgeVerdict:
    return ep.EdgeVerdict(starts_clean_audio=starts, ends_clean_audio=ends,
                          first_words="so let us", last_words="that is all", evidence="ok")


# ── (1) default OFF: the flag gate keeps the probe from ever running ──────────
def test_edge_probe_disabled_by_default_and_flag_gate(monkeypatch):
    assert config.EDGE_PROBE_ENABLED is False
    assert config.VIDEO_JUDGE_ENABLED is False
    # DEFAULTS carries the per-job override key defaulting to None → inherit config (OFF).
    assert config.DEFAULTS["edge_probe"] is None
    assert _edge_probe_enabled(dict(config.DEFAULTS)) is False
    assert _edge_probe_enabled({"edge_probe": None}) is False
    assert _edge_probe_enabled({"edge_probe": True}) is True      # explicit per-job dial
    monkeypatch.setattr(config, "EDGE_PROBE_ENABLED", True)
    assert _edge_probe_enabled({}) is True                        # inherits config when unset


def test_run_edge_probe_noop_without_video_path(monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(ep, "_probe_clip", lambda *a, **k: called.__setitem__("n", called["n"] + 1))
    specs = [_spec()]
    out = ep.run_edge_probe(specs, video_path="", settings={})
    assert out is specs and called["n"] == 0
    assert "starts_clean_audio" not in specs[0]                   # untouched


# ── (2) advisory mapping: False → warning; clean → none; never a kill/Rejection ──
def test_starts_not_clean_adds_only_a_warning(monkeypatch):
    monkeypatch.setattr(ep, "_cut_segment", lambda *a, **k: b"fake-mp4-bytes")
    monkeypatch.setattr(ep, "generate_json_video",
                        lambda *a, **k: _verdict(starts=False, ends=True).model_dump_json(),
                        raising=False)
    spec = _spec()
    ep.run_edge_probe([spec], video_path="/tmp/v.mp4", settings={})
    assert "starts_mid_sentence_audio" in spec["warnings"]
    assert "ends_mid_sentence_audio" not in spec["warnings"]
    assert spec["starts_clean_audio"] is False and spec["ends_clean_audio"] is True
    # advisory dock only — final_quality docked below the 0.84 baseline but the clip still SHIPS
    # (no drop, no Rejection). boundary weight 0.20 × 0.05 penalty ⇒ 0.84 → 0.83.
    assert spec["final_quality"] == 0.83 and spec["final_quality"] < 0.84


def test_both_edges_flagged(monkeypatch):
    monkeypatch.setattr(ep, "_cut_segment", lambda *a, **k: b"bytes")
    monkeypatch.setattr(ep, "generate_json_video",
                        lambda *a, **k: _verdict(starts=False, ends=False).model_dump_json(),
                        raising=False)
    spec = _spec()
    ep.run_edge_probe([spec], "/tmp/v.mp4", {})
    assert {"starts_mid_sentence_audio", "ends_mid_sentence_audio"} <= set(spec["warnings"])


def test_clean_verdict_adds_no_warning_and_no_dock(monkeypatch):
    monkeypatch.setattr(ep, "_cut_segment", lambda *a, **k: b"bytes")
    monkeypatch.setattr(ep, "generate_json_video",
                        lambda *a, **k: _verdict(True, True).model_dump_json(), raising=False)
    spec = _spec()
    ep.run_edge_probe([spec], "/tmp/v.mp4", {})
    assert spec["warnings"] == ()                                 # no edge warnings
    assert spec["final_quality"] == 0.84                          # no dock on a clean verdict
    assert spec["starts_clean_audio"] is True and spec["ends_clean_audio"] is True


def test_probe_never_kills_or_rejects(monkeypatch):
    # run_edge_probe returns the SAME list object, mutated in place — it can only add fields,
    # never drop a clip or emit a Rejection (it has no rejections channel at all).
    monkeypatch.setattr(ep, "_cut_segment", lambda *a, **k: b"bytes")
    monkeypatch.setattr(ep, "generate_json_video",
                        lambda *a, **k: _verdict(False, False).model_dump_json(), raising=False)
    specs = [_spec(), _spec(start=200.0, end=260.0)]
    out = ep.run_edge_probe(specs, "/tmp/v.mp4", {})
    assert out is specs and len(out) == 2                         # nothing dropped


# ── (3) fail-soft: an LLM error leaves the clip untouched, never raises ───────
def test_llm_error_is_fail_soft(monkeypatch):
    monkeypatch.setattr(ep, "_cut_segment", lambda *a, **k: b"bytes")

    def boom(*a, **k):
        raise RuntimeError("gemini 503")

    monkeypatch.setattr(ep, "generate_json_video", boom, raising=False)
    spec = _spec()
    out = ep.run_edge_probe([spec], "/tmp/v.mp4", {})             # must NOT raise
    assert out[0] is spec
    assert "starts_clean_audio" not in spec                       # untouched
    assert spec["warnings"] == () and spec["final_quality"] == 0.84


def test_ffmpeg_cut_failure_is_fail_soft(monkeypatch):
    # both head+tail cuts fail → _probe_clip returns None → clip untouched, no exception.
    monkeypatch.setattr(ep, "_cut_segment", lambda *a, **k: None)
    monkeypatch.setattr(ep, "generate_json_video",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call LLM")),
                        raising=False)
    spec = _spec()
    ep.run_edge_probe([spec], "/tmp/v.mp4", {})
    assert "starts_clean_audio" not in spec and spec["warnings"] == ()


def test_cut_segment_no_real_ffmpeg(monkeypatch):
    # _cut_segment must fail-soft (return None) if the subprocess errors — never let a real
    # ffmpeg run or an exception escape.
    import subprocess

    def fake_run(*a, **k):
        raise FileNotFoundError("ffmpeg missing")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert ep._cut_segment("/tmp/v.mp4", 0.0, 8.0) is None
    assert ep._cut_segment("/tmp/v.mp4", 0.0, 0.0) is None        # zero-duration guard
    assert ep._cut_segment("", 0.0, 8.0) is None                  # no source guard


# ── (4) the SDK enablers exist and call generate_content as expected ─────────
class _FakeResp:
    text = '{"starts_clean_audio": true, "ends_clean_audio": true}'


class _FakeModels:
    def __init__(self):
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return _FakeResp()


class _FakeClient:
    def __init__(self):
        self.models = _FakeModels()


def test_video_part_inline_builds_video_mp4_part():
    from google.genai import types
    p = gc.video_part_inline(b"abc", media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW)
    assert p.inline_data.mime_type == "video/mp4"
    assert p.media_resolution is not None                         # media_resolution attached
    p2 = gc.video_part_inline(b"abc")
    assert p2.inline_data.mime_type == "video/mp4" and p2.media_resolution is None


# ── embed-record threading: edge booleans surface only when the probe ran ────
def test_build_embed_clips_omits_edge_keys_when_probe_off():
    from backend.orchestrator import _build_embed_clips
    c = _build_embed_clips([_spec()], "vid")[0]                   # spec has no edge fields
    assert "starts_clean_audio" not in c and "ends_clean_audio" not in c


def test_build_embed_clips_surfaces_edge_booleans_when_present():
    from backend.orchestrator import _build_embed_clips
    spec = _spec(starts_clean_audio=False, ends_clean_audio=True)
    c = _build_embed_clips([spec], "vid")[0]
    assert c["starts_clean_audio"] is False and c["ends_clean_audio"] is True


def test_generate_json_video_calls_sdk_with_schema_and_video_model(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(gc, "get_client", lambda: fake)
    parts = [gc.text_part("beginning"), gc.video_part_inline(b"vid")]
    out = gc.generate_json_video("SYS", parts, ep.EdgeVerdict,
                                 media_resolution=gc.media_resolution_from_name("low"))
    assert out == _FakeResp.text
    call = fake.models.calls[0]
    assert call["model"] == config.VIDEO_JUDGE_MODEL              # flash-lite, not GEMINI_MODEL
    assert call["contents"] is parts
    cfg = call["config"]
    assert cfg.response_schema is ep.EdgeVerdict
    assert cfg.response_mime_type == "application/json"
    assert cfg.media_resolution is not None
    assert cfg.thinking_config is not None                       # thinking-off first attempt
