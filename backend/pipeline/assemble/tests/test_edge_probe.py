"""VID2 edge-probe (Tier-1 video judge) tests. Fully OFFLINE: the video-judge SDK call and
the ffmpeg cut are monkeypatched — ZERO network, ZERO live LLM, ZERO real ffmpeg.

Covers: the orchestrator-gated OFF path (no probe when disabled), the advisory warning
mapping (False → warning, clean → no warning, and NEVER a Rejection / kill), fail-soft on an
LLM error, and that the SDK enablers exist + call generate_content as expected.
"""
from __future__ import annotations

import pytest

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
    assert _edge_probe_enabled({"edge_probe": True}) is False
    monkeypatch.setattr(config, "EDGE_PROBE_ENABLED", True)
    assert _edge_probe_enabled({}) is False


def test_run_edge_probe_noop_without_video_path(monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(ep, "_probe_clip", lambda *a, **k: called.__setitem__("n", called["n"] + 1))
    specs = [_spec()]
    out = ep.run_edge_probe(specs, video_path="", settings={})
    assert out is specs and called["n"] == 0
    assert "starts_clean_audio" not in specs[0]                   # untouched


# ── (2) hard-disabled: even explicit settings cannot upload video ────────────
def test_run_edge_probe_never_cuts_or_calls_gemini(monkeypatch):
    monkeypatch.setattr(
        ep,
        "_probe_clip",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("video probe must remain disabled")
        ),
    )
    specs = [_spec(), _spec(start=200.0, end=260.0)]
    before = [dict(spec) for spec in specs]

    out = ep.run_edge_probe(
        specs,
        "/tmp/v.mp4",
        {"edge_probe": True},
    )

    assert out is specs
    assert specs == before


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


def test_video_helpers_are_hard_disabled():
    with pytest.raises(ValueError, match="video input is disabled"):
        gc.video_part_inline(b"abc")


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


def test_generate_json_video_never_calls_sdk(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(gc, "get_client", lambda: fake)
    with pytest.raises(ValueError, match="video input is disabled"):
        gc.generate_json_video("SYS", [], ep.EdgeVerdict)
    assert fake.models.calls == []
