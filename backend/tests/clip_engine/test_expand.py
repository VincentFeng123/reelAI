# backend/tests/clip_engine/test_expand.py
from backend.app.clip_engine import expand


def test_free_fallback_when_no_key(monkeypatch):
    monkeypatch.setattr(expand.config, "GEMINI_API_KEY", "")
    out = expand.expand_query("calculus", 4)
    assert out["provider_used"] == "free"
    assert out["queries"][0] == "calculus"
    assert len(out["queries"]) <= 4
    assert len(set(q.lower() for q in out["queries"])) == len(out["queries"])  # deduped


def test_gemini_path_parses_json(monkeypatch):
    monkeypatch.setattr(expand.config, "GEMINI_API_KEY", "g_test")
    monkeypatch.setattr(
        expand, "_gemini_expand_raw",
        lambda system, user, model: '{"corrected": "calculus", "queries": ["calculus", "derivatives", "integrals"]}',
    )
    out = expand.expand_query("calculas", 5)
    assert out["provider_used"] == "gemini"
    assert out["corrected"] == "calculus"
    assert out["queries"][:2] == ["calculus", "derivatives"]
