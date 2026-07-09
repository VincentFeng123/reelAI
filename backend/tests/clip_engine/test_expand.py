# backend/tests/clip_engine/test_expand.py
from backend.app.clip_engine import expand


def test_free_expand_includes_educational_templates():
    """Fallback expansion includes educational variants; dedup/order/contract intact."""
    out = expand.free_expand("photosynthesis", 10)
    queries_lower = [q.lower() for q in out["queries"]]
    assert "photosynthesis explained" in queries_lower
    assert "photosynthesis lecture" in queries_lower
    assert "how photosynthesis works" in queries_lower
    assert "photosynthesis course" in queries_lower
    assert "photosynthesis tutorial" in queries_lower
    # dedup intact
    assert len(set(q.lower() for q in out["queries"])) == len(out["queries"])
    # corrected/topic first
    assert out["queries"][0].lower() == "photosynthesis"
    # contract intact
    assert "corrected" in out and "queries" in out


def test_system_prompt_has_educational_steering():
    """_SYSTEM asserts educational steering exists and entertainment avoidance."""
    s = expand._SYSTEM.lower()
    # must steer toward teaching content
    assert any(kw in s for kw in ("educational", "lecture", "course", "teaching"))
    # must warn against entertainment
    assert any(kw in s for kw in ("entertainment", "reaction", "meme", "compilation"))


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


def test_expand_level_steering_lines(monkeypatch):
    from backend.app.clip_engine import expand as ex
    seen = {}

    def fake_raw(system, user, model):
        seen["system"] = system
        return '{"corrected": "physics", "queries": ["physics"]}'

    monkeypatch.setattr(ex.config, "GEMINI_API_KEY", "k")
    monkeypatch.setattr(ex, "_gemini_expand_raw", fake_raw)

    ex.expand_query("physics", 3, level="beginner")
    assert "beginner" in seen["system"].lower()
    assert "introduction to" in seen["system"]

    ex.expand_query("physics", 3, level="advanced")
    assert "advanced" in seen["system"].lower()
    assert "graduate" in seen["system"]

    ex.expand_query("physics", 3, level="intermediate")
    assert "graduate" not in seen["system"] and "for beginners" not in seen["system"]

    ex.expand_query("physics", 3)  # None unchanged
    assert "graduate" not in seen["system"] and "for beginners" not in seen["system"]
