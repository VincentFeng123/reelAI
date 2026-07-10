import threading

import pytest

from backend.app.clip_engine import expand
from backend.app.clip_engine.errors import CancellationError


def test_deterministic_expand_includes_educational_templates() -> None:
    output = expand.expand_query("photosynthesis", 10)
    queries = [query.casefold() for query in output["queries"]]
    assert "photosynthesis explained" in queries
    assert "photosynthesis lecture" in queries
    assert "how photosynthesis works" in queries
    assert "photosynthesis course" in queries
    assert "photosynthesis tutorial" in queries
    assert output["queries"][0] == "photosynthesis"
    assert output["provider_used"] == "deterministic"


def test_expand_query_has_no_gemini_provider_surface(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "must-not-be-used")
    output = expand.expand_query("calculus", 4)
    assert output["corrected"] == "calculus"
    assert len(output["queries"]) == 4
    assert not hasattr(expand, "_gemini_expand_raw")
    assert not hasattr(expand, "_gemini_expand_raw_async")


def test_level_templates_are_deterministic_and_truthful() -> None:
    beginner = expand.expand_query("physics", 4, level="beginner")["queries"]
    advanced = expand.expand_query("physics", 4, level="advanced")["queries"]
    intermediate = expand.expand_query("physics", 4, level="intermediate")["queries"]
    assert beginner[1:] == [
        "introduction to physics", "physics basics", "physics explained"
    ]
    assert advanced[1:] == [
        "advanced physics", "graduate physics lecture", "physics deep dive"
    ]
    assert "graduate physics lecture" not in intermediate
    assert "physics for beginners" not in intermediate


def test_unicode_normalization_prevents_duplicate_queries() -> None:
    assert expand._normalize(["CAFÉ", "cafe\u0301", "  café  "], 5) == ["CAFÉ"]


def test_expand_observes_cancellation_without_starting_provider_work() -> None:
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(CancellationError):
        expand.expand_query("calculus", 3, should_cancel=cancelled.is_set)
