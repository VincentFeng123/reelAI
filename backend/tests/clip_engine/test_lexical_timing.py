from __future__ import annotations

import json
import time

import httpx
import pytest

from backend.app.clip_engine import lexical_timing
from backend.app.clip_engine.errors import CancellationError
from backend.app.clip_engine.lexical_timing import (
    Json3CaptionTrack,
    LexicalWord,
    align_edge_anchor,
    fetch_json3_words,
    parse_json3_words,
    select_original_json3_track,
)


def _word(text: str, onset: float) -> LexicalWord:
    return LexicalWord(text=text, onset_sec=onset)


def test_selects_only_original_expected_language_json3_and_hides_url() -> None:
    secret = "https://captions.example/original?signature=secret"
    info = {
        "automatic_captions": {
            "en": [{"ext": "json3", "url": "https://captions.example/en?tlang=en"}],
            "fr-orig": [{"ext": "json3", "url": "https://captions.example/fr"}],
            "en-orig": [
                {"ext": "vtt", "url": "https://captions.example/original.vtt"},
                {"ext": "json3", "url": secret, "impersonate": True},
            ],
        },
        "subtitles": {"en": [{"ext": "json3", "url": "https://manual.example"}]},
    }

    track = select_original_json3_track(info, expected_language="en-US")

    assert track is not None
    assert track.language == "en"
    assert track.url == secret
    assert track.impersonate is True
    assert secret not in repr(track)


def test_original_key_precedes_validated_exact_language_asr_alias() -> None:
    original = "https://captions.example/timed?lang=en&kind=asr&source=orig"
    alias = "https://captions.example/timed?lang=en&kind=asr&source=alias"
    info = {
        "automatic_captions": {
            "en": [{"ext": "json3", "url": alias}],
            "en-orig": [{"ext": "json3", "url": original}],
        }
    }

    track = select_original_json3_track(info, expected_language="en-US")

    assert track is not None
    assert track.url == original


def test_validated_exact_language_asr_alias_is_safe_fallback() -> None:
    alias = "https://captions.example/timed?lang=en&kind=asr"

    track = select_original_json3_track(
        {"automatic_captions": {"en": [{"ext": "json3", "url": alias}]}},
        expected_language="en-US",
    )

    assert track == Json3CaptionTrack("en", alias)


@pytest.mark.parametrize(
    "url",
    [
        "https://captions.example/timed?lang=en",
        "https://captions.example/timed?lang=fr&kind=asr",
        "https://captions.example/timed?lang=en&kind=asr&tlang=en",
    ],
)
def test_unproven_asr_aliases_fail_closed(url: str) -> None:
    info = {"automatic_captions": {"en": [{"ext": "json3", "url": url}]}}

    assert select_original_json3_track(info, expected_language="en") is None


@pytest.mark.parametrize(
    ("info", "language"),
    [
        ({"automatic_captions": {"en": [{"ext": "json3", "url": "https://x.test"}]}}, "en"),
        ({"automatic_captions": {"en-orig": [{"ext": "json3", "url": "https://x.test?tlang=fr"}]}}, "en"),
        ({"automatic_captions": {"fr-orig": [{"ext": "json3", "url": "https://x.test"}]}}, "en"),
    ],
)
def test_track_selection_fails_closed_for_alias_translation_and_wrong_language(
    info: dict, language: str,
) -> None:
    assert select_original_json3_track(info, expected_language=language) is None


def test_json3_parser_uses_only_explicit_offsets_and_proven_first_word_zero() -> None:
    payload = {
        "events": [
            {
                "tStartMs": 1_000,
                "segs": [
                    {"utf8": "Alpha "},
                    {"utf8": "beta", "tOffsetMs": 250},
                ],
            },
            {"tStartMs": 2_000, "segs": [{"utf8": "untimed"}]},
            {"tStartMs": 3_000, "segs": [{"utf8": "two words", "tOffsetMs": 0}]},
            {"tStartMs": 4_000, "segs": [{"utf8": "Gamma", "tOffsetMs": 125}]},
        ]
    }

    assert parse_json3_words(payload) == (
        _word("alpha", 1.0),
        _word("beta", 1.25),
        _word("gamma", 4.125),
    )


def test_json3_parser_rejects_partially_timed_or_nonmonotonic_events() -> None:
    payload = {
        "events": [
            {
                "tStartMs": 1_000,
                "segs": [
                    {"utf8": "one", "tOffsetMs": 0},
                    {"utf8": "two"},
                ],
            },
            {
                "tStartMs": 2_000,
                "segs": [
                    {"utf8": "three", "tOffsetMs": 200},
                    {"utf8": "four", "tOffsetMs": 100},
                ],
            },
        ]
    }

    assert parse_json3_words(payload) == ()


def test_fetches_one_url_with_supplied_transport_bounds(monkeypatch) -> None:
    captured: dict = {"calls": []}
    payload = {
        "events": [
            {
                "tStartMs": 500,
                "segs": [
                    {"utf8": "exact", "tOffsetMs": 0},
                    {"utf8": "timing", "tOffsetMs": 100},
                ],
            }
        ]
    }

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, url, *, headers):
            captured["calls"].append((url, headers))
            return type(
                "Response",
                (),
                {"status_code": 200, "content": json.dumps(payload).encode("utf-8")},
            )()

    monkeypatch.setattr(lexical_timing.httpx, "AsyncClient", FakeClient)
    track = Json3CaptionTrack("en", "https://captions.example/json3?signature=secret")

    words = fetch_json3_words(
        track,
        headers={"User-Agent": "StudyReels"},
        proxy_url="http://proxy.example:8080",
        deadline=time.monotonic() + 0.5,
    )

    assert words == (_word("exact", 0.5), _word("timing", 0.6))
    assert len(captured["calls"]) == 1
    assert captured["calls"][0][1] == {"User-Agent": "StudyReels"}
    assert captured["client"]["proxy"] == "http://proxy.example:8080"
    assert captured["client"]["follow_redirects"] is False
    assert 0 < captured["client"]["timeout"].read <= 0.5


def test_impersonated_track_uses_curl_transport_with_same_bounds(monkeypatch) -> None:
    captured: dict = {}
    payload = {
        "events": [
            {
                "tStartMs": 500,
                "segs": [
                    {"utf8": "exact", "tOffsetMs": 0},
                    {"utf8": "timing", "tOffsetMs": 100},
                ],
            }
        ]
    }

    class FakeSession:
        def __init__(self, **kwargs):
            captured["session"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, url, **kwargs):
            captured["url"] = url
            captured["request"] = kwargs
            return type(
                "Response",
                (),
                {"status_code": 200, "content": json.dumps(payload).encode("utf-8")},
            )()

    monkeypatch.setattr(lexical_timing.curl_requests, "AsyncSession", FakeSession)
    track = Json3CaptionTrack(
        "en",
        "https://captions.example/json3?signature=secret",
        impersonate=True,
    )

    words = fetch_json3_words(
        track,
        headers={
            "User-Agent": "StudyReels",
            "Sec-CH-UA": '"StudyReels";v="1"',
            "Accept-Encoding": "gzip",
            "X-Study-Reels": "caption-timing",
        },
        proxy_url="http://proxy.example:8080",
        deadline=time.monotonic() + 0.5,
    )

    assert words == (_word("exact", 0.5), _word("timing", 0.6))
    assert captured["session"] == {
        "impersonate": "chrome",
        "max_clients": 1,
        "trust_env": False,
        "curl_options": {},
    }
    assert captured["request"]["headers"] == {
        "X-Study-Reels": "caption-timing"
    }
    assert captured["request"]["proxy"] == "http://proxy.example:8080"
    assert captured["request"]["allow_redirects"] is False
    assert 0 < captured["request"]["timeout"] <= 0.5


def test_direct_impersonated_fetch_explicitly_disables_environment_proxy(
    monkeypatch,
) -> None:
    captured: dict = {}
    payload = {
        "events": [
            {
                "tStartMs": 500,
                "segs": [
                    {"utf8": "exact", "tOffsetMs": 0},
                    {"utf8": "timing", "tOffsetMs": 100},
                ],
            }
        ]
    }

    class FakeSession:
        def __init__(self, **kwargs):
            captured["session"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, _url, **_kwargs):
            return type(
                "Response",
                (),
                {"status_code": 200, "content": json.dumps(payload).encode("utf-8")},
            )()

    monkeypatch.setattr(lexical_timing.curl_requests, "AsyncSession", FakeSession)

    words = fetch_json3_words(
        Json3CaptionTrack("en", "https://captions.example/json3", impersonate=True),
        deadline=time.monotonic() + 0.5,
    )

    assert words == (_word("exact", 0.5), _word("timing", 0.6))
    assert captured["session"]["trust_env"] is False
    assert captured["session"]["curl_options"] == {
        lexical_timing.CurlOpt.PROXY: ""
    }


def test_expired_deadline_makes_no_request_and_cancel_propagates(monkeypatch) -> None:
    calls = 0

    class FakeClient:
        def __init__(self, **_kwargs):
            nonlocal calls
            calls += 1

    monkeypatch.setattr(lexical_timing.httpx, "AsyncClient", FakeClient)
    track = Json3CaptionTrack("en", "https://captions.example/private")

    assert fetch_json3_words(track, deadline=time.monotonic() - 1) == ()
    assert calls == 0
    with pytest.raises(CancellationError):
        fetch_json3_words(
            track,
            deadline=time.monotonic() + 1,
            cancel_check=lambda: True,
        )
    assert calls == 0


def test_caption_fetch_has_a_small_latency_budget(monkeypatch) -> None:
    observed: list[float] = []

    async def fake_fetch(_track, **kwargs):
        observed.append(float(kwargs["timeout_sec"]))
        return None

    monkeypatch.setattr(lexical_timing, "_fetch_payload", fake_fetch)

    assert fetch_json3_words(
        Json3CaptionTrack("en", "https://captions.example/private"),
        deadline=time.monotonic() + 30.0,
    ) == ()
    assert observed == [lexical_timing.MAX_FETCH_TIMEOUT_SEC]
    assert observed[0] <= 2.0


def test_fetch_failure_does_not_expose_signed_url(monkeypatch) -> None:
    secret = "https://captions.example/json3?signature=do-not-log"

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, url, *, headers):
            request = httpx.Request("GET", url, headers=headers)
            raise httpx.ConnectError("signed request failed", request=request)

    monkeypatch.setattr(lexical_timing.httpx, "AsyncClient", FakeClient)
    track = Json3CaptionTrack("en", secret)

    assert fetch_json3_words(track, deadline=time.monotonic() + 1) == ()
    assert secret not in repr(track)


def test_impersonated_fetch_failure_remains_closed_and_hides_url(monkeypatch) -> None:
    secret = "https://captions.example/json3?signature=do-not-log"

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, _url, **_kwargs):
            raise lexical_timing.curl_requests.errors.RequestsError("request failed")

    monkeypatch.setattr(
        lexical_timing.curl_requests,
        "AsyncSession",
        lambda **_kwargs: FakeSession(),
    )
    track = Json3CaptionTrack("en", secret, impersonate=True)

    assert fetch_json3_words(track, deadline=time.monotonic() + 1) == ()
    assert secret not in repr(track)


def test_start_anchor_uses_unique_unicode_normalized_quote_and_prefix() -> None:
    words = [
        _word("Welcome", 1.0),
        _word("CAFÉ", 2.0),
        _word("science", 3.0),
        _word("begins", 4.0),
        _word("now", 5.0),
    ]

    anchor = align_edge_anchor(
        words,
        cue_text="Welcome — CAFÉ science begins now.",
        quote="Cafe\u0301 science",
        edge="start",
        cue_start_sec=0.95,
        cue_end_sec=5.1,
    )

    assert anchor is not None
    assert anchor.anchor_sec == 2.0
    assert anchor.quote_start_sec == 2.0
    assert anchor.quote_last_onset_sec == 3.0
    assert anchor.excluded_neighbor_onset_sec == 1.0


def test_end_anchor_is_first_excluded_suffix_onset() -> None:
    words = [
        _word("The", 1.0),
        _word("result", 2.0),
        _word("is", 3.0),
        _word("forty", 4.0),
        _word("two", 5.0),
        _word("anyway", 6.0),
        _word("thanks", 7.0),
    ]

    anchor = align_edge_anchor(
        words,
        cue_text="The result is forty two. Anyway, thanks.",
        quote="result is forty two",
        edge="end",
        cue_start_sec=0.95,
        cue_end_sec=7.1,
    )

    assert anchor is not None
    assert anchor.anchor_sec == 6.0
    assert anchor.quote_start_sec == 2.0
    assert anchor.quote_last_onset_sec == 5.0
    assert anchor.excluded_neighbor_onset_sec == 6.0


@pytest.mark.parametrize("edge", ["start", "end"])
def test_alignment_tolerates_provider_drift_away_from_verified_edge(
    edge: str,
) -> None:
    words = [
        _word("Hello", 1.0),
        _word("learners", 2.0),
        _word("Python", 3.0),
        _word("functions", 4.0),
        _word("package", 5.0),
        _word("reusable", 6.0),
        _word("instructions", 7.0),
        _word("thanks", 8.0),
        _word("everyone", 9.0),
    ]

    anchor = align_edge_anchor(
        words,
        cue_text=(
            "Welcome learners. Python functions package reusable instructions. "
            "Thanks for listening."
        ),
        quote=(
            "Python functions package"
            if edge == "start"
            else "reusable instructions"
        ),
        edge=edge,
        cue_start_sec=0.9,
        cue_end_sec=9.1,
    )

    assert anchor is not None
    assert anchor.anchor_sec == (3.0 if edge == "start" else 8.0)


@pytest.mark.parametrize(
    ("edge", "words"),
    [
        (
            "start",
            [
                _word("different", 1),
                _word("python", 2),
                _word("functions", 3),
                _word("package", 4),
            ],
        ),
        (
            "end",
            [
                _word("reusable", 1),
                _word("instructions", 2),
                _word("different", 3),
            ],
        ),
    ],
)
def test_alignment_requires_the_excluded_edge_neighbor(
    edge: str,
    words: list[LexicalWord],
) -> None:
    assert align_edge_anchor(
        words,
        cue_text=(
            "Welcome. Python functions package."
            if edge == "start"
            else "Reusable instructions. Thanks."
        ),
        quote=("Python functions package" if edge == "start" else "Reusable instructions"),
        edge=edge,
        cue_start_sec=0.5,
        cue_end_sec=5.0,
    ) is None


@pytest.mark.parametrize(
    ("cue_text", "quote", "edge", "words"),
    [
        (
            "Teach cells, then teach cells, finally stop",
            "teach cells",
            "end",
            [
                _word("teach", 1), _word("cells", 2), _word("then", 3),
                _word("teach", 4), _word("cells", 5), _word("finally", 6), _word("stop", 7),
            ],
        ),
        ("Teach cells then stop", "teach atoms", "end", []),
        (
            "Teach cells then stop",
            "Teach cells",
            "start",
            [_word("teach", 1), _word("cells", 2), _word("then", 3), _word("stop", 4)],
        ),
        (
            "First teach cells",
            "teach cells",
            "end",
            [_word("first", 1), _word("teach", 2), _word("cells", 3)],
        ),
        (
            "First teach cells then stop",
            "teach cells",
            "end",
            [_word("first", 1), _word("teach", 2), _word("then", 4), _word("stop", 5)],
        ),
    ],
)
def test_alignment_fails_closed_for_repeated_mismatch_whole_edge_and_untimed_context(
    cue_text: str,
    quote: str,
    edge: str,
    words: list[LexicalWord],
) -> None:
    assert align_edge_anchor(
        words,
        cue_text=cue_text,
        quote=quote,
        edge=edge,
        cue_start_sec=0.5,
        cue_end_sec=10.0,
    ) is None
