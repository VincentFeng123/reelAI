from __future__ import annotations

import sys
import types
import wave
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.app.clip_engine import groq_boundary_asr, silence
from backend.app.clip_engine import lexical_timing
from backend.app.clip_engine.lexical_timing import EdgeAnchor, LexicalWord


_POLYNOMIAL_CUE = (
    "So we're going to get -4 over x <unk>x. And so you can leave the answer "
    "like that if you want to. x <unk>x is also x^ 3. You could write it that "
    "way too, but that's the answer. Now the last example I'm going to go over "
    "is a polomial function. So let's say we have f(x)= x^2 - 5x + 9. "
    "What's the first derivative of this function? So first what's f x + h? "
    "Let's decide that."
)
_POLYNOMIAL_QUOTE = "So let's say we have f(x)= x^2 - 5x + 9."


def _words(texts: list[str], *, start: float, step: float = 0.16) -> tuple[LexicalWord, ...]:
    return tuple(
        LexicalWord(text=text, onset_sec=start + index * step)
        for index, text in enumerate(texts)
    )


def _write_wav(path: Path, *, duration_sec: float = 2.0) -> None:
    frame_count = round(groq_boundary_asr.EXPECTED_SAMPLE_RATE * duration_sec)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(groq_boundary_asr.EXPECTED_SAMPLE_RATE)
        handle.writeframes(b"\x00\x00" * frame_count)


def _prepared() -> silence.AudioPreparationResult:
    return silence.AudioPreparationResult(
        "ready",
        source=silence.PreparedAudioSource(
            url="https://media.example/signed-audio?secret=never-log-this",
            duration_sec=120.0,
            lexical_language="en-US",
        ),
    )


def _install_decode(monkeypatch: pytest.MonkeyPatch, wav_path: Path) -> dict[str, object]:
    captured: dict[str, object] = {}

    @contextmanager
    def decode(source: object, **kwargs: object):
        captured["source"] = source
        captured.update(kwargs)
        yield wav_path

    monkeypatch.setattr(silence, "decode_audio_window", decode)
    return captured


class _Transcriptions:
    def __init__(self, response: object = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.request: dict[str, object] | None = None

    def create(self, **kwargs: object) -> object:
        self.request = kwargs
        if self.error is not None:
            raise self.error
        return self.response


class _Client:
    def __init__(self, response: object = None, error: Exception | None = None) -> None:
        self.audio = SimpleNamespace(transcriptions=_Transcriptions(response, error))
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_uploads_only_bounded_wav_and_returns_absolute_word_onsets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    wav_path = tmp_path / "short.wav"
    _write_wav(wav_path)
    decode = _install_decode(monkeypatch, wav_path)
    client = _Client(
        {
            "words": [
                {"word": "cell", "start": 0.25, "end": 0.65},
                {"word": "division", "start": 0.65, "end": 1.4},
            ]
        }
    )
    client_args: dict[str, object] = {}

    def create_client(**kwargs: object) -> _Client:
        client_args.update(kwargs)
        return client

    monkeypatch.setenv("GROQ_API_KEY", "unit-test-key")
    monkeypatch.setattr(groq_boundary_asr, "_create_client", create_client)

    words = groq_boundary_asr.transcribe_boundary_words(
        _prepared(),
        window_start_sec=10.0,
        window_end_sec=12.0,
        timeout_sec=5.0,
    )

    assert [(word.text, word.onset_sec, word.end_sec) for word in words] == [
        ("cell", 10.25, 10.65),
        ("division", 10.65, 11.4),
    ]
    assert decode["window_start_sec"] == 10.0
    assert decode["window_end_sec"] == 12.0
    assert decode["max_duration_sec"] == groq_boundary_asr.MAX_BOUNDARY_WINDOW_SEC
    assert decode["timeout_sec"] == 5.0
    assert 0 < float(client_args["timeout_sec"]) <= 5.0
    assert client.closed is True

    request = client.audio.transcriptions.request
    assert request is not None
    assert request["model"] == "whisper-large-v3-turbo"
    assert request["response_format"] == "verbose_json"
    assert request["timestamp_granularities"] == ["word"]
    assert request["temperature"] == 0
    assert request["language"] == "en"
    assert 0 < float(request["timeout"]) <= 5.0
    assert set(request) == {
        "file",
        "model",
        "response_format",
        "temperature",
        "timestamp_granularities",
        "timeout",
        "language",
    }
    filename, payload, content_type = request["file"]
    assert filename == "boundary.wav"
    assert content_type == "audio/wav"
    assert isinstance(payload, bytes)
    assert payload.startswith(b"RIFF")
    assert b"media.example" not in payload


def test_trailing_word_straddling_wav_endpoint_keeps_valid_prefix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    wav_path = tmp_path / "short.wav"
    _write_wav(wav_path)
    _install_decode(monkeypatch, wav_path)
    client = _Client(
        {
            "words": [
                {"word": "cell", "start": 0.25, "end": 0.65},
                {"word": "division", "start": 0.65, "end": 1.4},
                {"word": "continues", "start": 1.75, "end": 2.12},
            ]
        }
    )
    monkeypatch.setenv("GROQ_API_KEY", "unit-test-key")
    monkeypatch.setattr(groq_boundary_asr, "_create_client", lambda **_kwargs: client)

    words = groq_boundary_asr.transcribe_boundary_words(
        _prepared(),
        window_start_sec=10.0,
        window_end_sec=12.0,
        timeout_sec=5.0,
    )

    assert [(word.text, word.onset_sec) for word in words] == [
        ("cell", 10.25),
        ("division", 10.65),
    ]


def test_client_disables_sdk_retries_and_sets_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    expected = object()
    module = types.ModuleType("groq")

    def groq(**kwargs: object) -> object:
        captured.update(kwargs)
        return expected

    module.Groq = groq
    monkeypatch.setitem(sys.modules, "groq", module)

    assert (
        groq_boundary_asr._create_client(api_key="unit-test-key", timeout_sec=3.5)
        is expected
    )
    assert captured == {
        "api_key": "unit-test-key",
        "timeout": 3.5,
        "max_retries": 0,
    }


def test_groq_edge_aligner_uses_full_generic_result_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = EdgeAnchor("start", 3.0, 3.0, 4.0, 2.5)
    calls: list[str] = []

    def align(_words: object, **kwargs: object) -> EdgeAnchor:
        calls.append(str(kwargs["quote"]))
        return expected

    monkeypatch.setattr(lexical_timing, "align_edge_anchor", align)

    assert groq_boundary_asr.align_groq_edge_anchor(
        (),
        cue_text="before exact quote after",
        quote="exact quote",
        edge="start",
        cue_start_sec=0.0,
        cue_end_sec=10.0,
    ) is expected
    assert calls == ["exact quote"]


def test_groq_edge_aligner_uses_longest_exact_prefix_for_real_math_expansion() -> None:
    words = _words(
        [
            "polomial",
            "function",
            "so",
            "let's",
            "say",
            "we",
            "have",
            "f",
            "of",
            "x",
            "equals",
            "x",
            "squared",
            "minus",
            "five",
            "x",
            "plus",
            "nine",
            "what's",
        ],
        start=1216.0,
        step=0.16,
    )
    # The generic full-quote path cannot bridge the spoken math expansion.
    assert lexical_timing.align_edge_anchor(
        words,
        cue_text=_POLYNOMIAL_CUE,
        quote=_POLYNOMIAL_QUOTE,
        edge="start",
        cue_start_sec=1193.6,
        cue_end_sec=1245.52,
    ) is None

    anchor = groq_boundary_asr.align_groq_edge_anchor(
        words,
        cue_text=_POLYNOMIAL_CUE,
        quote=_POLYNOMIAL_QUOTE,
        edge="start",
        cue_start_sec=1193.6,
        cue_end_sec=1245.52,
    )

    assert anchor == EdgeAnchor(
        edge="start",
        anchor_sec=1216.32,
        quote_start_sec=1216.32,
        quote_last_onset_sec=1217.12,
        excluded_neighbor_onset_sec=1216.16,
    )


def test_groq_edge_aligner_supports_exact_end_suffix() -> None:
    quote = "alpha beta gamma delta gives the final answer"
    words = _words(
        [
            "prior",
            "alpha",
            "spoken",
            "math",
            "expansion",
            "gives",
            "the",
            "final",
            "answer",
            "next",
        ],
        start=29.2,
        step=0.2,
    )
    cue = f"before {quote} next concept"
    assert lexical_timing.align_edge_anchor(
        words,
        cue_text=cue,
        quote=quote,
        edge="end",
        cue_start_sec=29.0,
        cue_end_sec=32.0,
    ) is None

    anchor = groq_boundary_asr.align_groq_edge_anchor(
        words,
        cue_text=cue,
        quote=quote,
        edge="end",
        cue_start_sec=29.0,
        cue_end_sec=32.0,
    )

    assert anchor == EdgeAnchor(
        edge="end",
        anchor_sec=31.0,
        quote_start_sec=30.2,
        quote_last_onset_sec=30.8,
        excluded_neighbor_onset_sec=31.0,
    )


def test_groq_edge_fragment_requires_at_least_four_exact_tokens() -> None:
    quote = "alpha beta gamma delta epsilon"
    words = _words(
        ["prior", "alpha", "beta", "gamma", "changed", "next"],
        start=1.0,
    )

    assert groq_boundary_asr.align_groq_edge_anchor(
        words,
        cue_text=f"before {quote} after",
        quote=quote,
        edge="start",
        cue_start_sec=0.0,
        cue_end_sec=5.0,
    ) is None


def test_groq_edge_fragment_must_be_unique_in_timed_words_even_with_occurrence() -> None:
    quote = "alpha beta gamma delta epsilon zeta"
    words = _words(
        [
            "prior",
            "alpha",
            "beta",
            "gamma",
            "delta",
            "epsilon",
            "changed",
            "alpha",
            "beta",
            "gamma",
            "delta",
            "epsilon",
            "other",
        ],
        start=1.0,
    )

    assert groq_boundary_asr.align_groq_edge_anchor(
        words,
        cue_text=f"before {quote} after",
        quote=quote,
        edge="start",
        cue_start_sec=0.0,
        cue_end_sec=5.0,
        occurrence="first",
    ) is None


def test_groq_edge_fragment_must_be_unique_in_caption_cue() -> None:
    quote = "alpha beta gamma delta epsilon zeta"
    words = _words(
        ["prior", "alpha", "beta", "gamma", "delta", "epsilon", "changed"],
        start=1.0,
    )
    cue = (
        "before alpha beta gamma delta epsilon one then "
        "alpha beta gamma delta epsilon two after"
    )

    assert groq_boundary_asr.align_groq_edge_anchor(
        words,
        cue_text=cue,
        quote=quote,
        edge="start",
        cue_start_sec=0.0,
        cue_end_sec=5.0,
    ) is None


def test_groq_edge_fragment_requires_real_excluded_timed_neighbor() -> None:
    quote = "alpha beta gamma delta epsilon zeta"
    words = _words(
        ["alpha", "beta", "gamma", "delta", "epsilon", "changed"],
        start=1.0,
    )

    assert groq_boundary_asr.align_groq_edge_anchor(
        words,
        cue_text=f"before {quote} after",
        quote=quote,
        edge="start",
        cue_start_sec=0.0,
        cue_end_sec=5.0,
    ) is None


@pytest.mark.parametrize(
    "words",
    [
        [{"word": "bad", "start": float("nan"), "end": 0.2}],
        [{"word": "bad", "start": -0.1, "end": 0.2}],
        [{"word": "bad", "start": 0.4, "end": 0.3}],
        [{"word": "bad", "start": 1.8, "end": 2.1}],
        [
            {"word": "first", "start": 0.2, "end": 0.8},
            {"word": "overlap", "start": 0.4, "end": 1.0},
        ],
        [
            {"word": "first", "start": 0.2, "end": 0.8},
            {"word": "interior", "start": 1.8, "end": 2.1},
            {"word": "tail", "start": 2.1, "end": 2.2},
        ],
        [
            {"word": "first", "start": 0.2, "end": 1.85},
            {"word": "overlap", "start": 1.5, "end": 2.1},
        ],
        [{"word": " ", "start": 0.1, "end": 0.2}],
    ],
)
def test_rejects_entire_response_for_invalid_or_unordered_word_timestamps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    words: list[dict[str, object]],
) -> None:
    wav_path = tmp_path / "short.wav"
    _write_wav(wav_path)
    _install_decode(monkeypatch, wav_path)
    client = _Client({"words": words})
    monkeypatch.setenv("GROQ_API_KEY", "unit-test-key")
    monkeypatch.setattr(groq_boundary_asr, "_create_client", lambda **_kwargs: client)

    assert groq_boundary_asr.transcribe_boundary_words(
        _prepared(),
        window_start_sec=10.0,
        window_end_sec=12.0,
        timeout_sec=5.0,
    ) == ()


def test_clamps_small_provider_timestamp_overlap_without_discarding_words(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    wav_path = tmp_path / "short.wav"
    _write_wav(wav_path)
    _install_decode(monkeypatch, wav_path)
    client = _Client({
        "words": [
            {"word": "the", "start": 0.24, "end": 0.42},
            {"word": "larger", "start": 0.24, "end": 0.66},
            {"word": "person", "start": 0.66, "end": 1.1},
        ]
    })
    monkeypatch.setenv("GROQ_API_KEY", "unit-test-key")
    monkeypatch.setattr(groq_boundary_asr, "_create_client", lambda **_kwargs: client)

    words = groq_boundary_asr.transcribe_boundary_words(
        _prepared(),
        window_start_sec=10.0,
        window_end_sec=12.0,
        timeout_sec=5.0,
    )

    assert [(word.text, word.onset_sec, word.end_sec) for word in words] == [
        ("the", 10.24, 10.42),
        ("larger", 10.42, 10.66),
        ("person", 10.66, 11.1),
    ]


class _ProviderFailure(RuntimeError):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"provider failed with private-unit-test-key ({status_code})")


@pytest.mark.parametrize(
    "failure",
    [_ProviderFailure(401), _ProviderFailure(429), _ProviderFailure(503), TimeoutError()],
)
def test_provider_failures_return_no_timing_without_exposing_secrets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    failure: Exception,
) -> None:
    wav_path = tmp_path / "short.wav"
    _write_wav(wav_path)
    _install_decode(monkeypatch, wav_path)
    client = _Client(error=failure)
    monkeypatch.setenv("GROQ_API_KEY", "private-unit-test-key")
    monkeypatch.setattr(groq_boundary_asr, "_create_client", lambda **_kwargs: client)

    assert groq_boundary_asr.transcribe_boundary_words(
        _prepared(),
        window_start_sec=10.0,
        window_end_sec=12.0,
        timeout_sec=5.0,
    ) == ()
    captured = capsys.readouterr()
    assert "private-unit-test-key" not in caplog.text
    assert "private-unit-test-key" not in captured.out
    assert "private-unit-test-key" not in captured.err
    assert client.closed is True


def test_no_key_or_empty_response_fails_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = _prepared()
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    def unexpected_decode(*_args: object, **_kwargs: object):
        raise AssertionError("audio must not be decoded without a key")

    monkeypatch.setattr(silence, "decode_audio_window", unexpected_decode)
    assert groq_boundary_asr.transcribe_boundary_words(
        prepared,
        window_start_sec=10.0,
        window_end_sec=12.0,
        timeout_sec=5.0,
    ) == ()

    wav_path = tmp_path / "short.wav"
    _write_wav(wav_path)
    _install_decode(monkeypatch, wav_path)
    monkeypatch.setenv("GROQ_API_KEY", "unit-test-key")
    monkeypatch.setattr(
        groq_boundary_asr,
        "_create_client",
        lambda **_kwargs: _Client({"words": []}),
    )
    assert groq_boundary_asr.transcribe_boundary_words(
        prepared,
        window_start_sec=10.0,
        window_end_sec=12.0,
        timeout_sec=5.0,
    ) == ()


def test_invalid_or_oversized_window_and_invalid_wav_fail_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "unit-test-key")
    calls = 0

    def create_client(**_kwargs: object) -> _Client:
        nonlocal calls
        calls += 1
        return _Client({"words": []})

    monkeypatch.setattr(groq_boundary_asr, "_create_client", create_client)
    assert groq_boundary_asr.transcribe_boundary_words(
        _prepared(),
        window_start_sec=10.0,
        window_end_sec=31.0,
        timeout_sec=5.0,
    ) == ()

    invalid_wav = tmp_path / "not-audio.wav"
    invalid_wav.write_bytes(b"RIFF video content disguised as audio")
    _install_decode(monkeypatch, invalid_wav)
    assert groq_boundary_asr.transcribe_boundary_words(
        _prepared(),
        window_start_sec=10.0,
        window_end_sec=12.0,
        timeout_sec=5.0,
    ) == ()
    assert calls == 0


def test_wav_cannot_extend_past_requested_window(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    wav_path = tmp_path / "too-long.wav"
    _write_wav(wav_path, duration_sec=2.0)
    _install_decode(monkeypatch, wav_path)
    calls = 0

    def create_client(**_kwargs: object) -> _Client:
        nonlocal calls
        calls += 1
        return _Client({"words": []})

    monkeypatch.setenv("GROQ_API_KEY", "unit-test-key")
    monkeypatch.setattr(groq_boundary_asr, "_create_client", create_client)

    assert groq_boundary_asr.transcribe_boundary_words(
        _prepared(),
        window_start_sec=10.0,
        window_end_sec=11.0,
        timeout_sec=5.0,
    ) == ()
    assert calls == 0


def test_decode_failure_and_cancellation_return_no_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "unit-test-key")

    @contextmanager
    def failed_decode(*_args: object, **_kwargs: object):
        raise RuntimeError("signed-audio-url-must-not-escape")
        yield Path("unreachable.wav")

    monkeypatch.setattr(silence, "decode_audio_window", failed_decode)
    assert groq_boundary_asr.transcribe_boundary_words(
        _prepared(),
        window_start_sec=10.0,
        window_end_sec=12.0,
        timeout_sec=5.0,
    ) == ()
    assert groq_boundary_asr.transcribe_boundary_words(
        _prepared(),
        window_start_sec=10.0,
        window_end_sec=12.0,
        timeout_sec=5.0,
        cancel_check=lambda: True,
    ) == ()


def test_public_decode_helper_passes_explicit_cap_and_removes_temp_wav(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared()
    captured: dict[str, object] = {}
    yielded_path: Path | None = None

    def decode(source: object, **kwargs: object) -> None:
        captured["source"] = source
        captured.update(kwargs)
        _write_wav(kwargs["output_path"], duration_sec=1.0)

    monkeypatch.setattr(silence, "_decode_window", decode)
    with silence.decode_audio_window(
        prepared.source,
        window_start_sec=4.0,
        window_end_sec=5.0,
        max_duration_sec=2.0,
        timeout_sec=3.0,
    ) as path:
        yielded_path = path
        assert path.is_file()

    assert yielded_path is not None
    assert not yielded_path.exists()
    assert captured["window_start_sec"] == 4.0
    assert captured["window_duration_sec"] == 1.0
    assert captured["max_duration_sec"] == 2.0
    assert float(captured["deadline"]) > 0
