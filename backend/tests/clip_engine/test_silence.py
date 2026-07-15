from __future__ import annotations

import math
import shutil
import threading
import time
import wave
from array import array
from pathlib import Path
from unittest import mock

import pytest

from backend.app.clip_engine import silence


def test_hosted_image_includes_yt_dlp_javascript_runtime() -> None:
    project_root = Path(__file__).resolve().parents[3]
    dockerfile = (project_root / "Dockerfile").read_text()
    requirements = (project_root / "backend" / "requirements.txt").read_text()

    assert "FROM denoland/deno:bin-2.9.2 AS deno_runtime" in dockerfile
    assert "COPY --from=deno_runtime /deno /usr/local/bin/deno" in dockerfile
    assert "yt-dlp[default,curl-cffi]==2026.7.4" in requirements
    assert "curl_cffi==0.15.0" in requirements
    assert "import yt_dlp.networking._curlcffi" in dockerfile
    assert "OMP_NUM_THREADS=1" in dockerfile
    assert "OPENBLAS_NUM_THREADS=1" in dockerfile
    assert "MKL_NUM_THREADS=1" in dockerfile
    assert "NUMEXPR_NUM_THREADS=1" in dockerfile
    assert "TOKENIZERS_PARALLELISM=false" in dockerfile


def _write_wav(path: Path, spans: list[tuple[float, float]], *, sample_rate: int = 16000) -> None:
    samples = array("h")
    for duration, amplitude in spans:
        count = round(duration * sample_rate)
        for index in range(count):
            value = 0 if amplitude == 0 else round(amplitude * math.sin(2 * math.pi * 440 * index / sample_rate))
            samples.append(value)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())


def _prepared() -> silence.AudioPreparationResult:
    return silence.AudioPreparationResult(
        "ready",
        source=silence.PreparedAudioSource(
            url="https://media.example/audio.m4a", format_id="140"
        ),
        diagnostics={"elapsed_ms": 12},
    )


def _context_aligned_search_context() -> dict:
    return {
        "selection_contract_version": "quality_silence_v15",
        "boundary_status": "context_aligned",
        "speech_corridor_verified": True,
        "selection_caption_cues": [
            {"cue_id": "selected", "start": 2.0, "end": 9.0, "text": "Teaching"}
        ],
        "boundary_diagnostics": {
            "method": "transcript_context",
            "context_aligned": True,
            "acoustic_verified": False,
            "transcript": {
                "context_aligned": True,
                "stage": "analyze",
                "reason": "start_silence_not_found",
                "required_speech_range": [2.0, 8.0],
                "semantic_range": [1.0, 9.0],
                "final_range": [2.0, 9.0],
            },
        },
    }


def test_context_aligned_boundary_is_usable_but_not_acoustically_verified() -> None:
    context = _context_aligned_search_context()

    assert silence.persisted_boundary_is_verified(context) is False
    assert silence.persisted_boundary_is_usable(context) is True
    assert silence.persisted_boundary_is_usable(
        context, t_start=2.0, t_end=9.0
    ) is True
    assert silence.persisted_boundary_is_usable(
        context, t_start=2.1, t_end=9.0
    ) is False


def test_current_strict_boundary_is_bound_to_persisted_range() -> None:
    context = {
        "selection_contract_version": "quality_silence_v15",
        "boundary_status": "verified",
        "boundary_diagnostics": {
            "acoustic_verified": True,
            "final_range": [2.0, 9.0],
            "acoustic": {
                "threshold_dbfs": -38.0,
                "start_quiet": [1.9, 2.1],
                "end_quiet": [8.9, 9.1],
            },
        },
    }

    assert silence.persisted_boundary_is_usable(
        context, t_start=2.0, t_end=9.0
    ) is True
    for start, end in (
        (float("nan"), 9.0),
        (9.0, 2.0),
        (-1.0, 9.0),
        (0.0, 900.0),
    ):
        assert silence.persisted_boundary_is_usable(
            context, t_start=start, t_end=end
        ) is False

    context["boundary_diagnostics"]["acoustic"].update(
        start_quiet=[20.0, 21.0], end_quiet=[30.0, 31.0]
    )
    assert silence.persisted_boundary_is_usable(
        context, t_start=2.0, t_end=9.0
    ) is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda context: context.update(boundary_status="verified"),
        lambda context: context.update(speech_corridor_verified=False),
        lambda context: context.update(selection_caption_cues=[]),
        lambda context: context["boundary_diagnostics"].update(context_aligned=False),
        lambda context: context["boundary_diagnostics"]["transcript"].update(
            final_range=[2.0, 7.0]
        ),
        lambda context: context["boundary_diagnostics"]["transcript"].update(
            final_range=[0.5, 9.0]
        ),
        lambda context: context["boundary_diagnostics"]["transcript"].update(
            required_speech_range=[-1.0, 8.0],
            semantic_range=[-2.0, 9.0],
            final_range=[-1.0, 9.0],
        ),
        lambda context: context.update(selection_caption_cues=[{}]),
        lambda context: context.update(selection_caption_cues=[{
            "start": 2.0, "end": 9.0, "text": "",
        }]),
        lambda context: context.update(selection_caption_cues=[{
            "start": float("nan"), "end": 9.0, "text": "Teaching",
        }]),
        lambda context: context.update(selection_caption_cues=[{
            "start": 3.0, "end": 8.0, "text": "Teaching",
        }]),
        lambda context: context.update(selection_caption_cues=[
            {"start": 8.0, "end": 9.0, "text": "Ending"},
            {"start": 2.0, "end": 8.0, "text": "Beginning"},
        ]),
        lambda context: context.update(selection_caption_cues=[
            {"start": 2.0, "end": 3.0, "text": "Before"},
            {"start": 8.0, "end": 9.0, "text": "After"},
        ]),
    ],
)
def test_malformed_context_aligned_boundary_is_not_usable(mutation) -> None:
    context = _context_aligned_search_context()
    mutation(context)

    assert silence.persisted_boundary_is_usable(context) is False


def test_verified_edges_preserve_required_quiet_cushions(tmp_path: Path) -> None:
    start_wav = tmp_path / "source-start.wav"
    end_wav = tmp_path / "source-end.wav"
    _write_wav(start_wav, [(2.7, 12000), (0.40, 0), (2.90, 12000)])
    _write_wav(end_wav, [(2.7, 12000), (0.40, 0), (2.90, 12000)])

    def fake_decode(_source, *, window_start_sec, output_path, **_kwargs):
        source = start_wav if window_start_sec < 8 else end_wav
        shutil.copyfile(source, output_path)

    with mock.patch.object(silence, "_decode_window", side_effect=fake_decode):
        result = silence.verify_acoustic_boundaries(
            "dQw4w9WgXcQ", 10.0, 20.3, prepared=_prepared()
        )

    assert result.verified
    assert result.start_sec == 10.0
    assert result.end_sec == 20.3
    assert result.diagnostics["start_quiet"] == [9.7, 10.1]
    assert result.diagnostics["end_quiet"] == [20.0, 20.4]
    assert result.diagnostics["threshold_dbfs"] == -38.0
    assert result.diagnostics["start_cushion_ms"] == 100
    assert result.diagnostics["end_cushion_ms"] == 100


def test_caption_handoff_observation_accepts_only_the_straddling_quiet_run(
    tmp_path: Path,
) -> None:
    def fake_decode(
        _source,
        *,
        window_start_sec,
        window_duration_sec,
        output_path,
        **_kwargs,
    ):
        if output_path.name == "start-0.wav":
            quiet_start = 9.70 - window_start_sec
            spans = [
                (quiet_start, 12000),
                (0.33, 0),
                (window_duration_sec - quiet_start - 0.33, 12000),
            ]
        else:
            quiet_start = 19.72 - window_start_sec
            spans = [
                (quiet_start, 12000),
                (0.36, 0),
                (window_duration_sec - quiet_start - 0.36, 12000),
            ]
        _write_wav(output_path, spans)

    with mock.patch.object(silence, "_decode_window", side_effect=fake_decode):
        result = silence.verify_acoustic_boundaries(
            "dQw4w9WgXcQ",
            10.0,
            20.0,
            search_start_limit_sec=10.0,
            search_end_limit_sec=20.0,
            require_speech_handoff=True,
            prepared=_prepared(),
        )

    assert result.verified
    assert result.start_sec == 9.93
    assert result.end_sec == 20.0
    assert result.diagnostics["speech_handoff_verified"] is True
    assert result.diagnostics["semantic_start_limit_sec"] == 10.0
    assert result.diagnostics["semantic_end_limit_sec"] == 20.0
    assert result.diagnostics["observation_start_limit_sec"] == 9.0
    assert result.diagnostics["observation_end_limit_sec"] == 21.0


def test_caption_handoff_never_moves_cuts_inside_required_speech(
    tmp_path: Path,
) -> None:
    def fake_decode(
        _source,
        *,
        window_start_sec,
        window_duration_sec,
        output_path,
        **_kwargs,
    ):
        quiet_start, quiet_end = (
            (9.90, 10.25)
            if output_path.name == "start-0.wav"
            else (19.80, 20.30)
        )
        before = quiet_start - window_start_sec
        after = window_duration_sec - before - (quiet_end - quiet_start)
        _write_wav(
            output_path,
            [(before, 12000), (quiet_end - quiet_start, 0), (after, 12000)],
        )

    with mock.patch.object(silence, "_decode_window", side_effect=fake_decode):
        result = silence.verify_acoustic_boundaries(
            "dQw4w9WgXcQ",
            10.0,
            20.0,
            search_start_limit_sec=10.0,
            search_end_limit_sec=21.0,
            require_start_speech_handoff=True,
            require_start_two_sided=True,
            prepared=_prepared(),
        )

    assert result.verified
    assert result.start_sec == 10.0
    assert result.end_sec == 20.0
    assert result.diagnostics["start_quiet"] == [9.9, 10.25]
    assert result.diagnostics["end_quiet"] == [19.8, 20.3]
    assert result.diagnostics["start_shift_sec"] == 0.0
    assert result.diagnostics["end_shift_sec"] == 0.0


def test_caption_handoff_rejects_nearby_silence_separated_by_sound(
    tmp_path: Path,
) -> None:
    def fake_decode(
        _source,
        *,
        window_start_sec,
        window_duration_sec,
        output_path,
        **_kwargs,
    ):
        if output_path.name == "start-0.wav":
            quiet_start = 9.70 - window_start_sec
            spans = [
                (quiet_start, 12000),
                (0.33, 0),
                (window_duration_sec - quiet_start - 0.33, 12000),
            ]
        else:
            quiet_start = 20.34 - window_start_sec
            spans = [
                (quiet_start, 12000),
                (0.30, 0),
                (window_duration_sec - quiet_start - 0.30, 12000),
            ]
        _write_wav(output_path, spans)

    with mock.patch.object(silence, "_decode_window", side_effect=fake_decode):
        result = silence.verify_acoustic_boundaries(
            "dQw4w9WgXcQ",
            10.0,
            20.0,
            search_start_limit_sec=10.0,
            search_end_limit_sec=20.0,
            require_speech_handoff=True,
            prepared=_prepared(),
        )

    assert result.status == "unavailable"
    assert result.diagnostics["reason"] == "end_silence_not_found"
    assert result.diagnostics["end_windows"] == [[17.0, 21.0]]


@pytest.mark.parametrize(
    ("end_quiet", "expected_verified"),
    [
        ((7.88, 8.18), True),
        ((8.18, 8.48), False),
        (None, False),
    ],
)
def test_rolling_caption_end_handoff_requires_the_same_straddling_quiet_run(
    tmp_path: Path,
    end_quiet: tuple[float, float] | None,
    expected_verified: bool,
) -> None:
    def fake_decode(
        _source,
        *,
        window_start_sec,
        window_duration_sec,
        output_path,
        **_kwargs,
    ):
        if output_path.name == "start-0.wav":
            quiet_start, quiet_end = 0.0, 0.30
        elif end_quiet is None:
            _write_wav(output_path, [(window_duration_sec, 12000)])
            return
        else:
            quiet_start, quiet_end = end_quiet
        before = quiet_start - window_start_sec
        after = window_duration_sec - before - (quiet_end - quiet_start)
        _write_wav(
            output_path,
            [(before, 12000), (quiet_end - quiet_start, 0), (after, 12000)],
        )

    with mock.patch.object(silence, "_decode_window", side_effect=fake_decode):
        result = silence.verify_acoustic_boundaries(
            "dQw4w9WgXcQ",
            0.01,
            8.0,
            search_start_limit_sec=0.0,
            search_end_limit_sec=8.0,
            require_end_speech_handoff=True,
            require_end_two_sided=True,
            prepared=_prepared(),
        )

    assert result.verified is expected_verified
    if expected_verified:
        assert result.end_sec == 8.0
        assert result.diagnostics["end_quiet"] == [7.88, 8.18]
        assert result.diagnostics["observation_end_limit_sec"] == 9.0
    else:
        assert result.diagnostics["reason"] == "end_silence_not_found"
        assert result.diagnostics["end_windows"] == [[5.0, 9.0]]


def test_lexical_ownership_corridor_accepts_real_one_hundred_ten_ms_gaps(
    tmp_path: Path,
) -> None:
    quiet_by_edge = {
        "start": (504.04, 504.15),
        "end": (530.23, 530.34),
    }

    def fake_decode(
        _source,
        *,
        window_start_sec,
        window_duration_sec,
        output_path,
        **_kwargs,
    ):
        edge = output_path.name.split("-", 1)[0]
        quiet_start, quiet_end = quiet_by_edge[edge]
        before = quiet_start - window_start_sec
        after = window_duration_sec - before - (quiet_end - quiet_start)
        _write_wav(
            output_path,
            [(before, 12000), (quiet_end - quiet_start, 0), (after, 12000)],
        )

    with mock.patch.object(silence, "_decode_window", side_effect=fake_decode):
        result = silence.verify_acoustic_boundaries(
            "sQK3Yr4Sc_k",
            504.08,
            530.16,
            search_start_limit_sec=503.68,
            search_end_limit_sec=530.399,
            require_start_two_sided=True,
            require_end_two_sided=True,
            prepared=_prepared(),
        )

    assert result.verified
    assert result.start_sec == pytest.approx(504.05)
    assert result.end_sec == pytest.approx(530.33)
    assert result.diagnostics["start_quiet"] == [504.04, 504.15]
    assert result.diagnostics["end_quiet"] == [530.23, 530.34]
    assert result.diagnostics["start_speech_handoff_verified"] is False
    assert result.diagnostics["end_speech_handoff_verified"] is False
    assert result.diagnostics["start_two_sided_required"] is True
    assert result.diagnostics["end_two_sided_required"] is True
    assert result.diagnostics["semantic_start_limit_sec"] == 503.68
    assert result.diagnostics["semantic_end_limit_sec"] == 530.399


def test_lexical_two_sided_halo_proves_safe_quiet_without_crossing_fences(
    tmp_path: Path,
) -> None:
    quiet_by_edge = {
        "start": (10.05, 10.25),
        "end": (44.92, 45.30),
    }

    def fake_decode(
        _source,
        *,
        window_start_sec,
        window_duration_sec,
        output_path,
        **_kwargs,
    ):
        edge = output_path.name.split("-", 1)[0]
        quiet_start, quiet_end = quiet_by_edge[edge]
        before = quiet_start - window_start_sec
        after = window_duration_sec - before - (quiet_end - quiet_start)
        _write_wav(
            output_path,
            [(before, 12000), (quiet_end - quiet_start, 0), (after, 12000)],
        )

    with mock.patch.object(silence, "_decode_window", side_effect=fake_decode):
        result = silence.verify_acoustic_boundaries(
            "3tisOnOkwzo",
            10.20,
            44.52,
            search_start_limit_sec=10.0,
            search_end_limit_sec=45.20,
            require_start_two_sided=True,
            require_end_two_sided=True,
            prepared=_prepared(),
        )

    assert result.verified
    assert result.start_sec == 10.15
    assert result.end_sec == 45.02
    assert result.start_sec >= 10.0
    assert result.end_sec <= 45.20
    assert result.diagnostics["start_quiet"] == [10.05, 10.25]
    assert result.diagnostics["end_quiet"] == [44.92, 45.3]
    assert result.diagnostics["observation_start_limit_sec"] == 9.0
    assert result.diagnostics["observation_end_limit_sec"] == 46.2


def test_lexical_start_halo_rejects_quiet_crossing_prior_word_fence(
    tmp_path: Path,
) -> None:
    quiet_by_edge = {
        "start": (9.92, 10.08),
        "end": (44.92, 45.30),
    }

    def fake_decode(
        _source,
        *,
        window_start_sec,
        window_duration_sec,
        output_path,
        **_kwargs,
    ):
        edge = output_path.name.split("-", 1)[0]
        quiet_start, quiet_end = quiet_by_edge[edge]
        before = quiet_start - window_start_sec
        after = window_duration_sec - before - (quiet_end - quiet_start)
        _write_wav(
            output_path,
            [(before, 12000), (quiet_end - quiet_start, 0), (after, 12000)],
        )

    with mock.patch.object(silence, "_decode_window", side_effect=fake_decode):
        result = silence.verify_acoustic_boundaries(
            "3tisOnOkwzo",
            10.20,
            44.52,
            search_start_limit_sec=10.0,
            search_end_limit_sec=45.20,
            require_start_two_sided=True,
            require_end_two_sided=True,
            prepared=_prepared(),
        )

    assert result.status == "unavailable"
    assert result.diagnostics["reason"] == "start_silence_not_found"
    assert (result.start_sec, result.end_sec) == (10.20, 44.52)


def test_lexical_two_sided_halo_rejects_quiet_wholly_beyond_end_fence(
    tmp_path: Path,
) -> None:
    quiet_by_edge = {
        "start": (10.05, 10.40),
        "end": (45.26, 45.56),
    }

    def fake_decode(
        _source,
        *,
        window_start_sec,
        window_duration_sec,
        output_path,
        **_kwargs,
    ):
        edge = output_path.name.split("-", 1)[0]
        quiet_start, quiet_end = quiet_by_edge[edge]
        before = quiet_start - window_start_sec
        after = window_duration_sec - before - (quiet_end - quiet_start)
        _write_wav(
            output_path,
            [(before, 12000), (quiet_end - quiet_start, 0), (after, 12000)],
        )

    with mock.patch.object(silence, "_decode_window", side_effect=fake_decode):
        result = silence.verify_acoustic_boundaries(
            "3tisOnOkwzo",
            10.20,
            44.52,
            search_start_limit_sec=10.0,
            search_end_limit_sec=45.20,
            require_start_two_sided=True,
            require_end_two_sided=True,
            prepared=_prepared(),
        )

    assert result.status == "unavailable"
    assert result.diagnostics["reason"] == "end_silence_not_found"
    assert (result.start_sec, result.end_sec) == (10.20, 44.52)


@pytest.mark.parametrize(
    ("end_quiet", "expected_end", "expected_reason"),
    [
        ((10.20, 10.35), 10.30, None),
        ((10.60, 10.90), 10.00, "end_silence_not_found"),
        ((10.45, 10.70), 10.00, "adjusted_range_invalid"),
    ],
)
def test_progressive_end_search_stays_inside_next_cue_fence_and_keeps_cushion(
    tmp_path: Path,
    end_quiet: tuple[float, float],
    expected_end: float,
    expected_reason: str | None,
) -> None:
    def fake_decode(
        _source,
        *,
        window_start_sec,
        window_duration_sec,
        output_path,
        **_kwargs,
    ):
        quiet_start, quiet_end = (
            (1.80, 2.20)
            if output_path.name == "start-0.wav"
            else end_quiet
        )
        before = quiet_start - window_start_sec
        after = window_duration_sec - before - (quiet_end - quiet_start)
        _write_wav(
            output_path,
            [(before, 12000), (quiet_end - quiet_start, 0), (after, 12000)],
        )

    with mock.patch.object(silence, "_decode_window", side_effect=fake_decode):
        result = silence.verify_acoustic_boundaries(
            "3tisOnOkwzo",
            2.0,
            10.0,
            search_start_limit_sec=0.0,
            search_end_limit_sec=10.5,
            require_end_two_sided=True,
            prepared=_prepared(),
        )

    if expected_reason is None:
        assert result.verified
        assert result.end_sec == expected_end
        assert result.end_sec <= 10.5
        assert result.end_sec >= end_quiet[0] + 0.1
        assert result.diagnostics["end_windows"] == [[7.0, 11.5]]
    else:
        assert result.status == "unavailable"
        assert result.diagnostics["reason"] == expected_reason
        assert (result.start_sec, result.end_sec) == (2.0, 10.0)


def test_caption_handoff_never_skips_sound_before_a_late_start_quiet_run(
    tmp_path: Path,
) -> None:
    def fake_decode(
        _source,
        *,
        window_start_sec,
        window_duration_sec,
        output_path,
        **_kwargs,
    ):
        quiet_at = 10.03 if output_path.name == "start-0.wav" else 19.80
        quiet_start = quiet_at - window_start_sec
        spans = [
            (quiet_start, 12000),
            (0.30, 0),
            (window_duration_sec - quiet_start - 0.30, 12000),
        ]
        _write_wav(output_path, spans)

    with mock.patch.object(silence, "_decode_window", side_effect=fake_decode):
        result = silence.verify_acoustic_boundaries(
            "dQw4w9WgXcQ",
            10.0,
            20.0,
            search_start_limit_sec=10.0,
            search_end_limit_sec=20.0,
            require_speech_handoff=True,
            prepared=_prepared(),
        )

    assert result.status == "unavailable"
    assert result.diagnostics["reason"] == "start_silence_not_found"


@pytest.mark.parametrize(
    ("one_sided_edge", "expected_reason"),
    [
        ("start", "start_silence_not_found"),
        ("end", "end_silence_not_found"),
    ],
)
def test_partial_cue_handoffs_require_sound_on_both_sides_of_the_quiet_gap(
    tmp_path: Path,
    one_sided_edge: str,
    expected_reason: str,
) -> None:
    """A source/caption edge is not proof of an in-cue semantic handoff."""

    def fake_decode(
        _source,
        *,
        window_start_sec,
        window_duration_sec,
        output_path,
        **_kwargs,
    ):
        if output_path.name == "start-0.wav":
            quiet_start = 10.0 - window_start_sec
            spans = (
                [(quiet_start, 12000), (0.30, 0), (window_duration_sec - quiet_start - 0.30, 12000)]
                if one_sided_edge != "start"
                else [(quiet_start + 0.30, 0), (window_duration_sec - quiet_start - 0.30, 12000)]
            )
        else:
            quiet_start = 19.70 - window_start_sec
            spans = (
                [(quiet_start, 12000), (0.40, 0), (window_duration_sec - quiet_start - 0.40, 12000)]
                if one_sided_edge != "end"
                else [(quiet_start, 12000), (window_duration_sec - quiet_start, 0)]
            )
        _write_wav(output_path, spans)

    with mock.patch.object(silence, "_decode_window", side_effect=fake_decode):
        result = silence.verify_acoustic_boundaries(
            "dQw4w9WgXcQ",
            10.0,
            20.0,
            search_start_limit_sec=10.0,
            search_end_limit_sec=20.0,
            require_speech_handoff=True,
            require_start_two_sided=True,
            require_end_two_sided=True,
            prepared=_prepared(),
        )

    assert result.status == "unavailable"
    assert result.diagnostics["reason"] == expected_reason


def test_caption_handoff_does_not_scan_an_entire_inter_caption_gap(
    tmp_path: Path,
) -> None:
    decoded: list[str] = []

    def fake_decode(
        _source,
        *,
        window_start_sec,
        window_duration_sec,
        output_path,
        **_kwargs,
    ):
        decoded.append(output_path.name)
        quiet_at = 9.70 if output_path.name == "start-0.wav" else 19.80
        quiet_start = quiet_at - window_start_sec
        spans = [
            (quiet_start, 12000),
            (0.30, 0),
            (window_duration_sec - quiet_start - 0.30, 12000),
        ]
        _write_wav(output_path, spans)

    with mock.patch.object(silence, "_decode_window", side_effect=fake_decode):
        result = silence.verify_acoustic_boundaries(
            "dQw4w9WgXcQ",
            10.0,
            20.0,
            search_start_limit_sec=0.0,
            search_end_limit_sec=30.0,
            require_speech_handoff=True,
            prepared=_prepared(),
        )

    assert result.verified
    assert sorted(decoded) == ["end-0.wav", "start-0.wav"]
    assert result.diagnostics["start_windows"] == [[7.0, 13.0]]
    assert result.diagnostics["end_windows"] == [[17.0, 23.0]]


@pytest.mark.parametrize(
    ("quiet_end", "expected_start"),
    [
        (20.0, 19.9),
        (20.4, 20.0),
    ],
)
def test_start_handoff_stitches_one_continuous_quiet_run_beyond_first_window(
    tmp_path: Path,
    quiet_end: float,
    expected_start: float,
) -> None:
    def fake_decode(
        _source,
        *,
        window_start_sec,
        window_duration_sec,
        output_path,
        **_kwargs,
    ):
        spans = [(window_duration_sec, 12000)]
        if output_path.name == "start-0.wav":
            quiet_duration = quiet_end - window_start_sec
            spans = [
                (quiet_duration, 0),
                (window_duration_sec - quiet_duration, 12000),
            ]
        elif output_path.name == "start-1.wav":
            spans = [(1.0, 12000), (window_duration_sec - 1.0, 0)]
        elif output_path.name == "end-0.wav":
            quiet_start = 39.8 - window_start_sec
            spans = [
                (quiet_start, 12000),
                (0.4, 0),
                (window_duration_sec - quiet_start - 0.4, 12000),
            ]
        _write_wav(output_path, spans)

    with mock.patch.object(silence, "_decode_window", side_effect=fake_decode):
        result = silence.verify_acoustic_boundaries(
            "dQw4w9WgXcQ",
            20.0,
            40.0,
            search_start_limit_sec=12.0,
            search_end_limit_sec=45.0,
            require_start_speech_handoff=True,
            require_start_two_sided=True,
            prepared=_prepared(),
        )

    assert result.verified
    assert result.start_sec == expected_start
    assert result.start_sec <= 20.0
    assert result.diagnostics["start_quiet"] == [12.0, quiet_end]
    assert result.diagnostics["start_windows"] == [[17.0, 23.0], [11.0, 17.0]]


def test_start_handoff_rejects_earlier_quiet_separated_by_sound(
    tmp_path: Path,
) -> None:
    decoded: list[str] = []

    def fake_decode(
        _source,
        *,
        window_start_sec,
        window_duration_sec,
        output_path,
        **_kwargs,
    ):
        decoded.append(output_path.name)
        spans = [(window_duration_sec, 12000)]
        if output_path.name == "start-0.wav":
            quiet_duration = 20.0 - window_start_sec
            spans = [
                (quiet_duration, 0),
                (window_duration_sec - quiet_duration, 12000),
            ]
        elif output_path.name == "start-1.wav":
            spans = [(1.0, 12000), (4.0, 0), (1.0, 12000)]
        elif output_path.name == "end-0.wav":
            quiet_start = 39.8 - window_start_sec
            spans = [
                (quiet_start, 12000),
                (0.4, 0),
                (window_duration_sec - quiet_start - 0.4, 12000),
            ]
        _write_wav(output_path, spans)

    with mock.patch.object(silence, "_decode_window", side_effect=fake_decode):
        result = silence.verify_acoustic_boundaries(
            "dQw4w9WgXcQ",
            20.0,
            40.0,
            search_start_limit_sec=12.0,
            search_end_limit_sec=45.0,
            require_start_speech_handoff=True,
            require_start_two_sided=True,
            prepared=_prepared(),
        )

    assert result.status == "unavailable"
    assert result.diagnostics["reason"] == "start_silence_not_found"
    assert "start-0.wav" in decoded
    assert "start-1.wav" in decoded
    assert "start-2.wav" not in decoded


def test_mixed_edge_modes_keep_exact_start_handoff_and_progressive_end_search(
    tmp_path: Path,
) -> None:
    def fake_decode(
        _source,
        *,
        window_start_sec,
        window_duration_sec,
        output_path,
        **_kwargs,
    ):
        spans = [(window_duration_sec, 12000)]
        if output_path.name == "start-0.wav":
            quiet_start = 19.8 - window_start_sec
            spans = [
                (quiet_start, 12000),
                (0.3, 0),
                (window_duration_sec - quiet_start - 0.3, 12000),
            ]
        elif output_path.name == "end-1.wav":
            quiet_start = 46.0 - window_start_sec
            spans = [
                (quiet_start, 12000),
                (0.3, 0),
                (window_duration_sec - quiet_start - 0.3, 12000),
            ]
        _write_wav(output_path, spans)

    with mock.patch.object(silence, "_decode_window", side_effect=fake_decode):
        result = silence.verify_acoustic_boundaries(
            "dQw4w9WgXcQ",
            20.0,
            40.0,
            search_start_limit_sec=20.0,
            search_end_limit_sec=60.0,
            require_start_speech_handoff=True,
            require_start_two_sided=True,
            prepared=_prepared(),
        )

    assert result.verified
    assert result.start_sec == 20.0
    assert result.end_sec == 46.1
    assert result.diagnostics["start_windows"] == [[19.0, 23.0]]
    assert result.diagnostics["end_windows"] == [[37.0, 43.0], [43.0, 49.0]]
    assert result.diagnostics["start_speech_handoff_verified"] is True
    assert result.diagnostics["end_speech_handoff_verified"] is False
    assert result.diagnostics["start_two_sided_required"] is True


def test_progressive_search_finds_silence_beyond_the_old_edge_window(
    tmp_path: Path,
) -> None:
    decoded: list[tuple[str, float, float, int]] = []

    def fake_decode(
        source,
        *,
        window_start_sec,
        window_duration_sec,
        output_path,
        **_kwargs,
    ):
        decoded.append(
            (
                output_path.name,
                window_start_sec,
                window_duration_sec,
                id(source),
            )
        )
        spans = [(window_duration_sec, 12000)]
        if output_path.name == "start-1.wav":
            spans = [(0.5, 12000), (0.3, 0), (window_duration_sec - 0.8, 12000)]
        elif output_path.name == "end-1.wav":
            spans = [(3.0, 12000), (0.3, 0), (window_duration_sec - 3.3, 12000)]
        _write_wav(output_path, spans)

    prepared = _prepared()
    with mock.patch.object(silence, "_prepare_audio_source") as resolve, mock.patch.object(
        silence, "_decode_window", side_effect=fake_decode
    ):
        result = silence.verify_acoustic_boundaries(
            "dQw4w9WgXcQ",
            20.0,
            40.0,
            search_start_limit_sec=0.0,
            search_end_limit_sec=60.0,
            prepared=prepared,
        )

    assert result.verified
    assert result.start_sec == 11.7
    assert result.end_sec == 46.1
    assert result.start_sec <= 20.0
    assert result.end_sec >= 40.0
    assert result.diagnostics["start_quiet"] == [11.5, 11.8]
    assert result.diagnostics["end_quiet"] == [46.0, 46.3]
    assert result.diagnostics["start_windows"] == [[17.0, 23.0], [11.0, 17.0]]
    assert result.diagnostics["end_windows"] == [[37.0, 43.0], [43.0, 49.0]]
    assert result.diagnostics["start_windows"][0][0] == result.diagnostics[
        "start_windows"
    ][1][1]
    assert result.diagnostics["end_windows"][0][1] == result.diagnostics[
        "end_windows"
    ][1][0]
    assert len({source_id for *_rest, source_id in decoded}) == 1
    resolve.assert_not_called()


def test_non_overlapping_windows_preserve_silence_transitions_at_chunk_edges(
    tmp_path: Path,
) -> None:
    def fake_decode(
        _source,
        *,
        window_duration_sec,
        output_path,
        **_kwargs,
    ):
        spans = [(window_duration_sec, 12000)]
        if output_path.name == "start-1.wav":
            spans = [(window_duration_sec - 0.3, 12000), (0.3, 0)]
        elif output_path.name == "end-1.wav":
            spans = [(0.3, 0), (window_duration_sec - 0.3, 12000)]
        _write_wav(output_path, spans)

    with mock.patch.object(silence, "_decode_window", side_effect=fake_decode):
        result = silence.verify_acoustic_boundaries(
            "dQw4w9WgXcQ",
            20.0,
            40.0,
            search_start_limit_sec=0.0,
            search_end_limit_sec=60.0,
            prepared=_prepared(),
        )

    assert result.verified
    assert result.start_sec == 16.9
    assert result.end_sec == 43.1
    assert result.diagnostics["start_quiet"] == [16.7, 17.0]
    assert result.diagnostics["end_quiet"] == [43.0, 43.3]


def test_backward_search_stitches_short_quiet_fragments_across_window_seam(
    tmp_path: Path,
) -> None:
    def fake_decode(
        _source,
        *,
        window_duration_sec,
        output_path,
        **_kwargs,
    ):
        spans = [(window_duration_sec, 12000)]
        if output_path.name == "start-0.wav":
            spans = [(0.08, 0), (window_duration_sec - 0.08, 12000)]
        elif output_path.name == "start-1.wav":
            spans = [(window_duration_sec - 0.08, 12000), (0.08, 0)]
        elif output_path.name == "end-0.wav":
            spans = [(3.0, 12000), (0.3, 0), (window_duration_sec - 3.3, 12000)]
        _write_wav(output_path, spans)

    with mock.patch.object(silence, "_decode_window", side_effect=fake_decode):
        result = silence.verify_acoustic_boundaries(
            "dQw4w9WgXcQ",
            20.0,
            40.0,
            search_start_limit_sec=0.0,
            search_end_limit_sec=60.0,
            prepared=_prepared(),
        )

    assert result.verified
    assert result.start_sec == 16.98
    assert result.diagnostics["start_quiet"] == [16.92, 17.08]
    assert result.diagnostics["start_windows"] == [[17.0, 23.0], [11.0, 17.0]]


def test_forward_search_stitches_short_quiet_fragments_across_window_seam(
    tmp_path: Path,
) -> None:
    def fake_decode(
        _source,
        *,
        window_duration_sec,
        output_path,
        **_kwargs,
    ):
        spans = [(window_duration_sec, 12000)]
        if output_path.name == "start-0.wav":
            spans = [(2.7, 12000), (0.3, 0), (window_duration_sec - 3.0, 12000)]
        elif output_path.name == "end-0.wav":
            spans = [(window_duration_sec - 0.06, 12000), (0.06, 0)]
        elif output_path.name == "end-1.wav":
            spans = [(0.06, 0), (window_duration_sec - 0.06, 12000)]
        _write_wav(output_path, spans)

    with mock.patch.object(silence, "_decode_window", side_effect=fake_decode):
        result = silence.verify_acoustic_boundaries(
            "dQw4w9WgXcQ",
            20.0,
            40.0,
            search_start_limit_sec=0.0,
            search_end_limit_sec=60.0,
            prepared=_prepared(),
        )

    assert result.verified
    assert result.end_sec == 43.04
    assert result.diagnostics["end_quiet"] == [42.94, 43.06]
    assert result.diagnostics["end_windows"] == [[37.0, 43.0], [43.0, 49.0]]


def test_progressive_search_fails_closed_when_limits_contain_no_silence(
    tmp_path: Path,
) -> None:
    def fake_decode(
        _source,
        *,
        window_duration_sec,
        output_path,
        **_kwargs,
    ):
        _write_wav(output_path, [(window_duration_sec, 12000)])

    with mock.patch.object(silence, "_decode_window", side_effect=fake_decode):
        result = silence.verify_acoustic_boundaries(
            "dQw4w9WgXcQ",
            20.0,
            40.0,
            search_start_limit_sec=0.0,
            search_end_limit_sec=60.0,
            prepared=_prepared(),
        )

    assert result.status == "unavailable"
    assert (result.start_sec, result.end_sec) == (20.0, 40.0)
    assert result.diagnostics["reason"] == "start_silence_not_found"
    assert result.diagnostics["start_windows"] == [
        [17.0, 23.0],
        [11.0, 17.0],
        [5.0, 11.0],
        [0.0, 5.0],
    ]


def test_source_edges_are_accepted_only_when_the_audio_at_each_edge_is_quiet(
    tmp_path: Path,
) -> None:
    def fake_decode(
        _source,
        *,
        window_duration_sec,
        output_path,
        **_kwargs,
    ):
        if output_path.name == "start-0.wav":
            spans = [(0.3, 0), (window_duration_sec - 0.3, 12000)]
        else:
            spans = [(window_duration_sec - 0.3, 12000), (0.3, 0)]
        _write_wav(output_path, spans)

    with mock.patch.object(silence, "_decode_window", side_effect=fake_decode):
        result = silence.verify_acoustic_boundaries(
            "dQw4w9WgXcQ",
            2.0,
            8.0,
            search_start_limit_sec=0.0,
            search_end_limit_sec=10.0,
            prepared=_prepared(),
        )

    assert result.verified
    assert result.start_sec == 0.2
    assert result.end_sec == 9.8
    assert result.diagnostics["start_quiet"] == [0.0, 0.3]
    assert result.diagnostics["end_quiet"] == [9.7, 10.0]


def test_true_media_duration_allows_post_caption_end_silence(
    tmp_path: Path,
) -> None:
    decoded_end_limits: list[float] = []

    def fake_decode(
        _source,
        *,
        window_start_sec,
        window_duration_sec,
        output_path,
        **_kwargs,
    ):
        if output_path.name == "start-0.wav":
            spans = [(0.3, 0), (window_duration_sec - 0.3, 12000)]
        else:
            decoded_end_limits.append(window_start_sec + window_duration_sec)
            leading_sound = 10.0 - window_start_sec
            spans = [
                (leading_sound, 12000),
                (0.3, 0),
                (window_duration_sec - leading_sound - 0.3, 12000),
            ]
        _write_wav(output_path, spans)

    prepared = silence.AudioPreparationResult(
        "ready",
        source=silence.PreparedAudioSource(
            url="https://media.example/audio.m4a",
            format_id="140",
            duration_sec=12.0,
        ),
    )
    with mock.patch.object(silence, "_decode_window", side_effect=fake_decode):
        result = silence.verify_acoustic_boundaries(
            "dQw4w9WgXcQ",
            2.0,
            10.0,
            search_start_limit_sec=0.0,
            search_end_limit_sec=20.0,
            prepared=prepared,
        )

    assert result.verified
    assert result.end_sec == 10.1
    assert decoded_end_limits == [12.0]
    assert result.diagnostics["search_end_limit_sec"] == 12.0
    assert result.diagnostics["source_duration_sec"] == 12.0


def test_exact_source_end_is_accepted_when_audio_reaches_edge_in_quiet(
    tmp_path: Path,
) -> None:
    def fake_decode(
        _source,
        *,
        window_duration_sec,
        output_path,
        **_kwargs,
    ):
        if output_path.name == "start-0.wav":
            spans = [(0.3, 0), (window_duration_sec - 0.3, 12000)]
        else:
            spans = [(window_duration_sec - 0.3, 12000), (0.3, 0)]
        _write_wav(output_path, spans)

    prepared = silence.AudioPreparationResult(
        "ready",
        source=silence.PreparedAudioSource(
            url="https://media.example/audio.m4a",
            format_id="140",
            duration_sec=10.0,
        ),
    )
    with mock.patch.object(silence, "_decode_window", side_effect=fake_decode):
        result = silence.verify_acoustic_boundaries(
            "dQw4w9WgXcQ",
            2.0,
            10.0,
            search_start_limit_sec=0.0,
            search_end_limit_sec=10.0,
            prepared=prepared,
        )

    assert result.verified
    assert result.end_sec == 10.0
    assert result.diagnostics["end_quiet"] == [9.7, 10.0]


def test_semantic_end_limit_is_not_treated_as_the_physical_source_edge(
    tmp_path: Path,
) -> None:
    def fake_decode(
        _source,
        *,
        window_duration_sec,
        output_path,
        **_kwargs,
    ):
        if output_path.name == "start-0.wav":
            spans = [(0.3, 0), (window_duration_sec - 0.3, 12000)]
        else:
            spans = [(window_duration_sec - 0.3, 12000), (0.3, 0)]
        _write_wav(output_path, spans)

    prepared = silence.AudioPreparationResult(
        "ready",
        source=silence.PreparedAudioSource(
            url="https://media.example/audio.m4a",
            format_id="140",
            duration_sec=20.0,
        ),
    )
    with mock.patch.object(silence, "_decode_window", side_effect=fake_decode):
        result = silence.verify_acoustic_boundaries(
            "dQw4w9WgXcQ",
            2.0,
            10.0,
            search_start_limit_sec=0.0,
            search_end_limit_sec=10.0,
            prepared=prepared,
        )

    assert result.status == "unavailable"
    assert result.diagnostics["reason"] == "end_silence_not_found"


def test_source_end_with_continuous_sound_is_not_assumed_to_be_silence(
    tmp_path: Path,
) -> None:
    def fake_decode(
        _source,
        *,
        window_duration_sec,
        output_path,
        **_kwargs,
    ):
        if output_path.name == "start-0.wav":
            spans = [(0.3, 0), (window_duration_sec - 0.3, 12000)]
        else:
            spans = [(window_duration_sec, 12000)]
        _write_wav(output_path, spans)

    with mock.patch.object(silence, "_decode_window", side_effect=fake_decode):
        result = silence.verify_acoustic_boundaries(
            "dQw4w9WgXcQ",
            2.0,
            10.0,
            search_start_limit_sec=0.0,
            search_end_limit_sec=10.0,
            prepared=_prepared(),
        )

    assert result.status == "unavailable"
    assert result.diagnostics["reason"] == "end_silence_not_found"


def test_final_cue_can_extend_into_measured_end_silence(tmp_path: Path) -> None:
    start_wav = tmp_path / "start.wav"
    end_wav = tmp_path / "end.wav"
    _write_wav(start_wav, [(2.7, 12000), (0.40, 0), (2.90, 12000)])
    _write_wav(end_wav, [(3.0, 12000), (0.30, 0), (2.70, 12000)])

    def fake_decode(_source, *, window_start_sec, output_path, **_kwargs):
        shutil.copyfile(start_wav if window_start_sec < 8 else end_wav, output_path)

    with mock.patch.object(silence, "_decode_window", side_effect=fake_decode):
        result = silence.verify_acoustic_boundaries(
            "dQw4w9WgXcQ", 10.0, 20.0, prepared=_prepared()
        )

    assert result.verified
    assert result.start_sec == 10.0
    assert result.end_sec == 20.1
    assert result.diagnostics["end_shift_sec"] == 0.1


def test_background_noise_above_required_threshold_is_unavailable(tmp_path: Path) -> None:
    start_wav = tmp_path / "noisy-start.wav"
    end_wav = tmp_path / "noisy-end.wav"
    _write_wav(start_wav, [(2.7, 12000), (0.40, 1000), (2.90, 12000)])
    _write_wav(end_wav, [(3.0, 12000), (0.30, 1000), (2.70, 12000)])

    def fake_decode(_source, *, window_start_sec, output_path, **_kwargs):
        shutil.copyfile(start_wav if window_start_sec < 8 else end_wav, output_path)

    with mock.patch.object(silence, "_decode_window", side_effect=fake_decode):
        result = silence.verify_acoustic_boundaries(
            "dQw4w9WgXcQ", 10.0, 20.0, prepared=_prepared()
        )

    assert result.status == "unavailable"
    assert result.diagnostics["reason"] == "start_silence_not_found"


def test_missing_start_silence_is_unavailable_and_keeps_original_range(tmp_path: Path) -> None:
    tone = tmp_path / "tone.wav"
    end_wav = tmp_path / "end.wav"
    _write_wav(tone, [(6.0, 12000)])
    _write_wav(end_wav, [(3.0, 12000), (0.3, 0), (2.7, 12000)])

    def fake_decode(_source, *, window_start_sec, output_path, **_kwargs):
        shutil.copyfile(tone if window_start_sec < 8 else end_wav, output_path)

    with mock.patch.object(silence, "_decode_window", side_effect=fake_decode):
        result = silence.verify_acoustic_boundaries(
            "dQw4w9WgXcQ", 10.0, 20.0, prepared=_prepared()
        )

    assert result.status == "unavailable"
    assert (result.start_sec, result.end_sec) == (10.0, 20.0)
    assert result.diagnostics["reason"] == "start_silence_not_found"


def test_end_requires_full_one_hundred_millisecond_cushion(tmp_path: Path) -> None:
    start_wav = tmp_path / "start.wav"
    short_end = tmp_path / "short-end.wav"
    _write_wav(start_wav, [(2.8, 12000), (0.25, 0), (2.95, 12000)])
    _write_wav(short_end, [(3.0, 12000), (0.09, 0), (2.91, 12000)])

    def fake_decode(_source, *, window_start_sec, output_path, **_kwargs):
        shutil.copyfile(start_wav if window_start_sec < 8 else short_end, output_path)

    with mock.patch.object(silence, "_decode_window", side_effect=fake_decode):
        result = silence.verify_acoustic_boundaries(
            "dQw4w9WgXcQ", 10.0, 20.0, prepared=_prepared()
        )

    assert result.status == "unavailable"
    assert result.diagnostics["reason"] == "end_silence_not_found"


def test_quiet_detector_enforces_dbfs_threshold_and_minimum_run(tmp_path: Path) -> None:
    wav = tmp_path / "threshold.wav"
    _write_wav(wav, [(0.1, 12000), (0.12, 300), (0.1, 12000), (0.11, 0), (0.1, 12000)])

    intervals = silence._quiet_intervals(
        wav,
        absolute_start_sec=5.0,
        threshold_dbfs=-38.0,
        min_quiet_ms=120,
    )

    assert len(intervals) == 1
    assert intervals[0].start_sec == pytest.approx(5.1)
    assert intervals[0].end_sec == pytest.approx(5.22)


def test_decode_failure_is_fail_soft_and_temp_files_are_cleaned(tmp_path: Path) -> None:
    observed_parent: Path | None = None

    def fail_decode(_source, *, output_path, **_kwargs):
        nonlocal observed_parent
        observed_parent = output_path.parent
        output_path.write_bytes(b"partial")
        raise silence._Unavailable("decode", "process_failed")

    with mock.patch.object(silence, "_decode_window", side_effect=fail_decode):
        result = silence.verify_acoustic_boundaries(
            "dQw4w9WgXcQ", 10.0, 20.0, prepared=_prepared()
        )

    assert result.status == "unavailable"
    assert result.diagnostics["stage"] == "decode"
    assert result.diagnostics["reason"] == "process_failed"
    assert observed_parent is not None and not observed_parent.exists()


def test_cancelled_before_work_never_resolves_or_decodes() -> None:
    with mock.patch.object(silence, "_prepare_audio_source") as resolve, mock.patch.object(
        silence, "_decode_window"
    ) as decode:
        result = silence.verify_acoustic_boundaries(
            "dQw4w9WgXcQ", 10.0, 20.0, cancel_check=lambda: True
        )

    assert result.status == "unavailable"
    assert result.diagnostics["reason"] == "cancelled"
    resolve.assert_not_called()
    decode.assert_not_called()


def test_command_timeout_kills_child_process() -> None:
    process = mock.Mock(returncode=None)
    process.communicate.return_value = (b"", b"")
    with mock.patch.object(
        silence.subprocess, "Popen", return_value=process
    ) as popen, mock.patch.object(
        silence.os, "killpg"
    ) as killpg, mock.patch.object(silence.time, "monotonic", side_effect=[0.0, 2.0]):
        with pytest.raises(silence._Unavailable, match="deadline_exceeded"):
            silence._run_command(
                ["fake"], deadline=1.0, cancel_check=None, stage="decode"
            )

    assert popen.call_args.kwargs["start_new_session"] is True
    killpg.assert_called_once_with(process.pid, silence.signal.SIGKILL)
    process.communicate.assert_called_once_with(timeout=1.0)


def test_ffmpeg_uses_bounded_seek_before_input(tmp_path: Path) -> None:
    output = tmp_path / "edge.wav"
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(list(command))
        _write_wav(output, [(4.0, 0)])
        return b"", b""

    source = silence.PreparedAudioSource(
        url="https://media.example/audio.m4a",
        headers={"User-Agent": "test"},
        proxy_url="https://proxy.example:443",
    )
    with mock.patch.object(silence, "_run_command", side_effect=fake_run):
        silence._decode_window(
            source,
            window_start_sec=91.0,
            window_duration_sec=99.0,
            output_path=output,
            ffmpeg_bin="ffmpeg",
            deadline=9999999999.0,
            cancel_check=None,
        )

    command = commands[0]
    assert command.index("-ss") < command.index("-i")
    assert float(command[command.index("-t") + 1]) == 6.0
    assert "-http_proxy" in command
    assert "-headers" in command
    assert command[command.index("-threads") + 1] == "1"
    assert command.index("-threads") < command.index("-i")
    assert command[command.index("-ar") + 1] == "16000"


@pytest.mark.parametrize(
    "reason",
    ["process_failed", "media_http_5xx", "media_network_error"],
)
def test_ffmpeg_decode_retries_one_transient_failure_without_refreshing(
    tmp_path: Path,
    reason: str,
) -> None:
    output = tmp_path / "edge.wav"
    calls = 0
    refresh = mock.Mock()
    source = silence.PreparedAudioSource(url="https://media.example/audio.m4a")
    source._decode_state.configure_refresh(refresh)

    def fake_run(_command, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            output.write_bytes(b"partial")
            raise silence._Unavailable("decode", reason)
        assert not output.exists()
        _write_wav(output, [(0.2, 0)])
        return b"", b""

    with mock.patch.object(silence, "_run_command", side_effect=fake_run):
        silence._decode_window(
            source,
            window_start_sec=0.0,
            window_duration_sec=0.2,
            output_path=output,
            ffmpeg_bin="ffmpeg",
            deadline=time.monotonic() + 10.0,
            cancel_check=None,
        )

    assert calls == 2
    refresh.assert_not_called()


@pytest.mark.parametrize(
    ("reason", "expected_calls"),
    [("process_failed", 2), ("cancelled", 1), ("deadline_exceeded", 1)],
)
def test_ffmpeg_decode_retry_is_bounded_and_reason_specific(
    tmp_path: Path,
    reason: str,
    expected_calls: int,
) -> None:
    output = tmp_path / f"{reason}.wav"
    run = mock.Mock(side_effect=silence._Unavailable("decode", reason))

    with mock.patch.object(silence, "_run_command", run), pytest.raises(
        silence._Unavailable,
        match=reason,
    ):
        silence._decode_window(
            silence.PreparedAudioSource(url="https://media.example/audio.m4a"),
            window_start_sec=0.0,
            window_duration_sec=0.2,
            output_path=output,
            ffmpeg_bin="ffmpeg",
            deadline=time.monotonic() + 10.0,
            cancel_check=None,
        )

    assert run.call_count == expected_calls


@pytest.mark.parametrize(
    ("stderr", "reason"),
    [
        (b"HTTP error 401 Unauthorized for ?token=secret", "media_http_401"),
        (b"HTTP error 403 Forbidden for ?token=secret", "media_http_403"),
        (b"HTTP error 404 Not Found for ?token=secret", "media_http_404"),
        (b"Server returned 410 Gone for ?token=secret", "media_http_410"),
        (b"HTTP error 429 Too Many Requests for ?token=secret", "media_http_429"),
        (b"HTTP error 503 Service Unavailable for ?token=secret", "media_http_5xx"),
        (b"Connection timed out for ?token=secret", "media_network_error"),
    ],
)
def test_ffmpeg_media_failures_are_safely_classified(
    stderr: bytes,
    reason: str,
) -> None:
    assert silence._process_failure_reason("decode", stderr) == reason
    assert "secret" not in silence._process_failure_reason("decode", stderr)


def test_concurrent_decode_failures_share_one_source_refresh(tmp_path: Path) -> None:
    source = silence.PreparedAudioSource(url="https://media.example/expired.m4a")
    refreshed = silence.PreparedAudioSource(url="https://media.example/healthy.m4a")
    old_route_barrier = threading.Barrier(2)
    refresh_calls = 0
    decoded_urls: list[str] = []
    lock = threading.Lock()

    def refresh(_deadline, _cancel_check):
        nonlocal refresh_calls
        with lock:
            refresh_calls += 1
        time.sleep(0.02)
        return refreshed

    def fake_run(command, **_kwargs):
        media_url = command[command.index("-i") + 1]
        with lock:
            decoded_urls.append(media_url)
        if media_url.endswith("expired.m4a"):
            old_route_barrier.wait(timeout=1.0)
            raise silence._Unavailable("decode", "media_http_403")
        _write_wav(Path(command[-1]), [(0.2, 0)])
        return b"", b""

    source._decode_state.configure_refresh(refresh)
    with mock.patch.object(silence, "_run_command", side_effect=fake_run):
        with silence.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    silence._decode_window,
                    source,
                    window_start_sec=float(index),
                    window_duration_sec=0.2,
                    output_path=tmp_path / f"shared-refresh-{index}.wav",
                    ffmpeg_bin="ffmpeg",
                    deadline=time.monotonic() + 5.0,
                    cancel_check=None,
                )
                for index in range(2)
            ]
            for future in futures:
                future.result(timeout=3.0)

    assert refresh_calls == 1
    assert decoded_urls.count("https://media.example/expired.m4a") == 2
    assert decoded_urls.count("https://media.example/healthy.m4a") == 2


def test_two_transient_failures_trip_source_and_suppress_later_decode(
    tmp_path: Path,
) -> None:
    source = silence.PreparedAudioSource(url="https://media.example/audio.m4a")
    run = mock.Mock(
        side_effect=silence._Unavailable("decode", "media_network_error")
    )

    with mock.patch.object(silence, "_run_command", run):
        for index in range(2):
            with pytest.raises(silence._Unavailable) as exc:
                silence._decode_window(
                    source,
                    window_start_sec=float(index),
                    window_duration_sec=0.2,
                    output_path=tmp_path / f"network-{index}.wav",
                    ffmpeg_bin="ffmpeg",
                    deadline=time.monotonic() + 5.0,
                    cancel_check=None,
                )
            assert exc.value.reason == "media_network_error"

    assert run.call_count == 2


def test_terminal_media_failure_trips_only_that_source_circuit(tmp_path: Path) -> None:
    failed = silence.PreparedAudioSource(url="https://media.example/failed.m4a")
    independent = silence.PreparedAudioSource(url="https://media.example/other.m4a")
    run = mock.Mock(side_effect=silence._Unavailable("decode", "media_http_403"))

    with mock.patch.object(silence, "_run_command", run):
        for index in range(2):
            with pytest.raises(silence._Unavailable) as exc:
                silence._decode_window(
                    failed,
                    window_start_sec=float(index),
                    window_duration_sec=0.2,
                    output_path=tmp_path / f"failed-{index}.wav",
                    ffmpeg_bin="ffmpeg",
                    deadline=time.monotonic() + 10.0,
                    cancel_check=None,
                )
            assert exc.value.reason == "media_http_403"

        with pytest.raises(silence._Unavailable):
            silence._decode_window(
                independent,
                window_start_sec=0.0,
                window_duration_sec=0.2,
                output_path=tmp_path / "other.wav",
                ffmpeg_bin="ffmpeg",
                deadline=time.monotonic() + 10.0,
                cancel_check=None,
            )

    assert run.call_count == 2


def test_ffmpeg_decodes_are_globally_bounded_to_four_and_per_source_to_two(
    tmp_path: Path,
) -> None:
    active = 0
    peak_active = 0
    active_by_source: dict[str, int] = {}
    peak_by_source: dict[str, int] = {}
    active_lock = threading.Lock()
    four_entered = threading.Event()

    def fake_run(command, **_kwargs):
        nonlocal active, peak_active
        source_url = command[command.index("-i") + 1]
        with active_lock:
            active += 1
            peak_active = max(peak_active, active)
            active_by_source[source_url] = active_by_source.get(source_url, 0) + 1
            peak_by_source[source_url] = max(
                peak_by_source.get(source_url, 0), active_by_source[source_url]
            )
            if active == 4:
                four_entered.set()
        assert four_entered.wait(timeout=1.0)
        time.sleep(0.02)
        _write_wav(Path(command[-1]), [(0.2, 0)])
        with active_lock:
            active -= 1
            active_by_source[source_url] -= 1
        return b"", b""

    sources = (
        silence.PreparedAudioSource(url="https://media.example/a.m4a"),
        silence.PreparedAudioSource(url="https://media.example/b.m4a"),
    )
    decode_slots = threading.BoundedSemaphore(4)
    with mock.patch.object(silence, "_decode_slots", decode_slots), mock.patch.object(
        silence, "_run_command", side_effect=fake_run
    ):
        with silence.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(
                    silence._decode_window,
                    sources[index % len(sources)],
                    window_start_sec=float(index),
                    window_duration_sec=0.2,
                    output_path=tmp_path / f"edge-{index}.wav",
                    ffmpeg_bin="ffmpeg",
                    deadline=time.monotonic() + 2.0,
                    cancel_check=None,
                )
                for index in range(8)
            ]
            for future in futures:
                future.result(timeout=2.0)

    assert peak_active == 4
    assert peak_by_source == {
        "https://media.example/a.m4a": 2,
        "https://media.example/b.m4a": 2,
    }


def test_ffmpeg_decode_slot_wait_is_cancellable(tmp_path: Path) -> None:
    cancelled = False

    class BusySlots:
        def acquire(self, *, timeout: float) -> bool:
            nonlocal cancelled
            assert timeout <= 0.1
            cancelled = True
            return False

        def release(self) -> None:
            raise AssertionError("an unacquired slot must not be released")

    with mock.patch.object(silence, "_decode_slots", BusySlots()), mock.patch.object(
        silence, "_run_command"
    ) as run:
        with pytest.raises(silence._Unavailable) as exc:
            silence._decode_window(
                silence.PreparedAudioSource(url="https://media.example/audio.m4a"),
                window_start_sec=0.0,
                window_duration_sec=1.0,
                output_path=tmp_path / "cancelled.wav",
                ffmpeg_bin="ffmpeg",
                deadline=time.monotonic() + 1.0,
                cancel_check=lambda: cancelled,
            )

    assert exc.value.reason == "cancelled"
    run.assert_not_called()


def test_global_decode_slot_wait_exits_when_source_circuit_trips(
    tmp_path: Path,
) -> None:
    waiting = threading.Event()
    source = silence.PreparedAudioSource(url="https://media.example/audio.m4a")

    class BusySlots:
        def acquire(self, *, timeout: float) -> bool:
            waiting.set()
            time.sleep(min(0.02, timeout))
            return False

        def release(self) -> None:
            raise AssertionError("an unacquired slot must not be released")

    with mock.patch.object(silence, "_decode_slots", BusySlots()), mock.patch.object(
        silence, "_run_command"
    ) as run:
        with silence.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                silence._decode_window,
                source,
                window_start_sec=0.0,
                window_duration_sec=1.0,
                output_path=tmp_path / "tripped.wav",
                ffmpeg_bin="ffmpeg",
                deadline=time.monotonic() + 2.0,
                cancel_check=None,
            )
            assert waiting.wait(timeout=1.0)
            source._decode_state.trip("media_http_403")
            with pytest.raises(silence._Unavailable) as exc:
                future.result(timeout=1.0)

    assert exc.value.reason == "media_http_403"
    run.assert_not_called()


def test_audio_entry_chooses_lowest_valid_requested_audio_without_format_override() -> None:
    selected = silence._audio_entry(
        {
            "requested_downloads": [
                {
                    "url": "https://media.example/high.opus",
                    "format_id": "251",
                    "acodec": "opus",
                    "vcodec": "none",
                    "tbr": 132.0,
                },
                {
                    "url": "https://media.example/selected-low.opus",
                    "format_id": "250",
                    "acodec": "opus",
                    "vcodec": "none",
                    "tbr": 64.0,
                },
            ],
            "url": "https://media.example/video.mp4",
            "format_id": "18",
            "acodec": "aac",
            "vcodec": "h264",
            "tbr": 500.0,
            "formats": [
                {
                    "url": "https://media.example/unknown.m4a",
                    "format_id": "none",
                    "acodec": "aac",
                    "vcodec": "none",
                    "tbr": 0,
                },
                {
                    "url": "https://media.example/low.m4a",
                    "format_id": "139",
                    "acodec": "aac",
                    "vcodec": "none",
                    "tbr": 49.0,
                },
            ],
        }
    )

    assert selected["format_id"] == "250"


def test_audio_entry_prefers_valid_selected_root_before_formats_fallback() -> None:
    selected = silence._audio_entry(
        {
            "url": "https://media.example/selected.m4a",
            "format_id": "140",
            "acodec": "aac",
            "vcodec": "none",
            "tbr": 64.0,
            "formats": [
                {
                    "url": "https://media.example/unselected.m4a",
                    "format_id": "139",
                    "acodec": "aac",
                    "vcodec": "none",
                    "tbr": 49.0,
                }
            ],
        }
    )

    assert selected["format_id"] == "140"


@pytest.mark.parametrize(
    "entry",
    [
        {
            "url": "https://media.example/video.mp4",
            "acodec": "aac",
            "vcodec": "h264",
            "tbr": 49.0,
        },
        {
            "url": "https://media.example/unknown.m4a",
            "tbr": 49.0,
        },
        {
            "url": "https://media.example/audio.m4a",
            "acodec": "aac",
            "vcodec": "none",
            "tbr": 0.0,
        },
    ],
)
def test_audio_entry_rejects_video_unknown_and_nonpositive_formats(entry) -> None:
    assert silence._audio_entry({"requested_downloads": [entry]}) == {}


def test_decode_http_403_refreshes_once_through_next_route_without_refetching_words(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        silence,
        "get_settings",
        lambda: mock.Mock(proxy_urls="", ytdlp_pot_provider_url=""),
    )
    resolved_commands: list[list[str]] = []
    decoded_urls: list[str] = []
    words = (silence.lexical_timing.LexicalWord("biology", 1.0),)

    def fake_run(command, **kwargs):
        if kwargs["stage"] == "resolve":
            resolved_commands.append(list(command))
            suffix = "expired" if len(resolved_commands) == 1 else "healthy"
            return (
                b'{"url":"https://media.example/'
                + suffix.encode()
                + b'.m4a","format_id":"140","acodec":"aac",'
                b'"vcodec":"none","tbr":49.0,"automatic_captions":'
                b'{"en-orig":[{"ext":"json3",'
                b'"url":"https://captions.example/en?lang=en&kind=asr"}]}}',
                b"",
            )
        media_url = command[command.index("-i") + 1]
        decoded_urls.append(media_url)
        if media_url.endswith("expired.m4a"):
            raise silence._Unavailable("decode", "media_http_403")
        _write_wav(Path(command[-1]), [(0.2, 0)])
        return b"", b""

    fetch = mock.Mock(return_value=words)
    with mock.patch.object(
        silence, "_run_command", side_effect=fake_run
    ), mock.patch.object(
        silence.lexical_timing, "fetch_json3_words", fetch
    ):
        result = silence.prepare_audio_source("dQw4w9WgXcQ")
        assert result.ready and result.source is not None
        silence._decode_window(
            result.source,
            window_start_sec=0.0,
            window_duration_sec=0.2,
            output_path=tmp_path / "refreshed.wav",
            ffmpeg_bin="ffmpeg",
            deadline=time.monotonic() + 10.0,
            cancel_check=None,
        )

    assert decoded_urls == [
        "https://media.example/expired.m4a",
        "https://media.example/healthy.m4a",
    ]
    assert len(resolved_commands) == 2
    assert "youtube:player_client=web_embedded;player_skip=webpage" in resolved_commands[1]
    assert fetch.call_count == 1


def test_prepare_reuses_cookie_proxy_and_pot_configuration(tmp_path: Path, monkeypatch) -> None:
    cookie_file = tmp_path / "cookies.txt"
    cookie_file.write_text("# Netscape HTTP Cookie File\n")
    monkeypatch.setenv("YT_COOKIES_FILE", str(cookie_file))
    monkeypatch.setattr(
        silence,
        "get_settings",
        lambda: mock.Mock(
            proxy_urls="http://one.example:8080,http://two.example:8080",
            ytdlp_pot_provider_url="http://pot.internal:4416",
        ),
    )
    payload = (
        b'{"url":"https://media.example/audio.m4a","format_id":"140",'
        b'"acodec":"aac","vcodec":"none","tbr":49.0,'
        b'"duration":123.456,"http_headers":{"User-Agent":"agent"}}'
    )
    with mock.patch.object(
        silence, "_run_command", return_value=(payload, b"")
    ) as run:
        result = silence.prepare_audio_source("dQw4w9WgXcQ")

    assert result.ready
    command = run.call_args.args[0]
    assert command[command.index("--cookies") + 1] == str(cookie_file)
    assert command[command.index("--proxy") + 1] == "http://one.example:8080"
    assert command[command.index("--extractor-args") + 1] == (
        "youtubepot-bgutilhttp:base_url=http://pot.internal:4416"
    )
    assert command[command.index("--impersonate") + 1] == "chrome"
    assert command[command.index("--format") + 1] == (
        "worstaudio[acodec!=none][vcodec=none]"
    )
    assert "--no-remote-components" in command
    assert "--remote-components" not in command
    assert result.source is not None and result.source.format_id == "140"
    assert result.source.duration_sec == 123.456
    assert "media.example" not in repr(result)


def test_prepare_fetches_json3_words_once_from_the_existing_ytdlp_metadata(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        silence,
        "get_settings",
        lambda: mock.Mock(proxy_urls="", ytdlp_pot_provider_url=""),
    )
    payload = (
        b'{"url":"https://media.example/audio.m4a","format_id":"140",'
        b'"acodec":"aac","vcodec":"none","tbr":49.0,'
        b'"automatic_captions":{"en-orig":[{"ext":"json3",'
        b'"url":"https://captions.example/timed?sig=secret"}]}}'
    )
    words = (
        silence.lexical_timing.LexicalWord("biology", 36.48),
        silence.lexical_timing.LexicalWord("is", 37.48),
    )
    fetch = mock.Mock(return_value=words)
    with mock.patch.object(
        silence, "_run_command", return_value=(payload, b"")
    ) as run, mock.patch.object(
        silence.lexical_timing,
        "fetch_json3_words",
        fetch,
    ):
        result = silence.prepare_audio_source("dQw4w9WgXcQ", language="en-US")

    assert result.ready and result.source is not None
    assert run.call_count == 1
    assert fetch.call_count == 1
    assert result.source.lexical_words == words
    assert result.source.lexical_language == "en"
    assert result.diagnostics["lexical_word_count"] == 2
    assert "captions.example" not in repr(result.source)


def test_prepare_gives_each_fallback_track_a_bounded_lexical_deadline(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        silence,
        "get_settings",
        lambda: mock.Mock(proxy_urls="", ytdlp_pot_provider_url=""),
    )
    payload = (
        b'{"url":"https://media.example/audio.m4a","format_id":"140",'
        b'"acodec":"aac","vcodec":"none","tbr":49.0,'
        b'"automatic_captions":{"en-orig":[{"ext":"json3",'
        b'"url":"https://captions.example/automatic?lang=en&kind=asr"}]},'
        b'"subtitles":{"en":[{"ext":"json3",'
        b'"url":"https://captions.example/manual?lang=en"}]}}'
    )
    words = (
        silence.lexical_timing.LexicalWord("biology", 36.48),
        silence.lexical_timing.LexicalWord("is", 37.48),
    )
    clock = [100.0]
    deadline_budgets: list[float] = []

    def fake_fetch(track, **kwargs):
        deadline_budgets.append(float(kwargs["deadline"]) - clock[0])
        if "automatic" in track.url:
            clock[0] += 1.9
            return ()
        return words

    with mock.patch.object(
        silence.time, "monotonic", side_effect=lambda: clock[0]
    ), mock.patch.object(
        silence, "_run_command", return_value=(payload, b"")
    ), mock.patch.object(
        silence.lexical_timing, "fetch_json3_words", side_effect=fake_fetch
    ) as fetch:
        result = silence.prepare_audio_source("dQw4w9WgXcQ", language="en-US")

    assert result.ready and result.source is not None
    assert fetch.call_count == 2
    assert deadline_budgets == [2.0, 2.0]
    assert result.source.lexical_words == words
    assert result.source.lexical_language == "en"
    assert "captions.example" not in repr(result.source)
    assert "captions.example" not in repr(result)


def test_prepare_attempts_at_most_two_tracks_and_keeps_missing_words_closed(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        silence,
        "get_settings",
        lambda: mock.Mock(proxy_urls="", ytdlp_pot_provider_url=""),
    )
    payload = (
        b'{"url":"https://media.example/audio.m4a","format_id":"140",'
        b'"acodec":"aac","vcodec":"none","tbr":49.0,'
        b'"automatic_captions":{"en-orig":[{"ext":"json3",'
        b'"url":"https://captions.example/one?lang=en&kind=asr"}],'
        b'"en":[{"ext":"json3",'
        b'"url":"https://captions.example/two?lang=en&kind=asr"}]},'
        b'"subtitles":{"en":[{"ext":"json3",'
        b'"url":"https://captions.example/three?lang=en"}]}}'
    )
    fetch = mock.Mock(return_value=())
    with mock.patch.object(
        silence, "_run_command", return_value=(payload, b"")
    ), mock.patch.object(
        silence.lexical_timing, "fetch_json3_words", fetch
    ):
        result = silence.prepare_audio_source("dQw4w9WgXcQ", language="en-US")

    assert result.ready and result.source is not None
    assert fetch.call_count == 2
    assert result.source.lexical_words == ()
    assert result.source.lexical_language == ""
    assert result.diagnostics["lexical_word_count"] == 0


@pytest.mark.parametrize("first_reason", ["proxy_failed", "youtube_bot_challenge"])
def test_prepare_rotates_to_the_next_configured_proxy(monkeypatch, first_reason: str) -> None:
    monkeypatch.setattr(
        silence,
        "get_settings",
        lambda: mock.Mock(
            proxy_urls="http://broken.example:8080,http://healthy.example:8080",
            ytdlp_pot_provider_url="",
        ),
    )
    payload = (
        b'{"url":"https://media.example/audio.m4a","format_id":"140",'
        b'"acodec":"aac","vcodec":"none","tbr":49.0}'
    )
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(list(command))
        if len(commands) == 1:
            raise silence._Unavailable("resolve", first_reason)
        return payload, b""

    with mock.patch.object(silence, "_run_command", side_effect=fake_run):
        result = silence.prepare_audio_source("dQw4w9WgXcQ")

    assert result.ready
    assert commands[0][commands[0].index("--proxy") + 1] == "http://broken.example:8080"
    assert commands[1][commands[1].index("--proxy") + 1] == "http://healthy.example:8080"
    assert result.source is not None
    assert result.source.proxy_url == "http://healthy.example:8080"


def test_prepare_uses_direct_route_after_all_configured_proxies_fail(monkeypatch) -> None:
    monkeypatch.setattr(
        silence,
        "get_settings",
        lambda: mock.Mock(
            proxy_urls="http://one.example:8080,http://two.example:8080",
            ytdlp_pot_provider_url="",
        ),
    )
    payload = (
        b'{"url":"https://media.example/audio.m4a","format_id":"251",'
        b'"acodec":"opus","vcodec":"none","tbr":49.0}'
    )
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(list(command))
        if len(commands) <= 2:
            raise silence._Unavailable("resolve", "proxy_failed")
        return payload, b""

    with mock.patch.object(silence, "_run_command", side_effect=fake_run):
        result = silence.prepare_audio_source("dQw4w9WgXcQ")

    assert result.ready
    assert "--proxy" in commands[0] and "--proxy" in commands[1]
    assert "--proxy" not in commands[2]
    assert result.source is not None and result.source.proxy_url == ""


def test_prepare_retries_bot_challenge_with_embedded_player(monkeypatch) -> None:
    monkeypatch.setattr(
        silence,
        "get_settings",
        lambda: mock.Mock(proxy_urls="", ytdlp_pot_provider_url=""),
    )
    payload = (
        b'{"url":"https://media.example/audio.m4a","format_id":"251",'
        b'"acodec":"opus","vcodec":"none","tbr":49.0}'
    )
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(list(command))
        if len(commands) == 1:
            raise silence._Unavailable("resolve", "youtube_bot_challenge")
        return payload, b""

    with mock.patch.object(silence, "_run_command", side_effect=fake_run):
        result = silence.prepare_audio_source("dQw4w9WgXcQ")

    assert result.ready
    assert "youtube:player_client=web_embedded" not in commands[0]
    assert (
        "youtube:player_client=web_embedded;player_skip=webpage" in commands[1]
    )


def test_prepare_uses_mweb_when_pot_provider_is_configured(monkeypatch) -> None:
    monkeypatch.setattr(
        silence,
        "get_settings",
        lambda: mock.Mock(
            proxy_urls="",
            ytdlp_pot_provider_url="http://pot.internal:4416",
        ),
    )
    payload = (
        b'{"url":"https://media.example/audio.m4a","format_id":"251",'
        b'"acodec":"opus","vcodec":"none","tbr":49.0}'
    )
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(list(command))
        if len(commands) == 1:
            raise silence._Unavailable("resolve", "youtube_bot_challenge")
        return payload, b""

    with mock.patch.object(silence, "_run_command", side_effect=fake_run):
        result = silence.prepare_audio_source("dQw4w9WgXcQ")

    assert result.ready
    assert "youtube:player_client=mweb;player_skip=webpage" in commands[1]
    assert "youtubepot-bgutilhttp:base_url=http://pot.internal:4416" in commands[1]


def test_prepare_caps_route_plan_and_gives_each_attempt_viable_time(monkeypatch) -> None:
    monkeypatch.setattr(
        silence,
        "get_settings",
        lambda: mock.Mock(
            proxy_urls=(
                "http://one.example:8080,http://two.example:8080,"
                "http://three.example:8080"
            ),
            ytdlp_pot_provider_url="",
        ),
    )
    budgets: list[float] = []
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(list(command))
        budgets.append(float(kwargs["deadline"]) - silence.time.monotonic())
        raise silence._Unavailable("resolve", "proxy_failed")

    with mock.patch.object(silence, "_run_command", side_effect=fake_run):
        result = silence.prepare_audio_source("dQw4w9WgXcQ")

    assert result.status == "unavailable"
    assert len(budgets) == 3
    assert min(budgets) >= 7.5
    assert "--proxy" in commands[0]
    assert "http://one.example:8080" in commands[0]
    assert "--proxy" in commands[1]
    assert "http://two.example:8080" in commands[1]
    assert "--proxy" not in commands[2]
    assert (
        "youtube:player_client=web_embedded;player_skip=webpage" in commands[2]
    )


def test_prepare_reserves_time_for_fallback_after_attempt_timeout(monkeypatch) -> None:
    monkeypatch.setattr(
        silence,
        "get_settings",
        lambda: mock.Mock(proxy_urls="", ytdlp_pot_provider_url=""),
    )
    payload = (
        b'{"url":"https://media.example/audio.m4a","format_id":"251",'
        b'"acodec":"opus","vcodec":"none","tbr":49.0}'
    )
    deadlines: list[float] = []

    def fake_run(_command, **kwargs):
        deadlines.append(float(kwargs["deadline"]))
        if len(deadlines) == 1:
            raise silence._Unavailable("resolve", "deadline_exceeded")
        return payload, b""

    with mock.patch.object(silence, "_run_command", side_effect=fake_run):
        result = silence.prepare_audio_source("dQw4w9WgXcQ")

    assert result.ready
    assert len(deadlines) == 2
    assert deadlines[0] < deadlines[1]


def test_prepare_preserves_actionable_reasons_across_failed_attempts(monkeypatch) -> None:
    monkeypatch.setattr(
        silence,
        "get_settings",
        lambda: mock.Mock(proxy_urls="", ytdlp_pot_provider_url=""),
    )
    failures = iter(("youtube_bot_challenge", "process_failed"))

    def fake_run(_command, **_kwargs):
        raise silence._Unavailable("resolve", next(failures))

    with mock.patch.object(silence, "_run_command", side_effect=fake_run):
        result = silence.prepare_audio_source("dQw4w9WgXcQ")

    assert result.status == "unavailable"
    assert result.diagnostics["reason"] == "youtube_bot_challenge"
    assert result.diagnostics["attempt_reasons"] == [
        "direct:default:youtube_bot_challenge",
        "direct:web_embedded:process_failed",
    ]


def test_preparation_attempt_reasons_reach_boundary_diagnostics() -> None:
    prepared = silence.AudioPreparationResult(
        "unavailable",
        diagnostics={
            "stage": "resolve",
            "reason": "youtube_bot_challenge",
            "attempt_reasons": [
                "proxy:default:proxy_failed",
                "direct:web_embedded:youtube_bot_challenge",
            ],
            "elapsed_ms": 123,
        },
    )

    result = silence.verify_acoustic_boundaries(
        "dQw4w9WgXcQ", 10.0, 20.0, prepared=prepared
    )

    assert result.status == "unavailable"
    assert result.diagnostics["attempt_reasons"] == prepared.diagnostics["attempt_reasons"]
    assert result.diagnostics["prepare_elapsed_ms"] == 123


def test_global_prepare_timeout_keeps_the_terminal_attempt_reason(monkeypatch) -> None:
    monkeypatch.setattr(
        silence,
        "get_settings",
        lambda: mock.Mock(proxy_urls="", ytdlp_pot_provider_url=""),
    )

    def fake_run(_command, **_kwargs):
        silence.time.sleep(0.02)
        raise silence._Unavailable("resolve", "deadline_exceeded")

    with mock.patch.object(silence, "_run_command", side_effect=fake_run):
        result = silence.prepare_audio_source("dQw4w9WgXcQ", timeout_sec=0.01)

    assert result.status == "unavailable"
    assert result.diagnostics["reason"] == "deadline_exceeded"
    assert result.diagnostics["attempt_reasons"] == [
        "direct:default:deadline_exceeded"
    ]


@pytest.mark.parametrize(
    ("stderr", "reason"),
    [
        (b"Sign in to confirm you're not a bot", "youtube_bot_challenge"),
        (b"Unable to connect to proxy", "proxy_failed"),
        (b"Requested format is not available", "format_unavailable"),
        (b"Remote component ejs failed", "component_failed"),
    ],
)
def test_resolver_process_failures_are_safely_classified(stderr: bytes, reason: str) -> None:
    process = mock.Mock(returncode=1)
    process.communicate.return_value = (b"", stderr)
    with mock.patch.object(silence.subprocess, "Popen", return_value=process):
        with pytest.raises(silence._Unavailable) as exc:
            silence._run_command(
                ["fake"], deadline=9999999999.0, cancel_check=None, stage="resolve"
            )

    assert exc.value.reason == reason


def test_resolver_classifies_terminal_error_not_prior_warning() -> None:
    stderr = (
        b"WARNING: Sign in to confirm you're not a bot\n"
        b"ERROR: Requested format is not available\n"
    )

    assert silence._process_failure_reason("resolve", stderr) == "format_unavailable"


def test_invalid_source_fails_before_yt_dlp() -> None:
    with mock.patch.object(silence, "_run_command") as run:
        result = silence.prepare_audio_source("https://attacker.example/private")

    assert result.status == "unavailable"
    assert result.diagnostics["reason"] == "invalid_youtube_source"
    run.assert_not_called()
