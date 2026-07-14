from __future__ import annotations

import math
import shutil
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
    assert result.diagnostics["end_cushion_ms"] == 200


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
                (0.26, 0),
                (window_duration_sec - quiet_start - 0.26, 12000),
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
    assert result.end_sec == 19.92
    assert result.diagnostics["speech_handoff_verified"] is True
    assert result.diagnostics["semantic_start_limit_sec"] == 10.0
    assert result.diagnostics["semantic_end_limit_sec"] == 20.0
    assert result.diagnostics["observation_start_limit_sec"] == 9.0
    assert result.diagnostics["observation_end_limit_sec"] == 21.0


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
                [(quiet_start, 12000), (0.30, 0), (window_duration_sec - quiet_start - 0.30, 12000)]
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
    assert result.end_sec == 46.2
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
    assert result.end_sec == 46.2
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
    assert result.end_sec == 43.2
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
            spans = [(window_duration_sec - 0.11, 12000), (0.11, 0)]
        elif output_path.name == "end-1.wav":
            spans = [(0.11, 0), (window_duration_sec - 0.11, 12000)]
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
    assert result.end_sec == 43.09
    assert result.diagnostics["end_quiet"] == [42.89, 43.11]
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
    assert result.end_sec == 9.9
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
    assert result.end_sec == 10.2
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
    assert result.end_sec == 20.2
    assert result.diagnostics["end_shift_sec"] == 0.2


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


def test_end_requires_full_two_hundred_millisecond_cushion(tmp_path: Path) -> None:
    start_wav = tmp_path / "start.wav"
    short_end = tmp_path / "short-end.wav"
    _write_wav(start_wav, [(2.8, 12000), (0.25, 0), (2.95, 12000)])
    _write_wav(short_end, [(3.0, 12000), (0.15, 0), (2.85, 12000)])

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
        b'"duration":123.456,"http_headers":{"User-Agent":"agent"}}'
    )
    with mock.patch.object(silence, "_run_command", return_value=(payload, b"")) as run:
        result = silence.prepare_audio_source("dQw4w9WgXcQ")

    assert result.ready
    command = run.call_args.args[0]
    assert command[command.index("--cookies") + 1] == str(cookie_file)
    assert command[command.index("--proxy") + 1] == "http://one.example:8080"
    assert command[command.index("--extractor-args") + 1] == (
        "youtubepot-bgutilhttp:base_url=http://pot.internal:4416"
    )
    assert command[command.index("--impersonate") + 1] == "chrome"
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
        b'"automatic_captions":{"en-orig":[{"ext":"json3",'
        b'"url":"https://captions.example/timed?sig=secret"}]}}'
    )
    words = (
        silence.lexical_timing.LexicalWord("biology", 36.48),
        silence.lexical_timing.LexicalWord("is", 37.48),
    )
    fetch = mock.Mock(return_value=words)
    with mock.patch.object(silence, "_run_command", return_value=(payload, b"")) as run, mock.patch.object(
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


def test_prepare_rotates_to_the_next_configured_proxy(monkeypatch) -> None:
    monkeypatch.setattr(
        silence,
        "get_settings",
        lambda: mock.Mock(
            proxy_urls="http://broken.example:8080,http://healthy.example:8080",
            ytdlp_pot_provider_url="",
        ),
    )
    payload = b'{"url":"https://media.example/audio.m4a","format_id":"140"}'
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(list(command))
        if len(commands) == 1:
            raise silence._Unavailable("resolve", "proxy_failed")
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
    payload = b'{"url":"https://media.example/audio.m4a","format_id":"251"}'
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
    payload = b'{"url":"https://media.example/audio.m4a","format_id":"251"}'
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(list(command))
        if len(commands) == 1:
            raise silence._Unavailable("resolve", "youtube_bot_challenge")
        return payload, b""

    with mock.patch.object(silence, "_run_command", side_effect=fake_run):
        result = silence.prepare_audio_source("dQw4w9WgXcQ")

    assert result.ready
    assert "youtube:player_client=web_embedded;player_skip=webpage" not in commands[0]
    assert "youtube:player_client=web_embedded;player_skip=webpage" in commands[1]


def test_prepare_reserves_time_for_fallback_after_attempt_timeout(monkeypatch) -> None:
    monkeypatch.setattr(
        silence,
        "get_settings",
        lambda: mock.Mock(proxy_urls="", ytdlp_pot_provider_url=""),
    )
    payload = b'{"url":"https://media.example/audio.m4a","format_id":"251"}'
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
        "direct:embedded:process_failed",
    ]


def test_preparation_attempt_reasons_reach_boundary_diagnostics() -> None:
    prepared = silence.AudioPreparationResult(
        "unavailable",
        diagnostics={
            "stage": "resolve",
            "reason": "youtube_bot_challenge",
            "attempt_reasons": [
                "proxy:default:proxy_failed",
                "direct:embedded:youtube_bot_challenge",
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


def test_invalid_source_fails_before_yt_dlp() -> None:
    with mock.patch.object(silence, "_run_command") as run:
        result = silence.prepare_audio_source("https://attacker.example/private")

    assert result.status == "unavailable"
    assert result.diagnostics["reason"] == "invalid_youtube_source"
    run.assert_not_called()
