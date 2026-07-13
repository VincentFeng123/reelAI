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
    assert "yt-dlp[default]==2026.3.17" in requirements


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
    _write_wav(end_wav, [(2.7, 12000), (0.30, 0), (3.00, 12000)])

    def fake_decode(_source, *, window_start_sec, output_path, **_kwargs):
        source = start_wav if window_start_sec < 8 else end_wav
        shutil.copyfile(source, output_path)

    with mock.patch.object(silence, "_decode_window", side_effect=fake_decode):
        result = silence.verify_acoustic_boundaries(
            "dQw4w9WgXcQ", 10.0, 20.3, prepared=_prepared()
        )

    assert result.verified
    assert result.start_sec == 10.0
    assert result.end_sec == 20.2
    assert result.diagnostics["start_quiet"] == [9.7, 10.1]
    assert result.diagnostics["end_quiet"] == [20.0, 20.3]
    assert result.diagnostics["start_cushion_ms"] == 100
    assert result.diagnostics["end_cushion_ms"] == 200


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


def test_background_noise_uses_bounded_adaptive_quiet_threshold(tmp_path: Path) -> None:
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

    assert result.verified
    assert result.diagnostics["adaptive_quiet"] is True
    assert result.diagnostics["start_threshold_dbfs"] == -24.0
    assert result.diagnostics["end_threshold_dbfs"] == -24.0


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
    with mock.patch.object(silence.subprocess, "Popen", return_value=process), mock.patch.object(
        silence.time, "monotonic", side_effect=[0.0, 2.0]
    ):
        with pytest.raises(silence._Unavailable, match="deadline_exceeded"):
            silence._run_command(
                ["fake"], deadline=1.0, cancel_check=None, stage="decode"
            )

    process.kill.assert_called_once_with()


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
        b'"http_headers":{"User-Agent":"agent"}}'
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
    assert result.source is not None and result.source.format_id == "140"
    assert "media.example" not in repr(result)


def test_invalid_source_fails_before_yt_dlp() -> None:
    with mock.patch.object(silence, "_run_command") as run:
        result = silence.prepare_audio_source("https://attacker.example/private")

    assert result.status == "unavailable"
    assert result.diagnostics["reason"] == "invalid_youtube_source"
    run.assert_not_called()
