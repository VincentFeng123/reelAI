# backend/pipeline/tests/test_output_safety.py
"""Atomic ffmpeg output finalization, range-keyed export names, zip guard. Offline."""
from __future__ import annotations

from pathlib import Path

import pytest

from backend.errors import PipelineError
from backend.main import _zipable_files
from backend.pipeline.export import _export_fname, finalize_output


# ── fix 3: atomic finalize ────────────────────────────────────────────────────
def test_finalize_success_renames(tmp_path):
    tmp, out = tmp_path / "c.tmp.mp4", tmp_path / "c.mp4"
    tmp.write_bytes(b"fake-mp4-bytes")
    finalize_output(tmp, out, rc=0, err="")
    assert out.exists() and not tmp.exists()
    assert out.read_bytes() == b"fake-mp4-bytes"


def test_finalize_failure_cleans_tmp_and_raises(tmp_path):
    tmp, out = tmp_path / "c.tmp.mp4", tmp_path / "c.mp4"
    tmp.write_bytes(b"partial")
    with pytest.raises(PipelineError):
        finalize_output(tmp, out, rc=1, err="boom\nlast line")
    assert not tmp.exists() and not out.exists()


def test_finalize_zero_byte_output_is_failure(tmp_path):
    tmp, out = tmp_path / "c.tmp.mp4", tmp_path / "c.mp4"
    tmp.write_bytes(b"")
    with pytest.raises(PipelineError):
        finalize_output(tmp, out, rc=0, err="")
    assert not tmp.exists() and not out.exists()


# ── fix 4: range-keyed export name ───────────────────────────────────────────
def test_export_fname_embeds_range():
    assert _export_fname(2, "definition", 12.345, 67.89) == "clip_2_definition_12345_67890.mp4"
    assert _export_fname(2, "definition", 12.345, 68.0) != _export_fname(2, "definition", 12.345, 67.89)


# ── fix 6: zip guard ─────────────────────────────────────────────────────────
def test_zipable_files_skips_none_and_missing(tmp_path):
    real = tmp_path / "clip_1_other.mp4"
    real.write_bytes(b"x")
    clips = [{"path": None},                                # embed mode
             {"path": "/clips/v/clip_1_other.mp4"},         # exists
             {"path": "/clips/v/clip_9_gone.mp4"}]          # missing on disk
    files = _zipable_files(clips, tmp_path)
    assert files == [real]


def test_zipable_files_empty_for_embed_job(tmp_path):
    assert _zipable_files([{"path": None}, {"path": None}], tmp_path) == []
