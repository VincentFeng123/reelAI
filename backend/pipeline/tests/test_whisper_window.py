"""_whisper_window: refine model + VAD kwargs, and returns (sents, wav_path). Model+ffmpeg stubbed."""
from __future__ import annotations

from pathlib import Path

import backend.pipeline.boundary as bmod


class _Seg:
    def __init__(self, start, end, text, words):
        self.start, self.end, self.text, self.words = start, end, text, words


class _W:
    def __init__(self, word, start, end):
        self.word, self.start, self.end = word, start, end


def test_whisper_window_returns_sents_and_wavpath(monkeypatch, tmp_path):
    captured = {}

    class _Model:
        def transcribe(self, path, **kw):
            captured["path"] = path
            captured["kw"] = kw
            segs = [_Seg(0.0, 1.0, "Hi there.", [_W("Hi", 0.0, 0.3), _W("there.", 0.4, 0.9)])]
            return segs, object()

    # ffmpeg is not run: fake subprocess.run and make the temp wav "exist"
    monkeypatch.setattr(bmod, "_get_refine_whisper", lambda: _Model())
    monkeypatch.setattr(bmod.subprocess, "run", lambda *a, **k: None)
    real_mkstemp = bmod.tempfile.mkstemp
    made = {}

    def _fake_mkstemp(suffix="", dir=None):
        fd, p = real_mkstemp(suffix=suffix, dir=dir)
        made["path"] = p
        return fd, p
    monkeypatch.setattr(bmod.tempfile, "mkstemp", _fake_mkstemp)

    sents, wav = bmod._whisper_window(Path(tmp_path) / "audio.m4a", 10.0, 20.0)

    assert wav is not None and str(wav) == made["path"]        # wav path returned, NOT deleted
    assert Path(wav).exists()
    assert captured["kw"].get("condition_on_previous_text") is False
    assert captured["kw"].get("temperature") == 0.0
    assert captured["kw"].get("beam_size") == 5
    assert captured["kw"].get("word_timestamps") is True
    assert captured["kw"].get("vad_filter") is True            # REFINE_VAD default on
    assert [s.text for s in sents]                              # sentences built + shifted
    Path(wav).unlink(missing_ok=True)


def test_pick_helpers_return_pick(monkeypatch):
    from backend.pipeline.boundary import Pick, _pick_end, _pick_start
    from backend.pipeline.sentences import Sentence

    def _s(i, a, b, term="."):
        return Sentence(idx=i, text=f"s{i}", start=a, end=b, terminator=term,
                        ends_with_period=(term in ".?!"), word_start_idx=i, word_end_idx=i,
                        align_confidence=1.0)

    pe = _pick_end([_s(0, 40.0, 44.5), _s(1, 44.6, 46.2)], rough=45.0, pad=10.0, allow_qe=False,
                   tail_pad=0.15, gap_min=0.12, end_extend_max=8.0)
    assert isinstance(pe, Pick)
