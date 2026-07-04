"""Refine-model Whisper singleton (Task 2). WhisperModel is stubbed — no real model loads."""
from __future__ import annotations

import backend.config as config
import backend.pipeline.transcribe as tr


class _FakeModel:
    def __init__(self, name, device=None, compute_type=None, num_workers=1):
        self.name, self.num_workers = name, num_workers


def _install_fake(monkeypatch):
    built = []

    class _FakeWM(_FakeModel):
        def __init__(self, name, **kw):
            super().__init__(name, **kw)
            built.append(name)

    # faster_whisper.WhisperModel is imported INSIDE the getters
    import faster_whisper
    monkeypatch.setattr(faster_whisper, "WhisperModel", _FakeWM)
    monkeypatch.setattr(tr, "_whisper_model", None, raising=False)
    monkeypatch.setattr(tr, "_refine_whisper_model", None, raising=False)
    return built


def test_refine_singleton_builds_refine_model(monkeypatch):
    built = _install_fake(monkeypatch)
    monkeypatch.setattr(config, "REFINE_WHISPER_MODEL", "medium")
    monkeypatch.setattr(config, "WHISPER_MODEL", "small")
    m = tr._get_refine_whisper()
    assert m.name == "medium"
    assert "medium" in built
    assert tr._get_refine_whisper() is m          # cached singleton


def test_refine_reuses_full_singleton_when_models_match(monkeypatch):
    built = _install_fake(monkeypatch)
    monkeypatch.setattr(config, "REFINE_WHISPER_MODEL", "small")
    monkeypatch.setattr(config, "WHISPER_MODEL", "small")
    full = tr._get_whisper()
    refine = tr._get_refine_whisper()
    assert refine is full                          # same object, one load
    assert built.count("small") == 1


def test_refine_singleton_uses_refine_workers(monkeypatch):
    _install_fake(monkeypatch)
    monkeypatch.setattr(config, "REFINE_WHISPER_MODEL", "medium")
    monkeypatch.setattr(config, "WHISPER_MODEL", "small")
    monkeypatch.setattr(config, "REFINE_WORKERS", 4)
    m = tr._get_refine_whisper()
    assert m.num_workers == 4
