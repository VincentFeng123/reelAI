"""W25-A structure integrity & freshness: build provenance stamping, the fingerprint
roundtrip through the cache, staleness classification against the LIVE sentence list,
and the explicit allow_stale (freeze) override. All offline — build_structure's LLM
stages are monkeypatched; everything else is pure model/cache logic."""
from __future__ import annotations

import json

from backend import config
from backend.adapters.detect import DetectionResult
from backend.pipeline.understand import build as build_mod
from backend.pipeline.understand.models import (
    ContentMap, DependencyGraph, Structure, Unit, load_structure, save_structure,
    sentence_fingerprint, structure_is_stale,
)

from .conftest import make_sents


def _fresh_structure(sents, video_id="vidF") -> Structure:
    """A Structure carrying valid W25-A provenance for `sents` (what build_structure stamps)."""
    return Structure(video_id=video_id,
                     n_sentences=len(sents),
                     sentence_fingerprint=sentence_fingerprint(sents),
                     prompt_version=config.UNDERSTANDING_PROMPT_VERSION,
                     built_at="2026-07-02T00:00:00+00:00")


# ── fingerprint function ──────────────────────────────────────────────────────
def test_fingerprint_is_deterministic_and_text_sensitive():
    a = make_sents(5)
    assert sentence_fingerprint(a) == sentence_fingerprint(make_sents(5))
    changed = make_sents(5, texts=["sentence number 0", "sentence number 1", "DIFFERENT",
                                   "sentence number 3", "sentence number 4"])
    assert sentence_fingerprint(a) != sentence_fingerprint(changed)


def test_fingerprint_is_boundary_sensitive():
    # ("ab","c") vs ("a","bc"): same concatenated text, different segmentation — the
    # record separator must keep them distinct (indices point at different sentences).
    ab_c = make_sents(2, texts=["ab", "c"])
    a_bc = make_sents(2, texts=["a", "bc"])
    assert sentence_fingerprint(ab_c) != sentence_fingerprint(a_bc)


# ── build_structure stamps provenance ────────────────────────────────────────
def test_build_structure_populates_provenance(monkeypatch):
    sents = make_sents(7)
    unit = Unit(unit_id="u0000", start=0.0, end=2.0, sentence_range=(0, 6))
    monkeypatch.setattr(build_mod, "build_content_map",
                        lambda sentences, settings, cb: ContentMap(engine="treeseg"))
    monkeypatch.setattr(build_mod, "extract_units",
                        lambda sentences, cm, adapter, settings, cb, perception: [unit])
    monkeypatch.setattr(build_mod, "build_dependency_graph",
                        lambda units, settings, cb: DependencyGraph())
    st = build_mod.build_structure("vidP", {"title": "t"}, sents, adapter=None,
                                   detection=DetectionResult(), settings={})
    assert st.n_sentences == 7
    assert st.sentence_fingerprint == sentence_fingerprint(sents)
    assert st.prompt_version == config.UNDERSTANDING_PROMPT_VERSION
    assert st.built_at                            # ISO stamp present (informational)
    assert structure_is_stale(st, sents) is None  # fresh against its own sentence list


# ── fingerprint roundtrip through the cache ───────────────────────────────────
def test_fingerprint_roundtrip_fresh_cache_loads(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORK_DIR", tmp_path)
    sents = make_sents(6)
    save_structure(_fresh_structure(sents))
    loaded = load_structure("vidF", sents)
    assert loaded is not None
    assert loaded.sentence_fingerprint == sentence_fingerprint(sents)
    assert loaded.n_sentences == 6


def test_n_sentences_mismatch_is_stale(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORK_DIR", tmp_path)
    save_structure(_fresh_structure(make_sents(6)))
    assert load_structure("vidF", make_sents(7)) is None   # different sentence universe


def test_fingerprint_mismatch_same_count_is_stale(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORK_DIR", tmp_path)
    save_structure(_fresh_structure(make_sents(6)))
    other = make_sents(6, texts=[f"other text {i}" for i in range(6)])
    assert load_structure("vidF", other) is None


def test_prompt_version_mismatch_is_stale(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORK_DIR", tmp_path)
    sents = make_sents(6)
    st = _fresh_structure(sents)
    st.prompt_version = "understand-v0"           # schema-compatible but old prompts
    save_structure(st)
    assert load_structure("vidF", sents) is None


def test_old_cache_missing_provenance_is_stale(tmp_path, monkeypatch):
    # a pre-W25-A cache validates fine (pydantic-additive defaults) but 'unknown' must
    # never pass for 'fresh' — this is qP's poisoned-cache state RIGHT NOW.
    monkeypatch.setattr(config, "WORK_DIR", tmp_path)
    save_structure(Structure(video_id="vidO"))    # defaults: n_sentences=0, fingerprint=""
    p = tmp_path / "vidO" / "structure.json"
    data = json.loads(p.read_text())
    for k in ("n_sentences", "sentence_fingerprint", "prompt_version", "built_at"):
        data.pop(k, None)                         # simulate the field-less on-disk shape
    p.write_text(json.dumps(data))
    sents = make_sents(6)
    assert load_structure("vidO", sents) is None
    reloaded = load_structure("vidO")             # sentences=None → no freshness gate
    assert reloaded is not None
    assert "missing build provenance" in structure_is_stale(reloaded, sents)


def test_allow_stale_override_warns_and_loads(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "WORK_DIR", tmp_path)
    save_structure(_fresh_structure(make_sents(6)))
    live = make_sents(9)                          # stale: 6 != 9
    st = load_structure("vidF", live, allow_stale=True)
    assert st is not None                         # the explicit override holds the cache…
    err = capsys.readouterr().err
    assert "STALE" in err and "vidF" in err       # …but LOUDLY, on stderr
    assert "n_sentences 6 != live 9" in err


def test_fresh_load_does_not_warn(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "WORK_DIR", tmp_path)
    sents = make_sents(6)
    save_structure(_fresh_structure(sents))
    assert load_structure("vidF", sents, allow_stale=True) is not None
    assert "STALE" not in capsys.readouterr().err


def test_no_sentences_keeps_legacy_schema_only_gate(tmp_path, monkeypatch):
    # callers that cannot supply a live index (sentences=None) keep the old behavior:
    # schema-version gate only — freshness needs something to be fresh AGAINST.
    monkeypatch.setattr(config, "WORK_DIR", tmp_path)
    save_structure(Structure(video_id="vidL"))
    assert load_structure("vidL") is not None
