"""Schema-v4 groundwork: engine field, version bump, cache gating, config block."""
from __future__ import annotations

import json

from backend import config
from backend.pipeline.understand.models import (
    SCHEMA_VERSION, ContentMap, Structure, load_structure, save_structure,
)


def test_schema_version_is_4():
    assert SCHEMA_VERSION == 4


def test_content_map_engine_field_defaults_empty():
    assert ContentMap().engine == ""


def test_engine_field_roundtrips_through_structure_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORK_DIR", tmp_path)
    st = Structure(video_id="vidA", content_map=ContentMap(engine="treeseg"))
    save_structure(st)
    loaded = load_structure("vidA")
    assert loaded is not None
    assert loaded.content_map.engine == "treeseg"


def test_v3_cache_is_invalidated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORK_DIR", tmp_path)
    st = Structure(video_id="vidB")
    save_structure(st)
    p = tmp_path / "vidB" / "structure.json"
    data = json.loads(p.read_text())
    data["schema_version"] = 3
    p.write_text(json.dumps(data))
    assert load_structure("vidB") is None


def test_treeseg_config_block():
    assert config.CONTENT_MAP_ENGINE in ("treeseg", "llm")
    assert config.TREESEG_TARGET_TOPIC_SEC == 120.0
    assert config.TREESEG_MIN_TOPICS == 2
    assert config.TREESEG_MAX_TOPICS == 24
    assert config.TREESEG_MIN_TOPIC_SENTS == 3
    assert config.TREESEG_COHERENCE_FLOOR == 0.0
    assert config.TREESEG_PAUSE_PRIOR == 0.15
    assert config.TREESEG_LABEL_BATCH == 12
    assert "content_map_engine" in config.DEFAULTS and config.DEFAULTS["content_map_engine"] is None
