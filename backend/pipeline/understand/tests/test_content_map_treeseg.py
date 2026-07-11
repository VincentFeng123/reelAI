"""Engine dispatch + treeseg ContentMap assembly + labeling alignment + fallback. Offline."""
from __future__ import annotations

import numpy as np
import pytest

import backend.llm as llm_mod
import backend.pipeline.understand.content_map as cm_mod
from backend.pipeline.understand.content_map import SegLabelLLM, SegLabelsLLM, build_content_map

from .conftest import block_emb, make_sents


@pytest.fixture
def two_block_embedder(monkeypatch):
    calls = {"n": 0}

    def fake_embed(sentences):
        calls["n"] += 1
        return block_emb([len(sentences) // 2, len(sentences) - len(sentences) // 2])

    monkeypatch.setattr(cm_mod, "embed_sentences", fake_embed)
    return calls


@pytest.fixture
def label_llm(monkeypatch):
    def fake_llm_json(system, user, schema, **kw):
        assert schema is SegLabelsLLM
        idxs = [int(t) for t in __import__("re").findall(r"\[(\d+)\]", user)]
        return SegLabelsLLM(labels=[
            SegLabelLLM(index=i, title=f"Label {i}", summary=f"About {i}", keywords=[f"k{i}"])
            for i in sorted(set(idxs))])

    monkeypatch.setattr(llm_mod, "llm_json", fake_llm_json)


def _mk_settings(engine=None):
    return {"content_map_engine": engine}


def test_treeseg_assembly(two_block_embedder, label_llm):
    sents = make_sents(12, sec=20.0)                     # 12×20 s → duration drives K ≥ 2
    cm = build_content_map(sents, _mk_settings())
    assert cm.engine == "treeseg"
    topics = cm.topics()
    assert [t.sentence_range for t in topics] == [(0, 5), (6, 11)]
    assert [t.title for t in topics] == ["Label 0", "Label 1"]
    assert topics[0].summary == "About 0" and topics[0].keywords == ["k0"]
    assert not cm.subtopics()                            # treeseg emits no subtopic nodes
    chapters = cm.chapters()
    assert chapters and all(ch.parent_id == "video" for ch in chapters)
    for t in topics:
        assert t.parent_id in {ch.node_id for ch in chapters}
        assert t.node_id in next(ch for ch in chapters if ch.node_id == t.parent_id).children_ids


def test_label_failure_keeps_partition(two_block_embedder, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("label LLM down")
    monkeypatch.setattr(llm_mod, "llm_json", boom)
    sents = make_sents(12, sec=20.0)
    cm = build_content_map(sents, _mk_settings())
    assert cm.engine == "treeseg"                        # label failure ≠ engine fallback
    topics = cm.topics()
    assert [t.sentence_range for t in topics] == [(0, 5), (6, 11)]
    assert all(t.title for t in topics)                  # deterministic fallback titles


def test_embed_failure_falls_back_to_llm_engine(monkeypatch, capsys):
    def boom(sentences):
        raise RuntimeError("no model")
    monkeypatch.setattr(cm_mod, "embed_sentences", boom)

    def fake_llm_json(system, user, schema, **kw):       # legacy path's topic pass
        from backend.pipeline.understand.content_map import ContentMapLLM, TopicLLM
        assert schema is ContentMapLLM
        return ContentMapLLM(topics=[TopicLLM(title="Legacy", sentence_start=0, sentence_end=11)])
    monkeypatch.setattr(llm_mod, "llm_json", fake_llm_json)

    cm = build_content_map(make_sents(12, sec=30.0), _mk_settings())
    assert cm.engine == "llm-fallback"
    assert cm.topics()[0].title == "Legacy"
    err = capsys.readouterr().err
    assert "treeseg failed" in err and "falling back" in err


def test_llm_engine_when_configured(two_block_embedder, monkeypatch):
    def fake_llm_json(system, user, schema, **kw):
        from backend.pipeline.understand.content_map import ContentMapLLM, TopicLLM
        return ContentMapLLM(topics=[TopicLLM(title="Legacy", sentence_start=0, sentence_end=11)])
    monkeypatch.setattr(llm_mod, "llm_json", fake_llm_json)
    cm = build_content_map(make_sents(12, sec=30.0), _mk_settings(engine="llm"))
    assert cm.engine == "llm"
    assert two_block_embedder["n"] == 0                  # embeddings never touched


def test_empty_sentences_video_only():
    cm = build_content_map([], _mk_settings())
    assert [n.level for n in cm.nodes] == ["video"]


def test_tiny_video_single_topic_no_embeddings(two_block_embedder, label_llm):
    sents = make_sents(4, sec=30.0)                      # n=4 < 2×TREESEG_MIN_TOPIC_SENTS
    cm = build_content_map(sents, _mk_settings())
    assert cm.engine == "treeseg"
    assert [t.sentence_range for t in cm.topics()] == [(0, 3)]
    assert two_block_embedder["n"] == 0                  # early-out before embedding


def test_build_structure_records_content_map_fallback(monkeypatch):
    from backend.adapters import generic
    from backend.adapters.detect import DetectionResult
    from backend.pipeline.understand import build as build_mod
    from backend.pipeline.understand.models import ContentMap, ContentNode

    def fake_cm(sentences, settings, progress=None):
        return ContentMap(nodes=[ContentNode(node_id="video", level="video",
                                             sentence_range=(0, len(sentences) - 1))],
                          engine="llm-fallback")
    monkeypatch.setattr(build_mod, "build_content_map", fake_cm)
    monkeypatch.setattr(build_mod, "extract_units", lambda *a, **kw: [])
    monkeypatch.setattr(build_mod, "build_dependency_graph",
                        lambda units, settings, progress=None: __import__(
                            "backend.pipeline.understand.models",
                            fromlist=["DependencyGraph"]).DependencyGraph())

    sents = make_sents(4)
    st = build_mod.build_structure("vidX", {"title": "t"}, sents,
                                   generic.GenericAdapter(), DetectionResult(), {})
    assert "content_map" in st.degraded


def test_build_structure_no_marker_when_not_fallback(monkeypatch):
    from backend.adapters import generic
    from backend.adapters.detect import DetectionResult
    from backend.pipeline.understand import build as build_mod
    from backend.pipeline.understand.models import ContentMap, ContentNode

    def fake_cm(sentences, settings, progress=None):
        return ContentMap(nodes=[ContentNode(node_id="video", level="video",
                                             sentence_range=(0, len(sentences) - 1))],
                          engine="treeseg")
    monkeypatch.setattr(build_mod, "build_content_map", fake_cm)
    monkeypatch.setattr(build_mod, "extract_units", lambda *a, **kw: [])
    monkeypatch.setattr(build_mod, "build_dependency_graph",
                        lambda units, settings, progress=None: __import__(
                            "backend.pipeline.understand.models",
                            fromlist=["DependencyGraph"]).DependencyGraph())

    sents = make_sents(4)
    st = build_mod.build_structure("vidX", {"title": "t"}, sents,
                                   generic.GenericAdapter(), DetectionResult(), {})
    assert "content_map" not in st.degraded
