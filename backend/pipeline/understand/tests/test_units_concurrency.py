"""Order-invariance of the concurrent unit-extraction pass (offline, mocked llm_json).

PASS A of extract_units fires one llm_json(UnitsLLM) call PER topic over a thread pool;
PASS B assigns unit_ids / clamps boundaries / truncates SERIALLY in original topic order.
These tests PIN that the produced units (unit_ids, sentence_ranges, roles, summaries,
start/end, and their order) are IDENTICAL to the serial (UNDERSTAND_WORKERS=1) build even
when the pool completes topics in the REVERSE of index order — i.e. completion order can
never leak into the output. The knob UNDERSTAND_WORKERS=1 must reproduce the serial path.
"""
from __future__ import annotations

import threading

import backend.config as config_mod
import backend.llm as llm_mod
from backend.adapters.lecture import LectureAdapter
from backend.pipeline.understand.models import ContentMap, ContentNode
from backend.pipeline.understand.units import UnitLLM, UnitsLLM, extract_units

from .conftest import make_sents

_PER = 3          # sentences per topic
_NTOPICS = 4      # → 12 sentences, 8 units total (u0000..u0007)


# ── fixtures ─────────────────────────────────────────────────────────────────
def _cm_multi() -> ContentMap:
    """A content map with _NTOPICS disjoint, contiguous topics named T0..T{k}."""
    nodes = [ContentNode(node_id="video", level="video",
                         sentence_range=(0, _NTOPICS * _PER - 1))]
    for k in range(_NTOPICS):
        a = k * _PER
        nodes.append(ContentNode(node_id=f"ch1.t{k}", level="topic",
                                 title=f"T{k}", sentence_range=(a, a + _PER - 1)))
    return ContentMap(root_id="video", nodes=nodes)


def _units_for(k: int) -> UnitsLLM:
    """Two distinct atomic units for topic k, so every topic's units are identifiable."""
    a = k * _PER
    b = a + _PER - 1
    return UnitsLLM(units=[
        UnitLLM(sentence_start=a, sentence_end=a, role="explanation", summary=f"t{k}.0"),
        UnitLLM(sentence_start=a + 1, sentence_end=b, role="explanation", summary=f"t{k}.1"),
    ])


def _topic_index(user: str) -> int:
    """Recover the topic index k from the built prompt (`TOPIC: T{k}`)."""
    title = user.split("TOPIC: ", 1)[1].split("\n", 1)[0]
    return int(title[1:])


def _sig(units) -> list[tuple]:
    return [(u.unit_id, u.sentence_range, u.role, u.summary, u.start, u.end) for u in units]


def _serial_units(monkeypatch) -> list[tuple]:
    """Baseline: force the exact serial path (UNDERSTAND_WORKERS=1), in-order, no blocking."""
    monkeypatch.setattr(config_mod, "UNDERSTAND_WORKERS", 1)

    def fake(system, user, schema, **kw):
        assert schema is UnitsLLM
        return _units_for(_topic_index(user))

    monkeypatch.setattr(llm_mod, "llm_json", fake)
    return _sig(extract_units(make_sents(_NTOPICS * _PER), _cm_multi(), LectureAdapter(),
                              settings={}))


# ── the pin: reverse completion order must not change the output ─────────────
def test_units_identical_when_pool_completes_topics_in_reverse(monkeypatch):
    serial = _serial_units(monkeypatch)

    # Enough workers that all topics run concurrently, then a chain of events forces
    # STRICT reverse completion: topic k may return only AFTER topic k+1 has returned.
    monkeypatch.setattr(config_mod, "UNDERSTAND_WORKERS", _NTOPICS)
    gate = {k: threading.Event() for k in range(_NTOPICS)}
    completion_order: list[int] = []
    lock = threading.Lock()

    def fake(system, user, schema, **kw):
        assert schema is UnitsLLM
        k = _topic_index(user)
        if k < _NTOPICS - 1:
            assert gate[k + 1].wait(timeout=10), f"topic {k+1} never completed (deadlock?)"
        with lock:
            completion_order.append(k)
        gate[k].set()
        return _units_for(k)

    monkeypatch.setattr(llm_mod, "llm_json", fake)
    parallel = _sig(extract_units(make_sents(_NTOPICS * _PER), _cm_multi(), LectureAdapter(),
                                  settings={}))

    # the pool really did complete topics in reverse index order …
    assert completion_order == list(range(_NTOPICS - 1, -1, -1))
    # … yet the produced units (ids, ranges, roles, summaries, times, order) are IDENTICAL.
    assert parallel == serial
    # concrete anchor: ids are dense & in topic order regardless of completion order.
    assert [u[0] for u in parallel] == [f"u{i:04d}" for i in range(_NTOPICS * 2)]
    assert [u[1] for u in parallel] == [
        (0, 0), (1, 2), (3, 3), (4, 5), (6, 6), (7, 8), (9, 9), (10, 11)]


def test_workers_one_reproduces_serial_path(monkeypatch):
    # WORKERS=1 is the revert switch: same output object-for-object as the multi-worker run.
    serial = _serial_units(monkeypatch)
    assert [s[0] for s in serial] == [f"u{i:04d}" for i in range(_NTOPICS * 2)]
    assert [s[3] for s in serial] == [f"t{k}.{j}" for k in range(_NTOPICS) for j in (0, 1)]
