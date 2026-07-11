"""Order-invariance pins for the perception concurrency lever (config.VISION_WORKERS).

The vision batch-captioning loop (vision.describe_keyframes) and the keyframe-extraction loop
(scenes.detect_and_extract) now run their INDEPENDENT Gemini/ffmpeg work concurrently. These tests
force OUT-OF-ORDER completion (later batches / higher-timestamp frames finish FIRST) and assert the
produced output — VisualEvent ids/ranges/order, and the deduped Scene list + on-disk files — is
byte-identical to the in-order (VISION_WORKERS=1) result. If the id-assignment / dedup post-pass ever
stops being a serial pass over the ORIGINAL index/time order, these fail (see the self-mutation note
in the PR): asserting equality alone isn't enough, so we ALSO assert completion actually reversed.
"""
from __future__ import annotations

import re
import threading
import time

import backend.config as config
from backend.pipeline.understand import scenes as scenes_mod
from backend.pipeline.understand import vision as vision_mod
from backend.pipeline.understand.models import Scene
from backend.pipeline.understand.vision import VisionBatch, VisionItem

N_SCENES = 20  # → 3 batches at VISION_BATCH=8 (8,8,4); indexes 0..19


# ── deterministic per-index caption content (a pure function of the scene index) ─────────────
def _item_for(k: int) -> VisionItem:
    if k % 3 == 0:  # face + empty text → exercises the drop path (must vanish identically)
        return VisionItem(index=k, kind="face", text="", description=f"desc-{k}", confidence=0.5)
    return VisionItem(index=k, kind="slide", text=f"onscreen-{k}",
                      description=f"desc-{k}", confidence=round(0.5 + (k % 4) * 0.1, 2))


def _make_fake_llm(reverse: bool, log: list, lock: threading.Lock):
    def fake_generate_json_mm(system, parts, schema, temperature=0.1):
        idxs = []
        for p in parts:
            if isinstance(p, str):
                m = re.search(r"index=(\d+)", p)
                if m:
                    idxs.append(int(m.group(1)))
        lo = min(idxs) if idxs else 0
        if reverse:
            # HIGH-index batches finish FIRST → completion order is the reverse of submit order
            time.sleep(0.01 * (N_SCENES - lo))
        with lock:
            log.append(lo)
        return VisionBatch(items=[_item_for(k) for k in idxs]).model_dump_json()

    return fake_generate_json_mm


def _run_describe(monkeypatch, tmp_path, *, workers: int, reverse: bool):
    kf = tmp_path / "kf.jpg"
    kf.write_bytes(b"\xff\xd8fake-jpeg")
    scenes = [Scene(index=i, start=float(i), end=float(i + 1),
                    keyframe_time=float(i), keyframe_path=str(kf)) for i in range(N_SCENES)]

    log: list = []
    lock = threading.Lock()
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(config, "VISION_WORKERS", workers)
    # patch on gemini_client — describe_keyframes lazily `from ...gemini_client import` at call time
    import backend.gemini_client as gc
    monkeypatch.setattr(gc, "generate_json_mm", _make_fake_llm(reverse, log, lock))
    monkeypatch.setattr(gc, "text_part", lambda s: s)
    monkeypatch.setattr(gc, "image_part", lambda data: b"IMG")

    events = vision_mod.describe_keyframes(scenes, lambda t: f"near-{t}")
    return events, log


def _dump(events):
    return [e.model_dump() for e in events]


def test_vision_events_identical_under_reversed_batch_completion(tmp_path, monkeypatch):
    serial, serial_log = _run_describe(monkeypatch, tmp_path, workers=1, reverse=False)
    parallel, par_log = _run_describe(monkeypatch, tmp_path, workers=8, reverse=True)

    # the lever actually did something: batches completed OUT of submit order in the parallel run
    assert par_log != sorted(par_log), f"completion was not reversed: {par_log}"
    assert serial_log == sorted(serial_log)  # sanity: serial path is in-order

    # OUTPUT is byte-identical: same event_ids, ranges, kinds, text, order — id is index-derived
    assert _dump(parallel) == _dump(serial)
    # and the drop path fired: k%3==0 (face+empty) dropped, so event_ids are dense ve_0000..
    assert [e.event_id for e in parallel] == [f"ve_{i:04d}" for i in range(len(parallel))]
    assert [e.text for e in parallel] == [f"onscreen-{k}" for k in range(N_SCENES) if k % 3]


# ── scenes: keyframe extraction concurrent, dHash dedup serial in time order ──────────────────
_DHASH_TABLE = {  # disjoint 4-bit blocks (hamming 8 between neighbours) except 3==2 (a near-dup)
    0: 0b1111, 1: 0xF00, 2: 0xF0000, 3: 0xF0000, 4: 0xF000000, 5: 0xF00000000, 6: 0xF0000000000,
}


def _make_fake_extract(reverse: bool, log: list, lock: threading.Lock, dur: float):
    def fake_extract(video_path, t, out):
        out.write_bytes(b"x" * 16)
        if reverse:
            # HIGH-timestamp frames finish FIRST → completion is the reverse of time order
            time.sleep(0.01 * (dur - t) / dur)
        with lock:
            log.append(t)
        return True

    return fake_extract


def _run_extract(monkeypatch, tmp_path, *, workers: int, reverse: bool):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    dur = 100.0
    log: list = []
    lock = threading.Lock()
    monkeypatch.setattr(config, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(config, "VISION_WORKERS", workers)
    monkeypatch.setattr(scenes_mod, "_scene_times", lambda vp, d: [])   # no ffmpeg scene scan
    monkeypatch.setattr(scenes_mod, "_extract_frame", _make_fake_extract(reverse, log, lock, dur))
    monkeypatch.setattr(scenes_mod, "_dhash",
                        lambda path: _DHASH_TABLE[int(re.search(r"kf_(\d+)\.jpg", path).group(1))])
    scenes = scenes_mod.detect_and_extract(str(video), "vid", dur)
    return scenes, log


def test_scene_dedup_identical_under_reversed_extraction_completion(tmp_path, monkeypatch):
    serial, serial_log = _run_extract(monkeypatch, tmp_path, workers=1, reverse=False)
    parallel, par_log = _run_extract(monkeypatch, tmp_path, workers=8, reverse=True)

    assert par_log != sorted(par_log), f"extraction completion was not reversed: {par_log}"
    assert serial_log == sorted(serial_log)

    # dedup ran in TIME order both times → identical scenes (index, spans, times, paths)
    assert [s.model_dump() for s in parallel] == [s.model_dump() for s in serial]
    # frame 3 (t=45, near-dup of frame 2) dropped both times → 6 kept scenes, re-indexed 0..5
    assert [s.index for s in parallel] == list(range(6))
    assert [round(s.keyframe_time, 1) for s in parallel] == [0.0, 15.0, 30.0, 60.0, 75.0, 90.0]
