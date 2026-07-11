"""Describe keyframes with Gemini — the PRIMARY visual pass (zero local deps).

For each de-duplicated keyframe we send the image plus its timestamp and the nearby
transcript to Gemini, which classifies what is on screen and transcribes any on-screen
text / equations / code VERBATIM. Talking-head frames with nothing on screen are dropped
so ``VisualEvent``s stay meaningful. Vision is Gemini-only (Groq has no image input here);
if ``GEMINI_API_KEY`` is unset it degrades to an empty list and the caller records ``"vision"``.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from pydantic import BaseModel, Field

from ... import config
from .models import Scene, VisualEvent

ProgressCb = Optional[Callable[[float, str], None]]

# frames that carry no clip-relevant on-screen content — drop when text is empty
_EMPTY_KINDS = frozenset({"face", "action", "other"})


class VisionItem(BaseModel):
    index: int
    kind: str = "other"          # slide|code|diagram|equation|chart|demo|text_overlay|face|action|other
    text: str = ""               # verbatim on-screen text / equations / code
    description: str = ""        # one-line semantic caption
    confidence: float = 0.5


class VisionBatch(BaseModel):
    items: list[VisionItem] = Field(default_factory=list)


VISION_SYSTEM = (
    "You are a visual analyst for a video. For each numbered keyframe (with its timestamp and the "
    "nearby transcript), classify what is ON SCREEN and extract any on-screen text, equations, or "
    "code VERBATIM. Kinds: slide, code, diagram, equation, chart, demo, text_overlay, face, action, "
    "other. Prefer the most specific kind (an equation on a whiteboard is 'equation', not 'slide'). "
    "Transcribe math faithfully (use plain symbols like Δv/Δt, x^2, sqrt). If the frame is just a "
    "talking head or scenery with nothing legible on screen, use kind='face'/'action' and text=''. "
    "Return exactly one item per keyframe index. Structured output only."
)


def available() -> bool:
    return bool(config.GEMINI_API_KEY)


def _caption_batch(batch: list[Scene], nearby_text_fn: Callable[[float], str],
                   generate_json_mm, image_part, text_part) -> tuple[list[Scene], dict, bool]:
    """The INDEPENDENT per-batch work (image read + one Gemini call + parse) run on a worker
    thread. Returns ``(included_scenes, {scene.index: VisionItem}, ok)`` tagged so the caller
    can reassemble deterministically. ``ok`` is True only when the LLM call succeeded — it
    reproduces the serial ``ok_batches`` accounting (a batch with no readable frame ⇒ not ok,
    no call made). Assigns NO event_id and mutates no shared state (order-neutral by design)."""
    parts = [text_part("Analyze these keyframes:")]
    included: list[Scene] = []
    for s in batch:
        try:
            with open(s.keyframe_path, "rb") as f:
                data = f.read()
        except OSError:
            continue
        near = (nearby_text_fn(s.keyframe_time) or "")[:240]
        parts.append(text_part(f"[keyframe index={s.index} t={s.keyframe_time:.1f}s] transcript: {near}"))
        parts.append(image_part(data))
        included.append(s)
    if not included:
        return included, {}, False
    try:
        raw = generate_json_mm(VISION_SYSTEM, parts, VisionBatch, temperature=0.1)
        vb = VisionBatch.model_validate_json(raw)
        ok = True
    except Exception:
        vb = VisionBatch()
        ok = False
    return included, {it.index: it for it in vb.items}, ok


def describe_keyframes(scenes: list[Scene], nearby_text_fn: Callable[[float], str],
                       progress: ProgressCb = None) -> list[VisualEvent]:
    """Caption keyframes in batches → ``VisualEvent``s (empty list if unavailable/failed).

    Latency lever: the per-batch Gemini captioning calls are INDEPENDENT and run concurrently
    over a ``ThreadPoolExecutor`` (``config.VISION_WORKERS``; =1 == the exact serial path). Every
    ordering-sensitive step is a SERIAL post-pass below: results are re-sorted by scene index and
    ``event_id`` is assigned there, so ids/order/contents are IDENTICAL regardless of which batch
    finishes first. ``event_id`` is cosmetic — ``_link_visual`` matches by time+text, not id."""
    usable = [s for s in scenes if s.keyframe_path]
    if not available() or not usable:
        return []
    from ...gemini_client import generate_json_mm, image_part, text_part

    batch_size = max(1, int(config.VISION_BATCH))
    batches = [usable[i:i + batch_size] for i in range(0, len(usable), batch_size)]

    # threads do the independent network/image work; results are tagged by original batch index.
    workers = max(1, min(int(config.VISION_WORKERS), len(batches)))
    results: list[Optional[tuple[list[Scene], dict, bool]]] = [None] * len(batches)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_caption_batch, batch, nearby_text_fn,
                            generate_json_mm, image_part, text_part): bi
                for bi, batch in enumerate(batches)}
        done = 0
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()
            done += 1
            if progress:
                progress(done / len(batches), f"Vision {done}/{len(batches)}")

    ok_batches = sum(1 for r in results if r and r[2])

    # SERIAL post-pass — flatten to (scene, item) pairs and SORT by scene index BEFORE assigning
    # event_id, so id/order/content are a pure function of the scenes, never of completion order.
    pairs: list[tuple[Scene, VisionItem]] = []
    for included, by_idx, _ok in results:              # results is index-ordered; we sort anyway
        for s in included:
            it = by_idx.get(s.index)
            if not it:
                continue
            pairs.append((s, it))
    pairs.sort(key=lambda p: (p[0].index, p[0].keyframe_time))

    events: list[VisualEvent] = []
    for s, it in pairs:
        kind = (it.kind or "other").strip().lower()
        text = (it.text or "").strip()
        if not text and kind in _EMPTY_KINDS:
            continue                                       # nothing clip-relevant on screen
        events.append(VisualEvent(
            event_id=f"ve_{len(events):04d}", start=s.start, end=s.end,
            kind=kind, text=text, description=(it.description or "").strip(),
            confidence=float(it.confidence or 0.5),
        ))

    # every batch call failed → a real failure (transient API/outage), not a visual-free video.
    # Signal it so the caller records "vision" in degraded and can retry on a later run; an empty
    # result with at least one successful call is a legitimate "nothing on screen" and is NOT degraded.
    if batches and ok_batches == 0:
        raise RuntimeError("vision: all keyframe batches failed")
    return events
