"""Optional local OCR of keyframes (default OFF — ``OCR_ENGINE="none"``).

Gemini-vision (``vision.py``) already reads on-screen text *semantically* (equations,
handwriting, code) better than glyph OCR, so this is off by default. When ``OCR_ENGINE`` is
set to ``easyocr`` or ``pytesseract`` it OCRs each keyframe and its text is merged into the
nearest ``VisualEvent`` (appended only when the vision pass missed it). Every path is guarded:
a missing engine → empty list, never a crash.
"""
from __future__ import annotations

from typing import Callable, Optional

from pydantic import BaseModel

from ... import config
from .models import Scene, VisualEvent

ProgressCb = Optional[Callable[[float, str], None]]

_reader = None                                             # cached easyocr.Reader


class OcrBlock(BaseModel):
    index: int
    keyframe_time: float
    text: str = ""


def available(engine: Optional[str] = None) -> bool:
    eng = (engine or config.OCR_ENGINE or "none").lower()
    if eng in ("none", ""):
        return False
    try:
        if eng == "easyocr":
            import easyocr  # noqa: F401
            return True
        if eng == "pytesseract":
            import shutil
            import pytesseract  # noqa: F401
            return shutil.which("tesseract") is not None
    except Exception:
        return False
    return False


def _easyocr_read(path: str) -> str:
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=False)
    return " ".join(_reader.readtext(path, detail=0) or [])


def _pytesseract_read(path: str) -> str:
    import pytesseract
    from PIL import Image
    return pytesseract.image_to_string(Image.open(path)) or ""


def ocr_keyframes(scenes: list[Scene], engine: Optional[str] = None,
                  progress: ProgressCb = None) -> list[OcrBlock]:
    """OCR each keyframe → text blocks (empty list if OCR is off/unavailable)."""
    eng = (engine or config.OCR_ENGINE or "none").lower()
    usable = [s for s in scenes if s.keyframe_path]
    if not available(eng) or not usable:
        return []
    read = _easyocr_read if eng == "easyocr" else _pytesseract_read
    blocks: list[OcrBlock] = []
    total = len(usable)
    for i, s in enumerate(usable):
        try:
            txt = " ".join((read(s.keyframe_path) or "").split()).strip()
        except Exception:
            txt = ""
        if txt:
            blocks.append(OcrBlock(index=s.index, keyframe_time=s.keyframe_time, text=txt))
        if progress:
            progress((i + 1) / total, f"OCR {i + 1}/{total}")
    return blocks


def merge_into_events(events: list[VisualEvent], blocks: list[OcrBlock]) -> None:
    """Append OCR text into the overlapping/nearest ``VisualEvent`` when vision missed it."""
    if not events or not blocks:
        return
    for blk in blocks:
        t = blk.keyframe_time
        target = next((ve for ve in events if ve.start <= t <= ve.end), None)
        if target is None:
            target = min(events, key=lambda ve: abs(((ve.start + ve.end) / 2) - t))
        low = target.text.lower()
        add = blk.text.strip()
        if add and add.lower() not in low:
            target.text = (target.text + " | " + add).strip(" |") if target.text else add
