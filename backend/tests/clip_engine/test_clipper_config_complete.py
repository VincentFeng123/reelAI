"""Completeness guard: every config.<CONSTANT> referenced in the vendored
gemini-path modules must be present on the shim (clipper/config.py).

Protects against the class of bug where a constant is read at call time
but was never added to the shim — which would raise AttributeError on a
real segment_clips → llm_json → gemini_client call.
"""
import re
from pathlib import Path

from backend.app.clip_engine.clipper import config as shim

_CLIPPER_ROOT = (
    Path(__file__).resolve().parent.parent.parent / "app" / "clip_engine" / "clipper"
)

_SCAN_MODULES = [
    _CLIPPER_ROOT / "llm.py",
    _CLIPPER_ROOT / "gemini_client.py",
    _CLIPPER_ROOT / "embed.py",
    _CLIPPER_ROOT / "pipeline" / "gemini_segment.py",
    _CLIPPER_ROOT / "pipeline" / "transcribe.py",
    _CLIPPER_ROOT / "supadata_client.py",
]

_CONFIG_REF = re.compile(r"\bconfig\.([A-Z][A-Z0-9_]+)\b")


def _collect_referenced_names() -> set[str]:
    names: set[str] = set()
    for path in _SCAN_MODULES:
        src = path.read_text(encoding="utf-8")
        names.update(_CONFIG_REF.findall(src))
    return names


def test_shim_exposes_all_referenced_constants():
    """Every config.NAME reference in the gemini-path modules must be on the shim."""
    missing = [name for name in sorted(_collect_referenced_names()) if not hasattr(shim, name)]
    assert not missing, (
        f"clipper/config.py is missing {len(missing)} constant(s) referenced in "
        f"the gemini-path modules: {missing}"
    )
