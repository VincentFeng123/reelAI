import os
import subprocess
import sys
from pathlib import Path


def test_gemini_path_callable_and_embed_url():
    import backend.app.clip_engine.clipper.pipeline.gemini_segment as gs
    import backend.app.clip_engine.clipper.embed as embed

    assert callable(gs.segment_clips)
    assert embed.embed_url("vid123", 10.4, 42.9) == "https://www.youtube.com/embed/vid123?start=10&end=43&rel=0"


def test_gemini_path_imports_without_heavy_deps():
    # Order-independent: prove in a FRESH interpreter that importing the gemini path
    # pulls no heavy deps. Asserting `sys.modules` in-process is unreliable because any
    # earlier test that imported `backend.app.main` / services loads torch first and
    # pollutes the shared module table (that made this a flaky false-failure).
    root = Path(__file__).resolve().parents[3]  # repo root: reelai/reelAI copy 2
    code = (
        "import sys\n"
        "import backend.app.clip_engine.clipper.pipeline.gemini_segment\n"
        "import backend.app.clip_engine.clipper.embed\n"
        "leaked = [m for m in ('torch', 'faster_whisper', 'yt_dlp') if m in sys.modules]\n"
        "assert not leaked, 'heavy import leaked: ' + repr(leaked)\n"
        "print('OK')\n"
    )
    env = {**os.environ, "PYTHONPATH": str(root)}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(root),
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, f"gemini path pulled a heavy dep:\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}"
    assert "OK" in proc.stdout
