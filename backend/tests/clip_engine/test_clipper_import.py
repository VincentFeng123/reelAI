import sys


def test_gemini_path_imports_without_torch():
    # Importing the gemini path must not pull torch / faster_whisper / yt_dlp.
    import backend.app.clip_engine.clipper.pipeline.gemini_segment as gs
    import backend.app.clip_engine.clipper.embed as embed
    assert callable(gs.segment_clips)
    assert embed.embed_url("vid123", 10.4, 42.9) == "https://www.youtube.com/embed/vid123?start=10&end=43&rel=0"
    assert "torch" not in sys.modules
    assert "faster_whisper" not in sys.modules
