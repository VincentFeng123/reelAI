from __future__ import annotations

from backend.app.clip_engine import bridge
from backend.app.ingestion.pipeline import IngestionPipeline, _PlatformRateLimiter
from backend.app.services.reels import ReelService


TRANSCRIPT = {
    "duration": 30.0,
    "segments": [
        {"cue_id": "before", "start": 0.0, "end": 10.0,
         "text": "and everything is great"},
        {"cue_id": "selected", "start": 9.9, "end": 20.0,
         "text": "Now the selected explanation is complete."},
        {"cue_id": "after", "start": 19.9, "end": 30.0,
         "text": "The next unrelated example begins here"},
    ],
}


def test_bridge_preserves_cue_ids() -> None:
    cues = bridge.to_cues(TRANSCRIPT)
    assert [cue.cue_id for cue in cues] == ["before", "selected", "after"]


def test_persist_engine_clip_uses_only_selected_cue_text(monkeypatch) -> None:
    pipeline = IngestionPipeline(
        embedding_service=None,
        rate_limiter=_PlatformRateLimiter(overrides={"yt": (100, 60.0)}),
        serverless_mode=False,
    )
    captured = {}

    def fake_persist(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(pipeline, "_persist_ingest", fake_persist)
    pipeline._persist_engine_clip(
        v={
            "id": "dQw4w9WgXcQ",
            "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "title": "Lesson",
            "channel": "Teacher",
            "duration": 30,
        },
        clip={
            "start": 9.95,
            "end": 20.05,
            "cue_ids": ["selected"],
            "title": "Selected lesson",
        },
        engine_out={"transcript": TRANSCRIPT},
        material_id="material",
        concept_id="concept",
        target_max=55,
    )

    assert captured["chosen"].text == "Now the selected explanation is complete."
    assert captured["snippet"] == "Now the selected explanation is complete."


def test_caption_projection_uses_selected_ids_and_explicit_end() -> None:
    service = ReelService(embedding_service=None, youtube_service=None)
    captions = service._build_caption_cues(
        transcript=TRANSCRIPT["segments"],
        clip_start=9.95,
        clip_end=20.05,
        selected_cue_ids=["selected"],
    )

    assert [cue["text"] for cue in captions] == [
        "Now the selected explanation is complete."
    ]
    assert captions[0]["start"] == 0.0
    assert captions[0]["end"] == 10.05


def test_caption_projection_never_truncates_selected_transcript_text() -> None:
    service = ReelService(embedding_service=None, youtube_service=None)
    selected_text = " ".join(
        [
            "Cells organize molecules into membranes and organelles while preserving the complete teaching claim."
        ]
        * 4
    )
    captions = service._build_caption_cues(
        transcript=[
            {
                "cue_id": "selected",
                "start": 0.0,
                "end": 30.0,
                "text": selected_text,
            }
        ],
        clip_start=0.0,
        clip_end=30.0,
        selected_cue_ids=["selected"],
    )

    assert len(selected_text) > 220
    assert captions == [{"start": 0.0, "end": 30.0, "text": selected_text}]
