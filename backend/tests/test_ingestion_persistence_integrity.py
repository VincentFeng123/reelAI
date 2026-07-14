import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("REELAI_INGEST_SKIP_IMPORT_SWEEP", "1")

from backend.app import db as db_module  # noqa: E402
from backend.app.config import get_settings  # noqa: E402
from backend.app.ingestion import (  # noqa: E402
    INGEST_SENTINEL_CONCEPT_ID,
    INGEST_SENTINEL_MATERIAL_ID,
)
from backend.app.ingestion import persistence as persistence_module  # noqa: E402
from backend.app.ingestion import pipeline as pipeline_module  # noqa: E402
from backend.app.ingestion.errors import InvalidReferenceError  # noqa: E402
from backend.app.ingestion.models import (  # noqa: E402
    IngestMetadata,
    IngestSegment,
    IngestTranscriptCue,
    YouTubeSourceRef,
)


class PersistenceIntegrityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.previous_data_dir = os.environ.get("DATA_DIR")
        os.environ["DATA_DIR"] = self.temp_dir.name
        self.addCleanup(self._restore_environment)
        db_module._db_ready = False
        get_settings.cache_clear()
        db_module.init_db()

    def _restore_environment(self) -> None:
        if self.previous_data_dir is None:
            os.environ.pop("DATA_DIR", None)
        else:
            os.environ["DATA_DIR"] = self.previous_data_dir
        db_module._db_ready = False
        get_settings.cache_clear()

    @staticmethod
    def _seed_identity(conn, material_id: str, concept_id: str) -> None:
        db_module.insert(
            conn,
            "materials",
            {
                "id": material_id,
                "subject_tag": "test",
                "raw_text": "test",
                "source_type": "topic",
                "source_path": None,
                "created_at": db_module.now_iso(),
            },
        )
        db_module.insert(
            conn,
            "concepts",
            {
                "id": concept_id,
                "material_id": material_id,
                "title": "Test concept",
                "keywords_json": "[]",
                "summary": "",
                "embedding_json": None,
                "created_at": db_module.now_iso(),
            },
        )

    def test_scratch_concepts_are_scoped_to_their_material(self) -> None:
        with db_module.get_conn(transactional=True) as conn:
            self._seed_identity(conn, "material-a", "concept-a")
            self._seed_identity(conn, "material-b", "concept-b")
            concept_a = persistence_module.ensure_sentinel_concept(conn, "material-a")
            concept_b = persistence_module.ensure_sentinel_concept(conn, "material-b")

        self.assertNotEqual(concept_a, concept_b)
        self.assertNotEqual(concept_a, INGEST_SENTINEL_CONCEPT_ID)
        self.assertNotEqual(concept_b, INGEST_SENTINEL_CONCEPT_ID)
        with db_module.get_conn() as conn:
            row_a = db_module.fetch_one(
                conn, "SELECT material_id FROM concepts WHERE id = ?", (concept_a,)
            )
            row_b = db_module.fetch_one(
                conn, "SELECT material_id FROM concepts WHERE id = ?", (concept_b,)
            )
        self.assertEqual(row_a["material_id"], "material-a")
        self.assertEqual(row_b["material_id"], "material-b")

    def test_global_sentinel_keeps_legacy_concept_id(self) -> None:
        with db_module.get_conn(transactional=True) as conn:
            material_id = persistence_module.ensure_sentinel_material(conn)
            concept_id = persistence_module.ensure_sentinel_concept(conn, material_id)

        self.assertEqual(material_id, INGEST_SENTINEL_MATERIAL_ID)
        self.assertEqual(concept_id, INGEST_SENTINEL_CONCEPT_ID)

    def test_corrupt_legacy_sentinel_is_not_reused_cross_material(self) -> None:
        with db_module.get_conn(transactional=True) as conn:
            self._seed_identity(conn, "material-a", INGEST_SENTINEL_CONCEPT_ID)
            material_id = persistence_module.ensure_sentinel_material(conn)
            concept_id = persistence_module.ensure_sentinel_concept(conn, material_id)

        self.assertNotEqual(concept_id, INGEST_SENTINEL_CONCEPT_ID)
        with db_module.get_conn() as conn:
            row = db_module.fetch_one(
                conn, "SELECT material_id FROM concepts WHERE id = ?", (concept_id,)
            )
        self.assertEqual(row["material_id"], INGEST_SENTINEL_MATERIAL_ID)

    def test_reference_resolution_rejects_missing_and_cross_material_ids(self) -> None:
        with db_module.get_conn(transactional=True) as conn:
            self._seed_identity(conn, "material-a", "concept-a")
            self._seed_identity(conn, "material-b", "concept-b")

            with self.assertRaises(InvalidReferenceError):
                persistence_module.resolve_material_concept(
                    conn, material_id="missing", concept_id=None
                )
            with self.assertRaises(InvalidReferenceError):
                persistence_module.resolve_material_concept(
                    conn, material_id="material-a", concept_id="missing"
                )
            with self.assertRaises(InvalidReferenceError):
                persistence_module.resolve_material_concept(
                    conn, material_id="material-a", concept_id="concept-b"
                )

    def test_concept_only_reference_uses_its_owning_material(self) -> None:
        with db_module.get_conn(transactional=True) as conn:
            self._seed_identity(conn, "material-a", "concept-a")
            material_id, concept_id = persistence_module.resolve_material_concept(
                conn, material_id=None, concept_id="concept-a"
            )

        self.assertEqual((material_id, concept_id), ("material-a", "concept-a"))

    def test_unexpected_reel_insert_error_propagates(self) -> None:
        with mock.patch.object(
            persistence_module, "insert", side_effect=RuntimeError("database unavailable")
        ):
            with self.assertRaisesRegex(RuntimeError, "database unavailable"):
                persistence_module.upsert_reel_row(
                    conn=object(),
                    reel_id="reel-a",
                    material_id="material-a",
                    concept_id="concept-a",
                    video_id="yt:video-a",
                    video_url="https://www.youtube.com/embed/video-a",
                    t_start=1.0,
                    t_end=2.0,
                    transcript_snippet="snippet",
                    takeaways=[],
                )

    def test_unique_reel_insert_error_returns_false(self) -> None:
        with mock.patch.object(
            persistence_module,
            "insert",
            side_effect=db_module.DatabaseIntegrityError("unique collision"),
        ):
            inserted = persistence_module.upsert_reel_row(
                conn=object(),
                reel_id="reel-a",
                material_id="material-a",
                concept_id="concept-a",
                video_id="yt:video-a",
                video_url="https://www.youtube.com/embed/video-a",
                t_start=1.0,
                t_end=2.0,
                transcript_snippet="snippet",
                takeaways=[],
            )

        self.assertFalse(inserted)

    def test_missing_row_after_unique_collision_raises(self) -> None:
        with db_module.get_conn(transactional=True) as conn:
            self._seed_identity(conn, "material-a", "concept-a")

        pipeline = pipeline_module.IngestionPipeline(
            youtube_service=None,
            embedding_service=None,
            serverless_mode=False,
        )
        adapter_result = YouTubeSourceRef(
            source_id="video-a",
            source_url="https://www.youtube.com/watch?v=video-a",
            playback_url="https://www.youtube.com/embed/video-a",
        )
        metadata = IngestMetadata(
            platform="yt",
            source_id="video-a",
            source_url=adapter_result.source_url,
            playback_url=adapter_result.playback_url,
            title="Video A",
        )

        with (
            mock.patch.object(pipeline_module, "upsert_reel_row", return_value=False),
            mock.patch.object(pipeline_module, "load_existing_reel", return_value=None),
        ):
            with self.assertRaises(db_module.DatabaseIntegrityError):
                pipeline._persist_ingest(
                    adapter_result=adapter_result,
                    metadata=metadata,
                    cues=[],
                    chosen=IngestSegment(t_start=1.0, t_end=2.0, text="snippet"),
                    snippet="snippet",
                    material_id="material-a",
                    concept_id="concept-a",
                    clip_window=(1.0, 2.0),
                    target_max=60,
                )

    def test_verified_retry_promotes_deferred_candidate_without_changing_reel_id(self) -> None:
        with db_module.get_conn(transactional=True) as conn:
            self._seed_identity(conn, "material-a", "concept-a")

        pipeline = pipeline_module.IngestionPipeline(
            youtube_service=None,
            embedding_service=None,
            serverless_mode=False,
        )
        adapter_result = YouTubeSourceRef(
            source_id="video-a",
            source_url="https://www.youtube.com/watch?v=video-a",
            playback_url="https://www.youtube.com/embed/video-a",
        )
        metadata = IngestMetadata(
            platform="yt",
            source_id="video-a",
            source_url=adapter_result.source_url,
            playback_url=adapter_result.playback_url,
            title="Video A",
        )
        candidate_id = "video-a::candidate-1"
        deferred = pipeline._persist_ingest(
            adapter_result=adapter_result,
            metadata=metadata,
            cues=[],
            chosen=IngestSegment(t_start=10.0, t_end=20.0, text="snippet"),
            snippet="rough snippet",
            material_id="material-a",
            concept_id="concept-a",
            clip_window=(10.0, 20.0),
            target_max=60,
            generation_id="generation-a",
            clip_details={
                "cue_ids": ["cue-1"],
                "search_context": {
                    "selection_candidate_id": candidate_id,
                    "surface_eligible": False,
                    "boundary_status": "unavailable",
                },
            },
        )
        verified = pipeline._persist_ingest(
            adapter_result=adapter_result,
            metadata=metadata,
            cues=[],
            chosen=IngestSegment(t_start=9.9, t_end=20.2, text="snippet"),
            snippet="verified snippet",
            material_id="material-a",
            concept_id="concept-a",
            clip_window=(9.9, 20.2),
            target_max=60,
            generation_id="generation-a",
            clip_details={
                "cue_ids": ["cue-1"],
                "search_context": {
                    "selection_candidate_id": candidate_id,
                    "surface_eligible": True,
                    "boundary_status": "verified",
                },
            },
        )

        self.assertEqual(verified.reel_id, deferred.reel_id)
        self.assertEqual((verified.t_start, verified.t_end), (9.9, 20.2))
        with db_module.get_conn() as conn:
            rows = db_module.fetch_all(
                conn,
                "SELECT id, video_url, t_start, t_end, transcript_snippet, "
                "search_context_json FROM reels WHERE generation_id = ?",
                ("generation-a",),
            )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], deferred.reel_id)
        self.assertEqual((rows[0]["t_start"], rows[0]["t_end"]), (9.9, 20.2))
        self.assertIn("start=9&end=21", rows[0]["video_url"])
        self.assertEqual(rows[0]["transcript_snippet"], "verified snippet")
        context = json.loads(rows[0]["search_context_json"])
        self.assertTrue(context["surface_eligible"])
        self.assertEqual(context["boundary_status"], "verified")

    def test_persisted_candidate_uses_projected_selection_caption_snapshot(self) -> None:
        with db_module.get_conn(transactional=True) as conn:
            self._seed_identity(conn, "material-a", "concept-a")

        pipeline = pipeline_module.IngestionPipeline(
            youtube_service=None,
            embedding_service=None,
            serverless_mode=False,
        )
        source_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        adapter_result = YouTubeSourceRef(
            source_id="dQw4w9WgXcQ",
            source_url=source_url,
            playback_url="https://www.youtube.com/embed/dQw4w9WgXcQ",
        )
        metadata = IngestMetadata(
            platform="yt",
            source_id="dQw4w9WgXcQ",
            source_url=source_url,
            playback_url=adapter_result.playback_url,
            title="Photosynthesis lesson",
            duration_sec=20.0,
        )
        projected_text = (
            "Photosynthesis converts light energy into chemical energy."
        )

        reel = pipeline._persist_ingest(
            adapter_result=adapter_result,
            metadata=metadata,
            cues=[
                IngestTranscriptCue(
                    cue_id="cue-1",
                    start=0.0,
                    end=20.0,
                    text=(
                        "Welcome back. Photosynthesis converts light energy into "
                        "chemical energy. Thanks for watching."
                    ),
                )
            ],
            chosen=IngestSegment(
                t_start=5.25,
                t_end=15.75,
                text=projected_text,
            ),
            snippet=(
                "Welcome back. Photosynthesis converts light energy into chemical "
                "energy. Thanks for watching."
            ),
            material_id="material-a",
            concept_id="concept-a",
            clip_window=(5.25, 15.75),
            target_max=0,
            generation_id="generation-a",
            clip_details={
                "cue_ids": ["cue-1"],
                "search_context": {
                    "selection_contract_version": "quality_silence_v6",
                    "selection_caption_cues": [
                        {
                            "cue_id": "cue-1",
                            "start": 5.25,
                            "end": 15.75,
                            "text": projected_text,
                            "lang": "en",
                        }
                    ],
                    "surface_eligible": True,
                    "boundary_status": "verified",
                },
            },
        )

        self.assertEqual(reel.transcript_snippet, projected_text)
        self.assertEqual(
            [cue.model_dump() for cue in reel.captions],
            [{"start": 0.0, "end": 10.5, "text": projected_text}],
        )
        with db_module.get_conn() as conn:
            row = db_module.fetch_one(
                conn,
                "SELECT transcript_snippet FROM reels WHERE id = ?",
                (reel.reel_id,),
            )
        self.assertEqual(row["transcript_snippet"], projected_text)


if __name__ == "__main__":
    unittest.main()
