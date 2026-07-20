import json
import os
import sys
import tempfile
import unittest
import uuid
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
from backend.pipeline import gemini_segment  # noqa: E402


def _strict_boundary_context(
    candidate_id: str,
    *,
    start: float,
    end: float,
    surface: bool,
) -> dict:
    return {
        "selection_candidate_id": candidate_id,
        "surface_eligible": surface,
        "selection_contract_version": "quality_silence_v38",
        "speech_corridor_verified": True,
        "boundary_status": "verified",
        "boundary_diagnostics": {
            "acoustic_verified": True,
            "final_range": [start, end],
            "acoustic": {
                "threshold_dbfs": -38.0,
                "start_quiet": [max(0.0, start - 0.1), start + 0.1],
                "end_quiet": [max(0.0, end - 0.1), end + 0.1],
            },
        },
    }


def _transcript_boundary_context(
    candidate_id: str,
    *,
    start: float,
    end: float,
    surface: bool,
) -> dict:
    return {
        "selection_candidate_id": candidate_id,
        "surface_eligible": surface,
        "selection_contract_version": "quality_silence_v38",
        "speech_corridor_verified": True,
        "boundary_status": "context_aligned",
        "selection_caption_cues": [
            {"cue_id": "cue-1", "start": start, "end": end, "text": "Teaching"}
        ],
        "boundary_diagnostics": {
            "method": "transcript_context",
            "context_aligned": True,
            "acoustic_verified": False,
            "transcript": {
                "context_aligned": True,
                "stage": "transcript",
                "reason": "complete_discourse_boundary",
                "required_speech_range": [start, end],
                "semantic_range": [max(0.0, start - 1.0), end + 1.0],
                "final_range": [start, end],
            },
        },
    }


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

    def _boundary_persistence_fixture(self):
        with db_module.get_conn(transactional=True) as conn:
            self._seed_identity(conn, "material-a", "concept-a")
        pipeline = pipeline_module.IngestionPipeline(
            youtube_service=None,
            embedding_service=None,
            serverless_mode=False,
        )
        adapter = YouTubeSourceRef(
            source_id="video-a",
            source_url="https://www.youtube.com/watch?v=video-a",
            playback_url="https://www.youtube.com/embed/video-a",
        )
        metadata = IngestMetadata(
            platform="yt",
            source_id="video-a",
            source_url=adapter.source_url,
            playback_url=adapter.playback_url,
            title="Video A",
            duration_sec=30.0,
        )
        return pipeline, adapter, metadata

    def _persist_boundary_candidate(
        self,
        *,
        pipeline,
        adapter,
        metadata,
        start: float,
        end: float,
        context: dict,
        cue_id: str = "cue-1",
    ):
        return pipeline._persist_ingest(
            adapter_result=adapter,
            metadata=metadata,
            cues=[
                IngestTranscriptCue(
                    cue_id=cue_id,
                    start=start,
                    end=end,
                    text="Teaching",
                )
            ],
            chosen=IngestSegment(t_start=start, t_end=end, text="Teaching"),
            snippet="Teaching",
            material_id="material-a",
            concept_id="concept-a",
            clip_window=(start, end),
            target_max=0,
            generation_id="generation-a",
            clip_details={
                "cue_ids": [cue_id],
                "selection_candidate_id": context["selection_candidate_id"],
                "search_context": context,
            },
        )

    def test_intent_role_and_coverage_survive_persistence(self) -> None:
        pipeline, adapter, metadata = self._boundary_persistence_fixture()
        context = _strict_boundary_context(
            "video-a::supporting",
            start=10.0,
            end=20.0,
            surface=True,
        )
        context.update({
            "intent_role": "supporting",
            "intent_coverage": 0.5,
        })

        reel = self._persist_boundary_candidate(
            pipeline=pipeline,
            adapter=adapter,
            metadata=metadata,
            start=10.0,
            end=20.0,
            context=context,
        )

        self.assertEqual(reel.selection_intent_role, "supporting")
        self.assertEqual(reel.selection_intent_coverage, 0.5)
        with db_module.get_conn() as conn:
            row = db_module.fetch_one(
                conn,
                "SELECT search_context_json FROM reels WHERE id = ?",
                (reel.reel_id,),
            )
        stored = json.loads(row["search_context_json"])
        self.assertEqual(stored["intent_role"], "supporting")
        self.assertEqual(stored["intent_coverage"], 0.5)

    def test_gemini_clip_concept_is_normalized_reused_and_persisted(self) -> None:
        pipeline, adapter, metadata = self._boundary_persistence_fixture()
        first_clip = gemini_segment._public_clips(
            [{"facet": "  Net   Force—Acceleration  ", "_private": "discard"}]
        )[0]
        second_clip = gemini_segment._public_clips(
            [{"facet": "net-force / acceleration", "_private": "discard"}]
        )[0]

        first_context = _strict_boundary_context(
            "video-a::net-force-1",
            start=2.0,
            end=8.0,
            surface=True,
        )
        first = pipeline._persist_ingest(
            adapter_result=adapter,
            metadata=metadata,
            cues=[
                IngestTranscriptCue(
                    cue_id="cue-1",
                    start=2.0,
                    end=8.0,
                    text="Net force produces acceleration.",
                )
            ],
            chosen=IngestSegment(
                t_start=2.0,
                t_end=8.0,
                text="Net force produces acceleration.",
            ),
            snippet="Net force produces acceleration.",
            material_id="material-a",
            concept_id="concept-a",
            clip_window=(2.0, 8.0),
            target_max=0,
            generation_id="generation-a",
            clip_title="Net force and acceleration",
            clip_details={
                **first_clip,
                "cue_ids": ["cue-1"],
                "search_context": first_context,
            },
        )

        second_context = _strict_boundary_context(
            "video-a::net-force-2",
            start=10.0,
            end=16.0,
            surface=True,
        )
        second = pipeline._persist_ingest(
            adapter_result=adapter,
            metadata=metadata,
            cues=[
                IngestTranscriptCue(
                    cue_id="cue-2",
                    start=10.0,
                    end=16.0,
                    text="Acceleration follows the net force.",
                )
            ],
            chosen=IngestSegment(
                t_start=10.0,
                t_end=16.0,
                text="Acceleration follows the net force.",
            ),
            snippet="Acceleration follows the net force.",
            material_id="material-a",
            concept_id="concept-a",
            clip_window=(10.0, 16.0),
            target_max=0,
            generation_id="generation-a",
            clip_title="Acceleration from net force",
            clip_details={
                **second_clip,
                "cue_ids": ["cue-2"],
                "search_context": second_context,
            },
        )

        self.assertEqual(first_clip["concept"], "Net Force—Acceleration")
        self.assertEqual(second_clip["concept"], "net-force / acceleration")
        self.assertEqual(first.concept_id, second.concept_id)
        self.assertNotEqual(first.concept_id, "concept-a")
        self.assertEqual(
            first.concept_id,
            str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    "reelai:clip-concept:material-a:net force acceleration",
                )
            ),
        )
        self.assertEqual(first.concept_title, "Net Force—Acceleration")
        self.assertEqual(second.concept_title, "Net Force—Acceleration")

        with db_module.get_conn() as conn:
            concept_rows = db_module.fetch_all(
                conn,
                "SELECT id, title FROM concepts WHERE material_id = ? ORDER BY id",
                ("material-a",),
            )
            reel_rows = db_module.fetch_all(
                conn,
                "SELECT concept_id, search_context_json FROM reels "
                "WHERE id IN (?, ?) ORDER BY t_start",
                (first.reel_id, second.reel_id),
            )

        self.assertCountEqual(
            [row["title"] for row in concept_rows],
            ["Net Force—Acceleration", "Test concept"],
        )
        self.assertEqual(
            [row["concept_id"] for row in reel_rows],
            [first.concept_id, first.concept_id],
        )
        first_provenance = json.loads(reel_rows[0]["search_context_json"])
        second_provenance = json.loads(reel_rows[1]["search_context_json"])
        for provenance, raw_concept in (
            (first_provenance, "Net Force—Acceleration"),
            (second_provenance, "net-force / acceleration"),
        ):
            self.assertEqual(provenance["acquisition_concept_id"], "concept-a")
            self.assertEqual(provenance["acquisition_concept_title"], "Test concept")
            self.assertEqual(provenance["clip_concept_raw"], raw_concept)
            self.assertEqual(provenance["clip_concept_key"], "net force acceleration")
            self.assertEqual(provenance["clip_concept_id"], first.concept_id)
            self.assertEqual(
                provenance["clip_concept_title"], "Net Force—Acceleration"
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
                    "selection_contract_version": "quality_silence_v38",
                    "speech_corridor_verified": True,
                    "boundary_status": "verified",
                    "boundary_diagnostics": {
                        "acoustic_verified": True,
                        "acoustic": {"threshold_dbfs": -38.0},
                    },
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

    def test_transcript_retry_never_downgrades_verified_boundary_or_return_value(self) -> None:
        pipeline, adapter, metadata = self._boundary_persistence_fixture()
        candidate_id = "video-a::monotonic-verified"
        verified = self._persist_boundary_candidate(
            pipeline=pipeline,
            adapter=adapter,
            metadata=metadata,
            start=9.8,
            end=20.2,
            context=_strict_boundary_context(
                candidate_id, start=9.8, end=20.2, surface=True
            ),
            cue_id="cue-verified",
        )
        retried = self._persist_boundary_candidate(
            pipeline=pipeline,
            adapter=adapter,
            metadata=metadata,
            start=10.0,
            end=20.0,
            context=_transcript_boundary_context(
                candidate_id,
                start=10.0,
                end=20.0,
                surface=True,
            ),
            cue_id="cue-retry",
        )

        self.assertEqual(retried.reel_id, verified.reel_id)
        self.assertEqual((retried.t_start, retried.t_end), (9.8, 20.2))
        self.assertEqual(retried.selected_cue_ids, ["cue-verified"])
        with db_module.get_conn() as conn:
            row = db_module.fetch_one(
                conn,
                "SELECT t_start, t_end, selected_cue_ids_json, search_context_json "
                "FROM reels WHERE id = ?",
                (verified.reel_id,),
            )
        self.assertEqual((row["t_start"], row["t_end"]), (9.8, 20.2))
        self.assertEqual(json.loads(row["selected_cue_ids_json"]), ["cue-verified"])
        self.assertEqual(
            json.loads(row["search_context_json"])["boundary_status"],
            "verified",
        )

    def test_verified_retry_promotes_transcript_boundary_even_when_level_deferred(self) -> None:
        pipeline, adapter, metadata = self._boundary_persistence_fixture()
        candidate_id = "video-a::monotonic-promotion"
        transcript = self._persist_boundary_candidate(
            pipeline=pipeline,
            adapter=adapter,
            metadata=metadata,
            start=10.0,
            end=20.0,
            context=_transcript_boundary_context(
                candidate_id,
                start=10.0,
                end=20.0,
                surface=True,
            ),
        )
        promoted = self._persist_boundary_candidate(
            pipeline=pipeline,
            adapter=adapter,
            metadata=metadata,
            start=9.8,
            end=20.2,
            context=_strict_boundary_context(
                candidate_id, start=9.8, end=20.2, surface=False
            ),
        )

        self.assertEqual(promoted.reel_id, transcript.reel_id)
        self.assertEqual((promoted.t_start, promoted.t_end), (9.8, 20.2))
        with db_module.get_conn() as conn:
            row = db_module.fetch_one(
                conn,
                "SELECT t_start, t_end, search_context_json FROM reels WHERE id = ?",
                (transcript.reel_id,),
            )
        context = json.loads(row["search_context_json"])
        self.assertEqual(context["boundary_status"], "verified")
        self.assertFalse(context["surface_eligible"])

    def test_unavailable_retry_never_downgrades_transcript_boundary(self) -> None:
        pipeline, adapter, metadata = self._boundary_persistence_fixture()
        candidate_id = "video-a::monotonic-transcript"
        transcript = self._persist_boundary_candidate(
            pipeline=pipeline,
            adapter=adapter,
            metadata=metadata,
            start=10.0,
            end=20.0,
            context=_transcript_boundary_context(
                candidate_id,
                start=10.0,
                end=20.0,
                surface=False,
            ),
        )
        retried = self._persist_boundary_candidate(
            pipeline=pipeline,
            adapter=adapter,
            metadata=metadata,
            start=11.0,
            end=19.0,
            context={
                "selection_candidate_id": candidate_id,
                "surface_eligible": True,
                "boundary_status": "unavailable",
            },
        )

        self.assertEqual(retried.reel_id, transcript.reel_id)
        self.assertEqual((retried.t_start, retried.t_end), (10.0, 20.0))
        with db_module.get_conn() as conn:
            row = db_module.fetch_one(
                conn,
                "SELECT t_start, t_end, search_context_json FROM reels WHERE id = ?",
                (transcript.reel_id,),
            )
        context = json.loads(row["search_context_json"])
        self.assertEqual(context["boundary_status"], "context_aligned")
        self.assertFalse(context["surface_eligible"])

    def test_current_transcript_retry_replaces_stale_strict_contract(self) -> None:
        pipeline, adapter, metadata = self._boundary_persistence_fixture()
        candidate_id = "video-a::stale-strict"
        stale_context = _strict_boundary_context(
            candidate_id, start=9.8, end=20.2, surface=True
        )
        stale_context["selection_contract_version"] = "quality_silence_v24"
        stale = self._persist_boundary_candidate(
            pipeline=pipeline,
            adapter=adapter,
            metadata=metadata,
            start=9.8,
            end=20.2,
            context=stale_context,
        )

        current = self._persist_boundary_candidate(
            pipeline=pipeline,
            adapter=adapter,
            metadata=metadata,
            start=10.0,
            end=20.0,
            context=_transcript_boundary_context(
                candidate_id,
                start=10.0,
                end=20.0,
                surface=True,
            ),
        )

        self.assertEqual(current.reel_id, stale.reel_id)
        self.assertEqual((current.t_start, current.t_end), (10.0, 20.0))
        with db_module.get_conn() as conn:
            row = db_module.fetch_one(
                conn,
                "SELECT search_context_json FROM reels WHERE id = ?",
                (stale.reel_id,),
            )
        context = json.loads(row["search_context_json"])
        self.assertEqual(
            context["selection_contract_version"], "quality_silence_v38"
        )
        self.assertEqual(context["boundary_status"], "context_aligned")

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
