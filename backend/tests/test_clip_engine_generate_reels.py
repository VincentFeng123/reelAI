"""
Tests for the clip-engine-routed ReelService.generate_reels (Task T4).

The legacy search+topic_cut internals of generate_reels were hard-replaced with
a per-concept loop that routes each extracted concept through
IngestionPipeline.ingest_topic (multi-clip). This test drives the real pipeline
with the heavy engine surfaces mocked (clip_engine_search.discover +
clip_engine_run.clip), against a temp file-backed SQLite DB, and asserts the T4
contract: reels persist under generation_id, ranked_feed reads them back,
MULTIPLE clips per video survive, on_reel_created fires once per reel, the
num_reels cap holds, and the returned dicts carry the _create_reel key shape.
"""

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

# Skip the import-time orphan sweep so tests don't poke at /tmp during import.
os.environ.setdefault("REELAI_INGEST_SKIP_IMPORT_SWEEP", "1")

from backend.app import db as db_module  # noqa: E402
from backend.app.config import get_settings  # noqa: E402
import backend.app.main as main_module  # noqa: E402
from backend.app.ingestion import pipeline as pipeline_module  # noqa: E402

# 11-char YouTube-style id so the embed video_url round-trips through
# clip_engine.metadata.extract_video_id in _reel_attribution_to_dict.
VIDEO_ID = "vidABCDE123"
MATERIAL_ID = "mat-t4-gen"
CONCEPT_ID = "concept-t4-gen"

SAP_ADVANCED_ATP_VIDEO_ID = "kRm4CviZPsc"
SAP_ADVANCED_ATP_TITLE = (
    "SAP S/4HANA aATP (Advanced ATP) – Part 1 | End-to-End Demo | SD aATP Tutorial"
)
SAP_ADVANCED_ATP_TRANSCRIPT = (
    "in the SAP S/4HANA training system that provides this ATP advanced ATP "
    "functionality. It is a part of S4H always on function business functions. "
    "Okay. So, now we know that this advanced ATP is uh activated for this test "
    "system. Here we will go, and we will check the config. So, going to this SPRO, "
    "you will see that uh uh first we will check if this check uh group is uh been "
    "activated for this advanced ATP checking. So, we'll go to this sales and "
    "distribution node, and there you will go to this uh basic function, and here "
    "you can select this node where you will find this advanced check and transfer "
    "of requirements. So, in this uh node, you will find another node. The subnode "
    "is for availability check. So, in this column AV, you will find this "
    "availability check. Look at this entry. So, this is a individual request "
    "change check. And in this column, you can see the last column in this table, "
    "you will see the last column as advanced ATP, which is activated."
)


def _discover_result(video_id: str = VIDEO_ID) -> dict:
    return {
        "corrected": "cellular respiration",
        "videos": [
            {
                "id": video_id,
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "title": "Cellular Respiration Explained",
                "channel": "Bio Channel",
                "duration": 300,
                "thumbnail": "",
                "view_count": 1000,
                "upload_date": None,
            }
        ],
        "credits_used": 1,
        "warning": None,
    }


def _quality_v2_engine_out(engine_out: dict) -> dict:
    transcript = engine_out["transcript"]
    transcript.update(
        source="supadata",
        artifact_key=f"supadata:{engine_out['video_id']}",
        native_mode=True,
    )
    segments = transcript["segments"]
    for index, segment in enumerate(segments):
        segment.setdefault("cue_id", f"cue-{index}")
    for index, clip in enumerate(engine_out["clips"]):
        selected = [
            segment
            for segment in segments
            if float(segment["start"]) >= float(clip["start"]) - 1e-6
            and float(segment["end"]) <= float(clip["end"]) + 1e-6
        ]
        if not selected:
            continue
        words = " ".join(str(segment["text"]) for segment in selected).split()
        if len(words) < 5:
            selected[0]["text"] = (
                f"{selected[0]['text']} with complete grounded explanatory context"
            )
            words = " ".join(str(segment["text"]) for segment in selected).split()
        clip.update(
            cue_ids=[str(segment["cue_id"]) for segment in selected],
            kind="educational",
            learning_objective=clip.get("learning_objective")
            or f"Explain {clip.get('title') or 'this concept'}.",
            facet=clip.get("facet") or f"facet-{index}",
            reason=clip.get("reason") or "Provides a complete grounded explanation.",
            self_contained=True,
            is_standalone=True,
            directly_teaches_topic=True,
            substantive=True,
            factually_grounded=True,
            topic_evidence_quote=" ".join(words[:40]),
            boundary_confidence=0.9,
            prerequisite_ids=[],
            selection_candidate_id=clip.get("selection_candidate_id")
            or f"candidate-{index}",
        )
        clip.setdefault("informativeness", 0.9)
        clip.setdefault("topic_relevance", 0.9)
        clip.setdefault("educational_importance", 0.9)
        clip.setdefault("difficulty", 0.15)
    return engine_out


def _multi_clip_engine_out() -> dict:
    """Two relevance-surviving clips from one video's transcript. Each clip's
    window text contains the topic tokens ('cellular respiration') so
    clip_engine_bridge.filter_by_query keeps both."""
    return _quality_v2_engine_out({
        "video_id": VIDEO_ID,
        "clips": [
            {
                "start": 0.0,
                "end": 45.0,
                "cut_end": 45.0,
                "title": "Cellular respiration overview",
                "facet": "biology",
                "reason": "core idea",
                "summary": "Cellular respiration releases usable energy in the mitochondria.",
                "takeaways": ["Mitochondria release usable energy", "Respiration transforms fuel"],
                "match_reason": "It explains how cellular respiration releases energy.",
                "informativeness": 0.92,
                "assessment": {
                    "prompt": "Where is usable energy released during cellular respiration?",
                    "options": ["In the mitochondria", "In the nucleus", "In the membrane", "Outside the cell"],
                    "correct_index": 0,
                    "explanation": "The transcript states that mitochondria release energy.",
                },
                "sequence_index": 0,
                "embed_url": "",
            },
            {
                "start": 50.0,
                "end": 95.0,
                "cut_end": 95.0,
                "title": "Cellular respiration in the cytoplasm",
                "facet": "biology",
                "reason": "second beat",
                "sequence_index": 1,
                "embed_url": "",
            },
        ],
        "transcript": {
            "segments": [
                {"start": 0.0, "end": 45.0, "text": "Cellular respiration releases energy in the mitochondria."},
                {"start": 50.0, "end": 95.0, "text": "The cellular respiration process continues in the cytoplasm."},
            ],
            "words": [],
            "duration": 300.0,
        },
        "notes": "",
    })


def _five_minute_engine_out() -> dict:
    """One 300s whole-topic clip — the kind the practice (Gemini) engine cuts for
    a long lecture segment. RAW-PRACTICE: it must persist and be served intact."""
    return _quality_v2_engine_out({
        "video_id": VIDEO_ID,
        "clips": [
            {
                "start": 0.0,
                "end": 300.0,
                "cut_end": 300.0,
                "title": "Cellular respiration full walkthrough",
                "facet": "biology",
                "reason": "whole topic",
                "sequence_index": 0,
                "embed_url": "",
            },
        ],
        "transcript": {
            "segments": [
                {"start": 0.0, "end": 300.0, "text": "Cellular respiration explained end to end."},
            ],
            "words": [],
            "duration": 300.0,
        },
        "notes": "",
    })


class ClipEngineGenerateReelsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.previous_data_dir = os.environ.get("DATA_DIR")
        os.environ["DATA_DIR"] = self.temp_dir.name
        self.addCleanup(self._restore_environment)
        db_module._db_ready = False
        get_settings.cache_clear()
        main_module.settings = get_settings()

        os.environ["REELAI_INGEST_SKIP_IMPORT_SWEEP"] = "1"

        # Bump rate limits so a single generation never trips the per-platform cap.
        from backend.app.ingestion.pipeline import _PlatformRateLimiter
        main_module.ingestion_pipeline._rate_limiter = _PlatformRateLimiter(
            overrides={"yt": (1000, 60.0)}
        )
        prepared = pipeline_module.clip_engine_silence.AudioPreparationResult(
            "ready",
            source=pipeline_module.clip_engine_silence.PreparedAudioSource(
                "https://audio.invalid/test"
            ),
        )
        self._prepare_patch = mock.patch.object(
            pipeline_module.clip_engine_silence,
            "prepare_audio_source",
            return_value=prepared,
        )
        self._verify_patch = mock.patch.object(
            pipeline_module.clip_engine_silence,
            "verify_acoustic_boundaries",
            side_effect=lambda _url, start, end, **_kwargs: (
                pipeline_module.clip_engine_silence.SilenceVerificationResult(
                    "verified", start, end, {"threshold_dbfs": -38.0}
                )
            ),
        )
        self._prepare_patch.start()
        self._verify_patch.start()
        self.addCleanup(self._prepare_patch.stop)
        self.addCleanup(self._verify_patch.stop)

        self._seed_material_and_concept()

    def _restore_environment(self) -> None:
        if self.previous_data_dir is None:
            os.environ.pop("DATA_DIR", None)
        else:
            os.environ["DATA_DIR"] = self.previous_data_dir
        db_module._db_ready = False
        get_settings.cache_clear()
        main_module.settings = get_settings()

    def _seed_material_and_concept(self) -> None:
        with db_module.get_conn(transactional=True) as conn:
            conn.execute(
                "INSERT INTO materials (id, subject_tag, raw_text, source_type, source_path, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    MATERIAL_ID,
                    "cellular respiration",
                    "Biology notes",
                    "text",
                    None,
                    "2026-07-06T00:00:00+00:00",
                ),
            )
            conn.execute(
                "INSERT INTO concepts (id, material_id, title, keywords_json, summary, embedding_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    CONCEPT_ID,
                    MATERIAL_ID,
                    "Cellular respiration",
                    '["cellular respiration", "mitochondria"]',
                    "How cells release energy.",
                    None,
                    "2026-07-06T00:01:00+00:00",
                ),
            )

    def _patched_engine(self, engine_out: dict):
        """Context managers patching discover + run.clip at the pipeline aliases."""
        mock_search = mock.patch.object(pipeline_module, "clip_engine_search")
        mock_run = mock.patch.object(pipeline_module, "clip_engine_run")
        search = mock_search.start()
        run = mock_run.start()
        self.addCleanup(mock_search.stop)
        self.addCleanup(mock_run.stop)
        search.discover.return_value = _discover_result()
        run.clip.return_value = _quality_v2_engine_out(engine_out)
        return search, run

    # ------------------------------------------------------------------ #
    # Happy path: one concept -> one video -> MULTIPLE clips persisted
    # ------------------------------------------------------------------ #
    def test_generate_reels_multi_clip_per_video(self) -> None:
        self._patched_engine(_multi_clip_engine_out())
        collector: list[dict] = []

        with db_module.get_conn() as conn:
            result = main_module.reel_service.generate_reels(
                conn,
                material_id=MATERIAL_ID,
                concept_id=None,
                num_reels=5,
                creative_commons_only=False,
                generation_id="gen-1",
                on_reel_created=collector.append,
            )

        # 1. Reels landed in the reels table under generation_id="gen-1".
        with db_module.get_conn() as conn:
            rows = db_module.fetch_all(
                conn,
                "SELECT id, video_id, ai_summary, match_reason, informativeness FROM reels WHERE generation_id = ?",
                ("gen-1",),
            )
        self.assertGreaterEqual(len(rows), 2, "expected >=2 persisted reels under gen-1")
        detailed = next(row for row in rows if row["ai_summary"])
        self.assertIn("mitochondria", detailed["ai_summary"].lower())
        self.assertTrue(str(detailed["match_reason"]).strip())
        self.assertAlmostEqual(float(detailed["informativeness"]), 0.92)
        with db_module.get_conn() as conn:
            question_count = db_module.fetch_one(
                conn,
                "SELECT COUNT(*) AS count FROM reel_assessment_questions",
            )
        self.assertEqual(int(question_count["count"]), 1)

        # 3. MULTIPLE clips per video kept (>=2 reels from the one mocked video).
        video_ids = {r["video_id"] for r in rows}
        self.assertEqual(video_ids, {f"yt:{VIDEO_ID}"})
        self.assertGreaterEqual(len(rows), 2)

        # 2. ranked_feed reads them back under the same generation_id.
        with db_module.get_conn() as conn:
            feed = main_module.reel_service.ranked_feed(
                conn, material_id=MATERIAL_ID, generation_id="gen-1"
            )
        self.assertGreaterEqual(len(feed), 2)

        # 4. on_reel_created fired once per persisted reel.
        self.assertEqual(len(collector), len(rows))

        # 5. num_reels cap respected (<= 5).
        self.assertLessEqual(len(result), 5)

        # 6. Returned items carry the _create_reel dict key shape.
        self.assertTrue(result)
        first = result[0]
        for key in ("reel_id", "video_url", "concept_id", "score", "video_id"):
            self.assertIn(key, first)
        self.assertEqual(first["concept_id"], CONCEPT_ID)
        self.assertEqual(first["video_id"], VIDEO_ID)
        self.assertTrue(first["video_url"].startswith(f"https://www.youtube.com/embed/{VIDEO_ID}"))
        self.assertTrue(str(first.get("ai_summary") or "").strip())
        self.assertTrue(str(first.get("match_reason") or "").strip())

    def test_below_threshold_quality_is_rejected(self) -> None:
        engine_out = _multi_clip_engine_out()
        engine_out["clips"] = [{
            **engine_out["clips"][0],
            "informativeness": 0.0,
            "topic_relevance": 0.0,
            "educational_importance": 0.0,
            "difficulty": 0.0,
            "boundary_confidence": 0.9,
            "directly_teaches_topic": True,
            "substantive": True,
            "factually_grounded": True,
            "topic_evidence_quote": (
                "Cellular respiration releases energy in the mitochondria"
            ),
            "is_standalone": True,
            "prerequisite_ids": [],
            "selection_candidate_id": "low-score",
        }]
        self._patched_engine(engine_out)

        with db_module.get_conn() as conn:
            generated = main_module.reel_service.generate_reels(
                conn,
                material_id=MATERIAL_ID,
                concept_id=CONCEPT_ID,
                num_reels=1,
                creative_commons_only=False,
                generation_id="gen-low-score",
            )
        with db_module.get_conn() as conn:
            stored = db_module.fetch_one(
                conn,
                "SELECT informativeness, difficulty FROM reels WHERE generation_id = ?",
                ("gen-low-score",),
            )
            feed = main_module.reel_service.ranked_feed(
                conn,
                material_id=MATERIAL_ID,
                generation_id="gen-low-score",
            )

        self.assertIsNone(stored)
        self.assertEqual(generated, [])
        self.assertEqual(feed, [])

    def test_one_pass_clips_never_call_legacy_summary_model(self) -> None:
        self._patched_engine(_multi_clip_engine_out())
        with mock.patch.object(
            main_module.reel_service,
            "_brief_ai_summary",
            side_effect=AssertionError("legacy summary model must not run"),
        ):
            with db_module.get_conn() as conn:
                generated = main_module.reel_service.generate_reels(
                    conn,
                    material_id=MATERIAL_ID,
                    concept_id=None,
                    num_reels=5,
                    creative_commons_only=False,
                    generation_id="gen-one-pass-details",
                )
            with db_module.get_conn() as conn:
                served = main_module.reel_service.ranked_feed(
                    conn,
                    material_id=MATERIAL_ID,
                    generation_id="gen-one-pass-details",
                )

        self.assertTrue(generated)
        self.assertTrue(served)
        self.assertTrue(all(reel["ai_summary"] for reel in served))

    def test_learner_level_override_reaches_discovery(self) -> None:
        search, _run = self._patched_engine(_multi_clip_engine_out())
        with db_module.get_conn() as conn:
            main_module.reel_service.generate_reels(
                conn,
                material_id=MATERIAL_ID,
                concept_id=CONCEPT_ID,
                num_reels=1,
                creative_commons_only=False,
                generation_id="gen-level-override",
                knowledge_level_override="advanced",
            )
        self.assertEqual(search.discover.call_args.kwargs.get("level"), "advanced")

    def test_full_material_generation_keeps_subject_as_literal_topic(self) -> None:
        with db_module.get_conn(transactional=True) as conn:
            conn.execute(
                "UPDATE materials SET subject_tag = ? WHERE id = ?",
                ("  biology   and\nlife  ", MATERIAL_ID),
            )

        with mock.patch.object(
            main_module.ingestion_pipeline,
            "ingest_topic",
            return_value=([], [VIDEO_ID]),
        ) as ingest_topic:
            with db_module.get_conn() as conn:
                main_module.reel_service.generate_reels(
                    conn,
                    material_id=MATERIAL_ID,
                    concept_id=None,
                    num_reels=1,
                    creative_commons_only=False,
                    generation_id="gen-full-material-literal-topic",
                )

        self.assertEqual(
            ingest_topic.call_args.kwargs["literal_topic"],
            "biology and life",
        )

    def test_explicit_concept_generation_keeps_concept_as_literal_topic(self) -> None:
        with db_module.get_conn(transactional=True) as conn:
            conn.execute(
                "UPDATE materials SET subject_tag = ? WHERE id = ?",
                ("  biology   and\nlife  ", MATERIAL_ID),
            )

        with mock.patch.object(
            main_module.ingestion_pipeline,
            "ingest_topic",
            return_value=([], [VIDEO_ID]),
        ) as ingest_topic:
            with db_module.get_conn() as conn:
                main_module.reel_service.generate_reels(
                    conn,
                    material_id=MATERIAL_ID,
                    concept_id=CONCEPT_ID,
                    num_reels=1,
                    creative_commons_only=False,
                    generation_id="gen-explicit-concept-literal-topic",
                )

        self.assertEqual(
            ingest_topic.call_args.kwargs["literal_topic"],
            "Cellular respiration",
        )

    def test_acronym_leaf_search_keeps_parent_topic_and_avoids_sap_atp(self) -> None:
        """Production regression: ``advanced ATP`` retrieved SAP S/4HANA for a
        cellular-respiration material. The leaf remains valid, but both search and
        transcript selection must receive its parent material context.
        """
        biological_video_id = "bioATP12345"
        biological_title = "ATP production during cellular respiration"
        biological_transcript = (
            "During cellular respiration, the mitochondrial electron transport "
            "chain builds a proton gradient that drives ATP synthase to make ATP."
        )
        with db_module.get_conn(transactional=True) as conn:
            conn.execute(
                "UPDATE materials SET subject_tag = 'cellular respiration' WHERE id = ?",
                (MATERIAL_ID,),
            )
            conn.execute(
                "UPDATE concepts SET title = 'Atp', keywords_json = '[\"atp\"]' WHERE id = ?",
                (CONCEPT_ID,),
            )

        search_patch = mock.patch.object(pipeline_module, "clip_engine_search")
        run_patch = mock.patch.object(pipeline_module, "clip_engine_run")
        search = search_patch.start()
        run = run_patch.start()
        self.addCleanup(search_patch.stop)
        self.addCleanup(run_patch.stop)

        def discovery(query: str, **_kwargs) -> dict:
            qualified = "cellular respiration" in query.casefold()
            video_id = biological_video_id if qualified else SAP_ADVANCED_ATP_VIDEO_ID
            title = biological_title if qualified else SAP_ADVANCED_ATP_TITLE
            return {
                "corrected": query,
                "videos": [{
                    "id": video_id,
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "title": title,
                    "channel": "Biology" if qualified else "SAP Training",
                    "duration": 300,
                    "thumbnail": "",
                    "view_count": 1000,
                    "upload_date": None,
                }],
                "credits_used": 1,
                "warning": None,
            }

        def clip(source_url: str, **_kwargs) -> dict:
            is_biological = biological_video_id in source_url
            video_id = biological_video_id if is_biological else SAP_ADVANCED_ATP_VIDEO_ID
            title = biological_title if is_biological else SAP_ADVANCED_ATP_TITLE
            transcript = biological_transcript if is_biological else SAP_ADVANCED_ATP_TRANSCRIPT
            return _quality_v2_engine_out({
                "video_id": video_id,
                "clips": [{
                    "start": 0.0,
                    "end": 60.0,
                    "cut_end": 60.0,
                    "title": title,
                    "facet": "ATP",
                    "reason": "ATP explanation",
                    "difficulty": 0.85,
                    "sequence_index": 0,
                    "embed_url": "",
                }],
                "transcript": {
                    "segments": [{"start": 0.0, "end": 60.0, "text": transcript}],
                    "words": [],
                    "duration": 300.0,
                },
                "notes": "",
            })

        search.discover.side_effect = discovery
        run.clip.side_effect = clip
        with db_module.get_conn() as conn:
            generated = main_module.reel_service.generate_reels(
                conn,
                material_id=MATERIAL_ID,
                concept_id=CONCEPT_ID,
                num_reels=1,
                creative_commons_only=False,
                generation_id="gen-atp-parent-context",
                knowledge_level_override="advanced",
            )

        self.assertEqual(search.discover.call_args.args[0], "Atp in cellular respiration")
        self.assertEqual(run.clip.call_args.kwargs.get("topic"), "Atp in cellular respiration")
        self.assertTrue(generated)
        self.assertEqual(generated[0]["video_id"], biological_video_id)
        self.assertNotIn("SAP S/4HANA", generated[0]["video_title"])

    def test_acronym_leaf_without_subject_tag_uses_sibling_material_context(self) -> None:
        """Text/file materials may have no subject tag. Their sibling concepts
        still provide enough source context to disambiguate a short leaf query.
        """
        with db_module.get_conn(transactional=True) as conn:
            conn.execute(
                "UPDATE materials SET subject_tag = NULL, source_type = 'text' WHERE id = ?",
                (MATERIAL_ID,),
            )
            conn.execute(
                "UPDATE concepts SET title = 'Atp', keywords_json = '[\"atp\"]' WHERE id = ?",
                (CONCEPT_ID,),
            )
            conn.execute(
                "INSERT INTO concepts "
                "(id, material_id, title, keywords_json, summary, embedding_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, NULL, ?)",
                (
                    "concept-cellular-respiration",
                    MATERIAL_ID,
                    "Cellular Respiration",
                    '["cellular respiration"]',
                    "Cellular respiration takes place in mitochondria.",
                    "2026-07-06T00:02:00+00:00",
                ),
            )
            conn.execute(
                "INSERT INTO concepts "
                "(id, material_id, title, keywords_json, summary, embedding_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, NULL, ?)",
                (
                    "concept-mitochondria",
                    MATERIAL_ID,
                    "Mitochondria",
                    '["mitochondria"]',
                    "Mitochondria carry out cellular respiration.",
                    "2026-07-06T00:03:00+00:00",
                ),
            )

        search, run = self._patched_engine(_multi_clip_engine_out())
        with db_module.get_conn() as conn:
            main_module.reel_service.generate_reels(
                conn,
                material_id=MATERIAL_ID,
                concept_id=CONCEPT_ID,
                num_reels=1,
                creative_commons_only=False,
                generation_id="gen-atp-sibling-context",
                knowledge_level_override="advanced",
            )

        self.assertEqual(
            search.discover.call_args.args[0],
            "Atp in cellular respiration mitochondria",
        )
        self.assertEqual(
            run.clip.call_args.kwargs.get("topic"),
            "Atp in cellular respiration mitochondria",
        )

    def test_ranked_feed_hides_cached_sap_atp_but_keeps_biochemical_atp(self) -> None:
        """Old completed inventories are replayed without regeneration, so the
        serving gate must reject the exact production SAP false positive while
        retaining biological ATP teaching for the same short leaf concept.
        """
        biological_video_id = "bioATP12345"
        biological_transcript = (
            "During cellular respiration, the mitochondrial electron transport "
            "chain builds a proton gradient that drives ATP synthase to make ATP."
        )
        with db_module.get_conn(transactional=True) as conn:
            conn.execute(
                "UPDATE materials SET subject_tag = 'cellular respiration' WHERE id = ?",
                (MATERIAL_ID,),
            )
            conn.execute(
                "UPDATE concepts SET title = 'Atp', keywords_json = '[\"atp\"]' WHERE id = ?",
                (CONCEPT_ID,),
            )
            for video_id, title, channel, transcript in (
                (
                    f"yt:{SAP_ADVANCED_ATP_VIDEO_ID}",
                    SAP_ADVANCED_ATP_TITLE,
                    "SAP Training",
                    SAP_ADVANCED_ATP_TRANSCRIPT,
                ),
                (
                    f"yt:{biological_video_id}",
                    "ATP production during cellular respiration",
                    "Biology",
                    biological_transcript,
                ),
            ):
                conn.execute(
                    "INSERT INTO videos "
                    "(id, title, channel_title, description, duration_sec, created_at) "
                    "VALUES (?, ?, ?, '', 300, '2026-07-13T00:00:00+00:00')",
                    (video_id, title, channel),
                )
                conn.execute(
                    "INSERT INTO reels "
                    "(id, material_id, concept_id, video_id, video_url, t_start, t_end, "
                    "transcript_snippet, takeaways_json, base_score, difficulty, generation_id, "
                    "search_context_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?, 0, 60, ?, '[]', 1.0, 0.7, ?, ?, "
                    "'2026-07-13T00:01:00+00:00')",
                    (
                        f"reel-{video_id}",
                        MATERIAL_ID,
                        CONCEPT_ID,
                        video_id,
                        f"https://www.youtube.com/embed/{video_id.removeprefix('yt:')}",
                        transcript,
                        "gen-cached-atp",
                        json.dumps({
                            "surface_eligible": True,
                            "boundary_status": "verified",
                            "boundary_diagnostics": {
                                "method": "energy_silence",
                                "acoustic_verified": True,
                            },
                        }),
                    ),
                )

        with db_module.get_conn() as conn:
            feed = main_module.reel_service.ranked_feed(
                conn,
                material_id=MATERIAL_ID,
                generation_id="gen-cached-atp",
                fast_mode=True,
            )

        served_ids = {row["video_id"] for row in feed}
        self.assertIn(f"yt:{biological_video_id}", served_ids)
        self.assertNotIn(f"yt:{SAP_ADVANCED_ATP_VIDEO_ID}", served_ids)

    def test_concept_priority_uses_only_current_learner_feedback(self) -> None:
        other_concept_id = "concept-t4-other"
        with db_module.get_conn(transactional=True) as conn:
            conn.execute(
                "INSERT INTO concepts (id, material_id, title, keywords_json, summary, embedding_json, created_at) "
                "VALUES (?, ?, 'ATP', '[]', '', NULL, '2026-07-06T00:01:00+00:00')",
                (other_concept_id, MATERIAL_ID),
            )
            for reel_id, concept_id, video_id in (
                ("reel-priority-primary", CONCEPT_ID, "video-priority-primary"),
                ("reel-priority-other", other_concept_id, "video-priority-other"),
            ):
                conn.execute(
                    "INSERT INTO videos (id, title, channel_title, duration_sec, created_at) "
                    "VALUES (?, ?, 'channel', 300, '2026-07-06T00:01:00+00:00')",
                    (video_id, video_id),
                )
                conn.execute(
                    "INSERT INTO reels "
                    "(id, material_id, concept_id, video_id, video_url, t_start, t_end, "
                    "transcript_snippet, takeaways_json, base_score, created_at) "
                    "VALUES (?, ?, ?, ?, '', 0, 30, '', '[]', 1.0, '2026-07-06T00:02:00+00:00')",
                    (reel_id, MATERIAL_ID, concept_id, video_id),
                )

            for learner_id, primary_helpful in (
                ("owner:learner-a", True),
                ("owner:learner-b", False),
            ):
                main_module.reel_service.record_feedback(
                    conn,
                    "reel-priority-primary",
                    helpful=primary_helpful,
                    confusing=not primary_helpful,
                    rating=3,
                    saved=False,
                    learner_id=learner_id,
                )
                main_module.reel_service.record_feedback(
                    conn,
                    "reel-priority-other",
                    helpful=not primary_helpful,
                    confusing=primary_helpful,
                    rating=3,
                    saved=False,
                    learner_id=learner_id,
                )

            concepts = db_module.fetch_all(
                conn,
                "SELECT * FROM concepts WHERE material_id = ? ORDER BY created_at, id",
                (MATERIAL_ID,),
            )
            learner_a = main_module.reel_service._order_concepts(
                conn, MATERIAL_ID, concepts, "owner:learner-a"
            )
            learner_b = main_module.reel_service._order_concepts(
                conn, MATERIAL_ID, concepts, "owner:learner-b"
            )

        self.assertEqual([item["id"] for item in learner_a], [other_concept_id, CONCEPT_ID])
        self.assertEqual([item["id"] for item in learner_b], [CONCEPT_ID, other_concept_id])

    # ------------------------------------------------------------------ #
    # RAW-PRACTICE: num_reels no longer truncates PERSISTENCE — a 2-clip video
    # with num_reels=1 persists BOTH clips; only the RESPONSE page is shaped.
    # ------------------------------------------------------------------ #
    def test_num_reels_shapes_response_not_persistence(self) -> None:
        self._patched_engine(_multi_clip_engine_out())
        collector: list[dict] = []

        with db_module.get_conn() as conn:
            result = main_module.reel_service.generate_reels(
                conn,
                material_id=MATERIAL_ID,
                concept_id=None,
                num_reels=1,
                creative_commons_only=False,
                generation_id="gen-cap",
                on_reel_created=collector.append,
            )

        # Both engine clips persist (no persistence cap) and stream once each.
        with db_module.get_conn() as conn:
            rows = db_module.fetch_all(
                conn,
                "SELECT id FROM reels WHERE generation_id = ?",
                ("gen-cap",),
            )
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(collector), 2)
        # The RESPONSE page is still shaped to num_reels by _finalize_generated_reels.
        self.assertEqual(len(result), 1)

        # ranked_feed serves BOTH persisted clips back regardless of num_reels.
        with db_module.get_conn() as conn:
            feed = main_module.reel_service.ranked_feed(
                conn, material_id=MATERIAL_ID, generation_id="gen-cap"
            )
        self.assertEqual(len(feed), 2)
        self.assertTrue(all(
            reel.get("selection_contract_version") == "quality_silence_v5"
            for reel in feed
        ))

    def test_quality_silence_v5_response_orders_stage_before_quality(self) -> None:
        generated = [
            {
                "reel_id": "a-late",
                "video_id": "source-a",
                "t_start": 90.75,
                "difficulty": 0.8,
                "score": 0.88,
                "_selection_quality_floor": 0.88,
                "_selection_quality_mean": 0.94,
                "_selection_topic_relevance": 0.96,
                "_selection_source_rank": 0,
                "selection_contract_version": "quality_silence_v5",
            },
            {
                "reel_id": "b",
                "video_id": "source-b",
                "t_start": 20.5,
                "difficulty": 0.1,
                "score": 0.84,
                "_selection_quality_floor": 0.84,
                "_selection_quality_mean": 0.97,
                "_selection_topic_relevance": 0.99,
                "_selection_source_rank": 1,
                "selection_contract_version": "quality_silence_v5",
            },
            {
                "reel_id": "a-early",
                "video_id": "source-a",
                "t_start": 10.25,
                "difficulty": 0.3,
                "score": 0.91,
                "_selection_quality_floor": 0.91,
                "_selection_quality_mean": 0.92,
                "_selection_topic_relevance": 0.93,
                "_selection_source_rank": 0,
                "selection_contract_version": "quality_silence_v5",
            },
        ]

        result = main_module.reel_service._finalize_generated_reels(
            generated=generated,
            num_reels=3,
            preferred_video_duration="any",
        )

        self.assertEqual(
            [reel["reel_id"] for reel in result],
            ["a-early", "b", "a-late"],
        )
        self.assertTrue(all(
            not any(key.startswith("_selection_") for key in reel)
            for reel in result
        ))

    def test_safe_batch_caps_progressive_ingest_to_requested_buffer(self) -> None:
        with mock.patch.object(
            main_module.ingestion_pipeline,
            "ingest_topic",
            return_value=([], [VIDEO_ID]),
        ) as ingest_topic:
            with db_module.get_conn() as conn:
                main_module.reel_service.generate_reels(
                    conn,
                    material_id=MATERIAL_ID,
                    concept_id=None,
                    num_reels=20,
                    creative_commons_only=False,
                    generation_id="gen-progressive",
                )

        self.assertEqual(ingest_topic.call_args.kwargs["max_reels"], 22)

    def test_two_stage_cap_and_profile_reach_ingestion(self) -> None:
        analyzed: set[str] = set()
        retrieved: set[str] = set()
        with mock.patch.object(
            main_module.ingestion_pipeline,
            "ingest_topic",
            return_value=([], [VIDEO_ID]),
        ) as ingest_topic:
            with db_module.get_conn() as conn:
                main_module.reel_service.generate_reels(
                    conn,
                    material_id=MATERIAL_ID,
                    concept_id=CONCEPT_ID,
                    num_reels=20,
                    creative_commons_only=False,
                    generation_id="gen-bootstrap-cap",
                    retrieval_profile="bootstrap",
                    max_new_reels=2,
                    analyzed_video_ids=analyzed,
                    retrieved_video_ids=retrieved,
                )

        kwargs = ingest_topic.call_args.kwargs
        self.assertEqual(kwargs["max_reels"], 2)
        self.assertEqual(kwargs["retrieval_profile"], "bootstrap")
        self.assertIs(kwargs["analyzed_video_ids"], analyzed)
        self.assertIs(kwargs["retrieved_video_ids"], retrieved)
        self.assertEqual(retrieved, {VIDEO_ID})

    def test_material_inventory_never_exceeds_300_reels(self) -> None:
        existing_video_id = "inventory-video"
        with db_module.get_conn(transactional=True) as conn:
            conn.execute(
                "INSERT INTO videos (id, title, channel_title, duration_sec, created_at) "
                "VALUES (?, 'Existing inventory', 'Channel', 1000, '2026-07-06T00:01:00+00:00')",
                (existing_video_id,),
            )
            conn.executemany(
                "INSERT INTO reels "
                "(id, generation_id, material_id, concept_id, video_id, video_url, t_start, t_end, "
                "transcript_snippet, takeaways_json, base_score, created_at) "
                "VALUES (?, 'gen-existing', ?, ?, ?, '', ?, ?, 'existing', '[]', 0.5, "
                "'2026-07-06T00:02:00+00:00')",
                [
                    (
                        f"existing-reel-{index}",
                        MATERIAL_ID,
                        CONCEPT_ID,
                        existing_video_id,
                        float(index * 2),
                        float(index * 2 + 1),
                    )
                    for index in range(299)
                ],
            )

        self._patched_engine(_multi_clip_engine_out())
        with db_module.get_conn() as conn:
            main_module.reel_service.generate_reels(
                conn,
                material_id=MATERIAL_ID,
                concept_id=None,
                num_reels=20,
                creative_commons_only=False,
                generation_id="gen-inventory-limit",
            )

        with db_module.get_conn() as conn:
            total = db_module.fetch_one(
                conn,
                "SELECT COUNT(*) AS count FROM reels WHERE material_id = ?",
                (MATERIAL_ID,),
            )
            added = db_module.fetch_one(
                conn,
                "SELECT COUNT(*) AS count FROM reels WHERE generation_id = ?",
                ("gen-inventory-limit",),
            )
        self.assertEqual(int(total["count"]), 300)
        self.assertEqual(int(added["count"]), 1)

    # ------------------------------------------------------------------ #
    # Complete teaching spans are not rejected because of their duration.
    # ------------------------------------------------------------------ #
    def test_five_minute_complete_clip_is_persisted_and_served(self) -> None:
        self._patched_engine(_five_minute_engine_out())

        with db_module.get_conn() as conn:
            main_module.reel_service.generate_reels(
                conn,
                material_id=MATERIAL_ID,
                concept_id=None,
                num_reels=5,
                creative_commons_only=False,
                generation_id="gen-5min",
            )

        with db_module.get_conn() as conn:
            rows = db_module.fetch_all(
                conn,
                "SELECT id, t_start, t_end FROM reels WHERE generation_id = ?",
                ("gen-5min",),
            )
        self.assertEqual(len(rows), 1)
        self.assertEqual(float(rows[0]["t_start"]), 0.0)
        self.assertEqual(float(rows[0]["t_end"]), 300.0)

        with db_module.get_conn() as conn:
            feed = main_module.reel_service.ranked_feed(
                conn, material_id=MATERIAL_ID, generation_id="gen-5min"
            )
        self.assertEqual(len(feed), 1)

    # ------------------------------------------------------------------ #
    # dry_run: discover-only viability probe, zero DB writes, non-empty
    # ------------------------------------------------------------------ #
    def test_generate_reels_dry_run_zero_db_writes(self) -> None:
        self._patched_engine(_multi_clip_engine_out())

        with db_module.get_conn() as conn:
            result = main_module.reel_service.generate_reels(
                conn,
                material_id=MATERIAL_ID,
                concept_id=None,
                num_reels=4,
                creative_commons_only=False,
                generation_id="gen-dry",
                dry_run=True,
            )

        # Non-empty when videos exist.
        self.assertTrue(result)
        for item in result:
            self.assertIn("reel_id", item)
            self.assertIn("video_id", item)

        # Zero DB writes under dry_run.
        with db_module.get_conn() as conn:
            rows = db_module.fetch_all(
                conn,
                "SELECT id FROM reels WHERE generation_id = ?",
                ("gen-dry",),
            )
        self.assertEqual(len(rows), 0)

    # ------------------------------------------------------------------ #
    # Finding #3: refinement/extension excludes prior-generation video ids
    # ------------------------------------------------------------------ #
    def test_excludes_prior_generation_video_ids_from_discover(self) -> None:
        search, _run = self._patched_engine(_multi_clip_engine_out())

        # gen-1: persist reels for VIDEO_ID (stored as yt:<id>).
        with db_module.get_conn() as conn:
            main_module.reel_service.generate_reels(
                conn,
                material_id=MATERIAL_ID,
                concept_id=None,
                num_reels=5,
                creative_commons_only=False,
                generation_id="gen-1",
            )
        with db_module.get_conn() as conn:
            rows = db_module.fetch_all(
                conn, "SELECT DISTINCT video_id FROM reels WHERE generation_id = ?", ("gen-1",)
            )
        self.assertEqual({r["video_id"] for r in rows}, {f"yt:{VIDEO_ID}"})

        search.discover.reset_mock()

        # gen-2 excludes gen-1 → discover must receive gen-1's BARE video id.
        with db_module.get_conn() as conn:
            main_module.reel_service.generate_reels(
                conn,
                material_id=MATERIAL_ID,
                concept_id=None,
                num_reels=5,
                creative_commons_only=False,
                generation_id="gen-2",
                exclude_generation_ids=["gen-1"],
            )

        self.assertTrue(search.discover.called)
        passed = search.discover.call_args.kwargs.get("exclude_video_ids") or []
        self.assertIn(VIDEO_ID, passed)
        self.assertNotIn(f"yt:{VIDEO_ID}", passed)

    # ------------------------------------------------------------------ #
    # Finding #4a: one concept's engine failure must not abort the run
    # ------------------------------------------------------------------ #
    def test_one_concept_failure_does_not_abort_generation(self) -> None:
        # Seed a second concept so two concepts are processed.
        with db_module.get_conn(transactional=True) as conn:
            conn.execute(
                "INSERT INTO concepts (id, material_id, title, keywords_json, summary, embedding_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "concept-t4-gen-2",
                    MATERIAL_ID,
                    "Cellular respiration stages",
                    '["cellular respiration", "atp"]',
                    "The stages of respiration.",
                    None,
                    "2026-07-06T00:02:00+00:00",
                ),
            )

        search, run = self._patched_engine(_multi_clip_engine_out())
        call_count = {"n": 0}

        def _discover_side_effect(topic, limit, exclude_video_ids=None, **kw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("engine down for the first concept")
            return _discover_result()

        search.discover.side_effect = _discover_side_effect

        with db_module.get_conn() as conn:
            result = main_module.reel_service.generate_reels(
                conn,
                material_id=MATERIAL_ID,
                concept_id=None,
                num_reels=5,
                creative_commons_only=False,
                generation_id="gen-partial",
            )

        # The surviving concept still produced reels despite the first concept failing.
        self.assertTrue(result, "a failing concept must not abort the whole generation")
        with db_module.get_conn() as conn:
            rows = db_module.fetch_all(
                conn, "SELECT id FROM reels WHERE generation_id = ?", ("gen-partial",)
            )
        self.assertGreaterEqual(len(rows), 1)

    def test_provider_failure_after_previous_pass_keeps_persisted_generation(self) -> None:
        _search, run = self._patched_engine(_multi_clip_engine_out())
        generation_id = "gen-provider-partial"

        with db_module.get_conn() as conn:
            first = main_module.reel_service.generate_reels(
                conn,
                material_id=MATERIAL_ID,
                concept_id=None,
                num_reels=5,
                creative_commons_only=False,
                generation_id=generation_id,
            )
        self.assertTrue(first)
        with db_module.get_conn() as conn:
            before = db_module.fetch_one(
                conn,
                "SELECT COUNT(*) AS count FROM reels WHERE generation_id = ?",
                (generation_id,),
            )

        run.clip.side_effect = pipeline_module._ClipProviderError(
            "provider unavailable",
            provider="test",
            operation="clip",
        )
        with db_module.get_conn() as conn:
            second = main_module.reel_service.generate_reels(
                conn,
                material_id=MATERIAL_ID,
                concept_id=None,
                num_reels=5,
                creative_commons_only=False,
                generation_id=generation_id,
            )

        with db_module.get_conn() as conn:
            after = db_module.fetch_one(
                conn,
                "SELECT COUNT(*) AS count FROM reels WHERE generation_id = ?",
                (generation_id,),
            )
        self.assertEqual(second, [])
        self.assertEqual(int(after["count"]), int(before["count"]))

    # ------------------------------------------------------------------ #
    # Finding #4b: rate-limit surfaces as HTTP 429 at the generate endpoint
    # ------------------------------------------------------------------ #
class LevelAwareFeedTests(ClipEngineGenerateReelsTests):
    """Serve-time level scoring: matched clips first, off-level kept at the
    back, and a level change re-sorts WITHOUT regeneration."""

    def _seed_two_reels_with_difficulty(self) -> None:
        # Two persisted reels on the same material: one easy, one hard.
        # videos table: id, title, channel_title, duration_sec, created_at
        # reels table: id, material_id, concept_id, video_id, video_url,
        #              t_start, t_end, transcript_snippet, takeaways_json,
        #              base_score, generation_id, difficulty, created_at
        # Use cellular-respiration content so both reels pass the relevance gate.
        with db_module.get_conn(transactional=True) as conn:
            for reel_id, vid, d in (("r-easy", "videasy00001", 0.15),
                                    ("r-hard", "vidhard00001", 0.85)):
                conn.execute(
                    "INSERT INTO videos (id, title, channel_title, duration_sec, created_at) "
                    "VALUES (?, 'Cellular respiration explained', 'Chan', 600, '2026-07-08T00:00:00+00:00')",
                    (vid,),
                )
                conn.execute(
                    "INSERT INTO reels (id, material_id, concept_id, video_id, video_url, t_start, t_end, "
                    "transcript_snippet, takeaways_json, base_score, generation_id, difficulty, created_at) "
                    "VALUES (?, ?, ?, ?, 'https://www.youtube.com/embed/x?start=0&end=30', 0, 30, "
                    "'cellular respiration process in mitochondria', '[]', 0.8, 'gen-lvl', ?, '2026-07-08T00:00:00+00:00')",
                    (reel_id, MATERIAL_ID, CONCEPT_ID, vid, d),
                )

    def test_beginner_feed_puts_easy_first_but_keeps_hard(self) -> None:
        self._seed_two_reels_with_difficulty()
        with db_module.get_conn(transactional=True) as conn:
            conn.execute("UPDATE materials SET knowledge_level='beginner', level_adjustment=0 WHERE id=?",
                         (MATERIAL_ID,))
        with db_module.get_conn() as conn:
            feed = main_module.reel_service.ranked_feed(conn, material_id=MATERIAL_ID, generation_id="gen-lvl")
        ids = [r["reel_id"] for r in feed]
        self.assertEqual(ids[0], "r-easy")
        self.assertIn("r-hard", ids)          # NEVER hidden — waits at the back

    def test_level_change_resorts_without_regeneration(self) -> None:
        self._seed_two_reels_with_difficulty()
        with db_module.get_conn(transactional=True) as conn:
            conn.execute("UPDATE materials SET knowledge_level='advanced', level_adjustment=0 WHERE id=?",
                         (MATERIAL_ID,))
        with db_module.get_conn() as conn:
            feed = main_module.reel_service.ranked_feed(conn, material_id=MATERIAL_ID, generation_id="gen-lvl")
        self.assertEqual(feed[0]["reel_id"], "r-hard")   # the back-of-feed clip re-entered

    def test_cache_version_includes_recall_and_stored_details(self) -> None:
        self.assertEqual(main_module.reel_service.RANKED_FEED_CACHE_VERSION, 18)


if __name__ == "__main__":
    unittest.main()
