"""Pipeline runner.

Two profiles share transcription and sentence indexing, then diverge:

- ``full`` (default) — the structure-first pipeline: understand the whole video (cached
  per video_id as a Structure), then assemble self-contained clips from that structure
  (anchors → context closure → clip-only judge/repair → boundary snap → context cards →
  chronological order).
- ``fast`` — the legacy single-pass selector (transcript window → snap). Also the
  graceful-degrade target: if the full path raises (missing keys/deps, LLM failure), we
  fall back to fast rather than failing the job.

Clips play from YouTube embeds at full quality; a single clip can be exported to mp4 on
demand. If TRANSCRIBER is a Whisper variant, audio is downloaded first for transcription.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

from . import config
from .embed import embed_url
from .errors import PipelineError, friendly_error
from .jobs import Job, ProgressEvent, Status, registry
from .pipeline.boundary import refine_clip_boundaries
from .pipeline.cut import cut_clips
from .pipeline.download import download, extract_video_id, probe_metadata
from .pipeline.refine import refine_and_snap
from .pipeline.select import select_segments
from .pipeline.punctuation.service import build_sentences
from .pipeline.transcribe import transcribe, transcribe_supadata


def _scale(lo: float, hi: float, frac: float) -> float:
    return lo + (hi - lo) * max(0.0, min(1.0, frac))


def _build_embed_clips(clips_spec: list[dict], video_id: str) -> list[dict]:
    """Shape the final clip dicts (contract-preserving) with optional structure fields."""
    clips = []
    for i, c in enumerate(clips_spec):
        start, end = float(c["start"]), float(c["end"])
        seq = int(c.get("sequence_index", i + 1))
        clip = {
            "n": seq,
            "video_id": video_id,
            "facet": c.get("facet", "other"),
            "reason": c.get("reason", ""),
            "start": round(start, 2),
            "end": round(end, 2),
            "duration": round(end - start, 2),
            "embed_url": embed_url(video_id, start, end),
            "path": None,
            # optional structure-first fields (frontends ignore unknown keys) --
            "role": c.get("role", ""),
            "title": c.get("title", ""),
            "context_card": c.get("context_card", ""),
            "sequence_index": seq,
            "prerequisite_clips": c.get("prerequisite_clips", []),
            # human-readable per-clip notes (e.g. P4b's 'continues clip N' watch-first cue)
            "notes": list(c.get("notes") or []),
            # FE2: quality signals the frontend surfaces (the doc's "verdict" == these derived
            # fields; there is no nested verdict object). final_quality is the shipped blend;
            # ship_flagged marks a clip that shipped with unverified judge concerns.
            "final_quality": round(float(c["final_quality"]), 3) if c.get("final_quality") is not None else None,
            "warnings": list(c.get("warnings") or []),
            "ship_flagged": bool(c.get("ship_flagged", False)),
        }
        # VID2 edge-probe outcomes — only present when the probe ran (default OFF ⇒ absent,
        # keeping the payload byte-identical). The warnings already ride in via 'warnings'.
        if c.get("starts_clean_audio") is not None:
            clip["starts_clean_audio"] = bool(c["starts_clean_audio"])
        if c.get("ends_clean_audio") is not None:
            clip["ends_clean_audio"] = bool(c["ends_clean_audio"])
        clips.append(clip)
    return clips


async def run_pipeline(job: Job, executor: ThreadPoolExecutor) -> None:
    loop = asyncio.get_running_loop()

    # ── per-stage wall-clock profiling (timing only; behavior-neutral) ──────
    # Every blocking stage flows through run() below, so wrapping it times the
    # whole pipeline keyed by function name (download vs perceive vs build_structure
    # vs assemble_clips vs refine…). Records live to PROFILE_TIMINGS_FILE when set,
    # so a killed long run still leaves the stages that completed on disk.
    timings: list[tuple[str, float]] = []
    job._stage_timings = timings  # type: ignore[attr-defined]  # readable by callers/artifacts
    _tfile = os.environ.get("PROFILE_TIMINGS_FILE")

    def _record(name: str, ms: float) -> None:
        timings.append((name, ms))
        if _tfile:
            try:
                with open(_tfile, "a") as f:
                    f.write(json.dumps({
                        "stage": name, "ms": round(ms, 1),
                        "cumulative_ms": round(sum(m for _, m in timings), 1),
                    }) + "\n")
            except Exception:
                pass

    def emit(stage: str, lo: float, hi: float):
        def cb(frac: float, msg: str = "") -> None:
            registry.publish(job, ProgressEvent(stage, _scale(lo, hi, frac), msg))
        return cb

    async def run(fn, *args):
        t0 = time.perf_counter()
        try:
            return await loop.run_in_executor(executor, fn, *args)
        finally:
            _record(getattr(fn, "__name__", str(fn)), (time.perf_counter() - t0) * 1000.0)

    # STEP 2 cross-stage pipelining: speculatively PREFETCH the perception video download so it
    # overlaps transcribe+punctuate. Gated below on a COLD structure cache so a warm re-clip (which
    # skips the download entirely in embed mode) never wastes a ~40s download. download() is URL-only,
    # idempotent + cache-keyed and has NO transcript dependency, so starting it early is byte-neutral.
    profile = str(job.settings.get("analysis_profile", config.ANALYSIS_PROFILE)).lower()
    prefetch_dl = None
    try:
        # ── TRANSCRIBE (shared) ─────────────────────────────────────────────
        if config.TRANSCRIBER == "supadata":
            video_id = extract_video_id(job.url)
            if not video_id:
                return registry.fail(job, "Couldn't read the YouTube video id from that URL.")
            registry.set_meta(job, video_id=video_id)
            # speculative prefetch — only on a cold cache-miss for the full+multimodal path.
            if (profile == "full" and _want_multimodal(job.settings) and config.STRUCTURE_CACHE
                    and not (config.WORK_DIR / video_id / "structure.json").exists()):
                prefetch_dl = asyncio.ensure_future(run(download, job.url, job.settings))
                # retrieve any error on completion so an unconsumed task never logs
                # "exception never retrieved" (the awaiter in _run_full also sees it).
                prefetch_dl.add_done_callback(lambda t: None if t.cancelled() else t.exception())
            registry.publish(job, ProgressEvent("transcribing", 10.0, "Fetching transcript…"))
            transcript = await run(
                transcribe_supadata, job.url, video_id, job.settings, emit("transcribing", 10, 40)
            )
        else:
            registry.publish(job, ProgressEvent("transcribing", 0.0, "Downloading audio…"))
            dl = await run(download, job.url, job.settings, emit("transcribing", 0, 25))
            video_id = dl["video_id"]
            registry.set_meta(job, video_id=video_id, title=dl.get("title", ""))
            transcript = await run(
                transcribe, dl["audio_path"], video_id, job.settings, emit("transcribing", 25, 40)
            )
        transcript.setdefault("title", job.title or "")

        sentences = await run(
            build_sentences, transcript, video_id, job.settings, emit("punctuating", 40, 44))
        if not sentences:
            return registry.fail(job, "Could not build a transcript index for this video.")

        # ── SELECT + ASSEMBLE (profile switch, with graceful degrade) ───────
        handled = False
        if profile == "full":
            try:
                await _run_full(job, transcript, video_id, sentences, run, emit, prefetch_dl)
                handled = True
            except (PipelineError, Exception) as e:  # noqa: BLE001 — degrade, don't fail
                if isinstance(e, asyncio.CancelledError):
                    raise
                registry.publish(job, ProgressEvent(
                    "selecting", 45.0, "Structure pass unavailable — using fast mode…"))
        if not handled:
            await _run_fast(job, transcript, video_id, sentences, run, emit)

    except PipelineError as e:
        registry.fail(job, str(e))
    except asyncio.CancelledError:
        job.status = Status.CANCELLED
        raise
    except Exception as e:  # noqa: BLE001
        registry.fail(job, friendly_error(e))
    finally:
        # reap the speculative prefetch if a bail-out path (no sentences / degrade / cancel) never
        # consumed it — cancel is a no-op once _run_full has awaited it; the in-flight executor
        # thread finishes harmlessly (it just populates the download cache).
        if prefetch_dl is not None and not prefetch_dl.done():
            prefetch_dl.cancel()


def _want_multimodal(settings: dict) -> bool:
    """Full-profile + multimodal enabled (per-job setting overrides the config default)."""
    mm = settings.get("multimodal")
    if mm is None:
        mm = config.MULTIMODAL
    return bool(mm) and str(settings.get("analysis_profile", config.ANALYSIS_PROFILE)).lower() == "full"


def _edge_probe_enabled(settings: dict) -> bool:
    """VID2 edge probe on/off (per-job setting overrides the config default; default OFF).
    Mirrors _want_multimodal's None → inherit-config resolution."""
    ep = settings.get("edge_probe")
    if ep is None:
        ep = config.EDGE_PROBE_ENABLED
    return bool(ep)


async def _run_full(job: Job, transcript: dict, video_id: str, sentences, run, emit,
                    prefetch_dl=None) -> None:
    """Structure-first path: (cached) understanding → topic-dependent assembly → clips.

    In multimodal mode the video is downloaded and perceived (scenes → keyframes →
    Gemini-vision) before understanding; any failure there degrades to transcript-only
    (staying on the full path, not falling back to fast). ``output_mode="cut"`` renders
    mp4s once a video is available.
    """
    from .adapters import get_adapter, select_adapter
    from .pipeline.assemble import assemble_clips
    from .pipeline.assemble.artifacts import write_run_artifacts
    from .pipeline.understand import Perception, build_structure, load_structure, save_structure

    settings = job.settings
    want_mm = _want_multimodal(settings)
    want_cut = str(settings.get("output_mode", config.OUTPUT_MODE)).lower() == "cut"
    video_path: str | None = None

    # freshness-gated load (W25-A): the cache must have been built on THESE sentences —
    # a stale/foreign-indexer cache returns None here and is rebuilt below.
    structure = load_structure(video_id, sentences) if config.STRUCTURE_CACHE else None
    if structure is None or not structure.units:
        # ── understanding + perception, OVERLAPPED (cross-stage pipelining) ──────────
        # content_map needs only `sentences`; the video branch (download → perceive) needs the
        # video plus the SAME `sentences`. Neither depends on the other and extract_units is the
        # join, so we run them concurrently. Both stages receive their exact serial inputs
        # (perceive gets the real transcript+sentences; content_map the real sentences), so the
        # resulting clips are byte-identical — only the SCHEDULE changes. This hides content_map
        # (+ the cheap adapter-detect) underneath the long download+perceive branch.
        from .pipeline.understand.content_map import build_content_map
        from .pipeline.understand.perceive import perceive

        perception = None
        meta: dict = {}                 # GEN1: yt-dlp metadata for the genre-detection signal
        downloaded_meta = False
        # content_map runs concurrently with the video branch below (progress stays None — the
        # bar is driven by perceive/segmenting; a precomputed map just skips build_structure's
        # 0–30% sub-band). It is joined before build_structure.
        cm_task = asyncio.ensure_future(run(build_content_map, sentences, settings, None))
        perceive_task = None
        try:
            if want_mm:
                try:
                    registry.publish(job, ProgressEvent("perceiving", 44.0, "Downloading video for perception…"))
                    # consume the speculative prefetch (already running since before transcribe);
                    # fall back to a fresh download when it wasn't started (warm-cache path / no prefetch).
                    if prefetch_dl is not None:
                        dl = await prefetch_dl
                    else:
                        dl = await run(download, job.url, settings, emit("perceiving", 44, 46))
                    video_path = dl.get("video_path")
                    meta = {k: dl.get(k) for k in ("categories", "tags", "artist", "track", "genre")}
                    downloaded_meta = True
                    if dl.get("title"):
                        registry.set_meta(job, title=dl["title"])
                    registry.publish(job, ProgressEvent("perceiving", 46.0, "Reading on-screen content…"))
                    perceive_task = asyncio.ensure_future(run(
                        perceive, video_path, video_id, transcript, sentences,
                        settings, emit("perceiving", 46, 55), dl.get("audio_path"),
                    ))
                except Exception:
                    # degrade to transcript-only (stay full), but record that perception was
                    # attempted and failed so /health and Structure.degraded reflect it.
                    perception = Perception(video_id=video_id, degraded=["perception"])

            # GEN1: in supadata/fast mode (no full download) probe metadata cheaply so the genre
            # signal still fires. Fail-soft; adds one network call only when nothing was downloaded.
            if not downloaded_meta:
                try:
                    meta = await run(probe_metadata, job.url)
                except Exception:
                    meta = {}

            # adapter detection runs while the perception branch is still captioning (both are
            # independent of each other), then we join perception + content_map.
            adapter, detection = await run(select_adapter, transcript, settings, meta)
            if perceive_task is not None:
                try:
                    perception = await perceive_task
                    perceive_task = None
                except Exception:
                    perception = Perception(video_id=video_id, degraded=["perception"])
            content_map = await cm_task     # raises → _run_full degrades to _run_fast, as before
        finally:
            # never leak the concurrent tasks if we bail out early (degrade / exception path);
            # cancelling only detaches the awaitable — the executor thread finishes harmlessly.
            for _t in (cm_task, perceive_task):
                if _t is not None and not _t.done():
                    _t.cancel()

        registry.publish(job, ProgressEvent("segmenting", 55.0, "Understanding the video…"))
        structure = await run(
            build_structure, video_id, transcript, sentences, adapter, detection,
            settings, emit("segmenting", 55, 72), perception, content_map,
        )
        if not structure.units:
            raise PipelineError("No structure could be built from this transcript.")
        if config.STRUCTURE_CACHE:
            save_structure(structure)
    else:
        adapter = get_adapter(structure.detection.domain)

    # need the source video to cut mp4s even when the structure came from cache
    if want_cut and not video_path:
        try:
            dl = await run(download, job.url, settings, emit("perceiving", 55, 58))
            video_path = dl.get("video_path")
        except Exception:
            video_path = None

    registry.publish(job, ProgressEvent("assembling", 72.0, "Assembling self-contained clips…"))
    stats: dict = {}                           # machine-readable run signals (I1/W25-G)
    clips_spec, notes, rejections = await run(
        assemble_clips, structure, job.topic, sentences, job.url, video_id,
        settings, adapter, emit("assembling", 72, 90), stats,
    )
    # W25-G: persist the run's plan/arcs/shipped/ledger (work/<id>/runs/<ts>/) — jobs are
    # in-memory and the ledger used to die right here; the writer never raises. Written
    # BEFORE the empty-result bail so failed runs leave their ledger too.
    await run(write_run_artifacts, video_id, clips_spec, rejections, stats)
    if rejections:
        registry.publish(job, ProgressEvent(
            "assembling", 90.0, f"{len(rejections)} candidate(s) dropped ({', '.join(sorted({r.stage for r in rejections}))})"))
    if not clips_spec:
        return registry.fail(
            job, notes or f"“{job.topic}” isn’t covered enough in this video to clip.", notes=notes)

    if config.PRECISE_BOUNDARIES and transcript.get("source") == "supadata":
        registry.publish(job, ProgressEvent("refining", 90.0, "Refining boundaries with Whisper…"))
        clips_spec = await run(
            refine_clip_boundaries, clips_spec, job.url, video_id, settings, emit("refining", 90, 96))

    # VID2 edge probe (default OFF): runs AFTER boundary refinement so probed offsets == shipped
    # offsets, and only when a source video is already on disk (multimodal/cut paths downloaded
    # it — we never force a download here). ADVISORY: mutates each clip's warnings + a small
    # final_quality dock; never kills a clip / creates a Rejection / raises into the pipeline.
    if _edge_probe_enabled(settings) and video_path:
        try:
            from .pipeline.assemble.edge_probe import run_edge_probe
            registry.publish(job, ProgressEvent("refining", 96.0, "Checking clip edges…"))
            clips_spec = await run(run_edge_probe, clips_spec, video_path, settings)
        except Exception:
            pass                                            # fail-soft: keep the shipped clips

    clips = _build_embed_clips(clips_spec, video_id)
    if want_cut and video_path:
        try:
            registry.publish(job, ProgressEvent("cutting", 96.0, "Cutting clips…"))
            cut_results = await cut_clips(video_path, video_id, clips_spec, settings, emit("cutting", 96, 100))
            by_n = {c["n"]: c for c in cut_results}
            for i, clip in enumerate(clips):                # cut n == segment index + 1
                cr = by_n.get(i + 1)
                if cr and cr.get("path"):
                    clip["path"] = cr["path"]
        except Exception:
            pass                                            # keep embed clips (path stays None)

    registry.finish(job, clips, notes=notes)


async def _run_fast(job: Job, transcript: dict, video_id: str, sentences, run, emit) -> None:
    """Legacy single-pass selection path (also the degrade target)."""
    registry.publish(job, ProgressEvent("selecting", 45.0, "Finding relevant moments…"))
    candidates, notes = await run(
        select_segments, sentences, job.topic, job.settings, emit("selecting", 45, 70))
    if not candidates:
        msg = notes or f"The topic “{job.topic}” doesn’t appear enough in this video to clip."
        return registry.fail(job, msg, notes=notes)

    registry.publish(job, ProgressEvent("refining", 70.0, "Snapping to boundaries…"))
    clips_spec = await run(refine_and_snap, candidates, sentences, job.settings, emit("refining", 70, 85))
    if not clips_spec:
        return registry.fail(job, "No clean clip boundaries could be formed.", notes=notes)

    if config.PRECISE_BOUNDARIES and transcript.get("source") == "supadata":
        registry.publish(job, ProgressEvent("refining", 86.0, "Refining boundaries with Whisper…"))
        clips_spec = await run(
            refine_clip_boundaries, clips_spec, job.url, video_id, job.settings, emit("refining", 86, 99))

    registry.finish(job, _build_embed_clips(clips_spec, video_id), notes=notes)
