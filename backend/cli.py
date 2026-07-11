"""Quick end-to-end pipeline runner (no web server), for verification/debugging.

Usage:
    source .venv/bin/activate
    python -m backend.cli "https://youtu.be/VIDEO_ID" "the derivative"
    CLIP_ENGINE=topic python -m backend.cli "https://youtu.be/VIDEO_ID" "topic"
    CLIP_ENGINE=unit python -m backend.cli "https://youtu.be/VIDEO_ID" "topic"

Runs the one-pass Gemini selector by default. The heavier topic and unit pipelines remain
available as explicit experiments through ``CLIP_ENGINE``.
"""
from __future__ import annotations

import asyncio
import sys

from . import config
from .pipeline.assemble import assemble_clips, _resolve_assemble_fn
from .pipeline.download import download, extract_video_id
from .pipeline.punctuation.service import build_sentences
from .pipeline.refine import refine_and_snap
from .pipeline.select import select_segments
from .pipeline.transcribe import transcribe, transcribe_supadata


def _drop_line(r) -> str:
    """One drop-ledger line. Stage-agnostic on purpose — it must format EVERY
    Rejection.stage literal (incl. post_snap_judge) without changes here."""
    return f"  [dropped/{r.stage}] {r.title[:60]} (score={r.score}, kinds={r.failure_kinds}, q={r.final_quality})"


def _p(stage: str):
    def cb(frac: float, msg: str = ""):
        sys.stdout.write(f"\r[{stage:12}] {frac * 100:5.1f}%  {msg[:60]:<60}")
        sys.stdout.flush()
        if frac >= 1.0:
            sys.stdout.write("\n")
    return cb


async def run(url: str, topic: str, profile: str = "full") -> None:
    settings = dict(config.DEFAULTS)
    settings["analysis_profile"] = profile

    # ── transcript (supadata: no download; whisper: download audio) ──
    if config.TRANSCRIBER == "supadata":
        video_id = extract_video_id(url)
        transcript = transcribe_supadata(url, video_id, settings, _p("transcribe"))
    else:
        dl = await asyncio.get_running_loop().run_in_executor(None, download, url, settings, _p("download"))
        video_id = dl["video_id"]
        transcript = transcribe(dl["audio_path"], video_id, settings, _p("transcribe"))
    transcript.setdefault("title", "")
    engine = str(settings.get("clip_engine") or config.CLIP_ENGINE).lower()
    if engine == "gemini":
        from .pipeline.gemini_segment import segment_clips
        clips_spec, notes = segment_clips(transcript, settings, _p("select"), topic, video_id)
        rejections = []
        sents = []
        print(f"  video_id={video_id}  {len(transcript.get('segments', []))} caption segments")
    else:
        sents = build_sentences(transcript, video_id, settings, _p("punctuate"))
        print(f"  video_id={video_id}  {len(sents)} sentences, {len(transcript.get('words', []))} words")

    if engine == "gemini":
        pass
    elif profile == "full":
        from .adapters import select_adapter
        from .pipeline.understand import build_structure

        # ── perceive (multimodal only; downloads the video, cached) ──
        perception = None
        want_mm = settings.get("multimodal")
        want_mm = config.MULTIMODAL if want_mm is None else want_mm
        if want_mm:
            try:
                from .pipeline.download import download
                from .pipeline.understand.perceive import perceive
                dl = download(url, settings, _p("download"))
                perception = perceive(dl["video_path"], video_id, transcript, sents, settings,
                                      _p("perceive"), dl.get("audio_path"))
                print(f"  perception: {len(perception.scenes)} scenes, "
                      f"{len(perception.visual_events)} visual events, degraded={perception.degraded}")
            except Exception as e:  # noqa: BLE001 — degrade to transcript-only
                print(f"  perception skipped ({type(e).__name__}: {e}) — transcript-only")
                perception = None

        adapter, det = select_adapter(transcript, settings)
        print(f"  detected: {det.content_type} → adapter={adapter.domain} (density={det.density})")
        st = build_structure(video_id, transcript, sents, adapter, det, settings, _p("understand"), perception)
        linked = sum(1 for u in st.units for d in u.visual_dependencies if d.visual_event_id)
        withdep = sum(1 for u in st.units if u.visual_dependencies)
        print(f"  structure: {len(st.units)} units, {len(st.dependencies.edges)} dep edges, "
              f"{len(st.content_map.topics())} topics")
        print(f"  visual: {len(st.visual_events)} events, has_perception={st.has_perception}, "
              f"degraded={st.degraded}; units w/ visual_deps={withdep} ({linked} linked)")
        clips_spec, notes, rejections = _resolve_assemble_fn(settings)(
            st, topic, sents, url, video_id, settings, adapter, _p("assemble"))
    else:
        rejections = []
        clips_spec, notes = select_segments(sents, topic, settings, _p("select"))
        clips_spec = refine_and_snap(clips_spec, sents, settings, _p("refine")) if clips_spec else []

    print(f"  notes: {notes!r}")
    for r in (rejections or []):
        print(_drop_line(r))
    print(f"\n=== {len(clips_spec)} CLIPS (profile={profile}) ===")
    for c in clips_spec:
        seq = c.get("sequence_index", "?")
        print(f"  #{seq} [{c['start']:7.2f}-{c['end']:7.2f}] ({c['end']-c['start']:5.1f}s) "
              f"facet={c.get('facet','?'):14} role={c.get('role','-'):18} q={c.get('final_quality', 0):.2f}")
        print(f"       {(c.get('title') or c.get('reason',''))[:74]}")
        if c.get("context_card"):
            print(f"       card: {c['context_card'][:100]}")
        if c.get("prerequisite_clips"):
            print(f"       watch first: {c['prerequisite_clips']}")
        if sents:
            end_sentence = next((s for s in sents if abs(s.end - c["end"]) < 0.3), None)
            ok = end_sentence and end_sentence.ends_with_period
            print(f"       ends_on_period={bool(ok)}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: python -m backend.cli "<youtube_url>" "<topic>" [full|fast]')
        sys.exit(1)
    prof = sys.argv[3] if len(sys.argv) > 3 else "full"
    asyncio.run(run(sys.argv[1], sys.argv[2], prof))
