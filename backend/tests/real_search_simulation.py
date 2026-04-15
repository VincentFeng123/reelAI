"""
REAL end-to-end simulation — hits YouTube via the Data API, fetches actual
transcripts via youtube-transcript-api, runs the full reel-cutting pipeline,
and prints the resulting reel URLs.

NOT a unit test with mock data. The search and transcripts are real.

Usage:
    cd backend && python tests/real_search_simulation.py [query]

Requires YOUTUBE_API_KEY in backend/.env.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Load .env so the services pick up YOUTUBE_API_KEY etc.
_backend_root = Path(__file__).resolve().parent.parent
_env_path = _backend_root / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

sys.path.insert(0, str(_backend_root))

import requests  # noqa: E402

from app.db import init_db  # noqa: E402
from app.services.embeddings import EmbeddingService  # noqa: E402
from app.services.reels import ReelService  # noqa: E402
from app.services.segmenter import normalize_terms  # noqa: E402
from app.services.youtube import YouTubeService  # noqa: E402


def _separator(title: str) -> None:
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)


def yt_search(query: str, max_results: int = 10) -> list[dict]:
    """Real YouTube Data API v3 search — returns video metadata."""
    api_key = os.environ["YOUTUBE_API_KEY"]

    # Step 1: search for videos
    search_r = requests.get(
        "https://www.googleapis.com/youtube/v3/search",
        params={
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": max_results,
            "key": api_key,
        },
        timeout=15,
    )
    search_r.raise_for_status()
    items = search_r.json().get("items", [])
    if not items:
        return []

    video_ids = [it["id"]["videoId"] for it in items]

    # Step 2: get duration + statistics for each
    detail_r = requests.get(
        "https://www.googleapis.com/youtube/v3/videos",
        params={
            "part": "contentDetails,snippet,statistics",
            "id": ",".join(video_ids),
            "key": api_key,
        },
        timeout=15,
    )
    detail_r.raise_for_status()
    details = {d["id"]: d for d in detail_r.json().get("items", [])}

    def _iso_duration_to_sec(iso: str) -> int:
        import re
        m = re.match(r"^PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$", iso or "")
        if not m:
            return 0
        h, mi, s = (int(x or 0) for x in m.groups())
        return h * 3600 + mi * 60 + s

    out: list[dict] = []
    for it in items:
        vid = it["id"]["videoId"]
        sn = it["snippet"]
        det = details.get(vid, {})
        duration_sec = _iso_duration_to_sec(det.get("contentDetails", {}).get("duration", ""))
        out.append({
            "id": vid,
            "title": sn.get("title", ""),
            "channel_title": sn.get("channelTitle", ""),
            "description": sn.get("description", ""),
            "published_at": sn.get("publishedAt", ""),
            "duration_sec": duration_sec,
            "view_count": int(det.get("statistics", {}).get("viewCount", 0) or 0),
        })
    return out


def fetch_transcript(video_id: str) -> tuple[list[dict], str]:
    """Fetch transcript — prefer manual (punctuated) over auto (unpunctuated).

    Returns (transcript, kind) where kind is 'manual', 'generated', or 'other'.
    Matches the production pipeline's preference order.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        print("  youtube-transcript-api not installed")
        return [], ""
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        try:
            tr = transcript_list.find_manually_created_transcript(["en"])
            kind = "manual"
        except Exception:
            try:
                tr = transcript_list.find_generated_transcript(["en"])
                kind = "generated"
            except Exception:
                tr = next(iter(transcript_list), None)
                kind = "other"
        if tr is None:
            return [], ""
        fetched = tr.fetch()
        return (
            [
                {"start": float(s.start), "duration": float(s.duration), "text": s.text}
                for s in fetched
            ],
            kind,
        )
    except Exception as e:
        print(f"  transcript fetch exception: {type(e).__name__}: {str(e)[:120]}")
        return [], ""


def run(query: str, *, clip_min: int = 30, clip_max: int = 55, num_videos: int = 8) -> None:
    print(f"REAL END-TO-END SIMULATION")
    print(f"  query = {query!r}")
    print(f"  user clip duration = {clip_min}-{clip_max}s")
    print(f"  num_videos to examine = {num_videos}")

    init_db()

    yt = YouTubeService()
    es = EmbeddingService()
    rs = ReelService(embedding_service=es, youtube_service=yt)

    _separator("STEP 1 — Real YouTube search")
    t0 = time.time()
    videos = yt_search(query, max_results=num_videos)
    print(f"found {len(videos)} videos in {time.time() - t0:.1f}s:")
    for v in videos:
        dur_m = v["duration_sec"] // 60
        dur_s = v["duration_sec"] % 60
        print(
            f"  - {v['id']:12} {dur_m:3}:{dur_s:02}  "
            f"chan={v['channel_title'][:28]:28}  "
            f"title={v['title'][:60]}"
        )

    _separator("STEP 2 — Content-tier classification")
    for v in videos:
        tier = rs._infer_channel_tier(channel=v["channel_title"].lower(), title=v["title"].lower())
        print(f"  {v['id']:12} -> {tier:22} | {v['title'][:60]}")

    _separator("STEP 3 — Drop entertainment_media / low_quality_compilation (ambiguous concept)")
    ambig = bool(normalize_terms([query]) & rs.AMBIGUOUS_CONCEPT_TOKENS)
    print(f"  '{query}' is in AMBIGUOUS_CONCEPT_TOKENS: {ambig}")
    if ambig:
        kept = []
        for v in videos:
            tier = rs._infer_channel_tier(
                channel=v["channel_title"].lower(), title=v["title"].lower(),
            )
            if tier in {"entertainment_media", "low_quality_compilation"}:
                print(f"  DROP: {v['id']}  tier={tier}  title={v['title'][:50]}")
                continue
            kept.append(v)
        videos = kept
        print(f"  after drop: {len(videos)} videos")

    _separator("STEP 4 — Fetch REAL transcripts and cut into topic reels")
    total_reels = 0
    max_videos_to_cut = 4
    cut_count = 0
    for v in videos:
        if cut_count >= max_videos_to_cut:
            break
        video_id = v["id"]
        duration_sec = v["duration_sec"]
        print()
        print(f"▶ {video_id}  duration={duration_sec}s")
        print(f"  channel: {v['channel_title']}")
        print(f"  title:   {v['title']}")

        if duration_sec <= 0:
            print("  SKIP: unknown duration")
            continue
        if duration_sec <= 60:
            print("  TYPE: YouTube Short — would emit the full video as one reel")
            total_reels += 1
            cut_count += 1
            continue

        transcript, kind = fetch_transcript(video_id)
        if not transcript:
            print("  SKIP: no transcript available")
            continue
        has_punct = any(
            c["text"].strip().endswith((".", "!", "?", "…", ".\"", "!\"", "?\""))
            for c in transcript[:40]
        )
        print(f"  transcript: {len(transcript)} cues, kind={kind}, punctuated={has_punct}, "
              f"spans [{transcript[0]['start']:.1f}, "
              f"{transcript[-1]['start'] + transcript[-1]['duration']:.1f}]s")

        segments = rs._topic_cut_segments_for_concept(
            transcript=transcript,
            video_id=video_id,
            video_duration_sec=duration_sec,
            clip_min_len=clip_min,
            clip_max_len=clip_max,
            max_segments=6,
            concept_terms=[query],
        )
        if not segments:
            print(f"  topic_cut found no clusters mentioning '{query}' ≥ 2 times")
            continue
        print(f"  {len(segments)} topic sub-segment(s) from mention clustering:")
        for s in segments:
            cg = getattr(s, "cluster_group_id", "") or "(single)"
            sub = getattr(s, "cluster_sub_index", 0)
            print(f"    raw [{s.t_start:7.1f}, {s.t_end:7.1f})  cluster={cg}  sub_idx={sub}")

        # Simulate the main candidate loop's refinement + chaining,
        # mirroring reels.py's logic exactly (in-cluster chain + cross-cluster
        # bridge for temporally adjacent segments).
        chain: dict[str, float] = {}
        topic_cut_last_refined_end: float | None = None
        BRIDGE_TOLERANCE_SEC = 2.0
        windows: list[tuple[float, float]] = []
        # Sort in TEMPORAL order (matches real pipeline).
        sorted_segs = sorted(
            segments,
            key=lambda s: (float(s.t_start), int(getattr(s, "cluster_sub_index", 0))),
        )
        for seg in sorted_segs:
            span = seg.t_end - seg.t_start
            cg = str(getattr(seg, "cluster_group_id", "") or "")
            prev = chain.get(cg) if cg else None
            if prev is not None:
                eff = float(prev)
            elif (
                topic_cut_last_refined_end is not None
                and abs(float(seg.t_start) - topic_cut_last_refined_end) <= BRIDGE_TOLERANCE_SEC
            ):
                eff = float(topic_cut_last_refined_end)
            else:
                eff = float(seg.t_start)
            if span > clip_max + 16:
                w = rs._split_into_consecutive_windows(
                    transcript=transcript, segment_start=eff, segment_end=seg.t_end,
                    video_duration_sec=duration_sec, min_len=clip_min, max_len=clip_max,
                )
            else:
                rmax = int(max(span + 16, clip_max))
                rmin = max(1, min(clip_min, int(max(1.0, span * 0.6))))
                single = rs._refine_clip_window_from_transcript(
                    transcript=transcript, proposed_start=eff, proposed_end=seg.t_end,
                    video_duration_sec=duration_sec, min_len=rmin, max_len=rmax, min_start=eff,
                )
                w = [single] if single else []
            windows.extend(w)
            if w:
                topic_cut_last_refined_end = float(w[-1][1])
                if cg:
                    chain[cg] = float(w[-1][1])

        print(f"  → PRODUCED {len(windows)} sentence-aligned reel(s):")
        for i, (a, b) in enumerate(windows):
            # Fetch the actual opening text at a=t_start
            open_text = ""
            close_text = ""
            for c in transcript:
                cs = float(c["start"])
                ce = cs + float(c.get("duration", 0))
                if cs <= a + 0.05 and ce >= a - 0.05 and not open_text:
                    open_text = str(c["text"]).replace("\n", " ")[:100]
                if cs <= b + 0.05 and ce >= b - 0.05:
                    close_text = str(c["text"]).replace("\n", " ")[:100]
            url = f"https://www.youtube.com/watch?v={video_id}&t={int(a)}s"
            print(f"    reel #{i + 1}: [{a:7.2f} → {b:7.2f}]  dur={b-a:5.1f}s")
            print(f"             URL:   {url}")
            if open_text:
                print(f"             opens: \"{open_text}\"")
            if close_text:
                print(f"             closes: \"{close_text}\"")
            total_reels += 1

        # Continuity verification. There are three relevant cases:
        #   1. Within a topic cluster (same cluster_group_id or bridged neighbours):
        #      reels MUST chain with zero overlap and near-zero gap.
        #   2. Across topic clusters that are far apart in time:
        #      a larger gap is EXPECTED — they're unrelated discussions.
        #   3. Small gap (<3s) between sentence-end and next sentence-start:
        #      natural silence, not a defect.
        sorted_by_sub = sorted(
            segments,
            key=lambda s: (float(s.t_start), int(getattr(s, "cluster_sub_index", 0))),
        )
        for i in range(len(windows) - 1):
            a_end = windows[i][1]
            b_start = windows[i + 1][0]
            gap = b_start - a_end
            if abs(gap) < 0.01:
                status = "CONTIGUOUS ✓ (chained)"
            elif -0.5 < gap < 0:
                status = f"TINY OVERLAP {gap:.2f}s ✗"
            elif 0 <= gap < 3.0:
                status = f"natural silence {gap:.2f}s (sentence→sentence)"
            else:
                status = f"cluster boundary (gap {gap:.1f}s — different topic)"
            print(f"    reel #{i + 1} → reel #{i + 2}: {status}")

        cut_count += 1

    _separator("SUMMARY")
    print(f"  videos searched:      {len(videos) + (sum(1 for v in videos if True))}")
    print(f"  videos after filter:  {len(videos)}")
    print(f"  videos processed:     {cut_count}")
    print(f"  total reels emitted:  {total_reels}")


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "calculus"
    run(query)
