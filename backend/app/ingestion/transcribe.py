"""
Transcript extraction with a four-strategy fallback chain:

    1. YouTube only: reuse the existing `YouTubeService.get_transcript` (caches via
       `transcript_cache` table; free; best quality when captions are authored).
    2. Any platform: parse yt-dlp's scraped subtitles from the `info_dict`
       (`automatic_captions` / `subtitles`). Free; ~90% of IG Reels with captions and
       most English YouTube videos land here.
    3. faster-whisper running LOCALLY on the extracted audio. Free; runs on
       CPU via CTranslate2; downloads a small model (~80 MB for `base.en`) on
       first use to `~/.cache/huggingface/hub/`. Tried before the paid OpenAI
       path so users without an OpenAI key still get usable transcripts.
       Skipped silently if `faster-whisper` is not installed.
    4. Fallback: OpenAI Whisper API (`whisper-1`) on the extracted audio. Paid
       (~$0.006/min) but works on anything that has speech. Used only when
       both 1-3 fail OR an OpenAI client is explicitly preferred.

Results are cached in the existing `llm_cache` table under key
`ingest_transcript:{platform}:{source_id}` so repeated ingests of the same URL skip
the expensive Whisper call.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from ..db import DatabaseIntegrityError, dumps_json, fetch_one, loads_json, now_iso, upsert
from ..services.transcript_validation import validate_transcript
from .errors import ServerlessUnavailable, TranscriptionError
from .ffmpeg_tools import extract_audio_wav
from .logging_config import get_ingest_logger, log_event
from .models import IngestTranscriptCue, IngestTranscriptWord, PlatformLiteral, WordSourceLiteral

logger: logging.Logger = get_ingest_logger(__name__)

_TRANSCRIPT_CACHE_PREFIX = "ingest_transcript:"
_WHISPER_MODEL = "whisper-1"
_WHISPER_MAX_FILE_BYTES = 24 * 1024 * 1024  # 24 MiB (Whisper API caps at 25 MiB)

# faster-whisper local model size. Override with FASTER_WHISPER_MODEL env var.
# Sizes (rough downloads): tiny.en=39MB, base.en=74MB, small.en=244MB,
# medium.en=769MB, large-v3=2.9GB. base.en is the sweet spot for quality/speed
# on CPU — small.en is noticeably better but ~3x slower.
_FASTER_WHISPER_MODEL = os.environ.get("FASTER_WHISPER_MODEL", "base.en")
# CPU is the safe default. Set FASTER_WHISPER_DEVICE=cuda to use GPU.
_FASTER_WHISPER_DEVICE = os.environ.get("FASTER_WHISPER_DEVICE", "cpu")
# int8 quantization gives ~2x speed on CPU with negligible accuracy loss.
_FASTER_WHISPER_COMPUTE_TYPE = os.environ.get("FASTER_WHISPER_COMPUTE_TYPE", "int8")


def _serialize_cues(cues: list[IngestTranscriptCue]) -> str:
    payload: list[dict[str, Any]] = []
    for c in cues:
        cue_dict: dict[str, Any] = {"start": c.start, "end": c.end, "text": c.text}
        # Only include words+word_source when words are present, to keep legacy
        # cache rows (that never had these fields) small and readable.
        if c.words:
            cue_dict["words"] = [
                {
                    "start": w.start,
                    "end": w.end,
                    "text": w.text,
                    **({"confidence": w.confidence} if w.confidence is not None else {}),
                }
                for w in c.words
            ]
            cue_dict["word_source"] = c.word_source
        payload.append(cue_dict)
    return dumps_json(payload)


def _deserialize_cues(raw: Any) -> list[IngestTranscriptCue]:
    payload = loads_json(raw, default=[])
    cues: list[IngestTranscriptCue] = []
    if not isinstance(payload, list):
        return cues
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            words_raw = item.get("words") or []
            words: list[IngestTranscriptWord] = []
            if isinstance(words_raw, list):
                for w in words_raw:
                    if not isinstance(w, dict):
                        continue
                    try:
                        words.append(
                            IngestTranscriptWord(
                                start=float(w.get("start", 0.0)),
                                end=float(w.get("end", 0.0)),
                                text=str(w.get("text", "")),
                                confidence=(
                                    float(w["confidence"])
                                    if isinstance(w.get("confidence"), (int, float))
                                    else None
                                ),
                            )
                        )
                    except Exception:
                        continue
            word_source_raw = str(item.get("word_source") or "legacy")
            word_source: WordSourceLiteral = (
                word_source_raw  # type: ignore[assignment]
                if word_source_raw in ("whisper", "openai", "proportional", "legacy")
                else "legacy"
            )
            cues.append(
                IngestTranscriptCue(
                    start=float(item.get("start", 0.0)),
                    end=float(item.get("end", 0.0)),
                    text=str(item.get("text", "")).strip(),
                    words=words,
                    word_source=word_source,
                )
            )
        except Exception:
            continue
    return cues


_WHITESPACE_RE = re.compile(r"\s+")


def _fill_proportional_words(cue: IngestTranscriptCue) -> None:
    """
    Populate `cue.words` via a proportional-character split when the upstream
    transcript source (youtube-transcript-api, yt-dlp VTT) doesn't provide
    word-level timing (Phase A.1 fallback).

    Accuracy is limited by the duration of the cue itself: on a 6s cue the
    per-word error is at most ~6s / n_words. Good enough for sentence-boundary
    picks on segment-level cues; not good enough for sub-word seeking. The
    `word_source="proportional"` marker lets the boundary engine degrade
    accuracy claims appropriately.
    """
    if cue.words:
        return
    text = (cue.text or "").strip()
    if not text:
        return
    # Split on whitespace, preserving punctuation attached to words.
    tokens = [t for t in _WHITESPACE_RE.split(text) if t]
    if not tokens:
        return
    total_chars = sum(len(t) for t in tokens) or len(tokens)
    duration = max(cue.end - cue.start, 0.01)
    cursor = cue.start
    words: list[IngestTranscriptWord] = []
    for tok in tokens:
        span = max(duration * (len(tok) / total_chars), 0.02)
        w_start = cursor
        w_end = min(cursor + span, cue.end)
        try:
            words.append(
                IngestTranscriptWord(
                    start=w_start,
                    end=w_end if w_end > w_start else w_start + 0.02,
                    text=tok,
                    confidence=None,
                )
            )
        except Exception:
            continue
        cursor = w_end
    if words:
        cue.words = words
        cue.word_source = "proportional"


def _cache_key(platform: PlatformLiteral, source_id: str, language: str) -> str:
    return f"{_TRANSCRIPT_CACHE_PREFIX}{platform}:{source_id}:{language}"


def _load_cached(conn: Any, key: str) -> list[IngestTranscriptCue] | None:
    try:
        row = fetch_one(
            conn,
            "SELECT response_json FROM llm_cache WHERE cache_key = ?",
            (key,),
        )
    except Exception:
        logger.exception("transcript cache read failed for key=%s", key)
        return None
    if not row or not row.get("response_json"):
        return None
    cues = _deserialize_cues(row["response_json"])
    return cues or None


def _store_cache(conn: Any, key: str, cues: list[IngestTranscriptCue]) -> None:
    if not cues:
        return
    try:
        upsert(
            conn,
            "llm_cache",
            {
                "cache_key": key,
                "response_json": _serialize_cues(cues),
                "created_at": now_iso(),
            },
            pk="cache_key",
        )
    except DatabaseIntegrityError:
        pass
    except Exception:
        logger.exception("transcript cache write failed for key=%s", key)


# --------------------------------------------------------------------- #
# Strategy 1: YouTube-only, via the existing service
# --------------------------------------------------------------------- #


def _cues_from_youtube_service(conn: Any, youtube_service: Any, yt_video_id: str) -> list[IngestTranscriptCue]:
    if youtube_service is None:
        return []
    try:
        raw = youtube_service.get_transcript(conn, yt_video_id)
    except Exception:
        logger.exception("youtube_service.get_transcript raised for video_id=%s", yt_video_id)
        return []
    cues: list[IngestTranscriptCue] = []
    if not isinstance(raw, list):
        return cues
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            start = float(entry.get("start", 0.0))
            duration = float(entry.get("duration", 0.0))
            text = str(entry.get("text", "")).replace("\n", " ").strip()
            if not text:
                continue
            cue = IngestTranscriptCue(start=start, end=start + max(duration, 0.01), text=text)
            # youtube-transcript-api doesn't expose word-level timing — fill
            # proportional-character fallback so the boundary engine (Phase A.3)
            # has *something* to work with instead of cue-start/cue-end only.
            _fill_proportional_words(cue)
            cues.append(cue)
        except Exception:
            continue
    return cues


# --------------------------------------------------------------------- #
# Strategy 2: yt-dlp's scraped subtitles / automatic captions
# --------------------------------------------------------------------- #


_VTT_TIMESTAMP_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[.,](\d{3})\s+-->\s+(\d{1,2}):(\d{2}):(\d{2})[.,](\d{3})"
)


def _vtt_time_to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _parse_vtt(text: str) -> list[IngestTranscriptCue]:
    """Parse a WEBVTT or SRT payload into cues. Tolerant of minor format variants."""
    cues: list[IngestTranscriptCue] = []
    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        match = _VTT_TIMESTAMP_RE.search(line)
        if not match:
            i += 1
            continue
        start = _vtt_time_to_seconds(*match.groups()[:4])
        end = _vtt_time_to_seconds(*match.groups()[4:])
        i += 1
        buffer: list[str] = []
        while i < n and lines[i].strip():
            clean = re.sub(r"<[^>]+>", "", lines[i])
            clean = clean.replace("&nbsp;", " ").strip()
            if clean:
                buffer.append(clean)
            i += 1
        text_joined = " ".join(buffer).strip()
        if text_joined:
            cue = IngestTranscriptCue(start=start, end=max(end, start + 0.01), text=text_joined)
            # VTT/SRT is segment-level; no native word timing. Fill proportional.
            _fill_proportional_words(cue)
            cues.append(cue)
    return cues


def _cues_from_info_dict_subtitles(info: dict[str, Any], language: str) -> list[IngestTranscriptCue]:
    """
    Parse cues from yt-dlp's info_dict. yt-dlp stores subtitles under `subtitles` (manual)
    and `automatic_captions` (auto-generated) keyed by language, each a list of format
    dicts with `url` and `ext`.
    """
    candidates: list[dict[str, Any]] = []
    for key in ("subtitles", "automatic_captions"):
        container = info.get(key)
        if not isinstance(container, dict):
            continue
        # Try the exact language first, then common English variants, then any.
        for lang_key in (language, "en", "en-US", "en-GB"):
            entries = container.get(lang_key)
            if isinstance(entries, list):
                candidates.extend(entries)
        if not candidates:
            for entries in container.values():
                if isinstance(entries, list):
                    candidates.extend(entries)
                    break
    if not candidates:
        return []

    # Some yt-dlp extractors inline the cues under `data`; others provide a `url` to fetch.
    import urllib.request

    for entry in candidates:
        if not isinstance(entry, dict):
            continue
        ext = (entry.get("ext") or "").lower()
        if ext not in ("vtt", "srv3", "srt", "ttml"):
            continue
        body: str | None = None
        inline = entry.get("data")
        if isinstance(inline, str) and inline.strip():
            body = inline
        else:
            url = entry.get("url")
            if isinstance(url, str):
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "ReelAIBot/1.0 (+https://reelai.app/bot)"})
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        body = resp.read().decode("utf-8", errors="replace")
                except Exception:
                    logger.exception("yt-dlp subtitle fetch failed for %s", url)
                    continue
        if not body:
            continue
        cues = _parse_vtt(body)
        if cues:
            return cues
    return []


# --------------------------------------------------------------------- #
# Strategy 3: faster-whisper running LOCALLY (free, no API)
# --------------------------------------------------------------------- #


def _load_faster_whisper_model() -> Any | None:
    """
    Lazy-load and cache the faster-whisper model. Returns None if the package
    isn't installed or model loading fails — every caller is expected to
    handle None gracefully and fall through to the OpenAI Whisper API path.

    The first call downloads the model (~80 MB for base.en) to
    `~/.cache/huggingface/hub/`. Subsequent calls reuse the cached weights so
    only the first transcription pays the cold-start cost.
    """
    cache = getattr(_load_faster_whisper_model, "_cache", None)
    if cache is not None:
        return cache
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception:
        logger.debug("faster-whisper is not installed; local Whisper path disabled")
        _load_faster_whisper_model._cache = None  # type: ignore[attr-defined]
        return None
    try:
        model = WhisperModel(
            _FASTER_WHISPER_MODEL,
            device=_FASTER_WHISPER_DEVICE,
            compute_type=_FASTER_WHISPER_COMPUTE_TYPE,
        )
    except Exception:
        logger.exception(
            "failed to load faster-whisper model %s on device=%s",
            _FASTER_WHISPER_MODEL, _FASTER_WHISPER_DEVICE,
        )
        _load_faster_whisper_model._cache = None  # type: ignore[attr-defined]
        return None
    _load_faster_whisper_model._cache = model  # type: ignore[attr-defined]
    return model


def _faster_whisper_transcribe(
    audio_path: Path,
    *,
    language: str,
) -> list[IngestTranscriptCue] | None:
    """
    Transcribe `audio_path` with faster-whisper running locally.

    Returns:
      * `list[IngestTranscriptCue]` on success.
      * None if faster-whisper isn't installed or the model fails to load —
        the caller falls through to the OpenAI Whisper API path.
      * Raises TranscriptionError if the model loaded but the transcription
        itself failed (e.g. corrupt audio file). The caller treats this as
        a hard failure for this strategy and moves on.

    `language` is passed through to faster-whisper as a hint. The English-only
    models (`*.en`) ignore it; multilingual models use it to bias detection.
    """
    model = _load_faster_whisper_model()
    if model is None:
        return None

    try:
        size = audio_path.stat().st_size
    except OSError as exc:
        raise TranscriptionError("extracted audio file is missing", detail=str(exc)) from exc
    if size == 0:
        raise TranscriptionError("extracted audio file is empty")

    try:
        # `transcribe` returns a generator of segments + an `info` object.
        # Iterating the generator drives the actual decoding work.
        # Phase A.1: request word-level timestamps so the ClipBoundaryEngine
        # can snap to sub-segment word positions. Small extra CPU cost
        # (~10-20% on CPU) but unlocks precise sentence-boundary cuts.
        segments_iter, _info = model.transcribe(
            str(audio_path),
            language=language if language else None,
            beam_size=1,
            vad_filter=True,
            word_timestamps=True,
        )
        cues: list[IngestTranscriptCue] = []
        for seg in segments_iter:
            text = (getattr(seg, "text", "") or "").strip()
            if not text:
                continue
            start = float(getattr(seg, "start", 0.0))
            end = float(getattr(seg, "end", start))
            cue = IngestTranscriptCue(
                start=start,
                end=max(end, start + 0.01),
                text=text,
                word_source="whisper",
            )
            seg_words = getattr(seg, "words", None) or []
            for w in seg_words:
                try:
                    w_start = float(getattr(w, "start", start))
                    w_end = float(getattr(w, "end", w_start))
                    w_text = (getattr(w, "word", "") or getattr(w, "text", "") or "").strip()
                    if not w_text:
                        continue
                    # faster-whisper exposes `probability` per word.
                    prob = getattr(w, "probability", None)
                    confidence = float(prob) if isinstance(prob, (int, float)) else None
                    cue.words.append(
                        IngestTranscriptWord(
                            start=w_start,
                            end=max(w_end, w_start + 0.01),
                            text=w_text,
                            confidence=confidence,
                        )
                    )
                except Exception:
                    continue
            # If the segment had text but no usable words object (rare), degrade
            # to proportional so downstream code still has something to snap to.
            if not cue.words:
                _fill_proportional_words(cue)
            cues.append(cue)
    except Exception as exc:
        raise TranscriptionError("faster-whisper transcription failed", detail=str(exc)) from exc

    if not cues:
        raise TranscriptionError("faster-whisper produced no usable segments")
    return cues


# --------------------------------------------------------------------- #
# Strategy 4: OpenAI Whisper API
# --------------------------------------------------------------------- #


def _whisper_transcribe(
    audio_path: Path,
    *,
    openai_client: Any,
    language: str,
) -> list[IngestTranscriptCue]:
    """
    Upload the wav to OpenAI Whisper API and return timestamped cues. Kept as a module-level
    function so tests can `patch.object(transcribe, "_whisper_transcribe", ...)` without
    spinning up the full pipeline.
    """
    if openai_client is None:
        raise TranscriptionError(
            "Whisper fallback requested but no OpenAI client is configured",
        )

    try:
        size = audio_path.stat().st_size
    except OSError as exc:
        raise TranscriptionError("extracted audio file is missing", detail=str(exc)) from exc

    if size == 0:
        raise TranscriptionError("extracted audio file is empty")
    if size > _WHISPER_MAX_FILE_BYTES:
        raise TranscriptionError(
            f"audio file too large for Whisper API (size={size}, max={_WHISPER_MAX_FILE_BYTES})"
        )

    try:
        with open(audio_path, "rb") as f:
            # Phase A.1: request both segment- and word-level timestamps.
            # OpenAI's whisper-1 returns a flat `words` array on the completion
            # object when word granularity is requested; we distribute those
            # words into segments by time containment below.
            completion = openai_client.audio.transcriptions.create(
                model=_WHISPER_MODEL,
                file=f,
                response_format="verbose_json",
                language=language if language else None,
                timestamp_granularities=["segment", "word"],
            )
    except Exception as exc:
        raise TranscriptionError("Whisper API call failed", detail=str(exc)) from exc

    segments = getattr(completion, "segments", None)
    if segments is None and isinstance(completion, dict):
        segments = completion.get("segments")
    if not segments:
        raw_text = getattr(completion, "text", None) or (completion.get("text") if isinstance(completion, dict) else "")
        if not raw_text:
            raise TranscriptionError("Whisper returned no segments and no text")
        cue = IngestTranscriptCue(
            start=0.0,
            end=max(1.0, float(audio_path.stat().st_size) / 32000.0),
            text=str(raw_text).strip(),
            word_source="openai",
        )
        _fill_proportional_words(cue)
        return [cue]

    # Extract the flat word list (available when timestamp_granularities includes "word").
    words_payload = getattr(completion, "words", None)
    if words_payload is None and isinstance(completion, dict):
        words_payload = completion.get("words")
    flat_words: list[dict[str, Any]] = []
    if isinstance(words_payload, list):
        for w in words_payload:
            if isinstance(w, dict):
                flat_words.append(w)
            else:
                try:
                    flat_words.append(
                        {
                            "start": float(getattr(w, "start", 0.0)),
                            "end": float(getattr(w, "end", 0.0)),
                            "word": getattr(w, "word", None) or getattr(w, "text", None) or "",
                        }
                    )
                except Exception:
                    continue
    flat_words.sort(key=lambda w: float(w.get("start") or 0.0))

    def _words_in(start_t: float, end_t: float) -> list[IngestTranscriptWord]:
        out: list[IngestTranscriptWord] = []
        for w in flat_words:
            try:
                ws = float(w.get("start") or 0.0)
                we = float(w.get("end") or ws)
            except (TypeError, ValueError):
                continue
            if we < start_t:
                continue
            if ws > end_t:
                break
            token = str(w.get("word") or w.get("text") or "").strip()
            if not token:
                continue
            try:
                out.append(
                    IngestTranscriptWord(
                        start=ws,
                        end=max(we, ws + 0.01),
                        text=token,
                        confidence=None,  # OpenAI whisper-1 doesn't return per-word confidence
                    )
                )
            except Exception:
                continue
        return out

    cues: list[IngestTranscriptCue] = []
    for segment in segments:
        if isinstance(segment, dict):
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", start))
            text = str(segment.get("text", "")).strip()
        else:
            start = float(getattr(segment, "start", 0.0))
            end = float(getattr(segment, "end", start))
            text = str(getattr(segment, "text", "")).strip()
        if not text:
            continue
        cue = IngestTranscriptCue(
            start=start,
            end=max(end, start + 0.01),
            text=text,
            word_source="openai",
        )
        cue.words = _words_in(start, max(end, start + 0.01))
        if not cue.words:
            _fill_proportional_words(cue)
        cues.append(cue)
    if not cues:
        raise TranscriptionError("Whisper segments contained no usable text")
    return cues


# --------------------------------------------------------------------- #
# Public orchestrator
# --------------------------------------------------------------------- #


def _check_transcript_coverage(
    cues: list[IngestTranscriptCue],
    video_duration_sec: float | None,
    *,
    min_coverage: float = 0.85,
) -> float:
    """Compute what fraction of the video the transcript covers.

    Delegates to the shared ``validate_transcript`` module for the actual
    calculation, but returns the simple coverage ratio that the fallback
    chain in :func:`transcribe` expects.
    """
    if not cues or not video_duration_sec or video_duration_sec <= 0:
        return 1.0
    quality = validate_transcript(cues, video_duration_sec, min_coverage=min_coverage)
    return quality.coverage_ratio


def transcribe(
    conn: Any,
    *,
    platform: PlatformLiteral,
    source_id: str,
    info_dict: dict[str, Any],
    video_path: Path,
    workspace: Path,
    youtube_service: Any,
    openai_client: Any,
    language: str = "en",
    serverless_mode: bool = False,
    allow_openai_in_serverless: bool = False,
    video_duration_sec: float | None = None,
) -> list[IngestTranscriptCue]:
    """
    Return a timestamped transcript for the ingested reel.

    Strategy chain:
      (1) cache hit
      (2) YouTube service (yt platform only) — reuses existing cache
      (3) yt-dlp scraped subtitles from the info_dict
      (4) faster-whisper running LOCALLY (free, no API)
      (5) OpenAI Whisper API fallback (paid, refused in SERVERLESS_MODE unless overridden)

    When `video_duration_sec` is provided, each strategy's output is checked for
    transcript coverage (last cue end / video duration). If coverage is below 85%,
    the strategy is treated as insufficient and the next one is tried. The best
    result (highest coverage) is kept as a final fallback if no strategy reaches 85%.

    Raises `TranscriptionError` if every strategy fails AND Whisper is unavailable.
    """
    MIN_COVERAGE = 0.85
    cache_key = _cache_key(platform, source_id, language)

    cached = _load_cached(conn, cache_key)
    if cached:
        log_event(logger, logging.INFO, "transcript_cache_hit", platform=platform, source_id=source_id, count=len(cached))
        return cached

    # Track best result across strategies for coverage fallback.
    best_cues: list[IngestTranscriptCue] = []
    best_coverage: float = 0.0

    def _try_accept(cues: list[IngestTranscriptCue], strategy_name: str) -> list[IngestTranscriptCue] | None:
        """Accept cues if coverage ≥ 85%, otherwise stash as best-so-far and return None."""
        nonlocal best_cues, best_coverage
        if not cues:
            return None
        coverage = _check_transcript_coverage(cues, video_duration_sec)
        if coverage > best_coverage:
            best_coverage = coverage
            best_cues = cues
        if coverage < MIN_COVERAGE and video_duration_sec and video_duration_sec > 0:
            logger.warning(
                "transcript from %s covers %.0f%% of video (%.0fs / %.0fs); "
                "trying next strategy for better coverage",
                strategy_name, coverage * 100,
                max(cue.end for cue in cues), video_duration_sec,
            )
            return None
        return cues

    # Strategy 2: YouTube transcript service
    if platform == "yt":
        cues = _cues_from_youtube_service(conn, youtube_service, source_id)
        accepted = _try_accept(cues, "youtube_service")
        if accepted:
            _store_cache(conn, cache_key, accepted)
            log_event(logger, logging.INFO, "transcript_from_youtube_service", source_id=source_id, count=len(accepted))
            return accepted

    # Strategy 3: yt-dlp scraped subtitles
    cues = _cues_from_info_dict_subtitles(info_dict, language)
    accepted = _try_accept(cues, "yt_dlp_subs")
    if accepted:
        _store_cache(conn, cache_key, accepted)
        log_event(
            logger,
            logging.INFO,
            "transcript_from_yt_dlp_subs",
            platform=platform,
            source_id=source_id,
            count=len(accepted),
        )
        return accepted

    # Strategy 4: faster-whisper running LOCALLY (free, no API).
    # We try this BEFORE the paid OpenAI Whisper API so users without an
    # OpenAI key still get usable transcripts. Skipped silently if the package
    # isn't installed — the OpenAI path picks up the slack.
    audio_path = workspace / "audio_16k.wav"
    audio_extracted = False

    def _ensure_audio_extracted() -> None:
        nonlocal audio_extracted
        if audio_extracted:
            return
        try:
            extract_audio_wav(video_path, audio_path)
        except Exception as exc:
            raise TranscriptionError(
                "Could not extract audio for Whisper fallback",
                detail=str(exc),
            ) from exc
        audio_extracted = True

    try:
        # Probe whether faster-whisper is even available before extracting
        # audio (audio extraction is ~50ms but worth skipping if neither
        # whisper backend can run).
        if _load_faster_whisper_model() is not None:
            _ensure_audio_extracted()
            local_cues = _faster_whisper_transcribe(audio_path, language=language)
            if local_cues:
                accepted = _try_accept(local_cues, "faster_whisper")
                if accepted:
                    _store_cache(conn, cache_key, accepted)
                    log_event(
                        logger,
                        logging.INFO,
                        "transcript_from_faster_whisper",
                        platform=platform,
                        source_id=source_id,
                        count=len(accepted),
                        model=_FASTER_WHISPER_MODEL,
                    )
                    return accepted
    except TranscriptionError:
        logger.exception(
            "faster-whisper transcribe failed for %s:%s; falling back to OpenAI Whisper",
            platform, source_id,
        )

    # Strategy 5: OpenAI Whisper API fallback (paid)
    if serverless_mode and not allow_openai_in_serverless:
        # Before failing, check if we have a best-effort result from earlier strategies.
        if best_cues:
            logger.warning(
                "Whisper unavailable in serverless mode; returning best-effort transcript "
                "(%.0f%% coverage, %d cues)",
                best_coverage * 100, len(best_cues),
            )
            _store_cache(conn, cache_key, best_cues)
            return best_cues
        raise ServerlessUnavailable(
            "Whisper fallback is unavailable in serverless mode. Set ALLOW_OPENAI_IN_SERVERLESS=1 to override."
        )
    if openai_client is None:
        # Return best-effort result if we have one, even if coverage is low.
        if best_cues:
            logger.warning(
                "No Whisper client configured; returning best-effort transcript "
                "(%.0f%% coverage, %d cues)",
                best_coverage * 100, len(best_cues),
            )
            _store_cache(conn, cache_key, best_cues)
            return best_cues
        raise TranscriptionError(
            "No transcript available from platform sources and neither faster-whisper nor OpenAI Whisper is configured."
        )

    _ensure_audio_extracted()
    cues = _whisper_transcribe(audio_path, openai_client=openai_client, language=language)
    _store_cache(conn, cache_key, cues)
    log_event(
        logger,
        logging.INFO,
        "transcript_from_whisper",
        platform=platform,
        source_id=source_id,
        count=len(cues),
    )
    return cues


__all__ = ["transcribe"]
