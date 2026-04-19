"""
Transcript extraction with a four-strategy fallback chain:

    1. YouTube only: reuse the existing `YouTubeService.get_transcript` (caches via
       `transcript_cache` table; free; best quality when captions are authored).
    2. Any platform: parse yt-dlp's scraped subtitles from the `info_dict`
       (`automatic_captions` / `subtitles`). Free; ~90% of IG Reels with captions and
       most English YouTube videos land here.
    3. faster-whisper running LOCALLY on the extracted audio. Free; runs on
       CPU via CTranslate2; downloads a small model (~80 MB for `base.en`) on
       first use to `~/.cache/huggingface/hub/`. Tried before the remote Groq
       path so users without a Groq key still get usable transcripts.
       Skipped silently if `faster-whisper` is not installed.
    4. Fallback: Groq Whisper API (`whisper-large-v3`) on the extracted audio,
       routed through `services.llm_router.transcribe_audio`. Used only when
       strategies 1-3 fail.

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
from .errors import TranscriptionError
from .ffmpeg_tools import extract_audio_wav
from .logging_config import get_ingest_logger, log_event
from .models import IngestTranscriptCue, IngestTranscriptWord, PlatformLiteral, WordSourceLiteral

logger: logging.Logger = get_ingest_logger(__name__)

# Cache key prefix. Versioned (v2) so Phase-1 faster-whisper alignment
# refreshes existing cache rows on first read — older proportional rows are
# simply missed and get rewritten with the upgraded word_source.
_TRANSCRIPT_CACHE_PREFIX = "ingest_transcript:v2:"
_WHISPER_MODEL = "whisper-large-v3"
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

# Phase 1 — when a caption-path strategy succeeds but returns
# word_source in {"proportional", "legacy"}, run faster-whisper over the
# audio purely for word alignment and transplant native timings onto the
# (higher-quality) caption text. Default ON for English; disable via
# FORCE_NATIVE_WORD_TIMESTAMPS=false if CPU/latency budget is tight.
_FORCE_NATIVE_WORD_TIMESTAMPS = str(
    os.environ.get("FORCE_NATIVE_WORD_TIMESTAMPS", "true")
).strip().lower() in {"1", "true", "yes", "on"}
# Minimum fraction of caption tokens that must match an ASR word for the
# alignment result to replace proportional timings. Below this threshold
# the caller keeps the original cues.
_ALIGNMENT_MIN_MATCH_RATE = 0.6


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
                if word_source_raw in (
                    "whisper",
                    "whisper_aligned",
                    "whisperx",
                    "openai",
                    "groq",
                    "proportional",
                    "legacy",
                )
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
    handle None gracefully and fall through to the Groq Whisper API path.

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
        the caller falls through to the Groq Whisper API path.
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
# Phase 1: faster-whisper word alignment over caption-path cues
# --------------------------------------------------------------------- #


_ALIGN_PUNCT_RE = re.compile(r"[^\w]+")


def _normalize_token(tok: str) -> str:
    return _ALIGN_PUNCT_RE.sub("", tok).lower()


def _align_caption_to_asr(
    caption_tokens: list[str],
    asr_words: list[IngestTranscriptWord],
    *,
    lookahead: int = 5,
) -> list[int | None]:
    """
    Greedy two-pointer alignment: for each caption token, find the nearest
    ASR word (ahead of the cursor, within `lookahead`) whose normalized
    text matches. Returns, for each caption token, the matched ASR word
    index or None. Preserves ASR ordering — caption tokens that don't match
    stay None and the caller handles interpolation.
    """
    matches: list[int | None] = [None] * len(caption_tokens)
    j = 0
    for i, ct in enumerate(caption_tokens):
        ct_norm = _normalize_token(ct)
        if not ct_norm:
            continue
        # Scan forward up to `lookahead` ASR words for a match.
        best = None
        for k in range(j, min(j + lookahead, len(asr_words))):
            asr_norm = _normalize_token(asr_words[k].text)
            if asr_norm == ct_norm:
                best = k
                break
        if best is not None:
            matches[i] = best
            j = best + 1
    return matches


def align_words_via_faster_whisper(
    audio_path: Path,
    caption_cues: list[IngestTranscriptCue],
    *,
    language: str,
) -> list[IngestTranscriptCue] | None:
    """
    Transplant native faster-whisper word timings onto caption-path cues.

    Replaces `word_source in {"proportional", "legacy"}` with
    `word_source="whisper_aligned"` when alignment match-rate meets
    ``_ALIGNMENT_MIN_MATCH_RATE``. Returns new cues on success, None on
    failure or low match — caller keeps the originals.

    Caption text is preserved (assumed higher quality than ASR for captioned
    videos); only word-level start/end timings change. Unmatched tokens get
    linearly interpolated timings between neighboring matches.
    """
    if not caption_cues:
        return None
    model = _load_faster_whisper_model()
    if model is None:
        return None

    try:
        asr_cues = _faster_whisper_transcribe(audio_path, language=language)
    except TranscriptionError:
        logger.info("faster-whisper alignment pass failed; keeping proportional timings")
        return None
    if not asr_cues:
        return None

    # Flatten ASR words across all cues, ordered by start time.
    asr_words: list[IngestTranscriptWord] = []
    for cue in asr_cues:
        asr_words.extend(cue.words)
    asr_words.sort(key=lambda w: w.start)
    if not asr_words:
        return None

    # Flatten caption tokens, tracking which cue each token belongs to.
    caption_tokens: list[str] = []
    token_cue_idx: list[int] = []
    for cue_idx, cue in enumerate(caption_cues):
        for tok in _WHITESPACE_RE.split((cue.text or "").strip()):
            if tok:
                caption_tokens.append(tok)
                token_cue_idx.append(cue_idx)
    if not caption_tokens:
        return None

    matches = _align_caption_to_asr(caption_tokens, asr_words)
    matched_count = sum(1 for m in matches if m is not None)
    match_rate = matched_count / len(caption_tokens)
    if match_rate < _ALIGNMENT_MIN_MATCH_RATE:
        logger.info(
            "faster-whisper alignment match rate %.2f below threshold %.2f; "
            "keeping proportional timings",
            match_rate, _ALIGNMENT_MIN_MATCH_RATE,
        )
        return None

    # Interpolate timings for unmatched tokens: find previous and next
    # anchor, linearly split the interval. Tokens before the first anchor
    # fall back to the cue's start bound; tokens after the last use cue end.
    n = len(caption_tokens)
    word_starts: list[float] = [0.0] * n
    word_ends: list[float] = [0.0] * n
    for i, m in enumerate(matches):
        if m is not None:
            word_starts[i] = asr_words[m].start
            word_ends[i] = asr_words[m].end
    # Forward pass: propagate last-known end to unmatched.
    i = 0
    while i < n:
        if matches[i] is not None:
            i += 1
            continue
        # Find next matched index.
        j = i + 1
        while j < n and matches[j] is None:
            j += 1
        # Left anchor: last matched before i (end), or cue.start.
        if i > 0 and matches[i - 1] is not None:
            left_t = word_ends[i - 1]
        else:
            left_t = caption_cues[token_cue_idx[i]].start
        # Right anchor: word_starts[j] if j<n, else cue.end of last unmatched.
        if j < n:
            right_t = word_starts[j]
        else:
            right_t = caption_cues[token_cue_idx[n - 1]].end
        span = max(right_t - left_t, 0.01)
        gap = j - i
        for k in range(i, j):
            frac_s = (k - i) / gap
            frac_e = (k - i + 1) / gap
            word_starts[k] = left_t + span * frac_s
            word_ends[k] = left_t + span * frac_e
        i = j

    # Build new cues preserving caption text; attach aligned words.
    aligned: list[IngestTranscriptCue] = []
    cursor = 0
    for cue_idx, cue in enumerate(caption_cues):
        tokens_in_cue: list[tuple[str, float, float, float | None]] = []
        while cursor < n and token_cue_idx[cursor] == cue_idx:
            tok = caption_tokens[cursor]
            ws = word_starts[cursor]
            we = max(word_ends[cursor], ws + 0.01)
            m = matches[cursor]
            conf = asr_words[m].confidence if m is not None else None
            tokens_in_cue.append((tok, ws, we, conf))
            cursor += 1
        if not tokens_in_cue:
            aligned.append(cue)
            continue
        new_cue = IngestTranscriptCue(
            start=cue.start,
            end=cue.end,
            text=cue.text,
            words=[
                IngestTranscriptWord(
                    start=ws, end=we, text=tok, confidence=conf,
                )
                for (tok, ws, we, conf) in tokens_in_cue
            ],
            word_source="whisper_aligned",
        )
        aligned.append(new_cue)
    logger.info(
        "faster-whisper alignment applied: match_rate=%.2f tokens=%d cues=%d",
        match_rate, len(caption_tokens), len(aligned),
    )
    return aligned


# --------------------------------------------------------------------- #
# Strategy 4: Groq Whisper API (via services.llm_router)
# --------------------------------------------------------------------- #


def _whisper_transcribe(
    audio_path: Path,
    *,
    language: str,
) -> list[IngestTranscriptCue]:
    """
    Upload the wav to Groq Whisper API (whisper-large-v3) and return timestamped
    cues. Kept as a module-level function so tests can
    `patch.object(transcribe, "_whisper_transcribe", ...)` without spinning up
    the full pipeline.
    """
    from ..services import llm_router

    if not llm_router.gemini_or_groq_available():
        raise TranscriptionError(
            "Whisper fallback requested but no Groq API key is configured",
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
        completion = llm_router.transcribe_audio(
            str(audio_path),
            language=language if language else None,
        )
    except Exception as exc:
        raise TranscriptionError("Whisper API call failed", detail=str(exc)) from exc

    if not isinstance(completion, dict):
        raise TranscriptionError("Whisper returned no usable response")

    segments = completion.get("segments")
    if not segments:
        raw_text = completion.get("text")
        if not raw_text:
            raise TranscriptionError("Whisper returned no segments and no text")
        cue = IngestTranscriptCue(
            start=0.0,
            end=max(1.0, float(audio_path.stat().st_size) / 32000.0),
            text=str(raw_text).strip(),
            word_source="groq",
        )
        _fill_proportional_words(cue)
        return [cue]

    words_payload = completion.get("words")
    flat_words: list[dict[str, Any]] = []
    if isinstance(words_payload, list):
        for w in words_payload:
            if isinstance(w, dict):
                flat_words.append(w)
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
                        confidence=None,
                    )
                )
            except Exception:
                continue
        return out

    cues: list[IngestTranscriptCue] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        cue = IngestTranscriptCue(
            start=start,
            end=max(end, start + 0.01),
            text=text,
            word_source="groq",
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
    language: str = "en",
    serverless_mode: bool = False,
    video_duration_sec: float | None = None,
) -> list[IngestTranscriptCue]:
    """
    Return a timestamped transcript for the ingested reel.

    Strategy chain:
      (1) cache hit
      (2) YouTube service (yt platform only) — reuses existing cache
      (3) yt-dlp scraped subtitles from the info_dict
      (4) faster-whisper running LOCALLY (free, no API)
      (5) Groq Whisper API fallback (whisper-large-v3 via llm_router)

    When `video_duration_sec` is provided, each strategy's output is checked for
    transcript coverage (last cue end / video duration). If coverage is below 85%,
    the strategy is treated as insufficient and the next one is tried. The best
    result (highest coverage) is kept as a final fallback if no strategy reaches 85%.

    Raises `TranscriptionError` if every strategy fails AND Whisper is unavailable.
    """
    from ..services import llm_router
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

    # Set up lazy audio extraction — needed by Phase-1 alignment AND by
    # faster-whisper/Groq fallback strategies below.
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

    def _maybe_align_caption_cues(
        cues: list[IngestTranscriptCue],
        strategy_name: str,
    ) -> list[IngestTranscriptCue]:
        """Phase 1: if caption cues have proportional/legacy word timings,
        upgrade them to faster-whisper-aligned native timings. Silently
        keeps originals if alignment fails, model unavailable, or match
        rate below threshold."""
        if not _FORCE_NATIVE_WORD_TIMESTAMPS:
            return cues
        if _load_faster_whisper_model() is None:
            return cues
        if all(c.word_source not in ("proportional", "legacy") for c in cues):
            return cues
        try:
            _ensure_audio_extracted()
        except TranscriptionError:
            logger.info(
                "audio extraction failed during %s alignment; keeping proportional",
                strategy_name,
            )
            return cues
        aligned = align_words_via_faster_whisper(
            audio_path, cues, language=language,
        )
        upgrade_source = "whisper_aligned"
        if aligned is None:
            # Phase 4(a): WhisperX forced-alignment fallback. Only runs when
            # Phase-1 returned None (match-rate below threshold) and WhisperX
            # is enabled via WHISPERX_FALLBACK_ENABLED (default true).
            try:
                from .whisperx_transcribe import (
                    whisperx_align,
                    whisperx_fallback_enabled,
                )
            except Exception:
                return cues
            if not whisperx_fallback_enabled():
                return cues
            aligned = whisperx_align(audio_path, cues, language=language)
            if aligned is None:
                return cues
            upgrade_source = "whisperx"
        log_event(
            logger, logging.INFO, "transcript_word_alignment_upgraded",
            platform=platform, source_id=source_id, strategy=strategy_name,
            cue_count=len(aligned), alignment_source=upgrade_source,
        )
        return aligned

    # Strategy 2: YouTube transcript service
    if platform == "yt":
        cues = _cues_from_youtube_service(conn, youtube_service, source_id)
        accepted = _try_accept(cues, "youtube_service")
        if accepted:
            accepted = _maybe_align_caption_cues(accepted, "youtube_service")
            _store_cache(conn, cache_key, accepted)
            log_event(logger, logging.INFO, "transcript_from_youtube_service", source_id=source_id, count=len(accepted))
            return accepted

    # Strategy 3: yt-dlp scraped subtitles
    cues = _cues_from_info_dict_subtitles(info_dict, language)
    accepted = _try_accept(cues, "yt_dlp_subs")
    if accepted:
        accepted = _maybe_align_caption_cues(accepted, "yt_dlp_subs")
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
    # We try this BEFORE the remote Groq Whisper API so users without a Groq
    # key still get usable transcripts. Skipped silently if the package isn't
    # installed — the Groq path picks up the slack.
    # (audio extraction helpers are defined above alongside Phase-1 alignment.)
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
            "faster-whisper transcribe failed for %s:%s; falling back to Groq Whisper",
            platform, source_id,
        )

    # Strategy 5: Groq Whisper API fallback (whisper-large-v3)
    if not llm_router.gemini_or_groq_available():
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
            "No transcript available from platform sources and neither faster-whisper nor Groq Whisper is configured."
        )

    _ensure_audio_extracted()
    cues = _whisper_transcribe(audio_path, language=language)
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
