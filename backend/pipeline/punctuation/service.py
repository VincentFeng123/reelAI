"""The punctuation stage entry points.

``restore_transcript_punctuation`` runs the full pipeline (chunk → infer → validate → reconcile →
reconstruct) with a per-chunk retry/fallback ladder, caching, and structured observability.

``build_sentences`` is the single gated seam the orchestrator and CLI call in place of the old
``_build_sentences``: it decides whether the transcript needs punctuation, runs the stage, converts
the result to the legacy ``Sentence`` model, and degrades to the legacy path on skip/failure — so the
pipeline never breaks.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from ... import config
from . import cache
from .chunker import _pause_gaps, _segment_boundary_indices, make_chunks
from .fallback import pause_based_annotations, pause_based_full
from .prompt import (
    PROMPT_VERSION,
    STRICT_REPAIR_SYSTEM,
    SYSTEM_PROMPT,
    build_user_prompt,
    repair_prompt,
    strict_repair_prompt,
)
from .provider import LLMPunctuationProvider, PunctuationProvider
from .reconciler import normalize_full, reconcile
from .reconstructor import build_punctuated_words
from .reconstructor import build_sentences as _build_ts
from .reconstructor import render_readable
from .types import (
    Annotation,
    PunctuationMetadata,
    PunctuationResult,
    TimedWord,
)
from .validator import Scope, densify, hard_errors, validate

log = logging.getLogger("clipper.punctuation")


def _to_timed_words(raw) -> list[TimedWord]:
    """Adapt the project's ``[{word,start,end}]`` (or ``TimedWord``) into ids-assigned tokens.

    Ids are the positional index (``w<i>``) so annotations map back exactly. Missing timestamps are
    coerced from neighbours; nothing is dropped."""
    out: list[TimedWord] = []
    prev_end = 0.0
    for i, w in enumerate(raw):
        if isinstance(w, TimedWord):
            word, start, end, speaker, conf = w.word, w.start, w.end, w.speaker, w.confidence
        else:
            word = str(w.get("word", ""))
            start = w.get("start")
            end = w.get("end")
            speaker = w.get("speaker")
            conf = w.get("confidence")
        s = float(start) if start is not None else prev_end
        e = float(end) if end is not None else s
        if e < s:
            e = s
        out.append(TimedWord(id=f"w{i}", word=word, start=s, end=e, speaker=speaker, confidence=conf))
        prev_end = e
    return out


def _est_tokens(system: str, user: str) -> int:
    return (len(system) + len(user)) // config.CHARS_PER_TOKEN + 1500


def _rollup(statuses, had_full_fallback: bool) -> str:
    statuses = list(statuses)
    if had_full_fallback or any(s == "degraded" for s in statuses):
        return "degraded"
    if any(s == "complete_with_repairs" for s in statuses):
        return "complete_with_repairs"
    return "complete"


def _avg_conf(merged: dict[int, Annotation]) -> Optional[float]:
    vals = [a.confidence for a in merged.values() if a.confidence is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def _annotate_chunk(chunk, words, provider_impl: PunctuationProvider, seg_bounds, gaps,
                    weak_timing, video_id, model):
    """Run one chunk up the ladder. Returns (dense_ann, status, retries, val_failures, cache_hit)."""
    key = cache.chunk_key(chunk, words, model, PROMPT_VERSION)
    hit = cache.load_chunk(video_id, key, chunk)
    if hit and hit[1] != "degraded":
        return hit[0], hit[1], 0, 0, 1
    # W25-B: a cached 'degraded' chunk is retry-eligible — one fresh ladder attempt per run
    # (degraded results used to be served forever: qP's live index was 183 run-on
    # "sentences" replayed from this cache). The cached annotations remain the floor: on a
    # failed retry we return them unchanged and never overwrite — never worse than today.

    scope = Scope(chunk.token_ids, is_full=False)
    est = _est_tokens(SYSTEM_PROMPT, build_user_prompt(chunk, words))
    retries = 0
    vfail = 0
    try:
        # attempt 0
        dense, derr = densify(chunk, words, provider_impl.infer(
            SYSTEM_PROMPT, build_user_prompt(chunk, words), est_tokens=est).edits)
        h = hard_errors(derr + validate(scope, words, dense))
        if not h:
            cache.save_chunk(video_id, key, dense, "complete")
            return dense, "complete", 0, 0, 0
        vfail += len(h)

        # attempt 1 — retry with the concrete validation errors
        retries += 1
        dense, derr = densify(chunk, words, provider_impl.infer(
            SYSTEM_PROMPT, repair_prompt(chunk, words, h), est_tokens=est).edits)
        h = hard_errors(derr + validate(scope, words, dense))
        if not h:
            cache.save_chunk(video_id, key, dense, "complete_with_repairs")
            return dense, "complete_with_repairs", retries, vfail, 0
        vfail += len(h)

        # attempt 2 — stricter minimal repair prompt
        retries += 1
        dense, derr = densify(chunk, words, provider_impl.infer(
            STRICT_REPAIR_SYSTEM, strict_repair_prompt(chunk, words), est_tokens=est).edits)
        h = hard_errors(derr + validate(scope, words, dense))
        if not h:
            cache.save_chunk(video_id, key, dense, "complete_with_repairs")
            return dense, "complete_with_repairs", retries, vfail, 0
        vfail += len(h)
    except Exception:  # noqa: BLE001 — provider/transport failure → safe fallback below
        log.warning("chunk %s inference failed; using pause fallback", chunk.id, exc_info=True)

    if hit:                                        # failed retry of a degraded chunk: keep the artifact
        return hit[0], "degraded", retries, vfail, 1
    dense = pause_based_annotations(chunk, words, seg_bounds, gaps, weak_timing)
    cache.save_chunk(video_id, key, dense, "degraded")
    return dense, "degraded", retries, vfail, 0


def restore_transcript_punctuation(words, *, language: str = "en", provider: Optional[str] = None,
                                   model: Optional[str] = None, speaker_aware: bool = True,
                                   video_id: Optional[str] = None, source: str = "",
                                   segments=None, progress=None,
                                   provider_impl: Optional[PunctuationProvider] = None
                                   ) -> PunctuationResult:
    t0 = time.monotonic()
    prov = (provider or config.PUNCTUATION_PROVIDER or config.LLM_PROVIDER).lower()
    model_id = model or config.PUNCTUATION_MODEL or (
        config.GEMINI_MODEL if prov == "gemini" else config.LLM_PRIMARY)

    timed = _to_timed_words(words)
    n = len(timed)
    if n == 0:
        return PunctuationResult(status="failed", metadata=PunctuationMetadata(
            provider=prov, model=model_id, promptVersion=PROMPT_VERSION))

    cached = cache.load_artifact(video_id, timed, source, model_id, PROMPT_VERSION)
    if cached is not None and cached.status != "degraded":
        cached.metadata.cacheHitCount = max(1, cached.metadata.cacheHitCount)
        return cached
    # W25-B: cached.status == 'degraded' falls THROUGH to one fresh attempt this run
    # (per-chunk caches make it cheap: complete chunks replay, degraded chunks retry).
    # If the retry rolls up 'degraded' again the cached artifact is returned and kept
    # on disk unchanged (see below) — never worse than today.

    weak = (source == "supadata")
    chunks = make_chunks(timed, config.PUNCT_TARGET_WORDS, config.PUNCT_OVERLAP_WORDS,
                         config.PUNCT_MAX_WORDS, segments=segments, weak_timing=weak,
                         min_words=config.PUNCT_MIN_WORDS)
    gaps = _pause_gaps(timed)
    seg_bounds = _segment_boundary_indices(timed, segments)
    provider_impl = provider_impl or LLMPunctuationProvider(prov, model or config.PUNCTUATION_MODEL)

    per_chunk_ann: dict[str, dict[int, Annotation]] = {}
    per_chunk_status: dict[str, str] = {}
    retries = val_failures = chunk_hits = 0
    workers = max(1, min(config.PUNCT_WORKERS, len(chunks)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_annotate_chunk, ch, timed, provider_impl, seg_bounds, gaps, weak,
                            video_id, model_id): ch for ch in chunks}
        done = 0
        for fut in as_completed(futs):
            ch = futs[fut]
            ann, status, r, vf, hit = fut.result()
            per_chunk_ann[ch.id], per_chunk_status[ch.id] = ann, status
            retries += r
            val_failures += vf
            chunk_hits += hit
            done += 1
            if progress:
                progress(done / len(chunks), f"Punctuating {done}/{len(chunks)}")

    merged, conflicts = reconcile(chunks, per_chunk_ann, timed, gaps)
    normalize_full(merged, timed)
    full_errs = hard_errors(validate(Scope(list(range(n)), is_full=True), timed, merged))
    had_full_fallback = False
    if full_errs:
        val_failures += len(full_errs)
        merged = normalize_full(pause_based_full(timed, seg_bounds, gaps, weak), timed)
        had_full_fallback = True

    status = _rollup(per_chunk_status.values(), had_full_fallback)
    pw = build_punctuated_words(timed, merged)
    sentences = _build_ts(timed, merged)
    readable = render_readable(sentences)

    warnings: list[str] = []
    if conflicts:
        warnings.append(f"{len(conflicts)} overlap conflict(s) reconciled")
    if status == "degraded":
        warnings.append("some chunks fell back to pause-based punctuation")

    meta = PunctuationMetadata(
        provider=prov, model=model_id, promptVersion=PROMPT_VERSION, inputWords=n,
        chunkCount=len(chunks), chunkSizes=[len(c.token_ids) for c in chunks],
        retryCount=retries, validationFailures=val_failures, conflictCount=len(conflicts),
        cacheHitCount=chunk_hits, processingTimeMs=int((time.monotonic() - t0) * 1000),
        averageConfidence=_avg_conf(merged))
    result = PunctuationResult(status=status, words=pw, sentences=sentences,
                               readableText=readable, warnings=warnings, metadata=meta)
    if cached is not None and status == "degraded":
        # W25-B failed retry of a degraded artifact: keep the existing artifact (the fresh
        # pass could even be WORSE — e.g. a full-validation fallback wiping chunks that were
        # fine in the cached run). Don't overwrite; return the cached result.
        cached.metadata.cacheHitCount = max(1, cached.metadata.cacheHitCount)
        log.info("punctuation[%s] degraded-artifact retry failed (status=%s); keeping cached "
                 "degraded artifact", video_id or "-", status)
        return cached
    cache.save_artifact(video_id, timed, source, model_id, PROMPT_VERSION, result)
    log.info("punctuation[%s] words=%d chunks=%d retries=%d conflicts=%d val_fail=%d "
             "cache_hits=%d status=%s avg_conf=%s ms=%d", video_id or "-", n, len(chunks),
             retries, len(conflicts), val_failures, chunk_hits, status,
             meta.averageConfidence, meta.processingTimeMs)
    return result


# ── the gated seam used by orchestrator + CLI ────────────────────────────────
def _needs_punctuation(transcript: dict) -> bool:
    if config.PUNCT_FORCE:
        return True
    if transcript.get("source") == "supadata":
        return True
    segs = transcript.get("segments") or []
    text = " ".join((s.get("text") or "") for s in segs)
    if not text.strip():
        return True
    words = transcript.get("words") or []
    terminals = sum(text.count(c) for c in ".?!")
    return (terminals / max(1, len(words))) < 0.02


def build_sentences(transcript: dict, video_id: Optional[str], settings: dict, progress=None):
    """Return legacy ``Sentence`` objects, preferring restored punctuation, else the legacy path."""
    from ..sentences import (
        build_sentence_index,
        sentences_from_chunks,
        sentences_from_punctuation,
    )

    if config.PUNCTUATION_ENABLED and transcript.get("words") and _needs_punctuation(transcript):
        try:
            result = restore_transcript_punctuation(
                transcript["words"], language=settings.get("language", "en"),
                provider=settings.get("punctuation_provider"),
                model=settings.get("punctuation_model"),
                speaker_aware=config.PUNCT_SPEAKER_AWARE, video_id=video_id,
                source=transcript.get("source", ""), segments=transcript.get("segments"),
                progress=progress)
            if result.status != "failed" and result.sentences:
                sents = sentences_from_punctuation(result, transcript["words"])
                if sents:
                    transcript["punctuated"] = result.model_dump()
                    return sents
        except Exception:  # noqa: BLE001 — golden rule: degrade, never crash the job
            log.warning("punctuation stage failed; using legacy sentences", exc_info=True)

    sents = build_sentence_index(transcript)
    if transcript.get("source") == "supadata":
        frac = (sum(1 for s in sents if s.ends_with_period) / len(sents)) if sents else 0.0
        avg = (sum(s.end - s.start for s in sents) / len(sents)) if sents else 999.0
        if len(sents) < 5 or frac < 0.3 or avg > 40.0:
            sents = sentences_from_chunks(transcript.get("chunks", []))
    return sents
