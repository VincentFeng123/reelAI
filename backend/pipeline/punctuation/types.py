"""Data model for punctuation restoration.

Three layers of representation flow through this package:

- ``TimedWord``      — the raw ASR token (Layer 1: source of truth; word + timing).
- ``Annotation``     — the dense, per-token punctuation decision we reconcile over.
- ``PunctuatedWord`` — Layer 2 output: the original token + capitalization/punctuation applied.

The LLM never returns rewritten prose. It returns a *sparse* ``ChunkEdits`` (only tokens that
deviate from the mid-sentence default), keyed by immutable ``w<index>`` token ids; we densify to
``Annotation`` per token internally, so the model can never alter word text or timing.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# Bump to invalidate work/<id>/punctuation.json cached artifacts.
SCHEMA_VERSION = 1

PunctuationMark = Literal["", ".", ",", "?", "!", ":", ";"]
PunctuationStatus = Literal["complete", "complete_with_repairs", "degraded", "failed"]

TERMINALS: frozenset[str] = frozenset({".", "?", "!"})
PUNCT_ENUM: frozenset[str] = frozenset({"", ".", ",", "?", "!", ":", ";"})


# ── Layer 1: raw timed word ──────────────────────────────────────────────────
class TimedWord(BaseModel):
    """A single ASR token. ``id`` is assigned deterministically from the word index."""
    id: str
    word: str
    start: float
    end: float
    speaker: Optional[str] = None
    confidence: Optional[float] = None


# ── LLM I/O (sparse) ─────────────────────────────────────────────────────────
class TokenEdit(BaseModel):
    """One deviation from the default. Only tokens that change are emitted by the model.

    ``p`` is a plain ``str`` (not the ``PunctuationMark`` Literal) on purpose: Gemini/Groq structured
    output rejects an empty-string enum member, and the default here is "". We validate ``p`` against
    ``PUNCT_ENUM`` ourselves in ``validator.densify``."""
    id: str
    p: str = ""                  # punctuationAfter; validated against PUNCT_ENUM downstream
    cap: bool = False            # capitalize
    ss: bool = False             # sentenceStart
    se: bool = False             # sentenceEnd
    pg: bool = False             # paragraphStart
    conf: Optional[float] = None


class ChunkEdits(BaseModel):
    """The full structured reply for one chunk — a list of sparse edits."""
    edits: list[TokenEdit] = Field(default_factory=list)


# ── dense internal annotation (one per token, reconciled) ────────────────────
class Annotation(BaseModel):
    capitalize: bool = False
    punctuationAfter: PunctuationMark = ""
    sentenceStart: bool = False
    sentenceEnd: bool = False
    paragraphStart: bool = False
    confidence: Optional[float] = None

    def key(self) -> tuple:
        """A total-order key for fully-deterministic tie-breaking (no dict/set ordering)."""
        return (
            self.capitalize,
            self.punctuationAfter,
            self.sentenceStart,
            self.sentenceEnd,
            self.paragraphStart,
            self.confidence if self.confidence is not None else -1.0,
        )


# ── chunking ─────────────────────────────────────────────────────────────────
class TranscriptChunk(BaseModel):
    id: str
    token_ids: list[int]                 # global word indices, in order
    primary_token_ids: list[int]
    left_overlap_token_ids: list[int]
    right_overlap_token_ids: list[int]


# ── Layer 2 output ───────────────────────────────────────────────────────────
class PunctuatedWord(BaseModel):
    id: str
    word: str                            # ORIGINAL token, unchanged (preservation invariant)
    displayWord: str                     # word + capitalization (never alters the underlying word)
    start: float
    end: float
    speaker: Optional[str] = None
    confidence: Optional[float] = None
    capitalize: bool = False
    punctuationAfter: PunctuationMark = ""
    sentenceStart: bool = False
    sentenceEnd: bool = False
    paragraphStart: bool = False
    punctuationConfidence: Optional[float] = None


class TranscriptSentence(BaseModel):
    id: str
    text: str
    start: float
    end: float
    speaker: Optional[str] = None
    tokenIds: list[str] = Field(default_factory=list)
    confidence: Optional[float] = None
    paragraphStart: bool = False


class Conflict(BaseModel):
    token: str
    won: str                             # winning chunk id
    lost: str                            # losing chunk id
    reason: str = ""
    field_diff: str = ""


# ── result + cache ───────────────────────────────────────────────────────────
class PunctuationMetadata(BaseModel):
    provider: str = ""
    model: str = ""
    promptVersion: str = ""
    inputWords: int = 0
    chunkCount: int = 0
    chunkSizes: list[int] = Field(default_factory=list)
    retryCount: int = 0
    validationFailures: int = 0
    conflictCount: int = 0
    cacheHitCount: int = 0
    processingTimeMs: int = 0
    averageConfidence: Optional[float] = None


class PunctuationResult(BaseModel):
    status: PunctuationStatus
    words: list[PunctuatedWord] = Field(default_factory=list)
    sentences: list[TranscriptSentence] = Field(default_factory=list)
    readableText: str = ""
    warnings: list[str] = Field(default_factory=list)
    metadata: PunctuationMetadata = Field(default_factory=PunctuationMetadata)


class PunctuationArtifact(BaseModel):
    """The on-disk cache bundle. Gated on schema_version + fingerprint + token_count because
    token ids are positional — a re-transcode under the same video_id must MISS, not mis-map."""
    schema_version: int = SCHEMA_VERSION
    video_id: str = ""
    transcript_fingerprint: str = ""
    token_count: int = 0
    model: str = ""
    prompt_version: str = ""
    result: PunctuationResult
