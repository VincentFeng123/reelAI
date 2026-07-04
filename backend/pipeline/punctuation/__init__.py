"""Transcript punctuation restoration.

Turns raw timed words into readable, correctly-segmented sentences while preserving every original
word and timestamp exactly. Runs between word-level transcription and semantic segmentation; the
word-level timed transcript stays the source of truth.
"""
from .prompt import PROMPT_VERSION  # noqa: F401
from .provider import LLMPunctuationProvider, PunctuationProvider  # noqa: F401
from .service import (  # noqa: F401
    build_sentences,
    restore_transcript_punctuation,
)
from .types import (  # noqa: F401
    Annotation,
    ChunkEdits,
    Conflict,
    PunctuatedWord,
    PunctuationMark,
    PunctuationMetadata,
    PunctuationResult,
    PunctuationStatus,
    TimedWord,
    TokenEdit,
    TranscriptChunk,
    TranscriptSentence,
)
