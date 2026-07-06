class EngineError(Exception):
    """Base error for the clip engine."""


class SearchError(EngineError):
    pass


class TranscriptError(EngineError):
    pass


class ClipError(EngineError):
    pass


class UnsupportedURLError(EngineError):
    pass
