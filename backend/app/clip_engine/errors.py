from __future__ import annotations

from typing import Any


JOB_GLOBAL_PROVIDER_ERROR_CODES = frozenset({
    "model_unavailable",
    "provider_authentication",
    "provider_budget_exceeded",
    "provider_configuration",
    "provider_quota_exhausted",
    "provider_rate_limited",
})


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


class CancellationError(EngineError):
    """The caller explicitly cancelled active provider work."""


class ProviderError(EngineError):
    """Typed provider failure safe to persist on a generation job."""

    code = "provider_error"
    retryable = False

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        operation: str,
        status_code: int | None = None,
        retry_after_sec: float | None = None,
        detail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.operation = operation
        self.status_code = status_code
        self.retry_after_sec = retry_after_sec
        self.detail = detail

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "provider": self.provider,
            "operation": self.operation,
            "retryable": self.retryable,
        }
        if self.status_code is not None:
            payload["status_code"] = self.status_code
        if self.retry_after_sec is not None:
            payload["retry_after_sec"] = self.retry_after_sec
        if self.detail:
            payload["detail"] = self.detail
        return payload


class ProviderConfigurationError(ProviderError):
    code = "provider_configuration"


class ProviderAuthenticationError(ProviderError):
    code = "provider_authentication"


class ProviderQuotaError(ProviderError):
    code = "provider_quota_exhausted"


class ProviderRateLimitError(ProviderError):
    code = "provider_rate_limited"
    retryable = True


class ProviderTransientError(ProviderError):
    code = "provider_transient"
    retryable = True


class ProviderResponseValidationError(ProviderError):
    """The provider responded, but local response validation rejected it."""

    code = "provider_response_invalid"
    retryable = True


class ProviderRequestError(ProviderError):
    code = "provider_request_rejected"


class TranscriptUnavailableError(ProviderError):
    code = "transcript_unavailable"


# Backward-compatible import name for callers/tests that predate hosted
# transcript generation. Availability now covers both native and generated
# timestamped transcripts rather than source captions alone.
CaptionsUnavailableError = TranscriptUnavailableError


class ProviderBudgetExceededError(ProviderError):
    code = "provider_budget_exceeded"


class ModelUnavailableError(ProviderError):
    code = "model_unavailable"
