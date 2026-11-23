"""Internal exception types, all mappable to OpenAI-style error bodies."""
from __future__ import annotations


class SwitchboardError(Exception):
    status = 500
    code = "internal_error"
    type = "api_error"
    retryable = False

    def __init__(self, message: str = "", *, retry_after: float | None = None) -> None:
        super().__init__(message or self.code)
        self.message = message or self.code
        self.retry_after = retry_after

    def to_body(self) -> dict:
        return {"error": {"message": self.message, "type": self.type, "code": self.code}}


class BadRequest(SwitchboardError):
    status, code, type = 400, "invalid_request", "invalid_request_error"


class ModelNotFound(SwitchboardError):
    status, code, type = 404, "model_not_found", "invalid_request_error"


class ContextLengthExceeded(SwitchboardError):
    status, code, type = 400, "context_length_exceeded", "invalid_request_error"


class Unauthorized(SwitchboardError):
    status, code, type = 401, "invalid_api_key", "authentication_error"


class Forbidden(SwitchboardError):
    status, code, type = 403, "forbidden", "permission_error"


class BudgetExceeded(SwitchboardError):
    status, code, type = 429, "insufficient_quota", "insufficient_quota"


class AdmissionRejected(SwitchboardError):
    """Our own limiter refused the request (client or provider capacity)."""
    status, code, type = 429, "rate_limit_exceeded", "rate_limit_error"
    retryable = True


class Deferred(SwitchboardError):
    """Batch-path signal: nothing is wrong, try again later (no attempt charged)."""
    status, code, type = 503, "deferred", "api_error"
    retryable = True


class ProviderError(SwitchboardError):
    status, code, type = 502, "provider_error", "api_error"
    retryable = True
    provider: str | None = None


class RateLimited(ProviderError):
    status, code = 429, "provider_rate_limited"


class ProviderUnavailable(ProviderError):
    status, code = 503, "provider_unavailable"


class ProviderTimeout(ProviderError):
    status, code = 504, "provider_timeout"


class ProviderAuthError(ProviderError):
    status, code = 502, "provider_auth_error"
    retryable = False


class ProviderBadRequest(ProviderError):
    status, code, type = 400, "provider_bad_request", "invalid_request_error"
    retryable = False


class StreamInterrupted(ProviderError):
    status, code = 502, "stream_interrupted"
