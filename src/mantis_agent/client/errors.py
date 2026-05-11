"""Exception hierarchy for the Mantis client.

All client-raised exceptions inherit from :class:`MantisError`, so callers
can write ``except MantisError`` once and catch every failure shape the
client can surface (HTTP errors, auth failures, polling timeouts, terminal
run failures). More specific subclasses let callers branch on the cause
without parsing string messages.
"""

from __future__ import annotations

from typing import Any, Optional


class MantisError(Exception):
    """Base class for every exception the Mantis client raises."""


class MantisAPIError(MantisError):
    """Generic HTTP-level failure from the Mantis server.

    Raised when the server returns a non-2xx status code that isn't more
    specifically classified (auth, rate-limit). Attributes carry the raw
    HTTP status and the parsed response body so callers can build their
    own retry / logging logic.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        response_body: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class MantisAuthError(MantisAPIError):
    """401 or 403 from the Mantis server.

    Usually means the ``X-Mantis-Token`` header is missing, expired, or
    scoped wrong for the requested action — or, for Baseten-hosted
    deployments, that ``Authorization: Api-Key ...`` is missing.
    """


class MantisRateLimitError(MantisAPIError):
    """429 from the Mantis server. Carries the server's ``Retry-After``."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 429,
        response_body: Any = None,
        retry_after_seconds: Optional[float] = None,
    ) -> None:
        super().__init__(
            message, status_code=status_code, response_body=response_body,
        )
        self.retry_after_seconds = retry_after_seconds


class MantisTimeoutError(MantisError):
    """``wait_for_completion`` exceeded the caller-supplied timeout.

    The run is *not* cancelled — it's still in flight on the server. The
    caller can either poll again or call :meth:`MantisClient.cancel`.
    """

    def __init__(self, message: str, *, run_id: str, elapsed_s: float) -> None:
        super().__init__(message)
        self.run_id = run_id
        self.elapsed_s = elapsed_s


class MantisRunFailed(MantisError):
    """A run reached a terminal non-success state (``failed`` or ``cancelled``).

    Raised by :meth:`MantisClient.run_to_completion` when the caller asked
    for the result-shaped convenience flow. The status snapshot is attached
    so the caller can inspect ``.error`` / ``.summary`` without a second
    round-trip.
    """

    def __init__(self, status: Any) -> None:
        super().__init__(
            f"run {status.run_id} ended with status={status.status!r}"
        )
        self.status = status
