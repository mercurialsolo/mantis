"""``MantisClient`` — typed HTTP wrapper for the Mantis API.

Wraps the five end-user-facing endpoints (``/v1/predict`` + ``status /
result / logs / cancel`` actions, ``/v1/cua``, ``/v1/runs/{id}/video``,
``/v1/health``) into a small typed surface so integrators don't rewrite
the same ``submit() / poll() / fetch_result()`` boilerplate every time.

The client deliberately stays sync and thin: one ``requests.Session``,
explicit retries are the caller's job, no hidden parallelism. Async / httpx
support can layer on later without changing the surface.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import requests
from pydantic import ValidationError

from mantis_agent.api_schemas import (
    DetachedRunHandle,
    PredictRequest,
    PureCUARequest,
    RunStatus,
)

from .errors import (
    MantisAPIError,
    MantisAuthError,
    MantisError,
    MantisRateLimitError,
    MantisRunFailed,
    MantisTimeoutError,
)

# Terminal status values returned by /v1/predict {action: status}.
TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})

# Default sync timeout for non-polling HTTP calls. Plans can take 5-30+ min
# in detached mode, but each individual call is cheap; 60 s is generous.
_DEFAULT_TIMEOUT_S = 60.0

# Default polling cadence — matches the recommended cadence on
# docs/client/runs-and-polling.md (30 s after the first minute).
_DEFAULT_POLL_INTERVAL_S = 30.0


class MantisClient:
    """Typed client for the Mantis HTTP API.

    Two ways to construct:

    .. code-block:: python

        client = MantisClient(
            endpoint="https://model-x.api.baseten.co/production/sync",
            api_key="...",            # platform key (Baseten); None for Modal
            mantis_token="...",       # per-tenant Mantis token
        )

        # Or pull from env (MANTIS_ENDPOINT, BASETEN_API_KEY, MANTIS_API_TOKEN):
        client = MantisClient.from_env()

    The session is owned by the client; callers can pass their own
    ``requests.Session`` for connection-pool reuse or custom adapters.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        api_key: Optional[str] = None,
        mantis_token: Optional[str] = None,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        session: Optional[requests.Session] = None,
        idempotency_key: Optional[str] = None,
    ) -> None:
        if not endpoint:
            raise ValueError("endpoint is required")
        # Strip a trailing /v1 if the caller pasted the OpenAI-compat base.
        # We always append /v1/... ourselves.
        normalized = endpoint.rstrip("/")
        if normalized.endswith("/v1"):
            normalized = normalized[: -len("/v1")]
        self.endpoint = normalized
        self.api_key = api_key
        self.mantis_token = mantis_token
        self.timeout_s = timeout_s
        self.idempotency_key = idempotency_key
        self._session = session or requests.Session()

    @classmethod
    def from_env(cls, **overrides: Any) -> "MantisClient":
        """Construct from ``MANTIS_ENDPOINT`` / ``BASETEN_API_KEY`` /
        ``MANTIS_API_TOKEN`` env vars.

        Any keyword passed in ``overrides`` wins over the env value — useful
        for tests or for talking to a non-default deployment from the same
        process.
        """
        endpoint = overrides.pop("endpoint", None) or os.environ.get(
            "MANTIS_ENDPOINT", "",
        )
        api_key = overrides.pop("api_key", None) or os.environ.get(
            "BASETEN_API_KEY", "",
        ) or None
        mantis_token = overrides.pop("mantis_token", None) or os.environ.get(
            "MANTIS_API_TOKEN", "",
        ) or None
        if not endpoint:
            raise ValueError(
                "MANTIS_ENDPOINT is not set; pass endpoint=... or export it.",
            )
        return cls(
            endpoint=endpoint,
            api_key=api_key,
            mantis_token=mantis_token,
            **overrides,
        )

    # ── HTTP plumbing ───────────────────────────────────────────────────

    def _headers(self, extra: Optional[dict] = None) -> dict:
        """Build the standard auth headers + caller overrides."""
        h: dict = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Api-Key {self.api_key}"
        if self.mantis_token:
            h["X-Mantis-Token"] = self.mantis_token
        if self.idempotency_key:
            h["Idempotency-Key"] = self.idempotency_key
        if extra:
            h.update(extra)
        return h

    def _raise_for_status(self, resp: requests.Response) -> None:
        """Convert non-2xx into the typed error hierarchy."""
        if resp.status_code < 400:
            return
        body: Any
        try:
            body = resp.json()
        except ValueError:
            body = resp.text
        message = f"HTTP {resp.status_code} from {resp.url}"
        if resp.status_code in (401, 403):
            raise MantisAuthError(
                message, status_code=resp.status_code, response_body=body,
            )
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            try:
                retry_after_s = float(retry_after) if retry_after else None
            except ValueError:
                retry_after_s = None
            raise MantisRateLimitError(
                message,
                response_body=body,
                retry_after_seconds=retry_after_s,
            )
        raise MantisAPIError(
            message, status_code=resp.status_code, response_body=body,
        )

    def _post_predict(self, payload: dict) -> dict:
        url = f"{self.endpoint}/v1/predict"
        resp = self._session.post(
            url, json=payload, headers=self._headers(), timeout=self.timeout_s,
        )
        self._raise_for_status(resp)
        return resp.json()

    # ── Predict + action variants ───────────────────────────────────────

    def predict(self, request: Union[PredictRequest, dict]) -> DetachedRunHandle:
        """Submit a plan. Returns a :class:`DetachedRunHandle` with ``run_id``.

        ``request`` is either a :class:`PredictRequest` (preferred — gives
        typed-field validation client-side) or a raw dict for callers that
        already speak the wire shape. Pydantic ``ValidationError`` from a
        malformed :class:`PredictRequest` is re-raised verbatim so callers
        can locate the offending field.
        """
        if isinstance(request, PredictRequest):
            payload = request.model_dump(exclude_none=True)
        elif isinstance(request, dict):
            # Validate the dict client-side too so callers get the same
            # error path. Skip the validation envelope only if the caller
            # explicitly passed an action-mode dict (status/result/logs/cancel).
            if not request.get("action"):
                try:
                    PredictRequest.model_validate(request)
                except ValidationError:
                    raise
            payload = dict(request)
        else:
            raise TypeError(
                f"request must be PredictRequest or dict, got {type(request).__name__}",
            )
        # ``predict`` is for the submit path; force detached unless caller
        # explicitly opts out. The wait_for_completion helper handles polling.
        payload.setdefault("detached", True)
        body = self._post_predict(payload)
        return DetachedRunHandle.model_validate(body)

    def status(self, run_id: str) -> RunStatus:
        """``POST /v1/predict {action: status, run_id}`` — current run state."""
        body = self._post_predict({"action": "status", "run_id": run_id})
        return RunStatus.model_validate(body)

    def result(self, run_id: str) -> dict:
        """``POST /v1/predict {action: result, run_id}`` — full result payload.

        Returned as a raw dict because the ``result`` shape is plan-dependent
        (lead lists, step traces, video metadata) and not centrally typed.
        """
        return self._post_predict({"action": "result", "run_id": run_id})

    def logs(self, run_id: str, *, tail: int = 200) -> list[str]:
        """``POST /v1/predict {action: logs, run_id, tail}`` — recent events.

        Returns the raw ``events`` list straight off the response so callers
        can grep / pipe / pretty-print at their own discretion.
        """
        if tail < 1 or tail > 10_000:
            raise ValueError("tail must be in [1, 10000]")
        body = self._post_predict(
            {"action": "logs", "run_id": run_id, "tail": tail},
        )
        events = body.get("events") or body.get("logs") or []
        if not isinstance(events, list):
            raise MantisAPIError(
                f"unexpected logs response shape: {type(events).__name__}",
                status_code=200,
                response_body=body,
            )
        return [str(e) for e in events]

    def cancel(self, run_id: str) -> dict:
        """``POST /v1/predict {action: cancel, run_id}`` — request cancellation.

        Cancels are cooperative — the runner finishes its current checkpoint
        before stopping. Status flips to ``cancelled`` 5-60 s later.
        """
        return self._post_predict({"action": "cancel", "run_id": run_id})

    # ── /v1/cua pass-through ────────────────────────────────────────────

    def cua_run(
        self,
        instruction: Union[str, PureCUARequest],
        **kwargs: Any,
    ) -> dict:
        """``POST /v1/cua`` — pure Holo3 pass-through (no Claude decomposition).

        Accepts a free-text ``instruction`` (with optional ``**kwargs`` for
        ``start_url``, ``max_steps``, etc.) or a pre-built
        :class:`PureCUARequest`. Returns the raw JSON response —
        either a detached run handle or a sync result depending on
        ``detached``.
        """
        if isinstance(instruction, PureCUARequest):
            payload = instruction.model_dump(exclude_none=True)
        else:
            payload = PureCUARequest(
                instruction=instruction, **kwargs,
            ).model_dump(exclude_none=True)

        url = f"{self.endpoint}/v1/cua"
        resp = self._session.post(
            url, json=payload, headers=self._headers(), timeout=self.timeout_s,
        )
        self._raise_for_status(resp)
        return resp.json()

    # ── Video + health ──────────────────────────────────────────────────

    def fetch_video(
        self,
        run_id: str,
        dest_path: Union[str, Path],
        *,
        polished: bool = True,
        chunk_size: int = 1 << 16,
    ) -> Path:
        """Stream ``GET /v1/runs/{run_id}/video`` to a local file.

        Returns the destination ``Path``. Set ``polished=False`` to fetch
        the raw recording instead of the captioned / overlay version (when
        the deployment produced both).
        """
        url = f"{self.endpoint}/v1/runs/{run_id}/video"
        params: dict[str, Any] = {}
        if not polished:
            params["polished"] = "false"
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with self._session.get(
            url,
            params=params,
            headers=self._headers(),
            timeout=self.timeout_s,
            stream=True,
        ) as resp:
            self._raise_for_status(resp)
            with dest.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        fh.write(chunk)
        return dest

    def health(self) -> dict:
        """``GET /v1/health`` — liveness probe. Returns the server's payload.

        Useful as a sanity check right after constructing the client; raises
        :class:`MantisError` if the server isn't reachable.
        """
        url = f"{self.endpoint}/v1/health"
        try:
            resp = self._session.get(
                url, headers=self._headers(), timeout=self.timeout_s,
            )
        except requests.RequestException as exc:
            raise MantisError(f"{url} unreachable: {exc}") from exc
        self._raise_for_status(resp)
        return resp.json()

    # ── Convenience: polling helpers ────────────────────────────────────

    def wait_for_completion(
        self,
        run_id: str,
        *,
        poll_interval_s: float = _DEFAULT_POLL_INTERVAL_S,
        timeout_s: Optional[float] = None,
        on_status: Optional[Any] = None,
    ) -> RunStatus:
        """Poll ``status`` until terminal (``succeeded`` / ``failed`` /
        ``cancelled``) or until ``timeout_s`` elapses.

        ``on_status`` is an optional callable invoked with each intermediate
        :class:`RunStatus` snapshot — useful for surfacing a progress
        indicator in a UI without re-implementing the polling loop.
        Raises :class:`MantisTimeoutError` on timeout (the run is *not*
        cancelled — call :meth:`cancel` explicitly if needed).
        """
        if poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be > 0")
        t0 = time.monotonic()
        while True:
            status = self.status(run_id)
            if on_status is not None:
                on_status(status)
            if status.status in TERMINAL_STATUSES:
                return status
            elapsed = time.monotonic() - t0
            if timeout_s is not None and elapsed >= timeout_s:
                raise MantisTimeoutError(
                    f"run {run_id} did not reach terminal state in "
                    f"{timeout_s:.0f}s (last status: {status.status!r})",
                    run_id=run_id,
                    elapsed_s=elapsed,
                )
            time.sleep(poll_interval_s)

    def run_to_completion(
        self,
        request: Union[PredictRequest, dict],
        *,
        poll_interval_s: float = _DEFAULT_POLL_INTERVAL_S,
        timeout_s: Optional[float] = None,
        on_status: Optional[Any] = None,
    ) -> dict:
        """Submit + wait + fetch result in one call.

        Equivalent to ``predict() → wait_for_completion() → result()``.
        Raises :class:`MantisRunFailed` if the run ends in ``failed`` or
        ``cancelled``; raises :class:`MantisTimeoutError` if the run
        doesn't terminate inside ``timeout_s``.
        """
        handle = self.predict(request)
        final = self.wait_for_completion(
            handle.run_id,
            poll_interval_s=poll_interval_s,
            timeout_s=timeout_s,
            on_status=on_status,
        )
        if final.status != "succeeded":
            raise MantisRunFailed(final)
        return self.result(handle.run_id)

    # ── Context manager — close the underlying session ─────────────────

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "MantisClient":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # Make ``list(client.iter_logs(run_id))`` a clean one-liner if a caller
    # ever wants page-by-page log retrieval — but only if the server grows
    # cursor-style pagination. Kept as a placeholder so the surface is
    # forward-compat-friendly.
    def iter_logs(self, run_id: str, *, batch: int = 1000) -> Iterable[str]:
        """Yield log lines in batches of ``batch``. Today this is a single
        call (server doesn't paginate); structure preserved for future
        cursor-style pagination.
        """
        yield from self.logs(run_id, tail=batch)
