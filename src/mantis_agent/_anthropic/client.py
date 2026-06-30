"""Shared Anthropic ``/v1/messages`` client with retry + tool_use helpers.

Lifted from :class:`mantis_agent.extraction.extractor.ClaudeExtractor`
under #406 so the same client backs:

- ``ClaudeExtractor`` (extraction calls — ``extract`` /
  ``find_all_listings`` / ``verify_gate``)
- ``ClaudeFormTargetProvider`` (grounding calls — ``find_form_target``
  / ``find_target_by_affordance`` / ``verify_dropdown_value``)

Both used to duplicate the API call shape because they lived on the
same class; splitting the grounding methods out of the extractor
revealed the duplication. Owning the client in one place means:

- One retry policy. The 529/Overloaded survival logic added in #403 /
  PR #404 applies to *every* Anthropic call now, not just the
  extraction ones.
- One image-encoding path. PNG → base64 happens here, callers pass
  ``PIL.Image`` objects.
- One time-accounting hook. ``_credit_claude_time`` is wired into the
  call helpers so the TimeMeter bookkeeping (#362) keeps working for
  both code paths without each caller re-implementing it.

Public surface intentionally narrow — only the three helpers
extraction and grounding actually share:

- :meth:`AnthropicToolUseClient.call_with_tool_schema` — single
  screenshot, schema-validated ``tool_use`` response.
- :meth:`AnthropicToolUseClient.call_with_tool_schema_multi` —
  multi-screenshot variant.
- :meth:`AnthropicToolUseClient.post_messages_with_retry` — raw POST
  with retry, for callers that need the full Response (none today,
  but ``_call`` / ``_call_many`` will migrate here in a follow-up).
"""

from __future__ import annotations

import base64
import logging
import os
import random
import time
from io import BytesIO
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)


# ── Image encoding for /v1/messages (#518) ─────────────────────────────


_SUPPORTED_IMAGE_FORMATS = {
    "jpeg": ("JPEG", "image/jpeg"),
    "jpg":  ("JPEG", "image/jpeg"),
    "png":  ("PNG",  "image/png"),
    "webp": ("WEBP", "image/webp"),
}


def _resolve_image_format() -> tuple[str, str]:
    """Read ``MANTIS_CLAUDE_IMAGE_FORMAT`` and return ``(pil_format, mime)``.

    Defaults to ``jpeg`` (~30% smaller than PNG at q=85 for typical
    browser screenshots, no impact on Claude vision recognition of
    text / buttons / form labels). Operators pin back to PNG via
    ``MANTIS_CLAUDE_IMAGE_FORMAT=png`` when they need pixel-perfect
    encoding (e.g. evaluating against a reference bundle).
    """
    raw = (os.environ.get("MANTIS_CLAUDE_IMAGE_FORMAT") or "jpeg").strip().lower()
    return _SUPPORTED_IMAGE_FORMATS.get(raw, _SUPPORTED_IMAGE_FORMATS["jpeg"])


def _resolve_image_quality() -> int:
    """Read ``MANTIS_CLAUDE_IMAGE_QUALITY`` (default 85, clamped to 1-95).

    JPEG q=85 is the standard sweet-spot for screenshot OCR; q>95 loses
    the size advantage; q<70 starts to blur small text.
    """
    raw = os.environ.get("MANTIS_CLAUDE_IMAGE_QUALITY")
    if not raw:
        return 85
    try:
        return max(1, min(95, int(raw)))
    except (TypeError, ValueError):
        return 85


def encode_screenshot_for_claude(
    screenshot: "Image.Image",
    *,
    format: str | None = None,  # noqa: A002 — matches PIL's kw
    quality: int | None = None,
) -> tuple[str, str]:
    """Encode a PIL screenshot for the Anthropic ``/v1/messages`` API (#518).

    Returns ``(base64_data, media_type)`` ready to plug into the request
    body's ``{"type": "image", "source": {"type": "base64", "media_type": ..., "data": ...}}``
    block. Image dimensions are preserved exactly — only the encoding
    (lossy JPEG by default) changes. This matters because grounding
    callers dispatch clicks at the x/y coordinates Claude returns,
    which are in input-pixel space — any downsample would silently
    mis-place every click.

    Caller may override ``format`` ("jpeg" / "png" / "webp") and
    ``quality`` (1-95, JPEG only); otherwise honours the
    ``MANTIS_CLAUDE_IMAGE_FORMAT`` + ``MANTIS_CLAUDE_IMAGE_QUALITY``
    env knobs.

    Defaults: JPEG q=85 — typically ~30-40% smaller than PNG for
    browser screenshots, no impact on Claude vision recognition.
    """
    if format is not None:
        pil_format, media_type = _SUPPORTED_IMAGE_FORMATS.get(
            format.lower(), _SUPPORTED_IMAGE_FORMATS["jpeg"],
        )
    else:
        pil_format, media_type = _resolve_image_format()
    q = quality if quality is not None else _resolve_image_quality()

    # JPEG / WEBP don't carry an alpha channel; PIL raises if you save
    # an RGBA image as JPEG. Browser screenshots from PIL.ImageGrab
    # are usually RGB but Xvfb captures land as RGBA in some paths.
    img = screenshot
    if pil_format in {"JPEG", "WEBP"} and img.mode not in {"RGB", "L"}:
        img = img.convert("RGB")

    buf = BytesIO()
    if pil_format == "JPEG":
        img.save(buf, format=pil_format, quality=q, optimize=True)
    elif pil_format == "PNG":
        img.save(buf, format=pil_format, optimize=True)
    else:
        img.save(buf, format=pil_format, quality=q)
    return base64.b64encode(buf.getvalue()).decode(), media_type


# HTTP status codes treated as transient (worth retrying with backoff).
# Anthropic-specific framing:
#   - 429: rate limited
#   - 502/503/504: upstream gateway hiccups (rare but real)
#   - 529: Anthropic's "Overloaded" code — the canonical motivating
#     case (issue #403). By far the most common reason a plan halts
#     mid-step during peak hours.
_TRANSIENT_STATUS_CODES: frozenset[int] = frozenset({429, 502, 503, 504, 529})


def _retry_delay(attempt: int, retry_after_header: str | None) -> float:
    """Compute the sleep before the next retry attempt.

    - Honours ``Retry-After`` when it's a numeric seconds value
      (RFC 7231 §7.1.3 — HTTP-date form not supported; Anthropic
      uses seconds in practice).
    - Otherwise exponential backoff with up to 25% jitter: 1s, 2s,
      4s, 8s, capped at 16s. Total worst-case wait across 4 attempts
      is ~15s + jitter, which clears typical 1–2 minute overload
      spikes without blowing the per-step budget.

    Standalone function (not a method) so tests can monkeypatch it
    directly without instantiating the client.
    """
    if retry_after_header:
        try:
            return max(0.0, float(retry_after_header))
        except (TypeError, ValueError):
            pass
    base = float(min(16, 2 ** attempt))
    jitter = random.uniform(0.0, base * 0.25)
    return base + jitter


def _credit_claude_time(bucket: str, t0: float) -> None:
    """Credit elapsed Anthropic API time to the runner's TimeMeter
    (epic #362). Best-effort — bookkeeping I/O must never break a
    Claude call. Same implementation that previously lived inside
    ``extractor.py``; lifted alongside the client so both consumers
    of the shared client report into the same bucket.
    """
    try:
        from ..gym.time_meter import record_to_current
        record_to_current(bucket, time.monotonic() - t0)
    except Exception as exc:  # noqa: BLE001 — observability, never fatal
        logger.debug("anthropic time_meter credit failed: %s", exc)


def credit_claude_tokens_from_response(response: Any) -> None:
    """Read ``usage.{input_tokens,output_tokens,...}`` off an Anthropic
    response and credit them to the currently-published CostMeter (#514).

    Accepts the parsed JSON dict OR a ``requests.Response`` object —
    callers that already have the dict can pass it directly; raw
    response objects get parsed here. Best-effort: a missing usage
    block, malformed JSON, or no published meter all silently no-op.

    The Anthropic API returns:

    * ``usage.input_tokens`` — total input tokens (includes cached)
    * ``usage.output_tokens`` — generated tokens
    * ``usage.cache_read_input_tokens`` — subset of input that hit cache
    * ``usage.cache_creation_input_tokens`` — input written to cache
      (billed at higher rate than standard input; for now we treat as
      ``input_tokens`` and let the standard rate apply)

    Source of truth for the surface:
    https://docs.anthropic.com/en/api/messages#response-usage
    """
    try:
        if response is None:
            return
        if hasattr(response, "json") and callable(response.json):
            try:
                payload = response.json()
            except Exception:  # noqa: BLE001
                return
        else:
            payload = response
        if not isinstance(payload, dict):
            return
        usage = payload.get("usage") or {}
        if not isinstance(usage, dict):
            return
        from ..gym.cost_meter import record_claude_tokens_to_current
        record_claude_tokens_to_current(
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
            cached_input_tokens=int(usage.get("cache_read_input_tokens", 0) or 0),
        )
    except Exception as exc:  # noqa: BLE001 — observability never fatal
        logger.debug("anthropic cost_meter token credit failed: %s", exc)


def _record_modelio_if_active(
    request_payload: dict[str, Any],
    response_json: dict[str, Any],
    duration_ms: int,
) -> None:
    """Forward to the modelio capture layer when a context is active.

    Local import to avoid pulling the observability module into every
    LLM call when no caller has published a layer context (which is the
    case at module load until PR B-2..B-5 wire individual call sites).
    """
    try:
        from ..observability.modelio import (
            current_modelio_context,
            record_anthropic_modelio,
        )
    except Exception:  # noqa: BLE001 — import-time issues never block calls
        return
    ctx = current_modelio_context()
    if ctx is None:
        return
    try:
        record_anthropic_modelio(
            request_payload=request_payload,
            response_json=response_json,
            duration_ms=duration_ms,
            ctx=ctx,
        )
    except Exception as exc:  # noqa: BLE001 — Augur spec §4.3
        logger.debug("modelio capture failed: %s", exc)


class AnthropicToolUseClient:
    """Anthropic ``/v1/messages`` client tuned for schema-validated tool_use.

    Holds ``api_key`` + ``model`` and exposes two call helpers that
    package a screenshot (or multiple) plus a prompt into a request
    body, force the model to emit a schema-validated ``tool_use``
    block, and parse it back to a dict.

    On any non-200 or shape-mismatch the helpers return ``None`` —
    callers treat that as a not-found / fall-through outcome. The
    retry policy (defined by :data:`_TRANSIENT_STATUS_CODES` +
    :func:`_retry_delay`) is internal to :meth:`post_messages_with_retry`
    so individual callers don't think about it.

    Why ``_log_prefix`` exists: extraction calls have historically
    logged warnings prefixed ``ClaudeExtractor`` so existing
    log-scraping (Modal log queries, the agentic-recovery reading
    these prefixes) keeps working. Grounding callers can pass a
    different prefix (``ClaudeFormTarget``) to disambiguate. Default
    matches the legacy extractor wording.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        log_prefix: str = "Anthropic",
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self._log_prefix = log_prefix
        # Extra request headers merged into every POST — e.g. the
        # ``anthropic-beta`` header that enables the computer-use tool
        # type for the CUA brain. Without this seam the brain's
        # ``computer_20251124`` tool was rejected with HTTP 400 on every
        # call (the beta header had nowhere to ride through this shared
        # client). Empty for the text/grounding/extraction callers.
        self._extra_headers = dict(extra_headers or {})

    def post_messages_with_retry(
        self,
        payload: dict[str, Any],
        *,
        timeout: float,
        max_attempts: int = 4,
    ):
        """POST ``payload`` to /v1/messages with transient-error retry.

        Retries on the status codes in :data:`_TRANSIENT_STATUS_CODES`
        (429 / 502 / 503 / 504 / 529) and on network exceptions
        (``requests.Timeout``, ``requests.ConnectionError``). Non-
        transient HTTP errors (4xx other than 429) return the Response
        object so the caller can log + parse the body — no retry,
        because retrying a malformed payload wastes budget.

        Returns:
            - The final ``requests.Response`` on success.
            - The final non-transient ``requests.Response`` (4xx).
            - The final transient ``requests.Response`` after the
              retry budget is exhausted.
            - ``None`` only when every attempt raised a network
              exception (no Response to return).

        Honours ``Retry-After`` header when numeric. Sleeps between
        attempts via ``time.sleep`` — tests monkeypatch that for speed.

        Side-effect: stashes the retry count on
        :attr:`last_retries_spent` (#841 user-feedback quality).
        Zero on first-attempt success, ``max_attempts - 1`` on
        exhaustion. Read by callers that want to surface "we retried
        N times" in the run summary or failure_help dict.
        """
        import requests

        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        headers.update(self._extra_headers)
        last_response = None
        self.last_retries_spent = 0
        _t0 = time.monotonic()
        for attempt in range(max_attempts):
            try:
                resp = requests.post(
                    url, headers=headers, json=payload, timeout=timeout,
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                if attempt == max_attempts - 1:
                    self.last_retries_spent = attempt
                    logger.warning(
                        "%s network error after %d attempts: %s",
                        self._log_prefix, max_attempts, exc,
                    )
                    return None
                delay = _retry_delay(attempt, None)
                logger.info(
                    "%s transient network error (%s) — retry %d/%d in %.1fs",
                    self._log_prefix, type(exc).__name__,
                    attempt + 1, max_attempts, delay,
                )
                time.sleep(delay)
                continue
            if resp.status_code not in _TRANSIENT_STATUS_CODES:
                self.last_retries_spent = attempt
                # #892 item 2: capture the modelio record at the single HTTP
                # chokepoint so EVERY caller is covered uniformly — including
                # ``brain_claude`` / ``plan_decomposer`` / grounding, which
                # call this method directly and so never hit the per-wrapper
                # capture that used to live in ``call_with_tool_schema*``.
                # Cheap context probe first so we don't parse JSON when no
                # layer is published (the common case).
                if resp.status_code == 200:
                    try:
                        from ..observability.modelio import (
                            current_modelio_context,
                        )
                        if current_modelio_context() is not None:
                            _record_modelio_if_active(
                                payload, resp.json(),
                                int((time.monotonic() - _t0) * 1000),
                            )
                    except Exception:  # noqa: BLE001 — telemetry never blocks
                        pass
                return resp
            last_response = resp
            if attempt == max_attempts - 1:
                self.last_retries_spent = attempt
                logger.warning(
                    "%s transient HTTP %s after %d attempts; giving up",
                    self._log_prefix, resp.status_code, max_attempts,
                )
                return resp
            delay = _retry_delay(attempt, resp.headers.get("Retry-After"))
            logger.info(
                "%s transient HTTP %s — retry %d/%d in %.1fs",
                self._log_prefix, resp.status_code,
                attempt + 1, max_attempts, delay,
            )
            time.sleep(delay)
        return last_response

    def call_with_tool_schema(
        self,
        screenshot: Image.Image,
        prompt: str,
        *,
        tool_name: str,
        tool_description: str,
        input_schema: dict[str, Any],
        max_tokens: int = 500,
        time_bucket: str = "claude_extract",
        cache_tools: bool = False,
    ) -> dict | None:
        """Send one screenshot + prompt, return the parsed tool_use dict.

        Anthropic's ``tool_use`` with ``tool_choice={"type": "tool",
        "name": ...}`` forces the model to emit a ``tool_use`` content
        block whose ``input`` field is server-side-validated against
        ``input_schema``. Replaces the prompt-only "output ONLY valid
        JSON" pattern (issue #219) which produced prose-only /
        truncated / malformed responses in production.

        Returns the validated input dict, or ``None`` on:

        - missing API key
        - retry budget exhausted with no Response object
        - non-200 final status
        - no ``tool_use`` block of the requested name in the reply
        - any unexpected exception during parsing

        Callers log the ``None`` and treat it as not-found / fall-
        through — same contract the original extractor methods relied
        on.

        ``time_bucket`` flows into :func:`_credit_claude_time` so per-
        step accounting in the TimeMeter still distinguishes extraction
        from grounding when both share this client.

        ``cache_tools`` opts the tool spec into Anthropic's ephemeral
        prompt cache (#421). The cache breakpoint sits on the last
        tool definition so the entire ``tools`` block — name +
        description + input schema — counts as cached input on hits.
        Worth it for callers that fire the same tool repeatedly with a
        stable schema (verify_gate is the canonical case); not worth
        the per-request overhead for one-off calls.
        """
        if not self.api_key:
            logger.warning("%s: no API key", self._log_prefix)
            return None

        # #518 — encode at MANTIS_CLAUDE_IMAGE_QUALITY / _FORMAT. Same
        # dimensions as the source screenshot (grounding coords must
        # match pixel space); only the byte encoding changes.
        b64, media_type = encode_screenshot_for_claude(screenshot)

        tool_def: dict[str, Any] = {
            "name": tool_name,
            "description": tool_description,
            "input_schema": input_schema,
        }
        if cache_tools:
            tool_def["cache_control"] = {"type": "ephemeral"}

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "tools": [tool_def],
            "tool_choice": {"type": "tool", "name": tool_name},
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        }

        t0 = time.monotonic()
        try:
            resp = self.post_messages_with_retry(payload, timeout=30)
            if resp is None:
                return None
            if resp.status_code != 200:
                logger.warning(
                    "%s tool_use API error %s: %s",
                    self._log_prefix, resp.status_code,
                    resp.text[:500],
                )
                return None
            payload_json = resp.json()
            credit_claude_tokens_from_response(payload_json)
            # #720 — extractor cache telemetry (single-screenshot path).
            if cache_tools:
                from .cache import extract_cache_telemetry
                tele = extract_cache_telemetry(payload_json)
                if tele.get("cache_read_input_tokens", 0) > 0 or tele.get(
                    "cache_creation_input_tokens", 0
                ) > 0:
                    logger.warning(
                        "  [cache] extractor: read=%d created=%d input=%d output=%d",
                        tele.get("cache_read_input_tokens", 0),
                        tele.get("cache_creation_input_tokens", 0),
                        tele.get("input_tokens", 0),
                        tele.get("output_tokens", 0),
                    )
            from ..observability.claude_cost_meter import record_from_response
            record_from_response(
                source=time_bucket or "extract_tool",
                model=self.model, response_json=payload_json,
            )
            # #523 modelio capture now happens centrally in
            # ``post_messages_with_retry`` (#892 item 2) — which this method
            # calls — so every caller is covered once, with no double-record.
            for block in payload_json.get("content", []):
                if block.get("type") == "tool_use" and block.get("name") == tool_name:
                    tool_input = block.get("input")
                    if isinstance(tool_input, dict):
                        return tool_input
            logger.warning(
                "%s tool_use returned no %s tool block in response",
                self._log_prefix, tool_name,
            )
        except Exception as e:  # noqa: BLE001 — surface unexpected
            logger.warning("%s tool_use failed: %s", self._log_prefix, e)
        finally:
            _credit_claude_time(time_bucket, t0)
        return None

    def call_with_tool_schema_multi(
        self,
        screenshots: list[Image.Image],
        prompt: str,
        *,
        tool_name: str,
        tool_description: str,
        input_schema: dict[str, Any],
        labels: list[str] | None = None,
        max_tokens: int = 500,
        time_bucket: str = "claude_extract",
        cache_tools: bool = False,
        image_quality: int | None = None,
    ) -> dict | None:
        """Multi-screenshot variant of :meth:`call_with_tool_schema`.

        Bundles a list of screenshots into a single messages payload —
        each image prefixed with its label so the model can refer to
        them in prose ("Screenshot 1: BEFORE click", "Screenshot 2:
        AFTER click"). Used by post-click navigation verification.

        Same return-shape contract as the single-screenshot helper —
        ``None`` on any failure (caller treats as fall-through).
        Slightly larger ``timeout`` (45s) than the single-screenshot
        variant because the larger payload upload + analysis takes
        longer.
        """
        if not self.api_key:
            logger.warning("%s: no API key", self._log_prefix)
            return None

        labels = labels or []
        content: list[dict] = [{"type": "text", "text": prompt}]
        for i, screenshot in enumerate(screenshots, 1):
            label = labels[i - 1] if i - 1 < len(labels) else f"screenshot {i}"
            # #518 — same encoding helper as the single-shot path.
            # ``image_quality`` opt-in lets cost-sensitive paths fall
            # back to the env-resolved default (q=85) while OCR-sensitive
            # paths (e.g. phone digit reads from body text) bump to
            # q=95 for cleaner glyphs at a small token-count cost.
            if image_quality is not None:
                b64, media_type = encode_screenshot_for_claude(
                    screenshot, quality=image_quality,
                )
            else:
                b64, media_type = encode_screenshot_for_claude(screenshot)
            content.append({"type": "text", "text": f"Screenshot {i}: {label}"})
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": b64},
            })

        tool_def: dict[str, Any] = {
            "name": tool_name,
            "description": tool_description,
            "input_schema": input_schema,
        }
        if cache_tools:
            tool_def["cache_control"] = {"type": "ephemeral"}
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "tools": [tool_def],
            "tool_choice": {"type": "tool", "name": tool_name},
            "messages": [{"role": "user", "content": content}],
        }

        t0 = time.monotonic()
        try:
            resp = self.post_messages_with_retry(payload, timeout=45)
            if resp is None:
                return None
            if resp.status_code != 200:
                logger.warning(
                    "%s multi tool_use API error %s: %s",
                    self._log_prefix, resp.status_code,
                    resp.text[:500],
                )
                return None
            payload_json = resp.json()
            credit_claude_tokens_from_response(payload_json)
            # #720 — emit cache telemetry at WARNING so operators can audit
            # extractor cache hit rate in Modal logs without an env flag.
            from .cache import extract_cache_telemetry
            tele = extract_cache_telemetry(payload_json)
            if tele.get("cache_read_input_tokens", 0) > 0 or tele.get(
                "cache_creation_input_tokens", 0
            ) > 0:
                logger.warning(
                    "  [cache] extractor_multi: read=%d created=%d input=%d output=%d",
                    tele.get("cache_read_input_tokens", 0),
                    tele.get("cache_creation_input_tokens", 0),
                    tele.get("input_tokens", 0),
                    tele.get("output_tokens", 0),
                )
            from ..observability.claude_cost_meter import record_from_response
            record_from_response(
                source=time_bucket or "extract_tool_multi",
                model=self.model, response_json=payload_json,
            )
            # #523 modelio capture is centralized in post_messages_with_retry.
            for block in payload_json.get("content", []):
                if block.get("type") == "tool_use" and block.get("name") == tool_name:
                    tool_input = block.get("input")
                    if isinstance(tool_input, dict):
                        return tool_input
            logger.warning(
                "%s multi tool_use returned no %s tool block",
                self._log_prefix, tool_name,
            )
        except Exception as e:  # noqa: BLE001 — surface unexpected
            logger.warning("%s multi tool_use failed: %s", self._log_prefix, e)
        finally:
            _credit_claude_time(time_bucket, t0)
        return None


__all__ = [
    "AnthropicToolUseClient",
    "_TRANSIENT_STATUS_CODES",
    "_retry_delay",
    "_credit_claude_time",
    "credit_claude_tokens_from_response",
    "encode_screenshot_for_claude",
]
