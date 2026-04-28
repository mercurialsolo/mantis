"""Tier 2 webhook callbacks for run-completion notifications.

When a tenant has a webhook URL configured (either at tenant level via
``TenantConfig.webhook_url`` or per-request via ``callback_url`` in the
PredictRequest), the server POSTs a JSON body to that URL once the run
reaches a terminal state.

Body shape:

    {
      "run_id": "...",
      "tenant_id": "...",
      "status": "succeeded|failed|cancelled",
      "summary": { ... },
      "delivered_at": "<ISO8601>"
    }

Delivery is signed with HMAC-SHA256 in ``X-Mantis-Signature`` so the
receiver can verify authenticity. The signing secret is read from a
container secret named by ``TenantConfig.webhook_secret_name`` (per tenant)
or by the ``MANTIS_WEBHOOK_SECRET_DEFAULT`` env var.

Retries: 3 attempts with exponential backoff (1s, 5s, 30s). Best-effort —
delivery state is logged but not persisted; callers should also be able
to poll ``/v1/predict {action: status, run_id}`` if they miss a webhook.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger("mantis_agent.webhooks")

DEFAULT_RETRY_DELAYS_SECONDS = (1.0, 5.0, 30.0)
DEFAULT_TIMEOUT_SECONDS = 10.0


@dataclass
class WebhookPayload:
    run_id: str
    tenant_id: str
    status: str
    summary: dict[str, Any]

    def to_body(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "tenant_id": self.tenant_id,
            "status": self.status,
            "summary": self.summary,
            "delivered_at": datetime.now(timezone.utc).isoformat(),
        }


def _read_secret(name: str) -> str:
    if not name:
        return ""
    path = Path("/secrets") / name
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def _resolve_signing_secret(secret_name: str) -> str:
    if secret_name:
        v = _read_secret(secret_name)
        if v:
            return v
    return os.environ.get("MANTIS_WEBHOOK_SECRET_DEFAULT", "").strip()


def sign_body(body_bytes: bytes, secret: str) -> str:
    """Return the hex-encoded HMAC-SHA256 of ``body_bytes`` under ``secret``."""
    if not secret:
        return ""
    return hmac.new(secret.encode("utf-8"), body_bytes, hashlib.sha256).hexdigest()


def deliver_webhook_sync(
    url: str,
    payload: WebhookPayload,
    secret: str = "",
    retry_delays: tuple[float, ...] = DEFAULT_RETRY_DELAYS_SECONDS,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    session: requests.Session | None = None,
) -> bool:
    """Deliver one webhook. Blocks until success or all retries exhausted.

    Returns True on a 2xx response, False otherwise.
    """
    if not url:
        return False
    body = json.dumps(payload.to_body(), separators=(",", ":")).encode("utf-8")
    sig = sign_body(body, secret) if secret else ""
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "mantis-cua/webhook",
    }
    if sig:
        headers["X-Mantis-Signature"] = f"sha256={sig}"

    sess = session or requests
    attempts = (0,) + retry_delays
    last_status: int | str = "no-attempt"
    for i, delay in enumerate(attempts):
        if delay:
            time.sleep(delay)
        try:
            r = sess.post(url, data=body, headers=headers, timeout=timeout)
        except requests.RequestException as exc:
            last_status = f"connection-error: {exc}"
            logger.warning(
                "webhook attempt %d/%d failed run_id=%s url=%s err=%s",
                i + 1, len(attempts), payload.run_id, url, exc,
            )
            continue
        last_status = r.status_code
        if 200 <= r.status_code < 300:
            logger.info(
                "webhook delivered run_id=%s tenant=%s url=%s status=%d attempt=%d",
                payload.run_id, payload.tenant_id, url, r.status_code, i + 1,
            )
            return True
        logger.warning(
            "webhook attempt %d/%d non-2xx run_id=%s url=%s status=%d",
            i + 1, len(attempts), payload.run_id, url, r.status_code,
        )
    logger.error(
        "webhook GAVE UP run_id=%s tenant=%s url=%s last=%s",
        payload.run_id, payload.tenant_id, url, last_status,
    )
    return False


def deliver_webhook_async(
    url: str,
    payload: WebhookPayload,
    secret_name: str = "",
    retry_delays: tuple[float, ...] = DEFAULT_RETRY_DELAYS_SECONDS,
) -> threading.Thread:
    """Fire-and-forget webhook delivery in a daemon thread.

    Returns the thread for tests that need to ``join()`` on completion.
    """
    secret = _resolve_signing_secret(secret_name)

    def _run() -> None:
        try:
            deliver_webhook_sync(url, payload, secret=secret, retry_delays=retry_delays)
        except Exception:  # noqa: BLE001 — webhook errors should never break the runner
            logger.exception(
                "webhook thread crashed run_id=%s url=%s",
                payload.run_id, url,
            )

    t = threading.Thread(
        target=_run,
        name=f"mantis-webhook-{payload.run_id[:8]}",
        daemon=True,
    )
    t.start()
    return t
