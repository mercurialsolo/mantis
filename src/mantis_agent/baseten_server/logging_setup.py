"""Logging configuration for the Baseten CUA workload.

JSON-per-record formatter with per-request tenant enrichment, plus a
``DetachedRunLogHandler`` that streams a thread's log records to the
runtime's per-run event log so the ``/v1/runs/{run_id}`` endpoint can
serve them back.

``configure_logging()`` is called explicitly by the routes module at
import time — keeping the side effect there means importing this module
is safe in tests.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .runtime import BasetenCUARuntime


class JsonLogFormatter(logging.Formatter):
    """One-line JSON-per-record formatter that attaches tenant_id and run_id
    when set in the process environment. Lets stdout consumers (Datadog,
    CloudWatch, Stackdriver) parse logs without ad-hoc regexes.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Per-request tenant context, set by /predict handler; empty in startup.
        tenant_id = os.environ.get("MANTIS_TENANT_ID")
        if tenant_id:
            payload["tenant_id"] = tenant_id
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    """One-time logging setup. JSON to stdout, level from LOG_LEVEL env."""
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    if os.environ.get("MANTIS_LOG_FORMAT", "json").lower() == "json":
        handler = logging.StreamHandler()
        handler.setFormatter(JsonLogFormatter())
        root = logging.getLogger()
        root.handlers[:] = [handler]
        root.setLevel(level)
    else:
        logging.basicConfig(level=level)


class DetachedRunLogHandler(logging.Handler):
    """Per-thread log handler that streams records to a detached run's event log."""

    def __init__(self, runtime: "BasetenCUARuntime", run_id: str, thread_id: int) -> None:
        super().__init__(level=logging.INFO)
        self.runtime = runtime
        self.run_id = run_id
        self.thread_id = thread_id

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread != self.thread_id:
            return
        try:
            self.runtime._append_detached_event(self.run_id, self.format(record))
        except Exception:
            self.handleError(record)
