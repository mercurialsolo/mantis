"""``grade_run`` — call the env's oracle, parse the response, return a result.

After ``MicroPlanRunner`` finishes a plan, the harness calls
:func:`grade_run` to get a server-side ground-truth verdict on whether
the agent accomplished the task. Oracle endpoint contract (see
``docs/envs/SPEC.md`` §"Oracle interface"):

* ``GET /__env__/oracle?task_id=<id>`` with header ``X-Env-Admin: <token>``
* Returns JSON: ``{"passed": bool, "score": float, "reasons": [...], "diff": {...}}``

Result type :class:`GradingResult` is a thin dataclass so it round-trips
through JSON cleanly. The CLI writes it to ``<output_dir>/oracle.json``
next to ``trace.jsonl`` (additive — nothing breaks if it isn't there).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


@dataclass
class GradingResult:
    """Oracle verdict for one plan run.

    ``error`` is populated when the oracle call itself failed (network,
    401, non-JSON body, etc.); callers can distinguish "plan failed
    according to oracle" (``passed=False, error=None``) from "we never
    got an oracle verdict" (``error=<reason>``). Both write through to
    ``oracle.json`` so the failure mode is captured in the run record.
    """

    task_id: str
    passed: bool = False
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    diff: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def grade_run(
    env_url: str,
    admin_token: str,
    task_id: str,
    *,
    timeout_s: float = 30.0,
    extra_headers: dict[str, str] | None = None,
) -> GradingResult:
    """Hit ``GET <env_url>/__env__/oracle?task_id=<task_id>`` with the admin token.

    Returns a :class:`GradingResult`. Never raises — oracle / network
    failures populate ``error`` so the run record always reflects what
    happened. This is the right shape for benchmarks: a flaky network
    should not lose the rest of the run record.

    ``extra_headers`` are merged onto the request (``X-Env-Admin`` still
    set first). A sim env served behind a preview proxy (e.g. Daytona's
    ``*.daytonaproxy01.net``) needs the proxy's skip-warning + preview-token
    headers on *every* request, including this admin oracle call, or the GET
    307s to the proxy's auth wall and the body comes back as HTML, not JSON.
    """
    if not env_url:
        return GradingResult(
            task_id=task_id,
            error="env_url is empty — no oracle to call",
        )
    if not task_id:
        return GradingResult(
            task_id="",
            error="task_id is empty — plans must declare a task_id under --env",
        )

    base = env_url.rstrip("/")
    url = f"{base}/__env__/oracle?task_id={quote(task_id)}"
    headers = {"X-Env-Admin": admin_token, **(extra_headers or {})}
    req = Request(url, headers=headers)
    try:
        with urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 — env URL only
            body = resp.read().decode("utf-8")
            payload = json.loads(body)
    except HTTPError as exc:
        return GradingResult(
            task_id=task_id,
            error=f"oracle returned HTTP {exc.code}: {exc.reason}",
        )
    except (URLError, ConnectionError, OSError, TimeoutError) as exc:
        return GradingResult(
            task_id=task_id,
            error=f"oracle network error: {exc!r}",
        )
    except json.JSONDecodeError as exc:
        return GradingResult(
            task_id=task_id,
            error=f"oracle returned non-JSON body: {exc!r}",
        )

    if not isinstance(payload, dict):
        return GradingResult(
            task_id=task_id,
            error=f"oracle returned non-dict payload: {type(payload).__name__}",
        )

    return GradingResult(
        task_id=task_id,
        passed=bool(payload.get("passed", False)),
        score=float(payload.get("score", 0.0)),
        reasons=list(payload.get("reasons", []) or []),
        diff=dict(payload.get("diff", {}) or {}),
    )
