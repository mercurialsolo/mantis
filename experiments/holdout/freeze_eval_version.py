"""Freeze a real Augur eval-version from producer candidates (#894/#901).

The producer pipeline (#901/#902) emits a ``task_spec`` + a ``mark_for_eval``
candidate on each eval-worthy run. This script is the operator step that turns
that live candidate pool into an immutable, run-backed **eval-version** — the
frozen holdout the slow-loop champion/challenger gate evaluates against.

Two distinct auth scopes (live-verified against the workspace 2026-06-14):

* **Read** (list candidates / versions) — accepts the producer key as
  ``Authorization: Bearer <AUGUR_API_KEY>``. Works with what's in ``.env``.
* **Write** (``POST /eval-versions``, the freeze) — is **operator-session
  gated**. The producer key is rejected (401 "not signed in") on every header
  variant; only a signed-in browser **session cookie** authorizes it. Supply it
  via ``AUGUR_SESSION_COOKIE`` (the raw ``Cookie:`` header value copied from an
  authenticated workspace tab) or ``--session-cookie``.

Usage::

    # see what's freezable today (read-only, no session needed)
    python experiments/holdout/freeze_eval_version.py --list

    # freeze a named version from explicit task_spec_ids (needs session cookie)
    AUGUR_SESSION_COOKIE='session=...' \\
      python experiments/holdout/freeze_eval_version.py \\
        --name mantis-holdout-v1 \\
        --task-spec-id boattrader.bt01_lead_capture.v1 \\
        --task-spec-id boattrader.bt02_spec_lookup.v1

    # freeze every producer candidate matching a prefix
    AUGUR_SESSION_COOKIE='session=...' \\
      python experiments/holdout/freeze_eval_version.py \\
        --name mantis-holdout-v1 --select-prefix boattrader.

The HTTP calls are injected (``http_get`` / ``http_post``) so the selection +
freeze logic is unit-testable with no network — see
``tests/test_freeze_eval_version.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from typing import Any

import requests

DEFAULT_BASE = "https://mantis-cua.ngrok-free.app/api/v1"


def _default_tenant() -> str:
    """Tenant from ``AUGUR_TENANT`` env, else the ``tenant=`` param of
    ``AUGUR_DSN`` — never hard-code a tenant/customer name in source
    (tests/test_docs_client_isolation)."""
    t = (os.environ.get("AUGUR_TENANT") or "").strip()
    if t:
        return t
    dsn = os.environ.get("AUGUR_DSN", "")
    if "tenant=" in dsn:
        return dsn.split("tenant=", 1)[1].split("&", 1)[0]
    return ""

# An (status_code, json_body) HTTP seam so the logic is testable without network.
HttpGet = Callable[[str, dict[str, str]], "tuple[int, Any]"]
HttpPost = Callable[[str, dict[str, str], dict[str, Any]], "tuple[int, Any]"]


def _augur_token() -> str:
    """The producer read key — AUGUR_API_KEY, or the ``token`` from AUGUR_DSN."""
    tok = (os.environ.get("AUGUR_API_KEY") or "").strip()
    if tok:
        return tok
    dsn = os.environ.get("AUGUR_DSN", "")
    if "token=" in dsn:
        return dsn.split("token=", 1)[1].split("&", 1)[0]
    return ""


def _live_get(url: str, headers: dict[str, str]) -> tuple[int, Any]:
    r = requests.get(url, headers=headers, timeout=30)
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, r.text


def _live_post(url: str, headers: dict[str, str], body: dict[str, Any]) -> tuple[int, Any]:
    r = requests.post(url, headers=headers, json=body, timeout=30)
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, r.text


def list_candidates(
    *, base: str, tenant: str, token: str, http_get: HttpGet, limit: int = 200
) -> list[dict[str, Any]]:
    """Return the producer/operator eval-candidate pool (read scope: Bearer key)."""
    url = f"{base}/eval-candidates?tenant={tenant}&limit={limit}"
    headers = {"Authorization": f"Bearer {token}", "ngrok-skip-browser-warning": "1"}
    status, body = http_get(url, headers)
    if status != 200:
        raise RuntimeError(f"list candidates failed [{status}]: {body}")
    return list(body.get("candidates", [])) if isinstance(body, dict) else []


def select_task_spec_ids(
    candidates: list[dict[str, Any]],
    *,
    explicit: list[str] | None = None,
    select_prefix: str | None = None,
) -> list[str]:
    """Resolve which ``task_spec_id`` s to freeze, de-duplicated, order-stable.

    * ``explicit`` — exactly these ids; each MUST be present in the candidate
      pool (you can't freeze a task no run ever produced).
    * ``select_prefix`` — every candidate id starting with the prefix.
    * neither — every candidate id (freeze the whole live pool).
    """
    pool = []
    seen: set[str] = set()
    for c in candidates:
        tid = str(c.get("task_spec_id") or "").strip()
        if tid and tid not in seen:
            seen.add(tid)
            pool.append(tid)

    if explicit:
        missing = [t for t in explicit if t not in seen]
        if missing:
            raise ValueError(
                f"these task_spec_ids have no candidate in the pool (run them first): {missing}"
            )
        return list(dict.fromkeys(explicit))
    if select_prefix:
        return [t for t in pool if t.startswith(select_prefix)]
    return pool


def freeze_version(
    *,
    base: str,
    tenant: str,
    name: str,
    task_spec_ids: list[str],
    session_cookie: str,
    http_post: HttpPost,
    description: str = "",
) -> dict[str, Any]:
    """POST /eval-versions (write scope: operator session cookie required)."""
    if not session_cookie:
        raise ValueError(
            "freeze needs an operator session cookie (AUGUR_SESSION_COOKIE / "
            "--session-cookie). The producer key authorizes reads only."
        )
    if not task_spec_ids:
        raise ValueError("refusing to freeze an empty eval-version")
    url = f"{base}/eval-versions?tenant={tenant}"
    headers = {
        "Content-Type": "application/json",
        "Cookie": session_cookie,
        "ngrok-skip-browser-warning": "1",
    }
    body: dict[str, Any] = {"name": name, "task_spec_ids": task_spec_ids}
    if description:
        body["description"] = description
    status, resp = http_post(url, headers, body)
    if status not in (200, 201):
        raise RuntimeError(f"freeze failed [{status}]: {resp}")
    return resp if isinstance(resp, dict) else {"raw": resp}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Freeze an Augur eval-version from producer candidates")
    ap.add_argument("--base", default=os.environ.get("AUGUR_BASE", DEFAULT_BASE))
    ap.add_argument("--tenant", default=_default_tenant())
    ap.add_argument("--list", action="store_true", help="list freezable candidates and exit (read-only)")
    ap.add_argument("--name", help="eval-version name to freeze")
    ap.add_argument("--description", default="")
    ap.add_argument("--task-spec-id", action="append", dest="task_spec_ids", default=[])
    ap.add_argument("--select-prefix", default=None)
    ap.add_argument("--session-cookie", default=os.environ.get("AUGUR_SESSION_COOKIE", ""))
    args = ap.parse_args(argv)

    token = _augur_token()
    if not token:
        print("ERROR: no AUGUR_API_KEY / AUGUR_DSN token in environment", file=sys.stderr)
        return 2

    candidates = list_candidates(
        base=args.base, tenant=args.tenant, token=token, http_get=_live_get
    )
    if args.list:
        print(f"{len(candidates)} candidate(s) in pool:")
        for tid in select_task_spec_ids(candidates):
            n = sum(1 for c in candidates if c.get("task_spec_id") == tid)
            print(f"  {n:3d}  {tid}")
        return 0

    if not args.name:
        print("ERROR: --name is required to freeze (or use --list)", file=sys.stderr)
        return 2

    selected = select_task_spec_ids(
        candidates, explicit=args.task_spec_ids or None, select_prefix=args.select_prefix
    )
    print(f"freezing '{args.name}' with {len(selected)} task(s): {selected}")
    result = freeze_version(
        base=args.base,
        tenant=args.tenant,
        name=args.name,
        task_spec_ids=selected,
        session_cookie=args.session_cookie,
        http_post=_live_post,
        description=args.description,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
