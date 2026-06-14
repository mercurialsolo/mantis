"""Oracles — server-side ground-truth graders for the mantis-auth plans.

Each oracle exports a ``grade(conn, *, now, seed_val)`` returning the
harness oracle shape::

    {"passed": bool, "score": float, "reasons": [...], "diff": {...}}

The dispatch table maps ``task_id`` → grader. New auth scenarios land a
grader (usually a one-liner over ``scenarios``/``_common``) plus a
``GRADERS`` entry.

Determinism: oracles read DB + ``mutations`` state only, never the
agent's transcript. Same end state → same verdict, so the "N reruns → N
identical scores" criterion holds by construction.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Callable

from . import scenarios

GraderFn = Callable[..., dict[str, Any]]


GRADERS: dict[str, GraderFn] = {
    "T01_password_login": scenarios.grade_password,
    "T02_oauth_google": scenarios.make_oauth_grader("google"),
    "T03_oauth_github": scenarios.make_oauth_grader("github"),
    "T04_oauth_microsoft": scenarios.make_oauth_grader("microsoft"),
    "T05_oauth_okta": scenarios.make_oauth_grader("okta"),
    "T06_magic_link_email": scenarios.grade_magic_link,
    "T07_email_otp": scenarios.grade_email_otp,
    "T08_passkey": scenarios.grade_passkey,
}


def grade(task_id: str, conn: sqlite3.Connection, *,
          now: str, seed_val: int) -> dict[str, Any]:
    fn = GRADERS.get(task_id)
    if fn is None:
        return {
            "passed": False,
            "score": 0.0,
            "task_id": task_id,
            "reasons": [f"no oracle registered for task_id={task_id!r}"],
            "diff": {},
        }
    result = fn(conn, now=now, seed_val=seed_val)
    result.setdefault("task_id", task_id)
    return result
