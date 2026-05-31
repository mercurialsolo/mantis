"""Oracles — server-side graders for boattrader plan tasks.

Each oracle module exports a single
``grade(store, *, now, seed_val) -> {"passed": bool, "score": float, "reasons": list, "diff": dict}``
function that reads the in-memory ``Store`` (catalog + leads +
``mutations`` audit log) and returns the harness oracle response.

The dispatch table at the bottom maps ``task_id`` → grader. New plans
land their grader as a new file + an entry in :data:`GRADERS`.

Design notes
------------

* Oracles are deterministic: same store snapshot → same verdict. The
  ``mutations`` log is append-only, so reading it after the run
  yields the exact same sequence regardless of how many times we
  call :func:`grade`.
* Each oracle computes both the "right set" (mutations that satisfy
  the task) and the "wrong set" (mutations that don't). Stray
  collateral-damage mutations are a fail, even if every target was
  also hit.
* In-memory state matches the rest of the env — no SQLite. The shape
  mirrors the CRM oracle dispatcher (which reads a SQLite ``conn``)
  except the first arg is the in-memory ``Store`` instance.
"""

from __future__ import annotations

from typing import Any, Callable

from .. import db
from . import (
    bt01_lead_capture_filtered_search,
    bt02_spec_lookup_engine,
    bt03_byowner_phone_reveal,
)

GraderFn = Callable[..., dict[str, Any]]


GRADERS: dict[str, GraderFn] = {
    "BT01_lead_capture_filtered_search": bt01_lead_capture_filtered_search.grade,
    "BT02_spec_lookup_engine": bt02_spec_lookup_engine.grade,
    "BT03_byowner_phone_reveal": bt03_byowner_phone_reveal.grade,
}


def grade(
    task_id: str,
    store: db.Store,
    *,
    now: str,
    seed_val: int,
) -> dict[str, Any]:
    """Dispatch a grade request to the registered grader for ``task_id``.

    Returns the canonical harness response shape even for unknown
    task ids so the caller (``/__env__/oracle`` route, ``gym/grading.py``)
    never has to special-case the missing-grader branch.
    """
    fn = GRADERS.get(task_id)
    if fn is None:
        return {
            "passed": False,
            "score": 0.0,
            "task_id": task_id,
            "reasons": [f"no oracle registered for task_id={task_id!r}"],
            "diff": {},
        }
    result = fn(store, now=now, seed_val=seed_val)
    result.setdefault("task_id", task_id)
    return result


__all__ = ["GRADERS", "GraderFn", "grade"]
