"""Per-run lead CSV writer.

Walks the micro-runner's terminal step results and writes one row per
lead to::

    /data/runs/<tenant_id>/<run_id>/leads.csv

Two extraction-shape paths are supported (use whichever the step
produced):

1. **Structured ``extracted_fields``** — schema-keyed dict produced by
   the listings-flow extract_data path (claude_step.py:475). Each
   key/value becomes a CSV column.

2. **Pipe-delimited ``VIABLE | k: v | ...`` string in ``data``** — the
   canonical row format produced by the marketplace recipe's deep-
   extraction path (see ListingDedup._VIABLE_PREFIX). Parsed back into
   a flat dict so the columns are equivalent between paths.

Without the second path, the writer misses every lead the marketplace
recipe produces — the lead-counter in ``run_reporter`` reads
``r.data``, so an empty CSV next to a "3 viable leads" log line is a
wiring gap, not zero leads.

Columns are the union of all field keys seen in the run, with
``step_index``, ``step_intent``, ``final_url``, and ``viable`` prepended
for traceability. ``run_id`` and ``profile_id`` are included so multi-
run CSVs (one per zip / proxy geo / model variant) can be concatenated
later with ``pandas.concat`` or shell ``cat``.

Best-effort throughout — errors are swallowed at DEBUG and the run
terminal never fails because of CSV I/O.
"""

from __future__ import annotations

import csv
import logging
import os
import re
from typing import Any, Iterable

logger = logging.getLogger(__name__)


_FIXED_HEAD_COLS = (
    "run_id",
    "profile_id",
    "step_index",
    "step_intent",
    "final_url",
    "viable",
)


_VIABLE_PREFIX = "VIABLE"
# Match "Field: value" pairs from "VIABLE | Year: 2022 | Make: Bayliner | ..."
_KV_PAIR_RE = re.compile(r"([A-Za-z][A-Za-z0-9 _]*?)\s*:\s*([^|]*?)(?=\s*\||$)")


def _parse_viable_row(data: str) -> dict[str, Any]:
    """Parse a marketplace-recipe VIABLE row into a {field: value} dict.

    Input shape (canonical, from ListingDedup):
        "VIABLE | Year: 2022 | Make: Bayliner | Model: T22CC | URL: ... | Phone: 555-..."
    Returns:
        {"Year": "2022", "Make": "Bayliner", ...} — keys preserved as-is
        from the source row. Empty dict on parse failure.
    """
    if not data or not data.startswith(_VIABLE_PREFIX):
        return {}
    out: dict[str, Any] = {}
    for m in _KV_PAIR_RE.finditer(data):
        key = m.group(1).strip()
        if not key or key.upper() == _VIABLE_PREFIX:
            continue
        out[key] = m.group(2).strip()
    return out


def _sanitize(s: str) -> str:
    return "".join(c if c.isalnum() or c in "_-." else "_" for c in (s or ""))[:120]


def _output_path(*, tenant_id: str, run_id: str) -> str:
    root = os.environ.get("MANTIS_RUN_ARTIFACTS_DIR", "/data/runs")
    safe_tenant = _sanitize(tenant_id) or "default"
    safe_run = _sanitize(run_id) or "unknown"
    return os.path.join(root, safe_tenant, safe_run, "leads.csv")


def _rows_from_results(
    results: Iterable[Any], *, run_id: str, profile_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in results:
        # Path 1: structured extracted_fields (listings-flow extract_data)
        fields = getattr(r, "extracted_fields", None)
        # Path 2: pipe-delimited VIABLE row in data (marketplace recipe)
        data = getattr(r, "data", None) or ""
        viable_fields: dict[str, Any] = {}
        if isinstance(data, str) and data.startswith(_VIABLE_PREFIX):
            viable_fields = _parse_viable_row(data)
        # Skip this step if neither path produced fields
        has_fields = bool(fields) if isinstance(fields, dict) else False
        if not has_fields and not viable_fields:
            continue
        row: dict[str, Any] = {
            "run_id": run_id,
            "profile_id": profile_id,
            "step_index": getattr(r, "step_index", ""),
            "step_intent": (getattr(r, "intent", "") or "")[:200],
            "final_url": getattr(r, "final_url", "") or "",
            "viable": bool(viable_fields),
        }
        # Merge both — extracted_fields keys win on overlap (they're
        # schema-typed, viable parse is regex-best-effort).
        merged: dict[str, Any] = {}
        for k, v in viable_fields.items():
            merged[str(k)] = v
        if isinstance(fields, dict):
            for k, v in fields.items():
                if isinstance(v, (str, int, float, bool)) or v is None:
                    merged[str(k)] = v
                else:
                    merged[str(k)] = repr(v)
        row.update(merged)
        rows.append(row)
    return rows


def finalize_leads_csv(
    *, run_id: str, tenant_id: str, profile_id: str,
    results: Iterable[Any],
) -> str:
    """Write one CSV row per step with non-empty ``extracted_fields``.

    Returns the path written, or empty string when there's nothing to
    write (no run_id, no extracting steps). Idempotent: rewrites the
    same path on each call.
    """
    if not run_id:
        return ""
    rows = _rows_from_results(
        results, run_id=run_id, profile_id=profile_id or "",
    )
    # Always emit a CSV at run terminal — even when no leads landed —
    # so operators see the writer ran. Empty-result file has just the
    # fixed header columns (run_id / profile_id / step_index / etc.),
    # which is enough to confirm the artifact path is correct and lets
    # downstream tooling treat "no leads" as a valid outcome instead of
    # a missing file.

    extra_cols: list[str] = []
    seen: set[str] = set(_FIXED_HEAD_COLS)
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                extra_cols.append(k)
    fieldnames = list(_FIXED_HEAD_COLS) + extra_cols

    path = _output_path(tenant_id=tenant_id, run_id=run_id)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.tmp.{os.getpid()}"
        with open(tmp_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        os.replace(tmp_path, path)
        logger.warning(
            "  [leads-csv] wrote %s: %d lead rows, %d columns",
            path, len(rows), len(fieldnames),
        )
        return path
    except Exception as exc:  # noqa: BLE001
        logger.debug("leads_csv finalize failed: %s", exc)
        return ""
