"""Human-actionable help for terminal-failure runs (#841 follow-up).

Pre-fix every terminal-failure response carried a raw Python exception
repr in ``status.error`` (e.g. ``ConnectionResetError(104, 'Connection
reset by peer')``) and a machine-readable token in ``halt_reason``
(e.g. ``page_blocked``). Both grep-able, neither human-actionable —
the user has no way to know "what does this mean / what should I do".

This module turns a ``halt_class`` (the stable wire constant set by
the runner) into a ``failure_help`` dict carrying:

- ``summary`` — one-sentence what-went-wrong, plain English.
- ``likely_causes`` — bulleted hypotheses ranked by probability.
- ``next_steps`` — bulleted operator actions, in order of cheapness.
- ``debug_surfaces`` — direct URL pointers to per-run debug surfaces
  (SSE event stream, Augur bundle, action=logs) so the user doesn't
  have to know they exist.

Adding a new failure class means adding one entry below — help text
stays in ONE place, not scattered across runtime call sites.

CUA-contract provenance: this is pure post-processing of taxonomy
data the runner already produces. No DOM access, no derived
grounding (`feedback_cua_no_dom_access.md`).
"""

from __future__ import annotations

import os
from typing import Any


# Stable wire constant → help dict. Keep entries terse — operators
# read these in JSON status payloads, not docs.
#
# Convention: every entry has a ``summary``, ``likely_causes``, and
# ``next_steps``. ``debug_surfaces`` is injected per-run by
# :func:`failure_help_for` (depends on run_id).
_HALT_CLASS_HELP: dict[str, dict[str, Any]] = {
    # Anthropic-reachability cluster
    "anthropic_unreachable": {
        "summary": (
            "Could not reach the Anthropic API after the retry budget "
            "was exhausted."
        ),
        "likely_causes": [
            "Transient Anthropic 5xx / 529 Overloaded during peak hours",
            "Per-IP rate limit on the Modal egress shared pool",
            "Deployed image is stale (>14 days) and its TLS / certifi "
            "stack hasn't been refreshed",
            "Anthropic API key revoked or hit account quota",
        ],
        "next_steps": [
            "Wait 60s and retry — the retry policy backs off exponentially",
            "Check ``GET /v1/version`` — if ``deploy_age_days > 14``, "
            "redeploy from current main (#840 surfaces this directly)",
            "Verify ANTHROPIC_API_KEY hasn't been rotated in the Modal Secret",
        ],
    },
    # CF / DataDome / Akamai blocking cluster
    "cf_challenge": {
        "summary": (
            "The target site returned a Cloudflare Turnstile challenge "
            "that the runner could not auto-solve."
        ),
        "likely_causes": [
            "Stealth posture is leaking signal (TLS / UA / WebGL mismatch)",
            "Residential proxy IP is on a CF reputation blocklist",
            "First-paint settle was too short — Turnstile JS hadn't loaded",
        ],
        "next_steps": [
            "Use the live-viewer URL on this run's status to attempt the "
            "challenge interactively (auto-pause path from #541)",
            "Try a different ``proxy_country`` / ``proxy_city`` to rotate "
            "the egress IP",
            "Run the fingerprint diagnostic (#827) to confirm stealth "
            "posture: ``POST /v1/diagnose/fingerprint``",
        ],
    },
    "page_blocked": {
        "summary": (
            "The browser landed on a page the runner classified as a "
            "block / interstitial (CF, paywall, geo-restriction)."
        ),
        "likely_causes": [
            "Cloudflare Turnstile or similar bot-detection",
            "Proxy egressed to a geo the site doesn't serve",
            "Cookie-consent overlay tripping the listings scanner "
            "(`feedback_page_blocked_is_find_all_listings_overlay`)",
        ],
        "next_steps": [
            "Open the live-viewer URL to see what the page actually shows",
            "If CF: see ``cf_challenge`` guidance — proxy rotation + "
            "fingerprint diagnostic",
            "If cookie banner: pre-seed the dismissing cookie via the "
            "header seam",
        ],
    },
    # Navigation-shape cluster
    "navigation_drift": {
        "summary": (
            "A step ran against a page different from the one the "
            "preceding navigate landed on."
        ),
        "likely_causes": [
            "Proxy or CF challenge redirected mid-step",
            "Decomposer emitted a click that left the expected page",
            "Slow first-paint racing into the next step (lower "
            "``wait_after_load_seconds`` than needed)",
        ],
        "next_steps": [
            "View the per-step event trace at ``/v1/runs/<run_id>/events?"
            "sse=true`` — look at the ``url`` field on each step",
            "Increase ``params.wait_after_load_seconds`` on the navigate step",
            "If the plan didn't intend to click: add ``read_only`` "
            "phrasing (#831 honors it) or set ``params.expect_url_contains`` "
            "explicitly",
        ],
    },
    "navigate_failed": {
        "summary": "The runner could not load the requested URL.",
        "likely_causes": [
            "DNS failure / proxy didn't tunnel CONNECT",
            "Site returned non-200 (4xx / 5xx)",
            "Bad URL in the plan",
        ],
        "next_steps": [
            "Check the navigate step's ``params.url`` is reachable from "
            "your browser",
            "If using a proxy: verify ``proxy_provider`` env vars are set; "
            "default is now ``oxylabs`` (see "
            "``feedback_proxy_provider.md``)",
            "View action=logs to read the runner's per-step failure text",
        ],
    },
    "bad_url": {
        "summary": (
            "Post-navigate URL classification flagged the page as bad "
            "(404, soft-404, wrong-domain, or DNS failure)."
        ),
        "likely_causes": [
            "Plan-author URL no longer exists",
            "Proxy redirected to a captive portal",
            "Site changed its URL structure",
        ],
        "next_steps": [
            "Run the URL through the host's own browser; if it 404s, "
            "update the plan",
            "View ``page_title`` in the run's status — confirms what the "
            "browser actually saw",
        ],
    },
    # Extraction-shape cluster
    "extract_data_failed": {
        "summary": (
            "Claude could not produce a viable extraction for a step "
            "that required one."
        ),
        "likely_causes": [
            "Schema mismatch — the requested fields aren't visible on the page",
            "No inline ``extract`` block and no recipe-bound schema "
            "(``no_schema_configured``) — see the WARNING at extract step entry",
            "Page didn't fully load before the screenshot",
        ],
        "next_steps": [
            "Add an inline ``extract`` block to the step (see "
            "``docs/client/plans.md#inline-extraction-schema``)",
            "Increase ``wait_after_load_seconds`` if the page is JS-heavy",
            "View ``/v1/runs/<run_id>/events?sse=true`` to see the "
            "extractor's per-step result",
        ],
    },
    "no_schema_configured": {
        "summary": (
            "An extract step ran with no schema bound — every row was "
            "rejected with ``no_schema_configured``."
        ),
        "likely_causes": [
            "Plan author didn't include an inline ``extract`` block AND "
            "no recipe was loaded",
        ],
        "next_steps": [
            "Add an inline ``extract`` block on the step (see "
            "``docs/client/plans.md#inline-extraction-schema``)",
            "Or register a recipe via ``POST /v1/recipes`` (#809)",
        ],
    },
    # Budget / time caps
    "budget_cap": {
        "summary": "The run hit the configured ``max_cost`` ceiling.",
        "likely_causes": [
            "Plan is too long for the budget",
            "A retry loop burned through the budget on transient failures",
        ],
        "next_steps": [
            "Raise ``max_cost`` on the next submission",
            "Inspect per-step costs via ``action=result`` — find which "
            "step was expensive",
        ],
    },
    "time_cap": {
        "summary": "The run hit the configured ``max_time_minutes`` ceiling.",
        "likely_causes": [
            "Plan is too long / has too many steps for the time budget",
            "Slow proxy / slow first-paint on every step",
        ],
        "next_steps": [
            "Raise ``max_time_minutes`` on the next submission",
            "Profile slow steps via ``mantis_step_latency_seconds{phase}`` "
            "in the per-action Grafana dashboard",
        ],
    },
    "halt_timeout": {
        "summary": "The run exceeded its lifecycle-level timeout.",
        "likely_causes": [
            "Plan is too long for the lifecycle timeout",
            "Run was stuck waiting on Anthropic / proxy / GPU and got "
            "killed by the heartbeat",
        ],
        "next_steps": [
            "Raise the per-run timeout when submitting",
            "View action=logs to find where the run was stuck",
        ],
    },
    # User-initiated
    "cancelled": {
        "summary": "The run was cancelled by an explicit ``POST /v1/runs/{id}/cancel``.",
        "likely_causes": ["Operator-initiated cancellation"],
        "next_steps": [
            "Resubmit the plan if the cancellation was a mistake",
        ],
    },
}


# Single-source-of-truth debug-surface URLs, parameterized per run.
def _debug_surfaces(run_id: str) -> dict[str, str]:
    return {
        "phase":  f"/v1/runs/{run_id}",
        "status": f"/v1/runs/{run_id}/status",
        "events": f"/v1/runs/{run_id}/events?sse=true",
        "result": f"/v1/runs/{run_id}/result",
        "augur":  f"/v1/runs/{run_id}/augur",
        "logs": (
            "POST /v1/predict {action: logs, run_id: " + run_id + "}"
        ),
    }


# Fallback help when the halt_class isn't in the table. Used so an
# unknown halt_class still gets a non-empty failure_help (operator at
# least sees the debug surface pointers).
_DEFAULT_HELP: dict[str, Any] = {
    "summary": (
        "The run terminated with a halt class that isn't in the help "
        "taxonomy yet — please file an issue with the halt_class value "
        "and we'll add it."
    ),
    "likely_causes": [
        "An unhandled exception path the runner hadn't classified yet",
    ],
    "next_steps": [
        "View the per-step event trace via the ``events`` debug surface",
        "Check ``action=logs`` for the raw runner output",
    ],
}


def failure_help_for(
    halt_class: str,
    *,
    run_id: str,
    retries_spent: int | None = None,
    extra_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the user-facing ``failure_help`` dict for a terminal run.

    Args:
        halt_class: Stable wire constant set by the runner
            (e.g. ``"cf_challenge"``, ``"anthropic_unreachable"``).
            Unknown values fall back to a default help dict that
            still surfaces the debug URLs.
        run_id: The run id used to parameterize the
            ``debug_surfaces`` URLs.
        retries_spent: Optional — surface the retry budget spent
            on the run. Pulled into ``next_steps`` reasoning when set.
        extra_context: Optional per-call additions (e.g. the actual
            URL that drifted, the step index that failed) merged
            into the response under ``context``.
    """
    halt_class = (halt_class or "").strip().lower()
    base = _HALT_CLASS_HELP.get(halt_class, _DEFAULT_HELP)
    out: dict[str, Any] = {
        "halt_class": halt_class or "unknown",
        "summary": base["summary"],
        "likely_causes": list(base["likely_causes"]),
        "next_steps": list(base["next_steps"]),
        "debug_surfaces": _debug_surfaces(run_id),
    }
    if retries_spent is not None and retries_spent > 0:
        out["retries_spent"] = int(retries_spent)
    if extra_context:
        out["context"] = dict(extra_context)
    # Sticky operator hint about deploy age when AUGUR or
    # MANTIS_DEPLOY_AGE_WARN env signals staleness (#840 placeholder).
    deploy_warn = os.environ.get("MANTIS_DEPLOY_AGE_WARN", "").strip()
    if deploy_warn:
        out["deploy_age_warning"] = deploy_warn
    return out


def known_halt_classes() -> list[str]:
    """Snapshot of currently-documented halt classes — exposed for tests
    and for the ``GET /v1/diagnose/halt-classes`` endpoint (future)."""
    return sorted(_HALT_CLASS_HELP.keys())


__all__ = [
    "failure_help_for",
    "known_halt_classes",
]
