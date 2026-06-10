"""Tests for ``run_failure_help`` — user-actionable failure messages (#841).

Each known halt class must produce a ``failure_help`` dict carrying
a summary, likely causes, next steps, and debug-surface URLs. Unknown
classes fall back to a default that still surfaces the URLs.
"""

from __future__ import annotations

import pytest

from mantis_agent.run_failure_help import (
    failure_help_for,
    known_halt_classes,
)


# ── Coverage of the taxonomy ──────────────────────────────────────


def test_known_halt_classes_returns_non_empty_list():
    classes = known_halt_classes()
    assert isinstance(classes, list)
    assert len(classes) >= 5  # at minimum: anthropic, cf, drift, extract, budget
    assert classes == sorted(classes)


@pytest.mark.parametrize("halt_class", [
    "anthropic_unreachable",
    "cf_challenge",
    "page_blocked",
    "navigation_drift",
    "navigate_failed",
    "bad_url",
    "extract_data_failed",
    "no_schema_configured",
    "budget_cap",
    "time_cap",
    "halt_timeout",
    "cancelled",
])
def test_every_documented_class_has_full_help(halt_class):
    """Each entry must have summary + likely_causes + next_steps."""
    help_dict = failure_help_for(halt_class, run_id="r1")
    assert help_dict["summary"]
    assert help_dict["likely_causes"]
    assert help_dict["next_steps"]
    assert len(help_dict["likely_causes"]) >= 1
    assert len(help_dict["next_steps"]) >= 1
    assert help_dict["halt_class"] == halt_class


# ── Debug-surface URLs ────────────────────────────────────────────


def test_debug_surfaces_carry_run_id():
    help_dict = failure_help_for("anthropic_unreachable", run_id="r-abc-123")
    surfaces = help_dict["debug_surfaces"]
    assert "/v1/runs/r-abc-123" in surfaces["phase"]
    assert "/v1/runs/r-abc-123/events?sse=true" in surfaces["events"]
    assert "/v1/runs/r-abc-123/augur" in surfaces["augur"]
    assert "r-abc-123" in surfaces["logs"]


def test_debug_surfaces_include_all_expected_keys():
    surfaces = failure_help_for("cf_challenge", run_id="r1")["debug_surfaces"]
    for key in ("phase", "status", "events", "result", "augur", "logs"):
        assert key in surfaces, f"missing debug surface: {key}"


# ── Unknown halt class falls back ─────────────────────────────────


def test_unknown_halt_class_returns_fallback_help():
    """Even an unrecognized halt_class must produce something useful —
    the user at least gets debug-surface pointers."""
    help_dict = failure_help_for("brand_new_failure_mode", run_id="r1")
    assert help_dict["summary"]
    assert help_dict["next_steps"]
    assert help_dict["debug_surfaces"]
    assert help_dict["halt_class"] == "brand_new_failure_mode"


def test_empty_halt_class_returns_unknown_label():
    help_dict = failure_help_for("", run_id="r1")
    assert help_dict["halt_class"] == "unknown"
    assert help_dict["debug_surfaces"]


def test_none_handled_gracefully():
    """Defensive — the runner sometimes passes None when it can't
    classify the halt."""
    help_dict = failure_help_for(None, run_id="r1")  # type: ignore[arg-type]
    assert help_dict["halt_class"] == "unknown"


# ── Optional fields ───────────────────────────────────────────────


def test_retries_spent_omitted_when_none():
    help_dict = failure_help_for("anthropic_unreachable", run_id="r1")
    assert "retries_spent" not in help_dict


def test_retries_spent_surfaced_when_provided():
    help_dict = failure_help_for(
        "anthropic_unreachable", run_id="r1", retries_spent=3,
    )
    assert help_dict["retries_spent"] == 3


def test_retries_spent_zero_is_omitted():
    """A successful first-attempt has retries_spent=0 — surfacing
    that adds noise without information."""
    help_dict = failure_help_for(
        "anthropic_unreachable", run_id="r1", retries_spent=0,
    )
    assert "retries_spent" not in help_dict


def test_extra_context_merged_under_context_key():
    help_dict = failure_help_for(
        "navigation_drift", run_id="r1",
        extra_context={
            "expected": "news.ycombinator.com/newest",
            "got": "news.ycombinator.com/login",
            "step_index": 4,
        },
    )
    assert help_dict["context"]["expected"] == "news.ycombinator.com/newest"
    assert help_dict["context"]["step_index"] == 4


# ── Deploy-age signal ─────────────────────────────────────────────


def test_deploy_age_warning_surfaced_when_env_set(monkeypatch):
    """When ``MANTIS_DEPLOY_AGE_WARN`` env is set (#840 placeholder),
    the failure_help carries it so the operator sees the staleness
    on every failed-run response."""
    monkeypatch.setenv(
        "MANTIS_DEPLOY_AGE_WARN",
        "deploy is 24 days old; consider redeploying",
    )
    help_dict = failure_help_for("anthropic_unreachable", run_id="r1")
    assert "deploy_age_warning" in help_dict
    assert "24 days" in help_dict["deploy_age_warning"]


def test_deploy_age_warning_omitted_by_default(monkeypatch):
    monkeypatch.delenv("MANTIS_DEPLOY_AGE_WARN", raising=False)
    help_dict = failure_help_for("anthropic_unreachable", run_id="r1")
    assert "deploy_age_warning" not in help_dict


# ── Canonical case: the user's actual failure ────────────────────


def test_anthropic_unreachable_help_mentions_deploy_age(monkeypatch):
    """The HN-feedback failure's help text should specifically point
    the user at #840 / deploy age — that was the root cause."""
    monkeypatch.delenv("MANTIS_DEPLOY_AGE_WARN", raising=False)
    help_dict = failure_help_for("anthropic_unreachable", run_id="r1")
    joined = " ".join(help_dict["next_steps"] + help_dict["likely_causes"])
    assert "deploy" in joined.lower() or "stale" in joined.lower()
    assert "version" in joined.lower()


def test_cf_challenge_help_mentions_viewer():
    """CF blocks are the canonical "open the live viewer" case."""
    help_dict = failure_help_for("cf_challenge", run_id="r1")
    joined = " ".join(help_dict["next_steps"])
    assert "viewer" in joined.lower()


def test_navigation_drift_help_mentions_read_only():
    """Drift is fixed by either expect_url_contains or read-only intent."""
    help_dict = failure_help_for("navigation_drift", run_id="r1")
    joined = " ".join(help_dict["next_steps"])
    assert "read_only" in joined or "read-only" in joined.lower() or "expect_url" in joined
