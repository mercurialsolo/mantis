"""Regression test for the ``VIABLE`` prefix convention in
``WorkflowRunner._extract_data``.

Bug: when the brain's done() summary already contains structured
fields (e.g. ``"Year: 2024 | Make: X | URL: https://..."``), the
Priority-1 branch returned the summary AS-IS — without the
``VIABLE | `` prefix. Downstream, ``ListingDedup.successful_lead_data``
filters StepResults via ``data.startswith("VIABLE")`` and silently
drops any lead that lacks the prefix. The Priority-2 (synthetic
reconstruction from thinking) branch DID prepend the prefix, but the
happy-path Priority-1 (brain emitted structured data itself) was
the regression source — runs that extracted real data reported
"0 viable leads" because of the convention mismatch.

Pin the fix: when Priority-1 fires, the returned string must start
with ``VIABLE | ``.
"""

from __future__ import annotations

from types import SimpleNamespace

from mantis_agent.actions import Action, ActionType
from mantis_agent.gym.workflow_runner import WorkflowRunner


def _result(summary: str):
    """Minimal RunResult duck with a done()-bearing trajectory."""
    done_step = SimpleNamespace(
        action=Action(ActionType.DONE, {"success": True, "summary": summary}),
        thinking="",
    )
    return SimpleNamespace(trajectory=[done_step])


def test_priority1_prepends_viable_prefix_when_missing() -> None:
    """Brain emitted structured data without the prefix — _extract_data
    must add it so ListingDedup counts the lead."""
    summary = (
        "Year: 2024 | Make: Beneteau | Model: Idylle 15.50 | "
        "Price: $130,000 | URL: https://example.test/boat/x"
    )
    out = WorkflowRunner._extract_data(_result(summary))
    assert out.startswith("VIABLE | ")
    # Original fields preserved after the prefix.
    assert "Year: 2024" in out
    assert "Make: Beneteau" in out
    assert "URL: https://example.test/boat/x" in out


def test_priority1_leaves_existing_viable_prefix_intact() -> None:
    """When the brain already emitted ``VIABLE | ...`` (rare but
    possible if a future plan teaches it the convention), don't
    double-prefix."""
    summary = "VIABLE | Year: 2022 | Make: SeaRay | URL: https://example.test/2"
    out = WorkflowRunner._extract_data(_result(summary))
    assert out.startswith("VIABLE | Year")
    # NOT ``VIABLE | VIABLE | ...``
    assert not out.startswith("VIABLE | VIABLE")


def test_priority1_prefix_holds_when_url_already_present() -> None:
    """The summary already contains a URL — Priority-1 returns the
    summary as-is (after prefix injection). Ensures the URL-backfill
    branch and the prefix-injection branch compose correctly."""
    summary = (
        "Year: 2024 | Make: Foo | URL: https://example.test/boat/2024-foo"
    )
    out = WorkflowRunner._extract_data(_result(summary))
    assert out.startswith("VIABLE | ")
    assert "URL: https://example.test/boat/2024-foo" in out


def test_priority2_synthetic_path_unchanged() -> None:
    """The Priority-2 branch (synthetic reconstruction from thinking
    when done() summary lacks Year) was already correct — pin that the
    fix didn't regress it. Returned string still starts with
    ``VIABLE | `` from the synthetic builder."""
    done_step = SimpleNamespace(
        action=Action(
            ActionType.DONE, {"success": True, "summary": "task complete"},
        ),
        thinking=(
            "I found a 2020 Sea Ray 320 at $145,000. Listing URL "
            "is https://example.test/boat/2020-searay/. Seller is "
            "John at 555-867-5309."
        ),
    )
    out = WorkflowRunner._extract_data(SimpleNamespace(trajectory=[done_step]))
    assert out.startswith("VIABLE | ")
