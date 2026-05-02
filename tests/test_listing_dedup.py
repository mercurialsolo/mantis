"""Tests for #115 step 5 — ListingDedup extracted from MicroPlanRunner."""

from __future__ import annotations

from mantis_agent.gym.checkpoint import StepResult
from mantis_agent.gym.listing_dedup import ListingDedup


def _viable(data: str, *, success: bool = True, idx: int = 0) -> StepResult:
    return StepResult(step_index=idx, intent="extract", success=success, data=data)


# ── successful_lead_data ─────────────────────────────────────────────────


def test_successful_lead_data_keeps_only_viable_success_rows() -> None:
    rows = [
        _viable("VIABLE | URL: https://x.test/a", idx=1),
        _viable("REJECTED | spam | URL: https://x.test/b", idx=2),
        _viable("VIABLE | URL: https://x.test/c", success=False, idx=3),
        _viable("", idx=4),
        _viable("VIABLE | URL: https://x.test/d", idx=5),
    ]
    out = ListingDedup.successful_lead_data(rows)
    assert out == [
        "VIABLE | URL: https://x.test/a",
        "VIABLE | URL: https://x.test/d",
    ]


def test_successful_lead_data_handles_none_data() -> None:
    rows = [_viable(None, idx=1)]  # type: ignore[arg-type]
    rows[0].data = None  # type: ignore[assignment]
    assert ListingDedup.successful_lead_data(rows) == []


# ── lead_key ─────────────────────────────────────────────────────────────


def test_lead_key_extracts_url_field() -> None:
    assert (
        ListingDedup.lead_key("VIABLE | Year: 2024 | URL: https://x.test/a/b")
        == "https://x.test/a/b"
    )


def test_lead_key_falls_back_to_first_100_chars_when_no_url() -> None:
    data = "VIABLE | Year: 2024 | Make: Sea Ray | (no url field present)" + "x" * 200
    assert ListingDedup.lead_key(data) == data[:100]


def test_lead_key_strips_whitespace_around_url() -> None:
    assert (
        ListingDedup.lead_key("VIABLE | URL:   https://x.test/  ")
        == "https://x.test/"
    )


# ── lead_has_phone ──────────────────────────────────────────────────────


def test_lead_has_phone_detects_full_us_number() -> None:
    assert (
        ListingDedup.lead_has_phone("VIABLE | URL: x | Phone: 786-555-1234")
        is True
    )


def test_lead_has_phone_rejects_null_sentinels() -> None:
    for sentinel in ("none", "N/A", "Unknown", "not visible", "not shown", ""):
        row = f"VIABLE | URL: x | Phone: {sentinel}"
        assert ListingDedup.lead_has_phone(row) is False, sentinel


def test_lead_has_phone_rejects_when_field_missing() -> None:
    assert ListingDedup.lead_has_phone("VIABLE | URL: x | Year: 2024") is False


def test_lead_has_phone_requires_at_least_10_digits() -> None:
    assert ListingDedup.lead_has_phone("VIABLE | URL: x | Phone: 555-1234") is False
    assert ListingDedup.lead_has_phone("VIABLE | URL: x | Phone: 555-555-1234") is True


def test_lead_has_phone_strips_format_chars() -> None:
    assert (
        ListingDedup.lead_has_phone("VIABLE | URL: x | Phone: (555) 555.1234")
        is True
    )


# ── unique_leads_from_results ───────────────────────────────────────────


def test_unique_leads_dedups_by_url() -> None:
    rows = [
        _viable("VIABLE | URL: https://x.test/a | Phone: 555-555-1234", idx=1),
        # Same URL, different metadata — second occurrence wins.
        _viable("VIABLE | URL: https://x.test/a | Phone: none", idx=2),
        _viable("VIABLE | URL: https://x.test/b | Phone: 555-555-9999", idx=3),
    ]
    out = ListingDedup.unique_leads_from_results(rows)
    assert len(out) == 2
    keys = {ListingDedup.lead_key(r) for r in out}
    assert keys == {"https://x.test/a", "https://x.test/b"}


def test_unique_leads_empty_when_no_viable_rows() -> None:
    rows = [
        _viable("REJECTED | spam", idx=1),
        _viable("", idx=2),
    ]
    assert ListingDedup.unique_leads_from_results(rows) == []


# ── lead_counts ─────────────────────────────────────────────────────────


def test_lead_counts_returns_unique_and_phone_totals() -> None:
    rows = [
        _viable("VIABLE | URL: https://x.test/a | Phone: 786-555-1234", idx=1),
        _viable("VIABLE | URL: https://x.test/b | Phone: none", idx=2),
        # Duplicate URL — should not double-count.
        _viable("VIABLE | URL: https://x.test/a | Phone: 786-555-1234", idx=3),
        _viable("VIABLE | URL: https://x.test/c | Phone: 305-555-9999", idx=4),
    ]
    total, with_phone = ListingDedup.lead_counts(rows)
    assert total == 3
    assert with_phone == 2


def test_lead_counts_zero_when_no_viable() -> None:
    assert ListingDedup.lead_counts([_viable("REJECTED | spam", idx=1)]) == (0, 0)


def test_lead_counts_handles_dedup_when_same_url_has_phone_in_one_record() -> None:
    """Last record per URL wins (dict assignment) — verifies we count
    by the *kept* row, not the first occurrence."""
    rows = [
        # First occurrence has phone…
        _viable("VIABLE | URL: https://x.test/a | Phone: 786-555-1234", idx=1),
        # …second occurrence overwrites with phone=none. Count reflects last.
        _viable("VIABLE | URL: https://x.test/a | Phone: none", idx=2),
    ]
    total, with_phone = ListingDedup.lead_counts(rows)
    assert total == 1
    assert with_phone == 0


# ── Backward-compat: MicroPlanRunner shims still resolve ───────────────


def test_runner_static_helpers_delegate_to_listing_dedup() -> None:
    from mantis_agent.gym.micro_runner import MicroPlanRunner

    rows = [
        _viable("VIABLE | URL: https://x.test/a | Phone: 786-555-1234", idx=1),
        _viable("VIABLE | URL: https://x.test/b | Phone: none", idx=2),
    ]
    assert MicroPlanRunner._lead_counts(rows) == (2, 1)
    assert MicroPlanRunner._unique_leads_from_results(rows) == [
        "VIABLE | URL: https://x.test/a | Phone: 786-555-1234",
        "VIABLE | URL: https://x.test/b | Phone: none",
    ]
    assert MicroPlanRunner._lead_key("VIABLE | URL: https://x.test/a") == (
        "https://x.test/a"
    )
    assert MicroPlanRunner._lead_has_phone("VIABLE | Phone: 555-555-1234") is True
