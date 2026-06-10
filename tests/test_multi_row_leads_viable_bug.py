"""Tests for the multi-row leads/viable counter bug (#820 follow-up).

User report this session: schema-based extractions return 0 viable
leads even when the multi-row primitive logs ``5/5 rows captured``.
Root cause: ``ListingDedup.successful_lead_data`` only counted step
results whose ``data`` field started with ``VIABLE | ``, but the
multi-row branch (#820 ``_execute_rows``) sets ``data="extract_rows:5/5"``.
The 5 rows landed correctly in the ``extracted_rows.json`` artifact
but never reached the leads list or the viable counter.

Fix: ``successful_lead_data`` now synthesizes one ``VIABLE`` summary
per row in ``extracted_rows`` so the counter and downstream consumers
agree with the artifact pipeline.
"""

from __future__ import annotations

from mantis_agent.gym.checkpoint import StepResult
from mantis_agent.gym.listing_dedup import ListingDedup


def test_single_row_legacy_path_still_works():
    """Pre-fix the legacy single-row path returned ``r.data`` verbatim.
    Multi-row support must not regress that."""
    r = StepResult(
        step_index=0, intent="x", success=True,
        data="VIABLE | Year: 2020 | Make: Sea Ray",
    )
    leads = ListingDedup.successful_lead_data([r])
    assert leads == ["VIABLE | Year: 2020 | Make: Sea Ray"]


def test_multi_row_emits_one_lead_per_row():
    """The exact failure shape: extract_rows captured 5 rows; pre-fix
    leads was empty. Post-fix we get 5 ``VIABLE | ...`` strings."""
    r = StepResult(
        step_index=1, intent="extract top 5",
        success=True,
        data="extract_rows:5/5",
        extracted_rows=[
            {"rank": "1", "title": "Building an HTML-first site"},
            {"rank": "2", "title": "AMA: Eric Ries"},
            {"rank": "3", "title": "PgDog is funded"},
            {"rank": "4", "title": "Apache Burr"},
            {"rank": "5", "title": "Show HN: New thing"},
        ],
    )
    leads = ListingDedup.successful_lead_data([r])
    assert len(leads) == 5
    assert "VIABLE | Rank: 1 | Title: Building an HTML-first site" in leads
    assert "VIABLE | Rank: 5 | Title: Show HN: New thing" in leads


def test_multi_row_omits_empty_fields():
    """Per-row summary skips empty values so we don't get
    ``Author: `` clutter on rows that don't have author populated."""
    r = StepResult(
        step_index=0, intent="x", success=True,
        data="extract_rows:1/5",
        extracted_rows=[{"rank": "1", "title": "Real Title", "author": ""}],
    )
    leads = ListingDedup.successful_lead_data([r])
    assert len(leads) == 1
    assert "Real Title" in leads[0]
    assert "Author:" not in leads[0]


def test_multi_row_humanizes_snake_case():
    """``snake_case`` field names render as ``Title Case`` so the
    output reads as a sentence, matching the single-row to_summary
    convention."""
    r = StepResult(
        step_index=0, intent="x", success=True,
        data="extract_rows:1/1",
        extracted_rows=[{"story_url": "https://x.com", "comments_count": "42"}],
    )
    leads = ListingDedup.successful_lead_data([r])
    assert "Story Url" in leads[0]
    assert "Comments Count" in leads[0]


def test_failed_step_contributes_nothing():
    """A failed multi-row step must not count even if rows were
    partially captured."""
    r = StepResult(
        step_index=0, intent="x", success=False,
        data="extract_rows:0/5",
        extracted_rows=[{"rank": "1", "title": "captured-but-failed"}],
    )
    leads = ListingDedup.successful_lead_data([r])
    assert leads == []


def test_empty_extracted_rows_falls_to_legacy_data():
    """When the multi-row branch wasn't taken (empty list), the
    legacy ``data`` field controls."""
    r = StepResult(
        step_index=0, intent="x", success=True,
        data="VIABLE | URL: https://example.com",
        extracted_rows=[],
    )
    leads = ListingDedup.successful_lead_data([r])
    assert leads == ["VIABLE | URL: https://example.com"]


def test_mixed_step_results_both_contribute():
    """A plan that has BOTH a multi-row step and a single-row step
    should report leads from both."""
    multi = StepResult(
        step_index=0, intent="multi", success=True,
        data="extract_rows:2/5",
        extracted_rows=[
            {"rank": "1", "title": "A"},
            {"rank": "2", "title": "B"},
        ],
    )
    single = StepResult(
        step_index=1, intent="single", success=True,
        data="VIABLE | Year: 2020 | Make: Sea Ray",
    )
    leads = ListingDedup.successful_lead_data([multi, single])
    assert len(leads) == 3
    assert any("Year: 2020" in lead for lead in leads)
    assert any("Rank: 1 | Title: A" in lead for lead in leads)


def test_non_dict_rows_filtered_out():
    """Defensive — never crash on bad shape."""
    r = StepResult(
        step_index=0, intent="x", success=True,
        data="extract_rows:1/5",
        extracted_rows=[
            "not a dict",  # type: ignore[list-item]
            {"rank": "1", "title": "Real"},
            None,  # type: ignore[list-item]
        ],
    )
    leads = ListingDedup.successful_lead_data([r])
    assert len(leads) == 1
    assert "Real" in leads[0]


def test_empty_row_filtered_out():
    r = StepResult(
        step_index=0, intent="x", success=True,
        data="extract_rows:0/5",
        extracted_rows=[{}],
    )
    leads = ListingDedup.successful_lead_data([r])
    assert leads == []


# ── End-to-end: the HN reproducer ────────────────────────────────


def test_hn_top_5_reproducer():
    """The exact failure shape from the user's HN probe — 5 rows
    captured under a strict schema, pre-fix returned 0 viable."""
    r = StepResult(
        step_index=1, intent="Extract HN front-page stories",
        success=True,
        data="extract_rows:5/5",
        extracted_rows=[
            {"rank": "1", "title": "Story A", "story_url": "https://a.com",
             "points": "100", "author": "alice"},
            {"rank": "2", "title": "Story B", "story_url": "https://b.com",
             "points": "80", "author": "bob"},
            {"rank": "3", "title": "Story C", "story_url": "https://c.com",
             "points": "60", "author": "carol"},
            {"rank": "4", "title": "Story D", "story_url": "https://d.com",
             "points": "40", "author": "david"},
            {"rank": "5", "title": "Story E", "story_url": "https://e.com",
             "points": "20", "author": "eve"},
        ],
    )
    leads = ListingDedup.successful_lead_data([r])
    assert len(leads) == 5
    # Each lead string round-trips through the dedup keyer cleanly.
    for lead in leads:
        assert lead.startswith("VIABLE | ")
        assert "Rank: " in lead
        assert "Title: " in lead
