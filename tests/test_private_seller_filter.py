import json

from mantis_agent.extraction import ExtractionResult
from mantis_agent.gym.micro_runner import MicroPlanRunner, RunCheckpoint, StepResult


def test_private_seller_is_viable():
    result = ExtractionResult(
        year="2006",
        make="Luhrs",
        model="41 Convertible",
        price="$235,000",
        phone="+507 6615-9404",
        url="boattrader.com/boat/2006-luhrs-41-convertible-10131961/",
        seller="Mario Vega",
    )

    assert result.is_private_seller()
    assert result.has_phone()
    assert result.is_viable()


def test_private_seller_without_phone_is_viable_but_not_phone_lead():
    result = ExtractionResult(
        year="1987",
        make="Beneteau",
        model="Idylle 15.50",
        price="$130,000",
        phone="",
        url="boattrader.com/boat/1987-beneteau-idylle-15-50-10139990/",
        seller="Private Seller",
    )

    assert result.is_private_seller()
    assert not result.has_phone()
    assert result.missing_required_reason() == ""
    assert result.is_viable()


def test_dealer_inventory_is_not_viable_even_with_phone():
    result = ExtractionResult(
        year="2026",
        make="Azimut",
        model="S8",
        price="Request a Price",
        phone="954-800-6512",
        url="boattrader.com/boat/2026-azimut-s8-10041625/",
        seller="MarineMax East Florida Yacht Center",
    )

    assert not result.is_private_seller()
    assert not result.is_viable()


def test_dealer_url_is_not_viable():
    result = ExtractionResult(
        year="2026",
        make="Azimut",
        model="Verve 48",
        price="Request a Price",
        phone="",
        url="boattrader.com/boats/dealerName-MarineMax/make-azimut/condition-new/",
    )

    assert not result.is_private_seller()
    assert not result.is_viable()


def test_lead_counts_split_phone_leads_from_total_leads():
    rows = [
        StepResult(
            step_index=5,
            intent="extract",
            success=True,
            data=(
                "VIABLE | Year: 1997 | Make: AcmeBoats | Model: Sample 52 | "
                "Price: $254,000 | Phone: +1-415-555-0123 | "
                "URL: example.com/boat/sample-52-1001/"
            ),
        ),
        StepResult(
            step_index=5,
            intent="extract",
            success=True,
            data=(
                "VIABLE | Year: 1987 | Make: AcmeBoats | Model: Idylle | "
                "Price: $130,000 | Phone: none | "
                "URL: example.com/boat/idylle-1002/"
            ),
        ),
    ]

    assert MicroPlanRunner._lead_counts(rows) == (2, 1)


def test_filter_tokens_require_boattrader_private_seller_filters():
    from mantis_agent.site_config import SiteConfig

    runner = object.__new__(MicroPlanRunner)
    runner.site_config = SiteConfig.default_boattrader()
    runner._required_filter_tokens = MicroPlanRunner._derive_filter_tokens(
        "https://www.boattrader.com/boats/state-fl/city-miami/zip-33101/by-owner/price-35000/"
    )

    assert runner._url_has_required_filters(
        "https://www.boattrader.com/boats/state-fl/city-miami/zip-33101/by-owner/price-35000/page-2/"
    )
    assert not runner._url_has_required_filters(
        "https://www.boattrader.com/boats/state-fl/city-miami/zip-33101/price-35000/"
    )
    assert not runner._url_has_required_filters(
        "https://www.boattrader.com/boat/sample-listing-1001/"
    )


def test_checkpoint_roundtrip_preserves_resume_state(tmp_path):
    path = tmp_path / "state.json"
    lead = (
        "VIABLE | Year: 1997 | Make: AcmeBoats | Model: Sample 52 | "
        "Phone: +1-415-555-0123 | URL: example.com/boat/1001/"
    )
    checkpoint = RunCheckpoint(
        run_key="boattrader-miami",
        plan_signature="sig",
        status="halted",
        halt_reason="page_blocked",
        step_index=5,
        seen_urls=["https://example.com/boat/1"],
        step_results=[StepResult(5, "extract", True, data=lead).to_dict()],
        loop_counters={"7": 3},
        listings_on_page=2,
        page_listings=[[100, 200, "Listing title"]],
        page_listing_index=1,
        viewport_stage=2,
        current_page=3,
        results_base_url="https://www.boattrader.com/boats/by-owner/price-35000/",
        required_filter_tokens=["by-owner", "price-35000"],
        scroll_state={
            "context": "detail_extract",
            "url": "https://example.com/boat/1",
            "page_downs": 3,
            "wheel_downs": 0,
            "label": "detail viewport 4",
        },
        last_extracted={
            "last_completed_url": "https://example.com/boat/1",
            "last_completed_key": "https://example.com/boat/1",
        },
        costs={"gpu_steps": 4},
    )

    checkpoint.save(str(path))
    loaded = RunCheckpoint.load(str(path))
    assert loaded is not None

    runner = object.__new__(MicroPlanRunner)
    runner.costs = {
        "gpu_steps": 0,
        "gpu_seconds": 0.0,
        "claude_extract": 0,
        "claude_grounding": 0,
        "proxy_mb": 0.0,
    }
    results, loop_counters, listings_on_page = runner._restore_from_checkpoint(loaded)

    assert results[0].data == lead
    assert loop_counters == {7: 3}
    assert listings_on_page == 2
    assert runner._page_listings == [(100, 200, "Listing title")]
    assert runner._current_page == 3
    assert runner._scroll_state["page_downs"] == 3
    assert runner._scroll_state["label"] == "detail viewport 4"
    assert runner._last_extracted["last_completed_url"] == "https://example.com/boat/1"


def test_checkpoint_load_ignores_unknown_fields(tmp_path):
    path = tmp_path / "legacy_state.json"
    path.write_text(json.dumps({"step_index": 4, "unknown_future_field": "ok"}))

    loaded = RunCheckpoint.load(str(path))

    assert loaded is not None
    assert loaded.step_index == 4


def test_current_results_page_url_preserves_filtered_page_number():
    from mantis_agent.site_config import SiteConfig

    runner = object.__new__(MicroPlanRunner)
    runner.site_config = SiteConfig.default_boattrader()
    runner._results_base_url = (
        "https://www.boattrader.com/boats/state-fl/city-miami/zip-33101/by-owner/price-35000/"
    )
    runner._current_page = 3

    assert runner._current_results_page_url() == (
        "https://www.boattrader.com/boats/state-fl/city-miami/zip-33101/by-owner/price-35000/page-3/"
    )
