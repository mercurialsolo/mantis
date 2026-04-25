from mantis_agent.verification.dynamic_plan_verifier import DynamicPlanVerifier


def test_dynamic_verifier_marks_running_coverage_pending():
    verifier = DynamicPlanVerifier(
        plan_name="generic-results-plan",
        required_filter_tokens=("by-owner", "price-35000"),
    )

    verifier.record_page_start(page=1, url="https://example.com/results/by-owner/price-35000/")
    verifier.record_filter_check(
        page=1,
        url="https://example.com/results/by-owner/price-35000/",
        passed=True,
        reason="canonical_url",
    )
    verifier.record_viewport_scan(
        page=1,
        viewport_stage=0,
        cards=[(100, 200, "Listing A"), (100, 400, "Listing B")],
        new_cards=[(100, 200, "Listing A"), (100, 400, "Listing B")],
    )
    verifier.record_item_attempt(page=1, item="Listing A", viewport_stage=0)
    verifier.record_item_opened(page=1, item="Listing A", url="https://example.com/item/a")
    verifier.record_item_completed(page=1, item="Listing A", url="https://example.com/item/a")

    report = verifier.report(status="running")

    assert report["verdict"] == "running"
    assert report["totals"]["found_items"] == 2
    assert report["totals"]["attempted_items"] == 1
    assert report["totals"]["missing_attempts"] == 1
    assert any(
        check["name"] == "page_1_found_items_attempted" and check["status"] == "pending"
        for check in report["checks"]
    )


def test_dynamic_verifier_passes_after_page_exhaustion_and_pagination_stop():
    verifier = DynamicPlanVerifier(required_filter_tokens=("by-owner",))

    verifier.record_page_start(page=1, url="https://example.com/results/by-owner/")
    verifier.record_filter_check(page=1, url="https://example.com/results/by-owner/", passed=True)
    verifier.record_viewport_scan(
        page=1,
        viewport_stage=0,
        cards=[{"title": "Listing A"}, {"title": "Listing B"}],
    )
    for title in ("Listing A", "Listing B"):
        verifier.record_item_attempt(page=1, item=title)
        verifier.record_item_opened(page=1, item=title, url=f"https://example.com/{title[-1].lower()}")
        verifier.record_item_completed(page=1, item=title, success=True)
    verifier.record_page_exhausted(page=1, reason="no_new_visible_items")
    verifier.record_pagination(page=1, success=False, method="all_layers", reason="no_next_page")

    report = verifier.report(status="completed")

    assert report["verdict"] == "pass"
    assert report["totals"]["found_items"] == 2
    assert report["totals"]["completed_items"] == 2
    assert report["totals"]["failed_items"] == 0
    assert all(check["status"] == "pass" for check in report["checks"])


def test_dynamic_verifier_fails_completed_run_with_unattempted_item():
    verifier = DynamicPlanVerifier()
    verifier.record_page_start(page=1, url="https://example.com/results")
    verifier.record_viewport_scan(page=1, viewport_stage=0, cards=[(1, 2, "Listing A")])
    verifier.record_page_exhausted(page=1)

    report = verifier.report(status="completed")

    assert report["verdict"] == "fail"
    assert any(
        check["name"] == "page_1_found_items_attempted" and check["status"] == "fail"
        for check in report["checks"]
    )


def test_dynamic_verifier_restores_from_report():
    verifier = DynamicPlanVerifier(required_filter_tokens=("initial",))
    verifier.record_page_start(page=2, url="https://example.com/page-2")
    verifier.record_filter_check(page=2, url="https://example.com/page-2", passed=True)
    verifier.record_viewport_scan(page=2, viewport_stage=1, cards=[(1, 2, "Listing C")])
    verifier.record_item_attempt(page=2, item="Listing C")
    verifier.record_item_completed(page=2, item="Listing C", success=False, reason="click_failed")
    report = verifier.report(status="halted")

    restored = DynamicPlanVerifier()
    restored.load_report(report)
    restored_report = restored.report(status="halted")

    assert restored_report["required_filter_tokens"] == ["initial"]
    assert restored_report["pages"][0]["page"] == 2
    assert restored_report["totals"]["failed_items"] == 1
    assert any(
        check["name"] == "page_2_completed_without_item_failures" and check["status"] == "fail"
        for check in restored_report["checks"]
    )
