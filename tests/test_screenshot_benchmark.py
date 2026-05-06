"""Tests for the offline screenshot CUA benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from training.screenshot_benchmark import (
    Prediction,
    Region,
    ScreenshotCase,
    load_cases,
    load_predictions,
    run_with_predictions,
    score_case,
    summarize,
)


def test_score_case_accepts_expected_click_in_allowed_region() -> None:
    case = ScreenshotCase(
        id="title-click",
        image="unused.png",
        task="Click the title text",
        expected_action_type="click",
        allowed_regions=(Region("title", 100, 100, 300, 160),),
        forbidden_regions=(Region("photo", 0, 0, 90, 180),),
        tags=("title_click",),
    )
    pred = Prediction(id="title-click", action_type="click", params={"x": 180, "y": 130})

    result = score_case(case, pred)

    assert result.passed
    assert result.checks == {
        "prediction_present": True,
        "action_type": True,
        "allowed_region": True,
        "forbidden_region": True,
    }


def test_score_case_rejects_forbidden_photo_click() -> None:
    case = ScreenshotCase(
        id="photo-click",
        image="unused.png",
        task="Click title text, not photo",
        expected_action_type="click",
        allowed_regions=(Region("title", 100, 100, 300, 160),),
        forbidden_regions=(Region("photo", 0, 0, 90, 180),),
        tags=("forbidden_region",),
    )
    pred = Prediction(id="photo-click", action_type="click", params={"x": 50, "y": 90})

    result = score_case(case, pred)

    assert not result.passed
    assert result.checks["allowed_region"] is False
    assert result.checks["forbidden_region"] is False
    assert any("forbidden" in failure for failure in result.failures)


def test_score_case_checks_done_summary_regex() -> None:
    case = ScreenshotCase(
        id="done-format",
        image="unused.png",
        task="Return extracted row",
        expected_action_type="done",
        expected_text_regex=r"VIABLE\s+\|\s+Year:\s+2026\s+\|\s+Make:\s+Tracker",
        tags=("done_format",),
    )
    pred = Prediction(
        id="done-format",
        action_type="done",
        params={"success": True, "summary": "VIABLE | Year: 2026 | Make: Tracker"},
    )

    assert score_case(case, pred).passed


def test_load_cases_and_predictions_jsonl(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.jsonl"
    preds_path = tmp_path / "predictions.jsonl"
    cases_path.write_text(
        json.dumps({
            "id": "c1",
            "image": "x.png",
            "task": "Click",
            "expected_action_type": "click",
            "allowed_regions": [{"label": "button", "x1": 1, "y1": 2, "x2": 5, "y2": 8}],
            "tags": ["smoke"],
        }) + "\n"
    )
    preds_path.write_text(
        json.dumps({"id": "c1", "action_type": "click", "params": {"x": 3, "y": 4}}) + "\n"
    )

    cases = load_cases(cases_path)
    preds = load_predictions(preds_path)
    results, summary = run_with_predictions(cases, preds)

    assert len(cases) == 1
    assert cases[0].allowed_regions[0].label == "button"
    assert results[0].passed
    assert summary["passed"] == 1
    assert summary["by_tag"]["smoke"]["pass_rate"] == 1.0


def test_summarize_reports_failures_by_tag() -> None:
    cases = [
        ScreenshotCase(
            id="ok",
            image="unused.png",
            task="Click",
            expected_action_type="click",
            allowed_regions=(Region("ok", 0, 0, 10, 10),),
            tags=("click",),
        ),
        ScreenshotCase(
            id="bad",
            image="unused.png",
            task="Click",
            expected_action_type="click",
            allowed_regions=(Region("ok", 0, 0, 10, 10),),
            tags=("click",),
        ),
    ]
    preds = {
        "ok": Prediction(id="ok", action_type="click", params={"x": 5, "y": 5}),
        "bad": Prediction(id="bad", action_type="click", params={"x": 50, "y": 50}),
    }

    results, summary = run_with_predictions(cases, preds)

    assert [r.passed for r in results] == [True, False]
    assert summary == summarize(results)
    assert summary["pass_rate"] == 0.5
    assert summary["by_tag"]["click"]["passed"] == 1
