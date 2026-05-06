#!/usr/bin/env python3
"""Offline screenshot benchmark for CUA action regressions.

The benchmark runs one screenshot + task prompt per case and scores the
resulting action against simple, verifiable expectations:

* action type
* click point inside allowed target regions
* click point outside forbidden regions
* optional text regex over model output / done summary

Use ``--predictions`` for CI or parser-only checks without a model endpoint.
Use ``--brain holo3`` to call a live Holo3-compatible endpoint.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from mantis_agent.actions import Action, ActionType


@dataclass(frozen=True)
class Region:
    """A rectangular scoring region in screen coordinates."""

    label: str
    x1: int
    y1: int
    x2: int
    y2: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Region":
        return cls(
            label=str(data.get("label", "")),
            x1=int(data["x1"]),
            y1=int(data["y1"]),
            x2=int(data["x2"]),
            y2=int(data["y2"]),
        )

    def contains(self, x: int, y: int) -> bool:
        left, right = sorted((self.x1, self.x2))
        top, bottom = sorted((self.y1, self.y2))
        return left <= x <= right and top <= y <= bottom


@dataclass(frozen=True)
class ScreenshotCase:
    """One offline benchmark case."""

    id: str
    image: str
    task: str
    expected_action_type: str = ""
    allowed_regions: tuple[Region, ...] = ()
    forbidden_regions: tuple[Region, ...] = ()
    expected_text_regex: str = ""
    tags: tuple[str, ...] = ()
    screen_size: tuple[int, int] = (1280, 720)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScreenshotCase":
        screen = data.get("screen_size") or [1280, 720]
        return cls(
            id=str(data["id"]),
            image=str(data["image"]),
            task=str(data["task"]),
            expected_action_type=str(data.get("expected_action_type", "")),
            allowed_regions=tuple(
                Region.from_dict(r) for r in data.get("allowed_regions", [])
            ),
            forbidden_regions=tuple(
                Region.from_dict(r) for r in data.get("forbidden_regions", [])
            ),
            expected_text_regex=str(data.get("expected_text_regex", "")),
            tags=tuple(str(t) for t in data.get("tags", [])),
            screen_size=(int(screen[0]), int(screen[1])),
        )


@dataclass(frozen=True)
class Prediction:
    """One model prediction for a benchmark case."""

    id: str
    action_type: str
    params: dict[str, Any] = field(default_factory=dict)
    text: str = ""
    thinking: str = ""

    @classmethod
    def from_action(
        cls,
        case_id: str,
        action: Action,
        *,
        text: str = "",
        thinking: str = "",
    ) -> "Prediction":
        return cls(
            id=case_id,
            action_type=action.action_type.value,
            params=dict(action.params),
            text=text,
            thinking=thinking,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Prediction":
        # Accept both our JSONL format and serialized Action-like dicts.
        action_type = data.get("action_type") or data.get("type")
        params = data.get("params") or data.get("action_params") or {}
        return cls(
            id=str(data["id"]),
            action_type=str(action_type or ""),
            params=dict(params),
            text=str(data.get("text", data.get("raw_output", ""))),
            thinking=str(data.get("thinking", "")),
        )

    def action_point(self) -> tuple[int, int] | None:
        """Return the screen point to score for click-like actions."""
        if self.action_type not in {ActionType.CLICK.value, ActionType.DOUBLE_CLICK.value}:
            return None
        try:
            return int(self.params["x"]), int(self.params["y"])
        except Exception:
            return None


@dataclass(frozen=True)
class CaseResult:
    """Score for one benchmark case."""

    id: str
    passed: bool
    checks: dict[str, bool]
    failures: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


def load_cases(path: Path) -> list[ScreenshotCase]:
    """Load newline-delimited benchmark cases."""
    cases: list[ScreenshotCase] = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                cases.append(ScreenshotCase.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: invalid case: {exc}") from exc
    return cases


def load_predictions(path: Path) -> dict[str, Prediction]:
    """Load newline-delimited predictions keyed by case id."""
    predictions: dict[str, Prediction] = {}
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                pred = Prediction.from_dict(json.loads(line))
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: invalid prediction: {exc}") from exc
            predictions[pred.id] = pred
    return predictions


def score_case(case: ScreenshotCase, prediction: Prediction | None) -> CaseResult:
    """Score a single prediction against one case."""
    checks: dict[str, bool] = {}
    failures: list[str] = []

    if prediction is None:
        return CaseResult(
            id=case.id,
            passed=False,
            checks={"prediction_present": False},
            failures=("missing prediction",),
            tags=case.tags,
        )
    checks["prediction_present"] = True

    if case.expected_action_type:
        ok = prediction.action_type == case.expected_action_type
        checks["action_type"] = ok
        if not ok:
            failures.append(
                f"action_type expected {case.expected_action_type!r}, got {prediction.action_type!r}"
            )

    point = prediction.action_point()
    if case.allowed_regions:
        ok = point is not None and any(r.contains(*point) for r in case.allowed_regions)
        checks["allowed_region"] = ok
        if not ok:
            failures.append(
                "point missing or outside allowed regions"
                if point is None else f"point {point} outside allowed regions"
            )

    if case.forbidden_regions:
        ok = point is None or not any(r.contains(*point) for r in case.forbidden_regions)
        checks["forbidden_region"] = ok
        if not ok:
            failures.append(f"point {point} inside forbidden region")

    if case.expected_text_regex:
        text = _prediction_text(prediction)
        ok = re.search(case.expected_text_regex, text, re.IGNORECASE | re.DOTALL) is not None
        checks["text_regex"] = ok
        if not ok:
            failures.append("expected text regex did not match")

    passed = all(checks.values()) if checks else True
    return CaseResult(
        id=case.id,
        passed=passed,
        checks=checks,
        failures=tuple(failures),
        tags=case.tags,
    )


def summarize(results: Iterable[CaseResult]) -> dict[str, Any]:
    """Aggregate pass rates overall and by tag."""
    rows = list(results)
    total = len(rows)
    passed = sum(1 for r in rows if r.passed)
    by_tag: dict[str, dict[str, int | float]] = {}
    for row in rows:
        for tag in row.tags or ("untagged",):
            bucket = by_tag.setdefault(tag, {"total": 0, "passed": 0, "pass_rate": 0.0})
            bucket["total"] = int(bucket["total"]) + 1
            bucket["passed"] = int(bucket["passed"]) + (1 if row.passed else 0)
    for bucket in by_tag.values():
        bucket["pass_rate"] = (
            float(bucket["passed"]) / float(bucket["total"]) if bucket["total"] else 0.0
        )
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": (passed / total) if total else 0.0,
        "by_tag": by_tag,
    }


def run_with_predictions(
    cases: list[ScreenshotCase],
    predictions: dict[str, Prediction],
) -> tuple[list[CaseResult], dict[str, Any]]:
    results = [score_case(case, predictions.get(case.id)) for case in cases]
    return results, summarize(results)


def run_with_holo3(
    cases: list[ScreenshotCase],
    *,
    base_url: str,
    model: str,
    api_key: str,
    max_cases: int = 0,
) -> tuple[list[CaseResult], dict[str, Any]]:
    """Run cases against a live Holo3Brain endpoint."""
    from mantis_agent.brain_holo3 import Holo3Brain

    brain = Holo3Brain(base_url=base_url, model=model, api_key=api_key)
    selected = cases[:max_cases] if max_cases > 0 else cases
    predictions: dict[str, Prediction] = {}
    for case in selected:
        image = Image.open(case.image)
        result = brain.think(
            frames=[image],
            task=case.task,
            action_history=None,
            screen_size=case.screen_size,
        )
        predictions[case.id] = Prediction.from_action(
            case.id,
            result.action,
            text=getattr(result, "raw_output", ""),
            thinking=getattr(result, "thinking", ""),
        )
    return run_with_predictions(selected, predictions)


def write_results(path: Path, results: list[CaseResult], summary: dict[str, Any]) -> None:
    """Write per-case JSONL plus a sibling summary JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for result in results:
            f.write(json.dumps(asdict(result), sort_keys=True) + "\n")
    summary_path = path.with_suffix(path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def _prediction_text(prediction: Prediction) -> str:
    done_summary = ""
    if prediction.action_type == ActionType.DONE.value:
        done_summary = str(prediction.params.get("summary", ""))
    return "\n".join(
        part for part in [prediction.text, prediction.thinking, done_summary] if part
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, help="Benchmark cases JSONL")
    parser.add_argument("--output", required=True, help="Per-case result JSONL")
    parser.add_argument("--predictions", help="Prediction JSONL keyed by case id")
    parser.add_argument("--brain", choices=["holo3"], help="Run a live brain instead of predictions")
    parser.add_argument("--base-url", default="https://api.hcompany.ai/v1")
    parser.add_argument("--model", default="holo3-35b-a3b")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--max-cases", type=int, default=0)
    args = parser.parse_args()

    cases = load_cases(Path(args.cases))
    if args.predictions:
        results, summary = run_with_predictions(cases, load_predictions(Path(args.predictions)))
    elif args.brain == "holo3":
        results, summary = run_with_holo3(
            cases,
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            max_cases=args.max_cases,
        )
    else:
        parser.error("provide either --predictions or --brain holo3")

    write_results(Path(args.output), results, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    sys.exit(0 if summary["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
