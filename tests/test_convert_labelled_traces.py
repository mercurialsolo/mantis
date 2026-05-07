"""Tests for #155 step 3 — labelled-trace → SFT chat converter.

Pins the schema match against ``train_holo3_distill.py``'s expected
input shape, the label-filter contract, and the screenshot resolution
logic. Side-by-side compatibility with ``convert_rollouts.py`` is
verified by structurally checking the output keys.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


# Sibling-script: training/ isn't on sys.path by default.
_TRAINING = Path(__file__).resolve().parent.parent / "training"
sys.path.insert(0, str(_TRAINING))
from convert_labelled_traces import (  # noqa: E402
    convert,
    labelled_step_to_sample,
    trace_to_samples,
)


def _step(
    *,
    step_index: int = 0,
    intent: str = "click first listing",
    last_action: dict[str, Any] | None = None,
    label: str = "positive",
    label_reason: str = "success_with_observed_delta",
    predicted_outcome: str = "page navigates",
    observed_outcome: str = "page navigated",
    success: bool = True,
    data: str = "",
) -> dict[str, Any]:
    return {
        "step_index": step_index,
        "intent": intent,
        "type": "click",
        "success": success,
        "data": data,
        "last_action": last_action
        or {"action_type": "click", "params": {"x": 100, "y": 200}},
        "predicted_outcome": predicted_outcome,
        "observed_outcome": observed_outcome,
        "label": label,
        "label_reason": label_reason,
    }


def _trace(steps: list[dict[str, Any]], **overrides) -> dict[str, Any]:
    base = {
        "run_id": "rid_abc",
        "tenant_id": "t-one",
        "status": "completed",
        "label_summary": {"positive": 1, "negative": 0, "neutral": 0},
        "steps": steps,
    }
    base.update(overrides)
    return base


# ── labelled_step_to_sample ────────────────────────────────────────────


def test_step_to_sample_emits_holo3_chat_shape():
    sample = labelled_step_to_sample(
        _step(),
        run_id="r1",
        tenant_id="t1",
        intent="Click first listing",
        screenshots_root=None,
        require_screenshot=False,
    )
    assert sample is not None
    # Three-turn chat: system, human, gpt
    assert len(sample["conversations"]) == 3
    assert sample["conversations"][0]["from"] == "system"
    assert sample["conversations"][1]["from"] == "human"
    assert sample["conversations"][2]["from"] == "gpt"
    assert "<image>" in sample["conversations"][1]["value"]
    assert sample["metadata"]["source"] == "labelled_trace"
    assert sample["metadata"]["label"] == "positive"


def test_step_to_sample_skips_unknown_action_type():
    """Action types the Holo3 distill format doesn't recognise are dropped."""
    step = _step(last_action={"action_type": "totally_unknown", "params": {}})
    sample = labelled_step_to_sample(
        step, run_id="r", tenant_id="t", intent="x",
        screenshots_root=None, require_screenshot=False,
    )
    assert sample is None


def test_step_to_sample_skips_when_screenshot_required_but_missing(tmp_path):
    sample = labelled_step_to_sample(
        _step(),
        run_id="r1", tenant_id="t1", intent="x",
        screenshots_root=tmp_path,  # no PNGs under here
        require_screenshot=True,
    )
    assert sample is None


def test_step_to_sample_emits_image_path_when_screenshot_present(tmp_path):
    # Layout matches what TraceExporter writes when MANTIS_TRACE_INCLUDE_SCREENSHOTS=1
    shots = tmp_path / "t-one" / "rid_abc_screens"
    shots.mkdir(parents=True)
    (shots / "0000.png").write_bytes(b"\x89PNG")
    sample = labelled_step_to_sample(
        _step(step_index=0),
        run_id="rid_abc", tenant_id="t-one", intent="x",
        screenshots_root=tmp_path,
        require_screenshot=True,
    )
    assert sample is not None
    assert sample["image"].endswith("0000.png")


def test_step_to_sample_uses_shared_when_tenant_blank(tmp_path):
    shots = tmp_path / "__shared__" / "rid_abc_screens"
    shots.mkdir(parents=True)
    (shots / "0000.png").write_bytes(b"\x89PNG")
    sample = labelled_step_to_sample(
        _step(step_index=0),
        run_id="rid_abc", tenant_id="", intent="x",
        screenshots_root=tmp_path,
        require_screenshot=True,
    )
    assert sample is not None
    assert "__shared__" in sample["image"]


def test_step_to_sample_truncates_long_intent():
    long_intent = "x" * 800
    sample = labelled_step_to_sample(
        _step(),
        run_id="r1", tenant_id="t1", intent=long_intent,
        screenshots_root=None, require_screenshot=False,
    )
    human_turn = sample["conversations"][1]["value"]
    # Truncation tag is appended.
    assert "..." in human_turn


# ── trace_to_samples (label filter) ────────────────────────────────────


def test_trace_to_samples_default_keeps_positive_only():
    trace = _trace([
        _step(step_index=0, label="positive"),
        _step(step_index=1, label="negative"),
        _step(step_index=2, label="neutral"),
    ])
    samples = trace_to_samples(
        trace, keep_labels=("positive",),
        screenshots_root=None, require_screenshot=False,
    )
    assert len(samples) == 1
    assert samples[0]["metadata"]["step"] == 0


def test_trace_to_samples_dpo_keeps_positive_and_negative():
    trace = _trace([
        _step(step_index=0, label="positive"),
        _step(step_index=1, label="negative"),
        _step(step_index=2, label="neutral"),
    ])
    samples = trace_to_samples(
        trace, keep_labels=("positive", "negative"),
        screenshots_root=None, require_screenshot=False,
    )
    assert len(samples) == 2
    labels = {s["metadata"]["label"] for s in samples}
    assert labels == {"positive", "negative"}


# ── convert (file + directory) ─────────────────────────────────────────


def test_convert_single_file(tmp_path):
    in_file = tmp_path / "labelled.json"
    in_file.write_text(json.dumps(_trace([_step(step_index=0)])))
    out = tmp_path / "out.jsonl"
    traces, samples = convert(
        in_file, out, screenshots_root=None,
        require_screenshot=False,
    )
    assert traces == 1
    assert samples == 1
    written = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(written) == 1


def test_convert_directory_walks_recursively(tmp_path):
    inp = tmp_path / "in"
    (inp / "acme").mkdir(parents=True)
    (inp / "globex").mkdir(parents=True)
    (inp / "acme" / "r1.json").write_text(json.dumps(_trace([_step(step_index=0)])))
    (inp / "globex" / "r2.json").write_text(json.dumps(_trace([
        _step(step_index=0),
        _step(step_index=1, label="negative"),
    ])))
    out = tmp_path / "out.jsonl"
    traces, samples = convert(
        inp, out, screenshots_root=None,
        require_screenshot=False,
    )
    assert traces == 2
    # Default keep-labels=positive → drops the negative row in r2.
    assert samples == 2


def test_convert_skips_broken_json(tmp_path):
    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "good.json").write_text(json.dumps(_trace([_step()])))
    (inp / "broken.json").write_text("{not json")
    out = tmp_path / "out.jsonl"
    traces, samples = convert(
        inp, out, screenshots_root=None,
        require_screenshot=False,
    )
    assert traces == 1  # only the good one counted
    assert samples == 1


def test_convert_writes_empty_output_when_no_matching_labels(tmp_path):
    """A trace where every step is ``neutral`` produces no SFT samples
    under the default ``--keep-labels=positive`` filter, but we still
    create the output file so a downstream ``cat … >> distill.jsonl``
    doesn't error."""
    in_file = tmp_path / "labelled.json"
    in_file.write_text(json.dumps(_trace([
        _step(step_index=0, label="neutral"),
        _step(step_index=1, label="neutral"),
    ])))
    out = tmp_path / "out.jsonl"
    traces, samples = convert(
        in_file, out, screenshots_root=None,
        require_screenshot=False,
    )
    assert traces == 1
    assert samples == 0
    assert out.exists()
    assert out.read_text() == ""
