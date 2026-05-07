#!/usr/bin/env python3
"""Convert labelled production traces → Holo3 SFT chat format (#155 step 3).

The third deliverable in the continual-fine-tuning pipeline:

    [export]   MicroPlanRunner emits trace JSON         (step 1, PR #194)
    [label]    mantis trace label adds positive/neg/neu (step 2, PR #195)
    [convert]  THIS SCRIPT — labelled JSON → SFT chat   (step 3)
    [train]    train_holo3_distill.py runs SFT          (existing)

Each labelled step → one chat sample, when:
  - ``label == "positive"`` (default) — the brain's action matched a
    desired outcome (gate-verify pass, success-with-observed-delta).
  - Optional: ``--include-negative`` keeps ``label == "negative"`` rows,
    output in DPO-friendly shape (chosen/rejected pair) where the
    rejected side is the model's actual action and the chosen side
    is left blank for human or follow-up labelling.

Sample shape:

    {
      "conversations": [
        {"from": "system", "value": HOLO3_SYSTEM},
        {"from": "human",  "value": "<image>\\nTask: ...\\nScreen size: ..."},
        {"from": "gpt",    "value": "<brief reasoning>\\n<action_text>"}
      ],
      "image": "<absolute path to step screenshot>",
      "metadata": {
        "source": "labelled_trace",
        "run_id": "...",
        "tenant_id": "...",
        "step": 0,
        "label": "positive",
        "label_reason": "gate_verify_pass"
      }
    }

The ``HOLO3_SYSTEM``, ``claude_action_to_holo3``, and ``thinking_to_brief``
helpers from ``convert_claude_trajectories.py`` are reused so every SFT
sample — whether sourced from rollouts, Claude trajectories, or
production traces — uses the same wire format.

Usage:

    # Labels from `mantis trace label` live alongside per-run screenshot dirs.
    python training/convert_labelled_traces.py \\
        --traces /data/labelled \\
        --screenshots-root /data/traces \\
        --output training/data/labelled_distill.jsonl

    # Then append into the standing SFT dataset:
    cat training/data/labelled_distill.jsonl \\
        >> training/data/holo3_distill_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys as _sys
from pathlib import Path
from pathlib import Path as _Path
from typing import Any, Iterable

# Sibling-script import: training/ holds the shared helpers; add it to
# sys.path the same way ``convert_rollouts.py`` does.
_sys.path.insert(0, str(_Path(__file__).resolve().parent))

from convert_claude_trajectories import (  # type: ignore[import-not-found]  # noqa: E402
    HOLO3_SYSTEM,
    claude_action_to_holo3,
    thinking_to_brief,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Sample assembly ────────────────────────────────────────────────────


def _resolve_screenshot(
    step_index: int,
    run_id: str,
    tenant_id: str,
    screenshots_root: Path | None,
) -> Path | None:
    """Locate the per-step PNG written by the trace exporter.

    Trace exporter layout:
      ``<root>/<tenant_id>/<run_id>_screens/<step:04d>.png``
      ``<root>/__shared__/<run_id>_screens/<step:04d>.png`` (legacy single-tenant)

    Returns the resolved path if the file exists, else ``None`` so the
    caller can skip the sample.
    """
    if screenshots_root is None:
        return None
    tenant = (tenant_id or "").strip() or "__shared__"
    candidate = (
        screenshots_root / tenant / f"{run_id}_screens" / f"{step_index:04d}.png"
    )
    return candidate if candidate.exists() else None


def labelled_step_to_sample(
    step: dict[str, Any],
    *,
    run_id: str,
    tenant_id: str,
    intent: str,
    screenshots_root: Path | None,
    screen_size: tuple[int, int] = (1280, 720),
    require_screenshot: bool = True,
) -> dict[str, Any] | None:
    """Render one labelled step as an SFT chat sample.

    Returns ``None`` when the sample isn't usable for training:
    - action couldn't be serialized (unknown type, missing params)
    - screenshot is required but missing on disk

    The ``thinking_to_brief`` pass keeps the assistant turn short — the
    brain's full chain-of-thought stays in the source trace, only the
    distilled action context lands in the sample.
    """
    last_action = step.get("last_action") or {}
    if not isinstance(last_action, dict):
        return None
    action_type = last_action.get("action_type", "")
    params = last_action.get("params") or {}
    action_text = claude_action_to_holo3(action_type, params)
    if action_text is None:
        return None

    screenshot_path: Path | None = None
    if screenshots_root is not None:
        screenshot_path = _resolve_screenshot(
            step.get("step_index", 0), run_id, tenant_id, screenshots_root,
        )
    if require_screenshot and screenshot_path is None:
        return None

    # Compose the assistant turn from predicted_outcome + action_text. The
    # predicted_outcome is the brain's own one-sentence rationale; on
    # successful steps it's the cleanest short reasoning we have.
    predicted = step.get("predicted_outcome", "") or ""
    brief = thinking_to_brief(predicted)
    assistant = f"{brief}\n{action_text}" if brief else action_text

    sample_intent = intent or step.get("intent", "")
    if len(sample_intent) > 500:
        sample_intent = sample_intent[:500] + "..."

    sample: dict[str, Any] = {
        "conversations": [
            {"from": "system", "value": HOLO3_SYSTEM},
            {
                "from": "human",
                "value": (
                    f"<image>\nTask: {sample_intent}"
                    f"\nScreen size: {screen_size[0]}x{screen_size[1]} pixels"
                ),
            },
            {"from": "gpt", "value": assistant},
        ],
        "metadata": {
            "source": "labelled_trace",
            "run_id": run_id,
            "tenant_id": tenant_id,
            "step": step.get("step_index", 0),
            "label": step.get("label", "neutral"),
            "label_reason": step.get("label_reason", ""),
            "action_type": action_type,
        },
    }
    if screenshot_path is not None:
        sample["image"] = str(screenshot_path)
    return sample


def trace_to_samples(
    trace: dict[str, Any],
    *,
    keep_labels: tuple[str, ...] = ("positive",),
    screenshots_root: Path | None,
    screen_size: tuple[int, int] = (1280, 720),
    require_screenshot: bool = True,
) -> list[dict[str, Any]]:
    """Convert one labelled trace to a list of SFT samples.

    ``keep_labels`` controls which buckets graduate into the training
    set. The default ``("positive",)`` is the conservative SFT recipe;
    pass ``("positive", "negative")`` for DPO-style chosen/rejected
    construction (callers will need to pair the rejected with their
    own chosen side).
    """
    run_id = str(trace.get("run_id", ""))
    tenant_id = str(trace.get("tenant_id", ""))
    samples: list[dict[str, Any]] = []
    for step in trace.get("steps", []):
        if step.get("label") not in keep_labels:
            continue
        sample = labelled_step_to_sample(
            step,
            run_id=run_id,
            tenant_id=tenant_id,
            intent=step.get("intent", ""),
            screenshots_root=screenshots_root,
            screen_size=screen_size,
            require_screenshot=require_screenshot,
        )
        if sample is not None:
            samples.append(sample)
    return samples


# ── Batch convert ──────────────────────────────────────────────────────


def _iter_traces(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
        return
    for path in sorted(input_path.glob("**/*.json")):
        if path.is_file():
            yield path


def convert(
    input_path: Path,
    output_jsonl: Path,
    *,
    screenshots_root: Path | None = None,
    keep_labels: tuple[str, ...] = ("positive",),
    require_screenshot: bool = True,
    screen_size: tuple[int, int] = (1280, 720),
) -> tuple[int, int]:
    """Convert one labelled-trace file (or directory) into SFT JSONL.

    Returns ``(traces_kept, samples_written)``.
    """
    traces_kept = 0
    samples_written = 0
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w") as dst:
        for trace_path in _iter_traces(input_path):
            try:
                trace = json.loads(trace_path.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("skipping %s: %s", trace_path, exc)
                continue
            traces_kept += 1
            for sample in trace_to_samples(
                trace,
                keep_labels=keep_labels,
                screenshots_root=screenshots_root,
                screen_size=screen_size,
                require_screenshot=require_screenshot,
            ):
                dst.write(json.dumps(sample) + "\n")
                samples_written += 1
    return traces_kept, samples_written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--traces", required=True,
        help="Labelled trace JSON file or directory (output of mantis trace label)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output Holo3 chat-format JSONL",
    )
    parser.add_argument(
        "--screenshots-root",
        default="",
        help="Root dir holding <tenant>/<run_id>_screens/<NNNN>.png "
        "(typically MANTIS_TRACE_EXPORT_DIR with --include-screenshots).",
    )
    parser.add_argument(
        "--keep-labels", default="positive",
        help="Comma-separated labels to keep (default: positive). Use "
        "'positive,negative' for DPO-style preference data.",
    )
    parser.add_argument(
        "--require-screenshot", action="store_true", default=True,
        help="Skip steps whose screenshot is missing (default).",
    )
    parser.add_argument(
        "--no-require-screenshot", dest="require_screenshot",
        action="store_false",
        help="Emit samples without an 'image' field when the PNG is "
        "missing — useful for text-only data exploration.",
    )
    parser.add_argument(
        "--screen-width", type=int, default=1280,
    )
    parser.add_argument(
        "--screen-height", type=int, default=720,
    )
    args = parser.parse_args(argv)

    screenshots_root = (
        Path(args.screenshots_root).expanduser() if args.screenshots_root else None
    )
    keep_labels = tuple(s.strip() for s in args.keep_labels.split(",") if s.strip())
    if not keep_labels:
        print("error: --keep-labels must list at least one label", file=_sys.stderr)
        return 1

    traces_kept, samples_written = convert(
        Path(args.traces).expanduser(),
        Path(args.output).expanduser(),
        screenshots_root=screenshots_root,
        keep_labels=keep_labels,
        require_screenshot=args.require_screenshot,
        screen_size=(args.screen_width, args.screen_height),
    )

    logger.info(
        "traces_kept=%d samples_written=%d → %s",
        traces_kept, samples_written, args.output,
    )
    return 0


if __name__ == "__main__":
    _sys.exit(main())
