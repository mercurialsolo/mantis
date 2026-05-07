#!/usr/bin/env python3
"""Benchmark how far a brain (Holo3) gets on a raw English plan with no decomposition.

Submits a single-task ``task_suite`` whose ``intent`` is the verbatim contents of a
text-plan file. Skips ``PlanDecomposer`` entirely (see runtime.py:_run_tasks).
``verify`` is set to never match so the run terminates only on:
  - the model emitting ``done``
  - an unhandled exception in the task loop
  - hitting ``max_steps`` / ``max_cost`` / ``max_time_minutes``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Reuse helpers from the existing client.
sys.path.insert(0, str(Path(__file__).parent))
from baseten_workload import (  # noqa: E402
    DEFAULT_ENVIRONMENT,
    DEFAULT_MODEL_ID,
    load_dotenv,
    post_json,
    print_json,
)


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    intent = Path(args.plan_file).read_text()
    # Substitute {key} placeholders from --sub key=value pairs. Lets us pass
    # secrets/inputs without baking them into the on-disk plan file.
    for sub in args.sub or []:
        if "=" not in sub:
            raise SystemExit(f"--sub requires key=value, got: {sub!r}")
        k, v = sub.split("=", 1)
        intent = intent.replace("{" + k.strip() + "}", v)
    payload: dict[str, Any] = {
        "detached": not args.attach,
        "state_key": args.state_key,
        "resume_state": False,
        "max_cost": args.max_cost,
        "max_time_minutes": args.max_time_minutes,
        "max_steps": args.max_steps,
        "decompose": args.decompose,
        "proxy_disabled": args.proxy_disabled,
        "record_video": args.record_video,
    }
    if args.shape == "task_suite":
        payload["task_suite"] = {
            "session_name": args.session_name,
            # Workaround: _run_tasks doesn't copy payload.proxy_disabled into the
            # task_suite, so the flag has to be embedded here directly to take effect.
            "_proxy_disabled": args.proxy_disabled,
            "tasks": [
                {
                    "task_id": "long_run",
                    "intent": intent,
                    "start_url": args.start_url,
                    "verify": {"type": "url_contains", "value": args.never_match_token},
                }
            ],
        }
    else:  # plan_text
        payload["plan_text"] = intent
        payload["start_url"] = args.start_url
        payload["session_name"] = args.session_name
    if args.run_id:
        payload["run_id"] = args.run_id
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-file", required=True, help="path to the raw text plan")
    parser.add_argument("--start-url", default="about:blank")
    parser.add_argument("--session-name", default="text_plan_bench")
    parser.add_argument("--state-key", default="text-plan-bench-1")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-cost", type=float, default=50.0)
    parser.add_argument("--max-time-minutes", type=int, default=200)
    parser.add_argument("--never-match-token", default="__bench_never_match__")
    parser.add_argument(
        "--shape",
        choices=("task_suite", "plan_text"),
        default="task_suite",
        help="payload shape; plan_text needs server-side decompose flag support",
    )
    parser.add_argument(
        "--decompose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="when False (default), free-text input runs verbatim; --decompose flips to PlanDecomposer",
    )
    parser.add_argument(
        "--proxy-disabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="when True, the runtime skips the residential proxy and connects direct",
    )
    parser.add_argument(
        "--record-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="when True, the runtime records the screen via ffmpeg (saved server-side)",
    )
    parser.add_argument(
        "--sub",
        action="append",
        default=[],
        help="substitute {key} → value in the plan text (repeatable: --sub zip_code=33101 --sub pop_password=...). "
        "Lets you keep secrets out of the on-disk plan file.",
    )
    parser.add_argument("--attach", action="store_true", help="block until Baseten returns a result")
    parser.add_argument("--run-id", help="optional deterministic run id")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--endpoint", help="full Baseten predict endpoint")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--environment", default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true", help="print payload without sending")

    args = parser.parse_args()
    load_dotenv(Path(args.env_file))

    payload = build_payload(args)
    if args.dry_run:
        print_json(payload)
        return 0

    response = post_json(args, payload)
    print_json(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
