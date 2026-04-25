#!/usr/bin/env python3
"""Small Baseten workload client for detached Mantis runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_MODEL_ID = "qvvgkneq"
DEFAULT_ENVIRONMENT = "production"
DEFAULT_MICRO = "plans/boattrader/extract_url_filtered.json"


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def baseten_api_key() -> str:
    for name in ("BASETEN_API_KEY", "BASETEN_KEY", "HAI_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    raise SystemExit("Missing BASETEN_API_KEY, BASETEN_KEY, or HAI_API_KEY in environment/.env")


def endpoint(args: argparse.Namespace) -> str:
    if getattr(args, "endpoint", None):
        return args.endpoint
    env_endpoint = os.environ.get("BASETEN_HOLO3_ENDPOINT") or os.environ.get("BASETEN_ENDPOINT")
    if env_endpoint:
        return env_endpoint
    model_id = getattr(args, "model_id", None) or os.environ.get("BASETEN_MODEL_ID", DEFAULT_MODEL_ID)
    environment = (
        getattr(args, "environment", None)
        or os.environ.get("BASETEN_ENVIRONMENT", DEFAULT_ENVIRONMENT)
    )
    return f"https://model-{model_id}.api.baseten.co/{environment}/predict"


def post_json(args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        endpoint(args),
        data=body,
        headers={
            "Authorization": f"Api-Key {baseten_api_key()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise SystemExit(f"Baseten request failed ({exc.code}): {detail}") from exc


def print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def write_csv(path: Path, leads: list[Any]) -> None:
    rows: list[dict[str, str]] = []
    fieldnames = ["status", "year", "make", "model", "price", "phone", "seller", "url", "raw"]
    for lead in leads:
        if isinstance(lead, dict):
            row = {key: "" for key in fieldnames}
            for key, value in lead.items():
                text = "" if value is None else str(value)
                if key not in row:
                    fieldnames.append(key)
                row[key] = text
            row["raw"] = json.dumps(lead, sort_keys=True)
        else:
            row = {key: "" for key in fieldnames}
            row["raw"] = str(lead)
        rows.append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def command_run(args: argparse.Namespace) -> int:
    payload: dict[str, Any] = {
        "detached": not args.attach,
        "micro": args.micro,
        "state_key": args.state_key,
        "resume_state": args.resume_state,
        "max_cost": args.max_cost,
        "max_time_minutes": args.max_time_minutes,
    }
    if args.proxy_city:
        payload["proxy_city"] = args.proxy_city
    if args.proxy_state:
        payload["proxy_state"] = args.proxy_state
    if args.run_id:
        payload["run_id"] = args.run_id
    response = post_json(args, payload)
    print_json(response)
    return 0


def command_status(args: argparse.Namespace) -> int:
    print_json(post_json(args, {"action": "status", "run_id": args.run_id}))
    return 0


def command_logs(args: argparse.Namespace) -> int:
    response = post_json(args, {"action": "logs", "run_id": args.run_id, "tail": args.tail})
    if args.raw:
        for line in response.get("events", []):
            print(line)
    else:
        print_json(response)
    return 0


def command_result(args: argparse.Namespace) -> int:
    response = post_json(args, {"action": "result", "run_id": args.run_id})
    if args.csv_out and isinstance(response.get("leads"), list):
        write_csv(Path(args.csv_out), response["leads"])
        response["local_csv_path"] = args.csv_out
    print_json(response)
    return 0


def command_watch(args: argparse.Namespace) -> int:
    last_seen_event: str | None = None
    while True:
        status = post_json(args, {"action": "status", "run_id": args.run_id})
        events = post_json(args, {"action": "logs", "run_id": args.run_id, "tail": args.tail}).get(
            "events", []
        )
        start = 0
        if last_seen_event is not None and last_seen_event in events:
            start = len(events) - 1 - events[::-1].index(last_seen_event) + 1
        for line in events[start:]:
            print(line, flush=True)
        if events:
            last_seen_event = events[-1]
        state = status.get("status")
        print(f"status={state} updated_at={status.get('updated_at')}", file=sys.stderr, flush=True)
        if state not in {"queued", "running"}:
            print_json(status)
            return 0 if state == "succeeded" else 1
        time.sleep(args.interval)


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env-file", default=".env", help="dotenv file to load before requests")
    parser.add_argument("--endpoint", help="full Baseten predict endpoint")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Baseten model id")
    parser.add_argument("--environment", default=DEFAULT_ENVIRONMENT, help="Baseten environment")
    parser.add_argument("--timeout", type=int, default=300, help="HTTP timeout in seconds")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="start a workload run")
    add_common(run)
    run.add_argument("--attach", action="store_true", help="block until Baseten returns a result")
    run.add_argument("--micro", default=DEFAULT_MICRO)
    run.add_argument("--state-key", default="boattrader-miami-private-v1")
    run.add_argument("--resume-state", action="store_true")
    run.add_argument("--max-cost", type=float, default=10.0)
    run.add_argument("--max-time-minutes", type=int, default=180)
    run.add_argument("--proxy-city", default="")
    run.add_argument("--proxy-state", default="")
    run.add_argument("--run-id", help="optional deterministic run id")
    run.set_defaults(func=command_run)

    status = subparsers.add_parser("status", help="get detached run status")
    add_common(status)
    status.add_argument("--run-id", required=True)
    status.set_defaults(func=command_status)

    logs = subparsers.add_parser("logs", help="get detached run events")
    add_common(logs)
    logs.add_argument("--run-id", required=True)
    logs.add_argument("--tail", type=int, default=200)
    logs.add_argument("--raw", action="store_true")
    logs.set_defaults(func=command_logs)

    result = subparsers.add_parser("result", help="get detached run result")
    add_common(result)
    result.add_argument("--run-id", required=True)
    result.add_argument("--csv-out", help="write result leads to a local CSV")
    result.set_defaults(func=command_result)

    watch = subparsers.add_parser("watch", help="poll status and run events until completion")
    add_common(watch)
    watch.add_argument("--run-id", required=True)
    watch.add_argument("--tail", type=int, default=200)
    watch.add_argument("--interval", type=float, default=20.0)
    watch.set_defaults(func=command_watch)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    load_dotenv(Path(args.env_file))
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
