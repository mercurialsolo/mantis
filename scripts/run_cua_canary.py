"""Fire the `/v1/cua` (Claude computer-use) canary against the mantis-linkedin
sim env — NOT real linkedin.com.

Why a sim-env target: the real-LinkedIn `/v1/predict` canary kept tripping
LinkedIn's anti-bot from Modal IPs (``form_target_not_found``), which is
noise, not a regression. The `mantis-linkedin` Modal sim env
(``deploy/sim_envs/modal_mantis_linkedin.py``) is a high-fidelity,
network-reachable LinkedIn mirror with no bot wall, so a CUA run there
exercises the executor end-to-end deterministically.

The task is deliberately **non-destructive read-only**: open the feed and
read the top post. It never clicks Connect / Like / Comment / Post / Apply,
never logs in, never submits a form — mirroring the safety constraints of
the existing cold-mount canary.

Submits to ``/v1/predict`` with ``cua_model="claude"`` (the Modal Claude
executor reads ``tasks[]``, not ``_micro_plan`` — see
``feedback_cua_model_holo3_vs_claude_shape``) and polls until terminal.

Usage::

    MANTIS_API_TOKEN=... uv run python scripts/run_cua_canary.py

Override the target with ``MANTIS_LINKEDIN_SIM_URL``.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

import requests

ENDPOINT = "https://getmason--mantis-cua-server-api.modal.run"

# The deployed mantis-linkedin Modal sim env. Override via env for a
# different sandbox / a locally-tunnelled instance.
SIM_URL = os.environ.get(
    "MANTIS_LINKEDIN_SIM_URL",
    "https://getmason--mantis-sim-env-mantis-linkedin-web.modal.run",
).rstrip("/")

PROFILE_ID = f"cua-canary-{int(time.time())}"
WORKFLOW_ID = f"cua-canary-{int(time.time())}"


def _token() -> str:
    tok = os.environ.get("MANTIS_API_TOKEN", "")
    if not tok:
        print("ERROR: MANTIS_API_TOKEN required in env", file=sys.stderr)
        sys.exit(2)
    return tok


def _post(path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    r = requests.post(
        f"{ENDPOINT}{path}",
        json=body,
        headers={"X-Mantis-Token": _token(), "Content-Type": "application/json"},
        timeout=60,
    )
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, {"raw": r.text}


def _build_task_suite() -> dict[str, Any]:
    """A read-only LinkedIn-feed task for the Claude executor.

    The Claude executor consumes ``tasks[]`` + ``base_url`` directly. Proxy
    is disabled (``_proxy_disabled``) — the sim env is a plain Modal URL
    with no bot wall, so routing through PrivateProxy would only add a
    failure surface.
    """
    return {
        "session_name": "cua_linkedin_canary",
        "base_url": SIM_URL,
        "tasks": [
            {
                "task_id": "linkedin_sim_feed_read",
                "intent": (
                    "On the LinkedIn feed page, read the author name and the "
                    "text of the single top-most post. Report what you read. "
                    "Do NOT click Connect, Like, Comment, Repost, Send, Start "
                    "a post, or Apply. Do NOT log in or submit any form. This "
                    "is a read-only task."
                ),
                "start_url": f"{SIM_URL}/feed/",
            },
        ],
        "_proxy_disabled": True,
        "_plan_name": "cua_linkedin_canary",
    }


def main() -> int:
    print(f"[canary] target sim env: {SIM_URL}")

    # Warm the API health endpoint twice — the cua-server cold-starts and a
    # first-hit ReadTimeout would look like a canary failure.
    for i in (1, 2):
        try:
            h = requests.get(f"{ENDPOINT}/v1/health", timeout=60)
            print(f"[health {i}] HTTP {h.status_code}: {h.text[:80]}")
            if h.status_code == 200:
                break
        except requests.RequestException as exc:
            print(f"[health {i}] {type(exc).__name__}: {exc}")

    submit_body = {
        "task_suite": _build_task_suite(),
        "profile_id": PROFILE_ID,
        "workflow_id": WORKFLOW_ID,
        "cua_model": "claude",  # routes to run_claude_cua (the /v1/cua executor)
        "max_steps": 12,
    }
    status, resp = _post("/v1/predict", submit_body)
    print(f"[submit] HTTP {status}  run_id={resp.get('run_id')!r}")
    if status != 200:
        print(f"[submit] body: {resp}", file=sys.stderr)
        return 1
    run_id = resp.get("run_id")
    if not run_id:
        print("[submit] no run_id in response", file=sys.stderr)
        return 1

    terminal = {
        "succeeded", "failed", "cancelled", "halted",
        "completed", "completed_with_failures",
    }
    started = time.time()
    last_status = ""
    poll_n = 0
    while True:
        poll_n += 1
        if time.time() - started > 20 * 60:
            print("TIMEOUT: 20 minutes without terminal status", file=sys.stderr)
            return 1
        _, s_resp = _post("/v1/predict", {"action": "status", "run_id": run_id})
        status_str = s_resp.get("status", "?")
        if status_str != last_status:
            print(f"[{poll_n:3d}] t={int(time.time() - started):4d}s  status={status_str}")
            last_status = status_str
        if status_str in terminal:
            print(f"\n[done] terminal status={status_str}")
            print(f"[done] payload: {s_resp}")
            # Honest baseline: with today's /v1/cua executor this read-only
            # feed task reliably HALTS — the Claude loop sees the post but
            # never emits a verified done within max_steps (cua-issues core
            # finding). After the #940 status-honesty fix, the wire status
            # truthfully reports that ``halted`` instead of a false
            # ``succeeded``. So a clean run that REACHES a terminal status is
            # a healthy canary EXECUTION (exit 0); judging the run's quality
            # — i.e. catching a false ``succeeded`` on an all-failed
            # trajectory, the regression we actually fear — is the hourly
            # watch's Augur job, not this script's.
            #
            # The one thing this script flags red: the run never reached a
            # terminal status (submit/infra failure) — handled by the submit
            # guard and the 20-min timeout above.
            return 0
        time.sleep(10)


if __name__ == "__main__":
    raise SystemExit(main())
