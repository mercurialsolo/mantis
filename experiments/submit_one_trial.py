"""Run ONE autoresearch trial against zip 33131 (Miami).

Reads CONFIG from experiments.experiment, submits a Modal /v1/predict
run with those knobs, polls to terminal, pulls the leads.csv +
claude_cost_by_path.json from the Modal volume, computes the
autoresearch metrics, and prints the summary block the agent grep-
extracts.

READ-ONLY orchestrator. DO NOT MODIFY. (The agent edits
experiments/experiment.py and optionally the plan / decomposer prompt /
grounding default / verifier-escalation default.)

Also enforces the $10 cumulative-cost budget cap. Before submitting,
sums ``cost_usd`` across all rows in experiments/results.tsv. If the
sum is >= $10, refuses to submit and exits 2 (``budget_exhausted``).

Output format (machine-readable trailing block):

    ---
    commit:           <sha7>
    config_hash:      <sha7>
    cost_usd:         <float>
    valid_leads:      <int>
    $_per_valid:      <float>
    total_leads:      <int>
    halt_reason:      "<str>"
    wall_seconds:     <float>
    status:           candidate_keep | candidate_discard | crash
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Import after sys.path setup so the mantis_agent package resolves
from mantis_agent.plan_decomposer import PlanDecomposer  # noqa: E402
from mantis_agent.server_utils import (  # noqa: E402
    build_micro_suite,
    merge_runtime,
    micro_plan_steps_to_dicts,
)


# ── Constants ────────────────────────────────────────────────────────

ZIP = "33131"  # Miami — proven test bed
SEARCH_RADIUS = "50"
POP_PASSWORD = "SelfService38#!"
ENDPOINT = "https://getmason--mantis-cua-server-api.modal.run"
BUDGET_CAP = 10.0  # USD across all trials
RESULTS_TSV = REPO_ROOT / "experiments" / "results.tsv"
PROXY_CITY = "miami"
PROXY_STATE = "florida"
PROXY_PROVIDER = "oxylabs"


# ── Helpers ──────────────────────────────────────────────────────────


def _git_short_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=7", "HEAD"], cwd=REPO_ROOT,
        )
        return out.decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _config_hash(config: dict) -> str:
    s = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:7]


def _cumulative_cost() -> float:
    """Sum cost_usd column across all rows in results.tsv."""
    if not RESULTS_TSV.exists():
        return 0.0
    total = 0.0
    with RESULTS_TSV.open(newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        try:
            next(reader)  # skip header
        except StopIteration:
            return 0.0
        for row in reader:
            if len(row) >= 2:
                try:
                    total += float(row[1])
                except ValueError:
                    continue
    return total


def _read_env(key: str) -> str:
    val = os.environ.get(key, "")
    if val:
        return val
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1].strip()
    return ""


def _post(path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    r = requests.post(
        f"{ENDPOINT}{path}",
        headers={
            "X-Mantis-Token": _read_env("MANTIS_API_TOKEN"),
            "Content-Type": "application/json",
        },
        data=json.dumps(body),
        timeout=120,
    )
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {"raw": r.text}


# ── Lead-counting (the autoresearch metric source of truth) ─────────


def _count_leads_csv(csv_path: Path) -> tuple[int, int]:
    """Return (total_leads, valid_leads).

    ``valid`` = phone-bearing, non-dealer rows. Phone column is
    considered populated when it's not in the null sentinel set;
    Seller_Type column is considered private when missing OR
    explicitly "private". A row with Seller_Type == "dealer" is
    excluded from valid even if it has a phone.
    """
    if not csv_path.exists():
        return 0, 0
    null_phones = {"", "none", "not listed", '""', "n/a", "na", "unknown"}
    total = 0
    valid = 0
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            total += 1
            phone = (row.get("Phone") or "").strip().strip('"').lower()
            seller_type = (row.get("Seller_Type") or "").strip().lower()
            has_phone = bool(phone) and phone not in null_phones
            is_dealer = seller_type == "dealer"
            if has_phone and not is_dealer:
                valid += 1
    return total, valid


# ── Pull artifacts from Modal volume ────────────────────────────────


def _pull_artifacts(profile: str, workflow: str, dest: Path) -> Path:
    """Pull leads.csv + claude_cost_by_path.json from the Modal volume
    to ``dest``. Returns ``dest`` (always — pulls are best-effort)."""
    dest.mkdir(parents=True, exist_ok=True)
    for fname in ("leads.csv", "claude_cost_by_path.json"):
        try:
            subprocess.run(
                [
                    "uv", "run", "modal", "volume", "get",
                    "osworld-data",
                    f"/runs/{profile}/{workflow}/{fname}",
                    str(dest / fname),
                    "--force",
                ],
                check=False, capture_output=True, timeout=120, cwd=REPO_ROOT,
            )
        except subprocess.TimeoutExpired:
            continue
    return dest


def _read_cost(json_path: Path) -> tuple[float, str]:
    """Return (cost_usd, halt_reason) from claude_cost_by_path.json."""
    if not json_path.exists():
        return 0.0, ""
    try:
        d = json.loads(json_path.read_text())
        cost = float((d.get("totals") or {}).get("cost_usd") or 0.0)
        outcome = d.get("outcome") or {}
        halt = str(outcome.get("halt_reason") or "")
        return cost, halt
    except Exception:  # noqa: BLE001
        return 0.0, ""


# ── Main ─────────────────────────────────────────────────────────────


def main() -> int:
    spent = _cumulative_cost()
    print(f"cumulative_spend_so_far: ${spent:.4f} of ${BUDGET_CAP:.2f} cap")
    if spent >= BUDGET_CAP:
        print()
        print("---")
        print(f"commit:           {_git_short_sha()}")
        print("config_hash:      n/a")
        print("cost_usd:         0.0000")
        print("valid_leads:      0")
        print("$_per_valid:      999.9999")
        print("total_leads:      0")
        print('halt_reason:      "budget_exhausted"')
        print("wall_seconds:     0.0")
        print("status:           budget_exhausted")
        return 2

    # Import CONFIG fresh — the agent may have edited it this trial
    from experiments.experiment import CONFIG  # noqa: PLC0415

    plan_path = REPO_ROOT / CONFIG["plan_path"]
    if not plan_path.exists():
        print(f"ERROR: plan not found at {plan_path}", file=sys.stderr)
        return 4

    # Substitute plan placeholders
    raw = plan_path.read_text()
    plan_text = (
        raw.replace("{zip_code}", ZIP)
        .replace("{search_radius}", SEARCH_RADIUS)
        .replace("{pop_password}", POP_PASSWORD)
    )

    # Decompose locally
    api_key = _read_env("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 4

    decomposer = PlanDecomposer(api_key=api_key, model="claude-opus-4-7")
    cache_dir = REPO_ROOT / "data" / "plan_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_template = str(cache_dir / "decomposed_{hash}.json")
    print(f"[decompose] {len(plan_text)} chars → PlanDecomposer")
    t_decompose_0 = time.time()
    micro_plan = decomposer.decompose_text(
        plan_text, cache_path_template=cache_template,
    )
    print(
        f"[decompose] → {len(micro_plan.steps)} steps "
        f"(plan_hash {micro_plan.plan_hash}) in "
        f"{time.time() - t_decompose_0:.1f}s",
    )

    # Build runtime + suite
    runtime = merge_runtime({
        "proxy_disabled": False,
        "proxy_provider": PROXY_PROVIDER,
        "proxy_city": PROXY_CITY,
        "proxy_state": PROXY_STATE,
        "max_cost": float(CONFIG["max_cost"]),
        "max_time_minutes": int(CONFIG["max_time_minutes"]),
    })

    profile_id = f"autoresearch-zip-{ZIP}"
    workflow_id = f"autoresearch-{ZIP}-{int(time.time())}"

    suite = build_micro_suite(
        micro_plan_steps_to_dicts(micro_plan.steps),
        micro_plan.domain or "boattrader_scrape",
        profile_id=profile_id,
        workflow_id=workflow_id,
        plan_hash=micro_plan.plan_hash,
        plan_evolution_scope_id=profile_id,
        extractor_model=str(CONFIG["extractor_model"]),
        **runtime,
    )

    # Fan-out opt-in
    fanout = int(CONFIG["fanout_phase1_workers"])
    if fanout > 1:
        suite["_fanout_phase1_workers"] = fanout

    submit_body = {
        "task_suite": suite,
        "profile_id": suite["_profile_id"],
        "workflow_id": suite["_workflow_id"],
        "cua_model": "holo3",
        "max_steps": 200,
        "detached": True,
        **runtime,
    }

    # Health probe
    try:
        h = requests.get(f"{ENDPOINT}/v1/health", timeout=30)
        if h.status_code != 200:
            print(f"ERROR: endpoint unhealthy ({h.status_code})", file=sys.stderr)
            return 3
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: health check failed: {exc}", file=sys.stderr)
        return 3

    # Submit
    print(f"[trial]    config_hash={_config_hash(CONFIG)} fanout={fanout}")
    print(f"[trial]    plan_path={CONFIG['plan_path']} extractor={CONFIG['extractor_model']}")
    print(f"[trial]    max_cost=${CONFIG['max_cost']:.2f} max_time={CONFIG['max_time_minutes']}m")
    t0 = time.time()
    status, resp = _post("/v1/predict", submit_body)
    if status != 200:
        print(f"ERROR: submit returned {status}: {resp}", file=sys.stderr)
        return 5
    run_id = resp.get("run_id", "")
    print(f"[submit]   HTTP 200  run_id={run_id}")

    # Poll until terminal
    terminal = {"succeeded", "failed", "cancelled", "halted"}
    last_status = ""
    halt_reason = ""
    started = time.time()
    while True:
        if time.time() - started > 60 * int(CONFIG["max_time_minutes"]) + 60:
            print("TIMEOUT in poll loop", file=sys.stderr)
            halt_reason = "poll_timeout"
            break
        s_status, s_resp = _post(
            "/v1/predict",
            {"action": "status", "run_id": run_id},
        )
        st = s_resp.get("status", "?")
        if st != last_status:
            print(
                f"[poll]     t={int(time.time() - started):4d}s  status={st}",
            )
            last_status = st
        if st in terminal:
            halt_reason = str(s_resp.get("halt_reason") or "")
            break
        time.sleep(15)

    wall = time.time() - t0

    # Pull artifacts
    artifacts_dir = REPO_ROOT / "data" / "leads" / "autoresearch" / workflow_id
    _pull_artifacts(profile_id, workflow_id, artifacts_dir)

    leads_csv = artifacts_dir / "leads.csv"
    cost_json = artifacts_dir / "claude_cost_by_path.json"

    total_leads, valid_leads = _count_leads_csv(leads_csv)
    cost_usd, halt_from_cost = _read_cost(cost_json)
    if not halt_reason and halt_from_cost:
        halt_reason = halt_from_cost
    dpv = (cost_usd / valid_leads) if valid_leads > 0 else 999.9999

    # Crash heuristic
    is_crash = total_leads == 0 and (
        halt_reason.startswith("required_failed")
        or halt_reason in ("cf_challenge", "external_pause", "poll_timeout")
        or not leads_csv.exists()
    )
    rec_status = (
        "crash" if is_crash else
        ("candidate_keep" if valid_leads > 0 else "candidate_discard")
    )

    print()
    print("---")
    print(f"commit:           {_git_short_sha()}")
    print(f"config_hash:      {_config_hash(CONFIG)}")
    print(f"cost_usd:         {cost_usd:.4f}")
    print(f"valid_leads:      {valid_leads}")
    print(f"$_per_valid:      {dpv:.4f}")
    print(f"total_leads:      {total_leads}")
    print(f'halt_reason:      "{halt_reason}"')
    print(f"wall_seconds:     {wall:.1f}")
    print(f"status:           {rec_status}")
    print(f"artifacts_dir:    {artifacts_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
