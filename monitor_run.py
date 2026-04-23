#!/usr/bin/env python3
"""Monitor a running Mantis CUA job on Modal.

Polls the Modal volume for results JSON files, shows live progress
including task scores, extracted data, loop iterations, and cost.

Usage:
    uv run python monitor_run.py                    # auto-detect latest run
    uv run python monitor_run.py --session bt_dynamic
    uv run python monitor_run.py --interval 15      # poll every 15s
"""

import argparse
import json
import subprocess
import time
from datetime import datetime


VOLUME = "osworld-data"
RESULTS_DIR = "results"

# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def modal_volume_ls(path: str) -> list[str]:
    """List files on Modal volume."""
    cmd = ["uv", "run", "modal", "volume", "ls", VOLUME, path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return []
        return [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
    except Exception:
        return []


def modal_volume_get(remote_path: str, local_path: str) -> bool:
    """Download a file from Modal volume."""
    cmd = ["uv", "run", "modal", "volume", "get", "--force", VOLUME, remote_path, local_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.returncode == 0
    except Exception:
        return False


def find_latest_results(session: str = "") -> str | None:
    """Find the most recent results file on the volume."""
    files = modal_volume_ls(RESULTS_DIR)
    if not files:
        return None

    # Filter by session if given
    json_files = [f for f in files if f.endswith(".json")]
    if session:
        json_files = [f for f in json_files if session in f]

    if not json_files:
        return None

    # Sort by name (contains timestamp) — latest last
    json_files.sort()
    # modal volume ls returns paths like "results/file.json" — strip prefix
    latest = json_files[-1]
    if latest.startswith(RESULTS_DIR + "/"):
        latest = latest[len(RESULTS_DIR) + 1:]
    return latest


def fetch_results(filename: str) -> dict | None:
    """Download and parse results JSON from volume."""
    remote_path = f"{RESULTS_DIR}/{filename}"
    local_path = f"/tmp/mantis_monitor_{filename.replace('/', '_')}"

    if not modal_volume_get(remote_path, local_path):
        return None

    try:
        with open(local_path) as f:
            return json.load(f)
    except Exception:
        return None


def display_progress(data: dict, prev_state: dict | None = None) -> dict:
    """Display run progress. Returns state dict for change detection."""
    run_id = data.get("run_id", "?")
    model = data.get("model", "?")
    session = data.get("session_name", "?")
    tasks_run = data.get("tasks_run", 0)
    completed = data.get("completed_at", "")
    gpu_time = data.get("total_gpu_time_s", 0)
    cost = data.get("estimated_cost_usd", 0)
    scores = data.get("scores", [])
    details = data.get("task_details", [])

    prev_state = prev_state or {"tasks_shown": 0, "loop_iters": 0}

    n_done = len(scores)
    n_pass = sum(1 for s in scores if s > 0)
    avg = (sum(scores) / len(scores) * 100) if scores else 0

    # Count total leads extracted across all loop tasks
    total_leads = 0
    total_iterations = 0
    for d in details:
        if d.get("viable"):
            total_leads += d["viable"]
        if d.get("iterations"):
            total_iterations += d["iterations"]

    # Header
    now = datetime.now().strftime("%H:%M:%S")
    status = f"{GREEN}COMPLETE{RESET}" if completed else f"{YELLOW}RUNNING{RESET}"
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}Mantis Monitor{RESET}  {DIM}[{now}]{RESET}  {status}")
    print(f"  Run:     {run_id}  |  Model: {model}  |  Session: {session}")
    print(f"  GPU:     {gpu_time}s  |  Cost: {BOLD}${cost:.2f}{RESET}")

    # Leads extracted summary (prominent)
    if total_leads > 0 or total_iterations > 0:
        print(f"  Leads:   {GREEN}{BOLD}{total_leads}{RESET} extracted from {total_iterations} listings scanned")

    # Progress bar
    if tasks_run > 0:
        pct = n_done / tasks_run
        bar_len = 40
        filled = int(bar_len * pct)
        bar = f"{'█' * filled}{'░' * (bar_len - filled)}"
        print(f"  Progress: [{bar}] {n_done}/{tasks_run}")
    print(f"  Score:   {n_pass}/{n_done} passed ({avg:.1f}%)")
    print(f"{BOLD}{'='*70}{RESET}")

    # Task details — show new tasks AND update in-progress loops
    for i, detail in enumerate(details):
        task_id = detail.get("task_id", "?")
        success = detail.get("success", False)
        steps = detail.get("steps", 0)
        duration = detail.get("duration_s", 0)
        reason = detail.get("termination_reason", "")
        extracted = detail.get("extracted_data", "")
        error = detail.get("error", "")
        iterations = detail.get("iterations", 0)
        viable = detail.get("viable", 0)

        is_loop_in_progress = reason == "loop_in_progress"
        is_new = i >= prev_state["tasks_shown"]
        loop_changed = is_loop_in_progress and iterations != prev_state.get("loop_iters", 0)

        if not is_new and not loop_changed:
            continue

        if is_loop_in_progress:
            icon = f"{YELLOW}~{RESET}"
            # Show a mini progress bar for the loop
            loop_max = 50  # Approximate
            loop_pct = min(iterations / loop_max, 1.0)
            loop_bar_len = 20
            loop_filled = int(loop_bar_len * loop_pct)
            loop_bar = f"{'█' * loop_filled}{'░' * (loop_bar_len - loop_filled)}"
            print(f"  {icon} {BOLD}{task_id}{RESET}  [{loop_bar}] {iterations} scanned, {duration}s")
            print(f"    {CYAN}{BOLD}Leads: {viable}{RESET}{CYAN} viable / {iterations} scanned ({viable/iterations*100:.0f}% hit rate){RESET}" if iterations > 0 else "")
            print(f"    {DIM}Cost so far: ${cost:.2f} | Rate: ${cost/max(duration,1)*3600:.2f}/hr{RESET}")
        else:
            icon = f"{GREEN}✓{RESET}" if success else f"{RED}✗{RESET}"
            print(f"  {icon} {BOLD}{task_id}{RESET}  ({steps} steps, {duration}s) — {reason}")

            if iterations:
                print(f"    {CYAN}{BOLD}Leads: {viable}{RESET}{CYAN} extracted from {iterations} listings{RESET}")

        if extracted:
            snippet = extracted[:200].replace("\n", " ")
            print(f"    {DIM}Data: {snippet}{RESET}")

        if error:
            print(f"    {RED}Error: {error[:150]}{RESET}")

        # Show latest lead data entries
        data_entries = detail.get("data", [])
        if data_entries:
            # Show last 3 (most recent) for in-progress, first 5 for complete
            if is_loop_in_progress:
                show = data_entries[-3:]
                start_idx = len(data_entries) - 3
                for j, entry in enumerate(show):
                    print(f"    {DIM}  [{start_idx+j+1}] {entry[:120]}{RESET}")
            else:
                for j, entry in enumerate(data_entries[:5]):
                    print(f"    {DIM}  [{j+1}] {entry[:120]}{RESET}")
                if len(data_entries) > 5:
                    print(f"    {DIM}  ... +{len(data_entries)-5} more{RESET}")

    return {
        "tasks_shown": n_done if not any(d.get("termination_reason") == "loop_in_progress" for d in details) else max(n_done - 1, prev_state["tasks_shown"]),
        "loop_iters": total_iterations,
    }


def check_containers() -> bool:
    """Check if there are active Modal containers."""
    cmd = ["uv", "run", "modal", "container", "list"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        # If output has content rows beyond the header
        lines = [
            line
            for line in result.stdout.splitlines()
            if line.strip()
            and "Container ID" not in line
            and "━" not in line
            and "─" not in line
        ]
        return len(lines) > 0
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Monitor Mantis CUA run on Modal")
    parser.add_argument("--session", default="", help="Filter by session name (e.g., bt_dynamic)")
    parser.add_argument("--interval", type=int, default=10, help="Poll interval in seconds")
    parser.add_argument("--once", action="store_true", help="Check once and exit")
    args = parser.parse_args()

    print(f"{BOLD}Mantis Run Monitor{RESET}")
    print(f"  Volume:   {VOLUME}")
    print(f"  Session:  {args.session or '(auto-detect)'}")
    print(f"  Interval: {args.interval}s")
    print()

    prev_file = None
    prev_state = {"tasks_shown": 0, "loop_iters": 0}
    no_change_count = 0

    while True:
        # Find latest results file
        filename = find_latest_results(args.session)

        if not filename:
            containers = check_containers()
            if containers:
                print(f"{YELLOW}Waiting for results... (containers running){RESET}", end="\r")
            else:
                print(f"{DIM}No results found. Waiting for run to start...{RESET}", end="\r")

            if args.once:
                break
            time.sleep(args.interval)
            continue

        # New file detected
        if filename != prev_file:
            prev_file = filename
            prev_state = {"tasks_shown": 0, "loop_iters": 0}
            no_change_count = 0
            print(f"\n{CYAN}Found: {filename}{RESET}")

        # Fetch and display
        data = fetch_results(filename)
        if data:
            new_state = display_progress(data, prev_state)

            if new_state == prev_state:
                no_change_count += 1
            else:
                no_change_count = 0
            prev_state = new_state

            # Check if run is complete
            if data.get("completed_at"):
                # If containers are still running, this is a stale completed file —
                # a new run hasn't written its results yet. Keep waiting.
                if check_containers():
                    if not getattr(main, '_warned_stale', False):
                        print(f"\n{YELLOW}Previous run complete. Waiting for new results from active container...{RESET}")
                        main._warned_stale = True
                    # Reset to look for a newer file next poll
                    prev_file = None
                    time.sleep(args.interval)
                    continue

                scores = data.get("scores", [])
                details = data.get("task_details", [])
                n_pass = sum(1 for s in scores if s > 0)
                avg = sum(scores) / len(scores) * 100 if scores else 0
                total_leads = sum(d.get("viable", 0) for d in details)
                total_cost = data.get("estimated_cost_usd", 0)
                print(f"\n{GREEN}{BOLD}Run complete: {n_pass}/{len(scores)} ({avg:.1f}%){RESET}")
                print(f"  Leads extracted: {BOLD}{total_leads}{RESET}")
                print(f"  Total cost: {BOLD}${total_cost:.2f}{RESET}")
                if total_leads > 0:
                    print(f"  Cost per lead: ${total_cost/total_leads:.2f}")
                break

            # Warn if stale (loop tasks can take 20+ min per task)
            if no_change_count >= 30:
                print(f"\n{YELLOW}No progress in {no_change_count * args.interval}s — run may be stuck{RESET}")

        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
