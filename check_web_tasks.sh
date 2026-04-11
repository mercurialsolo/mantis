#!/bin/bash
# Mantis — Web Task benchmark progress checker
# Usage: ./check_web_tasks.sh [session_name]
# Example: ./check_web_tasks.sh staffai_crm

cd /Users/barada/Sandbox/Mason/cua-agent

SESSION="${1:-staffai_crm}"

# Find the most recent results file for this session
rm -f /tmp/mantis_web_results.json /tmp/mantis_web_ls.txt 2>/dev/null
.venv/bin/modal volume ls osworld-data results/ 2>/dev/null | grep "web_results_${SESSION}_2" | sort -r | head -1 > /tmp/mantis_web_ls.txt

LATEST=$(cat /tmp/mantis_web_ls.txt | xargs)
if [ -n "$LATEST" ]; then
    .venv/bin/modal volume get osworld-data "$LATEST" /tmp/mantis_web_results.json 2>/dev/null
fi

if [ ! -f /tmp/mantis_web_results.json ]; then
    echo ""
    echo "  ╔═══════════════════════════════════════════════╗"
    echo "  ║  Mantis — Web Task Progress                   ║"
    echo "  ╚═══════════════════════════════════════════════╝"
    echo ""
    echo "  No results found for session: $SESSION"
    echo "  Run: modal run --detach modal_web_tasks.py --task-file tasks/crm/staffai_tasks.json"
    echo ""
    exit 0
fi

python3 << 'PYEOF'
import json, sys

session = sys.argv[1] if len(sys.argv) > 1 else "staffai_crm"

try:
    with open("/tmp/mantis_web_results.json") as f:
        content = f.read().strip()
    if not content:
        print("  Waiting for first result...")
        sys.exit(0)
    r = json.loads(content)
except json.JSONDecodeError:
    print("  Results file being written — try again in a moment")
    sys.exit(0)
except Exception as e:
    print(f"  Error: {e}")
    sys.exit(1)

scores = r.get("scores", [])
passed = sum(1 for s in scores if s > 0)
total_expected = r.get("tasks_run", len(scores))
avg = sum(scores) / len(scores) * 100 if scores else 0
model = r.get("model", "unknown")
gpu_time = r.get("total_gpu_time_s", 0)
cost = r.get("estimated_cost_usd", 0)
run_id = r.get("run_id", "?")
started = r.get("started_at", "")
completed = r.get("completed_at", "")
session_name = r.get("session_name", session)
base_url = r.get("base_url", "")

done = len(scores) >= total_expected

from datetime import datetime
def format_local(iso_str):
    if not iso_str:
        return ""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        local_dt = dt.astimezone()
        tz_name = local_dt.strftime("%Z")
        return local_dt.strftime(f"%Y-%m-%d %H:%M {tz_name}")
    except:
        return iso_str[:19]

started_short = format_local(started)
completed_short = format_local(completed)

print()
print("  ╔═══════════════════════════════════════════════╗")
print("  ║  Mantis — Web Task Progress                   ║")
print("  ╚═══════════════════════════════════════════════╝")
print()
print(f"  Run:       {run_id}")
print(f"  Session:   {session_name}")
print(f"  Target:    {base_url}")
if started_short:
    print(f"  Started:   {started_short}")
if done and completed_short:
    print(f"  Finished:  {completed_short}")
print(f"  Model:     {model}")
print(f"  Status:    {'COMPLETE ★' if done else 'RUNNING ●'}")
print()
print(f"  Completed: {len(scores)}/{total_expected} ({len(scores)/total_expected*100:.0f}%)" if total_expected else "")
print(f"  Passed:    {passed}/{len(scores)}")
print(f"  Score:     {avg:.1f}%")
if gpu_time:
    print(f"  GPU time:  {gpu_time/60:.0f} min | Cost: ${cost:.2f}")
print()
print(f"  ─── Results ────────────────────────────────────")

task_details = r.get("task_details", [])
for i in range(len(scores)):
    mark = "✓" if scores[i] > 0 else "✗"
    desc = f"Task {i+1}"
    steps = ""
    duration = ""
    verified = ""
    if i < len(task_details):
        td = task_details[i]
        desc = td.get("instruction", desc)[:50]
        s = td.get("steps", 0)
        d = td.get("duration_s", 0)
        if s: steps = f"{s} steps"
        if d: duration = f"{d:.0f}s"
        if td.get("verified"):
            verified = " ✔verified"
        if td.get("error"):
            desc = f"ERROR: {td['error'][:40]}"
    suffix = ""
    parts = [p for p in [steps, duration] if p]
    if parts:
        suffix = f"  ({' '.join(parts)}{verified})"
    print(f"  {mark} {i+1:2d}. {desc}{suffix}")

print(f"  ────────────────────────────────────────────────")
if done:
    print(f"  ★ COMPLETE — Final: {avg:.1f}% ({passed}/{len(scores)})")
else:
    remaining = total_expected - len(scores)
    print(f"  {remaining} tasks remaining")
print()
PYEOF
