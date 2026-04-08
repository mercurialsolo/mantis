#!/bin/bash
# Mantis — OSWorld benchmark progress checker
# Usage: ./check_benchmark.sh [domain]

cd /Users/barada/Sandbox/Mason/cua-agent

DOMAIN="${1:-all}"

# Fetch latest results from Modal volume
rm -f /tmp/mantis_results.json 2>/dev/null
.venv/bin/modal volume get osworld-data "results/osworld_results_${DOMAIN}.json" /tmp/mantis_results.json 2>/dev/null

# Fall back to legacy filename, then to "os"
if [ ! -f /tmp/mantis_results.json ]; then
    .venv/bin/modal volume get osworld-data results/osworld_results.json /tmp/mantis_results.json 2>/dev/null
fi
if [ ! -f /tmp/mantis_results.json ] && [ "$DOMAIN" = "all" ]; then
    .venv/bin/modal volume get osworld-data results/osworld_results_os.json /tmp/mantis_results.json 2>/dev/null
    DOMAIN="os"
fi

if [ ! -f /tmp/mantis_results.json ]; then
    echo ""
    echo "  ╔═══════════════════════════════════════════════╗"
    echo "  ║  Mantis — OSWorld Benchmark Progress          ║"
    echo "  ╚═══════════════════════════════════════════════╝"
    echo ""
    echo "  No results found on Modal volume"
    echo "  Run: modal run --detach modal_osworld_direct.py --domain $DOMAIN"
    echo ""
    exit 0
fi

python3 << 'PYEOF'
import json, sys, os
from datetime import datetime

try:
    with open("/tmp/mantis_results.json") as f:
        r = json.load(f)
except Exception as e:
    print(f"  Error reading results: {e}")
    sys.exit(1)

scores = r.get("scores", [])
passed = sum(1 for s in scores if s > 0)
domain = r.get("domain", "?")
total_expected = r.get("tasks_run", len(scores)) if len(scores) >= 24 else 24
avg = sum(scores) / len(scores) * 100 if scores else 0
model = r.get("model", "unknown")
gpu_time = r.get("total_gpu_time_s", 0)
cost = r.get("estimated_cost_usd", 0)
run_id = r.get("run_id", "?")
started = r.get("started_at", "")
completed = r.get("completed_at", "")
learnings = r.get("learnings", [])

done = len(scores) >= total_expected

# Format timestamps
started_short = ""
if started:
    try:
        dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
        started_short = dt.strftime("%Y-%m-%d %H:%M UTC")
    except: started_short = started[:19]

print()
print("  ╔═══════════════════════════════════════════════╗")
print("  ║  Mantis — OSWorld Benchmark Progress          ║")
print("  ╚═══════════════════════════════════════════════╝")
print()
print(f"  Run:       {run_id}")
if started_short:
    print(f"  Started:   {started_short}")
print(f"  Domain:    {domain}")
print(f"  Model:     {model}")
print(f"  Status:    {'COMPLETE ★' if done else 'RUNNING ●'}")
print()
print(f"  Completed: {len(scores)}/{total_expected} ({len(scores)/total_expected*100:.0f}%)")
print(f"  Passed:    {passed}/{len(scores)}")
print(f"  Score:     {avg:.1f}%")
if gpu_time:
    print(f"  GPU time:  {gpu_time/60:.0f} min | Cost: ${cost:.2f}")
if learnings:
    print(f"  Learnings: {len(learnings)} distilled")
print()
print(f"  ─── Results ────────────────────────────────────")

task_ids = r.get("task_ids", [])
task_details = r.get("task_details", [])
for i in range(len(scores)):
    mark = "✓" if scores[i] > 0 else "✗"
    desc = f"Task {i+1}"
    # Try task_details first, then config file
    if i < len(task_details) and task_details[i].get("instruction"):
        desc = task_details[i]["instruction"][:55]
    elif i < len(task_ids):
        tid = task_ids[i]
        for dom_dir in ["os", "chrome", "vs_code", "gimp", "vlc", "thunderbird",
                        "libreoffice_calc", "libreoffice_writer", "libreoffice_impress", "multi_apps"]:
            config_path = f"OSWorld/evaluation_examples/examples/{dom_dir}/{tid}.json"
            if os.path.exists(config_path):
                try:
                    with open(config_path) as f:
                        desc = json.load(f).get("instruction", desc)[:55]
                except: pass
                break
    print(f"  {mark} {i+1:2d}. {desc}")

print(f"  ────────────────────────────────────────────────")
if done:
    print(f"  ★ COMPLETE — Final: {avg:.1f}% ({passed}/{len(scores)})")
else:
    remaining = total_expected - len(scores)
    print(f"  {remaining} tasks remaining (~{remaining*3} min)")
print()
PYEOF
