#!/bin/bash
# Mantis — OSWorld benchmark progress checker (reads from Modal volume)
# Usage: ./check_benchmark.sh

cd /Users/barada/Sandbox/Mason/cua-agent

echo ""
echo "  ╔═══════════════════════════════════════════════╗"
echo "  ║  Mantis — OSWorld Benchmark Progress          ║"
echo "  ╚═══════════════════════════════════════════════╝"
echo ""

# Accept domain as argument (default: try "all", fall back to "os")
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
    echo "  No results found on Modal volume"
    echo "  Run: modal run --detach modal_osworld_direct.py --domain $DOMAIN"
    echo ""
    exit 0
fi

python3 << 'PYEOF'
import json, sys, os

try:
    with open("/tmp/mantis_results.json") as f:
        r = json.load(f)
except Exception as e:
    print(f"  Error reading results: {e}")
    sys.exit(1)

scores = r.get("scores", [])
passed = sum(1 for s in scores if s > 0)
total_expected = 24
avg = sum(scores) / len(scores) * 100 if scores else 0
model = r.get("model", "unknown")
task_details = r.get("task_details", [])
gpu_time = r.get("total_gpu_time_s", 0)
cost = r.get("estimated_cost_usd", 0)

done = len(scores) >= total_expected
status = "COMPLETE ★" if done else "RUNNING ●"
print(f"  Status:    {status}")
print(f"  Model:     {model}")
print(f"  Completed: {len(scores)}/{total_expected} ({len(scores)/total_expected*100:.0f}%)")
print(f"  Passed:    {passed}/{len(scores)}")
print(f"  Score:     {avg:.1f}%")
if gpu_time:
    print(f"  GPU time:  {gpu_time/60:.0f} min | Est. cost: ${cost:.2f}")
print()
print(f"  ─── Results ────────────────────────────────────")

# Try to get task descriptions from task_details or task_ids
task_ids = r.get("task_ids", [])
for i in range(len(scores)):
    mark = "✓" if scores[i] > 0 else "✗"
    # Try to load task description from config
    desc = f"Task {i+1}"
    if i < len(task_ids):
        tid = task_ids[i]
        config_path = f"OSWorld/evaluation_examples/examples/os/{tid}.json"
        if os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    ex = json.load(f)
                desc = ex.get("instruction", desc)[:55]
            except:
                pass
    print(f"  {mark} {i+1:2d}. {desc}")

print(f"  ────────────────────────────────────────────────")
remaining = total_expected - len(scores)
if remaining <= 0:
    print(f"  ★ COMPLETE — Final: {avg:.1f}% ({passed}/{len(scores)})")
else:
    print(f"  {remaining} tasks remaining (~{remaining*3} min)")
PYEOF

echo ""
