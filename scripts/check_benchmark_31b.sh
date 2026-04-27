#!/bin/bash
# Mantis — OSWorld benchmark progress checker for 31B F16 run
# Usage: ./check_benchmark_31b.sh

cd /Users/barada/Sandbox/Mason/cua-agent

echo ""
echo "  ╔═══════════════════════════════════════════════╗"
echo "  ║  Mantis — OSWorld Benchmark (31B F16)         ║"
echo "  ╚═══════════════════════════════════════════════╝"
echo ""

DOMAIN="${1:-all}"

rm -f /tmp/mantis_results_31b.json 2>/dev/null
.venv/bin/modal volume get osworld-data-31b-f16 "results/osworld_results_${DOMAIN}.json" /tmp/mantis_results_31b.json 2>/dev/null

if [ ! -f /tmp/mantis_results_31b.json ]; then
    .venv/bin/modal volume get osworld-data-31b-f16 results/osworld_results.json /tmp/mantis_results_31b.json 2>/dev/null
fi
if [ ! -f /tmp/mantis_results_31b.json ] && [ "$DOMAIN" = "all" ]; then
    .venv/bin/modal volume get osworld-data-31b-f16 results/osworld_results_os.json /tmp/mantis_results_31b.json 2>/dev/null
    DOMAIN="os"
fi

if [ ! -f /tmp/mantis_results_31b.json ]; then
    echo "  No results found on Modal volume (osworld-data-31b-f16)"
    echo "  Run: modal run --detach modal_osworld_31b_f16.py --domain $DOMAIN"
    echo ""
    exit 0
fi

python3 << 'PYEOF'
import json, sys, os

try:
    with open("/tmp/mantis_results_31b.json") as f:
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
    avg_per_task = r.get("avg_time_per_task_s", 0)
    if avg_per_task:
        remaining_tasks = total_expected - len(scores)
        eta_min = (remaining_tasks * avg_per_task) / 60
        remaining_cost = remaining_tasks * avg_per_task * 0.001261
        if remaining_tasks > 0:
            print(f"  ETA:       ~{eta_min:.0f} min | ~${remaining_cost:.2f} remaining")
print()
print(f"  ─── Results ────────────────────────────────────")

task_ids = r.get("task_ids", [])
for i in range(len(scores)):
    mark = "✓" if scores[i] > 0 else "✗"
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
    print(f"  {remaining} tasks remaining (~{remaining*5} min)")

# Compare with Q4_K_M results if available
try:
    with open("/tmp/mantis_results.json") as f:
        q4 = json.load(f)
    q4_scores = q4.get("scores", [])
    q4_avg = sum(q4_scores) / len(q4_scores) * 100 if q4_scores else 0
    q4_passed = sum(1 for s in q4_scores if s > 0)
    print()
    print(f"  ─── Comparison ─────────────────────────────────")
    print(f"  26B-A4B Q4_K_M: {q4_avg:.1f}% ({q4_passed}/{len(q4_scores)})")
    print(f"  31B F16:        {avg:.1f}% ({passed}/{len(scores)})")
    if len(scores) > 0 and len(q4_scores) > 0:
        delta = avg - q4_avg
        print(f"  Delta:          {'+' if delta >= 0 else ''}{delta:.1f}%")
    print(f"  ────────────────────────────────────────────────")
except:
    pass
PYEOF

echo ""
