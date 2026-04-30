#!/bin/bash
# Mantis — Web Task progress checker
# Usage: ./check_web_tasks.sh [session_name]
# Examples:
#   ./check_web_tasks.sh                    # latest boattrader result
#   ./check_web_tasks.sh crm_demo        # CRM results
#   ./check_web_tasks.sh bt_full_prod       # specific session
#   ./check_web_tasks.sh boattrader         # any boattrader result

cd /Users/barada/Sandbox/Mason/cua-agent

SESSION="${1:-boattrader}"

# Find the most recent results file matching the session name
LATEST=$(.venv/bin/modal volume ls osworld-data results/ 2>/dev/null | grep -i "$SESSION" | sort -r | head -1 | xargs)

if [ -z "$LATEST" ]; then
    echo ""
    echo "  ╔═══════════════════════════════════════════════╗"
    echo "  ║  Mantis — Web Task Progress                   ║"
    echo "  ╚═══════════════════════════════════════════════╝"
    echo ""
    echo "  No results found for: $SESSION"
    echo "  Available sessions:"
    .venv/bin/modal volume ls osworld-data results/ 2>/dev/null | sed 's/.*results\//    /' | sort -r | head -10
    echo ""
    exit 0
fi

echo "  Downloading: $LATEST"
rm -rf /tmp/mantis_web_check 2>/dev/null
.venv/bin/modal volume get osworld-data "$LATEST" /tmp/mantis_web_check 2>/dev/null

# Find the JSON file
JSONFILE=$(find /tmp/mantis_web_check -name "*.json" 2>/dev/null | head -1)
if [ -z "$JSONFILE" ]; then
    JSONFILE="/tmp/mantis_web_check"
fi

python3 << PYEOF
import json, sys, os

path = "$JSONFILE"
try:
    with open(path) as f:
        r = json.load(f)
except Exception as e:
    print(f"  Error reading {path}: {e}")
    sys.exit(1)

scores = r.get("scores", [])
passed = sum(1 for s in scores if s > 0)
total = len(scores)
tasks_run = r.get("tasks_run", total)
model = r.get("model", "?")
gpu_time = r.get("total_gpu_time_s", 0)
cost = r.get("estimated_cost_usd", 0)
run_id = r.get("run_id", "?")
session_name = r.get("session_name", "$SESSION")
avg = sum(scores) / total * 100 if total else 0
done = total >= tasks_run if tasks_run else True

from datetime import datetime
def fmt(iso_str):
    if not iso_str: return ""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.astimezone().strftime("%Y-%m-%d %H:%M %Z")
    except: return iso_str[:19]

print()
print("  ╔═══════════════════════════════════════════════╗")
print("  ║  Mantis — Web Task Progress                   ║")
print("  ╚═══════════════════════════════════════════════╝")
print()
print(f"  Run:       {run_id}")
print(f"  Session:   {session_name}")
s = fmt(r.get("started_at",""))
c = fmt(r.get("completed_at",""))
if s: print(f"  Started:   {s}")
if done and c: print(f"  Finished:  {c}")
print(f"  Model:     {model}")
print(f"  Status:    {'COMPLETE ★' if done else 'RUNNING ●'}")
print()
if tasks_run: print(f"  Completed: {total}/{tasks_run}")
print(f"  Passed:    {passed}/{total}")
print(f"  Score:     {avg:.1f}%")
if gpu_time: print(f"  GPU time:  {gpu_time//60}min | Cost: \${cost:.2f}")

contacts = 0
print()
print(f"  ─── Results ────────────────────────────────────")
for i, td in enumerate(r.get("task_details", [])):
    mark = "✓" if td.get("success") else "✗"
    tid = td.get("task_id", f"task_{i}")[:40]
    steps = td.get("steps", "?")
    dur = td.get("duration_s", 0)
    data = td.get("extracted_data", "")
    retry = " ↻" if td.get("retry") else ""

    parts = []
    if steps != "?": parts.append(f"{steps}st")
    if dur: parts.append(f"{dur}s")
    suffix = f" ({' '.join(parts)})" if parts else ""

    # Contact detection
    if data:
        dl = data.lower()
        cf = any(kw in dl for kw in ["cloudflare", "verify you are human"])
        has_phone = any(kw in dl for kw in ["phone", "viable", "seller phone", "contact"])
        if has_phone and not cf:
            contacts += 1
            suffix += " 📞"

    print(f"  {mark} {i+1:2d}. {tid}{suffix}{retry}")

    # Print extracted data (skip CF errors)
    if data:
        dl = data.lower()
        if not any(kw in dl for kw in ["cloudflare", "verify you"]):
            print(f"      📄 {data[:150]}")

print(f"  ────────────────────────────────────────────────")
if done:
    print(f"  ★ COMPLETE — {avg:.1f}% ({passed}/{total})")
else:
    print(f"  ● RUNNING — {total} tasks done so far")
if contacts:
    print(f"  📞 Contacts found: {contacts}")
print()
PYEOF
