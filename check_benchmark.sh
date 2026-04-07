#!/bin/bash
# Mantis — OSWorld benchmark progress checker
# Usage: ./check_benchmark.sh

cd /Users/barada/Sandbox/Mason/cua-agent

echo ""
echo "  ╔═══════════════════════════════════════════════╗"
echo "  ║  Mantis — OSWorld Benchmark Progress          ║"
echo "  ╚═══════════════════════════════════════════════╝"
echo ""

# Check if process is running
if ps aux | grep "modal run modal_osworld" | grep -v grep > /dev/null 2>&1; then
    echo "  Status: RUNNING ●"
else
    echo "  Status: STOPPED ○"
fi
echo ""

# Find the latest log (newest first)
LOG=""
for f in /tmp/modal_full_os_v4.log /tmp/modal_full_os_v3.log /tmp/modal_full_os_v2.log /tmp/modal_full_os.log; do
    if [ -f "$f" ] && [ -s "$f" ]; then
        LOG="$f"
        break
    fi
done

if [ -z "$LOG" ]; then
    echo "  No log found"
    echo ""
    exit 0
fi

echo "  Log: $(basename $LOG)"
echo ""

python3 << PYEOF
import re, sys

scores = []
tasks = []
retries = 0
setup_fails = 0
with open("$LOG") as f:
    for line in f:
        line = line.strip()
        m = re.search(r'Score:\s*([\d.]+)', line)
        if m:
            scores.append(float(m.group(1)))
        if '  Task:' in line or line.startswith('Task:'):
            task_text = line.split('Task:', 1)[1].strip()[:55]
            tasks.append(task_text)
        if 'Attempt' in line and 'failed' in line:
            retries += 1
        if 'SETUP FAILED' in line:
            setup_fails += 1

if not scores:
    if tasks:
        print(f'  Waiting for first result...')
        print(f'  ⏳ 1. {tasks[0]} (in progress)')
    else:
        print('  No results yet — benchmark starting...')
    sys.exit(0)

passed = sum(1 for s in scores if s > 0)
total = 24
avg = sum(scores) / len(scores) * 100

print(f'  Completed: {len(scores)}/{total} ({len(scores)/total*100:.0f}%)')
print(f'  Passed:    {passed}/{len(scores)}')
print(f'  Score:     {avg:.1f}%')
if retries:
    print(f'  Retries:   {retries} (self-verification recoveries attempted)')
if setup_fails:
    print(f'  Setup fails: {setup_fails}')
print()
print(f'  ─── Results ────────────────────────────────────')
for i in range(len(scores)):
    t = tasks[i] if i < len(tasks) else '?'
    mark = '✓' if scores[i] > 0 else '✗'
    print(f'  {mark} {i+1:2d}. {t}')
if len(tasks) > len(scores):
    print(f'  ⏳ {len(scores)+1:2d}. {tasks[len(scores)]} (in progress)')
print(f'  ────────────────────────────────────────────────')
remaining = total - len(scores)
if remaining <= 0:
    print(f'  ★ COMPLETE — Final: {avg:.1f}% ({passed}/{len(scores)})')
else:
    print(f'  {remaining} tasks remaining (~{remaining*3} min)')
PYEOF

echo ""
