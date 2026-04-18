#!/usr/bin/env bash
# Monitor a running BoatTrader extraction on Modal.
# Usage: ./monitor_run.sh [session_pattern]
#   e.g. ./monitor_run.sh boattrader
#        ./monitor_run.sh           (auto-detects latest)

set -euo pipefail

PATTERN="${1:-boattrader}"
POLL_INTERVAL=30  # seconds between checks

echo "=== Mantis CUA Monitor ==="
echo "  Pattern: $PATTERN"
echo "  Poll:    every ${POLL_INTERVAL}s"
echo "  Press Ctrl+C to stop"
echo ""

prev_viable=0
prev_total=0
start_time=$(date +%s)

while true; do
    # Find latest matching result file
    latest=$(uv run modal volume ls osworld-data results/ 2>/dev/null \
        | grep -i "$PATTERN" \
        | grep "holo3_results\|claude_results" \
        | head -1 \
        | sed 's/^[[:space:]]*//' \
        | awk '{print $NF}')

    if [ -z "$latest" ]; then
        echo "[$(date +%H:%M:%S)] No results yet for '$PATTERN'..."
        sleep "$POLL_INTERVAL"
        continue
    fi

    # Download latest
    uv run modal volume get osworld-data "$latest" /tmp/_monitor_results.json 2>/dev/null

    # Parse and display
    result=$(python3 -c "
import json, sys
try:
    d = json.load(open('/tmp/_monitor_results.json'))
    elapsed = d.get('total_time_s', 0)
    mins = elapsed // 60

    for td in d.get('task_details', []):
        if 'data' not in td:
            status = 'PASS' if td.get('success') else 'FAIL'
            print(f'  {td[\"task_id\"]}: {status} ({td.get(\"steps\",\"?\")} steps)')
            continue

        viable = td.get('viable', 0)
        total = td.get('iterations', 0)
        real = td.get('real_iterations', total)
        pf = td.get('parse_failures', 0)
        pct = viable / max(total, 1) * 100

        print(f'  {td[\"task_id\"]}: {viable}/{total} viable ({pct:.0f}%) | {real} real | {pf} parse failures')
        print(f'  Time: {mins}m | Status: {td.get(\"termination_reason\", \"running\")}')
        print()

        # Show last 5 data entries
        data = td.get('data', [])
        start = max(0, len(data) - 5)
        for i, entry in enumerate(data[start:], start + 1):
            short = entry[:150].replace(chr(10), ' ')
            # Mark viable vs skip
            tag = 'OK' if any(k in entry for k in ['VIABLE', 'Year:']) else '--'
            print(f'    [{i}] {tag} {short}')

except Exception as e:
    print(f'  Parse error: {e}')
" 2>&1)

    # Extract current viable count for delta detection
    cur_viable=$(echo "$result" | grep -oP '\d+(?=/\d+ viable)' | head -1 || echo "0")
    cur_total=$(echo "$result" | grep -oP '(?<=viable \()\d+' | head -1 || echo "0")

    now=$(date +%s)
    runtime=$(( (now - start_time) / 60 ))

    # Only print if something changed or every 5th poll
    if [ "$cur_viable" != "$prev_viable" ] || [ "$cur_total" != "$prev_total" ] || [ $((runtime % 3)) -eq 0 ]; then
        echo ""
        echo "[$(date +%H:%M:%S)] === $latest ==="
        echo "$result"
        prev_viable="$cur_viable"
        prev_total="$cur_total"
    else
        printf "."
    fi

    sleep "$POLL_INTERVAL"
done
