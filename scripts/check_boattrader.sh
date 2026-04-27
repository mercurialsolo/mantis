#!/bin/bash
# Mantis — BoatTrader extraction progress checker
# Usage: ./check_boattrader.sh [model]
#   model: evocua-8b (default), evocua-32b, gemma4-cua, claude

cd /Users/barada/Sandbox/Mason/cua-agent

MODEL="${1:-all}"

echo ""
echo "  ╔═══════════════════════════════════════════════╗"
echo "  ║  Mantis — BoatTrader Lead Extraction          ║"
echo "  ╚═══════════════════════════════════════════════╝"
echo ""

# Find latest results on Modal volume
rm -f /tmp/mantis_bt_*.json 2>/dev/null

if [ "$MODEL" = "all" ] || [ "$MODEL" = "evocua-8b" ]; then
    LATEST=$(.venv/bin/modal volume ls osworld-data results/ 2>/dev/null | grep "cua_results_bt_dynamic" | head -1)
    if [ -n "$LATEST" ]; then
        .venv/bin/modal volume get osworld-data "$LATEST" /tmp/mantis_bt_evocua.json 2>/dev/null
    fi
fi

if [ "$MODEL" = "all" ] || [ "$MODEL" = "gemma4-cua" ]; then
    LATEST=$(.venv/bin/modal volume ls osworld-data results/ 2>/dev/null | grep "gemma4cua_results_bt_dynamic" | head -1)
    if [ -n "$LATEST" ]; then
        .venv/bin/modal volume get osworld-data "$LATEST" /tmp/mantis_bt_gemma4.json 2>/dev/null
    fi
fi

if [ "$MODEL" = "all" ] || [ "$MODEL" = "claude" ]; then
    LATEST=$(.venv/bin/modal volume ls osworld-data results/ 2>/dev/null | grep "claude_results_bt_dynamic" | head -1)
    if [ -n "$LATEST" ]; then
        .venv/bin/modal volume get osworld-data "$LATEST" /tmp/mantis_bt_claude.json 2>/dev/null
    fi
fi

python3 -c "
import json, re, glob, os

files = sorted(glob.glob('/tmp/mantis_bt_*.json'), key=os.path.getmtime, reverse=True)
if not files:
    print('  No results found on Modal volume.')
    print('  Run: uv run modal run --detach modal_cua_server.py --task-file tasks/boattrader/dynamic_production.json --model evocua-8b --max-steps 40')
    exit(0)

def is_real_phone(phone):
    digits = re.sub(r'\D', '', phone)
    return len(digits) >= 10 and digits[3:6] != '555'

for f in files:
    try:
        d = json.load(open(f))
        gpu = d['total_gpu_time_s']
        completed = 'DONE' if d.get('completed_at') else 'RUNNING'
        cost = d['estimated_cost_usd']
        model = d['model']
        run_id = d['run_id']

        print(f'  ┌─ {model} ({completed}) ─ run {run_id}')
        print(f'  │  GPU: {gpu//60}min | Cost: \${cost}')

        total_scanned = 0
        total_viable = 0
        total_parse_fails = 0
        leads = []

        for t in d['task_details']:
            tid = t['task_id']
            success = t.get('success', False)
            iters = t.get('iterations', 0) or 0
            viable = t.get('viable', 0) or 0
            pf = t.get('parse_failures', 0) or 0
            reason = t.get('termination_reason', str(t.get('error',''))[:50])
            status = '✓' if success else '✗'
            extra = f' | {viable}/{iters} viable' if iters else ''
            if pf: extra += f' | {pf} parse_fails'
            print(f'  │  {status} {tid:25s} {reason}{extra}')

            if 'extract' in tid:
                total_scanned = iters
                total_viable = viable
                total_parse_fails = pf
                for item in t.get('data', []):
                    text = str(item)
                    phones = re.findall(r'\(?\d{3}\)?[\s\-\.]?\d{3}[\s\-\.]?\d{4}', text)
                    real = [p for p in phones if is_real_phone(p)]
                    prices = re.findall(r'\\\$[\d,]+', text)
                    if real:
                        leads.append({'phone': real[0], 'price': prices[0] if prices else '?', 'raw': text[:120]})

        hit = f'{total_viable/max(total_scanned,1)*100:.0f}%' if total_scanned else 'N/A'
        cpl = f'\${cost/max(total_viable,1):.2f}' if total_viable else 'N/A'
        speed = f'{(gpu/60)/max(total_scanned,1):.1f}' if total_scanned else 'N/A'

        print(f'  │')
        print(f'  │  Scanned: {total_scanned} | Viable: {total_viable} | Hit: {hit} | \$/lead: {cpl} | {speed} min/listing')

        if leads:
            print(f'  │')
            print(f'  │  LEADS:')
            for i, l in enumerate(leads, 1):
                print(f'  │    {i}. Phone: {l[\"phone\"]}  Price: {l[\"price\"]}')
                print(f'  │       {l[\"raw\"]}')

        print(f'  └─')
        print()

    except Exception as e:
        print(f'  Error reading {f}: {e}')
"
