#!/bin/bash
# Mantis — BoatTrader comprehensive monitoring
# Usage: ./monitor_boattrader.sh
#        ./monitor_boattrader.sh --watch   (refresh every 60s)

cd /Users/barada/Sandbox/Mason/cua-agent

WATCH=false
[ "$1" = "--watch" ] && WATCH=true

run_check() {
    clear 2>/dev/null || true
    echo ""
    echo "  ╔═══════════════════════════════════════════════════════════════╗"
    echo "  ║  Mantis — BoatTrader Lead Extraction Monitor                 ║"
    echo "  ║  $(date '+%Y-%m-%d %H:%M:%S')                                        ║"
    echo "  ╚═══════════════════════════════════════════════════════════════╝"
    echo ""

    # 1. Active Modal apps
    echo "  ── ACTIVE WORKERS ──────────────────────────────────────────────"
    APPS=$(.venv/bin/modal app list 2>/dev/null | grep ephemeral)
    if [ -z "$APPS" ]; then
        echo "  No active runs."
    else
        echo "$APPS" | while read line; do
            tasks=$(echo "$line" | sed 's/.*│[[:space:]]*\([0-9]*\)[[:space:]]*│.*/\1/')
            echo "  Active: ${tasks} worker(s) running"
        done
    fi
    echo ""

    # 2. Download latest results
    rm -f /tmp/mon_bt_*.json 2>/dev/null
    for f in $(.venv/bin/modal volume ls osworld-data results/ 2>/dev/null | head -20); do
        base=$(basename "$f" .json)
        .venv/bin/modal volume get osworld-data "$f" "/tmp/mon_bt_${base}.json" 2>/dev/null
    done

    # 3. Parse and display
    python3 -c "
import json, re, glob, os, sys
from datetime import datetime

def extract_phone(text):
    for m in re.finditer(r'\(?\d{3}\)?[\s\-\.]?\d{3}[\s\-\.]?\d{4}', text):
        ph = m.group(); digits = re.sub(r'\D', '', ph)
        if len(digits) < 10 or len(digits) > 11 or digits[3:6] == '555': continue
        ctx = text[max(0,m.start()-20):m.start()].lower()
        if any(c in ctx for c in ['http','://','url=','&','?']): continue
        return ph
    return None

# Group results by run timestamp (last 2 hours only)
files = sorted(glob.glob('/tmp/mon_bt_*.json'), key=os.path.getmtime, reverse=True)

# Separate page workers from sequential runs
page_runs = {}  # {run_ts: {page: data}}
seq_runs = {}   # {run_ts: data}

for f in files:
    try:
        d = json.load(open(f))
        rid = d['run_id']
        model = d['model']
        session = d.get('session_name', '')

        if 'worker' in session and 'page' in session:
            # Extract page number from session name
            page_match = re.search(r'page_(\d+)', session)
            if page_match:
                page = int(page_match.group(1))
                # Group by approximate run time (within 5 min = same batch)
                ts = rid[:13]  # YYYYMMDD_HHMM
                if ts not in page_runs:
                    page_runs[ts] = {'model': model, 'pages': {}, 'rid': rid}
                page_runs[ts]['pages'][page] = d
        else:
            ts = rid[:13]
            if ts not in seq_runs:
                seq_runs[ts] = d
    except:
        pass

# Display parallel runs
print('  ── PARALLEL RUNS (page workers) ─────────────────────────────────')
all_leads = []
total_scanned_all = 0
total_viable_all = 0
total_cost_all = 0.0

for ts in sorted(page_runs.keys(), reverse=True)[:2]:  # Last 2 batches
    run = page_runs[ts]
    model = run['model']
    pages = run['pages']
    print(f'')
    print(f'  {model} — batch {ts}')
    print(f'  {\"Page\":>6s} {\"Viable\":>8s} {\"Scanned\":>8s} {\"Hit%\":>6s} {\"GPU\":>6s} {\"Cost\":>7s} {\"Status\":>8s}  Issues')
    print(f'  {\"─\"*6} {\"─\"*8} {\"─\"*8} {\"─\"*6} {\"─\"*6} {\"─\"*7} {\"─\"*8}  {\"─\"*20}')

    batch_v, batch_s, batch_cost, batch_gpu = 0, 0, 0.0, 0
    for p in range(1, 6):
        if p not in pages:
            print(f'  {p:>6d} {\"—\":>8s} {\"—\":>8s} {\"—\":>6s} {\"—\":>6s} {\"—\":>7s} {\"no data\":>8s}')
            continue
        d = pages[p]
        gpu = d['total_gpu_time_s']
        cost = d['estimated_cost_usd']
        completed = 'DONE' if d.get('completed_at') else 'LIVE'
        batch_cost += cost
        batch_gpu += gpu

        for t in d['task_details']:
            iters = t.get('iterations', 0) or 0
            viable = t.get('viable', 0) or 0
            batch_v += viable
            batch_s += iters
            hit = f'{viable/iters*100:.0f}%' if iters else '—'

            # Detect issues
            issues = []
            for item in t.get('data', []):
                text = str(item).lower()
                if 'err_tunnel' in text: issues.append('PROXY')
                elif 'facebook' in text or 'instagram' in text: issues.append('OFF-SITE')
                elif 'page not found' in text or '404' in text: issues.append('404')
                ph = extract_phone(str(item))
                if ph: all_leads.append({'page': p, 'phone': ph, 'raw': str(item)[:120]})

            issue_str = ', '.join(set(issues))[:20] if issues else ''
            print(f'  {p:>6d} {viable:>8d} {iters:>8d} {hit:>6s} {gpu//60:>5d}m \${cost:>6.2f} {completed:>8s}  {issue_str}')

    total_scanned_all += batch_s
    total_viable_all += batch_v
    total_cost_all += batch_cost
    hit = f'{batch_v/batch_s*100:.0f}%' if batch_s else '—'
    print(f'  {\"TOTAL\":>6s} {batch_v:>8d} {batch_s:>8d} {hit:>6s} {batch_gpu//60:>5d}m \${batch_cost:>6.2f}')

# Display sequential runs
print(f'')
print('  ── SEQUENTIAL RUNS ──────────────────────────────────────────────')
for ts in sorted(seq_runs.keys(), reverse=True)[:3]:
    d = seq_runs[ts]
    gpu = d['total_gpu_time_s']
    cost = d['estimated_cost_usd']
    completed = 'DONE' if d.get('completed_at') else 'LIVE'
    model = d['model']
    print(f'')
    print(f'  {model} | {completed} | GPU:{gpu//60}min | \${cost:.2f}')
    for t in d['task_details']:
        iters = t.get('iterations', 0) or 0
        viable = t.get('viable', 0) or 0
        pf = t.get('parse_failures', 0) or 0
        reason = t.get('termination_reason', str(t.get('error',''))[:40])
        status = '✓' if t.get('success') else '✗'
        extra = f' | {viable}/{iters} viable' if iters else ''
        if pf: extra += f' | {pf}pf'
        print(f'    {status} {t[\"task_id\"]:25s} {reason}{extra}')

        if 'extract' in t.get('task_id', ''):
            total_scanned_all += iters
            total_viable_all += viable
            total_cost_all += cost
            for item in t.get('data', [])[:5]:
                ph = extract_phone(str(item))
                if ph: all_leads.append({'page': 0, 'phone': ph, 'raw': str(item)[:120]})

# Deduplicated leads
print(f'')
print('  ── LEADS FOUND ──────────────────────────────────────────────────')
seen_digits = set()
unique_leads = []
for l in all_leads:
    digits = re.sub(r'\D', '', l['phone'])
    if digits not in seen_digits:
        seen_digits.add(digits)
        unique_leads.append(l)

if unique_leads:
    for i, l in enumerate(unique_leads, 1):
        src = f'page {l[\"page\"]}' if l['page'] else 'sequential'
        print(f'  {i}. {l[\"phone\"]:15s} ({src})  {l[\"raw\"]}')
else:
    print('  No phone leads extracted yet.')

# Summary
print(f'')
print('  ── SUMMARY ──────────────────────────────────────────────────────')
print(f'  Total scanned:     {total_scanned_all}')
print(f'  Viable (phone):    {total_viable_all}')
hit = f'{total_viable_all/total_scanned_all*100:.0f}%' if total_scanned_all else '—'
print(f'  Hit rate:          {hit}')
print(f'  Unique leads:      {len(unique_leads)}')
cpl = f'\${total_cost_all/len(unique_leads):.2f}' if unique_leads else '—'
print(f'  Cost/lead:         {cpl}')
print(f'  Total GPU cost:    \${total_cost_all:.2f}')
print(f'  Target:            125 listings (5 pages × 25), phone in ~20%')
print()
"
}

if $WATCH; then
    while true; do
        run_check
        echo "  Refreshing in 60s... (Ctrl+C to stop)"
        sleep 60
    done
else
    run_check
fi
