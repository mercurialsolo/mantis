"""Scan /v1/cua audit trajectories on the Modal volume → per-category verdict.

Companion to ``run_cua_audit.py``. The Claude executor saves per-run
trajectories to ``/data/results/claude_trajectories_<session>_<run>.jsonl`` on
the ``osworld-data`` volume. This reads them for one audit session and extracts
the CAPABILITY signals that wire status can't show — did the brain emit a
verified ``type_text``, complete a click chain, reach a ``done`` — and prints a
scorecard.

Usage::

    uv run modal run scripts/scan_cua_audit.py --session cua_audit_1782883000
"""

from __future__ import annotations

import json
import subprocess

import modal

app = modal.App("scan-cua-audit")
vol = modal.Volume.from_name("osworld-data")


@app.function(image=modal.Image.debian_slim(), volumes={"/data": vol}, timeout=180)
def scan(session: str) -> list[dict]:
    vol.reload()
    listing = subprocess.run(
        f"ls -t /data/results/claude_trajectories_{session}_*.jsonl 2>/dev/null",
        shell=True, capture_output=True, text=True,
    ).stdout.split()
    rows: list[dict] = []
    for path in listing:
        try:
            text = open(path).read()
        except OSError:
            continue
        for line in text.splitlines():
            d = json.loads(line)
            traj = d.get("trajectory", [])
            typed = landed = clicked = focused = done = False
            for s in traj:
                at = s.get("action_type")
                fb = (s.get("feedback") or "").lower()
                if at == "type_text" and str(s.get("action_params", {}).get("text") or ""):
                    typed = True
                    if "verified" in fb:
                        landed = True
                if at in ("click", "double_click"):
                    clicked = True
                if "field focused" in fb:
                    focused = True
                if at == "done":
                    done = True
            rows.append({
                "task_id": d.get("task_id"),
                "steps": d.get("steps"),
                "termination": d.get("termination_reason"),
                "success": d.get("success"),
                "focused_input": focused,
                "typed": typed,
                "type_landed_verified": landed,
                "clicked": clicked,
                "emitted_done": done,
            })
    return rows


@app.local_entrypoint()
def main(session: str = "") -> None:
    if not session:
        print("pass --session cua_audit_<ts>")
        return
    rows = scan.remote(session)
    if not rows:
        print(f"no trajectories found for session {session!r}")
        return
    print(f"\n=== CAPABILITY SCORECARD — session {session} ===")
    hdr = f"{'task':<24}{'steps':<6}{'focus':<6}{'typed':<6}{'landed':<7}{'click':<6}{'done':<5}{'termination'}"
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: str(x.get("task_id"))):
        print(
            f"{str(r['task_id']):<24}{str(r['steps']):<6}"
            f"{'Y' if r['focused_input'] else '-':<6}"
            f"{'Y' if r['typed'] else '-':<6}"
            f"{'Y' if r['type_landed_verified'] else '-':<7}"
            f"{'Y' if r['clicked'] else '-':<6}"
            f"{'Y' if r['emitted_done'] else '-':<5}"
            f"{r['termination']}"
        )
    # headline verdicts for the audit's broken categories
    by_id = {r["task_id"]: r for r in rows}
    print("\n=== VERDICT vs audit's 'broken' categories ===")
    st = by_id.get("search_type", {})
    wt = by_id.get("write_contenteditable", {})
    ms = by_id.get("multistep_nav", {})
    print(f"  #3 search type-landing : {'FIXED — text landed (verified)' if st.get('type_landed_verified') else 'still failing' if st else 'no data'}")
    print(f"  #4 contenteditable write: {'FIXED — text landed (verified)' if wt.get('type_landed_verified') else 'still failing' if wt else 'no data'}")
    print(f"  multi-step click+nav    : {'brain acted (clicked+done)' if ms.get('clicked') and ms.get('emitted_done') else 'partial/failing' if ms else 'no data'}")
