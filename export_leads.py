#!/usr/bin/env python3
"""Export extracted boat listings to CSV.

Downloads latest results from Modal volume and exports all viable
listings as a CSV file.

Usage:
    python export_leads.py                    # Export latest run
    python export_leads.py --all              # Export all runs
    python export_leads.py --output leads.csv # Custom output file
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys


def download_results(pattern=""):
    """Download result files from Modal volume."""
    result = subprocess.run(
        [".venv/bin/modal", "volume", "ls", "osworld-data", "results/"],
        capture_output=True, text=True,
    )
    files = [line.strip() for line in result.stdout.split("\n") if line.strip()]

    if pattern:
        files = [f for f in files if pattern in f]

    downloaded = []
    for f in files[:20]:  # Limit to 20 most recent
        local = f"/tmp/export_{os.path.basename(f)}"
        subprocess.run(
            [".venv/bin/modal", "volume", "get", "osworld-data", f, local],
            capture_output=True,
        )
        if os.path.exists(local):
            downloaded.append(local)

    return downloaded


def extract_boat_info(text):
    """Parse boat data from extraction text."""
    info = {
        "year": "",
        "make": "",
        "model": "",
        "price": "",
        "phone": "",
        "seller": "",
        "url": "",
        "raw": text[:200],
    }

    # Year
    yr = re.search(r"(?:19|20)\d{2}", text)
    if yr:
        info["year"] = yr.group()

    # Price
    pr = re.search(r"\$[\d,]+", text)
    if pr:
        info["price"] = pr.group()

    # Phone
    ph = re.findall(r"\(?\d{3}\)?[\s\-\.]\d{3}[\s\-\.]\d{4}", text)
    real = [p for p in ph if re.sub(r"\D", "", p)[3:6] != "555"]
    if real:
        info["phone"] = real[0]

    # Boat names from text
    boat_patterns = [
        r"(\d{4}\s+[\w\-]+\s+[\w\-]+(?:\s+[\w\-]+)?)",  # "2018 Sea Ray 350 SLX"
        r"(\d{4}\s+[\w\-]+\s+\d+\w*)",  # "2025 Tracker 2072"
    ]
    for pat in boat_patterns:
        m = re.search(pat, text)
        if m:
            parts = m.group(1).split()
            if len(parts) >= 3:
                info["year"] = parts[0]
                info["make"] = parts[1]
                info["model"] = " ".join(parts[2:])
            break

    # URL
    url = re.search(r"boattrader\.com/boat[s]?/[\w\-/]+", text)
    if url:
        info["url"] = "https://www." + url.group()

    return info


def export_to_csv(results_files, output_path):
    """Export all viable listings from results files to CSV."""
    rows = []

    for f in results_files:
        try:
            d = json.load(open(f))
        except (json.JSONDecodeError, FileNotFoundError):
            continue

        model = d.get("model", "?")
        run_id = d.get("run_id", "?")
        session = d.get("session_name", "?")

        for t in d.get("task_details", []):
            for item in t.get("data", []):
                text = str(item)
                if len(text) < 10:
                    continue

                info = extract_boat_info(text)
                info["ai_model"] = model  # AI model, not boat model
                info["run_id"] = run_id
                info["session"] = session
                info["status"] = "VIABLE" if info["year"] else "SKIP"
                rows.append(info)

    # Write CSV
    fieldnames = ["status", "year", "make", "model", "price", "phone", "seller", "url", "ai_model", "run_id", "session", "raw"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            # Rename 'model' key to avoid collision with boat model
            row_out = dict(row)
            row_out["model"] = row.get("model", "")  # boat model, not AI model
            writer.writerow(row_out)

    viable = sum(1 for r in rows if r["status"] == "VIABLE")
    print(f"Exported {len(rows)} listings ({viable} viable) to {output_path}")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Export boat listings to CSV")
    parser.add_argument("--output", default="leads.csv", help="Output CSV file")
    parser.add_argument("--all", action="store_true", help="Export all runs (default: latest)")
    parser.add_argument("--pattern", default="holo3", help="Filter results by pattern")
    args = parser.parse_args()

    print("Downloading results from Modal volume...")
    files = download_results(args.pattern)
    print(f"Downloaded {len(files)} result files")

    rows = export_to_csv(files, args.output)

    # Print summary
    if rows:
        viable = [r for r in rows if r["status"] == "VIABLE"]
        print(f"\n=== LEADS SUMMARY ===")
        for i, r in enumerate(viable, 1):
            print(f"  {i}. {r['year']} {r['make']} {r['model']} | {r['price']} | ph={r['phone'] or 'none'}")


if __name__ == "__main__":
    main()
