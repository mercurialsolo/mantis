#!/usr/bin/env python3
"""Test extraction prompts against cached BoatTrader screenshots.

Quick iteration: change the prompt, run again, see if extraction improves.
No GPU, no browser, no proxy needed — just API key + cached screenshots.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python test_extraction_prompt.py

    # Or with a custom prompt:
    python test_extraction_prompt.py --prompt "your custom prompt"

    # Or test against a specific screenshot:
    python test_extraction_prompt.py --screenshot screenshots/bt_latest/0020.png
"""

import argparse
import base64
import json
import os
import sys
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

# The extraction prompt we're testing — edit this to iterate
DEFAULT_PROMPT = """\
You are on a BoatTrader boat listing page. Look at the screenshot carefully.

EXTRACT these fields:
1. Year, Make, Model — from the title/header area
2. Price — dollar amount shown on the page
3. Phone number — look in ALL of these locations:
   - "Contact Seller" section (right side or below photos)
   - Phone icon with a number next to it
   - Description text section (scroll area below photos)
   - "Seller Notes" or "More Details" sections
   - Any clickable phone link
4. Seller name — from the Contact section
5. URL — from the browser address bar

Phone formats: (305) 555-1234, 786-555-1234, 305.555.1234, +17865551234

OUTPUT — use exactly one of these formats:

If phone found:
VIABLE | Year: 2022 | Make: Century | Model: 3100 | Price: $89500 | Phone: 786-386-1420 | Seller: Melissa Gonzalez

If NO phone visible anywhere:
SKIPPED | Year: 2022 | Make: Century | Model: 3100 | Price: $89500 | no phone visible
"""


def test_screenshot(screenshot_path: str, prompt: str, model: str = "claude-sonnet-4-20250514") -> str:
    """Send a screenshot + prompt to Claude and get the extraction result."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    img = Image.open(screenshot_path)
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2025-01-24",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 500,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        },
        timeout=30,
    )

    if resp.status_code != 200:
        return f"API error: {resp.status_code} {resp.text[:200]}"

    data = resp.json()
    for block in data.get("content", []):
        if block.get("type") == "text":
            return block["text"]
    return "No text response"


def main():
    parser = argparse.ArgumentParser(description="Test extraction prompts against cached screenshots")
    parser.add_argument("--screenshot", default="", help="Single screenshot to test")
    parser.add_argument("--dir", default="screenshots/bt_latest", help="Directory of screenshots")
    parser.add_argument("--prompt", default="", help="Custom prompt (default: built-in extraction prompt)")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Claude model")
    parser.add_argument("--max", type=int, default=5, help="Max screenshots to test from dir")
    args = parser.parse_args()

    prompt = args.prompt or DEFAULT_PROMPT

    if args.screenshot:
        files = [args.screenshot]
    else:
        files = sorted(Path(args.dir).glob("*.png"))[:args.max]

    print(f"Testing {len(files)} screenshot(s) with model {args.model}")
    print(f"Prompt: {prompt[:100]}...")
    print("=" * 60)

    for f in files:
        fname = str(f)
        print(f"\n--- {Path(fname).name} ---")
        result = test_screenshot(fname, prompt, args.model)
        print(result)

        # Check if we got a viable lead
        if "VIABLE" in result and "Phone:" in result:
            print("  >>> LEAD FOUND <<<")
        elif "SKIPPED" in result:
            print("  (no phone)")


if __name__ == "__main__":
    main()
