"""Opus Planner — convert plain text plans to CUA-aware task suites.

Uses Claude Opus to analyze a plain text plan and generate a rich,
structured task suite with visual anchors, error handlers, and
CUA-specific instructions that smaller models can execute reliably.

Architecture:
    User plain text plan
      → Opus (API, text-only, no browsing)
        → Rich task suite JSON with:
          - Visual descriptions of what to click
          - Error handlers per step (404, gallery, off-site)
          - Negative examples (what NOT to click)
          - Action format hints (use tool-call syntax)
      → Gemma4/Holo3 executes the plan via xdotool

Usage:
    from mantis_agent.opus_planner import plan_with_opus
    task_suite = plan_with_opus("plans/boattrader/extract_only.txt")
    # Returns JSON task suite ready for modal_cua_server.py
"""

from __future__ import annotations

import json
import logging
import os

import requests

logger = logging.getLogger(__name__)

PLANNER_SYSTEM = """\
You are a CUA (Computer Use Agent) plan architect. You convert plain text task descriptions \
into structured execution plans for a vision model that controls a real browser via screenshots + mouse/keyboard.

The executing model (Gemma4 or Holo3):
- Sees ONLY screenshots (1280x720 pixels) — no DOM, no HTML, no element IDs
- Outputs actions: click(x, y), scroll(direction, amount), type_text(text), key_press(keys), done(success, summary)
- Has NO memory between iterations — each listing extraction is independent
- Gets confused by: social media icons, photo galleries, ad banners, dealer links, popups

Your job: generate a task suite JSON that gives the executing model SPECIFIC, VISUAL instructions \
so it clicks the right things and avoids traps.

CRITICAL RULES for your output:
1. Describe UI elements by VISUAL APPEARANCE, not by CSS/HTML (the model can't see code)
2. Include NEGATIVE examples — what NOT to click, described visually
3. Include ERROR HANDLERS — what to do when 404, gallery trap, off-site, cookies appear
4. Use the executor's ACTION SYNTAX in instructions: click(), scroll(), key_press(), done()
5. Each looped task must be SELF-CONTAINED — the model has no memory of prior iterations
6. Be SPECIFIC about coordinates: "center of page" (x~640), "top area" (y<100), "bottom area" (y>600)
"""

PLANNER_PROMPT = """\
Convert this plain text plan into a CUA task suite JSON.

PLAIN TEXT PLAN:
{plan_text}

Generate a JSON task suite with this structure:
{{
  "session_name": "<short_name>",
  "base_url": "<starting_url>",
  "tasks": [
    {{
      "task_id": "<unique_id>",
      "intent": "<DETAILED CUA instructions with visual anchors, negative examples, error handlers>",
      "loop": {{  // only if this task repeats
        "pagination_intent": "<how to go to next page>",
        "max_iterations": <number>,
        "max_pages": <number>,
        "max_steps_per_iteration": <number>
      }},
      "start_url": "<url>"
    }}
  ]
}}

For the "intent" field, include ALL of these sections:

WHAT TO CLICK — describe the visual appearance of the target element:
- Size (LARGE rectangle, SMALL icon)
- Position (CENTER of page, FOOTER, HEADER, SIDEBAR)
- Content (shows boat photo + Year/Make/Model + Price)
- Which part to click (TITLE TEXT, not the photo)

WHAT TO NEVER CLICK — describe visually:
- Social media icons (SMALL colored squares 20-40px in footer/sidebar)
- Navigation menus, dropdown headers
- Ad banners, dealer logos
- Photo images (open gallery traps)

STEPS — use the executor's action syntax:
- click() for mouse clicks
- scroll(direction="down", amount=5) for scrolling
- key_press(keys="alt+left") for keyboard
- done(success=true, summary="...") for completion

ERROR HANDLERS — what to do for each error state:
- 404/Page Not Found → key_press(keys="alt+left"), done(summary="SKIPPED | 404")
- Photo gallery ("1 of N") → key_press(keys="Escape"), key_press(keys="alt+left")
- Off-site (facebook/instagram) → key_press(keys="alt+left") immediately
- Cookie popup → click Accept ONCE, max 2 steps

OUTPUT FORMAT — exact format for done() calls

Return ONLY the JSON, no markdown fences, no explanation.
"""


def plan_with_opus(
    plan_path: str,
    api_key: str = "",
    model: str = "claude-sonnet-4-20250514",
    output_path: str = "",
    force: bool = False,
) -> dict:
    """Convert a plain text plan to a CUA-aware task suite using Opus.

    Args:
        plan_path: Path to plain text plan file.
        api_key: Anthropic API key (falls back to ANTHROPIC_API_KEY env var).
        model: Claude model to use for planning.
        output_path: Optional path to save the generated JSON.

    Returns:
        Task suite dict ready for modal_cua_server.py
    """
    import hashlib

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    # Cache check
    if not force and output_path and os.path.exists(output_path):
        try:
            cached = json.loads(open(output_path).read())
            plan_hash = hashlib.md5(open(plan_path, "rb").read()).hexdigest()[:8]
            if cached.get("_plan_hash") == plan_hash:
                logger.info(f"Using cached plan: {output_path}")
                return cached
        except (json.JSONDecodeError, KeyError):
            pass

    with open(plan_path) as f:
        plan_text = f.read()

    logger.info(f"Planning with {model}: {plan_path}")

    prompt = PLANNER_PROMPT.format(plan_text=plan_text)

    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 4096,
            "system": PLANNER_SYSTEM,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"Opus API error: {resp.status_code} {resp.text[:200]}")

    data = resp.json()
    text = ""
    for block in data.get("content", []):
        if block.get("type") == "text":
            text = block["text"]
            break

    # Parse JSON from response (strip any markdown fences)
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()

    task_suite = json.loads(text)

    # Tag for cache
    plan_hash = hashlib.md5(open(plan_path, "rb").read()).hexdigest()[:8]
    task_suite["_plan_hash"] = plan_hash
    task_suite["_generated_by"] = f"opus_planner:plan_with_opus:{model}"

    # Log stats
    tasks = task_suite.get("tasks", [])
    loop_tasks = sum(1 for t in tasks if t.get("loop"))
    logger.info(f"Generated {len(tasks)} tasks ({loop_tasks} with loops)")

    tokens = data.get("usage", {})
    cost = (tokens.get("input_tokens", 0) * 3 + tokens.get("output_tokens", 0) * 15) / 1_000_000
    task_suite["_cost_usd"] = round(cost, 4)
    logger.info(f"Opus tokens: {tokens.get('input_tokens', 0)} in + {tokens.get('output_tokens', 0)} out = ~${cost:.3f}")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(task_suite, f, indent=2)
        logger.info(f"Saved to {output_path} (hash={plan_hash})")

    return task_suite


BROWSE_PROMPT = """\
I'm showing you screenshots from a website at 1280x720 resolution. Analyze the layout with PRECISE PIXEL COORDINATES.

For EACH interactive element, report its bounding box as (x_start, y_start, x_end, y_end).

1. LISTING CARDS — For each card visible:
   - Overall card bounds: (x1,y1) to (x2,y2)
   - PHOTO AREA bounds: (x1,y1) to (x2,y_photo_end) — DO NOT CLICK HERE
   - TITLE TEXT bounds: (x1,y_title_start) to (x2,y_title_end) — CLICK HERE
   - PRICE TEXT bounds: approximate y range
   - "Contact Seller" or "View Details" button bounds if visible

2. DANGEROUS ZONES — Elements to NEVER click with their pixel bounds:
   - Social media icons (exact y range in footer)
   - Ad banners (exact position)
   - Navigation dropdown menus (exact position)
   - Dealer logo areas
   - Any element that opens a photo gallery

3. FILTERS SIDEBAR:
   - Zip/location input: approximate (x, y) center
   - Price filter: approximate (x, y) center
   - Seller type: approximate (x, y) center
   - Sort dropdown: approximate (x, y) center

4. PAGINATION: Where are page navigation controls? Exact y position.

5. DETAIL PAGE (if screenshot shows one):
   - Where does the photo gallery end and description text begin? Exact y position.
   - Where is the "Contact Seller" / phone number area?

Be EXTREMELY specific with pixel coordinates. The executing model can ONLY click at (x,y) coordinates — it cannot read element IDs or CSS.
"""


def browse_and_plan(
    plan_path: str,
    screenshot_dir: str,
    api_key: str = "",
    model: str = "claude-sonnet-4-20250514",
    output_path: str = "",
    max_screenshots: int = 5,
    force: bool = False,
) -> dict:
    """Browse-enhanced planning: analyze cached screenshots + plain text plan.

    Phase 1: Opus analyzes screenshots from prior runs to understand site layout
    Phase 2: Opus generates a CUA-aware task suite with site-specific visual hints

    This is a ONE-TIME cost (~$0.05) that produces a reusable plan.
    Plans are cached — if output_path exists and plan_path hasn't changed,
    returns the cached plan without calling Opus.
    Opus never enters the extraction loop — only Gemma4 executes.

    Args:
        plan_path: Path to plain text plan file.
        screenshot_dir: Directory with cached screenshots (from prior runs).
        api_key: Anthropic API key.
        model: Claude model for planning.
        output_path: Optional path to save the generated JSON.
        max_screenshots: Max screenshots to send for analysis.
        force: If True, regenerate even if cached plan exists.

    Returns:
        Task suite dict with site-specific visual hints.
    """
    import base64
    import hashlib
    from io import BytesIO
    from pathlib import Path

    from PIL import Image

    # Cache check: if output exists and plan text hasn't changed, use cached
    if not force and output_path and os.path.exists(output_path):
        try:
            cached = json.loads(open(output_path).read())
            plan_hash = hashlib.md5(open(plan_path, "rb").read()).hexdigest()[:8]
            if cached.get("_plan_hash") == plan_hash:
                logger.info(f"Using cached plan: {output_path} (hash={plan_hash})")
                return cached
            logger.info(f"Plan text changed (hash mismatch), regenerating...")
        except (json.JSONDecodeError, KeyError):
            pass

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    with open(plan_path) as f:
        plan_text = f.read()

    # Load screenshots
    screenshot_dir = Path(screenshot_dir)
    png_files = sorted(screenshot_dir.glob("*.png"))[:max_screenshots]

    if not png_files:
        logger.warning(f"No screenshots in {screenshot_dir}, falling back to text-only planning")
        return plan_with_opus(plan_path, api_key=api_key, model=model, output_path=output_path)

    logger.info(f"Browse-enhanced planning: {len(png_files)} screenshots from {screenshot_dir}")

    # Phase 1: Analyze screenshots to understand site layout
    browse_content = [{"type": "text", "text": BROWSE_PROMPT}]
    for f in png_files:
        img = Image.open(f)
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        browse_content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": b64},
        })
        browse_content.append({"type": "text", "text": f"[Screenshot: {f.name}]"})

    logger.info("Phase 1: Analyzing site layout from screenshots...")
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 2000,
            "messages": [{"role": "user", "content": browse_content}],
        },
        timeout=60,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"Browse API error: {resp.status_code} {resp.text[:500]}")

    site_analysis = ""
    for block in resp.json().get("content", []):
        if block.get("type") == "text":
            site_analysis = block["text"]
            break

    tokens1 = resp.json().get("usage", {})
    cost1 = (tokens1.get("input_tokens", 0) * 3 + tokens1.get("output_tokens", 0) * 15) / 1_000_000
    logger.info(f"Phase 1 cost: ~${cost1:.3f} ({tokens1.get('input_tokens', 0)} in)")
    logger.info(f"Site analysis: {site_analysis[:200]}...")

    # Phase 2: Generate plan with site-specific knowledge
    enhanced_prompt = PLANNER_PROMPT.format(plan_text=plan_text)
    enhanced_prompt += f"\n\nSITE-SPECIFIC VISUAL ANALYSIS (from real screenshots):\n{site_analysis}"

    logger.info("Phase 2: Generating CUA-aware plan with visual hints...")
    resp2 = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 4096,
            "system": PLANNER_SYSTEM,
            "messages": [{"role": "user", "content": enhanced_prompt}],
        },
        timeout=60,
    )

    if resp2.status_code != 200:
        raise RuntimeError(f"Plan API error: {resp2.status_code} {resp2.text[:200]}")

    text = ""
    for block in resp2.json().get("content", []):
        if block.get("type") == "text":
            text = block["text"]
            break

    # Parse JSON
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()

    task_suite = json.loads(text)

    tokens2 = resp2.json().get("usage", {})
    cost2 = (tokens2.get("input_tokens", 0) * 3 + tokens2.get("output_tokens", 0) * 15) / 1_000_000
    total_cost = cost1 + cost2
    logger.info(f"Phase 2 cost: ~${cost2:.3f}")
    logger.info(f"Total browse+plan cost: ~${total_cost:.3f}")

    tasks = task_suite.get("tasks", [])
    loop_tasks = sum(1 for t in tasks if t.get("loop"))
    logger.info(f"Generated {len(tasks)} tasks ({loop_tasks} with loops)")

    # Tag with plan hash for cache invalidation
    plan_hash = hashlib.md5(open(plan_path, "rb").read()).hexdigest()[:8]
    task_suite["_plan_hash"] = plan_hash
    task_suite["_generated_by"] = f"opus_planner:browse_and_plan:{model}"
    task_suite["_cost_usd"] = round(total_cost, 4)

    if output_path:
        with open(output_path, "w") as f:
            json.dump(task_suite, f, indent=2)
        logger.info(f"Saved to {output_path} (hash={plan_hash})")

    return task_suite


def main():
    """CLI: python -m mantis_agent.opus_planner <plan_file> [output_file] [--browse screenshots_dir]"""
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m mantis_agent.opus_planner <plan_file> [output_file]")
        print("  python -m mantis_agent.opus_planner <plan_file> [output_file] --browse <screenshots_dir>")
        sys.exit(1)

    plan_path = sys.argv[1]
    output_path = ""
    browse_dir = ""

    args = sys.argv[2:]
    while args:
        if args[0] == "--browse" and len(args) > 1:
            browse_dir = args[1]
            args = args[2:]
        elif not output_path:
            output_path = args[0]
            args = args[1:]
        else:
            args = args[1:]

    if not output_path:
        output_path = plan_path.replace(".txt", "_opus.json")

    if browse_dir:
        task_suite = browse_and_plan(plan_path, browse_dir, output_path=output_path)
    else:
        task_suite = plan_with_opus(plan_path, output_path=output_path)

    print(json.dumps(task_suite, indent=2))


if __name__ == "__main__":
    main()
