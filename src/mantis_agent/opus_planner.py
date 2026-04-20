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

KNOWN CUA MODEL LIMITATIONS (your plan MUST work around these):
1. SKIPS STEPS: The model often calls done() immediately without actually interacting with the page. \
   For filter/setup tasks, you MUST include explicit verification criteria and tell the model \
   "Do NOT call done() until you have actually clicked and typed in the filter controls."
2. PHOTO GALLERY TRAP: Clicking on large images opens fullscreen galleries ("1 of N"). \
   The model MUST click on text links, not images. Include explicit "NEVER click the photo" warnings.
3. ADDRESS BAR BLINDNESS: The model rarely reads the browser address bar unless explicitly told. \
   For data extraction tasks, ALWAYS instruct: "Read the URL from the browser address bar at the top \
   of the screen BEFORE scrolling. The URL is mandatory in your output."
4. CONTEXT ROT: After 8-10 iterations, model quality degrades. Each iteration must be self-contained.
5. OFF-SITE NAVIGATION: The model sometimes clicks social media icons (Facebook, Instagram) or \
   dealer links that navigate away from the target site. Include off-site recovery instructions.
6. TEMPLATE ECHOING: The model may output placeholder text like "Year: <from title>" instead of \
   actual values. Your output format examples must show realistic filled-in data, and WARN against echoing.

Your job: generate a task suite JSON that gives the executing model SPECIFIC, VISUAL instructions \
so it clicks the right things and avoids traps.

CRITICAL RULES for your output:
1. Describe UI elements by VISUAL APPEARANCE, not by CSS/HTML (the model can't see code)
2. Include NEGATIVE examples — what NOT to click, described visually
3. Include ERROR HANDLERS — what to do when 404, gallery trap, off-site, cookies appear
4. Use the executor's ACTION SYNTAX in instructions: click(), scroll(), key_press(), done()
5. Each looped task must be SELF-CONTAINED — the model has no memory of prior iterations
6. Be SPECIFIC about coordinates: "center of page" (x~640), "top area" (y<100), "bottom area" (y>600)
7. For FILTER/SETUP tasks: require verification of results (e.g. "result count should decrease") \
   and PROHIBIT calling done() without actual UI interaction
8. For EXTRACTION tasks: URL from the address bar is ALWAYS a required output field
9. MULTI-TASK CONTINUITY: The extraction task should set start_url to null so it \
   continues from where the setup task left off (with filters applied). \
   However, EVERY setup/filter task MUST end with a VERIFICATION step that checks:
   - The page heading/title still shows the expected filter (e.g. "by owner", "private")
   - The result count decreased from unfiltered
   - If verification fails, the model should RE-APPLY the most critical filter
   The setup intent MUST list filters in priority order (most important first) \
   and warn "Do NOT click navigation elements that change the page URL".
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

WHAT TO CLICK — describe by VISUAL APPEARANCE, not coordinates:
- What it LOOKS LIKE (e.g. "blue text link showing Year Make Model below the photo")
- Where it IS (e.g. "in the left sidebar, below the Location heading")
- NEVER use pixel coordinates like click(95, 510) — the layout shifts between sessions

WHAT TO NEVER CLICK — describe visually:
- Large PHOTOS/IMAGES — clicking opens a fullscreen gallery trap ("1 of N")
- Social media icons (small colored squares in footer)
- Navigation dropdown menus in the header
- Ad banners, dealer/brand logos
- Heart/favorite icons on cards

STEPS — use the executor's action syntax but describe targets visually:
- "click() on the text that says 'Year Make Model' BELOW the photo" (NOT the photo itself)
- scroll(direction="down", amount=5) for scrolling
- key_press(keys="alt+left") for going back
- done(success=true, summary="...") for completion
- CRITICAL: every click() instruction must say WHAT TEXT OR BUTTON to click, not coordinates

ERROR HANDLERS — what to do for each error state:
- 404/Page Not Found → key_press(keys="alt+left"), done(summary="SKIPPED | 404")
- Photo gallery (shows "1 of N" with fullscreen image) → key_press(keys="Escape"), key_press(keys="alt+left")
- Off-site (navigated away from the target site) → key_press(keys="alt+left") immediately
- Cookie popup → click the Accept button ONCE, max 2 steps

FILTER/SETUP TASK RULES:
- The model WILL try to call done() immediately without interacting. You MUST include:
  1. "Do NOT call done() in your first step. You must click and type first."
  2. A VERIFICATION step: "Read the result count on the page — it should be LESS than [threshold]"
  3. FAILURE criteria: "If the result count still shows [unfiltered count], filters were NOT applied"
- Keep filter tasks focused: fewer filters = higher success rate
- CRITICAL: Warn the model about TRAP BUTTONS in the sidebar that navigate away from \
  the page (e.g. "Calculate", "Get Pre-Qualified", "Get Started", "Apply Now"). \
  The model MUST be told: "Do NOT click any buttons — ONLY click filter options/checkboxes"
- If a filter is deep in the sidebar, tell the model exactly which sections to scroll PAST \
  before clicking, so it doesn't click intermediate interactive elements

EXTRACTION/DATA TASK RULES:
- URL from the browser address bar is a MANDATORY output field
- Include explicit instruction: "IMMEDIATELY after the detail page loads, read the URL \
  from the browser address bar at the top of the screen. It looks like: site.com/item/name-id/"
- Output format must show REALISTIC example data, not placeholders like <from title>
- WARN the model: "Do NOT output template text like 'Year: Year' — output actual values like 'Year: 2018'"

PHONE NUMBER EXTRACTION RULES:
- Phone numbers are the HIGHEST VALUE field — they enable direct seller contact
- Phones are typically buried DEEP in listing pages: 5-6 scrolls below photos/gallery
- Include explicit scrolling instructions: "scroll(direction='down', amount=5) AGGRESSIVELY \
  past the photo gallery to reach Description/Seller Notes where phone numbers appear"
- Phone formats to look for: (305)555-1234, 786-555-1234, 305.555.5678, 10+ digit numbers
- NOT phone numbers: prices ($45,000), years (2020), zip codes (33101), model numbers
- Do NOT click "Contact Seller" or "Request Info" buttons — they open popup forms, not phone numbers
- If the listing shows "Contact Seller" instead of a phone, the seller chose to hide their number — move on
- ALWAYS report phone: "Phone: 305-555-1234" or "Phone: none" — never omit the field

OUTPUT FORMAT — exact format for done() calls with realistic examples

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
I'm showing you screenshots from a website at 1280x720 resolution. Describe the layout so a VISION MODEL can find and click the right elements.

CRITICAL: Do NOT output pixel coordinates. The layout shifts between sessions (ads, banners). \
Instead, describe elements by their VISUAL APPEARANCE so the executing model can locate them visually.

For each element, describe:
- What it LOOKS LIKE (color, size relative to page, shape)
- WHERE on the page (left sidebar, center content, top header, bottom footer)
- What TEXT is visible on or near it
- What's ABOVE and BELOW it (spatial context)

1. LISTING CARDS:
   - What does a listing card look like? (overall shape, what's inside it)
   - Where is the PHOTO within the card vs the TITLE TEXT vs the PRICE?
   - What exactly should the model click to open the listing detail page?
   - What should the model NEVER click? (the photo opens a gallery trap)

2. DANGEROUS ELEMENTS — describe visually so model can avoid:
   - Social media icons: what do they look like, where are they?
   - Ad banners: what do they look like?
   - Dealer logos/badges: what do they look like?
   - Photo gallery: what happens if model clicks a photo?

3. FILTERS (if visible in sidebar):
   - What does the zip/location input look like?
   - What does the price filter look like?
   - What does the seller type filter look like?
   - What does the sort dropdown look like?

4. DETAIL PAGE (if any screenshot shows one):
   - What does the detail page look like?
   - Where is the description/seller notes text relative to the photos?
   - Where is the contact/phone info typically located?

5. URL STRUCTURE:
   - What URL patterns do you see in the address bar or listing links?
   - What does a filtered URL look like vs an unfiltered one?
   - What URL path segments correspond to which filters?

Describe everything VISUALLY. No pixel coordinates. No CSS selectors. Only visual descriptions.
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
    force = False

    args = sys.argv[2:]
    while args:
        if args[0] == "--browse" and len(args) > 1:
            browse_dir = args[1]
            args = args[2:]
        elif args[0] == "--force":
            force = True
            args = args[1:]
        elif not output_path:
            output_path = args[0]
            args = args[1:]
        else:
            args = args[1:]

    if not output_path:
        output_path = plan_path.replace(".txt", "_opus.json")

    if browse_dir:
        task_suite = browse_and_plan(plan_path, browse_dir, output_path=output_path, force=force)
    else:
        task_suite = plan_with_opus(plan_path, output_path=output_path, force=force)

    print(json.dumps(task_suite, indent=2))


if __name__ == "__main__":
    main()
