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
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

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

    # Log stats
    tasks = task_suite.get("tasks", [])
    loop_tasks = sum(1 for t in tasks if t.get("loop"))
    logger.info(f"Generated {len(tasks)} tasks ({loop_tasks} with loops)")

    tokens = data.get("usage", {})
    cost = (tokens.get("input_tokens", 0) * 3 + tokens.get("output_tokens", 0) * 15) / 1_000_000
    logger.info(f"Opus tokens: {tokens.get('input_tokens', 0)} in + {tokens.get('output_tokens', 0)} out = ~${cost:.3f}")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(task_suite, f, indent=2)
        logger.info(f"Saved to {output_path}")

    return task_suite


def main():
    """CLI: python -m mantis_agent.opus_planner plans/boattrader/extract_only.txt"""
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python -m mantis_agent.opus_planner <plan_file> [output_file]")
        sys.exit(1)

    plan_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else plan_path.replace(".txt", "_opus.json")

    task_suite = plan_with_opus(plan_path, output_path=output_path)
    print(json.dumps(task_suite, indent=2))


if __name__ == "__main__":
    main()
