"""Plan Decomposer — break plain text plans into executable micro-intents.

Takes a human-written plan like "Search BoatTrader for private seller boats..."
and decomposes it into an ordered list of atomic micro-intents, each executable
by Holo3 in 3-8 steps.

Architecture:
    Plain text plan → Claude Sonnet (one-time, ~$0.01)
      → List of MicroIntent with:
        - intent: 1 sentence for Holo3
        - type: click, scroll, navigate, extract, filter, paginate, loop
        - verify: expected outcome for Claude to check
        - budget: max Holo3 steps (3-8)
        - reverse: how to undo if failed
      → MicroPlanRunner executes sequentially with checkpoint/verify/reverse

Usage:
    decomposer = PlanDecomposer()
    micro_plan = decomposer.decompose("plans/boattrader/extract_only.txt")
    # Returns list of MicroIntent
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MicroIntent:
    """A single atomic instruction for the CUA executor."""
    intent: str             # 1 sentence for Holo3: "Click the blue title text below the photo"
    type: str               # click, scroll, navigate, extract_url, extract_data, filter, paginate, loop, navigate_back
    verify: str = ""        # Expected outcome: "URL contains boattrader.com/boat/"
    budget: int = 5         # Max Holo3 steps
    reverse: str = ""       # How to undo: "Press Alt+Left"
    grounding: bool = False # Enable ClaudeGrounding for this step
    claude_only: bool = False  # No Holo3 steps — Claude reads screenshot
    loop_target: int = -1   # For loop type: jump back to this step index
    loop_count: int = 0     # For loop type: how many times to repeat


@dataclass
class MicroPlan:
    """Ordered list of micro-intents decomposed from a plain text plan."""
    steps: list[MicroIntent] = field(default_factory=list)
    source_plan: str = ""
    domain: str = ""

    def summary(self) -> str:
        lines = [f"MicroPlan: {len(self.steps)} steps for {self.domain}"]
        for i, s in enumerate(self.steps):
            tag = "🤖" if not s.claude_only else "🧠"
            lines.append(f"  [{i:2d}] {tag} {s.type:15s} {s.intent[:60]}")
        return "\n".join(lines)


DECOMPOSE_PROMPT = """\
You are a CUA (Computer Use Agent) plan decomposer. Break this plain text plan into \
an ordered list of atomic micro-intents.

PLAIN TEXT PLAN:
{plan_text}

RULES:
1. Each micro-intent is ONE sentence that a vision model can execute in 3-8 actions
2. Use POSITIVE instructions only ("Click the blue text" not "Don't click the photo")
3. Keep intents SHORT — under 20 words each
4. Include a "verify" field: what the screen should look like after this step
5. Include a "reverse" field: how to undo if this step fails
6. For data extraction (reading text from screen), set claude_only=true — a separate \
   vision model reads the screenshot, no executor steps needed
7. For repeated actions (process each listing), use type="loop" with loop_target \
   pointing to the step index to jump back to
8. Order matters — each step builds on the previous one's verified state

STEP TYPES:
- navigate: Go to a URL or page
- filter: Click a filter/dropdown option
- click: Click a specific element on the page
- scroll: Scroll the page
- extract_url: Read URL from address bar (claude_only=true)
- extract_data: Read data from page content (claude_only=true)
- navigate_back: Go back to previous page
- paginate: Click Next/page number to advance
- loop: Repeat a sequence of steps

Output ONLY valid JSON array:
[
  {{"intent": "...", "type": "...", "verify": "...", "budget": N, "reverse": "...", "grounding": false, "claude_only": false, "loop_target": -1, "loop_count": 0}},
  ...
]
"""


class PlanDecomposer:
    """Decomposes plain text plans into micro-intents using Claude Sonnet."""

    def __init__(self, api_key: str = "", model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model

    def decompose(self, plan_path: str) -> MicroPlan:
        """Read a plan file and decompose into micro-intents.

        Args:
            plan_path: Path to plain text plan file.

        Returns:
            MicroPlan with ordered list of MicroIntent steps.
        """
        import hashlib
        import requests

        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")

        with open(plan_path) as f:
            plan_text = f.read()

        # Extract domain from plan text
        domain = ""
        m = re.search(r"(?:https?://)?(?:www\.)?([\w\-]+\.[\w]+)", plan_text)
        if m:
            domain = m.group(1)

        # Check cache
        plan_hash = hashlib.md5(plan_text.encode()).hexdigest()[:8]
        cache_path = plan_path.replace(".txt", f"_micro_{plan_hash}.json")
        if os.path.exists(cache_path):
            try:
                cached = json.loads(open(cache_path).read())
                plan = MicroPlan(source_plan=plan_text, domain=domain)
                plan.steps = [MicroIntent(**s) for s in cached]
                logger.info(f"Loaded cached micro-plan: {cache_path} ({len(plan.steps)} steps)")
                return plan
            except Exception:
                pass

        logger.info(f"Decomposing plan with {self.model}: {plan_path}")
        prompt = DECOMPOSE_PROMPT.format(plan_text=plan_text)

        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )

        if resp.status_code != 200:
            raise RuntimeError(f"Decompose API error: {resp.status_code} {resp.text[:200]}")

        text = ""
        for block in resp.json().get("content", []):
            if block.get("type") == "text":
                text = block["text"].strip()
                break

        # Parse JSON
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        steps_raw = json.loads(text)
        plan = MicroPlan(source_plan=plan_text, domain=domain)
        for s in steps_raw:
            plan.steps.append(MicroIntent(
                intent=s.get("intent", ""),
                type=s.get("type", "click"),
                verify=s.get("verify", ""),
                budget=s.get("budget", 5),
                reverse=s.get("reverse", ""),
                grounding=s.get("grounding", False),
                claude_only=s.get("claude_only", False),
                loop_target=s.get("loop_target", -1),
                loop_count=s.get("loop_count", 0),
            ))

        # Cache
        try:
            with open(cache_path, "w") as f:
                json.dump(steps_raw, f, indent=2)
            logger.info(f"Cached micro-plan: {cache_path}")
        except Exception:
            pass

        tokens = resp.json().get("usage", {})
        cost = (tokens.get("input_tokens", 0) * 3 + tokens.get("output_tokens", 0) * 15) / 1_000_000
        logger.info(f"Decomposed into {len(plan.steps)} micro-intents (~${cost:.3f})")
        logger.info(plan.summary())

        return plan
