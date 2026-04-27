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
    section: str = ""       # Section name: "setup", "extraction", "pagination"
    required: bool = False  # If True and step fails after retries, halt pipeline
    gate: bool = False      # If True, this is a verification gate — must pass before next section
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
You are a CUA (Computer Use Agent) plan decomposer.

WHY MICRO-PLANS:
The executing model (Holo3, 3B params) can ONLY handle 1-sentence instructions with \
3-8 actions. It passes 100% on isolated tasks but fails when instructions are combined. \
Your job: break a human plan into SECTIONS of atomic steps with DEPENDENCIES between sections.

SECTIONS AND DEPENDENCIES — THIS IS CRITICAL:
Plans MUST be organized into sections. Each section has a purpose and a gate:

1. SETUP section (required=true): Navigate + apply filters + VERIFY
   - Every filter step has required=true
   - The LAST step in setup is a GATE (gate=true): Claude verifies filters applied
   - If the gate FAILS, the entire pipeline HALTS — do not extract from wrong page
   - Example gate: "Verify page heading shows expected filters and result count is reasonable"

2. EXTRACTION section (depends on setup gate passing):
   - Click → URL → scroll → extract → back → loop
   - Only runs if setup gate passed
   - Click only organic target result cards. Skip sponsored, paid, or off-topic cards.
   - Extraction must inspect both contact areas AND expanded text sections.
   - Extraction must reject off-topic or spam listings even if data is visible.
   - If any content section is collapsed, the extraction step must expand it first.
   - Prefer safe reveal controls such as Show more, Read more, See more, Show phone,
     View phone, or Call. Never use generic contact forms or lead-generation buttons.

3. PAGINATION section (depends on extraction):
   - Paginate → loop back to extraction
   - Only runs after extraction exhausts current page

DYNAMIC PLAN VERIFICATION CONTRACT:
For any plan that browses search results, listings, products, profiles, rows, or
other repeated page items, structure the plan so the runtime can prove coverage:
   - The setup gate must state the required page/filter/search state.
   - The extraction loop must discover visible items, attempt each discovered
     item exactly once, open the item detail or row, and produce a terminal
     extraction decision for it.
   - Page exhaustion must be observable before pagination: the executor must
     scan down the results page until no new relevant items remain.
   - Pagination must be a separate step after page exhaustion, and the loop must
     continue until no next page/control is available.
   - Do not hardcode a fixed item count unless the user explicitly provides one;
     use a bounded loop with runtime exhaustion checks.

RULES:
- Each step: ONE action, ONE sentence, under 20 words
- POSITIVE framing only: "Click the blue title text" (not "Don't click the photo")
- Include WHAT + WHERE: "Click Private Seller text in left sidebar"
- For navigate steps: include the FULL URL in the intent
- Extraction steps (reading screen) use claude_only=true
- The "reverse" field must be a CUA-executable instruction
- Set section="setup", section="extraction", or section="pagination"
- Set required=true for all setup/filter steps
- Set gate=true for the verification step at the end of setup

LOOP STRUCTURE:
  Extraction loop: click → URL → scroll → extract → back → loop(target=click, count=N)
  Pagination loop: paginate → loop(target=click, count=pages)
  The listing loop runs INSIDE the pagination loop.

PLAIN TEXT PLAN:
{plan_text}

STEP TYPES:
- navigate: Go to a URL — include full URL in intent (budget=3)
- filter: Click a filter option (budget=8, grounding=true, required=true, section="setup")
- click: Click a specific element (budget=8, grounding=true, section="extraction")
- scroll: Scroll until target content visible (budget=10, section="extraction")
- extract_url: Read URL from address bar (claude_only=true, budget=0, section="extraction")
- extract_data: Inspect contact area and expanded description/details, then read structured data (claude_only=true, budget=0)
- navigate_back: Go back (budget=3, section="extraction")
- paginate: Click Next page (budget=10, grounding=true, section="pagination")
- loop: Jump back to step index (loop_target=N, loop_count=max)

Output ONLY valid JSON array of steps.
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

        # Check cache — include prompt version in hash to invalidate on schema changes
        prompt_version = "v8_step_type_fix"  # Bump this when DECOMPOSE_PROMPT changes
        plan_hash = hashlib.md5(f"{prompt_version}:{plan_text}".encode()).hexdigest()[:8]
        cache_path = plan_path.replace(".txt", f"_micro_{plan_hash}.json")
        if os.path.exists(cache_path):
            try:
                cached = json.loads(open(cache_path).read())
                plan = MicroPlan(source_plan=plan_text, domain=domain)
                for s in cached:
                    plan.steps.append(self._build_intent(s))

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
            plan.steps.append(self._build_intent(s))

        # Fix 3: Validate and fix loop targets — must point to the click step
        self._fix_loop_targets(plan)

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

    @staticmethod
    def _build_intent(s: dict) -> MicroIntent:
        """Build a MicroIntent from a raw dict — used by both cache and fresh paths."""
        step_type = s.get("type") or s.get("step_type") or s.get("action") or "click"
        reverse = s.get("reverse") or s.get("reverse_action") or ""

        # Infer section from step type if not provided
        section = s.get("section", "")
        if not section:
            if step_type in ("navigate", "filter"):
                section = "setup"
            elif step_type in ("paginate",):
                section = "pagination"
            elif step_type in ("click", "scroll", "extract_url", "extract_data", "navigate_back"):
                section = "extraction"

        return MicroIntent(
            intent=s.get("intent", ""),
            type=step_type,
            verify=s.get("verify", s.get("expected_outcome", "")),
            budget=s.get("budget", 5),
            reverse=reverse,
            grounding=s.get("grounding", step_type in ("click", "filter", "paginate")),
            claude_only=s.get("claude_only", step_type in ("extract_url", "extract_data")),
            loop_target=s.get("loop_target", -1),
            loop_count=s.get("loop_count", 0),
            section=section,
            required=s.get("required", step_type == "filter"),
            gate=s.get("gate", False),
        )

    @staticmethod
    def _fix_loop_targets(plan: MicroPlan):
        """Fix 3: Ensure loop targets point to the click step, not extract_url.

        The decomposer often generates loop→extract_url instead of loop→click.
        Find the first extraction click step and retarget all loops to it.
        """
        click_idx = None
        for i, s in enumerate(plan.steps):
            if s.type == "click" and s.section == "extraction":
                click_idx = i
                break

        if click_idx is None:
            return

        for s in plan.steps:
            if s.loop_target >= 0 and s.loop_target != click_idx:
                # Only fix if the target is close (off by 1-2, typical decomposer error)
                if abs(s.loop_target - click_idx) <= 2:
                    logger.info(f"  [fix] Loop target {s.loop_target} → {click_idx} (click step)")
                    s.loop_target = click_idx
