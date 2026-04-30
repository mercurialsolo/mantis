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
from dataclasses import dataclass, field, fields
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MicroIntent:
    """A single atomic instruction for the CUA executor."""
    intent: str             # 1 sentence for Holo3: "Click the blue title text below the photo"
    type: str               # click, scroll, navigate, extract_url, extract_data, filter, paginate, loop, navigate_back, fill_field, submit, select_option
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
    # Structured payload for form-shaped step types (#80). Keys vary by type:
    #   fill_field    : {"label": str, "value": str}
    #   submit        : {"label": str}
    #   select_option : {"dropdown_label": str, "option_label": str}
    # Empty for everything else; the runner falls back to parsing `intent`.
    params: dict[str, Any] = field(default_factory=dict)
    # Per-step grounding hints for the runner. Free-form, plan-driven —
    # never inferred from the runner's domain assumptions. Recognised keys:
    #   layout: "listings" | "single"
    #     "listings"  → use ClaudeExtractor.find_all_listings (results-page click)
    #     "single"    → use ClaudeExtractor.find_form_target (one labelled element)
    #     missing     → runner picks based on step.type and step.section
    #   spam_indicators: list[str] — domain-specific spam strings to filter
    #   spam_label: str — what to call spam in prompts (e.g. "recruiter", "broker")
    #   entity_name: str — what the items are called on this page (job, lead, property)
    # Anything that used to be hardcoded in the extractor should now flow
    # through this field. See issue #86 for the redesign.
    hints: dict[str, Any] = field(default_factory=dict)


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

    def to_dict(self) -> dict:
        """Serializable form. Round-trips through JSON for callers that want
        to ship a pre-built plan over the wire (e.g. vision_claude passing
        a hand-authored micro_plan into MantisOrchestratedBackend)."""
        return {
            "steps": [
                {
                    f.name: getattr(s, f.name)
                    for f in fields(MicroIntent)
                }
                for s in self.steps
            ],
            "source_plan": self.source_plan,
            "domain": self.domain,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "MicroPlan":
        """Construct a MicroPlan from a dict (typically a JSON payload).

        Accepts both shapes: ``{"steps": [...]}`` (full ``to_dict()`` output)
        and a bare list of step dicts at the top level.
        """
        if isinstance(payload, list):
            steps_raw = payload
            source_plan = ""
            domain = ""
        else:
            steps_raw = payload.get("steps", [])
            source_plan = payload.get("source_plan", "")
            domain = payload.get("domain", "")
        plan = cls(source_plan=source_plan, domain=domain)
        for s in steps_raw:
            plan.steps.append(PlanDecomposer._build_intent(s))
        return plan


DECOMPOSE_PROMPT = """\
You are a CUA (Computer Use Agent) plan decomposer.

WHY MICRO-PLANS:
The executing model (Holo3, 3B params) can ONLY handle 1-sentence instructions with \
3-8 actions. It passes 100% on isolated tasks but fails when instructions are combined. \
Your job: break a human plan into SECTIONS of atomic steps with DEPENDENCIES between sections.

TWO PLAN SHAPES — pick the right step types:

A) LISTINGS / EXTRACTION FLOWS — search → click result → extract → loop
   Use: navigate, filter, click, scroll, extract_url, extract_data, navigate_back, paginate, loop.
   Example: "Find boats on BoatTrader", "Extract 10 jobs from Greenhouse".

B) FORM-DRIVEN FLOWS — login, edit pages, settings, dropdowns, single-button clicks
   Use: navigate, fill_field, submit, select_option, extract_data (for verification).
   Example: "Log in with credentials", "Update the Industry Vertical to X then Save",
            "Open the dropdown and pick the third option".

A single plan can mix both shapes (e.g. log in, navigate, then extract listings).

SECTIONS AND DEPENDENCIES — THIS IS CRITICAL:
Plans MUST be organized into sections. Each section has a purpose and a gate:

1. SETUP section (required=true): Navigate + login + apply filters + VERIFY
   - Every filter / fill_field / submit step has required=true (default)
   - For listings flows, the LAST setup step is a GATE: Claude verifies filters applied
   - For form-only flows, end with an extract_data step that verifies the right page/state
   - If the gate FAILS, the entire pipeline HALTS — do not proceed
   - Example listings gate: "Verify page heading shows expected filters and result count is reasonable"
   - Example form gate: "Verify the Edit Lead page is shown for the correct lead"

2. EXTRACTION section (listings flows only — depends on setup gate passing):
   - Click → URL → scroll → extract → re-navigate or back → loop
   - Click only organic target result cards. Skip sponsored, paid, or off-topic cards.
   - Extraction must inspect both contact areas AND expanded text sections.
   - Extraction must reject off-topic or spam listings even if data is visible.
   - If any content section is collapsed, the extraction step must expand it first.
   - Prefer safe reveal controls such as Show more, Read more, See more, Show phone,
     View phone, or Call. Never use generic contact forms or lead-generation buttons.

3. PAGINATION section (listings flows only — depends on extraction):
   - Paginate → loop back to extraction
   - Only runs after extraction exhausts current page

DYNAMIC PLAN VERIFICATION CONTRACT (listings flows):
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

VERB → STEP-TYPE MAPPING (FORM FLOWS):
   "log in", "sign in", "authenticate"
       → fill_field for each credential, then submit for the login button.
   "enter X in the Y field", "type X into Y", "fill in Y with X", "set Y to X"
       → fill_field with params={"label": "Y", "value": "X"}.
   "click the Submit button", "click Save", "click Update Lead", "press Continue",
   "submit the form", "click the {Leads/Settings/etc} navigation link",
   "click the {Edit Lead/Cancel} button", "go to the {Y} page"
   (when {Y} is a tab/nav/menu item, NOT a URL)
       → submit with params={"label": "<button or link text>"}.
       Use submit for any SINGLE LABELLED CLICKABLE on a non-listings page —
       buttons, nav links, tab items, menu items, dock icons, inline
       action links. The runner uses find_form_target which locates one
       labelled element by visible text, no listings-grid assumption.
   "select X from the Y dropdown", "choose X under Y", "pick X in the Y selector"
       → select_option with params={"dropdown_label": "Y", "option_label": "X"}.
   "click the first / next / nth result/row/listing/card/job/product/property"
       → click (this IS a listings click — many similar items on one page,
       runner picks the next un-extracted one). DO NOT use click for nav
       links, buttons, or any single-element clickable.

When the source text says "Click the user ID field and enter sarah.connor", emit
a SINGLE fill_field step (with label="User ID", value="sarah.connor") — NOT a
click step. The runner clicks the field as part of fill_field.

RULES:
- Each step: ONE action, ONE sentence, under 20 words
- POSITIVE framing only: "Click the blue title text" (not "Don't click the photo")
- Include WHAT + WHERE: "Click Private Seller text in left sidebar"
- For navigate steps: include the FULL URL in the intent
- Extraction steps (reading screen) use claude_only=true
- For fill_field / submit / select_option, ALWAYS populate `params` (label/value/...)
  even if the same info is in `intent`. The runner trusts `params` over the prose.
- The "reverse" field must be a CUA-executable instruction
- Set section="setup", section="extraction", or section="pagination"
- Set required=true for all setup/filter/form steps (this is the default)
- Set gate=true for the verification step at the end of setup

LOOP STRUCTURE:
  Extraction loop: click → URL → scroll → extract → back → loop(target=click, count=N)
  Pagination loop: paginate → loop(target=click, count=pages)
  The listing loop runs INSIDE the pagination loop.
  Form-only flows do not need loops.

PLAIN TEXT PLAN:
{plan_text}

STEP TYPES:
- navigate: Go to a URL — include full URL in intent (budget=3)
- filter: Click a filter option (budget=8, grounding=true, required=true, section="setup")
- click: Click an element on a listings/results page (budget=8, grounding=true)
- scroll: Scroll until target content visible (budget=10, section="extraction")
- extract_url: Read URL from address bar (claude_only=true, budget=0, section="extraction")
- extract_data: Inspect page and read structured data or verify state (claude_only=true, budget=0)
- navigate_back: Go back (budget=3, section="extraction")
- paginate: Click Next page (budget=10, grounding=true, section="pagination")
- loop: Jump back to step index (loop_target=N, loop_count=max)
- fill_field: Click a labelled input and type a value
              (budget=4, params={"label": "<visible field label>", "value": "<text to type>"})
- submit: Click a SINGLE LABELLED CLICKABLE on a non-listings page — buttons
          (Login / Save / Submit / Update / Continue), navigation links,
          tab items, menu items, dock icons, action links.
          NOT for "click the next listing/result" — use `click` for that.
          (budget=4, params={"label": "<visible button or link text>"})
- select_option: Open a dropdown and pick an option by visible text
                 (budget=6, params={"dropdown_label": "<dropdown name>",
                                    "option_label": "<option text>"})

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
        with open(plan_path) as f:
            plan_text = f.read()
        cache_path = plan_path.replace(".txt", "_micro_{hash}.json")
        return self.decompose_text(plan_text, cache_path_template=cache_path)

    def decompose_text(
        self,
        plan_text: str,
        *,
        cache_path_template: str | None = None,
    ) -> MicroPlan:
        """Decompose a free-text plan into micro-intents.

        Used by callers (e.g. vision_claude's MantisOrchestratedBackend) that
        receive a prompt string rather than a path on disk.

        Args:
            plan_text: The plan text to decompose.
            cache_path_template: Optional path template containing ``{hash}``
                — when provided, the decomposed plan is cached at
                ``cache_path_template.replace("{hash}", <8-char-md5>)``.
                Pass ``None`` to skip the cache (typical for ad-hoc prompts).

        Returns:
            MicroPlan with ordered list of MicroIntent steps.
        """
        import hashlib
        import requests

        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")

        # Extract domain from plan text
        domain = ""
        m = re.search(r"(?:https?://)?(?:www\.)?([\w\-]+\.[\w]+)", plan_text)
        if m:
            domain = m.group(1)

        # Check cache — include prompt version in hash to invalidate on schema changes
        prompt_version = "v11_submit_covers_nav"  # Bump this when DECOMPOSE_PROMPT changes
        plan_hash = hashlib.md5(f"{prompt_version}:{plan_text}".encode()).hexdigest()[:8]
        cache_path = (
            cache_path_template.replace("{hash}", plan_hash)
            if cache_path_template
            else None
        )
        if cache_path and os.path.exists(cache_path):
            try:
                cached = json.loads(open(cache_path).read())
                plan = MicroPlan(source_plan=plan_text, domain=domain)
                for s in cached:
                    plan.steps.append(self._build_intent(s))

                logger.info(f"Loaded cached micro-plan: {cache_path} ({len(plan.steps)} steps)")
                return plan
            except Exception:
                pass

        logger.info(f"Decomposing plan with {self.model} ({len(plan_text)} chars)")
        # Use replace() instead of format() — the prompt has literal `{...}`
        # JSON examples (params={"label": ...}) that confuse str.format.
        prompt = DECOMPOSE_PROMPT.replace("{plan_text}", plan_text)

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
        if cache_path:
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

    # Step types that interact with form-shaped UIs (login, edit, settings).
    # Always claude-grounded; no listings find_all. See issue #80.
    FORM_STEP_TYPES = ("fill_field", "submit", "select_option")

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
            elif step_type in PlanDecomposer.FORM_STEP_TYPES:
                # Form steps default to "setup" — login + form-fill happens before extraction.
                section = "setup"

        # Form steps default to required=True — failing to fill a login field
        # or click Submit is fatal to the rest of the plan.
        default_required = step_type == "filter" or step_type in PlanDecomposer.FORM_STEP_TYPES

        params = s.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        hints = s.get("hints") or {}
        if not isinstance(hints, dict):
            hints = {}

        return MicroIntent(
            intent=s.get("intent", ""),
            type=step_type,
            verify=s.get("verify", s.get("expected_outcome", "")),
            budget=s.get("budget", 5),
            reverse=reverse,
            grounding=s.get(
                "grounding",
                step_type in ("click", "filter", "paginate") + PlanDecomposer.FORM_STEP_TYPES,
            ),
            claude_only=s.get("claude_only", step_type in ("extract_url", "extract_data")),
            loop_target=s.get("loop_target", -1),
            loop_count=s.get("loop_count", 0),
            section=section,
            required=s.get("required", default_required),
            gate=s.get("gate", False),
            params=params,
            hints=hints,
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
