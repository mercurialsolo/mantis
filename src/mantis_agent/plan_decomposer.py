"""Plan Decomposer — break plain text plans into executable micro-intents.

Takes a human-written plan like "Log in to the CRM, find lead X, update status"
or "Search a marketplace for items matching <criteria>, extract details" and
decomposes it into an ordered list of atomic micro-intents, each executable
by Holo3 in 3-8 steps.

The decomposer is **plan-shape aware**: it heuristically detects whether the
plan is a listings/extraction flow, a form-driven flow, a multi-step workflow,
or an inspect-only flow, and adapts the guidance Claude sees accordingly. It
does NOT hard-wire any specific domain (no boattrader / linkedin / shopify
assumptions in the prompt).

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
    micro_plan = decomposer.decompose_text("Extract jobs from ...")
    # Returns a MicroPlan with an ordered list of MicroIntent steps
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


def _extract_json_payload(text: str) -> Any:
    """Robustly extract the first JSON object or array from a model response.

    Handles three response shapes Claude empirically produces:

      1. Pure JSON:           ``{"shapes": [...], "steps": [...]}``
      2. Code-fenced JSON:    ``\\`\\`\\`json\\n {...} \\n\\`\\`\\``` (with or without ``json`` tag)
      3. Prose-wrapped JSON:  ``"Here's the decomposition:\\n{...}\\nLet me know..."``

    Returns the parsed Python object (dict or list), or ``None`` when no
    JSON could be found. The caller decides what to do with None — typically
    raises with the offending text so operators can debug.

    Issue #112 documented Claude's habit of prepending prose before the
    JSON fence. The previous parser only handled fence-wrapped output;
    this version finds the outermost balanced ``{...}`` or ``[...]``.
    """
    if not text:
        return None
    s = text.strip()

    # Strip code fences if present.
    if s.startswith("```"):
        # Drop the opening fence (which may include a language tag).
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
    if s.endswith("```"):
        s = s.rsplit("```", 1)[0]
    s = s.strip()

    # Fast path: the cleaned text is already valid JSON.
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: scan left-to-right for the first balanced JSON object OR
    # array. Either kind is acceptable — we prefer whichever comes first
    # because Claude's typical preamble is "Here's the result:\n\n{...}"
    # and we want the JSON immediately following it. String-literal state
    # tracking prevents braces inside intent strings from confusing the
    # depth counter.
    closer_for = {"{": "}", "[": "]"}
    for start in range(len(s)):
        opener = s[start]
        if opener not in closer_for:
            continue
        closer = closer_for[opener]
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(s)):
            ch = s[i]
            if escape:
                escape = False
                continue
            if ch == "\\" and in_str:
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    candidate = s[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except (json.JSONDecodeError, ValueError):
                        # Balanced but not valid JSON — keep scanning.
                        break
    return None


@dataclass
class MicroIntent:
    """A single atomic instruction for the CUA executor."""
    intent: str             # 1 sentence for Holo3: "Click the blue title text below the photo"
    type: str               # click, scroll, navigate, extract_url, extract_data, filter, paginate, loop, navigate_back, fill_field, submit, select_option, right_click
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
    #   fill_field    : {"label": str, "value": str, "aliases"?: list[str]}
    #   submit        : {"label": str, "aliases"?: list[str]}
    #   select_option : {"dropdown_label": str, "option_label": str}
    #   right_click   : {"label": str, "aliases"?: list[str]}  # opens native context menu
    #   navigate      : {"wait_after_load_seconds"?: int}
    # ``aliases`` (#89 §2) lets a plan list synonyms for a primary submit
    # button whose copy varies across products (e.g. "Save" / "Save Changes"
    # / "Apply"). Use only for primary submit — never for nav links or
    # unique-label clickables.
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
    # Vision-only conditional step support (#643 stage 2). The plan
    # composes a ``detect_visible`` step that asks Holo3/Claude
    # "is X visible?" against the current screenshot and binds the
    # boolean result to ``runner._state_vars[out_var]``. Subsequent
    # steps then carry a ``guard`` field naming that variable — if
    # the variable is False (or absent), the runner skips the step
    # for free (no vision call, no env action).
    #
    # ``guard`` is empty for unconditional steps (the default).
    # ``out_var`` is set on ``detect_visible`` steps; ignored elsewhere.
    # Pure-vision: detection is one Claude/Holo3 call against the
    # screenshot, never a DOM probe. CUA-purity preserved.
    guard: str = ""
    out_var: str = ""


@dataclass
class LoopGroup:
    """A loop step + the body it iterates, with a parallelizability tag.

    Produced by :meth:`PlanDecomposer._classify_loop_groups` after the loop
    targets are normalized (#605). Consumers — the fan-out runner in
    particular — read ``shape`` to decide whether the body is safe to
    partition across workers (issue #614).

    ``loop_step_idx`` is the index of the ``loop`` step itself.
    ``body_range`` is the half-open ``[start, end)`` slice of plan.steps
    covering the loop body (``loop_target`` … ``loop_step_idx``).
    ``shape`` is one of:

      * ``parallelizable_url_collect`` — body starts with ``click`` on a
        listing card and contains ``extract_url``. Each iteration is
        independent given the listing URL, so the body is safe to
        rewrite as ``navigate(url) → scroll → extract_data`` and
        partition across workers.
      * ``parallelizable_pagination`` — body's terminal step is
        ``paginate``. Each page is an independent unit.
      * ``sequential`` — anything else (forms, login, stateful flows).
        Don't fan out.
    """

    loop_step_idx: int
    body_range: tuple[int, int]
    shape: str = "sequential"


@dataclass
class MicroPlan:
    """Ordered list of micro-intents decomposed from a plain text plan."""
    steps: list[MicroIntent] = field(default_factory=list)
    source_plan: str = ""
    domain: str = ""
    # Plan shapes Claude classified at decomposition time (subset of
    # ``PlanDecomposer.KNOWN_PLAN_SHAPES``). Empty when Claude returned the
    # legacy bare-array schema or when classification was unreliable.
    shapes: list[str] = field(default_factory=list)
    # Loop bodies tagged with a parallelizability shape (issue #614).
    # Populated by :meth:`PlanDecomposer._classify_loop_groups` after
    # ``_fix_loop_targets`` has normalized ``loop_target``. Empty for
    # plans with no ``loop`` steps. Pure analysis — runtime behaviour
    # is unchanged until the fan-out runner (#616, #617) reads this.
    loop_groups: list[LoopGroup] = field(default_factory=list)
    # Per-plan pagination URL template (#629). When set, the Modal
    # fan-out runner uses this to synthesize per-page partition URLs
    # instead of the default ``{base}/page-{n}/``. Resolution order at
    # dispatch time:
    #   1. ``MicroPlan.pagination_url_template`` (this field)
    #   2. ``paginate_step.params['url_template']`` (back-compat)
    #   3. :data:`fanout_runner.DEFAULT_PAGINATION_URL_TEMPLATE`
    # Populated by the LLM at decompose time when the source plan
    # mentions a URL with a page indicator, OR set programmatically
    # by a future enhancer step that probes the site.
    pagination_url_template: str = ""

    def summary(self) -> str:
        shape_tag = f" [{','.join(self.shapes)}]" if self.shapes else ""
        lines = [f"MicroPlan: {len(self.steps)} steps for {self.domain}{shape_tag}"]
        for i, s in enumerate(self.steps):
            tag = "🤖" if not s.claude_only else "🧠"
            lines.append(f"  [{i:2d}] {tag} {s.type:15s} {s.intent[:60]}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serializable form. Round-trips through JSON for callers that want
        to ship a pre-built plan over the wire (e.g. the host integration passing
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
            "shapes": list(self.shapes),
            "loop_groups": [
                {
                    "loop_step_idx": g.loop_step_idx,
                    "body_range": list(g.body_range),
                    "shape": g.shape,
                }
                for g in self.loop_groups
            ],
            "pagination_url_template": self.pagination_url_template,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "MicroPlan":
        """Construct a MicroPlan from a dict (typically a JSON payload).

        Accepts both shapes: ``{"steps": [...]}`` (full ``to_dict()`` output)
        and a bare list of step dicts at the top level.
        """
        shapes_raw: Any = []
        if isinstance(payload, list):
            steps_raw = payload
            source_plan = ""
            domain = ""
        else:
            steps_raw = payload.get("steps", [])
            source_plan = payload.get("source_plan", "")
            domain = payload.get("domain", "")
            shapes_raw = payload.get("shapes", [])
        plan = cls(source_plan=source_plan, domain=domain)
        plan.shapes = PlanDecomposer._normalize_shapes(shapes_raw)
        for s in steps_raw:
            plan.steps.append(PlanDecomposer._build_intent(s))
        # Round-trip ``loop_groups`` when present (#614). Recompute when
        # absent so plans persisted before this field existed still get
        # classified on load.
        groups_raw = payload.get("loop_groups") if isinstance(payload, dict) else None
        if groups_raw:
            plan.loop_groups = [
                LoopGroup(
                    loop_step_idx=int(g.get("loop_step_idx", -1)),
                    body_range=tuple(g.get("body_range", (0, 0))),
                    shape=str(g.get("shape", "sequential")),
                )
                for g in groups_raw
                if isinstance(g, dict)
            ]
        else:
            PlanDecomposer._classify_loop_groups(plan)
        # #629: round-trip per-plan pagination URL template if present.
        if isinstance(payload, dict):
            plan.pagination_url_template = str(
                payload.get("pagination_url_template", "") or ""
            )
        return plan


DECOMPOSE_PROMPT = """\
You are a CUA (Computer Use Agent) plan decomposer.

WHY MICRO-PLANS:
The executing model (Holo3, 3B params) can ONLY handle 1-sentence instructions with \
3-8 actions. It passes 100% on isolated tasks but fails when instructions are combined. \
Your job: break a human plan into SECTIONS of atomic steps with DEPENDENCIES between sections.

FOUR PLAN SHAPES — pick the right step types per step:

STEP 0 — CLASSIFY THE PLAN'S SHAPE(S) BEFORE GENERATING STEPS.
Read the source plan and decide which of these four shapes it matches.
A plan can match multiple shapes (e.g. login=form + multi-page CRM=workflow).
Use ONLY these tokens in your output: "listings", "form", "workflow", "inspect".

A) LISTINGS / EXTRACTION FLOWS — search → click result → extract → loop.
   Use: navigate, filter, click, scroll, extract_url, extract_data, navigate_back,
   paginate, loop.
   Recognise this shape when the source says: "find/extract/list N items
   matching <criteria>", "go through each result", "scrape/collect data
   from the search results page", or describes pagination.

B) FORM-DRIVEN FLOWS — login, edit pages, settings panels, dropdowns,
   single labelled buttons.
   Use: navigate, fill_field, submit, select_option, extract_data (for verification).
   Recognise this shape when the source says: "log in / sign in",
   "enter/type/fill X into the Y field", "set Y to X", "choose/select/pick X
   from Y", "click the {Save|Submit|Update|Continue|Login} button".

C) WORKFLOW / MULTI-STEP TRANSITION FLOWS — log in → navigate → select an
   existing record → edit → save. CRMs, admin consoles, ticket systems,
   project boards, content editors, settings dashboards.
   Use: navigate, fill_field, submit, select_option, extract_data.
   Recognise this shape when the source says: "log in" + "go to/open the
   <name> page/tab" + "select/click the <type> with <attribute>" + "edit/
   change/update <field>" + "click <save/update/apply>". The hallmark is
   navigating through several authenticated pages without an extraction
   loop.
   IMPORTANT: do NOT use the listings click for "select the lead/ticket/
   record" — those usually open a single targeted item by visible text.
   Use submit (with the visible label) for single-target picks like
   "the first Qualified lead", "the open ticket from John", "the Acme
   account".

D) INSPECT-ONLY / VERIFICATION FLOWS — read the screen, confirm a value,
   take a screenshot.
   Use: navigate, extract_data (claude_only=true) for the read pass,
   plus submit only when navigation is required to reach the inspection
   target.
   Recognise this shape when the source asks "verify that", "check
   whether", "confirm X is Y", "what's the value of X" without any
   write actions.

A single plan can mix shapes (log in, navigate, then extract a list).
Decide step-by-step — DO NOT force a whole plan into one shape. The
runtime SHAPE HINT above tells you which shape(s) the heuristic
detected; trust it as a strong default but override per-step when the
source text contradicts.

SECTIONS AND DEPENDENCIES — THIS IS CRITICAL:
Plans MUST be organized into sections. Each section has a purpose and a gate:

1. SETUP section (required=true): Navigate + login + apply filters + VERIFY
   - Every filter / fill_field / submit step has required=true (default)
   - For listings flows, the LAST setup step is a GATE: Claude verifies filters applied
   - For form-only flows, end with an extract_data step that verifies the right
     page/state was reached (URL, header, and any read-back of the values just set)
   - If the gate FAILS, the entire pipeline HALTS — do not proceed
   - Example listings gate: "Verify page heading shows the expected filters and a
     reasonable result count"
   - Example form gate: "Verify the page shows the just-saved record with the
     updated values"

2. EXTRACTION section (listings flows only — depends on setup gate passing):
   - Click → URL → scroll → extract → re-navigate or back → loop
   - Click only organic target result items. Skip sponsored, paid, or
     off-topic cards.
   - Extraction must inspect both summary regions AND any expanded detail
     sections.
   - Extraction must reject off-topic or spam items even if data is visible.
   - If any content section is collapsed, the extraction step must expand it first.
   - Prefer safe reveal controls (Show more, Read more, See details, View …,
     Expand). Never use generic contact / lead-generation forms.

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

CRITICAL — PRESERVE LITERAL VALUES FROM THE SOURCE PLAN.
Whenever the source plan names a literal value the user wants entered
(a username, password, email, API key, search term, comment, dropdown
choice, etc.), that value MUST appear verbatim in `params.value` (or
`params.option_label` for selects) of the corresponding step. Never
emit a fill_field with an empty `value` when the source provided one
— the runner will type "" into the field and the form will be empty.

WORKED EXAMPLES — credential / value preservation:
  Source: "Log in with user ID alice password hunter2"
  → Two fill_field steps + one submit:
      [{"type": "fill_field", "intent": "Enter alice in the user ID field",
        "params": {"label": "user ID", "value": "alice"}, ...},
       {"type": "fill_field", "intent": "Enter hunter2 in the password field",
        "params": {"label": "password", "value": "hunter2"}, ...},
       {"type": "submit", "intent": "Click the Sign In button to authenticate",
        "params": {"label": "Sign In"}, ...}]
  Source: "Set the search box to acme corp"
  → {"type": "fill_field", "intent": "Type acme corp into the search box",
     "params": {"label": "search box", "value": "acme corp"}, ...}
  Source: "Update the Industry to Space Exploration"
  → {"type": "select_option", "intent": "Pick Space Exploration from Industry",
     "params": {"dropdown_label": "Industry",
                "option_label": "Space Exploration"}, ...}

Self-check after generation: for every fill_field/select_option step,
look at the source phrase that produced it. Did it name a literal
value? If yes, that string MUST appear in `params.value` /
`params.option_label`. If the source didn't provide a value (e.g. the
plan says "Enter your password" with no actual password), still emit
`fill_field` but expect the runner to use a placeholder or fail with
an explicit message — better than silently typing "".

RUNTIME HINTS — DO NOT DECOMPOSE:

Some source plans include guidance for the runtime / browser /
operator that is NOT a step the model needs to take. These are
PROMPT METADATA, not actions. DO NOT emit a step for any of these:

   • "Wait N seconds for the page to load", "Pause N seconds before
     interacting", "Let the page fully render", "Allow N seconds for
     hydration"
       → omit. The runner's navigate handler already waits up to 18s
         for first paint, the form handler adds another 4s settle,
         and find_form_target re-screenshots until the page is non-
         blank. Decomposing a "wait" instruction as ``extract_data``
         then runs the listing-data extractor on a not-yet-populated
         page and rejects the lead with REJECTED_INCOMPLETE — wrong
         outcome, halts the plan.

   • "FIRST ACTION RULE: ...", "BENCHMARK PREAMBLE", "INPUTS:",
     "PERMITTED ACTIONS:", "STRICTLY PROHIBITED:" headers and the
     bullet lists under them — these are operator notes, not steps.

   • "Open a browser at <URL>", "The runtime has already opened the
     browser at <URL>" — omit. The runner / host opens the browser
     externally; the first decomposable step in this case is
     whatever the source plan does AFTER the browser is open.

   • "Do NOT use Developer Tools", "Do NOT execute JavaScript",
     "Do NOT read page source" and similar capability constraints
     — omit. The brain's action vocabulary doesn't include these
     primitives in the first place.

   When in doubt: if the instruction describes an environment
   precondition or operator constraint rather than a clickable /
   typeable / scrollable action, omit it. The plan is shorter and
   more reliable for it.

VERB → STEP-TYPE MAPPING (FORM FLOWS):
   "log in", "sign in", "authenticate"
       → fill_field for each credential (with the literal username and
         password preserved in params.value as shown above), then
         submit for the login button.
   "enter X in the Y field", "type X into Y", "fill in Y with X", "set Y to X",
   "input X for Y"
       → fill_field with params={"label": "Y", "value": "X"}.
   submit handles ANY single labelled clickable on a non-listings page —
   buttons, nav links, tab items, menu items. The runner's
   find_form_target locates one labelled element by visible text. To
   help it pick the right element when several share a label, the
   step MUST include params.kind classifying the visual affordance:

   • params.kind="button" — form-submit / call-to-action buttons
       "click the Submit button", "click Save", "click the primary action button",
       "press Continue", "submit the form", "click the {Edit/Cancel} button"
       → submit with params={"label": "<button text>", "kind": "button"}.

   • params.kind="nav_link" — sidebar / top-nav navigation links and
                              "go to the X page" in-app transitions
       "click the {Y} navigation link", "click the {Y} link in the sidebar",
       "go to the {Y} page" (when {Y} is a tab/nav/menu item, NOT a URL),
       "open the {Y} section"
       → submit with params={"label": "Y", "kind": "nav_link"}.

   • params.kind="tab" — horizontal/vertical tab bars on a page
       "click the {Y} tab", "switch to the {Y} tab"
       → submit with params={"label": "Y", "kind": "tab"}.

   • params.kind="menu_item" — entries inside an open menu / kebab /
                               dropdown / context menu
       "click the {Y} menu item", "select {Y} from the menu",
       "choose {Y} from the kebab menu"
       → submit with params={"label": "Y", "kind": "menu_item"}.

   • params.kind="row_link" — a record-name link inside a data table
                              that opens that row's detail page. The
                              click target is the row's primary cell
                              text (often a name, ID, or title), NOT
                              the row checkbox, status badge, sort
                              header, or inline action icon.
       "click the lead {Y} to open its detail/edit page",
       "open the {Y} record from the {leads/orders/tickets} list",
       "click the first row whose Status is Qualified",
       "click the row for record X in the table"
       → submit with params={"label": "Y", "kind": "row_link"}.
     Use this whenever the source plan describes selecting / opening
     a single record from a multi-row list/table by its name or by a
     filtering attribute (status, owner, etc.).

   • params.kind="cell_link" — a hyperlinked value inside a table
                               cell that isn't the row's primary
                               record name (e.g. an account-name link
                               in the Account column, an email link in
                               the Email column).
       "click the {account-name} link in the {Account} column",
       "open the {related record} link in the row"
       → submit with params={"label": "Y", "kind": "cell_link"}.

   When in doubt, default to params.kind="button". The runner accepts
   submit steps without a kind and treats them as buttons for
   backward compatibility.
   "select X from the Y dropdown", "choose X under Y", "pick X in the Y selector"
       → select_option with params={"dropdown_label": "Y", "option_label": "X"}.
   "click the first / next / nth result/row/listing/card/job/product/property"
       → click (this IS a listings click — many similar items on one page,
       runner picks the next un-extracted one). DO NOT use click for nav
       links, buttons, or any single-element clickable.
   "right-click on Y", "open the context menu on Y",
   "press the right mouse button on Y", "secondary-click Y"
       → right_click with params={"label": "Y"}.
       Use ONLY when the workflow needs the browser's native context
       menu on a specific labelled target — typically "Open Link in
       New Tab", "Copy Link", "Inspect Element", or app-defined
       context menus on table rows / grid cells. The runner finds the
       element via find_form_target then dispatches a right mouse
       button click at its center. Do NOT use right_click as a generic
       "click harder" — a plain click step is correct unless the
       source explicitly calls for a right-click verb or a context
       menu is the only way to reach the next action.

When the source text says "Click the {field-name} field and enter {value}", emit
a SINGLE fill_field step (label={field-name}, value={value}) — NOT a click step.
The runner clicks the field as part of fill_field.

RULES:
- Each step: ONE action, ONE sentence, under 20 words
- POSITIVE framing only: state what to click/type, never what to avoid
- Include WHAT + WHERE: name the visible label of the target AND the
  region of the page (e.g. left sidebar, top header, modal dialog,
  results card) when it disambiguates. Use the labels the source plan
  itself names — never invent application-specific filter names.
- For navigate steps: include the FULL URL (http:// or https://) in the intent
  AND mirror the URL in `params.url`. The duplication is intentional — the
  runner will recover from `params.url` if the intent string ever drifts.
  CRITICAL RULE — no exceptions:
    • If the source plan DOES contain an http(s):// URL, emit `navigate`
      with the full URL in BOTH `intent` and `params.url`.
    • If the source plan does NOT contain an http(s):// URL for a step,
      you MUST emit `submit` with params={"label": "<page name>"}.
      The runner clicks the matching nav link — there is no URL to load.
  WORKED EXAMPLES:
    Source: "1. Go to https://app.example.com/login"
    → {"type": "navigate", "intent": "Navigate to https://app.example.com/login",
       "params": {"url": "https://app.example.com/login"}, ...}
    Source: "1. Navigate to https://app.example.com for billing"
    (do NOT paraphrase the URL away — keep the literal string)
    → {"type": "navigate", "intent": "Navigate to https://app.example.com",
       "params": {"url": "https://app.example.com"}, ...}
    Source: "3. Go the Leads Page"
    → {"type": "submit", "intent": "Open the Leads section",
       "params": {"label": "Leads"}, ...}
    Source: "5. Open Settings"
    → {"type": "submit", "intent": "Open the Settings section",
       "params": {"label": "Settings"}, ...}
  This rule is verifiable by inspecting your own output: every `navigate`
  step's intent string MUST contain the substring "http://" or "https://"
  AND `params.url` MUST contain the same URL. If it doesn't, the step type
  is wrong — switch it to `submit`.
- Extraction steps (reading screen) use claude_only=true
- For fill_field / submit / select_option, ALWAYS populate `params`
  (label/value/dropdown_label/option_label) using the labels the source
  plan provides. The runner trusts `params` over the prose.
- HINTS — emit `hints` whenever the source plan gives an explicit URL
  expectation OR a strong spatial cue. Three structured fields are
  recognised by the runner today:
    • `hints.expect_url_contains` — list of substrings the post-click
      URL MUST contain. Emit ONE entry per literal URL clue the source
      plan names. Examples:
        Source: "click 'Active' in the status sidebar to filter the
                 items list ... the URL should now include status=Active"
        → hints={"expect_url_contains": ["status=Active"]}
        Source: "click Apply ... URL now also includes priority=High"
        (already has status=Active from prior step — keep BOTH)
        → hints={"expect_url_contains": ["status=Active",
                                          "priority=High"]}
        Source: "land on an item detail page whose URL pattern
                 /items/<id>"
        → hints={"expect_url_contains": ["/items/"]}
      Without this hint, a click that "changed state" but landed on the
      wrong URL slides through as success — a common failure mode when
      a filter pill toggles a tab visually but doesn't navigate.
    • `hints.expect_url_excludes` — list of substrings the post-click
      URL MUST NOT contain. Useful for the "shouldn't drift to a
      detail page" cases:
        Source: "click 'Active' in the sidebar — do NOT
                 land on an individual item detail"
        → hints={"expect_url_contains": ["status=Active"],
                 "expect_url_excludes": ["/items/"]}
      Only emit these for submit / click / navigate steps (URL is
      meaningful). Skip on fill_field / select_option / extract_data.
    • `hints.fallback_url` — when the click / submit has a STRUCTURAL
      ALTERNATIVE that the agent can navigate to directly if the
      click keeps misfiring. The runner's recovery layer will REPLACE
      the click with a direct navigate after 2+ failures.
        Source: "click 'Active' to filter the items list ... the
                 URL should now include status=Active"
        → hints={"expect_url_contains": ["status=Active"],
                 "fallback_url": "/items?status=Active"}
        Source: "click the item's title to open its detail page
                 (URL ends in /items/<id>)"
        → hints={"expect_url_contains": ["/items/"]}
        (no fallback_url — the <id> isn't predictable at plan time)
      Emit fallback_url ONLY when the equivalent URL is PREDICTABLE
      from the plan text alone — typically filter / view / static-
      nav clicks where the URL pattern is named in the source. Skip
      for clicks on dynamic data (record rows, comment replies, etc.)
      where the destination URL contains an ID we won't know.

      CAVEAT — fallback_url assumes the site is URL-DRIVEN: that
      navigating directly to the URL produces the same observable
      state as clicking the original element. Many SPA admin
      consoles (CRMs, ticket systems, dashboards) accept the URL
      visually but strip the query string and apply default state,
      so direct navigation does NOT match the click's intended
      effect — the click handler maintains in-app state that the
      route doesn't. When you can see from the source plan that
      the site routes only via clicks (e.g. "the URL bar updates
      after you click but typing the same URL in the address bar
      loads the default view"), DO NOT emit fallback_url; let the
      frontier critic propose a different recovery instead. Default
      to emitting fallback_url unless the source explicitly warns
      about URL-stripping or SPA-only state.
- The "reverse" field must be a CUA-executable instruction
- Set section="setup", section="extraction", or section="pagination"
- Set required=true for all setup/filter/form steps (this is the default)
- Set gate=true for the verification step at the end of setup
- Set gate=true on EVERY verification-flavored extract_data step,
  not just end-of-setup gates. If the step's intent contains any of
  "verify", "confirm", "check that", "make sure", "wait for",
  "read the URL", "ensure" — it is a verification step. Set
  gate=true so the runner dispatches to verify_gate (no recipe
  schema validation). Authoritative extraction (read structured
  fields, scrape, harvest data) keeps gate=false.

LOOP STRUCTURE:
  Extraction loop: click → URL → scroll → extract → back → loop(target=click, count=30)
  Pagination loop: paginate → loop(target=click, count=10)
  The listing loop runs INSIDE the pagination loop.
  Form-only flows do not need loops.

  ALWAYS emit a concrete loop_count integer — never 0, never omitted:
    • Extraction-section loops: 30 (typical page has ≤25 cards; +20% covers
      lazy-loaded carousel rotation and dedup-skipped re-clicks).
    • Pagination-section loops: 10 (typical scrape covers ≤10 pages).
  If the source plan mentions an explicit count ("scan 50 listings", "go
  through all 8 pages"), use that number instead. Never default to 0 —
  the runtime falls back to a 200-iter ceiling and burns ~$2.50/loop on
  wasted dedup-skipped retries.

PLAIN TEXT PLAN:
{plan_text}

STEP TYPES:
- navigate: Go to an http(s):// URL — include the FULL URL in the intent
            (budget=3). REQUIRES an http:// or https:// URL in the source
            plan. For in-app navigation ("Go to the Leads page", "Open
            Settings tab"), use submit instead — the runner clicks the
            matching nav link.
            Optional params={"wait_after_load_seconds": <int>} when the page
            needs a longer first-paint wait (e.g. proxied SPA cold-start,
            heavy CRM splash). Default is 18s; use the param only when the
            source plan explicitly says the page is slow to load.
- filter: Click a filter option (budget=8, grounding=true, required=true, section="setup")
- click: Click an element on a listings/results page (budget=8, grounding=true)
- scroll: Scroll until target content visible (budget=10, section="extraction")
- extract_url: Read URL from address bar (claude_only=true, budget=0, section="extraction")
- extract_data: Inspect page and read structured data OR verify state.
                CRITICAL — these are TWO DISTINCT MODES (issue #244):
                  • Authoritative extraction (claude_only=true, gate=false):
                    runs the recipe-schema-validated deep-extract path.
                    Use when the source plan asks to *read structured
                    fields* off a detail/listing page (year, make,
                    price, title, lead name, …) for downstream use.
                  • Verification (claude_only=true, gate=true):
                    runs the verify_gate path — Claude reads the
                    screenshot and answers PASS/FAIL with a reason.
                    NO recipe schema, NO required-fields validation,
                    NO REJECTED_INCOMPLETE cascade. Use whenever the
                    source plan asks to *verify, confirm, check, make
                    sure, wait for, read the URL* — anything that
                    asserts a condition holds rather than producing a
                    structured row of data.
                Get this wrong and the plan halts on a search/list
                page where the schema's required fields don't exist
                in the rendered text. Set gate=true on every
                verification extract_data, not just end-of-setup.
- navigate_back: Go back (budget=3, section="extraction")
- paginate: Click Next page (budget=10, grounding=true, section="pagination")
- loop: Jump back to step index. ALWAYS emit a concrete loop_count integer:
        extraction-section: 30 (or source plan's explicit listing count)
        pagination-section: 10 (or source plan's explicit page count)
        Never 0; never omit. See LOOP STRUCTURE above for full rationale.
- fill_field: Click a labelled input and type a value
              (budget=4, params={"label": "<visible field label>", "value": "<text to type>"})
- submit: Click a SINGLE LABELLED CLICKABLE on a non-listings page — buttons
          (Login / Save / Submit / Update / Continue), navigation links,
          tab items, menu items, dock icons, action links.
          NOT for "click the next listing/result" — use `click` for that.
          (budget=4, params={"label": "<visible button or link text>",
                             "aliases": ["<synonym1>", "<synonym2>", ...]})
          Use `aliases` ONLY for primary submit buttons whose copy varies
          across products: "Save" / "Save Changes" / "Apply" all do the
          same thing. Do NOT add aliases for nav links, tab clicks, or
          any other unique-label element — they cause false matches.
- select_option: Open a dropdown and pick an option by visible text
                 (budget=6, params={"dropdown_label": "<dropdown name>",
                                    "option_label": "<option text>"})

OUTPUT FORMAT — emit ONE valid JSON object and nothing else.

CRITICAL: do NOT include prose preamble ("Here's the decomposition:"),
classification commentary ("I'll classify this as..."), epilogue
("Let me know if..."), or markdown fences. The first character of your
response MUST be `{` and the last MUST be `}`.

Schema:

{
  "shapes": ["listings" | "form" | "workflow" | "inspect", ...],
  "pagination_url_template": "{base}/page-{n}/",   // OPTIONAL — see PAGINATION URL TEMPLATE below
  "steps": [ <list of step objects, same shape as before> ]
}

The "shapes" array MUST contain at least one of the four canonical tokens.
Use multiple tokens when the plan mixes shapes (e.g. log in then extract
listings → ["form", "listings"]).

PAGINATION URL TEMPLATE — emit when the source plan contains an explicit
example or hint about how pagination URLs are formed. Use the literal
placeholders ``{base}`` (the setup-navigate URL stripped of trailing slash)
and ``{n}`` (the 1-indexed page number). Common patterns observed in the
wild:

  * Path segment with prefix:         "{base}/page-{n}/"
  * Generic query parameter:          "{base}?page={n}"
  * Ampersand append (mid-query):     "{base}&p={n}"
  * Indexed page-N suffix:            "{base}/p/{n}"

If the source plan shows either an example URL with a page number
(``...?page=2``, ``.../page-2/``, ``.../p/2``) OR explicit prose like
"page URLs use the ?page=N query parameter", emit ``pagination_url_template``
with the inferred form. When the source plan says nothing about URL
shape (or pagination is cursor-based / JS-driven), OMIT the field
entirely so the orchestrator falls back to the default
``{base}/page-{n}/``.

Never invent a template based on the domain alone — only emit when the
source plan provides direct evidence of the URL shape.

Backward-compatible fallback: if you cannot reliably classify the plan,
emit a bare JSON array of step objects (no top-level "shapes"). The
parser accepts both forms but prefers the object form.
"""


class PlanDecomposer:
    """Decomposes plain text plans into micro-intents using Claude Sonnet."""

    def __init__(self, api_key: str = "", model: str = "claude-opus-4-7"):
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

        Used by callers (e.g. the host integration's MantisOrchestratedBackend) that
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
        prompt_version = "v32_pagination_url_template_neutral"  # Bump this when DECOMPOSE_PROMPT changes
        plan_hash = hashlib.md5(f"{prompt_version}:{plan_text}".encode()).hexdigest()[:8]
        cache_path = (
            cache_path_template.replace("{hash}", plan_hash)
            if cache_path_template
            else None
        )
        if cache_path and os.path.exists(cache_path):
            try:
                cached = json.loads(open(cache_path).read())
                # Same dual-shape handling as the live path: either a bare
                # list of steps, or {"shapes": [...], "steps": [...]}.
                if isinstance(cached, dict):
                    cached_steps = cached.get("steps", [])
                    cached_shapes = cached.get("shapes", [])
                    cached_template = str(
                        cached.get("pagination_url_template", "") or ""
                    )
                else:
                    cached_steps = cached
                    cached_shapes = []
                    cached_template = ""
                plan = MicroPlan(source_plan=plan_text, domain=domain)
                plan.shapes = self._normalize_shapes(cached_shapes)
                if "{base}" in cached_template and "{n}" in cached_template:
                    plan.pagination_url_template = cached_template
                for s in cached_steps:
                    plan.steps.append(self._build_intent(s))

                # Issue #605: re-run loop-target normalization on cache
                # load too. Cached plans from before this fix have
                # ``loop_target: null`` (-> -1) on loop steps, which
                # causes the runner to self-spin. The fix is idempotent,
                # so running it on every load (cached or fresh) is safe
                # and means we don't have to invalidate all existing
                # cache files when shipping plan-normalization fixes.
                self._fix_loop_targets(plan)
                # #614: classify loop bodies for the fan-out runner.
                self._classify_loop_groups(plan)

                logger.info(f"Loaded cached micro-plan: {cache_path} ({len(plan.steps)} steps)")
                return plan
            except Exception:
                pass

        logger.info(f"Decomposing plan with {self.model} ({len(plan_text)} chars)")
        # Use replace() instead of format() — the prompt has literal `{...}`
        # JSON examples (params={"label": ...}) that confuse str.format.
        prompt = DECOMPOSE_PROMPT.replace("{plan_text}", plan_text)

        request_body = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }
        t0 = time.monotonic()
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=request_body,
            timeout=60,
        )

        if resp.status_code != 200:
            raise RuntimeError(f"Decompose API error: {resp.status_code} {resp.text[:200]}")

        # #523 PR B-2 — capture this call as a modelio record when an
        # upstream caller has published a layer context (planner). Silent
        # no-op otherwise (default path until the wrap fires in PR B-2-
        # integration or later). Best-effort: telemetry never breaks
        # decomposition.
        try:
            from .observability.modelio import (
                current_modelio_context,
                record_anthropic_modelio,
            )
            ctx = current_modelio_context()
            if ctx is not None:
                record_anthropic_modelio(
                    request_payload=request_body,
                    response_json=resp.json(),
                    duration_ms=int((time.monotonic() - t0) * 1000),
                    ctx=ctx,
                )
        except Exception as exc:  # noqa: BLE001 — observability never fatal
            logger.debug("plan_decomposer modelio capture failed: %s", exc)

        text = ""
        for block in resp.json().get("content", []):
            if block.get("type") == "text":
                text = block["text"].strip()
                break

        # Parse JSON — tolerant of:
        #   • bare ```json / ``` fences (legacy)
        #   • prose preamble before the JSON ("Here's the decomposition:")
        #   • prose epilogue after the JSON
        # Issue #112: Claude often prepends explanation text. Robustly find
        # the outermost JSON object or array in the response.
        parsed = _extract_json_payload(text)
        if parsed is None:
            raise RuntimeError(
                f"Decomposer response did not contain parseable JSON. "
                f"First 300 chars: {text[:300]!r}"
            )
        # Two accepted response shapes:
        #   {"shapes": [...], "steps": [...]}  ← preferred (LLM classified)
        #   [...]                              ← legacy bare array
        if isinstance(parsed, list):
            steps_raw = parsed
            shapes_raw: Any = []
            template_raw = ""
        elif isinstance(parsed, dict):
            steps_raw = parsed.get("steps", [])
            shapes_raw = parsed.get("shapes", [])
            # #629: optional pagination URL template — only present when
            # the LLM inferred one from the source plan.
            template_raw = str(parsed.get("pagination_url_template", "") or "")
        else:
            raise ValueError(
                f"Decomposer returned unexpected JSON type: {type(parsed).__name__}"
            )

        plan = MicroPlan(source_plan=plan_text, domain=domain)
        plan.shapes = self._normalize_shapes(shapes_raw)
        # Only accept the template when it contains both placeholders;
        # otherwise it's malformed and the orchestrator will reject it
        # anyway (better to surface in the decompose log).
        if "{base}" in template_raw and "{n}" in template_raw:
            plan.pagination_url_template = template_raw
            logger.info(
                "  [decomposer] inferred pagination_url_template=%s",
                template_raw,
            )
        for s in steps_raw:
            plan.steps.append(self._build_intent(s))

        if plan.shapes:
            logger.info(f"  [decomposer] Claude classified shape(s): {plan.shapes}")
        else:
            logger.info("  [decomposer] no shape classification returned (legacy schema)")

        # Fix 3: Validate and fix loop targets — must point to the click step
        self._fix_loop_targets(plan)
        # #614: classify loop bodies for the fan-out runner.
        self._classify_loop_groups(plan)
        # Fix 4 (#209 Symptom 4): repair urlless navigate steps by injecting
        # the matching source-plan URL into ``params.url``. The v20 prompt
        # asks Claude to mirror the URL there already; this pass catches
        # the residual cases where it paraphrases the URL away entirely.
        self._repair_navigate_urls(plan)
        # Fix 5 (#578): inject ``expect_url_contains`` hints onto
        # navigate-verify gates so the runner's #563 URL short-circuit
        # can fire — skips a ~$0.01-0.02 + ~10-15s Claude verify call
        # per gate whose preceding navigate had a literal URL with
        # distinctive path segments.
        self._inject_gate_url_hints(plan)

        # Cache the full parsed structure (object or legacy array) so the
        # cached path round-trips through both schemas. #629: also persist
        # the inferred pagination URL template when set, so cached plans
        # don't lose the inference on rehydration.
        if plan.shapes or plan.pagination_url_template:
            cache_payload: Any = {"shapes": plan.shapes, "steps": steps_raw}
            if plan.pagination_url_template:
                cache_payload["pagination_url_template"] = (
                    plan.pagination_url_template
                )
        else:
            cache_payload = steps_raw
        if cache_path:
            try:
                with open(cache_path, "w") as f:
                    json.dump(cache_payload, f, indent=2)
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

    # ``right_click`` (#373) shares the FormHandler dispatch shape —
    # find_form_target finds one labelled element, the handler then
    # dispatches an ``Action(CLICK, button="right")`` — but the
    # FORM_STEP_TYPES defaults (section=setup, required=True) are
    # wrong for right-click, which is typically an extraction-flow
    # primitive (open a context menu mid-extract). The build path
    # below treats it as its own group with extraction-friendly
    # defaults.

    # Verification-language triggers (issue #244). When an
    # ``extract_data`` step's intent contains any of these substrings
    # (case-insensitive) and the source dict didn't set ``gate``
    # explicitly, ``_build_intent`` auto-promotes ``gate=True`` so
    # the runner dispatches the step through ``verify_gate`` rather
    # than the deep-extract / recipe-schema path. Without this,
    # auto-injected verification steps like *"Verify the page
    # loaded"* hit the marketplace_listings recipe schema on a
    # search-results URL and reject as REJECTED_INCOMPLETE,
    # halting the plan.
    _VERIFICATION_INTENT_TRIGGERS = (
        "verify",
        "confirm",
        "check that",
        "make sure",
        "wait for",
        "read the url",
        "ensure",
    )

    @staticmethod
    def _is_verification_intent(intent: str) -> bool:
        """Return True iff ``intent`` reads as verification language —
        i.e. asserting a state holds rather than reading authoritative
        structured data. Case-insensitive substring match against
        :data:`_VERIFICATION_INTENT_TRIGGERS`. Used by
        :meth:`_build_intent` as the safety net behind issue #244."""
        if not intent:
            return False
        text = intent.casefold()
        return any(t in text for t in PlanDecomposer._VERIFICATION_INTENT_TRIGGERS)

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
            elif step_type in (
                "click", "scroll", "extract_url", "extract_data",
                "navigate_back", "right_click", "collect_urls",
            ):
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

        # Issue #244: auto-promote verification ``extract_data`` to
        # ``gate=True`` when the source dict didn't say either way.
        # Explicit operator overrides (``gate`` present in ``s``)
        # always win — the heuristic must not silently flip a
        # plan-author's deliberate choice.
        if "gate" in s:
            gate = bool(s["gate"])
        elif step_type == "extract_data" and PlanDecomposer._is_verification_intent(s.get("intent", "")):
            gate = True
        else:
            gate = False

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
            claude_only=s.get(
                "claude_only",
                step_type in ("extract_url", "extract_data", "detect_visible"),
            ),
            # Some plan-authoring styles (and Claude's older response
            # shapes) put loop_target / loop_count inside ``params``
            # rather than at the top level. Read top-level first;
            # fall back to ``params`` so both forms work. Without this
            # fallback, the canonical boattrader_scrape_urlnav plan
            # silently degraded to loop_target=-1 / loop_count=0 →
            # ``_handle_loop_step`` self-spun on its own index and
            # the inner extraction loop never iterated past the first
            # card (1 lead vs the expected 25-40). Live repro
            # 2026-05-24 verification run 20260524_172252_88d4e49e.
            loop_target=s.get(
                "loop_target",
                (s.get("params") or {}).get("loop_target", -1),
            ),
            loop_count=s.get(
                "loop_count",
                (s.get("params") or {}).get("loop_count", 0),
            ),
            section=section,
            required=s.get("required", default_required),
            gate=gate,
            params=params,
            hints=hints,
            guard=str(s.get("guard", "") or ""),
            out_var=str(s.get("out_var", "") or ""),
        )

    @staticmethod
    def _repair_navigate_urls(plan: MicroPlan) -> None:
        """Inject lost source-plan URLs into urlless navigate steps (#209 Symptom 4).

        The v20 ``DECOMPOSE_PROMPT`` instructs Claude to mirror every
        literal URL from the source plan into both the navigate step's
        ``intent`` and ``params.url``. Empirically, even with explicit
        rules and worked examples, Claude occasionally paraphrases the
        URL away ("Navigate to the leads management system") leaving
        the runner with no URL to load.

        This pass extracts ``https?://`` URLs from the source plan in
        source order, walks the decomposed navigate steps, and pairs
        each urlless navigate step (intent AND params.url both empty
        of an https URL) with the next unconsumed source URL by
        injecting it into ``params.url``. The runtime navigate handler
        falls back to ``params.url`` when the intent lacks a URL, so
        the repaired step executes correctly.

        A repaired step logs at WARNING (so prompt regressions are
        visible). An unrepairable step — more urlless navigates than
        source URLs — logs at ERROR; that's a decomposer bug worth a
        prompt iteration. Repair is best-effort and never fails the
        decompose, because production traffic must still flow.
        """
        url_re = re.compile(r'https?://[^\s"\'<>]+')
        source_urls = url_re.findall(plan.source_plan or "")
        if not source_urls:
            return

        # Pass 1: build the queue of source URLs the decomposer hasn't
        # already covered. We walk healthy navigate steps in plan order and
        # remove the first remaining occurrence of each URL they reference,
        # so duplicates in the source plan are honoured (each occurrence
        # can satisfy at most one navigate step).
        remaining = list(source_urls)
        for step in plan.steps:
            if step.type != "navigate":
                continue
            covered: str | None = None
            intent_match = url_re.search(step.intent or "")
            if intent_match:
                covered = intent_match.group()
            elif isinstance(step.params, dict):
                params_url = step.params.get("url")
                if isinstance(params_url, str):
                    pm = url_re.search(params_url)
                    if pm:
                        covered = pm.group()
            if covered and covered in remaining:
                remaining.remove(covered)

        # Pass 2: repair urlless navigate steps from the unused queue.
        cursor = 0
        for i, step in enumerate(plan.steps):
            if step.type != "navigate":
                continue
            if url_re.search(step.intent or ""):
                continue
            params = step.params if isinstance(step.params, dict) else {}
            existing = params.get("url")
            if isinstance(existing, str) and url_re.search(existing):
                continue

            if cursor >= len(remaining):
                logger.error(
                    f"  [decomposer] navigate step #{i} lost its URL and no "
                    f"source URL remains to repair from: {step.intent[:80]!r}"
                )
                continue

            recovered = remaining[cursor]
            cursor += 1
            if not isinstance(step.params, dict):
                step.params = {}
            step.params["url"] = recovered
            logger.warning(
                f"  [decomposer] navigate step #{i} dropped its URL "
                f"({step.intent[:60]!r}); repaired with source URL "
                f"{recovered}. Tighten the decomposer prompt if this recurs."
            )

    @staticmethod
    def _extract_url_hint_segments(url: str) -> list[str]:
        """Pull distinctive path segments from a URL for #563 short-circuit.

        Returns segments containing a digit OR a hyphen — captures
        boattrader-style filter slugs (``zip-33101``, ``state-fl``,
        ``by-owner``, ``radius-25``) and excludes bare resource
        names (``boats``, ``discover``) that would match unrelated
        pages on the same domain.

        Returns ``[]`` when the URL has no distinctive segments —
        the runner's short-circuit then falls through to vision
        verify (no false positives possible).
        """
        if not url:
            return []
        # Strip scheme + host so we walk only the path. Query strings
        # and fragments are intentionally excluded — they change
        # across navigations within the same page (?page=2, #anchor)
        # and would cause the short-circuit to false-negative.
        m = re.search(r"https?://[^/]+(/[^?#]*)", url)
        if not m:
            return []
        path = m.group(1)
        segments = [s for s in path.split("/") if s]
        # Distinctive = has digit OR hyphen. A bare "boats" matches
        # too many pages; "zip-33101" / "state-fl" / "by-owner" only
        # match the intended filtered listings.
        distinctive = [
            s for s in segments
            if any(ch.isdigit() for ch in s) or "-" in s
        ]
        return distinctive

    @staticmethod
    def _inject_gate_url_hints(plan: MicroPlan) -> None:
        r"""#578: emit ``expect_url_contains`` hints on navigate-verify gates.

        Walks the plan; for each ``extract_data`` step with
        ``gate=True`` and ``claude_only=True``, finds the nearest
        preceding ``navigate`` step. If that navigate carries a
        literal URL (in ``params.url`` or ``intent``), pulls
        distinctive path segments via :meth:`_extract_url_hint_segments`
        and merges them into the gate's ``hints.expect_url_contains``.

        Idempotent: if the plan author already supplied hints, they're
        preserved as-is. Empty / generic URLs (no distinctive segments)
        are skipped silently — the runner's short-circuit handles
        absent hints by falling through to vision.

        Activates the ``_try_url_shortcircuit_gate`` path shipped in
        PR #577 — that helper short-circuits the ~10-15s + \$0.01-0.02
        Claude verify call when these hints are present and the URL
        matches.
        """
        url_re = re.compile(r"https?://[^\s\"'<>]+")
        last_navigate_url: str = ""
        for step in plan.steps:
            if step.type == "navigate":
                # Prefer params.url, then any URL in the intent.
                params = step.params or {}
                candidate = str(params.get("url") or "").strip()
                if not candidate:
                    m = url_re.search(step.intent or "")
                    if m:
                        candidate = m.group(0)
                if candidate:
                    last_navigate_url = candidate
                continue
            if not (step.type == "extract_data" and step.gate and step.claude_only):
                continue
            if not last_navigate_url:
                continue
            hints = step.hints or {}
            existing = hints.get("expect_url_contains")
            if existing:
                # Plan author already supplied hints — don't override.
                continue
            segments = PlanDecomposer._extract_url_hint_segments(last_navigate_url)
            if not segments:
                continue
            new_hints = dict(hints)
            new_hints["expect_url_contains"] = segments
            step.hints = new_hints
            logger.info(
                "  [decomposer] injected expect_url_contains=%s on gate "
                "step (intent=%r) from navigate URL %s",
                segments, step.intent[:60], last_navigate_url[:120],
            )

    @staticmethod
    def _fix_loop_targets(plan: MicroPlan):
        """Fix 3: Ensure loop targets point to the click step, not extract_url.

        The decomposer often generates loop→extract_url instead of loop→click.
        Find the first extraction click step and retarget all loops to it.

        Issue #605: Claude also frequently OMITS ``loop_target`` entirely
        on loop steps, so the field defaults to ``-1`` at parse time
        (line ``loop_target=s.get("loop_target", -1)``). The runner's
        loop handler then computes ``target = state.step_index`` (the
        loop step's own index), causing the loop to self-spin instead
        of jumping back into the iteration body. Only one pass through
        the body ever runs and the plan extracts at most one lead.
        Default unset targets to ``click_idx`` so listings-extraction
        plans loop on click→extract→back→loop as designed.
        """
        click_idx = None
        for i, s in enumerate(plan.steps):
            if s.type == "click" and s.section == "extraction":
                click_idx = i
                break

        if click_idx is None:
            return

        for s in plan.steps:
            if s.type != "loop":
                continue
            if s.loop_target < 0:
                # Claude omitted loop_target; default to the extraction
                # click step. Without this the loop self-spins
                # (see issue #605).
                logger.info(
                    "  [fix] Loop target unset (-1) → %d (click step)",
                    click_idx,
                )
                s.loop_target = click_idx
            elif s.loop_target != click_idx and abs(s.loop_target - click_idx) <= 2:
                # Only fix close targets (off by 1-2, typical decomposer error)
                logger.info(
                    "  [fix] Loop target %d → %d (click step)",
                    s.loop_target, click_idx,
                )
                s.loop_target = click_idx

    # ── #614 loop classifier ────────────────────────────────────────────

    @staticmethod
    def _classify_loop_groups(plan: MicroPlan) -> None:
        """Tag every ``loop`` step's body with a parallelizability shape.

        Must run AFTER :meth:`_fix_loop_targets` so ``loop_target`` is
        normalized to point at the body's entry step.

        Classification rules (#614):

          * ``parallelizable_url_collect`` — body starts with a ``click``
            step in section ``extraction`` AND the body contains an
            ``extract_url`` step. This is the canonical extraction-loop
            shape from the decomposer prompt: each iteration depends only
            on the URL of one listing card.
          * ``parallelizable_pagination`` — body's terminal action is
            ``paginate``. Each page is an independent unit (this is the
            shape ``deploy/modal/modal_cua_server.py:_make_page_task``
            already partitions, just under a different rubric).
          * ``sequential`` — fallback. Forms, login flows, anything with
            stateful dependencies between iterations.

        Pure analysis: writes only ``plan.loop_groups`` and never
        mutates ``plan.steps`` or any other field.
        """
        plan.loop_groups = []
        for idx, step in enumerate(plan.steps):
            if step.type != "loop":
                continue
            target = step.loop_target if step.loop_target >= 0 else idx
            # Guard against a loop that targets ahead of itself (Claude
            # occasionally emits this when the body is empty); treat as
            # zero-length body — un-fan-outable by definition.
            start = max(0, min(target, idx))
            end = idx  # half-open
            body = plan.steps[start:end]
            shape = PlanDecomposer._loop_body_shape(body)
            plan.loop_groups.append(
                LoopGroup(loop_step_idx=idx, body_range=(start, end), shape=shape)
            )
            logger.info(
                "  [loop-classifier] step %d body=[%d, %d) → shape=%s",
                idx, start, end, shape,
            )

    @staticmethod
    def _loop_body_shape(body: list[MicroIntent]) -> str:
        if not body:
            return "sequential"
        types = [s.type for s in body]
        # Pagination outer loop: the body contains a ``paginate`` step.
        # The canonical shape (``plan_decomposer.py:588-592``) is
        # ``... → paginate → loop`` — the pagination loop's body
        # frequently includes the inner extraction body because
        # ``_fix_loop_targets`` rewinds BOTH loop_targets to the
        # extraction click step. So presence-of-paginate is a stronger
        # signal than "body looks like extraction" for the outer loop:
        # only the outer loop will ever have a paginate step in its
        # body (the inner extraction loop body is click → extract_url
        # → scroll → extract_data → navigate_back, no paginate).
        # Check this BEFORE the url-collect case so a nested
        # outer-pagination loop doesn't get mis-tagged as url-collect.
        if "paginate" in types:
            return "parallelizable_pagination"
        # URL-collect extraction loop: body opens on a ``click`` in the
        # extraction section AND contains ``extract_url``. This is the
        # canonical shape the prompt produces for "find each listing →
        # open → extract" plans. The presence of ``extract_url`` proves
        # the per-iteration unit is anchored on a distinct URL — exactly
        # the partition key the fan-out runner needs (#615 collects them
        # up front so the fan-out workers can navigate directly).
        first = body[0]
        if (
            first.type == "click"
            and first.section == "extraction"
            and "extract_url" in types
        ):
            return "parallelizable_url_collect"
        return "sequential"

    # ── Plan-shape extraction (parsed from Claude's JSON output) ────────

    # Canonical shape vocabulary. The prompt asks Claude to populate
    # ``shapes: [...]`` using only these tokens; the parser drops any
    # other tokens silently so a model hallucination can't poison
    # downstream observability.
    KNOWN_PLAN_SHAPES: ClassVar[frozenset[str]] = frozenset(
        {"listings", "form", "workflow", "inspect"}
    )

    @classmethod
    def _normalize_shapes(cls, raw: Any) -> list[str]:
        """Filter Claude's reported shapes to the canonical vocabulary,
        preserving the canonical display order so consumers see a
        deterministic ordering regardless of how Claude listed them.
        """
        canonical_order = ("listings", "form", "workflow", "inspect")
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, (list, tuple, set)):
            return []
        seen = {str(s).strip().lower() for s in raw}
        return [s for s in canonical_order if s in seen and s in cls.KNOWN_PLAN_SHAPES]

    # ── No regex semantic post-processing ───────────────────────────────
    #
    # The earlier branch of this work tried a regex-based
    # ``_rewrite_urlless_navigates`` post-process to catch a urlless
    # navigate (where the LLM emitted ``navigate`` for an in-app
    # transition like "Go to the Leads page"). That regex pass was
    # removed in favor of LLM-only generalization:
    #
    #   1. The prompt is strict — ``navigate`` REQUIRES an http(s):// URL.
    #      In-app phrases must be ``submit`` with the visible label.
    #      The four-shape classification + worked examples in the prompt
    #      give Claude enough context to follow this rule reliably.
    #
    #   2. If the LLM still slips, the runner will surface the broken
    #      step naturally rather than the decomposer silently rewriting
    #      based on a brittle English-pattern regex. Operators see the
    #      bad classification and can refine the prompt.
    #
    # No semantic matching by regex anywhere in the decomposer. Only the
    # LLM decides which step type fits a given source phrase.
