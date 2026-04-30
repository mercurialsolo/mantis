"""Plan Optimizer — preprocesses text plans for reliable CUA execution.

Takes a raw text plan and produces an optimized execution plan that
accounts for known CUA limitations:

1. AUTH SEPARATION: Detect login/OAuth flows → extract as pre-steps
   with session saving. The main workflow then uses saved sessions.

2. LOOP UNROLLING: Detect "for each" / "repeat" patterns → unroll
   into bounded sections with explicit iteration counts.

3. SITE BOUNDARY SECTIONING: Detect navigation between different
   domains → create section breaks with explicit URL targets.

4. FORM FILL DECOMPOSITION: Detect form-filling steps → add explicit
   field-by-field instructions with "click field, type value" pattern.

5. STEP BUDGETING: Assign appropriate max_steps and retries per section
   based on complexity (auth=120 steps, extraction=60, form=80).

Usage:
    from mantis_agent.gym.plan_optimizer import optimize_plan

    task_suite = optimize_plan(
        plan_text=open("plans/example/spec.md").read(),
        inputs={"zip_code": "33101", "admin_password": "xyz"},
    )
    # Returns a task suite JSON ready for modal_web_tasks_opencua.py

    # Or via CLI:
    python -m mantis_agent.gym.plan_optimizer plans/example/spec.md \
        --inputs "zip_code=33101,admin_password=xyz" \
        --output tasks/boattrader/optimized.json
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class OptimizedSection:
    """A section in the optimized execution plan."""
    task_id: str
    intent: str
    phase: str  # "pre_auth", "setup", "extract", "entry", "cleanup"
    max_steps: int = 60
    max_retries: int = 3
    save_session: bool = False
    require_session: bool = False
    start_url: str = ""
    verify_type: str = ""
    verify_value: str = ""
    loop: dict | None = None  # WorkflowRunner loop config

    def to_task(self) -> dict:
        task = {
            "task_id": self.task_id,
            "intent": self.intent,
            "start_url": self.start_url,
        }
        if self.save_session:
            task["save_session"] = True
        if self.require_session:
            task["require_session"] = True
        if self.verify_type:
            task["verify"] = {"type": self.verify_type, "value": self.verify_value}
        if self.loop:
            task["loop"] = self.loop
        return task


# ── CUA limitation patterns ──────────────────────────────────────────

AUTH_PATTERNS = [
    r"log\s*in", r"sign\s*in", r"authenticat", r"password",
    r"credential", r"oauth", r"google.*sign", r"2fa",
    r"impersonat", r"user\s*management",
]

LOOP_PATTERNS = [
    r"for\s+each", r"for\s+every", r"repeat\s+for",
    r"process\s+all", r"iterate", r"loop",
    r"next\s+page", r"paginat",
]

FORM_PATTERNS = [
    r"fill\s+(?:in|out)?.*form", r"enter.*field",
    r"submit", r"click.*button", r"select.*dropdown",
    r"type.*(?:into|in)\s+", r"outside\s+boat\s+lead",
]

SITE_BOUNDARY_PATTERN = re.compile(r'https?://([^/\s]+)')


LLM_PLAN_PROMPT = """\
You are a workflow optimizer. Given a plan text, break it into executable sections for a CUA (Computer Use Agent).

Output a JSON array of sections. Each section has:
- task_id: string (snake_case, e.g. "setup_search", "extract_all", "auth_login")
- intent: string (detailed step-by-step instruction for the agent)
- phase: "setup" | "extract" | "pre_auth" | "entry" | "cleanup"
- save_session: bool (true for auth and setup sections)
- require_session: bool (true for sections that need prior session state)
- start_url: string (URL to navigate to, include https://)

For extraction/iteration phases ("for each listing", "process all", "repeat"):
- Create ONE section with a "loop" field containing:
  - pagination_intent: instruction to click Next page
  - max_iterations: number (e.g. 30)
  - max_pages: number (e.g. 6)
- The intent should use {{ORDINAL}} placeholder (e.g. "Process the {{ORDINAL}} listing")
- Do NOT unroll loops into individual sections — the runtime handles iteration dynamically

Task ordering rules:
- Search/filter setup comes FIRST (before auth) when on a different site
- Auth/login sections come AFTER search if the auth is for a separate site
- Entry sections always come LAST
- Auth sections MUST have save_session=true

Plan text:
---
{plan_text}
---

Inputs:
{inputs_text}

Respond with ONLY a JSON array. No other text."""


def _try_llm_optimize(brain: Any, plan_text: str, inputs: dict[str, str],
                       session_name: str, max_listings: int) -> dict | None:
    """Try to optimize plan using LLM brain. Returns None on failure."""
    if brain is None or not hasattr(brain, "query"):
        return None

    inputs_text = "\n".join(f"  {k}: {v}" for k, v in inputs.items()) if inputs else "  (none)"
    prompt = LLM_PLAN_PROMPT.format(plan_text=plan_text[:4000], inputs_text=inputs_text)

    try:
        response = brain.query(prompt, response_format="json")
        if not response:
            return None

        # Extract JSON from response (may have markdown wrapping)
        json_text = response.strip()
        if json_text.startswith("```"):
            json_text = json_text.split("```")[1]
            if json_text.startswith("json"):
                json_text = json_text[4:]
        if json_text.endswith("```"):
            json_text = json_text[:-3]

        sections = json.loads(json_text.strip())
        if not isinstance(sections, list) or len(sections) == 0:
            return None

        # Validate and convert to task suite
        tasks = []
        for s in sections:
            if not isinstance(s, dict) or "task_id" not in s or "intent" not in s:
                continue
            task = {
                "task_id": s["task_id"],
                "intent": s["intent"],
                "start_url": s.get("start_url", ""),
            }
            if s.get("save_session"):
                task["save_session"] = True
            if s.get("require_session"):
                task["require_session"] = True
            if s.get("verify_type"):
                task["verify"] = {"type": s["verify_type"], "value": s.get("verify_value", "")}
            # Emit loop metadata if phase is "extract" and intent mentions iteration
            if s.get("loop"):
                task["loop"] = s["loop"]
            tasks.append(task)

        if not tasks:
            return None

        # Resolve input variables in all intents and URLs
        for task in tasks:
            for key, val in inputs.items():
                task["intent"] = task["intent"].replace(f"{{{{{key}}}}}", val).replace(f"{{{key}}}", val)
                if task.get("start_url"):
                    task["start_url"] = task["start_url"].replace(f"{{{{{key}}}}}", val)

        base_url = tasks[0].get("start_url", "")
        return {
            "session_name": session_name,
            "base_url": base_url,
            "tasks": tasks,
            "_optimization": {
                "method": "llm",
                "sections_generated": len(tasks),
            },
        }

    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"LLM plan optimization failed: {e}")
        return None


def optimize_plan(
    plan_text: str,
    inputs: dict[str, str] | None = None,
    session_name: str = "workflow",
    max_listings: int = 30,
    brain: Any = None,
) -> dict:
    """Optimize a text plan for CUA execution using an LLM brain.

    Requires a brain with query() (e.g. Gemma4 via llama.cpp or vLLM).
    The LLM analyzes the plan and produces structured task sections with
    proper ordering, session management, and loop configs.

    Args:
        plan_text: Free-text plan/spec describing the workflow.
        inputs: Variable substitutions (e.g. {"zip_code": "33101"}).
        session_name: Name for session persistence.
        max_listings: Cap for loop iterations.
        brain: LLM brain with query() method. Required.

    Returns:
        Task suite JSON dict (compatible with modal_web_tasks_opencua.py).

    Raises:
        ValueError: If no brain is provided.
    """
    inputs = inputs or {}

    if brain is None:
        raise ValueError(
            "optimize_plan requires a brain with query() for plan parsing. "
            "Pass a Gemma4/LlamaCpp brain, or use a pre-built task JSON instead."
        )

    result = _try_llm_optimize(brain, plan_text, inputs, session_name, max_listings)
    if result is not None:
        return result

    raise RuntimeError(
        "LLM plan optimization failed. Check that the brain is running and "
        "can handle text queries. Use a pre-built task JSON as fallback."
    )


def _detect_sites(text: str) -> list[str]:
    """Find all unique domain URLs in the plan."""
    urls = SITE_BOUNDARY_PATTERN.findall(text)
    seen = set()
    unique = []
    for domain in urls:
        if domain not in seen and "example" not in domain:
            seen.add(domain)
            unique.append(domain)
    return unique


def _detect_phases(text: str, sites: list[str]) -> list[dict]:
    """Detect logical phases in the plan."""
    phases = []
    text_lower = text.lower()

    # Check for auth
    has_auth = any(re.search(p, text_lower) for p in AUTH_PATTERNS)
    if has_auth:
        phases.append({"type": "auth", "sites": [s for s in sites if "admin" in s or "pop" in s.lower()]})

    # Check for search/setup
    if any(kw in text_lower for kw in ["search", "filter", "sort", "navigate to"]):
        phases.append({"type": "setup"})

    # Check for loops/extraction
    has_loop = any(re.search(p, text_lower) for p in LOOP_PATTERNS)
    if has_loop or "listing" in text_lower or "extract" in text_lower:
        phases.append({"type": "extract_loop"})

    # Check for form entry
    has_form = any(re.search(p, text_lower) for p in FORM_PATTERNS)
    if has_form:
        phases.append({"type": "entry"})

    return phases


def _generate_sections(
    phases: list[dict],
    text: str,
    inputs: dict[str, str],
    sites: list[str],
    max_listings: int,
) -> list[OptimizedSection]:
    """Generate optimized sections from detected phases."""
    sections: list[OptimizedSection] = []
    text_lower = text.lower()

    # Extract URLs for different sites
    urls = re.findall(r'https?://[^\s<>"]+', text)
    main_url = urls[0] if urls else ""
    auth_urls = [u for u in urls if "admin" in u.lower() or "auth" in u.lower() or "pop" in u.lower()]
    form_urls = [u for u in urls if "lead" in u.lower() or "form" in u.lower() or "entry" in u.lower()]

    # Phase 1: Auth pre-step (if detected)
    if any(p["type"] == "auth" for p in phases):
        # Extract auth details from plan
        auth_url = auth_urls[0] if auth_urls else ""
        auth_root = re.match(r'(https?://[^/]+)', auth_url).group(1) if auth_url else ""

        email = _extract_credential(text, "email")
        password = _extract_credential(text, "password") or inputs.get("admin_password", "")

        # Login section
        login_intent = _build_auth_intent(text, auth_root, email, password)
        sections.append(OptimizedSection(
            task_id="pre_auth_login",
            intent=login_intent,
            phase="pre_auth",
            max_steps=80,
            max_retries=1,  # Auth = 1 attempt per spec
            save_session=True,
            start_url=auth_root + "/" if auth_root else "",
            verify_type="url_contains",
            verify_value=auth_root.split("//")[-1] if auth_root else "",
        ))

        # Impersonation section (if mentioned)
        if "impersonat" in text_lower or "user management" in text_lower:
            impersonate_intent = _build_impersonate_intent(text, form_urls)
            sections.append(OptimizedSection(
                task_id="pre_auth_impersonate",
                intent=impersonate_intent,
                phase="pre_auth",
                max_steps=60,
                max_retries=2,
                require_session=True,
                save_session=True,
                start_url=auth_root + "/" if auth_root else "",
                verify_type="url_contains",
                verify_value=auth_root.split("//")[-1] if auth_root else "",
            ))

    # Phase 2: Search/setup
    if any(p["type"] == "setup" for p in phases):
        setup_intent = _build_setup_intent(text, inputs)
        sections.append(OptimizedSection(
            task_id="setup_search",
            intent=setup_intent,
            phase="setup",
            max_steps=80,
            save_session=True,
            start_url=main_url,
            verify_type="url_contains",
            verify_value=main_url.split("//")[-1].split("/")[0] if main_url else "",
        ))

    # Phase 3: Dynamic loop with pagination (uses WorkflowRunner)
    if any(p["type"] == "extract_loop" for p in phases):
        extract_intent_base = _build_extract_intent(text)
        sections.append(OptimizedSection(
            task_id="extract_all",
            intent=extract_intent_base,
            phase="extract",
            max_steps=60,
            require_session=True,
            start_url=main_url,
            verify_type="url_contains",
            verify_value=main_url.split("//")[-1].split("/")[0] if main_url else "",
            loop={
                "pagination_intent": (
                    "Scroll to the bottom of the search results. Look for a 'Next' button, "
                    "'>' arrow, or page numbers. If there is a next page, click it and wait "
                    "for results to load. Call terminate('success'). If there is no next page, "
                    "call terminate('failure') with 'no more pages'."
                ),
                "max_iterations": max_listings,
                "max_pages": max(max_listings // 5, 3),
            },
        ))

    # Phase 4: Entry sections (one per expected viable lead)
    if any(p["type"] == "entry" for p in phases):
        entry_url = form_urls[0] if form_urls else ""
        entry_intent = _build_entry_intent(text, entry_url)
        for i in range(1, 3):  # Default 2 lead entries
            sections.append(OptimizedSection(
                task_id=f"enter_lead_{i}",
                intent=entry_intent.replace("{LEAD_N}", str(i)),
                phase="entry",
                max_steps=80,
                max_retries=2,
                require_session=True,
                start_url=entry_url,
                verify_type="url_contains",
                verify_value="admin" if "admin" in entry_url else "",
            ))

    return sections


# ── Intent builders ──────────────────────────────────────────────────

def _extract_credential(text: str, cred_type: str) -> str:
    """Extract email or password from plan text."""
    if cred_type == "email":
        match = re.search(r'[Ee]mail:\s*(\S+@\S+)', text)
        return match.group(1) if match else ""
    elif cred_type == "password":
        match = re.search(r'[Pp]assword:\s*(\S+)', text)
        return match.group(1) if match else ""
    return ""


def _build_auth_intent(text: str, auth_url: str, email: str, password: str) -> str:
    """Build a clear auth section intent."""
    return (
        f"Log into the admin panel at {auth_url}/\n\n"
        f"1. Navigate to {auth_url}/\n"
        f"2. Click 'Sign in with Google' (or the login button)\n"
        f"3. Enter email: {email}, click Next\n"
        f"4. Enter password: {password}, click Next\n"
        f"5. If 2FA requested: terminate('failure') with '2FA required'\n"
        f"6. Wait for redirect back to admin panel\n\n"
        f"ONE attempt only. terminate('success') when logged in."
    )


def _build_impersonate_intent(text: str, form_urls: list[str]) -> str:
    """Build impersonation intent from plan details."""
    # Extract user name to impersonate
    user_match = re.search(r'[Ss]elect.*?["\'](.+?)["\']', text)
    user_name = user_match.group(1) if user_match else "the specified user"

    nav_url = form_urls[0] if form_urls else ""

    return (
        f"Set up user impersonation on the admin panel.\n\n"
        f"1. Scroll DOWN past the dashboard content\n"
        f"2. Find the 'User Management' section\n"
        f"3. Do NOT use the dropdown at the top-left corner\n"
        f"4. In User Management, select '{user_name}'\n"
        f"5. Navigate to: {nav_url}\n\n"
        f"terminate('success') when on the target form page."
    )


def _build_setup_intent(text: str, inputs: dict) -> str:
    """Build search/setup intent."""
    # Extract filter values from inputs or text
    zip_code = inputs.get("zip_code", "")
    radius = inputs.get("search_radius", "")
    min_price = re.search(r'(?:minimum|min).*?price.*?(\d+)', text.lower())
    price = min_price.group(1) if min_price else "35000"

    return (
        f"Apply search filters on the page:\n"
        f"1. Enter {zip_code} in the location/zip code field\n"
        f"2. Set search radius to {radius} miles\n"
        f"3. Set minimum price to {price}\n"
        f"4. Select 'Private Seller' in the seller type filter\n"
        f"5. Sort by 'Recently Updated' or newest first\n"
        f"6. Wait for results to load\n\n"
        f"terminate('success') and report how many results you see."
    )


def _build_extract_intent(text: str) -> str:
    """Build a template for listing extraction."""
    return (
        "You are on search results. Process the {ORDINAL} listing:\n"
        "1. Find the {ORDINAL} listing (scroll if needed)\n"
        "2. Click it to open\n"
        "3. Scroll through the page — read Description and More Details\n"
        "4. Look for a phone number: (555)555-5555, 555-555-5555, 10+ digits\n"
        "   NOT valid: prices, years, zip codes\n"
        "5. If phone found: extract year, make, model, price, phone, boat type,\n"
        "   seller name (or Unknown), read URL from address bar\n"
        "6. Go back to results (Alt+Left)\n\n"
        "terminate('success') with: VIABLE or SKIPPED + all extracted data"
    )


def _build_entry_intent(text: str, form_url: str) -> str:
    """Build lead entry intent."""
    return (
        f"Enter lead #{'{LEAD_N}'} into the form.\n\n"
        f"Navigate to: {form_url}\n\n"
        f"1. Select the appropriate lead type if a selector exists\n"
        f"2. For each form field: click it, type the value\n"
        f"3. Scroll through to find all fields\n"
        f"4. Fill: Year, Make, Model, Type, Price, Seller Name, Phone, URL\n"
        f"5. Click Submit\n"
        f"6. URL changed = success. Form cleared but URL same = rejection (do NOT retry)\n\n"
        f"terminate('success') if submitted, terminate('failure') if rejected."
    )


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Optimize a text plan for CUA execution")
    parser.add_argument("plan", help="Path to plan text file")
    parser.add_argument("--inputs", default="", help="key=value,key=value")
    parser.add_argument("--output", help="Output JSON path")
    parser.add_argument("--session-name", default="workflow")
    parser.add_argument("--max-listings", type=int, default=5)

    args = parser.parse_args()

    with open(args.plan) as f:
        plan_text = f.read()

    inputs = {}
    if args.inputs:
        for pair in args.inputs.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                inputs[k.strip()] = v.strip()

    result = optimize_plan(
        plan_text=plan_text,
        inputs=inputs,
        session_name=args.session_name,
        max_listings=args.max_listings,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Optimized plan saved to {args.output}")
    else:
        print(json.dumps(result, indent=2))

    opt = result.get("_optimization", {})
    print("\nOptimization summary:")
    print(f"  Sites: {opt.get('sites_detected', [])}")
    print(f"  Phases: {opt.get('phases', [])}")
    print(f"  Sections: {opt.get('sections_generated', 0)}")
    print(f"  Auth separated: {opt.get('auth_separated', False)}")
    print(f"  Loops unrolled: {opt.get('loops_unrolled', 0)}")


if __name__ == "__main__":
    main()
