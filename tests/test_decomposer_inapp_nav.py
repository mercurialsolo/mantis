"""Tests for the in-app-navigation rewrite (staffcrm verify follow-up).

The post-process guard ensures any ``navigate`` step without an http(s)://
URL is rewritten to ``submit`` with the page label, so a single decomposer
slip-up doesn't halt the whole run. Surfaced by the staffcrm Modal verify
where "Go the Leads Page" was emitted as a ``navigate`` step and crashed
with ``form_target_not_found``.
"""

from __future__ import annotations

from mantis_agent.plan_decomposer import MicroIntent, MicroPlan, PlanDecomposer


# ── _extract_in_app_page_label ─────────────────────────────────────────


def test_extract_label_from_go_to_phrase() -> None:
    cases = [
        ("Go to the Leads page", "Leads"),
        ("Go to the Leads Page", "Leads"),
        ("Go to Settings", "Settings"),
        ("Go to the Reports tab", "Reports"),
    ]
    for intent, expected in cases:
        assert PlanDecomposer._extract_in_app_page_label(intent) == expected, intent


def test_extract_label_from_navigate_to_phrase() -> None:
    cases = [
        ("Navigate to the Leads page", "Leads"),
        ("Navigate to the Leads Page", "Leads"),
        ("Navigate to Reports", "Reports"),
        ("Navigate to the Settings section", "Settings"),
    ]
    for intent, expected in cases:
        assert PlanDecomposer._extract_in_app_page_label(intent) == expected, intent


def test_extract_label_from_open_phrase() -> None:
    cases = [
        ("Open the Leads page", "Leads"),
        ("Open Settings", "Settings"),
        ("Open the Reports view", "Reports"),
    ]
    for intent, expected in cases:
        assert PlanDecomposer._extract_in_app_page_label(intent) == expected, intent


def test_extract_label_from_switch_to_phrase() -> None:
    cases = [
        ("Switch to the Reports tab", "Reports"),
        ("Switch to Dashboard", "Dashboard"),
    ]
    for intent, expected in cases:
        assert PlanDecomposer._extract_in_app_page_label(intent) == expected, intent


def test_extract_label_strips_trailing_punctuation() -> None:
    assert (
        PlanDecomposer._extract_in_app_page_label("Go to the Leads page.") == "Leads"
    )


def test_extract_label_strips_quotes() -> None:
    assert (
        PlanDecomposer._extract_in_app_page_label('Go to the "Leads" page')
        == "Leads"
    )


def test_extract_label_returns_empty_when_no_pattern_matches() -> None:
    cases = [
        "",
        "Click the Update Lead button",
        "Fill in the username field",
        "Verify the page shows the saved record",
    ]
    for intent in cases:
        assert PlanDecomposer._extract_in_app_page_label(intent) == "", intent


# ── _rewrite_urlless_navigates ──────────────────────────────────────────


def _navigate_step(intent: str) -> MicroIntent:
    return MicroIntent(
        intent=intent,
        type="navigate",
        section="setup",
    )


def test_rewrite_urlless_navigate_to_submit() -> None:
    """The exact failure mode from the staffcrm verify run."""
    plan = MicroPlan()
    plan.steps.append(_navigate_step("Navigate to the Leads page"))

    PlanDecomposer._rewrite_urlless_navigates(plan)

    step = plan.steps[0]
    assert step.type == "submit"
    assert step.params == {"label": "Leads"}


def test_keeps_navigate_when_intent_has_https_url() -> None:
    plan = MicroPlan()
    plan.steps.append(_navigate_step("Go to https://staffai-test-crm.exe.xyz/login"))

    PlanDecomposer._rewrite_urlless_navigates(plan)

    assert plan.steps[0].type == "navigate"
    assert plan.steps[0].params == {}


def test_keeps_navigate_when_intent_has_http_url() -> None:
    plan = MicroPlan()
    plan.steps.append(_navigate_step("Open http://localhost:3000/admin"))

    PlanDecomposer._rewrite_urlless_navigates(plan)

    assert plan.steps[0].type == "navigate"


def test_does_not_rewrite_non_navigate_steps() -> None:
    plan = MicroPlan()
    plan.steps.append(MicroIntent(intent="Click the Update Lead button", type="submit"))

    PlanDecomposer._rewrite_urlless_navigates(plan)

    assert plan.steps[0].type == "submit"


def test_leaves_intent_alone_when_no_label_recoverable() -> None:
    """If the intent doesn't match any in-app-nav pattern AND has no URL,
    we leave it as a navigate so the runner surfaces the planning error
    rather than this rewrite hiding it."""
    plan = MicroPlan()
    plan.steps.append(_navigate_step("Just do something useful"))

    PlanDecomposer._rewrite_urlless_navigates(plan)

    assert plan.steps[0].type == "navigate"


def test_rewrite_preserves_existing_params() -> None:
    """If the decomposer already populated params, merge label rather than
    overwriting unrelated keys."""
    step = _navigate_step("Go to the Settings page")
    step.params = {"wait_after_load_seconds": 30}
    plan = MicroPlan()
    plan.steps.append(step)

    PlanDecomposer._rewrite_urlless_navigates(plan)

    assert plan.steps[0].type == "submit"
    # setdefault preserves the wait_after_load_seconds key.
    assert plan.steps[0].params["wait_after_load_seconds"] == 30
    assert plan.steps[0].params["label"] == "Settings"


def test_rewrite_marks_step_required() -> None:
    """Login + nav + form steps are required by default. Match that."""
    plan = MicroPlan()
    plan.steps.append(_navigate_step("Go to the Leads page"))

    PlanDecomposer._rewrite_urlless_navigates(plan)

    assert plan.steps[0].required is True


def test_rewrite_handles_multiple_navigates_in_one_plan() -> None:
    plan = MicroPlan()
    plan.steps.append(_navigate_step("Go to https://x.test/login"))           # keep
    plan.steps.append(_navigate_step("Go to the Leads page"))                 # rewrite
    plan.steps.append(_navigate_step("Open Settings"))                        # rewrite
    plan.steps.append(_navigate_step("Navigate to https://x.test/dashboard")) # keep

    PlanDecomposer._rewrite_urlless_navigates(plan)

    assert plan.steps[0].type == "navigate"
    assert plan.steps[1].type == "submit"
    assert plan.steps[1].params["label"] == "Leads"
    assert plan.steps[2].type == "submit"
    assert plan.steps[2].params["label"] == "Settings"
    assert plan.steps[3].type == "navigate"


# ── Prompt content (regression guard for the cache key) ────────────────


def test_prompt_describes_navigate_url_requirement() -> None:
    """The prompt must explicitly require an http(s):// URL for navigate."""
    from mantis_agent.plan_decomposer import DECOMPOSE_PROMPT
    text = DECOMPOSE_PROMPT.lower()
    assert "http" in text and "https" in text
    # Direct tokens from the new rule the prompt now contains.
    assert "in-app" in text or "go to the leads" in text
    assert "submit" in text


# ── End-to-end via the dispatch path ────────────────────────────────────


def test_dispatch_path_runs_rewrite_after_loop_target_fix() -> None:
    """Smoke test: the runner-style ordered post-process (loop targets,
    then in-app rewrite) leaves both fixes applied."""
    plan = MicroPlan()
    plan.steps.append(_navigate_step("Go to the Leads page"))
    plan.steps.append(MicroIntent(
        intent="Loop back to first step",
        type="loop",
        loop_target=99,  # Out-of-range; not "close enough" — ignore in fix.
    ))

    # Apply the dispatch chain manually (mirrors decompose_text).
    PlanDecomposer._fix_loop_targets(plan)
    PlanDecomposer._rewrite_urlless_navigates(plan)

    assert plan.steps[0].type == "submit"
    assert plan.steps[0].params["label"] == "Leads"
