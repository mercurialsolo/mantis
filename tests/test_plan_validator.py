"""Tests for PlanValidator — structural checks on compiled MicroPlans."""

from mantis_agent.graph.plan_validator import PlanValidator
from mantis_agent.graph.objective import ObjectiveSpec
from mantis_agent.plan_decomposer import MicroIntent, MicroPlan


def _make_valid_plan() -> MicroPlan:
    """A structurally valid plan matching extract_url_filtered.json."""
    return MicroPlan(
        domain="boattrader.com",
        steps=[
            MicroIntent(intent="Navigate to https://example.com/results", type="navigate", budget=3, section="setup", required=True),
            MicroIntent(intent="Verify filters applied", type="extract_data", claude_only=True, budget=0, section="setup", gate=True, verify="Filters active"),
            MicroIntent(intent="Click listing", type="click", budget=8, grounding=True, section="extraction"),
            MicroIntent(intent="Read URL", type="extract_url", claude_only=True, budget=0, section="extraction"),
            MicroIntent(intent="Scroll to details", type="scroll", budget=10, section="extraction"),
            MicroIntent(intent="Extract data", type="extract_data", claude_only=True, budget=0, section="extraction"),
            MicroIntent(intent="Go back", type="navigate_back", budget=3, section="extraction"),
            MicroIntent(intent="Loop extraction", type="loop", loop_target=2, loop_count=50, section="extraction"),
            MicroIntent(intent="Next page", type="paginate", budget=10, grounding=True, section="pagination"),
            MicroIntent(intent="Loop pagination", type="loop", loop_target=2, loop_count=50, section="pagination"),
        ],
    )


def test_valid_plan_has_no_issues():
    validator = PlanValidator()
    issues = validator.validate(_make_valid_plan())
    assert issues == []


def test_empty_plan():
    validator = PlanValidator()
    issues = validator.validate(MicroPlan())
    assert len(issues) == 1
    assert issues[0].code == "empty_plan"


def test_missing_navigate():
    plan = MicroPlan(steps=[
        MicroIntent(intent="Click something", type="click", section="extraction"),
    ])
    validator = PlanValidator()
    issues = validator.validate(plan)
    codes = [i.code for i in issues]
    assert "missing_navigate" in codes


def test_missing_navigate_auto_fix():
    plan = MicroPlan(steps=[
        MicroIntent(intent="Click listing", type="click", section="extraction"),
    ])
    spec = ObjectiveSpec(raw_text="test", domains=["example.com"], start_url="https://example.com/search")
    validator = PlanValidator()
    enhanced = validator.enhance(plan, objective=spec)
    assert enhanced.steps[0].type == "navigate"
    assert "example.com" in enhanced.steps[0].intent


def test_no_gate_after_filters():
    plan = MicroPlan(steps=[
        MicroIntent(intent="Navigate", type="navigate", section="setup", required=True),
        MicroIntent(intent="Apply filter", type="filter", section="setup", required=True),
        MicroIntent(intent="Click", type="click", section="extraction"),
    ])
    validator = PlanValidator()
    issues = validator.validate(plan)
    codes = [i.code for i in issues]
    assert "no_gate_after_filters" in codes


def test_no_gate_auto_fix():
    plan = MicroPlan(steps=[
        MicroIntent(intent="Navigate", type="navigate", section="setup", required=True),
        MicroIntent(intent="Apply filter", type="filter", section="setup", required=True),
        MicroIntent(intent="Click", type="click", section="extraction"),
    ])
    spec = ObjectiveSpec(raw_text="test", target_entity="job posting", required_filters=["remote", "senior"])
    validator = PlanValidator()
    enhanced = validator.enhance(plan, objective=spec)
    gates = [s for s in enhanced.steps if s.gate]
    assert len(gates) == 1
    assert "job posting" in gates[0].intent
    assert gates[0].claude_only is True


def test_no_navigate_back_in_loop():
    plan = MicroPlan(steps=[
        MicroIntent(intent="Navigate", type="navigate", section="setup", required=True),
        MicroIntent(intent="Click", type="click", section="extraction"),
        MicroIntent(intent="Extract", type="extract_data", claude_only=True, section="extraction"),
        MicroIntent(intent="Loop", type="loop", loop_target=1, loop_count=10, section="extraction"),
    ])
    validator = PlanValidator()
    issues = validator.validate(plan)
    codes = [i.code for i in issues]
    assert "no_navigate_back_in_loop" in codes


def test_loop_target_out_of_range():
    plan = MicroPlan(steps=[
        MicroIntent(intent="Navigate", type="navigate", section="setup"),
        MicroIntent(intent="Loop", type="loop", loop_target=5, loop_count=10, section="extraction"),
    ])
    validator = PlanValidator()
    issues = validator.validate(plan)
    codes = [i.code for i in issues]
    assert "loop_target_out_of_range" in codes


def test_loop_target_forward():
    plan = MicroPlan(steps=[
        MicroIntent(intent="Navigate", type="navigate", section="setup"),
        MicroIntent(intent="Loop forward", type="loop", loop_target=2, loop_count=10, section="extraction"),
        MicroIntent(intent="Click", type="click", section="extraction"),
    ])
    validator = PlanValidator()
    issues = validator.validate(plan)
    codes = [i.code for i in issues]
    assert "loop_target_forward" in codes


def test_extract_not_claude_only():
    plan = MicroPlan(steps=[
        MicroIntent(intent="Navigate", type="navigate", section="setup"),
        MicroIntent(intent="Read URL", type="extract_url", claude_only=False, budget=5, section="extraction"),
    ])
    validator = PlanValidator()
    issues = validator.validate(plan)
    codes = [i.code for i in issues]
    assert "extract_not_claude_only" in codes


def test_extract_claude_only_auto_fix():
    plan = MicroPlan(steps=[
        MicroIntent(intent="Navigate", type="navigate", section="setup"),
        MicroIntent(intent="Read URL", type="extract_url", claude_only=False, budget=5, section="extraction"),
    ])
    validator = PlanValidator()
    enhanced = validator.enhance(plan)
    extract_step = [s for s in enhanced.steps if s.type == "extract_url"][0]
    assert extract_step.claude_only is True
    assert extract_step.budget == 0


def test_no_pagination():
    plan = MicroPlan(steps=[
        MicroIntent(intent="Navigate", type="navigate", section="setup"),
        MicroIntent(intent="Click", type="click", section="extraction"),
        MicroIntent(intent="Back", type="navigate_back", section="extraction"),
        MicroIntent(intent="Loop", type="loop", loop_target=1, loop_count=10, section="extraction"),
    ])
    validator = PlanValidator()
    issues = validator.validate(plan)
    codes = [i.code for i in issues]
    assert "no_pagination" in codes


def test_filters_in_objective_but_not_plan():
    plan = MicroPlan(steps=[
        MicroIntent(intent="Navigate", type="navigate", section="setup"),
        MicroIntent(intent="Click", type="click", section="extraction"),
    ])
    spec = ObjectiveSpec(raw_text="test", required_filters=["private seller", "zip 33101"])
    validator = PlanValidator()
    issues = validator.validate(plan, objective=spec)
    codes = [i.code for i in issues]
    assert "no_filter_steps" in codes


def test_enhance_preserves_valid_plan():
    plan = _make_valid_plan()
    validator = PlanValidator()
    enhanced = validator.enhance(plan)
    assert len(enhanced.steps) == len(plan.steps)
    for orig, enh in zip(plan.steps, enhanced.steps):
        assert orig.type == enh.type
        assert orig.intent == enh.intent
