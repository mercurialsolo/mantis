"""Tests for the shared server_utils module.

Verifies that both Modal and Baseten produce identical results
through the shared build_micro_result builder.
"""

import json
import tempfile
from pathlib import Path

import pytest

from mantis_agent.server_utils import (
    build_micro_result,
    build_micro_suite,
    build_proxy_config,
    build_task_loop_result,
    micro_plan_steps_to_dicts,
    parse_lead_row,
    plan_signature_from_steps,
    result_summary,
    safe_state_key,
    save_result_json,
    write_leads_csv,
)


# ── build_proxy_config: IPRoyal geo-targeting suffix shape ─────────────

class TestBuildProxyConfig:
    """The IPRoyal residential proxy is finicky about the suffix shape on the
    proxy password. Empirical contract — verified against geo.iproyal.com:12321:

      _country-us              ✅ accepted
      _city-miami              ✅ accepted
      _state-florida           ✅ accepted (full name, lowercase)
      _state-fl                ❌ rejected with 503 on CONNECT
      _session-<id>            ✅ accepted (sticky IP)

    These tests pin the build_proxy_config behaviour so a regression doesn't
    silently break every proxied run.
    """

    @pytest.fixture(autouse=True)
    def _proxy_env(self, monkeypatch):
        monkeypatch.setenv("PROXY_URL", "http://geo.iproyal.com:12321")
        monkeypatch.setenv("PROXY_USER", "test_user")
        monkeypatch.setenv("PROXY_PASS", "basepass")

    def test_returns_none_when_proxy_url_unset(self, monkeypatch):
        monkeypatch.delenv("PROXY_URL", raising=False)
        assert build_proxy_config(city="miami") is None

    def test_appends_city_suffix(self):
        cfg = build_proxy_config(city="miami")
        assert cfg["password"] == "basepass_city-miami"

    def test_drops_two_letter_state_abbreviation(self):
        """Empirical: IPRoyal returns 503 for `_state-fl`. The builder must
        not append it — caller likely meant a full state name and only had
        the abbreviation, so we silently skip rather than corrupt the suffix."""
        cfg = build_proxy_config(city="miami", state="fl")
        assert "_state-" not in cfg["password"], (
            f"two-letter state must NOT be appended: {cfg['password']!r}"
        )
        assert cfg["password"] == "basepass_city-miami"

    def test_appends_full_state_name_lowercased(self):
        cfg = build_proxy_config(city="miami", state="Florida")
        assert cfg["password"].endswith("_state-florida")

    def test_appends_session_suffix(self):
        cfg = build_proxy_config(session_id="abc123")
        assert cfg["password"].endswith("_session-abc123")

    def test_skips_already_present_suffixes(self):
        """If the env var already has _city- baked in, don't double-append."""
        import os
        os.environ["PROXY_PASS"] = "basepass_city-miami"
        try:
            cfg = build_proxy_config(city="miami")
            assert cfg["password"].count("_city-") == 1
        finally:
            os.environ["PROXY_PASS"] = "basepass"


class FakeStepResult:
    def __init__(self, step_index, intent, success, data="", steps_used=3):
        self.step_index = step_index
        self.intent = intent
        self.success = success
        self.data = data
        self.steps_used = steps_used


class FakeMicroRunner:
    """Minimal mock matching MicroPlanRunner interface used by build_micro_result."""

    def __init__(self, leads=None, costs=None, final_status="completed"):
        self._leads = leads or []
        self._final_costs = costs or {"gpu_steps": 10, "status": final_status}
        self._final_status = final_status

    def _successful_lead_data(self, step_results):
        return self._leads

    def _lead_key(self, lead):
        if isinstance(lead, dict):
            return lead.get("url", str(lead))
        return str(lead)

    def _lead_has_phone(self, lead):
        if isinstance(lead, dict):
            return bool(lead.get("phone"))
        return "phone" in str(lead).lower()

    def dynamic_verification_report(self, status=None):
        return {
            "status": status or self._final_status,
            "verdict": "pass",
            "totals": {"found_items": 5, "attempted_items": 5, "completed_items": 5},
            "checks": [{"name": "page_1_found_items_attempted", "status": "pass"}],
            "pages": [],
        }


def test_build_micro_result_includes_dynamic_verification():
    """The critical test: build_micro_result must always include dynamic_verification."""
    runner = FakeMicroRunner(
        leads=[
            {"year": "2020", "make": "Boston Whaler", "phone": "555-1234", "url": "https://example.com/1"},
            {"year": "2019", "make": "Grady-White", "phone": "", "url": "https://example.com/2"},
        ],
    )
    step_results = [
        FakeStepResult(0, "Navigate to page", True),
        FakeStepResult(1, "Click listing 1", True, data="lead_data_1"),
        FakeStepResult(2, "Click listing 2", True, data="lead_data_2"),
    ]

    result = build_micro_result(
        runner,
        step_results,
        run_id="20260424_120000",
        provider="modal",
        session_name="test_session",
        model_name="Holo3-35B-A3B",
        elapsed_seconds=120.5,
        state_key="test_key",
        checkpoint_path="/data/checkpoints/test.json",
        plan_signature="abc123",
        resume_state=False,
    )

    # Core fields
    assert result["run_id"] == "20260424_120000"
    assert result["provider"] == "modal"
    assert result["model"] == "Holo3-35B-A3B"
    assert result["mode"] == "micro_intent"
    assert result["total_time_s"] == 120  # round(120.5) uses banker's rounding
    assert result["steps_executed"] == 3
    assert result["viable"] == 2
    assert result["leads_with_phone"] == 1
    assert result["state_key"] == "test_key"

    # THE CRITICAL CHECK: dynamic_verification must be present
    assert "dynamic_verification" in result
    assert "dynamic_verification_summary" in result
    dv = result["dynamic_verification"]
    assert dv["verdict"] == "pass"
    assert dv["status"] == "completed"
    dvs = result["dynamic_verification_summary"]
    assert dvs["verdict"] == "pass"
    assert dvs["totals"]["found_items"] == 5
    assert len(dvs["checks"]) == 1

    # Step details
    assert len(result["step_details"]) == 3
    assert result["step_details"][0]["intent"] == "Navigate to page"

    # Leads
    assert len(result["leads"]) == 2


def test_build_micro_result_same_output_for_both_providers():
    """Modal and Baseten should produce structurally identical results."""
    runner = FakeMicroRunner(leads=[{"url": "https://example.com/a"}])
    steps = [FakeStepResult(0, "test", True)]

    modal_result = build_micro_result(
        runner, steps,
        run_id="run1", provider="modal", session_name="s", model_name="M",
        elapsed_seconds=10.0,
    )
    baseten_result = build_micro_result(
        runner, steps,
        run_id="run1", provider="baseten", session_name="s", model_name="M",
        elapsed_seconds=10.0,
    )

    # Same keys
    assert set(modal_result.keys()) == set(baseten_result.keys())

    # Both have dynamic_verification
    assert "dynamic_verification" in modal_result
    assert "dynamic_verification" in baseten_result
    assert modal_result["dynamic_verification"] == baseten_result["dynamic_verification"]


def test_plan_signature_deterministic():
    steps = [{"intent": "click button", "type": "click", "budget": 5}]
    sig1 = plan_signature_from_steps(steps)
    sig2 = plan_signature_from_steps(steps)
    assert sig1 == sig2
    assert len(sig1) == 64


def test_safe_state_key_sanitizes():
    assert safe_state_key("hello world!") == "hello_world"
    assert safe_state_key("") == "micro_state"
    assert safe_state_key("valid_key-123.txt") == "valid_key-123.txt"
    assert safe_state_key("...") == "micro_state"


def test_build_micro_suite_structure():
    steps = [{"intent": "nav", "type": "navigate"}]
    suite = build_micro_suite(steps, "example.com", max_cost=3.0, state_key="my_key")
    assert suite["session_name"] == "micro_example_com"
    assert suite["_max_cost"] == 3.0
    assert suite["_state_key"] == "my_key"
    assert suite["_micro_plan"] == steps
    assert suite["tasks"] == []
    assert suite["_checkpoint_path"].endswith("my_key.json")


def test_micro_plan_steps_to_dicts():
    class FakeIntent:
        intent = "click X"
        type = "click"
        verify = "X visible"
        budget = 5
        reverse = "Escape"
        grounding = True
        claude_only = False
        loop_target = -1
        loop_count = 0
        section = "extraction"
        required = False
        gate = False

    dicts = micro_plan_steps_to_dicts([FakeIntent()])
    assert len(dicts) == 1
    d = dicts[0]
    assert d["intent"] == "click X"
    assert d["type"] == "click"
    assert d["grounding"] is True
    assert d["section"] == "extraction"


def test_parse_lead_row_dict():
    row = parse_lead_row({"year": "2020", "make": "Grady", "phone": "555-1234"})
    assert row["year"] == "2020"
    assert row["phone"] == "555-1234"
    assert "raw" in row


def test_parse_lead_row_string():
    row = parse_lead_row("VIABLE | year:2020 | make:Boston | phone:555-1234")
    assert row["status"] == "VIABLE"
    assert row["year"] == "2020"
    assert row["phone"] == "555-1234"


def test_write_leads_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "leads.csv"
        leads = [{"year": "2020", "make": "BW", "phone": "555"}]
        write_leads_csv(csv_path, leads)
        assert csv_path.exists()
        content = csv_path.read_text()
        assert "2020" in content
        assert "555" in content


def test_save_result_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = {
            "run_id": "test_run",
            "leads": [{"year": "2020", "phone": "555"}],
        }
        path = save_result_json(result, Path(tmpdir), "holo3")
        assert path.exists()
        saved = json.loads(path.read_text())
        assert saved["run_id"] == "test_run"
        assert "result_path" in saved
        assert "csv_path" in saved


def test_result_summary_extracts_keys():
    full = {
        "run_id": "r1",
        "provider": "modal",
        "session_name": "s",
        "model": "M",
        "mode": "micro_intent",
        "viable": 5,
        "extra_field": "ignored",
        "dynamic_verification_summary": {"verdict": "pass"},
    }
    summary = result_summary(full)
    assert summary["run_id"] == "r1"
    assert summary["dynamic_verification_summary"]["verdict"] == "pass"
    assert "extra_field" not in summary


def test_build_task_loop_result():
    result = build_task_loop_result(
        run_id="r1",
        provider="baseten",
        session_name="s",
        model_name="holo3",
        elapsed_seconds=60.0,
        scores=[1.0, 0.0, 1.0],
        task_details=[{"task_id": "t1"}, {"task_id": "t2"}, {"task_id": "t3"}],
    )
    assert result["passed"] == 2
    assert result["total"] == 3
    assert abs(result["score"] - 66.67) < 0.1
    assert result["mode"] == "tasks"
