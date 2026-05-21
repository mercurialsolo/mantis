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
    load_plan_file,
    merge_runtime,
    micro_plan_steps_to_dicts,
    parse_lead_row,
    plan_signature_from_steps,
    resolve_ids,
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
        for key in (
            "MANTIS_PROXY_PROVIDER",
            "OXYLABS_ENTRYPOINT",
            "OXYLABS_CITY_ENTRYPOINT",
            "OXYLABS_USERNAME",
            "OXYLABS_USER",
            "OXYLABS_PASSWORD",
            "OXYLABS_PASS",
            "OXYLABS_COUNTRY",
            "OXYLABS_STATE",
            "OXYLABS_CITY",
            "PRIVATEPROXY_ENTRYPOINT",
            "PRIVATEPROXY_USERNAME",
            "PRIVATEPROXY_PASSWORD",
        ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("PROXY_URL", "http://geo.iproyal.com:12321")
        monkeypatch.setenv("PROXY_USER", "test_user")
        monkeypatch.setenv("PROXY_PASS", "basepass")

    def test_returns_none_when_proxy_url_unset(self, monkeypatch):
        monkeypatch.delenv("PROXY_URL", raising=False)
        assert build_proxy_config(city="miami") is None

    # NB: the default ``provider`` flipped from ``iproyal`` →
    # ``privateproxy`` in the stealth-parity PR. These tests pin the
    # IPRoyal username/password suffix shape, so they now pass
    # ``provider="iproyal"`` explicitly to keep the contract checks
    # intact regardless of the default.

    def test_appends_city_suffix(self):
        cfg = build_proxy_config(city="miami", provider="iproyal")
        assert cfg["password"] == "basepass_city-miami"

    def test_drops_two_letter_state_abbreviation(self):
        """Empirical: IPRoyal returns 503 for `_state-fl`. The builder must
        not append it — caller likely meant a full state name and only had
        the abbreviation, so we silently skip rather than corrupt the suffix."""
        cfg = build_proxy_config(city="miami", state="fl", provider="iproyal")
        assert "_state-" not in cfg["password"], (
            f"two-letter state must NOT be appended: {cfg['password']!r}"
        )
        assert cfg["password"] == "basepass_city-miami"

    def test_appends_full_state_name_lowercased(self):
        cfg = build_proxy_config(city="miami", state="Florida", provider="iproyal")
        assert cfg["password"].endswith("_state-florida")

    def test_appends_session_suffix(self):
        cfg = build_proxy_config(session_id="abc123", provider="iproyal")
        assert cfg["password"].endswith("_session-abc123")

    def test_skips_already_present_suffixes(self):
        """If the env var already has _city- baked in, don't double-append."""
        import os
        os.environ["PROXY_PASS"] = "basepass_city-miami"
        try:
            cfg = build_proxy_config(city="miami", provider="iproyal")
            assert cfg["password"].count("_city-") == 1
        finally:
            os.environ["PROXY_PASS"] = "basepass"

    def test_oxylabs_provider_targets_city_with_customer_username(self, monkeypatch):
        monkeypatch.setenv("OXYLABS_ENTRYPOINT", "pr.oxylabs.io:10000")
        monkeypatch.setenv("OXYLABS_USERNAME", "oxy_user")
        monkeypatch.setenv("OXYLABS_PASSWORD", "oxy_pass")

        cfg = build_proxy_config(
            city="miami",
            state="Florida",
            session_id="abc123",
            provider="oxylabs",
        )

        assert cfg == {
            "server": "http://pr.oxylabs.io:7777",
            "username": "customer-oxy_user-st-us_florida-city-miami",
            "password": "oxy_pass",
        }

    def test_oxylabs_provider_uses_raw_credentials_without_city(self, monkeypatch):
        monkeypatch.setenv("OXYLABS_ENTRYPOINT", "pr.oxylabs.io:10000")
        monkeypatch.setenv("OXYLABS_USERNAME", "oxy_user")
        monkeypatch.setenv("OXYLABS_PASSWORD", "oxy_pass")

        cfg = build_proxy_config(provider="oxylabs")

        assert cfg == {
            "server": "http://pr.oxylabs.io:10000",
            "username": "oxy_user",
            "password": "oxy_pass",
        }

    def test_oxylabs_provider_can_be_selected_by_env(self, monkeypatch):
        monkeypatch.setenv("MANTIS_PROXY_PROVIDER", "oxylabs")
        monkeypatch.setenv("OXYLABS_ENTRYPOINT", "http://pr.oxylabs.io:10000")
        monkeypatch.setenv("OXYLABS_USERNAME", "oxy_user")
        monkeypatch.setenv("OXYLABS_PASSWORD", "oxy_pass")

        cfg = build_proxy_config(city="miami")

        assert cfg["server"] == "http://pr.oxylabs.io:7777"
        assert cfg["username"] == "customer-oxy_user-cc-US-city-miami"
        assert cfg["password"] == "oxy_pass"

    def test_oxylabs_provider_reads_location_from_env(self, monkeypatch):
        monkeypatch.setenv("OXYLABS_ENTRYPOINT", "http://pr.oxylabs.io:10000")
        monkeypatch.setenv("OXYLABS_CITY_ENTRYPOINT", "pr.oxylabs.io:7777")
        monkeypatch.setenv("OXYLABS_USERNAME", "oxy_user")
        monkeypatch.setenv("OXYLABS_PASSWORD", "oxy_pass")
        monkeypatch.setenv("OXYLABS_CITY", "miami")
        monkeypatch.setenv("OXYLABS_STATE", "florida")

        cfg = build_proxy_config(provider="oxylabs")

        assert cfg["server"] == "http://pr.oxylabs.io:7777"
        assert cfg["username"] == "customer-oxy_user-st-us_florida-city-miami"

    def test_oxylabs_provider_respects_pre_targeted_username(self, monkeypatch):
        monkeypatch.setenv("OXYLABS_ENTRYPOINT", "pr.oxylabs.io:10000")
        monkeypatch.setenv("OXYLABS_USERNAME", "customer-oxy_user-cc-US-city-miami")
        monkeypatch.setenv("OXYLABS_PASSWORD", "oxy_pass")

        cfg = build_proxy_config(city="miami", provider="oxylabs")

        assert cfg["server"] == "http://pr.oxylabs.io:7777"
        assert cfg["username"] == "customer-oxy_user-cc-US-city-miami"

    def test_privateproxy_provider_appends_city_geo_to_username(self, monkeypatch):
        """Verified against ``edge1-us.privateproxy.me:8888`` on 2026-05-20 —
        bare username returns random global IPs (saw Munich + Romanian);
        ``-cc-us-city-miami`` reliably returns real Miami FL Comcast IPs."""
        for k in ("PRIVATEPROXY_COUNTRY", "PRIVATEPROXY_CC", "PRIVATEPROXY_CITY"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("PRIVATEPROXY_ENTRYPOINT", "privateproxy.example:8080")
        monkeypatch.setenv("PRIVATEPROXY_USERNAME", "private_user")
        monkeypatch.setenv("PRIVATEPROXY_PASSWORD", "private_pass")

        cfg = build_proxy_config(city="miami", state="Florida", provider="privateproxy")

        assert cfg == {
            "server": "http://privateproxy.example:8080",
            "username": "private_user-cc-us-city-miami",
            "password": "private_pass",
        }

    def test_privateproxy_runtime_country_overrides_pre_baked_cc_in_username(self, monkeypatch):
        """Runtime ``country="us"`` MUST strip + replace any -cc-XX
        suffix already baked into PRIVATEPROXY_USERNAME. Pre-PR-bug:
        ``already_targeted=True`` short-circuited the cc-us application
        and a UK-locked username from the Modal Secret kept producing
        Sheffield egress IPs for a US plan (run 20260521_051150)."""
        for k in ("PRIVATEPROXY_COUNTRY", "PRIVATEPROXY_CC", "PRIVATEPROXY_CITY"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("PRIVATEPROXY_ENTRYPOINT", "privateproxy.example:8080")
        # Pre-baked UK target in the username — simulates the Modal
        # Secret state we observed in production.
        monkeypatch.setenv("PRIVATEPROXY_USERNAME", "private_user-cc-gb")
        monkeypatch.setenv("PRIVATEPROXY_PASSWORD", "private_pass")

        cfg = build_proxy_config(provider="privateproxy", country="us")
        assert cfg["username"] == "private_user-cc-us", (
            f"runtime country must strip pre-baked -cc-gb and apply -cc-us; "
            f"got {cfg['username']!r}"
        )

    def test_privateproxy_runtime_country_with_city_replaces_pre_baked_target(self, monkeypatch):
        """Runtime country + city wins over a pre-baked
        ``-cc-XX-city-YYY`` username. Whole geo suffix is stripped
        and re-applied."""
        for k in ("PRIVATEPROXY_COUNTRY", "PRIVATEPROXY_CC", "PRIVATEPROXY_CITY"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("PRIVATEPROXY_ENTRYPOINT", "privateproxy.example:8080")
        monkeypatch.setenv("PRIVATEPROXY_USERNAME", "private_user-cc-gb-city-sheffield")
        monkeypatch.setenv("PRIVATEPROXY_PASSWORD", "private_pass")

        cfg = build_proxy_config(
            provider="privateproxy", country="us", city="miami",
        )
        assert cfg["username"] == "private_user-cc-us-city-miami"

    def test_privateproxy_no_runtime_country_keeps_pre_baked_target(self, monkeypatch):
        """Backward compat: when the caller does NOT pass ``country``,
        a pre-baked -cc-XX username is left alone. This preserves the
        existing behavior for plans that deliberately want whatever
        the Secret encodes."""
        for k in ("PRIVATEPROXY_COUNTRY", "PRIVATEPROXY_CC", "PRIVATEPROXY_CITY"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("PRIVATEPROXY_ENTRYPOINT", "privateproxy.example:8080")
        monkeypatch.setenv("PRIVATEPROXY_USERNAME", "private_user-cc-gb")
        monkeypatch.setenv("PRIVATEPROXY_PASSWORD", "private_pass")

        cfg = build_proxy_config(provider="privateproxy")
        # No -cc-us applied because already_targeted + no runtime
        # override.
        assert cfg["username"] == "private_user-cc-gb"

    def test_privateproxy_country_only_when_no_city(self, monkeypatch):
        """No city → ``-cc-us`` only. Default country is US (the most
        common deployment) — operators override via PRIVATEPROXY_COUNTRY."""
        for k in ("PRIVATEPROXY_COUNTRY", "PRIVATEPROXY_CC", "PRIVATEPROXY_CITY"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("PRIVATEPROXY_ENTRYPOINT", "privateproxy.example:8080")
        monkeypatch.setenv("PRIVATEPROXY_USERNAME", "private_user")
        monkeypatch.setenv("PRIVATEPROXY_PASSWORD", "private_pass")

        cfg = build_proxy_config(provider="privateproxy")
        assert cfg["username"] == "private_user-cc-us"

    def test_privateproxy_country_override_via_env(self, monkeypatch):
        """PRIVATEPROXY_COUNTRY env var overrides the US default."""
        for k in ("PRIVATEPROXY_CC", "PRIVATEPROXY_CITY"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("PRIVATEPROXY_ENTRYPOINT", "privateproxy.example:8080")
        monkeypatch.setenv("PRIVATEPROXY_USERNAME", "private_user")
        monkeypatch.setenv("PRIVATEPROXY_PASSWORD", "private_pass")
        monkeypatch.setenv("PRIVATEPROXY_COUNTRY", "uk")

        cfg = build_proxy_config(city="london", provider="privateproxy")
        assert cfg["username"] == "private_user-cc-uk-city-london"

    def test_privateproxy_skips_geo_when_username_already_targeted(self, monkeypatch):
        """If the operator has manually baked geo into the username
        (``mb5ku-cc-us-city-miami``), don't double-append. The check
        looks for ``-cc-`` or ``-city-`` substrings."""
        for k in ("PRIVATEPROXY_COUNTRY", "PRIVATEPROXY_CC", "PRIVATEPROXY_CITY"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("PRIVATEPROXY_ENTRYPOINT", "privateproxy.example:8080")
        monkeypatch.setenv("PRIVATEPROXY_USERNAME", "mb5ku-cc-us-city-miami")
        monkeypatch.setenv("PRIVATEPROXY_PASSWORD", "p")

        cfg = build_proxy_config(city="austin", provider="privateproxy")
        # Manual targeting preserved — even though we asked for austin,
        # the env-stamped username wins (caller-explicit intent).
        assert cfg["username"] == "mb5ku-cc-us-city-miami"

    def test_privateproxy_provider_accepts_four_part_endpoint(self, monkeypatch):
        for k in ("PRIVATEPROXY_COUNTRY", "PRIVATEPROXY_CC", "PRIVATEPROXY_CITY"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv(
            "PRIVATEPROXY_ENTRYPOINT",
            "privateproxy.example:8080:private_user:private_pass",
        )
        monkeypatch.delenv("PRIVATEPROXY_USERNAME", raising=False)
        monkeypatch.delenv("PRIVATEPROXY_PASSWORD", raising=False)

        cfg = build_proxy_config(provider="privateproxy")

        # Four-part endpoint still extracts user; default -cc-us appended
        # (no city since none was passed).
        assert cfg == {
            "server": "http://privateproxy.example:8080",
            "username": "private_user-cc-us",
            "password": "private_pass",
        }


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
        profile_id="alice",
        workflow_id="plan_v3",
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
    # #341: split identities echoed in the envelope.
    assert result["profile_id"] == "alice"
    assert result["workflow_id"] == "plan_v3"

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


def test_build_micro_result_includes_wall_time_breakdown_when_meter_present():
    """Epic #362 Phase B: a runner with a populated TimeMeter must
    surface the aggregate breakdown + per-step breakdown on the
    result envelope."""
    from mantis_agent.gym.time_meter import BUCKETS, TimeMeter

    meter = TimeMeter()
    # Stage known timings: 2s think on step 0, 0.5s act on step 1.
    meter.record("think", 2.0, step_idx=0)
    meter.record("act", 0.5, step_idx=1)

    runner = FakeMicroRunner()
    runner.time_meter = meter
    steps = [
        FakeStepResult(0, "Decide what to do", True),
        FakeStepResult(1, "Click submit", True),
    ]
    result = build_micro_result(
        runner, steps,
        run_id="run1", provider="modal", session_name="s", model_name="M",
        elapsed_seconds=3.0,
    )

    # Aggregate dict carries every bucket from the vocabulary.
    wt = result["wall_time_breakdown"]
    assert set(wt) == set(BUCKETS)
    assert wt["think"] == 2.0
    assert wt["act"] == 0.5
    # overhead is the residual against elapsed_seconds — non-negative,
    # ≤ elapsed.
    assert wt["overhead"] >= 0.0

    # Per-step breakdowns route to the right step.
    assert result["step_details"][0]["time_breakdown"]["think"] == 2.0
    assert result["step_details"][0]["time_breakdown"]["act"] == 0.0
    assert result["step_details"][1]["time_breakdown"]["act"] == 0.5
    assert result["step_details"][1]["time_breakdown"]["think"] == 0.0


def test_build_micro_result_falls_back_to_zeros_without_time_meter():
    """Pre-Phase-A runners (or test harnesses) shouldn't break the
    envelope shape — buckets land as zeros, every key still present."""
    from mantis_agent.gym.time_meter import BUCKETS

    runner = FakeMicroRunner()
    # No `time_meter` attribute on this runner.
    steps = [FakeStepResult(0, "nav", True)]
    result = build_micro_result(
        runner, steps,
        run_id="run1", provider="modal", session_name="s", model_name="M",
        elapsed_seconds=10.0,
    )
    assert "wall_time_breakdown" in result
    assert set(result["wall_time_breakdown"]) == set(BUCKETS)
    assert all(v == 0.0 for v in result["wall_time_breakdown"].values())
    # Per-step zeros too.
    assert result["step_details"][0]["time_breakdown"]["act"] == 0.0


def test_build_micro_result_bucket_sum_approximates_total_time():
    """Sum of bucket times should track total_time_s within ±5% on a
    realistic run — captures regressions where overhead drifts wildly."""
    from mantis_agent.gym.time_meter import TimeMeter

    meter = TimeMeter()
    # Allocate 100s split across plausible buckets.
    meter.record("think", 40.0, step_idx=0)
    meter.record("claude_extract", 30.0, step_idx=0)
    meter.record("act", 5.0, step_idx=0)
    meter.record("settle", 15.0, step_idx=0)
    meter.record("perceive", 8.0, step_idx=0)
    # ~2s residual lands in overhead via breakdown().

    runner = FakeMicroRunner()
    runner.time_meter = meter
    steps = [FakeStepResult(0, "do work", True)]
    result = build_micro_result(
        runner, steps,
        run_id="r", provider="modal", session_name="s", model_name="M",
        elapsed_seconds=100.0,
    )
    bucket_sum = sum(result["wall_time_breakdown"].values())
    # Both numbers are seconds; total_time_s is round(elapsed).
    # On a synthetic run with no real wall-clock pressure, the
    # sum + elapsed agree closely; ±5% gives generous slack.
    assert abs(bucket_sum - result["total_time_s"]) <= max(0.05 * result["total_time_s"], 1.0)


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
    # Phase 1 back-compat (#341): legacy state_key routes to both identities.
    assert suite["_profile_id"] == "my_key"
    assert suite["_workflow_id"] == "my_key"
    assert suite["_micro_plan"] == steps
    assert suite["tasks"] == []
    assert suite["_checkpoint_path"].endswith("my_key.json")


# ── #341: profile_id / workflow_id split ─────────────────────────────


def test_resolve_ids_legacy_state_key_routes_to_both():
    pid, wid = resolve_ids(state_key="abc", plan_signature="deadbeef")
    assert pid == "abc"
    assert wid == "abc"


def test_resolve_ids_new_fields_win_over_state_key():
    pid, wid = resolve_ids(
        state_key="legacy", profile_id="alice", workflow_id="plan_v3"
    )
    assert pid == "alice"
    assert wid == "plan_v3"


def test_resolve_ids_workflow_defaults_to_signature_prefix():
    pid, wid = resolve_ids(profile_id="alice", plan_signature="deadbeef1234567890")
    assert pid == "alice"
    assert wid == "deadbeef1234"  # first 12 hex chars


def test_resolve_ids_profile_defaults_to_default():
    pid, wid = resolve_ids(workflow_id="plan_v1", plan_signature="abc")
    assert pid == "default"
    assert wid == "plan_v1"


def test_resolve_ids_no_input_uses_defaults():
    pid, wid = resolve_ids(plan_signature="cafe000000000000")
    assert pid == "default"
    assert wid == "cafe00000000"


def test_resolve_ids_sanitizes():
    pid, wid = resolve_ids(profile_id="alice@prod!", workflow_id="plan v2")
    assert pid == "alice_prod"
    assert wid == "plan_v2"


def test_build_micro_suite_split_identities():
    steps = [{"intent": "nav", "type": "navigate"}]
    suite = build_micro_suite(
        steps,
        "example.com",
        profile_id="alice",
        workflow_id="plan_v3",
    )
    assert suite["_profile_id"] == "alice"
    assert suite["_workflow_id"] == "plan_v3"
    # state_key tracks workflow_id for downstream back-compat readers.
    assert suite["_state_key"] == "plan_v3"
    # Checkpoint path uses workflow_id, NOT profile_id.
    assert suite["_checkpoint_path"].endswith("plan_v3.json")


def test_profile_id_preserved_across_workflow_rotation():
    """Regression guard for the core motivation of #341.

    Rotating workflow_id (because the plan definition changed) must not
    invalidate the Chrome profile. The two suites below share a profile
    but use different checkpoints — exactly what coupling under one
    state_key prevented.
    """
    steps_v1 = [{"intent": "nav", "type": "navigate"}]
    steps_v2 = [{"intent": "nav", "type": "navigate"}, {"intent": "click", "type": "click"}]
    suite_v1 = build_micro_suite(steps_v1, "example.com", profile_id="alice", workflow_id="v1")
    suite_v2 = build_micro_suite(steps_v2, "example.com", profile_id="alice", workflow_id="v2")
    assert suite_v1["_profile_id"] == suite_v2["_profile_id"] == "alice"
    assert suite_v1["_workflow_id"] != suite_v2["_workflow_id"]
    assert suite_v1["_checkpoint_path"] != suite_v2["_checkpoint_path"]


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
        params = {"label": "X", "kind": "button"}
        hints = {"region": "form-footer"}

    dicts = micro_plan_steps_to_dicts([FakeIntent()])
    assert len(dicts) == 1
    d = dicts[0]
    assert d["intent"] == "click X"
    assert d["type"] == "click"
    assert d["grounding"] is True
    assert d["section"] == "extraction"
    # Plan fidelity (P0 #1): params + hints must survive the wire so
    # downstream form / region handlers don't have to re-parse intent
    # prose for structured fields the decomposer already extracted.
    assert d["params"] == {"label": "X", "kind": "button"}
    assert d["hints"] == {"region": "form-footer"}


def test_micro_plan_steps_to_dicts_defaults_when_params_hints_absent():
    """Legacy callers / minimal fakes that don't define ``params`` or
    ``hints`` at all (e.g. early test doubles) still serialise to dicts
    with both keys present as empty dicts. Receiving side reconstructs
    via ``MicroIntent(**d)`` which expects every field — empties are
    safe because :class:`MicroIntent` ``params`` / ``hints`` defaults
    are also ``{}``.
    """
    class MinimalIntent:
        intent = "navigate"
        type = "navigate"
        verify = ""
        budget = 3
        reverse = ""
        grounding = False
        claude_only = False
        loop_target = -1
        loop_count = 0
        section = ""
        required = False
        gate = False

    dicts = micro_plan_steps_to_dicts([MinimalIntent()])
    assert dicts[0]["params"] == {}
    assert dicts[0]["hints"] == {}


def test_micro_plan_steps_to_dicts_roundtrip_through_microintent():
    """End-to-end: a real :class:`MicroIntent` with params + hints
    serializes → dict → reconstructs via ``MicroIntent(**dict)``
    with every structured field intact. This is the contract
    ``baseten_server/runtime.py:1360`` and ``modal_cua_server.py:712``
    rely on to rebuild the runner-side plan from the wire payload.
    """
    from mantis_agent.plan_decomposer import MicroIntent

    original = MicroIntent(
        intent="Click Update Lead button",
        type="submit",
        params={"label": "Update Lead", "kind": "button",
                "aliases": ["Save", "Save Changes"]},
        hints={"region": "form-footer", "layout": "single"},
        section="edit",
        required=True,
    )
    serialised = micro_plan_steps_to_dicts([original])[0]
    rebuilt = MicroIntent(**serialised)
    assert rebuilt.intent == original.intent
    assert rebuilt.type == original.type
    assert rebuilt.params == original.params
    assert rebuilt.hints == original.hints
    assert rebuilt.required is True


def test_plan_signature_distinguishes_plans_with_differing_params():
    """Two plans with identical ``intent``/``type`` skeletons but
    different ``params`` (e.g. one targets ``Update Lead``, the other
    ``Save Changes``) must hash to DIFFERENT signatures. Without this,
    a resume could silently reuse a stale checkpoint against a
    logically-different plan and corrupt step indices.
    """
    from mantis_agent.plan_decomposer import MicroIntent, MicroPlan
    from mantis_agent.gym.checkpoint_manager import CheckpointManager

    plan_a = MicroPlan(domain="x", steps=[
        MicroIntent(intent="Click button", type="submit",
                    params={"label": "Update Lead"}),
    ])
    plan_b = MicroPlan(domain="x", steps=[
        MicroIntent(intent="Click button", type="submit",
                    params={"label": "Save Changes"}),
    ])
    sig_a = CheckpointManager.compute_plan_signature(plan_a)
    sig_b = CheckpointManager.compute_plan_signature(plan_b)
    assert sig_a != sig_b, (
        "plans differing only in params.label must produce different "
        "checkpoint signatures"
    )


def test_plan_signature_distinguishes_plans_with_differing_hints():
    """Same shape as above but for ``hints`` — a plan that pins
    ``hints.region: form-footer`` must NOT share a signature with one
    that defaults to no region hint.
    """
    from mantis_agent.plan_decomposer import MicroIntent, MicroPlan
    from mantis_agent.gym.checkpoint_manager import CheckpointManager

    plan_with_hint = MicroPlan(domain="x", steps=[
        MicroIntent(intent="Click", type="submit",
                    hints={"region": "form-footer"}),
    ])
    plan_no_hint = MicroPlan(domain="x", steps=[
        MicroIntent(intent="Click", type="submit"),
    ])
    sig_with = CheckpointManager.compute_plan_signature(plan_with_hint)
    sig_without = CheckpointManager.compute_plan_signature(plan_no_hint)
    assert sig_with != sig_without


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


# ── load_plan_file + merge_runtime ─────────────────────────────────


def _write_plan(tmp_path: Path, body: object) -> Path:
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(body))
    return p


def test_load_plan_file_bare_array(tmp_path: Path) -> None:
    p = _write_plan(tmp_path, [{"intent": "go", "type": "navigate"}])
    steps, runtime = load_plan_file(p)
    assert steps == [{"intent": "go", "type": "navigate"}]
    assert runtime == {}


def test_load_plan_file_wrapped_with_runtime(tmp_path: Path) -> None:
    p = _write_plan(tmp_path, {
        "runtime": {"proxy_disabled": True, "max_cost": 2.5},
        "steps": [{"intent": "go", "type": "navigate"}],
    })
    steps, runtime = load_plan_file(p)
    assert steps == [{"intent": "go", "type": "navigate"}]
    assert runtime == {"proxy_disabled": True, "max_cost": 2.5}


def test_load_plan_file_wrapped_without_runtime(tmp_path: Path) -> None:
    p = _write_plan(tmp_path, {"steps": [{"intent": "x", "type": "click"}]})
    steps, runtime = load_plan_file(p)
    assert steps == [{"intent": "x", "type": "click"}]
    assert runtime == {}


def test_load_plan_file_drops_unknown_runtime_keys(tmp_path: Path) -> None:
    """Forward-compat: unknown ``runtime`` keys never leak through to
    callers (who'd hit a ``build_micro_suite`` TypeError)."""
    p = _write_plan(tmp_path, {
        "runtime": {"proxy_disabled": True, "future_flag": "ignored"},
        "steps": [],
    })
    _steps, runtime = load_plan_file(p)
    assert runtime == {"proxy_disabled": True}
    assert "future_flag" not in runtime


def test_load_plan_file_rejects_non_list_steps(tmp_path: Path) -> None:
    p = _write_plan(tmp_path, {"steps": "oops"})
    with pytest.raises(ValueError, match="'steps' must be a list"):
        load_plan_file(p)


def test_load_plan_file_rejects_non_object_runtime(tmp_path: Path) -> None:
    p = _write_plan(tmp_path, {"runtime": "no", "steps": []})
    with pytest.raises(ValueError, match="'runtime' must be an object"):
        load_plan_file(p)


def test_load_plan_file_rejects_scalar(tmp_path: Path) -> None:
    p = _write_plan(tmp_path, 42)
    with pytest.raises(ValueError, match="expected array of steps"):
        load_plan_file(p)


def test_merge_runtime_plan_only() -> None:
    out = merge_runtime({"proxy_disabled": True, "max_cost": 2.0})
    assert out == {"proxy_disabled": True, "max_cost": 2.0}


def test_merge_runtime_submission_override_wins() -> None:
    """Explicit non-None overrides beat the plan's declared default."""
    out = merge_runtime(
        {"proxy_disabled": True, "max_cost": 2.0},
        proxy_disabled=False,
    )
    assert out["proxy_disabled"] is False
    assert out["max_cost"] == 2.0  # untouched override falls back to plan


def test_merge_runtime_none_override_falls_back_to_plan() -> None:
    """``None`` means 'caller didn't set this' — keep the plan default."""
    out = merge_runtime(
        {"proxy_disabled": True},
        proxy_disabled=None,
        max_cost=None,
    )
    assert out == {"proxy_disabled": True}


def test_merge_runtime_empty_plan() -> None:
    out = merge_runtime(None, proxy_disabled=True)
    assert out == {"proxy_disabled": True}


def test_merge_runtime_ignores_unknown_overrides() -> None:
    """Unknown kwargs don't leak into the merge output (so the result
    can be splatted into ``build_micro_suite(**runtime)`` safely)."""
    out = merge_runtime({"proxy_disabled": True}, future_flag="x")
    assert out == {"proxy_disabled": True}


def test_merge_runtime_output_splats_into_build_micro_suite(tmp_path: Path) -> None:
    """End-to-end: load → merge → build_micro_suite consumes without
    TypeError. Pins the contract callers depend on."""
    p = _write_plan(tmp_path, {
        "runtime": {"proxy_disabled": True, "max_cost": 3.0, "max_time_minutes": 15},
        "steps": [{"intent": "go", "type": "navigate"}],
    })
    steps, plan_runtime = load_plan_file(p)
    runtime = merge_runtime(plan_runtime)
    suite = build_micro_suite(steps, "test_domain", **runtime)
    assert suite["_proxy_disabled"] is True
    assert suite["_max_cost"] == 3.0
    assert suite["_max_time_minutes"] == 15


def test_runtime_proxy_provider_flows_to_suite(tmp_path: Path) -> None:
    """``proxy_provider`` declared in the plan lands on
    ``task_suite['_proxy_provider']`` which Modal executors read at
    submission time."""
    p = _write_plan(tmp_path, {
        "runtime": {"proxy_provider": "privateproxy", "proxy_city": "miami"},
        "steps": [{"intent": "go", "type": "navigate"}],
    })
    steps, plan_runtime = load_plan_file(p)
    runtime = merge_runtime(plan_runtime)
    suite = build_micro_suite(steps, "luma", **runtime)
    assert suite["_proxy_provider"] == "privateproxy"
    assert suite["_proxy_city"] == "miami"


def test_runtime_provider_override_wins() -> None:
    """Submission-time ``proxy_provider`` beats the plan's choice."""
    out = merge_runtime(
        {"proxy_provider": "privateproxy"},
        proxy_provider="oxylabs",
    )
    assert out["proxy_provider"] == "oxylabs"


# ── #508 extraction artifacts ─────────────────────────────────────


class FakeStepResultWithFields(FakeStepResult):
    """FakeStepResult that also exposes ``extracted_fields`` so the
    aggregator can pick up structured rows."""

    def __init__(self, *args, extracted_fields=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.extracted_fields = dict(extracted_fields or {})


def test_build_micro_result_emits_artifacts_for_extracted_fields():
    """Schema-keyed rows on StepResult land in result['artifacts'] as a
    structured_data entry. The legacy ``leads`` string list is
    untouched so existing callers keep working."""
    from mantis_agent.server_utils import build_micro_result

    runner = FakeMicroRunner(leads=["VIABLE|Title: ML | URL: https://a"])
    steps = [
        FakeStepResultWithFields(
            0, "extract row 1", True, data="VIABLE|...",
            extracted_fields={"title": "ML Engineer", "url": "https://a", "department": "Eng"},
        ),
        FakeStepResultWithFields(
            1, "extract row 2", True, data="VIABLE|...",
            extracted_fields={"title": "Designer", "url": "https://b", "location": "NY"},
        ),
    ]
    result = build_micro_result(
        runner, steps,
        run_id="r1", provider="modal", session_name="t",
        model_name="claude", elapsed_seconds=1.0,
    )

    assert "artifacts" in result
    arts = result["artifacts"]
    assert len(arts) == 1, arts
    structured = arts[0]
    assert structured["kind"] == "structured_data"
    assert structured["name"] == "extracted_rows"
    assert structured["mime_type"] == "application/json"
    # First-seen field order preserved across rows
    assert structured["schema"]["fields"] == ["title", "url", "department", "location"]
    assert structured["row_count"] == 2
    assert structured["data"][0] == {
        "title": "ML Engineer", "url": "https://a", "department": "Eng",
    }
    # Legacy leads list untouched
    assert result["leads"] == ["VIABLE|Title: ML | URL: https://a"]


def test_build_micro_result_artifacts_empty_when_no_structured_rows():
    """Runs that produce no extract_data rows yield artifacts=[]
    rather than an artifact with empty data — keeps the result schema
    predictable for callers that always iterate ``artifacts``."""
    from mantis_agent.server_utils import build_micro_result

    runner = FakeMicroRunner(leads=[])
    steps = [FakeStepResult(0, "navigate", True)]
    result = build_micro_result(
        runner, steps,
        run_id="r1", provider="modal", session_name="t",
        model_name="claude", elapsed_seconds=1.0,
    )
    assert result["artifacts"] == []


def test_persist_run_artifacts_writes_dynamic_csv_and_json(tmp_path: Path):
    """``extracted_rows.csv`` header matches the schema fields exactly —
    no marketplace columns leak in, missing values are empty strings."""
    from mantis_agent.server_utils import persist_run_artifacts

    result = {
        "run_id": "r1",
        "leads": [{"url": "https://a"}, {"url": "https://b"}],
        "artifacts": [{
            "name": "extracted_rows",
            "kind": "structured_data",
            "mime_type": "application/json",
            "schema": {"fields": ["title", "url", "department"]},
            "row_count": 2,
            "data": [
                {"title": "ML Engineer", "url": "https://a", "department": "Eng"},
                {"title": "Designer", "url": "https://b"},  # missing department
            ],
        }],
    }
    file_arts = persist_run_artifacts(result, tmp_path, run_id="r1")

    names = [a["name"] for a in file_arts]
    assert names == ["leads.csv", "extracted_rows.csv", "extracted_rows.json"]

    rows_csv = (tmp_path / "extracted_rows.csv").read_text().splitlines()
    assert rows_csv[0] == "title,url,department"
    assert rows_csv[1] == "ML Engineer,https://a,Eng"
    assert rows_csv[2] == "Designer,https://b,"

    rows_json = json.loads((tmp_path / "extracted_rows.json").read_text())
    assert rows_json[0]["title"] == "ML Engineer"
    assert rows_json[1].get("department", "") == ""

    # Legacy leads.csv still uses fixed columns — back-compat guard
    legacy_header = (tmp_path / "leads.csv").read_text().splitlines()[0]
    assert legacy_header == "status,year,make,model,price,phone,seller,url,raw"

    # download_url shape
    by_name = {a["name"]: a for a in file_arts}
    assert by_name["leads.csv"]["download_url"] == "/v1/runs/r1/artifacts/leads.csv"
    assert by_name["extracted_rows.csv"]["download_url"] == "/v1/runs/r1/artifacts/extracted_rows.csv"
    assert by_name["extracted_rows.json"]["download_url"] == "/v1/runs/r1/artifacts/extracted_rows.json"


def test_persist_run_artifacts_skips_when_no_data(tmp_path: Path):
    """No leads + no structured rows → no files written, empty list."""
    from mantis_agent.server_utils import persist_run_artifacts

    file_arts = persist_run_artifacts(
        {"run_id": "r1", "leads": [], "artifacts": []}, tmp_path, run_id="r1",
    )
    assert file_arts == []
    assert not (tmp_path / "leads.csv").exists()
    assert not (tmp_path / "extracted_rows.csv").exists()


def test_save_result_json_writes_dynamic_csv_in_artifacts_dir(tmp_path: Path):
    """End-to-end: ``save_result_json`` writes the schema CSV alongside
    the legacy one and the result envelope advertises both."""
    result = {
        "run_id": "r9",
        "leads": [{"url": "https://a"}],
        "artifacts": [{
            "name": "extracted_rows",
            "kind": "structured_data",
            "mime_type": "application/json",
            "schema": {"fields": ["title", "url"]},
            "row_count": 1,
            "data": [{"title": "T", "url": "https://a"}],
        }],
    }
    save_result_json(result, tmp_path, "session")

    legacy_csv = tmp_path / "session_leads_r9.csv"
    assert legacy_csv.exists()
    assert legacy_csv.read_text().splitlines()[0].startswith("status,year,make,model")

    arts_dir = tmp_path / "session_artifacts_r9"
    assert (arts_dir / "extracted_rows.csv").exists()
    assert (arts_dir / "extracted_rows.json").exists()
    assert (arts_dir / "extracted_rows.csv").read_text().splitlines()[0] == "title,url"

    # File artifacts merged into result["artifacts"]
    kinds = [(a["name"], a["kind"]) for a in result["artifacts"]]
    assert ("extracted_rows", "structured_data") in kinds
    assert ("leads.csv", "file") in kinds
    assert ("extracted_rows.csv", "file") in kinds
    assert ("extracted_rows.json", "file") in kinds


def test_extraction_schema_from_dict_round_trips_custom_fields():
    """Non-marketplace schema (title/department/location/url) survives
    through ``ExtractionSchema.from_dict`` with the right required set
    and entity name — proves payload-level extraction_schema is honored."""
    from mantis_agent.extraction.schema import ExtractionSchema

    s = ExtractionSchema.from_dict({
        "entity_name": "job",
        "fields": [
            {"name": "title", "required": True},
            {"name": "department", "required": False},
            {"name": "location", "required": False},
            {"name": "url", "required": True},
        ],
    })
    assert s.entity_name == "job"
    assert s.field_names() == ["title", "department", "location", "url"]
    assert sorted(s.required_fields) == ["title", "url"]


def test_extraction_schema_from_dict_validates_shape():
    """Missing / malformed inputs raise ValueError so callers fail fast
    instead of silently producing an empty-fields schema."""
    from mantis_agent.extraction.schema import ExtractionSchema

    with pytest.raises(ValueError):
        ExtractionSchema.from_dict({})
    with pytest.raises(ValueError):
        ExtractionSchema.from_dict({"fields": []})
    with pytest.raises(ValueError):
        ExtractionSchema.from_dict({"fields": [{"missing_name": "x"}]})


def test_predict_request_accepts_typed_extraction_schema():
    """The schema dict is a first-class typed PredictRequest field —
    previously accepted via extra='allow' but silently ignored."""
    from mantis_agent.api_schemas import PredictRequest

    req = PredictRequest(
        task_suite={"steps": []},
        extraction_schema={
            "fields": [{"name": "title", "required": True}],
            "entity_name": "post",
        },
    )
    assert req.extraction_schema is not None
    assert req.extraction_schema["entity_name"] == "post"


def test_extracted_rows_csv_omits_legacy_marketplace_columns(tmp_path: Path):
    """Regression guard: the dynamic CSV must NOT include
    ``status``/``year``/``make`` etc. when the schema doesn't ask for
    them. That's the whole point of #508."""
    from mantis_agent.server_utils import write_extracted_rows_csv

    write_extracted_rows_csv(
        tmp_path / "out.csv",
        rows=[{"title": "ML", "url": "https://a"}],
        fieldnames=["title", "url"],
    )
    header = (tmp_path / "out.csv").read_text().splitlines()[0]
    assert header == "title,url"
    for legacy in ("status", "year", "make", "model", "price", "phone", "seller", "raw"):
        assert legacy not in header
