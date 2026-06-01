"""Tests for the live Phase-2 run_fn adapter (no spend, no network).

:class:`~experiments.learning_allocator.live_runner.LiveRunFn` is the only
spending piece of the Phase-2 wiring, so its three I/O seams (``post_fn``,
``pull_cost_fn``, ``decompose_fn``) are all injected here with fakes. The tests
pin the behaviours that would silently corrupt a live run or leak spend:

* NO proxies in the submit (the sim env is reached directly),
* the substrate → hint-store-flag mapping (frozen vs S0),
* the ``run_result`` shape the reward channels read, and the verdict/cost wiring,
* one decompose shared across calls + a fresh ``workflow_id`` per submit,
* a hard refusal to start ``main`` without a sim-env URL.
"""

from __future__ import annotations

import json
from pathlib import Path

from mantis_agent.learning.reward import cost_channel, proxy_channel
from mantis_agent.learning.substrates.base import SubstrateResult
from mantis_agent.plan_decomposer import MicroIntent, MicroPlan

from experiments.learning_allocator import live_runner
from experiments.learning_allocator.live_runner import LiveRunFn

# ── fakes ────────────────────────────────────────────────────────────────


def _stub_plan() -> MicroPlan:
    return MicroPlan(
        steps=[MicroIntent(intent="open boattrader", type="navigate")],
        domain="boattrader_scrape",
        plan_hash="deadbeef",
    )


class FakeDecomposer:
    """Records every plan text it's asked to decompose; returns a stub plan."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    def __call__(self, text: str) -> MicroPlan:
        self.texts.append(text)
        return _stub_plan()


class FakePoster:
    """Scriptable stand-in for the Modal HTTP seam.

    ``statuses`` is the sequence returned on successive ``action=status`` polls
    (last value repeats); ``result_envelope`` is what ``action=result`` returns
    under ``result``. Submit (no ``action``) returns ``run_id``.
    """

    def __init__(
        self, *, submit_status: int = 200, run_id: str = "run-1",
        statuses: list[str] | None = None, result_envelope: dict | None = None,
    ) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.submit_status = submit_status
        self.run_id = run_id
        self.statuses = list(statuses or ["succeeded"])
        self.result_envelope = result_envelope
        self._idx = 0

    def __call__(self, path: str, body: dict) -> tuple[int, dict]:
        self.calls.append((path, body))
        action = body.get("action")
        if action == "status":
            st = self.statuses[min(self._idx, len(self.statuses) - 1)]
            self._idx += 1
            halt = "" if st == "succeeded" else "some_halt"
            return 200, {"status": st, "halt_reason": halt}
        if action == "result":
            return 200, {"status": "succeeded", "result": self.result_envelope}
        return self.submit_status, {"run_id": self.run_id}

    @property
    def submit_bodies(self) -> list[dict]:
        return [b for p, b in self.calls if "task_suite" in b]


def _fake_cost(profile_id: str, workflow_id: str) -> tuple[float, str]:  # noqa: ARG001
    return 0.37, ""


def _plan_file(tmp_path: Path) -> Path:
    p = tmp_path / "plan.txt"
    p.write_text(
        "Navigate to {env_url}/boats/.\n"
        "Search boats near {zip_code} within {search_radius} miles.\n",
    )
    return p


def _make(tmp_path: Path, poster: FakePoster, decomposer: FakeDecomposer) -> LiveRunFn:
    return LiveRunFn(
        plan_path=_plan_file(tmp_path),
        env_url="https://sim.example/env",
        post_fn=poster,
        pull_cost_fn=_fake_cost,
        decompose_fn=decomposer,
        poll_interval_s=0.0,
    )


def _result(substrate: str) -> SubstrateResult:
    return SubstrateResult(substrate=substrate, applied=True)


# ── _runtime: no proxies ─────────────────────────────────────────────────


def test_runtime_disables_proxy_and_carries_no_proxy_keys(tmp_path: Path) -> None:
    run = _make(tmp_path, FakePoster(), FakeDecomposer())
    runtime = run._runtime()
    assert runtime["proxy_disabled"] is True
    assert runtime["max_cost"] == 1.0
    assert runtime["max_time_minutes"] == 30
    # The sim env is reached directly — never route through a proxy.
    for key in ("proxy_provider", "proxy_city", "proxy_state", "proxy_country"):
        assert not runtime.get(key), f"{key} must be empty/absent for a sim-env run"


# ── substrate → hint-store flag mapping ──────────────────────────────────


def test_frozen_disables_hint_store(tmp_path: Path) -> None:
    run = _make(tmp_path, FakePoster(), FakeDecomposer())
    suite: dict = {}
    run._apply_substrate(suite, "frozen")
    assert suite["_hint_store_disabled"] is True
    assert "_hint_store_dict_name" not in suite


def test_s0_binds_shared_hint_dict(tmp_path: Path) -> None:
    run = _make(tmp_path, FakePoster(), FakeDecomposer())
    suite: dict = {}
    run._apply_substrate(suite, "S0_retrieval")
    assert suite["_hint_store_dict_name"] == "la-bt01-hints"
    assert "_hint_store_disabled" not in suite


# ── end-to-end __call__ (faked I/O) ──────────────────────────────────────


def test_call_returns_reward_readable_result_on_success(tmp_path: Path) -> None:
    poster = FakePoster(
        statuses=["succeeded"],
        result_envelope={"dynamic_verification_summary": {"verdict": "pass"}},
    )
    run = _make(tmp_path, poster, FakeDecomposer())

    out = run(_task(seed=42), None, _result("frozen"))

    # The reward channels must be able to read this dict verbatim.
    assert proxy_channel(out) == "pass"
    assert cost_channel(out) == 0.37
    assert out["_terminal_status"] == "succeeded"
    assert out["_substrate"] == "frozen"


def test_submit_body_has_no_proxy_and_frozen_flag(tmp_path: Path) -> None:
    poster = FakePoster(statuses=["succeeded"], result_envelope={})
    run = _make(tmp_path, poster, FakeDecomposer())

    run(_task(seed=42), None, _result("frozen"))

    body = poster.submit_bodies[0]
    assert body["proxy_disabled"] is True
    assert "proxy_provider" not in body
    assert "proxy_city" not in body
    assert body["task_suite"]["_hint_store_disabled"] is True
    assert body["cua_model"] == "holo3"
    assert body["detached"] is True


def test_suite_carries_daytona_skip_header_by_default(tmp_path: Path) -> None:
    # The sim env sits behind the Daytona preview proxy, which blocks the
    # runner's browser-UA requests with an interstitial unless this header
    # rides every request. The suite must carry it so modal_cua_server's
    # setup_env opens the persistent header session.
    poster = FakePoster(statuses=["succeeded"], result_envelope={})
    run = _make(tmp_path, poster, FakeDecomposer())

    run(_task(seed=42), None, _result("frozen"))

    headers = poster.submit_bodies[0]["task_suite"]["_browser_extra_headers"]
    assert headers == {"X-Daytona-Skip-Preview-Warning": "true"}


def test_empty_browser_headers_omits_suite_key(tmp_path: Path) -> None:
    # A direct (non-Daytona) env clears the default → no header key, so the
    # remote env's header session stays a no-op (production CF runs path).
    poster = FakePoster(statuses=["succeeded"], result_envelope={})
    run = _make(tmp_path, poster, FakeDecomposer())
    run.browser_extra_headers = {}

    run(_task(seed=42), None, _result("frozen"))

    assert "_browser_extra_headers" not in poster.submit_bodies[0]["task_suite"]


class FakeReset:
    """Records env resets so ordering against the submit can be asserted."""

    def __init__(self, log: list) -> None:
        self.log = log
        self.calls: list[tuple[str, str]] = []

    def __call__(self, env_url: str, admin_token: str) -> None:
        self.calls.append((env_url, admin_token))
        self.log.append("reset")


def test_reset_fires_before_submit_when_admin_token_set(tmp_path: Path) -> None:
    # Shared log captures the interleaving of reset vs submit.
    log: list[str] = []
    reset = FakeReset(log)

    class LoggingPoster(FakePoster):
        def __call__(self, path: str, body: dict) -> tuple[int, dict]:
            if "task_suite" in body:
                log.append("submit")
            return super().__call__(path, body)

    poster = LoggingPoster(statuses=["succeeded"], result_envelope={})
    run = LiveRunFn(
        plan_path=_plan_file(tmp_path),
        env_url="https://sim.example/env",
        admin_token="tok-123",
        post_fn=poster,
        pull_cost_fn=_fake_cost,
        decompose_fn=FakeDecomposer(),
        reset_fn=reset,
        poll_interval_s=0.0,
    )

    run(_task(seed=42), None, _result("frozen"))

    # Each run resets the cumulative store *before* it submits, so the
    # precision-sensitive oracle grades only this run's leads.
    assert reset.calls == [("https://sim.example/env", "tok-123")]
    assert log[0] == "reset" and log.index("reset") < log.index("submit")


def test_reset_skipped_without_admin_token(tmp_path: Path) -> None:
    reset = FakeReset([])
    run = LiveRunFn(
        plan_path=_plan_file(tmp_path),
        env_url="https://sim.example/env",  # admin_token left empty
        post_fn=FakePoster(statuses=["succeeded"], result_envelope={}),
        pull_cost_fn=_fake_cost,
        decompose_fn=FakeDecomposer(),
        reset_fn=reset,
        poll_interval_s=0.0,
    )

    run(_task(seed=42), None, _result("frozen"))

    # No admin token → offline-inert: never touch the env.
    assert reset.calls == []


def test_verdict_falls_back_to_fail_on_halt(tmp_path: Path) -> None:
    poster = FakePoster(statuses=["halted"])
    run = _make(tmp_path, poster, FakeDecomposer())

    out = run(_task(seed=7), None, _result("S0_retrieval"))

    assert proxy_channel(out) == "fail"
    assert out["_terminal_status"] == "halted"
    # A halted run must not query action=result (envelope unavailable).
    assert not any(b.get("action") == "result" for _, b in poster.calls)


def test_succeeded_run_without_summary_defaults_pass(tmp_path: Path) -> None:
    poster = FakePoster(statuses=["succeeded"], result_envelope={"leads": []})
    run = _make(tmp_path, poster, FakeDecomposer())

    out = run(_task(seed=42), None, _result("frozen"))

    assert proxy_channel(out) == "pass"


def test_submit_failure_yields_failed_result_without_spend(tmp_path: Path) -> None:
    poster = FakePoster(submit_status=503)
    run = _make(tmp_path, poster, FakeDecomposer())

    out = run(_task(seed=42), None, _result("frozen"))

    assert proxy_channel(out) == "fail"
    assert cost_channel(out) == 0.0
    # No poll happens after a failed submit.
    assert all(b.get("action") != "status" for _, b in poster.calls)


def test_plan_decomposed_once_with_substituted_placeholders(
    tmp_path: Path,
) -> None:
    decomposer = FakeDecomposer()
    poster = FakePoster(statuses=["succeeded"], result_envelope={})
    run = _make(tmp_path, poster, decomposer)

    run(_task(seed=42), None, _result("frozen"))
    run(_task(seed=7), None, _result("S0_retrieval"))

    # Decompose is memoised — one call across both submits.
    assert len(decomposer.texts) == 1
    text = decomposer.texts[0]
    assert "33131" in text  # {zip_code} filled
    assert "https://sim.example/env" in text  # {env_url} points the nav at the sim env
    assert "{env_url}" not in text
    # Each submit gets a fresh workflow_id.
    wf0 = poster.submit_bodies[0]["workflow_id"]
    wf1 = poster.submit_bodies[1]["workflow_id"]
    assert wf0 != wf1


# ── .json pre-decomposed plan path (no decompose, no spend) ──────────────


def _json_plan_file(tmp_path: Path) -> Path:
    """A minimal pre-decomposed micro-plan with a vision guard + {{ENV_URL}}."""
    p = tmp_path / "bt02_spec_lookup.json"
    p.write_text(json.dumps({
        "domain": "boattrader_scrape",
        "shapes": ["listings", "form"],
        "steps": [
            {
                "index": 0, "type": "navigate",
                "intent": "Navigate to {{ENV_URL}}/boats/",
                "params": {"url": "{{ENV_URL}}/boats/", "wait_after_load_seconds": 8},
                "section": "setup", "required": True,
            },
            {
                "index": 1, "type": "detect_visible",
                "intent": "Is the engine make exactly Caterpillar?",
                "out_var": "is_caterpillar",
                "params": {"out_var": "is_caterpillar"},
                "hints": {"out_var": "is_caterpillar"},
                "claude_only": True, "section": "extraction", "required": False,
            },
            {
                "index": 2, "type": "submit",
                "intent": "Click Contact Seller to send the inquiry",
                "params": {"label": "Contact Seller", "kind": "button"},
                "guard": "is_caterpillar",
                "hints": {"guard": "is_caterpillar"},
                "section": "extraction", "required": False,
            },
        ],
    }))
    return p


def test_json_plan_loads_directly_without_decompose(tmp_path: Path) -> None:
    decomposer = FakeDecomposer()
    poster = FakePoster(statuses=["succeeded"], result_envelope={})
    run = LiveRunFn(
        plan_path=_json_plan_file(tmp_path),
        env_url="https://sim.example/env",
        post_fn=poster,
        pull_cost_fn=_fake_cost,
        decompose_fn=decomposer,
        poll_interval_s=0.0,
    )

    run(_task(seed=42), None, _result("frozen"))

    # A .json plan is already decomposed — the LLM decomposer never runs
    # (the deterministic guard must not be re-paraphrased away).
    assert decomposer.texts == []
    steps = poster.submit_bodies[0]["task_suite"]["_micro_plan"]
    # {{ENV_URL}} is substituted with the live sim-env URL on the nav step.
    assert steps[0]["params"]["url"] == "https://sim.example/env/boats/"
    assert "{{ENV_URL}}" not in json.dumps(steps)
    # The conditional guard the decomposer would flatten survives verbatim,
    # both at top level (read by the suite builder) and in hints (the
    # runner's triple-fallback resolution).
    assert steps[1]["type"] == "detect_visible"
    assert steps[1]["out_var"] == "is_caterpillar"
    assert steps[2]["guard"] == "is_caterpillar"
    assert steps[2]["hints"]["guard"] == "is_caterpillar"


def test_json_plan_memoised_across_submits(tmp_path: Path) -> None:
    poster = FakePoster(statuses=["succeeded"], result_envelope={})
    run = LiveRunFn(
        plan_path=_json_plan_file(tmp_path),
        env_url="https://sim.example/env",
        post_fn=poster,
        pull_cost_fn=_fake_cost,
        decompose_fn=FakeDecomposer(),
        poll_interval_s=0.0,
    )

    run(_task(seed=42), None, _result("frozen"))
    run(_task(seed=7), None, _result("S0_retrieval"))

    # One load shared across both submits, each with a fresh workflow_id.
    wf0 = poster.submit_bodies[0]["workflow_id"]
    wf1 = poster.submit_bodies[1]["workflow_id"]
    assert wf0 != wf1
    assert poster.submit_bodies[0]["task_suite"]["_hint_store_disabled"] is True
    assert poster.submit_bodies[1]["task_suite"]["_hint_store_dict_name"]


def test_tasks_for_plan_matches_json_suffix_by_stem() -> None:
    """clusters.json names the logical plan (``bt02_spec_lookup``); the on-disk
    file may be ``bt02_spec_lookup.json``. Both must select the same tasks."""
    bare = {t.name for t in live_runner.tasks_for_plan("bt02_spec_lookup")}
    suffixed = {t.name for t in live_runner.tasks_for_plan("plans/bt02_spec_lookup.json")}
    assert bare == suffixed
    assert "bt02_spec_lookup_visible" in suffixed


def test_credential_not_hardcoded_in_source() -> None:
    """A real lead-site login password must never live in a tracked file."""
    src = Path(live_runner.__file__).read_text()
    assert "SelfService" not in src


# ── main() preflight ─────────────────────────────────────────────────────


def test_main_refuses_without_sim_env_url(monkeypatch) -> None:
    monkeypatch.setattr(live_runner, "_read_env", lambda key: "")
    assert live_runner.main([]) == 2


# ── helpers ──────────────────────────────────────────────────────────────


def _task(*, seed: int):
    from mantis_agent.learning.eval import EvalTask

    return EvalTask(
        name=f"bt01_s{seed}", cluster="capability", split="visible", seed=seed,
        task_id="BT01_lead_capture_filtered_search", plan="boattrader_scrape",
        status="ready",
    )
