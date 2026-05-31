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
        "Search boats near {zip_code} within {search_radius} miles.\n"
        "Log into PopYachts with password {pop_password}.\n",
    )
    return p


def _make(tmp_path: Path, poster: FakePoster, decomposer: FakeDecomposer) -> LiveRunFn:
    return LiveRunFn(
        plan_path=_plan_file(tmp_path),
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
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setenv("POP_PASSWORD", "s3cr3t-from-env")
    decomposer = FakeDecomposer()
    poster = FakePoster(statuses=["succeeded"], result_envelope={})
    run = _make(tmp_path, poster, decomposer)

    run(_task(seed=42), None, _result("frozen"))
    run(_task(seed=7), None, _result("S0_retrieval"))

    # Decompose is memoised — one call across both submits.
    assert len(decomposer.texts) == 1
    text = decomposer.texts[0]
    assert "33131" in text  # {zip_code} filled
    assert "s3cr3t-from-env" in text  # {pop_password} from env, never hardcoded
    assert "{pop_password}" not in text
    # Each submit gets a fresh workflow_id.
    wf0 = poster.submit_bodies[0]["workflow_id"]
    wf1 = poster.submit_bodies[1]["workflow_id"]
    assert wf0 != wf1


def test_pop_password_not_hardcoded_in_source() -> None:
    """The real PopYachts password must never live in a tracked file."""
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
