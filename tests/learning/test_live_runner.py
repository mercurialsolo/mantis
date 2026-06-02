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

import pytest

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


def test_s1_ships_exemplars_and_freezes_s0(tmp_path: Path) -> None:
    # S1's lift must be attributable to the exemplar replay alone, so the S0
    # anchor store is frozen while the worked steps ride along.
    run = _make(tmp_path, FakePoster(), FakeDecomposer())
    run.exemplars = [{"intent": "reveal phone", "type": "click"}]
    suite: dict = {}
    run._apply_substrate(suite, "S1_exemplar")
    assert suite["_hint_store_disabled"] is True
    assert suite["_exemplars"] == [{"intent": "reveal phone", "type": "click"}]
    assert "_hint_store_dict_name" not in suite


def test_s1_without_exemplars_omits_suite_key(tmp_path: Path) -> None:
    # No pre-extracted exemplars ⇒ no _exemplars key (the remote overlay is a
    # no-op), but S0 still frozen so the rung stays isolated.
    run = _make(tmp_path, FakePoster(), FakeDecomposer())
    suite: dict = {}
    run._apply_substrate(suite, "S1_exemplar")
    assert suite["_hint_store_disabled"] is True
    assert "_exemplars" not in suite


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


def test_suite_carries_daytona_proxy_and_consent_headers_by_default(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch",
) -> None:
    # The sim env sits behind the Daytona preview proxy: every browser request
    # needs the skip-warning header (suppresses the UA interstitial) AND the
    # per-sandbox preview token (else a 307 to Auth0). It also pre-seeds the
    # cookie-consent cookie so the OneTrust banner — which the listings
    # pre-scan reads as a consent wall → page_blocked — never renders. The
    # suite must carry all three so modal_cua_server's setup_env opens the
    # persistent header session with them.
    monkeypatch.setenv("LA_ENV_PREVIEW_TOKEN", "tok-xyz")
    poster = FakePoster(statuses=["succeeded"], result_envelope={})
    run = _make(tmp_path, poster, FakeDecomposer())

    run(_task(seed=42), None, _result("frozen"))

    headers = poster.submit_bodies[0]["task_suite"]["_browser_extra_headers"]
    assert headers == {
        "X-Daytona-Skip-Preview-Warning": "true",
        "x-daytona-preview-token": "tok-xyz",
        "Cookie": "bt_cookie_consent=decline",
    }


def test_suite_omits_preview_token_when_env_unset(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch",
) -> None:
    # No LA_ENV_PREVIEW_TOKEN (e.g. a direct env or an un-exported shell) → the
    # token header is dropped, but the skip + consent headers still ride.
    monkeypatch.delenv("LA_ENV_PREVIEW_TOKEN", raising=False)
    poster = FakePoster(statuses=["succeeded"], result_envelope={})
    run = _make(tmp_path, poster, FakeDecomposer())

    run(_task(seed=42), None, _result("frozen"))

    headers = poster.submit_bodies[0]["task_suite"]["_browser_extra_headers"]
    assert "x-daytona-preview-token" not in headers
    assert headers["X-Daytona-Skip-Preview-Warning"] == "true"
    assert headers["Cookie"] == "bt_cookie_consent=decline"


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


# ── incremental results streaming ────────────────────────────────────────


def _outcome(task, substrate: str, score: float, dollars: float):
    from mantis_agent.learning.orchestrator import TaskOutcome
    from mantis_agent.learning.reward import RewardRecord

    rr = RewardRecord(
        task_id=task.task_id or task.name, oracle_score=score,
        oracle_passed=score >= 0.5, proxy_verdict="pass", proxy_score=0.0,
        dollars=dollars, reward=score - 0.1 * dollars,
        false_pass=False, false_fail=False,
    )
    return TaskOutcome(
        task_name=task.name, task_id=task.task_id or task.name,
        cluster=task.cluster, substrate=substrate, reward=rr.reward,
        dollars=dollars, reward_record=rr,
    )


def test_incremental_writer_streams_each_row_to_disk(tmp_path: Path) -> None:
    from experiments.learning_allocator.runner import _OUTCOME_COLS

    path = tmp_path / "results.tsv"
    writer = live_runner._IncrementalResultsWriter(path, banner="# LIVE", echo=False)
    t = _task(seed=42)

    writer("frozen", t, _outcome(t, "frozen", 0.0, 0.40))
    # The row is already durable on disk after the first run — no waiting for
    # the matrix to finish, and a crash mid-run keeps what ran.
    lines = path.read_text().splitlines()
    assert lines[0] == "# LIVE"
    assert lines[1].split("\t") == _OUTCOME_COLS
    assert len(lines) == 3  # banner + header + 1 row

    writer("S0_only", t, _outcome(t, "S0_retrieval", 0.9, 0.42))
    lines = path.read_text().splitlines()
    assert len(lines) == 4  # header written once; second row appended
    row1, row2 = lines[2].split("\t"), lines[3].split("\t")
    assert row1[0] == "frozen" and row1[6] == "frozen"
    assert row2[0] == "S0_only" and row2[6] == "S0_retrieval"
    # split/seed come straight off the task.
    assert row1[4] == t.split and row1[5] == str(t.seed)


def test_incremental_writer_truncates_stale_file(tmp_path: Path) -> None:
    # A re-run into the same --out dir must not append onto a prior matrix's
    # rows; the first emission truncates.
    path = tmp_path / "results.tsv"
    path.write_text("STALE-BANNER\nSTALE-ROW\n")
    writer = live_runner._IncrementalResultsWriter(path, banner="# LIVE", echo=False)
    t = _task(seed=7)

    writer("frozen", t, _outcome(t, "frozen", 0.1, 0.4))

    text = path.read_text()
    assert "STALE" not in text
    assert text.splitlines()[0] == "# LIVE"


def test_incremental_writer_handles_skipped_outcome(tmp_path: Path) -> None:
    from mantis_agent.learning.orchestrator import TaskOutcome

    path = tmp_path / "results.tsv"
    writer = live_runner._IncrementalResultsWriter(path, banner="# LIVE", echo=False)
    t = _task(seed=1)
    skipped = TaskOutcome(
        task_name=t.name, task_id=t.task_id, cluster=t.cluster,
        skipped=True, note="budget exhausted",
    )

    writer("frozen", t, skipped)  # must not raise on the empty reward_record

    row = path.read_text().splitlines()[2].split("\t")
    assert row[-1] == "budget exhausted"
    assert row[-2] == "True"  # skipped column


# ── --exemplars loader (S1 backing) ──────────────────────────────────────


def test_load_exemplars_empty_path_returns_empty() -> None:
    # No --exemplars ⇒ S1 is a no-op (attributable in the logs), not an error.
    assert live_runner._load_exemplars("") == []


def test_load_exemplars_reads_json_list(tmp_path: Path) -> None:
    p = tmp_path / "ex.json"
    p.write_text(json.dumps([{"type": "submit", "intent": "reveal phone"}]))
    assert live_runner._load_exemplars(str(p)) == [
        {"type": "submit", "intent": "reveal phone"},
    ]


def test_load_exemplars_rejects_non_list(tmp_path: Path) -> None:
    # A malformed file must fail loudly — a silent [] makes S1 look like frozen
    # and voids the comparison.
    p = tmp_path / "ex.json"
    p.write_text(json.dumps({"type": "submit"}))
    with pytest.raises(ValueError, match="expected a JSON list"):
        live_runner._load_exemplars(str(p))


def test_committed_bt03_exemplar_matches_reveal_not_lead_submit() -> None:
    """The shipped BT03 exemplar must out-match the *reveal* submit step over
    the *lead* submit step, or S1 stamps the wrong sub-goal. Guards the file
    against an edit that drifts its intent tokens (matching is type + overlap).
    """
    from mantis_agent.gym.exemplar_memory import _tokens

    path = (
        Path(live_runner.__file__).resolve().parent
        / "eval" / "bt03_byowner_exemplar.json"
    )
    exemplars = live_runner._load_exemplars(str(path))
    assert len(exemplars) == 1
    ex = exemplars[0]
    assert ex["type"] == "submit"  # the decomposed reveal step is a `submit`
    ex_tok = _tokens(ex["intent"])
    reveal_tok = _tokens(
        "Click the Show Phone Number button in the Contact Private Seller "
        "area to reveal the seller phone"
    )
    lead_tok = _tokens(
        "Click the Contact Seller submit button on the Contact Private "
        "Seller form"
    )
    assert len(ex_tok & reveal_tok) > len(ex_tok & lead_tok), (
        "exemplar must overlap the reveal step more than the lead-submit step"
    )


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
