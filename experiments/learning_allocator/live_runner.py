"""Live Phase-2 run_fn — the spend boundary the offline runner deferred.

:mod:`experiments.learning_allocator.runner` ships the *no-spend* half of the
Phase-2 comparison: it drives every policy through the eval but scores runs
from outcomes baked into a ``run_result`` dict. Its docstring defers the live
adapter on purpose — "that adapter lands when we cross the spend line". This
module is that adapter.

:class:`LiveRunFn` is a ``run_fn`` (the orchestrator's execute seam,
``run_fn(task, plan, substrate_result) -> dict``) that submits each eval task's
plan to the **Modal CUA server** running against the **Daytona boattrader sim
env** — no proxies (the sandbox URL is reached directly). For each call it:

* loads the plan once, memoised across every task and policy — a ``.json``
  plan is a pre-decomposed micro-plan loaded directly (the deterministic,
  guard-carrying path); any other extension is decomposed via the LLM,
* builds a micro-suite with a fresh ``workflow_id`` and maps the allocator's
  chosen substrate onto the remote hint-store backend flags (``frozen`` →
  frozen / ``NullHintStore``; ``S0_retrieval`` → a Modal-Dict hint store shared
  across runs so retrieval actually accrues),
* submits, polls to terminal, reads the LLM-verifier verdict off the result
  envelope and the dollar cost off the run's Modal volume, and
* returns the ``run_result`` dict the reward channels read
  (``dynamic_verification_summary.verdict`` + ``costs.total``).

The **oracle** is *not* read here: ``main`` leaves ``reward_fn`` at its default
(:func:`~mantis_agent.learning.reward.reward_from_run`), so each finished run is
graded live against the sim env's ``/__env__/oracle``. That is the ground-truth
channel; this adapter only carries the proxy + cost signals the run produced.

Three I/O seams are injected so the adapter is unit-testable with **no spend and
no network**:

* ``post_fn(path, body) -> (status_code, dict)`` — the Modal HTTP call (submit +
  poll). Default posts to the live CUA server.
* ``pull_cost_fn(profile_id, workflow_id) -> (cost_usd, halt_reason)`` — reads
  ``claude_cost_by_path.json`` off the Modal volume. Default shells out to
  ``modal volume get``.
* ``decompose_fn(plan_text) -> MicroPlan`` — the local plan decompose. Default
  builds a :class:`PlanDecomposer` (needs ``ANTHROPIC_API_KEY``).

THIS MODULE SPENDS. Importing it is free; calling :class:`LiveRunFn` or ``main``
submits real Modal GPU runs against the sim env. ``main`` refuses to start
without ``LA_ENV_URL`` / ``LA_ENV_ADMIN_TOKEN`` pointing at a live sim env.

    uv run python -m experiments.learning_allocator.live_runner --out /tmp/la_live
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from mantis_agent.gym.grading import grade_run
from mantis_agent.learning.eval import EvalTask, load_manifest
from mantis_agent.learning.orchestrator import TaskOutcome
from mantis_agent.learning.reward import (
    DEFAULT_LAMBDA,
    RewardRecord,
    compute_reward,
    cost_channel,
    proxy_channel,
)
from mantis_agent.learning.substrates.base import SubstrateResult
from mantis_agent.plan_decomposer import MicroPlan, PlanDecomposer
from mantis_agent.server_utils import (
    build_micro_suite,
    merge_runtime,
    micro_plan_steps_to_dicts,
)
from mantis_agent.sim_envs.templating import substitute_env_url

from experiments.learning_allocator.runner import (
    FROZEN,
    S0,
    S1,
    _OUTCOME_COLS,
    _outcome_row,
    build_table1,
    format_table1,
    run_experiment,
    write_results,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ENDPOINT = "https://getmason--mantis-cua-server-api.modal.run"

# Run states the poll loop stops on (mirrors submit_one_trial's contract).
_TERMINAL = frozenset({"succeeded", "failed", "cancelled", "halted"})

# The injected I/O seams.
PostFn = Callable[[str, dict[str, Any]], tuple[int, dict[str, Any]]]
PullCostFn = Callable[[str, str], tuple[float, str]]
DecomposeFn = Callable[[str], MicroPlan]
ResetFn = Callable[[str, str], None]


# ── env + default I/O seams (the live, spending implementations) ────────────


def _read_env(key: str) -> str:
    """Read ``key`` from the process env, falling back to the repo ``.env``."""
    val = os.environ.get(key, "")
    if val:
        return val
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1].strip()
    return ""


def _load_exemplars(path: str) -> list[dict[str, Any]]:
    """Load the S1 worked-step exemplars from a JSON list file.

    The file is a hand-authored stand-in for the positive-labelled steps a
    distillation run's ``ExemplarSubstrate`` would emit — a JSON array of
    ``{type, intent, last_action, observed_outcome, source_run}`` dicts. Empty
    path ⇒ no exemplars (S1 degrades to a no-op, attributable in the logs).
    Raises on a malformed file: a silent empty list would make S1 look like
    frozen and quietly void the comparison.
    """
    if not path:
        return []
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list) or not all(isinstance(x, dict) for x in data):
        raise ValueError(
            f"--exemplars {path}: expected a JSON list of objects, "
            f"got {type(data).__name__}"
        )
    return data


def _default_post(path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """POST to the live Modal CUA server with the tenant token header."""
    r = requests.post(
        f"{ENDPOINT}{path}",
        headers={
            "X-Mantis-Token": _read_env("MANTIS_API_TOKEN"),
            "Content-Type": "application/json",
        },
        data=json.dumps(body),
        timeout=120,
    )
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, {"raw": r.text}


def _daytona_proxy_headers() -> dict[str, str]:
    """Proxy headers every request to the Daytona-served sim env needs.

    The sandbox is fronted by ``*.daytonaproxy01.net``, whose preview proxy
    (a) shows a "Preview - Warning" interstitial to real-browser User-Agents
    unless ``X-Daytona-Skip-Preview-Warning`` is set, and (b) 307s to an Auth0
    wall unless the per-sandbox ``x-daytona-preview-token`` is presented. BOTH
    are required on EVERY request — browser navigations *and* the admin
    reset/oracle calls alike. The token rotates per sandbox and is read from
    the env (never hard-coded — this repo is public). Empty token → skip the
    token header so a direct (non-Daytona) env still works.
    """
    hdr = {"X-Daytona-Skip-Preview-Warning": "true"}
    token = _read_env("LA_ENV_PREVIEW_TOKEN")
    if token:
        hdr["x-daytona-preview-token"] = token
    return hdr


def _default_browser_headers() -> dict[str, str]:
    """Headers applied to every *browser* request on the remote run.

    Two concerns, both sim-env-specific:

    * the Daytona proxy headers (see :func:`_daytona_proxy_headers`) — without
      the preview token the CUA browser 307s to Auth0 and never reaches the
      boats page;
    * a pre-seeded ``bt_cookie_consent`` cookie. The sim env renders a OneTrust
      cookie-consent banner on first visit (fresh Modal profile → no consent
      cookie → banner every run). The listings pre-scan
      (``ClaudeExtractor.find_all_listings``) reads that full-page overlay as a
      consent/sign-in wall and halts ``page_blocked`` *before* the brain can
      dismiss it. The server gates the banner on the cookie's mere presence, so
      seeding it suppresses the banner; ``decline`` is the privacy-preserving
      choice. Eval-only noise removal — the banner would block FROZEN and S0
      equally, so it adds nothing but variance to the substrate comparison.
    """
    return {**_daytona_proxy_headers(), "Cookie": "bt_cookie_consent=decline"}


def _default_reset(env_url: str, admin_token: str) -> None:
    """Reset the sim env to a clean, deterministically-reseeded state.

    The BT02/BT03 oracles grade the *cumulative* leads in the store and are
    precision-sensitive (one wrong lead fails the run), so without a reset
    between runs the first run's leads contaminate every later run's grade —
    in both directions. ``/__env__/reset`` clears the leads and rebuilds the
    catalog from the fixed ``SEED`` (same boats, so S0's cross-run hints stay
    valid). Best-effort: a reset failure must not crash the run loop, so it is
    swallowed the way ``grade_run`` swallows oracle errors.

    ``/__env__/reset`` is an admin route *behind* the Daytona preview proxy, so
    it needs the proxy headers in addition to ``X-Env-Admin`` — without them the
    POST 307s to the proxy auth wall and the reset silently no-ops.
    """
    try:
        requests.post(
            f"{env_url.rstrip('/')}/__env__/reset",
            headers={"X-Env-Admin": admin_token, **_daytona_proxy_headers()},
            timeout=30,
        )
    except requests.RequestException:
        pass


def _default_pull_cost(profile_id: str, workflow_id: str) -> tuple[float, str]:
    """Pull ``claude_cost_by_path.json`` off the Modal volume → (cost, halt)."""
    dest = Path(tempfile.mkdtemp(prefix="la-cost-")) / "claude_cost_by_path.json"
    try:
        subprocess.run(
            [
                "uv", "run", "modal", "volume", "get", "osworld-data",
                f"/runs/{profile_id}/{workflow_id}/claude_cost_by_path.json",
                str(dest), "--force",
            ],
            check=False, capture_output=True, timeout=120, cwd=REPO_ROOT,
        )
    except subprocess.TimeoutExpired:
        return 0.0, ""
    if not dest.exists():
        return 0.0, ""
    try:
        d = json.loads(dest.read_text())
    except (OSError, ValueError):
        return 0.0, ""
    cost = float((d.get("totals") or {}).get("cost_usd") or 0.0)
    halt = str((d.get("outcome") or {}).get("halt_reason") or "")
    return cost, halt


def _default_decompose(plan_text: str) -> MicroPlan:
    """Decompose the plan with a live :class:`PlanDecomposer` (cached to disk)."""
    api_key = _read_env("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set — cannot decompose the plan")
    cache_dir = REPO_ROOT / "data" / "plan_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    decomposer = PlanDecomposer(api_key=api_key, model="claude-opus-4-7")
    return decomposer.decompose_text(
        plan_text, cache_path_template=str(cache_dir / "decomposed_{hash}.json"),
    )


# ── the live run_fn ─────────────────────────────────────────────────────────


@dataclass
class LiveRunFn:
    """A ``run_fn`` that submits eval-task plans to the live Modal CUA server.

    The only state is the memoised decomposed plan (one decompose shared across
    every task and policy) and a submission counter that mints unique
    ``workflow_id``s. Call it as ``run_fn(task, plan, substrate_result)``.
    """

    plan_path: Path
    profile_id: str = "la-bt01"
    hint_dict_name: str = "la-bt01-hints"
    # S1 backing: positive-labelled exemplar steps (ExemplarSubstrate.apply's
    # delta_artifacts["exemplars"]) pre-extracted for this plan. The S1 branch
    # of _apply_substrate ships these to the remote run, where
    # modal_cua_server stamps them onto the micro plan via apply_exemplar_overlay.
    exemplars: list[dict[str, Any]] = field(default_factory=list)
    cua_model: str = "holo3"
    max_steps: int = 200
    max_cost: float = 1.0
    max_time_minutes: int = 30
    extractor_model: str = "claude-haiku-4-5-20251001"
    env_url: str = ""
    admin_token: str = ""
    # Request headers applied to every browser request on the remote run —
    # Daytona proxy headers (skip-warning + preview token) plus a pre-seeded
    # cookie-consent choice that suppresses the sim env's OneTrust banner.
    # See :func:`_default_browser_headers`. Set to ``{}`` for a direct
    # (non-Daytona) env with no consent banner.
    browser_extra_headers: dict[str, str] = field(
        default_factory=_default_browser_headers
    )
    zip_code: str = "33131"
    search_radius: str = "50"
    poll_interval_s: float = 15.0
    post_fn: PostFn = _default_post
    pull_cost_fn: PullCostFn = _default_pull_cost
    decompose_fn: DecomposeFn = _default_decompose
    reset_fn: ResetFn = _default_reset
    _micro_plan: MicroPlan | None = field(default=None, init=False, repr=False)
    _submits: int = field(default=0, init=False, repr=False)

    # ── orchestrator seam ──────────────────────────────────────────────

    def __call__(
        self, task: EvalTask, plan: Any, substrate_result: SubstrateResult,
    ) -> dict[str, Any]:
        """Submit one (task, substrate) run; return its ``run_result`` dict.

        ``plan`` (the orchestrator's in-process overlay target) is unused: the
        remote run decomposes its own plan text, so the in-process S0 overlay is
        a no-op here. The substrate's remote effect is the hint-store flag set
        in :meth:`_apply_substrate`.
        """
        del plan  # remote submit decomposes its own plan_text
        substrate = substrate_result.substrate
        # Isolate this run's grade: clear the prior run's leads so the
        # cumulative, precision-sensitive oracle scores only what THIS run
        # submits. No-op when admin_token is unset (keeps offline tests inert).
        if self.env_url and self.admin_token:
            self.reset_fn(self.env_url, self.admin_token)
        runtime = self._runtime()
        self._submits += 1
        workflow_id = f"la-{substrate}-s{task.seed}-{self._submits}-{int(time.time())}"
        suite = self._build_suite(substrate, workflow_id, runtime)

        body = {
            "task_suite": suite,
            "profile_id": suite["_profile_id"],
            "workflow_id": suite["_workflow_id"],
            "cua_model": self.cua_model,
            "max_steps": self.max_steps,
            "detached": True,
            **runtime,
        }
        status, resp = self.post_fn("/v1/predict", body)
        if status != 200:
            return self._failed_result(f"submit HTTP {status}: {resp}", workflow_id)

        run_id = str(resp.get("run_id") or "")
        terminal, halt_reason = self._poll(run_id)
        verdict = self._verdict(run_id, terminal)
        cost_usd, cost_halt = self.pull_cost_fn(self.profile_id, workflow_id)

        return {
            "dynamic_verification_summary": {"verdict": verdict},
            "costs": {"total": cost_usd},
            # Trace-only extras — ignored by the reward channels.
            "_run_id": run_id,
            "_terminal_status": terminal,
            "_halt_reason": halt_reason or cost_halt,
            "_substrate": substrate,
            "_workflow_id": workflow_id,
        }

    # ── suite construction ─────────────────────────────────────────────

    def _runtime(self) -> dict[str, Any]:
        """Runtime knobs for the submit — NO proxies (sim env reached directly).

        Inverts the real-site submit (``proxy_disabled=False`` + Oxylabs Miami):
        routing a sandbox URL through a residential proxy would send the run away
        from the env. ``merge_runtime`` drops the absent proxy keys entirely.
        """
        return merge_runtime({
            "proxy_disabled": True,
            "max_cost": float(self.max_cost),
            "max_time_minutes": int(self.max_time_minutes),
        })

    def _build_suite(
        self, substrate: str, workflow_id: str, runtime: dict[str, Any],
    ) -> dict[str, Any]:
        micro_plan = self._ensure_plan()
        suite = build_micro_suite(
            micro_plan_steps_to_dicts(micro_plan.steps),
            micro_plan.domain or "boattrader_scrape",
            profile_id=self.profile_id,
            workflow_id=workflow_id,
            plan_hash=micro_plan.plan_hash,
            plan_evolution_scope_id=self.profile_id,
            extractor_model=self.extractor_model,
            **runtime,
        )
        if self.browser_extra_headers:
            # Read by modal_cua_server's setup_env call sites → xdotool_env's
            # persistent header session (applied to every browser request).
            suite["_browser_extra_headers"] = dict(self.browser_extra_headers)
        self._apply_substrate(suite, substrate)
        return suite

    def _apply_substrate(self, suite: dict[str, Any], substrate: str) -> None:
        """Map the allocator's chosen rung onto the remote hint-store backend.

        The ladder's remote effect is which hint store the Modal run binds (see
        ``build_hint_store``): ``frozen`` freezes it (``NullHintStore``, no
        cross-run learning); ``S0_retrieval`` points it at a shared Modal Dict so
        hints accrue across runs. ``S1_exemplar`` ships pre-extracted worked
        steps that the remote stamps onto the plan as ``exemplar_replay`` hints,
        with the S0 anchor store frozen so the two rungs stay isolated.
        """
        if substrate == FROZEN:
            suite["_hint_store_disabled"] = True
        elif substrate == S0:
            suite["_hint_store_dict_name"] = self.hint_dict_name
        elif substrate == S1:
            # Freeze the S0 anchor mechanism so S1's lift is attributable to
            # the exemplar replay alone, then ship the worked steps.
            suite["_hint_store_disabled"] = True
            if self.exemplars:
                suite["_exemplars"] = list(self.exemplars)

    def _ensure_plan(self) -> MicroPlan:
        if self._micro_plan is None:
            if self.plan_path.suffix.lower() == ".json":
                self._micro_plan = self._load_json_plan()
            else:
                self._micro_plan = self.decompose_fn(self._plan_text())
        return self._micro_plan

    def _load_json_plan(self) -> MicroPlan:
        """Load a pre-decomposed micro-plan JSON directly — no decompose call.

        Mirrors ``cli.py``'s plan-format dispatch (``_looks_like_text_plan``): a
        ``.json`` plan is an already-decomposed micro-plan, so the LLM
        decomposer is bypassed entirely (a deterministic, hand-authored plan
        must not be re-paraphrased — e.g. the conditional ``detect_visible``
        guard the decomposer flattens). ``{{ENV_URL}}`` is substituted with the
        live sim-env URL via the repo's standard JSON-plan templating so the nav
        steps reach the sandbox directly (the text path uses its own
        ``{env_url}`` token). ``decompose_fn`` is unused on this branch.
        """
        payload = json.loads(self.plan_path.read_text())
        payload = substitute_env_url(payload, self.env_url)
        return MicroPlan.from_dict(payload)

    def _plan_text(self) -> str:
        """Plan text with placeholders filled. ``{env_url}`` points the nav at the
        live sim env so the run reaches the sandbox directly (no proxy)."""
        raw = self.plan_path.read_text()
        return (
            raw.replace("{env_url}", self.env_url)
            .replace("{zip_code}", self.zip_code)
            .replace("{search_radius}", self.search_radius)
        )

    # ── poll + verdict ─────────────────────────────────────────────────

    def _poll(self, run_id: str) -> tuple[str, str]:
        """Poll ``action=status`` until terminal; return (status, halt_reason)."""
        if not run_id:
            return "failed", "no_run_id"
        deadline = time.time() + 60 * self.max_time_minutes + 120
        while time.time() < deadline:
            _, resp = self.post_fn(
                "/v1/predict", {"action": "status", "run_id": run_id},
            )
            st = str(resp.get("status") or "")
            if st in _TERMINAL:
                return st, str(resp.get("halt_reason") or "")
            time.sleep(self.poll_interval_s)
        return "halted", "poll_timeout"

    def _verdict(self, run_id: str, terminal: str) -> str:
        """Proxy verdict: the LLM-verifier's call off the result envelope.

        ``action=status`` doesn't carry the verifier summary; ``action=result``
        does, but only once the run *succeeded*. On a non-success terminal the
        envelope is unavailable, so the terminal status stands in: a run that
        didn't succeed didn't pass.
        """
        if terminal == "succeeded" and run_id:
            _, resp = self.post_fn(
                "/v1/predict", {"action": "result", "run_id": run_id},
            )
            result = resp.get("result")
            if isinstance(result, dict):
                summary = result.get("dynamic_verification_summary") or {}
                verdict = summary.get("verdict") if isinstance(summary, dict) else None
                if verdict:
                    return str(verdict).strip().lower()
            return "pass"
        return "fail"

    def _failed_result(self, note: str, workflow_id: str) -> dict[str, Any]:
        return {
            "dynamic_verification_summary": {"verdict": "fail"},
            "costs": {"total": 0.0},
            "_terminal_status": "failed",
            "_halt_reason": note,
            "_workflow_id": workflow_id,
        }


# ── live driver ─────────────────────────────────────────────────────────────

_LIVE_BANNER = (
    "# SOURCE=live-modal-daytona — REAL agent runs against the boattrader sim "
    "env. Spend incurred. Oracle-graded; proxy + cost are from the agent run."
)


def tasks_for_plan(plan_name: str) -> list[EvalTask]:
    """Runnable eval tasks wired to ``plan_name``. Matching on the *stem* keeps
    ``plans/foo``, a bare ``foo``, and ``foo.json`` equal — clusters.json names
    the logical plan (``bt02_spec_lookup``) while the on-disk file may carry a
    ``.json`` suffix (a pre-decomposed micro-plan)."""
    stem = Path(plan_name).stem
    return [
        t for t in load_manifest().runnable()
        if t.plan and Path(t.plan).stem == stem
    ]


class _IncrementalResultsWriter:
    """Append one ``results.tsv`` row per completed run, as it lands.

    A live matrix is slow — each run is a multi-minute CUA job — and the
    batch :func:`write_results` only writes once everything finishes, so a
    watcher sees an empty file for the whole run and a crash/time-cap loses
    every row. This streams instead: the first call truncates and writes the
    banner + header, every call appends a row and ``flush``es. The final
    :func:`write_results` overwrites ``results.tsv`` with the identical
    complete set (and adds table1/fig1, which genuinely need every row), so
    streamed and final never diverge — same row formatter, same column order.
    """

    def __init__(self, path: Path, *, banner: str, echo: bool = True) -> None:
        self.path = Path(path)
        self.banner = banner
        self.echo = echo
        self._started = False

    def __call__(self, policy: str, task: EvalTask, outcome: TaskOutcome) -> None:
        # split/seed off the live task; equal to ExperimentResult's maps the
        # batch write sources from (runner stamps them from the same task).
        row = _outcome_row(policy, outcome, task.split, task.seed)
        mode = "a" if self._started else "w"
        with self.path.open(mode, newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            if not self._started:
                fh.write(self.banner + "\n")
                w.writerow(_OUTCOME_COLS)
                self._started = True
            w.writerow(row)
            fh.flush()
        if self.echo:
            print(_progress_line(policy, task, outcome), flush=True)


def _progress_line(policy: str, task: EvalTask, o: TaskOutcome) -> str:
    """A one-line, crash-safe console echo of a completed run."""
    if o.skipped:
        return f"[live] {policy:<10} {task.name:<28} SKIP ({o.note})"
    rr = o.reward_record
    score = f"{rr.oracle_score:.2f}" if rr else "  - "
    passed = rr.oracle_passed if rr else "-"
    reward = o.reward if o.reward is not None else 0.0
    return (
        f"[live] {policy:<10} {task.name:<28} sub={o.substrate or '-':<14} "
        f"oracle={score} pass={passed!s:<5} ${o.dollars or 0:.2f} "
        f"reward={reward:+.3f}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Learning Allocator Phase-2 LIVE runner (REAL SPEND).",
    )
    parser.add_argument("--out", default="", help="dir to write LIVE results TSVs")
    parser.add_argument("--budget", type=float, default=5.0)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-cost", type=float, default=1.0)
    parser.add_argument("--max-time-minutes", type=int, default=30)
    parser.add_argument(
        "--policies", default="frozen,S0_only",
        help="comma-separated policies to compare",
    )
    parser.add_argument(
        "--plan", default="plans/boattrader_scrape",
        help="plan file to submit (relative to the repo root)",
    )
    parser.add_argument(
        "--hint-dict-suffix", default="",
        help=(
            "appended to the S0 hint-store Modal Dict name "
            "(``{slug}-hints{suffix}``). Use a fresh suffix (e.g. ``-v4``) to "
            "isolate a run from anchors a prior run accumulated in the shared "
            "dict; empty (default) reuses the canonical ``{slug}-hints`` dict."
        ),
    )
    parser.add_argument(
        "--exemplars", default="",
        help=(
            "path to a JSON list of worked-step exemplar dicts (S1 backing). "
            "Each item carries ``type`` + ``intent`` (matched to a plan step) "
            "and a coordinate-free ``last_action``/``observed_outcome`` the "
            "remote stamps as an ``exemplar_replay`` hint. Read only by the "
            "S1 branch; frozen/S0 ignore it. Empty (default) ⇒ S1 is a no-op."
        ),
    )
    args = parser.parse_args(argv)

    env_url = _read_env("LA_ENV_URL")
    admin_token = _read_env("LA_ENV_ADMIN_TOKEN")
    if not env_url or not admin_token:
        print(
            "REFUSING TO RUN: LA_ENV_URL and LA_ENV_ADMIN_TOKEN must point at a "
            "live Daytona boattrader sim env (the /__env__/oracle endpoint). "
            "Boot the env and export both before a live run.",
            file=sys.stderr,
        )
        return 2

    # Stem (drop any ``.json`` suffix) so the profile slug and task wiring stay
    # clean for a pre-decomposed plan: ``plans/bt02_spec_lookup.json`` →
    # ``bt02_spec_lookup`` → slug ``la-bt02-spec-lookup`` and a clusters.json
    # match. ``plan_path`` below keeps the full suffixed path for reading.
    plan_name = Path(args.plan).stem
    tasks = tasks_for_plan(plan_name)
    if not tasks:
        print(
            f"No runnable task is wired to plan {plan_name!r} — "
            "set the cluster's plan field in clusters.json.",
            file=sys.stderr,
        )
        return 2

    policies = tuple(p.strip() for p in args.policies.split(",") if p.strip())
    slug = f"la-{plan_name.replace('_', '-')}"
    hint_dict_name = f"{slug}-hints{args.hint_dict_suffix}"
    exemplars = _load_exemplars(args.exemplars)
    live = LiveRunFn(
        plan_path=REPO_ROOT / args.plan,
        env_url=env_url,
        admin_token=admin_token,
        profile_id=slug,
        hint_dict_name=hint_dict_name,
        exemplars=exemplars,
        max_cost=args.max_cost,
        max_time_minutes=args.max_time_minutes,
    )

    print(
        "Learning Allocator — Phase-2 LIVE RUN (REAL SPEND).\n"
        f"  env_url={env_url}\n"
        f"  policies={policies} budget=${args.budget:.2f} tasks={len(tasks)}\n"
        f"  plan={args.plan} max_cost=${args.max_cost:.2f}/run\n"
        f"  rounds={args.rounds} S0_hint_dict={hint_dict_name} "
        f"S1_exemplars={len(exemplars)}\n",
    )
    # Proxy-aware oracle grading: the default reward_from_run hits
    # ``/__env__/oracle`` with only ``X-Env-Admin``, which the Daytona preview
    # proxy 307s to its auth wall (→ HTML, not JSON → every run graded
    # oracle-error). This closure is reward_from_run with the proxy headers
    # threaded onto the oracle GET; the arithmetic is identical.
    oracle_hdr = _daytona_proxy_headers()

    def _proxied_reward_fn(
        *, env_url: str, admin_token: str, task_id: str,
        run_result: dict[str, Any], lam: float = DEFAULT_LAMBDA,
    ) -> RewardRecord:
        graded = grade_run(env_url, admin_token, task_id, extra_headers=oracle_hdr)
        return compute_reward(
            task_id=task_id,
            oracle_score=graded.score,
            oracle_passed=graded.passed,
            proxy_verdict=proxy_channel(run_result),
            dollars=cost_channel(run_result),
            lam=lam,
            oracle_error=graded.error,
            extras={"oracle_reasons": graded.reasons, "oracle_diff": graded.diff},
        )

    # Stream each run's row to results.tsv as it lands — live visibility into
    # a slow matrix and durability if it crashes or hits the time cap. The
    # final write_results below overwrites with the identical complete set.
    streamer: _IncrementalResultsWriter | None = None
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        streamer = _IncrementalResultsWriter(
            out_dir / "results.tsv", banner=_LIVE_BANNER,
        )

    result = run_experiment(
        tasks=tasks,
        run_fn=live,
        reward_fn=_proxied_reward_fn,  # live oracle, proxy-header aware
        env_url=env_url,
        admin_token=admin_token,
        budget=args.budget,
        rounds=args.rounds,
        epsilon=args.epsilon,
        seed=args.seed,
        policies=policies,
        on_outcome=streamer,
    )
    print(format_table1(build_table1(result)))
    if args.out:
        paths = write_results(result, args.out, banner=_LIVE_BANNER)
        print("\nwrote (LIVE):")
        for label, p in paths.items():
            print(f"  {label:<8} {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
