"""Producer-side eval curation (#901).

The slow-loop trainer's champion/challenger gate needs a frozen holdout eval
set. Augur stores + freezes eval candidates, but the *content* must originate
here in Mantis: only the producer knows a run's canonical task definition
(instruction / env / success criteria) and which runs are eval-worthy.

This module owns the two pure pieces:

* :func:`compose_task_spec` / :func:`default_success_conditions` — build an
  ``augur_sdk.TaskSpec`` dict (with ``success_conditions``) from known fields.
* :func:`task_spec_from_runner` — compose a fallback TaskSpec for the
  general/micro/prod path, which (unlike the fan-out path) opens its
  DebugSession without one. Returns ``None`` for runs with no canonical task
  identity (health / trigger / ad-hoc), which must NOT become eval tasks.

The ``mark_for_eval`` *signal* (which runs to keep) is emitted by the runtime
via :meth:`observability.augur.AugurAdapter.mark_for_eval`.
"""

from __future__ import annotations

import os
from typing import Any


def emit_success_conditions() -> bool:
    """Whether to include ``success_conditions`` on the emitted TaskSpec.

    OFF by default: the deployed augur-sdk **0.6.1 server rejects the field**
    on TaskSpec — including it stalls run finalization (live-verified: a run
    whose task_spec carried ``success_conditions`` never left ``running``;
    the same task_spec without it finalized ``succeeded``). The consumer
    (mantis-trainer gate ``EvalTask.from_dict``) already defaults criteria to
    ``[{"type":"task_success"}]`` when absent, so omitting it loses nothing
    functional. Flip ``MANTIS_EVAL_SUCCESS_CONDITIONS=1`` once the deployed
    Augur schema supports it (the #901 parallel data-plane ask).
    """
    return os.environ.get("MANTIS_EVAL_SUCCESS_CONDITIONS", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def default_success_conditions() -> list[dict[str, Any]]:
    """The baseline criterion every oracle-graded task supports: the run
    succeeded. The gate's runner re-runs the task and the env oracle decides
    ``task_success``. Richer deterministic criteria (``url_contains`` /
    ``output_contains`` / ``status_eq``) are layered on when the task/oracle
    definition specifies them."""
    return [{"type": "task_success"}]


def compose_task_spec(
    *,
    task_spec_id: str,
    instruction: str = "",
    task_class: str = "",
    env_id: str = "",
    max_steps: int = 0,
    success_conditions: list[dict[str, Any]] | None = None,
    subgoals: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a TaskSpec dict, emitting only non-empty keys (the 0.6.0 schema
    treats every field as optional). ``success_conditions`` defaults to the
    baseline when not given."""
    spec: dict[str, Any] = {}
    if task_spec_id:
        spec["task_spec_id"] = task_spec_id
    if instruction:
        spec["instruction"] = instruction
    if task_class:
        spec["task_class"] = task_class
    if env_id:
        spec["env_id"] = env_id
    if max_steps and max_steps > 0:
        spec["max_steps"] = int(max_steps)
    # success_conditions only when explicitly provided AND emission is enabled —
    # the deployed augur 0.6.1 server rejects the field (see
    # ``emit_success_conditions``). Default: omit (consumer falls back to
    # task_success).
    if success_conditions and emit_success_conditions():
        spec["success_conditions"] = success_conditions
    if subgoals:
        spec["subgoals"] = subgoals
    return spec


def task_spec_from_runner(runner: Any, plan: Any) -> dict[str, Any] | None:
    """Compose a fallback TaskSpec for a non-fan-out run from runner + plan.

    Anchors the task identity on ``<domain>.<plan_name>.v1`` (or ``<plan_name>.v1``
    when no domain). Returns ``None`` when there's no plan name to anchor on —
    health / trigger / ad-hoc runs have no canonical task and should not be
    eligible for the holdout.
    """
    plan_name = str(getattr(runner, "plan_name", "") or "").strip()
    if not plan_name:
        return None
    domain = str(getattr(plan, "domain", "") or "").strip()
    task_spec_id = f"{domain}.{plan_name}.v1" if domain else f"{plan_name}.v1"

    steps = list(getattr(plan, "steps", []) or [])
    instruction = plan_name.replace("_", " ")
    if not instruction and steps:
        instruction = str(getattr(steps[0], "intent", "") or "")

    session_name = str(getattr(runner, "session_name", "") or "run")
    return compose_task_spec(
        task_spec_id=task_spec_id,
        instruction=instruction or task_spec_id,
        task_class=domain,
        env_id=f"mantis:{session_name}",
        max_steps=len(steps),
        success_conditions=default_success_conditions(),
    )
