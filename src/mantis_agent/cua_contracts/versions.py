"""Version metadata stamping for canonical trajectory events (#488).

Every committed :class:`~.types.TrajectoryEvent` carries a
``versions: dict[str, str]`` slot that this module owns. The
purpose is regression attribution — when a run starts behaving
differently after a deploy, an operator should be able to read the
event stream and answer "did the planner model change?" / "did the
grounding prompt change?" / "did the browser image change?" without
spelunking through deploy history.

The canonical key set lives here so writer (this module) and reader
(downstream consumers — model registry, shadow-router diff, eval
attribution) agree on what to look for.

Keys (additive — readers must tolerate unknown / missing entries
on v1 events, since rollout populates the dict incrementally):

* ``planner_model`` — model id used to decompose the plan
  (claude-opus-4-7, ...).
* ``planner_prompt`` — short hash / tag of the decomposer prompt
  template that produced the plan.
* ``grounding_model`` — model id used by the form-target /
  click-target grounding path (claude-haiku-4-5-..., holo3-35b-a3b).
* ``grounding_prompt`` — short hash / tag of the grounding prompt
  template.
* ``actor_model`` — model id driving the Holo3 / Claude action
  loop when a brain is in the loop.
* ``actor_prompt`` — prompt hash for the actor's tool-call /
  scoped-task template.
* ``verifier_model`` — model id used by ``verify_gate`` /
  ``StepVerifier`` (typically Haiku since #421).
* ``verifier_prompt`` — verifier prompt hash.
* ``action_ontology`` — schema_version of :class:`ActionTyped`
  (always populated; pinned per release).
* ``contracts_schema`` — schema_version of the cua_contracts
  package (always populated; pinned per release).
* ``browser_image`` — Modal / docker image tag of the executor
  container.
* ``sandbox_runtime`` — runtime identifier (e.g. ``modal/holo3``,
  ``baseten/claude``).

Population semantics:

* Static keys (``action_ontology``, ``contracts_schema``) populate
  at emitter construction time from module constants.
* Runtime keys (``browser_image``, ``sandbox_runtime``) populate
  from process env vars where the deploy script stashes them
  (Modal sets ``MODAL_IMAGE_BUILDER_VERSION``; Baseten sets
  ``BASETEN_DEPLOYMENT_ID``; etc).
* Model / prompt keys populate from the runner's
  ``runtime_versions`` dict — handlers / brains that know their
  model id stash it there. Missing keys land as empty string in
  v1; the validator allows that.
"""

from __future__ import annotations

import os
from typing import Any

from .ontology import ActionTyped
from .types import SCHEMA_VERSION


# Canonical ordered list — writers populate from this set, readers
# match against it. New keys added here only; the validator allows
# unknown keys for forward-compat but the dashboards / eval
# attribution will key off this list.
VERSION_KEYS: tuple[str, ...] = (
    # Model + prompt pairs per role.
    "planner_model", "planner_prompt",
    "grounding_model", "grounding_prompt",
    "actor_model", "actor_prompt",
    "verifier_model", "verifier_prompt",
    # Static contract / ontology versions.
    "action_ontology", "contracts_schema",
    # Runtime / deploy stamps.
    "browser_image", "sandbox_runtime",
)


# Environment-variable names the deploy scripts use to stash
# runtime stamps. Centralised here so a deploy-config change has
# one place to update.
_BROWSER_IMAGE_ENV: str = "MANTIS_BROWSER_IMAGE"
_SANDBOX_RUNTIME_ENV: str = "MANTIS_SANDBOX_RUNTIME"


def _action_ontology_version() -> str:
    """The action-ontology version tag, derived from the enum's
    membership snapshot. Bumping the enum is a contract change and
    surfaces here so downstream consumers see a different
    ``action_ontology`` value in events post-bump.

    Format: ``v1.<member_count>`` — short, grep-able, monotonically
    increases on additive changes. A non-additive change (rename /
    drop) bumps SCHEMA_VERSION too, which surfaces in
    ``contracts_schema`` independently.
    """
    return f"v{SCHEMA_VERSION}.{len(ActionTyped)}"


def _contracts_schema_version() -> str:
    """The cua_contracts package's pinned schema_version stamp.
    Single source of truth for "are reader and writer talking the
    same contract?"."""
    return f"v{SCHEMA_VERSION}"


def _runtime_env_stamp() -> dict[str, str]:
    """Pull deploy-time stamps off the process env. Missing env
    vars stay missing in the dict — the validator allows empty
    values in v1 so a partially-populated env doesn't break the
    write."""
    out: dict[str, str] = {}
    browser = os.environ.get(_BROWSER_IMAGE_ENV, "").strip()
    if browser:
        out["browser_image"] = browser
    sandbox = os.environ.get(_SANDBOX_RUNTIME_ENV, "").strip()
    if sandbox:
        out["sandbox_runtime"] = sandbox
    return out


def collect_versions(runner: Any | None = None) -> dict[str, str]:
    """Build the version dict for the emitter at construction time.

    Combines:

    * Static contract / ontology stamps (always populated).
    * Runtime / deploy env stamps (populated when env vars are set).
    * Per-runner model + prompt stamps from
      ``runner.runtime_versions`` — handlers / brains that know
      their model id stash it on the runner before the executor
      builds the emitter.

    Returns a *new* dict on every call so the emitter's stored
    snapshot doesn't share state with future runner mutations.
    The dict is intentionally permissive — populates whatever's
    available, leaves the rest absent. Validators on the reader
    side surface missing keys as warnings, not failures.
    """
    versions: dict[str, str] = {
        "action_ontology": _action_ontology_version(),
        "contracts_schema": _contracts_schema_version(),
    }
    versions.update(_runtime_env_stamp())
    if runner is not None:
        from_runner = getattr(runner, "runtime_versions", None)
        if isinstance(from_runner, dict):
            for k, v in from_runner.items():
                if isinstance(k, str) and isinstance(v, str) and v:
                    versions[k] = v
    return versions
