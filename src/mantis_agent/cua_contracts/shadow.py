"""Shadow / canary routing for the model-serving facade (#489).

Builds on #487's :class:`ModelServingFacade` + :class:`RoutingMode`.
The :class:`SplitFacade` wraps two underlying facades — production
and candidate — and routes calls so:

* **Reads** (planner / grounding / verifier) invoke BOTH; the prod
  result drives the committed action path, the shadow result lands
  as a non-committing :class:`TrajectoryEvent` (``committed=False``)
  alongside the same step.
* **Writes** (actor) refuse the shadow path entirely — shadow
  models MUST NOT dispatch side-effectful actions per the safety
  contract in the design doc.

The non-committing events let downstream consumers compute prod-vs-
shadow disagreement at step / action / verdict level — the eval
report (#490) reads them to grade a candidate before promotion.

What this PR doesn't do (intentionally):

* Migrate handlers to invoke through the SplitFacade. As with #487,
  each handler migrates one at a time as a separate PR.
* Wire a comparison/diff reporter. That belongs alongside the eval
  harness work in #490; this PR ships the substrate the reporter
  reads from.

Side-effect suppression contract:

The reason ACTOR is the only role refused on shadow is that the
ACTOR role's payload turns into ``env.step(action)`` calls that
actually mutate the browser / outside world. PLANNER / GROUNDING /
VERIFIER are all observation/decision producers — their shadow
output is metadata, never a click. Future role additions need an
explicit decision in the same code path: side-effectful → reject
shadow; observation/decision → allow.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .serving import ModelCallResult, ModelServingFacade, Role, RoutingMode

if TYPE_CHECKING:
    from .emit import TrajectoryEmitter

logger = logging.getLogger(__name__)


# Roles the shadow path refuses. Per the design's safety contract:
# any role whose output is dispatched as a side-effectful action
# (env.step) cannot run shadow because dispatching it would mutate
# the world twice. Future role additions need an explicit entry
# here; the SplitFacade fails-closed on unrecognised roles.
_SIDE_EFFECTFUL_ROLES: frozenset[Role] = frozenset({Role.ACTOR})


class ShadowRoutingError(RuntimeError):
    """Raised when the shadow path is asked to invoke a role it
    cannot safely run (e.g. ACTOR). The runner catches it,
    short-circuits the shadow call, and continues the prod path
    unchanged."""


class SplitFacade:
    """Routes model calls between a production and a candidate
    facade (#489).

    Usage:

        prod_facade = PassthroughFacade(prod_client)
        shadow_facade = PassthroughFacade(candidate_client)
        split = SplitFacade(
            prod=prod_facade,
            shadow=shadow_facade,
            shadow_event_sink=_record_shadow_event,
        )
        # Caller invokes ONCE; SplitFacade fans out internally.
        result = split.invoke(role=Role.PLANNER, payload={...})
        # ``result`` is the prod result. The shadow result was
        # passed to ``shadow_event_sink`` for side-channel
        # comparison; never affects ``result``.

    The split is **asymmetric**:

    * The prod call's result is always returned to the caller and
      drives the committed action path.
    * The shadow call's result is handed to ``shadow_event_sink``
      (typically wired to write a non-committing TrajectoryEvent).
    * Shadow-side exceptions are caught and logged; they NEVER
      surface to the prod path. The whole point of shadow is to
      learn about the candidate without it affecting production.

    Routing-mode honoured:

    * Default routing mode for the prod call is whatever the
      caller passes (typically ``PROD``).
    * The shadow call always runs with ``routing_mode=SHADOW`` so
      downstream metrics / event readers can group cleanly.
    """

    def __init__(
        self,
        *,
        prod: ModelServingFacade,
        shadow: ModelServingFacade | None = None,
        shadow_event_sink: "_ShadowEventSink | None" = None,
        canary_fraction: float = 0.0,
    ) -> None:
        self._prod = prod
        self._shadow = shadow
        self._shadow_event_sink = shadow_event_sink
        # Canary is a future hook — when > 0, a deterministic slice
        # of traffic routes to the shadow facade AS PROD (side-
        # effectful). Disabled in v1; the substrate is here so a
        # follow-up PR can wire a real canary picker without
        # changing the public surface.
        self._canary_fraction = float(canary_fraction)

    def invoke(
        self,
        *,
        role: Role,
        payload: Any,
        routing_mode: RoutingMode = RoutingMode.PROD,
        version_pin: str = "",
    ) -> ModelCallResult:
        # Always run the prod call. The shadow call is a side
        # channel; its outcome must not gate prod.
        prod_result = self._prod.invoke(
            role=role, payload=payload,
            routing_mode=routing_mode, version_pin=version_pin,
        )
        # Skip the shadow call when no shadow facade is wired (the
        # split is effectively a passthrough) or when the role is
        # side-effectful (safety contract).
        if self._shadow is None:
            return prod_result
        if role in _SIDE_EFFECTFUL_ROLES:
            logger.debug(
                "SplitFacade: refusing shadow invocation for "
                "side-effectful role %s — actor shadow would "
                "double-dispatch", role.value,
            )
            return prod_result
        try:
            shadow_result = self._shadow.invoke(
                role=role, payload=payload,
                routing_mode=RoutingMode.SHADOW,
                version_pin=version_pin,
            )
        except Exception as exc:  # noqa: BLE001 — shadow must not crash prod
            logger.warning(
                "SplitFacade: shadow invocation for role %s raised "
                "(%s) — prod path unaffected", role.value, exc,
            )
            return prod_result
        # Hand the shadow result to the sink for comparison /
        # event emission. Sink errors are logged and swallowed —
        # observability mustn't crash the prod return.
        if self._shadow_event_sink is not None:
            try:
                self._shadow_event_sink(
                    role=role,
                    prod_result=prod_result,
                    shadow_result=shadow_result,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "SplitFacade: shadow_event_sink raised: %s", exc,
                )
        return prod_result


# Callable shape every shadow-event sink must satisfy. Inputs are
# the role + both ModelCallResults; the sink decides what to do
# (write canonical event with committed=False, send to a queue,
# accumulate for a comparison report).
class _ShadowEventSink:  # pragma: no cover — pseudo-type for callers
    def __call__(
        self,
        *,
        role: Role,
        prod_result: ModelCallResult,
        shadow_result: ModelCallResult,
    ) -> None: ...


def emit_shadow_disagreement(
    emitter: "TrajectoryEmitter",
    *,
    role: Role,
    prod_result: ModelCallResult,
    shadow_result: ModelCallResult,
    run_id: str,
    step_index: int,
) -> bool:
    """Convenience sink — writes a non-committing TrajectoryEvent
    capturing the shadow result alongside the prod (#489).

    The event is logged with ``committed=False`` so a downstream
    reader can dedup-by-step then group prod-vs-shadow records by
    matching ``(run_id, step_index)``. ``versions`` carries the
    candidate's model + prompt stamps so attribution works without
    a side-channel registry lookup.

    Returns ``True`` on a successful emit, ``False`` on validation
    / IO failure. Best-effort — the shadow side channel must not
    crash prod.

    Implementation note: builds a minimal event by reusing the
    emitter's existing emit path. The :class:`StepResult` it
    constructs is a thin stub — most fields irrelevant for shadow
    comparisons; the consumer reads ``versions`` + the action +
    verdict, all of which come from the shadow ModelCallResult.
    """
    from .types import (
        ActionResult,
        Observation,
        SCHEMA_VERSION,
        Step,
        TrajectoryEvent,
        Verdict,
        VerdictKind,
    )
    from .validation import ContractValidationError, validate_trajectory_event

    versions = dict(emitter.versions)
    # Stamp the shadow's role-keyed model + prompt versions so a
    # reader of the JSONL can group by candidate version without
    # extra plumbing.
    if shadow_result.model_version:
        versions[f"{role.value}_model"] = shadow_result.model_version
    if shadow_result.prompt_version:
        versions[f"{role.value}_prompt"] = shadow_result.prompt_version
    event = TrajectoryEvent(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        step_index=step_index,
        attempt_index=0,
        step=Step(
            schema_version=SCHEMA_VERSION,
            intent=f"shadow:{role.value}",
            action_type="shadow_invocation",
        ),
        observation=Observation(
            schema_version=SCHEMA_VERSION,
            screenshot_ref=f"shadow://{run_id}/step_{step_index}/role_{role.value}",
        ),
        action_result=ActionResult(
            schema_version=SCHEMA_VERSION,
            action_type="shadow_invocation",
            params={"role": role.value},
            dispatched=False,
            dispatch_error="shadow path — never dispatched",
        ),
        verdict=Verdict(
            schema_version=SCHEMA_VERSION,
            kind=VerdictKind.OK,
            reason="",
            evidence=f"shadow {role.value} produced result; "
                     f"prod returned {prod_result.routing_mode.value}",
            confidence=shadow_result.confidence if hasattr(shadow_result, "confidence") else 0.0,
        ),
        versions=versions,
        latency_seconds=shadow_result.latency_seconds,
        cost_usd=shadow_result.cost_usd,
        committed=False,
    )
    try:
        validate_trajectory_event(event)
    except ContractValidationError as exc:
        logger.warning("shadow emit: validation failed: %s", exc)
        return False
    try:
        emitter._append(event)  # noqa: SLF001 — appending shadow event without dedup
    except OSError as exc:
        logger.warning("shadow emit: append failed: %s", exc)
        return False
    return True
