"""Role-versioned model-serving facade (#487).

Model calls are currently spread across the brain, grounding,
extraction, verifier, and runtime-setup paths. Each path constructs
its own model client, picks its own model id, and threads its own
prompt-version (or doesn't). The CUA design's reliability story
needs **one** invocation surface so:

* planner / grounding / actor / verifier decisions can be **routed**
  (prod vs shadow vs canary) independently;
* every call automatically attaches ``model_version`` +
  ``prompt_version`` to the runner's ``runtime_versions`` dict,
  where #488's emit hook picks them up;
* a future model-registry change (#490) can pin a role to a
  specific version without rewriting handlers;
* shadow / canary routing (#489) has a typed seam to hook into.

The facade is intentionally **narrow** for v1. It wraps existing
model clients rather than replacing them — so today's handlers
keep their concrete dependencies, and the migration to facade-
mediated calls happens incrementally.

Public surface:

* :class:`Role` — one of PLANNER / GROUNDING / ACTOR / VERIFIER.
* :class:`RoutingMode` — PROD / SHADOW / CANARY. The facade
  consults the mode when deciding *which* underlying client to
  invoke (production vs candidate).
* :class:`ModelCallResult` — what every facade call returns:
  the raw payload + the model + prompt versions that produced
  it + the routing mode that was active.
* :class:`ModelServingFacade` — Protocol. Substrate today;
  concrete adapters land in follow-up work as each handler migrates.
* :class:`PassthroughFacade` — default implementation that wraps a
  single underlying client and treats every role as a passthrough
  call to that client. Useful as a bring-up step before per-role
  routing lands.

What this PR doesn't do (intentionally):

* Migrate handlers to call through the facade. The form / click /
  brain / extractor handlers stay on their concrete clients —
  introducing the facade in one PR + migrating callers in
  follow-ups keeps each change reviewable.
* Wire shadow / canary execution. The :class:`RoutingMode` enum
  + the facade signature accept it; the actual shadow-dispatch
  logic lands in #489 once the facade has real users.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol, runtime_checkable


class Role(str, Enum):
    """Role of the model call (#487 / design doc §model-serving).

    String values are stable so they can be used as dict keys,
    log labels, and runtime_versions key prefixes (the facade
    stamps ``planner_model``, ``grounding_prompt``, etc. — keys
    derived from ``Role.value + suffix``).
    """

    PLANNER = "planner"
    GROUNDING = "grounding"
    ACTOR = "actor"
    VERIFIER = "verifier"


class RoutingMode(str, Enum):
    """How the facade should route a call (#487 / #489).

    * ``PROD`` — production model, committed action path. Default
      for every call until shadow / canary routing is wired.
    * ``SHADOW`` — candidate model invoked alongside prod for
      comparison. Result MUST NOT drive side-effectful actions
      (the canonical event for a shadow call lands with
      ``committed=False`` per #489's design).
    * ``CANARY`` — candidate model handles a small slice of prod
      traffic. Side-effectful — graded by eval / metric impact.
    """

    PROD = "prod"
    SHADOW = "shadow"
    CANARY = "canary"


@dataclass(frozen=True)
class ModelCallResult:
    """Typed return shape from every facade ``invoke`` call.

    The payload is opaque (each model produces its own native
    shape — Claude tool_use dict, Holo3 action, ClaudeExtractor
    schema record). The facade's job is to wrap the payload with
    the metadata downstream consumers need:

    * ``model_version`` — actual model id the call used
      (``claude-haiku-4-5-20251001``, ``holo3-35b-a3b``). Lands
      in the canonical event as ``<role>_model``.
    * ``prompt_version`` — short hash / tag of the prompt template.
      Lands as ``<role>_prompt``. Empty when the model client
      doesn't surface one (legacy path).
    * ``routing_mode`` — which mode this call ran under. Lands on
      the canonical event so a shadow comparison report can group
      by prod-vs-shadow without re-deriving.
    * ``latency_seconds`` — wall-time of the underlying call.
      Cheaper to capture here than to re-time at every callsite.
    * ``cost_usd`` — best-effort cost estimate (the cost meters
      that already exist can be threaded through; empty when
      unknown).
    """

    role: Role
    routing_mode: RoutingMode
    payload: Any
    model_version: str = ""
    prompt_version: str = ""
    latency_seconds: float = 0.0
    cost_usd: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ModelServingFacade(Protocol):
    """The single invocation surface for every CUA model call (#487).

    Implementations wrap existing clients (Claude, Holo3, local
    llama-server, etc.) and:

    * Apply the routing decision implied by ``routing_mode``
      (prod vs shadow vs canary).
    * Stamp the actual ``model_version`` + ``prompt_version`` onto
      the returned :class:`ModelCallResult` so #488's emit hook
      can pull them onto every canonical event.
    * Honour ``version_pin`` when set — overrides the default
      version selection (the model-registry promotion knob from
      #490 will set this per role).

    ``runtime_checkable`` so handlers can ``isinstance`` a wired
    facade for assertion clarity.

    The protocol is intentionally narrow in v1 — invoke() takes a
    free-form payload and returns a typed result. As handlers
    migrate, per-role specialised signatures can be added on top
    without rewriting the substrate.
    """

    def invoke(
        self,
        *,
        role: Role,
        payload: Any,
        routing_mode: RoutingMode = RoutingMode.PROD,
        version_pin: str = "",
    ) -> ModelCallResult:
        """Run the model call for ``role`` with ``payload``.

        The facade picks the concrete client based on
        ``role`` + ``routing_mode`` + ``version_pin``. Returns a
        :class:`ModelCallResult` carrying the model output AND the
        version metadata needed for event attribution.
        """
        ...


class PassthroughFacade:
    """Default :class:`ModelServingFacade` implementation —
    delegates every call to a single underlying client (#487).

    Useful as a bring-up step before per-role routing is wired:

    * Wrap a single existing client (e.g. AnthropicToolUseClient).
    * Every role calls through to that client.
    * Routing-mode is honoured for logging / event tagging but
      doesn't change *which* client is called (no shadow split yet).
    * model_version / prompt_version stamps come from the wrapped
      client's attributes when present.

    Once #489's shadow routing lands, a follow-up adds a
    :class:`SplitFacade` (or similar) that takes a {role: client}
    map and a router policy. The PassthroughFacade stays as the
    minimal default for tests + bring-up.

    The ``invoker`` callable is the actual model invocation — it
    takes the payload dict and returns the model output. Lets
    callers wire any underlying client without the facade
    knowing about Claude / Holo3 / etc concrete APIs.

    ``version_lookup`` is an optional callable that returns the
    current (model_version, prompt_version) tuple — for clients
    that expose those as attributes, callers can pass
    ``lambda: (client.model, "")`` etc. Without one, the result's
    version fields stay empty (handler-side stamping still works).
    """

    def __init__(
        self,
        invoker: Callable[[Any], Any],
        *,
        version_lookup: Callable[[], tuple[str, str]] | None = None,
    ) -> None:
        self._invoker = invoker
        self._version_lookup = version_lookup

    def invoke(
        self,
        *,
        role: Role,
        payload: Any,
        routing_mode: RoutingMode = RoutingMode.PROD,
        version_pin: str = "",
    ) -> ModelCallResult:
        import time

        t0 = time.monotonic()
        try:
            output = self._invoker(payload)
        except Exception:
            raise
        elapsed = time.monotonic() - t0

        model_version = ""
        prompt_version = ""
        if self._version_lookup is not None:
            try:
                model_version, prompt_version = self._version_lookup()
            except Exception:
                # Best-effort — never let a version probe break a
                # model call. Empty stamps just mean "not known".
                pass
        # ``version_pin`` is an explicit override the model-registry
        # promotion knob will set per-role; it wins over whatever
        # the underlying client reports.
        if version_pin:
            model_version = version_pin

        return ModelCallResult(
            role=role,
            routing_mode=routing_mode,
            payload=output,
            model_version=model_version,
            prompt_version=prompt_version,
            latency_seconds=elapsed,
        )


def stamp_runtime_versions(
    runner: Any, result: ModelCallResult,
) -> None:
    """Merge a :class:`ModelCallResult`'s version metadata into the
    runner's ``runtime_versions`` dict so #488's emit hook surfaces
    it on every canonical event (#487).

    Keys land as ``<role>_model`` / ``<role>_prompt`` — matching
    the canonical key set in :mod:`.versions`. Skipped when the
    result didn't carry the field (empty stamps stay absent rather
    than being recorded as blank strings).

    Best-effort: if the runner doesn't have / can't accept a
    ``runtime_versions`` attribute, log + skip rather than raise.
    The facade's value is the invocation contract; the stamp
    propagation is observability that mustn't break a call.
    """
    versions = getattr(runner, "runtime_versions", None)
    if versions is None:
        versions = {}
        try:
            runner.runtime_versions = versions
        except Exception:
            return
    if not isinstance(versions, dict):
        return
    role_key = result.role.value
    if result.model_version:
        versions[f"{role_key}_model"] = result.model_version
    if result.prompt_version:
        versions[f"{role_key}_prompt"] = result.prompt_version
