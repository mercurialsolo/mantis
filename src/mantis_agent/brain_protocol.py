"""Brain protocol + registry — the pluggable model boundary.

The repo currently ships several concrete brains (Holo3, Claude, OpenCUA,
llama.cpp, Gemma4) that grew independently and don't share a common type. This
module is the target interface they should converge on, and the registry
callers should consult instead of importing concrete classes.

A "brain" is the thing that turns (frames, task, history) → next ``Action``.
Whether it does perception + reasoning + grounding in one shot (Gemma4, Holo3)
or splits work across surgical Claude calls (Claude grounding / extraction)
is an implementation detail of the brain, not a contract on the runner.

Wire a new backend in three steps:

    from mantis_agent.brain_protocol import Brain, register_brain

    class MyBrain:
        def load(self) -> None: ...
        def think(self, frames, task, action_history=None,
                  screen_size=(1920, 1080)) -> "InferenceResult": ...

    register_brain("my-brain", lambda: MyBrain())

Then run with ``MANTIS_BRAIN=my-brain``. The runner asks
``resolve_brain(os.environ["MANTIS_BRAIN"])`` and gets a fresh instance.

This file deliberately has no runtime dependency on torch / transformers /
anthropic — it is pure typing + registry so the orchestrator install can
import it without pulling GPU deps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

if TYPE_CHECKING:
    from PIL import Image

    from .actions import Action


@runtime_checkable
class Brain(Protocol):
    """The minimum surface the runner needs from a model backend.

    Concrete brains may expose more (cost accounting, streaming, batched
    grounding) — those are optional and discovered via ``hasattr`` at the
    call site, not part of this contract.
    """

    def load(self) -> None:
        """Load weights / open clients. Idempotent. Called once before think()."""

    def think(
        self,
        frames: "list[Image.Image]",
        task: str,
        action_history: "list[Action] | None" = None,
        screen_size: tuple[int, int] = (1920, 1080),
    ) -> Any:  # InferenceResult — Any to avoid an import cycle
        """One perception → reasoning → action cycle.

        ``frames`` is oldest-first; the last frame is the current screen state.
        Returns an ``InferenceResult``-shaped object exposing at least
        ``.action`` (an ``Action``) and ``.raw_output`` (str).
        """


# ── Registry ────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, Callable[[], Brain]] = {}


def register_brain(name: str, factory: Callable[[], Brain]) -> None:
    """Register a brain factory under ``name``.

    ``factory`` is called with no arguments and must return a fresh ``Brain``.
    Calling ``register_brain`` twice with the same name overrides the previous
    entry — last-write-wins so plugin packages can opt-in upgrade.
    """
    if not name or not isinstance(name, str):
        raise ValueError("brain name must be a non-empty string")
    _REGISTRY[name] = factory


def resolve_brain(name: str) -> Brain:
    """Look up a registered brain factory and instantiate it.

    Raises ``KeyError`` if no factory is registered. The runner is expected
    to call ``brain.load()`` before the first ``think()``.
    """
    try:
        factory = _REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise KeyError(
            f"no brain registered as {name!r}; available: {available}"
        ) from exc
    return factory()


def list_brains() -> list[str]:
    """Names of all registered brains, sorted."""
    return sorted(_REGISTRY)


# ── Built-in brain registration ─────────────────────────────────────────────
#
# Each factory imports its concrete class lazily *inside* the lambda. That
# keeps ``register_brain`` cheap — registration always succeeds — and defers
# heavyweight imports (torch, transformers, anthropic) until the brain is
# actually resolved. A slim install can list the registered names without
# pulling GPU / API dependencies.
#
# ``MANTIS_BRAIN`` is the public selector. The legacy ``MANTIS_MODEL`` env
# var maps onto the same names; the runtime prefers ``MANTIS_BRAIN`` when
# both are set.


def _holo3_factory() -> Brain:
    from .brain_holo3 import Holo3Brain
    return Holo3Brain()


def _claude_factory() -> Brain:
    from .brain_claude import ClaudeBrain
    return ClaudeBrain()


def _opencua_factory() -> Brain:
    from .brain_opencua import OpenCUABrain
    return OpenCUABrain()


def _llamacpp_factory() -> Brain:
    from .brain_llamacpp import LlamaCppBrain
    return LlamaCppBrain()


def _gemma4_factory() -> Brain:
    from .brain import Gemma4Brain
    return Gemma4Brain()


def _agents_factory() -> Brain:
    from .brain_agents import AgentSBrain
    return AgentSBrain()


def _register_builtins() -> None:
    """Register the in-tree brains. Idempotent."""
    register_brain("holo3", _holo3_factory)
    register_brain("claude", _claude_factory)
    register_brain("opencua", _opencua_factory)
    register_brain("llamacpp", _llamacpp_factory)
    register_brain("gemma4", _gemma4_factory)
    register_brain("agent-s", _agents_factory)


_register_builtins()


def resolve_from_env(default: str = "holo3") -> Brain:
    """Resolve the brain selected by ``MANTIS_BRAIN`` (or legacy ``MANTIS_MODEL``).

    Used by the deployment runtimes (``baseten_server``, ``modal_runtime``)
    so the env-var contract lives in one place. Falls back to ``default``
    if neither variable is set.
    """
    import os

    name = (
        os.environ.get("MANTIS_BRAIN")
        or os.environ.get("MANTIS_MODEL")
        or default
    )
    # Map a legacy alias: ``MANTIS_MODEL=gemma4-cua`` → ``gemma4``.
    if name == "gemma4-cua":
        name = "gemma4"
    return resolve_brain(name)
