"""SoM-anchored click dispatch for unstructured CLICK actions (#300).

When ``RoutingPolicy.som_for_unstructured_clicks`` is on AND the env
exposes :meth:`XdotoolGymEnv.cdp_click_at_point` (or equivalent), the
runner hands a brain-emitted CLICK off to a CDP-dispatched
``el.click()`` instead of the legacy xdotool mouse pipeline. The
substituted trajectory step is tagged ``executor_backend=som``.

This pins:

* Policy gating: ``som_for_unstructured_clicks=False`` keeps current
  behaviour (every CLICK goes through xdotool and tags ``vision``).
* Capability gating: an env without ``cdp_click_at_point`` falls
  through to xdotool even with the policy on.
* Successful dispatch swaps the CLICK for a no-op WAIT so :meth:`env.step`
  doesn't double-click, and the trajectory carries ``som``.
* CDP failure (returns False) preserves the original CLICK and the
  trajectory tags ``vision``.

Plus :class:`PageDiscovery`'s CDP backend (its plan-step
``_try_discovery_execution`` consumer is exercised by the existing
``_try_discovery_execution`` integration; here we cover the
PageDiscovery internals end-to-end via a fake CDP env).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image

from mantis_agent.actions import Action, ActionType
from mantis_agent.gym.base import GymEnvironment, GymObservation, GymResult
from mantis_agent.gym.page_discovery import PageDiscovery
from mantis_agent.gym.runner import GymRunner, RoutingPolicy


# ── Fakes ───────────────────────────────────────────────────────────────


@dataclass
class _BrainResult:
    action: Action
    thinking: str = ""
    predicted_outcome: str = ""


class _ScriptedBrain:
    def __init__(self, script: list[Action]) -> None:
        self.script = list(script)
        self.calls = 0

    def think(
        self, frames: Any, task: str, action_history: Any = None,
        screen_size: tuple[int, int] = (100, 100),
    ) -> _BrainResult:
        if self.calls >= len(self.script):
            self.calls += 1
            return _BrainResult(Action(
                ActionType.DONE, {"success": True, "summary": "done"},
            ))
        action = self.script[self.calls]
        self.calls += 1
        return _BrainResult(action)


class _CdpEnv(GymEnvironment):
    """Env that exposes a stub ``cdp_click_at_point`` and counts xdotool steps."""

    def __init__(self, cdp_returns: list[bool] | None = None) -> None:
        self._frame = Image.new("RGB", (100, 100), color=(200, 200, 200))
        self.xdotool_clicks: list[tuple[int, int]] = []
        self.cdp_clicks: list[tuple[int, int]] = []
        self._cdp_returns = list(cdp_returns or [])

    @property
    def screen_size(self) -> tuple[int, int]:
        return (100, 100)

    def reset(self, task: str, **kwargs: Any) -> GymObservation:
        return GymObservation(screenshot=self._frame)

    def step(self, action: Action) -> GymResult:
        # Only record xdotool clicks the env would have executed. A
        # WAIT means the runner already SoM-dispatched via CDP and
        # asked the env to no-op.
        if action.action_type == ActionType.CLICK:
            self.xdotool_clicks.append((
                int(action.params.get("x", 0)),
                int(action.params.get("y", 0)),
            ))
        return GymResult(
            GymObservation(screenshot=self._frame),
            reward=0.0, done=False, info={"url": "https://x.test/a", "title": "A"},
        )

    def close(self) -> None:
        pass

    def cdp_click_at_point(self, x: int, y: int) -> bool:
        self.cdp_clicks.append((x, y))
        if self._cdp_returns:
            return self._cdp_returns.pop(0)
        return True


class _NoCdpEnv(_CdpEnv):
    """Env without the CDP shim — covers capability gating."""

    cdp_click_at_point = None  # type: ignore[assignment]


# ── Runner integration ─────────────────────────────────────────────────


def test_som_dispatch_swaps_click_when_policy_on_and_cdp_succeeds() -> None:
    """Policy on + CDP available + CDP returns True ⇒ trajectory tagged
    ``som`` and xdotool sees no CLICK."""
    brain = _ScriptedBrain([Action(ActionType.CLICK, {"x": 42, "y": 7})])
    env = _CdpEnv(cdp_returns=[True])
    runner = GymRunner(
        brain=brain, env=env, max_steps=3,
        routing_policy=RoutingPolicy(som_for_unstructured_clicks=True),
    )
    result = runner.run("som task")

    assert env.cdp_clicks == [(42, 7)]
    assert env.xdotool_clicks == [], "env should not see an xdotool click"
    backends = [
        s.executor_backend for s in result.trajectory if s.executor_backend
    ]
    assert "som" in backends, backends
    assert result.executor_backend_counts.get("som", 0) == 1


def test_som_dispatch_falls_back_to_vision_when_cdp_fails() -> None:
    """CDP returns False ⇒ original CLICK reaches the env via xdotool
    and trajectory carries ``vision``."""
    brain = _ScriptedBrain([Action(ActionType.CLICK, {"x": 10, "y": 11})])
    env = _CdpEnv(cdp_returns=[False])
    runner = GymRunner(
        brain=brain, env=env, max_steps=3,
        routing_policy=RoutingPolicy(som_for_unstructured_clicks=True),
    )
    result = runner.run("som-miss task")

    assert env.cdp_clicks == [(10, 11)]
    assert env.xdotool_clicks == [(10, 11)], "env should see the xdotool fallback click"
    backends = [
        s.executor_backend for s in result.trajectory if s.executor_backend
    ]
    assert "som" not in backends
    assert "vision" in backends


def test_som_dispatch_disabled_by_default() -> None:
    """Default policy (``som_for_unstructured_clicks=False``) keeps the
    legacy xdotool path even when CDP is available."""
    brain = _ScriptedBrain([Action(ActionType.CLICK, {"x": 50, "y": 60})])
    env = _CdpEnv()
    runner = GymRunner(brain=brain, env=env, max_steps=3)
    result = runner.run("legacy task")

    assert env.cdp_clicks == [], "CDP must not fire when policy is off"
    assert env.xdotool_clicks == [(50, 60)]
    assert all(
        s.executor_backend in ("", "vision")
        for s in result.trajectory
    ), [s.executor_backend for s in result.trajectory]


def test_som_dispatch_skips_when_env_lacks_cdp() -> None:
    """An env that doesn't expose ``cdp_click_at_point`` falls through
    to xdotool even with the policy on — capability gating."""
    brain = _ScriptedBrain([Action(ActionType.CLICK, {"x": 5, "y": 6})])
    env = _NoCdpEnv()
    runner = GymRunner(
        brain=brain, env=env, max_steps=3,
        routing_policy=RoutingPolicy(som_for_unstructured_clicks=True),
    )
    result = runner.run("no-cdp task")

    assert env.xdotool_clicks == [(5, 6)]
    backends = [
        s.executor_backend for s in result.trajectory if s.executor_backend
    ]
    assert "som" not in backends


# ── PageDiscovery CDP backend ──────────────────────────────────────────


class _CdpDiscoveryEnv:
    """Minimal env exposing :meth:`cdp_evaluate` + :meth:`cdp_click_at_point`
    for the PageDiscovery CDP backend tests. Not a full GymEnvironment —
    only the shims PageDiscovery needs."""

    def __init__(self, elements: list[dict]) -> None:
        self._elements = elements
        self.evaluate_calls: list[str] = []
        self.cdp_clicks: list[tuple[int, int]] = []

    def cdp_evaluate(self, expression: str):
        self.evaluate_calls.append(expression)
        # The discovery JS expects a list of element dicts; everything
        # else returns None.
        if "querySelectorAll" in expression:
            return list(self._elements)
        return None

    def cdp_click_at_point(self, x: int, y: int) -> bool:
        self.cdp_clicks.append((x, y))
        return True


def test_page_discovery_cdp_backend_discovers_elements() -> None:
    """PageDiscovery without a Playwright page should drive discovery
    through the env's ``cdp_evaluate`` shim."""
    env = _CdpDiscoveryEnv(elements=[
        {"tag": "button", "text": "Login", "type": "submit", "role": "",
         "name": "submit", "id": "login-btn", "placeholder": "",
         "ariaLabel": "", "value": "", "href": "",
         "bbox": {"x": 10, "y": 20, "w": 80, "h": 30}},
        {"tag": "a", "text": "Forgot password", "type": "", "role": "",
         "name": "", "id": "", "placeholder": "", "ariaLabel": "",
         "value": "", "href": "/forgot",
         "bbox": {"x": 10, "y": 60, "w": 120, "h": 20}},
    ])

    discovery = PageDiscovery(env=env)
    elements = discovery.discover()

    assert env.evaluate_calls, "PageDiscovery should have called cdp_evaluate"
    assert len(elements) == 2
    assert elements[0].text == "Login"
    assert elements[1].href == "/forgot"


def test_page_discovery_cdp_backend_click_uses_cdp_click_at_point() -> None:
    """``click_element`` on the CDP backend dispatches via
    :meth:`cdp_click_at_point` using the bbox center."""
    env = _CdpDiscoveryEnv(elements=[
        {"tag": "button", "text": "Save", "bbox": {"x": 100, "y": 200, "w": 40, "h": 20}},
    ])
    discovery = PageDiscovery(env=env)
    discovery.discover()
    assert discovery.click_element(0) is True
    # bbox center: (100 + 40//2, 200 + 20//2) = (120, 210)
    assert env.cdp_clicks == [(120, 210)]


def test_page_discovery_returns_empty_when_no_backend() -> None:
    """No Playwright page, no CDP shim ⇒ discover() returns an empty list
    (existing behaviour preserved)."""
    discovery = PageDiscovery()  # neither page nor env wired
    assert discovery.discover() == []
