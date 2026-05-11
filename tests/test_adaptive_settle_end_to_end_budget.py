"""End-to-end synthetic ablation: measure sleep-budget reduction
from adaptive_content_settle in ``_extract_listing_data_deep``.

The 4 production smoke attempts (staff-crm-priority, lu.ma, Zillow
no-proxy, Zillow Oxylabs) all failed to exercise the deep-extract
path — wrong code path, click mistargets, or bot CAPTCHAs. This
test runs the same code path against a controlled synthetic env
and measures the total sleep budget end-to-end.

The two configurations under test:

- **adaptive** (current PR #260 behavior): pixel-diff polling,
  exits when consecutive screenshots stabilize
- **fixed** (pre-PR-#260 behavior): four hard sleeps totaling
  ~10s minimum on a 6-viewport scan with zero reveal clicks
  (1.0s pre-settle + 1.5s Home + 5 × 1.0s Page_Down = 7.5s)

Setup mocks ``env.screenshot()`` to return the same stable image
on every call (the static-page case) and the extractor's helpers
so the path runs without Claude API calls. ``time.sleep`` is
patched to a *counter* that records seconds without actually
waiting — so the test runs in milliseconds while measuring the
sleep budget the production runner would have paid.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from PIL import Image


def _stable_image(color: int = 128) -> Image.Image:
    return Image.new("L", (320, 200), color).convert("RGB")


def _run_deep_extract(
    *,
    stable_pages: bool,
    monkeypatch,
) -> tuple[float, list[float]]:
    """Run ``_extract_listing_data_deep`` end-to-end with a fake env
    that returns either a stable image stream (page settled instantly)
    or an ever-changing one (page never settles).

    Returns ``(total_sleep_seconds, sleep_log)``. ``total_sleep_seconds``
    is what the production runner *would have* paid in wall-clock waits.
    ``sleep_log`` is the per-call list so test assertions can pinpoint
    where the budget went.
    """
    from mantis_agent.gym.step_context import StepContext
    from mantis_agent.gym.step_handlers.claude_step import ClaudeStepHandler

    # Virtual clock — sleep advances ``virtual_now`` instead of actually
    # waiting, AND time.time() returns ``virtual_now``. This way
    # adaptive_content_settle's ``while time.time() - start <
    # max_seconds`` check fires correctly on the "changing page"
    # path; otherwise the mocked sleep never advances real time and
    # the loop spins thousands of iterations before max_seconds is
    # hit via loop-iteration wall clock.
    state = {"now": 1000.0}
    sleep_log: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleep_log.append(float(seconds))
        state["now"] += float(seconds)

    def fake_time() -> float:
        return state["now"]

    monkeypatch.setattr("mantis_agent.gym.step_handlers.claude_step.time.sleep", fake_sleep)
    monkeypatch.setattr("mantis_agent.gym._runner_helpers.time.sleep", fake_sleep)
    monkeypatch.setattr("mantis_agent.gym._runner_helpers.time.time", fake_time)

    # Build a fake env. screenshot() returns a fixed image (stable)
    # or alternating images (changing).
    env = MagicMock()
    if stable_pages:
        stable = _stable_image(128)
        env.screenshot.return_value = stable
    else:
        # Alternating black/white — page never settles.
        toggle = {"i": 0}

        def changing_screenshot():
            toggle["i"] += 1
            return _stable_image(0 if toggle["i"] % 2 else 255)

        env.screenshot.side_effect = changing_screenshot

    env.step.return_value = None

    # Mock the runner shim: deep-extract reads runner.costs,
    # _set_scroll_state, _last_known_url, parent helpers.
    runner = MagicMock()
    runner.costs = {"claude_extract": 0, "gpu_steps": 0, "gpu_seconds": 0}
    runner._last_known_url = "https://www.example.com/boat/1"
    runner._set_scroll_state = MagicMock()

    # Extractor mock — find_listing_content_control returns None
    # (no reveal clicks fire; we're measuring the pre/Home/Page_Down
    # settles, not the reveal-click settle), extract_multi returns a
    # viable result so the path completes cleanly.
    extractor = MagicMock()
    extractor.find_listing_content_control.return_value = None
    extractor.extract_multi.return_value = MagicMock(
        is_viable=MagicMock(return_value=True),
    )

    ctx = StepContext(
        env=env, brain=None, extractor=extractor, grounding=None,
        cost_meter=None, dynamic_verifier=None,
        scanner=runner.scanner, site_config=None,
        tool_channel=None, extraction_cache=None,
        state={"index": 0},
    )

    handler = ClaudeStepHandler(runner)
    initial_shot = _stable_image(128)
    handler._extract_listing_data_deep(initial_shot, ctx)

    return sum(sleep_log), sleep_log


def test_stable_page_sleep_budget_under_legacy_max(monkeypatch) -> None:
    """The whole point of PR #260: on a static page, adaptive settle
    exits early at each of the 4 sleep sites.

    Legacy fixed budget (max possible on this code path with zero
    reveal clicks) is:
      pre-settle (1.0s)
      + Home settle (1.5s)
      + 5 × Page_Down settle (5 × 1.0s = 5.0s)
      = 7.5s minimum

    Adaptive settle on a stable page should exit much earlier at
    every site — at most ~min_seconds + 1 poll = 0.5s per site.
    Stable-page budget ceiling:
      pre-settle    (0.2 min + ~0.3 poll  ≈ 0.5s)
      Home settle   (0.3 min + ~0.3 poll  ≈ 0.6s)
      5 × Page_Down (5 × ≈0.5s            ≈ 2.5s)
      = ~3.6s total — *well* below the 7.5s legacy budget.
    """
    total_sleep, log = _run_deep_extract(stable_pages=True, monkeypatch=monkeypatch)

    # Hard ceiling: the new budget must be strictly less than the
    # legacy fixed-sleep budget (7.5s).
    assert total_sleep < 7.5, (
        f"adaptive settle budget {total_sleep:.2f}s should be < 7.5s legacy"
    )
    # Soft target: should be at most ~5s on a stable page (claimed
    # ~50% reduction in PR #260's description).
    assert total_sleep < 5.0, (
        f"adaptive settle on stable page paid {total_sleep:.2f}s; "
        f"expected <5s (~50% reduction). Log: {log}"
    )


def test_unstable_page_pays_close_to_legacy_max(monkeypatch) -> None:
    """The non-regressive guarantee: pages that never settle pay
    close to the original fixed-sleep budget.

    On an ever-changing page, adaptive settle hits ``max_seconds``
    at every site. Total ceiling matches legacy ~7.5s (pre + Home
    + 5×Page_Down) within poll-granularity tolerance.
    """
    total_sleep, log = _run_deep_extract(stable_pages=False, monkeypatch=monkeypatch)

    # Should be at or near the legacy budget — within ~30%
    # tolerance for poll-granularity rounding.
    assert total_sleep >= 6.0, (
        f"adaptive settle on changing page paid only {total_sleep:.2f}s; "
        f"non-regression contract says it should pay near legacy 7.5s. "
        f"Log: {log}"
    )
    # And not wildly over — caps should prevent runaway.
    assert total_sleep <= 10.0, (
        f"adaptive settle on changing page paid {total_sleep:.2f}s; "
        f"max budget is 1.0 + 1.5 + 5×1.0 = 7.5s + small overhead. Log: {log}"
    )


def test_savings_on_stable_vs_unstable_pages(monkeypatch) -> None:
    """The headline metric: per-extraction sleep-budget savings
    on a static page vs. an XHR-churning one.

    Static-page savings should be at least 3 seconds. That's the
    floor the PR description committed to (\"~5s saved per static-
    page extraction\"). Below this the PR isn't earning its keep
    even on the most favorable case.
    """
    # Run sequentially to avoid monkeypatch state bleed.
    # First: changing page (worst case)
    changing_total, _ = _run_deep_extract(stable_pages=False, monkeypatch=monkeypatch)
    # Then: stable page (best case) — fresh monkeypatch state
    monkeypatch.undo()
    stable_total, _ = _run_deep_extract(stable_pages=True, monkeypatch=monkeypatch)

    savings = changing_total - stable_total
    assert savings >= 3.0, (
        f"static vs changing budget delta is {savings:.2f}s; "
        f"PR #260 claimed ~5s savings. "
        f"changing={changing_total:.2f}s, stable={stable_total:.2f}s"
    )
