"""Tests for issue #255 — already-seen URL skip envelope.

Fifth tactical sibling in the skip-envelope family (#246 recipe
rejection, #250 navigation halt, #254 context budget, #248
exploration substrate). Same envelope contract, fifth trigger
source: when ``extract_data`` is about to fire against a URL the
host has already processed in a prior run, the runner short-circuits
with ``StepResult.skip=True / skip_reason='already_seen'`` and
never invokes ClaudeExtractor.

Design split (per architectural review on #255):

- **Data + policy stay host-side.** The host owns ``master.csv``
  or whatever its persistence layer is; the host decides what
  "seen" means (URL match, content hash, CRM lookup, …).
- **Mantis owns only the timing window.** The post-navigate
  pre-extract window is the only place a duplicate URL can be
  short-circuited cheaply, and that window is only observable
  inside the runner.

So the surface is a single predicate callable. The host writes a
one-liner that consults its own state; mantis bakes in zero
URL-set assumptions. A future helper can wrap a URL set into a
predicate if multiple hosts want that pattern, but v1 ships the
minimum surface.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from mantis_agent.gym.step_context import StepContext
from mantis_agent.gym.step_handlers.claude_step import ClaudeStepHandler
from mantis_agent.plan_decomposer import MicroIntent


# ── MicroPlanRunner constructor opt-in ──────────────────────────────


def test_runner_default_seen_url_predicate_is_none() -> None:
    """Default preserves today's behavior — no predicate means no
    short-circuit at extract_data, every detail page extracts."""
    from mantis_agent.gym.micro_runner import MicroPlanRunner
    runner = MicroPlanRunner(brain=MagicMock(), env=MagicMock())
    assert runner.seen_url_predicate is None


def test_runner_accepts_seen_url_predicate_kwarg() -> None:
    from mantis_agent.gym.micro_runner import MicroPlanRunner

    def predicate(url: str) -> bool:
        return "foo" in url

    runner = MicroPlanRunner(
        brain=MagicMock(), env=MagicMock(), seen_url_predicate=predicate,
    )
    assert runner.seen_url_predicate is predicate


# ── extract_data short-circuit ──────────────────────────────────────


def _build_ctx_for_extract_data(
    *,
    predicate=None,
    is_detail_page: bool = True,
    current_url: str = "https://www.example.com/boat/foo",
) -> tuple[ClaudeStepHandler, MicroIntent, StepContext, dict]:
    """Construct the minimum runner+ctx needed to exercise the
    extract_data branch. Returns ``(handler, step, ctx, call_log)``
    where ``call_log`` records whether the extractor's deep-extract
    helper was invoked."""
    runner = MagicMock()
    runner.seen_url_predicate = predicate
    runner._best_effort_current_url = MagicMock(return_value=current_url)
    runner.site_config = MagicMock()
    runner.site_config.is_detail_page = MagicMock(return_value=is_detail_page)
    runner.costs = {"claude_extract": 0, "gpu_steps": 0, "gpu_seconds": 0.0}
    runner._last_extracted = {}
    runner._current_page = 1
    runner._current_item_label = MagicMock(return_value="item")
    runner.scanner.is_duplicate.return_value = False
    runner.scanner.mark_seen = MagicMock()

    extractor = MagicMock()
    extractor.schema = MagicMock()
    extractor.schema.rejection_intents = {}

    env = MagicMock()
    env.screenshot.return_value = MagicMock()

    extraction_cache = MagicMock()
    extraction_cache.read_enabled = False
    extraction_cache.get.return_value = None

    dynamic_verifier = MagicMock()

    ctx = StepContext(
        env=env, brain=None, extractor=extractor, grounding=None,
        cost_meter=None, dynamic_verifier=dynamic_verifier,
        scanner=runner.scanner, site_config=runner.site_config,
        tool_channel=None, extraction_cache=extraction_cache,
        state={"index": 5},
    )

    handler = ClaudeStepHandler(runner)

    call_log = {"deep_extract_called": False}

    def fake_deep_extract(shot, ctx):
        call_log["deep_extract_called"] = True
        data = MagicMock()
        data.url = current_url
        data.is_viable.return_value = True
        data.dealer_reason.return_value = ""
        data.missing_required_reason.return_value = ""
        data.to_summary.return_value = "summary"
        data.extracted_fields = {}
        return (data, 1)

    handler._extract_listing_data_deep = fake_deep_extract

    step = MicroIntent(
        intent="Read structured fields off this detail page",
        type="extract_data",
        claude_only=True,
    )
    return handler, step, ctx, call_log


def test_extract_data_short_circuits_when_predicate_returns_true(monkeypatch) -> None:
    """Happy path: detail-page URL + host predicate returns True →
    StepResult.skip=True, ClaudeExtractor never called."""
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.claude_step.time.sleep",
        lambda *_: None,
    )
    url = "https://www.example.com/boat/already-seen-slug"
    handler, step, ctx, call_log = _build_ctx_for_extract_data(
        predicate=lambda u: "already-seen-slug" in u,
        is_detail_page=True,
        current_url=url,
    )
    result = handler.execute(step, ctx)

    assert result.success is False
    assert result.skip is True
    assert result.skip_reason == "already_seen"
    # Surface the URL in data so post-mortem / dashboards know which
    # one was deduped.
    assert "already_seen" in result.data
    assert url in result.data
    # Deep-extract never invoked → no Claude vision call → no $$.
    assert call_log["deep_extract_called"] is False


def test_extract_data_proceeds_when_predicate_returns_false(monkeypatch) -> None:
    """Predicate returns False — runner runs the normal extract
    path."""
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.claude_step.time.sleep",
        lambda *_: None,
    )
    handler, step, ctx, call_log = _build_ctx_for_extract_data(
        predicate=lambda u: False,
        is_detail_page=True,
        current_url="https://www.example.com/boat/fresh",
    )
    result = handler.execute(step, ctx)

    # Normal extract path ran (success because our fake helper
    # returns viable data).
    assert result.skip is False
    assert call_log["deep_extract_called"] is True


def test_extract_data_proceeds_when_predicate_is_none(monkeypatch) -> None:
    """No predicate → no short-circuit. Default behavior preserved
    for hosts that don't opt in."""
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.claude_step.time.sleep",
        lambda *_: None,
    )
    handler, step, ctx, call_log = _build_ctx_for_extract_data(
        predicate=None,
        is_detail_page=True,
        current_url="https://www.example.com/boat/any",
    )
    result = handler.execute(step, ctx)

    assert result.skip is False
    assert call_log["deep_extract_called"] is True


def test_extract_data_proceeds_when_not_a_detail_page(monkeypatch) -> None:
    """Applicability gate: if the URL isn't a detail page, the
    predicate isn't consulted. A search/results URL shouldn't be
    short-circuited even if it happens to be in the host's seen
    set (which would be a bug in the host but mantis defends in
    depth)."""
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.claude_step.time.sleep",
        lambda *_: None,
    )
    handler, step, ctx, call_log = _build_ctx_for_extract_data(
        predicate=lambda u: True,  # would short-circuit if applied
        is_detail_page=False,       # but applicability gate is False
        current_url="https://www.example.com/boats/search?state=fl",
    )
    result = handler.execute(step, ctx)

    assert result.skip is False
    assert call_log["deep_extract_called"] is True


def test_extract_data_proceeds_when_current_url_is_empty(monkeypatch) -> None:
    """If the runner can't read the current URL (cold env, CDP
    glitch, …), the predicate isn't consulted — fall through to
    the normal extract path rather than short-circuiting on the
    empty string. Defends against a predicate that matches \"\"."""
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.claude_step.time.sleep",
        lambda *_: None,
    )
    handler, step, ctx, call_log = _build_ctx_for_extract_data(
        predicate=lambda u: True,
        is_detail_page=True,
        current_url="",
    )
    result = handler.execute(step, ctx)

    assert result.skip is False
    assert call_log["deep_extract_called"] is True


def test_extract_data_short_circuits_before_cache_check(monkeypatch) -> None:
    """Order matters: the predicate runs *before* the in-session
    cache check, so a known-seen URL skips even if it's not in
    the warm cache. Today's cache is an intra-session
    optimization; the predicate is cross-session host state."""
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.claude_step.time.sleep",
        lambda *_: None,
    )
    url = "https://www.example.com/boat/known"
    handler, step, ctx, call_log = _build_ctx_for_extract_data(
        predicate=lambda u: u == url,
        is_detail_page=True,
        current_url=url,
    )
    # Make the cache "warm" but on a different URL — the predicate
    # should still fire and short-circuit before the cache is
    # consulted.
    ctx.extraction_cache.read_enabled = True
    ctx.extraction_cache.get.return_value = None

    result = handler.execute(step, ctx)

    assert result.skip is True
    assert result.skip_reason == "already_seen"
    # The cache was never consulted (extraction_cache.get not called)
    # because the predicate short-circuited first.
    ctx.extraction_cache.get.assert_not_called()


def test_extract_data_predicate_exception_does_not_break_run(monkeypatch) -> None:
    """A buggy host predicate (raises on a malformed URL, e.g.)
    should not crash the runner. Fall through to the normal
    extract path on any predicate exception — better to over-
    extract than to halt the whole run."""
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.claude_step.time.sleep",
        lambda *_: None,
    )

    def buggy_predicate(url):
        raise RuntimeError("host predicate crashed")

    handler, step, ctx, call_log = _build_ctx_for_extract_data(
        predicate=buggy_predicate,
        is_detail_page=True,
        current_url="https://www.example.com/boat/foo",
    )
    result = handler.execute(step, ctx)

    # Normal extract ran; the buggy predicate was swallowed.
    assert result.skip is False
    assert call_log["deep_extract_called"] is True


def test_extract_data_does_not_short_circuit_for_extract_url_step(monkeypatch) -> None:
    """``extract_url`` is a different step type (one-shot URL
    capture, not the full extract). The predicate doesn't apply
    there — extract_url is too cheap to deserve the gate, and the
    URL it reports is the *output* of the step, not an input."""
    monkeypatch.setattr(
        "mantis_agent.gym.step_handlers.claude_step.time.sleep",
        lambda *_: None,
    )
    handler, step, ctx, call_log = _build_ctx_for_extract_data(
        predicate=lambda u: True,
        is_detail_page=True,
        current_url="https://www.example.com/boat/foo",
    )
    # Switch step type to extract_url.
    step = MicroIntent(
        intent="Read URL",
        type="extract_url",
        claude_only=True,
    )

    # The extract_url branch needs extractor.extract — give it a
    # minimal response.
    extractor = ctx.extractor
    data = MagicMock()
    data.url = "https://www.example.com/boat/foo"
    extractor.extract.return_value = data

    result = handler.execute(step, ctx)

    # No skip envelope — extract_url ran normally.
    assert result.skip is False
    extractor.extract.assert_called_once()
