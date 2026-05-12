"""NavigateHandler — drive the browser to a URL and arm the listings scanner.

First real handler extraction under EPIC #161 Phase 2. The body is a
verbatim lift from ``MicroPlanRunner._execute_navigate``; the runner
keeps a thin delegating shim so existing tests and external callers
(host integrations, tools that drive the runner directly) continue
to work unchanged.

What the handler reads from :class:`StepContext`:

- ``env``               — for ``reset(task=..., start_url=...)`` and ``step(KEY_PRESS)``
- ``scanner``           — to set ``results_base_url`` and ``required_filter_tokens``
- ``dynamic_verifier``  — to seed the per-page check schedule

What the handler reads from the runner (via parent back-reference):

- ``_derive_filter_tokens``   — static method on the runner; pure function over URL string
- ``_current_page``           — runner state, set to 1 on navigate
- ``_last_known_url``         — runner state, set to the navigated URL
- ``_reset_results_scan_state`` — runner method, clears scanner viewport / page-listing cache
- ``browser_state.set_scroll_state`` — re-seats scroll context to ``results_top``

The runner state attributes (``_current_page``, ``_last_known_url``,
``_scroll_state``) are *not* on the scanner today — they live on the
runner because BrowserState (#115 step 4) was the first split. Phase 4
of #161 can promote them onto a unified ``RunState`` dataclass; for now
we read/write through the parent reference, the same pattern
:class:`~.browser_state.BrowserState` already uses.
"""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING

from ...actions import Action, ActionType
from .. import adaptive_settle
from ..checkpoint import StepResult
from ..step_context import StepContext

if TYPE_CHECKING:
    from ..micro_runner import MicroPlanRunner
    from ...plan_decomposer import MicroIntent

logger = logging.getLogger(__name__)


class NavigateHandler:
    """Implements :class:`~..step_context.StepHandler` for ``navigate`` steps.

    Verbatim port of ``MicroPlanRunner._execute_navigate``: same URL
    extraction regex, same wait-resolution order (step.params >
    MANTIS_NAV_WAIT_SECONDS env > 18s default), same env.reset call,
    same Home key + 2s settle, same scanner / dynamic-verifier
    bookkeeping. The only structural difference is that ``index`` is
    read from ``ctx.state["index"]`` so the handler signature matches
    the protocol.
    """

    step_type = "navigate"

    def __init__(self, runner: "MicroPlanRunner") -> None:
        self.parent = runner

    def execute(self, step: "MicroIntent", ctx: StepContext) -> StepResult:
        index = int(ctx.state.get("index", 0))
        url_match = re.search(r'https?://[^\s"]+', step.intent)
        url = url_match.group() if url_match else ""

        # Defense-in-depth (#209 Symptom 4): when the decomposer paraphrases
        # the intent and drops the URL, recover from ``params.url`` before
        # giving up. The repair pass in ``PlanDecomposer`` populates
        # ``params.url`` whenever the source plan named a URL the step lost.
        if not url:
            params_url = str((step.params or {}).get("url", ""))
            pm = re.search(r'https?://[^\s"]+', params_url)
            if pm:
                url = pm.group()
                logger.info(
                    f"  [navigate] URL absent from intent; recovered from params.url: {url}"
                )

        if not url:
            logger.warning(f"  [navigate] No URL found in intent: {step.intent[:60]}")
            return StepResult(step_index=index, intent=step.intent, success=False)

        try:
            wait_seconds = float(
                (step.params or {}).get("wait_after_load_seconds")
                or os.environ.get("MANTIS_NAV_WAIT_SECONDS")
                or 18
            )
        except (TypeError, ValueError):
            wait_seconds = 18.0
        wait_seconds = max(0.0, min(wait_seconds, 120.0))

        logger.info(f"  [navigate] Loading {url} (first-paint wait {wait_seconds:.0f}s)")
        env = ctx.env
        scanner = ctx.scanner
        dynamic_verifier = ctx.dynamic_verifier
        runner = self.parent

        try:
            env.reset(task="navigate", start_url=url)
            # Wait for Cloudflare challenge to auto-solve + page render.
            # #294: cap at the configured budget (default 18s); exit early
            # when the frame hash stabilises — most sites finish first
            # paint in 2-5s and the remaining budget is pure tax.
            adaptive_settle.settle_after_action(env, max_seconds=wait_seconds)
            env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Home"}))
            adaptive_settle.settle_after_action(env, max_seconds=2.0)

            # Anchor the results scan state to this URL.
            if scanner is not None:
                scanner.results_base_url = url
                scanner.required_filter_tokens = runner._derive_filter_tokens(url)
            else:  # legacy fallback for tests that pass ctx without scanner
                runner._results_base_url = url
                runner._required_filter_tokens = runner._derive_filter_tokens(url)

            runner._current_page = 1
            runner._last_known_url = url
            runner._reset_results_scan_state()
            runner.browser_state.set_scroll_state(
                context="results_top", url=url, page_downs=0, wheel_downs=0,
            )

            if dynamic_verifier is not None:
                dynamic_verifier.set_required_filter_tokens(
                    runner._required_filter_tokens
                )
                dynamic_verifier.record_page_start(page=runner._current_page, url=url)
            return StepResult(step_index=index, intent=step.intent, success=True)
        except Exception as e:
            logger.error(f"  [navigate] Failed: {e}")
            return StepResult(step_index=index, intent=step.intent, success=False)
