r"""Tests for #563 — URL-substring gate short-circuit.

Before this PR, every ``extract_data`` step with ``gate=True`` paid
for a Claude vision verify (~10-15s + \$0.01-0.02). On plans like
boattrader where step 1 is a navigate-verify against a known URL
pattern, that cost is gratuitous — the URL alone is enough signal.

This helper checks ``step.hints.expect_url_contains`` (all substrings
present) and ``expect_url_excludes`` (none present) against
``runner.env.current_url`` BEFORE dispatching the vision call. On a
clean match, returns a PASS StepResult and the runner moves on. On
any ambiguity, returns ``None`` and the caller falls through to the
existing Claude path — never a false positive.
"""

from __future__ import annotations

from types import SimpleNamespace

from mantis_agent.gym._runner_helpers import _try_url_shortcircuit_gate


def _step(hints: dict | None = None, intent: str = "Verify page") -> SimpleNamespace:
    return SimpleNamespace(intent=intent, hints=hints or {})


def _runner(url: str) -> SimpleNamespace:
    env = SimpleNamespace(current_url=url)
    return SimpleNamespace(env=env)


# ── Happy path: all substrings present → PASS ──────────────────────


def test_all_substrings_present_returns_pass() -> None:
    step = _step({"expect_url_contains": ["zip-33101", "by-owner"]})
    runner = _runner("https://www.boattrader.com/boats/state-fl/zip-33101/by-owner/")
    result = _try_url_shortcircuit_gate(runner, step, index=1)
    assert result is not None
    assert result.success is True
    assert "url_shortcircuit" in result.data


def test_single_substring_present_returns_pass() -> None:
    step = _step({"expect_url_contains": ["lu.ma/discover"]})
    runner = _runner("https://lu.ma/discover")
    result = _try_url_shortcircuit_gate(runner, step, index=0)
    assert result is not None
    assert result.success is True


# ── Fall-through: some substring missing → None (use Claude path) ─


def test_missing_substring_returns_none() -> None:
    step = _step({"expect_url_contains": ["zip-33101", "by-owner"]})
    runner = _runner("https://www.boattrader.com/boats/state-fl/zip-33101/")
    # by-owner missing → can't short-circuit
    assert _try_url_shortcircuit_gate(runner, step, index=1) is None


def test_no_expect_list_returns_none() -> None:
    # No signal strong enough to skip vision.
    assert _try_url_shortcircuit_gate(_runner("https://x.com/"), _step({}), 0) is None
    assert _try_url_shortcircuit_gate(_runner("https://x.com/"), _step(), 0) is None
    assert _try_url_shortcircuit_gate(_runner("https://x.com/"), _step({"expect_url_contains": []}), 0) is None


# ── Excludes guard against false positives ─────────────────────────


def test_excludes_blocks_shortcircuit() -> None:
    step = _step({
        "expect_url_contains": ["boattrader.com"],
        "expect_url_excludes": ["error", "404"],
    })
    runner = _runner("https://www.boattrader.com/error/404")
    # boattrader.com is present, but so is "error" — fall through to vision.
    assert _try_url_shortcircuit_gate(runner, step, 0) is None


def test_excludes_absent_allows_shortcircuit() -> None:
    step = _step({
        "expect_url_contains": ["boattrader.com"],
        "expect_url_excludes": ["error", "404"],
    })
    runner = _runner("https://www.boattrader.com/boats/")
    result = _try_url_shortcircuit_gate(runner, step, 0)
    assert result is not None
    assert result.success is True


# ── Robust against malformed hints / missing env ───────────────────


def test_malformed_expect_falls_through() -> None:
    # Not a list — defensive parse, no crash, no short-circuit.
    step = _step({"expect_url_contains": "boattrader.com"})  # str, not list
    runner = _runner("https://www.boattrader.com/")
    assert _try_url_shortcircuit_gate(runner, step, 0) is None


def test_no_hints_attr_returns_none() -> None:
    step = SimpleNamespace(intent="Verify", hints=None)
    runner = _runner("https://www.boattrader.com/")
    assert _try_url_shortcircuit_gate(runner, step, 0) is None


def test_empty_current_url_returns_none() -> None:
    step = _step({"expect_url_contains": ["x"]})
    runner = _runner("")
    assert _try_url_shortcircuit_gate(runner, step, 0) is None


def test_env_current_url_raises_returns_none() -> None:
    class _BadEnv:
        @property
        def current_url(self) -> str:
            raise RuntimeError("env not ready")

    runner = SimpleNamespace(env=_BadEnv())
    step = _step({"expect_url_contains": ["x"]})
    assert _try_url_shortcircuit_gate(runner, step, 0) is None


def test_case_sensitive_match() -> None:
    # URL paths are case-sensitive on most servers; refuse to
    # case-fold the match.
    step = _step({"expect_url_contains": ["BoatTrader"]})
    runner = _runner("https://www.boattrader.com/boats/")
    assert _try_url_shortcircuit_gate(runner, step, 0) is None


# ── String coercion of hint values ─────────────────────────────────


def test_non_string_hint_entries_coerced() -> None:
    # Plans may emit ints (e.g. zip codes) — coerce to str before match.
    step = _step({"expect_url_contains": [33101, "by-owner"]})
    runner = _runner("https://www.boattrader.com/zip-33101/by-owner/")
    result = _try_url_shortcircuit_gate(runner, step, 0)
    assert result is not None
    assert result.success is True
