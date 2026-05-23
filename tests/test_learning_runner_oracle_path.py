"""Unit tests for the oracle-path verifier helpers on ``LearningRunner``.

Issue #594 wired the per-step oracle primitives shipped in #590 / PR #593
into ``gym/learning_runner.py``. These tests cover the three private
helpers added there:

* ``_env_current_url`` — accessor that tolerates env adapters with
  different URL conventions (``current_url`` callable / attribute,
  ``url`` attribute, ``get_url()`` method, or no surface at all).
* ``_oracle_verify_url_signals`` — case-insensitive substring check
  used in place of ``verify_filter`` and ``verify_on_correct_page``.
* ``_oracle_verify_state_change`` — mutation-log delta check used in
  place of ``verify_step`` on state-changing actions.
* ``_sync_mutation_cursor`` — phase-start helper that advances the
  cursor past any reset/seed mutations so the first step's verifier
  doesn't see them.

The LearningRunner itself isn't booted — these tests instantiate it
directly with stub env / verifier objects and exercise the private
helpers. End-to-end `learn()` coverage stays for the integration
suite under ``tests/integration/`` (not in this repo today).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

# Import ``mantis_agent.rewards`` ahead of ``recipes.marketplace_listings``
# to sidestep the latent circular noted in #590's PR — not needed here
# (no rewards imports) but keeping the ordering consistent.
from mantis_agent.gym.learning_runner import LearningRunner


# ── Test scaffolding ───────────────────────────────────────────────────


def _stub_env(*, url: str = "", url_callable: bool = False):
    """Build a minimal env stub exposing the URL via the requested shape."""
    env = SimpleNamespace()
    if url_callable:
        env.current_url = lambda: url
    else:
        env.current_url = url
    return env


def _stub_session(url: str = "http://env.test", token: str = "admin-tok"):
    """Build a minimal EnvSession-shaped object."""
    return SimpleNamespace(url=url, admin_token=token)


def _make_runner(*, env=None, oracle_session=None) -> LearningRunner:
    """Construct LearningRunner without booting any real component."""
    return LearningRunner(
        brain=SimpleNamespace(),
        env=env if env is not None else _stub_env(url="http://env.test/boats/"),
        verifier=SimpleNamespace(),
        oracle_session=oracle_session,
    )


# ── _env_current_url ───────────────────────────────────────────────────


def test_env_current_url_reads_callable_current_url():
    env = _stub_env(url="http://example.test/boats/?make=Sea+Ray", url_callable=True)
    runner = _make_runner(env=env)
    assert runner._env_current_url() == "http://example.test/boats/?make=Sea+Ray"


def test_env_current_url_reads_attribute_current_url():
    env = _stub_env(url="http://example.test/boats/", url_callable=False)
    runner = _make_runner(env=env)
    assert runner._env_current_url() == "http://example.test/boats/"


def test_env_current_url_falls_back_to_get_url_method():
    env = SimpleNamespace()
    env.get_url = lambda: "http://example.test/from-getter"
    runner = _make_runner(env=env)
    assert runner._env_current_url() == "http://example.test/from-getter"


def test_env_current_url_returns_empty_when_no_accessor():
    env = SimpleNamespace()
    runner = _make_runner(env=env)
    assert runner._env_current_url() == ""


def test_env_current_url_swallows_callable_errors():
    """A throwing accessor shouldn't crash the verifier — return empty."""
    env = SimpleNamespace()
    def _boom():
        raise RuntimeError("env not ready")
    env.current_url = _boom
    runner = _make_runner(env=env)
    assert runner._env_current_url() == ""


def test_env_current_url_handles_none_env():
    """A None env shouldn't crash the accessor — must return ``""``."""
    runner = LearningRunner(
        brain=SimpleNamespace(),
        env=None,
        verifier=SimpleNamespace(),
    )
    assert runner._env_current_url() == ""


# ── _oracle_verify_url_signals ─────────────────────────────────────────


def test_url_signals_verifies_on_substring_match():
    runner = _make_runner(env=_stub_env(url="http://x.test/boats/?seller=by-owner"))
    result = runner._oracle_verify_url_signals(["private seller", "by-owner"])
    assert result.verified is True
    assert result.issue == ""


def test_url_signals_case_insensitive():
    runner = _make_runner(env=_stub_env(url="http://x.test/Boats/?Make=Sea+Ray"))
    result = runner._oracle_verify_url_signals(["sea+ray"])
    assert result.verified is True


def test_url_signals_fails_with_no_match():
    runner = _make_runner(env=_stub_env(url="http://x.test/about"))
    result = runner._oracle_verify_url_signals(["private seller", "by-owner"])
    assert result.verified is False
    assert result.issue == "filter_lost"


def test_url_signals_returns_no_url_when_env_silent():
    runner = _make_runner(env=SimpleNamespace())
    result = runner._oracle_verify_url_signals(["anything"])
    assert result.verified is False
    assert result.issue == "no_url"


def test_url_signals_empty_signal_list_is_verified():
    """An empty signal list with a valid URL is a degenerate-but-OK case."""
    runner = _make_runner(env=_stub_env(url="http://x.test/boats/"))
    result = runner._oracle_verify_url_signals([])
    assert result.verified is True


def test_url_signals_skips_empty_strings():
    """Empty strings in the signal list shouldn't accidentally match the URL."""
    runner = _make_runner(env=_stub_env(url="http://x.test/about"))
    result = runner._oracle_verify_url_signals(["", "by-owner"])
    assert result.verified is False


# ── _oracle_verify_state_change ────────────────────────────────────────


def test_state_change_no_session_returns_no_oracle():
    """Helper requires an oracle_session; without one, it returns no_oracle_session."""
    runner = _make_runner(oracle_session=None)
    result = runner._oracle_verify_state_change()
    assert result.verified is False
    assert result.issue == "no_oracle_session"


def test_state_change_verified_on_any_mutation_without_expected_ops():
    runner = _make_runner(oracle_session=_stub_session())
    fake = {"mutations": [
        {"id": 1, "operation": "consent_set"},
    ]}
    with patch(
        "mantis_agent.sim_envs.oracle_client.fetch_mutations",
        return_value=fake,
    ):
        result = runner._oracle_verify_state_change()
    assert result.verified is True
    assert runner._last_mutation_id == 1


def test_state_change_verified_when_expected_op_matches():
    runner = _make_runner(oracle_session=_stub_session())
    fake = {"mutations": [
        {"id": 1, "operation": "consent_set"},
        {"id": 2, "operation": "lead_submitted"},
    ]}
    with patch(
        "mantis_agent.sim_envs.oracle_client.fetch_mutations",
        return_value=fake,
    ):
        result = runner._oracle_verify_state_change(
            expected_ops={"lead_submitted"},
        )
    assert result.verified is True
    assert runner._last_mutation_id == 2


def test_state_change_unverified_when_expected_op_missing():
    """Mutations landed but none match the expected set — wrong_op."""
    runner = _make_runner(oracle_session=_stub_session())
    fake = {"mutations": [
        {"id": 1, "operation": "consent_set"},
    ]}
    with patch(
        "mantis_agent.sim_envs.oracle_client.fetch_mutations",
        return_value=fake,
    ):
        result = runner._oracle_verify_state_change(
            expected_ops={"lead_submitted"},
        )
    assert result.verified is False
    assert result.issue == "wrong_op"
    # Cursor still advances so the non-matching mutation doesn't re-fire.
    assert runner._last_mutation_id == 1


def test_state_change_empty_delta_is_no_state_change():
    runner = _make_runner(oracle_session=_stub_session())
    with patch(
        "mantis_agent.sim_envs.oracle_client.fetch_mutations",
        return_value={"mutations": []},
    ):
        result = runner._oracle_verify_state_change()
    assert result.verified is False
    assert result.issue == "no_state_change"
    # Cursor unchanged on empty delta.
    assert runner._last_mutation_id == 0


def test_state_change_fetch_error_surfaces_as_oracle_unreachable():
    runner = _make_runner(oracle_session=_stub_session())
    with patch(
        "mantis_agent.sim_envs.oracle_client.fetch_mutations",
        return_value={"mutations": [], "error": "HTTP 503: down"},
    ):
        result = runner._oracle_verify_state_change()
    assert result.verified is False
    assert result.issue == "oracle_unreachable"
    assert "HTTP 503" in result.suggestion


def test_state_change_passes_since_id_from_cursor():
    runner = _make_runner(oracle_session=_stub_session())
    runner._last_mutation_id = 42
    captured = {}

    def fake_fetch(url, token, *, since_id, **kw):
        captured["since_id"] = since_id
        return {"mutations": []}

    with patch(
        "mantis_agent.sim_envs.oracle_client.fetch_mutations",
        side_effect=fake_fetch,
    ):
        runner._oracle_verify_state_change()
    assert captured["since_id"] == 42


def test_state_change_cursor_monotonic_under_out_of_order_responses():
    """A response with id < current cursor should NOT regress the cursor."""
    runner = _make_runner(oracle_session=_stub_session())
    runner._last_mutation_id = 10
    fake = {"mutations": [{"id": 5, "operation": "consent_set"}]}
    with patch(
        "mantis_agent.sim_envs.oracle_client.fetch_mutations",
        return_value=fake,
    ):
        runner._oracle_verify_state_change()
    assert runner._last_mutation_id == 10


# ── _sync_mutation_cursor ──────────────────────────────────────────────


def test_sync_cursor_advances_past_current_tail():
    runner = _make_runner(oracle_session=_stub_session())
    fake = {"mutations": [
        {"id": 1, "operation": "env_reset"},
        {"id": 2, "operation": "consent_set"},
        {"id": 3, "operation": "env_reset"},
    ]}
    with patch(
        "mantis_agent.sim_envs.oracle_client.fetch_mutations",
        return_value=fake,
    ):
        runner._sync_mutation_cursor()
    assert runner._last_mutation_id == 3


def test_sync_cursor_noop_without_session():
    runner = _make_runner(oracle_session=None)
    runner._last_mutation_id = 5
    runner._sync_mutation_cursor()
    assert runner._last_mutation_id == 5


def test_sync_cursor_noop_on_empty_log():
    runner = _make_runner(oracle_session=_stub_session())
    with patch(
        "mantis_agent.sim_envs.oracle_client.fetch_mutations",
        return_value={"mutations": []},
    ):
        runner._sync_mutation_cursor()
    assert runner._last_mutation_id == 0


# ── Constructor wiring ─────────────────────────────────────────────────


def test_oracle_session_defaults_to_none():
    """The new constructor param defaults to None — vision-only path
    is unchanged for callers that don't pass a session."""
    runner = LearningRunner(
        brain=SimpleNamespace(),
        env=SimpleNamespace(),
        verifier=SimpleNamespace(),
    )
    assert runner.oracle_session is None
    assert runner._last_mutation_id == 0


def test_oracle_session_accepts_env_session_shape():
    """The runner accepts anything with ``.url`` and ``.admin_token``."""
    session = _stub_session(url="http://env.test", token="tok")
    runner = LearningRunner(
        brain=SimpleNamespace(),
        env=SimpleNamespace(),
        verifier=SimpleNamespace(),
        oracle_session=session,
    )
    assert runner.oracle_session is session
