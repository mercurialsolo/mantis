"""#906 — producer-side oracle reward: _apply_oracle_reward stamps the
ground-truth env-oracle verdict onto the terminal step so Augur's reward
reflects what actually happened (not the agent's self-verifier).

Pins:
1. no oracle config → returns None (caller falls back to plan-completion).
2. oracle PASS → set_score(terminal, 1.0, comparator="verifier") + returns True.
3. oracle FAIL → set_score(terminal, 0.0, ...) + returns False.
4. oracle HTTP error → returns None (telemetry never breaks the run).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from mantis_agent.gym.run_executor import RunExecutor


def _executor(runner):
    ex = object.__new__(RunExecutor)  # bypass __init__; we only exercise one method
    ex.parent = runner
    return ex


def _runner(**oracle):
    augur = MagicMock()
    r = SimpleNamespace(_augur=augur, **oracle)
    return r, augur


def test_no_oracle_config_returns_none():
    r, augur = _runner()  # no _oracle_url / _oracle_task_id
    assert _executor(r)._apply_oracle_reward([1, 2, 3]) is None
    augur.set_score.assert_not_called()


def test_oracle_pass_stamps_verifier_score_1():
    r, augur = _runner(
        _oracle_url="https://env.example/", _oracle_task_id="t01",
        _oracle_admin_token="adm", _oracle_preview_token="pv",
    )
    resp = MagicMock()
    resp.json.return_value = {"passed": True}
    with patch("requests.get", return_value=resp) as g:
        out = _executor(r)._apply_oracle_reward(["s0", "s1", "s2"])
    assert out is True
    # terminal step is index 2 (len-1); score 1.0; comparator verifier
    args, kwargs = augur.set_score.call_args
    assert args[0] == 2 and args[1] == 1.0
    assert kwargs["comparator"] == "verifier"
    # oracle_pass + process/progress shaping components (so all-pass/all-fail
    # groups still vary by effort/progress — the GRPO degeneracy fix).
    comps = kwargs["components"]
    assert comps["oracle_pass"] == 1.0
    assert comps["progress"] == 1.0  # passed → full progress
    assert 0.0 <= comps["process"] <= 1.0  # efficiency in [0,1]
    # graded the right task against /__env__/oracle
    assert g.call_args[0][0] == "https://env.example/__env__/oracle"
    assert g.call_args[1]["params"] == {"task_id": "t01"}


def test_oracle_fail_stamps_score_0():
    r, augur = _runner(_oracle_url="https://e/", _oracle_task_id="t01")
    resp = MagicMock()
    resp.json.return_value = {"passed": False}
    with patch("requests.get", return_value=resp):
        out = _executor(r)._apply_oracle_reward(["only"])
    assert out is False
    args, kwargs = augur.set_score.call_args
    assert args[0] == 0 and args[1] == 0.0  # single step → terminal index 0


def test_oracle_http_error_returns_none():
    r, augur = _runner(_oracle_url="https://e/", _oracle_task_id="t01")
    with patch("requests.get", side_effect=RuntimeError("boom")):
        assert _executor(r)._apply_oracle_reward(["s0"]) is None
    augur.set_score.assert_not_called()
