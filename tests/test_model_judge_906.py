"""#906 extension — model-as-judge outcome grader (RLAIF reward for oracle-less tasks).

Pins the verdict parsing + the inject-a-stub-post_fn contract so the judge logic
is testable with no Anthropic call.
"""

from __future__ import annotations

from mantis_agent.learning.model_judge import (
    DEFAULT_JUDGE_MODEL,
    JudgeVerdict,
    ModelJudge,
)


class _Resp:
    def __init__(self, text, status=200):
        self.status_code = status
        self._text = text

    def json(self):
        return {"content": [{"type": "text", "text": self._text}]}


def test_default_model_is_not_holo3():
    # guardrail: judge must be a different family than the Holo3 actor.
    assert "claude" in DEFAULT_JUDGE_MODEL.lower()
    assert "holo" not in DEFAULT_JUDGE_MODEL.lower()


def test_verdict_score_monotone():
    assert JudgeVerdict(True, 0.9, "ok").score == 0.9
    assert JudgeVerdict(False, 0.9, "no").score == 0.0  # fail → 0 regardless of conf
    assert JudgeVerdict(True, 0.0, "ok").status == "passed"


def test_judge_parses_pass_verdict():
    j = ModelJudge(post_fn=lambda p: _Resp('{"passed": true, "confidence": 0.82, "reason": "saved"}'))
    v = j.judge("save job_00007", b"\x89PNG fake")
    assert v.passed is True and v.confidence == 0.82 and v.reason == "saved"
    assert v.score == 0.82


def test_judge_parses_fail_with_fenced_json():
    j = ModelJudge(post_fn=lambda p: _Resp('```json\n{"passed": false, "confidence": 0.7, "reason": "form open"}\n```'))
    v = j.judge("apply to job", b"png")
    assert v.passed is False and v.score == 0.0


def test_judge_none_on_empty_screenshot():
    j = ModelJudge(post_fn=lambda p: _Resp("{}"))
    assert j.judge("x", b"") is None


def test_judge_none_on_non_200():
    j = ModelJudge(post_fn=lambda p: _Resp("err", status=500))
    assert j.judge("x", b"png") is None


def test_judge_none_on_post_exception():
    def boom(_p):
        raise RuntimeError("network")
    assert ModelJudge(post_fn=boom).judge("x", b"png") is None


def test_judge_id_format():
    assert ModelJudge(model="claude-sonnet-4-6").judge_id == "model-judge:claude-sonnet-4-6"


def test_sends_image_and_instruction():
    seen = {}

    def cap(payload):
        seen["payload"] = payload
        return _Resp('{"passed": true, "confidence": 1.0, "reason": "ok"}')

    ModelJudge(post_fn=cap).judge("MY TASK", b"PNGBYTES")
    content = seen["payload"]["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert "MY TASK" in content[1]["text"]
