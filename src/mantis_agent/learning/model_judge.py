"""Model-as-judge outcome grader (#906) — RLAIF reward for oracle-less tasks.

For sim-env tasks the env oracle is ground truth (see
``run_executor._apply_oracle_reward``). But open-ended web tasks have **no
programmatic oracle** — there a model judge is the only outcome signal, and it
also densifies reward for credit assignment in long CUA episodes.

This module is the judge itself: given a task instruction + the run's final
screenshot, a Claude model (a **different family than the Holo3 actor** — a
guardrail against the self-grading collusion #906 warns about) returns a
pass/fail verdict + confidence. The runtime writes it to Augur as a
``comparator="model-judge"`` verdict (populating the ``rm_outcome`` reward term)
plus a ``judge_type="model"`` decision for provenance — kept **below** the oracle
where both exist, and never promoted to the operative verdict (the
champion/challenger gate stays oracle-only).

Pure + injectable: the HTTP call is the only side effect and is behind the
shared Anthropic retry client, so the verdict logic is unit-testable with a
stub ``post_fn``.
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)

# A capable vision model from a DIFFERENT family than the Holo3 actor — judging
# Holo3 with Holo3 shares blind spots and colludes (#906 guardrail).
DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"

_PROMPT = (
    "You are an impartial evaluator grading whether a computer-use agent "
    "ACCOMPLISHED a task, judging ONLY from the final screenshot of the "
    "browser after the agent finished.\n\n"
    "Task the agent was asked to do:\n\"{instruction}\"\n\n"
    "Decide if the final state shows the task was completed successfully. Be "
    "strict: partial progress, an open form that was never submitted, or an "
    "error/blank page is NOT success. Respond with ONLY a JSON object, no prose:\n"
    '{{"passed": <true|false>, "confidence": <0.0-1.0>, "reason": "<one sentence>"}}'
)


@dataclass(frozen=True)
class JudgeVerdict:
    passed: bool
    confidence: float
    reason: str

    @property
    def status(self) -> str:
        return "passed" if self.passed else "failed"

    @property
    def score(self) -> float:
        """Continuous reward in [0,1]: confidence toward the verdict direction.

        passed → confidence; failed → 0. Keeps the reward monotone in
        "evidence the task succeeded" so the rm_outcome term behaves."""
        return max(0.0, min(1.0, self.confidence)) if self.passed else 0.0


# (payload: dict) -> response-with .status_code/.json(), mirroring the
# AnthropicToolUseClient.post_messages_with_retry seam.
PostFn = Callable[[dict[str, Any]], Any]


def _default_post_fn(model: str) -> PostFn:
    """Lazily build the shared Anthropic retry client on first call (so a judge
    constructed without a network call — e.g. for ``judge_id`` — never needs a
    key, and tests can inject a stub ``post_fn``)."""
    client_box: list[Any] = []

    def _post(payload: dict[str, Any]) -> Any:
        if not client_box:
            import os
            from .._anthropic.client import AnthropicToolUseClient
            client_box.append(AnthropicToolUseClient(
                api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
                model=model, log_prefix="[model-judge]",
            ))
        return client_box[0].post_messages_with_retry(payload, timeout=60)

    return _post


def _parse(text: str) -> JudgeVerdict:
    """Extract the JSON verdict from the model's reply (tolerant of fences)."""
    s = text.strip()
    if "{" in s and "}" in s:
        s = s[s.index("{"): s.rindex("}") + 1]
    obj = json.loads(s)
    return JudgeVerdict(
        passed=bool(obj.get("passed")),
        confidence=float(obj.get("confidence", 0.0) or 0.0),
        reason=str(obj.get("reason", "") or "")[:300],
    )


class ModelJudge:
    """Grades a finished run from its final screenshot (#906)."""

    def __init__(
        self, *, model: str = DEFAULT_JUDGE_MODEL,
        max_tokens: int = 256, post_fn: PostFn | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._post = post_fn or _default_post_fn(model)

    @property
    def judge_id(self) -> str:
        return f"model-judge:{self.model}"

    def judge(self, instruction: str, screenshot_png: bytes) -> JudgeVerdict | None:
        """Return the model's verdict, or ``None`` on any failure (telemetry
        never breaks the run — the caller falls back to no judge signal)."""
        if not screenshot_png:
            return None
        b64 = base64.b64encode(screenshot_png).decode("utf-8")
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/png", "data": b64}},
                    {"type": "text", "text": _PROMPT.format(instruction=instruction)},
                ],
            }],
        }
        try:
            resp = self._post(payload)
            if resp is None or getattr(resp, "status_code", 0) != 200:
                logger.warning("[model-judge] call failed (%s)",
                               getattr(resp, "status_code", "no-response"))
                return None
            for block in resp.json().get("content", []):
                if block.get("type") == "text":
                    return _parse(block["text"])
            return None
        except Exception as exc:  # noqa: BLE001 — best-effort; never propagate
            logger.warning("[model-judge] verdict failed: %s", exc)
            return None
