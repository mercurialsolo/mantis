from __future__ import annotations

from mantis_agent.gym.workflow_runner import LoopConfig, WorkflowRunner


class _Env:
    current_url = "https://www.boattrader.com/boats/state-fl/city-miami/zip-33101/"


def test_workflow_runner_anchors_recovery_to_current_url_when_start_url_missing() -> None:
    runner = WorkflowRunner(
        brain=object(),
        env=_Env(),
        loop_config=LoopConfig(
            iteration_intent="Process the next listing.",
            pagination_intent="Click Next.",
        ),
        start_url="",
    )

    assert runner.start_url == "https://www.boattrader.com/boats/state-fl/city-miami/zip-33101/"
