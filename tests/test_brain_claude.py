from __future__ import annotations

from mantis_agent.actions import ActionType
from mantis_agent.brain_claude import ClaudeBrain
from mantis_agent.gym.claude_director import _decision_to_action


def test_claude_key_action_accepts_text_payload() -> None:
    action = ClaudeBrain._parse_computer_action({"action": "key", "text": "Tab"})

    assert action.action_type == ActionType.KEY_PRESS
    assert action.params == {"keys": "tab"}


def test_claude_computer_wait_action_is_supported() -> None:
    action = ClaudeBrain._parse_computer_action({"action": "wait", "seconds": 15})

    assert action.action_type == ActionType.WAIT
    assert action.params == {"seconds": 15}


def test_claude_empty_key_action_infers_tab_from_reasoning() -> None:
    brain = ClaudeBrain(api_key="test-key")
    result = brain._parse_response({
        "content": [
            {
                "type": "thinking",
                "thinking": "The username is filled. Let me use the Tab key to move to the password field.",
            },
            {
                "type": "tool_use",
                "name": "computer",
                "input": {"action": "key"},
            },
        ],
    })

    assert result.action.action_type == ActionType.KEY_PRESS
    assert result.action.params == {"keys": "tab"}


def test_claude_empty_key_action_without_hint_becomes_wait() -> None:
    brain = ClaudeBrain(api_key="test-key")
    result = brain._parse_response({
        "content": [
            {"type": "thinking", "thinking": "I need a different approach."},
            {
                "type": "tool_use",
                "name": "computer",
                "input": {"action": "key"},
            },
        ],
    })

    assert result.action.action_type == ActionType.WAIT
    assert result.action.params == {"seconds": 1.0}


def test_claude_director_rejects_empty_keypress() -> None:
    assert _decision_to_action({"action_type": "key_press", "keys": ""}) is None
