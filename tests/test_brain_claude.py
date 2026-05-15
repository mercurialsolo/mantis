from __future__ import annotations

from PIL import Image

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


# ── #435 item 5: image stand-ins for older frames ────────────────────


def _img() -> Image.Image:
    return Image.new("RGB", (10, 10), color=(255, 255, 255))


def test_claude_brain_keeps_recent_frames_as_images() -> None:
    """The most recent _FRAMES_KEEP_AS_IMAGE frames are sent as raw
    image bytes (the default is 2 per cua_notes.md §1's 1-3 sweet spot).
    """
    brain = ClaudeBrain(api_key="k")
    messages = brain._build_messages(
        frames=[_img(), _img()], task="task",
        action_history=None, screen_size=(10, 10),
    )
    content = messages[0]["content"]
    image_blocks = [b for b in content if b.get("type") == "image"]
    assert len(image_blocks) == 2


def test_claude_brain_drops_older_frames_to_text_placeholders() -> None:
    """When >2 frames are passed, the older ones become
    ``[screenshot omitted — text-only at this distance]`` placeholders
    instead of raw image bytes. Position markers (``[Frame t-N]``) stay
    so the model can still reason about turn ordering.
    """
    brain = ClaudeBrain(api_key="k")
    messages = brain._build_messages(
        frames=[_img(), _img(), _img(), _img(), _img()],
        task="task", action_history=None, screen_size=(10, 10),
    )
    content = messages[0]["content"]
    # 2 images for the last 2 frames; 3 placeholder texts for the older 3.
    image_blocks = [b for b in content if b.get("type") == "image"]
    placeholder_blocks = [
        b for b in content
        if b.get("type") == "text"
        and "screenshot omitted" in b.get("text", "")
    ]
    # Position markers still present for all 5 frames.
    frame_markers = [
        b for b in content
        if b.get("type") == "text"
        and b.get("text", "").startswith("[Frame ")
    ]
    assert len(image_blocks) == 2
    assert len(placeholder_blocks) == 3
    assert len(frame_markers) == 5


def test_claude_brain_uses_current_beta_headers() -> None:
    """#435 item 6: brain_claude.py uses the current
    ``computer-use-2025-11-24`` + ``context-management-2025-06-27``
    betas. The headers must keep them in sync with the tool type
    (paired in ``_build_tools``).
    """
    brain = ClaudeBrain(api_key="k")
    headers = brain._headers()
    beta = headers["anthropic-beta"]
    assert "computer-use-2025-11-24" in beta
    assert "context-management-2025-06-27" in beta
    # Regression: the previous beta is replaced, not just appended.
    assert "computer-use-2025-01-24" not in beta


def test_claude_brain_tool_type_matches_beta() -> None:
    """The tool ``type`` must match the beta version — Anthropic
    rejects mismatched pairs. ``computer_20251124`` pairs with
    ``computer-use-2025-11-24``.
    """
    brain = ClaudeBrain(api_key="k")
    tools = brain._build_tools((1280, 800))
    computer_tool = next(t for t in tools if t.get("name") == "computer")
    assert computer_tool["type"] == "computer_20251124"


def test_claude_brain_marks_current_frame_distinctly() -> None:
    """The newest frame is labelled CURRENT, older ones t-N."""
    brain = ClaudeBrain(api_key="k")
    messages = brain._build_messages(
        frames=[_img(), _img(), _img()], task="task",
        action_history=None, screen_size=(10, 10),
    )
    text_markers = [
        b["text"] for b in messages[0]["content"]
        if b.get("type") == "text" and b["text"].startswith("[Frame ")
    ]
    assert text_markers == ["[Frame t-2]", "[Frame t-1]", "[Frame CURRENT]"]
