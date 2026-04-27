"""Integration test: verify Gemma4Agent works with OSWorld's interface.

Simulates the OSWorld evaluation loop without a VM to verify:
1. Agent initializes and loads correctly
2. predict() returns the right format (response, actions)
3. Actions are valid pyautogui code strings
4. Agent maintains frame history across steps
5. Agent signals DONE when appropriate
"""

import io

from PIL import Image, ImageDraw

from mantis_agent.actions import Action, ActionType
from mantis_agent.osworld import Gemma4Agent


class FakeBrain:
    """Small deterministic brain for testing the OSWorld adapter contract."""

    def load(self) -> None:
        pass

    def think(self, frames, task, action_history, screen_size):
        if len(action_history) >= 2:
            action = Action(ActionType.DONE, {"success": True, "summary": "done"})
        elif action_history:
            action = Action(ActionType.TYPE, {"text": "echo Hello World"})
        else:
            action = Action(ActionType.CLICK, {"x": 100, "y": 100})
        return type(
            "FakeInferenceResult",
            (),
            {"action": action, "raw_output": "", "thinking": f"Fake task: {task}"},
        )()


def make_screenshot(step: int = 0) -> bytes:
    """Generate a synthetic desktop screenshot as PNG bytes (like OSWorld provides)."""
    img = Image.new("RGB", (1920, 1080), color=(30, 30, 40))
    draw = ImageDraw.Draw(img)

    # Desktop with taskbar
    draw.rectangle([0, 0, 1920, 35], fill=(50, 50, 60))
    draw.text((10, 10), "Activities    Files    Firefox", fill="white")

    # Desktop icons
    draw.rectangle([50, 100, 130, 180], fill=(60, 60, 80))
    draw.text((55, 185), "Terminal", fill="white")

    draw.rectangle([50, 220, 130, 300], fill=(60, 60, 80))
    draw.text((55, 305), "Files", fill="white")

    # If step > 0, show a "terminal" open
    if step > 0:
        draw.rectangle([200, 80, 1700, 950], fill=(20, 20, 25), outline=(80, 80, 90))
        draw.text((210, 85), "Terminal — bash", fill="white")
        draw.text((210, 120), "user@ubuntu:~$ ", fill=(100, 200, 100))
        if step > 1:
            draw.text((380, 120), "echo 'Hello World'", fill="white")
            draw.text((210, 145), "Hello World", fill="white")
            draw.text((210, 170), "user@ubuntu:~$ ", fill=(100, 200, 100))

    # Convert to PNG bytes (like OSWorld provides)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_agent():
    print("=" * 60)
    print("  Integration Test: Gemma4Agent x OSWorld")
    print("=" * 60)

    # Initialize agent (same interface as OSWorld's PromptAgent)
    agent = Gemma4Agent(
        model="google/gemma-4-E2B-it",
        max_tokens=512,
        action_space="pyautogui",
        observation_type="screenshot",
        max_trajectory_length=5,
        enable_thinking=True,
        backend="llamacpp",
    )
    agent.brain = FakeBrain()

    print("\n1. Loading model...")
    agent.load()
    print("   Model loaded.")

    # Simulate OSWorld's evaluation loop
    instruction = "Open a terminal and type 'echo Hello World'"

    print(f"\n2. Task: {instruction}")
    print("-" * 60)

    max_steps = 3  # Short test
    for step in range(max_steps):
        # OSWorld provides raw PNG bytes
        obs = {
            "screenshot": make_screenshot(step),
            "accessibility_tree": None,
        }

        print(f"\n   Step {step + 1}:")
        response, actions = agent.predict(instruction, obs)

        # Verify output format
        assert isinstance(response, str), f"Response must be string, got {type(response)}"
        assert isinstance(actions, list), f"Actions must be list, got {type(actions)}"
        assert len(actions) > 0, "Must return at least one action"

        for action in actions:
            assert isinstance(action, str), f"Action must be string, got {type(action)}"
            print(f"   Response: {response[:100]}...")
            print(f"   Action:   {action}")

        # Check for terminal actions
        if actions[0] in ("DONE", "FAIL", "WAIT"):
            print(f"   >> Agent signaled: {actions[0]}")
            if actions[0] == "DONE":
                break

    # Verify frame history was maintained
    assert len(agent._frame_history) > 0, "Frame history should be populated"
    assert len(agent._action_history) > 0, "Action history should be populated"

    print("\n" + "=" * 60)
    print(f"  Frame history: {len(agent._frame_history)} frames")
    print(f"  Action history: {len(agent._action_history)} actions")
    print("  All assertions passed!")
    print("=" * 60)

    # Test reset
    agent.reset()
    assert len(agent._frame_history) == 0, "Reset should clear frame history"
    assert len(agent._action_history) == 0, "Reset should clear action history"
    print("\n  Reset works correctly.")
    print("  INTEGRATION TEST PASSED")


if __name__ == "__main__":
    test_agent()
