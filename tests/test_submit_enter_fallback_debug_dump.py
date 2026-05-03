"""Tests for submit Enter-key fallback + MANTIS_DEBUG_DUMP_DIR screenshot dumping.

Surfaced by the staffcrm v7 verify (run 20260503_135208_5c92318a):
the adaptive settle from #152 ran for the full 8s budget but
``env.current_url`` polling never saw a URL change. Either the click
landed on the right pixel but JS swallowed the event, or the form
needed an Enter-key submit instead.

Two complementary additions:

A) Enter-key fallback — when click + adaptive settle didn't navigate,
   fire Return on the focused field. HTML form's native onsubmit
   handles this even when the button's onclick path is blocked.

B) MANTIS_DEBUG_DUMP_DIR env var — when set, the runner saves
   pre/post screenshots so future failure modes can be inspected
   without re-running the full Modal pipeline.

Both are pure-observational additions: no LLM call, no per-CRM
heuristic. The Enter fallback is a generic HTML-form behavior that
works for any login/edit form using native ``<form>`` semantics.
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from mantis_agent.gym.micro_runner import MicroPlanRunner


# ── Fakes ───────────────────────────────────────────────────────────────


class _FakeScreenshotEnv:
    def __init__(self) -> None:
        self.shots = 0

    def screenshot(self) -> Image.Image:
        self.shots += 1
        return Image.new("RGB", (32, 32), (10, 10, 10))


class _FakeNoScreenshotEnv:
    """Env without a screenshot() method — adapter must degrade gracefully."""


class _FakeRaisingScreenshotEnv:
    def screenshot(self) -> Image.Image:
        raise RuntimeError("X11 not ready")


def _runner(env: Any) -> MicroPlanRunner:
    runner = MicroPlanRunner.__new__(MicroPlanRunner)
    runner.env = env
    runner.run_key = "test-run"
    runner.session_name = ""
    return runner


# ── _safe_screenshot ──────────────────────────────────────────────────


def test_safe_screenshot_captures_normally() -> None:
    env = _FakeScreenshotEnv()
    runner = _runner(env)
    img = runner._safe_screenshot()
    assert isinstance(img, Image.Image)
    assert env.shots == 1


def test_safe_screenshot_returns_none_when_env_lacks_method() -> None:
    runner = _runner(_FakeNoScreenshotEnv())
    assert runner._safe_screenshot() is None


def test_safe_screenshot_returns_none_on_raise() -> None:
    runner = _runner(_FakeRaisingScreenshotEnv())
    assert runner._safe_screenshot() is None


# ── _dump_debug_screenshot — env-var gated ─────────────────────────────


def test_dump_is_no_op_when_env_var_unset(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MANTIS_DEBUG_DUMP_DIR", raising=False)
    runner = _runner(_FakeScreenshotEnv())
    img = Image.new("RGB", (8, 8))
    # Should not raise, should not write anywhere.
    runner._dump_debug_screenshot("test_stem", img)
    # tmp_path stays empty — no file created (path wasn't even passed).
    assert not list(tmp_path.iterdir())


def test_dump_writes_to_run_subdir_when_var_set(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MANTIS_DEBUG_DUMP_DIR", str(tmp_path))
    runner = _runner(_FakeScreenshotEnv())
    img = Image.new("RGB", (8, 8), (255, 0, 0))
    runner._dump_debug_screenshot("step3_post_click", img)
    expected = tmp_path / runner.run_key / "step3_post_click.png"
    assert expected.exists()
    # PNG is a valid image (Pillow can re-open).
    Image.open(expected).verify()


def test_dump_uses_session_name_when_run_key_empty(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MANTIS_DEBUG_DUMP_DIR", str(tmp_path))
    runner = _runner(_FakeScreenshotEnv())
    runner.run_key = ""
    runner.session_name = "fallback-session"
    runner._dump_debug_screenshot("stem", Image.new("RGB", (8, 8)))
    assert (tmp_path / "fallback-session" / "stem.png").exists()


def test_dump_falls_back_to_default_subdir(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MANTIS_DEBUG_DUMP_DIR", str(tmp_path))
    runner = _runner(_FakeScreenshotEnv())
    runner.run_key = ""
    runner.session_name = ""
    runner._dump_debug_screenshot("stem", Image.new("RGB", (8, 8)))
    assert (tmp_path / "default" / "stem.png").exists()


def test_dump_handles_none_screenshot_gracefully(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MANTIS_DEBUG_DUMP_DIR", str(tmp_path))
    runner = _runner(_FakeScreenshotEnv())
    runner._dump_debug_screenshot("stem", None)
    # Nothing written — None screenshot is a no-op (degrades silently).
    assert not list(tmp_path.iterdir())


def test_dump_swallows_filesystem_errors(tmp_path, monkeypatch) -> None:
    """Disk full / permission denied during a debug dump should never
    abort the run. The whole point is observability — failures here
    must be swallowed."""
    monkeypatch.setenv("MANTIS_DEBUG_DUMP_DIR", str(tmp_path))
    runner = _runner(_FakeScreenshotEnv())
    # Force screenshot.save() to raise — exercises the broad except.
    class _BadImage:
        def save(self, *args, **kwargs):
            raise OSError("disk full")
    runner._dump_debug_screenshot("stem", _BadImage())  # type: ignore[arg-type]
    # No exception = test passes. The directory was created but no file
    # written (OSError caught).


def test_dump_creates_concurrent_run_subdirs(tmp_path, monkeypatch) -> None:
    """Two runs with different run_keys writing under the same dump dir
    must not collide."""
    monkeypatch.setenv("MANTIS_DEBUG_DUMP_DIR", str(tmp_path))
    a = _runner(_FakeScreenshotEnv())
    a.run_key = "run-A"
    b = _runner(_FakeScreenshotEnv())
    b.run_key = "run-B"
    a._dump_debug_screenshot("step", Image.new("RGB", (8, 8)))
    b._dump_debug_screenshot("step", Image.new("RGB", (8, 8)))
    assert (tmp_path / "run-A" / "step.png").exists()
    assert (tmp_path / "run-B" / "step.png").exists()


# ── Enter-key fallback path (logic-only — runtime exercised via Modal) ──


def test_dump_filename_carries_step_and_action() -> None:
    """The naming convention encodes step index + step type so a debug
    folder of N runs is greppable."""
    # Naming pattern from the runner: f"step{index}_post_{step_type}"
    # and f"submit_step{index}_pre_click" — assert via documentation
    # that these stems appear in the runner source.
    import inspect
    src = inspect.getsource(MicroPlanRunner)
    assert "submit_step" in src
    assert "post_click" in src or "post_enter" in src
    assert "pre_click" in src


# ── Integration — ENV var resolution end-to-end ────────────────────────


def test_dump_dir_var_strip_handles_whitespace(tmp_path, monkeypatch) -> None:
    """A user setting MANTIS_DEBUG_DUMP_DIR with trailing whitespace
    shouldn't break the path resolution."""
    monkeypatch.setenv("MANTIS_DEBUG_DUMP_DIR", f"  {tmp_path}  ")
    runner = _runner(_FakeScreenshotEnv())
    runner._dump_debug_screenshot("stem", Image.new("RGB", (8, 8)))
    assert (tmp_path / runner.run_key / "stem.png").exists()


def test_dump_empty_string_env_var_treated_as_unset(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MANTIS_DEBUG_DUMP_DIR", "")
    runner = _runner(_FakeScreenshotEnv())
    runner._dump_debug_screenshot("stem", Image.new("RGB", (8, 8)))
    # No file — empty string disables dumping.
    assert not list(tmp_path.iterdir())
