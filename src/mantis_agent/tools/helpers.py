"""Helper functions provided to the agent as a Python prelude.

These helpers run *inside* the OSWorld VM via the python service at port 5050.
They are prepended to every action the model emits, so the model can call them
as if they were already imported.

Implementation choice: pyautogui is used for click/key/scroll because it's the
only library reliably available in the OSWorld VM. xdotool is NOT installed.
For ``type_text``, pyautogui's character-by-character typing mangles special
chars, so we prefer a clipboard-paste path (xclip or pyperclip) and fall back
to ``pyautogui.write`` only as a last resort.

Why scaling lives here?
- The model receives 1280x720 screenshots but the actual screen is 1920x1080.
- Putting the scale factor in the helper means a single source of truth — no
  regex rewriting on action strings, no coupling to specific function names.
"""

from __future__ import annotations


HELPERS_PRELUDE = '''
import subprocess as _sp
import time as _time

# Coordinate scale: model coords (1280x720) -> screen coords (1920x1080)
_SCALE_X = 1920 / 1280  # 1.5
_SCALE_Y = 1080 / 720   # 1.5

# pyautogui is the only mouse/keyboard library reliably available in OSWorld VMs.
# Import lazily so import failures surface clearly in feedback.
def _pg():
    import pyautogui as _p
    _p.FAILSAFE = False
    return _p

def _xy(x, y):
    return int(round(float(x) * _SCALE_X)), int(round(float(y) * _SCALE_Y))

def _autosudo(cmd):
    """Wrap a sudo command for non-interactive execution."""
    s = cmd.lstrip()
    if s.startswith("sudo ") and "-S" not in s and "| sudo" not in s and "echo " not in s.split("sudo")[0]:
        return "echo 'password' | sudo -S " + s[5:]
    return cmd

def _fix_find_exec(cmd):
    """Auto-repair malformed find -exec commands AND teach the correction.

    Models often produce a trailing backslash (line continuation) at the end
    of a find -exec command when they mean to terminate with backslash-semicolon.
    The trailing backslash makes bash wait for more input and the find never runs.

    Returns (corrected_cmd, teaching_note). The note is printed to stdout so
    the captured-output feedback loop teaches the model the right syntax for
    future commands in the same session.
    """
    if "-exec" not in cmd:
        return cmd, ""
    stripped = cmd.rstrip()
    # NOTE: helpers run as a triple-quoted prelude, so each backslash here is
    # written as four chars in the outer source ("\\\\") to become two chars
    # in the loaded code ("\\") which represent a single literal backslash.
    if stripped.endswith("\\\\") and not stripped.endswith("\\\\;") and not stripped.endswith("\\\\+"):
        fixed = stripped + ";"
        note = (
            "Note: auto-corrected your find -exec command — added the missing "
            "semicolon terminator. find -exec must end with backslash-semicolon "
            "(escaped as \\\\; in shell), not just a bare trailing backslash. "
            "Remember this for future find commands."
        )
        return fixed, note
    return cmd, ""

def click(x, y, button="left"):
    """Click at (x, y) — coordinates are in the screenshot's pixel space."""
    rx, ry = _xy(x, y)
    try:
        _pg().click(rx, ry, button=button)
    except Exception as e:
        print(f"click({x},{y}) failed: {type(e).__name__}: {e}")

def double_click(x, y):
    """Double click at (x, y)."""
    rx, ry = _xy(x, y)
    try:
        _pg().doubleClick(rx, ry)
    except Exception as e:
        print(f"double_click({x},{y}) failed: {type(e).__name__}: {e}")

def right_click(x, y):
    """Right click at (x, y)."""
    click(x, y, button="right")

def _set_clipboard(text):
    """Set the X11 clipboard to text. Try xclip then xsel. Returns True on success."""
    for tool, args in (("xclip", ["xclip", "-selection", "clipboard"]),
                       ("xsel",  ["xsel",  "--clipboard", "--input"])):
        try:
            r = _sp.run(args, input=text, text=True, capture_output=True, timeout=5)
            if r.returncode == 0:
                return True
        except FileNotFoundError:
            continue
        except Exception:
            continue
    return False

def type_text(text):
    """Type any text into the focused element. Handles ALL special characters.

    Strategy: copy to clipboard then paste with Ctrl+V. Falls back to
    pyautogui.write only if no clipboard tool is available.
    """
    if not isinstance(text, str):
        text = str(text)
    if _set_clipboard(text):
        try:
            _pg().hotkey("ctrl", "v")
            return
        except Exception as e:
            print(f"type_text paste failed: {type(e).__name__}: {e}")
    # Last-resort fallback (mangles special chars but typed something is better than nothing)
    try:
        _pg().write(text, interval=0.01)
    except Exception as e:
        print(f"type_text fallback write failed: {type(e).__name__}: {e}")

def _normalize_key(combo):
    """pyautogui wants lowercase letter keys (e.g. 't', not 'T'). F-keys and named
    keys (Return, Tab, etc.) are kept as-is.
    """
    parts = combo.split("+")
    out = []
    for p in parts:
        s = p.strip()
        if len(s) == 1 and s.isalpha() and s.isupper():
            out.append(s.lower())
        else:
            out.append(s)
    return out

def key(combo):
    """Press a key or combo. Examples: key('enter'), key('ctrl+c'), key('alt+f4').

    Letter keys are auto-lowercased. Named keys (Return, Tab, F1) are preserved.
    """
    try:
        parts = _normalize_key(str(combo))
        if len(parts) == 1:
            _pg().press(parts[0])
        else:
            _pg().hotkey(*parts)
    except Exception as e:
        print(f"key({combo!r}) failed: {type(e).__name__}: {e}")

def hotkey(*keys):
    """Press multiple keys as a combo. Example: hotkey('ctrl', 'shift', 't')."""
    try:
        parts = _normalize_key("+".join(str(k) for k in keys))
        _pg().hotkey(*parts)
    except Exception as e:
        print(f"hotkey({keys!r}) failed: {type(e).__name__}: {e}")

def run_command(cmd, timeout=120):
    """Run a shell command via subprocess and PRINT the output.

    PREFERRED for shell commands. Blocks until done, prints stdout/stderr.
    Sudo is auto-wrapped. Common shell mistakes (like missing find -exec
    terminators) are auto-corrected AND the model is told what was fixed
    so it learns the right syntax for next time.
    """
    if not isinstance(cmd, str):
        cmd = str(cmd)
    cmd, find_note = _fix_find_exec(cmd)
    cmd = _autosudo(cmd)
    try:
        result = _sp.run(["bash", "-lc", cmd], capture_output=True, text=True, timeout=timeout)
        out = (result.stdout or "") + (result.stderr or "")
        if find_note:
            print(find_note)
        print(f"$ {cmd}")
        if out.strip():
            print(out.strip())
        print(f"[exit {result.returncode}]")
        return result
    except _sp.TimeoutExpired:
        if find_note:
            print(find_note)
        print(f"$ {cmd}")
        print(f"[TIMEOUT after {timeout}s]")
        return None

def run_terminal(cmd):
    """Open a terminal WINDOW and type a command into it (slow, GUI-context).

    Use ``run_command`` for almost all shell commands. ``run_terminal`` only when
    you specifically need a visible interactive terminal window.
    """
    if not isinstance(cmd, str):
        cmd = str(cmd)
    cmd = _autosudo(cmd)
    try:
        _pg().hotkey("ctrl", "alt", "t")
    except Exception as e:
        print(f"run_terminal: failed to open terminal: {e}")
        return
    _time.sleep(1.5)
    type_text(cmd)
    _time.sleep(0.2)
    try:
        _pg().press("enter")
    except Exception:
        pass

def shell(cmd):
    """Type a shell command into a terminal that's already open and press Enter."""
    if not isinstance(cmd, str):
        cmd = str(cmd)
    cmd = _autosudo(cmd)
    type_text(cmd)
    _time.sleep(0.2)
    try:
        _pg().press("enter")
    except Exception:
        pass

def wait(seconds=1):
    """Pause and let the screen settle."""
    _time.sleep(float(seconds))

def scroll(amount, x=None, y=None):
    """Scroll. Positive=up, negative=down. Optional x,y to scroll at a position."""
    try:
        amount = int(amount)
        p = _pg()
        if x is not None and y is not None:
            rx, ry = _xy(x, y)
            p.scroll(amount, x=rx, y=ry)
        else:
            p.scroll(amount)
    except Exception as e:
        print(f"scroll({amount}) failed: {type(e).__name__}: {e}")
'''


def wrap_action(code: str) -> str:
    """Prepend the helpers prelude to a model-generated action snippet."""
    return HELPERS_PRELUDE + "\n" + code
