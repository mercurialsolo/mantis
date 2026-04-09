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
    """Wrap sudo invocations for non-interactive execution.

    Handles three cases:
      1. ``sudo COMMAND``                            -> ``echo 'password' | sudo -S COMMAND``
      2. ``A | sudo COMMAND``                        -> still wrap the sudo with -S (NOT just
                                                        the leading echo, since A might be
                                                        feeding stdin to COMMAND, e.g.
                                                        ``echo 'user:pw' | sudo chpasswd``).
                                                        We use ``sudo -S -A`` no — instead
                                                        we use ``sudo -n`` after a one-shot
                                                        ``sudo -v`` priming, but the simplest
                                                        reliable form is to use the
                                                        ``SUDO_ASKPASS`` env var with a script
                                                        that prints the password. We avoid
                                                        that complexity here by inserting
                                                        ``echo 'password' |`` into a heredoc.
      3. Already wrapped (contains ``-S`` or matches ``echo ... | sudo``) -> leave alone.

    Also walks through ``&&`` / ``;`` separated sub-commands so each ``sudo`` gets handled
    independently.
    """
    if "sudo" not in cmd:
        return cmd

    # Strategy: ensure passwordless sudo for the lifetime of this command.
    # Run a one-shot ``sudo -v`` with the password piped in, then chain the
    # original command. After ``sudo -v`` succeeds, subsequent sudos within
    # the same session don't need a password for ~5 minutes (timestamp_timeout).
    # This handles ALL cases: leading sudo, piped sudo, multiple sudos,
    # sudo inside command substitution, etc.
    if "-S" in cmd or "SUDO_ASKPASS" in cmd:
        return cmd  # already self-sufficient
    return "echo 'password' | sudo -S -v && " + cmd

def _fix_find_exec(cmd):
    """Auto-repair malformed find -exec commands AND teach the correction.

    Handles two common mistakes:
      1. Trailing bare backslash: ``find ... -exec cp {} dest/ \\``
         (model meant ``\\;`` but wrote line-continuation)
      2. No terminator at all: ``find ... -exec cp {} dest/``
         (model forgot the terminator entirely)

    In both cases, find emits "missing argument to '-exec'" or hangs. We
    append the proper ``\\;`` and print a teaching note so the model learns
    the right syntax via the captured-output feedback loop.
    """
    if "-exec" not in cmd:
        return cmd, ""
    stripped = cmd.rstrip()
    # NOTE: helpers run as a triple-quoted prelude, so each backslash here is
    # written as four chars in the outer source ("\\\\") to become two chars
    # in the loaded code ("\\") which represent a single literal backslash.

    # Case 1: trailing bare backslash (line continuation)
    if stripped.endswith("\\\\") and not stripped.endswith("\\\\;") and not stripped.endswith("\\\\+"):
        fixed = stripped + ";"
        note = (
            "Note: auto-corrected your find -exec command — added the missing "
            "semicolon terminator. find -exec must end with \\\\; (backslash + "
            "semicolon), not just a bare trailing backslash. Remember this."
        )
        return fixed, note

    # Case 2: no terminator at all. Detect by checking that the command
    # contains -exec but does NOT contain \; or + or \\; anywhere after -exec.
    exec_idx = stripped.find("-exec")
    after_exec = stripped[exec_idx:]
    if "\\\\;" not in after_exec and " ;" not in after_exec and after_exec.rstrip()[-1] != "+":
        fixed = stripped + " \\\\;"
        note = (
            "Note: auto-corrected your find -exec command — appended the "
            "missing \\\\; terminator. find -exec MUST end with \\\\; or + "
            "or it fails with 'missing argument to -exec'. Remember this."
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
