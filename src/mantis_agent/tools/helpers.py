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
    """Wrap a leading sudo command for non-interactive execution.

    Only wraps the simple case of a command starting with ``sudo ``. More
    complex cases (piped sudo, chained sudo, multi-sudo) rely on the
    passwordless-sudo setup that the harness runs once at task start
    (see ``_setup_passwordless_sudo`` in deploy/modal/modal_osworld_direct.py). After
    that setup, every ``sudo`` in any position "just works" without any
    wrapping needed.

    This keeps the helper simple and avoids the non-tty credential-cache
    problems that bite ``sudo -v`` under ``bash -lc``.
    """
    s = cmd.lstrip()
    if s.startswith("sudo ") and "-S" not in s and "| sudo" not in s and "echo " not in s.split("sudo")[0]:
        return "echo 'password' | sudo -S " + s[5:]
    return cmd

def _fix_find_exec(cmd):
    """Auto-repair malformed find -exec commands AND teach the correction.

    A valid find -exec looks like one of:
        find ... -exec CMD \\;        (backslash-semicolon terminator)
        find ... -exec CMD {} +       (plus terminator — multiple args)

    The terminator lives in the ``-exec SECTION``, which runs from the
    ``-exec`` token up to the first shell pipe (``" | "``) or end of the
    command. Past mistakes we auto-repair:
      1. Trailing bare backslash at section end  (\\ instead of \\;)
      2. No terminator at all in the section

    IMPORTANT regression avoided: a command like
        find . -exec cat {} + | wc -l
    is VALID — the ``+`` terminator comes before a shell pipe. A naive
    check that only looks at the last character of the whole command
    would miss this and incorrectly append \\; to the end, breaking wc.

    NOTE: helpers run as a triple-quoted prelude, so each literal backslash
    here is written as four chars in the source ("\\\\") which become two
    chars in the loaded code ("\\") which represent a single backslash at
    exec time. The comments below use RUNTIME syntax (single backslash).
    """
    if "-exec" not in cmd:
        return cmd, ""

    # 1. Identify the -exec section: from "-exec" to the first " | " (shell
    #    pipe surrounded by spaces) or end of command. Pipes inside argument
    #    values are uncommon in real finds, so splitting on " | " is safe.
    exec_idx = cmd.find("-exec")
    pipe_idx = cmd.find(" | ", exec_idx)
    section_end = pipe_idx if pipe_idx != -1 else len(cmd)
    section = cmd[exec_idx:section_end].rstrip()

    # 2. Check if the section already has a valid terminator.
    #    RUNTIME checks:
    #       ends with "\\;"        -> valid (backslash-semicolon)
    #       ends with " +"         -> valid (plus terminator after a space)
    #       ends with "{}+"        -> valid (plus after braces with no space)
    if section.endswith("\\\\;"):
        return cmd, ""
    if section.endswith(" +") or section.endswith("{}+"):
        return cmd, ""

    # 3. Broken case A: trailing bare backslash (line continuation).
    #    RUNTIME: section ends with "\\" but NOT "\\;".
    if section.endswith("\\\\"):
        # Replace the trailing \\ with \\;
        fixed = cmd[:exec_idx] + section[:-1] + "\\\\;" + cmd[section_end:]
        note = (
            "Note: auto-corrected your find -exec — replaced trailing backslash "
            "with backslash-semicolon. find -exec needs \\\\; (the semicolon is "
            "required), not just a bare \\\\. Remember this."
        )
        return fixed, note

    # 4. Broken case B: no terminator at all. Insert \\; at the end of the
    #    -exec section (just before the pipe, or at end of command).
    fixed = cmd[:exec_idx] + section + " \\\\;" + cmd[section_end:]
    note = (
        "Note: auto-corrected your find -exec — inserted the missing \\\\; "
        "terminator. find -exec MUST end with \\\\; or + or it fails with "
        "'missing argument to -exec'. Remember this."
    )
    return fixed, note

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
