"""Curriculum knowledge bank for the Mantis agent.

The curriculum is a collection of small, focused technique snippets
organized by topic. The agent's prompt does NOT include all of them — that
would overwhelm a small model. Instead, ``select_techniques()`` picks the
1-3 most relevant snippets for the current task based on keywords in the
instruction and the evaluator config, and the hint engine injects them
inline.

This keeps the prompt lean for simple tasks (no irrelevant noise) while
still teaching the model the right pattern when it matters.

Adding a new technique:
    1. Add a constant ``TECHNIQUE_<NAME>`` to one of the topic constants below
    2. Add corresponding keywords/triggers to the matching entry in ``TOPICS``
    3. (Optional) Add a unit test in tests/test_curriculum.py

Topic structure:
    Each topic has:
      - ``triggers``: list of substrings/regexes that match relevant tasks
      - ``techniques``: list of snippet strings
"""

from __future__ import annotations

import re

# ── Topic: shell command basics ───────────────────────────────────────────────
SHELL_BASICS = """\
Shell command techniques (you have run_command(cmd) which runs via bash):
- find -exec MUST terminate with `\\;` — like `find . -name '*.txt' -exec cp {} dest/ \\;`. A bare trailing `\\` is line continuation and breaks the command.
- For multiple commands in one call, use `&&` or `;` between them: `run_command('cd dir && find . -name foo')`.
- Capture output for grep filtering: `run_command('snap list | grep spotify')` works directly.
- run_command starts at the home directory, not at any terminal's cwd.\
"""

# ── Topic: file operations ────────────────────────────────────────────────────
FILE_OPERATIONS = """\
File operation techniques:
- Copy files: `cp source dest`. Copy directories recursively: `cp -r source dest`.
- Preserve directory hierarchy when copying matched files: `find . -name '*.txt' -exec cp --parents {} dest/ \\;` (the --parents flag preserves the relative path under dest).
- Move (rename): `mv old new`. This is NOT compression — see "Compression" below.
- Find by name pattern: `find <root> -name '*.ext'` (use single quotes around the pattern to prevent shell expansion).
- Use ABSOLUTE paths or `cd` first when the task references a specific directory — run_command starts at home, not at the terminal's cwd.\
"""

# ── Topic: compression ────────────────────────────────────────────────────────
COMPRESSION = """\
Compression techniques:
- "Compress" means reducing file size, NOT moving. Common tools:
  - `gzip file.txt` → produces `file.txt.gz` (replaces original)
  - `tar -czf archive.tar.gz dir/` → tarball with gzip compression
  - `zip -r archive.zip dir/` → zip archive
- A "compressed" file should have a `.gz`, `.tar.gz`, `.zip`, etc. extension.
- If the task says "compress files older than 30 days", you must actually run gzip/tar — `mv` alone is not compression.\
"""

# ── Topic: gsettings / desktop preferences ────────────────────────────────────
GSETTINGS = """\
gsettings techniques:
- Get a value: `run_command("gsettings get org.gnome.desktop.interface text-scaling-factor")`
- Set a scalar value: `run_command("gsettings set org.gnome.desktop.session idle-delay 0")`
- Set an array value (uses nested quotes — type_text/run_command handle the escaping):
  `run_command("gsettings set org.gnome.shell favorite-apps \\"['firefox.desktop', 'thunderbird.desktop']\\"")`
- If gsettings reports "No such schema", you may need DBUS context:
  `run_command("export DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus' && gsettings ...")`
- After setting, verify with `gsettings get` and you should see the new value.\
"""

# ── Topic: package management ─────────────────────────────────────────────────
PACKAGE_MANAGEMENT = """\
Package install techniques:
- snap install: `run_command("sudo snap install spotify")` — sudo is auto-wrapped, no password needed.
- apt install: `run_command("sudo apt install -y <package>")` — `-y` skips the confirmation prompt.
- apt may be locked by `packagekitd` — if `Could not get lock` appears, use snap or wait.
- After installing, verify with `which <binary>` rather than `snap list` (snap list can lag).
- Snap installs may show `[install-snap change in progress]` — that means it IS installing, just wait or check `snap changes`.\
"""

# ── Topic: system settings (timezone, locale, etc.) ───────────────────────────
SYSTEM_SETTINGS = """\
System settings techniques:
- Timezone: `run_command("sudo timedatectl set-timezone UTC")` then verify with `timedatectl`.
- Default Python: use update-alternatives, e.g. `update-alternatives --config python3`.
- Locale: `localectl set-locale LANG=en_US.UTF-8`.
- Permissions: `chmod 644 file` for files, `chmod 755 dir` for directories. Recursive: `chmod -R 644 dir/`.\
"""

# ── Topic: verification & finishing ──────────────────────────────────────────
VERIFICATION = """\
How to finish a task efficiently:
1. After your state-changing action, run ONE verification command (e.g. `gsettings get`, `which spotify`, `ls dest/`).
2. If the captured output matches what you expect, output `DONE` immediately. Don't keep verifying.
3. If it doesn't match, adjust and try again.
4. The captured-output feedback shows you exactly what your last command produced — read it carefully before deciding the next step.\
"""


# ── Topic registry: each entry has triggers and techniques ────────────────────
TOPICS: list[dict] = [
    {
        "name": "shell_basics",
        "triggers": [
            r"\bfind\b", r"\bgrep\b", r"\b-exec\b", r"\bawk\b", r"\bsed\b",
            r"\bxargs\b", r"\bpipe\b", r"recursive", r"directory tree",
        ],
        "content": SHELL_BASICS,
    },
    {
        "name": "file_operations",
        "triggers": [
            r"\bcopy\b", r"\bcp\b", r"\bmove\b", r"\brename\b", r"\bdirectory\b",
            r"\bfolder\b", r"\bfile(s)?\b", r"hierarchy", r"\.jpg", r"\.txt",
            r"\.ipynb", r"\.json", r"\.csv", r"\.md",
        ],
        "content": FILE_OPERATIONS,
    },
    {
        "name": "compression",
        "triggers": [
            r"compress", r"\bgzip\b", r"\btar\b", r"\bzip\b", r"archive",
            r"\.gz\b", r"\.tar\b", r"\.zip\b",
        ],
        "content": COMPRESSION,
    },
    {
        "name": "gsettings",
        "triggers": [
            r"gsettings", r"favorite", r"dim screen", r"idle.?delay",
            r"text scaling", r"magnif", r"notification", r"dconf",
            r"\bdesktop\b.*setting", r"gnome",
        ],
        "content": GSETTINGS,
    },
    {
        "name": "package_management",
        "triggers": [
            r"\binstall\b", r"\bapt\b", r"\bsnap\b", r"\bdpkg\b",
            r"package", r"spotify", r"firefox", r"\bvscode\b",
        ],
        "content": PACKAGE_MANAGEMENT,
    },
    {
        "name": "system_settings",
        "triggers": [
            r"timezone", r"time zone", r"timedatectl", r"locale",
            r"python.?version", r"permission", r"\bchmod\b", r"\bchown\b",
        ],
        "content": SYSTEM_SETTINGS,
    },
    {
        "name": "verification",
        "triggers": [
            # Always-relevant — the model needs to know how to finish.
            # Empty triggers list with always=True makes this a default.
        ],
        "content": VERIFICATION,
        "always": True,
    },
]


def select_techniques(
    instruction: str,
    hint_text: str = "",
    domain: str = "",
    max_topics: int = 3,
) -> str:
    """Pick the most relevant curriculum snippets for a task and return them.

    Selection works by scanning the task instruction (and any hint text already
    derived from the evaluator config) for trigger keywords. Topics that match
    are ranked by trigger count and the top ``max_topics`` are returned.

    Topics marked ``always=True`` are always included (in addition to keyword
    matches). The total returned text is capped to keep prompts lean.
    """
    haystack = (instruction + " " + hint_text).lower()
    scored: list[tuple[int, dict]] = []
    always: list[dict] = []
    for topic in TOPICS:
        if topic.get("always"):
            always.append(topic)
            continue
        score = 0
        for pat in topic["triggers"]:
            try:
                if re.search(pat, haystack):
                    score += 1
            except re.error:
                continue
        if score > 0:
            scored.append((score, topic))

    scored.sort(key=lambda x: -x[0])
    chosen = [t for _, t in scored[:max_topics]] + always
    if not chosen:
        return ""
    return "\n\n".join(t["content"] for t in chosen)


__all__ = ["select_techniques", "TOPICS"]
