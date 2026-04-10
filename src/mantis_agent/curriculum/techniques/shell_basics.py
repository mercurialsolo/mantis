"""Shell command basics: find, grep, pipes, and run_command quirks."""

NAME = "Shell basics"
TAGS = ["shell", "bash", "find", "grep", "pipe", "command", "terminal", "subprocess"]
TRIGGERS = [
    r"\bfind\b", r"\bgrep\b", r"\b-exec\b", r"\bawk\b", r"\bsed\b",
    r"\bxargs\b", r"\bpipe\b", r"recursive", r"directory tree",
    r"\bsearch\b", r"\bmatch\b",
]
ALWAYS = False

CONTENT = """\
Shell command techniques (you have run_command(cmd) which runs via bash):
- find -exec MUST terminate with `\\;` — like `find . -name '*.txt' -exec cp {} dest/ \\;`. A bare trailing `\\` is line continuation and breaks the command.
- For multiple commands in one call, use `&&` or `;` between them: `run_command('cd dir && find . -name foo')`.
- Capture output for grep filtering: `run_command('snap list | grep spotify')` works directly.
- run_command starts at the home directory, not at any terminal's cwd. Use absolute paths or `cd` first if the task references a specific directory.\
"""
