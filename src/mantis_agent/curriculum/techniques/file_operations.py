"""File operations: copy, move, find with patterns, hierarchy preservation."""

NAME = "File operations"
TAGS = ["file", "copy", "move", "rename", "directory", "folder", "hierarchy",
        "path", "desktop", "trash", "restore", "delete"]
TRIGGERS = [
    r"\bcopy\b", r"\bcp\b", r"\bmove\b", r"\brename\b", r"\bdirectory\b",
    r"\bfolder\b", r"\bfile(s)?\b", r"hierarchy", r"\.jpg", r"\.png",
    r"\.txt", r"\.ipynb", r"\.json", r"\.csv", r"\.md", r"\.webp",
]
ALWAYS = False

CONTENT = """\
File operation techniques:
- Copy files: `cp source dest`. Copy directories recursively: `cp -r source dest`.
- Preserve directory hierarchy when copying matched files: `find . -name '*.txt' -exec cp --parents {} dest/ \\;` (the --parents flag preserves the relative path under dest).
- Move (rename): `mv old new`. This is NOT compression — see the compression technique if the task says compress.
- Find by name pattern: `find <root> -name '*.ext'` (use single quotes around the pattern to prevent shell expansion).
- Use ABSOLUTE paths or `cd` first when the task references a specific directory — run_command starts at home, not at the terminal's cwd.\
"""
