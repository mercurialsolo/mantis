"""How to finish a task efficiently — always-on technique."""

NAME = "Verification"
TAGS = ["verify", "done", "finish", "complete"]
TRIGGERS: list[str] = []
ALWAYS = True  # Always inject — every task needs to know how to finish.

CONTENT = """\
How to finish a task efficiently:
1. After your state-changing action, run ONE verification command (e.g. `gsettings get`, `which spotify`, `ls dest/`).
2. If the captured output matches what you expect, output `DONE` immediately. Don't keep verifying.
3. If it doesn't match, adjust and try again — but don't retry the IDENTICAL command, and don't keep polling indefinitely on async operations (`wait(N)` is your friend).
4. The captured-output feedback shows you exactly what your last command produced — read it carefully before deciding the next step.\
"""
