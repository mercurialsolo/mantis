"""Package installation: snap, apt, dpkg."""

NAME = "Package management"
TAGS = ["install", "package", "snap", "apt", "dpkg", "uninstall", "software"]
TRIGGERS = [
    r"\binstall\b", r"\bapt\b", r"\bsnap\b", r"\bdpkg\b",
    r"package", r"spotify", r"firefox", r"\bvscode\b", r"\bvs code\b",
    r"\buninstall\b",
]
ALWAYS = False

CONTENT = """\
Package install techniques:
- snap install: `run_command("sudo snap install spotify")` — passwordless sudo is set up automatically.
- apt install: `run_command("sudo apt install -y <package>")` — `-y` skips the confirmation prompt.
- apt may be locked by `packagekitd` — if `Could not get lock` appears, use snap or wait.
- After installing, verify with `which <binary>` rather than `snap list` (snap list can lag).
- Snap installs may show `[install-snap change in progress]` — that means it IS installing in the background, just wait or check `snap changes` status.\
"""
