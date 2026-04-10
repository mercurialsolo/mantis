"""System settings: timezone, locale, permissions, user management."""

NAME = "System settings"
TAGS = ["timezone", "locale", "permission", "chmod", "user", "system", "ssh"]
TRIGGERS = [
    r"timezone", r"time zone", r"timedatectl", r"locale",
    r"python.?version", r"permission", r"\bchmod\b", r"\bchown\b",
    r"\busermod\b", r"\buseradd\b", r"\bgroupadd\b", r"\bsshd?\b",
]
ALWAYS = False

CONTENT = """\
System settings techniques:
- Timezone: `run_command("sudo timedatectl set-timezone UTC")` then verify with `timedatectl`.
- Default Python: use update-alternatives, e.g. `update-alternatives --config python3`.
- Locale: `localectl set-locale LANG=en_US.UTF-8`.
- Permissions: `chmod 644 file` for files, `chmod 755 dir` for directories. Recursive: `chmod -R 644 dir/`.
- User management: `sudo useradd -m -d /home/USER -s /bin/bash USERNAME` then set password via `echo 'USERNAME:PASSWORD' | sudo chpasswd`.\
"""
