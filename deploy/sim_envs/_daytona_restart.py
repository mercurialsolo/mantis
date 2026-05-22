"""Force-restart uvicorn inside an existing Daytona sandbox.

Uses Python's /proc walk to find uvicorn-running PIDs (slim image lacks
``ps``/``pkill``/``curl``), SIGKILLs them, then relaunches.

    .venv-daytona/bin/python deploy/sim_envs/_daytona_restart.py <sandbox-id>
"""

from __future__ import annotations

import os
import pathlib
import sys
import time


KILL_PY = """\
import os, signal
killed = []
for pid_s in os.listdir('/proc'):
    if not pid_s.isdigit():
        continue
    pid = int(pid_s)
    if pid == os.getpid():
        continue
    try:
        with open(f'/proc/{pid}/cmdline', 'rb') as f:
            cmd = f.read().decode('utf-8', 'replace').replace('\\x00', ' ')
    except (IOError, OSError):
        continue
    if 'uvicorn' in cmd or 'app.main:app' in cmd:
        try:
            os.kill(pid, signal.SIGKILL)
            killed.append((pid, cmd.strip()[:80]))
        except (PermissionError, ProcessLookupError):
            pass
print('killed:', killed)
"""


PROBE_PY = """\
import os
out = []
for pid_s in os.listdir('/proc'):
    if not pid_s.isdigit():
        continue
    try:
        with open(f'/proc/{pid_s}/cmdline', 'rb') as f:
            cmd = f.read().decode('utf-8', 'replace').replace('\\x00', ' ').strip()
    except (IOError, OSError):
        continue
    if 'python' in cmd or 'uvicorn' in cmd:
        out.append(f'{pid_s}: {cmd[:100]}')
print('procs:\\n  ' + '\\n  '.join(out) if out else 'no python/uvicorn procs')
"""


HEALTH_PY = """\
import urllib.request, sys
try:
    r = urllib.request.urlopen('http://127.0.0.1:8080/__env__/health', timeout=3)
    print('health:', r.status, r.read().decode()[:200])
except Exception as exc:
    print('health-err:', exc)
"""


def _load_env() -> None:
    for env_path in [
        pathlib.Path("/Users/barada/Sandbox/Mason/mantis/.env"),
        pathlib.Path(__file__).resolve().parents[2] / ".env",
    ]:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> None:
    sandbox_id = sys.argv[1]
    _load_env()
    from daytona import Daytona, DaytonaConfig
    daytona = Daytona(DaytonaConfig(api_key=os.environ["DAYTONA_API_KEY"]))
    sandbox = daytona.get(sandbox_id)

    def run(cmd: str) -> str:
        return sandbox.process.exec(cmd).result or ""

    def run_py(py: str) -> str:
        # Pipe Python source via stdin so we don't deal with shell quoting.
        import base64
        b64 = base64.b64encode(py.encode()).decode()
        return run(f'python -c "import base64,sys;exec(base64.b64decode(\\"{b64}\\"))"')

    print("BEFORE:")
    print(run_py(PROBE_PY))

    print("\nkilling uvicorn …")
    print(run_py(KILL_PY))
    time.sleep(2)

    print("\nAFTER kill:")
    print(run_py(PROBE_PY))

    print("\nstarting fresh uvicorn (detached) …")
    sandbox.process.exec(
        "cd /srv && setsid python -m uvicorn app.main:app "
        "--host 0.0.0.0 --port 8080 > /tmp/uvicorn.log 2>&1 < /dev/null &"
    )
    time.sleep(4)

    print("\nAFTER start:")
    print(run_py(PROBE_PY))

    print("\nhealth from inside:")
    print(run_py(HEALTH_PY))


if __name__ == "__main__":
    main()
