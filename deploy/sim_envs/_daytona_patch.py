"""Patch an existing Daytona sandbox in place — upload modified files
and restart uvicorn. Faster than a full image rebuild during iteration.

    .venv-daytona/bin/python deploy/sim_envs/_daytona_patch.py <sandbox-id>
"""

from __future__ import annotations

import os
import pathlib
import sys


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: _daytona_patch.py <sandbox-id>")
    sandbox_id = sys.argv[1]

    # Load .env for DAYTONA_API_KEY.
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

    from daytona import Daytona, DaytonaConfig

    daytona = Daytona(DaytonaConfig(api_key=os.environ["DAYTONA_API_KEY"]))
    sandbox = daytona.get(sandbox_id)
    state = getattr(sandbox, "state", None)
    print(f"sandbox: {sandbox.id} (state: {state})")
    if state and str(state).endswith("STOPPED"):
        print("→ starting stopped sandbox …")
        daytona.start(sandbox)
        sandbox = daytona.get(sandbox_id)
        print(f"  state now: {getattr(sandbox, 'state', '?')}")

    app_root = pathlib.Path(__file__).parent / "mantis_boattrader" / "app"
    paths_to_patch = [
        app_root / "main.py",
        app_root / "seed.py",
        app_root / "db.py",
        app_root / "templates" / "base.html",
        app_root / "templates" / "home.html",
        app_root / "templates" / "boats.html",
        app_root / "templates" / "boat_detail.html",
        app_root / "templates" / "_listing_card.html",
        app_root / "static" / "app.css",
        app_root / "static" / "bt_logo.svg",
    ]

    for p in paths_to_patch:
        rel = p.relative_to(app_root.parent)
        remote = f"/srv/{rel.as_posix()}"
        data = p.read_bytes()
        sandbox.fs.upload_file(data, remote)
        print(f"  uploaded {remote}  ({len(data)} bytes)")

    # Kill uvicorn via /proc (the slim image has no ps/pkill).
    kill_py = (
        "import os, signal\n"
        "for pid_s in os.listdir('/proc'):\n"
        "    if not pid_s.isdigit() or int(pid_s) == os.getpid():\n"
        "        continue\n"
        "    try:\n"
        "        with open(f'/proc/{pid_s}/cmdline','rb') as f:\n"
        "            cmd=f.read().decode('utf-8','replace').replace('\\x00',' ')\n"
        "    except (IOError, OSError):\n"
        "        continue\n"
        "    if 'uvicorn' in cmd or 'app.main:app' in cmd:\n"
        "        try: os.kill(int(pid_s), signal.SIGKILL)\n"
        "        except Exception: pass\n"
    )
    import base64
    b64 = base64.b64encode(kill_py.encode()).decode()
    sandbox.process.exec(
        f'python -c "import base64,sys;exec(base64.b64decode(\\"{b64}\\"))"'
    )
    sandbox.process.exec(
        "cd /srv && setsid python -m uvicorn app.main:app "
        "--host 0.0.0.0 --port 8080 > /tmp/uvicorn.log 2>&1 < /dev/null &"
    )
    print("uvicorn restarted")

    preview = sandbox.get_preview_link(8080)
    print(f"URL: {preview.url}")
    print(f"Token: {preview.token}")


if __name__ == "__main__":
    main()
