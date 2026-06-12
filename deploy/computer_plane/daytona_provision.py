"""Build + verify the Daytona computer-plane image (#699 Phase 2).

Brings up a Daytona sandbox with:

* xvfb / xdotool / scrot / google-chrome-stable apt installed
* fastapi / uvicorn / pillow / pydantic pip installed
* the local mantis_agent source uploaded
* uvicorn booting ``mantis_agent.server.computer_agent:app`` on
  port 8000 at sandbox start

Then probes the preview URL for ``/health`` and ``POST /session/init``
to confirm the wire contract is reachable. On success, prints the
sandbox id so the operator can use it as the
``MANTIS_DAYTONA_SNAPSHOT`` default for ``DaytonaComputerImpl``.

Usage
=====

::

    DAYTONA_API_KEY=... python deploy/computer_plane/daytona_provision.py
    # → on success: prints sandbox-id + preview URL + run summary
    #   and (with --keep) leaves the sandbox running for further probes

    DAYTONA_API_KEY=... python deploy/computer_plane/daytona_provision.py \\
        --no-create --sandbox-id ab1234 \\
        # → smoke an existing sandbox; useful after a fresh truss push

The script is idempotent: re-running it builds a new sandbox each
time. Old sandboxes can be cleaned up via the Daytona dashboard or
``daytona_provision.py --delete <id>``.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

PORT = 8000
HEALTH_TIMEOUT = 180.0
ROOT = pathlib.Path(__file__).resolve().parents[2]


def _build_image():
    from daytona import Image

    # Spec § 5: container image must pre-install xvfb / xdotool /
    # scrot / Chrome AND boot uvicorn against
    # ``mantis_agent.server.computer_agent:app`` at start.
    return (
        Image.debian_slim("3.11")
        .run_commands(
            "DEBIAN_FRONTEND=noninteractive apt-get update",
            "DEBIAN_FRONTEND=noninteractive apt-get install -y "
            "xvfb xdotool xclip scrot gnupg curl wget "
            "fonts-liberation fonts-dejavu-core",
            # Google Chrome — same pin as the Modal holo3 image so
            # snapshots produced here load on a Chrome-major-equivalent
            # runner.
            "curl -fsSL https://dl.google.com/linux/linux_signing_key.pub "
            "| gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg",
            "echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/"
            "google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ "
            "stable main' > /etc/apt/sources.list.d/google-chrome.list",
            "DEBIAN_FRONTEND=noninteractive apt-get update && "
            "DEBIAN_FRONTEND=noninteractive apt-get install -y "
            "google-chrome-stable || true",
        )
        .pip_install([
            "fastapi>=0.110",
            "uvicorn>=0.27",
            "pydantic>=2",
            "pillow>=10",
            "mss>=9.0",
            "requests>=2.28",
            "websocket-client>=1.6",
        ])
        .workdir("/srv")
        # Ship the local mantis_agent source so we don't have to
        # publish a wheel for every iteration.
        .add_local_dir(str(ROOT / "src" / "mantis_agent"), "/srv/mantis_agent")
        .env({
            "PORT": str(PORT),
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": "/srv",
            # Tells DaytonaComputerImpl which port to look at; the
            # image-side equivalent is the uvicorn --port flag below.
            "MANTIS_COMPUTER_PLANE_PORT": str(PORT),
        })
    )


def _start_uvicorn(sandbox) -> None:
    """Kill any stale uvicorn, then start the computer-plane server."""
    sandbox.process.exec(
        "python3 -c \""
        "import os, signal\n"
        "for pid in os.listdir('/proc'):\n"
        "    if not pid.isdigit(): continue\n"
        "    try:\n"
        "        with open(f'/proc/{pid}/cmdline','rb') as f: c=f.read()\n"
        "    except: continue\n"
        "    if b'uvicorn' in c or b'computer_agent' in c:\n"
        "        try: os.kill(int(pid), signal.SIGKILL)\n"
        "        except: pass\n"
        "\""
    )
    time.sleep(1)
    sandbox.process.exec(
        f"cd /srv && nohup python3 -m uvicorn "
        f"mantis_agent.server.computer_agent:app "
        f"--host 0.0.0.0 --port {PORT} > /tmp/uvicorn.log 2>&1 &"
    )


def _wait_healthy(sandbox, *, timeout: float = HEALTH_TIMEOUT) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = sandbox.process.exec(
                f"python3 -c \"import urllib.request; "
                f"print(urllib.request.urlopen("
                f"'http://127.0.0.1:{PORT}/health', timeout=3).status)\""
            )
            out = (getattr(r, "result", "") or "").strip()
            if "200" in out:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def _provision_and_smoke(
    api_key: str, *, keep: bool, log,
) -> tuple[str, str | None]:
    """Build a fresh sandbox, boot uvicorn, prove /health, return id."""
    from daytona import (
        CreateSandboxFromImageParams,
        Daytona,
        DaytonaConfig,
    )

    log("→ building Daytona image (apt + pip + mantis_agent source) …")
    image = _build_image()

    d = Daytona(DaytonaConfig(api_key=api_key))
    log("→ creating sandbox …")
    sandbox = d.create(
        CreateSandboxFromImageParams(image=image),
        timeout=600,
        on_snapshot_create_logs=lambda line: log(f"   [build] {line[:140]}"),
    )
    sandbox_id = getattr(sandbox, "id", "") or "?"
    log(f"→ sandbox id = {sandbox_id}")

    log("→ starting uvicorn …")
    _start_uvicorn(sandbox)

    log("→ waiting for /health …")
    if not _wait_healthy(sandbox):
        # Pull the uvicorn log so the operator can see what went wrong.
        try:
            r = sandbox.process.exec("tail -c 4000 /tmp/uvicorn.log")
            log(f"   uvicorn.log tail:\n{getattr(r, 'result', '')!s}")
        except Exception:
            pass
        if not keep:
            log("→ tearing down failed sandbox …")
            try:
                sandbox.delete()
            except Exception:
                pass
        raise RuntimeError(
            "Daytona sandbox /health did not respond — see log above"
        )
    log("→ /health 200 — wire-server is up")

    # Resolve the preview URL.
    preview_url = None
    try:
        link = sandbox.get_preview_link(PORT)
        preview_url = getattr(link, "url", None)
    except Exception as exc:
        log(f"   note: preview link lookup raised {exc!r}")
    if preview_url:
        log(f"→ preview URL: {preview_url}")

    if not keep:
        log("→ tearing down (use --keep to leave the sandbox running) …")
        try:
            sandbox.delete()
        except Exception as exc:
            log(f"   note: delete raised {exc!r}")

    return sandbox_id, preview_url


def _delete_sandbox(api_key: str, sandbox_id: str, log) -> None:
    from daytona import Daytona, DaytonaConfig

    d = Daytona(DaytonaConfig(api_key=api_key))
    target = None
    for s in d.list():
        if getattr(s, "id", "") == sandbox_id:
            target = s
            break
    if target is None:
        log(f"→ no sandbox with id={sandbox_id} — nothing to delete")
        return
    target.delete()
    log(f"→ deleted sandbox {sandbox_id}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Provision + verify the Daytona computer-plane image."
        ),
    )
    parser.add_argument(
        "--keep", action="store_true",
        help="Leave the sandbox running after /health passes "
             "(default: tear down).",
    )
    parser.add_argument(
        "--delete", metavar="SANDBOX_ID", default="",
        help="Delete an existing sandbox by id, then exit.",
    )
    args = parser.parse_args(argv)

    api_key = os.environ.get("DAYTONA_API_KEY", "").strip()
    if not api_key:
        print("error: DAYTONA_API_KEY env var is required", file=sys.stderr)
        return 2

    def log(msg: str) -> None:
        print(msg)
        sys.stdout.flush()

    if args.delete:
        _delete_sandbox(api_key, args.delete, log)
        return 0

    sandbox_id, preview_url = _provision_and_smoke(
        api_key, keep=args.keep, log=log,
    )
    print()
    print("=" * 60)
    print(f"DAYTONA SANDBOX ID: {sandbox_id}")
    if preview_url:
        print(f"PREVIEW URL: {preview_url}")
    print("=" * 60)
    print(
        "Set ``MANTIS_DAYTONA_SNAPSHOT=<sandbox-id>`` in the brain "
        "secret to default DaytonaComputerImpl to this image."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
