"""Hang-tolerant deploy for mantis-shopify on Daytona.

Adapted from daytona_mantis_mirror_v2.py — see that file for the
explanation of the create() hang workaround.

Usage:
    uv run python deploy/sim_envs/daytona_mantis_shopify.py
"""

from __future__ import annotations

import os
import pathlib
import secrets
import threading
import time

PORT = 8080
TITLE = "mantis-shopify"


def _load_env() -> None:
    for env_path in [pathlib.Path("/Users/barada/Sandbox/Mason/mantis/.env")]:
        if not env_path.is_file():
            continue
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _build_image(app_root: pathlib.Path, admin_token: str):
    from daytona import Image
    return (
        Image.debian_slim("3.11")
        .pip_install([
            "fastapi>=0.110",
            "uvicorn>=0.27",
            "jinja2>=3.1",
            "python-multipart>=0.0.9",
            "starlette>=0.27,<1.0",
            "itsdangerous>=2.0",
        ])
        .workdir("/srv")
        .add_local_dir(str(app_root), "/srv/app")
        .env({
            "PORT": str(PORT),
            "PYTHONUNBUFFERED": "1",
            "ENV_ADMIN_TOKEN": admin_token,
            "SEED": "42",
            "FAKE_NOW": "2026-06-09T09:00:00Z",
            "ENV_REQUIRE_AUTH": "0",
        })
    )


def _identify_sandbox(d, exclude_ids: set[str]):
    title_marker = f'title="{TITLE}"'
    for s in d.list():
        sid = getattr(s, "id", "")
        if sid in exclude_ids:
            continue
        state = str(getattr(s, "state", "")).replace("SandboxState.", "")
        if state != "STARTED":
            continue
        try:
            r = s.process.exec(
                "grep -h 'title=' /srv/app/main.py 2>/dev/null | head -1"
            )
            out = (getattr(r, "result", "") or "")
            if title_marker in out:
                return s
        except Exception:
            continue
    return None


def _start_uvicorn(sandbox) -> None:
    sandbox.process.exec(
        "python3 -c \""
        "import os, signal\n"
        "for pid in os.listdir('/proc'):\n"
        "    if not pid.isdigit(): continue\n"
        "    try:\n"
        "        with open(f'/proc/{pid}/cmdline','rb') as f: c=f.read()\n"
        "    except: continue\n"
        "    if b'uvicorn' in c or b'app.main' in c:\n"
        "        try: os.kill(int(pid), signal.SIGKILL)\n"
        "        except: pass\n"
        "\""
    )
    time.sleep(1)
    sandbox.process.exec(
        f"cd /srv && nohup python3 -m uvicorn app.main:app "
        f"--host 0.0.0.0 --port {PORT} > /tmp/uvicorn.log 2>&1 &"
    )


def _wait_healthy(sandbox, timeout: int = 45) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = sandbox.process.exec(
                f"python3 -c \"import urllib.request; "
                f"print(urllib.request.urlopen('http://127.0.0.1:{PORT}/__env__/health',"
                f"timeout=3).read().decode())\""
            )
            out = (getattr(r, "result", "") or "").strip()
            if '"ok"' in out:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def main() -> None:
    from daytona import (
        CreateSandboxFromImageParams,
        Daytona,
        DaytonaConfig,
    )

    _load_env()
    if not os.environ.get("DAYTONA_API_KEY"):
        raise SystemExit("error: DAYTONA_API_KEY not set")

    app_root = pathlib.Path(__file__).parent / "mantis_shopify" / "app"
    if not app_root.is_dir():
        raise SystemExit(f"error: {app_root} not found")

    admin_token = secrets.token_urlsafe(24)
    d = Daytona(DaytonaConfig(api_key=os.environ["DAYTONA_API_KEY"]))
    pre_existing = {getattr(s, "id", "") for s in d.list()}

    image = _build_image(app_root, admin_token)
    print("→ creating sandbox …", flush=True)

    result_holder: dict = {}
    def _create():
        try:
            sb = d.create(
                CreateSandboxFromImageParams(image=image),
                timeout=300,
                on_snapshot_create_logs=lambda line: print(f"  [build] {line[:140]}", flush=True),
            )
            result_holder["sb"] = sb
        except Exception as exc:
            result_holder["err"] = exc

    t = threading.Thread(target=_create, daemon=True)
    t.start()
    t.join(360)

    sb = result_holder.get("sb")
    err = result_holder.get("err")
    if err and not sb:
        raise RuntimeError(f"create() failed: {err}")
    if sb is None:
        print("SDK create() hung — identifying sandbox via list() …", flush=True)
        for _ in range(8):
            sb = _identify_sandbox(d, pre_existing)
            if sb:
                break
            time.sleep(5)
        if sb is None:
            raise RuntimeError("create() hung and could not identify sandbox post-hoc")
        print(f"identified hung sandbox: {sb.id}", flush=True)

    sid = getattr(sb, "id", "?")
    print(f"  sandbox id: {sid}", flush=True)

    try:
        sb.set_autostop_interval(180)
    except Exception as exc:
        print(f"  warning: autostop not set ({exc})", flush=True)

    print("→ starting uvicorn …", flush=True)
    _start_uvicorn(sb)

    print("→ waiting for health …", flush=True)
    healthy = _wait_healthy(sb, timeout=60)

    preview = sb.get_preview_link(PORT)
    flag = "OK" if healthy else "WARN (health did not return ok yet)"
    print()
    print("=" * 64)
    print(f"=== {TITLE} — {flag} ===")
    print(f"  URL:           {preview.url}")
    print(f"  Preview token: {getattr(preview, 'token', None)}")
    print(f"  Admin token:   {admin_token}")
    print(f"  Sandbox id:    {sid}")


if __name__ == "__main__":
    main()
