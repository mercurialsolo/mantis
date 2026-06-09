"""Hang-tolerant deploy for mantis mirror sim envs.

The Daytona 0.183.0 SDK has a known hang in `daytona.create()` after
debian_slim image build completes — see memory note
`feedback_daytona_fresh_build_hang_reuse_or_snapshot.md`. The sandbox
IS created and reaches STARTED state; the SDK call just never returns.

This script works around it by running `create()` in a thread with a
deadline, then falling back to identifying the just-created sandbox via
`d.list()` if the call hasn't returned. It also handles uvicorn start
robustly (kills any stale uvicorn first; uses python urllib for health
since the slim image has no curl).

Usage:
    uv run python deploy/sim_envs/daytona_mantis_mirror_v2.py --env all
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import pathlib
import secrets
import threading
import time

PORT = 8080

ENVS = {
    "mercor":   {"dir": "mantis_mercor",   "title": "mantis-mercor"},
    "fiverr":   {"dir": "mantis_fiverr",   "title": "mantis-fiverr"},
    "linkedin": {"dir": "mantis_linkedin", "title": "mantis-linkedin"},
    "indeed":   {"dir": "mantis_indeed",   "title": "mantis-indeed"},
}


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
            "FAKE_NOW": "2026-06-08T09:00:00Z",
            "ENV_REQUIRE_AUTH": "0",
        })
    )


def _identify_sandbox(d, slug: str, exclude_ids: set[str]):
    """After a create() hang, find the just-built sandbox by inspecting
    /srv/app. We look at STARTED sandboxes not in `exclude_ids` and grep
    for the title slot in main.py."""
    from daytona import Daytona  # noqa: F401
    title_marker = f'title="{ENVS[slug]["title"]}"'
    for s in d.list():
        sid = getattr(s, "id", "")
        if sid in exclude_ids:
            continue
        state = str(getattr(s, "state", "")).replace("SandboxState.", "")
        if state != "STARTED":
            continue
        try:
            r = s.process.exec("grep -h 'title=' /srv/app/main.py 2>/dev/null | head -1")
            out = (getattr(r, "result", "") or "")
            if title_marker in out:
                return s
        except Exception:
            continue
    return None


def _start_uvicorn(sandbox) -> None:
    """Kill any stale uvicorn (no pgrep in slim — use /proc), then start fresh."""
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


def _wait_healthy(sandbox, timeout: int = 30) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = sandbox.process.exec(
                f"python3 -c \"import urllib.request; "
                f"print(urllib.request.urlopen('http://127.0.0.1:{PORT}/__env__/health',timeout=3).read().decode())\""
            )
            out = (getattr(r, "result", "") or "").strip()
            if '"ok"' in out:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def deploy(slug: str, log) -> dict:
    from daytona import (
        CreateSandboxFromImageParams,
        Daytona,
        DaytonaConfig,
    )

    spec = ENVS[slug]
    app_root = pathlib.Path(__file__).parent / spec["dir"] / "app"
    if not app_root.is_dir():
        raise SystemExit(f"error: {app_root} not found")

    api_key = os.environ["DAYTONA_API_KEY"]
    admin_token = secrets.token_urlsafe(24)
    d = Daytona(DaytonaConfig(api_key=api_key))

    pre_existing = {getattr(s, "id", "") for s in d.list()}

    image = _build_image(app_root, admin_token)
    log(f"[{slug}] → creating sandbox …")

    result_holder = {}
    def _create():
        try:
            sb = d.create(
                CreateSandboxFromImageParams(image=image),
                timeout=300,
                on_snapshot_create_logs=lambda line: log(f"[{slug}]   [build] {line[:140]}"),
            )
            result_holder["sb"] = sb
        except Exception as exc:
            result_holder["err"] = exc

    t = threading.Thread(target=_create, daemon=True)
    t.start()

    # Allow generous time for the build + a grace period for the hang
    SDK_DEADLINE = 360  # 6 min
    t.join(SDK_DEADLINE)

    sb = result_holder.get("sb")
    err = result_holder.get("err")
    if err and not sb:
        raise RuntimeError(f"create() failed: {err}")
    if sb is None:
        log(f"[{slug}] SDK create() hung — identifying sandbox via list() …")
        # Give Daytona a moment to register the sandbox
        for _ in range(8):
            sb = _identify_sandbox(d, slug, pre_existing)
            if sb:
                break
            time.sleep(5)
        if sb is None:
            raise RuntimeError("create() hung and could not identify sandbox post-hoc")
        log(f"[{slug}] identified hung sandbox: {sb.id}")

    sid = getattr(sb, "id", "?")
    log(f"[{slug}]   sandbox id: {sid}")

    try:
        sb.set_autostop_interval(180)
    except Exception as exc:
        log(f"[{slug}]   warning: autostop not set ({exc})")

    log(f"[{slug}] → starting uvicorn …")
    _start_uvicorn(sb)

    log(f"[{slug}] → waiting for health …")
    healthy = _wait_healthy(sb, timeout=45)

    preview = sb.get_preview_link(PORT)
    return {
        "slug": slug,
        "title": spec["title"],
        "sandbox_id": sid,
        "url": preview.url,
        "preview_token": getattr(preview, "token", None),
        "admin_token": admin_token,
        "health_ok": healthy,
    }


def _print_result(r: dict) -> None:
    flag = "OK" if r["health_ok"] else "WARN (health did not return ok)"
    print()
    print(f"=== {r['title']} — {flag} ===")
    print(f"  URL:           {r['url']}")
    print(f"  Preview token: {r['preview_token']}")
    print(f"  Admin token:   {r['admin_token']}")
    print(f"  Sandbox id:    {r['sandbox_id']}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=[*ENVS.keys(), "all"])
    args = p.parse_args()

    _load_env()
    if not os.environ.get("DAYTONA_API_KEY"):
        raise SystemExit("error: DAYTONA_API_KEY not set")

    slugs = list(ENVS.keys()) if args.env == "all" else [args.env]
    log_lock = threading.Lock()

    def log(msg):
        with log_lock:
            print(msg, flush=True)

    results: list[dict] = []
    with cf.ThreadPoolExecutor(max_workers=len(slugs)) as ex:
        futs = {ex.submit(deploy, s, log): s for s in slugs}
        for fut in cf.as_completed(futs):
            s = futs[fut]
            try:
                results.append(fut.result())
            except Exception as exc:
                log(f"[{s}] FAILED: {exc}")
                results.append({"slug": s, "title": ENVS[s]["title"], "error": str(exc)})

    print()
    print("=" * 64)
    print("Deploy summary")
    print("=" * 64)
    for r in results:
        if "error" in r:
            print(f"  {r['title']:20} FAILED — {r['error']}")
        else:
            _print_result(r)


if __name__ == "__main__":
    main()
