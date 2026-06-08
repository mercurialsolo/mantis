"""Deploy a mantis mirror sim env (mercor / fiverr / linkedin / indeed) to Daytona.

Generic counterpart to ``daytona_mantis_boattrader.py`` — same pattern,
parameterised on the env slug. Build a debian-slim image with the four
common FastAPI deps, upload the env's ``app/`` tree, start uvicorn on
8080, print the preview URL + admin token.

## Usage

    # 1. Make sure DAYTONA_API_KEY is set (already in repo .env).
    export $(grep -v '^#' .env | xargs)

    # 2. Install the SDK once:
    uv pip install daytona

    # 3. Deploy one env:
    uv run python deploy/sim_envs/daytona_mantis_mirror.py --env mercor

    # 4. Or all four in parallel via the driver below:
    uv run python deploy/sim_envs/daytona_mantis_mirror.py --env all
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import pathlib
import secrets
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PORT = 8080

ENVS = {
    "mercor":   {"dir": "mantis_mercor",   "title": "mantis-mercor"},
    "fiverr":   {"dir": "mantis_fiverr",   "title": "mantis-fiverr"},
    "linkedin": {"dir": "mantis_linkedin", "title": "mantis-linkedin"},
    "indeed":   {"dir": "mantis_indeed",   "title": "mantis-indeed"},
}


def _load_env_from_repo() -> None:
    here = pathlib.Path(__file__).resolve()
    candidates = [p / ".env" for p in [here, *here.parents]]
    main_root = pathlib.Path("/Users/barada/Sandbox/Mason/mantis/.env")
    if main_root not in candidates:
        candidates.append(main_root)
    for env_path in candidates:
        if not env_path.exists() or not env_path.is_file():
            continue
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def deploy(slug: str) -> dict:
    spec = ENVS[slug]
    src_root = pathlib.Path(__file__).parent / spec["dir"]
    app_root = src_root / "app"
    if not app_root.is_dir():
        raise SystemExit(f"error: {app_root} not found")

    _load_env_from_repo()
    api_key = os.environ.get("DAYTONA_API_KEY")
    if not api_key:
        raise SystemExit("error: DAYTONA_API_KEY not set in .env or environment")

    try:
        from daytona import (  # type: ignore[import-not-found]
            CreateSandboxFromImageParams,
            Daytona,
            DaytonaConfig,
            Image,
        )
    except ImportError:
        raise SystemExit("error: daytona SDK missing. Install with: uv pip install daytona")

    admin_token = secrets.token_urlsafe(24)

    image = (
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
            "SEED": os.environ.get("SEED", "42"),
            "FAKE_NOW": os.environ.get("FAKE_NOW", "2026-06-08T09:00:00Z"),
            "ENV_REQUIRE_AUTH": os.environ.get("ENV_REQUIRE_AUTH", "0"),
        })
    )

    cfg = DaytonaConfig(api_key=api_key)
    daytona = Daytona(cfg)

    print(f"[{slug}] → creating Daytona sandbox …")
    sandbox = daytona.create(
        CreateSandboxFromImageParams(image=image),
        timeout=300,
        on_snapshot_create_logs=lambda line: print(f"[{slug}]   [build] {line}"),
    )
    sid = getattr(sandbox, "id", "?")
    print(f"[{slug}]   sandbox id: {sid}")

    try:
        sandbox.set_autostop_interval(180)
    except Exception as exc:  # noqa: BLE001
        print(f"[{slug}]   warning: could not set autostop interval ({exc})")

    print(f"[{slug}] → starting uvicorn …")
    cmd = (
        f"cd /srv && nohup python -m uvicorn app.main:app "
        f"--host 0.0.0.0 --port {PORT} "
        f"> /tmp/uvicorn.log 2>&1 &"
    )
    sandbox.process.exec(cmd)

    print(f"[{slug}] → waiting for health …")
    deadline = time.time() + 45
    health_ok = False
    while time.time() < deadline:
        try:
            res = sandbox.process.exec(
                f"curl -fs http://127.0.0.1:{PORT}/__env__/health || echo ___NOT_READY___"
            )
            out = getattr(res, "result", "") or ""
            if '"ok"' in out and "___NOT_READY___" not in out:
                health_ok = True
                break
        except Exception:
            pass
        time.sleep(1.5)

    preview = sandbox.get_preview_link(PORT)
    url = preview.url
    token = getattr(preview, "token", None)

    return {
        "slug": slug,
        "title": spec["title"],
        "sandbox_id": sid,
        "url": url,
        "preview_token": token,
        "admin_token": admin_token,
        "health_ok": health_ok,
    }


def _print_result(r: dict) -> None:
    flag = "OK" if r["health_ok"] else "WARN (health did not return ok in 45s)"
    print()
    print(f"=== {r['title']} — {flag} ===")
    print(f"  URL:           {r['url']}")
    print(f"  Preview token: {r['preview_token']}")
    print(f"  Admin token:   {r['admin_token']}")
    print(f"  Sandbox id:    {r['sandbox_id']}")
    print(f"  Try:  curl '{r['url']}/__env__/health' -H 'x-daytona-preview-token: {r['preview_token']}'")
    print(f"        open  '{r['url']}/'")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deploy mantis mirror sim envs to Daytona")
    p.add_argument(
        "--env",
        required=True,
        choices=[*ENVS.keys(), "all"],
        help="Which env to deploy. 'all' deploys mercor+fiverr+linkedin+indeed in parallel.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    slugs = list(ENVS.keys()) if args.env == "all" else [args.env]

    results: list[dict] = []
    if len(slugs) == 1:
        results.append(deploy(slugs[0]))
    else:
        with cf.ThreadPoolExecutor(max_workers=len(slugs)) as ex:
            futs = {ex.submit(deploy, s): s for s in slugs}
            for fut in cf.as_completed(futs):
                s = futs[fut]
                try:
                    results.append(fut.result())
                except Exception as exc:
                    print(f"[{s}] FAILED: {exc}")
                    results.append({"slug": s, "title": ENVS[s]["title"], "error": str(exc)})

    print()
    print("================================================================")
    print("Deploy summary")
    print("================================================================")
    for r in results:
        if "error" in r:
            print(f"  {r['title']:20} FAILED — {r['error']}")
        else:
            _print_result(r)


if __name__ == "__main__":
    main()
