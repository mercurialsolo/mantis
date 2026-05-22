"""Deploy mantis-boattrader to Daytona.

Builds a debian-slim image with our FastAPI dependencies, uploads the
``app/`` source tree, then starts uvicorn on port 8080 and prints the
public preview URL.

## Usage

    # 1. Make sure DAYTONA_API_KEY is set (already in repo .env).
    export $(grep -v '^#' .env | xargs)

    # 2. Install the SDK (one-time):
    uv pip install daytona python-dotenv

    # 3. Deploy:
    uv run python deploy/sim_envs/daytona_mantis_boattrader.py

The script prints:

    Deployed:
      URL:    https://<sandbox-host>/...
      Token:  <preview-token>
      Admin:  <generated-env-admin-token>

The URL is public; the harness can hit it via the ``/__env__/*``
endpoints by sending ``X-Env-Admin: <admin-token>`` (and for sandbox
previews the Daytona ``x-daytona-preview-token`` header).

Latency / failure injection is configurable via env vars set at create
time, or at runtime via ``POST /__env__/config`` with the admin token.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import secrets
import sys
import time
from typing import Iterable


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_ROOT = pathlib.Path(__file__).parent / "mantis_boattrader"
APP_ROOT = SRC_ROOT / "app"
PORT = 8080


def _load_env_from_repo() -> None:
    """Lightweight .env loader so we don't add python-dotenv as a hard dep.

    Walks up from this file looking for a ``.env``. When running inside a
    git worktree the worktree root won't carry .env — the original repo
    root will. We also probe ``git rev-parse --show-toplevel``'s parent
    in case the user keeps .env one level above.
    """
    candidates: list[pathlib.Path] = []
    here = pathlib.Path(__file__).resolve()
    for p in [here, *here.parents]:
        candidates.append(p / ".env")
    # Common shared location for worktree-aware setups: original repo root.
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
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            os.environ.setdefault(k, v)
        # First file wins — but iterate all to populate missing keys
        # only via setdefault above.


def _iter_source_files(root: pathlib.Path) -> Iterable[pathlib.Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        # Skip cache / hidden / binary artifacts.
        if any(p.startswith(".") or p in {"__pycache__"} for p in rel.parts):
            continue
        yield path


def deploy(latency_min_ms: int, latency_max_ms: int, failure_rate: float,
           keep_open: bool) -> str:
    _load_env_from_repo()
    api_key = os.environ.get("DAYTONA_API_KEY")
    if not api_key:
        sys.exit("error: DAYTONA_API_KEY not set in .env or environment")

    try:
        from daytona import (  # type: ignore[import-not-found]
            CreateSandboxFromImageParams,
            Daytona,
            DaytonaConfig,
            Image,
        )
    except ImportError:
        sys.exit("error: daytona SDK missing. Install with: uv pip install daytona")

    admin_token = os.environ.get("ENV_ADMIN_TOKEN") or secrets.token_urlsafe(24)

    image = (
        Image.debian_slim("3.11")
        .pip_install([
            "fastapi>=0.110",
            "uvicorn>=0.27",
            "jinja2>=3.1",
            "python-multipart>=0.0.9",
            "starlette>=0.27,<1.0",
        ])
        .workdir("/srv")
        .add_local_dir(str(APP_ROOT), "/srv/app")
        .env({
            "PORT": str(PORT),
            "PYTHONUNBUFFERED": "1",
            "ENV_ADMIN_TOKEN": admin_token,
            "SEED": os.environ.get("SEED", "42"),
            "FAKE_NOW": os.environ.get("FAKE_NOW", "2026-01-15T09:00:00Z"),
            "LATENCY_MS_MIN": str(latency_min_ms),
            "LATENCY_MS_MAX": str(latency_max_ms),
            "LATENCY_FAILURE_RATE": str(failure_rate),
        })
    )

    cfg = DaytonaConfig(api_key=api_key)
    daytona = Daytona(cfg)

    print(f"→ creating Daytona sandbox (image: debian-slim/python 3.11, port {PORT}) …")
    sandbox = daytona.create(
        CreateSandboxFromImageParams(image=image),
        timeout=300,
        on_snapshot_create_logs=lambda line: print(f"  [build] {line}"),
    )
    print(f"  sandbox id: {getattr(sandbox, 'id', '?')}")

    print("→ starting uvicorn …")
    cmd = (
        f"cd /srv && nohup python -m uvicorn app.main:app "
        f"--host 0.0.0.0 --port {PORT} "
        f"> /tmp/uvicorn.log 2>&1 &"
    )
    sandbox.process.exec(cmd)

    # Wait for /__env__/health to come up before printing the URL.
    print("→ waiting for health …")
    deadline = time.time() + 30
    health_ok = False
    while time.time() < deadline:
        try:
            res = sandbox.process.exec(
                f"curl -fs http://127.0.0.1:{PORT}/__env__/health || echo ___NOT_READY___"
            )
            out = getattr(res, "result", "") or ""
            if "ok" in out and "___NOT_READY___" not in out:
                health_ok = True
                break
        except Exception:
            pass
        time.sleep(1)

    if not health_ok:
        print("warning: health check did not return ok within 30s — uvicorn.log may explain")

    preview = sandbox.get_preview_link(PORT)
    print()
    print("Deployed mantis-boattrader on Daytona")
    print(f"  URL:    {preview.url}")
    print(f"  Token:  {getattr(preview, 'token', '<not exposed>')}")
    print(f"  Admin:  {admin_token}")
    print()
    print("Try:")
    print(f"  curl '{preview.url}/__env__/health' -H 'x-daytona-preview-token: <token>'")
    print(f"  open  '{preview.url}/'")
    print(f"  open  '{preview.url}/boats/?make=Sea+Ray&condition=used&sort=price-asc'")
    print(f"  open  '{preview.url}/boat/<slug>/'")
    print()
    if not keep_open:
        print("Sandbox will continue running. Stop it via Daytona dashboard when done.")
    return preview.url


def _upload(sandbox, remote_path: str, data: bytes) -> None:
    """Upload one file. The SDK has changed the filesystem API across
    versions, so we try the known shapes and surface a clear error if
    none match.
    """
    fs = getattr(sandbox, "fs", None)
    if fs is None:
        raise RuntimeError("sandbox.fs not present — SDK shape mismatch")
    # Signature: upload_file(src: str | bytes, dst: str). When src is
    # bytes, the SDK treats it as the file content directly.
    if hasattr(fs, "upload_file"):
        fs.upload_file(data, remote_path)
        return
    # Fallback: write_file
    if hasattr(fs, "write_file"):
        fs.write_file(remote_path, data)
        return
    raise RuntimeError("no compatible upload method on sandbox.fs")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deploy mantis-boattrader to Daytona")
    p.add_argument("--latency-min", type=int, default=120, help="Min injected latency in ms")
    p.add_argument("--latency-max", type=int, default=480, help="Max injected latency in ms")
    p.add_argument("--failure-rate", type=float, default=0.0,
                   help="Probability of returning 503 per request (0..1)")
    p.add_argument("--keep-open", action="store_true",
                   help="Leave the sandbox running (default)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    deploy(
        latency_min_ms=args.latency_min,
        latency_max_ms=args.latency_max,
        failure_rate=args.failure_rate,
        keep_open=args.keep_open,
    )
