"""Local runtime backend — boots a sim env on the developer's laptop.

Two boot modes, picked automatically:

1. **Docker** (preferred for real envs): if the env name resolves to a
   container image (registered in :func:`_image_for_env` or via the
   ``MANTIS_SIM_ENV_IMAGE_<NAME>`` env var) and ``docker`` is on PATH,
   we ``docker run`` the image with ``--network none``, port-published
   to a free local port, and the env vars (``SEED``, ``FAKE_NOW``,
   ``ENV_ADMIN_TOKEN``) populated.

2. **Subprocess** (used by the stub env + tests): if no image is
   registered, we boot ``python -m mantis_agent.sim_envs.stub_app`` as
   a subprocess. This keeps tests fast and lets the harness be
   exercised end-to-end without requiring a docker daemon.

Both modes return the same :class:`RuntimeHandle`; the caller (CLI,
benchmark, env_up.py) never has to know which mode was used.

Picking a free port: we ask the kernel for one (`socket.bind(("", 0))`)
then close the socket. There's a tiny race between close and the child
process binding; the health-wait loop retries health for 30s so a lost
race is recovered transparently. We do not try to reserve the port via
``SO_REUSEADDR`` magic — that buys little and adds complexity.

What this backend deliberately does NOT do:

* Manage a pool / cache of warm envs (v1.5 territory).
* Multi-tenant scoping inside one container (v1.5 — see #336).
* Talk to a remote docker host. ``DOCKER_HOST`` works because we shell
  out, but cross-machine docker is not something we test.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import shutil
import socket
import subprocess
import sys
import time
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from .runtime import RuntimeHandle

logger = logging.getLogger(__name__)


# ── image registry ─────────────────────────────────────────────────────


def _image_for_env(env_name: str) -> str | None:
    """Resolve an env name to a Docker image tag.

    Real envs land their own entry here in their per-env PRs. The
    canonical pattern is ``mantis/sim-env-<env>:latest`` extending the
    ``mantis/sim-env-base:latest`` shared base image (see #336 §9).

    The env var override ``MANTIS_SIM_ENV_IMAGE_<NAME>`` (uppercased,
    hyphens → underscores) lets a dev point at a freshly-built image
    without editing this table.

    Returns ``None`` for envs without a registered image — the local
    backend then falls back to the subprocess stub mode.
    """
    override_key = f"MANTIS_SIM_ENV_IMAGE_{env_name.upper().replace('-', '_')}"
    override = os.environ.get(override_key, "").strip()
    if override:
        return override
    # Canonical per-env images. Child env PRs append to this table.
    registry = {
        "mantis-auth": "mantis/sim-env-mantis-auth:latest",
        "mantis-crm": "mantis/sim-env-mantis-crm:latest",
        "mantis-helpdesk": "mantis/sim-env-mantis-helpdesk:latest",
        "mantis-shop": "mantis/sim-env-mantis-shop:latest",
    }
    return registry.get(env_name)


# ── port allocation ────────────────────────────────────────────────────


def _pick_free_port() -> int:
    """Ask the kernel for a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# ── http health poll ───────────────────────────────────────────────────


def _http_get_json(url: str, *, timeout: float = 1.0, headers: dict[str, str] | None = None) -> Any:
    req = Request(url, headers=headers or {})
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310 — local-only health check
        return json.loads(resp.read().decode("utf-8"))


# ── docker mode ────────────────────────────────────────────────────────


def _docker_start(
    image: str,
    *,
    port: int,
    admin_token: str,
    seed: int,
    now: str,
) -> str:
    """``docker run`` the image; return the container id."""
    cmd = [
        "docker", "run", "-d",
        "--rm",
        "--network", "none",  # no outbound egress, see #336 §"Open questions"
        "-p", f"127.0.0.1:{port}:8080",
        "-e", f"SEED={seed}",
        "-e", f"FAKE_NOW={now}",
        "-e", f"ENV_ADMIN_TOKEN={admin_token}",
        "-e", "PORT=8080",
        image,
    ]
    logger.info("local backend: docker run %s", image)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"docker run failed (image={image}): {proc.stderr.strip() or proc.stdout.strip()}"
        )
    container_id = proc.stdout.strip()
    if not container_id:
        raise RuntimeError("docker run returned empty container id")
    return container_id


def _docker_stop(container_id: str) -> None:
    if not container_id:
        return
    # ``docker stop`` is idempotent enough — if the container is already
    # gone we silently ignore. ``--rm`` on start auto-removes on stop.
    subprocess.run(
        ["docker", "stop", container_id],
        capture_output=True, text=True, check=False,
    )


# ── subprocess (stub) mode ─────────────────────────────────────────────


def _stub_subprocess_start(
    *,
    port: int,
    admin_token: str,
    seed: int,
    now: str,
) -> subprocess.Popen[bytes]:
    """Launch the stub env as a Python subprocess."""
    cmd = [
        sys.executable, "-m", "mantis_agent.sim_envs.stub_app",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--admin-token", admin_token,
        "--seed", str(seed),
        "--now", now,
    ]
    env = dict(os.environ)
    env["ENV_ADMIN_TOKEN"] = admin_token
    logger.info("local backend: subprocess stub on port %d", port)
    return subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ── backend ────────────────────────────────────────────────────────────


class LocalBackend:
    """Local-only :class:`RuntimeBackend` — Docker preferred, stub fallback."""

    name = "local"

    def start(
        self,
        env_name: str,
        *,
        seed: int = 42,
        now: str = "2026-01-15T09:00:00Z",
        admin_token: str | None = None,
    ) -> RuntimeHandle:
        token = admin_token or secrets.token_urlsafe(32)
        port = _pick_free_port()
        url = f"http://127.0.0.1:{port}"

        image = _image_for_env(env_name)
        if image and shutil.which("docker"):
            container_id = _docker_start(
                image, port=port, admin_token=token, seed=seed, now=now,
            )
            return RuntimeHandle(
                env_name=env_name,
                url=url,
                admin_token=token,
                backend=self.name,
                started_at=time.time(),
                extra={"mode": "docker", "container_id": container_id, "port": port},
            )

        # No image / no docker → subprocess stub mode. This is the path
        # the harness acceptance tests exercise; real envs land an image
        # in `_image_for_env` and switch to the docker branch.
        proc = _stub_subprocess_start(port=port, admin_token=token, seed=seed, now=now)
        return RuntimeHandle(
            env_name=env_name,
            url=url,
            admin_token=token,
            backend=self.name,
            started_at=time.time(),
            extra={"mode": "subprocess", "proc": proc, "port": port},
        )

    def wait_healthy(self, handle: RuntimeHandle, *, timeout_s: float = 180.0) -> None:
        """Poll ``/__env__/health`` until 200 or ``timeout_s`` elapses.

        Default bumped to 180 s after the 90 s ceiling continued flaking
        across PRs #449 and #453 (this session, 2026-05-17), each time
        on a different ``test_env_up.py`` sub-test that hit the
        FastAPI-cold-start race. History:

        - 60 s (#364) — flaked
        - 90 s (#???) — flaked on PRs #449, #453 in one investigation session
        - 180 s (current) — covers the slowest CI runners we've observed

        The poll loop exits as soon as the env responds 200, so the
        higher cap costs nothing on the happy path (healthy boots
        complete in 1-3 s) — it only changes whether we surface a
        noisy TimeoutError on a slow boot. The right place to invest
        further is making the stub env's first-paint faster (lazy
        FastAPI imports, deferred uvicorn worker bind), not bumping
        the cap a third time.
        """
        deadline = time.time() + timeout_s
        last_err: Exception | None = None
        while time.time() < deadline:
            try:
                payload = _http_get_json(f"{handle.url}/__env__/health", timeout=1.0)
                if isinstance(payload, dict) and payload.get("ok") is True:
                    return
                last_err = RuntimeError(f"health endpoint returned: {payload!r}")
            except (URLError, ConnectionError, OSError, json.JSONDecodeError) as exc:
                last_err = exc
            time.sleep(0.25)
        raise TimeoutError(
            f"env {handle.env_name!r} did not become healthy in {timeout_s:.0f}s "
            f"(last error: {last_err!r})"
        )

    def get_url(self, handle: RuntimeHandle) -> str:
        return handle.url

    def fetch_events(
        self,
        handle: RuntimeHandle,
        *,
        since: float | None = None,
    ) -> list[dict[str, Any]]:
        since_ts = since if since is not None else handle.started_at
        try:
            payload = _http_get_json(
                f"{handle.url}/__env__/events?since={since_ts}",
                timeout=5.0,
                headers={"X-Env-Admin": handle.admin_token},
            )
        except (URLError, ConnectionError, OSError, json.JSONDecodeError):
            return []
        if isinstance(payload, dict):
            events = payload.get("events", [])
            if isinstance(events, list):
                return events
        return []

    def stop(self, handle: RuntimeHandle) -> None:
        mode = handle.extra.get("mode")
        if mode == "docker":
            _docker_stop(handle.extra.get("container_id", ""))
            return
        if mode == "subprocess":
            proc: subprocess.Popen[bytes] | None = handle.extra.get("proc")
            if proc is None:
                return
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2.0)
            return
        # Unknown mode — nothing to do, but don't crash. ``stop`` is idempotent.
        logger.debug("local backend: unknown mode %r — stop is no-op", mode)
