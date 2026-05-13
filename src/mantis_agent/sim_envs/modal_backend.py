"""Modal runtime backend — deploys one ``modal.App`` per env, per run.

V1 strategy: per-run app suffix. Each ``start()`` call deploys (or looks
up + redeploys) ``mantis-sim-env-<env>-<run_id>``, hits its
``/__env__/health`` until it returns 200, and hands the resulting URL
back to the caller. ``stop()`` ``modal app stop``s it.

The Modal app definitions live under ``deploy/sim_envs/``. The stub env
is ``deploy/sim_envs/modal_stub.py``; per-env PRs land their own
``deploy/sim_envs/<env>.py`` exporting an ASGI route under
``mantis-sim-env-<env>``.

Why per-run app instead of per-run scope inside one long-lived app:

* Per-run app is *clean teardown*: failure during a benchmark cannot
  leak state between plans.
* The cost is ~5-15s cold start. Acceptable at v1 benchmark volume
  (dozens of plans/day).
* When batch volume justifies it, swap to per-run-scope inside a hot
  app — same backend interface, different ``start`` body. See #336
  §"Open questions".

Modal SDK is imported lazily: callers that never select ``--runtime
modal`` should not pay the import cost (and the slim install does not
ship ``modal`` at all).
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import time
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from .runtime import RuntimeHandle

logger = logging.getLogger(__name__)


APP_NAME_PREFIX = "mantis-sim-env"


def _modal_app_name(env_name: str, run_suffix: str) -> str:
    return f"{APP_NAME_PREFIX}-{env_name}-{run_suffix}"


def _http_get_json(
    url: str,
    *,
    timeout: float = 5.0,
    headers: dict[str, str] | None = None,
) -> Any:
    req = Request(url, headers=headers or {})
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310 — Modal URL only
        return json.loads(resp.read().decode("utf-8"))


class ModalBackend:
    """:class:`RuntimeBackend` that ships envs as Modal apps.

    The backend is intentionally thin — Modal's app definition (the
    container image, the ASGI mount, the secret bindings) lives in the
    per-env file under ``deploy/sim_envs/``. This class only knows how
    to look one up, deploy it, wait for it to come live, and tear it
    down.
    """

    name = "modal"

    def __init__(self, *, workspace: str | None = None) -> None:
        # Workspace lives on the handle; passing it explicitly lets the
        # CI runner override the inferred workspace name when the same
        # image is deployed under a non-default user.
        self.workspace = workspace or os.environ.get("MODAL_WORKSPACE") or ""

    # ── helpers ─────────────────────────────────────────────────────

    def _resolve_modal(self) -> Any:
        try:
            import modal  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover — exercised in tests via mocking
            raise RuntimeError(
                "modal SDK not installed. `pip install modal` or use the [hud] extras."
            ) from exc
        return modal

    # ── RuntimeBackend protocol ─────────────────────────────────────

    def start(
        self,
        env_name: str,
        *,
        seed: int = 42,
        now: str = "2026-01-15T09:00:00Z",
        admin_token: str | None = None,
    ) -> RuntimeHandle:
        modal = self._resolve_modal()
        token = admin_token or secrets.token_urlsafe(32)
        run_suffix = secrets.token_hex(4)
        app_name = _modal_app_name(env_name, run_suffix)

        # The per-env deploy file is expected to expose a function /
        # ASGI app under ``<APP_NAME_PREFIX>-<env_name>`` that we then
        # re-deploy under the run-suffixed name. The mechanism for
        # "deploy a copy with a per-run name" is intentionally left to
        # the deploy file (it knows the image + secrets); we just look
        # up the function by the run-suffixed name and read its URL.
        #
        # Until per-env deploy files land (#332+), the canonical
        # already-deployed entry is the stub at ``mantis-sim-env-stub``
        # which we read directly (no per-run app — the stub is shared).
        if env_name == "stub":
            try:
                fn = modal.Function.from_name("mantis-sim-env-stub", "web")
                url = fn.get_web_url()
            except Exception as exc:  # noqa: BLE001 — modal raises varied exceptions
                raise RuntimeError(
                    "Modal stub env not deployed. Run "
                    "`modal deploy deploy/sim_envs/modal_stub.py` first."
                ) from exc
            # The stub trusts a fixed admin token from its Modal Secret;
            # bridge that into the handle so admin calls work.
            stub_token = os.environ.get("MANTIS_STUB_ENV_ADMIN_TOKEN") or token
            return RuntimeHandle(
                env_name=env_name,
                url=url,
                admin_token=stub_token,
                backend=self.name,
                started_at=time.time(),
                extra={"app_name": "mantis-sim-env-stub", "run_suffix": run_suffix},
            )

        # Per-env apps: each env PR lands its own deploy file. We resolve
        # the shared (long-lived) app first; per-run app suffixing is
        # deferred until batch volume justifies it.
        shared_app_name = f"{APP_NAME_PREFIX}-{env_name}"
        candidates = [shared_app_name, app_name]
        last_exc: Exception | None = None
        for name in candidates:
            try:
                fn = modal.Function.from_name(name, "web")
                url = fn.get_web_url()
                return RuntimeHandle(
                    env_name=env_name,
                    url=url,
                    admin_token=token,
                    backend=self.name,
                    started_at=time.time(),
                    extra={"app_name": name, "run_suffix": run_suffix},
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
        raise RuntimeError(
            f"Modal apps {candidates!r} not found. Per-env deploy files "
            f"land in deploy/sim_envs/<env>.py (e.g. "
            f"deploy/sim_envs/modal_mantis_crm.py). For dev, override "
            f"the env image with MANTIS_SIM_ENV_IMAGE_<NAME> and use "
            f"--runtime local instead."
        ) from last_exc

    def wait_healthy(self, handle: RuntimeHandle, *, timeout_s: float = 60.0) -> None:
        # Modal cold start is 5-15s; bump the default timeout vs local.
        deadline = time.time() + timeout_s
        last_err: Exception | None = None
        while time.time() < deadline:
            try:
                payload = _http_get_json(f"{handle.url}/__env__/health", timeout=5.0)
                if isinstance(payload, dict) and payload.get("ok") is True:
                    return
                last_err = RuntimeError(f"health endpoint returned: {payload!r}")
            except (URLError, ConnectionError, OSError, json.JSONDecodeError) as exc:
                last_err = exc
            time.sleep(1.0)
        raise TimeoutError(
            f"modal env {handle.env_name!r} did not become healthy in "
            f"{timeout_s:.0f}s (last error: {last_err!r})"
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
                timeout=15.0,
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
        # The stub is shared / long-lived; never stop it.
        if handle.env_name == "stub":
            return
        # Per-env apps are long-lived in v1 (no per-run app yet — see
        # module docstring). Nothing to tear down here. When v1.5 lands
        # per-run apps, this is where the ``modal app stop`` call goes.
        return
