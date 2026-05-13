"""env_up.py — local lifecycle wrapper for simulated envs.

Usage::

    python scripts/env_up.py start --env stub --seed 42
    # → prints JSON: {"url": "http://127.0.0.1:NNNN", "admin_token": "...", "handle_file": "..."}

    python scripts/env_up.py stop --handle <path-to-handle-file>

This is the thin CLI on top of :class:`LocalBackend`. ``start`` writes
the handle (env name, url, token, backend-specific state) to a JSON file
so a separate ``stop`` invocation in another shell can find it. The
default handle file lives under ``$TMPDIR/mantis-env-<env>-<pid>.json``
unless ``--handle`` overrides.

Most callers don't reach for this directly — ``mantis plan run --env
... --runtime local`` does start/stop in the same process. ``env_up.py``
is for the cases where you want to keep the env hot across multiple
manual invocations (debugging a flaky plan, poking the UI by hand).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

# Add ``src/`` to path so the script works both installed and from a
# fresh checkout without ``pip install -e .``.
_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if _REPO_SRC.exists() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from mantis_agent.sim_envs.local import LocalBackend  # noqa: E402
from mantis_agent.sim_envs.runtime import RuntimeHandle  # noqa: E402


def _default_handle_path(env_name: str) -> Path:
    return Path(tempfile.gettempdir()) / f"mantis-env-{env_name}-{os.getpid()}.json"


def _serialize_handle(handle: RuntimeHandle) -> dict[str, Any]:
    """Persistable view of the handle.

    The local backend's ``extra.proc`` is a ``Popen`` — not JSON-serialisable.
    We drop it from the on-disk view; ``stop`` reconstructs enough state
    from the container id / port to terminate the process group by PID
    (subprocess mode stores the pid; docker mode stores the container id).
    """
    extra = dict(handle.extra)
    proc = extra.pop("proc", None)
    if proc is not None and hasattr(proc, "pid"):
        extra["pid"] = proc.pid
    return {
        "env_name": handle.env_name,
        "url": handle.url,
        "admin_token": handle.admin_token,
        "backend": handle.backend,
        "started_at": handle.started_at,
        "extra": extra,
    }


def _stop_from_serialized(payload: dict[str, Any]) -> None:
    """Stop an env using only the JSON view of its handle.

    Subprocess: kill by PID. Docker: ``docker stop`` by container id.
    We avoid round-tripping through ``LocalBackend.stop`` because the
    Popen object is gone — we'd just re-implement the same two branches
    with a slightly different shape.
    """
    extra = payload.get("extra", {}) or {}
    mode = extra.get("mode")

    if mode == "docker":
        from mantis_agent.sim_envs.local import _docker_stop  # noqa: PLC0415

        _docker_stop(extra.get("container_id", ""))
        return

    if mode == "subprocess":
        pid = int(extra.get("pid") or 0)
        if pid > 0:
            try:
                os.kill(pid, 15)  # SIGTERM
            except ProcessLookupError:
                pass
        return

    print(f"warning: unknown mode {mode!r} — nothing to stop", file=sys.stderr)


# ── subcommands ─────────────────────────────────────────────────────────


def cmd_start(args: argparse.Namespace) -> int:
    backend = LocalBackend()
    handle = backend.start(args.env, seed=args.seed, now=args.now)
    try:
        backend.wait_healthy(handle, timeout_s=args.timeout)
    except TimeoutError as exc:
        backend.stop(handle)
        print(f"error: {exc}", file=sys.stderr)
        return 1

    handle_path = Path(args.handle) if args.handle else _default_handle_path(args.env)
    handle_path.write_text(json.dumps(_serialize_handle(handle), indent=2) + "\n")

    out = {
        "env": args.env,
        "url": handle.url,
        "admin_token": handle.admin_token,
        "handle_file": str(handle_path),
        "mode": handle.extra.get("mode"),
    }
    print(json.dumps(out, indent=2))
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    handle_path = Path(args.handle)
    if not handle_path.exists():
        print(f"error: handle file not found: {handle_path}", file=sys.stderr)
        return 1
    payload = json.loads(handle_path.read_text())
    _stop_from_serialized(payload)
    handle_path.unlink(missing_ok=True)
    print(json.dumps({"stopped": payload.get("env_name", "?")}))
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    """``env_up.py health --handle <path>`` — one-shot health check.

    Mostly a convenience for shell scripts driving env_up in a Makefile
    or CI step.
    """
    import urllib.request

    handle_path = Path(args.handle)
    payload = json.loads(handle_path.read_text())
    url = payload.get("url", "").rstrip("/")
    with urllib.request.urlopen(f"{url}/__env__/health", timeout=5.0) as resp:  # noqa: S310
        print(resp.read().decode("utf-8"))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="mantis sim env local lifecycle")
    sub = p.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="Boot an env and write a handle file")
    start.add_argument("--env", required=True, help="Env name, e.g. 'stub', 'mantis-crm'")
    start.add_argument("--seed", type=int, default=42)
    start.add_argument("--now", default="2026-01-15T09:00:00Z")
    start.add_argument("--timeout", type=float, default=30.0,
                       help="Health-wait timeout in seconds (default: 30)")
    start.add_argument("--handle", default=None,
                       help="Where to write the handle JSON (default: a tmp path)")
    start.set_defaults(func=cmd_start)

    stop = sub.add_parser("stop", help="Stop an env using a handle file from `start`")
    stop.add_argument("--handle", required=True)
    stop.set_defaults(func=cmd_stop)

    health = sub.add_parser("health", help="Hit /__env__/health on a running env")
    health.add_argument("--handle", required=True)
    health.set_defaults(func=cmd_health)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
