"""Baseten workload server for Mantis CUA.

This module is used by the Baseten custom-server Trusses under ``deploy/baseten/``.
It starts a local llama.cpp server for either Holo3 or Gemma4, then exposes a
small FastAPI surface that runs the existing CUA task and micro-plan runners.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


import requests

try:
    from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
    from fastapi.responses import JSONResponse
    from starlette.concurrency import run_in_threadpool
except ImportError as exc:  # pragma: no cover - container-only deps
    raise ImportError(
        "mantis_agent.baseten_server requires fastapi + uvicorn. "
        "Install via: pip install -e '.[server]'  (or run inside the "
        "Baseten Truss image, which provisions them in build_commands)."
    ) from exc

from mantis_agent.api_schemas import (
    MAX_COST_USD,
    MAX_RUNTIME_MINUTES,
    PredictRequest,
    assert_hosts_allowed,
    extract_navigate_hosts,
)
from mantis_agent.gym.xdotool_env import XdotoolGymEnv
from mantis_agent.idempotency import get_idempotency_cache
from mantis_agent import metrics as mantis_metrics
from mantis_agent.rate_limit import get_rate_limiter
from mantis_agent.server_utils import (
    build_micro_result,
    build_micro_suite,
    build_proxy_config,
    micro_plan_steps_to_dicts,
    parse_lead_row,
    plan_signature_from_steps,
    result_summary,
    safe_state_key,
    save_result_json,
    start_local_proxy,
    utc_now,
    wait_for_openai_server,
    write_leads_csv,
)
from mantis_agent.tenant_auth import (
    DEFAULT_TENANT,
    TenantConfig,
    get_key_store,
)
from mantis_agent.webhooks import WebhookPayload, deliver_webhook_async

class _JsonLogFormatter(logging.Formatter):
    """One-line JSON-per-record formatter that attaches tenant_id and run_id
    when set in the process environment. Lets stdout consumers (Datadog,
    CloudWatch, Stackdriver) parse logs without ad-hoc regexes.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Per-request tenant context, set by /predict handler; empty in startup.
        tenant_id = os.environ.get("MANTIS_TENANT_ID")
        if tenant_id:
            payload["tenant_id"] = tenant_id
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _configure_logging() -> None:
    """One-time logging setup. JSON to stdout, level from LOG_LEVEL env."""
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    if os.environ.get("MANTIS_LOG_FORMAT", "json").lower() == "json":
        handler = logging.StreamHandler()
        handler.setFormatter(_JsonLogFormatter())
        root = logging.getLogger()
        root.handlers[:] = [handler]
        root.setLevel(level)
    else:
        logging.basicConfig(level=level)


_configure_logging()
logger = logging.getLogger("mantis_agent.baseten_server")

app = FastAPI(title="Mantis CUA Baseten Workload", docs_url=None, redoc_url=None)


SECRET_ENV_MAP = {
    "anthropic_api_key": "ANTHROPIC_API_KEY",
    "proxy_url": "PROXY_URL",
    "proxy_user": "PROXY_USER",
    "proxy_pass": "PROXY_PASS",
    "hf_access_token": "HF_TOKEN",
    "mantis_api_token": "MANTIS_API_TOKEN",
}

def _require_mantis_token(
    x_mantis_token: str | None = Header(default=None, alias="X-Mantis-Token"),
) -> TenantConfig:
    """Container-level auth → resolved TenantConfig.

    Uses a custom header (``X-Mantis-Token``) instead of ``Authorization: Bearer``
    so it does not collide with Baseten's gateway auth, which sends
    ``Authorization: Api-Key <baseten_key>`` to the container.

    Backwards-compat: if MANTIS_TENANT_KEYS_PATH is unset and MANTIS_API_TOKEN
    matches, returns DEFAULT_TENANT (single-tenant mode). Multi-tenant mode is
    enabled by mounting a JSON keys file and setting MANTIS_TENANT_KEYS_PATH.
    """
    store = get_key_store()
    if not store.is_multi_tenant and not os.environ.get("MANTIS_API_TOKEN", "").strip():
        raise HTTPException(status_code=503, detail="server auth not configured")
    if not x_mantis_token:
        raise HTTPException(status_code=401, detail="missing X-Mantis-Token header")
    tenant = store.resolve(x_mantis_token)
    if tenant is None:
        raise HTTPException(status_code=401, detail="invalid X-Mantis-Token")
    return tenant


def _require_run_scope(tenant: TenantConfig = Depends(_require_mantis_token)) -> TenantConfig:
    if not tenant.has_scope("run"):
        raise HTTPException(status_code=403, detail="tenant lacks 'run' scope")
    return tenant


def _read_secret(name: str) -> str:
    path = Path("/secrets") / name
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def _resolve_anthropic_key(tenant: TenantConfig) -> str:
    """Return the Anthropic key this tenant should use.

    Reads from the secret named by the tenant's `anthropic_secret_name`
    (each tenant can have its own Anthropic billing). Falls back to the
    legacy ANTHROPIC_API_KEY env var if the per-tenant secret isn't present
    on disk.
    """
    name = tenant.anthropic_secret_name or DEFAULT_TENANT.anthropic_secret_name
    value = _read_secret(name)
    if value:
        return value
    return os.environ.get("ANTHROPIC_API_KEY", "").strip()


def _load_secret_environment() -> None:
    for secret_name, env_name in SECRET_ENV_MAP.items():
        if os.environ.get(env_name):
            continue
        value = _read_secret(secret_name)
        if value:
            os.environ[env_name] = value


def _data_root() -> Path:
    """Top-level data dir. Per-tenant subdirs live under this."""
    root = Path(os.environ.get("MANTIS_DATA_DIR", "/workspace/mantis-data"))
    root.mkdir(parents=True, exist_ok=True)
    for child in ("results", "runs", "screenshots", "checkpoints", "chrome-profile", "tenants"):
        (root / child).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MANTIS_DEBUG_DIR", str(root / "screenshots" / "claude_debug"))
    return root


def _tenant_root(tenant: TenantConfig) -> Path:
    """Per-tenant subtree of the data volume. Caller cannot escape this prefix.

    Layout:
      /workspace/mantis-data/tenants/<tenant_id>/
        ├── runs/<run_id>/{status,result,leads,events}
        ├── checkpoints/<state_key>.json
        ├── chrome-profile/<state_key>/
        └── screenshots/<run_id>/
    """
    root = _data_root() / "tenants" / safe_state_key(tenant.tenant_id)
    for child in ("runs", "checkpoints", "chrome-profile", "screenshots"):
        (root / child).mkdir(parents=True, exist_ok=True)
    return root


def _tenant_state_key(tenant: TenantConfig, caller_state_key: str | None) -> str:
    """Server-namespaced state key. Caller's value is sanitized + prefixed."""
    base = safe_state_key(caller_state_key or "default")
    return f"{safe_state_key(tenant.tenant_id)}__{base}"


def _tenant_chrome_profile(tenant: TenantConfig, state_key: str) -> Path:
    """Per-tenant, per-state-key Chrome profile dir."""
    profile = _tenant_root(tenant) / "chrome-profile" / safe_state_key(state_key)
    profile.mkdir(parents=True, exist_ok=True)
    return profile


def _repo_root() -> Path:
    return Path(os.environ.get("MANTIS_REPO_ROOT", "/workspace/cua-agent"))


_safe_state_key = safe_state_key  # backward compat alias


_utc_now = utc_now  # backward compat alias


def _new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


class _DetachedRunLogHandler(logging.Handler):
    def __init__(self, runtime: "BasetenCUARuntime", run_id: str, thread_id: int) -> None:
        super().__init__(level=logging.INFO)
        self.runtime = runtime
        self.run_id = run_id
        self.thread_id = thread_id

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread != self.thread_id:
            return
        try:
            self.runtime._append_detached_event(self.run_id, self.format(record))
        except Exception:
            self.handleError(record)


_parse_lead_row = parse_lead_row  # backward compat alias


_write_leads_csv = write_leads_csv  # backward compat alias


_plan_signature_from_steps = plan_signature_from_steps  # backward compat alias


def _first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("none of these paths exist: " + ", ".join(str(p) for p in paths))


def _find_gguf(model_dir: Path, preferred: str = "") -> Path:
    if preferred:
        return _first_existing([Path(preferred), model_dir / preferred])

    candidates = [
        path
        for path in model_dir.glob("*.gguf")
        if "mmproj" not in path.name.lower()
    ]
    if not candidates:
        raise FileNotFoundError(f"no model GGUF found in {model_dir}")

    def rank(path: Path) -> tuple[int, str]:
        name = path.name.lower()
        if "q8_0" in name:
            return (0, name)
        if "q4_k_m" in name:
            return (1, name)
        return (2, name)

    return sorted(candidates, key=rank)[0]


def _find_mmproj(model_dir: Path, preferred: str = "") -> Path | None:
    if preferred:
        path = Path(preferred)
        return path if path.exists() else model_dir / preferred
    candidates = sorted(model_dir.glob("*mmproj*.gguf"))
    return candidates[0] if candidates else None


_wait_for_openai_server = wait_for_openai_server  # backward compat alias


_start_local_proxy = start_local_proxy  # backward compat alias


_build_proxy_config = build_proxy_config  # backward compat alias


class BasetenCUARuntime:
    def __init__(self) -> None:
        self.model_kind = os.environ.get("MANTIS_MODEL", "holo3")
        self.port = int(os.environ.get("MANTIS_LLAMA_PORT", "18080"))
        self.llama_proc: subprocess.Popen | None = None
        self._llama_log_fh: Any = None
        self.brain: Any = None
        self.lock = threading.Lock()
        self.detached_threads: dict[str, threading.Thread] = {}
        self.loaded = False

    def load(self) -> None:
        if self.loaded:
            return
        _load_secret_environment()
        data_root = _data_root()
        os.environ.setdefault("HF_HOME", str(data_root / "hf"))

        if self.model_kind == "holo3":
            self.brain = self._load_holo3()
        elif self.model_kind == "gemma4-cua":
            self.brain = self._load_gemma4()
        else:
            raise RuntimeError(f"unsupported MANTIS_MODEL={self.model_kind!r}")
        self.loaded = True

    def _start_llama(self, model_path: Path, mmproj_path: Path | None, extra_args: list[str]) -> None:
        cmd = [
            "/opt/llama.cpp/build/bin/llama-server",
            "-m", str(model_path),
            "--host", "0.0.0.0",
            "--port", str(self.port),
            "-ngl", "99",
            "-c", os.environ.get("MANTIS_CONTEXT_SIZE", "8192"),
            "-ub", os.environ.get("MANTIS_UBATCH_SIZE", "2048"),
            "--fit", "off",
        ]
        if mmproj_path:
            cmd.extend(["--mmproj", str(mmproj_path)])
        cmd.extend(extra_args)

        logger.info("starting llama.cpp: %s", " ".join(cmd))
        # Open the log file via context-managed handle that survives the
        # Popen call but is owned by the runtime so it's closed at shutdown
        # instead of leaking on every restart.
        self._llama_log_fh = open("/tmp/llama.log", "w")
        try:
            self.llama_proc = subprocess.Popen(
                cmd,
                stdout=self._llama_log_fh,
                stderr=subprocess.STDOUT,
            )
        except Exception:
            self._llama_log_fh.close()
            self._llama_log_fh = None
            raise
        _wait_for_openai_server(self.port, self.llama_proc, "llama.cpp")

    def _load_holo3(self) -> Any:
        from mantis_agent.brain_holo3 import Holo3Brain

        model_dir = Path(os.environ.get("MANTIS_HOLO3_MODEL_DIR", "/models/holo3"))
        model_path = _find_gguf(model_dir, os.environ.get("MANTIS_HOLO3_GGUF", ""))
        mmproj_path = _find_mmproj(model_dir, os.environ.get("MANTIS_HOLO3_MMPROJ", ""))
        self._start_llama(model_path, mmproj_path, ["--jinja", "--flash-attn", "on"])

        brain = Holo3Brain(
            base_url=f"http://127.0.0.1:{self.port}/v1",
            model="holo3",
            api_key="",
            max_tokens=2048,
            temperature=0.0,
            screen_size=(1280, 720),
            use_tool_calling=True,
        )
        brain.load()
        return brain

    def _load_gemma4(self) -> Any:
        from mantis_agent.brain_llamacpp import LlamaCppBrain

        model_dir = Path(os.environ.get("MANTIS_GEMMA4_MODEL_DIR", "/models/gemma4"))
        model_path = _find_gguf(model_dir, os.environ.get("MANTIS_GEMMA4_GGUF", ""))
        mmproj_path = _find_mmproj(model_dir, os.environ.get("MANTIS_GEMMA4_MMPROJ", ""))
        self._start_llama(
            model_path,
            mmproj_path,
            ["--jinja", "--reasoning-budget", "512", "--flash-attn", "on"],
        )

        brain = LlamaCppBrain(
            base_url=f"http://127.0.0.1:{self.port}/v1",
            model="gemma4-cua",
            max_tokens=512,
            temperature=0.0,
            use_tool_calling=True,
        )
        brain.load()
        return brain

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        action = str(payload.get("action") or payload.get("op") or "").lower()
        if action in {"status", "result", "logs"}:
            return self._detached_action(action, payload)
        if action == "graph_learn":
            return self._graph_learn(payload)

        self.load()
        if payload.get("detached"):
            return self._start_detached(payload)

        with self.lock:
            task_suite = self._task_suite_from_payload(payload)
            if task_suite.get("_micro_plan"):
                return self._run_micro(task_suite, payload)
            return self._run_tasks(task_suite, payload)

    def _graph_learn(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run graph learning phase (probe + graph generation).

        Does NOT require GPU — uses Claude screenshots only.
        Returns the compiled MicroPlan as a task_suite for execution.
        """
        from mantis_agent.graph import GraphLearner, GraphCompiler, GraphStore
        from mantis_agent.server_utils import micro_plan_steps_to_dicts

        objective_text = str(payload.get("objective") or payload.get("plan_text") or "")
        start_url = str(payload.get("start_url") or "")
        force = bool(payload.get("force_relearn", False))

        if not objective_text:
            raise ValueError("graph_learn requires 'objective' or 'plan_text'")

        data_root = _data_root()
        learner = GraphLearner(store=GraphStore(base_path=str(data_root / "graphs")))
        graph = learner.learn(objective_text, start_url=start_url, n_samples=0, force_relearn=force)

        compiler = GraphCompiler()
        micro_plan = compiler.compile(graph)

        steps_dicts = micro_plan_steps_to_dicts(micro_plan.steps)
        return {
            "mode": "graph_learn",
            "domain": graph.domain,
            "phases": len(graph.phases),
            "edges": len(graph.edges),
            "compiled_steps": len(micro_plan.steps),
            "objective_hash": graph.objective_hash[:12],
            "micro_plan": steps_dicts,
        }

    def _run_path(self, run_id: str, *, create: bool = False) -> Path:
        safe_run_id = _safe_state_key(run_id)
        if not safe_run_id or safe_run_id != run_id:
            raise ValueError(f"invalid run_id: {run_id!r}")
        path = _data_root() / "runs" / run_id
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def _write_json_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2))
        tmp_path.replace(path)

    def _read_json_file(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(str(path))
        return json.loads(path.read_text())

    def _write_detached_status(self, run_id: str, status: dict[str, Any]) -> dict[str, Any]:
        run_dir = self._run_path(run_id, create=True)
        status_path = run_dir / "status.json"
        existing: dict[str, Any] = {}
        if status_path.exists():
            try:
                existing = json.loads(status_path.read_text())
            except Exception:
                existing = {}

        merged = {
            **existing,
            **status,
            "run_id": run_id,
            "updated_at": _utc_now(),
            "status_path": str(status_path),
            "result_path": str(run_dir / "result.json"),
            "csv_path": str(run_dir / "leads.csv"),
            "events_path": str(run_dir / "events.log"),
        }
        self._write_json_atomic(status_path, merged)
        return merged

    def _append_detached_event(self, run_id: str, message: str) -> None:
        run_dir = self._run_path(run_id, create=True)
        line = json.dumps({"ts": _utc_now(), "message": message}, sort_keys=True)
        with (run_dir / "events.log").open("a") as handle:
            handle.write(line + "\n")

    def _start_detached(self, payload: dict[str, Any]) -> dict[str, Any]:
        run_id = _safe_state_key(str(payload.get("run_id") or _new_run_id()))
        if run_id in self.detached_threads and self.detached_threads[run_id].is_alive():
            raise RuntimeError(f"detached run already exists and is active: {run_id}")

        run_payload = dict(payload)
        run_payload.pop("detached", None)
        run_payload["_detached_run_id"] = run_id
        run_payload["_detached_started_at"] = _utc_now()

        status = self._write_detached_status(
            run_id,
            {
                "status": "queued",
                "created_at": run_payload["_detached_started_at"],
                "model": self.model_kind,
                "mode": "detached",
                "payload": {
                    key: value
                    for key, value in run_payload.items()
                    if key not in {"task_file_contents"}
                },
            },
        )
        self._append_detached_event(run_id, "queued")

        thread = threading.Thread(
            target=self._run_detached_worker,
            args=(run_id, run_payload),
            name=f"baseten-detached-{run_id}",
            daemon=True,
        )
        self.detached_threads[run_id] = thread
        thread.start()
        return status

    def _run_detached_worker(self, run_id: str, payload: dict[str, Any]) -> None:
        handler = _DetachedRunLogHandler(self, run_id, threading.get_ident())
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        agent_logger = logging.getLogger("mantis_agent")
        agent_logger.addHandler(handler)
        try:
            self._append_detached_event(run_id, "waiting_for_runtime_lock")
            with self.lock:
                self._append_detached_event(run_id, "running")
                self._write_detached_status(run_id, {"status": "running", "started_at": _utc_now()})
                task_suite = self._task_suite_from_payload(payload)
                if task_suite.get("_micro_plan"):
                    result = self._run_micro(task_suite, payload, run_id=run_id)
                else:
                    result = self._run_tasks(task_suite, payload, run_id=run_id)
                self._save_detached_result(run_id, result)
                self._write_detached_status(
                    run_id,
                    {
                        "status": "succeeded",
                        "finished_at": _utc_now(),
                        "summary": self._result_summary(result),
                    },
                )
                self._append_detached_event(run_id, "succeeded")
        except Exception as exc:
            logger.exception("detached run %s failed", run_id)
            self._write_detached_status(
                run_id,
                {
                    "status": "failed",
                    "finished_at": _utc_now(),
                    "error": str(exc),
                    "traceback": traceback.format_exc()[-4000:],
                },
            )
            self._append_detached_event(run_id, f"failed: {exc}")
        finally:
            agent_logger.removeHandler(handler)
            handler.close()

    def _save_detached_result(self, run_id: str, result: dict[str, Any]) -> None:
        run_dir = self._run_path(run_id, create=True)
        run_result_path = run_dir / "result.json"
        result["detached_result_path"] = str(run_result_path)
        leads = result.get("leads")
        if isinstance(leads, list):
            csv_path = run_dir / "leads.csv"
            _write_leads_csv(csv_path, leads)
            result["detached_csv_path"] = str(csv_path)
        self._write_json_atomic(run_result_path, result)

    def _result_summary(self, result: dict[str, Any]) -> dict[str, Any]:
        return result_summary(result)

    def _detached_action(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        run_id = str(payload.get("run_id") or "")
        if not run_id:
            raise ValueError("run_id is required")
        run_dir = self._run_path(run_id)

        if action == "status":
            status = self._read_json_file(run_dir / "status.json")
            thread = self.detached_threads.get(run_id)
            if thread and thread.is_alive() and status.get("status") not in {"running", "queued"}:
                status["in_memory_thread_alive"] = True
            return status

        if action == "result":
            result_path = run_dir / "result.json"
            if result_path.exists():
                return self._read_json_file(result_path)
            status = self._read_json_file(run_dir / "status.json")
            return {"run_id": run_id, "status": status.get("status", "unknown"), "result_ready": False}

        tail = int(payload.get("tail", 200))
        events_path = run_dir / "events.log"
        if not events_path.exists():
            return {"run_id": run_id, "events": []}
        lines = events_path.read_text(errors="ignore").splitlines()[-tail:]
        return {"run_id": run_id, "events": lines}

    def _task_suite_from_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if "task_suite" in payload:
            return dict(payload["task_suite"])
        if "task_file_contents" in payload:
            contents = payload["task_file_contents"]
            return json.loads(contents) if isinstance(contents, str) else dict(contents)
        if "task_file" in payload:
            path = self._resolve_path(payload["task_file"])
            return json.loads(path.read_text())
        # plan_text: free-text → PlanDecomposer.decompose_text → micro suite.
        # Documented in onboarding docs as the one-shot ad-hoc shape; build the
        # suite here so callers get the same dispatch as the file-based path.
        plan_text = payload.get("plan_text")
        if plan_text:
            return self._micro_suite_from_text(str(plan_text), payload)

        micro_path = (
            payload.get("micro")
            or payload.get("micro_path")
            or os.environ.get("MANTIS_DEFAULT_MICRO", "")
        )
        if not micro_path:
            raise ValueError(
                "Request must provide one of: 'plan_text', 'micro', "
                "'micro_path', or set MANTIS_DEFAULT_MICRO on the deployment. "
                "No default plan ships with the public image."
            )
        return self._micro_suite_from_path(str(micro_path), payload)

    def _micro_suite_from_text(self, plan_text: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Decompose free-text → MicroPlan → suite. Used by the plan_text shape."""
        from mantis_agent.plan_decomposer import PlanDecomposer

        cache_dir = _data_root() / "plan_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_template = str(cache_dir / "decomposed_{hash}.json")

        decomposer = PlanDecomposer()
        micro_plan = decomposer.decompose_text(
            plan_text, cache_path_template=cache_template,
        )

        steps_dicts = micro_plan_steps_to_dicts(micro_plan.steps)
        data_root = _data_root()
        return build_micro_suite(
            steps_dicts,
            micro_plan.domain or "plan_text",
            max_cost=float(payload.get("max_cost", 10.0)),
            max_time_minutes=int(payload.get("max_time_minutes", 180)),
            resume_state=bool(payload.get("resume_state", False)),
            state_key=str(payload.get("state_key") or ""),
            checkpoint_dir=str(data_root / "checkpoints"),
            proxy_city=str(payload.get("proxy_city") or os.environ.get("MANTIS_PROXY_CITY", "")),
            proxy_state=str(payload.get("proxy_state") or os.environ.get("MANTIS_PROXY_STATE", "")),
            proxy_disabled=bool(payload.get("proxy_disabled", False)),
        )

    def _resolve_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path
        candidates = [_repo_root() / path]
        if path.parts and path.parts[0] == "plans":
            candidates.append(_repo_root().joinpath(*path.parts[1:]))
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _micro_suite_from_path(self, raw_path: str, payload: dict[str, Any]) -> dict[str, Any]:
        from mantis_agent.plan_decomposer import MicroIntent, MicroPlan, PlanDecomposer

        path = self._resolve_path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"micro plan not found: {path}")

        objective_dict = None

        if path.suffix == ".json":
            raw_steps = json.loads(path.read_text())
            domain = path.stem
            micro_plan = MicroPlan(domain=domain)
            for step in raw_steps:
                micro_plan.steps.append(PlanDecomposer._build_intent(step))
        else:
            # Text plan: embed raw text + heuristic objective for the detached thread.
            # The actual decomposition/enhancement happens inside _run_micro where
            # there's no request timeout and the browser env is available for probing.
            plan_text = path.read_text()
            objective_dict = None
            try:
                from mantis_agent.graph.objective import ObjectiveSpec
                obj = ObjectiveSpec._parse_heuristic(plan_text)
                objective_dict = obj.to_dict()
                objective_dict["_raw_plan_text"] = plan_text
                objective_dict["_plan_path"] = str(path)
            except Exception:
                pass
            # When the deployment ships with a baked-in default plan, prefer
            # it as a starting point — the detached thread will replace it
            # with an enhanced plan via graph learning. When no default is
            # configured, the minimal navigate-only fallback below kicks in.
            default_micro = os.environ.get("MANTIS_DEFAULT_MICRO", "")
            fallback_candidates: list[Path] = []
            if default_micro:
                fallback_candidates.append(self._resolve_path(default_micro))
                fallback_candidates.append(Path(default_micro))
            fallback_loaded = False
            for fallback_path in fallback_candidates:
                if fallback_path.exists():
                    try:
                        raw_steps = json.loads(fallback_path.read_text())
                        domain = obj.domains[0] if objective_dict and obj.domains else "unknown"
                        micro_plan = MicroPlan(domain=domain)
                        for step in raw_steps:
                            micro_plan.steps.append(PlanDecomposer._build_intent(step))
                        fallback_loaded = True
                        logger.info("Baseten: loaded fallback plan from %s", fallback_path)
                        break
                    except Exception:
                        continue
            if not fallback_loaded:
                # Last resort: create a minimal navigate-only plan from the objective
                domain = obj.domains[0] if objective_dict and obj.domains else "unknown"
                start_url = obj.start_url if objective_dict else "about:blank"
                micro_plan = MicroPlan(domain=domain, steps=[
                    MicroIntent(intent=f"Navigate to {start_url}", type="navigate", budget=3, section="setup", required=True),
                ])
                logger.warning("Baseten: no fallback plan found, using minimal navigate-only plan")

        steps_dicts = micro_plan_steps_to_dicts(micro_plan.steps)
        data_root = _data_root()
        suite = build_micro_suite(
            steps_dicts,
            micro_plan.domain,
            max_cost=float(payload.get("max_cost", 10.0)),
            max_time_minutes=int(payload.get("max_time_minutes", 180)),
            resume_state=bool(payload.get("resume_state", False)),
            state_key=str(payload.get("state_key") or ""),
            checkpoint_dir=str(data_root / "checkpoints"),
            proxy_city=str(payload.get("proxy_city") or os.environ.get("MANTIS_PROXY_CITY", "")),
            proxy_state=str(payload.get("proxy_state") or os.environ.get("MANTIS_PROXY_STATE", "")),
            proxy_disabled=bool(payload.get("proxy_disabled", False)),
            objective=objective_dict,
        )
        return suite

    def _make_env(self, task_suite: dict[str, Any], run_id: str, settle_time: float) -> tuple[XdotoolGymEnv, Any]:
        from mantis_agent.task_loop import setup_env

        data_root = _data_root()
        session_name = task_suite.get("session_name", "baseten_cua")
        return setup_env(
            base_url=task_suite.get("base_url", ""),
            run_id=run_id,
            session_name=session_name,
            settle_time=settle_time,
            proxy_city=str(task_suite.get("_proxy_city") or ""),
            proxy_state=str(task_suite.get("_proxy_state") or ""),
            proxy_disabled=bool(task_suite.get("_proxy_disabled", False)),
            browser=os.environ.get("MANTIS_BROWSER", "google-chrome"),
            profile_dir=str(data_root / "chrome-profile"),
            save_screenshots_dir=str(data_root / "screenshots"),
        )

    def _maybe_record(
        self, payload: dict[str, Any], run_id: str
    ) -> tuple[Any, Any]:
        """Spawn a ScreenRecorder if payload.record_video is set.

        Returns ``(recorder, click_log)`` so the caller can wrap the env
        with a ``ClickRecordingEnv`` to capture click coordinates that
        feed the polished video's ripple animations. Either may be None
        when recording is disabled.
        """
        if not payload.get("record_video"):
            return (None, None)
        from mantis_agent.presentation import ClickEventLog
        from mantis_agent.recorder import ScreenRecorder

        tenant_id = safe_state_key(
            os.environ.get("MANTIS_TENANT_ID", DEFAULT_TENANT.tenant_id)
        )
        runs_dir = _data_root() / "tenants" / tenant_id / "runs" / safe_state_key(run_id)
        runs_dir.mkdir(parents=True, exist_ok=True)
        fmt = str(payload.get("video_format", "mp4"))
        output = runs_dir / f"recording.{fmt}"
        rec = ScreenRecorder(
            output=output,
            fps=int(payload.get("video_fps", 5)),
            fmt=fmt,  # type: ignore[arg-type]
        )
        click_log = ClickEventLog()
        if not rec.start():
            logger.warning(
                "recorder requested but failed to start: %s",
                rec.result.error if rec.result else "unknown",
            )
            return (rec, click_log)
        # Re-anchor the click log to the moment ffmpeg started capturing,
        # so click timestamps align with the raw video timeline.
        if hasattr(rec, "_started_at") and rec._started_at:
            click_log._anchor = rec._started_at
        return (rec, click_log)

    def _attach_recording_metadata(
        self, result: dict[str, Any], recorder: Any, click_log: Any = None,
    ) -> None:
        if not recorder:
            return
        rec_result = recorder.stop()
        actions = {
            "clicks": len(getattr(click_log, "clicks", []) if click_log else []),
            "keys": len(getattr(click_log, "keys", []) if click_log else []),
            "types": len(getattr(click_log, "types", []) if click_log else []),
            "scrolls": len(getattr(click_log, "scrolls", []) if click_log else []),
            "drags": len(getattr(click_log, "drags", []) if click_log else []),
        }
        result["video"] = {
            "path": str(rec_result.output_path) if rec_result.output_path else None,
            "format": result.get("video_format")
            or (recorder._fmt if hasattr(recorder, "_fmt") else "mp4"),
            "duration_seconds": round(rec_result.duration_seconds, 2),
            "bytes": rec_result.bytes_written,
            "error": rec_result.error,
            "actions": actions,
            "clicks": actions["clicks"],  # backwards-compat field
        }
        # Polish the raw recording into a feature-walkthrough video.
        if rec_result.succeeded and rec_result.output_path:
            polished_path = self._polish_recording(
                raw_video=rec_result.output_path,
                result=result,
                recorder=recorder,
                click_log=click_log,
            )
            if polished_path is not None:
                result["video"]["polished_path"] = str(polished_path)

    def _polish_recording(
        self, raw_video: Any, result: dict[str, Any], recorder: Any,
        click_log: Any = None,
    ) -> Any:
        """Compose title + raw-with-captions + outro into a polished video.

        Always best-effort — if any step fails (PIL, ffmpeg, missing fonts),
        the raw recording is preserved and the run still succeeds.
        """
        from mantis_agent import presentation

        try:
            tenant_id = os.environ.get("MANTIS_TENANT_ID", "default")
            run_id = result.get("run_id") or "unknown"
            session_name = result.get("session_name") or "run"
            fmt = recorder._fmt if hasattr(recorder, "_fmt") else "mp4"
            width = recorder._width if hasattr(recorder, "_width") else 1280
            height = recorder._height if hasattr(recorder, "_height") else 720
            run_dir = raw_video.parent
            polished_path = run_dir / f"recording_polished.{fmt}"

            title_card_cfg = presentation.title_card_for_run(
                plan_label=session_name,
                tenant_id=tenant_id,
                run_id=run_id,
            )
            title_card_path = run_dir / "_title.png"
            presentation.write_card(title_card_path, width, height, title_card_cfg)

            summary = result.get("summary") or {}
            outro_cfg = presentation.outro_card_from_summary(
                summary=summary,
                plan_label=session_name,
                cost_total=result.get("cost_total"),
                duration_seconds=result.get("elapsed_seconds"),
            )
            outro_card_path = run_dir / "_outro.png"
            presentation.write_card(outro_card_path, width, height, outro_cfg)

            srt_path = None
            timings = self._collect_step_timings(result)
            if timings:
                # Title card runs for ~3s before the agent footage starts;
                # offset captions accordingly.
                captions = presentation.captions_from_step_timings(
                    timings, title_offset=0.0,
                )
                srt_path = run_dir / "_captions.srt"
                srt_path.write_text(presentation.captions_to_srt(captions), encoding="utf-8")

            # Action overlay (works for any computer-use action —
            # browser, file manager, terminal, dialog, anything visible
            # on the Xvfb display). Click ripples + keyboard chord
            # badges + scroll arrows + type captions + drag trails all
            # composite into the same PNG sequence.
            ripples_dir: Any = None
            if click_log is not None and len(click_log) > 0:
                ripples_dir = run_dir / "_ripples"
                duration = (
                    recorder.result.duration_seconds
                    if recorder.result and recorder.result.duration_seconds
                    else 60.0
                )
                ripples_dir = presentation.render_action_overlay_pngs(
                    ripples_dir,
                    duration_seconds=duration,
                    fps=30,
                    width=width,
                    height=height,
                    clicks=getattr(click_log, "clicks", click_log.events),
                    keys=getattr(click_log, "keys", []),
                    types=getattr(click_log, "types", []),
                    scrolls=getattr(click_log, "scrolls", []),
                    drags=getattr(click_log, "drags", []),
                )

            ok = presentation.compose_polished_video(
                raw_video=raw_video,
                title_card=title_card_path,
                outro_card=outro_card_path,
                subtitles_srt=srt_path,
                ripples_dir=ripples_dir,
                output=polished_path,
                width=width,
                height=height,
                fmt=fmt,
            )
            return polished_path if ok else None
        except Exception as exc:  # noqa: BLE001 — polish is best-effort
            logger.warning("polished video compose failed: %s", exc)
            return None

    @staticmethod
    def _collect_step_timings(
        result: dict[str, Any],
    ) -> list[tuple[float, str, str]]:
        """Pull (elapsed_seconds, intent, status) from a result for SRT.

        Walks ``result["steps"]`` (micro-runner output). Each entry is
        expected to expose ``intent``, ``success``, and a ``duration`` (or
        ``elapsed``) float. Steps with no intent are skipped.
        """
        steps = result.get("steps") or []
        if not isinstance(steps, list):
            return []
        timings: list[tuple[float, str, str]] = []
        cum = 0.0
        for s in steps:
            if not isinstance(s, dict):
                continue
            intent = (s.get("intent") or "").strip()
            if not intent:
                continue
            duration = float(s.get("duration") or s.get("elapsed") or 0.0)
            status = "completed" if s.get("success") else "failed"
            timings.append((cum, intent, status))
            cum += max(duration, 0.0)
        return timings

    def _run_micro(
        self,
        task_suite: dict[str, Any],
        payload: dict[str, Any],
        run_id: str | None = None,
    ) -> dict[str, Any]:
        from mantis_agent.extraction import ClaudeExtractor
        from mantis_agent.grounding import ClaudeGrounding
        from mantis_agent.gym.micro_runner import MicroPlanRunner
        from mantis_agent.plan_decomposer import MicroIntent, MicroPlan

        run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        t0 = time.time()
        session_name = task_suite.get("session_name", "baseten_micro")
        env, proxy_proc = self._make_env(
            task_suite,
            run_id,
            settle_time=4.0 if self.model_kind == "holo3" else 2.0,
        )
        recorder, click_log = self._maybe_record(payload, run_id)
        if click_log is not None:
            from mantis_agent.presentation import ClickRecordingEnv
            env = ClickRecordingEnv(env, click_log)

        try:
            micro_plan = MicroPlan(domain=session_name)
            for step in task_suite["_micro_plan"]:
                micro_plan.steps.append(MicroIntent(**step))

            # If objective available, probe site inside container and re-enhance
            objective_data = task_suite.get("_objective")
            if objective_data and env:
                try:
                    from mantis_agent.graph.objective import ObjectiveSpec as _OS
                    from mantis_agent.graph.probe import SiteProber
                    from mantis_agent.graph.enhancer import PlanEnhancer
                    from mantis_agent.graph import GraphCompiler, PlanValidator
                    from mantis_agent.graph.graph import WorkflowGraph
                    from mantis_agent.verification.playbook import Playbook as _PB

                    obj = _OS.from_dict(objective_data)
                    if obj.start_url:
                        logger.info("Baseten: probing %s inside container", obj.start_url)
                        prober = SiteProber(env=env)
                        probe = prober.probe(obj.start_url, obj)
                        enhancer = PlanEnhancer()
                        enhancement = enhancer.enhance(obj, probe)
                        phases, edges = enhancer.build_enhanced_phases(obj, probe, enhancement)
                        graph = WorkflowGraph(
                            objective=obj, phases=phases, edges=edges,
                            playbook=_PB(domain=obj.domains[0] if obj.domains else "", listings_per_page=probe.estimated_listings_per_page or 25),
                            domain=obj.domains[0] if obj.domains else "",
                            objective_hash=obj.objective_hash,
                        )
                        compiler = GraphCompiler()
                        micro_plan = compiler.compile(graph)
                        validator = PlanValidator()
                        issues = validator.validate(micro_plan, objective=obj)
                        if issues:
                            micro_plan = validator.enhance(micro_plan, objective=obj)
                        logger.info("Baseten: probe-enhanced plan: %d steps", len(micro_plan.steps))
                except Exception as e:
                    logger.warning("Baseten: probe enhancement failed: %s", e)

            grounding = ClaudeGrounding()
            schema = None
            objective_data = task_suite.get("_objective")
            if objective_data:
                from mantis_agent.graph.objective import ObjectiveSpec
                from mantis_agent.extraction import ExtractionSchema
                objective = ObjectiveSpec.from_dict(objective_data)
                schema = ExtractionSchema.from_objective(objective)
            extractor = ClaudeExtractor(schema=schema)
            resume_state = bool(task_suite.get("_resume_state", False))
            checkpoint_path = task_suite.get("_checkpoint_path")
            runner = MicroPlanRunner(
                brain=self.brain,
                env=env,
                grounding=grounding,
                extractor=extractor,
                checkpoint_path=checkpoint_path,
                run_key=task_suite.get("_state_key", session_name),
                session_name=session_name,
                plan_signature=task_suite.get("_plan_signature", ""),
                resume_state=resume_state,
                max_cost=task_suite.get("_max_cost", 10.0),
                max_time_minutes=task_suite.get("_max_time_minutes", 180),
            )
            step_results = runner.run(micro_plan, resume=resume_state)
            result = build_micro_result(
                runner,
                step_results,
                run_id=run_id,
                provider="baseten",
                session_name=session_name,
                model_name=self.model_kind,
                elapsed_seconds=time.time() - t0,
                state_key=task_suite.get("_state_key", ""),
                checkpoint_path=checkpoint_path,
                plan_signature=task_suite.get("_plan_signature", ""),
                resume_state=resume_state,
            )
            self._attach_recording_metadata(result, recorder, click_log=click_log)
            self._save_result(result, prefix=self.model_kind.replace("-", "_"))
            return result
        finally:
            # Stop the recorder BEFORE closing env so the final frames flush
            # while the Xvfb display still exists. _attach_recording_metadata
            # is idempotent (ScreenRecorder.stop() is locked) so calling it
            # again here covers the exception path.
            if recorder is not None and recorder.result is None:
                try:
                    recorder.stop()
                except Exception:
                    logger.exception("recorder stop in finally failed")
            env.close()
            if proxy_proc:
                proxy_proc.terminate()

    def _run_tasks(
        self,
        task_suite: dict[str, Any],
        payload: dict[str, Any],
        run_id: str | None = None,
    ) -> dict[str, Any]:
        from mantis_agent.grounding import ClaudeGrounding
        from mantis_agent.task_loop import TaskLoopConfig, run_executor_lifecycle

        run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        t0 = time.time()
        session_name = task_suite.get("session_name", "baseten_tasks")
        env, proxy_proc = self._make_env(
            task_suite,
            run_id,
            settle_time=4.0 if self.model_kind == "holo3" else 2.0,
        )
        recorder, click_log = self._maybe_record(payload, run_id)
        if click_log is not None:
            from mantis_agent.presentation import ClickRecordingEnv
            env = ClickRecordingEnv(env, click_log)
        grounding = ClaudeGrounding()

        try:
            config = TaskLoopConfig(
                run_id=run_id,
                session_name=session_name,
                model_name=self.model_kind,
                results_prefix=self.model_kind.replace("-", "_"),
                brain=self.brain,
                env=env,
                grounding=grounding,
                max_steps=int(payload.get("max_steps", 30)),
                frames_per_inference=1 if self.model_kind == "holo3" else 2,
                results_dir=str(_data_root() / "results"),
            )
            result = run_executor_lifecycle(
                task_suite, config,
                proxy_proc=proxy_proc, t0=t0,
            )
            result["provider"] = "baseten"
            self._attach_recording_metadata(result, recorder, click_log=click_log)
            self._save_result(result, prefix=self.model_kind.replace("-", "_"))
            return result
        finally:
            if recorder is not None and recorder.result is None:
                try:
                    recorder.stop()
                except Exception:
                    logger.exception("recorder stop in finally failed")

    def _save_result(self, result: dict[str, Any], prefix: str) -> None:
        save_result_json(result, _data_root() / "results", prefix)


runtime = BasetenCUARuntime()


@app.on_event("startup")
def startup() -> None:
    runtime.load()


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": runtime.loaded, "model": runtime.model_kind}


@app.get("/v1/health")
def health_v1() -> dict[str, Any]:
    """Versioned alias for /health.

    The unversioned /health endpoint is what platform liveness probes target;
    /v1/health is the same payload available under the public API path.
    """
    return {"ok": runtime.loaded, "model": runtime.model_kind}


@app.get("/v1/runs/{run_id}/video")
def get_run_video(
    run_id: str,
    request: Request,
    tenant: TenantConfig = Depends(_require_mantis_token),
) -> Any:
    """Download the screencast for a run.

    Default: serves the **polished** version (title card + captioned run
    footage + outro card). Pass ``?raw=1`` to fetch the raw screencast
    without overlays.

    Resolves to the per-tenant run dir; returns 404 if no recording exists
    (recording wasn't requested or ffmpeg failed). Auth requires a valid
    token but not specifically the ``run`` scope.
    """
    from fastapi.responses import FileResponse
    from mantis_agent.recorder import content_type_for

    raw_only = request.query_params.get("raw", "").lower() in {"1", "true", "yes"}

    safe_run_id = safe_state_key(run_id)
    tenant_dir = _data_root() / "tenants" / safe_state_key(tenant.tenant_id)
    runs_dir = tenant_dir / "runs" / safe_run_id

    # Prefer polished by default; fall back to raw if polished is missing
    # (e.g., ffmpeg compose failed). ?raw=1 skips polished entirely.
    prefixes = ("recording",) if raw_only else ("recording_polished", "recording")
    for prefix in prefixes:
        for fmt in ("mp4", "webm", "gif"):
            candidate = runs_dir / f"{prefix}.{fmt}"
            if candidate.exists() and candidate.stat().st_size > 0:
                return FileResponse(
                    candidate,
                    media_type=content_type_for(fmt),  # type: ignore[arg-type]
                    filename=f"{safe_run_id}.{fmt}",
                )
    raise HTTPException(
        status_code=404,
        detail="no recording for this run "
        "(record_video=true on /v1/predict required)",
    )


@app.get("/metrics")
def metrics_endpoint() -> Any:
    """Prometheus scrape endpoint.

    Returns 503 if prometheus_client isn't installed in the container.
    """
    if not mantis_metrics.is_available():
        raise HTTPException(status_code=503, detail="prometheus_client not installed")
    return Response(
        content=mantis_metrics.render_text(),
        media_type=mantis_metrics.CONTENT_TYPE_LATEST,
    )


@app.get("/v1/models")
def models() -> dict[str, Any]:
    """OpenAI-compatible model listing.

    Public so clients can discover the model id before sending requests.
    Auth is enforced on the inference path itself (/v1/chat/completions).
    """
    return {
        "object": "list",
        "data": [
            {
                "id": runtime.model_kind,
                "object": "model",
                "owned_by": "mantis",
            }
        ],
    }


# Sentinel headers the upstream llama.cpp shouldn't see. We strip them so the
# Mantis-side auth credential never reaches the inference layer.
_PROXY_DROP_HEADERS = {
    "host",
    "content-length",
    "transfer-encoding",
    "connection",
    "x-mantis-token",
    "authorization",
    "cookie",
}


@app.post("/v1/chat/completions")
async def chat_completions_proxy(
    request: Request,
    tenant: TenantConfig = Depends(_require_run_scope),
) -> Any:
    """Auth-gated reverse proxy to the in-pod llama.cpp Holo3 server.

    Designed for OpenAI-compat clients (the host integration's BrainHolo3 client,
    direct `openai.OpenAI(...)` callers) that want to use Holo3 inference
    without the full /predict orchestrator.

    What this endpoint does:
      • Validates ``X-Mantis-Token`` and resolves the tenant (must have
        ``run`` scope).
      • Strips Mantis-side auth headers before forwarding so the upstream
        llama.cpp never sees them.
      • Forwards the JSON body verbatim to the in-pod llama.cpp server at
        ``MANTIS_LLAMA_PORT``.
      • Passes upstream status codes and JSON bodies through.
    """
    body = await request.body()
    upstream_port = os.environ.get("MANTIS_LLAMA_PORT", "18080")
    upstream = f"http://127.0.0.1:{upstream_port}/v1/chat/completions"

    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in _PROXY_DROP_HEADERS
    }
    headers["Content-Type"] = "application/json"

    logger.info(
        "v1/chat/completions tenant=%s upstream=%s bytes=%d",
        tenant.tenant_id,
        upstream,
        len(body),
    )

    try:
        r = await run_in_threadpool(
            requests.post,
            upstream,
            data=body,
            headers=headers,
            timeout=180,
        )
    except requests.RequestException as exc:
        mantis_metrics.CHAT_COMPLETIONS.labels(
            tenant_id=tenant.tenant_id, outcome="upstream_error"
        ).inc()
        logger.exception("v1/chat/completions upstream error tenant=%s", tenant.tenant_id)
        raise HTTPException(status_code=502, detail=f"upstream error: {exc}") from exc

    mantis_metrics.CHAT_COMPLETIONS.labels(
        tenant_id=tenant.tenant_id,
        outcome="ok" if 200 <= r.status_code < 300 else f"status_{r.status_code // 100}xx",
    ).inc()

    try:
        payload = r.json()
    except ValueError:
        # Upstream returned non-JSON (rare; usually means it crashed). Surface
        # the raw text so callers can debug.
        return JSONResponse(
            content={"error": {"message": r.text[:1000], "type": "upstream_error"}},
            status_code=r.status_code if r.status_code >= 400 else 502,
        )
    return JSONResponse(content=payload, status_code=r.status_code)


async def _handle_predict(
    request: Request, tenant: TenantConfig
) -> dict[str, Any]:
    """Shared handler for /predict and /v1/predict.

    Tier-1 + Tier-2 pipeline:

    1. Pydantic validation, global cap clamp.
    2. Per-tenant cap clamp.
    3. State-key + Chrome-profile namespacing per tenant.
    4. Per-tenant Anthropic key resolution.
    5. (Tier-2) URL allowlist enforcement on the plan.
    6. (Tier-2) Idempotency-key cache lookup.
    7. (Tier-2) Rate-limit token consumption.
    8. (Tier-2) Concurrency-slot acquisition (released in finally).
    9. (Tier-2) Webhook callback registered with the runtime if requested.
    10. Forward to the runtime.
    """
    try:
        raw = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="request body must be JSON") from exc
    if not isinstance(raw, dict):
        raise HTTPException(status_code=400, detail="request body must be a JSON object")

    try:
        req = PredictRequest.model_validate(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid request: {exc}") from exc

    payload = req.model_dump(exclude_none=True)
    payload["max_cost"] = min(
        float(payload.get("max_cost", MAX_COST_USD)),
        tenant.max_cost_per_run,
    )
    payload["max_time_minutes"] = min(
        int(payload.get("max_time_minutes", MAX_RUNTIME_MINUTES)),
        tenant.max_time_minutes_per_run,
    )
    payload["state_key"] = _tenant_state_key(tenant, payload.get("state_key"))

    os.environ["ANTHROPIC_API_KEY"] = _resolve_anthropic_key(tenant)
    os.environ["MANTIS_TENANT_ID"] = tenant.tenant_id
    profile_dir = _tenant_chrome_profile(tenant, payload["state_key"])
    os.environ["MANTIS_CHROME_PROFILE_DIR"] = str(profile_dir)

    is_run_mode = req.action is None

    # ── Tier-2: URL allowlist ────────────────────────────────────────────
    if is_run_mode and tenant.allowed_domains:
        plan_obj: Any = None
        if req.task_suite is not None:
            plan_obj = req.task_suite
        elif req.task_file_contents:
            try:
                plan_obj = json.loads(req.task_file_contents)
            except json.JSONDecodeError:
                plan_obj = None
        if plan_obj is not None:
            try:
                hosts = extract_navigate_hosts(plan_obj)
                assert_hosts_allowed(hosts, tenant.is_domain_allowed)
            except PermissionError as exc:
                mantis_metrics.PREDICT_REQUESTS.labels(
                    tenant_id=tenant.tenant_id, mode="run", outcome="denied_allowlist"
                ).inc()
                raise HTTPException(status_code=403, detail=str(exc)) from exc

    # ── Tier-2: Idempotency-key cache ────────────────────────────────────
    idempotency_key = request.headers.get("Idempotency-Key", "").strip()
    if is_run_mode and idempotency_key:
        cached = get_idempotency_cache().get(tenant.tenant_id, idempotency_key)
        if cached is not None:
            logger.info(
                "predict idempotency-hit tenant=%s key=%s run_id=%s",
                tenant.tenant_id, idempotency_key, cached.run_id,
            )
            mantis_metrics.PREDICT_REQUESTS.labels(
                tenant_id=tenant.tenant_id, mode="run", outcome="idempotent_hit"
            ).inc()
            return cached.response

    # ── Tier-2: Rate limit (token bucket) — applied to run-mode only ─────
    limiter = get_rate_limiter()
    if is_run_mode:
        rate_decision = limiter.try_consume_rate_token(
            tenant.tenant_id, tenant.rate_limit_per_minute
        )
        if not rate_decision.allowed:
            mantis_metrics.RATE_LIMIT_REJECTIONS.labels(
                tenant_id=tenant.tenant_id, kind="rate"
            ).inc()
            mantis_metrics.PREDICT_REQUESTS.labels(
                tenant_id=tenant.tenant_id, mode="run", outcome="rate_limited"
            ).inc()
            raise HTTPException(
                status_code=429,
                detail=rate_decision.reason,
                headers={"Retry-After": str(int(rate_decision.retry_after_seconds) + 1)},
            )

    # ── Tier-2: Concurrency slot ─────────────────────────────────────────
    concurrency_acquired = False
    if is_run_mode:
        decision = limiter.try_acquire_concurrency_slot(
            tenant.tenant_id, tenant.max_concurrent_runs
        )
        if not decision.allowed:
            mantis_metrics.RATE_LIMIT_REJECTIONS.labels(
                tenant_id=tenant.tenant_id, kind="concurrent"
            ).inc()
            mantis_metrics.PREDICT_REQUESTS.labels(
                tenant_id=tenant.tenant_id, mode="run", outcome="rate_limited"
            ).inc()
            raise HTTPException(
                status_code=429,
                detail=decision.reason,
                headers={"Retry-After": str(int(decision.retry_after_seconds) + 1)},
            )
        concurrency_acquired = True
        mantis_metrics.CONCURRENT_RUNS.labels(tenant_id=tenant.tenant_id).set(
            decision.concurrent
        )

    # ── Tier-2: Webhook URL — caller may override the tenant default ─────
    webhook_url = (
        raw.get("callback_url") or tenant.webhook_url or ""
    ).strip()
    if webhook_url:
        payload["_webhook_url"] = webhook_url
        payload["_webhook_secret_name"] = tenant.webhook_secret_name
        payload["_tenant_id"] = tenant.tenant_id

    logger.info(
        "predict tenant=%s scope=run state_key=%s detached=%s action=%s",
        tenant.tenant_id,
        payload["state_key"],
        payload.get("detached", True),
        req.action or "run",
    )

    try:
        response = await run_in_threadpool(runtime.run, payload)
        mode = req.action or "run"
        mantis_metrics.PREDICT_REQUESTS.labels(
            tenant_id=tenant.tenant_id, mode=mode, outcome="ok"
        ).inc()
        if is_run_mode and idempotency_key and isinstance(response, dict):
            get_idempotency_cache().store(
                tenant.tenant_id, idempotency_key, response.get("run_id", ""), response
            )
        if is_run_mode and webhook_url and isinstance(response, dict):
            run_id = response.get("run_id", "")
            if response.get("status") in {"succeeded", "failed", "cancelled"}:
                deliver_webhook_async(
                    webhook_url,
                    WebhookPayload(
                        run_id=run_id,
                        tenant_id=tenant.tenant_id,
                        status=str(response.get("status", "")),
                        summary=response.get("summary") or {},
                    ),
                    secret_name=tenant.webhook_secret_name,
                )
        return response
    except ValueError as exc:
        mantis_metrics.PREDICT_REQUESTS.labels(
            tenant_id=tenant.tenant_id, mode="run", outcome="bad_request"
        ).inc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        mantis_metrics.PREDICT_REQUESTS.labels(
            tenant_id=tenant.tenant_id, mode="run", outcome="error"
        ).inc()
        logger.exception("predict failed tenant=%s", tenant.tenant_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if concurrency_acquired:
            limiter.release_concurrency_slot(tenant.tenant_id)
            mantis_metrics.CONCURRENT_RUNS.labels(tenant_id=tenant.tenant_id).set(
                limiter.get_concurrent(tenant.tenant_id)
            )


@app.post("/v1/predict")
async def predict_v1(
    request: Request,
    tenant: TenantConfig = Depends(_require_mantis_token),
) -> dict[str, Any]:
    """Tier-1 multi-tenant /predict. Validated, per-tenant capped and isolated."""
    return await _handle_predict(request, tenant)


@app.post("/predict")
async def predict(
    request: Request,
    tenant: TenantConfig = Depends(_require_mantis_token),
) -> dict[str, Any]:
    """Backwards-compat alias for /v1/predict.

    Kept indefinitely for callers built against the v1.0 deployment shape.
    Identical behavior to /v1/predict.
    """
    return await _handle_predict(request, tenant)
