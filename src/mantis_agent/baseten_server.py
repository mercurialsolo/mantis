"""Baseten workload server for Mantis CUA.

This module is used by the Baseten custom-server Trusses under ``baseten/``.
It starts a local llama.cpp server for either Holo3 or Gemma4, then exposes a
small FastAPI surface that runs the existing CUA task and micro-plan runners.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from mantis_agent.gym.runner import GymRunner
from mantis_agent.gym.xdotool_env import XdotoolGymEnv
from mantis_agent.server_utils import (
    build_micro_result,
    build_micro_suite,
    build_proxy_config,
    micro_plan_steps_to_dicts,
    parse_lead_row,
    plan_signature_from_steps,
    resolve_proxy_server,
    result_summary,
    safe_state_key,
    save_result_json,
    start_local_proxy,
    utc_now,
    wait_for_openai_server,
    write_leads_csv,
)

logger = logging.getLogger("mantis_agent.baseten_server")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

app = FastAPI(title="Mantis CUA Baseten Workload", docs_url=None, redoc_url=None)


SECRET_ENV_MAP = {
    "anthropic_api_key": "ANTHROPIC_API_KEY",
    "proxy_url": "PROXY_URL",
    "proxy_user": "PROXY_USER",
    "proxy_pass": "PROXY_PASS",
    "hf_access_token": "HF_TOKEN",
}


def _read_secret(name: str) -> str:
    path = Path("/secrets") / name
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def _load_secret_environment() -> None:
    for secret_name, env_name in SECRET_ENV_MAP.items():
        if os.environ.get(env_name):
            continue
        value = _read_secret(secret_name)
        if value:
            os.environ[env_name] = value


def _data_root() -> Path:
    root = Path(os.environ.get("MANTIS_DATA_DIR", "/workspace/mantis-data"))
    root.mkdir(parents=True, exist_ok=True)
    for child in ("results", "runs", "screenshots", "checkpoints", "chrome-profile"):
        (root / child).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MANTIS_DEBUG_DIR", str(root / "screenshots" / "claude_debug"))
    return root


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
        self.llama_proc = subprocess.Popen(
            cmd,
            stdout=open("/tmp/llama.log", "w"),
            stderr=subprocess.STDOUT,
        )
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

        micro_path = payload.get("micro") or payload.get("micro_path") or os.environ.get(
            "MANTIS_DEFAULT_MICRO",
            "plans/boattrader/extract_url_filtered.json",
        )
        return self._micro_suite_from_path(str(micro_path), payload)

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
        from mantis_agent.plan_decomposer import MicroPlan, PlanDecomposer

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
            # Text plan: use graph learning path for enhancement
            plan_text = path.read_text()
            try:
                from mantis_agent.graph import GraphLearner, GraphCompiler, GraphStore
                from mantis_agent.graph.plan_validator import PlanValidator

                data_root = _data_root()
                learner = GraphLearner(
                    store=GraphStore(base_path=str(data_root / "graphs")),
                )
                graph = learner.learn(objective_text=plan_text, n_samples=0)
                compiler = GraphCompiler()
                micro_plan = compiler.compile(graph)
                validator = PlanValidator()
                issues = validator.validate(micro_plan, objective=graph.objective)
                if issues:
                    micro_plan = validator.enhance(micro_plan, objective=graph.objective)
                objective_dict = graph.objective.to_dict()
                logger.info(
                    "Baseten: graph-enhanced plan from %s (%d steps)",
                    path.name, len(micro_plan.steps),
                )
            except Exception as e:
                logger.warning("Baseten: graph learning failed (%s), falling back to PlanDecomposer", e)
                decomposer = PlanDecomposer()
                micro_plan = decomposer.decompose(str(path))

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
            browser=os.environ.get("MANTIS_BROWSER", "google-chrome"),
            profile_dir=str(data_root / "chrome-profile"),
            save_screenshots_dir=str(data_root / "screenshots"),
        )

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
            self._save_result(result, prefix=self.model_kind.replace("-", "_"))
            return result
        finally:
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
        grounding = ClaudeGrounding()

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
        self._save_result(result, prefix=self.model_kind.replace("-", "_"))
        return result

    def _save_result(self, result: dict[str, Any], prefix: str) -> None:
        save_result_json(result, _data_root() / "results", prefix)


runtime = BasetenCUARuntime()


@app.on_event("startup")
def startup() -> None:
    runtime.load()


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": runtime.loaded, "model": runtime.model_kind}


@app.get("/v1/models")
def models() -> dict[str, Any]:
    return {"data": [{"id": runtime.model_kind, "object": "model"}]}


@app.post("/predict")
async def predict(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="request body must be JSON") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="request body must be a JSON object")
    try:
        return await run_in_threadpool(runtime.run, payload)
    except Exception as exc:
        logger.exception("predict failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
