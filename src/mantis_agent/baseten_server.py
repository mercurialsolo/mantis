"""Baseten workload server for Mantis CUA.

This module is used by the Baseten custom-server Trusses under ``baseten/``.
It starts a local llama.cpp server for either Holo3 or Gemma4, then exposes a
small FastAPI surface that runs the existing CUA task and micro-plan runners.
"""

from __future__ import annotations

import base64
import hashlib
import http.server
import json
import logging
import os
import re
import socket
import socketserver
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, HTTPException, Request

from mantis_agent.gym.runner import GymRunner
from mantis_agent.gym.xdotool_env import XdotoolGymEnv

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
    for child in ("results", "screenshots", "checkpoints", "chrome-profile"):
        (root / child).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MANTIS_DEBUG_DIR", str(root / "screenshots" / "claude_debug"))
    return root


def _repo_root() -> Path:
    return Path(os.environ.get("MANTIS_REPO_ROOT", "/workspace/cua-agent"))


def _safe_state_key(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._")
    return cleaned or "micro_state"


def _plan_signature_from_steps(steps: list[dict[str, Any]]) -> str:
    payload = json.dumps(steps, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def _wait_for_openai_server(port: int, proc: subprocess.Popen, label: str) -> None:
    for i in range(180):
        try:
            resp = requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=2)
            if resp.status_code == 200:
                logger.info("%s ready after %ss", label, i * 2)
                return
        except Exception:
            pass
        if proc.poll() is not None:
            log_path = Path("/tmp/llama.log")
            log = log_path.read_text(errors="ignore")[-3000:] if log_path.exists() else ""
            raise RuntimeError(f"{label} crashed during startup:\n{log[-1000:]}")
        time.sleep(2)
    log = Path("/tmp/llama.log").read_text(errors="ignore")[-2000:]
    raise RuntimeError(f"{label} startup timeout:\n{log}")


def _start_local_proxy(upstream_proxy: dict[str, str], local_port: int = 3128) -> subprocess.Popen:
    proxy_server = upstream_proxy.get("server", "")
    proxy_user = upstream_proxy.get("username", "")
    proxy_pass = upstream_proxy.get("password", "")

    class ProxyHandler(http.server.BaseHTTPRequestHandler):
        upstream = proxy_server
        auth = base64.b64encode(f"{proxy_user}:{proxy_pass}".encode()).decode()

        def do_CONNECT(self):
            from urllib.parse import urlparse

            parsed = urlparse(self.upstream)
            try:
                upstream_socket = socket.create_connection((parsed.hostname, parsed.port), timeout=30)
                connect_req = (
                    f"CONNECT {self.path} HTTP/1.1\r\n"
                    f"Host: {self.path}\r\n"
                    f"Proxy-Authorization: Basic {self.auth}\r\n\r\n"
                )
                upstream_socket.sendall(connect_req.encode())
                response = upstream_socket.recv(4096)
                if b"200" not in response:
                    self.send_error(502)
                    return
                self.send_response(200)
                self.end_headers()

                def forward(src, dst):
                    try:
                        while True:
                            data = src.recv(65536)
                            if not data:
                                break
                            dst.sendall(data)
                    except OSError:
                        pass

                client = self.request
                t1 = threading.Thread(target=forward, args=(client, upstream_socket), daemon=True)
                t2 = threading.Thread(target=forward, args=(upstream_socket, client), daemon=True)
                t1.start()
                t2.start()
                t1.join()
                t2.join()
            except Exception as exc:
                self.send_error(502, str(exc))

        def log_message(self, *_args):
            return

    server = socketserver.ThreadingTCPServer(("127.0.0.1", local_port), ProxyHandler)

    def serve():
        with server:
            server.serve_forever()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()

    class ProxyProcess:
        def terminate(self):
            server.shutdown()

    logger.info("local proxy forwarder listening on :%s", local_port)
    return ProxyProcess()  # type: ignore[return-value]


def _build_proxy_config(city: str = "", state: str = "", session_id: str = "") -> dict[str, str] | None:
    proxy_url = os.environ.get("PROXY_URL", "")
    if not proxy_url:
        return None
    proxy: dict[str, str] = {"server": proxy_url}
    proxy_user = os.environ.get("PROXY_USER", "")
    proxy_pass = os.environ.get("PROXY_PASS", "")
    if proxy_user:
        if city and "_city-" not in proxy_pass:
            proxy_pass = f"{proxy_pass}_city-{city}"
        if state and "_state-" not in proxy_pass:
            proxy_pass = f"{proxy_pass}_state-{state}"
        if session_id and "_session-" not in proxy_pass:
            proxy_pass = f"{proxy_pass}_session-{session_id}"
        proxy["username"] = proxy_user
        proxy["password"] = proxy_pass
    return proxy


class BasetenCUARuntime:
    def __init__(self) -> None:
        self.model_kind = os.environ.get("MANTIS_MODEL", "holo3")
        self.port = int(os.environ.get("MANTIS_LLAMA_PORT", "18080"))
        self.llama_proc: subprocess.Popen | None = None
        self.brain: Any = None
        self.lock = threading.Lock()
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
        self.load()
        with self.lock:
            task_suite = self._task_suite_from_payload(payload)
            if task_suite.get("_micro_plan"):
                return self._run_micro(task_suite, payload)
            return self._run_tasks(task_suite, payload)

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

        if path.suffix == ".json":
            raw_steps = json.loads(path.read_text())
            domain = path.stem
            micro_plan = MicroPlan(domain=domain)
            for step in raw_steps:
                micro_plan.steps.append(PlanDecomposer._build_intent(step))
        else:
            decomposer = PlanDecomposer()
            micro_plan = decomposer.decompose(str(path))

        micro_plan_steps = [
            {
                "intent": s.intent,
                "type": s.type,
                "verify": s.verify,
                "budget": s.budget,
                "reverse": s.reverse,
                "grounding": s.grounding,
                "claude_only": s.claude_only,
                "loop_target": s.loop_target,
                "loop_count": s.loop_count,
                "section": s.section,
                "required": s.required,
                "gate": s.gate,
            }
            for s in micro_plan.steps
        ]
        plan_signature = _plan_signature_from_steps(micro_plan_steps)
        default_state_key = f"micro_{micro_plan.domain.replace('.', '_')}_{plan_signature[:12]}"
        state_key = _safe_state_key(str(payload.get("state_key") or default_state_key))
        data_root = _data_root()
        return {
            "session_name": f"micro_{micro_plan.domain.replace('.', '_')}",
            "base_url": "",
            "_max_cost": float(payload.get("max_cost", 10.0)),
            "_max_time_minutes": int(payload.get("max_time_minutes", 180)),
            "_resume_state": bool(payload.get("resume_state", False)),
            "_state_key": state_key,
            "_checkpoint_path": str(data_root / "checkpoints" / f"{state_key}.json"),
            "_plan_signature": plan_signature,
            "_micro_plan": micro_plan_steps,
            "tasks": [],
        }

    def _make_env(self, task_suite: dict[str, Any], run_id: str, settle_time: float) -> tuple[XdotoolGymEnv, Any]:
        data_root = _data_root()
        session_name = task_suite.get("session_name", "baseten_cua")
        proxy = _build_proxy_config(city="miami", session_id=f"mantis{run_id.replace('_', '')}")
        proxy_proc = None
        proxy_server = ""
        if proxy:
            if proxy.get("username"):
                proxy_proc = _start_local_proxy(proxy, local_port=3128)
                proxy_server = "http://127.0.0.1:3128"
            else:
                proxy_server = proxy["server"]

        env = XdotoolGymEnv(
            start_url=task_suite.get("base_url", ""),
            viewport=(1280, 720),
            browser=os.environ.get("MANTIS_BROWSER", "google-chrome"),
            settle_time=settle_time,
            human_speed=False,
            proxy_server=proxy_server,
            profile_dir=str(data_root / "chrome-profile"),
            save_screenshots=str(data_root / "screenshots" / f"{session_name}_{run_id}"),
        )
        return env, proxy_proc

    def _run_micro(self, task_suite: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        from mantis_agent.extraction import ClaudeExtractor
        from mantis_agent.grounding import ClaudeGrounding
        from mantis_agent.gym.micro_runner import MicroPlanRunner
        from mantis_agent.plan_decomposer import MicroIntent, MicroPlan

        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
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

            grounding = ClaudeGrounding()
            extractor = ClaudeExtractor()
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
            lead_rows = runner._successful_lead_data(step_results)
            unique_leads = {runner._lead_key(lead): lead for lead in lead_rows}
            leads = list(unique_leads.values())
            costs = getattr(runner, "_final_costs", {})
            result = {
                "run_id": run_id,
                "provider": "baseten",
                "session_name": session_name,
                "model": self.model_kind,
                "mode": "micro_intent",
                "total_time_s": round(time.time() - t0),
                "steps_executed": len(step_results),
                "viable": len(leads),
                "leads_with_phone": sum(1 for lead in leads if runner._lead_has_phone(lead)),
                "state_key": task_suite.get("_state_key", ""),
                "checkpoint_path": checkpoint_path,
                "plan_signature": task_suite.get("_plan_signature", ""),
                "resume_state": resume_state,
                "costs": costs,
                "leads": leads,
                "step_details": [
                    {
                        "step": r.step_index,
                        "intent": r.intent[:120],
                        "success": r.success,
                        "steps": r.steps_used,
                        "data": r.data[:300] if r.data else "",
                    }
                    for r in step_results
                ],
            }
            self._save_result(result, prefix=self.model_kind.replace("-", "_"))
            return result
        finally:
            env.close()
            if proxy_proc:
                proxy_proc.terminate()

    def _run_tasks(self, task_suite: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        from mantis_agent.grounding import ClaudeGrounding

        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        t0 = time.time()
        tasks = task_suite.get("tasks", [])
        session_name = task_suite.get("session_name", "baseten_tasks")
        env, proxy_proc = self._make_env(
            task_suite,
            run_id,
            settle_time=4.0 if self.model_kind == "holo3" else 2.0,
        )
        grounding = ClaudeGrounding()
        scores: list[float] = []
        task_details: list[dict[str, Any]] = []

        try:
            for i, task in enumerate(tasks):
                task_start = time.time()
                task_id = task.get("task_id", f"task_{i + 1}")
                runner = GymRunner(
                    brain=self.brain,
                    env=env,
                    max_steps=int(payload.get("max_steps", 30)),
                    frames_per_inference=1 if self.model_kind == "holo3" else 2,
                    grounding=grounding,
                )
                try:
                    result = runner.run(
                        task=task["intent"],
                        task_id=task_id,
                        start_url=task.get("start_url", ""),
                    )
                    success = bool(result.success)
                    scores.append(1.0 if success else 0.0)
                    task_details.append(
                        {
                            "task_id": task_id,
                            "success": success,
                            "steps": result.total_steps,
                            "duration_s": round(time.time() - task_start),
                            "termination_reason": result.termination_reason,
                        }
                    )
                except Exception as exc:
                    logger.exception("task %s failed", task_id)
                    scores.append(0.0)
                    task_details.append(
                        {
                            "task_id": task_id,
                            "success": False,
                            "error": str(exc),
                            "duration_s": round(time.time() - task_start),
                        }
                    )
            passed = sum(1 for score in scores if score > 0)
            result = {
                "run_id": run_id,
                "provider": "baseten",
                "session_name": session_name,
                "model": self.model_kind,
                "mode": "tasks",
                "passed": passed,
                "total": len(scores),
                "score": (sum(scores) / len(scores) * 100) if scores else 0,
                "total_time_s": round(time.time() - t0),
                "task_details": task_details,
            }
            self._save_result(result, prefix=self.model_kind.replace("-", "_"))
            return result
        finally:
            env.close()
            if proxy_proc:
                proxy_proc.terminate()

    def _save_result(self, result: dict[str, Any], prefix: str) -> None:
        result_path = _data_root() / "results" / f"{prefix}_results_{result['run_id']}.json"
        result_path.write_text(json.dumps(result, indent=2))
        result["result_path"] = str(result_path)


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
        return runtime.run(payload)
    except Exception as exc:
        logger.exception("predict failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
