"""Shared utilities for Mantis CUA server deployments (Modal, Baseten, local).

Consolidates duplicated logic across modal_cua_server.py and baseten_server.py:
  - Proxy configuration and local proxy forwarding
  - Plan signature hashing and state key sanitization
  - Micro-intent result building with dynamic verification
  - Lead CSV writing and parsing
  - Micro-plan suite construction from file paths
"""

from __future__ import annotations

import base64
import csv
import hashlib
import http.server
import json
import logging
import os
import re
import socket
import socketserver
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Proxy utilities ───────────────────────────────────────────────


def build_proxy_config(
    city: str = "",
    state: str = "",
    session_id: str = "",
) -> dict[str, str] | None:
    """Build proxy config from environment variables.

    IPRoyal residential proxy supports geo-targeting via password suffixes:
      _country-us          -> US IPs (default in .env)
      _city-miami          -> Miami residential IP
      _state-florida       -> Florida state
      _session-{id}        -> sticky session (same IP across requests)
    """
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


def start_local_proxy(
    upstream_proxy: dict[str, str],
    local_port: int = 3128,
) -> Any:
    """Start a local forward proxy that handles authentication with the upstream.

    Chrome --proxy-server doesn't support user:pass in URLs.
    This starts a tiny Python CONNECT proxy on localhost that forwards to the
    authenticated upstream (IPRoyal etc).

    Returns an object with a .terminate() method.
    """
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
                upstream_socket = socket.create_connection(
                    (parsed.hostname, parsed.port), timeout=30
                )
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
    return ProxyProcess()


def resolve_proxy_server(proxy: dict[str, str] | None, local_port: int = 3128) -> tuple[str, Any]:
    """Resolve proxy config into a proxy_server URL string and optional proxy process.

    Returns (proxy_server_url, proxy_process_or_None).
    """
    if not proxy:
        return "", None
    if proxy.get("username"):
        proc = start_local_proxy(proxy, local_port=local_port)
        return f"http://127.0.0.1:{local_port}", proc
    return proxy["server"], None


# ── State key and plan signature ──────────────────────────────────


def plan_signature_from_steps(steps: list[dict[str, Any]]) -> str:
    """Create a deterministic SHA256 hash of plan steps for deduplication."""
    payload = json.dumps(steps, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def safe_state_key(raw: str) -> str:
    """Sanitize a string into a safe filesystem/state key."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._")
    return cleaned or "micro_state"


# ── Time and ID utilities ─────────────────────────────────────────


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id() -> str:
    import uuid

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


# ── Lead CSV writing ──────────────────────────────────────────────

LEAD_FIELDS = ("status", "year", "make", "model", "price", "phone", "seller", "url")


def parse_lead_row(lead: Any) -> dict[str, str]:
    """Normalize a lead (dict or pipe-delimited string) into a flat row dict."""
    raw = json.dumps(lead, sort_keys=True) if isinstance(lead, dict) else str(lead)
    row = {field: "" for field in LEAD_FIELDS}
    row["raw"] = raw

    if isinstance(lead, dict):
        for field in LEAD_FIELDS:
            value = lead.get(field)
            if value is not None:
                row[field] = str(value)
        return row

    parts = [part.strip() for part in raw.split("|") if part.strip()]
    if parts and ":" not in parts[0]:
        row["status"] = parts[0]
        parts = parts[1:]
    for part in parts:
        key, sep, value = part.partition(":")
        if not sep:
            continue
        normalized = key.strip().lower().replace(" ", "_")
        if normalized in row:
            row[normalized] = value.strip()
    return row


def write_leads_csv(path: Path, leads: list[Any]) -> None:
    """Write leads to a CSV file with standard field ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(LEAD_FIELDS) + ["raw"],
        )
        writer.writeheader()
        for lead in leads:
            writer.writerow(parse_lead_row(lead))


# ── Micro-intent result builder ───────────────────────────────────


def build_micro_result(
    runner: Any,
    step_results: list[Any],
    *,
    run_id: str,
    provider: str,
    session_name: str,
    model_name: str,
    elapsed_seconds: float,
    state_key: str = "",
    checkpoint_path: str = "",
    plan_signature: str = "",
    resume_state: bool = False,
    intent_truncate: int = 120,
) -> dict[str, Any]:
    """Build a standardized micro-intent result dict with dynamic verification.

    This is the single source of truth for micro-intent run results.
    Both Modal and Baseten use this to ensure consistent output including
    the dynamic_verification report.
    """
    lead_rows = runner._successful_lead_data(step_results)
    unique_leads = {runner._lead_key(lead): lead for lead in lead_rows}
    leads = list(unique_leads.values())
    costs = getattr(runner, "_final_costs", {})
    status = costs.get("status") or getattr(runner, "_final_status", "unknown")

    dynamic_verification = runner.dynamic_verification_report(status=status)
    dynamic_verification_summary = {
        "status": dynamic_verification.get("status"),
        "verdict": dynamic_verification.get("verdict"),
        "totals": dynamic_verification.get("totals", {}),
        "checks": dynamic_verification.get("checks", []),
    }

    return {
        "run_id": run_id,
        "provider": provider,
        "session_name": session_name,
        "model": model_name,
        "mode": "micro_intent",
        "total_time_s": round(elapsed_seconds),
        "steps_executed": len(step_results),
        "viable": len(leads),
        "leads_with_phone": sum(1 for lead in leads if runner._lead_has_phone(lead)),
        "state_key": state_key,
        "checkpoint_path": checkpoint_path,
        "plan_signature": plan_signature,
        "resume_state": resume_state,
        "costs": costs,
        "dynamic_verification": dynamic_verification,
        "dynamic_verification_summary": dynamic_verification_summary,
        "leads": leads,
        "step_details": [
            {
                "step": r.step_index,
                "intent": r.intent[:intent_truncate],
                "success": r.success,
                "steps": r.steps_used,
                "data": r.data[:300] if r.data else "",
            }
            for r in step_results
        ],
    }


# Keys to include in a compact result summary (for detached run status etc.)
RESULT_SUMMARY_KEYS = (
    "run_id",
    "provider",
    "session_name",
    "model",
    "mode",
    "total_time_s",
    "steps_executed",
    "viable",
    "leads_with_phone",
    "passed",
    "total",
    "score",
    "result_path",
    "csv_path",
    "detached_result_path",
    "detached_csv_path",
    "dynamic_verification_summary",
)


def result_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Extract a compact summary from a full result dict."""
    return {key: result[key] for key in RESULT_SUMMARY_KEYS if key in result}


# ── Task loop result builder ─────────────────────────────────────


def build_task_loop_result(
    *,
    run_id: str,
    provider: str,
    session_name: str,
    model_name: str,
    elapsed_seconds: float,
    scores: list[float],
    task_details: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a standardized task-loop result dict."""
    passed = sum(1 for s in scores if s > 0)
    result = {
        "run_id": run_id,
        "provider": provider,
        "session_name": session_name,
        "model": model_name,
        "mode": "tasks",
        "passed": passed,
        "total": len(scores),
        "score": (sum(scores) / len(scores) * 100) if scores else 0,
        "total_time_s": round(elapsed_seconds),
        "task_details": task_details,
    }
    if extra:
        result.update(extra)
    return result


# ── Result persistence ────────────────────────────────────────────


def save_result_json(
    result: dict[str, Any],
    results_dir: Path,
    prefix: str,
) -> Path:
    """Save a result dict to JSON and optionally write leads CSV.

    Returns the path to the saved JSON file.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"{prefix}_results_{result['run_id']}.json"
    result["result_path"] = str(result_path)

    leads = result.get("leads")
    if isinstance(leads, list):
        csv_path = results_dir / f"{prefix}_leads_{result['run_id']}.csv"
        write_leads_csv(csv_path, leads)
        result["csv_path"] = str(csv_path)

    result_path.write_text(json.dumps(result, indent=2))
    return result_path


# ── Micro-plan suite construction ─────────────────────────────────


def micro_plan_steps_to_dicts(steps: list[Any]) -> list[dict[str, Any]]:
    """Serialize a list of MicroIntent objects into plain dicts."""
    return [
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
        for s in steps
    ]


def build_micro_suite(
    micro_plan_steps: list[dict[str, Any]],
    domain: str,
    *,
    max_cost: float = 10.0,
    max_time_minutes: int = 180,
    resume_state: bool = False,
    state_key: str = "",
    checkpoint_dir: str = "/data/checkpoints",
    proxy_city: str = "",
    proxy_state: str = "",
) -> dict[str, Any]:
    """Build a task_suite dict for micro-intent execution.

    Used by both Modal main() and Baseten _micro_suite_from_path().
    """
    signature = plan_signature_from_steps(micro_plan_steps)
    safe_domain = domain.replace(".", "_")
    default_key = f"micro_{safe_domain}_{signature[:12]}"
    resolved_key = safe_state_key(state_key or default_key)
    checkpoint_path = f"{checkpoint_dir}/{resolved_key}.json"

    return {
        "session_name": f"micro_{safe_domain}",
        "base_url": "",
        "_max_cost": max_cost,
        "_max_time_minutes": max_time_minutes,
        "_resume_state": resume_state,
        "_state_key": resolved_key,
        "_checkpoint_path": checkpoint_path,
        "_plan_signature": signature,
        "_proxy_city": proxy_city,
        "_proxy_state": proxy_state,
        "_micro_plan": micro_plan_steps,
        "tasks": [],
    }


# ── Server readiness polling ─────────────────────────────────────


def wait_for_openai_server(
    port: int,
    proc: Any,
    label: str,
    timeout_seconds: int = 360,
    poll_interval: float = 2.0,
    log_path: str = "/tmp/llama.log",
) -> None:
    """Poll an OpenAI-compatible /v1/models endpoint until ready or crash."""
    import requests

    for i in range(int(timeout_seconds / poll_interval)):
        try:
            resp = requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=2)
            if resp.status_code == 200:
                logger.info("%s ready after %ss", label, int(i * poll_interval))
                return
        except Exception:
            pass
        if proc.poll() is not None:
            log_text = ""
            log_file = Path(log_path)
            if log_file.exists():
                log_text = log_file.read_text(errors="ignore")[-3000:]
            raise RuntimeError(f"{label} crashed during startup:\n{log_text[-1000:]}")
        time.sleep(poll_interval)

    log_text = Path(log_path).read_text(errors="ignore")[-2000:] if Path(log_path).exists() else ""
    raise RuntimeError(f"{label} startup timeout:\n{log_text}")
