"""Shared utilities for Mantis CUA server deployments (Modal, Baseten, local).

Consolidates duplicated logic across deploy/modal/modal_cua_server.py and baseten_server.py:
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


def _slug_proxy_location(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9]+", "_", value.strip())).strip("_")


_US_STATE_SLUGS = {
    "al": "alabama",
    "ak": "alaska",
    "az": "arizona",
    "ar": "arkansas",
    "ca": "california",
    "co": "colorado",
    "ct": "connecticut",
    "de": "delaware",
    "fl": "florida",
    "ga": "georgia",
    "hi": "hawaii",
    "id": "idaho",
    "il": "illinois",
    "in": "indiana",
    "ia": "iowa",
    "ks": "kansas",
    "ky": "kentucky",
    "la": "louisiana",
    "me": "maine",
    "md": "maryland",
    "ma": "massachusetts",
    "mi": "michigan",
    "mn": "minnesota",
    "ms": "mississippi",
    "mo": "missouri",
    "mt": "montana",
    "ne": "nebraska",
    "nv": "nevada",
    "nh": "new_hampshire",
    "nj": "new_jersey",
    "nm": "new_mexico",
    "ny": "new_york",
    "nc": "north_carolina",
    "nd": "north_dakota",
    "oh": "ohio",
    "ok": "oklahoma",
    "or": "oregon",
    "pa": "pennsylvania",
    "ri": "rhode_island",
    "sc": "south_carolina",
    "sd": "south_dakota",
    "tn": "tennessee",
    "tx": "texas",
    "ut": "utah",
    "vt": "vermont",
    "va": "virginia",
    "wa": "washington",
    "wv": "west_virginia",
    "wi": "wisconsin",
    "wy": "wyoming",
}


def _build_oxylabs_username(
    username: str,
    *,
    city: str,
    state: str = "",
    country: str = "US",
) -> str:
    """Build an Oxylabs username with optional city targeting.

    Oxylabs city targeting requires the ``customer-`` username prefix. Plain
    account usernames work for generic endpoints, but city suffixes return 407
    unless the prefixed credential form is used.
    """
    if not city:
        return username

    lowered = username.lower()
    if "-city-" in lowered or "-cc-" in lowered or "-st-" in lowered:
        return username

    city_slug = _slug_proxy_location(city).lower()
    if not city_slug:
        return username

    base = username if lowered.startswith("customer-") else f"customer-{username}"
    state_slug = _slug_proxy_location(state or os.environ.get("OXYLABS_STATE", "")).lower()
    state_slug = _US_STATE_SLUGS.get(state_slug, state_slug)
    if state_slug and country.upper() == "US":
        return f"{base}-st-us_{state_slug}-city-{city_slug}"
    return f"{base}-cc-{country.upper()}-city-{city_slug}"


def _oxylabs_endpoint_for_targeting(endpoint: str, *, city: str) -> str:
    if not city:
        return endpoint
    return os.environ.get("OXYLABS_CITY_ENTRYPOINT", "").strip() or "pr.oxylabs.io:7777"


def build_proxy_config(
    city: str = "",
    state: str = "",
    session_id: str = "",
    provider: str = "",
) -> dict[str, str] | None:
    """Build proxy config from environment variables.

    Providers:
      iproyal  -> PROXY_URL / PROXY_USER / PROXY_PASS
      oxylabs  -> OXYLABS_ENTRYPOINT / OXYLABS_USERNAME / OXYLABS_PASSWORD
      privateproxy -> PRIVATEPROXY_ENTRYPOINT / PRIVATEPROXY_USERNAME / PRIVATEPROXY_PASSWORD

    IPRoyal residential proxy supports geo-targeting via password suffixes:
      _country-us          -> US IPs (default in .env)
      _city-miami          -> Miami residential IP
      _state-<full-name>   -> e.g. _state-florida (NOT _state-fl)
      _session-{id}        -> sticky session (same IP across requests)

    The two-letter state abbreviation form (`_state-fl`) is rejected by
    IPRoyal with a 503 on CONNECT — verified empirically. Pass the full
    lowercase name (`_state-florida`) or omit the state entirely (city +
    country alone is plenty of geo-targeting for most use cases).
    """
    provider = (provider or os.environ.get("MANTIS_PROXY_PROVIDER") or "iproyal").strip().lower()
    if provider in {"privateproxy", "private_proxy", "private"}:
        proxy_endpoint = os.environ.get("PRIVATEPROXY_ENTRYPOINT", "").strip()
        proxy_user = os.environ.get("PRIVATEPROXY_USERNAME", "")
        proxy_pass = os.environ.get("PRIVATEPROXY_PASSWORD", "")
        # PrivateProxy commonly exports proxies as IP:port:username:password.
        # Support that shape as well as separate env vars.
        if "://" not in proxy_endpoint:
            host, port, user, password = (proxy_endpoint.split(":", 3) + ["", "", "", ""])[:4]
            if host and port and user and password and not proxy_user:
                proxy_endpoint = f"{host}:{port}"
                proxy_user = user
                proxy_pass = password

        proxy_url = proxy_endpoint
        if not proxy_url:
            return None
        if "://" not in proxy_url:
            proxy_url = f"http://{proxy_url}"

        proxy: dict[str, str] = {"server": proxy_url}
        if proxy_user:
            proxy["username"] = proxy_user
            proxy["password"] = proxy_pass
        return proxy

    if provider in {"oxylabs", "oxy"}:
        proxy_user = (
            os.environ.get("OXYLABS_USERNAME", "")
            or os.environ.get("OXYLABS_USER", "")
        )
        target_city = city or os.environ.get("OXYLABS_CITY", "")
        target_state = state or os.environ.get("OXYLABS_STATE", "")
        proxy_url = _oxylabs_endpoint_for_targeting(
            os.environ.get("OXYLABS_ENTRYPOINT", "").strip(),
            city=target_city if proxy_user else "",
        )
        if not proxy_url:
            return None
        if "://" not in proxy_url:
            proxy_url = f"http://{proxy_url}"

        proxy: dict[str, str] = {"server": proxy_url}
        proxy_pass = (
            os.environ.get("OXYLABS_PASSWORD", "")
            or os.environ.get("OXYLABS_PASS", "")
        )
        if proxy_user:
            country = os.environ.get("OXYLABS_COUNTRY", "US").strip() or "US"
            proxy["username"] = _build_oxylabs_username(
                proxy_user,
                city=target_city,
                state=target_state,
                country=country,
            )
            proxy["password"] = proxy_pass
        return proxy

    if provider not in {"iproyal", "iproyal_residential"}:
        raise ValueError(f"unknown proxy provider: {provider}")

    proxy_url = os.environ.get("PROXY_URL", "")
    if not proxy_url:
        return None

    proxy: dict[str, str] = {"server": proxy_url}
    proxy_user = os.environ.get("PROXY_USER", "")
    proxy_pass = os.environ.get("PROXY_PASS", "")
    if proxy_user:
        if city and "_city-" not in proxy_pass:
            proxy_pass = f"{proxy_pass}_city-{city}"
        # Skip _state- when state looks like a 2-letter abbreviation —
        # IPRoyal rejects those and the CONNECT fails with 503.
        if state and "_state-" not in proxy_pass and len(state) > 2:
            proxy_pass = f"{proxy_pass}_state-{state.lower()}"
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
    """Sanitize a string into a safe filesystem/state key.

    Despite the legacy name, this is now used for ``profile_id``,
    ``workflow_id``, ``tenant_id``, and any other identifier that lands on
    disk. Kept as ``safe_state_key`` for back-compat with all existing
    callers; new code can call it whatever fits semantically.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._")
    return cleaned or "micro_state"


def resolve_ids(
    *,
    state_key: str = "",
    profile_id: str = "",
    workflow_id: str = "",
    plan_signature: str = "",
) -> tuple[str, str]:
    """Resolve caller inputs into ``(profile_id, workflow_id)`` (#341).

    Behavior matrix:

    * Caller set ``profile_id`` / ``workflow_id`` → use them (new shape).
      Missing ``workflow_id`` defaults to ``plan_signature[:12]``;
      missing ``profile_id`` defaults to ``"default"``.
    * Caller set only legacy ``state_key`` → route to both (Phase 1
      back-compat; preserves today's "one key controls profile + checkpoint"
      behavior). Eventually deprecated; see issue #341.
    * Neither set → ``profile_id="default"``, ``workflow_id=plan_signature[:12]``.

    Outputs are sanitized via :func:`safe_state_key`.
    """
    if profile_id or workflow_id:
        wf_default = plan_signature[:12] if plan_signature else "default"
        wid = safe_state_key(workflow_id or wf_default)
        pid = safe_state_key(profile_id or "default")
        return pid, wid

    if state_key:
        sk = safe_state_key(state_key)
        return sk, sk

    wf_default = plan_signature[:12] if plan_signature else "default"
    return safe_state_key("default"), safe_state_key(wf_default)


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
    profile_id: str = "",
    workflow_id: str = "",
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

    # #300: per-backend trajectory aggregate. ``plan`` / ``som`` /
    # ``vision`` counts mirror :class:`gym.runner.RunResult.executor_backend_counts`
    # on the /v1/cua path. Empty dict on legacy hosts whose handlers
    # don't tag a backend (no behavior change for those callers).
    executor_backend_counts: dict[str, int] = {}
    for r in step_results:
        backend = getattr(r, "executor_backend", "") or ""
        if backend:
            executor_backend_counts[backend] = (
                executor_backend_counts.get(backend, 0) + 1
            )

    # Epic #377 Phase C: surface the self-healing audit log on the
    # result envelope so operators can see what the framework did
    # (rewrites, demotions, handler escalations, critic insertions).
    # Empty list when no healing fired.
    from .gym import healing_events as _healing
    healing_log = _healing.snapshot(runner)

    # Epic #362 Phase B: surface the TimeMeter's bucket breakdown
    # on the result envelope. ``wall_time_breakdown`` is the aggregate
    # 9-bucket dict; ``step_details[i].time_breakdown`` is the same
    # shape scoped to one step. Both are always-present additive
    # keys — runners without a TimeMeter (pre-Phase-A or test
    # harnesses) yield all-zero dicts rather than missing keys, so
    # consumers can branch on bucket presence rather than key
    # existence.
    time_meter = getattr(runner, "time_meter", None)
    if time_meter is not None:
        wall_time_breakdown = {k: round(v, 3) for k, v in time_meter.breakdown().items()}
    else:
        from .gym.time_meter import BUCKETS as _BUCKETS
        wall_time_breakdown = {b: 0.0 for b in _BUCKETS}

    return {
        "run_id": run_id,
        "provider": provider,
        "session_name": session_name,
        "model": model_name,
        "mode": "micro_intent",
        "total_time_s": round(elapsed_seconds),
        "wall_time_breakdown": wall_time_breakdown,
        "steps_executed": len(step_results),
        "viable": len(leads),
        "leads_with_phone": sum(1 for lead in leads if runner._lead_has_phone(lead)),
        "state_key": state_key or workflow_id,
        "profile_id": profile_id or state_key,
        "workflow_id": workflow_id or state_key,
        "checkpoint_path": checkpoint_path,
        "plan_signature": plan_signature,
        "resume_state": resume_state,
        "costs": costs,
        "healing_events": healing_log,
        "dynamic_verification": dynamic_verification,
        "dynamic_verification_summary": dynamic_verification_summary,
        "leads": leads,
        "executor_backend_counts": executor_backend_counts,
        "step_details": [
            {
                "step": r.step_index,
                "intent": r.intent[:intent_truncate],
                "success": r.success,
                "steps": r.steps_used,
                "data": r.data[:300] if r.data else "",
                "executor_backend": getattr(r, "executor_backend", "") or "",
                "time_breakdown": _step_time_breakdown(time_meter, r.step_index),
            }
            for r in step_results
        ],
    }


def _step_time_breakdown(time_meter: Any, step_index: int) -> dict[str, float]:
    """Return the per-step bucket dict, rounded for JSON friendliness.

    Falls back to an all-zeros bucket dict when ``time_meter`` is None
    or the step was never measured — keeps the schema stable across
    callers that haven't adopted the TimeMeter yet.
    """
    if time_meter is None:
        from .gym.time_meter import BUCKETS as _BUCKETS
        return {b: 0.0 for b in _BUCKETS}
    return {k: round(v, 3) for k, v in time_meter.step_breakdown(step_index).items()}


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
    profile_id: str = "",
    workflow_id: str = "",
    checkpoint_dir: str = "/data/checkpoints",
    proxy_city: str = "",
    proxy_state: str = "",
    proxy_disabled: bool = False,
    objective: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a task_suite dict for micro-intent execution.

    Used by both Modal main() and Baseten _micro_suite_from_path().
    When ``objective`` is provided, it's embedded so downstream code
    (e.g. ClaudeExtractor) can build an ExtractionSchema from it.

    Identity resolution (#341):

    * ``profile_id`` keys the Chrome user-data-dir.
    * ``workflow_id`` keys the runner checkpoint file.
    * ``state_key`` is the legacy single-field input; when set alone it
      routes to both (Phase 1 back-compat).

    The returned suite carries ``_profile_id`` + ``_workflow_id`` for new
    consumers, plus ``_state_key`` (= ``workflow_id``) for downstream
    code that hasn't migrated yet.
    """
    signature = plan_signature_from_steps(micro_plan_steps)
    safe_domain = domain.replace(".", "_")
    default_wf = f"micro_{safe_domain}_{signature[:12]}"

    if not (profile_id or workflow_id or state_key):
        workflow_id = default_wf

    resolved_profile, resolved_workflow = resolve_ids(
        state_key=state_key,
        profile_id=profile_id,
        workflow_id=workflow_id,
        plan_signature=signature,
    )
    checkpoint_path = f"{checkpoint_dir}/{resolved_workflow}.json"

    suite: dict[str, Any] = {
        "session_name": f"micro_{safe_domain}",
        "base_url": "",
        "_max_cost": max_cost,
        "_max_time_minutes": max_time_minutes,
        "_resume_state": resume_state,
        "_profile_id": resolved_profile,
        "_workflow_id": resolved_workflow,
        "_state_key": resolved_workflow,
        "_checkpoint_path": checkpoint_path,
        "_plan_signature": signature,
        "_proxy_city": proxy_city,
        "_proxy_state": proxy_state,
        "_proxy_disabled": bool(proxy_disabled),
        "_micro_plan": micro_plan_steps,
        "tasks": [],
    }
    if objective:
        suite["_objective"] = objective
    return suite


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
                log_text = log_file.read_text(errors="ignore")[-20000:]
            raise RuntimeError(f"{label} crashed during startup:\n{log_text}")
        time.sleep(poll_interval)

    log_text = Path(log_path).read_text(errors="ignore")[-20000:] if Path(log_path).exists() else ""
    raise RuntimeError(f"{label} startup timeout:\n{log_text}")
