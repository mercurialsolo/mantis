"""``BasetenCUARuntime`` — model lifecycle, run dispatch, llama-server.

Owns:

- The model singleton (Holo3 / Gemma4 / etc.) and its in-pod
  llama.cpp server lifecycle
- The detached-run thread pool and per-run state directory
- The Anthropic key resolver per tenant
- The CUA / micro-plan runners that the FastAPI routes invoke

Routes live in :mod:`.routes`. Path helpers live in :mod:`.paths`.
Auth + secret middleware live in :mod:`.middleware`.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


from ..gym.xdotool_env import XdotoolGymEnv
from ..server_utils import (
    build_micro_result,
    build_micro_suite,
    build_proxy_config,
    micro_plan_steps_to_dicts,
    parse_lead_row,
    persist_run_artifacts,
    plan_signature_from_steps,
    result_summary,
    safe_state_key,
    save_result_json,
    start_local_proxy,
    utc_now,
    wait_for_openai_server,
    write_leads_csv,
)
from ..tenant_auth import DEFAULT_TENANT
from .logging_setup import DetachedRunLogHandler
from .middleware import load_secret_environment, read_secret, resolve_anthropic_key
from .paths import (
    data_root,
    find_gguf,
    find_mmproj,
    new_run_id,
    repo_root,
    tenant_chrome_profile,
    tenant_root,
    tenant_state_key,
)


logger = logging.getLogger("mantis_agent.baseten_server.runtime")


# Backwards-compat single-underscore aliases used inside the class body.
# The class was carved out of the old single-file module where these names
# lived alongside the runtime; rather than rewrite ~50 call sites, we keep
# the aliases pointing at the canonical names from sibling modules.
_data_root = data_root
_tenant_root = tenant_root
_tenant_state_key = tenant_state_key
_tenant_chrome_profile = tenant_chrome_profile
_repo_root = repo_root
_new_run_id = new_run_id
_find_gguf = find_gguf
_find_mmproj = find_mmproj
_resolve_anthropic_key = resolve_anthropic_key
_load_secret_environment = load_secret_environment
_read_secret = read_secret
_DetachedRunLogHandler = DetachedRunLogHandler
# Aliases for server_utils helpers used by name throughout the carved
# class body. The originals live in ``server_utils``; these names are kept
# only so the moved code does not need a global rename.
_safe_state_key = safe_state_key
_utc_now = utc_now
_parse_lead_row = parse_lead_row
_write_leads_csv = write_leads_csv
_plan_signature_from_steps = plan_signature_from_steps
_wait_for_openai_server = wait_for_openai_server
_start_local_proxy = start_local_proxy
_build_proxy_config = build_proxy_config


# ── #311: container-scoped Chrome session cache ───────────────────────
#
# A fresh ``/v1/cua`` request pays ~10 s for Xvfb + Chrome launch + first
# navigation before the brain runs. Cache keyed on
# ``(profile_dir, proxy_server)`` so successive requests with the same
# tenant + same profile + same proxy reuse the live browser process.
# ``reset(start_url=...)`` inside the cached env handles the in-tab
# navigation in <500 ms.
#
# Cross-tenant AND cross-profile isolation is preserved by
# :func:`_chrome_profile_dir_for_suite`, which derives the path from
# ``task_suite["_profile_id"]`` — a value already namespaced as
# ``<tenant_id>__<caller_profile_id>`` by
# :func:`server_utils.build_micro_suite`. Different profile_ids land
# on different cache keys (and on disk in different user-data-dirs)
# so cookies / localStorage don't leak across them. Earlier code passed
# a flat ``data_root / "chrome-profile"`` here — the cache key was the
# same string across tenants, and the on-disk profile was shared too.
#
# Set ``MANTIS_CHROME_REUSE=disabled`` to fall back to the per-request
# launch path (the legacy behaviour). Callers can also opt out per request
# via ``payload["reuse_session"] = False``.
_chrome_env_cache: dict[tuple[str, str], tuple[XdotoolGymEnv, Any]] = {}
_chrome_env_cache_lock = threading.Lock()


def _chrome_reuse_enabled() -> bool:
    return os.environ.get("MANTIS_CHROME_REUSE", "enabled").lower() != "disabled"


def _should_ground_cua_clicks(payload: dict[str, Any], *, has_anthropic_key: bool) -> bool:
    """#931 P2: whether /v1/cua should refine the brain's clicks with the
    screenshot grounding model.

    True only when the caller opted in (``ground_clicks``) AND a key is
    available — without a key ClaudeGrounding can't run, so we fall back to
    brain-only rather than hard-failing the request.
    """
    return bool(payload.get("ground_clicks")) and has_anthropic_key


def _chrome_profile_dir_for_suite(task_suite: dict[str, Any]) -> str:
    """Per-tenant, per-profile Chrome user-data-dir for a Baseten request.

    Mirrors the same isolation guarantee the Modal path got in PR #426.
    The earlier Baseten code passed ``data_root / "chrome-profile"`` — a
    flat path shared across every request — which the surrounding
    comment claimed was tenant-scoped but wasn't. Cookies / localStorage /
    IndexedDB leaked across tenants AND across ``profile_id``s on the
    same container.

    ``task_suite["_profile_id"]`` is already server-prefixed (the
    ``<tenant_id>__<caller_profile_id>`` shape produced by
    :func:`server_utils.build_micro_suite`), so a single path
    segment gives us both isolations at once. Falling back to
    ``"default"`` keeps legacy / unscoped callers working, just with
    them sharing one default profile rather than the previous "everyone
    shares one chrome-profile" behaviour.

    The session-reuse cache (``_chrome_env_cache``) keys on this same
    path, so two requests with different ``profile_id``s no longer
    accidentally share a cached Chrome instance.
    """
    profile_id = (
        task_suite.get("_profile_id")
        or task_suite.get("_state_key")
        or "default"
    )
    return str(_data_root() / "chrome-profile" / _safe_state_key(str(profile_id)))


def _shutdown_chrome_env_cache() -> None:
    """Force-close every cached env. Wired into container shutdown hooks
    so reused processes don't leak across container lifetimes."""
    with _chrome_env_cache_lock:
        for env, proxy_proc in list(_chrome_env_cache.values()):
            try:
                env.shutdown()
            except Exception as exc:
                logger.warning("chrome-env shutdown failed: %s", exc)
            if proxy_proc is not None:
                try:
                    proxy_proc.terminate()
                except Exception:
                    pass
        _chrome_env_cache.clear()


class BasetenCUARuntime:
    def __init__(self) -> None:
        # ``MANTIS_BRAIN`` is the new public selector; ``MANTIS_MODEL`` stays
        # supported as a fallback for one minor release. ``gemma4-cua`` is
        # an alias for ``gemma4`` carried over from the legacy name.
        self.model_kind = (
            os.environ.get("MANTIS_BRAIN")
            or os.environ.get("MANTIS_MODEL")
            or "holo3"
        )
        self.port = int(os.environ.get("MANTIS_LLAMA_PORT", "18080"))
        self.llama_proc: subprocess.Popen | None = None
        self._llama_log_fh: Any = None
        self.brain: Any = None
        # #911: served-model-name override set when a vLLM LoRA challenger boots.
        self._lora_served_name: str | None = None
        # Replica-wide concurrency limiter. Pre-fix this was a single
        # ``threading.Lock`` that every run path acquired — so
        # ``max_concurrent_runs=N`` allowed N submits but only ONE
        # actually ran (the others sat queued for the full duration of
        # the active run). Replaced with a counting semaphore that
        # honours per-tenant ``max_concurrent_runs`` by allowing
        # ``MANTIS_RUNTIME_CONCURRENCY`` concurrent runs server-wide.
        # Default 1 preserves the legacy serialized behaviour; the
        # operator bumps it to match GPU / RAM capacity. ``with
        # self.lock:`` continues to work because Semaphore is a
        # context manager (acquire on __enter__, release on __exit__).
        try:
            _runtime_concurrency = max(
                1, int(os.environ.get("MANTIS_RUNTIME_CONCURRENCY") or "1")
            )
        except ValueError:
            _runtime_concurrency = 1
        self.lock: threading.Semaphore = threading.Semaphore(_runtime_concurrency)
        self._runtime_concurrency = _runtime_concurrency
        self.detached_threads: dict[str, threading.Thread] = {}
        self.loaded = False
        # #117: process-level shared grounding cache. Listing-card layouts
        # repeat both across pages within a run AND across runs / tenants on
        # the same site, so a single cache amortises Claude grounding cost
        # across the whole server lifetime. Bounded by GroundingCache's LRU
        # cap; entries TTL out at 1 h by default.
        from ..grounding_cache import GroundingCache
        self.grounding_cache: GroundingCache = GroundingCache()
        # Cross-replica run-state store (#866 pattern, ported to mantis-server).
        # pause_state.json + payload.json live on the /data Modal Volume, whose
        # cross-container visibility is eventually-consistent — a resume landing
        # on a different replica can miss them (→ wedged-paused / re-decompose,
        # or "pause_state missing"). Mirror them into a modal.Dict so resume
        # reads are immediate and replica-independent; disk stays the durable
        # backstop. Gated on Modal — ``modal.Dict.from_name`` BLOCKS off-Modal
        # (no exception/timeout), so MODAL_TASK_ID is the only safe gate
        # (feedback_modal_dict_blocks_off_modal).
        self._run_state_store = self._build_run_state_store()

    @staticmethod
    def _build_run_state_store() -> Any:
        from ..run_state_store import NullRunStateStore, RunStateStore
        if not os.environ.get("MODAL_TASK_ID"):
            return NullRunStateStore()
        try:
            name = os.environ.get("MANTIS_RUN_STATE_DICT", "mantis-server-run-state")
            return RunStateStore.from_name(name)
        except Exception as exc:  # noqa: BLE001 — store fault ≠ run fault; disk backstops
            logger.warning("run-state store unavailable; disk-only resume: %s", exc)
            return NullRunStateStore()

    @staticmethod
    def _store_tenant() -> str:
        return _safe_state_key(
            os.environ.get("MANTIS_TENANT_ID") or DEFAULT_TENANT.tenant_id
        )

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
        elif self.model_kind == "fara":
            self.brain = self._load_fara()
        else:
            raise RuntimeError(f"unsupported MANTIS_MODEL={self.model_kind!r}")

        # #118: optional SpeculativeBrain wrapping. The wrapper overlaps
        # think() with the post-action settle using a worker thread.
        #
        # The strict validator (Hamming distance 0) makes this quality-
        # safe — speculative results never drive an action when the
        # screen changed.
        #
        # **HOWEVER**, on single-llama.cpp deployments (Holo3 on Modal,
        # current production config) the speculative HTTP request and
        # the sync HTTP request serialize on the same GPU. A 55% hit
        # rate on lu.ma still produced a +52% wall-time regression
        # because the speculative call holds GPU time across the action
        # dispatch and the sync fallback waits for GPU to free.
        #
        # Default is therefore ``disabled`` — enable explicitly on
        # multi-replica / multi-GPU deployments where the two HTTP
        # requests land on separate inference workers.
        # See docs/reference/speculative-inference.md for the full
        # ablation data.
        if os.environ.get(
            "MANTIS_SPECULATIVE_INFERENCE", "disabled",
        ).lower() == "enabled":
            from ..speculative_brain import SpeculativeBrain
            self.brain = SpeculativeBrain(self.brain)
            logger.info("brain: wrapped in SpeculativeBrain (#118)")

        self.loaded = True

    def _boot_lora_args(self, backend_model: str) -> tuple[list[str], str | None]:
        """#911 parity: resolve a **deployment-level** LoRA challenger adapter.

        Unlike Modal (which boots a fresh inference server per run and so can take
        a per-request ``_lora_adapter``), the Baseten pod boots one shared
        inference server at model-load. So the adapter is fixed for the
        deployment via ``MANTIS_LORA_ADAPTER`` (+ optional ``MANTIS_LORA_SCALE`` /
        ``MANTIS_LORA_NAME``): the *champion* deployment leaves it unset (serves
        the base); a *challenger* deployment sets it (serves base + adapter). The
        gate then points its two endpoints at the two deployments.

        ``backend_model`` is a representative model for the boot path's runtime
        (``"holo3"`` for llama.cpp, ``"fara"`` for vLLM) so the shared serving
        logic picks the right flags. Returns ``(extra_server_args,
        served_model_name)``; ``([], None)`` when no adapter is configured.
        """
        ref = (os.environ.get("MANTIS_LORA_ADAPTER") or "").strip()
        if not ref:
            return [], None
        from mantis_agent.serving.lora_serving import plan_serving

        suite: dict[str, Any] = {"_lora_adapter": ref}
        if os.environ.get("MANTIS_LORA_SCALE"):
            suite["_lora_scale"] = float(os.environ["MANTIS_LORA_SCALE"])
        if os.environ.get("MANTIS_LORA_NAME"):
            suite["_lora_name"] = os.environ["MANTIS_LORA_NAME"]
        plan = plan_serving(
            cua_model=backend_model,
            suite=suite,
            mounts={},  # Baseten adapters are mounted via the truss `weights:` block
            gguf_cache_root=os.environ.get(
                "MANTIS_LORA_CACHE", f"{os.environ.get('MANTIS_DATA_DIR', '/data')}/lora_cache"
            ),
        )
        if plan.convert_cmd:
            raise RuntimeError(
                "[#911] Baseten LoRA needs a pre-converted .gguf adapter — point "
                "MANTIS_LORA_ADAPTER at the .gguf (mounted via the truss weights: "
                f"block), not a PEFT dir. got: {ref!r}"
            )
        logger.warning(
            "[#911] Baseten serving challenger adapter tag=%s args=%s served=%s",
            plan.challenger_tag, plan.extra_server_args, plan.served_model_name,
        )
        return plan.extra_server_args, plan.served_model_name

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
        # #911: deployment-level LoRA challenger (llama.cpp --lora). No-op unless
        # MANTIS_LORA_ADAPTER is set. Folds into the base → served name unchanged.
        lora_args, _ = self._boot_lora_args("holo3")
        cmd.extend(lora_args)

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

    def _start_vllm(self, model_path: Path, extra_args: list[str]) -> None:
        """Start a vLLM OpenAI-compatible server in the pod.

        Used by Fara (Qwen2.5-VL native) — no llama.cpp / GGUF needed.
        Mirrors ``_start_llama``'s lifecycle: subprocess pinned to
        ``self.llama_proc`` (the field is generically the inference
        server handle; shutdown wires already terminate it).
        """
        import sys

        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", str(model_path),
            "--served-model-name", "model",
            "--host", "0.0.0.0",
            "--port", str(self.port),
            "--trust-remote-code",
            "--gpu-memory-utilization",
            os.environ.get("MANTIS_VLLM_GPU_UTIL", "0.90"),
            "--max-model-len",
            os.environ.get("MANTIS_CONTEXT_SIZE", "32768"),
            "--dtype", os.environ.get("MANTIS_VLLM_DTYPE", "auto"),
        ]
        cmd.extend(extra_args)
        # #911: deployment-level LoRA challenger (vLLM --enable-lora). vLLM serves
        # the adapter under its own name, so stash it for the brain to request.
        lora_args, served = self._boot_lora_args("fara")
        cmd.extend(lora_args)
        self._lora_served_name = served

        logger.info("starting vllm: %s", " ".join(cmd))
        self._llama_log_fh = open("/tmp/vllm.log", "w")
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
        _wait_for_openai_server(
            self.port, self.llama_proc, "vllm",
            log_path="/tmp/vllm.log",
            timeout_seconds=int(os.environ.get("MANTIS_VLLM_TIMEOUT_S", "900")),
        )

    def _load_fara(self) -> Any:
        """Load Microsoft Fara-7B via vLLM and return a ``FaraBrain``.

        Expects weights pre-mounted at ``MANTIS_FARA_MODEL_DIR``
        (default ``/models/fara``) by the Baseten ``weights:`` block;
        falls back to the HF repo id ``microsoft/Fara-7B`` so a dev
        deployment with HF auth can pull on demand.
        """
        from mantis_agent.brain_fara import FaraBrain

        model_dir = Path(
            os.environ.get("MANTIS_FARA_MODEL_DIR", "/models/fara")
        )
        model_ref: str | Path = model_dir if model_dir.exists() else (
            os.environ.get("MANTIS_FARA_REPO", "microsoft/Fara-7B")
        )
        # vLLM 0.11+ requires both flags to honour ``tool_choice="auto"``.
        # Without them, every FaraBrain.think() request 400s with
        # `"auto" tool choice requires --enable-auto-tool-choice and
        # --tool-call-parser to be set`. ``hermes`` is the Qwen-family
        # parser that matches Fara's Qwen2.5-VL base and OpenAI-format
        # tool_calls. Override via MANTIS_FARA_TOOL_PARSER for fine-tunes
        # that emit a different schema.
        tool_parser = os.environ.get("MANTIS_FARA_TOOL_PARSER", "hermes")
        self._start_vllm(
            Path(str(model_ref)),
            extra_args=[
                "--enable-auto-tool-choice",
                "--tool-call-parser", tool_parser,
            ],
        )

        brain = FaraBrain(
            base_url=f"http://127.0.0.1:{self.port}/v1",
            # #911: when a LoRA challenger is active, vLLM serves it under a
            # distinct name — request that, not the base "model".
            model=getattr(self, "_lora_served_name", None) or "model",
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
        # ``cancel`` and ``pause`` MUST dispatch ahead of the
        # ``detached`` check — ``PredictRequest`` defaults ``detached``
        # to true, so an action-only call would otherwise fall into
        # ``_start_detached`` and either 400 on an already-running run
        # or overwrite a finished run as a new submission. Mirrors the
        # Modal CUA fix in ``modal_cua_server.predict`` (#866).
        if action in {
            "status", "result", "logs", "resume", "cancel", "pause",
            "reasoning_trace",
        }:
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
        """Resolve the on-disk run directory.

        Tenant-scoped (``<root>/tenants/<tenant>/runs/<run_id>/``) to
        match Modal's ``_run_dir`` and the Baseten artifact reader at
        ``routes.get_run_artifact``. Pre-fix this method returned the
        un-scoped ``<root>/runs/<run_id>/`` so writes from
        ``_save_detached_result`` (``leads.csv`` / ``extracted_rows.json``
        / ``result.json``) landed in a directory the artifact endpoint
        never read — every artifact request 404'd.

        On a read, we fall back to the legacy un-scoped location when
        the scoped dir doesn't exist, so runs persisted before this fix
        stay reachable until the next container restart.

        ``MANTIS_TENANT_ID`` env var feeds the tenant — single-tenant
        container model (each replica binds one tenant).
        """
        safe_run_id = _safe_state_key(run_id)
        if not safe_run_id or safe_run_id != run_id:
            raise ValueError(f"invalid run_id: {run_id!r}")
        tenant_id = _safe_state_key(
            os.environ.get("MANTIS_TENANT_ID") or DEFAULT_TENANT.tenant_id
        )
        scoped = _data_root() / "tenants" / tenant_id / "runs" / run_id
        if create:
            scoped.mkdir(parents=True, exist_ok=True)
            return scoped
        # Read path: prefer the scoped location; fall back to the legacy
        # un-scoped layout for runs persisted before this fix landed.
        if scoped.exists():
            return scoped
        legacy = _data_root() / "runs" / run_id
        if legacy.exists():
            return legacy
        return scoped

    def _write_json_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2))
        tmp_path.replace(path)

    def _read_json_file(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(str(path))
        return json.loads(path.read_text())

    # First-terminal-wins guard (mirrors ``modal_cua_server`` #866).
    # Once a run reaches any of these the late worker write can't
    # clobber an API-side cancel — the brain's terminal status loses
    # to the operator's intent.
    _TERMINAL_STATUSES: frozenset[str] = frozenset({
        "succeeded", "failed", "cancelled", "completed_with_failures",
        "timeout", "halted",
    })

    def _write_detached_status(self, run_id: str, status: dict[str, Any]) -> dict[str, Any]:
        run_dir = self._run_path(run_id, create=True)
        status_path = run_dir / "status.json"
        existing: dict[str, Any] = {}
        if status_path.exists():
            try:
                existing = json.loads(status_path.read_text())
            except Exception:
                existing = {}

        existing_phase = str(existing.get("status", "")).lower()
        incoming_phase = str(status.get("status", "")).lower()
        if (
            existing_phase in self._TERMINAL_STATUSES
            and incoming_phase
            and incoming_phase != existing_phase
        ):
            # Refuse the terminal overwrite — return the existing
            # status unchanged so the caller's polling stays
            # coherent. Important so a late worker write doesn't
            # clobber an operator's cancel.
            logger.warning(
                "refusing to overwrite terminal status "
                "run=%s cur=%r new=%r",
                run_id, existing_phase, incoming_phase,
            )
            return existing

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
        # Mirror into the cross-replica store so a resume / poll landing on a
        # different replica sees the current phase immediately (the resume
        # handler's status=='paused' gate would otherwise 400 on an un-synced
        # Volume read).
        from ..run_state_store import KIND_STATUS
        self._run_state_store.put(self._store_tenant(), run_id, KIND_STATUS, merged)
        return merged

    def _read_detached_status(self, run_id: str) -> dict[str, Any]:
        """Status read, cache-first (cross-replica) with disk fallback."""
        from ..run_state_store import KIND_STATUS, read_with_store

        def _disk() -> dict[str, Any] | None:
            try:
                return self._read_json_file(self._run_path(run_id) / "status.json")
            except FileNotFoundError:
                return None

        return read_with_store(
            self._run_state_store, tenant_id=self._store_tenant(),
            run_id=run_id, kind=KIND_STATUS, disk_reader=_disk,
        ) or {}

    def _append_detached_event(self, run_id: str, message: str) -> None:
        run_dir = self._run_path(run_id, create=True)
        line = json.dumps({"ts": _utc_now(), "message": message}, sort_keys=True)
        with (run_dir / "events.log").open("a") as handle:
            handle.write(line + "\n")

    # ── Augur metadata sidecar (parity with Modal #838) ───────────
    #
    # The runner mints an Augur ``run_id`` distinct from the API
    # run id (the Augur SDK groups its Runs view on it). We persist
    # the mapping into ``<run_dir>/augur.json`` so the lifecycle
    # endpoints can cross-link the API run id to the Augur bundle
    # without the executor and the API speaking directly.

    def _augur_bundle_dir(self, augur_run_id: str) -> Path:
        """Resolve the on-disk Augur bundle directory.

        Mirrors :func:`mantis_agent.observability.augur.default_out_dir`
        but does not import augur-sdk (the bundle root is the file
        layout we own; we don't need the SDK in the API hot path).
        Honors ``MANTIS_AUGUR_DIR`` for the same override the runner
        reads.
        """
        override = os.environ.get("MANTIS_AUGUR_DIR", "").strip()
        if override:
            return Path(override) / augur_run_id
        return _data_root() / "augur" / augur_run_id

    def _write_augur_metadata(
        self, run_id: str, augur_run_id: str,
    ) -> None:
        """Persist the API-run-id → augur-run-id mapping.

        Best-effort — telemetry must never break a finishing run, so
        callers wrap with try/except. The lifecycle / augur endpoints
        read this on every poll.
        """
        if not augur_run_id:
            return
        run_dir = self._run_path(run_id, create=True)
        blob = {
            "augur_run_id": augur_run_id,
            "bundle_dir": str(self._augur_bundle_dir(augur_run_id)),
            "dsn_workspace": os.environ.get(
                "AUGUR_DSN_WORKSPACE_URL", "",
            ) or "",
        }
        self._write_json_atomic(run_dir / "augur.json", blob)

    def _read_augur_metadata(self, run_id: str) -> dict[str, Any] | None:
        path = self._run_path(run_id) / "augur.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    # ── Status enrichment (parity with Modal #841 + viewer surface) ──

    def _enrich_status(
        self, run_id: str, status: dict[str, Any],
    ) -> dict[str, Any]:
        """Annotate a status blob with derived fields the polling
        callers expect.

        - ``viewer_url`` from any sidecar viewer.json (file is dropped
          alongside status.json by the viewer setup path; falls back
          to a value already merged into status.json).
        - ``failure_help`` synthesized from ``halt_class`` when the
          run reached a terminal failure phase and the runner hasn't
          already attached one.

        Idempotent — re-running on an already-enriched dict is a no-op.
        """
        # viewer_url sidecar (#416 parity). ``_maybe_start_live_viewer``
        # already merges the URL into status.json, but a tenant-side
        # tool may also drop a ``viewer.json`` file separately; honor
        # both shapes.
        if not status.get("viewer_url"):
            viewer_path = self._run_path(run_id) / "viewer.json"
            if viewer_path.exists():
                try:
                    blob = json.loads(viewer_path.read_text())
                    if isinstance(blob, dict) and blob.get("viewer_url"):
                        status["viewer_url"] = str(blob["viewer_url"])
                except (OSError, ValueError, json.JSONDecodeError):
                    pass

        # Identity-field surface for Modal parity. Modal exposes these
        # at top-level on status.json so callers can switch endpoints
        # without re-reading task_suite. The fields live inside
        # ``status['payload']['task_suite']`` on Baseten today; lift
        # them when they aren't already at top-level.
        payload = status.get("payload") if isinstance(status.get("payload"), dict) else {}
        suite = payload.get("task_suite") if isinstance(payload.get("task_suite"), dict) else {}
        for field, source_keys in (
            ("profile_id", ("_profile_id", "profile_id")),
            ("workflow_id", ("_workflow_id", "workflow_id")),
            ("state_key", ("_state_key", "state_key")),
            ("tenant_id", ("_tenant_id", "tenant_id")),
        ):
            if not status.get(field):
                for source in source_keys:
                    if suite.get(source):
                        status[field] = str(suite[source])
                        break
                    if payload.get(source):
                        status[field] = str(payload[source])
                        break
        if not status.get("max_steps"):
            for src in (payload, suite):
                if isinstance(src.get("max_steps"), (int, float)):
                    status["max_steps"] = int(src["max_steps"])
                    break

        cur_phase = str(status.get("status", "")).lower()
        if cur_phase in self._TERMINAL_STATUSES and not status.get("failure_help"):
            halt_class = str(status.get("halt_class") or "").strip()
            if not halt_class:
                # Some halts carry the wire reason on ``halt_reason``
                # rather than ``halt_class`` (the runner's older
                # field). Treat that as the same signal so the help
                # taxonomy still fires.
                halt_class = str(status.get("halt_reason") or "").strip()
            if halt_class or cur_phase in {
                "halted", "failed", "completed_with_failures",
                "timeout", "cancelled",
            }:
                try:
                    from ..run_failure_help import failure_help_for
                    status["failure_help"] = failure_help_for(
                        halt_class or cur_phase, run_id=run_id,
                    )
                except Exception as exc:  # noqa: BLE001 — diagnostic only
                    logger.debug(
                        "failure_help synthesis failed for %s: %s",
                        run_id, exc,
                    )
        return status

    def _save_pause_state(self, run_id: str, pause_state: dict[str, Any]) -> Path:
        """Persist a paused run's :class:`PauseState` blob (#344).

        Mirrors into the cross-replica store so a resume on a different replica
        reads it immediately (disk is the durable backstop)."""
        run_dir = self._run_path(run_id, create=True)
        path = run_dir / "pause_state.json"
        self._write_json_atomic(path, pause_state)
        from ..run_state_store import KIND_PAUSE_STATE
        self._run_state_store.put(self._store_tenant(), run_id, KIND_PAUSE_STATE, pause_state)
        return path

    def _read_pause_state(self, run_id: str) -> dict[str, Any]:
        """Read pause_state — cache-first (cross-replica), disk fallback (#344)."""
        from ..run_state_store import KIND_PAUSE_STATE, read_with_store

        def _disk() -> dict[str, Any] | None:
            path = self._run_path(run_id) / "pause_state.json"
            if not path.exists():
                return None
            try:
                return json.loads(path.read_text())
            except Exception:  # noqa: BLE001
                return None

        return read_with_store(
            self._run_state_store, tenant_id=self._store_tenant(),
            run_id=run_id, kind=KIND_PAUSE_STATE, disk_reader=_disk,
        ) or {}

    def _save_resume_payload(self, run_id: str, payload: dict[str, Any]) -> Path:
        """Persist the original payload so resume can rebuild the run (#344)."""
        run_dir = self._run_path(run_id, create=True)
        path = run_dir / "payload.json"
        # Skip transient fields that would be wrong to re-apply on resume.
        keep = {k: v for k, v in payload.items() if not k.startswith("_resume")}
        self._write_json_atomic(path, keep)
        from ..run_state_store import KIND_RESUME_PAYLOAD
        self._run_state_store.put(self._store_tenant(), run_id, KIND_RESUME_PAYLOAD, keep)
        return path

    def _read_resume_payload(self, run_id: str) -> dict[str, Any]:
        from ..run_state_store import KIND_RESUME_PAYLOAD, read_with_store

        def _disk() -> dict[str, Any] | None:
            path = self._run_path(run_id) / "payload.json"
            if not path.exists():
                return None
            try:
                return json.loads(path.read_text())
            except Exception:  # noqa: BLE001
                return None

        return read_with_store(
            self._run_state_store, tenant_id=self._store_tenant(),
            run_id=run_id, kind=KIND_RESUME_PAYLOAD, disk_reader=_disk,
        ) or {}

    def _start_detached(self, payload: dict[str, Any]) -> dict[str, Any]:
        run_id = _safe_state_key(str(payload.get("run_id") or _new_run_id()))
        if run_id in self.detached_threads and self.detached_threads[run_id].is_alive():
            raise RuntimeError(f"detached run already exists and is active: {run_id}")

        run_payload = dict(payload)
        run_payload.pop("detached", None)
        run_payload["_detached_run_id"] = run_id
        run_payload["_detached_started_at"] = _utc_now()
        # Snapshot the payload so a later action=resume can rebuild the run
        # (#344). Skipped for resume continuations (the original payload is
        # already on disk).
        if not run_payload.get("_resume_pause_state"):
            self._save_resume_payload(run_id, run_payload)

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
                # Mint a per-session Augur run id and stamp the
                # sidecar (Modal #838 parity). Distinct from the API
                # ``run_id`` so the Augur Runs view doesn't pile
                # overlapping rows under one identifier. Best-effort:
                # an augur write failure must never break a finishing
                # run, and a redo on resume is fine because the existing
                # sidecar wins via _read_augur_metadata.
                if not self._read_augur_metadata(run_id):
                    try:
                        import uuid as _uuid_for_augur
                        workflow_id = str(payload.get("workflow_id") or "")
                        augur_run_id = (
                            f"{workflow_id}-{_uuid_for_augur.uuid4().hex[:8]}"
                            if workflow_id
                            else _uuid_for_augur.uuid4().hex[:12]
                        )
                        self._write_augur_metadata(run_id, augur_run_id)
                    except Exception as exc:  # noqa: BLE001 — telemetry
                        logger.warning(
                            "augur metadata write failed run=%s: %s",
                            run_id, exc,
                        )
                task_suite = None
                if payload.get("_mode") == "pure_cua":
                    result = self._run_pure_cua(payload, run_id=run_id)
                else:
                    task_suite = self._task_suite_from_payload(payload)
                    if task_suite.get("_micro_plan"):
                        result = self._run_micro(task_suite, payload, run_id=run_id)
                    else:
                        result = self._run_tasks(task_suite, payload, run_id=run_id)
                # #344: paused branch — runner raised PauseRequested through
                # the default request_user_input tool. Stash the PauseState
                # snapshot so action=resume can rebuild the runner, and
                # surface prompt / reason on status.json. The poller hits
                # this exact shape on the next ``action=status`` round-trip.
                if result.get("_paused"):
                    pause_payload = result.pop("pause_state", None) or {}
                    # Persist the RESOLVED micro-suite in the checkpoint so
                    # action=resume restores the exact paused plan verbatim —
                    # never re-decompose (LLM, non-deterministic) or re-load the
                    # container-local ``decomposed_<hash>.json`` cache, which can
                    # MISS on a different replica → fresh plan → signature
                    # mismatch → run wedged in 'paused' forever (end-user bug,
                    # run 20260615_074614). Keyed implicitly by run_id (the file).
                    if task_suite and not pause_payload.get("resolved_task_suite"):
                        pause_payload["resolved_task_suite"] = task_suite
                    self._save_pause_state(run_id, pause_payload)
                    self._save_detached_result(run_id, result)
                    self._write_detached_status(
                        run_id,
                        {
                            "status": "paused",
                            "paused_at": _utc_now(),
                            "prompt": str(result.get("prompt", "")),
                            "reason": str(result.get("reason", "user_input")),
                            "summary": self._result_summary(result),
                        },
                    )
                    self._append_detached_event(run_id, "paused")
                    return
                self._save_detached_result(run_id, result)
                finished_at = _utc_now()
                # #audit item 4: read the honest terminal_status from
                # the runner's result envelope and use it as the
                # canonical detached-status. The legacy practice of
                # writing "succeeded" on any non-exception result hid
                # halts (REQUIRED step failures, budget/time caps) as
                # successful runs.
                #
                # Wire-level status mapping for callers that consume
                # action=status:
                #   completed              → succeeded (back-compat)
                #   completed_with_failures → completed_with_failures
                #   halted / budget_exceeded / time_exceeded → halted
                #     (with status_detail carrying the specific reason)
                #   anything else → succeeded (defensive — old plans
                #     don't surface terminal_status)
                rt_status = ""
                halt_reason = ""
                if isinstance(result, dict):
                    rt_status = str(result.get("terminal_status") or "")
                    halt_reason = str(result.get("halt_reason") or "")
                if rt_status == "completed":
                    wire_status = "succeeded"
                elif rt_status == "completed_with_failures":
                    wire_status = "completed_with_failures"
                elif rt_status in ("halted", "budget_exceeded", "time_exceeded"):
                    wire_status = "halted"
                else:
                    wire_status = "succeeded"
                status_blob: dict[str, Any] = {
                    "status": wire_status,
                    "finished_at": finished_at,
                    "summary": self._result_summary(result),
                }
                if rt_status:
                    status_blob["terminal_status"] = rt_status
                if halt_reason:
                    status_blob["halt_reason"] = halt_reason
                self._write_detached_status(run_id, status_blob)
                self._append_detached_event(run_id, wire_status)
                self._append_runs_log(
                    run_id, payload, result,
                    status=wire_status, finished_at=finished_at,
                )
        except Exception as exc:
            logger.exception("detached run %s failed", run_id)
            finished_at = _utc_now()
            self._write_detached_status(
                run_id,
                {
                    "status": "failed",
                    "finished_at": finished_at,
                    "error": str(exc),
                    "traceback": traceback.format_exc()[-4000:],
                },
            )
            self._append_detached_event(run_id, f"failed: {exc}")
            self._append_runs_log(
                run_id, payload, result=None,
                status="failed", finished_at=finished_at, error=str(exc),
            )
        finally:
            agent_logger.removeHandler(handler)
            handler.close()

    def _append_runs_log(
        self,
        run_id: str,
        payload: dict[str, Any],
        result: dict[str, Any] | None,
        *,
        status: str,
        finished_at: str,
        error: str | None = None,
    ) -> None:
        """Append a JSONL row to ``$MANTIS_DATA_DIR/runs_log/<YYYY-MM>.jsonl``
        on terminal status (epic #362 Phase C). Best-effort — bookkeeping
        I/O must never break a finishing run.
        """
        try:
            from ..runs_log import append_run, row_from_result

            tenant_id = os.environ.get("MANTIS_TENANT_ID", "default")
            row = row_from_result(
                run_id=run_id,
                tenant_id=tenant_id,
                profile_id=str(payload.get("profile_id") or ""),
                workflow_id=str(payload.get("workflow_id") or ""),
                plan_signature=str((result or {}).get("plan_signature") or ""),
                model=str((result or {}).get("model") or ""),
                status=status,
                created_at=str(payload.get("_created_at") or ""),
                finished_at=finished_at,
                result=result,
                error=error,
            )
            append_run(row)
        except Exception as exc:  # noqa: BLE001 — observability, never fatal
            logger.debug("runs_log: append failed for %s: %s", run_id, exc)

    def _save_detached_result(self, run_id: str, result: dict[str, Any]) -> None:
        run_dir = self._run_path(run_id, create=True)
        run_result_path = run_dir / "result.json"
        result["detached_result_path"] = str(run_result_path)
        leads = result.get("leads")
        if isinstance(leads, list):
            csv_path = run_dir / "leads.csv"
            _write_leads_csv(csv_path, leads)
            result["detached_csv_path"] = str(csv_path)
        # #508: also write the schema-driven CSV / JSON next to leads.csv
        # in the same run dir, and append file artifacts so the artifact
        # endpoint (GET /v1/runs/{id}/artifacts/{name}) has stable
        # download targets. ``persist_run_artifacts`` re-writes leads.csv
        # too, which is harmless — same content, same path.
        file_artifacts = persist_run_artifacts(result, run_dir, run_id=run_id)
        if file_artifacts:
            result["artifacts"] = list(result.get("artifacts") or []) + file_artifacts
        self._write_json_atomic(run_result_path, result)

        # Phase 2 M3 (#699) — capture the post-run Chrome profile dir
        # into the snapshot bucket. Env-gated; no-op when
        # MANTIS_PROFILE_SNAPSHOT_BUCKET is unset. Best-effort.
        try:
            from ..observability.snapshot_lifecycle import (
                maybe_capture_snapshot,
            )
            profile_id_raw = str(
                result.get("profile_id") or ""
            )
            tenant_id_raw = (
                os.environ.get("MANTIS_TENANT_ID")
                or DEFAULT_TENANT.tenant_id
            )
            # Chrome profile dir on Baseten is keyed on the raw
            # profile_id (see chrome_profile_dir() at line 150).
            from ..server_utils import safe_state_key as _ssk
            profile_dir = (
                _data_root() / "chrome-profile" / _ssk(profile_id_raw)
            )
            maybe_capture_snapshot(
                tenant_id=str(tenant_id_raw),
                profile_id=profile_id_raw,
                source_profile_dir=profile_dir,
                notes=f"executor=baseten terminal={result.get('terminal_status', '')}",
                captured_in={
                    "executor": "baseten_runtime",
                    "run_id": run_id,
                    "workflow_id": str(result.get("workflow_id") or ""),
                },
            )
        except Exception as _capture_exc:  # noqa: BLE001 — paranoia
            logger.warning(
                "snapshot capture wrapper raised: %s", _capture_exc,
            )

    def _result_summary(self, result: dict[str, Any]) -> dict[str, Any]:
        return result_summary(result)

    def _detached_action(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        run_id = str(payload.get("run_id") or "")
        if not run_id:
            raise ValueError("run_id is required")
        run_dir = self._run_path(run_id)

        # ``cancel`` / ``pause`` MUST be a 404 when the run isn't
        # known — otherwise the dispatch is back-pressured into
        # the run-submission path (the bug this fix prevents). The
        # status read below is permissive (returns ``{}`` on
        # missing); explicitly check existence so misrouted cancels
        # surface as an error the caller can act on.
        if action in {"cancel", "pause"}:
            status_path = run_dir / "status.json"
            if not status_path.exists():
                raise FileNotFoundError(f"unknown run_id: {run_id}")

        if action == "cancel":
            # First-terminal-wins (mirrors ``modal_cua_server`` #866).
            # A cancel on a finished run reports the existing status
            # without overwriting; a cancel on an active run drops
            # the cancel sentinel + flips status to ``cancelled``.
            status = self._read_json_file(status_path)
            cur = str(status.get("status", "")).lower()
            terminal = {
                "succeeded", "failed", "cancelled",
                "completed_with_failures", "timeout", "halted",
            }
            if cur in terminal:
                return self._enrich_status(run_id, status)
            now_iso = _utc_now()
            cancel_lookup_error: str | None = None
            # Loosely mirrors Modal ``modal_cua_server.py:2731-2745`` —
            # surface infra failures (sentinel write OSError, thread
            # signal failure) on the response WITHOUT failing the cancel
            # itself. The operator's intent always lands; the diagnostic
            # tells them whether the worker thread will observe it.
            try:
                (run_dir / "cancel_request.json").write_text(
                    json.dumps({"requested_at": now_iso}),
                    encoding="utf-8",
                )
            except OSError as exc:
                cancel_lookup_error = f"sentinel_write_failed: {exc}"
                logger.warning(
                    "cancel sentinel write failed for run=%s: %s",
                    run_id, exc,
                )
            # Best-effort thread-state sanity check. The worker thread
            # observes the sentinel via the gym checkpoint pump; we
            # don't kill it here (Python lacks a safe thread-kill),
            # but we do surface whether the thread is reachable for
            # the operator to interpret.
            try:
                thread = self.detached_threads.get(run_id)
                if thread is not None and not thread.is_alive():
                    # Worker already exited but a stale status said
                    # ``running`` — surface so the caller can take
                    # appropriate cleanup action.
                    if cancel_lookup_error is None:
                        cancel_lookup_error = (
                            "worker_thread_not_alive: cancel sentinel "
                            "written but no worker is polling it"
                        )
            except Exception as exc:  # noqa: BLE001 — diagnostic only
                if cancel_lookup_error is None:
                    cancel_lookup_error = (
                        f"thread_lookup_failed: {type(exc).__name__}: {exc}"
                    )
            status["status"] = "cancelled"
            status["cancelled_at"] = now_iso
            status["updated_at"] = now_iso
            if cancel_lookup_error:
                status["cancel_lookup_error"] = cancel_lookup_error
            self._write_detached_status(run_id, status)
            self._append_detached_event(run_id, "cancelled")
            return self._enrich_status(run_id, status)

        if action == "pause":
            # External pause (mirrors Modal #541). The detached worker
            # checks the sentinel between micro steps via
            # :func:`gym.external_pause.wait_while_paused` and stalls
            # until ``action=resume`` clears it.
            status = self._read_json_file(status_path)
            cur = str(status.get("status", "")).lower()
            if cur not in {"queued", "running"}:
                raise ValueError(
                    f"action='pause' requires a running run; "
                    f"run_id={run_id!r} is in status={status.get('status')!r}"
                )
            reason = str(payload.get("reason") or "external") or "external"
            now_iso = _utc_now()
            try:
                (run_dir / "pause_request.json").write_text(
                    json.dumps({
                        "reason": reason,
                        "requested_at": now_iso,
                    }),
                    encoding="utf-8",
                )
            except OSError as exc:
                raise RuntimeError(
                    f"failed to write pause sentinel: {exc}"
                ) from exc
            status["status"] = "paused"
            status["pause_reason"] = reason
            status["paused_at"] = now_iso
            status["updated_at"] = now_iso
            self._write_detached_status(run_id, status)
            self._append_detached_event(run_id, f"paused:{reason}")
            return status

        if action == "status":
            status = self._read_json_file(run_dir / "status.json")
            thread = self.detached_threads.get(run_id)
            if thread and thread.is_alive() and status.get("status") not in {"running", "queued"}:
                status["in_memory_thread_alive"] = True
            # #344: inline the PauseState blob so the polling caller has
            # everything they need to resume without a second round-trip.
            if status.get("status") == "paused":
                status["pause_state"] = self._read_pause_state(run_id)
            return self._enrich_status(run_id, status)

        if action == "reasoning_trace":
            # Parity with Modal ``modal_cua_server.py:2686-2710``.
            # Tail the per-run reasoning.jsonl with an optional ``since``
            # ISO-8601 cursor on ``ts``. Returns the same envelope as
            # Modal so callers can switch endpoints transparently.
            status = self._read_json_file(run_dir / "status.json")
            self._enrich_status(run_id, status)
            since_ts = str(payload.get("since") or "") or None
            jsonl_path = run_dir / "reasoning.jsonl"
            events: list[dict[str, Any]] = []
            if jsonl_path.exists():
                try:
                    with jsonl_path.open("r", encoding="utf-8") as handle:
                        for line in handle:
                            stripped = line.strip()
                            if not stripped:
                                continue
                            try:
                                event = json.loads(stripped)
                            except (json.JSONDecodeError, ValueError):
                                continue
                            if not isinstance(event, dict):
                                continue
                            if since_ts and str(event.get("ts", "")) <= since_ts:
                                continue
                            events.append(event)
                except OSError as exc:
                    logger.warning(
                        "reasoning_trace read failed for %s: %s", run_id, exc,
                    )
            return {**status, "events": events, "count": len(events)}

        if action == "result":
            # Modal parity (#862 follow-up): wrap the executor result
            # under a ``result`` key on the same envelope the status
            # response uses, so callers can switch endpoints without
            # re-keying. Previously Baseten returned the raw
            # ``result.json`` body at top-level, conflating the status
            # envelope with the executor payload.
            status = self._read_json_file(run_dir / "status.json")
            status = self._enrich_status(run_id, status)
            result_path = run_dir / "result.json"
            if result_path.exists():
                result_blob = self._read_json_file(result_path)
                return {**status, "result": result_blob}
            return {
                **status,
                "result": None,
                "result_ready": False,
            }

        if action == "resume":
            # #344: rehydrate a paused detached run. Read status cache-first so
            # a resume landing on a different replica sees status=='paused'
            # even before the /data Volume has propagated (#909-followup).
            status = self._read_detached_status(run_id)
            if not status:
                # Empty in BOTH the cross-replica store and on disk → the
                # run_id is genuinely unknown (→ 404), not merely a Volume that
                # hasn't propagated to this replica.
                raise FileNotFoundError(f"unknown run_id: {run_id}")
            if status.get("status") != "paused":
                raise ValueError(
                    f"action='resume' requires a paused run; "
                    f"run_id={run_id!r} is in status={status.get('status')!r}"
                )
            user_input = payload.get("user_input")
            if user_input is None:
                raise ValueError("action='resume' requires user_input")
            pause_state = self._read_pause_state(run_id)
            if not pause_state:
                raise FileNotFoundError(
                    f"pause_state.json missing for run {run_id!r}; cannot resume"
                )
            original_payload = self._read_resume_payload(run_id)
            if not original_payload:
                raise FileNotFoundError(
                    f"payload.json missing for run {run_id!r}; cannot resume"
                )
            existing_thread = self.detached_threads.get(run_id)
            if existing_thread and existing_thread.is_alive():
                raise RuntimeError(
                    f"resume in progress for {run_id!r}; poll status before retrying"
                )
            resume_payload = dict(original_payload)
            resume_payload["_detached_run_id"] = run_id
            resume_payload["_resume_pause_state"] = pause_state
            resume_payload["_resume_user_input"] = user_input
            # Restore the EXACT paused plan from the checkpoint — verbatim, no
            # re-decompose and no hash-cache lookup. Passing it as ``task_suite``
            # makes _task_suite_from_payload return it as-is, so the plan can't
            # drift across replicas and the signature always matches. (Fixes the
            # end-user "plan signature mismatch on resume → wedged paused" bug.)
            stored_suite = pause_state.get("resolved_task_suite")
            if stored_suite:
                resume_payload["task_suite"] = dict(stored_suite)
            else:
                # Legacy pauses (pre-checkpoint-suite) fall back to the
                # re-derive + signature guard so a genuinely changed plan still
                # surfaces as a 400 rather than running the wrong plan.
                stored_sig = str(pause_state.get("plan_signature", ""))
                current_sig = ""
                try:
                    rebuilt_suite = self._task_suite_from_payload(dict(original_payload))
                    current_sig = str(rebuilt_suite.get("_plan_signature", ""))
                except Exception:  # noqa: BLE001 — surface the mismatch, not the rebuild error
                    current_sig = ""
                if stored_sig and current_sig and stored_sig != current_sig:
                    raise ValueError(
                        "plan signature mismatch on resume: stored "
                        f"{stored_sig[:12]!r} ≠ current {current_sig[:12]!r}. "
                        "The plan referenced by this run_id has changed since it paused."
                    )
            now = _utc_now()
            self._write_detached_status(
                run_id,
                {
                    "status": "running",
                    "resumed_at": now,
                    # Wipe stale pause-surface fields so the next status
                    # poll doesn't keep showing the old prompt.
                    "prompt": "",
                    "reason": "",
                },
            )
            self._append_detached_event(run_id, "resuming")
            thread = threading.Thread(
                target=self._run_detached_worker,
                args=(run_id, resume_payload),
                daemon=True,
            )
            self.detached_threads[run_id] = thread
            thread.start()
            return {
                "run_id": run_id,
                "status": "running",
                "resumed_at": now,
            }

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
        decompose = bool(payload.get("decompose", True))
        # plan_text: free-text → PlanDecomposer.decompose_text → micro suite.
        # Documented in onboarding docs as the one-shot ad-hoc shape; build the
        # suite here so callers get the same dispatch as the file-based path.
        # decompose=False short-circuits the rewrite and runs the raw text as
        # a single-intent task_suite — used by long-plan-following benchmarks.
        plan_text = payload.get("plan_text")
        if plan_text:
            if not decompose:
                return self._raw_text_suite(str(plan_text), payload)
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
        # .txt micro paths are free-text; honor the same decompose flag.
        if not decompose:
            resolved = self._resolve_path(str(micro_path))
            if resolved.exists() and resolved.suffix != ".json":
                return self._raw_text_suite(resolved.read_text(), payload)
        return self._micro_suite_from_path(str(micro_path), payload)

    # #audit item 1: long raw plans are brittle — the executor maps
    # them to a single Holo3 task and the brain has no scaffolding to
    # decompose internally. The threshold below is the budget the
    # /v1/predict surface enforces unless the caller explicitly opts
    # in via ``payload.allow_raw_long_plan = True``. Word count
    # rather than char count because line breaks / whitespace in the
    # source text dominate at small scale.
    _RAW_LONG_PLAN_WORD_LIMIT: int = 80

    def _raw_text_suite(self, plan_text: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Wrap raw free-text in a single-task task_suite without decomposition.

        Long plans should always decompose — the executor's done-gate
        and recovery loop assume single-goal tasks. Raw long plans
        either oscillate or terminate prematurely. Callers that
        genuinely want to feed a long text directly (benchmark
        scaffolding, ablation harness) opt in via
        ``payload.allow_raw_long_plan = True``.
        """
        words = len((plan_text or "").split())
        allow_long = bool(payload.get("allow_raw_long_plan", False))
        if words > self._RAW_LONG_PLAN_WORD_LIMIT and not allow_long:
            raise ValueError(
                f"raw plan_text has {words} words; server cap for "
                f"decompose=False is {self._RAW_LONG_PLAN_WORD_LIMIT}. "
                f"Either drop decompose=False (the decomposer handles "
                f"long plans), or pass allow_raw_long_plan=true if you "
                f"intend the brain to run the full text as a single task."
            )
        return {
            "session_name": str(payload.get("session_name") or "plan_text_raw"),
            "base_url": str(payload.get("base_url") or ""),
            "tasks": [
                {
                    "task_id": "plan_text",
                    "intent": plan_text,
                    "start_url": str(payload.get("start_url") or ""),
                }
            ],
        }

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
        # #audit item 5: validate the parsed plan against MAX_STEPS /
        # MAX_LOOP_ITERATIONS / required-field rules BEFORE handing it
        # to build_micro_suite. Catches obvious problems (a 1000-step
        # plan from a runaway decomposer, missing intent/type on a
        # step) before the run consumes any Modal time.
        from mantis_agent.api_schemas import validate_micro_steps
        steps_dicts = validate_micro_steps(steps_dicts)
        data_root = _data_root()
        return build_micro_suite(
            steps_dicts,
            micro_plan.domain or "plan_text",
            max_cost=float(payload.get("max_cost", 10.0)),
            max_time_minutes=int(payload.get("max_time_minutes", 180)),
            resume_state=bool(payload.get("resume_state", False)),
            state_key=str(payload.get("state_key") or ""),
            profile_id=str(payload.get("profile_id") or ""),
            workflow_id=str(payload.get("workflow_id") or ""),
            checkpoint_dir=str(data_root / "checkpoints"),
            proxy_city=str(payload.get("proxy_city") or os.environ.get("MANTIS_PROXY_CITY", "")),
            proxy_state=str(payload.get("proxy_state") or os.environ.get("MANTIS_PROXY_STATE", "")),
            proxy_disabled=bool(payload.get("proxy_disabled", False)),
            brain_budgets=payload.get("brain_budgets"),
            pause_on_captcha=payload.get("pause_on_captcha"),
            settle_ceiling_seconds=payload.get("settle_ceiling_seconds"),
            max_recoveries_per_run=payload.get("max_recoveries_per_run"),
            max_recoveries_per_step=payload.get("max_recoveries_per_step"),
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
        # #audit item 5: same validation gate as the from-text path —
        # catches malformed JSON plans (missing required fields, step
        # count over the cap) before the run starts.
        from mantis_agent.api_schemas import validate_micro_steps
        steps_dicts = validate_micro_steps(steps_dicts)
        data_root = _data_root()
        suite = build_micro_suite(
            steps_dicts,
            micro_plan.domain,
            max_cost=float(payload.get("max_cost", 10.0)),
            max_time_minutes=int(payload.get("max_time_minutes", 180)),
            resume_state=bool(payload.get("resume_state", False)),
            state_key=str(payload.get("state_key") or ""),
            profile_id=str(payload.get("profile_id") or ""),
            workflow_id=str(payload.get("workflow_id") or ""),
            checkpoint_dir=str(data_root / "checkpoints"),
            proxy_city=str(payload.get("proxy_city") or os.environ.get("MANTIS_PROXY_CITY", "")),
            proxy_state=str(payload.get("proxy_state") or os.environ.get("MANTIS_PROXY_STATE", "")),
            proxy_disabled=bool(payload.get("proxy_disabled", False)),
            objective=objective_dict,
            brain_budgets=payload.get("brain_budgets"),
            pause_on_captcha=payload.get("pause_on_captcha"),
            settle_ceiling_seconds=payload.get("settle_ceiling_seconds"),
            max_recoveries_per_run=payload.get("max_recoveries_per_run"),
            max_recoveries_per_step=payload.get("max_recoveries_per_step"),
        )
        return suite

    def _make_env(
        self,
        task_suite: dict[str, Any],
        run_id: str,
        settle_time: float,
        *,
        reuse_session: bool = False,
    ) -> tuple[XdotoolGymEnv, Any]:
        from mantis_agent.task_loop import setup_env

        data_root = _data_root()
        session_name = task_suite.get("session_name", "baseten_cua")
        env, proxy_proc, _proxy_diag = setup_env(
            base_url=task_suite.get("base_url", ""),
            run_id=run_id,
            session_name=session_name,
            settle_time=settle_time,
            proxy_city=str(task_suite.get("_proxy_city") or ""),
            proxy_state=str(task_suite.get("_proxy_state") or ""),
            proxy_disabled=bool(task_suite.get("_proxy_disabled", False)),
            browser=os.environ.get("MANTIS_BROWSER", "google-chrome"),
            profile_dir=_chrome_profile_dir_for_suite(task_suite),
            save_screenshots_dir=str(data_root / "screenshots"),
            reuse_session=reuse_session,
        )
        return env, proxy_proc

    def _maybe_start_live_viewer(
        self, payload: dict[str, Any], run_id: str, env: Any = None,
    ) -> Any:
        """Start the MJPEG live viewer if ``payload.live_viewer`` is set (#416).

        Mirrors the structure of :meth:`_maybe_record`: a no-op when the
        feature isn't requested, an idempotent ``ensure_display_ready``
        call before the capture thread starts so it doesn't race the
        env's lazy Xvfb spawn, and a defensive try/except so a viewer
        failure never breaks the run.

        Returns the viewer context object (from
        :func:`task_loop.setup_viewer`) so the caller can call
        ``viewer_ctx.__exit__(None, None, None)`` in its ``finally``.
        ``None`` when the feature is off / setup failed — the caller
        should treat that as the no-viewer case.

        Surfaces the tunnel URL by merging ``{"viewer_url": url}`` into
        ``status.json`` via :meth:`_write_detached_status`. Callers
        polling ``action=status`` see the URL on the next round-trip
        after this method has been called.
        """
        if not payload.get("live_viewer"):
            return None
        if env is not None and hasattr(env, "ensure_display_ready"):
            try:
                env.ensure_display_ready()
            except Exception as exc:  # noqa: BLE001 — viewer is best-effort
                logger.warning(
                    "ensure_display_ready failed before live-viewer start: %s",
                    exc,
                )
        try:
            from ..task_loop import setup_viewer
            viewer_ctx, _event_bus, viewer_url = setup_viewer(True)
        except Exception as exc:  # noqa: BLE001 — never fatal
            logger.warning("live-viewer setup failed: %s", exc)
            return None
        if viewer_ctx is None:
            return None
        if viewer_url:
            self._write_detached_status(run_id, {"viewer_url": viewer_url})
            logger.info("live viewer ready for run %s at %s", run_id, viewer_url)
        return viewer_ctx

    def _stop_live_viewer(self, viewer_ctx: Any) -> None:
        """Idempotent cleanup for :meth:`_maybe_start_live_viewer`.

        Wrapped because ``viewer_ctx.__exit__`` can raise (the FastAPI
        server's shutdown can race the capture thread), and we don't
        want a viewer-cleanup error to mask whatever failed during the
        run itself.
        """
        if viewer_ctx is None:
            return
        try:
            viewer_ctx.__exit__(None, None, None)
        except Exception as exc:  # noqa: BLE001 — cleanup, never fatal
            logger.warning("live-viewer cleanup raised: %s", exc)

    def _maybe_record(
        self, payload: dict[str, Any], run_id: str, env: Any = None,
    ) -> tuple[Any, Any]:
        """Spawn a ScreenRecorder if payload.record_video is set.

        Returns ``(recorder, click_log)`` so the caller can wrap the env
        with a ``ClickRecordingEnv`` to capture click coordinates that
        feed the polished video's ripple animations. Either may be None
        when recording is disabled.

        When ``env`` is provided and exposes ``ensure_display_ready``,
        the X display is brought up *before* ffmpeg fires. Without this
        the recorder races ``env.reset()``'s lazy Xvfb spawn and ffmpeg
        fails with ``Cannot open display :99`` — the recurring error
        integrators were reporting on /v1/predict runs with
        ``record_video=true``.
        """
        if not payload.get("record_video"):
            return (None, None)
        from mantis_agent.presentation import ClickEventLog
        from mantis_agent.recorder import ScreenRecorder

        # Bring up Xvfb before ffmpeg attaches — fixes the startup race
        # where ``_make_env`` returns an env whose ``reset()`` hasn't
        # run yet, so the X display ffmpeg targets doesn't exist.
        # ``ensure_display_ready`` is idempotent: a no-op on warm
        # containers where Xvfb is already alive.
        display: str | None = None
        if env is not None and hasattr(env, "ensure_display_ready"):
            try:
                display = env.ensure_display_ready()
            except Exception as exc:  # noqa: BLE001 — best-effort
                logger.warning(
                    "ensure_display_ready failed before recorder spawn: %s",
                    exc,
                )

        tenant_id = safe_state_key(
            os.environ.get("MANTIS_TENANT_ID", DEFAULT_TENANT.tenant_id)
        )
        runs_dir = _data_root() / "tenants" / tenant_id / "runs" / safe_state_key(run_id)
        runs_dir.mkdir(parents=True, exist_ok=True)
        fmt = str(payload.get("video_format", "mp4"))
        output = runs_dir / f"recording.{fmt}"
        rec_kwargs: dict[str, Any] = {
            "output": output,
            "fps": int(payload.get("video_fps", 5)),
            "fmt": fmt,
        }
        if display:
            rec_kwargs["display"] = display
        rec = ScreenRecorder(**rec_kwargs)  # type: ignore[arg-type]
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

        _ripples_tmp: Any = None
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
                # Write the thousands of overlay frame PNGs to container-local
                # temp, NOT the persistent /data Volume — they're consumed by
                # the ffmpeg compositing below and then discarded. Dumping them
                # on the Volume exhausted its 500k-inode budget and 500'd every
                # submit (outage 2026-06-15); cleaned in the finally.
                import tempfile
                _ripples_tmp = tempfile.mkdtemp(prefix="mantis_ripples_")
                ripples_dir = Path(_ripples_tmp)
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
        finally:
            # Always discard the ephemeral overlay frames (temp, off-Volume).
            if _ripples_tmp:
                import shutil
                shutil.rmtree(_ripples_tmp, ignore_errors=True)

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

    def _build_extraction_cache(
        self,
        task_suite: dict[str, Any],
        payload: dict[str, Any],
    ):
        """Construct an ExtractionCache iff the request opted into caching.

        Returns ``None`` when both ``cache_read`` and ``cache_write`` are
        false (legacy behavior: no cache, no disk I/O). The cache file is
        scoped to ``(tenant_id, cache_key)`` where ``cache_key`` falls
        back to the resolved ``state_key``.
        """
        cache_read = bool(payload.get("cache_read", False))
        cache_write = bool(payload.get("cache_write", False))
        if not (cache_read or cache_write):
            return None
        from mantis_agent.extraction.cache import ExtractionCache

        tenant_id = os.environ.get("MANTIS_TENANT_ID", "default")
        cache_key = (
            payload.get("cache_key")
            or task_suite.get("_state_key")
            or payload.get("state_key")
            or task_suite.get("session_name")
            or "default"
        )
        safe_key = _safe_state_key(str(cache_key)) or "default"
        cache_dir = _data_root() / "tenants" / tenant_id / "cache"
        path = cache_dir / f"{safe_key}.json"
        ttl = int(payload.get("cache_ttl_seconds", 86400))
        return ExtractionCache(
            path,
            read_enabled=cache_read,
            write_enabled=cache_write,
            ttl_seconds=ttl,
        )

    def _run_micro(
        self,
        task_suite: dict[str, Any],
        payload: dict[str, Any],
        run_id: str | None = None,
    ) -> dict[str, Any]:
        from mantis_agent.extraction import ClaudeExtractor
        from mantis_agent.grounding import ClaudeGrounding
        from mantis_agent.gym.checkpoint import PauseRequested, PauseState
        from mantis_agent.gym.micro_runner import MicroPlanRunner
        from mantis_agent.plan_decomposer import MicroIntent, MicroPlan

        run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        t0 = time.time()
        session_name = task_suite.get("session_name", "baseten_micro")
        # Propagate top-level proxy_disabled from the request body onto the
        # inline task_suite so _make_env honors it. Inline _micro_plan
        # submissions don't go through build_micro_suite (which would set
        # _proxy_disabled), so without this propagation _make_env reads
        # task_suite.get("_proxy_disabled", False) → False even when the
        # caller asked to disable. Caught when news.ycombinator.com
        # extracts halted with ERR_TUNNEL_CONNECTION_FAILED despite
        # proxy_disabled: true.
        if "proxy_disabled" in payload and "_proxy_disabled" not in task_suite:
            task_suite["_proxy_disabled"] = bool(payload.get("proxy_disabled", False))
        env, proxy_proc = self._make_env(
            task_suite,
            run_id,
            settle_time=4.0 if self.model_kind == "holo3" else 2.0,
        )
        recorder, click_log = self._maybe_record(payload, run_id, env=env)
        if click_log is not None:
            from mantis_agent.presentation import ClickRecordingEnv
            env = ClickRecordingEnv(env, click_log)
        # #416: start live MJPEG viewer alongside the recorder when
        # the caller requested it. URL surfaces into status.json so
        # the polling client can hot-link it.
        viewer_ctx = self._maybe_start_live_viewer(payload, run_id, env=env)

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

            grounding = ClaudeGrounding(cache=self.grounding_cache)
            schema = None
            # #508: payload-level ``extraction_schema`` wins over the
            # plan-derived ObjectiveSpec schema. Callers that want a
            # purpose-built schema (custom fields, different rejection
            # vocabulary) used to be silently ignored — the dict was
            # accepted via PredictRequest.extra='allow' but never read.
            from mantis_agent.extraction import ExtractionSchema
            payload_schema = payload.get("extraction_schema") if isinstance(payload, dict) else None
            if isinstance(payload_schema, dict):
                schema = ExtractionSchema.from_dict(payload_schema)
            else:
                objective_data = task_suite.get("_objective")
                if objective_data:
                    from mantis_agent.graph.objective import ObjectiveSpec
                    objective = ObjectiveSpec.from_dict(objective_data)
                    schema = ExtractionSchema.from_objective(objective)
            # Honor _extractor_model from the suite (set via
            # build_micro_suite ← MANTIS_EXTRACTOR_MODEL env var on the
            # caller). Without this, ClaudeExtractor always uses its
            # hardcoded default (claude-sonnet-4-6) and the cheaper
            # Haiku extraction never lands — the suite field is silently
            # ignored. Empty string keeps the legacy default.
            _extractor_model = str(task_suite.get("_extractor_model") or "").strip()
            if _extractor_model:
                extractor = ClaudeExtractor(model=_extractor_model, schema=schema)
            else:
                extractor = ClaudeExtractor(schema=schema)
            resume_state = bool(task_suite.get("_resume_state", False))
            checkpoint_path = task_suite.get("_checkpoint_path")
            if not checkpoint_path:
                # Inline task_suite._micro_plan submissions don't go through
                # build_micro_suite, so no checkpoint path is set. Derive one
                # under the run dir so the runner can persist incrementally.
                checkpoint_path = str(_data_root() / "checkpoints" / f"{run_id}.json")

            # Per-request extraction cache. Both flags default off so legacy
            # callers see no behavior change; opt-in saves Claude tokens on
            # previously-seen URLs (~$0.04/item per cache hit).
            cache = self._build_extraction_cache(task_suite, payload)

            # #300 follow-up: per-request RoutingPolicy override.
            # ``payload["route_som_clicks"]`` flips SoM-anchored
            # CDP clicks on / off for the run; ``None`` defers to
            # ``MANTIS_ROUTE_SOM_CLICKS``. Same override shape as
            # /v1/cua so the ablation harness threads identically
            # through both endpoints.
            from ..gym.runner import RoutingPolicy
            routing_policy = RoutingPolicy.from_env()
            som_override = payload.get("route_som_clicks")
            if som_override is not None:
                routing_policy = RoutingPolicy(
                    plan_executor_enabled=routing_policy.plan_executor_enabled,
                    som_enabled=routing_policy.som_enabled,
                    som_for_unstructured_clicks=bool(som_override),
                )

            workflow_id = task_suite.get("_workflow_id") or task_suite.get("_state_key", "")
            profile_id = task_suite.get("_profile_id") or task_suite.get("_state_key", "")
            runner = MicroPlanRunner(
                brain=self.brain,
                env=env,
                grounding=grounding,
                extractor=extractor,
                checkpoint_path=checkpoint_path,
                run_key=workflow_id or session_name,
                session_name=session_name,
                plan_signature=task_suite.get("_plan_signature", ""),
                resume_state=resume_state,
                max_cost=task_suite.get("_max_cost", 10.0),
                max_time_minutes=task_suite.get("_max_time_minutes", 180),
                brain_budgets=task_suite.get("_brain_budgets"),
                pause_on_captcha=task_suite.get("_pause_on_captcha"),
                settle_ceiling_seconds=task_suite.get("_settle_ceiling_seconds"),
                max_recoveries_per_run=task_suite.get("_max_recoveries_per_run"),
                max_recoveries_per_step=task_suite.get("_max_recoveries_per_step"),
                extraction_cache=cache,
                routing_policy=routing_policy,
            )
            # Cost-meter finalize gate: ``micro_runner.run`` calls
            # ``finalize_to_disk(run_id=self._api_run_id, tenant_id=...)``
            # only when ``_api_run_id`` is truthy. Without these attrs
            # the per-source cost JSON never lands on disk and the new
            # artifact-side instrumentation has nothing to expose.
            # ``tenant_id`` here is just a path-sanitization scope for
            # the meter (not auth) — workflow_id is a stable enough
            # bucket on Baseten where the API token is per-tenant.
            runner._api_run_id = run_id
            runner._api_tenant_id = workflow_id or session_name or "default"
            # #344: default ``request_user_input`` host tool. Brains that
            # emit ``Action(TOOL_CALL, name="request_user_input")`` get a
            # paused-run snapshot on the first call, and the staged
            # ``user_input`` on the second (after action=resume rehydrates).
            def _request_user_input(args: dict[str, Any]) -> Any:
                staged = runner.consume_pause_input(default=None)
                if staged is None:
                    raise PauseRequested(
                        reason="user_input",
                        prompt=str(args.get("prompt", "")),
                    )
                return staged
            runner.register_tool(
                "request_user_input",
                {
                    "type": "object",
                    "properties": {"prompt": {"type": "string"}},
                    "additionalProperties": False,
                },
                _request_user_input,
            )

            # #344: resume continuation. When the caller submits
            # action=resume, the worker re-enters this method with the
            # stored PauseState and the caller's user_input layered onto
            # the payload. ``runner.resume(...)`` replays the recorded
            # steps and continues from the paused step.
            resume_blob = payload.get("_resume_pause_state")
            # Bind a fresh per-source ClaudeCostMeter for this run.
            # ``record_from_response`` (called from brain_claude /
            # extractor / grounding / agentic_recovery / _anthropic.
            # client) pulls the active meter via ``current_meter()``,
            # which reads the ContextVar this binds. Without this
            # binding every record_from_response is a silent no-op and
            # the per-source cost JSON never accumulates anything to
            # write. The ContextVar shape means concurrent runs in the
            # same process get isolated meters automatically; on
            # Baseten predict_concurrency=1 anyway.
            from mantis_agent.observability.claude_cost_meter import (
                ClaudeCostMeter as _ClaudeCostMeter,
                set_current_meter as _set_current_meter,
            )
            run_cost_meter = _ClaudeCostMeter()
            _set_current_meter(run_cost_meter)
            try:
                if resume_blob is not None:
                    pause_state_obj = (
                        PauseState.from_dict(resume_blob)
                        if isinstance(resume_blob, dict) else resume_blob
                    )
                    runner_result = runner.resume(
                        pause_state_obj,
                        user_input=payload.get("_resume_user_input"),
                        plan=micro_plan,
                    )
                else:
                    # #885-followup: stage an up-front ``user_input`` on
                    # the initial submit (no pause/resume round-trip) for
                    # ``{{user_input}}`` substitution — the "natural" login
                    # path. ``None`` (the common case) is a no-op.
                    runner_result = runner.run_with_status(
                        micro_plan, resume=resume_state,
                        user_input=payload.get("user_input"),
                    )
            finally:
                # ``micro_runner.run`` itself calls finalize_to_disk via
                # current_meter() once it returns, so we keep the bind
                # active across the call AND clear it afterwards to
                # avoid leaking across runs in the same container.
                _set_current_meter(None)
            step_results = runner_result.steps
            if cache is not None:
                try:
                    cache.save()
                except Exception as exc:  # noqa: BLE001 — cache persist is best-effort
                    logger.warning("extraction cache save failed: %s", exc)
            result = build_micro_result(
                runner,
                step_results,
                run_id=run_id,
                provider="baseten",
                session_name=session_name,
                model_name=self.model_kind,
                elapsed_seconds=time.time() - t0,
                state_key=task_suite.get("_state_key", ""),
                profile_id=profile_id,
                workflow_id=workflow_id,
                checkpoint_path=checkpoint_path,
                plan_signature=task_suite.get("_plan_signature", ""),
                resume_state=resume_state,
            )
            self._attach_recording_metadata(result, recorder, click_log=click_log)
            # #344: surface the paused snapshot so the detached worker can
            # write status=paused (rather than succeeded) and persist
            # pause_state.json for the next action=resume.
            if runner_result.paused and runner_result.pause_state is not None:
                result["_paused"] = True
                result["pause_state"] = runner_result.pause_state.to_dict()
                result["prompt"] = runner_result.pause_state.prompt
                result["reason"] = runner_result.pause_state.pending_reason
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
            # #416: stop the live viewer before env close so the FastAPI
            # capture thread can shut down cleanly while Xvfb is still up.
            self._stop_live_viewer(viewer_ctx)
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
        # Mirror _run_micro: surface payload-level proxy/start_url controls into
        # the task_suite dict so _make_env honors them. Otherwise these flags
        # are silently dropped on the task_suite path.
        if "proxy_disabled" in payload and "_proxy_disabled" not in task_suite:
            task_suite["_proxy_disabled"] = bool(payload.get("proxy_disabled", False))
        env, proxy_proc = self._make_env(
            task_suite,
            run_id,
            settle_time=4.0 if self.model_kind == "holo3" else 2.0,
        )
        recorder, click_log = self._maybe_record(payload, run_id, env=env)
        if click_log is not None:
            from mantis_agent.presentation import ClickRecordingEnv
            env = ClickRecordingEnv(env, click_log)
        # #416: live MJPEG viewer alongside the recorder; same lifetime
        # contract as in :meth:`_run_micro`.
        viewer_ctx = self._maybe_start_live_viewer(payload, run_id, env=env)
        grounding = ClaudeGrounding(cache=self.grounding_cache)

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
            result["run_id"] = run_id
            result["session_name"] = session_name
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
            # #416: stop live viewer (no-op if it wasn't started).
            self._stop_live_viewer(viewer_ctx)

    def run_pure_cua(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Entrypoint for ``POST /v1/cua``.

        Pure CUA loop: the configured brain (Holo3 / Gemma4) drives
        :class:`GymRunner` against ``XdotoolGymEnv`` directly. No
        ``PlanDecomposer``, no ``ClaudeGrounding``, no ``ClaudeExtractor``
        — every action the brain emits (click / type_text / scroll /
        drag / key_press / wait / done) is executed verbatim by xdotool.

        Detached mode reuses the standard worker pool; the dispatch in
        :meth:`_run_detached_worker` branches on ``_mode == 'pure_cua'``
        and re-enters :meth:`_run_pure_cua` from inside the worker.
        """
        self.load()
        payload = {**payload, "_mode": "pure_cua"}
        if payload.get("detached"):
            return self._start_detached(payload)
        with self.lock:
            return self._run_pure_cua(payload)

    def _run_pure_cua(
        self,
        payload: dict[str, Any],
        run_id: str | None = None,
    ) -> dict[str, Any]:
        from mantis_agent.gym.runner import GymRunner

        run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        t0 = time.time()
        instruction = str(payload.get("instruction") or "").strip()
        if not instruction:
            raise ValueError("instruction is required")
        start_url = str(payload.get("start_url") or "")
        max_steps = int(payload.get("max_steps", 30))
        frames = int(payload.get("frames_per_inference", 1))
        settle_time = float(payload.get("settle_time", 4.0 if self.model_kind == "holo3" else 2.0))

        # Per-request overrides for env-var-driven runner toggles. Lets
        # the ablation harness do single-deploy A/B without redeploys.
        # ``/v1/cua`` runs serially per container, so a try/finally
        # os.environ patch is safe. ``None`` (default) keeps whatever
        # the container's env var said. Pairs with the existing
        # ``reuse_session`` / ``speculation`` per-request overrides.
        _toggle_env_map: dict[str, str] = {
            "perceptual_verify": "MANTIS_PERCEPTUAL_VERIFY",
            "loop_recovery": "MANTIS_LOOP_RECOVERY",
            "done_gate": "MANTIS_DONE_GATE",
            "predicate_verify": "MANTIS_PREDICATE_VERIFY",
            "adaptive_settle": "MANTIS_ADAPTIVE_SETTLE",
            "form_controller": "MANTIS_FORM_CONTROLLER",
            # #298: adaptive loop-detector windows (default on).
            "loop_adaptive": "MANTIS_LOOP_ADAPTIVE",
            # #296: screen-DPI / element-class drift tolerance (default on).
            "adaptive_click_tol": "MANTIS_ADAPTIVE_CLICK_TOL",
        }
        _toggle_overrides: dict[str, str] = {}
        _toggle_restore: dict[str, str | None] = {}
        for _payload_key, _env_var in _toggle_env_map.items():
            _val = payload.get(_payload_key)
            if _val is None:
                continue
            _toggle_restore[_env_var] = os.environ.get(_env_var)
            _toggle_overrides[_env_var] = "enabled" if bool(_val) else "disabled"
            os.environ[_env_var] = _toggle_overrides[_env_var]

        # _make_env reads proxy + base_url off a task_suite-shaped dict;
        # build a minimal one here rather than threading a separate path.
        session_name = "pure_cua"
        task_suite: dict[str, Any] = {
            "session_name": session_name,
            "base_url": start_url,
            "_proxy_city": payload.get("proxy_city") or "",
            "_proxy_state": payload.get("proxy_state") or "",
            "_proxy_disabled": bool(payload.get("proxy_disabled", False)),
        }
        # #311 container-scoped Chrome session cache. When enabled
        # (default), successive requests with the same profile_dir +
        # proxy_server reuse the live Xvfb + Chrome instead of paying
        # the ~10 s cold-launch tax. Per-request opt-out via
        # ``payload["reuse_session"]=False``; container-wide opt-out via
        # ``MANTIS_CHROME_REUSE=disabled``.
        reuse_session_payload = payload.get("reuse_session")
        if reuse_session_payload is None:
            reuse_session_for_request = _chrome_reuse_enabled()
        else:
            reuse_session_for_request = bool(reuse_session_payload)

        cache_profile_dir = _chrome_profile_dir_for_suite(task_suite)
        # _make_env reads proxy_server via setup_env → build_proxy_config;
        # the cache key needs to include it so two requests with different
        # proxy configs don't share a browser. Recreate the proxy URL here
        # using the same inputs setup_env will see.
        cache_proxy_key = "" if task_suite.get("_proxy_disabled") else (
            f"{task_suite.get('_proxy_city') or ''}__{task_suite.get('_proxy_state') or ''}"
        )
        cache_key: tuple[str, str] = (cache_profile_dir, cache_proxy_key)

        env: XdotoolGymEnv
        proxy_proc: Any = None
        env_was_cached: bool = False
        if reuse_session_for_request:
            with _chrome_env_cache_lock:
                cached = _chrome_env_cache.get(cache_key)
                if cached is not None:
                    cand_env, cand_proxy = cached
                    cand_proc = getattr(cand_env, "_browser_proc", None)
                    if cand_proc is not None and cand_proc.poll() is None:
                        env = cand_env
                        proxy_proc = cand_proxy
                        env_was_cached = True
                        logger.info(
                            "chrome-env: reusing cached session "
                            "profile=%s proxy=%s",
                            cache_profile_dir, cache_proxy_key or "<none>",
                        )
                    else:
                        _chrome_env_cache.pop(cache_key, None)
                        try:
                            cand_env.shutdown()
                        except Exception:
                            pass
                if not env_was_cached:
                    env, proxy_proc = self._make_env(
                        task_suite, run_id,
                        settle_time=settle_time,
                        reuse_session=True,
                    )
                    _chrome_env_cache[cache_key] = (env, proxy_proc)
                    logger.info(
                        "chrome-env: caching new session "
                        "profile=%s proxy=%s",
                        cache_profile_dir, cache_proxy_key or "<none>",
                    )
        else:
            env, proxy_proc = self._make_env(
                task_suite, run_id,
                settle_time=settle_time,
                reuse_session=False,
            )

        recorder, click_log = self._maybe_record(payload, run_id, env=env)
        if click_log is not None:
            from mantis_agent.presentation import ClickRecordingEnv
            env = ClickRecordingEnv(env, click_log)

        # #118: reset speculative counters per episode so per-run hit
        # rates surface cleanly. No-op when the brain isn't wrapped.
        if hasattr(self.brain, "reset") and callable(self.brain.reset):
            try:
                self.brain.reset()
            except Exception as exc:
                logger.debug("speculative reset failed (ignored): %s", exc)

        # #118 per-request opt-out: ``payload["speculation"] = False``
        # passes the inner (synchronous) brain to this run without
        # touching the cached wrapper or container env. Lets a single
        # deploy serve both arms of an A/B ablation.
        brain_for_run: Any = self.brain
        speculation_payload = payload.get("speculation")
        if speculation_payload is False and hasattr(self.brain, "inner"):
            brain_for_run = self.brain.inner
            logger.info("brain: per-request override → synchronous (inner)")

        try:
            # #300: attach :class:`PageDiscovery` whenever the env
            # exposes the CDP DOM shim (``cdp_evaluate``). The SoM
            # branch in :meth:`GymRunner.run` only fires when both a
            # plan-step / site-config opts in AND the runner has a
            # discovery instance, so attaching it unconditionally is
            # a no-op on legacy callers but unlocks the plan-step
            # SoM path on production xdotool.
            page_discovery: Any = None
            if hasattr(env, "cdp_evaluate"):
                from ..gym.page_discovery import PageDiscovery
                page_discovery = PageDiscovery(env=env)
            # #300 follow-up: per-request RoutingPolicy override. The
            # request can flip ``route_som_clicks`` on / off; ``None``
            # defers to ``MANTIS_ROUTE_SOM_CLICKS``.
            from ..gym.runner import RoutingPolicy
            routing_policy = RoutingPolicy.from_env()
            override = payload.get("route_som_clicks")
            if override is not None:
                routing_policy = RoutingPolicy(
                    plan_executor_enabled=routing_policy.plan_executor_enabled,
                    som_enabled=routing_policy.som_enabled,
                    som_for_unstructured_clicks=bool(override),
                )
            # #931 P2: opt-in screenshot grounding for click precision. Pure
            # CUA is brain-only by default (grounding=None); when the caller
            # sets ``ground_clicks`` and an Anthropic key is present, refine
            # the brain's click coords with the screenshot grounding model —
            # the CUA-clean fix for small/ambiguous targets (vs DOM SoM).
            cua_grounding = None
            if _should_ground_cua_clicks(
                payload, has_anthropic_key=bool(os.environ.get("ANTHROPIC_API_KEY"))
            ):
                from mantis_agent.grounding import ClaudeGrounding
                cua_grounding = ClaudeGrounding(cache=self.grounding_cache)
                logger.info("  [pure_cua] ground_clicks=true — screenshot grounding enabled")
            elif bool(payload.get("ground_clicks")):
                logger.warning(
                    "  [pure_cua] ground_clicks requested but no ANTHROPIC_API_KEY "
                    "— running brain-only"
                )
            runner = GymRunner(
                brain=brain_for_run,
                env=env,
                max_steps=max_steps,
                frames_per_inference=frames,
                grounding=cua_grounding,  # None = pure brain; set when ground_clicks opts in
                page_discovery=page_discovery,
                routing_policy=routing_policy,
            )
            gym_result = runner.run(
                task=instruction,
                task_id="cua",
                start_url=start_url,
            )
            elapsed = time.time() - t0
            trajectory = list(getattr(gym_result, "trajectory", []) or [])
            # #291 ablation signal: how many predicates the brain emitted,
            # how many were evaluable (env exposed the signal), and how many
            # the brain got right. Lets a /v1/cua run double as an ablation
            # data point without dumping the full trajectory.
            predicate_total = 0
            predicate_evaluated = 0
            predicate_correct = 0
            for tstep in trajectory:
                for r in getattr(tstep, "predicate_results", None) or []:
                    predicate_total += 1
                    if r.get("result") is not None:
                        predicate_evaluated += 1
                        if r.get("result") is True:
                            predicate_correct += 1
            result: dict[str, Any] = {
                "run_id": run_id,
                "mode": "pure_cua",
                "provider": "baseten",
                "session_name": session_name,
                "model": self.model_kind,
                "instruction": instruction,
                "start_url": start_url,
                "success": bool(gym_result.success),
                "termination_reason": getattr(gym_result, "termination_reason", ""),
                "steps": int(gym_result.total_steps),
                "duration_s": round(elapsed),
                "elapsed_seconds": elapsed,
                "trajectory_len": len(trajectory),
                "predicate_summary": {
                    "total": predicate_total,
                    "evaluated": predicate_evaluated,
                    "correct": predicate_correct,
                    "accuracy": (
                        predicate_correct / predicate_evaluated
                        if predicate_evaluated else None
                    ),
                },
                # #303 ablation signal: per-reason count of done(success=true)
                # rejections by the deterministic gate. Empty dict when the
                # gate is disabled or never rejected anything.
                # #311 ablation signal: True when this run reused a cached
                # Xvfb + Chrome process from an earlier request.
                "reused_session": env_was_cached,
                # #118 ablation signal: speculative-inference hit rate.
                # ``hits`` = think() calls served from a pre-launched
                # speculation that validated against the post-settle frame.
                # ``misses`` = pre-launched speculations whose post-frame
                # failed validation (page changed during settle).
                # ``synchronous_starts`` = no pending speculation existed
                # (first call after reset; typically 1 per run).
                "speculation_summary": {
                    "hits": int(getattr(brain_for_run, "hits", 0)),
                    "misses": int(getattr(brain_for_run, "misses", 0)),
                    "synchronous_starts": int(
                        getattr(brain_for_run, "synchronous_starts", 0),
                    ),
                    "hit_rate": float(
                        brain_for_run.hit_rate()
                        if hasattr(brain_for_run, "hit_rate") else 0.0
                    ),
                    "enabled": (
                        brain_for_run is self.brain
                        and os.environ.get(
                            "MANTIS_SPECULATIVE_INFERENCE", "disabled",
                        ).lower() == "enabled"
                    ),
                },
                # #293 ablation signal: perceptual-diff verifier
                # aggregate. ``checked`` = high-risk actions the
                # verifier evaluated; ``no_effect`` = those where the
                # action visibly did nothing. Empty when the verifier
                # never fired (toggle off or no high-risk actions).
                "perceptual_summary": dict(
                    getattr(gym_result, "perceptual_summary", None) or {},
                ),
                # #302 ablation signal: per-reason count of loop-recovery
                # substitutions (type_pending_value, tab_to_next_field,
                # press_return_for_submit). Empty when the policy never
                # fired.
                "loop_recoveries_by_reason": dict(
                    getattr(gym_result, "loop_recoveries_by_reason", None) or {},
                ),
                "done_rejections_by_reason": dict(
                    getattr(gym_result, "done_rejections_by_reason", None) or {},
                ),
                # #295 / #300 ablation signal: per-backend trajectory-step
                # counts. ``plan`` = :class:`PlanExecutor` deterministic
                # dispatch, ``som`` = :class:`PageDiscovery` Set-of-Mark
                # dispatch, ``vision`` = brain-driven raw-coordinate
                # dispatch. Empty dict on hosts that don't wire a
                # PlanExecutor or PageDiscovery (the routing falls
                # straight through to ``vision``).
                "executor_backend_counts": dict(
                    getattr(gym_result, "executor_backend_counts", None) or {},
                ),
            }
            self._attach_recording_metadata(result, recorder, click_log=click_log)
            self._save_result(result, prefix="pure_cua")
            return result
        finally:
            if recorder is not None and getattr(recorder, "result", None) is None:
                try:
                    recorder.stop()
                except Exception:
                    logger.exception("recorder stop in finally failed")
            # #311: env.close() is a no-op when reuse_session=True — keeps
            # the Xvfb + Chrome process alive in the container-scoped cache.
            # The non-cached path still terminates as before.
            env.close()
            if proxy_proc and not reuse_session_for_request:
                proxy_proc.terminate()
            # Restore the env-var overrides put in place at the top of
            # this request. Pairs with the ablation harness so paired
            # ON/OFF requests don't leak state across the boundary.
            for _env_var, _orig in _toggle_restore.items():
                if _orig is None:
                    os.environ.pop(_env_var, None)
                else:
                    os.environ[_env_var] = _orig

    def _save_result(self, result: dict[str, Any], prefix: str) -> None:
        save_result_json(result, _data_root() / "results", prefix)


runtime = BasetenCUARuntime()


