"""Serve a trained LoRA adapter on top of a base CUA model (#911).

The slow-loop trainer (``mantis-trainer``) emits **LoRA adapter checkpoints**
(e.g. a bf16 SFT of ``Hcompany/Holo3-35B-A3B``, registered as a challenger). The
promotion gate (:mod:`mantis_agent.learning.promotion_gate`) evaluates that
challenger by running the frozen holdout against it via ``POST /v1/predict`` — so
the CUA server needs a path to serve ``base + adapter`` behind the usual
endpoint. This module is the pure-logic core of that path; the Modal GPU glue in
``deploy/modal/modal_cua_server.py`` executes the plan it produces.

Two serving backends, **auto-selected by the base model's runtime**:

* **llama.cpp** (GGUF bases — ``holo3``, ``gemma4-cua``). ``llama-server`` applies
  a GGUF-format adapter via ``--lora``; we convert the (small) PEFT adapter to
  GGUF once and cache it. No full-model merge — the base GGUF is reused as-is.
  This is the path the first real challenger (Holo3) takes, because vLLM lacks
  the ``qwen3_5_moe`` arch (see ``modal_cua_server`` Holo3 notes).
* **vLLM** (native bases — ``fara``, ``opencua-*``, ``evocua-*``). vLLM serves the
  adapter directly via ``--enable-lora --lora-modules <name>=<dir>``; the adapter
  is then addressable as a distinct *served model name* (so the brain must
  request that name, not the base name, to actually exercise the adapter).

The adapter is referenced from the request **suite** under ``_lora_adapter`` (the
same ``_``-prefixed convention as ``_sampling_temperature`` etc.), so no executor
spawn-signature change is needed. A reference is either ``"<volume>:<path>"``
(e.g. ``"mantis-trainer-vol:/checkpoints/sft-c3e0d799f432"``) or a bare local
path already visible on a mounted volume.

Everything here is side-effect-free: :func:`plan_serving` returns a
:class:`ServingPlan` describing *what* to run (extra server args, the served
model name, an optional adapter-conversion command). The caller runs the
subprocess (idempotently, cached). Unit-tested in
``tests/test_lora_serving.py``.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

# Base models whose runtime is llama.cpp (GGUF) vs vLLM. Keep in sync with
# CUA_MODELS in deploy/modal/modal_cua_server.py.
LLAMACPP_BASES = frozenset({"holo3", "gemma4-cua"})
VLLM_BASES = frozenset(
    {"evocua-8b", "evocua-32b", "opencua-32b", "opencua-72b", "fara"}
)
# No local weights to adapt — LoRA is meaningless for the hosted-API brain.
API_BASES = frozenset({"claude"})

DEFAULT_LORA_NAME = "challenger"
# vLLM rejects adapters whose rank exceeds --max-lora-rank; trainer SFT/GRPO
# runs use rank<=64, so default the ceiling there (overridable per call).
DEFAULT_MAX_LORA_RANK = 64


class LoraServingError(ValueError):
    """Raised when a LoRA serving request can't be honored as specified."""


@dataclass(frozen=True)
class AdapterRef:
    """A parsed ``_lora_adapter`` reference.

    ``volume`` is the Modal volume name when the ref is ``"<volume>:<path>"``,
    else ``None`` for a bare path. ``path`` is always the path *within* that
    volume (or the bare local path). ``raw`` is the original string.
    """

    path: str
    raw: str
    volume: str | None = None

    @property
    def is_volume_ref(self) -> bool:
        return self.volume is not None


@dataclass(frozen=True)
class ServingPlan:
    """Declarative description of how to serve base (+ optional adapter).

    The Modal glue reads these fields to build the llama-server / vLLM command
    and to tell the brain which served-model-name to request.
    """

    backend: str  # "llamacpp" | "vllm"
    lora_active: bool
    served_model_name: str
    # Extra args to append to the base server command (e.g. ``--lora <gguf>`` or
    # ``--enable-lora --lora-modules challenger=<dir>``). Empty when no adapter.
    extra_server_args: list[str] = field(default_factory=list)
    # llama.cpp only: the local PEFT adapter dir + the GGUF cache path it must be
    # converted to before serving. ``convert_cmd`` is the command to run when the
    # GGUF isn't cached yet (None when no conversion is needed).
    adapter_local_dir: str | None = None
    adapter_gguf_path: str | None = None
    convert_cmd: list[str] | None = None
    # Short, stable id for tagging the run / Augur (so the gate can attribute a
    # result to the right challenger). Empty string when serving the base.
    challenger_tag: str = ""


# ── reference parsing / resolution ────────────────────────────────────

_VOLUME_RE = re.compile(r"^(?P<vol>[A-Za-z0-9][A-Za-z0-9_.-]*):(?P<path>/.+)$")


def parse_adapter_ref(ref: str) -> AdapterRef:
    """Parse a ``_lora_adapter`` string into an :class:`AdapterRef`.

    Accepts ``"<volume>:<absolute_path>"`` or a bare absolute path. Rejects
    empty / relative / whitespace-only refs (fail fast — a typo'd checkpoint
    must not silently serve the base).
    """
    if not ref or not ref.strip():
        raise LoraServingError("empty LoRA adapter reference")
    ref = ref.strip()
    m = _VOLUME_RE.match(ref)
    if m:
        return AdapterRef(path=m.group("path"), raw=ref, volume=m.group("vol"))
    if not ref.startswith("/"):
        raise LoraServingError(
            f"adapter reference must be '<volume>:/abs/path' or '/abs/path', got {ref!r}"
        )
    return AdapterRef(path=ref, raw=ref, volume=None)


def local_adapter_dir(ref: AdapterRef, mounts: dict[str, str]) -> str:
    """Resolve a parsed ref to a local filesystem dir using ``mounts``.

    ``mounts`` maps a Modal volume name → its mountpoint on the executor
    (e.g. ``{"mantis-trainer-vol": "/trainer"}``). A bare-path ref is returned
    verbatim. A volume ref whose volume isn't mounted is an error (so we never
    serve the base when the operator meant to serve a challenger).
    """
    if not ref.is_volume_ref:
        return ref.path
    mount = mounts.get(ref.volume or "")
    if not mount:
        raise LoraServingError(
            f"volume {ref.volume!r} for adapter {ref.raw!r} is not mounted "
            f"(known mounts: {sorted(mounts)})"
        )
    # ref.path is absolute within the volume; splice it under the mountpoint.
    return mount.rstrip("/") + ref.path


def challenger_tag(ref: AdapterRef) -> str:
    """A short, stable id for the adapter — the checkpoint dir basename plus a
    short hash of the full ref (disambiguates same-named checkpoints across
    volumes)."""
    base = ref.path.rstrip("/").rsplit("/", 1)[-1] or "adapter"
    digest = hashlib.sha1(ref.raw.encode()).hexdigest()[:8]
    return f"{base}-{digest}"


def gguf_adapter_cache_path(adapter_local_dir: str, cache_root: str) -> str:
    """Deterministic cache path for the GGUF-converted adapter.

    Keyed by a hash of the source dir so re-serving the same checkpoint reuses
    the converted GGUF instead of re-running the (minutes-long) conversion."""
    digest = hashlib.sha1(adapter_local_dir.encode()).hexdigest()[:12]
    return f"{cache_root.rstrip('/')}/lora_{digest}.gguf"


# ── backend selection ─────────────────────────────────────────────────


def serving_backend(cua_model: str) -> str:
    """Return ``"llamacpp"`` or ``"vllm"`` for a base model, or raise for a base
    that can't host a local adapter (e.g. the hosted-API ``claude`` brain)."""
    if cua_model in LLAMACPP_BASES:
        return "llamacpp"
    if cua_model in VLLM_BASES:
        return "vllm"
    if cua_model in API_BASES:
        raise LoraServingError(
            f"base model {cua_model!r} is a hosted API — it can't serve a LoRA adapter"
        )
    raise LoraServingError(f"unknown base model {cua_model!r}; can't pick a serving backend")


# ── command builders ──────────────────────────────────────────────────


def build_llamacpp_lora_args(adapter_gguf_path: str, scale: float = 1.0) -> list[str]:
    """llama-server args to apply a GGUF adapter. ``--lora`` for unit scale,
    ``--lora-scaled`` otherwise. The served model name is unchanged (the adapter
    is folded into the base at load)."""
    if scale == 1.0:
        return ["--lora", adapter_gguf_path]
    return ["--lora-scaled", adapter_gguf_path, str(scale)]


def build_vllm_lora_args(
    adapter_dir: str, name: str = DEFAULT_LORA_NAME, max_lora_rank: int = DEFAULT_MAX_LORA_RANK
) -> list[str]:
    """vLLM args to serve an adapter as a distinct served-model-name."""
    return [
        "--enable-lora",
        "--lora-modules",
        f"{name}={adapter_dir}",
        "--max-lora-rank",
        str(max_lora_rank),
    ]


def build_convert_lora_to_gguf_cmd(
    *,
    python_exe: str,
    convert_script: str,
    adapter_dir: str,
    out_gguf: str,
    base_model: str | None = None,
) -> list[str]:
    """Build the llama.cpp ``convert_lora_to_gguf.py`` command.

    ``base_model`` (HF id or local dir) is needed when the adapter's
    ``adapter_config.json`` references a base the script must read tensor metadata
    from; passed via ``--base`` when given.
    """
    cmd = [python_exe, convert_script, adapter_dir, "--outfile", out_gguf, "--outtype", "f16"]
    if base_model:
        cmd += ["--base", base_model]
    return cmd


# ── top-level planner ─────────────────────────────────────────────────


def plan_serving(
    *,
    cua_model: str,
    suite: dict,
    mounts: dict[str, str],
    gguf_cache_root: str,
    base_served_name: str = "model",
    llamacpp_python: str = "python3",
    llamacpp_convert_script: str = "/opt/llama.cpp/convert_lora_to_gguf.py",
    llamacpp_base_model: str | None = None,
) -> ServingPlan:
    """Produce a :class:`ServingPlan` for a request.

    Reads ``suite['_lora_adapter']`` (and optional ``_lora_name``,
    ``_lora_scale``, ``_lora_max_rank``). When absent, returns a base-only plan
    (no adapter args, base served-name) — the common production path is
    untouched. When present, returns the backend-appropriate plan.
    """
    backend = serving_backend(cua_model)
    raw_ref = str(suite.get("_lora_adapter") or "").strip()
    if not raw_ref:
        return ServingPlan(
            backend=backend, lora_active=False, served_model_name=base_served_name
        )

    ref = parse_adapter_ref(raw_ref)
    adapter_dir = local_adapter_dir(ref, mounts)
    tag = challenger_tag(ref)

    if backend == "llamacpp":
        scale = float(suite.get("_lora_scale", 1.0) or 1.0)
        # Preferred path: the trainer already emitted a GGUF-format adapter
        # (``…:/checkpoints/x/adapter.gguf``). Serve it directly — no conversion,
        # so the serving image needs no torch/transformers/gguf deps. Only a raw
        # PEFT dir requires the (heavy) convert step.
        if adapter_dir.endswith(".gguf"):
            return ServingPlan(
                backend=backend,
                lora_active=True,
                served_model_name=base_served_name,  # llama.cpp folds the adapter in
                extra_server_args=build_llamacpp_lora_args(adapter_dir, scale=scale),
                adapter_local_dir=adapter_dir,
                adapter_gguf_path=adapter_dir,
                convert_cmd=None,
                challenger_tag=tag,
            )
        gguf = gguf_adapter_cache_path(adapter_dir, gguf_cache_root)
        return ServingPlan(
            backend=backend,
            lora_active=True,
            served_model_name=base_served_name,  # llama.cpp folds the adapter in
            extra_server_args=build_llamacpp_lora_args(gguf, scale=scale),
            adapter_local_dir=adapter_dir,
            adapter_gguf_path=gguf,
            convert_cmd=build_convert_lora_to_gguf_cmd(
                python_exe=llamacpp_python,
                convert_script=llamacpp_convert_script,
                adapter_dir=adapter_dir,
                out_gguf=gguf,
                base_model=llamacpp_base_model,
            ),
            challenger_tag=tag,
        )

    # vLLM
    name = str(suite.get("_lora_name") or DEFAULT_LORA_NAME)
    max_rank = int(suite.get("_lora_max_rank") or DEFAULT_MAX_LORA_RANK)
    return ServingPlan(
        backend=backend,
        lora_active=True,
        served_model_name=name,  # request the adapter, not the base
        extra_server_args=build_vllm_lora_args(adapter_dir, name=name, max_lora_rank=max_rank),
        adapter_local_dir=adapter_dir,
        challenger_tag=tag,
    )
