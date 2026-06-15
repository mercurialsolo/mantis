"""#911 parity — Baseten deployment-level LoRA challenger boot args.

The Baseten pod boots one shared inference server at model-load, so the adapter
is fixed for the deployment via ``MANTIS_LORA_ADAPTER`` (champion unset →
base; challenger set → base + adapter). These pin the env → llama-server/vLLM
arg translation in ``BasetenCUARuntime._boot_lora_args``.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def runtime(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MANTIS_BRAIN", "holo3")
    monkeypatch.setenv("MANTIS_SKIP_BRAIN_LOAD", "1")
    from mantis_agent.baseten_server.runtime import BasetenCUARuntime
    rt = BasetenCUARuntime()
    rt.load = lambda: None  # type: ignore[assignment]
    return rt


def test_no_adapter_env_is_noop(runtime, monkeypatch):
    monkeypatch.delenv("MANTIS_LORA_ADAPTER", raising=False)
    args, served = runtime._boot_lora_args("holo3")
    assert args == []
    assert served is None


def test_llamacpp_adapter_appends_lora(runtime, monkeypatch):
    monkeypatch.setenv("MANTIS_LORA_ADAPTER", "/models/holo3_adapter/adapter.gguf")
    args, served = runtime._boot_lora_args("holo3")
    assert args == ["--lora", "/models/holo3_adapter/adapter.gguf"]
    # llama.cpp folds the adapter into the base → served name unchanged.
    assert served == "model"


def test_llamacpp_scaled_adapter(runtime, monkeypatch):
    monkeypatch.setenv("MANTIS_LORA_ADAPTER", "/models/holo3_adapter/adapter.gguf")
    monkeypatch.setenv("MANTIS_LORA_SCALE", "0.5")
    args, _ = runtime._boot_lora_args("holo3")
    assert args == ["--lora-scaled", "/models/holo3_adapter/adapter.gguf", "0.5"]


def test_llamacpp_raw_peft_dir_rejected(runtime, monkeypatch):
    # No .gguf suffix → would need conversion, which the serving image can't do.
    monkeypatch.setenv("MANTIS_LORA_ADAPTER", "/models/holo3_adapter")
    with pytest.raises(RuntimeError, match="pre-converted .gguf"):
        runtime._boot_lora_args("holo3")


def test_vllm_adapter_enables_lora_and_returns_served_name(runtime, monkeypatch):
    monkeypatch.setenv("MANTIS_LORA_ADAPTER", "/models/fara_adapter")
    args, served = runtime._boot_lora_args("fara")
    assert "--enable-lora" in args
    assert "challenger=/models/fara_adapter" in args
    # vLLM serves the adapter under its own name → the brain must request it.
    assert served == "challenger"


def test_vllm_custom_lora_name(runtime, monkeypatch):
    monkeypatch.setenv("MANTIS_LORA_ADAPTER", "/models/fara_adapter")
    monkeypatch.setenv("MANTIS_LORA_NAME", "cand12")
    args, served = runtime._boot_lora_args("fara")
    assert served == "cand12"
    assert "cand12=/models/fara_adapter" in args
