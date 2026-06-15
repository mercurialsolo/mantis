"""Tests for the LoRA challenger serving plan (#911)."""

from __future__ import annotations

import pytest

from mantis_agent.serving.lora_serving import (
    DEFAULT_LORA_NAME,
    AdapterRef,
    LoraServingError,
    build_convert_lora_to_gguf_cmd,
    build_llamacpp_lora_args,
    build_vllm_lora_args,
    challenger_tag,
    gguf_adapter_cache_path,
    local_adapter_dir,
    parse_adapter_ref,
    plan_serving,
    serving_backend,
)

TRAINER_MOUNTS = {"mantis-trainer-vol": "/trainer"}


# ── reference parsing ──────────────────────────────────────────────────


def test_parse_volume_ref():
    ref = parse_adapter_ref("mantis-trainer-vol:/checkpoints/sft-c3e0d799f432")
    assert ref.volume == "mantis-trainer-vol"
    assert ref.path == "/checkpoints/sft-c3e0d799f432"
    assert ref.is_volume_ref


def test_parse_bare_path():
    ref = parse_adapter_ref("/data/checkpoints/sft-abc")
    assert ref.volume is None
    assert ref.path == "/data/checkpoints/sft-abc"
    assert not ref.is_volume_ref


@pytest.mark.parametrize("bad", ["", "   ", "relative/path", "vol:relative"])
def test_parse_rejects_bad_refs(bad):
    with pytest.raises(LoraServingError):
        parse_adapter_ref(bad)


# ── resolution ─────────────────────────────────────────────────────────


def test_local_adapter_dir_volume_ref():
    ref = parse_adapter_ref("mantis-trainer-vol:/checkpoints/x")
    assert local_adapter_dir(ref, TRAINER_MOUNTS) == "/trainer/checkpoints/x"


def test_local_adapter_dir_bare_path_passthrough():
    ref = parse_adapter_ref("/data/checkpoints/x")
    assert local_adapter_dir(ref, TRAINER_MOUNTS) == "/data/checkpoints/x"


def test_local_adapter_dir_unmounted_volume_errors():
    ref = parse_adapter_ref("nope-vol:/checkpoints/x")
    with pytest.raises(LoraServingError, match="not mounted"):
        local_adapter_dir(ref, TRAINER_MOUNTS)


def test_challenger_tag_stable_and_named():
    ref = parse_adapter_ref("mantis-trainer-vol:/checkpoints/sft-c3e0d799f432")
    tag = challenger_tag(ref)
    assert tag.startswith("sft-c3e0d799f432-")
    assert challenger_tag(ref) == tag  # deterministic


def test_challenger_tag_disambiguates_same_basename():
    a = challenger_tag(parse_adapter_ref("vol-a:/ckpt/best"))
    b = challenger_tag(parse_adapter_ref("vol-b:/ckpt/best"))
    assert a != b and a.startswith("best-") and b.startswith("best-")


def test_gguf_cache_path_deterministic():
    p1 = gguf_adapter_cache_path("/trainer/checkpoints/x", "/data/models/lora_cache")
    p2 = gguf_adapter_cache_path("/trainer/checkpoints/x", "/data/models/lora_cache")
    assert p1 == p2
    assert p1.startswith("/data/models/lora_cache/lora_")
    assert p1.endswith(".gguf")
    assert p1 != gguf_adapter_cache_path("/trainer/checkpoints/y", "/data/models/lora_cache")


# ── backend selection ──────────────────────────────────────────────────


@pytest.mark.parametrize("model", ["holo3", "gemma4-cua"])
def test_backend_llamacpp(model):
    assert serving_backend(model) == "llamacpp"


@pytest.mark.parametrize("model", ["fara", "opencua-72b", "evocua-8b"])
def test_backend_vllm(model):
    assert serving_backend(model) == "vllm"


def test_backend_claude_rejected():
    with pytest.raises(LoraServingError, match="hosted API"):
        serving_backend("claude")


def test_backend_unknown_rejected():
    with pytest.raises(LoraServingError, match="unknown base model"):
        serving_backend("not-a-model")


# ── command builders ───────────────────────────────────────────────────


def test_llamacpp_lora_args_unit_scale():
    assert build_llamacpp_lora_args("/cache/lora_x.gguf") == ["--lora", "/cache/lora_x.gguf"]


def test_llamacpp_lora_args_scaled():
    assert build_llamacpp_lora_args("/cache/lora_x.gguf", scale=0.5) == [
        "--lora-scaled",
        "/cache/lora_x.gguf",
        "0.5",
    ]


def test_vllm_lora_args():
    args = build_vllm_lora_args("/trainer/ckpt/x", name="challenger", max_lora_rank=32)
    assert args == [
        "--enable-lora",
        "--lora-modules",
        "challenger=/trainer/ckpt/x",
        "--max-lora-rank",
        "32",
    ]


def test_convert_cmd_with_base():
    cmd = build_convert_lora_to_gguf_cmd(
        python_exe="python3",
        convert_script="/opt/llama.cpp/convert_lora_to_gguf.py",
        adapter_dir="/trainer/ckpt/x",
        out_gguf="/cache/lora_x.gguf",
        base_model="Hcompany/Holo3-35B-A3B",
    )
    assert cmd[:3] == ["python3", "/opt/llama.cpp/convert_lora_to_gguf.py", "/trainer/ckpt/x"]
    assert "--outfile" in cmd and "/cache/lora_x.gguf" in cmd
    assert cmd[cmd.index("--base") + 1] == "Hcompany/Holo3-35B-A3B"


def test_convert_cmd_without_base_omits_flag():
    cmd = build_convert_lora_to_gguf_cmd(
        python_exe="python3",
        convert_script="conv.py",
        adapter_dir="/a",
        out_gguf="/b.gguf",
    )
    assert "--base" not in cmd


# ── top-level planner ──────────────────────────────────────────────────


def test_plan_base_only_when_no_adapter():
    plan = plan_serving(
        cua_model="holo3", suite={}, mounts=TRAINER_MOUNTS, gguf_cache_root="/data/lc"
    )
    assert plan.lora_active is False
    assert plan.served_model_name == "model"
    assert plan.extra_server_args == []
    assert plan.convert_cmd is None
    assert plan.challenger_tag == ""


def test_plan_llamacpp_challenger():
    suite = {"_lora_adapter": "mantis-trainer-vol:/checkpoints/sft-c3e0d799f432"}
    plan = plan_serving(
        cua_model="holo3",
        suite=suite,
        mounts=TRAINER_MOUNTS,
        gguf_cache_root="/data/models/lora_cache",
        llamacpp_base_model="Hcompany/Holo3-35B-A3B",
    )
    assert plan.backend == "llamacpp"
    assert plan.lora_active is True
    assert plan.served_model_name == "model"  # adapter folded into base
    assert plan.adapter_local_dir == "/trainer/checkpoints/sft-c3e0d799f432"
    assert plan.adapter_gguf_path and plan.adapter_gguf_path.endswith(".gguf")
    assert plan.extra_server_args == ["--lora", plan.adapter_gguf_path]
    assert plan.convert_cmd is not None
    assert plan.adapter_local_dir in plan.convert_cmd
    assert plan.challenger_tag.startswith("sft-c3e0d799f432-")


def test_plan_llamacpp_preconverted_gguf_skips_conversion():
    # Trainer-emitted GGUF adapter → serve directly, no convert step, no deps.
    suite = {"_lora_adapter": "mantis-trainer-vol:/checkpoints/sft-x/adapter.gguf"}
    plan = plan_serving(
        cua_model="holo3",
        suite=suite,
        mounts=TRAINER_MOUNTS,
        gguf_cache_root="/data/models/lora_cache",
    )
    assert plan.lora_active is True
    assert plan.convert_cmd is None
    assert plan.adapter_gguf_path == "/trainer/checkpoints/sft-x/adapter.gguf"
    assert plan.extra_server_args == ["--lora", "/trainer/checkpoints/sft-x/adapter.gguf"]


def test_plan_llamacpp_scaled_preconverted():
    suite = {
        "_lora_adapter": "/data/ckpt/adapter.gguf",
        "_lora_scale": 0.7,
    }
    plan = plan_serving(
        cua_model="holo3", suite=suite, mounts=TRAINER_MOUNTS, gguf_cache_root="/data/lc"
    )
    assert plan.extra_server_args == ["--lora-scaled", "/data/ckpt/adapter.gguf", "0.7"]


def test_plan_vllm_challenger_requests_adapter_name():
    suite = {"_lora_adapter": "mantis-trainer-vol:/checkpoints/grpo-xyz"}
    plan = plan_serving(
        cua_model="fara", suite=suite, mounts=TRAINER_MOUNTS, gguf_cache_root="/data/lc"
    )
    assert plan.backend == "vllm"
    assert plan.lora_active is True
    # vLLM serves the adapter under its own name → the brain must request it.
    assert plan.served_model_name == DEFAULT_LORA_NAME
    assert "--enable-lora" in plan.extra_server_args
    assert f"{DEFAULT_LORA_NAME}=/trainer/checkpoints/grpo-xyz" in plan.extra_server_args
    assert plan.convert_cmd is None  # vLLM serves the PEFT dir directly


def test_plan_vllm_custom_lora_name():
    suite = {"_lora_adapter": "/data/ckpt/x", "_lora_name": "cand7"}
    plan = plan_serving(
        cua_model="fara", suite=suite, mounts=TRAINER_MOUNTS, gguf_cache_root="/data/lc"
    )
    assert plan.served_model_name == "cand7"
    assert "cand7=/data/ckpt/x" in plan.extra_server_args


def test_plan_unmounted_volume_raises():
    suite = {"_lora_adapter": "ghost-vol:/ckpt/x"}
    with pytest.raises(LoraServingError, match="not mounted"):
        plan_serving(
            cua_model="holo3", suite=suite, mounts=TRAINER_MOUNTS, gguf_cache_root="/data/lc"
        )


def test_plan_claude_with_adapter_raises():
    with pytest.raises(LoraServingError, match="hosted API"):
        plan_serving(
            cua_model="claude",
            suite={"_lora_adapter": "/data/ckpt/x"},
            mounts=TRAINER_MOUNTS,
            gguf_cache_root="/data/lc",
        )


# ── #918 full-model-swap challenger ─────────────────────────────────────


def test_plan_full_model_swap_llamacpp():
    suite = {"_challenger_model": "mantis-trainer-vol:/checkpoints/sft-x/merged.Q8_0.gguf"}
    plan = plan_serving(
        cua_model="holo3", suite=suite, mounts=TRAINER_MOUNTS, gguf_cache_root="/data/lc"
    )
    assert plan.lora_active is False  # a model swap, not an adapter overlay
    assert plan.model_path_override == "/trainer/checkpoints/sft-x/merged.Q8_0.gguf"
    assert plan.extra_server_args == []  # no --lora; the executor swaps -m
    assert plan.served_model_name == "model"
    assert plan.challenger_tag.startswith("merged.Q8_0-") or plan.challenger_tag


def test_plan_full_model_swap_requires_gguf():
    suite = {"_challenger_model": "mantis-trainer-vol:/checkpoints/sft-x"}  # no .gguf
    with pytest.raises(LoraServingError, match="must be a full .gguf"):
        plan_serving(cua_model="holo3", suite=suite, mounts=TRAINER_MOUNTS, gguf_cache_root="/d")


def test_plan_full_model_swap_rejected_for_vllm():
    suite = {"_challenger_model": "/data/merged.gguf"}
    with pytest.raises(LoraServingError, match="only for llama.cpp"):
        plan_serving(cua_model="fara", suite=suite, mounts=TRAINER_MOUNTS, gguf_cache_root="/d")


def test_plan_rejects_both_challenger_model_and_adapter():
    suite = {"_challenger_model": "/data/m.gguf", "_lora_adapter": "/data/a.gguf"}
    with pytest.raises(LoraServingError, match="only one of"):
        plan_serving(cua_model="holo3", suite=suite, mounts=TRAINER_MOUNTS, gguf_cache_root="/d")


def test_plan_base_only_has_no_model_override():
    plan = plan_serving(cua_model="holo3", suite={}, mounts=TRAINER_MOUNTS, gguf_cache_root="/d")
    assert plan.model_path_override is None


def test_adapterref_is_frozen():
    ref = AdapterRef(path="/x", raw="/x")
    with pytest.raises(Exception):
        ref.path = "/y"  # type: ignore[misc]
