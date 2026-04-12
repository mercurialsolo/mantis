"""Train Gemma4-CUA on Modal — full pipeline.

Downloads AgentNet, converts to Gemma4 format, fine-tunes with QLoRA,
exports GGUF for llama.cpp deployment.

Usage:
    # Full pipeline (download + convert + train + export)
    modal run training/modal_train.py

    # Just training (data already converted)
    modal run training/modal_train.py --skip-download --skip-convert

    # Small test run
    modal run training/modal_train.py --max-tasks 100 --epochs 1
"""

import json
import os
import subprocess
import sys
import time

import modal

app = modal.App("gemma4-cua-training")

vol = modal.Volume.from_name("osworld-data", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "build-essential", "curl", "wget")
    .pip_install(
        # Core training deps (NO unsloth at build time — needs GPU)
        "transformers>=4.52", "torch>=2.1", "datasets",
        "trl>=0.14", "peft>=0.15", "bitsandbytes>=0.45",
        "accelerate>=1.5", "huggingface-hub",
        "pillow",
    )
    .add_local_dir("training", remote_path="/root/training")
)


@app.function(
    gpu="A100-80GB",
    image=image,
    volumes={"/data": vol},
    timeout=86400,  # 24 hours max
    memory=65536,
    cpu=8,
)
def train_gemma4_cua(
    max_tasks: int = 5000,
    epochs: int = 3,
    batch_size: int = 2,
    lr: float = 2e-4,
    lora_rank: int = 32,
    skip_download: bool = False,
    skip_convert: bool = False,
    export_gguf: bool = True,
):
    """Full training pipeline on A100."""
    import subprocess

    # Install unsloth at runtime (needs GPU, can't install at image build time)
    print("Installing unsloth (requires GPU)...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "unsloth[colab-new]", "--quiet",
    ], check=False)

    data_dir = "/data/training"
    agentnet_dir = os.path.join(data_dir, "agentnet")
    converted_path = os.path.join(data_dir, "gemma4_cua_train.jsonl")
    output_dir = os.path.join(data_dir, "gemma4-31b-cua")

    os.makedirs(data_dir, exist_ok=True)

    # ── Step 1: Download AgentNet ──
    if not skip_download:
        meta_path = os.path.join(agentnet_dir, "meta_data_merged.jsonl")
        if os.path.exists(meta_path):
            print(f"AgentNet already downloaded at {agentnet_dir}")
        else:
            print("Downloading AgentNet dataset (200GB)...")
            print("NOTE: This is a large download. First run will take 30-60 minutes.")
            os.makedirs(agentnet_dir, exist_ok=True)

            from huggingface_hub import snapshot_download
            snapshot_download(
                "xlangai/AgentNet",
                repo_type="dataset",
                local_dir=agentnet_dir,
                # Only download the metadata JSONL (not 200GB of images)
                # Images can be downloaded separately if needed for multimodal training
                allow_patterns=["*.jsonl", "*.json"],
            )
            vol.commit()
            print("AgentNet metadata downloaded.")
    else:
        print("Skipping download")

    # ── Step 2: Convert to Gemma4 format ──
    if not skip_convert:
        meta_path = os.path.join(agentnet_dir, "meta_data_merged.jsonl")
        if not os.path.exists(meta_path):
            # Try alternative paths
            for alt in ["train.jsonl", "data.jsonl"]:
                alt_path = os.path.join(agentnet_dir, alt)
                if os.path.exists(alt_path):
                    meta_path = alt_path
                    break

        if not os.path.exists(meta_path):
            print(f"ERROR: AgentNet metadata not found at {meta_path}")
            print(f"Contents of {agentnet_dir}: {os.listdir(agentnet_dir) if os.path.exists(agentnet_dir) else 'N/A'}")
            return {"error": "AgentNet metadata not found"}

        print(f"Converting AgentNet → Gemma4 format (max {max_tasks} tasks)...")
        sys.path.insert(0, "/root")
        from training.convert_agentnet import main as convert_main

        # Simulate CLI args
        sys.argv = [
            "convert_agentnet.py",
            "--input", meta_path,
            "--output", converted_path,
            "--screen-width", "1280",
            "--screen-height", "720",
            "--max-tasks", str(max_tasks),
            "--min-alignment", "6",
            "--min-steps", "2",
            "--max-steps", "25",
        ]
        convert_main()
        vol.commit()

        # Count samples
        n_samples = sum(1 for _ in open(converted_path))
        print(f"Converted {n_samples} training samples")
    else:
        print("Skipping conversion")

    if not os.path.exists(converted_path):
        print(f"ERROR: Training data not found at {converted_path}")
        return {"error": "Training data not found"}

    # ── Step 3: Train with QLoRA ──
    print(f"\n{'='*60}")
    print(f"Training Gemma4-31B CUA")
    print(f"  Data:    {converted_path}")
    print(f"  Output:  {output_dir}")
    print(f"  Epochs:  {epochs}")
    print(f"  Batch:   {batch_size}")
    print(f"  LR:      {lr}")
    print(f"  LoRA:    rank={lora_rank}")
    print(f"{'='*60}")

    from training.train_gemma4_cua import train as train_fn
    import argparse

    args = argparse.Namespace(
        data=converted_path,
        output=output_dir,
        max_samples=0,
        model="google/gemma-4-31b-it",
        max_seq_length=4096,
        lora_rank=lora_rank,
        lora_alpha=lora_rank * 2,
        epochs=epochs,
        batch_size=batch_size,
        gradient_accumulation=4,
        lr=lr,
        export_gguf=export_gguf,
    )

    train_fn(args)
    vol.commit()

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Model saved to: {output_dir}")
    if export_gguf:
        print(f"GGUF saved to: {output_dir}/gguf/")
    print(f"{'='*60}")

    return {
        "status": "complete",
        "output_dir": output_dir,
        "epochs": epochs,
    }


@app.local_entrypoint()
def main(
    max_tasks: int = 5000,
    epochs: int = 3,
    batch_size: int = 2,
    lr: float = 2e-4,
    lora_rank: int = 32,
    skip_download: bool = False,
    skip_convert: bool = False,
    export_gguf: bool = True,
):
    """Train Gemma4-CUA on Modal A100."""
    print(f"Mantis — Gemma4 CUA Training (Modal)")
    print(f"  Tasks:  {max_tasks}")
    print(f"  Epochs: {epochs}")
    print(f"  LoRA:   rank={lora_rank}")
    print()

    result = train_gemma4_cua.remote(
        max_tasks=max_tasks,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        lora_rank=lora_rank,
        skip_download=skip_download,
        skip_convert=skip_convert,
        export_gguf=export_gguf,
    )
    print(f"\nResult: {json.dumps(result, indent=2)}")
