"""Train Holo3-CUA on Modal — distillation from Claude trajectories.

Downloads Claude trajectory data from the Modal volume, converts to
training format, fine-tunes Holo3 with QLoRA, exports GGUF.

Usage:
    # Full pipeline:
    modal run training/modal_train_holo3.py

    # Skip data prep (already converted):
    modal run training/modal_train_holo3.py --skip-convert

    # Quick test:
    modal run training/modal_train_holo3.py --epochs 1 --max-samples 20
"""

import json
import os
import subprocess
import sys

import modal

app = modal.App("holo3-cua-training")
vol = modal.Volume.from_name("osworld-data", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "build-essential", "curl", "wget")
    .pip_install(
        "transformers>=5.2", "torch>=2.1", "datasets",
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
    timeout=86400,
    memory=81920,
    cpu=8,
)
def train_holo3_cua(
    epochs: int = 5,
    batch_size: int = 1,
    lr: float = 1e-4,
    lora_rank: int = 16,
    max_samples: int = 0,
    skip_convert: bool = False,
    export_gguf: bool = True,
):
    """Full training pipeline: convert + train + export GGUF."""

    # Install unsloth at runtime (needs GPU)
    print("Installing unsloth...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "unsloth[colab-new]", "--quiet"],
        check=False,
    )

    data_dir = "/data/training/holo3_distill"
    os.makedirs(data_dir, exist_ok=True)

    converted_path = os.path.join(data_dir, "holo3_distill_train.jsonl")
    output_dir = os.path.join(data_dir, "holo3-cua-distilled")

    # ── Step 1: Find and convert trajectory data ──
    if not skip_convert or not os.path.exists(converted_path):
        print("\n=== Step 1: Converting Claude trajectories ===")

        # Find all trajectory files on the volume
        results_dir = "/data/results"
        traj_files = [
            f for f in os.listdir(results_dir)
            if f.startswith("claude_trajectories_") and f.endswith(".jsonl")
        ]
        print(f"Found {len(traj_files)} trajectory files")

        # Find matching screenshot directories
        screenshots_base = "/data/screenshots"
        all_samples = []

        for traj_file in sorted(traj_files):
            traj_path = os.path.join(results_dir, traj_file)

            # Extract run_id from filename: claude_trajectories_bt_extract_20260418_022726.jsonl
            # → screenshots/bt_extract_20260418_022726/
            parts = traj_file.replace("claude_trajectories_", "").replace(".jsonl", "")
            screenshot_dir = os.path.join(screenshots_base, parts)

            if not os.path.isdir(screenshot_dir):
                print(f"  No screenshots for {traj_file}, skipping")
                continue

            png_count = len([f for f in os.listdir(screenshot_dir) if f.endswith(".png")])
            print(f"  {traj_file}: {png_count} screenshots in {screenshot_dir}")

            # Run converter
            sys.path.insert(0, "/root")
            from training.convert_claude_trajectories import convert_trajectory

            with open(traj_path) as f:
                step_offset = 0
                for line in f:
                    line = line.strip()
                    if not line or not line.startswith("{"):
                        continue
                    traj = json.loads(line)

                    from pathlib import Path
                    samples = convert_trajectory(
                        traj,
                        Path(screenshot_dir),
                        step_offset=step_offset,
                    )
                    all_samples.extend(samples)
                    step_offset += traj.get("steps", 0)

        # Write combined training data
        with open(converted_path, "w") as f:
            for sample in all_samples:
                f.write(json.dumps(sample) + "\n")

        print(f"Total training samples: {len(all_samples)}")
        vol.commit()
    else:
        # Count existing samples
        with open(converted_path) as f:
            count = sum(1 for _ in f)
        print(f"Using existing training data: {count} samples")

    # ── Step 2: Train ──
    print(f"\n=== Step 2: Training Holo3-CUA ({epochs} epochs) ===")

    cmd = [
        sys.executable, "/root/training/train_holo3_distill.py",
        "--data", converted_path,
        "--output", output_dir,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--lora-rank", str(lora_rank),
    ]
    if max_samples > 0:
        cmd.extend(["--max-samples", str(max_samples)])
    if export_gguf:
        cmd.append("--export-gguf")

    print(f"Command: {' '.join(cmd[-10:])}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"Training failed with exit code {result.returncode}")
        return

    vol.commit()

    # ── Step 3: Report ──
    print("\n=== Training Complete ===")
    print(f"Model saved to: {output_dir}")
    if export_gguf:
        gguf_dir = os.path.join(output_dir, "gguf")
        if os.path.isdir(gguf_dir):
            gguf_files = os.listdir(gguf_dir)
            print(f"GGUF files: {gguf_files}")
        else:
            print("GGUF export may have failed — check logs")

    print("\nTo deploy the fine-tuned model:")
    print(f"  1. GGUF is on Modal volume at: {output_dir}/gguf/")
    print("  2. Update HOLO3_MODEL_DIR in modal_cua_server.py to point to it")
    print("  3. Run: modal run modal_cua_server.py --model holo3 --task-file ...")


@app.local_entrypoint()
def main(
    epochs: int = 5,
    batch_size: int = 1,
    lr: float = 1e-4,
    lora_rank: int = 16,
    max_samples: int = 0,
    skip_convert: bool = False,
    export_gguf: bool = True,
):
    """Launch Holo3 training on Modal."""
    print("Holo3-CUA Distillation Training")
    print(f"  Epochs:     {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  LR:         {lr}")
    print(f"  LoRA rank:  {lora_rank}")
    print(f"  Export GGUF: {export_gguf}")

    train_holo3_cua.remote(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        lora_rank=lora_rank,
        max_samples=max_samples,
        skip_convert=skip_convert,
        export_gguf=export_gguf,
    )
