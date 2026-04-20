#!/usr/bin/env python3
"""Fine-tune Holo3 on Claude distillation data using QLoRA.

Takes the JSONL from convert_claude_trajectories.py (screenshot + action pairs)
and fine-tunes Holo3-35B-A3B to learn BoatTrader-specific CUA behavior:
- Click listing TITLE TEXT, not photos
- Scroll past gallery to description
- Format done() with VIABLE | Year: ... | Phone: ...
- Escape gallery traps

Architecture: Qwen3.5 MoE (35B total, 3B active) — needs special handling:
- Use Unsloth's FastVisionModel for memory-efficient QLoRA
- MoE gate + expert layers are frozen, only attention + projection fine-tuned
- Vision encoder (mmproj) stays frozen — we're teaching action behavior, not perception

Usage:
    # Via Modal (recommended):
    modal run training/modal_train_holo3.py

    # Local on A100:
    python training/train_holo3_distill.py \
        --data training/data/holo3_distill_train.jsonl \
        --output training/output/holo3-cua \
        --epochs 5 --batch-size 1 --lr 1e-4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def load_dataset(data_path: str, max_samples: int = 0):
    """Load distillation training data with image paths."""
    from datasets import Dataset

    samples = []
    with open(data_path) as f:
        for line in f:
            sample = json.loads(line.strip())
            conversations = sample.get("conversations", [])
            image_path = sample.get("image", "")

            # Verify image exists
            if image_path and not os.path.exists(image_path):
                continue

            # Format as text for SFT
            text = format_for_sft(conversations)
            if text:
                samples.append({
                    "text": text,
                    "image": image_path,
                })
            if max_samples > 0 and len(samples) >= max_samples:
                break

    logger.info(f"Loaded {len(samples)} training samples ({len(samples)} with images)")
    return Dataset.from_list(samples)


def format_for_sft(conversations: list[dict]) -> str:
    """Format conversations into Qwen3.5 chat template.

    Qwen3.5 (Holo3's base) uses:
    <|im_start|>system\n...<|im_end|>
    <|im_start|>user\n...<|im_end|>
    <|im_start|>assistant\n...<|im_end|>
    """
    parts = []
    for msg in conversations:
        role_map = {"system": "system", "human": "user", "gpt": "assistant"}
        role = role_map.get(msg["from"], msg["from"])
        value = msg["value"]
        parts.append(f"<|im_start|>{role}\n{value}<|im_end|>")

    return "\n".join(parts) if parts else ""


def train(args):
    """Run QLoRA fine-tuning."""
    try:
        from unsloth import FastVisionModel
        logger.info("Using Unsloth for memory-efficient training")
        use_unsloth = True
    except ImportError:
        logger.warning("Unsloth not available, falling back to standard HF")
        use_unsloth = False

    # 1. Load model
    model_name = args.model
    logger.info(f"Loading {model_name}...")

    if use_unsloth:
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name,
            max_seq_length=args.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )

        model = FastVisionModel.get_peft_model(
            model,
            r=args.lora_rank,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import get_peft_model, LoraConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    # 2. Load dataset
    logger.info(f"Loading training data from {args.data}...")
    dataset = load_dataset(args.data, max_samples=args.max_samples)
    logger.info(f"Training on {len(dataset)} samples")

    # 3. Training config
    from transformers import TrainingArguments
    from trl import SFTTrainer

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_8bit" if use_unsloth else "adamw_torch",
        report_to="none",
        max_grad_norm=1.0,
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,  # Don't pack — each sample has its own image context
    )

    # 4. Train
    logger.info("Starting training...")
    trainer.train()

    # 5. Save
    logger.info(f"Saving to {args.output}...")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    # 6. Export to GGUF
    if args.export_gguf and use_unsloth:
        gguf_dir = os.path.join(args.output, "gguf")
        logger.info(f"Exporting GGUF to {gguf_dir}...")
        model.save_pretrained_gguf(
            gguf_dir,
            tokenizer,
            quantization_method="q8_0",  # Q8_0 for best quality (fits 1x A100)
        )
        logger.info("GGUF export complete")

    logger.info("Training complete!")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Holo3 on Claude distillation data")

    parser.add_argument("--data", required=True, help="Training JSONL from convert_claude_trajectories.py")
    parser.add_argument("--output", default="training/output/holo3-cua", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit samples (0=all)")

    parser.add_argument("--model", default="Hcompany/Holo3-35B-A3B", help="Base model")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Max sequence length")

    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank (16 for small dataset)")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")

    parser.add_argument("--epochs", type=int, default=5, help="Training epochs (more for small data)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (1 for 35B MoE)")
    parser.add_argument("--gradient-accumulation", type=int, default=8, help="Gradient accumulation")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (lower for distillation)")

    parser.add_argument("--export-gguf", action="store_true", help="Export GGUF for llama.cpp")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
