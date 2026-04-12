#!/usr/bin/env python3
"""Fine-tune Gemma4 31B for CUA tasks using QLoRA + Unsloth.

Trains Gemma4 31B on AgentNet trajectories converted to Gemma4's
native tool-calling format. QLoRA keeps VRAM at ~22GB on single A100.

The resulting model should:
- Click/type/scroll accurately from screenshots (like EvoCUA)
- Use native Gemma4 tool-calling tokens (cleaner than raw pyautogui)
- Retain general reasoning + CLI capabilities (unlike from-scratch CUA)

Usage:
    # Local training on A100
    python training/train_gemma4_cua.py \
        --data training/data/gemma4_cua_train.jsonl \
        --output training/output/gemma4-31b-cua \
        --epochs 3 --batch-size 4 --lr 2e-4

    # Via Modal (recommended)
    modal run training/modal_train.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def load_dataset(data_path: str, max_samples: int = 0):
    """Load the converted training data."""
    from datasets import Dataset

    samples = []
    with open(data_path) as f:
        for line in f:
            sample = json.loads(line.strip())
            # Flatten conversations to a single text for SFT
            text = format_for_sft(sample["conversations"])
            if text:
                samples.append({"text": text})
            if max_samples > 0 and len(samples) >= max_samples:
                break

    logger.info(f"Loaded {len(samples)} training samples")
    return Dataset.from_list(samples)


def format_for_sft(conversations: list[dict]) -> str:
    """Format conversations into Gemma4 chat template string.

    Uses the standard Gemma4 format:
    <start_of_turn>user\n...<end_of_turn>
    <start_of_turn>model\n...<end_of_turn>
    """
    parts = []
    for msg in conversations:
        role = msg["from"]
        value = msg["value"]

        if role == "system":
            parts.append(f"<start_of_turn>user\nSystem: {value}<end_of_turn>")
        elif role == "human":
            parts.append(f"<start_of_turn>user\n{value}<end_of_turn>")
        elif role == "gpt":
            parts.append(f"<start_of_turn>model\n{value}<end_of_turn>")

    if not parts:
        return ""

    return "\n".join(parts)


def train(args):
    """Run QLoRA fine-tuning with Unsloth."""
    try:
        from unsloth import FastVisionModel
        logger.info("Using Unsloth for 2x faster training")
        use_unsloth = True
    except ImportError:
        logger.warning("Unsloth not available, using standard HuggingFace")
        use_unsloth = False

    # 1. Load model
    model_name = args.model
    logger.info(f"Loading {model_name}...")

    if use_unsloth:
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name,
            max_seq_length=args.max_seq_length,
            dtype=None,  # Auto-detect
            load_in_4bit=True,  # QLoRA
        )

        # Add LoRA adapters
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
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
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
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
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
        packing=True,
    )

    # 4. Train
    logger.info("Starting training...")
    trainer.train()

    # 5. Save
    logger.info(f"Saving to {args.output}...")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    # 6. Optionally export to GGUF for llama.cpp
    if args.export_gguf and use_unsloth:
        gguf_path = os.path.join(args.output, "gguf")
        logger.info(f"Exporting GGUF to {gguf_path}...")
        model.save_pretrained_gguf(
            gguf_path,
            tokenizer,
            quantization_method="q4_k_m",
        )
        logger.info("GGUF export complete")

    logger.info("Training complete!")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma4 for CUA")

    # Data
    parser.add_argument("--data", required=True, help="Training data JSONL (from convert_agentnet.py)")
    parser.add_argument("--output", default="training/output/gemma4-31b-cua", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit samples (0=all)")

    # Model
    parser.add_argument("--model", default="google/gemma-4-31b-it", help="Base model")
    parser.add_argument("--max-seq-length", type=int, default=4096, help="Max sequence length")

    # LoRA
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha")

    # Training
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per GPU")
    parser.add_argument("--gradient-accumulation", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")

    # Export
    parser.add_argument("--export-gguf", action="store_true", help="Export to GGUF for llama.cpp")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
