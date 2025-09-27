#!/usr/bin/env python3
"""
Fin-R1 CPU-Optimized Training Script for Fluence
------------------------------------------------
Efficient CPU-based LoRA fine-tuning optimized for Fluence VMs.
Uses memory-efficient techniques for 7B model training on CPU.
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import torch
from datasets import Dataset
from loguru import logger
from trl import SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType


@dataclass
class CPUTrainingConfig:
    """CPU-optimized training configuration for Fluence VMs."""
    base_model: str = "Qwen/Qwen2-7B-Instruct"
    max_seq_length: int = 1024  # Shorter for CPU efficiency

    # LoRA parameters optimized for CPU
    lora_r: int = 32  # Moderate rank for CPU
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    target_modules: List[str] = None

    # Training parameters for CPU
    batch_size: int = 1
    gradient_accumulation_steps: int = 8  # Effective batch size = 8
    learning_rate: float = 1e-4
    num_epochs: int = 1
    warmup_steps: int = 10

    # CPU optimizations
    use_cpu: bool = True
    fp16: bool = False  # CPU doesn't support fp16
    bf16: bool = False  # Most CPUs don't support bf16
    dataloader_num_workers: int = 2
    gradient_checkpointing: bool = True

    # Memory optimizations
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500

    def __post_init__(self):
        if self.target_modules is None:
            # Focus on key attention layers for efficiency
            self.target_modules = ["q_proj", "v_proj", "o_proj"]


class CPUTradingDataProcessor:
    """CPU-optimized data processing for trading instruction tuning."""

    def __init__(self, data_path: str, config: CPUTrainingConfig):
        self.data_path = data_path
        self.config = config

        # Load data with memory optimization
        self.data = pd.read_csv(data_path, dtype={'side_raw': 'category'})

        # Load metadata
        metadata_path = Path(data_path).parent / "metadata.json"
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        logger.info("Loaded {} samples for CPU training", len(self.data))

    def create_instruction_dataset(self) -> Dataset:
        """Create memory-efficient instruction dataset."""
        instructions = []

        seq_len = self.metadata['seq_len']
        core_features = self.metadata['core_feature_names']

        # Process in chunks to manage memory
        chunk_size = 1000
        for chunk_start in range(0, len(self.data), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(self.data))
            chunk = self.data.iloc[chunk_start:chunk_end]

            for _, row in chunk.iterrows():
                # Extract features
                latest_features = {}
                for feature in core_features:
                    col_name = f"{feature}_{seq_len-1}"
                    if col_name in row:
                        latest_features[feature] = row[col_name]

                # Create instruction
                instruction = self._create_trading_instruction(latest_features)
                response = self._create_trading_response(row)

                # Combine for training
                full_text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"

                instructions.append({"text": full_text})

        # Create dataset
        dataset = Dataset.from_list(instructions)
        logger.info("Created instruction dataset with {} samples", len(dataset))

        return dataset

    def _create_trading_instruction(self, features: Dict[str, float]) -> str:
        """Create concise trading instruction."""
        return (
            f"Analyze this ETH/USDC market data and provide a trading decision:\n\n"
            f"Price: ${features.get('price', 0):.4f}\n"
            f"Volatility: {features.get('volatility', 0)*100:.2f}%\n"
            f"Volume: {features.get('volume', 0):.4f} ETH\n"
            f"Volume MA: {features.get('volume_ma', 0):.4f}\n"
            f"Liquidity Impact: {features.get('liquidity_impact', 0):.2f}\n"
            f"Price Change: {features.get('price_change', 0)*100:.2f}%\n\n"
            f"Provide structured trading decision with action, size, slippage, and timeline."
        )

    def _create_trading_response(self, row: pd.Series) -> str:
        """Create structured trading response."""
        return (
            f"**Trading Decision:**\n"
            f"Action: {row['side_raw']}\n"
            f"Size: {row['size']:.3f} ETH\n"
            f"Slippage: {row['slippage_bps']:.0f} bps\n"
            f"Deadline: {row['deadline_s']:.0f} seconds\n\n"
            f"**Rationale:** Based on current market conditions, this {row['side_raw'].lower()} "
            f"position with {row['size']:.3f} ETH exposure provides optimal risk-adjusted returns "
            f"given the volatility and liquidity profile."
        )


def setup_cpu_model_and_tokenizer(config: CPUTrainingConfig):
    """Setup model and tokenizer optimized for CPU training."""
    logger.info("Loading model for CPU training: {}", config.base_model)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        trust_remote_code=True,
        use_fast=True
    )

    # Ensure padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model with CPU optimizations
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.float32,  # Use float32 for CPU
        device_map=None,  # Keep on CPU
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    # Move to CPU explicitly
    model = model.to('cpu')

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none"
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Enable gradient checkpointing for memory efficiency
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    logger.info("Model setup complete for CPU training")
    return model, tokenizer


def train_finr1_cpu(config: CPUTrainingConfig, data_dir: str, output_dir: str):
    """Train Fin-R1 on CPU with memory optimizations."""

    logger.info("🚀 Starting CPU-optimized Fin-R1 training")

    # Setup model and tokenizer
    model, tokenizer = setup_cpu_model_and_tokenizer(config)

    # Load and process data
    train_processor = CPUTradingDataProcessor(
        data_path=str(Path(data_dir) / "train.csv"),
        config=config
    )
    train_dataset = train_processor.create_instruction_dataset()

    # Validation dataset (smaller for CPU efficiency)
    val_processor = CPUTradingDataProcessor(
        data_path=str(Path(data_dir) / "val.csv"),
        config=config
    )
    val_dataset = val_processor.create_instruction_dataset()

    # Take smaller validation set for CPU
    val_dataset = val_dataset.select(range(min(100, len(val_dataset))))

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Training arguments optimized for CPU
    training_args = TrainingArguments(
        output_dir=str(output_path),
        overwrite_output_dir=True,

        # Training parameters
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,

        # Optimization
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        warmup_steps=config.warmup_steps,

        # Precision (CPU optimized)
        fp16=False,  # CPU doesn't support fp16
        bf16=False,  # Most CPUs don't support bf16

        # Logging and saving
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",

        # Memory optimizations
        gradient_checkpointing=config.gradient_checkpointing,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=False,  # Disable for CPU

        # Monitoring
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Disable external logging
        report_to=None,

        # CPU-specific optimizations
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        max_length=config.max_seq_length,
        return_tensors="pt"
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        packing=False,  # Disable packing for stability
    )

    # Start training
    logger.info("🔥 Starting training on CPU...")
    trainer.train()

    # Save model
    logger.info("💾 Saving trained model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_path)

    # Save config
    config_dict = {
        "base_model": config.base_model,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "target_modules": config.target_modules,
        "max_seq_length": config.max_seq_length,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "num_epochs": config.num_epochs,
        "training_type": "cpu_optimized"
    }

    with open(output_path / "training_config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)

    logger.success("✅ CPU training complete!")
    logger.info("📁 Model saved to: {}", output_path)

    # Test inference
    test_inference(model, tokenizer, config)


def test_inference(model, tokenizer, config: CPUTrainingConfig):
    """Test the trained model with sample inference."""
    logger.info("🧪 Testing model inference...")

    test_prompt = """### Instruction:
Analyze this ETH/USDC market data and provide a trading decision:

Price: $3500.0000
Volatility: 2.50%
Volume: 0.0850 ETH
Volume MA: 0.0800
Liquidity Impact: 1.20
Price Change: +0.15%

Provide structured trading decision with action, size, slippage, and timeline.

### Response:
"""

    # Tokenize
    inputs = tokenizer(
        test_prompt,
        return_tensors="pt",
        max_length=config.max_seq_length,
        truncation=True
    )

    # Generate
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = response[len(test_prompt):].strip()

    logger.info("Sample prediction: {}", generated[:200] + "..." if len(generated) > 200 else generated)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train Fin-R1 on CPU with Fluence")
    parser.add_argument("--data-dir", required=True, help="Training data directory")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--base-model", default="Qwen/Qwen2-7B-Instruct", help="Base model")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8, help="Gradient accumulation")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--max-seq-length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--use-cpu", action="store_true", help="Force CPU usage")

    args = parser.parse_args()

    # Create config
    config = CPUTrainingConfig(
        base_model=args.base_model,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        lora_r=args.lora_r,
        max_seq_length=args.max_seq_length,
        use_cpu=args.use_cpu
    )

    # Train
    train_finr1_cpu(config, args.data_dir, args.output_dir)

    print(f"\n🎉 Fin-R1 CPU training complete!")
    print(f"📁 Output: {args.output_dir}")
    print(f"🚀 Ready for deployment!")


if __name__ == "__main__":
    main()