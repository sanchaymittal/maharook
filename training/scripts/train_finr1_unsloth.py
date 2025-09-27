#!/usr/bin/env python3
"""
Fin-R1 Unsloth LoRA Training Script
-----------------------------------
Efficient LoRA fine-tuning of Fin-R1/Qwen2 using Unsloth for trading decisions.
Optimized for GPU training with memory efficiency.
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
from transformers import TrainingArguments

# Unsloth imports
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template


@dataclass
class FinR1TrainingConfig:
    """Fin-R1 LoRA training configuration optimized for Unsloth."""
    base_model: str = "Qwen/Qwen2-7B-Instruct"  # Use this as Fin-R1 base
    max_seq_length: int = 2048
    load_in_4bit: bool = True  # QLoRA for memory efficiency

    # LoRA parameters
    lora_r: int = 64  # Higher rank for 7B model
    lora_alpha: int = 16
    lora_dropout: float = 0.0  # Unsloth optimizes dropout
    target_modules: List[str] = None

    # Training parameters
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 1
    warmup_steps: int = 5
    max_steps: int = -1

    # Optimization
    fp16: bool = False
    bf16: bool = True  # Better for large models
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"

    # Logging and saving
    logging_steps: int = 1
    save_steps: int = 100
    eval_steps: int = 50
    output_dir: str = "finr1_lora_output"

    def __post_init__(self):
        if self.target_modules is None:
            # Target all linear layers for Qwen2
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]


class TradingDataProcessor:
    """Process trading data for Fin-R1 instruction fine-tuning."""

    def __init__(self, data_path: str, config: FinR1TrainingConfig):
        self.data_path = data_path
        self.config = config

        # Load trading data
        self.data = pd.read_csv(data_path)

        # Load metadata
        metadata_path = Path(data_path).parent / "metadata.json"
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        logger.info("Loaded {} trading samples for Fin-R1 training", len(self.data))

    def create_instruction_dataset(self) -> Dataset:
        """Convert trading data to instruction-following format."""
        instructions = []

        seq_len = self.metadata['seq_len']
        core_features = self.metadata['core_feature_names']

        for _, row in self.data.iterrows():
            # Extract latest market features
            latest_features = {}
            for feature in core_features:
                col_name = f"{feature}_{seq_len-1}"
                if col_name in row:
                    latest_features[feature] = row[col_name]

            # Create instruction prompt
            instruction = self._create_market_instruction(latest_features)

            # Create response
            response = self._create_trading_response(row)

            # Format as conversation
            conversation = {
                "instruction": instruction,
                "input": "",
                "output": response
            }

            instructions.append(conversation)

        # Convert to Hugging Face dataset
        dataset = Dataset.from_list(instructions)
        logger.info("Created instruction dataset with {} samples", len(dataset))

        return dataset

    def _create_market_instruction(self, features: Dict[str, float]) -> str:
        """Create market analysis instruction."""
        return (
            "You are a professional cryptocurrency trader analyzing ETH/USDC market conditions. "
            "Based on the following market data, provide a detailed trading decision including "
            "action (BUY/SELL/HOLD), position size, slippage tolerance, and execution timeline.\n\n"
            f"Market Data:\n"
            f"• Current Price: ${features.get('price', 0):.4f}\n"
            f"• Volatility: {features.get('volatility', 0)*100:.2f}%\n"
            f"• Volume (10m): {features.get('volume', 0):.4f} ETH\n"
            f"• Volume Moving Average: {features.get('volume_ma', 0):.4f}\n"
            f"• Liquidity Impact: {features.get('liquidity_impact', 0):.2f}\n"
            f"• Price Change (10m): {features.get('price_change', 0)*100:.2f}%\n\n"
            "Provide your trading analysis and decision:"
        )

    def _create_trading_response(self, row: pd.Series) -> str:
        """Create structured trading response."""
        return (
            f"Based on the current market conditions, here is my trading analysis:\n\n"
            f"**Market Assessment:**\n"
            f"The market shows specific patterns that warrant a {row['side_raw'].lower()} position.\n\n"
            f"**Trading Decision:**\n"
            f"• Action: {row['side_raw']}\n"
            f"• Position Size: {row['size']:.3f} ETH\n"
            f"• Slippage Tolerance: {row['slippage_bps']:.0f} basis points\n"
            f"• Execution Deadline: {row['deadline_s']:.0f} seconds\n\n"
            f"**Risk Management:**\n"
            f"This position size represents appropriate risk management given current volatility "
            f"and liquidity conditions. The slippage tolerance accounts for market impact, "
            f"and the execution timeline balances opportunity capture with market risk."
        )


def format_prompts(examples, tokenizer):
    """Format prompts for Unsloth training."""
    texts = []
    for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
        # Create chat format
        messages = [
            {"role": "user", "content": instruction + ("\n" + input_text if input_text else "")},
            {"role": "assistant", "content": output}
        ]

        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)

    return {"text": texts}


def train_finr1_lora(config: FinR1TrainingConfig, data_dir: str):
    """Train Fin-R1 LoRA adapter using Unsloth."""

    logger.info("🚀 Starting Fin-R1 LoRA training with Unsloth")

    # Load model with Unsloth optimizations
    logger.info("Loading model: {}", config.base_model)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.base_model,
        max_seq_length=config.max_seq_length,
        dtype=torch.bfloat16 if config.bf16 else torch.float16,
        load_in_4bit=config.load_in_4bit,
        trust_remote_code=True
    )

    # Configure LoRA
    logger.info("Configuring LoRA with rank={}, alpha={}", config.lora_r, config.lora_alpha)
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=config.target_modules,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,  # Rank stabilized LoRA
        loftq_config=None  # LoftQ quantization
    )

    # Setup chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="qwen-2.5",  # Use Qwen chat format
    )

    # Load and process data
    train_processor = TradingDataProcessor(
        data_path=str(Path(data_dir) / "train.csv"),
        config=config
    )
    train_dataset = train_processor.create_instruction_dataset()

    val_processor = TradingDataProcessor(
        data_path=str(Path(data_dir) / "val.csv"),
        config=config
    )
    val_dataset = val_processor.create_instruction_dataset()

    # Format datasets
    train_dataset = train_dataset.map(
        lambda examples: format_prompts(examples, tokenizer),
        batched=True
    )
    val_dataset = val_dataset.map(
        lambda examples: format_prompts(examples, tokenizer),
        batched=True
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps if config.max_steps > 0 else None,
        num_train_epochs=config.num_epochs if config.max_steps <= 0 else None,
        learning_rate=config.learning_rate,
        fp16=config.fp16 and not is_bfloat16_supported(),
        bf16=config.bf16 and is_bfloat16_supported(),
        logging_steps=config.logging_steps,
        optim=config.optim,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        seed=42,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None,  # Disable wandb for now
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        dataset_num_proc=2,
        packing=False,  # Disable packing for better quality
        args=training_args,
    )

    # Train
    logger.info("🔥 Starting training...")
    trainer.train()

    # Save model
    logger.info("💾 Saving trained model...")
    trainer.save_model()

    # Save Unsloth format for faster loading
    model.save_pretrained_merged(
        config.output_dir + "_merged",
        tokenizer,
        save_method="merged_16bit"
    )

    # Save GGUF for deployment
    model.save_pretrained_gguf(
        config.output_dir + "_gguf",
        tokenizer,
        quantization_method="q4_k_m"
    )

    logger.info("✅ Training complete!")
    logger.info("📁 LoRA adapter: {}", config.output_dir)
    logger.info("📁 Merged model: {}_merged", config.output_dir)
    logger.info("📁 GGUF model: {}_gguf", config.output_dir)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train Fin-R1 LoRA with Unsloth")
    parser.add_argument("--data-dir", required=True, help="Training data directory")
    parser.add_argument("--output-dir", default="finr1_lora_output", help="Output directory")
    parser.add_argument("--base-model", default="Qwen/Qwen2-7B-Instruct", help="Base model")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--lora-r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Max sequence length")

    args = parser.parse_args()

    # Create config
    config = FinR1TrainingConfig(
        base_model=args.base_model,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_seq_length=args.max_seq_length
    )

    # Train
    train_finr1_lora(config, args.data_dir)

    print(f"\n🎉 Fin-R1 LoRA training complete!")
    print(f"📁 Output: {args.output_dir}")
    print(f"🔧 Ready for deployment with Ollama!")


if __name__ == "__main__":
    main()