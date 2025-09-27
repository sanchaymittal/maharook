#!/usr/bin/env python3
"""
ROOK LoRA Training Script - Real LoRA Fine-tuning
-------------------------------------------------
Fine-tunes parent models using LoRA adapters for trading decisions.
Converts trading features to text and uses actual transformer fine-tuning.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


@dataclass
class LoRATrainingConfig:
    """LoRA fine-tuning configuration."""
    parent_model: str = "microsoft/DialoGPT-small"
    max_length: int = 512
    batch_size: int = 8
    learning_rate: float = 2e-4
    num_epochs: int = 3
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: list[str] = None
    warmup_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 250
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01

    def __post_init__(self):
        if self.target_modules is None:
            # Target attention layers for LoRA
            self.target_modules = ["c_attn", "c_proj", "c_fc"]


class TradingTextDataset(Dataset):
    """Dataset that converts trading features to text for language model training."""

    def __init__(self, data_path: str, tokenizer: AutoTokenizer, config: LoRATrainingConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.data_path = data_path

        # Load labeled data
        self.data = pd.read_csv(data_path)

        # Ensure tokenizer has pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Convert trading data to text format
        self.texts = self._convert_to_text()
        logger.info("Created {} text samples for training", len(self.texts))

    def _convert_to_text(self) -> list[str]:
        """Convert trading features and labels to natural language text."""
        texts = []

        # Load metadata to understand the structure
        metadata_path = Path(self.data_path).parent / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        seq_len = metadata['seq_len']
        core_features = metadata['core_feature_names']

        for _, row in self.data.iterrows():
            # Extract latest values from the sequence (last timestep)
            latest_features = {}
            for feature in core_features:
                # Get the last value in the sequence for this feature
                col_name = f"{feature}_{seq_len-1}"
                if col_name in row:
                    latest_features[feature] = row[col_name]

            # Format market features as text using latest values
            market_text = (
                f"Market Analysis: "
                f"Price: ${latest_features.get('price', 0):.6f}, "
                f"Volatility: {latest_features.get('volatility', 0)*100:.2f}%, "
                f"Volume: {latest_features.get('volume', 0):.4f} ETH, "
                f"Volume MA: {latest_features.get('volume_ma', 0):.4f}, "
                f"Liquidity Impact: {latest_features.get('liquidity_impact', 0):.2f}, "
                f"Price Change: {latest_features.get('price_change', 0)*100:.2f}%. "
            )

            # Format trading decision as text
            decision_text = (
                f"Trading Decision: {row['side_raw']} "
                f"Size: {row['size']:.3f} "
                f"Slippage: {row['slippage_bps']:.0f}bps "
                f"Deadline: {row['deadline_s']:.0f}s"
            )

            # Combine into training text
            full_text = f"{market_text}{decision_text}"
            texts.append(full_text)

        return texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get tokenized text for training."""
        text = self.texts[idx]

        # Tokenize with padding and truncation
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.config.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze()  # For causal LM loss
        }


class LoRAROOKTrainer:
    """LoRA fine-tuning trainer for ROOK agents."""

    def __init__(self, config: LoRATrainingConfig, data_dir: str, output_dir: str):
        self.config = config
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load tokenizer and model
        logger.info("Loading parent model: {}", config.parent_model)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.parent_model,
            cache_dir="training/models/parent"
        )

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            config.parent_model,
            cache_dir="training/models/parent",
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )

        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            bias="none"
        )

        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        # Create datasets
        self.train_dataset = TradingTextDataset(
            str(self.data_dir / "train.csv"),
            self.tokenizer,
            config
        )

        self.val_dataset = TradingTextDataset(
            str(self.data_dir / "val.csv"),
            self.tokenizer,
            config
        )

        logger.info("LoRA trainer initialized: {} train, {} val samples",
                   len(self.train_dataset), len(self.val_dataset))

    def train(self):
        """Train the LoRA adapter."""
        logger.info("Starting LoRA fine-tuning for {} epochs", self.config.num_epochs)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            overwrite_output_dir=True,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            warmup_steps=self.config.warmup_steps,
            logging_steps=50,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,  # Disable wandb
            dataloader_pin_memory=False,
            fp16=torch.cuda.is_available(),
            remove_unused_columns=False,
            push_to_hub=False,
        )

        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
            return_tensors="pt"
        )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        # Train
        trainer.train()

        # Save the LoRA adapter
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        # Save training config
        config_path = self.output_dir / "lora_config.json"
        config_dict = {
            "parent_model": self.config.parent_model,
            "lora_r": self.config.lora_r,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
            "target_modules": self.config.target_modules,
            "max_length": self.config.max_length,
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
            "num_epochs": self.config.num_epochs
        }

        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        logger.info("LoRA training complete! Adapter saved to {}", self.output_dir)

    def test_inference(self):
        """Test the trained LoRA adapter with a sample trading scenario."""
        logger.info("Testing LoRA adapter inference...")

        # Sample market condition
        test_prompt = (
            "Market Analysis: "
            "Price: $3500.00, "
            "Volatility: 2.50%, "
            "Volume: 0.085 ETH, "
            "Volume MA: 0.080, "
            "Liquidity Impact: 1.20, "
            "Price Change: +0.15%. "
            "Trading Decision:"
        )

        # Tokenize
        inputs = self.tokenizer(test_prompt, return_tensors="pt")

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        decision = generated_text[len(test_prompt):].strip()

        logger.info("Sample prediction: '{}'", decision)
        return decision


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train ROOK LoRA adapter")
    parser.add_argument("--data-dir", required=True, help="Directory with train/val/test CSV files")
    parser.add_argument("--output-dir", required=True, help="Output directory for LoRA adapter")
    parser.add_argument("--parent-model", default="microsoft/DialoGPT-small", help="Parent model name")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")

    args = parser.parse_args()

    # Create config
    config = LoRATrainingConfig(
        parent_model=args.parent_model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_length=args.max_length
    )

    # Train LoRA adapter
    trainer = LoRAROOKTrainer(config, args.data_dir, args.output_dir)
    trainer.train()

    # Test inference
    trainer.test_inference()

    print(f"\nLoRA training complete!")
    print(f"Adapter saved to: {args.output_dir}")
    print(f"Parent model: {args.parent_model}")
    print(f"LoRA parameters: r={args.lora_r}, alpha={args.lora_alpha}")


if __name__ == "__main__":
    main()