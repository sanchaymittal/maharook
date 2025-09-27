#!/usr/bin/env python3
"""
MLX-LM LoRA Training Script for Fin-R1
--------------------------------------
Fine-tunes Fin-R1 (Qwen2-based) model using MLX-LM LoRA adapters for trading decisions.
Optimized for Apple Silicon with memory-efficient training.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from loguru import logger
import mlx.core as mx
import mlx.optimizers as optim
from mlx_lm import load, generate
from mlx_lm.tuner import train, TrainingArgs, linear_to_lora_layers
from mlx_lm.tuner.datasets import load_dataset


@dataclass
class MLXLoRAConfig:
    """MLX LoRA training configuration."""
    model_path: str = "training/models/qwen2-1.5b-mlx"
    max_length: int = 512
    batch_size: int = 4
    learning_rate: float = 1e-4
    num_epochs: int = 3
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = None
    warmup_steps: int = 100
    save_steps: int = 500
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0

    def __post_init__(self):
        if self.target_modules is None:
            # Target attention and MLP layers for LoRA
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class TradingDataProcessor:
    """Processes trading data for MLX-LM text training."""

    def __init__(self, data_path: str, config: MLXLoRAConfig):
        self.data_path = data_path
        self.config = config
        self.data = pd.read_csv(data_path)

        # Load metadata to understand structure
        metadata_path = Path(data_path).parent / "metadata.json"
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        logger.info("Loaded {} trading samples for MLX training", len(self.data))

    def create_training_texts(self) -> List[str]:
        """Convert trading features and labels to natural language text."""
        texts = []
        seq_len = self.metadata['seq_len']
        core_features = self.metadata['core_feature_names']

        for _, row in self.data.iterrows():
            # Extract latest values from sequence (last timestep)
            latest_features = {}
            for feature in core_features:
                col_name = f"{feature}_{seq_len-1}"
                if col_name in row:
                    latest_features[feature] = row[col_name]

            # Format as natural language instruction
            market_analysis = (
                f"Analyze this market condition and provide a trading decision:\n"
                f"Price: ${latest_features.get('price', 0):.4f}\n"
                f"Volatility: {latest_features.get('volatility', 0)*100:.2f}%\n"
                f"Volume: {latest_features.get('volume', 0):.4f} ETH\n"
                f"Volume MA: {latest_features.get('volume_ma', 0):.4f}\n"
                f"Liquidity Impact: {latest_features.get('liquidity_impact', 0):.2f}\n"
                f"Price Change: {latest_features.get('price_change', 0)*100:.2f}%\n\n"
            )

            # Format expected response
            trading_decision = (
                f"Trading Decision: {row['side_raw']}\n"
                f"Position Size: {row['size']:.3f} ETH\n"
                f"Slippage Tolerance: {row['slippage_bps']:.0f} bps\n"
                f"Execution Deadline: {row['deadline_s']:.0f} seconds"
            )

            # Combine as instruction-response pair
            full_text = f"{market_analysis}{trading_decision}"
            texts.append(full_text)

        return texts

    def create_jsonl_dataset(self, output_path: str):
        """Create JSONL dataset for MLX-LM training."""
        texts = self.create_training_texts()

        with open(output_path, 'w') as f:
            for text in texts:
                # Format as instruction-following dataset
                entry = {
                    "text": text
                }
                f.write(json.dumps(entry) + '\n')

        logger.info("Created JSONL dataset with {} samples at {}", len(texts), output_path)
        return len(texts)


class MLXLoRATrainer:
    """MLX-LM LoRA trainer for Fin-R1 trading agent."""

    def __init__(self, config: MLXLoRAConfig, data_dir: str, output_dir: str):
        self.config = config
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load MLX model
        logger.info("Loading MLX model from {}", config.model_path)
        self.model, self.tokenizer = load(config.model_path)

        logger.info("MLX LoRA trainer initialized for Fin-R1")

    def prepare_data(self):
        """Prepare training data in JSONL format."""
        # Process training data
        train_processor = TradingDataProcessor(
            str(self.data_dir / "train.csv"),
            self.config
        )
        train_samples = train_processor.create_jsonl_dataset(
            str(self.output_dir / "train.jsonl")
        )

        # Process validation data
        val_processor = TradingDataProcessor(
            str(self.data_dir / "val.csv"),
            self.config
        )
        val_samples = val_processor.create_jsonl_dataset(
            str(self.output_dir / "val.jsonl")
        )

        logger.info("Prepared {} train, {} val samples", train_samples, val_samples)
        return train_samples, val_samples

    def train(self):
        """Train LoRA adapter using MLX-LM."""
        logger.info("Starting MLX LoRA fine-tuning for {} epochs", self.config.num_epochs)

        # Prepare data
        train_samples, val_samples = self.prepare_data()

        # Apply LoRA to model
        logger.info("Converting model to LoRA")
        linear_to_lora_layers(
            model=self.model,
            num_layers=16,  # Convert last 16 layers to LoRA
            config={
                "rank": self.config.lora_rank,
                "alpha": self.config.lora_alpha,
                "dropout": self.config.lora_dropout,
                "scale": self.config.lora_alpha / self.config.lora_rank
            }
        )

        # Setup optimizer
        optimizer = optim.AdamW(learning_rate=self.config.learning_rate)

        # Create args object for dataset loading
        class DatasetArgs:
            def __init__(self, data_path, max_seq_length):
                self.data = data_path
                self.max_seq_length = max_seq_length
                self.train_test_split = 0.9
                self.hf_dataset = False

        # Load datasets
        train_args = DatasetArgs(str(self.output_dir / "train.jsonl"), self.config.max_length)
        val_args = DatasetArgs(str(self.output_dir / "val.jsonl"), self.config.max_length)

        train_dataset, _, _ = load_dataset(train_args, self.tokenizer)
        val_dataset, _, _ = load_dataset(val_args, self.tokenizer)

        # Training arguments
        training_args = TrainingArgs(
            batch_size=self.config.batch_size,
            iters=train_samples // self.config.batch_size * self.config.num_epochs,
            val_batches=val_samples // self.config.batch_size,
            steps_per_report=50,
            steps_per_eval=self.config.save_steps,
            steps_per_save=self.config.save_steps,
            max_seq_length=self.config.max_length,
            adapter_file=str(self.output_dir / "adapters.safetensors")
        )

        # Run LoRA training
        try:
            logger.info("Starting LoRA training with MLX...")

            train(
                model=self.model,
                optimizer=optimizer,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                args=training_args
            )

            logger.info("LoRA training completed successfully!")

        except Exception as e:
            logger.error("Training failed: {}", e)
            raise

        # Save configuration
        self._save_config()

    def _save_config(self):
        """Save training configuration."""
        config_dict = {
            "model_path": self.config.model_path,
            "lora_rank": self.config.lora_rank,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
            "target_modules": self.config.target_modules,
            "max_length": self.config.max_length,
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
            "num_epochs": self.config.num_epochs
        }

        config_path = self.output_dir / "mlx_lora_config.json"
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        logger.info("Training config saved to {}", config_path)

    def test_inference(self):
        """Test the trained LoRA adapter."""
        logger.info("Testing MLX LoRA adapter inference...")

        # Sample trading scenario
        test_prompt = (
            "Analyze this market condition and provide a trading decision:\n"
            "Price: $3500.0000\n"
            "Volatility: 2.50%\n"
            "Volume: 0.0850 ETH\n"
            "Volume MA: 0.0800\n"
            "Liquidity Impact: 1.20\n"
            "Price Change: +0.15%\n\n"
        )

        # Generate response
        try:
            response = generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=test_prompt,
                max_tokens=100,
                temperature=0.7
            )

            logger.info("Sample prediction: '{}'", response)
            return response

        except Exception as e:
            logger.error("Inference test failed: {}", e)
            return None


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train Fin-R1 MLX LoRA adapter")
    parser.add_argument("--data-dir", required=True, help="Directory with train/val CSV files")
    parser.add_argument("--output-dir", required=True, help="Output directory for LoRA adapter")
    parser.add_argument("--model-path", default="training/models/qwen2-1.5b-mlx", help="MLX model path")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")

    args = parser.parse_args()

    # Create config
    config = MLXLoRAConfig(
        model_path=args.model_path,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        max_length=args.max_length
    )

    # Train LoRA adapter
    trainer = MLXLoRATrainer(config, args.data_dir, args.output_dir)
    trainer.train()

    # Test inference
    trainer.test_inference()

    print(f"\nMLX LoRA training complete!")
    print(f"Adapter saved to: {args.output_dir}")
    print(f"Base model: {args.model_path}")
    print(f"LoRA parameters: rank={args.lora_rank}, alpha={args.lora_alpha}")


if __name__ == "__main__":
    main()