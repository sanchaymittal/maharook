#!/usr/bin/env python3
"""
ROOK Training Script - Standalone Model Training
------------------------------------------------
Trains standalone PyTorch models for trading decisions using supervised learning.
Implements multi-task learning for trading actions: side, size, slippage, deadline.

NOTE: This is NOT LoRA fine-tuning - it's training a standalone LSTM-based model from scratch.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader, Dataset

# Optional imports for different model architectures
try:
    import peft  # noqa: F401
    import transformers  # noqa: F401
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("Transformers/PEFT not available - using basic PyTorch implementation")
    TRANSFORMERS_AVAILABLE = False


@dataclass
class TrainingConfig:
    """Training configuration for standalone ROOK model."""
    model_name: str = "standalone_rook_lstm"  # Descriptive name for our architecture
    batch_size: int = 32
    learning_rate: float = 5e-4
    num_epochs: int = 10
    seq_len: int = 30
    hidden_dim: int = 256  # LSTM hidden dimension
    intermediate_dim: int = 128  # Fusion layer dimension
    dropout: float = 0.1
    warmup_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 100
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01


class ROOKTrainingDataset(Dataset):
    """PyTorch dataset for ROOK training data."""

    def __init__(self, data_path: str, metadata_path: str, normalization_path: str):
        """Initialize dataset from CSV files."""
        self.metadata = self._load_metadata(metadata_path)
        self.normalization = self._load_normalization(normalization_path)

        # Load data
        self.data = pd.read_csv(data_path)

        # Extract features and targets
        self._extract_features_and_targets()

        logger.info("Dataset loaded: {} samples", len(self.data))

    def _load_metadata(self, path: str) -> dict[str, Any]:
        """Load dataset metadata."""
        with open(path) as f:
            return json.load(f)

    def _load_normalization(self, path: str) -> dict[str, Any]:
        """Load normalization parameters."""
        with open(path) as f:
            return json.load(f)

    def _extract_features_and_targets(self):
        """Extract feature arrays and targets from DataFrame."""
        seq_len = self.metadata['seq_len']
        n_core_features = self.metadata['n_core_features']

        # Core feature sequences (flattened in CSV)
        # Build exact column names: feature_name_timestep
        core_cols = []
        for feat in self.metadata['core_feature_names']:
            for i in range(seq_len):
                core_cols.append(f"{feat}_{i}")

        # Ensure all expected columns exist
        missing_cols = [col for col in core_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing core feature columns: {missing_cols[:5]}...")

        core_data = self.data[core_cols].values

        # Reshape: (n_samples, seq_len * n_features) -> (n_samples, seq_len, n_features)
        n_samples = core_data.shape[0]
        self.X_core = core_data.reshape(n_samples, seq_len, n_core_features)

        # Condition features
        condition_cols = self.metadata['condition_feature_names']
        self.X_conditions = self.data[condition_cols].values

        # Targets
        self.y_side = self.data['side'].values
        self.y_size = self.data['size'].values
        self.y_slippage = self.data['slippage_bps'].values
        self.y_deadline = self.data['deadline_s'].values

        logger.info("Features extracted: core={}, conditions={}",
                   self.X_core.shape, self.X_conditions.shape)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single training sample."""
        return {
            'core_features': torch.FloatTensor(self.X_core[idx]),
            'conditions': torch.FloatTensor(self.X_conditions[idx]),
            'side': torch.LongTensor([self.y_side[idx]]),
            'size': torch.FloatTensor([self.y_size[idx]]),
            'slippage': torch.FloatTensor([self.y_slippage[idx]]),
            'deadline': torch.FloatTensor([self.y_deadline[idx]]),
        }


class ROOKModelPyTorch(nn.Module):
    """Standalone PyTorch ROOK model - LSTM-based trading decision model."""

    def __init__(self, config: TrainingConfig, metadata: dict[str, Any]):
        super().__init__()
        self.config = config
        self.metadata = metadata

        # Input dimensions
        self.core_dim = metadata['n_core_features']
        self.condition_dim = metadata['n_condition_features']
        self.seq_len = metadata['seq_len']

        # Hidden dimensions from config
        self.hidden_dim = config.hidden_dim
        self.intermediate_dim = config.intermediate_dim

        # Core sequence encoder (LSTM)
        self.core_encoder = nn.LSTM(
            input_size=self.core_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=config.dropout
        )

        # Condition encoder
        self.condition_encoder = nn.Sequential(
            nn.Linear(self.condition_dim, self.intermediate_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.intermediate_dim, self.intermediate_dim)
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim + self.intermediate_dim, self.intermediate_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )

        # Output heads
        self.side_head = nn.Linear(self.intermediate_dim, 3)  # BUY, SELL, HOLD
        self.size_head = nn.Linear(self.intermediate_dim, 1)
        self.slippage_head = nn.Linear(self.intermediate_dim, 1)
        self.deadline_head = nn.Linear(self.intermediate_dim, 1)

        # Loss functions
        self.side_loss_fn = nn.CrossEntropyLoss()
        self.regression_loss_fn = nn.HuberLoss()

    def forward(self, core_features: torch.Tensor, conditions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass."""
        # Encode core sequences
        core_output, (hidden, _) = self.core_encoder(core_features)
        # Use last hidden state
        core_repr = hidden[-1]  # Shape: (batch_size, hidden_dim)

        # Encode conditions
        condition_repr = self.condition_encoder(conditions)

        # Fuse representations
        fused = torch.cat([core_repr, condition_repr], dim=1)
        fused_repr = self.fusion(fused)

        # Generate outputs
        outputs = {
            'side_logits': self.side_head(fused_repr),
            'size': torch.sigmoid(self.size_head(fused_repr)),  # Bound to [0,1]
            'slippage': torch.relu(self.slippage_head(fused_repr)),  # Positive values
            'deadline': torch.relu(self.deadline_head(fused_repr))   # Positive values
        }

        return outputs

    def compute_loss(self, outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute multi-task loss."""
        side_loss = self.side_loss_fn(outputs['side_logits'], targets['side'].squeeze())
        size_loss = self.regression_loss_fn(outputs['size'], targets['size'])
        slippage_loss = self.regression_loss_fn(outputs['slippage'], targets['slippage'])
        deadline_loss = self.regression_loss_fn(outputs['deadline'], targets['deadline'])

        # Weighted combination
        total_loss = (
            2.0 * side_loss +      # Higher weight for side prediction
            1.0 * size_loss +
            1.0 * slippage_loss +
            0.5 * deadline_loss    # Lower weight for deadline
        )

        return total_loss, {
            'side_loss': side_loss.item(),
            'size_loss': size_loss.item(),
            'slippage_loss': slippage_loss.item(),
            'deadline_loss': deadline_loss.item(),
            'total_loss': total_loss.item()
        }


class ROOKTrainer:
    """Training orchestrator for ROOK models."""

    def __init__(self, config: TrainingConfig, data_dir: str, output_dir: str):
        self.config = config
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load metadata
        metadata_path = self.data_dir / "metadata.json"
        normalization_path = self.data_dir / "normalization.json"

        with open(metadata_path) as f:
            self.metadata = json.load(f)

        # Create datasets
        self.train_dataset = ROOKTrainingDataset(
            str(self.data_dir / "train.csv"),
            str(metadata_path),
            str(normalization_path)
        )

        self.val_dataset = ROOKTrainingDataset(
            str(self.data_dir / "val.csv"),
            str(metadata_path),
            str(normalization_path)
        )

        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=2
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=2
        )

        # Initialize model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: {}", self.device)

        self.model = ROOKModelPyTorch(config, self.metadata).to(self.device)

        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs
        )

        logger.info("Trainer initialized: {} train samples, {} val samples",
                   len(self.train_dataset), len(self.val_dataset))

    def train_epoch(self) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_losses = {'total_loss': 0, 'side_loss': 0, 'size_loss': 0, 'slippage_loss': 0, 'deadline_loss': 0}
        num_batches = 0

        for batch in self.train_loader:
            # Move to device
            core_features = batch['core_features'].to(self.device)
            conditions = batch['conditions'].to(self.device)
            targets = {
                'side': batch['side'].to(self.device),
                'size': batch['size'].to(self.device),
                'slippage': batch['slippage'].to(self.device),
                'deadline': batch['deadline'].to(self.device)
            }

            # Forward pass
            outputs = self.model(core_features, conditions)
            loss, loss_dict = self.model.compute_loss(outputs, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            # Accumulate losses
            for key, value in loss_dict.items():
                total_losses[key] += value
            num_batches += 1

        # Average losses
        avg_losses = {key: value / num_batches for key, value in total_losses.items()}
        return avg_losses

    def validate(self) -> dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_losses = {'total_loss': 0, 'side_loss': 0, 'size_loss': 0, 'slippage_loss': 0, 'deadline_loss': 0}
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                # Move to device
                core_features = batch['core_features'].to(self.device)
                conditions = batch['conditions'].to(self.device)
                targets = {
                    'side': batch['side'].to(self.device),
                    'size': batch['size'].to(self.device),
                    'slippage': batch['slippage'].to(self.device),
                    'deadline': batch['deadline'].to(self.device)
                }

                # Forward pass
                outputs = self.model(core_features, conditions)
                loss, loss_dict = self.model.compute_loss(outputs, targets)

                # Accumulate losses
                for key, value in loss_dict.items():
                    total_losses[key] += value
                num_batches += 1

        # Average losses
        avg_losses = {key: value / num_batches for key, value in total_losses.items()}
        return avg_losses

    def train(self):
        """Train the model."""
        logger.info("Starting training for {} epochs", self.config.num_epochs)

        best_val_loss = float('inf')
        training_history = []

        for epoch in range(self.config.num_epochs):
            # Train
            train_losses = self.train_epoch()

            # Validate
            val_losses = self.validate()

            # Step scheduler
            self.scheduler.step()

            # Log progress
            logger.info(
                "Epoch {}/{}: train_loss={:.4f}, val_loss={:.4f}, side_acc={:.4f}",
                epoch + 1, self.config.num_epochs,
                train_losses['total_loss'], val_losses['total_loss'],
                1.0 - val_losses['side_loss']  # Rough accuracy approximation
            )

            # Save checkpoint if best
            if val_losses['total_loss'] < best_val_loss:
                best_val_loss = val_losses['total_loss']
                self.save_checkpoint(epoch, val_losses)

            # Save training history
            training_history.append({
                'epoch': epoch + 1,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'lr': self.optimizer.param_groups[0]['lr']
            })

        # Save final model and history
        self.save_final_model(training_history)
        logger.info("Training complete! Best val loss: {:.4f}", best_val_loss)

    def save_checkpoint(self, epoch: int, val_losses: dict[str, float]):
        """Save model checkpoint."""
        checkpoint_path = self.output_dir / "best_model.pt"

        # Save only the model state dict to avoid serialization issues
        torch.save(self.model.state_dict(), checkpoint_path)

        # Save metadata separately as JSON
        metadata_path = self.output_dir / "checkpoint_info.json"
        checkpoint_info = {
            'epoch': epoch,
            'val_losses': val_losses,
            'model_config': {
                'seq_len': self.metadata['seq_len'],
                'n_core_features': self.metadata['n_core_features'],
                'n_condition_features': self.metadata['n_condition_features'],
                'side_classes': self.metadata.get('side_classes', ['BUY', 'HOLD', 'SELL'])
            }
        }

        with open(metadata_path, 'w') as f:
            json.dump(checkpoint_info, f, indent=2)

        logger.info("Saved checkpoint at epoch {} with val_loss={:.4f}",
                   epoch + 1, val_losses['total_loss'])

    def save_final_model(self, training_history: list) -> None:
        """Save final model and training artifacts."""
        # Save model
        model_path = self.output_dir / "final_model.pt"
        torch.save(self.model.state_dict(), model_path)

        # Save training history
        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)

        # Save config
        config_path = self.output_dir / "config.json"
        config_dict = {
            'model_name': self.config.model_name,
            'batch_size': self.config.batch_size,
            'learning_rate': self.config.learning_rate,
            'num_epochs': self.config.num_epochs,
            'seq_len': self.config.seq_len,
            'hidden_dim': self.config.hidden_dim,
            'intermediate_dim': self.config.intermediate_dim,
            'dropout': self.config.dropout
        }
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        logger.info("Saved final model and training artifacts to {}", self.output_dir)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train ROOK model")
    parser.add_argument("--data-dir", required=True, help="Directory with train/val/test splits")
    parser.add_argument("--output-dir", required=True, help="Output directory for model")
    parser.add_argument("--model-name", default="microsoft/DialoGPT-small", help="Base model name")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--hidden-dim", type=int, default=256, help="LSTM hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    args = parser.parse_args()

    # Create config
    config = TrainingConfig(
        model_name=args.model_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    )

    # Train model
    trainer = ROOKTrainer(config, args.data_dir, args.output_dir)
    trainer.train()

    print("\nTraining complete!")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
