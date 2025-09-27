#!/usr/bin/env python3
"""
Dataset Creation for ROOK Training
----------------------------------
Creates train/val/test splits with sequence data for time-series model training.
Handles feature normalization and sequence generation for transformer models.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import LabelEncoder, StandardScaler


class ROOKDataset:
    """Dataset preparation for ROOK model training."""

    def __init__(self,
                 labeled_data_path: str,
                 conditions_path: str,
                 output_dir: str,
                 seq_len: int = 50,
                 test_split: float = 0.15,
                 val_split: float = 0.15):
        """Initialize dataset creator.

        Args:
            labeled_data_path: Path to labeled_dataset.csv
            conditions_path: Path to core_conditions.csv
            output_dir: Output directory for train/val/test files
            seq_len: Sequence length for time-series features
            test_split: Fraction for test set
            val_split: Fraction for validation set
        """
        self.labeled_data_path = Path(labeled_data_path)
        self.conditions_path = Path(conditions_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.seq_len = seq_len
        self.test_split = test_split
        self.val_split = val_split

        # Feature scalers
        self.core_scaler = StandardScaler()
        self.conditions_scaler = StandardScaler()
        self.side_encoder = LabelEncoder()

        logger.info("Dataset creator initialized: seq_len={}, splits={:.1%}/{:.1%}/{:.1%}",
                   seq_len, 1-test_split-val_split, val_split, test_split)

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load labeled dataset and conditions."""
        logger.info("Loading labeled data from {}", self.labeled_data_path)
        labeled_df = pd.read_csv(self.labeled_data_path, parse_dates=['timestamp'])

        logger.info("Loading conditions from {}", self.conditions_path)
        conditions_df = pd.read_csv(self.conditions_path, parse_dates=['timestamp'])

        logger.info("Loaded {} labeled samples and {} condition records",
                   len(labeled_df), len(conditions_df))

        return labeled_df, conditions_df

    def prepare_features(self, labeled_df: pd.DataFrame, conditions_df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Prepare and normalize features for training."""
        logger.info("Preparing features...")

        # Core time-series features (will be sequenced)
        core_feature_cols = [
            'price', 'volatility', 'volume', 'volume_ma',
            'liquidity_impact', 'price_change'
        ]

        # Condition features (single values per sample)
        condition_feature_cols = [col for col in conditions_df.columns
                                 if col not in ['timestamp'] and 'timestamp' not in col.lower()]

        # Extract core features
        core_features = labeled_df[core_feature_cols].values

        # Handle NaN and infinite values
        core_features = np.nan_to_num(core_features, nan=0.0, posinf=1e6, neginf=-1e6)

        # Normalize core features
        core_features_normalized = self.core_scaler.fit_transform(core_features)

        # Extract conditions (broadcast to match labeled data length)
        conditions_values = conditions_df[condition_feature_cols].values
        if len(conditions_values) == 1:
            # Single condition record - broadcast to all samples
            conditions_features = np.tile(conditions_values[0], (len(labeled_df), 1))
        else:
            # Multiple conditions - use as is (assuming alignment)
            conditions_features = conditions_values[:len(labeled_df)]

        # Handle NaN and infinite values in conditions
        conditions_features = np.nan_to_num(conditions_features, nan=0.0, posinf=1e6, neginf=-1e6)

        # Normalize conditions
        conditions_normalized = self.conditions_scaler.fit_transform(conditions_features)

        # Prepare target variables
        sides = labeled_df['side'].values
        sizes = labeled_df['size'].values
        slippage_bps = labeled_df['slippage_bps'].values
        deadline_s = labeled_df['deadline_s'].values

        # Encode categorical side
        sides_encoded = self.side_encoder.fit_transform(sides)

        logger.info("Features prepared: core={}, conditions={}, targets=4",
                   core_features_normalized.shape, conditions_normalized.shape)

        return {
            'core_features': core_features_normalized,
            'conditions': conditions_normalized,
            'sides': sides_encoded,
            'sides_raw': sides,
            'sizes': sizes,
            'slippage_bps': slippage_bps,
            'deadline_s': deadline_s,
            'timestamps': labeled_df['timestamp'].values,
            'core_feature_names': core_feature_cols,
            'condition_feature_names': condition_feature_cols,
        }

    def create_sequences(self, features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Create sequences from time-series features."""
        logger.info("Creating sequences of length {}", self.seq_len)

        core_features = features['core_features']
        conditions = features['conditions']

        n_samples = len(core_features)
        n_sequences = n_samples - self.seq_len + 1

        if n_sequences <= 0:
            raise ValueError(f"Not enough data for sequences: {n_samples} samples, {self.seq_len} seq_len")

        # Create core feature sequences
        core_sequences = np.zeros((n_sequences, self.seq_len, core_features.shape[1]))

        for i in range(n_sequences):
            core_sequences[i] = core_features[i:i + self.seq_len]

        # Conditions and targets correspond to the last timestamp in each sequence
        sequence_conditions = conditions[self.seq_len - 1:]
        sequence_sides = features['sides'][self.seq_len - 1:]
        sequence_sides_raw = features['sides_raw'][self.seq_len - 1:]
        sequence_sizes = features['sizes'][self.seq_len - 1:]
        sequence_slippage = features['slippage_bps'][self.seq_len - 1:]
        sequence_deadline = features['deadline_s'][self.seq_len - 1:]
        sequence_timestamps = features['timestamps'][self.seq_len - 1:]

        logger.info("Created {} sequences from {} samples", n_sequences, n_samples)

        return {
            'X_core': core_sequences,
            'X_conditions': sequence_conditions,
            'y_sides': sequence_sides,
            'y_sides_raw': sequence_sides_raw,
            'y_sizes': sequence_sizes,
            'y_slippage': sequence_slippage,
            'y_deadline': sequence_deadline,
            'timestamps': sequence_timestamps,
            'core_feature_names': features['core_feature_names'],
            'condition_feature_names': features['condition_feature_names'],
        }

    def create_splits(self, sequences: dict[str, np.ndarray]) -> tuple[dict, dict, dict]:
        """Create train/validation/test splits using time-based splitting."""
        logger.info("Creating time-based train/val/test splits")

        n_sequences = len(sequences['X_core'])

        # Time-based splits (preserve temporal order)
        test_start = int(n_sequences * (1 - self.test_split))
        val_start = int(n_sequences * (1 - self.test_split - self.val_split))

        # Create splits
        train_data = {}
        val_data = {}
        test_data = {}

        for key, data in sequences.items():
            if isinstance(data, np.ndarray):
                train_data[key] = data[:val_start]
                val_data[key] = data[val_start:test_start]
                test_data[key] = data[test_start:]
            else:
                # Non-array data (like feature names)
                train_data[key] = data
                val_data[key] = data
                test_data[key] = data

        logger.info("Splits created: train={}, val={}, test={}",
                   len(train_data['X_core']), len(val_data['X_core']), len(test_data['X_core']))

        # Log time ranges
        logger.info("Train period: {} to {}",
                   pd.to_datetime(train_data['timestamps'][0]),
                   pd.to_datetime(train_data['timestamps'][-1]))
        logger.info("Val period: {} to {}",
                   pd.to_datetime(val_data['timestamps'][0]),
                   pd.to_datetime(val_data['timestamps'][-1]))
        logger.info("Test period: {} to {}",
                   pd.to_datetime(test_data['timestamps'][0]),
                   pd.to_datetime(test_data['timestamps'][-1]))

        return train_data, val_data, test_data

    def save_splits(self, train_data: dict, val_data: dict, test_data: dict):
        """Save train/val/test splits and metadata."""
        logger.info("Saving dataset splits to {}", self.output_dir)

        # Save data splits
        for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
            # Convert to DataFrames for easier loading

            # Core sequences: flatten for CSV storage
            core_flat = split_data['X_core'].reshape(len(split_data['X_core']), -1)
            core_df = pd.DataFrame(core_flat, columns=[
                f"{feat}_{i}" for feat in split_data['core_feature_names']
                for i in range(self.seq_len)
            ])

            # Conditions
            conditions_df = pd.DataFrame(split_data['X_conditions'],
                                       columns=split_data['condition_feature_names'])

            # Targets
            targets_df = pd.DataFrame({
                'side': split_data['y_sides'],
                'side_raw': split_data['y_sides_raw'],
                'size': split_data['y_sizes'],
                'slippage_bps': split_data['y_slippage'],
                'deadline_s': split_data['y_deadline'],
                'timestamp': split_data['timestamps']
            })

            # Combine all features
            combined_df = pd.concat([core_df, conditions_df, targets_df], axis=1)

            # Save to CSV
            output_path = self.output_dir / f"{split_name}.csv"
            combined_df.to_csv(output_path, index=False)
            logger.info("Saved {} split: {} samples", split_name, len(combined_df))

        # Save metadata and normalization parameters
        metadata = {
            'seq_len': self.seq_len,
            'n_core_features': len(train_data['core_feature_names']),
            'n_condition_features': len(train_data['condition_feature_names']),
            'core_feature_names': train_data['core_feature_names'],
            'condition_feature_names': train_data['condition_feature_names'],
            'side_classes': self.side_encoder.classes_.tolist(),
            'splits': {
                'train_size': len(train_data['X_core']),
                'val_size': len(val_data['X_core']),
                'test_size': len(test_data['X_core']),
            }
        }

        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save normalization parameters
        normalization = {
            'core_scaler_mean': self.core_scaler.mean_.tolist(),
            'core_scaler_scale': self.core_scaler.scale_.tolist(),
            'conditions_scaler_mean': self.conditions_scaler.mean_.tolist(),
            'conditions_scaler_scale': self.conditions_scaler.scale_.tolist(),
        }

        norm_path = self.output_dir / "normalization.json"
        with open(norm_path, 'w') as f:
            json.dump(normalization, f, indent=2)

        logger.info("Saved metadata and normalization parameters")

    def create_dataset(self):
        """Run the complete dataset creation pipeline."""
        # Load data
        labeled_df, conditions_df = self.load_data()

        # Prepare features
        features = self.prepare_features(labeled_df, conditions_df)

        # Create sequences
        sequences = self.create_sequences(features)

        # Create splits
        train_data, val_data, test_data = self.create_splits(sequences)

        # Save everything
        self.save_splits(train_data, val_data, test_data)

        logger.info("Dataset creation complete!")

        return train_data, val_data, test_data


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Create ROOK training dataset")
    parser.add_argument("--labeled-data", required=True, help="Path to labeled_dataset.csv")
    parser.add_argument("--conditions", required=True, help="Path to core_conditions.csv")
    parser.add_argument("--output", required=True, help="Output directory for splits")
    parser.add_argument("--seq-len", type=int, default=50, help="Sequence length")
    parser.add_argument("--test-split", type=float, default=0.15, help="Test split fraction")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation split fraction")

    args = parser.parse_args()

    # Create dataset
    dataset = ROOKDataset(
        labeled_data_path=args.labeled_data,
        conditions_path=args.conditions,
        output_dir=args.output,
        seq_len=args.seq_len,
        test_split=args.test_split,
        val_split=args.val_split
    )

    train_data, val_data, test_data = dataset.create_dataset()

    print("\nDataset creation complete!")
    print(f"Train: {len(train_data['X_core'])} sequences")
    print(f"Val: {len(val_data['X_core'])} sequences")
    print(f"Test: {len(test_data['X_core'])} sequences")
    print(f"Sequence length: {args.seq_len}")
    print(f"Core features: {len(train_data['core_feature_names'])}")
    print(f"Condition features: {len(train_data['condition_feature_names'])}")


if __name__ == "__main__":
    main()
