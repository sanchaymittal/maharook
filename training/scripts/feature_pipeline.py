#!/usr/bin/env python3
"""
Feature Pipeline for ROOK Training
---------------------------------
Processes raw Uniswap V4 data into trading-ready features.
Generates core_features.csv and core_conditions.csv for model training.
"""

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger


class FeaturePipeline:
    """Feature extraction pipeline for Uniswap V4 trading data."""

    def __init__(self, raw_data_path: str, output_dir: str):
        """Initialize pipeline with data paths."""
        self.raw_data_path = Path(raw_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_raw_data(self) -> pd.DataFrame:
        """Load and preprocess raw Uniswap data."""
        logger.info("Loading raw data from {}", self.raw_data_path)

        df = pd.read_csv(self.raw_data_path)

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

        # Convert numeric columns to proper types
        numeric_columns = ['sqrt_price_x96', 'amount0', 'amount1', 'liquidity', 'tick']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Remove any rows with invalid numeric data
        df = df.dropna(subset=numeric_columns)

        # Calculate price from sqrt_price_x96
        # price = (sqrt_price_x96 / 2^96)^2
        df['price'] = (df['sqrt_price_x96'] / (2**96)) ** 2

        # Convert amounts to human-readable units
        # amount0 is ETH (18 decimals), amount1 is USDC (6 decimals)
        df['amount0_eth'] = df['amount0'] / 1e18
        df['amount1_usdc'] = df['amount1'] / 1e6

        # Calculate volume in ETH terms
        df['volume'] = abs(df['amount0_eth'])

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        logger.info("Loaded {} records from {} to {}",
                   len(df), df['timestamp'].min(), df['timestamp'].max())

        return df

    def extract_core_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract core time-series features for each transaction."""
        logger.info("Extracting core features...")

        features = []

        for i in range(len(df)):
            row = df.iloc[i]

            # Look back window for calculations
            lookback_end = i
            lookback_start = max(0, i - 100)  # Last 100 transactions

            window_df = df.iloc[lookback_start:lookback_end] if i > 0 else df.iloc[:1]

            if len(window_df) == 0:
                window_df = df.iloc[:1]

            # Time-based features
            if i > 0:
                time_diff = (row['timestamp'] - df.iloc[i-1]['timestamp']).total_seconds()
                price_change = (row['price'] - df.iloc[i-1]['price']) / df.iloc[i-1]['price']
            else:
                time_diff = 0
                price_change = 0

            # Rolling volatility (price changes)
            if len(window_df) > 1:
                price_changes = window_df['price'].pct_change().dropna()
                volatility = price_changes.std() if len(price_changes) > 0 else 0
            else:
                volatility = 0

            # Volume moving average
            volume_ma = window_df['volume'].mean()

            # Tick change
            tick_change = row['tick'] - df.iloc[i - 1]['tick'] if i > 0 else 0

            # Liquidity impact (volume relative to liquidity)
            liquidity_impact = row['volume'] / (row['liquidity'] / 1e18) if row['liquidity'] > 0 else 0

            feature_row = {
                'timestamp': row['timestamp'],
                'price': row['price'],
                'amount0_eth': row['amount0_eth'],
                'amount1_usdc': row['amount1_usdc'],
                'volume': row['volume'],
                'volatility': volatility,
                'volume_ma': volume_ma,
                'tick_change': tick_change,
                'time_diff': time_diff,
                'liquidity_impact': liquidity_impact,
                'price_change': price_change,
                'tick': row['tick'],
                'liquidity': row['liquidity'],
            }

            features.append(feature_row)

        features_df = pd.DataFrame(features)

        # Forward fill any NaN values
        features_df = features_df.fillna(method='ffill').fillna(0)

        logger.info("Extracted {} core features", len(features_df))
        return features_df

    def extract_conditions(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Extract market condition features (aggregated stats)."""
        logger.info("Extracting market conditions...")

        # Calculate rolling statistics over different windows
        windows = [10, 50, 100]  # Number of transactions
        conditions = {}

        for window in windows:
            recent_data = features_df.tail(window) if len(features_df) >= window else features_df

            conditions.update({
                f'avg_price_{window}': recent_data['price'].mean(),
                f'price_volatility_{window}': recent_data['price'].std(),
                f'avg_volume_{window}': recent_data['volume'].mean(),
                f'avg_tick_change_{window}': recent_data['tick_change'].mean(),
                f'avg_time_diff_{window}': recent_data['time_diff'].mean(),
                f'liquidity_impact_avg_{window}': recent_data['liquidity_impact'].mean(),
            })

        # Overall market conditions
        conditions.update({
            'total_transactions': len(features_df),
            'price_trend': features_df['price_change'].tail(20).mean(),  # Recent trend
            'volume_trend': features_df['volume'].tail(20).mean() / features_df['volume'].mean() if len(features_df) > 20 else 1.0,
            'current_price': features_df['price'].iloc[-1],
            'current_liquidity': features_df['liquidity'].iloc[-1],
            'timestamp': features_df['timestamp'].iloc[-1],
        })

        # Convert to DataFrame
        conditions_df = pd.DataFrame([conditions])

        logger.info("Extracted market conditions with {} features", len(conditions_df.columns))
        return conditions_df

    def save_features(self, core_features: pd.DataFrame, conditions: pd.DataFrame):
        """Save features to CSV files."""
        core_path = self.output_dir / "core_features.csv"
        conditions_path = self.output_dir / "core_conditions.csv"

        logger.info("Saving core features to {}", core_path)
        core_features.to_csv(core_path, index=False)

        logger.info("Saving conditions to {}", conditions_path)
        conditions.to_csv(conditions_path, index=False)

        # Save feature info
        info_path = self.output_dir / "feature_info.txt"
        with open(info_path, 'w') as f:
            f.write("Feature Pipeline Output\n")
            f.write("======================\n\n")
            f.write(f"Core features: {len(core_features)} rows, {len(core_features.columns)} columns\n")
            f.write(f"Conditions: {len(conditions)} rows, {len(conditions.columns)} columns\n\n")

            f.write("Core feature columns:\n")
            for col in core_features.columns:
                f.write(f"  - {col}\n")

            f.write("\nCondition columns:\n")
            for col in conditions.columns:
                f.write(f"  - {col}\n")

        logger.info("Feature extraction complete!")

    def run(self):
        """Run the complete feature pipeline."""
        # Load raw data
        raw_df = self.load_raw_data()

        # Extract features
        core_features = self.extract_core_features(raw_df)
        conditions = self.extract_conditions(core_features)

        # Save results
        self.save_features(core_features, conditions)

        return core_features, conditions


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Extract features from Uniswap V4 data")
    parser.add_argument("--input", "-i", required=True, help="Path to raw CSV file")
    parser.add_argument("--output", "-o", required=True, help="Output directory for features")

    args = parser.parse_args()

    # Run pipeline
    pipeline = FeaturePipeline(args.input, args.output)
    core_features, conditions = pipeline.run()

    print("\nFeature extraction complete!")
    print(f"Core features: {len(core_features)} rows")
    print(f"Conditions: {len(conditions)} rows")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
