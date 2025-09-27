#!/usr/bin/env python3
"""
Heuristic Trading Rules for ROOK Training
-----------------------------------------
Implements rule-based trading strategies to generate labeled training data.
Applies Mean Reversion, Breakout, Volatility, and Liquidity Impact rules.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger


@dataclass
class TradingAction:
    """Structured trading action label."""
    side: str  # "BUY", "SELL", "HOLD"
    size: float  # Amount to trade (0-1 fraction of available balance)
    slippage_bps: float  # Maximum slippage in basis points
    deadline_s: int  # Deadline in seconds
    confidence: float = 0.5  # Rule confidence score
    rule_triggered: str = "NONE"  # Which rule triggered this action


class HeuristicLabeler:
    """Generate trading labels using heuristic rules."""

    def __init__(self, config: dict[str, Any] = None):
        """Initialize with trading rule configuration."""
        self.config = config or self._default_config()
        logger.info("Heuristic labeler initialized with config: {}", self.config)

    def _default_config(self) -> dict[str, Any]:
        """Default trading rule configuration."""
        return {
            # Mean Reversion Rules
            "mr_price_deviation_threshold": 0.02,  # 2% price deviation
            "mr_volume_multiplier": 2.0,  # Volume must be 2x average
            "mr_position_size": 0.05,  # 5% of balance

            # Breakout Rules
            "breakout_volatility_threshold": 0.005,  # 0.5% volatility
            "breakout_volume_threshold": 1.5,  # 1.5x average volume
            "breakout_price_momentum": 0.01,  # 1% price movement
            "breakout_position_size": 0.08,  # 8% of balance

            # Volatility Filter
            "max_volatility_threshold": 0.05,  # 5% max volatility for trading
            "min_volatility_threshold": 0.001,  # 0.1% min volatility

            # Liquidity Impact Rules
            "max_liquidity_impact": 0.01,  # 1% max impact
            "min_liquidity_depth": 1000.0,  # $1000 minimum liquidity

            # Position Sizing
            "max_position_size": 0.1,  # 10% max position
            "min_position_size": 0.01,  # 1% min position

            # Risk Management
            "max_slippage_bps": 50,  # 50 bps max slippage
            "default_deadline_s": 300,  # 5 minutes default deadline

            # Market Hours (UTC timestamps)
            "active_hours_start": 8,  # 8 AM UTC
            "active_hours_end": 20,   # 8 PM UTC
        }

    def generate_labels(self, features_df: pd.DataFrame, conditions_df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading action labels for each timestamp."""
        logger.info("Generating heuristic labels for {} timestamps", len(features_df))

        labels = []

        for i, row in features_df.iterrows():
            # Get current market conditions
            conditions = conditions_df.iloc[0].to_dict() if len(conditions_df) > 0 else {}

            # Apply trading rules
            action = self._apply_trading_rules(row, conditions, i, features_df)

            # Convert to label format
            label = {
                'timestamp': row['timestamp'],
                'side': action.side,
                'size': action.size,
                'slippage_bps': action.slippage_bps,
                'deadline_s': action.deadline_s,
                'confidence': action.confidence,
                'rule_triggered': action.rule_triggered,

                # Context features for debugging
                'price': row['price'],
                'volatility': row['volatility'],
                'volume': row['volume'],
                'volume_ma': row['volume_ma'],
                'liquidity_impact': row['liquidity_impact'],
                'price_change': row['price_change'],
            }

            labels.append(label)

        labels_df = pd.DataFrame(labels)

        # Log statistics
        action_counts = labels_df['side'].value_counts()
        logger.info("Label distribution: {}", action_counts.to_dict())
        logger.info("Generated {} trading labels", len(labels_df))

        return labels_df

    def _apply_trading_rules(
        self,
        row: pd.Series,
        conditions: dict[str, Any],
        idx: int,
        features_df: pd.DataFrame
    ) -> TradingAction:
        """Apply heuristic trading rules to generate action."""

        # Default action
        action = TradingAction(
            side="HOLD",
            size=0.0,
            slippage_bps=self.config["max_slippage_bps"],
            deadline_s=self.config["default_deadline_s"]
        )

        # Check market hours (optional filter)
        hour = row['timestamp'].hour
        if not (self.config["active_hours_start"] <= hour <= self.config["active_hours_end"]):
            action.rule_triggered = "MARKET_HOURS"
            return action

        # Volatility filter - reject if too volatile or too quiet
        if (row['volatility'] > self.config["max_volatility_threshold"] or
            row['volatility'] < self.config["min_volatility_threshold"]):
            action.rule_triggered = "VOLATILITY_FILTER"
            return action

        # Liquidity filter
        if row['liquidity_impact'] > self.config["max_liquidity_impact"]:
            action.rule_triggered = "LIQUIDITY_FILTER"
            return action

        # Mean Reversion Strategy
        mr_action = self._mean_reversion_rule(row, conditions, idx, features_df)
        if mr_action.side != "HOLD":
            return mr_action

        # Breakout Strategy
        breakout_action = self._breakout_rule(row, conditions, idx, features_df)
        if breakout_action.side != "HOLD":
            return breakout_action

        # Default to HOLD
        action.rule_triggered = "NO_SIGNAL"
        return action

    def _mean_reversion_rule(
        self,
        row: pd.Series,
        conditions: dict[str, Any],  # noqa: ARG002
        idx: int,
        features_df: pd.DataFrame
    ) -> TradingAction:
        """Mean reversion trading rule."""

        # Need enough history for mean reversion
        if idx < 20:
            return TradingAction(side="HOLD", size=0.0,
                               slippage_bps=self.config["max_slippage_bps"],
                               deadline_s=self.config["default_deadline_s"])

        # Calculate recent price statistics
        lookback_window = min(50, idx)
        recent_prices = features_df.iloc[idx-lookback_window:idx]['price']
        mean_price = recent_prices.mean()

        # Price deviation from mean
        price_deviation = (row['price'] - mean_price) / mean_price

        # Volume confirmation
        volume_ratio = row['volume'] / row['volume_ma'] if row['volume_ma'] > 0 else 0

        # Mean reversion conditions
        mr_threshold = self.config["mr_price_deviation_threshold"]
        mr_volume_mult = self.config["mr_volume_multiplier"]

        # Price oversold (below mean) + high volume = BUY signal
        if (price_deviation < -mr_threshold and
            volume_ratio > mr_volume_mult):

            return TradingAction(
                side="BUY",
                size=self.config["mr_position_size"],
                slippage_bps=self.config["max_slippage_bps"],
                deadline_s=self.config["default_deadline_s"],
                confidence=min(0.8, abs(price_deviation) * 10 + volume_ratio * 0.1),
                rule_triggered="MEAN_REVERSION_BUY"
            )

        # Price overbought (above mean) + high volume = SELL signal
        elif (price_deviation > mr_threshold and
              volume_ratio > mr_volume_mult):

            return TradingAction(
                side="SELL",
                size=self.config["mr_position_size"],
                slippage_bps=self.config["max_slippage_bps"],
                deadline_s=self.config["default_deadline_s"],
                confidence=min(0.8, abs(price_deviation) * 10 + volume_ratio * 0.1),
                rule_triggered="MEAN_REVERSION_SELL"
            )

        return TradingAction(side="HOLD", size=0.0,
                           slippage_bps=self.config["max_slippage_bps"],
                           deadline_s=self.config["default_deadline_s"])

    def _breakout_rule(
        self,
        row: pd.Series,
        conditions: dict[str, Any],  # noqa: ARG002
        idx: int,
        features_df: pd.DataFrame
    ) -> TradingAction:
        """Momentum/breakout trading rule."""

        # Need recent price momentum
        if idx < 10:
            return TradingAction(side="HOLD", size=0.0,
                               slippage_bps=self.config["max_slippage_bps"],
                               deadline_s=self.config["default_deadline_s"])

        # Calculate momentum indicators
        recent_returns = features_df.iloc[idx-10:idx]['price_change']
        momentum = recent_returns.sum()  # Cumulative return over 10 periods

        # Volume and volatility confirmation
        volume_ratio = row['volume'] / row['volume_ma'] if row['volume_ma'] > 0 else 0
        volatility = row['volatility']

        # Breakout conditions
        momentum_threshold = self.config["breakout_price_momentum"]
        vol_threshold = self.config["breakout_volatility_threshold"]
        volume_threshold = self.config["breakout_volume_threshold"]

        # Strong upward momentum + volume + volatility = BUY
        if (momentum > momentum_threshold and
            volatility > vol_threshold and
            volume_ratio > volume_threshold):

            return TradingAction(
                side="BUY",
                size=self.config["breakout_position_size"],
                slippage_bps=min(self.config["max_slippage_bps"], 30),  # Tighter slippage for breakouts
                deadline_s=120,  # Shorter deadline for momentum trades
                confidence=min(0.9, momentum * 20 + volatility * 10),
                rule_triggered="BREAKOUT_BUY"
            )

        # Strong downward momentum + volume + volatility = SELL
        elif (momentum < -momentum_threshold and
              volatility > vol_threshold and
              volume_ratio > volume_threshold):

            return TradingAction(
                side="SELL",
                size=self.config["breakout_position_size"],
                slippage_bps=min(self.config["max_slippage_bps"], 30),
                deadline_s=120,
                confidence=min(0.9, abs(momentum) * 20 + volatility * 10),
                rule_triggered="BREAKOUT_SELL"
            )

        return TradingAction(side="HOLD", size=0.0,
                           slippage_bps=self.config["max_slippage_bps"],
                           deadline_s=self.config["default_deadline_s"])

    def save_labels(self, labels_df: pd.DataFrame, output_path: str):
        """Save labeled dataset to CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Saving {} labels to {}", len(labels_df), output_path)
        labels_df.to_csv(output_path, index=False)

        # Save labeling statistics
        stats_path = output_path.parent / "labeling_stats.txt"
        with open(stats_path, 'w') as f:
            f.write("Heuristic Labeling Statistics\n")
            f.write("============================\n\n")

            # Action distribution
            f.write("Action Distribution:\n")
            action_counts = labels_df['side'].value_counts()
            for action, count in action_counts.items():
                pct = count / len(labels_df) * 100
                f.write(f"  {action}: {count} ({pct:.1f}%)\n")

            f.write("\nRule Trigger Distribution:\n")
            rule_counts = labels_df['rule_triggered'].value_counts()
            for rule, count in rule_counts.items():
                pct = count / len(labels_df) * 100
                f.write(f"  {rule}: {count} ({pct:.1f}%)\n")

            # Confidence statistics
            f.write("\nConfidence Statistics:\n")
            f.write(f"  Mean: {labels_df['confidence'].mean():.3f}\n")
            f.write(f"  Std: {labels_df['confidence'].std():.3f}\n")
            f.write(f"  Min: {labels_df['confidence'].min():.3f}\n")
            f.write(f"  Max: {labels_df['confidence'].max():.3f}\n")

            # Position sizing
            trading_labels = labels_df[labels_df['side'] != 'HOLD']
            if len(trading_labels) > 0:
                f.write("\nPosition Sizing (Non-HOLD actions):\n")
                f.write(f"  Mean size: {trading_labels['size'].mean():.3f}\n")
                f.write(f"  Std size: {trading_labels['size'].std():.3f}\n")
                f.write(f"  Min size: {trading_labels['size'].min():.3f}\n")
                f.write(f"  Max size: {trading_labels['size'].max():.3f}\n")

        logger.info("Labeling complete! Saved labels and statistics.")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate heuristic trading labels")
    parser.add_argument("--features", "-f", required=True, help="Path to core_features.csv")
    parser.add_argument("--conditions", "-c", required=True, help="Path to core_conditions.csv")
    parser.add_argument("--output", "-o", required=True, help="Output path for labeled_dataset.csv")
    parser.add_argument("--config", help="Optional config JSON file for trading rules")

    args = parser.parse_args()

    # Load configuration if provided
    config = None
    if args.config:
        import json
        with open(args.config) as f:
            config = json.load(f)

    # Load features
    logger.info("Loading features from {} and {}", args.features, args.conditions)
    features_df = pd.read_csv(args.features, parse_dates=['timestamp'])
    conditions_df = pd.read_csv(args.conditions, parse_dates=['timestamp'])

    # Generate labels
    labeler = HeuristicLabeler(config)
    labels_df = labeler.generate_labels(features_df, conditions_df)

    # Save results
    labeler.save_labels(labels_df, args.output)

    print("\nHeuristic labeling complete!")
    print(f"Generated {len(labels_df)} labels")
    print("Action distribution:")
    action_counts = labels_df['side'].value_counts()
    for action, count in action_counts.items():
        pct = count / len(labels_df) * 100
        print(f"  {action}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
