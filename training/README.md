# ROOK Training Pipeline

This directory contains the complete training pipeline for ROOK (Reasoning-Optimized Operations Kernel) trading agents.

## Overview

ROOK agents are standalone PyTorch models trained for trading decisions:

1. **Current Implementation**: LSTM-based supervised learning from heuristic labels
2. **Future Enhancements**: Could be extended with transformer architectures or RL

**IMPORTANT**: This is NOT LoRA fine-tuning or parent model adaptation. We train
standalone LSTM models from scratch using our trading data.

## Directory Structure

```
training/
├── scripts/           # Training and data processing scripts
│   ├── feature_pipeline.py    # Extract features from raw data
│   ├── heuristics.py         # Generate trading action labels
│   ├── dataset.py            # Create train/val/test splits
│   └── train_rook.py         # Standalone model training script
├── data/
│   ├── processed/            # Processed datasets
│   │   ├── core_features.csv
│   │   ├── core_conditions.csv
│   │   ├── labeled_dataset.csv
│   │   ├── train.csv / val.csv / test.csv
│   │   ├── metadata.json
│   │   └── normalization.json
│   └── raw/                  # Raw Uniswap data
├── models/                   # Trained model artifacts
│   └── {pair}/              # Per trading pair
│       ├── best_model.pt     # Trained model weights
│       ├── config.yaml       # Trading configuration
│       └── norm.json         # Normalization parameters
└── configs/                  # Training configurations
    └── *.json               # Heuristic rule configs
```

## Quick Start

### 1. Data Preparation

```bash
# Extract features from raw Uniswap data
uv run python training/scripts/feature_pipeline.py \
    --input data/raw/uniswap_v4_eth_usdc.csv \
    --output training/data/processed

# Generate trading labels using heuristics
uv run python training/scripts/heuristics.py \
    --features training/data/processed/core_features.csv \
    --conditions training/data/processed/core_conditions.csv \
    --output training/data/processed/labeled_dataset.csv \
    --config training/configs/data_driven_heuristics.json

# Create train/val/test splits
uv run python training/scripts/dataset.py \
    --labeled-data training/data/processed/labeled_dataset.csv \
    --conditions training/data/processed/core_conditions.csv \
    --output training/data/processed \
    --seq-len 30
```

### 2. Model Training

```bash
# Train standalone ROOK model
uv run python training/scripts/train_rook.py \
    --data-dir training/data/processed \
    --output-dir training/models/eth_usdc_500 \
    --num-epochs 10 \
    --batch-size 32 \
    --hidden-dim 256
```

### 3. Using Trained Models

```python
from maharook.agents.rook.trained_brain import TrainedROOKBrain

# Load trained model
brain = TrainedROOKBrain(
    model_path="training/models/eth_usdc_500",
    pair="ETH_USDC"
)

# Make trading decisions
action = brain.decide(features, portfolio_state, market_state)
```

## Data Flow

```
Raw Uniswap Data → Feature Pipeline → Core Features + Conditions
                                            ↓
Trading Labels ← Heuristic Rules ← Feature Analysis
       ↓
Dataset Creation → Train/Val/Test Splits
       ↓
Model Training → Trained ROOK Agent
       ↓
Runtime Inference → Trading Actions
```

## Feature Engineering

### Core Features (Time Series)
- `price`: ETH/USDC price from sqrt_price_x96
- `volatility`: Rolling price change standard deviation
- `volume`: Transaction volume in ETH
- `volume_ma`: Volume moving average
- `liquidity_impact`: Volume relative to pool liquidity
- `price_change`: Price change from previous transaction

### Condition Features (Market Context)
- Rolling statistics (10, 50, 100 transaction windows)
- Price trends and volatility measures
- Volume patterns
- Market microstructure indicators

## Heuristic Trading Rules

### Mean Reversion Strategy
- **Trigger**: Price deviation > threshold + volume confirmation
- **Action**: Buy when oversold, sell when overbought
- **Position Size**: 5% of available balance

### Breakout Strategy
- **Trigger**: Price momentum + volatility + volume confirmation
- **Action**: Follow strong directional moves
- **Position Size**: 8% of available balance

### Risk Filters
- Volatility bounds (min/max trading volatility)
- Liquidity impact limits
- Market hours filtering
- Position size limits

## Model Architecture

### PyTorch Implementation
- **Core Encoder**: LSTM for sequence processing
- **Condition Encoder**: Feed-forward network
- **Fusion Layer**: Combined representation
- **Multi-Head Output**: Side (classification) + Size/Slippage/Deadline (regression)

### Loss Function
```
Total Loss = 2.0 * CrossEntropy(side) +
             1.0 * Huber(size) +
             1.0 * Huber(slippage) +
             0.5 * Huber(deadline)
```

## Training Results

Current ETH/USDC 500 bps model:
- **Training Samples**: 6,979 sequences
- **Validation Samples**: 1,496 sequences
- **Test Samples**: 1,496 sequences
- **Sequence Length**: 30 timesteps
- **Action Distribution**: 78.7% HOLD, 11.5% SELL, 9.8% BUY

## Performance Monitoring

### Training Metrics
- Multi-task loss components
- Side prediction accuracy
- Regression MSE for continuous targets
- Validation performance tracking

### Deployment Metrics
- Trading frequency and distribution
- P&L tracking per action type
- Risk metrics (drawdown, Sharpe ratio)
- Model confidence distribution

## Configuration

### Training Configuration
```json
{
  "batch_size": 32,
  "learning_rate": 5e-4,
  "num_epochs": 10,
  "seq_len": 30,
  "lora_r": 16,
  "lora_alpha": 32
}
```

### Trading Configuration (per pair)
```yaml
rook_config:
  pair: "ETH/USDC"
  max_position_size: 0.1
  max_slippage_bps: 50
  confidence_threshold: 0.6
```

## Next Steps

1. **Stage 2**: Implement chain-of-draft distillation
2. **Stage 3**: Add offline RL fine-tuning
3. **Multi-Pair**: Train models for additional pairs
4. **Evaluation**: Comprehensive backtesting framework
5. **Production**: Integration with live trading system

## Troubleshooting

### Common Issues

1. **Shape Mismatches**: Ensure feature extraction matches dataset creation
2. **Memory Issues**: Reduce batch size or sequence length
3. **Slow Training**: Consider using GPU or reducing model complexity
4. **Poor Convergence**: Adjust learning rate or check data quality

### Debug Commands

```bash
# Check data shapes
uv run python -c "
import pandas as pd
df = pd.read_csv('training/data/processed/train.csv')
print(f'Columns: {len(df.columns)}, Rows: {len(df)}')
"

# Validate model loading
uv run python -c "
from maharook.agents.rook.trained_brain import TrainedROOKBrain
brain = TrainedROOKBrain('training/models/eth_usdc_500')
print(brain.get_model_info())
"
```