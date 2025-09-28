# ROOK Training Pipeline

This directory contains the training pipeline for ROOK (Reasoning-Optimized Operations Kernel) trading agents.

## Overview

ROOK agents use LoRA fine-tuning of pre-trained language models for trading decisions:

1. **Current Implementation**: LoRA fine-tuning of DialoGPT-medium model
2. **Future Enhancements**: Could be extended with larger models or different base models

**IMPORTANT**: This uses LoRA (Low-Rank Adaptation) fine-tuning to adapt pre-trained
language models for financial trading decisions using our trading data.

## Directory Structure

```
training/
├── scripts/           # Training and deployment scripts
│   ├── finr1_lora_train.py   # LoRA fine-tuning script
│   ├── fluence_deploy.py     # Fluence cloud deployment
│   ├── fluence_quickstart.sh # Quick deployment script
│   ├── setup_linux_training.sh # Linux environment setup
│   └── transfer_to_linux.sh # Transfer script for Linux machines
├── data/
│   └── processed/            # Processed datasets (ready for training)
│       ├── core_features.csv # Core trading features
│       ├── core_conditions.csv # Market conditions
│       ├── labeled_dataset.csv # Labeled training data
│       ├── train.csv / val.csv / test.csv # Train/validation/test splits
│       ├── metadata.json     # Dataset metadata
│       └── normalization.json # Feature normalization parameters
├── models/                   # Trained model artifacts (output directory)
└── configs/                  # Training configurations
    ├── data_driven_heuristics.json # Trading rule parameters
    └── relaxed_heuristics.json     # Relaxed trading rules
```

## Quick Start

### 1. Local Training

```bash
# Train LoRA model locally
uv run python training/scripts/finr1_lora_train.py \
    --data-dir training/data/processed \
    --output-dir training/models/finr1_lora \
    --epochs 3 \
    --max-samples 1000 \
    --model microsoft/DialoGPT-medium
```

### 2. Cloud Training (Fluence)

```bash
# Quick deployment to Fluence cloud
./training/scripts/fluence_quickstart.sh

# Or manual deployment
python training/scripts/fluence_deploy.py \
    --ssh-key ~/.ssh/fluence_key.pub \
    --ssh-private-key ~/.ssh/fluence_key \
    --data-package fluence_training_package.tar.gz \
    --output-dir ./fluence_results
```

### 3. Using Trained Models

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load LoRA fine-tuned model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
base_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
model = PeftModel.from_pretrained(base_model, "training/models/finr1_lora")

# Generate trading decisions
inputs = tokenizer("Based on market data: price $3400, volume 150.0, volatility 0.002, should I buy or sell?", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Data Flow

```
Processed Trading Data → LoRA Fine-tuning Dataset
                                ↓
DialoGPT-medium + LoRA Adapters → Fine-tuned Model
                                ↓
Trading Prompts → Model Inference → Trading Decisions
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

### LoRA Fine-tuning Implementation
- **Base Model**: DialoGPT-medium (117M parameters)
- **LoRA Configuration**:
  - Rank (r): 16
  - Alpha: 32
  - Target modules: ["c_attn", "c_proj"]
  - Dropout: 0.1
- **Task**: Causal language modeling for trading decisions

### Training Configuration
```python
TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    use_cpu=True  # CPU-optimized training
)
```

## Training Results

Current LoRA fine-tuned model:
- **Training Samples**: 1,000 (configurable via --max-samples)
- **Base Model**: DialoGPT-medium
- **Training Format**: Instruction-response pairs
- **Action Distribution** (from underlying data): 78.7% HOLD, 11.5% SELL, 9.8% BUY
- **Training Time**: ~5-10 minutes (CPU), faster on GPU

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