# ROOK Multi-Model Usage Guide

This document explains how to run ROOK agents with different trained models using the MAHAROOK framework.

## Available Configurations

### 1. Qwen2.5 LoRA Fine-tuned Model
- **Config**: `configs/rook_qwen_lora.yaml`
- **Description**: Trained LoRA adapter with 86% loss reduction
- **Base Model**: `Qwen/Qwen2.5-1.5B`
- **Model**: `models/qwen25_lora/` (163MB download from VM)
- **Specialization**: Financial trading decisions with buy/sell recommendations

### 2. Qwen2.5 via Ollama
- **Config**: `configs/rook_qwen_ollama.yaml`
- **Description**: Base Qwen2.5-1.5B model via Ollama API
- **Model**: `qwen2.5:1.5b` (986MB via Ollama)
- **Specialization**: General purpose financial AI

### 3. Fin-R1 LoRA Fine-tuned Model
- **Config**: `configs/rook_finr1_lora.yaml`
- **Description**: Trained LoRA adapter on DialoGPT-medium base
- **Base Model**: `microsoft/DialoGPT-medium`
- **Model**: `models/finr1_lora/` (to be downloaded from VM)
- **Specialization**: Financial trading with conversational AI capabilities

### 4. Fin-R1 via Ollama
- **Config**: `configs/rook_finr1_ollama.yaml`
- **Description**: Specialized Fin-R1 model via Ollama API
- **Model**: `hf.co/Mungert/Fin-R1-GGUF:latest` (4.7GB via Ollama)
- **Specialization**: Advanced financial reasoning and trading strategies

## Quick Start

### Prerequisites
```bash
# Install dependencies with uv
uv add pydantic web3 pandas transformers torch peft

# For Ollama configuration - ensure Ollama is running
ollama serve
ollama pull qwen2.5:1.5b
```

### Running Agents

#### Single Agent - LoRA Model
```bash
# Run trained LoRA model for 60 minutes
uv run python run_rook_models.py --config configs/rook_qwen_lora.yaml --duration 60
```

#### Single Agent - Ollama Models
```bash
# Run Qwen2.5 Ollama model for 30 minutes
uv run python run_rook_models.py --config configs/rook_qwen_ollama.yaml --duration 30

# Run Fin-R1 Ollama model for 30 minutes
uv run python run_rook_models.py --config configs/rook_finr1_ollama.yaml --duration 30
```

#### Multiple Agents in Parallel
```bash
# Terminal 1: Qwen2.5 LoRA agent
uv run python run_rook_models.py --config configs/rook_qwen_lora.yaml --duration 120 &

# Terminal 2: Fin-R1 Ollama agent
uv run python run_rook_models.py --config configs/rook_finr1_ollama.yaml --duration 120 &

# Terminal 3: Qwen2.5 Ollama agent
uv run python run_rook_models.py --config configs/rook_qwen_ollama.yaml --duration 120 &
```

### Debug Mode
```bash
# Run with detailed logging
uv run python run_rook_models.py --config configs/rook_qwen_lora.yaml --duration 10 --log-level DEBUG
```

## Integration with MAHAROOK

The runner integrates directly with MAHAROOK's:
- **RookAgent**: Main agent orchestrator
- **Brain**: Model-agnostic decision making (supports LoRA + Ollama)
- **Portfolio**: Balance and allocation management
- **Executor**: Trade execution via UniswapV4

## Configuration Structure

```yaml
# Agent identity
agent:
  name: "ROOK-Agent-Name"
  pair: "WETH_USDC"

# Model configuration
model:
  model_name: "Qwen/Qwen2-1.5B"  # or "qwen2.5:1.5b" for Ollama
  model_provider: "transformers"  # or "ollama"
  adapter_path: "models/qwen25_lora"  # For LoRA only

# Portfolio settings
portfolio:
  target_allocation: 0.5
  initial_eth_balance: 10.0
  initial_usdc_balance: 10000.0

# Risk management
risk:
  max_slippage: 0.005
  max_position_size: 0.1
  min_confidence: 0.3
```

## Model Performance

### Qwen2.5 LoRA Training Results
- **Training Loss**: 2.8769 → 0.3952 (86% reduction)
- **Training Time**: 33.4 minutes
- **Epochs**: 3
- **Samples**: 200
- **Model Size**: 163MB (adapter only)

### Ollama Model
- **Model Size**: 986MB (full model)
- **Inference**: API-based, efficient resource usage
- **Latency**: ~5-10 seconds per decision

## Advanced Usage

### Fork Testing
For comprehensive testing with multiple models:
```bash
# Use the existing fork tester
uv run python testing/rook_fork_tester.py --models models/qwen25_lora configs/rook_ollama.yaml
```

### Custom Configurations
Copy and modify existing configs:
```bash
cp configs/rook_qwen_lora.yaml configs/my_custom_agent.yaml
# Edit my_custom_agent.yaml
uv run python run_rook_models.py --config configs/my_custom_agent.yaml
```

## Architecture

```
MAHAROOK Framework
├── RookAgent (run_rook_models.py)
│   ├── Brain (model inference)
│   ├── Portfolio (balance management)
│   └── Executor (trade execution)
├── Models
│   ├── qwen25_lora/ (trained adapter)
│   └── Ollama qwen2.5:1.5b
└── Configs
    ├── rook_qwen_lora.yaml
    └── rook_ollama.yaml
```

This system provides a streamlined way to run multiple ROOK agents with different model configurations, leveraging the full MAHAROOK framework for professional trading capabilities.