# ROOK Agent Runner

Universal agent system for MAHAROOK that supports multiple models and inference methods.

## Features

- **Model-agnostic**: Works with any model through configuration
- **Multiple providers**: Supports Transformers, Ollama, OpenAI, Anthropic
- **LoRA support**: Load and use fine-tuned LoRA adapters
- **Configuration-driven**: YAML-based configuration system
- **Multi-agent**: Run multiple agents simultaneously
- **Real-time trading**: Live market data processing and decision making

## Quick Start

### Single Agent

```bash
# Run Qwen2.5 LoRA agent
python run_rook.py --config configs/rook_qwen.yaml

# Run Ollama-based agent
python run_rook.py --config configs/rook_finr1.yaml
```

### Multiple Agents

```bash
# Run multiple agents in parallel
python run_rook.py --config configs/rook_qwen.yaml --daemon &
python run_rook.py --config configs/rook_finr1.yaml --daemon &
```

### Debug Mode

```bash
# Run with detailed logging
python run_rook.py --config configs/rook_qwen.yaml --log-level DEBUG
```

## Configuration

Each agent requires a YAML configuration file. See `configs/template.yaml` for a complete example.

### Available Configurations

- **`rook_qwen.yaml`**: Qwen2.5 LoRA fine-tuned model
- **`rook_finr1.yaml`**: Qwen2.5 via Ollama
- **`template.yaml`**: Configuration template

### Configuration Sections

#### Agent Section
```yaml
agent:
  name: "ROOK-Agent-Name"
  model_provider: "transformers"  # or "ollama"
  model_path: "path/to/model"     # For local models
  base_model: "base/model/name"   # For LoRA models
  model_name: "model:tag"         # For Ollama models
```

#### Inference Section
```yaml
inference:
  method: "lora"          # lora, api, local, remote
  max_tokens: 512
  temperature: 0.7
  timeout: 30
  api_url: "http://..."   # For API-based inference
```

#### Trading Section
```yaml
trading:
  strategy: "financial_advisor"
  data_source: "data/market.csv"
  risk_level: "moderate"
  position_size: 0.1
```

#### API Section
```yaml
api:
  port: 8000
  host: "localhost"
  cors_enabled: true
```

## Model Providers

### Transformers
- Local model loading
- LoRA adapter support
- CPU/GPU inference

### Ollama
- API-based inference
- Multiple model support
- Efficient resource usage

## Directory Structure

```
run_agent/
├── run_rook.py              # Main runner script
├── configs/                 # Configuration files
│   ├── rook_qwen.yaml      # Qwen2.5 LoRA config
│   ├── rook_finr1.yaml     # Ollama config
│   └── template.yaml       # Configuration template
├── utils/                   # Utility modules
│   ├── config_validator.py # Configuration validation
│   ├── model_loader.py     # Model loading utilities
│   └── __init__.py
└── README.md               # This file
```

## Dependencies

### Core Dependencies
```bash
pip install pyyaml loguru requests
```

### For Transformers Provider
```bash
pip install torch transformers peft
```

### For Ollama Provider
Ollama must be installed and running:
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
```

## Usage Examples

### Basic Trading Agent
```bash
python run_rook.py --config configs/rook_qwen.yaml
```

### Multi-Agent Trading System
```bash
# Terminal 1: Qwen2.5 LoRA agent
python run_rook.py --config configs/rook_qwen.yaml --daemon

# Terminal 2: Ollama agent
python run_rook.py --config configs/rook_finr1.yaml --daemon
```

### Custom Configuration
```bash
# Copy template and modify
cp configs/template.yaml configs/my_agent.yaml
# Edit my_agent.yaml with your settings
python run_rook.py --config configs/my_agent.yaml
```

## Architecture

The system is designed to be:

- **Modular**: Each component has a single responsibility
- **Extensible**: Easy to add new model providers
- **Configurable**: Everything controlled through YAML files
- **Scalable**: Support for multiple concurrent agents

## Troubleshooting

### Model Loading Issues
- Check model paths in configuration
- Verify dependencies are installed
- Check available disk space

### API Connection Issues
- Verify Ollama is running (`ollama serve`)
- Check API URLs and ports
- Test network connectivity

### Configuration Errors
- Validate YAML syntax
- Check required fields are present
- Use template.yaml as reference