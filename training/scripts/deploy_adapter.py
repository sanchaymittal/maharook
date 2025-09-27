#!/usr/bin/env python3
"""
Fin-R1 LoRA Adapter Deployment Script
-------------------------------------
Prepares trained LoRA adapter for deployment back to MacBook via Ollama.
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any

from loguru import logger


def create_ollama_modelfile(
    base_model: str,
    adapter_path: Path,
    output_path: Path,
    model_name: str = "finr1-trading"
):
    """Create Ollama Modelfile for LoRA deployment."""

    modelfile_content = f"""FROM {base_model}

# Set custom parameters for trading
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

# Custom system prompt for trading
SYSTEM \"\"\"You are Fin-R1, an expert cryptocurrency trading advisor specializing in ETH/USDC markets.

You analyze market conditions and provide structured trading decisions including:
- Action (BUY/SELL/HOLD)
- Position size
- Slippage tolerance
- Execution timeline
- Risk assessment

Always provide clear, actionable trading advice based on technical analysis and market conditions.
\"\"\"

# Load LoRA adapter (if supported in future Ollama versions)
# ADAPTER {adapter_path}

TEMPLATE \"\"\"{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ end }}{{ .Response }}<|im_end|>{{ if .Response }}
{{ end }}\"\"\"
"""

    with open(output_path, 'w') as f:
        f.write(modelfile_content)

    logger.info("Created Ollama Modelfile: {}", output_path)


def package_for_transfer(
    model_dir: Path,
    output_package: Path,
    include_gguf: bool = True
):
    """Package trained model for transfer to MacBook."""

    logger.info("📦 Packaging model for transfer...")

    # Create package directory
    package_dir = output_package / "finr1_package"
    package_dir.mkdir(parents=True, exist_ok=True)

    # Copy LoRA adapter
    adapter_source = model_dir
    adapter_dest = package_dir / "lora_adapter"

    if adapter_source.exists():
        shutil.copytree(adapter_source, adapter_dest, dirs_exist_ok=True)
        logger.info("✅ Copied LoRA adapter")

    # Copy merged model if available
    merged_source = Path(str(model_dir) + "_merged")
    if merged_source.exists():
        merged_dest = package_dir / "merged_model"
        shutil.copytree(merged_source, merged_dest, dirs_exist_ok=True)
        logger.info("✅ Copied merged model")

    # Copy GGUF if available and requested
    if include_gguf:
        gguf_source = Path(str(model_dir) + "_gguf")
        if gguf_source.exists():
            gguf_dest = package_dir / "gguf_model"
            shutil.copytree(gguf_source, gguf_dest, dirs_exist_ok=True)
            logger.info("✅ Copied GGUF model")

    # Create deployment instructions
    instructions = {
        "model_info": {
            "name": "Fin-R1 Trading LoRA",
            "base_model": "Qwen/Qwen2-7B-Instruct",
            "training_data": "ETH/USDC trading sequences",
            "lora_rank": 64,
            "domain": "cryptocurrency trading"
        },
        "deployment_steps": [
            "1. Transfer this package to your MacBook",
            "2. Install the GGUF model in Ollama: ollama create finr1-trading -f Modelfile",
            "3. Test the model: ollama run finr1-trading",
            "4. Integrate with your trading system using the MLX LoRA brain module"
        ],
        "files": {
            "lora_adapter/": "LoRA adapter weights (safetensors format)",
            "merged_model/": "Full model with LoRA merged (if available)",
            "gguf_model/": "GGUF quantized model for Ollama",
            "Modelfile": "Ollama configuration file",
            "deployment_guide.md": "Detailed deployment instructions"
        }
    }

    with open(package_dir / "deployment_info.json", 'w') as f:
        json.dump(instructions, f, indent=2)

    # Create Ollama Modelfile
    create_ollama_modelfile(
        base_model="finr1-base",  # You'll need to import base model to Ollama first
        adapter_path=package_dir / "gguf_model",
        output_path=package_dir / "Modelfile"
    )

    # Create deployment guide
    guide_content = f"""# Fin-R1 Trading Model Deployment Guide

## Overview
This package contains a fine-tuned Fin-R1 model specialized for ETH/USDC trading decisions.

## Files Included
- `lora_adapter/`: LoRA adapter weights for inference
- `merged_model/`: Complete model with LoRA weights merged
- `gguf_model/`: Quantized model optimized for Ollama
- `Modelfile`: Ollama configuration
- `deployment_info.json`: Technical details

## Deployment Steps

### 1. Setup on MacBook
```bash
# Navigate to your maharook project
cd ~/github/maharook

# Copy the package contents
cp -r finr1_package/* training/models/finr1_trained/
```

### 2. Install in Ollama
```bash
# Create Ollama model from GGUF
ollama create finr1-trading -f training/models/finr1_trained/Modelfile

# Test the model
ollama run finr1-trading "Analyze ETH price at $3500, volatility 2.5%, volume 0.085 ETH"
```

### 3. Integration with ROOK
Update your brain configuration:
```python
from maharook.agents.rook.mlx_lora_brain import MLXLoRAROOKBrain

# Use the trained model
brain = MLXLoRAROOKBrain(
    adapter_path="training/models/finr1_trained/lora_adapter",
    pair="ETH_USDC"
)
```

## Performance Notes
- Model optimized for financial trading decisions
- QLoRA training preserves 16-bit precision for critical layers
- GGUF format enables efficient local inference
- LoRA adapters are lightweight (typically <100MB)

## Troubleshooting
1. **Ollama import fails**: Ensure base model is available
2. **Memory issues**: Use q4_k_m quantization
3. **Poor predictions**: Check training data format matches inference

Enjoy your fine-tuned Fin-R1 trading model! 🚀
"""

    with open(package_dir / "deployment_guide.md", 'w') as f:
        f.write(guide_content)

    # Create compressed archive
    archive_path = output_package / "finr1_training_package.tar.gz"
    subprocess.run([
        "tar", "-czf", str(archive_path), "-C", str(package_dir.parent), package_dir.name
    ], check=True)

    logger.info("📦 Created deployment package: {}", archive_path)
    logger.info("📁 Package directory: {}", package_dir)

    return archive_path, package_dir


def test_model_locally(model_path: Path):
    """Test the trained model before packaging."""
    logger.info("🧪 Testing trained model...")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))

        # Test prompt
        test_prompt = """You are a professional cryptocurrency trader. Analyze this market:

Price: $3500.0000
Volatility: 2.50%
Volume: 0.0850 ETH
Volume MA: 0.0800
Liquidity Impact: 1.20
Price Change: +0.15%

Provide your trading decision:"""

        # Generate response
        inputs = tokenizer(test_prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        decision = response[len(test_prompt):].strip()

        logger.info("✅ Model test successful!")
        logger.info("Sample response: {}", decision[:200] + "..." if len(decision) > 200 else decision)

        return True

    except Exception as e:
        logger.error("❌ Model test failed: {}", e)
        return False


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Package Fin-R1 LoRA for deployment")
    parser.add_argument("--model-dir", required=True, help="Trained model directory")
    parser.add_argument("--output-dir", default="deployment_package", help="Output package directory")
    parser.add_argument("--test-model", action="store_true", help="Test model before packaging")
    parser.add_argument("--skip-gguf", action="store_true", help="Skip GGUF packaging")

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)

    if not model_dir.exists():
        logger.error("Model directory not found: {}", model_dir)
        return

    # Test model if requested
    if args.test_model:
        if not test_model_locally(model_dir):
            logger.warning("Model test failed, but continuing with packaging...")

    # Package for deployment
    archive_path, package_dir = package_for_transfer(
        model_dir=model_dir,
        output_package=output_dir,
        include_gguf=not args.skip_gguf
    )

    print("\n" + "="*60)
    print("🎉 DEPLOYMENT PACKAGE READY")
    print("="*60)
    print(f"📦 Archive: {archive_path}")
    print(f"📁 Package: {package_dir}")
    print("\n🚀 Next steps:")
    print("1. Transfer package to your MacBook")
    print("2. Follow deployment_guide.md instructions")
    print("3. Test with your trading system")
    print("="*60)


if __name__ == "__main__":
    main()