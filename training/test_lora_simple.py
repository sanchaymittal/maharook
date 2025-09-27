#!/usr/bin/env python3
"""
Simple LoRA Test - Verify the infrastructure works
"""

import json
import torch
from pathlib import Path
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger


def test_lora_setup():
    """Test basic LoRA setup and text generation."""
    logger.info("Testing LoRA setup...")

    # Load parent model
    model_name = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="training/models/parent")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir="training/models/parent",
        torch_dtype=torch.float16,
        device_map="auto"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj", "c_fc"],
        bias="none"
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Test inference
    prompt = (
        "Market Analysis: Price: $3500.0000, Volatility: 2.50%, "
        "Volume: 0.0850 ETH, Volume MA: 0.0800, Liquidity Impact: 1.20, "
        "Price Change: 0.15%. Trading Decision:"
    )

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    decision = generated_text[len(prompt):].strip()

    logger.info("Generated decision: '{}'", decision)

    # Save a minimal LoRA adapter for testing
    output_dir = Path("training/models/test_lora")
    output_dir.mkdir(exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save config
    config = {
        "parent_model": model_name,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "target_modules": ["c_attn", "c_proj", "c_fc"],
        "max_length": 512
    }

    with open(output_dir / "lora_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    logger.info("Test LoRA adapter saved to {}", output_dir)
    return True


if __name__ == "__main__":
    test_lora_setup()
    print("✅ LoRA infrastructure test completed!")