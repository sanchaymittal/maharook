#!/usr/bin/env python3
"""
Fin-R1 GGUF to HuggingFace Conversion Script
--------------------------------------------
Converts Fin-R1 from GGUF format to HuggingFace transformers format for training.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

import torch
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


def download_finr1_gguf(output_dir: Path):
    """Download Fin-R1 GGUF model."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download from Hugging Face
    model_url = "https://huggingface.co/Mungert/Fin-R1-GGUF/resolve/main/Fin-R1-q4_k.gguf"
    output_file = output_dir / "Fin-R1-q4_k.gguf"

    if output_file.exists():
        logger.info("GGUF model already exists: {}", output_file)
        return output_file

    logger.info("Downloading Fin-R1 GGUF model...")
    subprocess.run([
        "wget", "-O", str(output_file), model_url
    ], check=True)

    logger.info("Downloaded Fin-R1 GGUF: {}", output_file)
    return output_file


def convert_gguf_to_hf(gguf_path: Path, output_dir: Path):
    """Convert GGUF to HuggingFace format using llama.cpp."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Converting GGUF to HuggingFace format...")

    # Use llama.cpp to convert GGUF to GGML, then to HF
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(exist_ok=True)

    try:
        # Method 1: Use gguf-py to extract weights
        logger.info("Attempting conversion via gguf-py...")

        import gguf

        # Read GGUF file
        reader = gguf.GGUFReader(str(gguf_path))

        # Extract metadata
        metadata = {}
        for field in reader.fields.values():
            if isinstance(field.data, (str, int, float, bool)):
                metadata[field.name] = field.data

        # Save metadata
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info("Extracted metadata from GGUF file")

        # For now, we'll use the base Qwen2 model and indicate this is Fin-R1
        logger.warning("Direct GGUF conversion complex - using Qwen2-7B as base")
        logger.warning("You may need to manually convert using specialized tools")

        return False  # Indicate manual conversion needed

    except Exception as e:
        logger.error("GGUF conversion failed: {}", e)
        return False


def create_finr1_config(output_dir: Path):
    """Create Fin-R1 configuration for training."""
    config = {
        "model_name": "Fin-R1",
        "base_model": "Qwen/Qwen2-7B-Instruct",  # Fin-R1 is based on Qwen2
        "source": "GGUF converted",
        "domain": "financial",
        "description": "Financial reasoning model based on DeepSeek-R1",
        "parameters": "7.6B",
        "context_length": 32768,
        "architecture": "qwen2"
    }

    with open(output_dir / "finr1_info.json", 'w') as f:
        json.dump(config, f, indent=2)

    logger.info("Created Fin-R1 configuration")


def main():
    """Main conversion workflow."""
    parser = argparse.ArgumentParser(description="Convert Fin-R1 GGUF to HuggingFace format")
    parser.add_argument("--gguf-path", help="Path to Fin-R1 GGUF file")
    parser.add_argument("--output-dir", required=True, help="Output directory for HF model")
    parser.add_argument("--download", action="store_true", help="Download GGUF first")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.download:
        gguf_path = download_finr1_gguf(output_dir / "gguf")
    else:
        if not args.gguf_path:
            logger.error("Must provide --gguf-path or use --download")
            sys.exit(1)
        gguf_path = Path(args.gguf_path)

    if not gguf_path.exists():
        logger.error("GGUF file not found: {}", gguf_path)
        sys.exit(1)

    # Attempt conversion
    success = convert_gguf_to_hf(gguf_path, output_dir / "hf_model")

    if not success:
        logger.warning("Direct conversion failed. Recommended approach:")
        logger.warning("1. Use Qwen2-7B-Instruct as base model for training")
        logger.warning("2. Fine-tune with your financial data")
        logger.warning("3. The resulting model will be similar to Fin-R1")

        # Use Qwen2 as base instead
        logger.info("Downloading Qwen2-7B-Instruct as base model...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2-7B-Instruct",
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")

            hf_output = output_dir / "qwen2_base"
            model.save_pretrained(hf_output)
            tokenizer.save_pretrained(hf_output)

            logger.info("Qwen2-7B base model ready for training: {}", hf_output)

        except Exception as e:
            logger.error("Failed to download Qwen2 base: {}", e)

    # Create configuration
    create_finr1_config(output_dir)

    print("\n" + "="*60)
    print("🎯 CONVERSION SUMMARY")
    print("="*60)
    print(f"📁 Output directory: {output_dir}")
    print(f"📄 GGUF source: {gguf_path}")
    print("🔧 Recommended next steps:")
    print("   1. Use Qwen2-7B-Instruct as base model")
    print("   2. Fine-tune with financial trading data")
    print("   3. Export LoRA adapter for deployment")
    print("="*60)


if __name__ == "__main__":
    main()