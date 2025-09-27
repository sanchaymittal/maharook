#!/bin/bash
# Setup Script for Fin-R1 LoRA Training on Linux/GPU Machine
# Run this first on your Linux machine with GPU

set -e

echo "🚀 Setting up Fin-R1 LoRA Training Environment..."

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  No NVIDIA GPU detected. Training will be slower on CPU."
fi

# Create project directory
mkdir -p ~/fin_r1_training
cd ~/fin_r1_training

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip

# Core ML libraries
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.36.0
pip install peft>=0.7.0
pip install bitsandbytes>=0.41.0
pip install datasets>=2.14.0
pip install accelerate>=0.24.0

# Unsloth for efficient training
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Additional utilities
pip install loguru
pip install pandas
pip install numpy
pip install tqdm
pip install wandb  # Optional: for experiment tracking

# GGUF tools for model conversion
pip install gguf
pip install sentencepiece

# Install llama.cpp for GGUF handling
echo "🔧 Installing llama.cpp..."
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make clean && make -j$(nproc)
cd ..

echo "✅ Environment setup complete!"
echo "📁 Working directory: $(pwd)"
echo "🔧 Next steps:"
echo "   1. Transfer your trading data to this machine"
echo "   2. Download Fin-R1 model"
echo "   3. Run the training scripts"