#!/bin/bash
# Transfer Training Package to Linux Machine
# Usage: ./transfer_to_linux.sh <user@hostname>

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <user@hostname>"
    echo "Example: $0 user@gpu-server.local"
    exit 1
fi

REMOTE_HOST="$1"
PROJECT_DIR=$(pwd)

echo "🚀 Transferring Fin-R1 training setup to $REMOTE_HOST"

# Create transfer package
echo "📦 Creating training package..."
tar -czf training_package_full.tar.gz \
    training/data/processed/ \
    training/configs/ \
    training/scripts/setup_linux_training.sh \
    training/scripts/convert_finr1_gguf_to_hf.py \
    training/scripts/train_finr1_unsloth.py \
    training/scripts/deploy_adapter.py \
    training/README_LINUX_TRAINING.md

# Transfer package
echo "📤 Transferring package to $REMOTE_HOST..."
scp training_package_full.tar.gz "$REMOTE_HOST:~/"

# Create remote setup script
echo "📋 Creating remote setup commands..."
cat << 'EOF' > remote_setup.sh
#!/bin/bash
echo "🎯 Setting up Fin-R1 training on remote machine..."

# Extract package
cd ~/
tar -xzf training_package_full.tar.gz

# Make scripts executable
chmod +x training/scripts/*.sh
chmod +x training/scripts/*.py

# Run setup
./training/scripts/setup_linux_training.sh

echo "✅ Setup complete! Next steps:"
echo "1. Navigate to training directory: cd ~/training"
echo "2. Review README_LINUX_TRAINING.md"
echo "3. Run training: python scripts/train_finr1_unsloth.py --data-dir data/processed --output-dir finr1_lora"
echo "4. Package results: python scripts/deploy_adapter.py --model-dir finr1_lora"
EOF

# Transfer and run setup script
scp remote_setup.sh "$REMOTE_HOST:~/"
ssh "$REMOTE_HOST" "chmod +x remote_setup.sh && ./remote_setup.sh"

# Cleanup
rm training_package_full.tar.gz remote_setup.sh

echo ""
echo "🎉 Transfer complete!"
echo "📁 Package size: $(du -h training_data_package.tar.gz | cut -f1)"
echo "🔗 Remote host: $REMOTE_HOST"
echo ""
echo "Next steps on remote machine:"
echo "1. ssh $REMOTE_HOST"
echo "2. cd ~/training"
echo "3. Follow README_LINUX_TRAINING.md"
echo ""
echo "Training command:"
echo "python scripts/train_finr1_unsloth.py \\"
echo "    --data-dir data/processed \\"
echo "    --output-dir finr1_lora \\"
echo "    --base-model Qwen/Qwen2-7B-Instruct \\"
echo "    --batch-size 2 \\"
echo "    --num-epochs 1"