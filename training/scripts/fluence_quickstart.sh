#!/bin/bash
# Fluence Fin-R1 Training Quick Start
# One-command deployment to Fluence platform

set -e

echo "🚀 Fluence Fin-R1 Training Quick Start"
echo "======================================"

# Check prerequisites
if [ ! -f ~/.ssh/fluence_key ]; then
    echo "🔑 Generating SSH key for Fluence..."
    ssh-keygen -t rsa -b 4096 -f ~/.ssh/fluence_key -N ""
    chmod 600 ~/.ssh/fluence_key
    echo "✅ SSH key generated: ~/.ssh/fluence_key"
    echo "📋 Public key to add to Fluence Console:"
    cat ~/.ssh/fluence_key.pub
    echo ""
    read -p "Press Enter after adding this key to Fluence Console..."
fi

# Install dependencies
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+"
    exit 1
fi

pip3 install requests loguru --quiet

# Check for package
if [ ! -f fluence_training_package.tar.gz ]; then
    echo "📦 Creating training package..."
    tar -czf fluence_training_package.tar.gz \
        training/data/processed/ \
        training/scripts/finr1_lora_train.py \
        training/scripts/fluence_deploy.py \
        training/configs/ \
        training/FLUENCE_DEPLOYMENT_GUIDE.md
fi

echo "📊 Package info:"
echo "   Size: $(du -h fluence_training_package.tar.gz | cut -f1)"
echo "   Contains: Training data, scripts, configs"

# Get API key if not set
if [ -z "$FLUENCE_API_KEY" ]; then
    echo ""
    echo "🔐 Fluence API Key Setup:"
    echo "   1. Go to console.fluence.network"
    echo "   2. Generate API key in Settings"
    echo "   3. Enter it below (or set FLUENCE_API_KEY env var)"
    echo ""
    read -p "Enter Fluence API Key (optional for manual deployment): " api_key

    if [ ! -z "$api_key" ]; then
        export FLUENCE_API_KEY="$api_key"
    fi
fi

echo ""
echo "🎯 Deployment Options:"
echo "   1. Automatic deployment (requires API key)"
echo "   2. Manual deployment (console.fluence.network)"
echo ""
read -p "Choose option (1 or 2): " option

if [ "$option" = "1" ]; then
    if [ -z "$FLUENCE_API_KEY" ]; then
        echo "❌ API key required for automatic deployment"
        exit 1
    fi

    echo "🚀 Starting automatic deployment..."
    python3 training/scripts/fluence_deploy.py \
        --ssh-key ~/.ssh/fluence_key.pub \
        --ssh-private-key ~/.ssh/fluence_key \
        --data-package fluence_training_package.tar.gz \
        --output-dir ./fluence_results \
        --vm-name finr1-training

    echo ""
    echo "🎉 Deployment complete!"
    echo "📁 Results will be in: ./fluence_results/"

elif [ "$option" = "2" ]; then
    echo ""
    echo "📋 Manual Deployment Instructions:"
    echo "=================================="
    echo ""
    echo "1. 🌐 Go to console.fluence.network and login"
    echo ""
    echo "2. 🔍 Search for VM with these settings:"
    echo "   - Configuration: cpu-8-ram-16gb-storage-25gb"
    echo "   - Max price: \$2.00/day"
    echo "   - Datacenter: US, DE, or GB"
    echo "   - Storage: SSD/NVMe"
    echo ""
    echo "3. ⚙️  Configure VM:"
    echo "   - Name: finr1-training"
    echo "   - OS: Ubuntu 22.04 LTS"
    echo "   - SSH Key: $(cat ~/.ssh/fluence_key.pub)"
    echo "   - Open Ports: 22, 8080"
    echo ""
    echo "4. 🚀 Deploy VM and note the IP address"
    echo ""
    echo "5. 📤 Transfer training package:"
    echo "   scp -i ~/.ssh/fluence_key fluence_training_package.tar.gz ubuntu@<VM_IP>:~/"
    echo ""
    echo "6. 🔗 Connect and setup:"
    echo "   ssh -i ~/.ssh/fluence_key ubuntu@<VM_IP>"
    echo "   tar -xzf fluence_training_package.tar.gz"
    echo "   # Follow FLUENCE_DEPLOYMENT_GUIDE.md"
    echo ""
    echo "7. 🔥 Start training:"
    echo "   cd ~/finr1_training"
    echo "   # Setup environment as per guide"
    echo "   python3 scripts/finr1_lora_train.py --data-dir data/processed --output-dir models/finr1_lora"
    echo ""
    echo "8. 📥 Retrieve results when complete:"
    echo "   scp -i ~/.ssh/fluence_key ubuntu@<VM_IP>:~/finr1_training/models/finr1_lora.tar.gz ./"
    echo ""

    echo "📖 Full guide: training/FLUENCE_DEPLOYMENT_GUIDE.md"

else
    echo "❌ Invalid option. Please choose 1 or 2."
    exit 1
fi

echo ""
echo "💰 Expected Costs:"
echo "   - VM: ~\$1-2/day (8-core, 16GB RAM)"
echo "   - Training time: ~6-8 hours"
echo "   - Total cost: ~\$0.50-1.00"
echo ""
echo "🎁 Alpha Program: \$256 in credits available!"
echo "📊 Monitor at: console.fluence.network"
echo ""
echo "✅ Setup complete! Happy training! 🚀"