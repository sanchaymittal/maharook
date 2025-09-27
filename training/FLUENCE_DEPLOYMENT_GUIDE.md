# Fluence Deployment Guide for Fin-R1 Training

Complete guide for deploying Fin-R1 LoRA training on Fluence's decentralized compute platform.

## 🎯 Overview

Train a 7.6B parameter Fin-R1 model on Fluence VMs with:
- **75% cheaper** than traditional cloud
- **CPU-optimized** training (GPU support coming soon)
- **Memory-efficient** LoRA fine-tuning
- **Automated deployment** and monitoring
- **$256 Alpha credits** available

## 📋 Prerequisites

### 1. Fluence Account Setup
```bash
# Register at console.fluence.network with:
# - Email used in Alpha VM program application
# - Google account, or GitHub account
# Note: Web3Auth creates self-custodial wallet
```

### 2. SSH Key Preparation
```bash
# Generate SSH key pair if needed
ssh-keygen -t rsa -b 4096 -f ~/.ssh/fluence_key
chmod 600 ~/.ssh/fluence_key
```

### 3. API Access (Optional)
- Get API key from Fluence Console
- Set environment variable: `export FLUENCE_API_KEY="your_key"`

## 🚀 Quick Start

### Step 1: Prepare Training Package
```bash
cd ~/github/maharook

# Create complete training package
tar -czf fluence_training_package.tar.gz \
    training/data/processed/ \
    training/scripts/train_finr1_cpu.py \
    training/scripts/fluence_deploy.py \
    training/configs/

# Verify package size (should be ~2-3MB)
ls -lh fluence_training_package.tar.gz
```

### Step 2: Deploy to Fluence
```bash
# Install deployment dependencies
pip install requests loguru

# Deploy with automation
python training/scripts/fluence_deploy.py \
    --ssh-key ~/.ssh/fluence_key.pub \
    --ssh-private-key ~/.ssh/fluence_key \
    --data-package fluence_training_package.tar.gz \
    --output-dir ./fluence_results \
    --vm-name finr1-training \
    --auto-cleanup
```

### Step 3: Monitor Training
```bash
# The script will provide SSH command for monitoring:
ssh -i ~/.ssh/fluence_key ubuntu@<vm_ip> 'tail -f ~/finr1_training/logs/training.log'

# Check system resources
ssh -i ~/.ssh/fluence_key ubuntu@<vm_ip> 'htop'
```

## 🔧 Manual Deployment (Advanced)

### 1. VM Configuration Selection

Recommended configurations for Fin-R1 training:

| Configuration | vCPU | RAM | Storage | Est. Cost/Day | Training Time |
|---------------|------|-----|---------|---------------|---------------|
| `cpu-4-ram-8gb-storage-25gb` | 4 | 8GB | 25GB | ~$0.50 | ~12-16 hours |
| `cpu-8-ram-16gb-storage-25gb` | 8 | 16GB | 25GB | ~$1.00 | ~6-8 hours |
| `cpu-16-ram-32gb-storage-50gb` | 16 | 32GB | 50GB | ~$2.00 | ~4-6 hours |

### 2. Manual VM Deployment via Console

1. **Login** to console.fluence.network
2. **Search** for VM offerings:
   - Configuration: `cpu-8-ram-16gb-storage-25gb`
   - Max price: $2.00/day
   - Datacenter: US, DE, or GB
   - Storage: SSD/NVMe preferred

3. **Configure VM**:
   - Name: `finr1-training`
   - OS: Ubuntu 22.04 LTS
   - SSH Key: Upload your public key
   - Open Ports: 22 (SSH), 8080 (monitoring)

4. **Deploy** and wait for IP address

### 3. Manual Environment Setup

```bash
# Connect to VM
ssh -i ~/.ssh/fluence_key ubuntu@<vm_ip>

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install -y python3 python3-pip python3-venv git build-essential

# Create working directory
mkdir -p ~/finr1_training && cd ~/finr1_training

# Setup Python environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch CPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install training dependencies
pip install transformers peft datasets accelerate trl loguru pandas numpy
```

### 4. Transfer and Extract Data

```bash
# From local machine
scp -i ~/.ssh/fluence_key fluence_training_package.tar.gz ubuntu@<vm_ip>:~/finr1_training/

# On VM
cd ~/finr1_training
tar -xzf fluence_training_package.tar.gz
```

### 5. Start Training

```bash
# On VM
source venv/bin/activate

python scripts/train_finr1_cpu.py \
    --data-dir data/processed \
    --output-dir models/finr1_lora \
    --base-model Qwen/Qwen2-7B-Instruct \
    --batch-size 1 \
    --gradient-accumulation-steps 8 \
    --num-epochs 1 \
    --lora-r 32 \
    --max-seq-length 1024 \
    --use-cpu > logs/training.log 2>&1 &

# Monitor progress
tail -f logs/training.log
```

## 📊 Training Optimization

### CPU Performance Tuning

```bash
# Check CPU info
lscpu

# Monitor resource usage
htop
iostat -x 1

# Optimize for CPU training
export OMP_NUM_THREADS=8  # Match vCPU count
export MKL_NUM_THREADS=8
```

### Memory Management

```python
# In training script, these optimizations are already included:
- gradient_checkpointing=True
- dataloader_pin_memory=False
- low_cpu_mem_usage=True
- torch_dtype=torch.float32  # Optimal for CPU
```

### Cost Optimization

```bash
# Monitor daily costs
# Billing occurs at 5:55 PM UTC daily

# Estimated costs:
# - 8-core VM: ~$1/day
# - Training time: ~6-8 hours
# - Total cost: ~$0.50 (if completed in same day)
```

## 📥 Results Retrieval

### Automated Retrieval
```bash
# The deployment script handles this automatically
# Results will be in ./fluence_results/finr1_results.tar.gz
```

### Manual Retrieval
```bash
# On VM, package results
cd ~/finr1_training
tar -czf finr1_results.tar.gz models/ logs/

# Download from local machine
scp -i ~/.ssh/fluence_key ubuntu@<vm_ip>:~/finr1_training/finr1_results.tar.gz ./

# Extract and verify
tar -xzf finr1_results.tar.gz
ls -la models/finr1_lora/
```

## 🔄 Integration with MacBook

### 1. Process Results
```bash
cd ~/github/maharook

# Extract results
tar -xzf finr1_results.tar.gz

# Move to training directory
mv models/finr1_lora training/models/finr1_fluence/
```

### 2. Convert for Ollama Deployment
```bash
# Use existing deployment script
python training/scripts/deploy_adapter.py \
    --model-dir training/models/finr1_fluence \
    --output-dir deployment_package \
    --test-model
```

### 3. Deploy to Ollama
```bash
# Install in Ollama
ollama create finr1-trading -f deployment_package/finr1_package/Modelfile

# Test
ollama run finr1-trading "Analyze ETH at $3500, volatility 2.5%"
```

## 🐛 Troubleshooting

### Common Issues

#### VM Deployment Fails
```bash
# Check offerings first
curl -H "Authorization: Bearer $FLUENCE_API_KEY" \
     "https://api.fluence.dev/offerings/v3?basicConfiguration=cpu-8-ram-16gb-storage-25gb"

# Try different datacenter
# Increase max price limit
# Use smaller configuration
```

#### Training Out of Memory
```bash
# Reduce batch size
--batch-size 1 --gradient-accumulation-steps 16

# Reduce sequence length
--max-seq-length 512

# Use smaller model (if available)
--base-model Qwen/Qwen2-1.5B-Instruct
```

#### Slow Training
```bash
# Check CPU utilization
htop

# Optimize threading
export OMP_NUM_THREADS=$(nproc)

# Use smaller LoRA rank
--lora-r 16
```

#### Network Issues
```bash
# Check VM connectivity
ping google.com

# Verify SSH access
ssh -v -i ~/.ssh/fluence_key ubuntu@<vm_ip>

# Check open ports
sudo netstat -tlnp
```

### Recovery Procedures

#### Training Interrupted
```bash
# Resume from checkpoint (if saved)
python scripts/train_finr1_cpu.py \
    --resume-from-checkpoint models/finr1_lora/checkpoint-<step>
```

#### VM Terminated Unexpectedly
```bash
# Check VM status via API
curl -H "Authorization: Bearer $FLUENCE_API_KEY" \
     "https://api.fluence.dev/vms/v3/<vm_id>"

# Deploy new VM and restore from backup
```

## 💡 Best Practices

### 1. Cost Management
- Use smallest suitable VM configuration
- Monitor training progress regularly
- Set up automatic termination after completion
- Keep training data compressed

### 2. Performance Optimization
- Use CPU-optimized PyTorch build
- Enable gradient checkpointing
- Use appropriate threading settings
- Monitor memory usage

### 3. Data Security
- Use SSH key authentication
- Transfer data in compressed format
- Clean up VM after training
- Keep local backups of results

### 4. Training Quality
- Start with 1 epoch for validation
- Monitor loss convergence
- Test with sample prompts
- Save intermediate checkpoints

## 📞 Support Resources

- **Fluence Documentation**: https://fluence.dev/docs/build/overview
- **Fluence Console**: https://console.fluence.network
- **Alpha Program**: $256 credits available
- **Community**: Fluence Discord/Telegram

The Fluence platform provides an excellent cost-effective solution for training large language models, with the added benefit of decentralized infrastructure and transparent pricing. Your Fin-R1 model will be ready for deployment in just a few hours at a fraction of traditional cloud costs! 🚀