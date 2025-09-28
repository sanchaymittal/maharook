#!/usr/bin/env python3
"""
Fluence VM Deployment Script for Fin-R1 Training
------------------------------------------------
Automatically provisions VMs on Fluence platform and deploys LoRA training.
Optimized for CPU-based training with memory efficiency.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any

import requests
from loguru import logger


class FluenceDeployer:
    """Deploy and manage Fin-R1 training on Fluence VMs."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("FLUENCE_API_KEY")
        self.base_url = "https://api.fluence.dev"
        self.session = requests.Session()

        if not self.api_key:
            logger.error("❌ Fluence API key required!")
            logger.info("📝 Get your API key from: https://fluence.dev/")
            logger.info("💡 Set FLUENCE_API_KEY environment variable or use --api-key")
            raise ValueError("API key is required")

        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })

    def get_offerings(self, config: str = "cpu-8-ram-16gb-storage-25gb") -> List[Dict]:
        """Get available VM offerings from Fluence marketplace."""
        logger.info("🔍 Searching for VM offerings...")

        url = f"{self.base_url}/offerings/v3"
        params = {
            "basicConfiguration": config,
            "maxTotalPricePerEpochUsd": "2.0",  # Max $2/day
            "datacenterCountries": ["US", "DE", "GB"]  # Prefer these locations
        }

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            offerings = response.json()

            logger.info("Found {} VM offerings", len(offerings))

            # Sort by price (cheapest first)
            offerings.sort(key=lambda x: float(x.get("totalPricePerEpochUsd", "999")))

            for i, offer in enumerate(offerings[:3]):
                logger.info("Option {}: ${:.2f}/day - {} in {}",
                           i+1,
                           float(offer.get("totalPricePerEpochUsd", 0)),
                           offer.get("basicConfiguration", "unknown"),
                           offer.get("datacenterCountry", "unknown"))

            return offerings

        except Exception as e:
            logger.error("Failed to get offerings: {}", e)
            return []

    def deploy_vm(self, ssh_public_key: str, vm_name: str = "finr1-training") -> Dict:
        """Deploy VM on Fluence for training."""
        logger.info("🚀 Deploying {} VM on Fluence...", vm_name)

        # Configuration optimized for Fin-R1 training
        deployment_config = {
            "constraints": {
                "basicConfiguration": "cpu-8-ram-16gb-storage-25gb",  # Good for 7B model
                "maxTotalPricePerEpochUsd": "2.0",
                "datacenterCountries": ["US", "DE", "GB"],
                "cpuManufacturers": ["AMD", "Intel"],
                "storageTypes": ["SSD", "NVMe"]  # Fast storage for model loading
            },
            "instances": 1,
            "vmConfiguration": {
                "name": vm_name,
                "hostname": vm_name.replace("-", ""),
                "openPorts": [
                    {"port": 22, "protocol": "tcp"},    # SSH
                    {"port": 8080, "protocol": "tcp"},  # Optional monitoring
                    {"port": 6006, "protocol": "tcp"}   # TensorBoard
                ],
                "osImage": "https://cloud-images.ubuntu.com/minimal/releases/jammy/release/ubuntu-22.04-minimal-cloudimg-amd64.img",
                "sshKeys": [ssh_public_key]
            }
        }

        try:
            url = f"{self.base_url}/vms/v3"
            logger.debug("Sending request to: {}", url)
            logger.debug("Request payload: {}", deployment_config)

            response = self.session.post(url, json=deployment_config)

            # Log response details before raising for status
            logger.debug("Response status: {}", response.status_code)
            logger.debug("Response body: {}", response.text[:500])

            response.raise_for_status()

            result = response.json()

            # Handle both list and dict response formats
            if isinstance(result, list) and len(result) > 0:
                vm_info = result[0]
                vm_id = vm_info.get("vmId")
                vm_name_returned = vm_info.get("vmName", vm_name)
            else:
                vm_id = result.get("vmIds", [None])[0] if isinstance(result, dict) else None
                vm_name_returned = vm_name

            if vm_id:
                logger.success("✅ VM deployed successfully: {}", vm_id)
                return {"vm_id": vm_id, "name": vm_name_returned, "config": deployment_config}
            else:
                logger.error("❌ VM deployment failed: {}", result)
                return {}

        except Exception as e:
            logger.error("❌ VM deployment error: {}", e)
            if hasattr(e, 'response') and e.response is not None:
                logger.error("Response content: {}", e.response.text)
            return {}

    def get_vm_status(self, vm_id: str) -> Dict:
        """Get VM status and connection details."""
        try:
            url = f"{self.base_url}/vms/v3/{vm_id}"
            response = self.session.get(url)
            response.raise_for_status()

            vm_info = response.json()
            return vm_info

        except Exception as e:
            logger.error("Failed to get VM status: {}", e)
            return {}

    def wait_for_vm_ready(self, vm_id: str, timeout: int = 300) -> str:
        """Wait for VM to be ready and return IP address."""
        logger.info("⏳ Waiting for VM to be ready...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            vm_info = self.get_vm_status(vm_id)

            status = vm_info.get("status", "")
            ip_address = vm_info.get("publicIpAddress", "")

            logger.info("VM Status: {} - IP: {}", status, ip_address or "pending")

            if status == "Running" and ip_address:
                logger.success("✅ VM ready at {}", ip_address)
                return ip_address

            time.sleep(10)

        logger.error("❌ VM failed to become ready within {} seconds", timeout)
        return ""

    def setup_training_environment(self, ip_address: str, ssh_key_path: str):
        """Setup training environment on the VM via SSH."""
        logger.info("🔧 Setting up training environment...")

        # Commands to run on VM
        setup_commands = [
            # Update system
            "sudo apt update && sudo apt upgrade -y",

            # Install Python and dependencies
            "sudo apt install -y python3 python3-pip python3-venv git wget curl",

            # Install build tools
            "sudo apt install -y build-essential cmake",

            # Create working directory
            "mkdir -p ~/finr1_training && cd ~/finr1_training",

            # Create Python virtual environment
            "python3 -m venv venv && source venv/bin/activate",

            # Install PyTorch CPU version (no GPU on Fluence yet)
            "pip install --upgrade pip",
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",

            # Install training dependencies
            "pip install transformers>=4.36.0 peft>=0.7.0 datasets>=2.14.0",
            "pip install accelerate>=0.24.0 loguru pandas numpy tqdm",

            # Install Unsloth CPU version
            "pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'",

            # Create training directory structure
            "mkdir -p data models scripts logs"
        ]

        for i, cmd in enumerate(setup_commands):
            logger.info("Step {}/{}: {}", i+1, len(setup_commands), cmd[:50] + "...")

            try:
                result = subprocess.run([
                    "ssh",
                    "-i", ssh_key_path,
                    "-o", "StrictHostKeyChecking=no",
                    "-o", "UserKnownHostsFile=/dev/null",
                    f"ubuntu@{ip_address}",
                    cmd
                ], capture_output=True, text=True, timeout=120)

                if result.returncode != 0:
                    logger.warning("Command failed: {}", result.stderr)
                else:
                    logger.debug("✅ Command successful")

            except subprocess.TimeoutExpired:
                logger.warning("Command timed out: {}", cmd[:50])
            except Exception as e:
                logger.error("SSH command failed: {}", e)

        logger.success("✅ Environment setup complete")

    def transfer_training_data(self, ip_address: str, ssh_key_path: str, data_package: str):
        """Transfer training data to VM."""
        logger.info("📤 Transferring training data...")

        try:
            # Transfer compressed data
            subprocess.run([
                "scp",
                "-i", ssh_key_path,
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                data_package,
                f"ubuntu@{ip_address}:~/finr1_training/"
            ], check=True)

            # Extract data on VM
            package_name = Path(data_package).name
            subprocess.run([
                "ssh",
                "-i", ssh_key_path,
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                f"ubuntu@{ip_address}",
                f"cd ~/finr1_training && tar -xzf {package_name}"
            ], check=True)

            logger.success("✅ Training data transferred")

        except Exception as e:
            logger.error("❌ Data transfer failed: {}", e)

    def start_training(self, ip_address: str, ssh_key_path: str, config: Dict):
        """Start Fin-R1 training on the VM."""
        logger.info("🔥 Starting Fin-R1 training...")

        # Training command using actual LoRA training script
        training_cmd = f"""
        cd ~/finr1_training &&
        source venv/bin/activate &&
        python scripts/finr1_lora_train.py \\
            --data-dir data/processed \\
            --output-dir models/finr1_lora \\
            --epochs 1 \\
            --max-samples 1000 \\
            --model microsoft/DialoGPT-medium > logs/training.log 2>&1 &
        """

        try:
            subprocess.run([
                "ssh",
                "-i", ssh_key_path,
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                f"ubuntu@{ip_address}",
                training_cmd
            ], check=True)

            logger.success("✅ Training started in background")
            logger.info("📊 Monitor progress: ssh -i {} ubuntu@{} 'tail -f ~/finr1_training/logs/training.log'",
                       ssh_key_path, ip_address)

        except Exception as e:
            logger.error("❌ Training start failed: {}", e)

    def monitor_training(self, ip_address: str, ssh_key_path: str):
        """Monitor training progress."""
        logger.info("📊 Monitoring training progress...")

        try:
            # Check training log
            result = subprocess.run([
                "ssh",
                "-i", ssh_key_path,
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                f"ubuntu@{ip_address}",
                "tail -20 ~/finr1_training/logs/training.log"
            ], capture_output=True, text=True)

            if result.stdout:
                logger.info("Training log:\n{}", result.stdout)
            else:
                logger.warning("No training output yet")

        except Exception as e:
            logger.error("❌ Monitoring failed: {}", e)

    def retrieve_results(self, ip_address: str, ssh_key_path: str, output_dir: str):
        """Retrieve trained model from VM."""
        logger.info("📥 Retrieving training results...")

        try:
            # Package results on VM
            subprocess.run([
                "ssh",
                "-i", ssh_key_path,
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                f"ubuntu@{ip_address}",
                "cd ~/finr1_training && tar -czf finr1_results.tar.gz models/ logs/"
            ], check=True)

            # Download results
            subprocess.run([
                "scp",
                "-i", ssh_key_path,
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                f"ubuntu@{ip_address}:~/finr1_training/finr1_results.tar.gz",
                output_dir
            ], check=True)

            logger.success("✅ Results retrieved to {}", output_dir)

        except Exception as e:
            logger.error("❌ Results retrieval failed: {}", e)

    def cleanup_vm(self, vm_id: str):
        """Terminate VM to stop billing."""
        logger.info("🧹 Cleaning up VM...")

        try:
            url = f"{self.base_url}/vms/v3/{vm_id}"
            response = self.session.delete(url)
            response.raise_for_status()

            logger.success("✅ VM terminated: {}", vm_id)

        except Exception as e:
            logger.error("❌ VM cleanup failed: {}", e)


def main():
    """Main deployment workflow."""
    parser = argparse.ArgumentParser(description="Deploy Fin-R1 training on Fluence")
    parser.add_argument("--ssh-key", required=True, help="Path to SSH public key")
    parser.add_argument("--ssh-private-key", required=True, help="Path to SSH private key")
    parser.add_argument("--data-package", required=True, help="Training data package")
    parser.add_argument("--output-dir", default=".", help="Output directory for results")
    parser.add_argument("--vm-name", default="finr1-training", help="VM name")
    parser.add_argument("--api-key", help="Fluence API key (or set FLUENCE_API_KEY)")
    parser.add_argument("--auto-cleanup", action="store_true", help="Auto-terminate VM after training")

    args = parser.parse_args()

    # Read SSH public key
    with open(args.ssh_key, 'r') as f:
        ssh_public_key = f.read().strip()

    # Initialize deployer
    deployer = FluenceDeployer(api_key=args.api_key)

    # Skip offerings check for now and proceed with deployment
    logger.info("🚀 Proceeding with VM deployment...")

    # Deploy VM
    vm_result = deployer.deploy_vm(ssh_public_key, args.vm_name)
    if not vm_result:
        logger.error("❌ VM deployment failed")
        return

    vm_id = vm_result["vm_id"]

    try:
        # Wait for VM to be ready
        ip_address = deployer.wait_for_vm_ready(vm_id)
        if not ip_address:
            return

        # Setup environment
        deployer.setup_training_environment(ip_address, args.ssh_private_key)

        # Transfer data
        deployer.transfer_training_data(ip_address, args.ssh_private_key, args.data_package)

        # Start training
        deployer.start_training(ip_address, args.ssh_private_key, {})

        # Monitor progress
        logger.info("🎯 Training started on Fluence VM!")
        logger.info("📊 Monitor: ssh -i {} ubuntu@{} 'tail -f ~/finr1_training/logs/training.log'",
                   args.ssh_private_key, ip_address)
        logger.info("💰 Estimated cost: ~$1-2/day")
        logger.info("⏱️  Training time: ~4-8 hours (CPU)")

        # Wait for training completion (or user intervention)
        input("\nPress Enter when training is complete to retrieve results...")

        # Retrieve results
        deployer.retrieve_results(ip_address, args.ssh_private_key, args.output_dir)

    finally:
        # Cleanup if requested
        if args.auto_cleanup:
            deployer.cleanup_vm(vm_id)
        else:
            logger.info("🔧 To terminate VM later: curl -X DELETE -H 'Authorization: Bearer <api_key>' {}/vms/v3/{}",
                       deployer.base_url, vm_id)

    print("\n" + "="*60)
    print("🎉 FLUENCE TRAINING COMPLETE")
    print("="*60)
    print(f"📁 Results: {args.output_dir}/finr1_results.tar.gz")
    print(f"🆔 VM ID: {vm_id}")
    print("🚀 Ready for deployment back to MacBook!")
    print("="*60)


if __name__ == "__main__":
    main()