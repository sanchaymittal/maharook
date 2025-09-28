"""
Model Loader for ROOK Agent
Universal model loading for different providers and inference methods
"""

import asyncio
import requests
from typing import Dict, Any, Optional
from pathlib import Path
from loguru import logger

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class ModelLoader:
    """Universal model loader for different providers."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent_config = config['agent']
        self.inference_config = config['inference']

        self.model = None
        self.tokenizer = None
        self.provider = self.agent_config['model_provider']
        self.method = self.inference_config['method']

    async def load_model(self):
        """Load model based on configuration."""
        logger.info("🔄 Loading model with provider: {}", self.provider)

        if self.provider == 'transformers':
            await self._load_transformers_model()
        elif self.provider == 'ollama':
            await self._load_ollama_model()
        else:
            raise ValueError(f"Unsupported model provider: {self.provider}")

        logger.success("✅ Model loaded successfully")

    async def _load_transformers_model(self):
        """Load model using transformers library."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not available. Install with: pip install transformers torch peft")

        if self.method == 'lora':
            await self._load_lora_model()
        elif self.method == 'local':
            await self._load_local_model()
        else:
            raise ValueError(f"Unsupported inference method for transformers: {self.method}")

    async def _load_lora_model(self):
        """Load LoRA fine-tuned model."""
        base_model = self.agent_config.get('base_model')
        lora_path = self.agent_config.get('model_path')

        if not base_model or not lora_path:
            raise ValueError("LoRA method requires both 'base_model' and 'model_path'")

        logger.info("Loading base model: {}", base_model)
        logger.info("Loading LoRA adapters: {}", lora_path)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        base_model_obj = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True
        )

        # Load LoRA adapters
        if Path(lora_path).exists():
            self.model = PeftModel.from_pretrained(base_model_obj, lora_path)
            logger.info("✅ LoRA adapters loaded from local path")
        else:
            raise FileNotFoundError(f"LoRA model path not found: {lora_path}")

    async def _load_local_model(self):
        """Load local model without LoRA."""
        model_path = self.agent_config.get('model_path')
        if not model_path:
            raise ValueError("Local method requires 'model_path'")

        logger.info("Loading local model: {}", model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True
        )

    async def _load_ollama_model(self):
        """Initialize Ollama model connection."""
        model_name = self.agent_config.get('model_name')
        api_url = self.inference_config.get('api_url', 'http://localhost:11434')

        if not model_name:
            raise ValueError("Ollama provider requires 'model_name'")

        logger.info("Connecting to Ollama: {}", api_url)
        logger.info("Model: {}", model_name)

        # Test connection
        try:
            response = requests.get(f"{api_url}/api/tags", timeout=10)
            if response.status_code != 200:
                raise ConnectionError(f"Failed to connect to Ollama API: {response.status_code}")

            # Check if model is available
            models = response.json()
            available_models = [model['name'] for model in models.get('models', [])]

            if model_name not in available_models:
                logger.warning("⚠️  Model {} not found in Ollama. Available: {}",
                             model_name, available_models)

            # Store connection info
            self.model = {
                'api_url': api_url,
                'model_name': model_name
            }

        except requests.RequestException as e:
            raise ConnectionError(f"Failed to connect to Ollama: {e}")

    async def generate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signal based on market data."""
        if self.provider == 'transformers':
            return await self._generate_transformers_signal(market_data)
        elif self.provider == 'ollama':
            return await self._generate_ollama_signal(market_data)
        else:
            raise ValueError(f"Signal generation not supported for provider: {self.provider}")

    async def _generate_transformers_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate signal using transformers model."""
        price = market_data.get('price', 0)
        volume = market_data.get('volume', 0)
        volatility = market_data.get('volatility', 0)

        # Create prompt
        prompt = f"Based on market data: price ${price:.2f}, volume {volume:.2f}, volatility {volatility:.4f}, should I buy or sell?"

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.inference_config['max_tokens'],
                temperature=self.inference_config['temperature'],
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()

        # Extract action
        action = "BUY" if "buy" in response.lower() else "SELL" if "sell" in response.lower() else "HOLD"

        return {
            'action': action,
            'response': response,
            'confidence': 0.85  # Placeholder
        }

    async def _generate_ollama_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate signal using Ollama API."""
        price = market_data.get('price', 0)
        volume = market_data.get('volume', 0)
        volatility = market_data.get('volatility', 0)

        # Create prompt
        prompt = f"Based on market data: price ${price:.2f}, volume {volume:.2f}, volatility {volatility:.4f}, should I buy or sell?"

        try:
            # Make API request
            response = requests.post(
                f"{self.model['api_url']}/api/generate",
                json={
                    "model": self.model['model_name'],
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.inference_config['temperature'],
                        "num_predict": self.inference_config['max_tokens']
                    }
                },
                timeout=self.inference_config.get('timeout', 30)
            )

            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '')

                # Extract action
                action = "BUY" if "buy" in response_text.lower() else "SELL" if "sell" in response_text.lower() else "HOLD"

                return {
                    'action': action,
                    'response': response_text,
                    'confidence': 0.80  # Placeholder
                }
            else:
                logger.error("Ollama API error: {}", response.text)
                return {'action': 'HOLD', 'response': 'API_ERROR', 'confidence': 0.0}

        except requests.RequestException as e:
            logger.error("Ollama request failed: {}", e)
            return {'action': 'HOLD', 'response': 'CONNECTION_ERROR', 'confidence': 0.0}

    async def cleanup(self):
        """Clean up model resources."""
        logger.info("🧹 Cleaning up model resources")

        if self.provider == 'transformers' and self.model is not None:
            # Clear GPU memory if using CUDA
            if hasattr(self.model, 'cpu'):
                self.model.cpu()
            del self.model
            del self.tokenizer

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.model = None
        self.tokenizer = None

        logger.info("✅ Model cleanup complete")