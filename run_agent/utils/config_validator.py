"""
Configuration Validator for ROOK Agent
Validates and loads YAML configuration files
"""

import yaml
from pathlib import Path
from typing import Dict, Any
from loguru import logger

class ConfigValidator:
    """Validates ROOK agent configuration files."""

    REQUIRED_SECTIONS = ['agent', 'inference', 'trading']
    REQUIRED_AGENT_FIELDS = ['name', 'model_provider']
    REQUIRED_INFERENCE_FIELDS = ['method', 'max_tokens', 'temperature']
    REQUIRED_TRADING_FIELDS = ['strategy', 'data_source']

    VALID_MODEL_PROVIDERS = ['transformers', 'ollama', 'openai', 'anthropic']
    VALID_INFERENCE_METHODS = ['lora', 'api', 'local', 'remote']

    async def load_and_validate(self, config_path: str) -> Dict[str, Any]:
        """Load and validate a configuration file."""
        logger.info("📋 Loading configuration: {}", config_path)

        # Load YAML file
        config = self._load_yaml(config_path)

        # Validate structure
        self._validate_structure(config)

        # Validate agent section
        self._validate_agent_section(config['agent'])

        # Validate inference section
        self._validate_inference_section(config['inference'])

        # Validate trading section
        self._validate_trading_section(config['trading'])

        # Apply defaults
        config = self._apply_defaults(config)

        logger.success("✅ Configuration validated successfully")
        return config

    def _load_yaml(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)

            if config is None:
                raise ValueError("Configuration file is empty")

            return config

        except FileNotFoundError:
            raise ValueError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load configuration: {e}")

    def _validate_structure(self, config: Dict[str, Any]):
        """Validate overall configuration structure."""
        missing_sections = []

        for section in self.REQUIRED_SECTIONS:
            if section not in config:
                missing_sections.append(section)

        if missing_sections:
            raise ValueError(f"Missing required sections: {missing_sections}")

    def _validate_agent_section(self, agent_config: Dict[str, Any]):
        """Validate agent configuration section."""
        missing_fields = []

        for field in self.REQUIRED_AGENT_FIELDS:
            if field not in agent_config:
                missing_fields.append(field)

        if missing_fields:
            raise ValueError(f"Missing required agent fields: {missing_fields}")

        # Validate model provider
        provider = agent_config['model_provider']
        if provider not in self.VALID_MODEL_PROVIDERS:
            raise ValueError(f"Invalid model provider: {provider}. "
                           f"Valid options: {self.VALID_MODEL_PROVIDERS}")

        # Provider-specific validation
        if provider == 'transformers':
            if 'model_path' not in agent_config and 'base_model' not in agent_config:
                raise ValueError("Transformers provider requires 'model_path' or 'base_model'")

        elif provider == 'ollama':
            if 'model_name' not in agent_config:
                raise ValueError("Ollama provider requires 'model_name'")

    def _validate_inference_section(self, inference_config: Dict[str, Any]):
        """Validate inference configuration section."""
        missing_fields = []

        for field in self.REQUIRED_INFERENCE_FIELDS:
            if field not in inference_config:
                missing_fields.append(field)

        if missing_fields:
            raise ValueError(f"Missing required inference fields: {missing_fields}")

        # Validate inference method
        method = inference_config['method']
        if method not in self.VALID_INFERENCE_METHODS:
            raise ValueError(f"Invalid inference method: {method}. "
                           f"Valid options: {self.VALID_INFERENCE_METHODS}")

        # Validate numeric ranges
        max_tokens = inference_config['max_tokens']
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")

        temperature = inference_config['temperature']
        if not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
            raise ValueError("temperature must be a number between 0 and 2")

    def _validate_trading_section(self, trading_config: Dict[str, Any]):
        """Validate trading configuration section."""
        missing_fields = []

        for field in self.REQUIRED_TRADING_FIELDS:
            if field not in trading_config:
                missing_fields.append(field)

        if missing_fields:
            raise ValueError(f"Missing required trading fields: {missing_fields}")

        # Validate data source exists if it's a file path
        data_source = trading_config['data_source']
        if isinstance(data_source, str) and not data_source.startswith('http'):
            if not Path(data_source).exists():
                logger.warning("⚠️  Data source file not found: {}", data_source)

    def _apply_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply default values to configuration."""
        # API defaults
        if 'api' not in config:
            config['api'] = {}

        api_defaults = {
            'host': 'localhost',
            'port': 8000,
            'cors_enabled': True
        }

        for key, default_value in api_defaults.items():
            if key not in config['api']:
                config['api'][key] = default_value

        # Inference defaults
        inference_defaults = {
            'timeout': 30,
            'retry_attempts': 3,
            'batch_size': 1
        }

        for key, default_value in inference_defaults.items():
            if key not in config['inference']:
                config['inference'][key] = default_value

        # Trading defaults
        trading_defaults = {
            'risk_level': 'moderate',
            'position_size': 0.1,
            'stop_loss': 0.02,
            'take_profit': 0.04
        }

        for key, default_value in trading_defaults.items():
            if key not in config['trading']:
                config['trading'][key] = default_value

        return config