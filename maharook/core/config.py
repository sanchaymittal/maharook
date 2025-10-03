"""
Core Configuration Management
-----------------------------
Centralized configuration following constitutional principles.
"""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings

from .exceptions import ConfigurationError


class TokenConfig(BaseModel):
    """Token configuration."""
    symbol: str
    address: str
    decimals: int

    @field_validator('decimals')
    @classmethod
    def validate_decimals(cls, v: int) -> int:
        if v < 0 or v > 30:
            raise ValueError("Token decimals must be between 0 and 30")
        return v


class ContractConfig(BaseModel):
    """Smart contract configuration."""
    name: str
    address: str
    abi_path: Optional[str] = None


class NetworkConfig(BaseModel):
    """Network configuration."""
    name: str
    chain_id: int
    rpc_url: str
    ws_url: Optional[str] = None
    explorer_url: Optional[str] = None


class TradingConfig(BaseModel):
    """Trading configuration."""
    max_slippage_bps: int = 50
    gas_price_gwei: Optional[int] = None
    max_gas_limit: int = 500000
    min_trade_size_eth: float = 0.001

    # ROOK Agent Configuration
    default_pair: str = "ETH/USDC"
    default_fee_tier: float = 0.0005
    default_target_allocation: float = 0.5
    max_slippage: float = 0.03
    max_position_size: float = 0.1
    max_daily_trades: int = 50
    min_confidence: float = 0.3
    default_slippage: float = 0.005
    default_deadline_minutes: int = 20
    high_volatility_threshold: float = 0.1
    min_portfolio_value_usd: float = 10.0
    default_model_name: str = "fin-r1"
    default_model_provider: str = "ollama"

    @field_validator('max_slippage_bps')
    @classmethod
    def validate_slippage(cls, v: int) -> int:
        if v < 1 or v > 1000:  # 0.01% to 10%
            raise ValueError("Slippage must be between 1 and 1000 basis points")
        return v


class CoreSettings(BaseSettings):
    """Core application settings loaded from config.yaml."""

    # Environment
    environment: str = "development"
    log_level: str = "INFO"

    # Network settings - loaded from config.yaml
    network: Optional[NetworkConfig] = None

    # Trading settings - loaded from config.yaml
    trading: Optional[TradingConfig] = None

    # API Keys
    basescan_api_key: Optional[str] = None
    coingecko_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    fluence_api_key: Optional[str] = None

    # Paths
    config_dir: Path = Path("config")
    data_dir: Path = Path("data")
    logs_dir: Path = Path("logs")

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "forbid"  # Fail fast on unknown config keys


class ConfigManager:
    """Configuration manager that loads from files and validates."""

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self.config_path: Path = config_path or Path("config")
        self.settings: CoreSettings = CoreSettings()
        self._tokens: dict[str, TokenConfig] = {}
        self._contracts: dict[str, ContractConfig] = {}

        self._load_configurations()

    def _load_configurations(self) -> None:
        """Load all configuration from single config.yaml file."""
        try:
            self._load_from_single_file()
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")

    def _load_from_single_file(self) -> None:
        """Load all configuration from config.yaml - single source of truth."""
        config_file = self.config_path / "config.yaml"
        if not config_file.exists():
            raise ConfigurationError(
                f"Configuration file not found: {config_file}. "
                f"Please create config/config.yaml with all configurations."
            )

        with open(config_file) as f:
            config_data = yaml.safe_load(f)

        # Load tokens
        for token_data in config_data.get('tokens', []):
            token = TokenConfig(**token_data)
            self._tokens[token.symbol] = token

        # Load contracts
        for contract_data in config_data.get('contracts', []):
            contract = ContractConfig(**contract_data)
            self._contracts[contract.name] = contract

        # Load network configuration from config.yaml
        if 'network' in config_data:
            network_data = config_data['network']
            self.settings.network = NetworkConfig(
                name=network_data.get('name', 'base'),
                chain_id=network_data.get('chain_id', 8453),
                rpc_url=network_data.get('rpc_url', ''),
                ws_url=network_data.get('ws_url'),
                explorer_url=network_data.get('block_explorer')
            )

        # Load trading configuration from config.yaml
        if 'trading' in config_data:
            trading_data = config_data['trading']
            self.settings.trading = TradingConfig(
                max_slippage_bps=int(trading_data.get('max_slippage', 0.03) * 10000),
                gas_price_gwei=trading_data.get('max_gas_price', 50) // 1000000000,
                max_gas_limit=trading_data.get('default_gas_limit', 300000),
                min_trade_size_eth=trading_data.get('min_trade_size_eth', 0.001),
                default_pair=trading_data.get('default_pair', 'ETH/USDC'),
                default_fee_tier=trading_data.get('default_fee_tier', 0.0005),
                default_target_allocation=trading_data.get('default_target_allocation', 0.5),
                max_slippage=trading_data.get('max_slippage', 0.03),
                max_position_size=trading_data.get('max_position_size', 0.1),
                max_daily_trades=trading_data.get('max_daily_trades', 50),
                min_confidence=trading_data.get('min_confidence', 0.3),
                default_slippage=trading_data.get('default_slippage', 0.005),
                default_deadline_minutes=trading_data.get('default_deadline_minutes', 20),
                high_volatility_threshold=trading_data.get('high_volatility_threshold', 0.1),
                min_portfolio_value_usd=trading_data.get('min_portfolio_value_usd', 10.0),
                default_model_name=config_data.get('model', {}).get('default_name', 'fin-r1'),
                default_model_provider=config_data.get('model', {}).get('default_provider', 'ollama')
            )

    def get_token(self, symbol: str) -> TokenConfig:
        """Get token configuration by symbol."""
        if symbol not in self._tokens:
            raise ConfigurationError(f"Token {symbol} not found in configuration")
        return self._tokens[symbol]

    def get_contract(self, name: str) -> ContractConfig:
        """Get contract configuration by name."""
        if name not in self._contracts:
            raise ConfigurationError(f"Contract {name} not found in configuration")
        return self._contracts[name]

    def get_all_tokens(self) -> dict[str, TokenConfig]:
        """Get all token configurations."""
        return self._tokens.copy()

    def get_all_contracts(self) -> dict[str, ContractConfig]:
        """Get all contract configurations."""
        return self._contracts.copy()

    def validate_configuration(self) -> bool:
        """Validate all configuration is properly set."""
        try:
            # Validate required tokens exist
            required_tokens = ["ETH", "USDC", "WETH"]
            for token in required_tokens:
                self.get_token(token)

            # Validate required contracts exist
            required_contracts = ["UNISWAP_V4_UNIVERSAL_ROUTER"]
            for contract in required_contracts:
                self.get_contract(contract)

            # Validate network configuration
            if not self.settings.network or not self.settings.network.rpc_url:
                raise ConfigurationError("Network configuration with RPC URL is required")

            # Validate trading configuration
            if not self.settings.trading:
                raise ConfigurationError("Trading configuration is required")

            return True

        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {e}")


# Global configuration instance
config_manager = ConfigManager()
settings = config_manager.settings
