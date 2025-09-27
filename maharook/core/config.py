"""
Core Configuration Management
-----------------------------
Centralized configuration following constitutional principles.
"""

from pathlib import Path

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
    abi_path: str | None = None


class NetworkConfig(BaseModel):
    """Network configuration."""
    name: str
    chain_id: int
    rpc_url: str
    ws_url: str | None = None
    explorer_url: str | None = None


class TradingConfig(BaseModel):
    """Trading configuration."""
    max_slippage_bps: int = 50
    gas_price_gwei: int | None = None
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
    default_model_name: str = "fin-r1"
    default_model_provider: str = "ollama"

    @field_validator('max_slippage_bps')
    @classmethod
    def validate_slippage(cls, v: int) -> int:
        if v < 1 or v > 1000:  # 0.01% to 10%
            raise ValueError("Slippage must be between 1 and 1000 basis points")
        return v


class CoreSettings(BaseSettings):
    """Core application settings."""

    # Environment
    environment: str = "development"
    log_level: str = "INFO"

    # Network settings
    network: NetworkConfig = NetworkConfig(
        name="base",
        chain_id=8453,
        rpc_url="https://mainnet.base.org",
        ws_url="wss://mainnet.base.org",
        explorer_url="https://basescan.org"
    )

    # Trading settings
    trading: TradingConfig = TradingConfig()

    # API Keys
    basescan_api_key: str | None = None
    coingecko_api_key: str | None = None
    openrouter_api_key: str | None = None

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

    def __init__(self, config_path: Path | None = None):
        self.config_path = config_path or Path("config")
        self.settings = CoreSettings()
        self._tokens: dict[str, TokenConfig] = {}
        self._contracts: dict[str, ContractConfig] = {}

        self._load_configurations()

    def _load_configurations(self):
        """Load all configuration files."""
        try:
            self._load_tokens()
            self._load_contracts()
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")

    def _load_tokens(self):
        """Load token configurations."""
        tokens_file = self.config_path / "tokens.yaml"
        if tokens_file.exists():
            with open(tokens_file) as f:
                tokens_data = yaml.safe_load(f)

            for token_data in tokens_data.get('tokens', []):
                token = TokenConfig(**token_data)
                self._tokens[token.symbol] = token
        else:
            # No hardcoded values - configuration file is required
            raise ConfigurationError(
                f"Token configuration file not found: {tokens_file}. "
                f"Please create config/tokens.yaml with token configurations."
            )

    def _load_contracts(self):
        """Load contract configurations."""
        contracts_file = self.config_path / "contracts.yaml"
        if contracts_file.exists():
            with open(contracts_file) as f:
                contracts_data = yaml.safe_load(f)

            for contract_data in contracts_data.get('contracts', []):
                contract = ContractConfig(**contract_data)
                self._contracts[contract.name] = contract
        else:
            # No hardcoded values - configuration file is required
            raise ConfigurationError(
                f"Contract configuration file not found: {contracts_file}. "
                f"Please create config/contracts.yaml with contract configurations."
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
            required_tokens = ["ETH", "USDC"]
            for token in required_tokens:
                self.get_token(token)

            # Validate required contracts exist
            required_contracts = ["UNISWAP_V3_ROUTER"]
            for contract in required_contracts:
                self.get_contract(contract)

            # Validate network configuration
            if not self.settings.network.rpc_url:
                raise ConfigurationError("RPC URL is required")

            return True

        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {e}")


# Global configuration instance
config_manager = ConfigManager()
settings = config_manager.settings
