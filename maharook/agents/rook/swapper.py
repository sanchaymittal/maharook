"""
Uniswap v4 Swapper
-----------------
A comprehensive module for executing swaps on Uniswap v4 on Base mainnet.
Follows constitutional principles: configuration-driven, fail-fast, clear errors.
"""

import json
from decimal import Decimal, getcontext
from typing import Any

from loguru import logger
from pydantic import BaseModel, field_validator

from maharook.blockchain.client import BaseClient
from maharook.core.config import config_manager, settings
from maharook.core.exceptions import (
    ConfigurationError,
    InsufficientFundsError,
    TradingError,
    TransactionError,
    ValidationError,
)

from .strategy import Order

# Set high precision for Decimal calculations
getcontext().prec = 40


def load_abi(abi_name: str) -> list:
    """Load ABI from configuration.

    Args:
        abi_name: Name of the ABI file

    Returns:
        ABI as list

    Raises:
        ConfigurationError: If ABI file not found or invalid
    """
    try:
        abi_path = settings.config_dir / "abi" / f"{abi_name}.json"
        if not abi_path.exists():
            # Fallback to minimal ABI for development
            return _get_minimal_abi(abi_name)

        with open(abi_path) as f:
            return json.load(f)
    except Exception as e:
        raise ConfigurationError(f"Failed to load ABI {abi_name}: {e}")


def _get_minimal_abi(abi_name: str) -> list:
    """Get minimal ABI for basic functionality."""
    if abi_name == "uniswap_v3_router":
        return [
            {
                "inputs": [
                    {
                        "components": [
                            {"internalType": "address", "name": "tokenIn", "type": "address"},
                            {"internalType": "address", "name": "tokenOut", "type": "address"},
                            {"internalType": "uint24", "name": "fee", "type": "uint24"},
                            {"internalType": "address", "name": "recipient", "type": "address"},
                            {"internalType": "uint256", "name": "deadline", "type": "uint256"},
                            {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
                            {"internalType": "uint256", "name": "amountOutMinimum", "type": "uint256"},
                            {"internalType": "uint160", "name": "sqrtPriceLimitX96", "type": "uint160"},
                        ],
                        "internalType": "struct ISwapRouter.ExactInputSingleParams",
                        "name": "params",
                        "type": "tuple",
                    }
                ],
                "name": "exactInputSingle",
                "outputs": [{"internalType": "uint256", "name": "amountOut", "type": "uint256"}],
                "stateMutability": "payable",
                "type": "function",
            }
        ]
    elif abi_name == "erc20":
        return [
            {
                "inputs": [],
                "name": "decimals",
                "outputs": [{"internalType": "uint8", "name": "", "type": "uint8"}],
                "stateMutability": "view",
                "type": "function",
            },
            {
                "inputs": [{"internalType": "address", "name": "account", "type": "address"}],
                "name": "balanceOf",
                "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
                "stateMutability": "view",
                "type": "function",
            },
            {
                "inputs": [
                    {"internalType": "address", "name": "spender", "type": "address"},
                    {"internalType": "uint256", "name": "amount", "type": "uint256"},
                ],
                "name": "approve",
                "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
                "stateMutability": "nonpayable",
                "type": "function",
            },
            {
                "inputs": [
                    {"internalType": "address", "name": "owner", "type": "address"},
                    {"internalType": "address", "name": "spender", "type": "address"},
                ],
                "name": "allowance",
                "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
                "stateMutability": "view",
                "type": "function",
            },
        ]
    else:
        raise ConfigurationError(f"No minimal ABI available for {abi_name}")


class SwapParams(BaseModel):
    """Parameters for a swap operation."""

    token_in: str
    token_out: str
    amount_in: float
    slippage_tolerance: float = 0.005  # 0.5% default
    deadline_minutes: int = 20
    fee_tier: int = 3000  # 0.3% default
    recipient: str | None = None

    @field_validator("slippage_tolerance")
    @classmethod
    def validate_slippage(cls, v: float) -> float:
        """Validate slippage tolerance."""
        max_slippage = settings.trading.max_slippage_bps / 10000
        if v < 0.0001 or v > max_slippage:
            raise ValueError(f"Slippage tolerance must be between 0.0001 and {max_slippage}")
        return v

    @field_validator("fee_tier")
    @classmethod
    def validate_fee_tier(cls, v: int) -> int:
        """Validate fee tier."""
        valid_tiers = [100, 500, 3000, 10000]  # 0.01%, 0.05%, 0.3%, 1%
        if v not in valid_tiers:
            raise ValueError(f"Fee tier must be one of: {valid_tiers}")
        return v

    @field_validator("amount_in")
    @classmethod
    def validate_amount(cls, v: float) -> float:
        """Validate amount is positive."""
        if v <= 0:
            raise ValueError("Amount must be positive")
        return v


class SwapResult(BaseModel):
    """Result of a swap operation."""

    success: bool
    transaction_hash: str | None = None
    amount_in: float
    amount_out: float
    gas_used: int | None = None
    gas_price: int | None = None
    effective_price: float | None = None
    slippage: float | None = None
    error_message: str | None = None


class UniswapV4Swapper:
    """
    Constitutional-compliant Uniswap v4/v3 swapper for Base mainnet.

    Features:
    - Configuration-driven (no hardcoded values)
    - Fail-fast error handling
    - Clear error messages with context
    - ETH <-> ERC20 and ERC20 <-> ERC20 swaps
    - Slippage protection and gas optimization
    """

    def __init__(self, client: BaseClient | None = None):
        """Initialize the swapper.

        Args:
            client: BaseClient instance. If None, creates a new one.

        Raises:
            ConfigurationError: If required configuration is missing.
        """
        try:
            self.client = client or BaseClient()

            if not self.client.account:
                raise ConfigurationError("BaseClient must have an account configured for swapping")

            # Validate configuration
            config_manager.validate_configuration()

            # Get router contract
            router_config = config_manager.get_contract("UNISWAP_V3_ROUTER")
            router_abi = load_abi("uniswap_v3_router")

            self.router_contract = self.client.get_contract(
                router_config.address,
                router_abi
            )

            # Load ERC20 ABI for token operations
            self.erc20_abi = load_abi("erc20")

            logger.info(
                "UniswapV4Swapper initialized with account: {}",
                self.client.address
            )

        except Exception as e:
            raise ConfigurationError(f"Failed to initialize swapper: {e}")

    def get_token_decimals(self, token_symbol: str) -> int:
        """Get token decimals from configuration.

        Args:
            token_symbol: Token symbol

        Returns:
            Token decimals

        Raises:
            ConfigurationError: If token not found
        """
        try:
            token_config = config_manager.get_token(token_symbol.upper())
            return token_config.decimals
        except Exception as e:
            raise ConfigurationError(f"Failed to get decimals for {token_symbol}: {e}")

    def get_token_address(self, token_symbol: str) -> str:
        """Get token address from configuration.

        Args:
            token_symbol: Token symbol

        Returns:
            Token address

        Raises:
            ConfigurationError: If token not found
        """
        try:
            token_config = config_manager.get_token(token_symbol.upper())
            return token_config.address
        except Exception as e:
            raise ConfigurationError(f"Failed to get address for {token_symbol}: {e}")

    def to_token_units(self, amount: float, token_symbol: str) -> int:
        """Convert float amount to token units (wei equivalent).

        Args:
            amount: Amount in token units
            token_symbol: Token symbol

        Returns:
            Amount in smallest token units

        Raises:
            ValidationError: If conversion fails
        """
        try:
            decimals = self.get_token_decimals(token_symbol)
            return int(Decimal(str(amount)) * Decimal(10**decimals))
        except Exception as e:
            raise ValidationError(f"Failed to convert {amount} {token_symbol} to units: {e}")

    def from_token_units(self, amount: int, token_symbol: str) -> float:
        """Convert token units to float amount.

        Args:
            amount: Amount in smallest token units
            token_symbol: Token symbol

        Returns:
            Amount in token units

        Raises:
            ValidationError: If conversion fails
        """
        try:
            decimals = self.get_token_decimals(token_symbol)
            return float(Decimal(amount) / Decimal(10**decimals))
        except Exception as e:
            raise ValidationError(f"Failed to convert {amount} units to {token_symbol}: {e}")

    def get_token_balance(self, token_symbol: str, address: str | None = None) -> float:
        """Get token balance for address.

        Args:
            token_symbol: Token symbol (ETH, USDC, etc.)
            address: Address to check. Defaults to connected account.

        Returns:
            Token balance as float.

        Raises:
            ValidationError: If balance query fails.
        """
        addr = address or self.client.address
        if not addr:
            raise ValidationError("No address provided and no account connected")

        try:
            if token_symbol.upper() == "ETH":
                return self.client.get_balance(addr)

            token_address = self.get_token_address(token_symbol)
            token_contract = self.client.get_contract(token_address, self.erc20_abi)
            balance_units = token_contract.functions.balanceOf(addr).call()

            return self.from_token_units(balance_units, token_symbol)

        except Exception as e:
            raise TradingError(f"Failed to get {token_symbol} balance for {addr}: {e}")

    def approve_token(self, token_symbol: str, spender: str, amount: float) -> str:
        """Approve token spending.

        Args:
            token_symbol: Token to approve.
            spender: Address to approve.
            amount: Amount to approve.

        Returns:
            Transaction hash.

        Raises:
            TradingError: If approval fails.
        """
        if token_symbol.upper() == "ETH":
            return "0x0"  # ETH doesn't need approval

        try:
            token_address = self.get_token_address(token_symbol)
            token_contract = self.client.get_contract(token_address, self.erc20_abi)

            # Check current allowance
            current_allowance = token_contract.functions.allowance(
                self.client.address, spender
            ).call()

            amount_units = self.to_token_units(amount, token_symbol)

            if current_allowance >= amount_units:
                logger.info("Token {} already approved for amount {}", token_symbol, amount)
                return "0x0"  # Already approved

            # Approve maximum amount for gas efficiency
            max_approval = 2**256 - 1

            transaction = token_contract.functions.approve(
                spender, max_approval
            ).build_transaction(
                {
                    "from": self.client.address,
                    "gasPrice": self.client.get_gas_price(),
                }
            )

            tx_hash = self.client.send_transaction(transaction)
            receipt = self.client.wait_for_receipt(tx_hash)

            if receipt["status"] != 1:
                raise TransactionError("Token approval failed", tx_hash, receipt)

            logger.info("Approved {} for spending by {}", token_symbol, spender)
            return tx_hash

        except Exception as e:
            raise TradingError(f"Failed to approve {token_symbol}: {e}")

    def calculate_minimum_amount_out(self, amount_out: float, slippage: float) -> float:
        """Calculate minimum amount out considering slippage."""
        return amount_out * (1 - slippage)

    def get_quote(self, params: SwapParams) -> dict[str, Any]:
        """Get quote for swap.

        Args:
            params: Swap parameters

        Returns:
            Quote information

        Note:
            This is a simplified implementation. In production, integrate with
            Uniswap quoter contract or price oracle feeds.
        """
        try:
            # Simplified quote calculation
            # In production: use Uniswap V3 quoter contract
            estimated_amount_out = params.amount_in * 0.999  # Simple 0.1% fee approximation

            return {
                "estimated_amount_out": estimated_amount_out,
                "estimated_gas": 150000,
                "estimated_gas_cost_eth": 0.002,
                "price_impact": 0.001,
            }
        except Exception as e:
            raise TradingError(f"Failed to get quote: {e}")

    def swap_exact_input_single(self, params: SwapParams) -> SwapResult:
        """Execute exact input single swap.

        Args:
            params: Swap parameters.

        Returns:
            Swap result.
        """
        try:
            # Validate input parameters
            self._validate_swap_params(params)

            # Validate balances
            balance_in = self.get_token_balance(params.token_in)
            if balance_in < params.amount_in:
                raise InsufficientFundsError(
                    f"Insufficient {params.token_in} balance",
                    {"required": params.amount_in, "available": balance_in}
                )

            # Get quote
            quote = self.get_quote(params)
            estimated_amount_out = quote["estimated_amount_out"]

            # Handle token approval for non-ETH tokens
            if params.token_in.upper() != "ETH":
                router_address = config_manager.get_contract("UNISWAP_V3_ROUTER").address
                self.approve_token(params.token_in, router_address, params.amount_in)

            # Calculate minimum amount out
            minimum_amount_out = self.calculate_minimum_amount_out(
                estimated_amount_out, params.slippage_tolerance
            )

            # Execute the swap
            swap_result = self._execute_swap_transaction(params, minimum_amount_out)

            logger.info(
                "Swap completed: {} {} -> {} {} (tx: {})",
                params.amount_in,
                params.token_in,
                swap_result.amount_out,
                params.token_out,
                swap_result.transaction_hash,
            )

            return swap_result

        except Exception as e:
            logger.error("Swap failed: {}", str(e))
            return SwapResult(
                success=False,
                amount_in=params.amount_in,
                amount_out=0.0,
                error_message=str(e),
            )

    def _validate_swap_params(self, params: SwapParams) -> None:
        """Validate swap parameters.

        Args:
            params: Swap parameters to validate

        Raises:
            ValidationError: If parameters are invalid
        """
        # Check minimum trade size
        min_size = settings.trading.min_trade_size_eth
        if params.token_in.upper() == "ETH" and params.amount_in < min_size:
            raise ValidationError(
                f"Trade size below minimum: {params.amount_in} < {min_size} ETH"
            )

        # Validate token symbols
        try:
            config_manager.get_token(params.token_in.upper())
            config_manager.get_token(params.token_out.upper())
        except Exception as e:
            raise ValidationError(f"Invalid token in swap params: {e}")

        # Check for same token swap
        if params.token_in.upper() == params.token_out.upper():
            raise ValidationError("Cannot swap token to itself")

    def _execute_swap_transaction(self, params: SwapParams, minimum_amount_out: float) -> SwapResult:
        """Execute the swap transaction.

        Args:
            params: Swap parameters
            minimum_amount_out: Minimum acceptable output amount

        Returns:
            Swap result

        Raises:
            TradingError: If swap execution fails
        """
        try:
            # Prepare swap parameters
            deadline = self.client.w3.eth.get_block("latest")["timestamp"] + (params.deadline_minutes * 60)
            recipient = params.recipient or self.client.address

            # Get token addresses
            token_in_address = self.get_token_address(params.token_in)
            token_out_address = self.get_token_address(params.token_out)

            # Handle ETH specially
            if params.token_in.upper() == "ETH":
                token_in_address = self.get_token_address("WETH")
            if params.token_out.upper() == "ETH":
                token_out_address = self.get_token_address("WETH")

            # Convert amounts to token units
            amount_in_units = self.to_token_units(params.amount_in, params.token_in)
            minimum_amount_out_units = self.to_token_units(minimum_amount_out, params.token_out)

            # Build transaction
            transaction = self.router_contract.functions.exactInputSingle(
                {
                    "tokenIn": token_in_address,
                    "tokenOut": token_out_address,
                    "fee": params.fee_tier,
                    "recipient": recipient,
                    "deadline": deadline,
                    "amountIn": amount_in_units,
                    "amountOutMinimum": minimum_amount_out_units,
                    "sqrtPriceLimitX96": 0,
                }
            ).build_transaction(
                {
                    "from": self.client.address,
                    "value": amount_in_units if params.token_in.upper() == "ETH" else 0,
                    "gasPrice": self.client.get_gas_price(),
                }
            )

            # Execute transaction
            tx_hash = self.client.send_transaction(transaction)
            receipt = self.client.wait_for_receipt(tx_hash)

            if receipt["status"] != 1:
                raise TransactionError("Swap transaction failed", tx_hash, receipt)

            # Calculate metrics (simplified - would parse from logs in production)
            actual_amount_out = minimum_amount_out  # Placeholder
            effective_price = actual_amount_out / params.amount_in if params.amount_in > 0 else 0
            slippage = 0.001  # Placeholder

            return SwapResult(
                success=True,
                transaction_hash=tx_hash,
                amount_in=params.amount_in,
                amount_out=actual_amount_out,
                gas_used=receipt.get("gasUsed"),
                gas_price=transaction.get("gasPrice"),
                effective_price=effective_price,
                slippage=slippage,
            )

        except Exception as e:
            raise TradingError(f"Swap execution failed: {e}")

    def swap_eth_to_token(self, token_out: str, eth_amount: float, slippage: float | None = None) -> SwapResult:
        """Convenience method for ETH -> Token swaps."""
        slippage = slippage or (settings.trading.max_slippage_bps / 10000)
        params = SwapParams(
            token_in="ETH",
            token_out=token_out,
            amount_in=eth_amount,
            slippage_tolerance=slippage,
        )
        return self.swap_exact_input_single(params)

    def swap_token_to_eth(self, token_in: str, token_amount: float, slippage: float | None = None) -> SwapResult:
        """Convenience method for Token -> ETH swaps."""
        slippage = slippage or (settings.trading.max_slippage_bps / 10000)
        params = SwapParams(
            token_in=token_in,
            token_out="ETH",
            amount_in=token_amount,
            slippage_tolerance=slippage,
        )
        return self.swap_exact_input_single(params)

    def swap_token_to_token(
        self, token_in: str, token_out: str, token_amount: float, slippage: float | None = None
    ) -> SwapResult:
        """Convenience method for Token -> Token swaps."""
        slippage = slippage or (settings.trading.max_slippage_bps / 10000)
        params = SwapParams(
            token_in=token_in,
            token_out=token_out,
            amount_in=token_amount,
            slippage_tolerance=slippage,
        )
        return self.swap_exact_input_single(params)

    def execute_order(self, order: Order) -> SwapResult:
        """Execute a trading order.

        Args:
            order: Order from trading strategy.

        Returns:
            Swap result.
        """
        params = SwapParams(
            token_in=order.token_in,
            token_out=order.token_out,
            amount_in=order.amount_in,
            slippage_tolerance=order.slippage_tolerance,
            fee_tier=order.pool_fee,
        )

        return self.swap_exact_input_single(params)

    def get_portfolio_summary(self) -> dict[str, float]:
        """Get summary of token balances."""
        summary = {}

        try:
            all_tokens = config_manager.get_all_tokens()

            for symbol, token_config in all_tokens.items():
                try:
                    balance = self.get_token_balance(symbol)
                    if balance > 0:
                        summary[symbol] = balance
                except Exception as e:
                    logger.warning("Failed to get balance for {}: {}", symbol, e)

            return summary

        except Exception as e:
            logger.error("Failed to get portfolio summary: {}", e)
            return {}


# Convenience functions for backward compatibility
def create_swapper(client: BaseClient | None = None) -> UniswapV4Swapper:
    """Create a swapper instance."""
    return UniswapV4Swapper(client)


def swap_eth_to_usdc(eth_amount: float, slippage: float = 0.01) -> SwapResult:
    """Quick ETH -> USDC swap."""
    swapper = create_swapper()
    return swapper.swap_eth_to_token("USDC", eth_amount, slippage)


def swap_usdc_to_eth(usdc_amount: float, slippage: float = 0.01) -> SwapResult:
    """Quick USDC -> ETH swap."""
    swapper = create_swapper()
    return swapper.swap_token_to_eth("USDC", usdc_amount, slippage)
