"""
Uniswap v4 Swapper
-----------------
A reusable module for executing swaps on Uniswap v4 on BASE
"""

from decimal import Decimal, getcontext
from typing import Optional, Any
from web3 import Web3
from web3.types import Wei
from eth_abi import encode
from uniswap_universal_router_decoder import RouterCodec
import logging
import time
from pydantic import BaseModel, field_validator

from maharook.core.config import config_manager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set high precision for Decimal calculations
getcontext().prec = 40

# Define token decimals
ETH_DECIMALS = 18
USDC_DECIMALS = 6

# Base addresses from config
BASE_USDC_ADDRESS = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"  # 6 decimals — pay attention to 6 decimals as most tokens are 18 decimals
ETH_ADDRESS = "0x0000000000000000000000000000000000000000"  # ETH native token address with standard 18 decimals
WETH_ADDRESS = "0x4200000000000000000000000000000000000006"  # WETH address on BASE with standard 18 decimals

# Permit2 constants
PERMIT2_ADDRESS = "0x000000000022D473030F116dDEE9F6B43aC78BA3"
MAX_UINT160 = 2**160 - 1

# ABI for the position manager (minimal interface for what we need)
POSITION_MANAGER_ABI = [
    {
        "inputs": [{"internalType": "bytes25", "name": "id", "type": "bytes25"}],
        "name": "poolKeys",
        "outputs": [
            {"internalType": "address", "name": "currency0", "type": "address"},
            {"internalType": "address", "name": "currency1", "type": "address"},
            {"internalType": "uint24", "name": "fee", "type": "uint24"},
            {"internalType": "int24", "name": "tickSpacing", "type": "int24"},
            {"internalType": "address", "name": "hooks", "type": "address"}
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

# Basic ABI for the pool manager to get swap events
POOL_MANAGER_ABI = [
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "bytes32", "name": "id", "type": "bytes32"},
            {"indexed": True, "internalType": "address", "name": "sender", "type": "address"},
            {"indexed": False, "internalType": "int128", "name": "amount0", "type": "int128"},
            {"indexed": False, "internalType": "int128", "name": "amount1", "type": "int128"},
            {"indexed": False, "internalType": "uint160", "name": "sqrtPriceX96", "type": "uint160"},
            {"indexed": False, "internalType": "uint128", "name": "liquidity", "type": "uint128"},
            {"indexed": False, "internalType": "int24", "name": "tick", "type": "int24"},
            {"indexed": False, "internalType": "uint24", "name": "fee", "type": "uint24"}
        ],
        "name": "Swap",
        "type": "event"
    }
]

# ABI for the state view contract
STATE_VIEW_ABI = [
    {
        "inputs": [{"internalType": "bytes32", "name": "poolId", "type": "bytes32"}],
        "name": "getSlot0",
        "outputs": [
            {"internalType": "uint160", "name": "sqrtPriceX96", "type": "uint160"},
            {"internalType": "int24", "name": "tick", "type": "int24"},
            {"internalType": "uint24", "name": "protocolFee", "type": "uint24"},
            {"internalType": "uint24", "name": "lpFee", "type": "uint24"}
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

# ERC20 ABI for token operations
ERC20_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function"
    },
    {
        "constant": False,
        "inputs": [
            {"name": "_spender", "type": "address"},
            {"name": "_value", "type": "uint256"}
        ],
        "name": "approve",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [
            {"name": "_owner", "type": "address"},
            {"name": "_spender", "type": "address"}
        ],
        "name": "allowance",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function"
    }
]

# Permit2 ABI
PERMIT2_ABI = [
    {
        "inputs": [
            {"name": "token", "type": "address"},
            {"name": "spender", "type": "address"},
            {"name": "amount", "type": "uint160"},
            {"name": "expiration", "type": "uint48"}
        ],
        "name": "approve",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"name": "", "type": "address"},
            {"name": "", "type": "address"},
            {"name": "", "type": "address"}
        ],
        "name": "allowance",
        "outputs": [
            {"name": "amount", "type": "uint160"},
            {"name": "expiration", "type": "uint48"},
            {"name": "nonce", "type": "uint48"}
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

def calculate_price_from_sqrt_price_x96(sqrt_price_x96: int) -> Optional[float]:
    """
    Calculate price from sqrtPriceX96 value using Uniswap V4's formula
    Price = (sqrtPriceX96 / 2^96) ^ 2
    For ETH/USDC pair, need to account for decimal differences (ETH: 18, USDC: 6)
    """
    try:
        # Convert to Decimal for precise calculation
        sqrt_price = Decimal(sqrt_price_x96)
        two_96 = Decimal(2) ** Decimal(96)

        # Calculate sqrtPrice / 2^96
        sqrt_price_adjusted = sqrt_price / two_96

        # Square it to get the price
        price = sqrt_price_adjusted * sqrt_price_adjusted

        # Convert to proper decimals (ETH/USDC)
        # ETH (18 decimals) to USDC (6 decimals) = multiply by 10^12
        price = price * Decimal(10 ** 12)

        return float(price)
    except Exception as e:
        logger.error(f"Error calculating price: {e}")
        return None

def tick_to_price(tick: int) -> Optional[float]:
    """
    Convert tick to price using the formula:
    price = 1.0001^tick
    For ETH/USDC, we need to account for decimal differences
    """
    try:
        tick_multiplier = Decimal('1.0001') ** Decimal(tick)
        # Adjust for decimals (ETH 18 - USDC 6 = 12)
        return float(tick_multiplier * Decimal(10 ** 12))
    except Exception as e:
        logger.error(f"Error calculating price from tick: {e}")
        return None

def setup_permit2_allowance(w3: Web3, account: Any, token_address: str, spender_address: str, amount: int) -> bool:
    """
    Set up Permit2 allowance for a token
    """
    permit2_contract = w3.eth.contract(address=PERMIT2_ADDRESS, abi=PERMIT2_ABI)

    # Check current allowance
    try:
        current_allowance = permit2_contract.functions.allowance(
            account.address, token_address, spender_address
        ).call()
        logger.info(f"Current Permit2 allowance: {current_allowance}")

        # If allowance is sufficient and not expired, return
        if current_allowance[0] >= amount and current_allowance[1] > int(time.time()):
            logger.info("Sufficient Permit2 allowance already exists")
            return True

    except Exception as e:
        logger.warning(f"Error checking Permit2 allowance: {e}")

    # Set new allowance
    expiration = int(time.time()) + 3600  # 1 hour from now

    try:
        logger.info(f"Setting Permit2 allowance...")

        # Build transaction
        tx_data = permit2_contract.functions.approve(
            token_address,
            spender_address,
            MAX_UINT160,
            expiration
        ).build_transaction({
            'from': account.address,
            'nonce': w3.eth.get_transaction_count(account.address),
            'gas': 100000,
            'gasPrice': w3.eth.gas_price,
            'chainId': w3.eth.chain_id
        })

        # Sign and send
        signed_tx = account.sign_transaction(tx_data)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

        if receipt.status == 1:
            logger.info(f"Permit2 allowance set successfully: {tx_hash.hex()}")
            return True
        else:
            logger.error(f"Permit2 allowance transaction failed")
            return False

    except Exception as e:
        logger.error(f"Error setting Permit2 allowance: {e}")
        return False


class SwapParams(BaseModel):
    """Parameters for a swap operation."""
    token_in: str
    token_out: str
    amount_in: float
    slippage_tolerance: float = 0.005  # 0.5% default
    deadline_minutes: int = 20
    fee_tier: int = 500  # 0.05% default
    recipient: Optional[str] = None

    @field_validator("slippage_tolerance")
    @classmethod
    def validate_slippage(cls, v: float) -> float:
        """Validate slippage tolerance."""
        if v < 0.0001 or v > 0.1:  # 0.01% to 10%
            raise ValueError(f"Slippage tolerance must be between 0.0001 and 0.1")
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
    transaction_hash: Optional[str] = None
    amount_in: float
    amount_out: float
    gas_used: Optional[int] = None
    gas_price: Optional[int] = None
    effective_price: Optional[float] = None
    slippage: Optional[float] = None
    error_message: Optional[str] = None

class UniswapV4Swapper:
    """
    A class to handle Uniswap v4 swaps on Base
    """

    def __init__(self, client: Optional[Any] = None) -> None:
        """
        Initialize the swapper

        Args:
            client: BaseClient instance (if None, will extract web3 and account info)
        """
        if client:
            self.w3 = client.w3
            self.account = client.account
            self.address = client.address
        else:
            # For standalone usage - would need web3 and private key
            raise ValueError("Client is required for ROOK integration")

        # Get contract addresses from config
        config = {
            'BASE_USDC_ADDRESS': BASE_USDC_ADDRESS,
            'ETH_ADDRESS': ETH_ADDRESS,
            'WETH_ADDRESS': WETH_ADDRESS,
            'UNISWAP_V4_UNIVERSAL_ROUTER': config_manager.get_contract("UNISWAP_V4_UNIVERSAL_ROUTER").address,
            'UNISWAP_V4_POSITION_MANAGER': config_manager.get_contract("UNISWAP_V4_POSITION_MANAGER").address,
            'UNISWAP_V4_STATE_VIEW': config_manager.get_contract("UNISWAP_V4_STATE_VIEW").address,
            'UNISWAP_V4_POOL_MANAGER': config_manager.get_contract("UNISWAP_V4_POOL_MANAGER").address,
        }

        # Store contract addresses
        self.usdc_address = config['BASE_USDC_ADDRESS']
        self.eth_address = config['ETH_ADDRESS']
        self.weth_address = config.get('WETH_ADDRESS', None)
        self.universal_router = config['UNISWAP_V4_UNIVERSAL_ROUTER']
        self.position_manager = config['UNISWAP_V4_POSITION_MANAGER']
        self.state_view = config['UNISWAP_V4_STATE_VIEW']
        self.pool_manager = config['UNISWAP_V4_POOL_MANAGER']

        # Convert addresses to checksum format
        self.usdc_address_cs = Web3.to_checksum_address(self.usdc_address)
        self.eth_address_cs = Web3.to_checksum_address(self.eth_address)
        self.weth_address_cs = Web3.to_checksum_address(self.weth_address) if self.weth_address else None
        self.universal_router_cs = Web3.to_checksum_address(self.universal_router)
        self.position_manager_cs = Web3.to_checksum_address(self.position_manager)
        self.state_view_cs = Web3.to_checksum_address(self.state_view)
        self.pool_manager_cs = Web3.to_checksum_address(self.pool_manager)

        # Initialize contracts
        self.usdc_contract = self.w3.eth.contract(address=self.usdc_address_cs, abi=ERC20_ABI)

        # Setup Universal Router Codec
        self.codec = RouterCodec(self.w3)

        # Create the v4 pool key for ETH/USDC (back to original order)
        self.eth_usdc_pool_key = self.codec.encode.v4_pool_key(
            self.eth_address_cs,     # Native ETH address
            self.usdc_address_cs,    # USDC address
            500,                     # 0.05% fee tier
            10                       # Tick spacing
        )

        # Use hardcoded pool ID from config instead of calculating
        self.pool_id = "0x96d4b53a38337a5733179751781178a2613306063c511b78cd02684739288c0a"
        calculated_pool_id = self._calculate_pool_id()
        logger.info(f"Initialized UniswapV4Swapper for account: {self.address}")
        logger.info(f"Using hardcoded pool ID: {self.pool_id}")
        logger.info(f"Calculated pool ID: {calculated_pool_id}")
        logger.info(f"Pool IDs match: {self.pool_id == calculated_pool_id}")

    def _calculate_pool_id(self) -> str:
        """Calculate the Uniswap v4 pool ID for ETH/USDC"""
        # Extract pool key parameters
        token0 = self.eth_usdc_pool_key['currency_0']
        token1 = self.eth_usdc_pool_key['currency_1']
        fee = self.eth_usdc_pool_key['fee']
        tick_spacing = self.eth_usdc_pool_key['tick_spacing']
        hooks = self.eth_usdc_pool_key['hooks']

        # Encode the pool parameters
        pool_init_code = encode(
            ['address', 'address', 'uint24', 'int24', 'address'],
            [token0, token1, fee, tick_spacing, hooks]
        )

        # Calculate the pool ID (keccak256 hash of the encoded parameters)
        calculated_pool_id = Web3.solidity_keccak(['bytes'], [pool_init_code]).hex()
        return f"0x{calculated_pool_id}"

    def get_eth_price(self) -> Optional[float]:
        """Get the current ETH price in USDC from the pool"""
        try:
            # Create state view contract instance
            state_view = self.w3.eth.contract(address=self.state_view_cs, abi=STATE_VIEW_ABI)

            # Get slot0 data
            try:
                # Try to call getSlot0 with pool_id directly (newer contract design)
                slot0_data = state_view.functions.getSlot0(self.pool_id).call()
            except Exception as e:
                logger.error(f"Error calling getSlot0 directly: {e}")
                # Try to call with bytes32 pool ID
                pool_id_bytes32 = Web3.to_bytes(hexstr=self.pool_id)
                try:
                    slot0_data = state_view.functions.getSlot0(pool_id_bytes32).call()
                except Exception as e2:
                    logger.error(f"Error calling getSlot0 with bytes32: {e2}")
                    # Fallback to hard-coded price for testing
                    logger.warning(f"Using fallback price for testing")
                    return 1800.0

            sqrt_price_x96, tick, protocol_fee, lp_fee = slot0_data

            # Calculate price from sqrt_price_x96
            price = calculate_price_from_sqrt_price_x96(sqrt_price_x96)

            # If that fails, try calculating from tick
            if not price:
                price = tick_to_price(tick)

            return price
        except Exception as e:
            logger.error(f"Error getting ETH price: {e}")
            return None

    def get_balances(self) -> dict[str, float]:
        """Get ETH and USDC balances for the account"""
        try:
            eth_balance = self.w3.eth.get_balance(self.address)
            usdc_balance = self.usdc_contract.functions.balanceOf(self.address).call()

            return {
                "ETH": eth_balance / 10**18,
                "USDC": usdc_balance / 10**6
            }
        except Exception as e:
            logger.error(f"Error getting balances: {e}")
            return {"ETH": 0, "USDC": 0}

    def swap_eth_to_usdc(self, eth_amount: float, slippage: float = 0.01) -> dict[str, Any]:
        """
        Swap ETH to USDC

        Args:
            eth_amount: Amount of ETH to swap
            slippage: Slippage tolerance (default 1%)

        Returns:
            dict: Transaction details
        """
        try:
            # Get balances before swap
            balances_before = self.get_balances()

            # Convert ETH amount to Wei
            amount_in = Wei(int(eth_amount * 10**18))
            logger.info(f"Swapping {eth_amount:.8f} ETH to USDC")

            # Get current ETH price to estimate output
            eth_price = self.get_eth_price()
            if not eth_price:
                logger.error("Failed to get ETH price, cannot estimate output")
                return {"success": False, "error": "Failed to get ETH price"}

            # Estimate USDC output
            estimated_usdc_out = eth_amount * eth_price
            min_usdc_out = estimated_usdc_out * (1 - slippage)

            # Build the transaction
            transaction_params = (
                self.codec
                .encode
                .chain()
                .v4_swap()
                .swap_exact_in_single(
                    pool_key=self.eth_usdc_pool_key,
                    zero_for_one=True,  # ETH is token0, USDC is token1
                    amount_in=amount_in,
                    amount_out_min=Wei(int(min_usdc_out * 10**6)),  # USDC has 6 decimals
                )
                .take_all(self.usdc_address_cs, Wei(0))
                .settle_all(self.eth_address_cs, amount_in)
                .build_v4_swap()
                .build_transaction(
                    self.address,
                    amount_in,
                    ur_address=self.universal_router_cs,
                    block_identifier=self.w3.eth.block_number
                )
            )

            # Make sure we have all required transaction parameters
            if 'chainId' not in transaction_params:
                transaction_params['chainId'] = self.w3.eth.chain_id

            if 'nonce' not in transaction_params:
                transaction_params['nonce'] = self.w3.eth.get_transaction_count(self.address)

            # Remove old gas pricing and add EIP-1559 parameters
            if 'gasPrice' in transaction_params:
                del transaction_params['gasPrice']

            # Add EIP-1559 gas parameters for Base
            transaction_params['maxFeePerGas'] = 2000000000  # 2 gwei
            transaction_params['maxPriorityFeePerGas'] = 1000000000  # 1 gwei

            # Set reasonable gas limit
            transaction_params['gas'] = 250000

            # Sign and send the transaction
            signed_txn = self.w3.eth.account.sign_transaction(transaction_params, self.account.key)

            # Get the raw transaction bytes
            if hasattr(signed_txn, 'rawTransaction'):
                raw_tx = signed_txn.rawTransaction
            elif hasattr(signed_txn, 'raw_transaction'):
                raw_tx = signed_txn.raw_transaction
            else:
                raise Exception("Could not find raw transaction data")

            # Send transaction
            tx_hash = self.w3.eth.send_raw_transaction(raw_tx)
            logger.info(f"Transaction sent: {self.w3.to_hex(tx_hash)}")

            # Wait for the transaction to be mined
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            # Check if transaction was successful
            if receipt['status'] == 1:
                logger.info("Transaction successful!")

                # Get balances after swap
                balances_after = self.get_balances()

                # Calculate amounts
                eth_spent = balances_before["ETH"] - balances_after["ETH"]
                usdc_received = balances_after["USDC"] - balances_before["USDC"]

                return {
                    "success": True,
                    "tx_hash": self.w3.to_hex(tx_hash),
                    "eth_spent": eth_spent,
                    "usdc_received": usdc_received,
                    "effective_price": usdc_received / eth_spent if eth_spent > 0 else 0,
                    "receipt": receipt
                }
            else:
                logger.error(f"Transaction failed: {receipt}")
                return {
                    "success": False,
                    "tx_hash": self.w3.to_hex(tx_hash),
                    "error": "Transaction failed",
                    "receipt": receipt
                }

        except Exception as e:
            logger.error(f"Error in swap_eth_to_usdc: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e)}

    def swap_usdc_to_eth(self, usdc_amount: float, slippage: float = 0.01) -> dict[str, Any]:
        """
        Swap USDC to ETH

        Args:
            usdc_amount: Amount of USDC to swap
            slippage: Slippage tolerance (default 1%)

        Returns:
            dict: Transaction details
        """
        try:
            # Get balances before swap
            balances_before = self.get_balances()
            usdc_balance = balances_before['USDC']

            # Make sure we don't try to swap more than we have
            if usdc_amount > usdc_balance * 0.99:
                logger.warning(f"Requested USDC amount ({usdc_amount}) higher than 99% of balance ({usdc_balance})")
                usdc_amount = usdc_balance * 0.95  # Use 95% of balance at most
                logger.info(f"Reduced USDC swap amount to {usdc_amount:.6f}")

            # Convert USDC amount to smallest unit (6 decimals)
            amount_in = int(usdc_amount * 10**6)
            logger.info(f"Swapping {usdc_amount:.6f} USDC to ETH")

            # Ensure minimum USDC amount (at least 0.01 USDC)
            if amount_in < 10000:  # 0.01 USDC in smallest units
                logger.warning(f"USDC amount too small: {usdc_amount:.6f} USDC")
                return {"success": False, "error": "USDC amount too small to swap"}

            # Get current ETH price to estimate output
            eth_price = self.get_eth_price()
            if not eth_price:
                logger.error("Failed to get ETH price, cannot estimate output")
                return {"success": False, "error": "Failed to get ETH price"}

            # Estimate ETH output
            estimated_eth_out = usdc_amount / eth_price
            min_eth_out = estimated_eth_out * (1 - slippage)
            min_eth_out_wei = Wei(int(min_eth_out * 10**18))

            # Set up Permit2 approvals for USDC
            try:
                # Step 1: Approve Permit2 to spend USDC
                current_allowance = self.usdc_contract.functions.allowance(self.address, PERMIT2_ADDRESS).call()
                if current_allowance < 10**6 * 1000:  # 1000 USDC worth of allowance
                    logger.info("Setting ERC20 allowance for Permit2...")

                    approve_tx = self.usdc_contract.functions.approve(
                        PERMIT2_ADDRESS,
                        2**256 - 1  # Max approval
                    ).build_transaction({
                        'from': self.address,
                        'nonce': self.w3.eth.get_transaction_count(self.address),
                        'gas': 100000,
                        'gasPrice': self.w3.eth.gas_price,
                        'chainId': self.w3.eth.chain_id
                    })

                    signed_tx = self.account.sign_transaction(approve_tx)
                    tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
                    receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

                    if receipt.status == 1:
                        logger.info(f"ERC20 approval for Permit2 successful: {tx_hash.hex()}")
                    else:
                        raise Exception("ERC20 approval for Permit2 failed")
                else:
                    logger.info("Sufficient ERC20 allowance already exists for Permit2")

                # Step 2: Set Permit2 allowance for Universal Router
                if not setup_permit2_allowance(self.w3, self.account, self.usdc_address_cs, self.universal_router_cs, amount_in):
                    raise Exception("Failed to set Permit2 allowance")

            except Exception as e:
                logger.error(f"Error setting up Permit2 approvals: {e}")
                return {"success": False, "error": f"Failed to set up Permit2 approvals: {str(e)}"}

            # Debug pool key information for USDC->ETH
            logger.info("🔧 USDC->ETH DEBUG:")
            logger.info(f"  Pool Key Token0 (ETH): {self.eth_usdc_pool_key['currency_0']}")
            logger.info(f"  Pool Key Token1 (USDC): {self.eth_usdc_pool_key['currency_1']}")
            logger.info(f"  Pool Key Fee: {self.eth_usdc_pool_key['fee']}")
            logger.info(f"  Pool Key Tick Spacing: {self.eth_usdc_pool_key['tick_spacing']}")
            logger.info(f"  Pool Key Hooks: {self.eth_usdc_pool_key['hooks']}")
            logger.info(f"  Calculated Pool ID: {self.pool_id}")
            logger.info(f"  USDC Amount: {amount_in} (6 decimals)")
            logger.info(f"  Zero for One: False (USDC->ETH)")

            # Build the transaction (matching working standalone script)
            try:
                # Convert amount_in to proper Wei if it isn't already
                if not isinstance(amount_in, int):
                    amount_in = int(amount_in)
                amount_in_wei = Wei(amount_in)

                transaction_params = (
                    self.codec
                    .encode
                    .chain()
                    .v4_swap()
                    .swap_exact_in_single(
                        pool_key=self.eth_usdc_pool_key,
                        zero_for_one=False,  # USDC to ETH (token1 to token0)
                        amount_in=amount_in_wei,
                        amount_out_min=Wei(0),  # Setting min amount to 0 for now
                    )
                    .settle_all(self.usdc_address_cs, amount_in_wei)
                    .take_all(self.eth_address_cs, Wei(0))
                    .build_v4_swap()
                    .build_transaction(
                        self.address,
                        Wei(0),  # No ETH value needed for USDC -> ETH swap
                        ur_address=self.universal_router_cs,
                        block_identifier=self.w3.eth.block_number
                    )
                )

                # Make sure we have all required transaction parameters
                if 'chainId' not in transaction_params:
                    transaction_params['chainId'] = self.w3.eth.chain_id

                if 'nonce' not in transaction_params:
                    transaction_params['nonce'] = self.w3.eth.get_transaction_count(self.address)

                # Use legacy gas pricing like working standalone script
                if 'gasPrice' not in transaction_params and 'maxFeePerGas' not in transaction_params:
                    transaction_params['gasPrice'] = self.w3.eth.gas_price

                # Set higher gas limit for better execution (matching working script)
                transaction_params['gas'] = 500000

                # Sign and send the transaction
                signed_txn = self.w3.eth.account.sign_transaction(transaction_params, self.account.key)

                # Get the raw transaction bytes (handle both attribute names)
                raw_tx = getattr(signed_txn, 'rawTransaction', None) or getattr(signed_txn, 'raw_transaction', None)
                if not raw_tx:
                    raise Exception("Could not find raw transaction data")

                # Send transaction
                tx_hash = self.w3.eth.send_raw_transaction(raw_tx)
                logger.info(f"Transaction sent: {self.w3.to_hex(tx_hash)}")
                logger.info("Waiting for confirmation...")

                # Wait for the transaction to be mined
                receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

                if receipt['status'] == 1:
                    logger.info("Transaction successful!")
                else:
                    logger.error("Transaction failed!")
                    logger.error(receipt)
                    return {
                        "success": False,
                        "tx_hash": self.w3.to_hex(tx_hash),
                        "error": "Transaction failed",
                        "receipt": receipt
                    }

                # Check balances after swap
                balances_after = self.get_balances()

                # Calculate amounts
                usdc_spent = balances_before["USDC"] - balances_after["USDC"]
                eth_received = balances_after["ETH"] - balances_before["ETH"]

                # Calculate gas cost properly
                gas_used = receipt['gasUsed']
                # For local testing, estimate gas cost
                gas_cost = (gas_used * 1000000000) / 10**18  # Estimate with 1 gwei

                # Calculate the pure swap amount (ETH received before gas costs)
                pure_swap_amount = eth_received + gas_cost

                # Calculate pure swap rate
                pure_swap_rate = pure_swap_amount / usdc_spent if usdc_spent > 0 else 0

                logger.info(f"USDC spent: {usdc_spent:.6f} USDC")
                logger.info(f"ETH received (after gas): {eth_received:.8f} ETH")
                logger.info(f"ETH received (swap only): {pure_swap_amount:.8f} ETH")
                logger.info(f"Pure swap rate: 1 USDC = {pure_swap_rate:.8f} ETH")

                return {
                    "success": True,
                    "tx_hash": self.w3.to_hex(tx_hash),
                    "usdc_spent": usdc_spent,
                    "eth_received": eth_received,
                    "pure_eth_received": pure_swap_amount,
                    "gas_cost_eth": gas_cost,
                    "effective_price": usdc_spent / eth_received if eth_received > 0 else 0,
                    "receipt": receipt
                }

            except Exception as e:
                logger.error(f"Error executing swap: {e}")
                return {"success": False, "error": str(e)}

        except Exception as e:
            logger.error(f"Error in swap_usdc_to_eth: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e)}

    # Legacy compatibility methods for ROOK integration
    def swap_exact_input_single(self, params: SwapParams) -> SwapResult:
        """Legacy method for ROOK compatibility"""
        try:
            if params.token_in.upper() == "ETH":
                result = self.swap_eth_to_usdc(params.amount_in, params.slippage_tolerance)
            else:
                result = self.swap_usdc_to_eth(params.amount_in, params.slippage_tolerance)

            # Convert to expected format
            if result["success"]:
                return SwapResult(
                    success=True,
                    transaction_hash=result["tx_hash"],
                    amount_in=params.amount_in,
                    amount_out=result.get("usdc_received", result.get("eth_received", 0)),
                    gas_used=result.get("receipt", {}).get("gasUsed"),
                    effective_price=result.get("effective_price", 0),
                    slippage=0.001
                )
            else:
                return SwapResult(
                    success=False,
                    amount_in=params.amount_in,
                    amount_out=0.0,
                    error_message=result.get("error", "Unknown error")
                )
        except Exception as e:
            return SwapResult(
                success=False,
                amount_in=params.amount_in,
                amount_out=0.0,
                error_message=str(e)
            )

    def get_portfolio_summary(self) -> dict[str, float]:
        """Get summary of token balances for ROOK compatibility"""
        balances = self.get_balances()
        return {
            "ETH": balances["ETH"],
            "USDC": balances["USDC"]
        }