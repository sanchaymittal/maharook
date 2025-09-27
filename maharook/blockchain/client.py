"""Web3 client for Base blockchain interaction."""

from typing import Any

from eth_account import Account
from loguru import logger
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware

from maharook.core.config import settings
from maharook.core.exceptions import ConfigurationError, NetworkError


class BaseClient:
    """Web3 client for Base blockchain interaction."""

    def __init__(self, rpc_url: str | None = None, private_key: str | None = None):
        """Initialize Base client.

        Args:
            rpc_url: RPC endpoint URL. Defaults to settings.
            private_key: Private key for transactions. Defaults to settings.

        Raises:
            ConfigurationError: If required configuration is missing.
            NetworkError: If connection to blockchain fails.
        """
        self.rpc_url = rpc_url or settings.network.rpc_url
        self.private_key = private_key

        if not self.rpc_url:
            raise ConfigurationError("RPC URL is required but not provided")

        try:
            # Initialize Web3 client
            self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))

            # Add PoA middleware for Base compatibility
            self.w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

            # Set up account if private key provided
            self.account: Account | None = None
            if self.private_key:
                if not self.private_key.startswith('0x'):
                    self.private_key = f"0x{self.private_key}"
                self.account = Account.from_key(self.private_key)
                logger.info("Account initialized: {}", self.account.address)

            # Verify connection
            self._verify_connection()

        except Exception as e:
            raise NetworkError(f"Failed to initialize Base client: {e}")

    def _verify_connection(self) -> None:
        """Verify blockchain connection.

        Raises:
            NetworkError: If connection verification fails.
        """
        try:
            if not self.w3.is_connected():
                raise NetworkError("Web3 client is not connected")

            block_number = self.w3.eth.block_number
            chain_id = self.w3.eth.chain_id

            # Verify we're on the correct network
            expected_chain_id = settings.network.chain_id
            if chain_id != expected_chain_id:
                raise NetworkError(
                    f"Wrong network: expected chain ID {expected_chain_id}, got {chain_id}"
                )

            logger.info("Connected to {} (Chain ID: {}, Block: {})",
                       settings.network.name, chain_id, block_number)

        except Exception as e:
            raise NetworkError(f"Connection verification failed: {e}")

    @property
    def address(self) -> str | None:
        """Get wallet address."""
        return self.account.address if self.account else None

    def get_balance(self, address: str | None = None) -> float:
        """Get ETH balance for address.

        Args:
            address: Address to check. Defaults to connected account.

        Returns:
            Balance in ETH.

        Raises:
            ValueError: If no address provided and no account connected.
            NetworkError: If balance query fails.
        """
        addr = address or self.address
        if not addr:
            raise ValueError("No address provided and no account connected")

        try:
            balance_wei = self.w3.eth.get_balance(addr)
            return float(self.w3.from_wei(balance_wei, 'ether'))
        except Exception as e:
            raise NetworkError(f"Failed to get balance for {addr}: {e}")

    def get_gas_price(self) -> int:
        """Get current gas price in Wei.

        Returns:
            Gas price in Wei.

        Raises:
            NetworkError: If gas price query fails.
        """
        try:
            if settings.trading.gas_price_gwei:
                return self.w3.to_wei(settings.trading.gas_price_gwei, 'gwei')
            return self.w3.eth.gas_price
        except Exception as e:
            raise NetworkError(f"Failed to get gas price: {e}")

    def estimate_gas(self, transaction: dict[str, Any]) -> int:
        """Estimate gas for transaction.

        Args:
            transaction: Transaction parameters.

        Returns:
            Estimated gas limit.

        Raises:
            NetworkError: If gas estimation fails.
        """
        try:
            estimated = self.w3.eth.estimate_gas(transaction)

            # Apply safety margin
            safety_margin = 1.2  # 20% margin
            return int(estimated * safety_margin)

        except Exception as e:
            raise NetworkError(f"Gas estimation failed: {e}")

    def send_transaction(self, transaction: dict[str, Any]) -> str:
        """Send signed transaction.

        Args:
            transaction: Transaction parameters.

        Returns:
            Transaction hash.

        Raises:
            ConfigurationError: If no account connected.
            NetworkError: If transaction fails.
        """
        if not self.account:
            raise ConfigurationError("No account connected for signing transactions")

        try:
            # Set required defaults
            transaction.setdefault('from', self.account.address)
            transaction.setdefault('nonce', self.w3.eth.get_transaction_count(self.account.address))
            transaction.setdefault('gasPrice', self.get_gas_price())

            # Estimate gas if not provided
            if 'gas' not in transaction:
                transaction['gas'] = min(
                    self.estimate_gas(transaction),
                    settings.trading.max_gas_limit
                )

            # Validate gas limit
            if transaction['gas'] > settings.trading.max_gas_limit:
                raise NetworkError(f"Gas limit {transaction['gas']} exceeds maximum {settings.trading.max_gas_limit}")

            # Sign and send
            signed_txn = self.w3.eth.account.sign_transaction(transaction, self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)

            tx_hash_hex = tx_hash.hex()
            logger.info("Transaction sent: {}", tx_hash_hex)
            return tx_hash_hex

        except Exception as e:
            raise NetworkError(f"Transaction failed: {e}")

    def wait_for_receipt(self, tx_hash: str, timeout: int = 120) -> dict[str, Any]:
        """Wait for transaction receipt.

        Args:
            tx_hash: Transaction hash.
            timeout: Timeout in seconds.

        Returns:
            Transaction receipt.

        Raises:
            NetworkError: If receipt retrieval fails or times out.
        """
        try:
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=timeout)

            if receipt.status == 1:
                logger.info("Transaction {} confirmed in block {}", tx_hash, receipt.blockNumber)
            else:
                logger.error("Transaction {} failed with status {}", tx_hash, receipt.status)

            return dict(receipt)

        except Exception as e:
            raise NetworkError(f"Failed to get receipt for {tx_hash}: {e}")

    def get_contract(self, address: str, abi: list) -> Any:
        """Get contract instance.

        Args:
            address: Contract address.
            abi: Contract ABI.

        Returns:
            Contract instance.

        Raises:
            NetworkError: If contract instantiation fails.
        """
        try:
            checksum_address = Web3.to_checksum_address(address)
            return self.w3.eth.contract(address=checksum_address, abi=abi)
        except Exception as e:
            raise NetworkError(f"Failed to create contract instance for {address}: {e}")

    def get_network_info(self) -> dict[str, Any]:
        """Get current network information.

        Returns:
            Network information dictionary.
        """
        try:
            return {
                "chain_id": self.w3.eth.chain_id,
                "block_number": self.w3.eth.block_number,
                "gas_price": self.get_gas_price(),
                "connected": self.w3.is_connected(),
                "account": self.address
            }
        except Exception as e:
            logger.error("Failed to get network info: {}", e)
            return {
                "error": str(e),
                "connected": False
            }
