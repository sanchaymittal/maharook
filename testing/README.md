# ROOK + Fin-R1 Base Fork Testing

Clean, minimal setup for testing ROOK with Fin-R1 on Base mainnet fork.

## Files (Single Source of Truth)

- **`setup_simple.py`** - WETH deposit and approval setup
- **`rook_fork_tester.py`** - Main ROOK testing with Fin-R1
- **`base_market_simulator.py`** - Realistic market data simulation
- **`run_test.sh`** - Complete test runner script
- **`test_imports.py`** - Verify module imports work

## Configuration (Single Source of Truth)

All configuration is stored in `/maharook/config/` YAML files:
- **`config/contracts.yaml`** - Uniswap V4 contract addresses
- **`config/tokens.yaml`** - Token addresses and metadata

Testing scripts read from these YAML files directly.

## Quick Start

```bash
# Start complete test
./run_test.sh

# Or manual steps:
# 1. Start Anvil fork
anvil --fork-url https://mainnet.base.org --port 8545 --chain-id 8453 --accounts 5 --balance 100

# 2. Setup WETH
uv run python setup_simple.py

# 3. Run ROOK test (from maharook root)
cd .. && uv run python testing/rook_fork_tester.py --fork-url http://localhost:8545 --duration 1 --interval 60 --eth-balance 50.0 --usdc-balance 10000.0 --models "ollama/hf.co/Mungert/Fin-R1-GGUF:latest"
```