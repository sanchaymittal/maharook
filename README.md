# Maharook ♟️

**Agentic Trading Arena for DeFi**
Built on Uniswap v4 with specialized AI agents.

## Overview

Maharook is a multi-agent trading system where autonomous trading agents (ROOKs) compete on-chain. Each ROOK specializes in a specific trading pair, while the WAZIR orchestrator manages competitions and resource allocations. All outcomes are settled via a Payment Agent using smart contracts.

## 🏗️ Architecture

### Core Components

**ROOK Agents**
AI-driven traders with modular architecture:
- **Brain**: ML models for market analysis and prediction
- **Portfolio**: Position management and risk control
- **Executor**: On-chain transaction submission

**WAZIR Orchestrator**
Central competition manager:
- Assigns trading budgets to ROOKs
- Schedules and monitors trading duels
- Tracks performance metrics and rankings

**Payment Agent**
Settlement infrastructure:
- Manages escrow and stake deposits
- Executes automated settlements
- Distributes rewards based on performance

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- uv package manager
- Ethereum wallet with testnet funds
- Infura/Alchemy API key

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/maharook.git
cd maharook

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

### Configuration

All configuration is managed through environment files. Copy the example configuration:

```bash
cp config/example.env config/.env
```

Edit `config/.env` with your settings:

```env
# Network Configuration
ETHEREUM_RPC_URL=https://mainnet.infura.io/v3/YOUR_KEY
CHAIN_ID=1

# Trading Parameters
DEFAULT_SLIPPAGE_TOLERANCE=0.005
MAX_GAS_PRICE_GWEI=100

# Agent Configuration
ROOK_COUNT=3
WAZIR_ALLOCATION_STRATEGY=proportional
```

### Running the System

```bash
# Start the WAZIR orchestrator
python -m maharook.wazir start

# Deploy ROOK agents
python -m maharook.rook deploy --count 3

# Monitor trading arena
python -m maharook.monitor
```

## 📊 Trading Strategies

ROOKs can implement various trading strategies:

- **Arbitrage**: Cross-DEX price discrepancies
- **Market Making**: Liquidity provision with spread capture
- **Momentum**: Trend-following algorithms
- **Mean Reversion**: Statistical arbitrage

## 🔧 Development

### Project Structure

```
maharook/
├── src/
│   ├── agents/          # ROOK and WAZIR implementations
│   ├── contracts/       # Smart contract interfaces
│   ├── strategies/      # Trading strategy modules
│   └── utils/          # Shared utilities
├── tests/
│   ├── unit/           # Unit tests
│   ├── integration/    # Integration tests
│   └── contracts/      # Contract tests
├── config/             # Configuration files
└── docs/              # Documentation
```

### Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=maharook --cov-report=html

# Run specific test suite
uv run pytest tests/unit
```

### Code Quality

```bash
# Format code
uv run ruff format

# Lint code
uv run ruff check

# Type checking
uv run mypy src/
```

## 🔐 Security

- All smart contract interactions are validated before execution
- Private keys are never stored in code or configuration files
- Use hardware wallets for production deployments
- Regular security audits of smart contracts

## 📈 Performance Metrics

The system tracks:
- **PnL**: Profit and loss per ROOK
- **Win Rate**: Successful trades percentage
- **Sharpe Ratio**: Risk-adjusted returns
- **Gas Efficiency**: Transaction cost optimization

## 🤝 Contributing

Please read [CONTRIBUTING.md](docs/CONTRIBUTING.md) for contribution guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes following conventional commits
4. Run tests and ensure coverage
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🔗 Resources

- [Documentation](https://maharook.readthedocs.io)
- [Uniswap v4 Docs](https://docs.uniswap.org/contracts/v4/overview)
- [Discord Community](https://discord.gg/maharook)
- [Twitter Updates](https://twitter.com/maharook)

## ⚠️ Disclaimer

This software is for educational and research purposes. Trading cryptocurrencies carries significant risk. Always perform your own research and never invest more than you can afford to lose.

---

*Built with adherence to [Maharook Constitution v1.0.0](.specify/memory/constitution.md)*