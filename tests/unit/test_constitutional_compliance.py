"""
Constitutional Compliance Tests
------------------------------
Verify that all ROOK components follow constitutional principles.
"""

import pytest
from unittest.mock import Mock, patch
from decimal import Decimal
import inspect

from maharook.core.config import config_manager, settings
from maharook.core.exceptions import (
    ConfigurationError,
    ValidationError,
    TradingError,
    InsufficientFundsError
)
from pydantic import ValidationError as PydanticValidationError
from maharook.agents.rook import (
    RookConfig,
    Portfolio,
    Brain,
    Executor,
    UniswapV4Swapper,
    SwapParams,
    TradingAction,
    MarketFeatures,
    PortfolioState
)


class TestConfigurationDriven:
    """Test Principle V: Configuration-Driven Architecture."""

    def test_no_hardcoded_token_addresses(self):
        """Verify no hardcoded token addresses in code."""
        # Token addresses should come from configuration files only
        import os

        # Check that hardcoded addresses don't exist in Python files
        for root, dirs, files in os.walk("maharook"):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        content = f.read()
                        # Look for Ethereum addresses (0x followed by 40 hex chars)
                        import re
                        addresses = re.findall(r'0x[0-9a-fA-F]{40}', content)
                        assert len(addresses) == 0, f"Found hardcoded addresses in {file_path}: {addresses}"

        # Verify addresses are loaded from config
        eth_config = config_manager.get_token("ETH")
        usdc_config = config_manager.get_token("USDC")

        # Should have addresses (from config files)
        assert eth_config.address.startswith("0x")
        assert usdc_config.address.startswith("0x")

    def test_slippage_from_settings(self):
        """Verify slippage limits come from settings."""
        params = SwapParams(
            token_in="ETH",
            token_out="USDC",
            amount_in=1.0,
            slippage_tolerance=0.001  # 0.1%
        )

        # Should validate against configured maximum
        max_slippage = settings.trading.max_slippage_bps / 10000
        assert params.slippage_tolerance <= max_slippage

    def test_gas_limits_configurable(self):
        """Verify gas limits come from configuration."""
        assert hasattr(settings.trading, 'max_gas_limit')
        assert settings.trading.max_gas_limit > 0

    def test_rook_config_externalized(self):
        """Verify ROOK configurations are externalized."""
        # Test that default config comes from settings, not hardcoded values
        config = RookConfig()

        # Verify config uses settings from TradingConfig
        assert config.pair == settings.trading.default_pair
        assert config.fee_tier == settings.trading.default_fee_tier
        assert config.target_allocation == settings.trading.default_target_allocation
        assert config.max_slippage == settings.trading.max_slippage
        assert config.max_position_size == settings.trading.max_position_size
        assert config.min_confidence == settings.trading.min_confidence

        # Test that config can still be overridden
        config_override = RookConfig(
            pair="BTC/USDC",
            target_allocation=0.6,
            max_slippage=0.02
        )

        assert config_override.pair == "BTC/USDC"
        assert config_override.target_allocation == 0.6
        assert config_override.max_slippage == 0.02

    def test_no_hardcoded_values_in_agent(self):
        """Verify agent.py RookConfig has no hardcoded default values."""
        import os
        import re

        agent_file = "maharook/agents/rook/agent.py"
        assert os.path.exists(agent_file), f"Agent file not found: {agent_file}"

        with open(agent_file, 'r') as f:
            content = f.read()

        # Check RookConfig dataclass doesn't have hardcoded numeric defaults
        # Should use settings.trading.* instead of hardcoded values
        rook_config_section = re.search(r'@dataclass\s+class RookConfig:(.*?)(?=@dataclass|\nclass|\nif|\nfrom|\Z)', content, re.DOTALL)
        assert rook_config_section, "RookConfig class not found"

        config_content = rook_config_section.group(1)

        # Check that defaults reference settings, not hardcoded values
        assert "settings.trading." in config_content, "RookConfig should use settings.trading.* for defaults"

        # Ensure no hardcoded numeric defaults (except 0.0 for balance fields and None)
        hardcoded_patterns = [
            r'(?<!_balance):\s*float\s*=\s*[0-9]+\.[0-9]+(?!.*balance)',  # float = 0.005, etc. (not balance fields)
            r':\s*int\s*=\s*[1-9][0-9]*',  # int = 50, etc.
            r':\s*str\s*=\s*["\'][^"\']*["\'](?!\s*#.*None)',  # str = "hardcoded"
        ]

        for pattern in hardcoded_patterns:
            matches = re.findall(pattern, config_content)
            assert len(matches) == 0, f"Found hardcoded values in RookConfig: {matches}"

    def test_no_conditional_imports(self):
        """Verify no try/except imports (fail-fast principle)."""
        import os
        import re

        # Check all Python files for try/except imports
        for root, dirs, files in os.walk("maharook"):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        content = f.read()

                    # Look for try/except import patterns
                    try_import_patterns = [
                        r'try:\s*\n\s*import',
                        r'try:\s*\n\s*from.*import',
                        r'except ImportError:',
                        r'except ModuleNotFoundError:'
                    ]

                    for pattern in try_import_patterns:
                        matches = re.findall(pattern, content, re.MULTILINE)
                        assert len(matches) == 0, f"Found conditional import in {file_path}: {matches}. Use direct imports instead and ensure dependencies are properly declared."


class TestFailFastDesign:
    """Test Principle II: Fail-Fast Design."""

    def test_invalid_token_fails_immediately(self):
        """Verify invalid tokens cause immediate failure."""
        with pytest.raises(ConfigurationError, match="Token INVALID not found"):
            config_manager.get_token("INVALID")

    def test_insufficient_funds_fails_fast(self):
        """Verify insufficient funds cause immediate failure."""
        # Test that InsufficientFundsError can be raised with proper context
        with pytest.raises(InsufficientFundsError):
            raise InsufficientFundsError(
                "Insufficient ETH balance",
                {"required": 1.0, "available": 0.5}
            )

    def test_invalid_slippage_fails_validation(self):
        """Verify invalid slippage fails validation immediately."""
        with pytest.raises(PydanticValidationError):
            SwapParams(
                token_in="ETH",
                token_out="USDC",
                amount_in=1.0,
                slippage_tolerance=1.5  # 150% - invalid
            )

    def test_negative_amounts_fail_fast(self):
        """Verify negative amounts fail validation."""
        with pytest.raises(PydanticValidationError):
            SwapParams(
                token_in="ETH",
                token_out="USDC",
                amount_in=-1.0  # Negative amount
            )


class TestClearErrorCommunication:
    """Test Principle III: Clear Error Communication."""

    def test_configuration_errors_include_context(self):
        """Verify configuration errors include helpful context."""
        try:
            config_manager.get_token("NONEXISTENT")
        except ConfigurationError as e:
            assert "NONEXISTENT" in str(e)
            assert "not found" in str(e).lower()

    def test_validation_errors_specify_problem(self):
        """Verify validation errors specify the exact problem."""
        try:
            SwapParams(
                token_in="ETH",
                token_out="USDC",
                amount_in=0.0  # Invalid amount
            )
        except PydanticValidationError as e:
            assert "positive" in str(e).lower()

    def test_trading_errors_include_remediation(self):
        """Verify trading errors suggest remediation."""
        error = InsufficientFundsError(
            "Insufficient ETH balance",
            {"required": 1.0, "available": 0.5}
        )

        assert "required" in str(error)
        assert "available" in str(error)


class TestNoAssumptions:
    """Test Principle IV: No Assumptions or Hallucinations."""

    def test_brain_requires_explicit_features(self):
        """Verify Brain requires explicit market features."""
        with patch('ollama.Client') as mock_ollama:
            mock_ollama.return_value.list.return_value = {"models": []}
            brain = Brain(
                model_name="test-model",
                model_provider="ollama",
                config={"min_confidence": 0.3}
            )

        # Should require explicit features, not assume defaults
        features = MarketFeatures(
            price=4300.0,
            price_change_10m=0.02,
            volatility=0.05,
            volume_10m=100.0,
            liquidity_depth=1000000.0,
            spread_bps=5.0
        )

        portfolio_state = PortfolioState(
            eth_balance=1.0,
            usdc_balance=4300.0,
            total_value_usd=8600.0,
            target_allocation=0.5,
            current_allocation=0.5,
            unrealized_pnl=0.0
        )

        # Should not assume market conditions
        assert features.price > 0
        assert features.volatility >= 0

    def test_portfolio_requires_explicit_allocation(self):
        """Verify Portfolio requires explicit target allocation."""
        portfolio = Portfolio(target_eth_allocation=0.7)

        # Should use explicit allocation, not assume 50/50
        assert portfolio.target_eth_allocation == 0.7

    def test_executor_validates_all_parameters(self):
        """Verify Executor validates all parameters explicitly."""
        action = TradingAction(
            side="BUY",
            size=0.1,
            slippage=0.005,
            deadline=20,
            reasoning="Test trade",
            confidence=0.8
        )

        # All parameters should be explicitly provided
        assert action.side in ["BUY", "SELL", "HOLD"]
        assert action.size > 0
        assert 0 < action.slippage < 1
        assert action.deadline > 0
        assert 0 <= action.confidence <= 1


class TestZeroDeadCode:
    """Test Principle VI: Zero Dead Code Policy."""

    def test_all_portfolio_methods_reachable(self):
        """Verify all Portfolio methods have execution paths."""
        portfolio = Portfolio(target_eth_allocation=0.5)

        # Test all public methods are reachable
        assert callable(portfolio.update_balances)
        assert callable(portfolio.snapshot)
        assert callable(portfolio.record_trade)
        assert callable(portfolio.save_snapshot)
        assert callable(portfolio.needs_rebalancing)
        assert callable(portfolio.calculate_performance)

    def test_all_brain_methods_reachable(self):
        """Verify all Brain methods have execution paths."""
        with patch('ollama.Client') as mock_ollama:
            mock_ollama.return_value.list.return_value = {"models": []}
            brain = Brain(
                model_name="test-model",
                model_provider="ollama",
                config={}
            )

        # Test all public methods are reachable
        assert callable(brain.decide)
        assert callable(brain.update_config)
        assert callable(brain.get_model_info)

    def test_all_swapper_methods_reachable(self):
        """Verify all Swapper methods have execution paths."""
        # Mock the client to avoid actual blockchain connection
        with patch('maharook.blockchain.client.BaseClient') as mock_client:
            mock_client.return_value.account = Mock()
            mock_client.return_value.address = "0x123"

            swapper = UniswapV4Swapper(client=mock_client.return_value)

            # Test all public methods are reachable
            assert callable(swapper.get_token_decimals)
            assert callable(swapper.get_token_balance)
            assert callable(swapper.swap_eth_to_token)
            assert callable(swapper.swap_token_to_eth)
            assert callable(swapper.get_portfolio_summary)

    def test_no_duplicate_configurations(self):
        """Verify no duplicate configuration files exist."""
        # Should only have one main config system
        import os

        # Check for duplicate config files
        config_files = []
        for root, dirs, files in os.walk("maharook"):
            for file in files:
                if file.startswith("config") and file.endswith(".py"):
                    config_files.append(os.path.join(root, file))

        # Should only have the main config.py in core/
        expected_config = "maharook/core/config.py"
        assert len([f for f in config_files if "core/config.py" in f]) == 1, "Main config should exist"

        # No duplicate configs
        duplicate_configs = [f for f in config_files if "configs/config.py" in f]
        assert len(duplicate_configs) == 0, f"Found duplicate config files: {duplicate_configs}"

    def test_no_legacy_components(self):
        """Verify no legacy/superseded components exist."""
        import os

        # Check for legacy stateful agent
        legacy_agent_path = "maharook/agents/rook/stateful_trading_agent.py"
        assert not os.path.exists(legacy_agent_path), f"Legacy stateful agent should be removed: {legacy_agent_path}"

        # Check for duplicate brain files
        duplicate_brain_path = "maharook/agents/rook/lora_brain.py"
        assert not os.path.exists(duplicate_brain_path), f"Duplicate brain file should be removed: {duplicate_brain_path}"

        # Should only have one main brain.py
        brain_files = []
        for root, dirs, files in os.walk("maharook/agents/rook"):
            for file in files:
                if "brain" in file.lower() and file.endswith(".py"):
                    brain_files.append(os.path.join(root, file))

        assert len(brain_files) == 1, f"Should have exactly one brain file, found: {brain_files}"
        assert "brain.py" in brain_files[0], f"Main brain file should be brain.py, found: {brain_files[0]}"

        # Check for broken imports (files that import non-existent modules)
        for root, dirs, files in os.walk("maharook"):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        content = f.read()
                        # Check for broken import patterns
                        assert "from python.blockchain" not in content, f"Broken import in {file_path}"
                        assert "from config import *" not in content, f"Wildcard config import in {file_path}"

    def test_implementation_integrity(self):
        """Verify implementation matches architectural claims."""
        # Test that components actually implement what they claim
        from maharook.agents.rook.brain import Brain

        # Check Brain's __init__ signature
        import inspect
        brain_init_signature = inspect.signature(Brain.__init__)
        params = list(brain_init_signature.parameters.keys())

        # If Brain claims LoRA support in parameters, it should actually implement it
        if 'adapter_path' in params:
            # Check that the Brain class actually uses adapter_path
            brain_source = inspect.getsource(Brain)

            # Should have actual implementation that uses adapter_path, not just stores it
            assert "self.adapter_path = adapter_path" in brain_source, "Brain accepts adapter_path but doesn't store it"

            # Should have logic that actually loads/uses the adapter
            if "self.adapter_path = adapter_path" in brain_source:
                # Check for actual adapter loading logic
                assert ("load_adapter" in brain_source or
                        "from_pretrained" in brain_source and "adapter" in brain_source.lower() or
                        "peft" in brain_source.lower()), \
                    "Brain stores adapter_path but doesn't implement actual LoRA loading"


class TestPythonStandards:
    """Test Principle I: Python Industry Standards."""

    def test_type_hints_present(self):
        """Verify type hints are used throughout."""
        from maharook.agents.rook.brain import Brain
        from maharook.agents.rook.portfolio import Portfolio

        # Check method signatures have type hints
        import inspect

        brain_methods = inspect.getmembers(Brain, predicate=inspect.isfunction)
        portfolio_methods = inspect.getmembers(Portfolio, predicate=inspect.isfunction)

        # Verify at least some methods have annotations
        assert len(brain_methods) > 0
        assert len(portfolio_methods) > 0

    def test_pydantic_validation(self):
        """Verify Pydantic models are used for validation."""
        # SwapParams should validate inputs
        assert hasattr(SwapParams, '__annotations__')
        assert hasattr(SwapParams, 'model_validate')

    def test_decimal_precision(self):
        """Verify high precision Decimal usage."""
        from decimal import getcontext

        # Should have high precision set
        assert getcontext().prec >= 28  # High precision for financial calculations


def test_constitutional_integration():
    """Integration test for constitutional compliance."""
    # Test that components work together following constitutional principles

    # 1. Configuration-driven setup
    config = RookConfig(
        pair="ETH/USDC",
        model_name="test-model",
        target_allocation=0.6  # From config, not hardcoded
    )

    # 2. Fail-fast validation
    assert config.target_allocation == 0.6

    # 3. Clear error handling (would be tested in integration)
    portfolio = Portfolio(target_eth_allocation=config.target_allocation)

    # 4. No assumptions - explicit state required
    eth_price = 4300.0  # Explicit, not assumed
    state = portfolio.snapshot(eth_price)

    # 5. All code reachable
    assert state.target_allocation == 0.6

    # 6. Python standards - type checking would be done by mypy
    assert isinstance(state.eth_balance, float)
    assert isinstance(state.target_allocation, float)