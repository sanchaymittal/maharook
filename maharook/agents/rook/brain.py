"""
Brain Module for ROOK Agents
----------------------------
Model-agnostic decision-making component that can use LLM prompts or direct model inference.
Supports LoRA adapter loading for specialized trading strategies.
"""

import re
from dataclasses import dataclass
from typing import Any

from loguru import logger

from openai import OpenAI
import ollama
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class TradingAction:
    """Structured trading action output from Brain."""
    side: str  # "BUY", "SELL", "HOLD"
    size: float  # Amount to trade
    slippage: float  # Maximum slippage tolerance
    deadline: int  # Deadline in minutes
    reasoning: str  # Decision rationale
    confidence: float = 0.5  # Confidence score 0-1


@dataclass
class MarketFeatures:
    """Market features for decision making."""
    price: float
    price_change_10m: float
    volatility: float
    volume_10m: float
    liquidity_depth: float
    spread_bps: float


@dataclass
class PortfolioState:
    """Current portfolio state."""
    eth_balance: float
    usdc_balance: float
    total_value_usd: float
    target_allocation: float
    current_allocation: float
    unrealized_pnl: float


class Brain:
    """
    Model-agnostic decision-making brain for ROOK agents.

    Supports:
    - LLM-based prompting (OpenAI, Ollama)
    - Direct model inference with LoRA adapters
    - Configurable decision strategies
    """

    def __init__(
        self,
        model_name: str,
        model_provider: str = "ollama",
        adapter_path: str | None = None,
        config: dict[str, Any] | None = None
    ):
        """Initialize the Brain.

        Args:
            model_name: Name of the base model
            model_provider: Provider (ollama, openrouter, transformers)
            adapter_path: Path to LoRA adapter (optional)
            config: Additional configuration
        """
        self.model_name = model_name
        self.model_provider = model_provider
        self.adapter_path = adapter_path
        self.config = config or {}

        # Initialize model clients
        self.llm_client = None
        self.tokenizer = None
        self.model = None

        self._initialize_model()

        logger.info(
            "Brain initialized: {} via {} {}",
            model_name,
            model_provider,
            f"with adapter {adapter_path}" if adapter_path else ""
        )

    def _initialize_model(self):
        """Initialize the appropriate model based on provider."""
        if self.model_provider == "openrouter":
            self._initialize_openrouter()
        elif self.model_provider == "ollama":
            self._initialize_ollama()
        elif self.model_provider == "transformers":
            self._initialize_transformers()
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")

    def _initialize_openrouter(self):
        """Initialize OpenRouter client."""
        api_key = self.config.get("openrouter_api_key")
        if not api_key:
            raise ValueError("OpenRouter API key required")

        self.llm_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        # Test connection
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
                timeout=15
            )
            logger.info("OpenRouter client initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize OpenRouter: {}", e)
            raise

    def _initialize_ollama(self):
        """Initialize Ollama client."""
        ollama_url = self.config.get("ollama_url", "http://localhost:11434")
        self.llm_client = ollama.Client(host=ollama_url)

        try:
            models = self.llm_client.list()
            logger.info("Ollama client initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize Ollama: {}", e)
            raise

    def _initialize_transformers(self):
        """Initialize Transformers model with optional LoRA adapter."""
        try:
            # Load base model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )

            # Load LoRA adapter if specified
            if self.adapter_path and PeftModel is not None:
                logger.info("Loading LoRA adapter from {}", self.adapter_path)
                self.model = PeftModel.from_pretrained(self.model, self.adapter_path)

            logger.info("Transformers model initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize Transformers model: {}", e)
            raise

    def decide(
        self,
        features: MarketFeatures,
        portfolio_state: PortfolioState,
        market_state: dict[str, Any]
    ) -> TradingAction:
        """Make a trading decision based on current state.

        Args:
            features: Current market features
            portfolio_state: Current portfolio state
            market_state: Additional market context

        Returns:
            TradingAction with decision details
        """
        if self.model_provider in ["openrouter", "ollama"]:
            return self._decide_via_llm(features, portfolio_state, market_state)
        elif self.model_provider == "transformers":
            return self._decide_via_inference(features, portfolio_state, market_state)
        else:
            raise ValueError(f"Unsupported decision method for {self.model_provider}")

    def _decide_via_llm(
        self,
        features: MarketFeatures,
        portfolio_state: PortfolioState,
        market_state: dict[str, Any]
    ) -> TradingAction:
        """Make decision using LLM prompting."""
        prompt = self._build_trading_prompt(features, portfolio_state, market_state)

        messages = [
            {
                "role": "system",
                "content": "You are an expert cryptocurrency trading agent. Analyze market conditions and provide clear trading recommendations."
            },
            {"role": "user", "content": prompt}
        ]

        response = self._get_llm_response(messages)
        action = self._parse_trading_response(response, features, portfolio_state)

        logger.debug("LLM decision: {} {} ETH", action.side, action.size)
        return action

    def _decide_via_inference(
        self,
        features: MarketFeatures,
        portfolio_state: PortfolioState,
        market_state: dict[str, Any]
    ) -> TradingAction:
        """Make decision using direct model inference."""
        # Convert features to model input format
        input_text = self._format_features_for_inference(features, portfolio_state)

        # Tokenize and generate
        inputs = self.tokenizer(input_text, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode and parse response
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(input_text):].strip()

        action = self._parse_trading_response(response, features, portfolio_state)

        logger.debug("Model inference decision: {} {} ETH", action.side, action.size)
        return action

    def _build_trading_prompt(
        self,
        features: MarketFeatures,
        portfolio_state: PortfolioState,
        market_state: dict[str, Any]
    ) -> str:
        """Build comprehensive trading prompt."""
        prompt = f"""
Current Market Conditions:
- ETH Price: ${features.price:.2f}
- Price Change (10min): {features.price_change_10m:.2f}%
- Volatility: {features.volatility * 100:.2f}%
- Volume (10min): {features.volume_10m:.2f} ETH
- Liquidity Depth: ${features.liquidity_depth:.0f}
- Spread: {features.spread_bps:.1f} bps

Current Portfolio:
- ETH Balance: {portfolio_state.eth_balance:.6f} ETH
- USDC Balance: {portfolio_state.usdc_balance:.2f} USDC
- Total Value: ${portfolio_state.total_value_usd:.2f}
- Current ETH Allocation: {portfolio_state.current_allocation:.1%}
- Target ETH Allocation: {portfolio_state.target_allocation:.1%}
- Unrealized P&L: {portfolio_state.unrealized_pnl:+.2%}

Market Context:
{self._format_market_context(market_state)}

Based on this information, should I:
1. BUY more ETH (convert USDC to ETH)
2. SELL ETH (convert ETH to USDC)
3. HOLD current position

Please provide your recommendation with reasoning.
Format your response as:
DECISION: [BUY/SELL/HOLD]
SIZE: [amount in ETH if trading, 0 if HOLD]
SLIPPAGE: [maximum slippage tolerance as decimal, e.g., 0.005 for 0.5%]
DEADLINE: [deadline in minutes, e.g., 20]
CONFIDENCE: [confidence score 0-1]
REASONING: [your analysis]
"""
        return prompt.strip()

    def _format_features_for_inference(
        self,
        features: MarketFeatures,
        portfolio_state: PortfolioState
    ) -> str:
        """Format features for direct model inference."""
        return f"""
Market: price={features.price:.2f} change={features.price_change_10m:.3f} vol={features.volatility:.3f}
Portfolio: eth={portfolio_state.eth_balance:.6f} usdc={portfolio_state.usdc_balance:.2f} alloc={portfolio_state.current_allocation:.3f}
Decision:"""

    def _format_market_context(self, market_state: dict[str, Any]) -> str:
        """Format additional market context."""
        context_items = []

        if "recent_trades" in market_state:
            trades = market_state["recent_trades"]
            context_items.append(f"- Recent trades: {len(trades)} in last hour")

        if "trend" in market_state:
            trend = market_state["trend"]
            context_items.append(f"- Market trend: {trend}")

        if "support_resistance" in market_state:
            sr = market_state["support_resistance"]
            context_items.append(f"- Support: ${sr.get('support', 0):.2f}, Resistance: ${sr.get('resistance', 0):.2f}")

        return "\n".join(context_items) if context_items else "- No additional context available"

    def _get_llm_response(self, messages: list[dict[str, str]]) -> str:
        """Get response from LLM."""
        try:
            if self.model_provider == "openrouter":
                response = self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=500,
                    temperature=0.6,
                    timeout=60
                )
                return response.choices[0].message.content

            elif self.model_provider == "ollama":
                response = self.llm_client.chat(
                    model=self.model_name,
                    messages=messages
                )
                return response['message']['content']

        except Exception as e:
            logger.error("Error getting LLM response: {}", e)
            return "ERROR: Could not get model response"

    def _parse_trading_response(
        self,
        response: str,
        features: MarketFeatures,
        portfolio_state: PortfolioState
    ) -> TradingAction:
        """Parse LLM response into structured TradingAction."""
        try:
            response_upper = response.upper()

            # Extract decision
            decision = "HOLD"
            if "DECISION: BUY" in response_upper:
                decision = "BUY"
            elif "DECISION: SELL" in response_upper:
                decision = "SELL"
            elif "DECISION: HOLD" in response_upper:
                decision = "HOLD"
            else:
                # Fallback parsing
                if "BUY" in response_upper and "ETH" in response_upper:
                    decision = "BUY"
                elif "SELL" in response_upper and "ETH" in response_upper:
                    decision = "SELL"

            # Extract size
            size = 0.0
            size_match = re.search(r"SIZE:\s*([\d.]+)", response, re.IGNORECASE)
            if size_match:
                size = float(size_match.group(1))
            elif decision in ["BUY", "SELL"]:
                # Default to small position size
                if decision == "SELL":
                    size = min(0.01, portfolio_state.eth_balance * 0.1)
                else:  # BUY
                    size = min(0.01, portfolio_state.usdc_balance / features.price * 0.1)

            # Extract slippage
            slippage = 0.005  # Default 0.5%
            slippage_match = re.search(r"SLIPPAGE:\s*([\d.]+)", response, re.IGNORECASE)
            if slippage_match:
                slippage = float(slippage_match.group(1))

            # Extract deadline
            deadline = 20  # Default 20 minutes
            deadline_match = re.search(r"DEADLINE:\s*(\d+)", response, re.IGNORECASE)
            if deadline_match:
                deadline = int(deadline_match.group(1))

            # Extract confidence
            confidence = 0.5  # Default moderate confidence
            confidence_match = re.search(r"CONFIDENCE:\s*([\d.]+)", response, re.IGNORECASE)
            if confidence_match:
                confidence = float(confidence_match.group(1))

            # Extract reasoning
            reasoning_match = re.search(r"REASONING:\s*(.+?)(?:\n|$)", response, re.IGNORECASE | re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else response

            return TradingAction(
                side=decision,
                size=size,
                slippage=slippage,
                deadline=deadline,
                reasoning=reasoning,
                confidence=confidence
            )

        except Exception as e:
            logger.error("Error parsing trading response: {}", e)
            return TradingAction(
                side="HOLD",
                size=0.0,
                slippage=0.005,
                deadline=20,
                reasoning=f"Error parsing response: {e}",
                confidence=0.0
            )

    def update_config(self, new_config: dict[str, Any]):
        """Update brain configuration."""
        self.config.update(new_config)
        logger.info("Brain configuration updated")

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "model_provider": self.model_provider,
            "adapter_path": self.adapter_path,
            "config": self.config
        }
