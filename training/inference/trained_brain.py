"""
Trained ROOK Brain Module
------------------------
Supports loading and inference with trained ROOK models.
Extends the base Brain with trained model capabilities.
"""

import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

from .brain import Brain, TradingAction, MarketFeatures, PortfolioState


class ROOKModelPyTorch(nn.Module):
    """PyTorch implementation of ROOK model (matches training script)."""

    def __init__(self, config: Dict[str, Any], metadata: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.metadata = metadata

        # Input dimensions
        self.core_dim = metadata['n_core_features']
        self.condition_dim = metadata['n_condition_features']
        self.seq_len = metadata['seq_len']

        # Hidden dimensions
        self.hidden_dim = 256
        self.intermediate_dim = 128

        # Core sequence encoder (LSTM + attention)
        self.core_encoder = nn.LSTM(
            input_size=self.core_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        # Condition encoder
        self.condition_encoder = nn.Sequential(
            nn.Linear(self.condition_dim, self.intermediate_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.intermediate_dim, self.intermediate_dim)
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim + self.intermediate_dim, self.intermediate_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Output heads
        self.side_head = nn.Linear(self.intermediate_dim, 3)  # BUY, SELL, HOLD
        self.size_head = nn.Linear(self.intermediate_dim, 1)
        self.slippage_head = nn.Linear(self.intermediate_dim, 1)
        self.deadline_head = nn.Linear(self.intermediate_dim, 1)

    def forward(self, core_features: torch.Tensor, conditions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        batch_size = core_features.size(0)

        # Encode core sequences
        core_output, (hidden, _) = self.core_encoder(core_features)
        # Use last hidden state
        core_repr = hidden[-1]  # Shape: (batch_size, hidden_dim)

        # Encode conditions
        condition_repr = self.condition_encoder(conditions)

        # Fuse representations
        fused = torch.cat([core_repr, condition_repr], dim=1)
        fused_repr = self.fusion(fused)

        # Generate outputs
        outputs = {
            'side_logits': self.side_head(fused_repr),
            'size': torch.sigmoid(self.size_head(fused_repr)),  # Bound to [0,1]
            'slippage': torch.relu(self.slippage_head(fused_repr)),  # Positive values
            'deadline': torch.relu(self.deadline_head(fused_repr))   # Positive values
        }

        return outputs


class TrainedROOKBrain(Brain):
    """Brain that uses trained ROOK models for inference."""

    def __init__(
        self,
        model_path: str,
        pair: str = "ETH_USDC",
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize with trained model.

        Args:
            model_path: Path to trained model directory
            pair: Trading pair name
            config: Additional configuration
        """
        self.model_path = Path(model_path)
        self.pair = pair
        self.config = config or {}

        # Load model artifacts
        self._load_model_artifacts()

        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the trained model
        self._initialize_trained_model()

        # Feature buffer for sequence building
        self.feature_buffer = []

        logger.info("TrainedROOKBrain initialized for {} with model from {}",
                   pair, model_path)

    def _load_model_artifacts(self):
        """Load model metadata, config, and normalization parameters."""
        # Load metadata
        metadata_path = self.model_path / "metadata.json"
        if not metadata_path.exists():
            # Try to find metadata in parent data directory
            data_dir = self.model_path.parent / "data" / "processed"
            metadata_path = data_dir / "metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Cannot find metadata.json for model at {self.model_path}")

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Load model config
        config_path = self.model_path / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.model_config = json.load(f)
        else:
            # Default config
            self.model_config = {
                'model_name': 'trained_rook',
                'seq_len': self.metadata['seq_len']
            }

        # Load normalization parameters
        norm_path = self.model_path / "normalization.json"
        if not norm_path.exists():
            # Try to find in data directory
            data_dir = self.model_path.parent / "data" / "processed"
            norm_path = data_dir / "normalization.json"

        if norm_path.exists():
            with open(norm_path, 'r') as f:
                self.normalization = json.load(f)
        else:
            logger.warning("No normalization parameters found - using identity scaling")
            self.normalization = None

        logger.info("Loaded model artifacts: seq_len={}, features={}+{}",
                   self.metadata['seq_len'],
                   self.metadata['n_core_features'],
                   self.metadata['n_condition_features'])

    def _initialize_trained_model(self):
        """Initialize and load the trained PyTorch model."""
        # Create model instance
        self.model = ROOKModelPyTorch(self.model_config, self.metadata)

        # Load trained weights
        model_file = self.model_path / "best_model.pt"
        if not model_file.exists():
            model_file = self.model_path / "final_model.pt"

        if not model_file.exists():
            raise FileNotFoundError(f"No trained model found at {self.model_path}")

        # Load checkpoint info if available
        checkpoint_info_path = self.model_path / "checkpoint_info.json"
        checkpoint_info = {}
        if checkpoint_info_path.exists():
            with open(checkpoint_info_path, 'r') as f:
                checkpoint_info = json.load(f)

        # Load state dict
        try:
            state_dict = torch.load(model_file, map_location=self.device)
            self.model.load_state_dict(state_dict)

            epoch_info = f"epoch {checkpoint_info.get('epoch', 'unknown')}" if checkpoint_info else "unknown epoch"
            logger.info("Loaded model state dict from {}", epoch_info)
        except Exception as e:
            logger.error("Failed to load model: {}", e)
            raise

        self.model.to(self.device)
        self.model.eval()

        # Side class mapping
        self.side_classes = self.metadata.get('side_classes', ['BUY', 'HOLD', 'SELL'])

        logger.info("Trained model loaded successfully")

    def decide(
        self,
        features: MarketFeatures,
        portfolio_state: PortfolioState,
        market_state: Dict[str, Any]
    ) -> TradingAction:
        """Make trading decision using trained model."""
        try:
            # Convert features to model input format
            core_features, conditions = self._prepare_model_input(
                features, portfolio_state, market_state
            )

            # Run inference
            with torch.no_grad():
                outputs = self.model(core_features, conditions)

            # Convert outputs to trading action
            action = self._parse_model_output(outputs, features, portfolio_state)

            logger.debug("Trained model decision: {} {} ETH (conf: {:.2f})",
                        action.side, action.size, action.confidence)

            return action

        except Exception as e:
            logger.error("Error in trained model inference: {}", e)
            # Fallback to safe action
            return TradingAction(
                side="HOLD",
                size=0.0,
                slippage=0.005,
                deadline=300,
                reasoning=f"Model inference error: {e}",
                confidence=0.0
            )

    def _prepare_model_input(
        self,
        features: MarketFeatures,
        portfolio_state: PortfolioState,
        market_state: Dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare features for model input."""
        # Extract core features in same order as training
        current_features = [
            features.price,
            features.volatility,
            features.volume_10m,
            features.volume_10m,  # volume_ma (using same as volume for now)
            features.liquidity_depth / 1000.0,  # liquidity_impact approximation
            features.price_change_10m / 100.0,  # price_change
        ]

        # Add to buffer
        self.feature_buffer.append(current_features)

        # Maintain sequence length
        seq_len = self.metadata['seq_len']
        if len(self.feature_buffer) > seq_len:
            self.feature_buffer = self.feature_buffer[-seq_len:]

        # Pad if needed
        while len(self.feature_buffer) < seq_len:
            # Pad with last available features or zeros
            if len(self.feature_buffer) > 0:
                self.feature_buffer.insert(0, self.feature_buffer[0])
            else:
                self.feature_buffer.append([0.0] * len(current_features))

        # Create core sequence
        core_sequence = np.array(self.feature_buffer[-seq_len:])

        # Normalize if parameters available
        if self.normalization:
            core_mean = np.array(self.normalization['core_scaler_mean'])
            core_scale = np.array(self.normalization['core_scaler_scale'])
            core_sequence = (core_sequence - core_mean) / core_scale

        # Create condition features (mock for now)
        condition_features = self._create_condition_features(
            features, portfolio_state, market_state
        )

        # Convert to tensors
        core_tensor = torch.FloatTensor(core_sequence).unsqueeze(0).to(self.device)  # (1, seq_len, n_features)
        condition_tensor = torch.FloatTensor(condition_features).unsqueeze(0).to(self.device)  # (1, n_conditions)

        return core_tensor, condition_tensor

    def _create_condition_features(
        self,
        features: MarketFeatures,
        portfolio_state: PortfolioState,
        market_state: Dict[str, Any]
    ) -> np.ndarray:
        """Create condition features for current state."""
        # Mock condition features - in practice these would be computed from recent history
        conditions = [
            features.price,  # avg_price_10
            features.volatility * 0.1,  # price_volatility_10
            features.volume_10m,  # avg_volume_10
            0.0,  # avg_tick_change_10
            60.0,  # avg_time_diff_10
            features.liquidity_depth / 1000.0,  # liquidity_impact_avg_10
            features.price,  # avg_price_50
            features.volatility * 0.15,  # price_volatility_50
            features.volume_10m * 0.8,  # avg_volume_50
            0.0,  # avg_tick_change_50
            120.0,  # avg_time_diff_50
            features.liquidity_depth / 1200.0,  # liquidity_impact_avg_50
            features.price,  # avg_price_100
            features.volatility * 0.2,  # price_volatility_100
            features.volume_10m * 0.6,  # avg_volume_100
            0.0,  # avg_tick_change_100
            180.0,  # avg_time_diff_100
            features.liquidity_depth / 1500.0,  # liquidity_impact_avg_100
            100.0,  # total_transactions
            features.price_change_10m / 100.0,  # price_trend
            1.0,  # volume_trend
            features.price,  # current_price
            features.liquidity_depth,  # current_liquidity
        ]

        condition_array = np.array(conditions)

        # Normalize if parameters available
        if self.normalization:
            condition_mean = np.array(self.normalization['conditions_scaler_mean'])
            condition_scale = np.array(self.normalization['conditions_scaler_scale'])
            condition_array = (condition_array - condition_mean) / condition_scale

        return condition_array

    def _parse_model_output(
        self,
        outputs: Dict[str, torch.Tensor],
        features: MarketFeatures,
        portfolio_state: PortfolioState
    ) -> TradingAction:
        """Parse model output into TradingAction."""
        # Extract predictions
        side_logits = outputs['side_logits'].cpu().numpy()[0]
        size_pred = outputs['size'].cpu().numpy()[0, 0]
        slippage_pred = outputs['slippage'].cpu().numpy()[0, 0]
        deadline_pred = outputs['deadline'].cpu().numpy()[0, 0]

        # Convert side prediction
        side_idx = np.argmax(side_logits)
        side = self.side_classes[side_idx]
        side_confidence = torch.softmax(outputs['side_logits'], dim=1).cpu().numpy()[0, side_idx]

        # Scale and bound predictions
        size = np.clip(size_pred, 0.01, 0.1)  # 1% to 10%
        slippage_bps = max(10.0, min(100.0, slippage_pred * 100))  # 10-100 bps
        deadline_minutes = max(60, min(1800, deadline_pred * 60))  # 1-30 minutes

        return TradingAction(
            side=side,
            size=size,
            slippage=slippage_bps / 10000,  # Convert bps to decimal
            deadline=int(deadline_minutes),
            reasoning=f"Trained model prediction: side_conf={side_confidence:.3f}",
            confidence=side_confidence
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_type": "trained_rook",
            "model_path": str(self.model_path),
            "pair": self.pair,
            "seq_len": self.metadata['seq_len'],
            "core_features": self.metadata['n_core_features'],
            "condition_features": self.metadata['n_condition_features'],
            "side_classes": self.side_classes,
            "config": self.config
        }