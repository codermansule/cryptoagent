"""
Reinforcement Learning Position Sizing Agent
Uses PPO (Proximal Policy Optimization) to learn optimal position sizing.

The agent learns to:
- Adjust position size based on market regime
- Scale down in volatile conditions
- Scale up in trending conditions with high confidence

Usage:
    sizer = RLPositionSizer()
    await sizer.start()
    size_mult = await sizer.get_position_multiplier(
        confidence=75.0,
        regime="trending_up",
        volatility=0.02,
        portfolio_risk=1.0,
    )
    await sizer.stop()
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.get_logger(__name__)

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False


@dataclass
class PositionSizingState:
    """State representation for the RL agent."""
    confidence: float      # 0-100
    regime_encoded: int   # 0=unknown, 1=trending_up, 2=trending_down, 3=ranging, 4=volatile, 5=breakout
    volatility: float     # ATR / price (normalized)
    portfolio_risk: float # Current risk usage (0-100%)
    recent_return: float  # Last N bars return
    drawdown: float       # Current drawdown (0-100%)


class RLPositionSizer:
    """
    Reinforcement Learning agent for dynamic position sizing.
    
    The agent receives market state and outputs a position size multiplier.
    """

    # State dimensions
    STATE_DIM = 5
    
    # Action: position multiplier (0.0 to 1.5)
    ACTION_DIM = 1

    def __init__(
        self,
        model_path: str = "models/rl_position_sizer.zip",
        train_mode: bool = False,
    ):
        if not RL_AVAILABLE:
            raise RuntimeError(
                "stable-baselines3 not installed. "
                "Run: pip install stable-baselines3"
            )
        
        self.model_path = Path(model_path)
        self.train_mode = train_mode
        self._model: Optional[PPO] = None
        self._env: Optional[DummyVecEnv] = None
        
        # Fallback multipliers when RL is not available or not trained
        self._default_rules = {
            ("trending_up", 0.7): 1.2,
            ("trending_down", 0.7): 1.2,
            ("trending_up", 0.5): 1.0,
            ("trending_down", 0.5): 1.0,
            ("ranging", 0.5): 0.7,
            ("volatile", 0.5): 0.5,
            ("breakout", 0.7): 1.3,
        }

    async def start(self) -> None:
        """Initialize and load the model."""
        if self.model_path.exists():
            try:
                self._model = PPO.load(self.model_path)
                logger.info("RL position sizer loaded from %s", self.model_path)
            except Exception as exc:
                logger.warning("Failed to load RL model: %s", exc)
                self._model = None
        else:
            logger.info("No RL model found at %s, using rule-based fallback", self.model_path)
        
        if self.train_mode and not self.model_path.exists():
            await self._train()

    async def stop(self) -> None:
        """Save model and cleanup."""
        if self._model and self.train_mode:
            self._model.save(self.model_path)
            logger.info("RL position sizer model saved")

    async def get_position_multiplier(
        self,
        confidence: float,
        regime: str,
        volatility: float,
        portfolio_risk: float,
        recent_return: float = 0.0,
        drawdown: float = 0.0,
    ) -> float:
        """
        Get position size multiplier based on current state.
        
        Args:
            confidence: Signal confidence (0-100)
            regime: Market regime (trending_up, trending_down, ranging, volatile, breakout)
            volatility: Normalized volatility (ATR/price)
            portfolio_risk: Current portfolio risk usage (0-100%)
            recent_return: Recent return for momentum signal
            drawdown: Current drawdown percentage
            
        Returns:
            Position size multiplier (0.0 to 1.5)
        """
        if self._model is None:
            return self._get_default_multiplier(
                confidence, regime, volatility, portfolio_risk, drawdown
            )
        
        try:
            # Encode regime
            regime_map = {
                "unknown": 0, "trending_up": 1, "trending_down": 2,
                "ranging": 3, "volatile": 4, "breakout": 5
            }
            regime_encoded = regime_map.get(regime, 0)
            
            # Normalize state
            state = np.array([
                confidence / 100.0,
                regime_encoded / 5.0,
                min(volatility * 10, 1.0),  # Scale volatility
                portfolio_risk / 100.0,
                drawdown / 100.0,
            ], dtype=np.float32)
            
            # Get action from model
            action, _ = self._model.predict(state, deterministic=True)
            multiplier = float(action[0])
            
            # Clamp to valid range
            multiplier = max(0.0, min(1.5, multiplier))
            
            return multiplier
            
        except Exception as exc:
            logger.warning("RL prediction failed: %s", exc)
            return self._get_default_multiplier(
                confidence, regime, volatility, portfolio_risk, drawdown
            )

    def _get_default_multiplier(
        self,
        confidence: float,
        regime: str,
        volatility: float,
        portfolio_risk: float,
        drawdown: float,
    ) -> float:
        """Rule-based fallback position sizing."""
        # Base multiplier from regime
        regime_mult = {
            "trending_up": 1.0,
            "trending_down": 1.0,
            "ranging": 0.7,
            "volatile": 0.5,
            "breakout": 1.2,
            "unknown": 0.5,
        }.get(regime, 0.5)
        
        # Confidence scaling
        conf_mult = confidence / 100.0
        
        # Volatility reduction
        vol_mult = max(0.3, 1.0 - volatility * 5)
        
        # Risk reduction
        risk_mult = max(0.5, 1.0 - portfolio_risk / 100.0)
        
        # Drawdown reduction
        dd_mult = max(0.3, 1.0 - drawdown / 20.0)
        
        # Combined
        multiplier = regime_mult * conf_mult * vol_mult * risk_mult * dd_mult
        
        return max(0.1, min(1.5, multiplier))

    async def _train(self) -> None:
        """Train the RL agent (placeholder - would need historical data)."""
        logger.info("RL training not implemented - requires historical trading data")
        # In production, this would:
        # 1. Generate training episodes from historical data
        # 2. Define reward function (e.g., Sharpe ratio improvement)
        # 3. Train PPO agent
        # 4. Save model

    def add_experience(
        self,
        state: PositionSizingState,
        action: float,
        reward: float,
        done: bool,
    ) -> None:
        """
        Add experience to the replay buffer for online learning.
        
        Note: Would implement experience replay for continual learning.
        """
        # Placeholder for online learning
        pass


async def get_rl_position_sizer() -> RLPositionSizer:
    """Get the global RL position sizer instance."""
    if not hasattr(get_rl_position_sizer, "_instance"):
        get_rl_position_sizer._instance = RLPositionSizer()
        await get_rl_position_sizer._instance.start()
    return get_rl_position_sizer._instance
