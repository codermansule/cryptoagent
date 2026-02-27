"""
Portfolio Manager
Manages multi-position portfolios, tracks correlation, and allocates capital across trading strategies.

Features:
- Position tracking across multiple symbols
- Correlation-based risk management
- Dynamic rebalancing
- P&L attribution
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from src.core.config import get_settings
from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Position:
    """Single position tracking."""
    symbol: str
    side: str  # "long" or "short"
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    entry_time: datetime
    stop_loss: float = 0.0
    take_profit: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class PortfolioMetrics:
    """Portfolio-level performance metrics."""
    total_value: float
    cash: float
    positions_value: float
    unrealized_pnl: float
    realized_pnl: float
    daily_pnl: float
    daily_return_pct: float
    max_drawdown_pct: float
    Sharpe_ratio: float
    num_positions: int
    leverage: float


class PortfolioManager:
    """
    Manages the overall portfolio including:
    - Position tracking
    - Risk metrics (correlation, concentration)
    - Capital allocation
    - Rebalancing
    """

    def __init__(self, initial_capital: float = 10000.0):
        self._settings = get_settings()
        self._initial_capital = initial_capital
        self._cash = initial_capital
        self._positions: dict[str, Position] = {}
        self._closed_positions: list[Position] = []
        self._equity_history: list[tuple[datetime, float]] = []
        self._peak_equity = initial_capital
        
        # Correlation tracking
        self._correlation_matrix: Optional[pd.DataFrame] = None
        self._returns_df: Optional[pd.DataFrame] = None
        
        # Risk limits
        self._max_position_size_pct = 0.2  # 20% of portfolio per position
        self._max_correlation = 0.7
        
        logger.info("Portfolio manager initialized", initial_capital=initial_capital)

    # ── Position Management ────────────────────────────────────────────────────

    def add_position(
        self,
        symbol: str,
        side: str,
        size: float,
        entry_price: float,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
    ) -> bool:
        """Add a new position. Returns True if successful."""
        # Check position size limit
        position_value = size * entry_price
        max_value = self._initial_capital * self._max_position_size_pct
        
        if position_value > max_value:
            logger.warning(
                "Position too large",
                symbol=symbol,
                requested=position_value,
                max_allowed=max_value,
            )
            return False
        
        # Check available cash
        if position_value > self._cash:
            logger.warning(
                "Insufficient cash",
                symbol=symbol,
                required=position_value,
                available=self._cash,
            )
            return False
        
        # Add position
        self._positions[symbol] = Position(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            current_price=entry_price,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            entry_time=datetime.now(timezone.utc),
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        
        self._cash -= position_value
        logger.info(
            "Position opened",
            symbol=symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            cash_remaining=self._cash,
        )
        return True

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        reason: str = "manual",
    ) -> float:
        """Close a position. Returns realized P&L."""
        if symbol not in self._positions:
            logger.warning("Position not found", symbol=symbol)
            return 0.0
        
        pos = self._positions[symbol]
        
        # Calculate P&L
        if pos.side == "long":
            pnl = (exit_price - pos.entry_price) * pos.size
        else:
            pnl = (pos.entry_price - exit_price) * pos.size
        
        pos.realized_pnl = pnl
        pos.current_price = exit_price
        pos.unrealized_pnl = 0.0
        
        # Move to closed
        self._closed_positions.append(pos)
        
        # Return capital + P&L to cash
        position_value = pos.size * pos.entry_price
        self._cash += position_value + pnl
        
        logger.info(
            "Position closed",
            symbol=symbol,
            reason=reason,
            pnl=pnl,
            cash=self._cash,
        )
        
        del self._positions[symbol]
        return pnl

    def update_prices(self, prices: dict[str, float]) -> None:
        """Update current prices for all positions."""
        for symbol, price in prices.items():
            if symbol in self._positions:
                pos = self._positions[symbol]
                pos.current_price = price
                
                # Update unrealized P&L
                if pos.side == "long":
                    pos.unrealized_pnl = (price - pos.entry_price) * pos.size
                else:
                    pos.unrealized_pnl = (pos.entry_price - price) * pos.size

    # ── Risk Management ─────────────────────────────────────────────────────

    def check_correlation_risk(self, symbol: str) -> bool:
        """Check if adding a position would create excessive correlation."""
        if not self._positions or self._correlation_matrix is None:
            return True
        
        # Check average correlation with existing positions
        correlations = []
        for existing_symbol in self._positions.keys():
            if existing_symbol in self._correlation_matrix.columns:
                corr = self._correlation_matrix.loc[symbol, existing_symbol]
                correlations.append(corr)
        
        if not correlations:
            return True
        
        avg_corr = np.mean(correlations)
        return avg_corr < self._max_correlation

    def get_concentration_risk(self) -> float:
        """Get current concentration risk (Herfindahl index)."""
        if not self._positions:
            return 0.0
        
        total_value = sum(p.size * p.current_price for p in self._positions.values())
        if total_value == 0:
            return 0.0
        
        weights = [(p.size * p.current_price) / total_value for p in self._positions.values()]
        herfindahl = sum(w ** 2 for w in weights)
        
        return herfindahl  # 0 = diversified, 1 = concentrated

    # ── Metrics ─────────────────────────────────────────────────────────────

    def get_metrics(self) -> PortfolioMetrics:
        """Calculate current portfolio metrics."""
        positions_value = sum(p.size * p.current_price for p in self._positions.values())
        total_value = self._cash + positions_value
        
        unrealized_pnl = sum(p.unrealized_pnl for p in self._positions.values())
        realized_pnl = sum(p.realized_pnl for p in self._closed_positions)
        
        # Daily P&L (simplified - would track from midnight)
        daily_pnl = unrealized_pnl  # Would compare to yesterday
        
        # Daily return
        prev_equity = self._equity_history[-1][1] if self._equity_history else self._initial_capital
        daily_return_pct = (total_value - prev_equity) / prev_equity * 100 if prev_equity else 0
        
        # Max drawdown
        self._peak_equity = max(self._peak_equity, total_value)
        max_dd_pct = (self._peak_equity - total_value) / self._peak_equity * 100 if self._peak_equity else 0
        
        # Sharpe (simplified)
        sharpe = daily_return_pct * np.sqrt(252) / max(abs(daily_pnl), 1)  # Rough approximation
        
        # Leverage
        leverage = positions_value / total_value if total_value > 0 else 0
        
        return PortfolioMetrics(
            total_value=total_value,
            cash=self._cash,
            positions_value=positions_value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            daily_pnl=daily_pnl,
            daily_return_pct=daily_return_pct,
            max_drawdown_pct=max_dd_pct,
            Sharpe_ratio=sharpe,
            num_positions=len(self._positions),
            leverage=leverage,
        )

    def record_equity(self) -> None:
        """Record current equity for historical tracking."""
        total_value = self._cash + sum(p.size * p.current_price for p in self._positions.values())
        self._equity_history.append((datetime.now(timezone.utc), total_value))
        
        # Keep only last 30 days
        if len(self._equity_history) > 30 * 24 * 60:  # Assuming 1-min intervals
            self._equity_history = self._equity_history[-30 * 24 * 60:]

    # ── State ───────────────────────────────────────────────────────────────

    def get_positions(self) -> dict[str, Position]:
        """Get all open positions."""
        return self._positions.copy()

    def get_closed_positions(self) -> list[Position]:
        """Get closed position history."""
        return self._closed_positions.copy()

    def get_cash(self) -> float:
        """Get available cash."""
        return self._cash

    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity history as DataFrame."""
        if not self._equity_history:
            return pd.DataFrame(columns=["time", "equity"])
        
        df = pd.DataFrame(self._equity_history, columns=["time", "equity"])
        df["time"] = pd.to_datetime(df["time"])
        return df


async def get_portfolio_manager() -> PortfolioManager:
    """Get the global portfolio manager instance."""
    if not hasattr(get_portfolio_manager, "_instance"):
        settings = get_settings()
        initial_capital = getattr(settings, 'paper_starting_balance', 10000)
        get_portfolio_manager._instance = PortfolioManager(initial_capital)
    return get_portfolio_manager._instance
