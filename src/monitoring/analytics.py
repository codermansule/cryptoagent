"""
Performance Analytics
Computes trading performance metrics from a list of trade records or a P&L series.
Used by both the dashboard and reporting scripts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 365 * 24 * 4) -> float:
    """Annualised Sharpe ratio (assumes zero risk-free rate)."""
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, periods_per_year: int = 365 * 24 * 4) -> float:
    """Annualised Sortino ratio."""
    downside = returns[returns < 0]
    if len(downside) < 2 or downside.std() == 0:
        return 0.0
    return float(returns.mean() / downside.std() * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    """Maximum drawdown as a negative fraction (e.g. -0.15 = -15%)."""
    if len(equity) < 2:
        return 0.0
    peak = equity.cummax()
    dd   = (equity - peak) / peak
    return float(dd.min())


def calmar_ratio(equity: pd.Series) -> float:
    """Calmar = total return / abs(max drawdown)."""
    total_ret = (equity.iloc[-1] / equity.iloc[0] - 1) if equity.iloc[0] else 0
    mdd       = abs(max_drawdown(equity))
    return float(total_ret / mdd) if mdd > 0 else 0.0


def compute_summary(
    equity_curve: pd.Series,
    trade_log: Optional[pd.DataFrame] = None,
    periods_per_year: int = 365 * 24 * 4,
) -> dict:
    """
    Compute a full performance summary dict.
    equity_curve: index = datetime or int, values = equity in USDC
    trade_log: DataFrame with columns pnl, pnl_pct, exit_reason (optional)
    """
    ret = equity_curve.pct_change().dropna()

    summary = {
        "initial_equity":   float(equity_curve.iloc[0]),
        "final_equity":     float(equity_curve.iloc[-1]),
        "total_return_pct": float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1),
        "max_drawdown_pct": max_drawdown(equity_curve),
        "sharpe_ratio":     sharpe_ratio(ret, periods_per_year),
        "sortino_ratio":    sortino_ratio(ret, periods_per_year),
        "calmar_ratio":     calmar_ratio(equity_curve),
        "volatility_ann":   float(ret.std() * np.sqrt(periods_per_year)),
    }

    if trade_log is not None and len(trade_log) > 0:
        pnls  = trade_log["pnl"].values
        wins  = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        summary.update({
            "total_trades":     len(trade_log),
            "win_rate":         len(wins) / len(trade_log),
            "avg_win_usdc":     float(wins.mean())   if len(wins)   > 0 else 0.0,
            "avg_loss_usdc":    float(losses.mean()) if len(losses) > 0 else 0.0,
            "expectancy_usdc":  float(pnls.mean()),
            "profit_factor":    float(abs(wins.sum() / losses.sum()))
                                if losses.sum() != 0 else float("inf"),
            "tp_pct":           float((trade_log["exit_reason"] == "tp").mean())
                                if "exit_reason" in trade_log.columns else 0.0,
            "sl_pct":           float((trade_log["exit_reason"] == "sl").mean())
                                if "exit_reason" in trade_log.columns else 0.0,
        })

    return summary
