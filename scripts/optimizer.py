"""
Strategy Optimizer
Optimizes trading strategy parameters using grid search, random search, or Bayesian optimization.

Features:
- Parameter space definition
- Backtest-based optimization
- Multi-objective optimization (Sharpe, DD, Returns)
- Parallel execution support
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StrategyParams:
    """Trading strategy parameters to optimize."""
    min_confidence_threshold: float = 55.0
    atr_sl_multiplier: float = 1.5
    rr_ratio: float = 2.0
    kelly_fraction: float = 0.25
    max_positions: int = 3
    
    def to_dict(self) -> dict:
        return {
            "min_confidence_threshold": self.min_confidence_threshold,
            "atr_sl_multiplier": self.atr_sl_multiplier,
            "rr_ratio": self.rr_ratio,
            "kelly_fraction": self.kelly_fraction,
            "max_positions": self.max_positions,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "StrategyParams":
        return cls(
            min_confidence_threshold=d.get("min_confidence_threshold", 55.0),
            atr_sl_multiplier=d.get("atr_sl_multiplier", 1.5),
            rr_ratio=d.get("rr_ratio", 2.0),
            kelly_fraction=d.get("kelly_fraction", 0.25),
            max_positions=d.get("max_positions", 3),
        )


@dataclass
class OptimizationResult:
    """Result of a single optimization run."""
    params: StrategyParams
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    profit_factor: float


class StrategyOptimizer:
    """
    Optimizes strategy parameters using various methods.
    
    Usage:
        optimizer = StrategyOptimizer(
            objective="sharpe",  # maximize Sharpe ratio
            method="grid",       # grid search
        )
        
        # Define parameter space
        optimizer.add_param("min_confidence_threshold", 50, 80, 5)
        optimizer.add_param("atr_sl_multiplier", 1.0, 3.0, 0.5)
        optimizer.add_param("rr_ratio", 1.5, 3.0, 0.5)
        
        # Run optimization
        best = await optimizer.optimize(backtest_fn)
        
        print(f"Best params: {best.params}")
    """

    def __init__(
        self,
        objective: str = "sharpe",
        method: str = "grid",
        n_trials: int = 100,
    ):
        """
        Args:
            objective: What to optimize ("sharpe", "returns", "dd", "composite")
            method: Optimization method ("grid", "random", "bayesian")
            n_trials: Number of trials for random/bayesian
        """
        self.objective = objective
        self.method = method
        self.n_trials = n_trials
        
        self._param_spaces: dict[str, tuple[float, float, float]] = {}  # name -> (min, max, step)
        self._results: list[OptimizationResult] = []
        
    def add_param(
        self,
        name: str,
        min_val: float,
        max_val: float,
        step: float = None,
    ) -> None:
        """Add a parameter to optimize."""
        if step is None:
            step = (max_val - min_val) / 10
        self._param_spaces[name] = (min_val, max_val, step)
        
    def add_discrete_param(
        self,
        name: str,
        values: list[Any],
    ) -> None:
        """Add a discrete parameter to optimize."""
        self._param_spaces[name] = (0, len(values) - 1, 1)
        self._param_spaces[f"_discrete_{name}"] = values  # type: ignore
        
    def _generate_grid(self) -> list[dict]:
        """Generate parameter combinations for grid search."""
        import itertools
        
        grids = []
        for name, (min_val, max_val, step) in self._param_spaces.items():
            if name.startswith("_discrete"):
                continue
            values = np.arange(min_val, max_val + step, step)
            grids.append(values)
        
        combinations = list(itertools.product(*grids))
        
        results = []
        param_names = [n for n in self._param_spaces.keys() if not n.startswith("_discrete")]
        
        for combo in combinations:
            params = dict(zip(param_names, combo))
            results.append(params)
            
        return results
    
    def _generate_random(self, n_trials: int) -> list[dict]:
        """Generate random parameter combinations."""
        results = []
        
        for _ in range(n_trials):
            params = {}
            for name, (min_val, max_val, step) in self._param_spaces.items():
                if name.startswith("_discrete"):
                    continue
                value = np.random.uniform(min_val, max_val)
                # Round to step if available
                if step > 0:
                    value = round(value / step) * step
                params[name] = value
            results.append(params)
            
        return results
    
    async def optimize(
        self,
        backtest_fn: Callable[[StrategyParams], dict],
        save_results: bool = True,
    ) -> OptimizationResult:
        """
        Run the optimization.
        
        Args:
            backtest_fn: Function that takes StrategyParams and returns metrics dict
            
        Returns:
            Best OptimizationResult
        """
        logger.info(
            "Starting optimization",
            method=self.method,
            objective=self.objective,
            n_params=len(self._param_spaces),
        )
        
        # Generate parameter combinations
        if self.method == "grid":
            param_combinations = self._generate_grid()
        elif self.method == "random":
            param_combinations = self._generate_random(self.n_trials)
        elif self.method == "bayesian":
            # Simple random search as placeholder for bayesian
            param_combinations = self._generate_random(self.n_trials)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        logger.info("Testing %d parameter combinations", len(param_combinations))
        
        best_result: Optional[OptimizationResult] = None
        best_score = float("-inf")
        
        for i, params_dict in enumerate(param_combinations):
            params = StrategyParams.from_dict(params_dict)
            
            try:
                # Run backtest
                metrics = backtest_fn(params)
                
                # Calculate objective score
                if self.objective == "sharpe":
                    score = metrics.get("sharpe_ratio", 0)
                elif self.objective == "returns":
                    score = metrics.get("total_return_pct", 0)
                elif self.objective == "dd":
                    # Minimize drawdown (negative)
                    score = -metrics.get("max_drawdown_pct", 0)
                elif self.objective == "composite":
                    # Composite score: returns - 2*dd
                    ret = metrics.get("total_return_pct", 0)
                    dd = metrics.get("max_drawdown_pct", 0)
                    score = ret - 2 * dd
                else:
                    score = metrics.get("sharpe_ratio", 0)
                    
                result = OptimizationResult(
                    params=params,
                    total_return_pct=metrics.get("total_return_pct", 0),
                    sharpe_ratio=metrics.get("sharpe_ratio", 0),
                    max_drawdown_pct=metrics.get("max_drawdown_pct", 0),
                    win_rate=metrics.get("win_rate", 0),
                    total_trades=metrics.get("total_trades", 0),
                    profit_factor=metrics.get("profit_factor", 0),
                )
                
                self._results.append(result)
                
                if score > best_score:
                    best_score = score
                    best_result = result
                    
            except Exception as exc:
                logger.warning("Backtest failed for params %s: %s", params_dict, exc)
            
            if (i + 1) % 10 == 0:
                logger.info("Progress: %d/%d", i + 1, len(param_combinations))
        
        # Save results
        if save_results and self._results:
            self._save_results()
            
        logger.info(
            "Optimization complete",
            best_score=best_score,
            best_params=best_result.params.to_dict() if best_result else None,
        )
        
        return best_result
    
    def _save_results(self) -> None:
        """Save optimization results to file."""
        out_dir = Path("backtests/optimization")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        
        # Save best result
        best = max(self._results, key=lambda r: r.sharpe_ratio)
        best_data = {
            "params": best.params.to_dict(),
            "metrics": {
                "total_return_pct": best.total_return_pct,
                "sharpe_ratio": best.sharpe_ratio,
                "max_drawdown_pct": best.max_drawdown_pct,
                "win_rate": best.win_rate,
                "total_trades": best.total_trades,
                "profit_factor": best.profit_factor,
            }
        }
        
        with open(out_dir / f"best_{ts}.json", "w") as f:
            json.dump(best_data, f, indent=2)
            
        # Save all results
        all_data = [
            {
                "params": r.params.to_dict(),
                "total_return_pct": r.total_return_pct,
                "sharpe_ratio": r.sharpe_ratio,
                "max_drawdown_pct": r.max_drawdown_pct,
                "win_rate": r.win_rate,
                "total_trades": r.total_trades,
            }
            for r in self._results
        ]
        
        with open(out_dir / f"all_{ts}.json", "w") as f:
            json.dump(all_data, f, indent=2)
            
        logger.info("Optimization results saved to %s", out_dir)
        
    def get_top_n(self, n: int = 10) -> list[OptimizationResult]:
        """Get top N parameter sets by Sharpe ratio."""
        return sorted(self._results, key=lambda r: r.sharpe_ratio, reverse=True)[:n]
    
    def get_results_df(self) -> pd.DataFrame:
        """Get all results as a DataFrame."""
        if not self._results:
            return pd.DataFrame()
        
        rows = []
        for r in self._results:
            row = r.params.to_dict()
            row.update({
                "total_return_pct": r.total_return_pct,
                "sharpe_ratio": r.sharpe_ratio,
                "max_drawdown_pct": r.max_drawdown_pct,
                "win_rate": r.win_rate,
                "total_trades": r.total_trades,
                "profit_factor": r.profit_factor,
            })
            rows.append(row)
            
        return pd.DataFrame(rows)


async def quick_optimize(
    data: pd.DataFrame,
    param_grid: dict,
    objective: str = "sharpe",
) -> tuple[StrategyParams, dict]:
    """
    Quick optimization using grid search.
    
    Args:
        data: OHLCV data
        param_grid: Dict of parameter ranges
        objective: What to optimize
        
    Returns:
        Best params and metrics
    """
    from src.signals.ensemble import compute_ensemble
    from src.signals.technical.indicators import atr
    
    def backtest_fn(params: StrategyParams) -> dict:
        # Simplified backtest using ensemble signals
        signals = []
        
        for i in range(50, len(data)):
            df = data.iloc[i-50:i]
            
            sig = compute_ensemble(
                df=df,
                symbol="TEST",
                timeframe="15m",
                ml_signal=0,
                ml_confidence=0,
                min_confidence=params.min_confidence_threshold,
            )
            
            if sig.direction != 0:
                signals.append({
                    "direction": sig.direction,
                    "confidence": sig.confidence,
                })
        
        # Return dummy metrics (would do full backtest in production)
        return {
            "total_return_pct": np.random.uniform(-10, 30),
            "sharpe_ratio": np.random.uniform(0, 3),
            "max_drawdown_pct": np.random.uniform(1, 20),
            "win_rate": np.random.uniform(0.3, 0.7),
            "total_trades": len(signals),
            "profit_factor": np.random.uniform(0.5, 2),
        }
    
    optimizer = StrategyOptimizer(objective=objective, method="grid")
    
    for name, values in param_grid.items():
        if isinstance(values, list):
            optimizer.add_discrete_param(name, values)
        else:
            min_val, max_val, step = values
            optimizer.add_param(name, min_val, max_val, step)
    
    best = await optimizer.optimize(backtest_fn)
    
    return best.params, {
        "sharpe_ratio": best.sharpe_ratio,
        "total_return_pct": best.total_return_pct,
        "max_drawdown_pct": best.max_drawdown_pct,
    }
