"""
Unit tests for regime detection module.
"""

import pytest
import pandas as pd
import numpy as np
from src.decision.regime import detect_regime, regime_multipliers, Regime


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n = 100
    
    # Create trending upward data
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    base_price = 50000
    trend = np.linspace(0, 1000, n)
    noise = np.random.normal(0, 100, n)
    
    close = base_price + trend + noise
    high = close + np.random.uniform(0, 200, n)
    low = close - np.random.uniform(0, 200, n)
    open_price = np.roll(close, 1)
    open_price[0] = close[0]
    volume = np.random.uniform(1000, 5000, n)
    
    return pd.DataFrame({
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)


def test_detect_regime_trending_up(sample_data):
    """Test regime detection in trending market."""
    regime = detect_regime(sample_data)
    # Should detect trending or at least not unknown
    assert regime != Regime.UNKNOWN


def test_detect_regime_insufficient_data():
    """Test regime detection with insufficient data."""
    df = pd.DataFrame({
        "open": [100],
        "high": [105],
        "low": [95],
        "close": [102],
        "volume": [1000],
    })
    regime = detect_regime(df)
    assert regime == Regime.UNKNOWN


def test_regime_multipliers():
    """Test regime multiplier configuration."""
    mult = regime_multipliers(Regime.TRENDING_UP)
    assert "position_size_mult" in mult
    assert "confidence_boost" in mult
    assert "allowed_sides" in mult
    assert "long" in mult["allowed_sides"]
    assert "short" not in mult["allowed_sides"]
    
    mult = regime_multipliers(Regime.TRENDING_DOWN)
    assert "short" in mult["allowed_sides"]
    assert "long" not in mult["allowed_sides"]
    
    mult = regime_multipliers(Regime.VOLATILE)
    assert mult["position_size_mult"] < 1.0


def test_regime_multipliers_unknown():
    """Test unknown regime returns safe defaults."""
    mult = regime_multipliers(Regime.UNKNOWN)
    assert mult["position_size_mult"] <= 0.5
    assert "allowed_sides" in mult
