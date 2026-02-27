"""
Unit tests for technical indicators.
"""

import pytest
import pandas as pd
import numpy as np
from src.signals.technical.indicators import (
    ema, sma, rsi, macd, atr, bollinger_bands, vwap, cvd, obv,
    ema_crossover, supertrend, adx, stoch_rsi, bb_squeeze, compute_all,
)


@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data."""
    np.random.seed(42)
    n = 100
    
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    close = 50000 + np.cumsum(np.random.randn(n) * 100)
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


def test_ema(sample_ohlcv):
    """Test EMA calculation."""
    result = ema(sample_ohlcv["close"], 20)
    assert len(result) == len(sample_ohlcv)
    assert not result.isna().all()


def test_sma(sample_ohlcv):
    """Test SMA calculation."""
    result = sma(sample_ohlcv["close"], 20)
    assert len(result) == len(sample_ohlcv)


def test_rsi(sample_ohlcv):
    """Test RSI calculation."""
    result = rsi(sample_ohlcv["close"], 14)
    assert len(result) == len(sample_ohlcv)
    # RSI should be between 0 and 100
    valid = result.dropna()
    assert valid.min() >= 0
    assert valid.max() <= 100


def test_macd(sample_ohlcv):
    """Test MACD calculation."""
    result = macd(sample_ohlcv["close"])
    assert "macd" in result.columns
    assert "signal" in result.columns
    assert "hist" in result.columns


def test_atr(sample_ohlcv):
    """Test ATR calculation."""
    result = atr(sample_ohlcv, 14)
    assert len(result) == len(sample_ohlcv)
    # ATR should be non-negative for valid values (NaN in warmup period is OK)
    valid = result.dropna()
    assert len(valid) > 0
    assert (valid >= 0).all()


def test_bollinger_bands(sample_ohlcv):
    """Test Bollinger Bands calculation."""
    result = bollinger_bands(sample_ohlcv["close"])
    assert "upper" in result.columns
    assert "mid" in result.columns
    assert "lower" in result.columns
    assert "bandwidth" in result.columns
    assert "pct_b" in result.columns


def test_vwap(sample_ohlcv):
    """Test VWAP calculation."""
    result = vwap(sample_ohlcv)
    assert len(result) == len(sample_ohlcv)
    assert (result > 0).all()


def test_cvd(sample_ohlcv):
    """Test Cumulative Volume Delta."""
    result = cvd(sample_ohlcv)
    assert len(result) == len(sample_ohlcv)


def test_obv(sample_ohlcv):
    """Test On-Balance Volume."""
    result = obv(sample_ohlcv)
    assert len(result) == len(sample_ohlcv)


def test_ema_crossover(sample_ohlcv):
    """Test EMA crossover."""
    result = ema_crossover(sample_ohlcv, 9, 21)
    assert "ema_fast" in result.columns
    assert "ema_slow" in result.columns
    assert "cross" in result.columns


def test_supertrend(sample_ohlcv):
    """Test Supertrend indicator."""
    result = supertrend(sample_ohlcv, 10, 3.0)
    assert "supertrend" in result.columns
    assert "direction" in result.columns


def test_adx(sample_ohlcv):
    """Test ADX calculation."""
    result = adx(sample_ohlcv, 14)
    assert "adx" in result.columns
    assert "plus_di" in result.columns
    assert "minus_di" in result.columns


def test_stoch_rsi(sample_ohlcv):
    """Test Stochastic RSI."""
    result = stoch_rsi(sample_ohlcv["close"])
    assert "k" in result.columns
    assert "d" in result.columns


def test_bb_squeeze(sample_ohlcv):
    """Test Bollinger Band squeeze."""
    result = bb_squeeze(sample_ohlcv)
    assert len(result) == len(sample_ohlcv)


def test_compute_all(sample_ohlcv):
    """Test compute_all function."""
    result = compute_all(sample_ohlcv)
    assert len(result) == len(sample_ohlcv)
    # Should have many indicator columns added
    assert len(result.columns) > len(sample_ohlcv.columns)
