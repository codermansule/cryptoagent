"""
Integration tests for the trading agent.
These tests verify the interaction between components.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch


class TestEnsembleSignal:
    """Test ensemble signal computation."""
    
    @pytest.fixture
    def sample_data(self):
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

    def test_ensemble_signal_output(self, sample_data):
        """Test that ensemble signal produces valid output."""
        from src.signals.ensemble import compute_ensemble
        
        signal = compute_ensemble(
            df=sample_data,
            symbol="BTC-USDC",
            timeframe="1h",
            ml_signal=1,
            ml_confidence=0.7,
            min_confidence=55.0,
        )
        
        # Check that signal has expected attributes
        assert hasattr(signal, "direction")
        assert hasattr(signal, "confidence")
        assert hasattr(signal, "regime")
        assert hasattr(signal, "breakdown")
        
        # Direction should be -1, 0, or 1
        assert signal.direction in [-1, 0, 1]
        
        # Confidence should be 0-100
        assert 0 <= signal.confidence <= 100

    def test_ensemble_with_insufficient_data(self, sample_data):
        """Test ensemble with insufficient data."""
        from src.signals.ensemble import compute_ensemble
        
        # Use only 10 rows
        small_df = sample_data.head(10)
        
        signal = compute_ensemble(
            df=small_df,
            symbol="BTC-USDC",
            timeframe="1h",
        )
        
        # Should return zero signal for insufficient data
        assert signal.direction == 0


class TestConfluence:
    """Test multi-timeframe confluence."""
    
    @pytest.fixture
    def sample_signals(self):
        """Create sample ensemble signals."""
        from src.signals.ensemble import EnsembleSignal
        
        return [
            EnsembleSignal("BTC-USDC", "15m", 1, 65.0, "trending_up"),
            EnsembleSignal("BTC-USDC", "1h", 1, 60.0, "trending_up"),
            EnsembleSignal("BTC-USDC", "4h", 1, 70.0, "trending_up"),
        ]

    def test_confluence_valid(self, sample_signals):
        """Test valid confluence detection."""
        from src.decision.confluence import compute_confluence
        
        # Use explicit htf/ltf that match the test signals
        result = compute_confluence(
            sample_signals, 
            min_agreeing=2,
            htf_timeframes=["1h", "4h", "1D"],
            ltf_timeframes=["5m", "15m", "30m"],
        )
        
        assert result is not None
        assert result.direction == 1
        assert result.htf_aligned
        assert result.ltf_trigger

    def test_confluence_no_agreement(self, sample_signals):
        """Test when signals don't agree."""
        from src.signals.ensemble import EnsembleSignal
        from src.decision.confluence import compute_confluence
        
        mixed_signals = [
            EnsembleSignal("BTC-USDC", "15m", 1, 65.0, "trending_up"),
            EnsembleSignal("BTC-USDC", "1h", -1, 60.0, "trending_down"),
        ]
        
        result = compute_confluence(mixed_signals, min_agreeing=2)
        
        # Should return None when directions are tied
        assert result is None


class TestPaperEngine:
    """Test paper trading engine."""
    
    def test_paper_engine_initialization(self):
        """Test paper engine initializes correctly."""
        from src.execution.paper import PaperEngine
        from unittest.mock import patch
        
        with patch('src.execution.paper.get_settings') as mock_settings:
            mock_settings.return_value.paper_starting_balance = 10000
            mock_settings.return_value.paper_fee_pct = 0.05
            mock_settings.return_value.paper_slippage_pct = 0.02
            mock_settings.return_value.default_leverage = 5
            
            engine = PaperEngine()
            
            assert engine._balance_usdt == 10000
            assert len(engine._positions) == 0

    def test_place_market_order(self):
        """Test placing a market order."""
        from src.execution.paper import PaperEngine
        from src.exchanges.base import OrderSide, OrderType
        from unittest.mock import patch
        
        with patch('src.execution.paper.get_settings') as mock_settings:
            mock_settings.return_value.paper_starting_balance = 10000
            mock_settings.return_value.paper_fee_pct = 0.05
            mock_settings.return_value.paper_slippage_pct = 0.02
            mock_settings.return_value.default_leverage = 5
            
            engine = PaperEngine()
            engine.update_price("BTC-USDC", 50000)
            
            order = engine.place_order(
                symbol="BTC-USDC",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                size=0.1,  # 0.1 BTC
                price=50000,
            )
            
            assert order is not None
            assert order.status.value == "filled"
            assert len(engine._positions) == 1

    def test_stop_loss_trigger(self):
        """Test stop loss triggers correctly."""
        from src.execution.paper import PaperEngine
        from src.exchanges.base import OrderSide, OrderType
        from unittest.mock import patch
        
        with patch('src.execution.paper.get_settings') as mock_settings:
            mock_settings.return_value.paper_starting_balance = 10000
            mock_settings.return_value.paper_fee_pct = 0.05
            mock_settings.return_value.paper_slippage_pct = 0.02
            mock_settings.return_value.default_leverage = 5
            
            engine = PaperEngine()
            engine.update_price("BTC-USDC", 50000)
            
            # Place order with SL
            engine.place_order(
                symbol="BTC-USDC",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                size=0.1,
                price=50000,
                sl_price=49500,  # 1% stop loss
            )
            
            # Simulate price drop to trigger SL
            engine.update_price("BTC-USDC", 49400)
            
            # Position should be closed
            assert len(engine._positions) == 0


class TestDecisionEngine:
    """Test decision engine."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n = 100
        
        dates = pd.date_range("2024-01-01", periods=n, freq="15min")
        close = 50000 + np.cumsum(np.random.randn(n) * 50)
        high = close + np.random.uniform(0, 100, n)
        low = close - np.random.uniform(0, 100, n)
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

    @pytest.mark.asyncio
    async def test_decision_engine_initialization(self):
        """Test decision engine initializes correctly."""
        from src.decision.core import DecisionEngine
        
        engine = DecisionEngine(
            min_confidence=60.0,
            max_open_positions=5,
            max_single_risk_pct=0.5,
        )
        
        assert engine.min_confidence == 60.0
        assert engine.max_open_positions == 5
        assert engine.max_single_risk_pct == 0.5

    @pytest.mark.asyncio
    async def test_on_candle_processes_data(self, sample_data):
        """Test that on_candle processes data correctly."""
        from src.decision.core import DecisionEngine
        
        engine = DecisionEngine(
            min_confidence=60.0,
            max_open_positions=5,
        )
        
        # This should not raise
        await engine.on_candle("BTC-USDC", "15m", sample_data)
        
        # Buffer should be populated
        assert "BTC-USDC" in engine._buffers
        assert "15m" in engine._buffers["BTC-USDC"]
