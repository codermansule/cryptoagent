"""
Technical Indicators Library
Pure pandas/numpy implementation — no numba/ta-lib dependency required.
All functions accept a pandas DataFrame with columns: open, high, low, close, volume
and return a Series or DataFrame of indicator values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Trend ─────────────────────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def ema_crossover(df: pd.DataFrame, fast: int = 9, slow: int = 21) -> pd.DataFrame:
    """Returns fast EMA, slow EMA, and crossover signal (+1 bull, -1 bear, 0 none)."""
    fast_ema = ema(df["close"], fast)
    slow_ema = ema(df["close"], slow)
    prev_fast = fast_ema.shift(1)
    prev_slow = slow_ema.shift(1)
    signal = pd.Series(0, index=df.index, dtype=int)
    signal[(fast_ema > slow_ema) & (prev_fast <= prev_slow)] = 1   # bullish cross
    signal[(fast_ema < slow_ema) & (prev_fast >= prev_slow)] = -1  # bearish cross
    return pd.DataFrame({"ema_fast": fast_ema, "ema_slow": slow_ema, "cross": signal})


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """Supertrend indicator. Returns trend direction (+1/-1) and supertrend line."""
    hl2 = (df["high"] + df["low"]) / 2
    atr_val = atr(df, period)
    upper_band = hl2 + multiplier * atr_val
    lower_band = hl2 - multiplier * atr_val

    supertrend_vals = pd.Series(np.nan, index=df.index)
    direction = pd.Series(0, index=df.index, dtype=int)
    in_uptrend = pd.Series(True, index=df.index)

    for i in range(1, len(df)):
        if df["close"].iloc[i] > upper_band.iloc[i - 1]:
            in_uptrend.iloc[i] = True
        elif df["close"].iloc[i] < lower_band.iloc[i - 1]:
            in_uptrend.iloc[i] = False
        else:
            in_uptrend.iloc[i] = in_uptrend.iloc[i - 1]

        if in_uptrend.iloc[i]:
            supertrend_vals.iloc[i] = max(lower_band.iloc[i], supertrend_vals.iloc[i - 1] if not np.isnan(supertrend_vals.iloc[i - 1]) else lower_band.iloc[i])
            direction.iloc[i] = 1
        else:
            supertrend_vals.iloc[i] = min(upper_band.iloc[i], supertrend_vals.iloc[i - 1] if not np.isnan(supertrend_vals.iloc[i - 1]) else upper_band.iloc[i])
            direction.iloc[i] = -1

    return pd.DataFrame({"supertrend": supertrend_vals, "direction": direction})


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average Directional Index (ADX) with +DI and -DI."""
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0

    tr = atr(df, period, return_raw=True)
    atr_s = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr_s)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr_s)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx_val = dx.rolling(period).mean()
    return pd.DataFrame({"adx": adx_val, "plus_di": plus_di, "minus_di": minus_di})


# ── Momentum ──────────────────────────────────────────────────────────────────

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def rsi_divergence(df: pd.DataFrame, rsi_period: int = 14, lookback: int = 5) -> pd.Series:
    """
    Detect RSI divergence.
    Returns: +1 (bullish div), -1 (bearish div), 0 (none)
    """
    rsi_vals = rsi(df["close"], rsi_period)
    signal = pd.Series(0, index=df.index, dtype=int)
    for i in range(lookback, len(df)):
        price_slice = df["close"].iloc[i - lookback:i + 1]
        rsi_slice = rsi_vals.iloc[i - lookback:i + 1]
        # Bullish divergence: price makes lower low but RSI makes higher low
        if price_slice.iloc[-1] < price_slice.min() and rsi_slice.iloc[-1] > rsi_slice.min():
            signal.iloc[i] = 1
        # Bearish divergence: price makes higher high but RSI makes lower high
        elif price_slice.iloc[-1] > price_slice.max() and rsi_slice.iloc[-1] < rsi_slice.max():
            signal.iloc[i] = -1
    return signal


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal_period: int = 9) -> pd.DataFrame:
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": histogram})


def stoch_rsi(series: pd.Series, rsi_period: int = 14, stoch_period: int = 14,
              k_period: int = 3, d_period: int = 3) -> pd.DataFrame:
    rsi_vals = rsi(series, rsi_period)
    min_rsi = rsi_vals.rolling(stoch_period).min()
    max_rsi = rsi_vals.rolling(stoch_period).max()
    stoch = (rsi_vals - min_rsi) / (max_rsi - min_rsi).replace(0, np.nan) * 100
    k = stoch.rolling(k_period).mean()
    d = k.rolling(d_period).mean()
    return pd.DataFrame({"k": k, "d": d})


# ── Volatility ────────────────────────────────────────────────────────────────

def atr(df: pd.DataFrame, period: int = 14, return_raw: bool = False) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    if return_raw:
        return tr
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    mid = sma(series, period)
    std = series.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    bandwidth = (upper - lower) / mid
    pct_b = (series - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame({"upper": upper, "mid": mid, "lower": lower,
                         "bandwidth": bandwidth, "pct_b": pct_b})


def bb_squeeze(df: pd.DataFrame, bb_period: int = 20, kc_period: int = 20,
               kc_mult: float = 1.5) -> pd.Series:
    """
    Bollinger Band squeeze: BB inside Keltner Channel.
    Returns True when market is in squeeze (low volatility coiling).
    """
    bb = bollinger_bands(df["close"], bb_period)
    kc_mid = ema(df["close"], kc_period)
    kc_range = kc_mult * atr(df, kc_period)
    kc_upper = kc_mid + kc_range
    kc_lower = kc_mid - kc_range
    return (bb["upper"] < kc_upper) & (bb["lower"] > kc_lower)


# ── Volume ────────────────────────────────────────────────────────────────────

def vwap(df: pd.DataFrame) -> pd.Series:
    """Session VWAP (resets each day for intraday frames)."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    cum_vol = df["volume"].cumsum()
    cum_tp_vol = (typical * df["volume"]).cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


def cvd(df: pd.DataFrame) -> pd.Series:
    """
    Cumulative Volume Delta — proxy from candle data.
    Positive CVD = buyers dominate, Negative = sellers.
    """
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    buy_vol = df["volume"] * (df["close"] - df["low"]) / candle_range
    sell_vol = df["volume"] * (df["high"] - df["close"]) / candle_range
    delta = (buy_vol - sell_vol).fillna(0)
    return delta.cumsum()


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(df["close"].diff()).fillna(0)
    return (df["volume"] * direction).cumsum()


# ── Composite Signals ─────────────────────────────────────────────────────────

def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all indicators on a OHLCV DataFrame.
    Returns a DataFrame with all indicator columns appended.
    """
    out = df.copy()

    # Trend
    ema_df = ema_crossover(df, 9, 21)
    out["ema9"] = ema_df["ema_fast"]
    out["ema21"] = ema_df["ema_slow"]
    out["ema50"] = ema(df["close"], 50)
    out["ema200"] = ema(df["close"], 200)
    out["ema_cross"] = ema_df["cross"]

    st = supertrend(df, 10, 3.0)
    out["supertrend"] = st["supertrend"]
    out["supertrend_dir"] = st["direction"]

    adx_df = adx(df, 14)
    out["adx"] = adx_df["adx"]
    out["plus_di"] = adx_df["plus_di"]
    out["minus_di"] = adx_df["minus_di"]

    # Momentum
    out["rsi14"] = rsi(df["close"], 14)
    out["rsi_div"] = rsi_divergence(df, 14)
    macd_df = macd(df["close"])
    out["macd"] = macd_df["macd"]
    out["macd_signal"] = macd_df["signal"]
    out["macd_hist"] = macd_df["hist"]
    srsi = stoch_rsi(df["close"])
    out["srsi_k"] = srsi["k"]
    out["srsi_d"] = srsi["d"]

    # Volatility
    out["atr14"] = atr(df, 14)
    bb = bollinger_bands(df["close"])
    out["bb_upper"] = bb["upper"]
    out["bb_mid"] = bb["mid"]
    out["bb_lower"] = bb["lower"]
    out["bb_bandwidth"] = bb["bandwidth"]
    out["bb_pct_b"] = bb["pct_b"]
    out["bb_squeeze"] = bb_squeeze(df).astype(int)

    # Volume
    out["vwap"] = vwap(df)
    out["cvd"] = cvd(df)
    out["obv"] = obv(df)

    # Price features
    out["returns"] = df["close"].pct_change()
    out["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    out["hl_ratio"] = (df["high"] - df["low"]) / df["close"]
    out["co_ratio"] = (df["close"] - df["open"]) / df["open"]

    return out
