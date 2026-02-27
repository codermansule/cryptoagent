"""
Feature Engineering Pipeline
Transforms raw OHLCV + indicators into a clean ML-ready feature matrix.

Design principles:
  - All features are stationary (returns, ratios, z-scores — NOT raw prices)
  - Leakage-free: only information available at bar close is used
  - NaN rows are dropped after feature construction
  - Labels: forward returns over N bars, thresholded to {+1, 0, -1}
  
Enhanced with:
  - Sentiment features (Fear & Greed, news, Twitter, Reddit)
  - On-chain features (funding rates, OI, liquidations, BTC dominance)
  - Cross-exchange features
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from src.signals.technical.indicators import compute_all


# ── Label Generation ──────────────────────────────────────────────────────────

def make_labels(
    df: pd.DataFrame,
    horizon: int = 4,
    threshold: float = 0.005,
) -> pd.Series:
    """
    Generate directional labels from future returns.

    Args:
        horizon:   number of bars ahead to measure return
        threshold: minimum absolute return to generate a directional label
                   (returns within ±threshold → label 0, i.e. "flat")

    Returns:
        Series of int: +1 (long), -1 (short), 0 (flat/skip)
        Last `horizon` rows will be NaN (no future available).
    """
    fwd_return = df["close"].shift(-horizon) / df["close"] - 1
    labels = pd.Series(0, index=df.index, dtype=int)
    labels[fwd_return >  threshold] = 1
    labels[fwd_return < -threshold] = -1
    labels[fwd_return.isna()] = np.nan
    return labels


# ── Feature Construction ──────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a feature DataFrame from raw OHLCV data.

    Steps:
      1. Compute all technical indicators via compute_all()
      2. Normalise price-level features to stationary forms
      3. Add lagged features (t-1, t-2, t-3) for key signals
      4. Add cross-timeframe ratio proxies
      5. Drop NaN rows

    Returns a DataFrame with only numeric feature columns (no raw price/volume).
    """
    ind = compute_all(df)

    feat = pd.DataFrame(index=df.index)

    # ── Price action ──────────────────────────────────────────────────────
    feat["returns_1"]  = ind["returns"]
    feat["returns_3"]  = df["close"].pct_change(3)
    feat["returns_6"]  = df["close"].pct_change(6)
    feat["returns_12"] = df["close"].pct_change(12)
    feat["log_ret_1"]  = ind["log_returns"]
    feat["hl_ratio"]   = ind["hl_ratio"]
    feat["co_ratio"]   = ind["co_ratio"]

    # ── Trend features (normalised) ───────────────────────────────────────
    feat["ema9_dist"]  = (df["close"] - ind["ema9"])  / df["close"]
    feat["ema21_dist"] = (df["close"] - ind["ema21"]) / df["close"]
    feat["ema50_dist"] = (df["close"] - ind["ema50"]) / df["close"]
    feat["ema200_dist"]= (df["close"] - ind["ema200"])/ df["close"]
    feat["ema9_21_spread"]  = (ind["ema9"]  - ind["ema21"]) / df["close"]
    feat["ema21_50_spread"] = (ind["ema21"] - ind["ema50"]) / df["close"]
    feat["ema_cross"]  = ind["ema_cross"]             # integer {-1, 0, 1}
    feat["supertrend_dir"] = ind["supertrend_dir"]    # integer {-1, 0, 1}
    feat["adx"]        = ind["adx"] / 100.0           # normalise to [0,1]
    feat["plus_di"]    = ind["plus_di"] / 100.0
    feat["minus_di"]   = ind["minus_di"] / 100.0
    feat["di_spread"]  = (ind["plus_di"] - ind["minus_di"]) / 100.0

    # ── Momentum features ─────────────────────────────────────────────────
    feat["rsi14"]      = ind["rsi14"] / 100.0
    feat["rsi_ob"]     = (ind["rsi14"] > 70).astype(int)
    feat["rsi_os"]     = (ind["rsi14"] < 30).astype(int)
    feat["rsi_div"]    = ind["rsi_div"]
    feat["macd_hist"]  = ind["macd_hist"]
    feat["macd_hist_norm"] = ind["macd_hist"] / (df["close"].rolling(20).std().replace(0, np.nan))
    feat["macd_cross"] = np.sign(ind["macd_hist"]).diff().fillna(0).astype(int)
    feat["srsi_k"]     = ind["srsi_k"] / 100.0
    feat["srsi_d"]     = ind["srsi_d"] / 100.0
    feat["srsi_kd"]    = feat["srsi_k"] - feat["srsi_d"]

    # ── Volatility features ───────────────────────────────────────────────
    feat["atr_pct"]    = ind["atr14"] / df["close"]
    feat["bb_pct_b"]   = ind["bb_pct_b"]
    feat["bb_bw"]      = ind["bb_bandwidth"]
    feat["bb_squeeze"] = ind["bb_squeeze"]
    # Relative ATR: compare current ATR to its own moving average
    atr_ma = ind["atr14"].rolling(20).mean()
    feat["atr_ratio"]  = ind["atr14"] / atr_ma.replace(0, np.nan)

    # ── Volume features ───────────────────────────────────────────────────
    feat["vwap_dist"]  = (df["close"] - ind["vwap"]) / df["close"]
    vol_ma = df["volume"].rolling(20).mean()
    feat["vol_ratio"]  = df["volume"] / vol_ma.replace(0, np.nan)
    feat["cvd_norm"]   = ind["cvd"].diff() / (df["volume"].rolling(20).mean().replace(0, np.nan))
    feat["obv_diff"]   = ind["obv"].diff() / (df["volume"].rolling(20).mean().replace(0, np.nan))

    # ── Lagged features (t-1, t-2, t-3) ──────────────────────────────────
    lag_cols = ["returns_1", "rsi14", "macd_hist_norm", "atr_pct",
                "ema_cross", "supertrend_dir", "bb_pct_b", "vwap_dist"]
    for col in lag_cols:
        if col in feat.columns:
            feat[f"{col}_lag1"] = feat[col].shift(1)
            feat[f"{col}_lag2"] = feat[col].shift(2)
            feat[f"{col}_lag3"] = feat[col].shift(3)

    # ── Rolling statistics on returns ─────────────────────────────────────
    ret = ind["returns"]
    feat["ret_std_10"]   = ret.rolling(10).std()
    feat["ret_std_20"]   = ret.rolling(20).std()
    feat["ret_skew_20"]  = ret.rolling(20).skew()
    feat["ret_kurt_20"]  = ret.rolling(20).kurt()
    feat["ret_sum_5"]    = ret.rolling(5).sum()
    feat["ret_sum_10"]   = ret.rolling(10).sum()
    feat["pos_ret_frac"] = (ret > 0).rolling(10).mean()   # fraction of up bars

    # ── Replace inf with NaN, then forward-fill up to 2 bars ─────────────
    feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    feat.ffill(limit=2, inplace=True)

    return feat


def add_sentiment_features(df: pd.DataFrame, sentiment_data: dict = None) -> pd.DataFrame:
    """
    Add sentiment and on-chain features to the feature matrix.
    
    Args:
        df: Feature DataFrame
        sentiment_data: Dict with keys:
            - fear_greed: FearGreedIndex value (0-100)
            - news_sentiment: float (-1 to 1)
            - twitter_sentiment: float (-1 to 1)
            - reddit_sentiment: float (-1 to 1)
            - funding_diff: float (cross-exchange funding diff)
            - oi_change: float (24h OI change %)
            - liq_dominance: float (-1 to 1, negative=longs liquidated)
            - btc_dominance: float (0-100)
            - altcoin_season: float (0-100)
    
    Returns:
        DataFrame with added sentiment features
    """
    sentiment_data = sentiment_data or {}
    
    feat = df.copy()
    
    feat["fear_greed_norm"] = sentiment_data.get("fear_greed", 50) / 100.0
    feat["fear_greed_extreme"] = ((sentiment_data.get("fear_greed", 50) < 25) | 
                                   (sentiment_data.get("fear_greed", 50) > 75)).astype(int)
    
    feat["news_sentiment"] = sentiment_data.get("news_sentiment", 0)
    feat["twitter_sentiment"] = sentiment_data.get("twitter_sentiment", 0)
    feat["reddit_sentiment"] = sentiment_data.get("reddit_sentiment", 0)
    
    feat["social_sentiment_avg"] = (
        feat["news_sentiment"] * 0.4 +
        feat["twitter_sentiment"] * 0.3 +
        feat["reddit_sentiment"] * 0.3
    )
    
    feat["funding_diff"] = sentiment_data.get("funding_diff", 0)
    feat["funding_arbitrage"] = (abs(feat["funding_diff"]) > 0.001).astype(int)
    
    feat["oi_change"] = sentiment_data.get("oi_change", 0) / 100.0
    feat["oi_high"] = (abs(feat["oi_change"]) > 0.1).astype(int)
    
    feat["liq_dominance"] = sentiment_data.get("liq_dominance", 0)
    feat["liq_extreme"] = (abs(feat["liq_dominance"]) > 0.5).astype(int)
    
    feat["btc_dominance"] = sentiment_data.get("btc_dominance", 50) / 100.0
    feat["altcoin_season"] = sentiment_data.get("altcoin_season", 50) / 100.0
    feat["btc_dominance_rising"] = (sentiment_data.get("btc_dominance", 50) > 55).astype(int)
    feat["altcoin_season_active"] = (sentiment_data.get("altcoin_season", 50) > 80).astype(int)
    
    feat["sentiment_regime"] = (
        (feat["fear_greed_norm"] < 0.3).astype(int) * 1 +
        (feat["fear_greed_norm"] > 0.7).astype(int) * -1 +
        feat["social_sentiment_avg"] * 2
    )
    
    feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    feat.ffill(limit=2, inplace=True)
    
    return feat


def get_all_feature_names() -> list[str]:
    """Return list of all possible feature names including sentiment features."""
    base_features = [
        "returns_1", "returns_3", "returns_6", "returns_12", "log_ret_1",
        "hl_ratio", "co_ratio", "ema9_dist", "ema21_dist", "ema50_dist",
        "ema200_dist", "ema9_21_spread", "ema21_50_spread", "ema_cross",
        "supertrend_dir", "adx", "plus_di", "minus_di", "di_spread",
        "rsi14", "rsi_ob", "rsi_os", "rsi_div", "macd_hist", "macd_hist_norm",
        "macd_cross", "srsi_k", "srsi_d", "srsi_kd", "atr_pct", "bb_pct_b",
        "bb_bw", "bb_squeeze", "atr_ratio", "vwap_dist", "vol_ratio",
        "cvd_norm", "obv_diff", "ret_std_10", "ret_std_20", "ret_skew_20",
        "ret_kurt_20", "ret_sum_5", "ret_sum_10", "pos_ret_frac",
    ]
    
    lagged_features = [f"{col}_lag{i}" for col in [
        "returns_1", "rsi14", "macd_hist_norm", "atr_pct",
        "ema_cross", "supertrend_dir", "bb_pct_b", "vwap_dist"
    ] for i in range(1, 4)]
    
    sentiment_features = [
        "fear_greed_norm", "fear_greed_extreme", "news_sentiment",
        "twitter_sentiment", "reddit_sentiment", "social_sentiment_avg",
        "funding_diff", "funding_arbitrage", "oi_change", "oi_high",
        "liq_dominance", "liq_extreme", "btc_dominance", "altcoin_season",
        "btc_dominance_rising", "altcoin_season_active", "sentiment_regime"
    ]
    
    return base_features + lagged_features + sentiment_features


def prepare_dataset(
    df: pd.DataFrame,
    horizon: int = 4,
    threshold: float = 0.005,
    drop_flat: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build (X, y) ready for model training.

    Args:
        df:         Raw OHLCV DataFrame
        horizon:    Forward bars for label generation
        threshold:  Directional threshold
        drop_flat:  If True, remove rows where label == 0 (trains only on directional)

    Returns:
        (X, y) — both have the same index, NaN rows removed
    """
    features = build_features(df)
    labels   = make_labels(df, horizon, threshold)

    combined = features.copy()
    combined["__label__"] = labels

    # Drop rows with any NaN (indicator warm-up + last horizon bars)
    combined.dropna(inplace=True)

    if drop_flat:
        combined = combined[combined["__label__"] != 0]

    y = combined.pop("__label__").astype(int)
    X = combined

    return X, y


# ── Feature Names (for model persistence) ────────────────────────────────────

def get_feature_names(df_sample: pd.DataFrame) -> list[str]:
    """Return the ordered list of feature column names."""
    X, _ = prepare_dataset(df_sample.head(200))
    return list(X.columns)
