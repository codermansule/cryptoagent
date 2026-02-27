"""
LightGBM Directional Classifier
Predicts next-bar direction: +1 (long), -1 (short), 0 (flat).

Features:
  - Trained on prepared feature matrix from features.py
  - Walk-forward validation support
  - Probability output for confidence scoring in ensemble
  - Incremental re-training on new data
  - Model persistence (save/load with joblib)
  - Feature importance logging
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import classification_report, accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from src.data.preprocessing.features import prepare_dataset, build_features

logger = logging.getLogger(__name__)


# ── Default hyper-parameters ──────────────────────────────────────────────────

DEFAULT_PARAMS = {
    "objective":        "multiclass",
    "num_class":        3,
    "metric":           "multi_logloss",
    "num_leaves":       63,
    "learning_rate":    0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "min_child_samples": 20,
    "n_estimators":     300,
    "n_jobs":           -1,
    "verbosity":        -1,
    "random_state":     42,
}

# LightGBM encodes classes as 0,1,2 — map back to -1,0,+1
CLASS_MAP = {0: -1, 1: 0, 2: 1}
LABEL_MAP = {-1: 0,  0: 1,  1: 2}   # inverse: raw label → lgbm class index


# ── Classifier ────────────────────────────────────────────────────────────────

class DirectionalClassifier:
    """
    Wraps a LightGBM multiclass classifier for directional price prediction.

    Usage:
        clf = DirectionalClassifier()
        clf.train(df_history)
        signal, confidence = clf.predict(df_recent)
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        params: Optional[dict] = None,
        horizon: int = 4,
        threshold: float = 0.005,
    ):
        if not LGBM_AVAILABLE:
            raise RuntimeError("lightgbm not installed. Run: pip install lightgbm")
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn not installed. Run: pip install scikit-learn")

        self.params    = {**DEFAULT_PARAMS, **(params or {})}
        self.horizon   = horizon
        self.threshold = threshold
        self.model: Optional[lgb.LGBMClassifier] = None
        self.feature_names: list[str] = []
        self.model_path = Path(model_path) if model_path else None
        self._label_encoder = LabelEncoder()

        if self.model_path and self.model_path.exists():
            self.load(self.model_path)

    # ── Training ──────────────────────────────────────────────────────────

    def train(
        self,
        df: pd.DataFrame,
        drop_flat: bool = False,
        eval_split: float = 0.2,
        verbose: bool = True,
    ) -> dict:
        """
        Train the classifier on historical OHLCV data.

        Args:
            df:          Historical OHLCV DataFrame (min ~500 bars recommended)
            drop_flat:   Remove flat-label samples (trains only on directional moves)
            eval_split:  Fraction of tail data to use for evaluation
            verbose:     Print classification report

        Returns:
            dict with training metrics
        """
        X, y = prepare_dataset(df, self.horizon, self.threshold, drop_flat)
        if len(X) < 100:
            logger.warning("Insufficient training samples (%d). Need at least 100.", len(X))
            return {}

        self.feature_names = list(X.columns)

        # Map labels to LightGBM class indices
        y_encoded = y.map(LABEL_MAP).astype(int)

        # Temporal split (no random shuffle for time-series)
        split_idx = int(len(X) * (1 - eval_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y_encoded.iloc[:split_idx], y_encoded.iloc[split_idx:]

        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False),
                       lgb.log_evaluation(-1)],
        )

        # Evaluation
        y_pred = self.model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)

        # Convert back to readable labels for reporting
        y_val_readable  = [CLASS_MAP[c] for c in y_val.tolist()]
        y_pred_readable = [CLASS_MAP[c] for c in y_pred.tolist()]

        if verbose:
            logger.info("Classifier trained. Val accuracy: %.3f", acc)
            report = classification_report(y_val_readable, y_pred_readable,
                                           labels=[-1, 0, 1],
                                           target_names=["short", "flat", "long"],
                                           zero_division=0)
            logger.info("\n%s", report)

        metrics = {"val_accuracy": acc, "n_train": len(X_train), "n_val": len(X_val)}

        if self.model_path:
            self.save(self.model_path)

        return metrics

    def walk_forward_train(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        drop_flat: bool = False,
    ) -> list[dict]:
        """
        Walk-forward cross-validation.
        Returns per-fold metrics. The model is retrained on full data at the end.
        """
        X, y = prepare_dataset(df, self.horizon, self.threshold, drop_flat)
        if len(X) < 200:
            logger.warning("Insufficient data for walk-forward validation.")
            return []

        y_encoded = y.map(LABEL_MAP).astype(int)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_vl = y_encoded.iloc[train_idx], y_encoded.iloc[val_idx]

            m = lgb.LGBMClassifier(**self.params)
            m.fit(X_tr, y_tr,
                  eval_set=[(X_vl, y_vl)],
                  callbacks=[lgb.early_stopping(20, verbose=False),
                              lgb.log_evaluation(-1)])
            pred = m.predict(X_vl)
            acc  = accuracy_score(y_vl, pred)
            fold_metrics.append({"fold": fold + 1, "val_accuracy": acc,
                                  "n_train": len(X_tr), "n_val": len(X_vl)})
            logger.info("Fold %d/%d  val_acc=%.3f", fold + 1, n_splits, acc)

        # Retrain on full data
        self.feature_names = list(X.columns)
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X, y_encoded,
                       callbacks=[lgb.log_evaluation(-1)])
        if self.model_path:
            self.save(self.model_path)

        return fold_metrics

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> tuple[int, float]:
        """
        Predict direction for the latest completed bar.

        Returns:
            (signal, confidence) where:
              signal     ∈ {-1, 0, +1}
              confidence ∈ [0.0, 1.0]  — max class probability
        """
        if self.model is None:
            return 0, 0.0

        features = build_features(df)
        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        features.ffill(limit=2, inplace=True)

        # Use the last complete row (iloc[-1])
        row = features.iloc[-1]

        # Align to training feature set
        if self.feature_names:
            missing = [c for c in self.feature_names if c not in row.index]
            extra   = [c for c in row.index if c not in self.feature_names]
            for c in missing:
                row[c] = 0.0
            row = row[self.feature_names]

        if row.isna().any():
            return 0, 0.0

        X_pred = pd.DataFrame([row])
        proba  = self.model.predict_proba(X_pred)[0]   # shape [3]
        cls    = int(np.argmax(proba))
        conf   = float(proba[cls])
        signal = CLASS_MAP[cls]

        return signal, conf

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict on all rows of df (for backtesting / analysis).

        Returns DataFrame with columns: signal, confidence, prob_short, prob_flat, prob_long
        """
        if self.model is None:
            return pd.DataFrame()

        features = build_features(df)
        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        features.dropna(inplace=True)

        if self.feature_names:
            for c in self.feature_names:
                if c not in features.columns:
                    features[c] = 0.0
            features = features[self.feature_names]

        proba  = self.model.predict_proba(features)    # shape [n, 3]
        cls    = np.argmax(proba, axis=1)
        conf   = proba[np.arange(len(cls)), cls]
        signals = [CLASS_MAP[c] for c in cls]

        result = pd.DataFrame({
            "signal":     signals,
            "confidence": conf,
            "prob_short": proba[:, 0],
            "prob_flat":  proba[:, 1],
            "prob_long":  proba[:, 2],
        }, index=features.index)

        return result

    # ── Feature Importance ────────────────────────────────────────────────

    def feature_importance(self, top_n: int = 20) -> pd.Series:
        """Return top_n features by LightGBM gain importance."""
        if self.model is None or not self.feature_names:
            return pd.Series(dtype=float)
        imp = pd.Series(
            self.model.booster_.feature_importance(importance_type="gain"),
            index=self.feature_names,
        ).sort_values(ascending=False)
        return imp.head(top_n)

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        if not JOBLIB_AVAILABLE:
            logger.warning("joblib not installed, cannot save model.")
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model":         self.model,
            "feature_names": self.feature_names,
            "params":        self.params,
            "horizon":       self.horizon,
            "threshold":     self.threshold,
        }
        joblib.dump(payload, path)
        logger.info("Model saved to %s", path)

    def load(self, path: str | Path) -> None:
        if not JOBLIB_AVAILABLE:
            raise RuntimeError("joblib not installed. Run: pip install joblib")
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        payload = joblib.load(path)
        self.model         = payload["model"]
        self.feature_names = payload["feature_names"]
        self.params        = payload["params"]
        self.horizon       = payload["horizon"]
        self.threshold     = payload["threshold"]
        logger.info("Model loaded from %s  (%d features)", path, len(self.feature_names))


# ── Convenience factory ───────────────────────────────────────────────────────

def load_or_train(
    df: pd.DataFrame,
    model_path: str | Path = "models/lgbm_classifier.joblib",
    force_retrain: bool = False,
    **kwargs,
) -> DirectionalClassifier:
    """
    Load a saved model if it exists, otherwise train from scratch.

    Args:
        df:            Historical OHLCV data for training if needed
        model_path:    Where to save/load the model
        force_retrain: Always retrain even if model file exists
    """
    path = Path(model_path)
    clf  = DirectionalClassifier(model_path=path, **kwargs)

    if force_retrain or clf.model is None:
        logger.info("Training new DirectionalClassifier on %d bars...", len(df))
        clf.train(df)

    return clf
