"""
LSTM Sequence Model for Directional Price Prediction.

Architecture:
  - 2-layer bidirectional LSTM (hidden=128) with dropout 0.3
  - Fully-connected head → 3-class softmax (short / flat / long)
  - 60-bar lookback window over the full feature matrix from features.py
  - Per-sequence z-score normalisation (avoids look-ahead leakage)

Interface matches DirectionalClassifier:
    clf = LSTMClassifier(model_path="models/lstm.pt")
    clf.train(df_history)
    signal, confidence = clf.predict(df_recent)

Switch from paper → live just by instantiating with a saved model path.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.model_selection import TimeSeriesSplit
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from src.data.preprocessing.features import build_features, make_labels

logger = logging.getLogger(__name__)

# Same label mapping as LightGBM classifier for consistency
CLASS_MAP = {0: -1, 1: 0, 2: 1}   # model index → signal direction
LABEL_MAP = {-1: 0, 0: 1, 1: 2}   # signal direction → model index


# ── PyTorch Module ────────────────────────────────────────────────────────────

class _LSTMNet(nn.Module if TORCH_AVAILABLE else object):
    """
    Two-layer bidirectional LSTM → global average pool → FC → softmax.
    Input:  (batch, seq_len, n_features)
    Output: (batch, 3) log-probabilities
    """

    def __init__(self, n_features: int, hidden: int = 128, layers: int = 2,
                 dropout: float = 0.3) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, 3)   # ×2 for bidirectional

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        out, _ = self.lstm(x)           # (batch, seq, hidden*2)
        out = out.mean(dim=1)           # global average pool over time
        out = self.dropout(out)
        return self.fc(out)             # (batch, 3) — raw logits


# ── Sequence Dataset ──────────────────────────────────────────────────────────

class _SeqDataset(Dataset if TORCH_AVAILABLE else object):
    """Sliding-window sequences over a feature matrix with labels."""

    def __init__(
        self,
        X: np.ndarray,   # (n_bars, n_features) — already normalised per-seq below
        y: np.ndarray,   # (n_bars,) — integer class indices
        seq_len: int,
    ) -> None:
        self.X = X
        self.y = y
        self.seq_len = seq_len
        self.valid = [i for i in range(seq_len - 1, len(X)) if not np.isnan(y[i])]

    def __len__(self) -> int:
        return len(self.valid)

    def __getitem__(self, idx: int):
        end = self.valid[idx]
        start = end - self.seq_len + 1
        seq = self.X[start: end + 1].copy()   # (seq_len, features)

        # Per-sequence z-score normalisation (no look-ahead)
        mu = seq.mean(axis=0, keepdims=True)
        sigma = seq.std(axis=0, keepdims=True) + 1e-8
        seq = (seq - mu) / sigma

        return (
            torch.tensor(seq, dtype=torch.float32),
            torch.tensor(int(self.y[end]), dtype=torch.long),
        )


# ── Main Classifier ───────────────────────────────────────────────────────────

class LSTMClassifier:
    """
    LSTM-based directional classifier with the same interface as
    DirectionalClassifier (LightGBM).

    Usage:
        clf = LSTMClassifier(model_path="models/lstm.pt")
        clf.train(df_history, epochs=30)
        signal, confidence = clf.predict(df_recent)
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        seq_len: int = 60,
        hidden: int = 128,
        layers: int = 2,
        dropout: float = 0.3,
        horizon: int = 4,
        threshold: float = 0.005,
        device: Optional[str] = None,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not installed. Run: pip install torch")

        self.seq_len   = seq_len
        self.hidden    = hidden
        self.layers    = layers
        self.dropout   = dropout
        self.horizon   = horizon
        self.threshold = threshold
        self.model_path = Path(model_path) if model_path else None

        self._device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._net: Optional[_LSTMNet] = None
        self._feature_names: list[str] = []

        if self.model_path and self.model_path.exists():
            self.load(self.model_path)

    # ── Training ──────────────────────────────────────────────────────────

    def train(
        self,
        df: pd.DataFrame,
        epochs: int = 30,
        batch_size: int = 64,
        lr: float = 3e-4,
        eval_split: float = 0.15,
        verbose: bool = True,
    ) -> dict:
        """
        Train the LSTM on historical OHLCV data.

        Args:
            df:         Historical OHLCV DataFrame (min ~300 bars recommended)
            epochs:     Training epochs
            batch_size: Mini-batch size
            lr:         Adam learning rate
            eval_split: Tail fraction used for evaluation
            verbose:    Log epoch losses

        Returns:
            dict with training metrics
        """
        X_raw, y_raw = self._build_arrays(df)
        if X_raw is None or len(X_raw) < self.seq_len + 50:
            logger.warning("Insufficient data for LSTM training (%d bars)", len(df))
            return {}

        n_features = X_raw.shape[1]
        self._net = _LSTMNet(n_features, self.hidden, self.layers, self.dropout)
        self._net.to(self._device)

        # Temporal split
        split = int(len(X_raw) * (1 - eval_split))
        X_tr, X_vl = X_raw[:split], X_raw[split:]
        y_tr, y_vl = y_raw[:split], y_raw[split:]

        train_ds = _SeqDataset(X_tr, y_tr, self.seq_len)
        val_ds   = _SeqDataset(X_vl, y_vl, self.seq_len)

        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.Adam(self._net.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(1, epochs + 1):
            # Train
            self._net.train()
            train_loss = 0.0
            for xb, yb in train_dl:
                xb, yb = xb.to(self._device), yb.to(self._device)
                optimizer.zero_grad()
                logits = self._net(xb)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * len(xb)
            train_loss /= max(len(train_ds), 1)

            # Validate
            val_loss, val_acc = self._evaluate(val_dl, criterion)
            scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self._net.state_dict().items()}

            if verbose and (epoch % 5 == 0 or epoch == 1):
                logger.info(
                    "LSTM epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.3f",
                    epoch, epochs, train_loss, val_loss, val_acc,
                )

        # Restore best checkpoint
        if best_state is not None:
            self._net.load_state_dict({k: v.to(self._device) for k, v in best_state.items()})

        _, final_acc = self._evaluate(val_dl, criterion)
        logger.info("LSTM trained. Best val_loss=%.4f  final_val_acc=%.3f", best_val_loss, final_acc)

        if self.model_path:
            self.save(self.model_path)

        return {
            "best_val_loss": best_val_loss,
            "final_val_acc": final_acc,
            "n_train": len(train_ds),
            "n_val":   len(val_ds),
            "n_features": n_features,
        }

    def walk_forward_train(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        epochs: int = 20,
        batch_size: int = 64,
        lr: float = 3e-4,
    ) -> list[dict]:
        """
        Walk-forward cross-validation. Retrains on full data at the end.
        Returns per-fold metrics.
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available, skipping walk-forward. Falling back to train().")
            result = self.train(df, epochs=epochs, batch_size=batch_size, lr=lr)
            return [result] if result else []

        X_raw, y_raw = self._build_arrays(df)
        if X_raw is None or len(X_raw) < self.seq_len + 100:
            logger.warning("Insufficient data for LSTM walk-forward (%d bars)", len(df))
            return []

        n_features = X_raw.shape[1]
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_metrics = []
        criterion = nn.CrossEntropyLoss()

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_raw)):
            X_tr, X_vl = X_raw[train_idx], X_raw[val_idx]
            y_tr, y_vl = y_raw[train_idx], y_raw[val_idx]

            if len(X_tr) < self.seq_len + 20:
                continue

            net = _LSTMNet(n_features, self.hidden, self.layers, self.dropout).to(self._device)
            opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

            train_dl = DataLoader(_SeqDataset(X_tr, y_tr, self.seq_len),
                                  batch_size=batch_size, shuffle=False)
            val_dl   = DataLoader(_SeqDataset(X_vl, y_vl, self.seq_len),
                                  batch_size=batch_size, shuffle=False)

            for _ in range(epochs):
                net.train()
                for xb, yb in train_dl:
                    xb, yb = xb.to(self._device), yb.to(self._device)
                    opt.zero_grad()
                    loss = criterion(net(xb), yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step()
                sch.step()

            _, acc = self._evaluate(val_dl, criterion, net)
            fold_metrics.append({"fold": fold + 1, "val_accuracy": acc,
                                  "n_train": len(train_idx), "n_val": len(val_idx)})
            logger.info("LSTM fold %d/%d  val_acc=%.3f", fold + 1, n_splits, acc)

        # Retrain on full data
        logger.info("LSTM walk-forward done — retraining on full dataset")
        self.train(df, epochs=epochs, batch_size=batch_size, lr=lr, verbose=False)

        return fold_metrics

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> tuple[int, float]:
        """
        Predict direction for the latest completed bar.

        Returns:
            (signal, confidence) where:
              signal     ∈ {-1, 0, +1}
              confidence ∈ [0.0, 1.0]   — max class probability
        """
        if self._net is None:
            return 0, 0.0

        X_raw, _ = self._build_arrays(df, include_labels=False)
        if X_raw is None or len(X_raw) < self.seq_len:
            return 0, 0.0

        seq = X_raw[-self.seq_len:].copy()     # last seq_len bars

        # Per-sequence z-score (same as training)
        mu    = seq.mean(axis=0, keepdims=True)
        sigma = seq.std(axis=0, keepdims=True) + 1e-8
        seq   = (seq - mu) / sigma

        tensor = torch.tensor(seq[np.newaxis], dtype=torch.float32).to(self._device)

        self._net.eval()
        with torch.no_grad():
            logits = self._net(tensor)                # (1, 3)
            proba  = torch.softmax(logits, dim=1)[0]  # (3,)

        cls    = int(proba.argmax().item())
        conf   = float(proba[cls].item())
        signal = CLASS_MAP[cls]
        return signal, conf

    def predict_batch(self, df: pd.DataFrame, batch_size: int = 256) -> pd.DataFrame:
        """
        Predict on all available rows (for backtesting / analysis).

        Returns DataFrame with columns: signal, confidence, prob_short, prob_flat, prob_long
        """
        if self._net is None:
            return pd.DataFrame()

        X_raw, _ = self._build_arrays(df, include_labels=False)
        if X_raw is None or len(X_raw) < self.seq_len:
            return pd.DataFrame()

        features = build_features(df.copy())
        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        features.ffill(limit=2, inplace=True)
        features.fillna(0, inplace=True)

        if self._feature_names:
            for c in self._feature_names:
                if c not in features.columns:
                    features[c] = 0.0
            features = features[self._feature_names]

        X = features.values.astype(np.float32)
        valid_indices = list(range(self.seq_len - 1, len(X)))

        all_proba = []
        self._net.eval()
        with torch.no_grad():
            for i in range(0, len(valid_indices), batch_size):
                batch_idx = valid_indices[i: i + batch_size]
                seqs = []
                for end in batch_idx:
                    start = end - self.seq_len + 1
                    seq = X[start: end + 1].copy()
                    mu    = seq.mean(axis=0, keepdims=True)
                    sigma = seq.std(axis=0, keepdims=True) + 1e-8
                    seqs.append((seq - mu) / sigma)
                batch_tensor = torch.tensor(np.stack(seqs), dtype=torch.float32).to(self._device)
                logits = self._net(batch_tensor)
                proba  = torch.softmax(logits, dim=1).cpu().numpy()
                all_proba.append(proba)

        if not all_proba:
            return pd.DataFrame()

        proba_all = np.vstack(all_proba)
        cls_all   = np.argmax(proba_all, axis=1)
        conf_all  = proba_all[np.arange(len(cls_all)), cls_all]
        sig_all   = [CLASS_MAP[c] for c in cls_all]

        result_idx = features.index[valid_indices]
        return pd.DataFrame({
            "signal":     sig_all,
            "confidence": conf_all,
            "prob_short": proba_all[:, 0],
            "prob_flat":  proba_all[:, 1],
            "prob_long":  proba_all[:, 2],
        }, index=result_idx)

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        assert self._net is not None, "No model to save — train first"
        payload = {
            "state_dict":    self._net.state_dict(),
            "feature_names": self._feature_names,
            "seq_len":       self.seq_len,
            "hidden":        self.hidden,
            "layers":        self.layers,
            "dropout":       self.dropout,
            "horizon":       self.horizon,
            "threshold":     self.threshold,
        }
        torch.save(payload, path)
        logger.info("LSTM model saved to %s", path)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"LSTM model file not found: {path}")
        payload = torch.load(path, map_location=self._device, weights_only=False)
        self.seq_len       = payload["seq_len"]
        self.hidden        = payload["hidden"]
        self.layers        = payload["layers"]
        self.dropout       = payload["dropout"]
        self.horizon       = payload["horizon"]
        self.threshold     = payload["threshold"]
        self._feature_names = payload["feature_names"]

        n_features = len(self._feature_names)
        self._net = _LSTMNet(n_features, self.hidden, self.layers, self.dropout)
        self._net.load_state_dict(payload["state_dict"])
        self._net.to(self._device)
        self._net.eval()
        logger.info("LSTM model loaded from %s  (%d features, seq=%d)", path, n_features, self.seq_len)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _build_arrays(
        self, df: pd.DataFrame, include_labels: bool = True
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Build raw (X, y) numpy arrays from OHLCV df using feature pipeline."""
        try:
            features = build_features(df.copy())
            features.replace([np.inf, -np.inf], np.nan, inplace=True)
            features.ffill(limit=2, inplace=True)
            features.fillna(0, inplace=True)

            # Store feature names on first call (training)
            if not self._feature_names:
                self._feature_names = list(features.columns)

            # Align to stored feature set
            if self._feature_names:
                for c in self._feature_names:
                    if c not in features.columns:
                        features[c] = 0.0
                features = features[self._feature_names]

            X = features.values.astype(np.float32)

            if not include_labels:
                return X, None

            labels = make_labels(df, horizon=self.horizon, threshold=self.threshold)
            labels = labels.reindex(features.index)

            # Map direction labels → class indices (with NaN preserved as np.nan)
            y = np.full(len(labels), np.nan)
            for idx_i, val in enumerate(labels):
                if not pd.isna(val):
                    y[idx_i] = LABEL_MAP.get(int(val), 1)   # default flat

            return X, y

        except Exception as exc:
            logger.warning("LSTM feature build failed: %s", exc)
            return None, None

    def _evaluate(
        self,
        dataloader: "DataLoader",
        criterion: "nn.Module",
        net: Optional["_LSTMNet"] = None,
    ) -> tuple[float, float]:
        """Return (avg_loss, accuracy) on a DataLoader."""
        model = net if net is not None else self._net
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in dataloader:
                xb, yb = xb.to(self._device), yb.to(self._device)
                logits = model(xb)
                total_loss += criterion(logits, yb).item() * len(xb)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += len(xb)
        if total == 0:
            return 0.0, 0.0
        return total_loss / total, correct / total


# ── Convenience factory ───────────────────────────────────────────────────────

def load_or_train(
    df: pd.DataFrame,
    model_path: str | Path = "models/lstm.pt",
    force_retrain: bool = False,
    **kwargs,
) -> LSTMClassifier:
    """
    Load saved LSTM if it exists, otherwise train from scratch.

    Args:
        df:            Historical OHLCV data for training if needed
        model_path:    Where to save/load the model
        force_retrain: Always retrain even if model file exists
    """
    path = Path(model_path)
    clf  = LSTMClassifier(model_path=path, **kwargs)

    if force_retrain or clf._net is None:
        logger.info("Training new LSTMClassifier on %d bars...", len(df))
        clf.train(df)

    return clf
