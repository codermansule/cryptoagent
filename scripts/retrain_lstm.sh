#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
source .venv_312/Scripts/activate 2>/dev/null || source .venv_312/bin/activate 2>/dev/null || true

SYMBOLS="BTC-USDC ETH-USDC SOL-USDC XRP-USDC DOGE-USDC BNB-USDC SUI-USDC ENA-USDC LINK-USDC ADA-USDC"

echo "=== LSTM MODELS (all 10 symbols) ==="
for sym in $SYMBOLS; do
  echo "Training LSTM $sym 15m..."
  python scripts/train_models.py \
    --symbol "$sym" --timeframe 15m \
    --candles 18000 --splits 10 --threshold 0.002 \
    --drop-flat --force-retrain --train-lstm
  sleep 2
done
