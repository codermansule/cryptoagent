#!/usr/bin/env bash
# Weekly LGBM + LSTM retraining — 8 active symbols x 4 TFs + LSTM 15m
# Recommended cron: 0 2 * * 0  (Sunday 02:00 UTC)
# Add to crontab: crontab -e
#   0 2 * * 0 /path/to/cryptoagent/scripts/retrain_weekly.sh >> /path/to/cryptoagent/logs/retrain.log 2>&1

cd "$(dirname "$0")/.."
source .venv_312/Scripts/activate 2>/dev/null || source .venv_312/bin/activate 2>/dev/null || true

# Active symbols (BNB/SUI/PIPPIN disabled — negative OOS Sharpe, removed Mar-07)
SYMBOLS="BTC-USDC ETH-USDC SOL-USDC XRP-USDC DOGE-USDC ENA-USDC LINK-USDC ADA-USDC"

NOW=$(python -c "from datetime import datetime, timezone; print(datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'))")
echo "============================================================"
echo " CryptoAgent -- Weekly Model Retraining (8 symbols x 4 TFs)"
echo " $NOW"
echo "============================================================"

total=$((8 * 4))
done=0

for sym in $SYMBOLS; do
  for tf in 15m 5m 3m 1m; do
    done=$((done + 1))
    thresh=0.002
    [ "$sym" = "BTC-USDC" ] && [ "$tf" = "15m" ] && thresh=0.003
    [ "$tf" = "5m"  ] && thresh=0.001
    [ "$tf" = "3m"  ] && thresh=0.0007
    [ "$tf" = "1m"  ] && thresh=0.0005

    shap_flag=""
    [ "$sym" = "BTC-USDC" ] && [ "$tf" = "15m" ] && shap_flag="--shap"

    echo ""
    echo "[$done/$total] Training $sym $tf (threshold=$thresh)..."
    python scripts/train_models.py \
      --symbol "$sym" --timeframe "$tf" \
      --candles 18000 --splits 10 \
      --threshold "$thresh" \
      --drop-flat --force-retrain $shap_flag
    echo "  Done $sym $tf"
    sleep 2
  done
done

echo ""
echo "=== LSTM MODELS (8 active symbols, 15m only) ==="
for sym in $SYMBOLS; do
  echo "Training LSTM $sym 15m..."
  python scripts/train_models.py \
    --symbol "$sym" --timeframe 15m \
    --candles 18000 --splits 10 --threshold 0.002 \
    --drop-flat --force-retrain --train-lstm
  sleep 2
done

echo ""
echo "[Validate] BTC OOS backtest..."
python scripts/backtest.py \
  --symbol BTC-USDC --candles 2000 \
  --min-confidence 25 --trailing-activation-atr 0 --oos-split 0.25

echo ""
echo "Retraining complete: $(python -c "from datetime import datetime, timezone; print(datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'))")"
echo "============================================================"
