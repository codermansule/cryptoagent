#!/usr/bin/env bash
# Weekly LGBM retraining script — all 10 symbols, all 4 timeframes (1m/3m/5m/15m)
# Recommended cron: 0 2 * * 0  (Sunday 02:00 UTC)
# Add to crontab: crontab -e
#   0 2 * * 0 /path/to/cryptoagent/scripts/retrain_weekly.sh >> /path/to/cryptoagent/logs/retrain.log 2>&1

set -e
cd "$(dirname "$0")/.."
source .venv_312/Scripts/activate 2>/dev/null || source .venv_312/bin/activate 2>/dev/null || true

SYMBOLS="BTC-USDC ETH-USDC SOL-USDC XRP-USDC DOGE-USDC BNB-USDC SUI-USDC ENA-USDC LINK-USDC ADA-USDC"

echo "============================================================"
echo " CryptoAgent -- Weekly Model Retraining (10 symbols x 4 TFs)"
echo " $(date -u)"
echo "============================================================"

total=$((10 * 4))
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
echo "=== LSTM MODELS (all 10 symbols) ==="
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
echo "Retraining complete: $(date -u)"
echo "============================================================"
