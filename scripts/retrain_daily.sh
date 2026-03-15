#!/usr/bin/env bash
# Daily LGBM retraining — 8 active symbols x 4 timeframes (15m/5m/3m/1m)
# Faster than weekly: 5 splits instead of 10, no LSTM, restarts agent after.
# Cron (daily 02:00 UTC):  0 2 * * *  /path/to/cryptoagent/scripts/retrain_daily.sh >> /path/to/cryptoagent/logs/retrain.log 2>&1
#
# On Windows Task Scheduler: Action → "C:\Program Files\Git\bin\bash.exe" -c "/c/Users/ZESTRO/Desktop/cryptoagent/scripts/retrain_daily.sh"

cd "$(dirname "$0")/.."
source .venv_312/Scripts/activate 2>/dev/null || source .venv_312/bin/activate 2>/dev/null || true

# ── Active symbols (BNB/SUI/PIPPIN disabled — negative OOS Sharpe) ───────────
SYMBOLS="BTC-USDC ETH-USDC SOL-USDC XRP-USDC DOGE-USDC ENA-USDC LINK-USDC ADA-USDC"

NOW=$(python -c "from datetime import datetime, timezone; print(datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'))")
echo "============================================================"
echo " CryptoAgent -- Daily Model Retraining (8 symbols x 4 TFs)"
echo " $NOW"
echo "============================================================"

total=$((8 * 4))
done=0
failed=0

for sym in $SYMBOLS; do
  for tf in 15m 5m 3m 1m; do
    done=$((done + 1))

    # Per-TF thresholds (tuned to label noise at each timeframe)
    thresh=0.002
    [ "$sym" = "BTC-USDC" ] && [ "$tf" = "15m" ] && thresh=0.003
    [ "$tf" = "5m"  ] && thresh=0.001
    [ "$tf" = "3m"  ] && thresh=0.0007
    [ "$tf" = "1m"  ] && thresh=0.0005

    echo ""
    echo "[$done/$total] $sym $tf (threshold=$thresh) ..."

    if python scripts/train_models.py \
        --symbol "$sym" --timeframe "$tf" \
        --candles 18000 --splits 5 \
        --threshold "$thresh" \
        --drop-flat --force-retrain; then
      echo "  ✓ $sym $tf done"
    else
      echo "  ✗ $sym $tf FAILED — continuing"
      failed=$((failed + 1))
    fi

    sleep 2   # avoid BloFin 429 rate limit
  done
done

echo ""
DONE_TIME=$(python -c "from datetime import datetime, timezone; print(datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'))")
echo "============================================================"
echo " LGBM retrain complete — $done models ($failed failed)"
echo " Finished: $DONE_TIME"
echo "============================================================"

# ── Restart the agent to load fresh models ───────────────────────────────────
echo ""
echo "Restarting agent to load new models..."

# Use systemd if available (Azure VM), otherwise fall back to manual PID management
if systemctl is-active --quiet cryptoagent 2>/dev/null; then
  echo "Restarting via systemd..."
  sudo systemctl restart cryptoagent
  sleep 5
  NEW_PID=$(systemctl show -p MainPID --value cryptoagent 2>/dev/null || echo "unknown")
  echo "Agent restarted via systemd (PID $NEW_PID)"
elif [ -f agent.pid ]; then
  OLD_PID=$(cat agent.pid)
  echo "Killing old agent PID $OLD_PID..."
  kill -9 "$OLD_PID" 2>/dev/null || true
  sleep 3
  PYTHONUTF8=1 PYTHONUNBUFFERED=1 python -u -m src.agent >> logs/agent.log 2>&1 &
  NEW_PID=$!
  echo $NEW_PID > agent.pid
  echo "New agent started with PID $NEW_PID"
fi

echo ""
echo "Daily retrain complete: $(python -c "from datetime import datetime, timezone; print(datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'))")"
