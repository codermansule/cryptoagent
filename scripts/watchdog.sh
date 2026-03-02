#!/usr/bin/env bash
# Watchdog — keeps agent and dashboard alive.
# Run once: source .venv_312/Scripts/activate && bash scripts/watchdog.sh &

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"
source .venv_312/Scripts/activate

echo "[watchdog] Started at $(date -u)"

while true; do
    # ── Check agent ──────────────────────────────────────────────────────────
    AGENT_PID=$(cat agent.pid 2>/dev/null)
    if [ -z "$AGENT_PID" ] || ! tasklist 2>/dev/null | grep -q "^python.exe *$AGENT_PID "; then
        echo "[watchdog] Agent not running — starting..."
        PYTHONUTF8=1 PYTHONUNBUFFERED=1 python -u -m src.agent >> logs/agent.log 2>&1 &
        echo $! > agent.pid
        echo "[watchdog] Agent started PID: $(cat agent.pid)"
    fi

    # ── Check dashboard ──────────────────────────────────────────────────────
    if ! tasklist 2>/dev/null | grep -q "streamlit.exe"; then
        echo "[watchdog] Dashboard not running — starting..."
        streamlit run src/monitoring/dashboard.py --server.port 8501 --server.headless true >> dashboard_startup.log 2>&1 &
        echo $! > dashboard.pid
        echo "[watchdog] Dashboard started PID: $(cat dashboard.pid)"
    fi

    sleep 30
done
