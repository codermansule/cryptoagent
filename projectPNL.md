# CryptoAgent — PnL Report
**Last updated:** 2026-03-09

---

## PAPER AGENT (Bot) — 50 Trades

| Metric | Value |
|--------|-------|
| Total trades | 50 |
| Win rate | 14/50 = **28.0%** |
| Avg winner | **+13.83%** |
| Avg loser | **-3.66%** |
| Avg SL hit | **-7.28%** |
| Profit factor | **1.47** |
| Total return | **+61.84%** combined (~+$124 est.) |

### Exit Breakdown

| Exit Type | Count | Avg PnL |
|-----------|-------|---------|
| Full Take Profit | 4 | **+24.12%** |
| Partial Take Profit | 8 | **+10.63%** |
| Breakeven Stop | 19 | **-0.42%** (fees only) |
| Stop Loss | 17 | **-7.28%** |
| Manual Close | 2 | ~+5.5% |

### By Symbol

| Symbol | Trades | W | L | WR% | Total% | Best% | Worst% | Status |
|--------|--------|---|---|-----|--------|-------|--------|--------|
| LINK-USDC | 5 | 3 | 2 | 60% | **+43.35%** | +24.78% | -6.71% | Active |
| SOL-USDC | 6 | 3 | 3 | 50% | **+37.82%** | +30.84% | -15.84% | Active |
| BTC-USDC | 7 | 2 | 5 | 29% | **+7.65%** | +21.50% | -10.97% | Active |
| DOGE-USDC | 8 | 3 | 5 | 38% | **+7.74%** | +19.34% | -14.63% | Active |
| ENA-USDC | 3 | 1 | 2 | 33% | **+5.42%** | +12.99% | -6.79% | Active |
| ADA-USDC | 5 | 1 | 4 | 20% | **+0.39%** | +5.75% | -3.36% | Active |
| ETH-USDC | 9 | 1 | 8 | 11% | **-19.10%** | +2.59% | -13.84% | Muted (threshold 65%) |
| BNB-USDC | 5 | 0 | 5 | 0% | **-9.00%** | -0.27% | -4.71% | Disabled |
| SUI-USDC | 2 | 0 | 2 | 0% | **-12.43%** | -0.51% | -11.92% | Disabled |

### Top 5 Winners

| Symbol | Side | PnL% | Hold Time | Exit |
|--------|------|------|-----------|------|
| SOL-USDC | Long | **+30.84%** | 4.5h | Take Profit |
| LINK-USDC | Long | **+24.78%** | 2.4h | Take Profit |
| BTC-USDC | Long | **+21.50%** | 4.0h | Take Profit |
| DOGE-USDC | Long | **+19.34%** | 1.1h | Take Profit |
| SOL-USDC | Long | **+15.45%** | 3.8h | Partial TP |

### Top 5 Losers

| Symbol | Side | PnL% | Hold Time | Exit | Note |
|--------|------|------|-----------|------|------|
| SOL-USDC | Long | **-15.84%** | 10.7h | Stop Loss | Regime filter bug (now fixed) |
| DOGE-USDC | Short | **-14.63%** | 11.7h | Stop Loss | max_hold_hours bug (now fixed) |
| ETH-USDC | Long | **-13.84%** | 2.5h | Stop Loss | Regime filter bug (now fixed) |
| SUI-USDC | Long | **-11.92%** | 8.1h | Stop Loss | Symbol now disabled |
| BTC-USDC | Long | **-10.97%** | 5.9h | Stop Loss | Regime filter bug (now fixed) |

---

## REAL ACCOUNT — Manual Phone Trades

> All losses below are from manual trades placed on BloFin mobile app.
> The bot was NOT responsible for any of these entries.

### Liquidation Event 1 — March 5-6, 2026

All positions used **Cross margin** — one liquidation cascaded into all others simultaneously at 18:52:25 on March 6.

| Symbol | Side | Leverage | Entry | Exit | Loss (USD) | Loss% |
|--------|------|----------|-------|------|------------|-------|
| LINK-USDT | Long | **51X** | 9.359 | 8.966 | **-$20.88** | -220% |
| SUI-USDT | Long | **45X** | 0.953 | 0.9234 | **-$14.34** | -145% |
| ENA-USDC | Long | 25X | 0.1189 | 0.1079 | **-$15.54** | -234% |
| BNB-USDC | Long | 25X | 655.85 | 624.54 | **-$4.81** | -122% |
| PIPPIN-USDT | Long | 29X | 0.34814 | 0.33684 | **-$8.90** | -98% |
| DOGE-USDT | Long | 31X | 0.09431 | 0.09156 | **-$3.03** | -94% |
| DOOD-USDT | Long | 3X | 0.003119 | 0.003004 | **-$0.43** | -11% |
| **Event 1 Total** | | | | | **-$67.93** | |

### Liquidation Event 2 — March 7-9, 2026

PIPPIN held for 58 hours after explicit warning to close at -$24. Liquidation at 0.31213 cascaded into ENA and ADA shorts.

| Symbol | Side | Leverage | Entry | Exit | Loss (USD) | Loss% | Note |
|--------|------|----------|-------|------|------------|-------|------|
| PIPPIN-USDT | Long | **29X** | 0.36981 | 0.31213 | **-$85.40** | -456% | Warned to close at -$24 on Mar-07 03:46 |
| ENA-USDT | Short | **50X** | 0.0997 | 0.1019 | **-$8.42** | -114% | Cascade from PIPPIN |
| ADA-USDT | Short | 20X | 0.2499 | 0.2581 | **-$5.99** | -68% | Cascade from PIPPIN |
| ENA-USDT #2 | Short | 30X | — | — | unknown | — | Closed (not liquidated) |
| **Event 2 Total** | | | | | **~-$100+** | |

---

## Overall Summary

| Account | P&L |
|---------|-----|
| Paper Agent (bot) | **+$124 est. profit** |
| Real Account Event 1 | **-$67.93** |
| Real Account Event 2 | **-$100+** |
| **Net Real-World Result** | **-$46+ loss** |

---

## Key Lessons

| # | Mistake | Cost |
|---|---------|------|
| 1 | Cross margin on all positions | Cascade wiped 6-7 positions simultaneously, twice |
| 2 | 25–51X leverage with no stop loss | 2% market move = liquidation |
| 3 | PIPPIN held 58h after warning | -$24 refusal turned into -$85 loss |
| 4 | Trading symbols not in validated universe | PIPPIN, DOOD, SUI — no backtest data |
| 5 | Manual trades copying bot signals | Bot uses 5X + ATR stop. Manual had 29–51X + no stop |

## Rules Going Forward

1. **Isolated margin only — always, no exceptions**
2. **Maximum 5X leverage on any manual trade**
3. **Set stop loss immediately on every entry**
4. **Only trade the 8 validated symbols: BTC, ETH, SOL, XRP, DOGE, ENA, LINK, ADA**
5. **Do not trade until paper agent has 100+ profitable trades documented**

---

## Bot Fixes Applied (Mar 7, 2026)

| Fix | Impact |
|-----|--------|
| Regime filter enforced | No longs in downtrends, no shorts in uptrends |
| BTC 4h macro filter | Blocks all alt longs when BTC < EMA50 on 4h |
| BNB + SUI + PIPPIN removed | Negative OOS Sharpe — provably losing symbols |
| ETH threshold 40% → 65% | Muted until retrain (1W/4L live record) |
| RR ratio 2.0 → 2.5 → 2.0 | 5m scalp mode calibrated |
| Breakeven ATR 0.5 → 0.8 | Reduces chop-stop rate |
| max_hold_hours 6 → 3 | 5m scalp mode, no overnight positions |
| Primary TF sort fixed | ATR now uses 5m candle data (not 15m) |
| max_hold reads settings.yaml | Was hardcoded — DOGE held 11.7h instead of 6h |
| 32 LGBM models retrained | Fresh data as of Mar-06 22:33 UTC |
