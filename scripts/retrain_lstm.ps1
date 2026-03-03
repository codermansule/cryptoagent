$Symbols = "BTC-USDC", "ETH-USDC", "SOL-USDC", "XRP-USDC", "DOGE-USDC", "BNB-USDC", "SUI-USDC", "ENA-USDC", "LINK-USDC", "ADA-USDC"

Write-Host "=== LSTM MODELS (all 10 symbols) ==="
if (Test-Path ".\.venv_312\Scripts\Activate.ps1") {
    . .\.venv_312\Scripts\Activate.ps1
}
foreach ($sym in $Symbols) {
    Write-Host "Training LSTM $sym 15m..."
    python scripts/train_models.py --symbol $sym --timeframe 15m --candles 18000 --splits 10 --threshold 0.002 --drop-flat --force-retrain --train-lstm
    Start-Sleep -Seconds 2
}
