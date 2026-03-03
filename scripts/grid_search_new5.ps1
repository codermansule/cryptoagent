# ── Grid Search: New 5 Symbols ─────────────────────────────────────────────────
# Runs optimize_params.py for BNB / SUI / ENA / LINK / ADA sequentially.
# Usage:  .\scripts\grid_search_new5.ps1
# After completion: update config/settings.yaml symbol_overrides
# ──────────────────────────────────────────────────────────────────────────────

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (Test-Path ".\.venv_312\Scripts\Activate.ps1") {
    . .\.venv_312\Scripts\Activate.ps1
}

$Symbols = @("BNB-USDC", "SUI-USDC", "ENA-USDC", "LINK-USDC", "ADA-USDC")
$Confs = @("20", "25", "30", "35", "40", "45", "50", "55", "60")
$Atrs = @("1.5", "2.0", "2.5", "3.0")

$StartTime = Get-Date
Write-Host "`n=================================================" -ForegroundColor Cyan
Write-Host "  GRID SEARCH -- New 5 Symbols" -ForegroundColor Cyan
Write-Host "  Conf: $($Confs -join ', ')  ATR: $($Atrs -join ', ')" -ForegroundColor Cyan
Write-Host "  Bars: 3000  OOS split: 25%  RR: 2.0" -ForegroundColor Cyan
Write-Host "  Started: $StartTime" -ForegroundColor Cyan
Write-Host "=================================================`n" -ForegroundColor Cyan

$Results = @{}

foreach ($Sym in $Symbols) {
    Write-Host "--------------------------------------------------" -ForegroundColor DarkCyan
    Write-Host "  >> $Sym" -ForegroundColor White
    Write-Host "--------------------------------------------------" -ForegroundColor DarkCyan
    $SymStart = Get-Date

    $ArgList = @(
        "scripts/optimize_params.py",
        "--symbol", $Sym,
        "--timeframe", "15m",
        "--candles", "3000",
        "--conf"
    ) + $Confs + @(
        "--atr"
    ) + $Atrs + @(
        "--rr", "2.0",
        "--min-agreeing", "2",
        "--window", "300",
        "--warmup", "100",
        "--oos-split", "0.25",
        "--equity", "10000",
        "--fee", "0.05",
        "--slippage", "0.02",
        "--kelly", "0.25",
        "--top", "5"
    )

    python @ArgList
    $ExitCode = $LASTEXITCODE
    $Elapsed = (Get-Date) - $SymStart

    if ($ExitCode -eq 0) {
        Write-Host "  [OK] $Sym done in $($Elapsed.Minutes)m $($Elapsed.Seconds)s`n" -ForegroundColor Green
        $Results[$Sym] = "OK"
    }
    else {
        Write-Host "  [FAIL] $Sym exit code $ExitCode`n" -ForegroundColor Red
        $Results[$Sym] = "FAILED (exit $ExitCode)"
    }
}

$TotalElapsed = (Get-Date) - $StartTime
Write-Host "`n=================================================" -ForegroundColor Cyan
Write-Host "  GRID SEARCH COMPLETE" -ForegroundColor Cyan
Write-Host "  Total time: $($TotalElapsed.Hours)h $($TotalElapsed.Minutes)m $($TotalElapsed.Seconds)s" -ForegroundColor Cyan
Write-Host "=================================================`n" -ForegroundColor Cyan

foreach ($Sym in $Symbols) {
    $Status = $Results[$Sym]
    $Color = if ($Status -eq "OK") { "Green" } else { "Red" }
    Write-Host "  $Sym  -->  $Status" -ForegroundColor $Color
}

Write-Host "`n  Results in: backtests/optimize_<sym>_15m_*.csv"
Write-Host "  Next: update config/settings.yaml symbol_overrides`n"
