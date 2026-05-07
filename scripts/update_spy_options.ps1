param(
    [string]$Symbol = "SPY",
    [int]$MaxExpirations = 8,
    [double]$MinDteDays = 20.0,
    [double]$MaxDteDays = 45.0,
    [string]$QuoteTimestamp = "latest-market-date",
    [string]$DatasetOptionType = "call",
    [string]$VenvPath = ".venv"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

$PythonExe = Join-Path $RepoRoot "$VenvPath\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

& $PythonExe scripts\download_option_chain.py `
    --symbol $Symbol `
    --max-expirations $MaxExpirations `
    --min-dte-days $MinDteDays `
    --max-dte-days $MaxDteDays `
    --option-type both `
    --quote-timestamp $QuoteTimestamp `
    --build-dataset `
    --dataset-option-type $DatasetOptionType
