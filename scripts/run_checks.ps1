param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [switch]$SkipCompile
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

if (-not (Test-Path $Python)) {
    $Python = "python"
}

Write-Host "Using Python: $Python"

if (-not $SkipCompile) {
    Write-Host "Compiling Python files"
    & $Python -m compileall -q src scripts tests
}

Write-Host "Running tests"
& $Python -m pytest -q

Write-Host "Checks passed."

