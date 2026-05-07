param(
    [switch]$WithML,
    [switch]$NoDev,
    [switch]$SkipTests,
    [string]$VenvPath = ".venv",
    [string]$Python = "python",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cpu"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

if (-not (Test-Path $VenvPath)) {
    Write-Host "Creating virtual environment at $VenvPath"
    & $Python -m venv $VenvPath
}

$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Could not find Python executable at $VenvPython"
}

Write-Host "Upgrading pip"
& $VenvPython -m pip install --upgrade pip

if ($WithML) {
    Write-Host "Installing PyTorch from $TorchIndexUrl"
    & $VenvPython -m pip install "torch>=2.6" --index-url $TorchIndexUrl
}

if ($NoDev) {
    Write-Host "Installing project runtime dependencies"
    & $VenvPython -m pip install -e .
} else {
    Write-Host "Installing project with development dependencies"
    & $VenvPython -m pip install -e ".[dev]"
}

Write-Host "Compiling Python files"
& $VenvPython -m compileall -q src tests scripts

if (-not $SkipTests -and -not $NoDev) {
    Write-Host "Running test suite"
    & $VenvPython -m pytest -q
}

Write-Host ""
Write-Host "Environment ready."
Write-Host "Activate it with:"
Write-Host "  .\$VenvPath\Scripts\Activate.ps1"
