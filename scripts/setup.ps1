$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python is not on PATH. Install Python 3.10+ and retry."
}

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment in .venv ..."
    python -m venv .venv
}

$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Pip = Join-Path $Root ".venv\Scripts\pip.exe"

& $Python -m pip install --upgrade pip
& $Pip install -e ".[all,dev]"
& $Python -m pre_commit install

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example"
}

New-Item -ItemType Directory -Force -Path "data\whisper", "data\panns" | Out-Null

Write-Host ""
Write-Host "Setup complete."
Write-Host "  Activate:  .\.venv\Scripts\Activate.ps1"
Write-Host "  Run API:   .\scripts\run.ps1"
Write-Host "  Hooks:     pre-commit installed (runs on git commit)"
Write-Host "  Docker:    docker compose up -d --build"
