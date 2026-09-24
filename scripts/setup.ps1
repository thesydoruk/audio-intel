# Local install into .venv: the service, dev tools, and WhisperX alignment.
#
# For a CUDA build of torch, set TORCH_INDEX to the matching PyTorch wheel index
# before running, e.g. $env:TORCH_INDEX = "https://download.pytorch.org/whl/cu126"
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python is not on PATH. Install Python 3.10+ and retry."
}
foreach ($tool in "ffmpeg", "ffprobe") {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "$tool is not on PATH. Install ffmpeg and retry."
    }
}

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment in .venv ..."
    python -m venv .venv
}

$Python = Join-Path $Root ".venv\Scripts\python.exe"

function Invoke-Pip {
    & $Python -m pip @args
    if ($LASTEXITCODE -ne 0) { throw "pip $args failed" }
}

Invoke-Pip install --upgrade pip
if ($env:TORCH_INDEX) {
    Invoke-Pip install "torch>=2.8" "torchaudio>=2.8" --index-url $env:TORCH_INDEX
}
Invoke-Pip install -e ".[all,dev]"
# WhisperX pins pyannote.audio < 4 but its alignment never imports it: install it
# without dependencies, plus what alignment needs (same as the Dockerfile).
Invoke-Pip install --no-deps "whisperx==3.7.9"
Invoke-Pip install "nltk>=3.9.1" "pandas>=2.2.3" "transformers>=4.48.0,<5"
& $Python -m pre_commit install

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example"
}

New-Item -ItemType Directory -Force -Path "data\whisper", "data\panns" | Out-Null

Write-Host ""
Write-Host "Setup complete."
Write-Host "  Activate:   .\.venv\Scripts\Activate.ps1"
Write-Host "  Run API:    .\scripts\run.ps1"
Write-Host "  Check:      curl.exe http://localhost:8080/health"
Write-Host "  Uninstall:  .\scripts\uninstall.ps1   (add -Purge to also delete models and .env)"
