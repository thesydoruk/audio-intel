# Undo scripts/setup.ps1: remove the virtual environment, the pre-commit hook, and
# build/test caches. Downloaded models (data\whisper, data\panns) and .env are kept
# unless -Purge is given. Docker resources are not touched — see the README.
#
# Usage: .\scripts\uninstall.ps1 [-Purge]
param([switch]$Purge)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (Test-Path $Python) {
    & $Python -m pre_commit uninstall *> $null
}

foreach ($path in ".venv", "audio_intel.egg-info", ".pytest_cache", ".ruff_cache") {
    if (Test-Path $path) { Remove-Item -Recurse -Force $path }
}
Get-ChildItem -Path "src", "tests" -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force
Write-Host "Removed .venv, the pre-commit hook, and caches."

if ($Purge) {
    # Keep the .gitkeep placeholders the repository tracks.
    foreach ($dir in "data\whisper", "data\panns") {
        if (Test-Path $dir) {
            Get-ChildItem -Path $dir -Force | Where-Object { $_.Name -ne ".gitkeep" } |
                Remove-Item -Recurse -Force
        }
    }
    if (Test-Path ".env") { Remove-Item -Force ".env" }
    Write-Host "Purged downloaded models (data\whisper, data\panns) and .env."
} else {
    Write-Host "Kept data\whisper, data\panns and .env (use -Purge to delete them)."
}
