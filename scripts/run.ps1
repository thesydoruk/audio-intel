$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path ".venv")) {
    & (Join-Path $Root "scripts\setup.ps1")
}

$Python = Join-Path $Root ".venv\Scripts\python.exe"
& $Python -m audio_intel.server
