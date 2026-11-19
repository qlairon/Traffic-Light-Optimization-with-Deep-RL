# Create a virtual environment and install dependencies (Windows PowerShell).
# Usage (from anywhere):  .\scripts\setup_env.ps1
# If script execution is blocked, run first:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

$ErrorActionPreference = 'Stop'

# Run from the repository root regardless of where the script is called from.
Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not (Test-Path .venv)) {
    python -m venv .venv
}
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Write-Host "`nDone. Activate later with: .\.venv\Scripts\Activate.ps1"
