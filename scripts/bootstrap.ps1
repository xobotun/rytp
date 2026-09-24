# Create .venv at the repository root and install rytp with its development
# dependencies. ffmpeg, ffprobe and yt-dlp are system binaries you install
# yourself; this script covers Python dependencies only.
$ErrorActionPreference = "Stop"

Set-Location (Join-Path $PSScriptRoot "..")

py -3 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -e ".[dev]"

Write-Host "Done. Activate it with:  .\.venv\Scripts\Activate.ps1"
Write-Host "Then run the suite with: python -m pytest"
