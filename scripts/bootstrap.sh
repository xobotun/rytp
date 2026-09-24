#!/bin/sh
# Create .venv at the repository root and install rytp with its development
# dependencies. ffmpeg, ffprobe and yt-dlp are system binaries you install
# yourself; this script covers Python dependencies only.
set -eu

cd "$(dirname "$0")/.."

python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -e ".[dev]"

echo "Done. Activate it with:  . .venv/bin/activate"
echo "Then run the suite with: python -m pytest"
