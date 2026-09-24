"""Walk captions -> index -> search -> transcript through the real CLI.

Needs no model and no binary: caption ingest is pure Python. Run it from
the repo root with the virtualenv activated:

    python scripts/manual_check_index.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / ".rytp-manual"

LINES = (
    (0, "Добрый вечер дорогие друзья"),
    (4_000, "Сегодня у нас странное ощущение"),
)


def seed() -> None:
    from rytp.config import ensure_dir, paths
    from rytp.db import Database
    from rytp.transcribe.captions import ingest_captions

    ensure_dir(paths().root)
    db = Database(paths().db)
    db.migrate()
    db.conn.execute(
        "INSERT INTO videos (source, kind, external_id, url, title, created_at)"
        " VALUES ('ytdlp', 'video', 'VIDEO_A', 'https://example.invalid/w/VIDEO_A',"
        " 'Sample evening', '2026-09-21T00:00:00+00:00')"
    )
    events = [
        {
            "tStartMs": base,
            "segs": [
                {"utf8": token, "tOffsetMs": index * 400}
                for index, token in enumerate(line.split())
            ],
        }
        for base, line in LINES
    ]
    path = paths().root / "captions.json3"
    path.write_text(json.dumps({"events": events}), encoding="utf-8")
    print("words:", ingest_captions(db, 1, path))
    db.close()


def run(*args: str) -> None:
    print(f"\n$ rytp {' '.join(args)}")
    result = subprocess.run([sys.executable, "-m", "rytp", *args], check=False)
    if result.returncode != 0:
        print(f"  (exit {result.returncode})")


def main() -> int:
    if DATA.exists():
        shutil.rmtree(DATA)
    os.environ["RYTP_DATA"] = str(DATA)
    seed()
    run("index", "build")
    run("search", "words", "добрый вечер")
    run("search", "words", "ощущения")
    run("search", "words", "добрый вечер", "--cuttable")
    run("transcript", "build", "1")
    print("\n--- transcripts/1.md ---")
    print((DATA / "transcripts" / "1.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
