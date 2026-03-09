#!/usr/bin/env python3
"""One-time script: fetch 60 movie Wikipedia pages into data/movies/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.config import MOVIE_TITLES
from src.ingest.fetcher import fetch_all

if __name__ == "__main__":
    print(f"Fetching {len(MOVIE_TITLES)} movies...")
    saved = fetch_all(MOVIE_TITLES)
    print(f"\nDone. {len(saved)} files in data/movies/")
