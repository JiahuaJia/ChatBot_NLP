"""Fetch movie Wikipedia pages and persist as JSON."""

import hashlib
import json
import re
import time
from pathlib import Path

import wikipedia

from src.common.config import DATA_DIR, MOVIE_TITLES


def _slugify(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


def fetch_movie(title: str) -> dict:
    page = wikipedia.page(title, auto_suggest=False)
    content = page.content
    return {
        "title": page.title,
        "url": page.url,
        "content": content,
        "doc_id": hashlib.sha256(content.encode()).hexdigest()[:16],
    }


def fetch_all(titles: list[str] = MOVIE_TITLES, delay: float = 0.5) -> list[Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for title in titles:
        path = DATA_DIR / f"{_slugify(title)}.json"
        if path.exists():
            print(f"  [skip] {title}")
            saved.append(path)
            continue
        try:
            data = fetch_movie(title)
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
            print(f"  [ok]   {title} → {path.name}")
            saved.append(path)
        except wikipedia.exceptions.DisambiguationError as e:
            # Try the first option
            try:
                data = fetch_movie(e.options[0])
                path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
                print(f"  [ok]   {title} (via '{e.options[0]}') → {path.name}")
                saved.append(path)
            except Exception as inner:
                print(f"  [err]  {title}: {inner}")
        except Exception as exc:
            print(f"  [err]  {title}: {exc}")
        time.sleep(delay)
    return saved
