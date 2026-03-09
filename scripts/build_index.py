#!/usr/bin/env python3
"""One-time script: chunk all movie JSON files and upsert into Chroma."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.config import DATA_DIR
from src.ingest.chunker import chunk_document
from src.ingest.embedder import upsert_chunks

if __name__ == "__main__":
    movie_files = list(DATA_DIR.glob("*.json"))
    if not movie_files:
        print("No movie JSON files found. Run fetch_movies.py first.")
        sys.exit(1)

    print(f"Building index from {len(movie_files)} movie files...")
    all_chunks = []
    for path in movie_files:
        doc = json.loads(path.read_text())
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"  {doc['title']}: {len(chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")
    print("Upserting into Chroma...")
    total = upsert_chunks(all_chunks)
    print(f"\nDone. {total} chunks indexed in chroma_db/")
