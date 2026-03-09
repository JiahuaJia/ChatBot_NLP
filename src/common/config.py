"""Central configuration — single source of truth for all constants."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data" / "movies"
CHROMA_DIR = ROOT_DIR / "chroma_db"

# ── Chroma ─────────────────────────────────────────────────────────────────
COLLECTION_NAME = "movies_rag_v1"

# ── OpenAI ─────────────────────────────────────────────────────────────────
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
EMBED_DIMENSIONS = 1536

# ── Chunking ───────────────────────────────────────────────────────────────
CHUNK_SIZE_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 50

# ── Retrieval ──────────────────────────────────────────────────────────────
RETRIEVAL_CANDIDATES = 20   # BM25 and semantic each fetch this many
TOP_K = 5                   # final results after RRF fusion
RRF_K = 60                  # RRF constant

# ── Memory ─────────────────────────────────────────────────────────────────
MAX_HISTORY_TURNS = 10      # each turn = one user + one assistant message

# ── Movies to fetch ────────────────────────────────────────────────────────
MOVIE_TITLES = [
    # Classics
    "The Godfather",
    "The Shawshank Redemption",
    "Schindler's List",
    "Casablanca",
    "Citizen Kane",
    "Vertigo",
    "Rear Window",
    "Some Like It Hot",
    "Sunset Boulevard",
    "Singin' in the Rain",
    "12 Angry Men (1957 film)",
    "Lawrence of Arabia",
    "Psycho (1960 film)",
    "Chinatown",
    "All About Eve",
    "The Wizard of Oz",
    "It's a Wonderful Life",
    "On the Waterfront",
    "Double Indemnity",
    "Apocalypse Now",
    "The Maltese Falcon (1941 film)",
    "Dr. Strangelove",
    "Taxi Driver",
    "Raging Bull",
    "Annie Hall",
    "Network (1976 film)",
    "Barry Lyndon",
    "2001: A Space Odyssey",
    "Blade Runner",
    "The Silence of the Lambs (film)",
    # Modern
    "The Dark Knight",
    "Inception",
    "Pulp Fiction",
    "Fight Club",
    "Forrest Gump",
    "The Matrix",
    "Goodfellas",
    "Interstellar (film)",
    "The Lion King",
    "Jurassic Park",
    "Avatar (2009 film)",
    "Avengers: Endgame",
    "Parasite (2019 film)",
    "Get Out",
    "Mad Max: Fury Road",
    "La La Land",
    "Whiplash (film)",
    "Arrival (film)",
    "Her (film)",
    "No Country for Old Men",
    "There Will Be Blood",
    "The Social Network",
    "Gone Girl (film)",
    "Birdman (film)",
    "Moonlight (2016 film)",
    "1917 (film)",
    "Joker (2019 film)",
    "Knives Out",
    "Everything Everywhere All at Once",
    "Oppenheimer (film)",
]
