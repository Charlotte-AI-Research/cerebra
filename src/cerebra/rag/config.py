import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR  = DATA_DIR / "processed"

DISCORD_DATA_DIR = BASE_DIR / "cerebra_discord" / "data"
CAIR_OVERVIEW_FILE = DISCORD_DATA_DIR / "cair_overview.md"
PAST_EVENTS_FILE = DISCORD_DATA_DIR / "past_events.md"

# ChromaDB Cloud
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE", "default")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "cerebra")

# OpenAI 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"

# Chunking
CHUNK_SIZE = 400  # words per chunk (for processed/ files)
CHUNK_OVERLAP = 50   # word overlap between chunks

# Ingest
INGEST_BATCH_SIZE = 50  # documents per upsert call to Chroma Cloud