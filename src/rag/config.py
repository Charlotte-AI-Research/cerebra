import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# -------------------------------------------------
# 1. Paths
# -------------------------------------------------
BASE_DIR      = Path(__file__).resolve().parent.parent.parent
DATA_DIR      = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# All data files live under data/
CAIR_OVERVIEW_FILE = DATA_DIR / "cair_overview.md"
PAST_EVENTS_FILE   = DATA_DIR / "past_events.md"

# -------------------------------------------------
# 2. ChromaDB Cloud
# -------------------------------------------------
CHROMA_API_KEY  = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT   = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE", "default")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "cerebra")

# -------------------------------------------------
# 3. Model APIs
# -------------------------------------------------
# Local Embedding Model (vLLM)
# Start the embedding server (terminal 1):
#   vllm serve Qwen/Qwen3-Embedding-0.6B --task embed --port 8001
VLLM_EMBED_BASE_URL = os.getenv("VLLM_EMBED_BASE_URL", "http://localhost:8001/v1")
VLLM_EMBED_MODEL    = os.getenv("VLLM_EMBED_MODEL",    "Qwen/Qwen3-Embedding-0.6B")
# Optional: bearer token sent to the embedding endpoint. Useful when your embedding
# server is exposed via the gateway (recommended) or any reverse proxy with auth.
VLLM_EMBED_API_KEY  = os.getenv("VLLM_EMBED_API_KEY") or os.getenv("OPENAI_API_KEY") or "dummy-local-key"

# Remote Chat LLM (Kronos API)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://kronos-labs--vllm-gpt-oss-120b-serve.modal.run/v1")
LLM_MODEL    = os.getenv("LLM_MODEL",    "openai/gpt-oss-120b")
LLM_API_KEY  = os.getenv("LLM_API_KEY")

# -------------------------------------------------
# 4. Processing & Ingest Parameters
# -------------------------------------------------
# Chunking
CHUNK_SIZE    = 250 # words per chunk (for processed/ files)
CHUNK_OVERLAP = 50  # word overlap between chunks

# Ingest
INGEST_BATCH_SIZE =50 # documents per upsert call to Chroma Cloud

 