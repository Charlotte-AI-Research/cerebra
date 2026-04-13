import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
# config.py lives at src/rag/config.py
# .parent        → src/rag/
# .parent.parent → src/
# .parent.parent.parent → project root (CEREBRA/)

BASE_DIR      = Path(__file__).parent.parent.parent
DATA_DIR      = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# All data files live under data/
CAIR_OVERVIEW_FILE = DATA_DIR / "cair_overview.md"
PAST_EVENTS_FILE   = DATA_DIR / "past_events.md"

# ChromaDB Cloud
CHROMA_API_KEY  = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT   = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE", "default")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "cerebra")

# vLLM — two separate servers, one for embeddings, one for the chat LLM.
#
# Start the embedding server (terminal 1):
#   vllm serve Qwen/Qwen3-Embedding-4B --task embed --port 8001
#
# Start the chat LLM server (terminal 2):
#   vllm serve <your-chat-model> --port 8000

VLLM_BASE_URL       = os.getenv("VLLM_BASE_URL",       "http://localhost:8000")
VLLM_EMBED_BASE_URL = os.getenv("VLLM_EMBED_BASE_URL", "http://localhost:8001")
VLLM_EMBED_MODEL    = os.getenv("VLLM_EMBED_MODEL",    "Qwen/Qwen3-Embedding-0.6B")
VLLM_LLM_MODEL      = os.getenv("VLLM_LLM_MODEL",      "Qwen/Qwen2.5-7B-Instruct")
VLLM_API_KEY        = os.getenv("VLLM_API_KEY",        "EMPTY")

# Chunking
CHUNK_SIZE    = 200  # words per chunk (for processed/ files)
CHUNK_OVERLAP = 30   # word overlap between chunks

# Ingest
INGEST_BATCH_SIZE = 50  # documents per upsert call to Chroma Cloud