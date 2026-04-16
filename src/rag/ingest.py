"""
ingest.py — Cerebra data ingestion pipeline

Sources:
  1. data/processed/**/*.md  — scraped university pages  (priority: high)
  2. data/cair_overview.md   — first-party CAIR info     (priority: high)
  3. data/past_events.md     — CAIR event history        (priority: low)

Processed files are word-chunked.
Markdown files are section-chunked (split on ## headers).

Ingestion is incremental — deterministic chunk IDs mean upsert only
embeds chunks that are new or changed.

Embeddings → Local Qwen3-Embedding-0.6B via vLLM
  Start with: vllm serve Qwen/Qwen3-Embedding-0.6B --task embed --port 8001
"""

import hashlib
import re
import time
from typing import List

import requests
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

from .config import (
    CHROMA_API_KEY,
    CHROMA_TENANT,
    CHROMA_DATABASE,
    COLLECTION_NAME,
    VLLM_EMBED_BASE_URL,
    VLLM_EMBED_MODEL,
    VLLM_EMBED_API_KEY,
    PROCESSED_DIR,
    CAIR_OVERVIEW_FILE,
    PAST_EVENTS_FILE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    INGEST_BATCH_SIZE,
)
from .logging_utils import get_logger

log = get_logger("rag.ingest")


# vLLM embedding function (OpenAI-compatible /v1/embeddings endpoint)

class VLLMEmbeddingFunction(EmbeddingFunction):
    """
    ChromaDB-compatible embedding function that calls a local vLLM server
    via the OpenAI-compatible REST endpoint.
    """

    def __init__(
        self,
        base_url: str = VLLM_EMBED_BASE_URL,
        model: str = VLLM_EMBED_MODEL,
        api_key: str = VLLM_EMBED_API_KEY,
    ):
        self.base_url = base_url.rstrip("/")  # e.g., http://localhost:8001/v1
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._check_connection()

    def _check_connection(self):
        try:
            # We removed the hardcoded /v1 here since it's now in the config's base_url
            resp = requests.get(f"{self.base_url}/models", headers=self.headers, timeout=5)
            resp.raise_for_status()
            available = [m["id"] for m in resp.json().get("data", [])]
            if self.model not in available:
                log.warning(
                    "Embedding model not found on vLLM server",
                    extra={"extra": {"model": self.model, "available": available, "base_url": self.base_url}},
                )
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            raise RuntimeError(
                f"Cannot connect to vLLM embedding server at {self.base_url}. "
                f"Start it with: vllm serve {self.model} --task embed --port 8001"
            )
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(
                f"vLLM embedding server returned an error at {self.base_url}/models: {e}"
            )

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        # Removed hardcoded /v1
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                resp = requests.post(
                    f"{self.base_url}/embeddings",
                    headers=self.headers,
                    json={"model": self.model, "input": texts},
                    timeout=120,
                )
                resp.raise_for_status()
                data = sorted(resp.json()["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in data]
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.HTTPError) as e:
                last_exc = e
                wait = 2 ** attempt
                log.warning(
                    "vLLM embedding request failed; retrying",
                    extra={"extra": {"attempt": attempt + 1, "wait_s": wait, "base_url": self.base_url}},
                )
                time.sleep(wait)

        raise RuntimeError(f"vLLM embedding request failed after retries: {last_exc}")

    def __call__(self, input: Documents) -> Embeddings:
        return self._embed_batch(list(input))


# ChromaDB client

def get_collection():
    if not CHROMA_API_KEY or not CHROMA_TENANT:
        raise RuntimeError(
            "Missing required Chroma settings. Ensure CHROMA_API_KEY and CHROMA_TENANT are set."
        )

    log.info(
        "Connecting to Chroma Cloud",
        extra={
            "extra": {
                "tenant": CHROMA_TENANT,
                "database": CHROMA_DATABASE,
                "collection": COLLECTION_NAME,
            }
        },
    )

    client = chromadb.CloudClient(api_key=CHROMA_API_KEY, tenant=CHROMA_TENANT, database=CHROMA_DATABASE)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=VLLMEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )


# Parsing helpers

def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Extract YAML-style frontmatter and body from a markdown file."""
    if not content.startswith("---"):
        return {}, content

    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content

    metadata = {}
    for line in parts[1].strip().splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            metadata[key.strip()] = value.strip()

    return metadata, parts[2].strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)

    return chunks


def chunk_by_sections(content: str) -> list[tuple[str, str]]:
    """
    Split a markdown file by ## headers into (section_title, section_body) pairs.
    Falls back to treating the whole file as one chunk if no ## headers found.
    """
    parts = re.split(r"^(##\s+.+)$", content, flags=re.MULTILINE)

    if len(parts) == 1:
        return [("General", content.strip())]

    sections = []
    for i in range(1, len(parts), 2):
        header = parts[i].strip().lstrip("#").strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if body:
            sections.append((header, body))

    return sections


# ID generation

def make_id(namespace: str, index: int) -> str:
    """Deterministic chunk ID from a namespace string + index."""
    raw = f"{namespace}::{index}"
    return hashlib.md5(raw.encode()).hexdigest()


# Document builders

def build_processed_docs(md_files: list) -> tuple[list, list, list]:
    """
    Parse processed/ markdown files (with frontmatter).
    Returns (documents, metadatas, ids).
    """
    documents, metadatas, ids = [], [], []

    for md_file in md_files:
        try:
            content = md_file.read_text(encoding="utf-8")
            metadata, body = parse_frontmatter(content)

            if not body.strip():
                continue

            title     = metadata.get("title", "")
            url       = metadata.get("url", "")
            college   = metadata.get("college", md_file.parent.name)
            # Stable per-page identifier used to "expand" chunk hits into the full page later.
            # Prefer an explicit frontmatter id; otherwise use the file path relative to processed/.
            doc_id = metadata.get("id") or str(md_file.relative_to(PROCESSED_DIR))

            for i, chunk in enumerate(chunk_text(body)):
                enriched = ""
                if title:
                    enriched += f"Title: {title}\n"
                if url:
                    enriched += f"URL: {url}\n"
                enriched += f"\n{chunk}"

                documents.append(enriched)
                metadatas.append({
                    "type":        "scraped_content",
                    "priority":    "high",
                    "title":       title,
                    "url":         url,
                    "source_file": metadata.get("source_file", md_file.name),
                    "college":     college,
                    "depth":       metadata.get("depth", ""),
                    "doc_id":      doc_id,
                    "chunk_index": i,
                })
                ids.append(make_id(doc_id, i))

        except Exception as e:
            print(f"  [ERROR] {md_file}: {e}")

    return documents, metadatas, ids


def build_section_docs(file_path, source_name: str, priority: str) -> tuple[list, list, list]:
    """
    Parse a markdown file by ## section headers.
    Returns (documents, metadatas, ids).
    """
    documents, metadatas, ids = [], [], []

    if not file_path.exists():
        print(f"  [WARNING] File not found: {file_path}")
        return documents, metadatas, ids

    content = file_path.read_text(encoding="utf-8")
    sections = chunk_by_sections(content)

    for i, (section_title, body) in enumerate(sections):
        enriched = f"Source: {source_name}\nSection: {section_title}\n\n{body}"

        documents.append(enriched)
        metadatas.append({
            "type":     "cair_info",
            "priority": priority,
            "title":    section_title,
            "url":      "",
        })
        ids.append(make_id(source_name, i))

    return documents, metadatas, ids


# Upsert

def upsert_batched(collection, documents: list, metadatas: list, ids: list, label: str = ""):
    """Upsert documents to ChromaDB Cloud in batches with simple retry."""
    total = len(documents)
    if total == 0:
        return

    for i in range(0, total, INGEST_BATCH_SIZE):
        batch_docs  = documents[i : i + INGEST_BATCH_SIZE]
        batch_metas = metadatas[i : i + INGEST_BATCH_SIZE]
        batch_ids   = ids[i : i + INGEST_BATCH_SIZE]

        for attempt in range(3):
            try:
                collection.upsert(
                    documents=batch_docs,
                    metadatas=batch_metas,
                    ids=batch_ids,
                )
                break
            except Exception as e:
                if attempt < 2:
                    wait = 2 ** attempt
                    print(f"\n  [WARN] Upsert failed ({e}), retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"\n  [ERROR] Batch {i}–{i+INGEST_BATCH_SIZE} failed after 3 attempts: {e}")

        done = min(i + INGEST_BATCH_SIZE, total)
        tag  = f" [{label}]" if label else ""
        print(f"  Upserted {done}/{total} chunks...{tag}", end="\r")

    print()


# Main

def main():
    log.info("Starting ingestion")
    log.info(
        "Ingest settings",
        extra={
            "extra": {
                "collection": COLLECTION_NAME,
                "processed_dir": str(PROCESSED_DIR),
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "batch_size": INGEST_BATCH_SIZE,
                "vllm_base_url": VLLM_EMBED_BASE_URL,
                "vllm_model": VLLM_EMBED_MODEL,
            }
        },
    )
    collection = get_collection()
    existing_count = collection.count()
    log.info(
        "Collection ready",
        extra={"extra": {"collection": COLLECTION_NAME, "existing_count": existing_count}},
    )

    # Processed scraped pages
    if not PROCESSED_DIR.exists():
        log.warning("Processed directory not found", extra={"extra": {"path": str(PROCESSED_DIR.resolve())}})
    else:
        md_files = list(PROCESSED_DIR.glob("**/*.md"))
        log.info("Found scraped markdown files", extra={"extra": {"count": len(md_files)}})
        docs, metas, ids = build_processed_docs(md_files)
        log.info("Generated scraped chunks", extra={"extra": {"chunks": len(docs)}})
        upsert_batched(collection, docs, metas, ids, label="scraped")

    # CAIR Overview (high priority)
    log.info("Ingesting CAIR overview", extra={"extra": {"file": str(CAIR_OVERVIEW_FILE)}})
    docs, metas, ids = build_section_docs(CAIR_OVERVIEW_FILE, "cair_overview", priority="high")
    log.info("Generated CAIR overview chunks", extra={"extra": {"chunks": len(docs)}})
    upsert_batched(collection, docs, metas, ids, label="cair_overview")

    # Past Events (low priority)
    log.info("Ingesting past events", extra={"extra": {"file": str(PAST_EVENTS_FILE)}})
    docs, metas, ids = build_section_docs(PAST_EVENTS_FILE, "past_events", priority="low")
    log.info("Generated past events chunks", extra={"extra": {"chunks": len(docs)}})
    upsert_batched(collection, docs, metas, ids, label="past_events")

    new_count = collection.count()
    added = new_count - existing_count
    log.info("Ingestion complete", extra={"extra": {"new_count": new_count, "added": added}})


if __name__ == "__main__":
    main()