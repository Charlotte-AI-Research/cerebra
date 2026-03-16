"""
ingest.py — Cerebra data ingestion pipeline

Sources:
  1. data/processed/**/*.md               — scraped university pages  (priority: medium)
  2. cerebra_discord/data/cair_overview.md — first-party CAIR info    (priority: high)
  3. cerebra_discord/data/past_events.md   — CAIR event history       (priority: low)

Processed files are word-chunked.
Discord md files are section-chunked (split on ## headers).

Ingestion is incremental — deterministic chunk IDs mean upsert only
embeds chunks that are new or changed.
"""

import hashlib
import re
import time

import chromadb
from chromadb.utils import embedding_functions

from config import (
    CHROMA_API_KEY,
    CHROMA_TENANT,
    CHROMA_DATABASE,
    COLLECTION_NAME,
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    PROCESSED_DIR,
    CAIR_OVERVIEW_FILE,
    PAST_EVENTS_FILE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    INGEST_BATCH_SIZE,
)


# ChromaDB client 

def get_collection():
    client = chromadb.CloudClient(
        api_key=CHROMA_API_KEY,
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE,
    )

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBEDDING_MODEL,
    )

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef,
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

            title   = metadata.get("title", "")
            url     = metadata.get("url", "")
            college = md_file.parent.name
            # Use frontmatter id if present, otherwise relative path
            namespace = metadata.get("id") or str(md_file.relative_to(PROCESSED_DIR))

            for i, chunk in enumerate(chunk_text(body)):
                enriched = ""
                if title:
                    enriched += f"Title: {title}\n"
                if url:
                    enriched += f"URL: {url}\n"
                enriched += f"\n{chunk}"

                documents.append(enriched)
                metadatas.append({
                    "source":      metadata.get("source", "processed"),
                    "type":        "scraped_content",
                    "priority":    "medium",
                    "title":       title,
                    "url":         url,
                    "section":     metadata.get("section", ""),
                    "source_file": metadata.get("source_file", md_file.name),
                    "college":     college,
                    "depth":       metadata.get("depth", ""),
                })
                ids.append(make_id(namespace, i))

        except Exception as e:
            print(f"  [ERROR] {md_file}: {e}")

    return documents, metadatas, ids


def build_section_docs(file_path, source_name: str, priority: str) -> tuple[list, list, list]:
    """
    Parse a no-frontmatter markdown file by ## section headers.
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
            "source":   source_name,
            "type":     "cair_info",
            "priority": priority,
            "title":    section_title,
            "url":      "",
            "section":  section_title,
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


def main():
    print("Connecting to ChromaDB Cloud...")
    collection = get_collection()
    existing_count = collection.count()
    print(f"Collection '{COLLECTION_NAME}' currently has {existing_count} documents\n")

    # Processed scraped pages
    if not PROCESSED_DIR.exists():
        print(f"[WARNING] Processed directory not found: {PROCESSED_DIR.resolve()}")
    else:
        md_files = list(PROCESSED_DIR.glob("**/*.md"))
        print(f"[1/3] Found {len(md_files)} scraped markdown files in processed/")
        docs, metas, ids = build_processed_docs(md_files)
        print(f"      Generated {len(docs)} chunks")
        upsert_batched(collection, docs, metas, ids, label="scraped")

    # CAIR Overview (high priority) 
    print(f"[2/3] Ingesting CAIR overview: {CAIR_OVERVIEW_FILE.name}")
    docs, metas, ids = build_section_docs(CAIR_OVERVIEW_FILE, "cair_overview", priority="high")
    print(f"      Generated {len(docs)} section chunks")
    upsert_batched(collection, docs, metas, ids, label="cair_overview")

    # Past Events (low priority)
    print(f"[3/3] Ingesting past events: {PAST_EVENTS_FILE.name}")
    docs, metas, ids = build_section_docs(PAST_EVENTS_FILE, "past_events", priority="low")
    print(f"      Generated {len(docs)} section chunks")
    upsert_batched(collection, docs, metas, ids, label="past_events")

    # Summary 
    new_count = collection.count()
    added = new_count - existing_count
    print(f"\nDone. Collection now has {new_count} documents (+{added} new)\n")


if __name__ == "__main__":
    main()