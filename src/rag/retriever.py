"""
retriever.py — Cerebra retrieval layer

Queries ChromaDB Cloud and returns the most relevant chunks,
with high-priority documents (cair_overview) surfaced first.

Embeddings → Local Qwen3-Embedding-0.6B via vLLM
"""

from __future__ import annotations

from typing import List

import chromadb
from langchain.schema import BaseRetriever, Document
from pydantic import Field

from .config import (
    CHROMA_API_KEY,
    CHROMA_TENANT,
    CHROMA_DATABASE,
    COLLECTION_NAME,
)

# Re-use the embedding function from ingest to keep a single source of truth
from .ingest import VLLMEmbeddingFunction
from .logging_utils import get_logger

log = get_logger("rag.retriever")

_collection = None


def get_collection():
    global _collection
    if _collection is not None:
        return _collection

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

    vllm_ef = VLLMEmbeddingFunction()

    try:
        _collection = client.get_collection(name=COLLECTION_NAME, embedding_function=vllm_ef)
    except Exception:
        log.exception(
            "Failed to get Chroma collection",
            extra={"extra": {"collection": COLLECTION_NAME, "tenant": CHROMA_TENANT, "database": CHROMA_DATABASE}},
        )
        raise

    return _collection


def _expand_scraped_docs(collection, chunk_hits: list[dict]) -> list[dict]:
    """
    Given top-k chunk hits, expand scraped-content hits into full documents by doc_id.

    Returns a list of dicts with:
      - text: full concatenated document text
      - metadata: representative metadata (title/url/etc) + doc_id
    """
    doc_ids: list[str] = []
    for hit in chunk_hits:
        meta = hit.get("metadata") or {}
        if meta.get("type") != "scraped_content":
            continue
        doc_id = meta.get("doc_id")
        if isinstance(doc_id, str) and doc_id and doc_id not in doc_ids:
            doc_ids.append(doc_id)

    if not doc_ids:
        return []

    expanded: list[dict] = []
    for doc_id in doc_ids:
        try:
            got = collection.get(
                where={"doc_id": doc_id},
                include=["documents", "metadatas"],
            )
        except Exception:
            log.exception("Failed to expand doc_id", extra={"extra": {"doc_id": doc_id}})
            continue

        docs = got.get("documents") or []
        metas = got.get("metadatas") or []
        if not docs or not metas:
            log.warning("No chunks found for doc_id expansion", extra={"extra": {"doc_id": doc_id}})
            continue

        paired = list(zip(docs, metas))
        paired.sort(key=lambda dm: int((dm[1] or {}).get("chunk_index", 0)))

        full_text = "\n\n".join(d for d, _m in paired if isinstance(d, str) and d.strip())
        rep_meta = dict(paired[0][1] or {})
        rep_meta["doc_id"] = doc_id
        rep_meta["expanded_chunks"] = len(paired)
        rep_meta["retrieval_mode"] = "expanded_full_doc"

        expanded.append({"text": full_text, "metadata": rep_meta})

        log.info(
            "Expanded doc_id into full document",
            extra={"extra": {"doc_id": doc_id, "chunks": len(paired), "chars": len(full_text)}},
        )

    return expanded


def retrieve(query: str, n_results: int = 10) -> list[dict]:
    """
    Query ChromaDB and return a list of result dicts sorted by priority.
    Each dict has 'text' and 'metadata' keys.
    """
    collection = get_collection()
    q = (query or "").strip()
    if not q:
        log.info("Empty query passed to retriever; returning no results")
        return []

    log.info(
        "Running Chroma query",
        extra={"extra": {"query": q, "n_results": int(n_results)}},
    )

    try:
        # Avoid gating on count(); count() can be slow/fragile and a transient failure
        # looks like "no data". Just ask for top-k.
        results = collection.query(
            query_texts=[q],
            n_results=max(1, int(n_results)),
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        log.exception("Chroma query failed", extra={"extra": {"n_results": n_results, "query": q}})
        return []

    docs      = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Bundle into dicts
    chunks = [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(docs, metadatas, distances)
    ]

    # Log every retrieved chunk with a text preview so you can verify what's being pulled
    log.info(
        "Chroma query returned chunks",
        extra={"extra": {"total_chunks": len(chunks)}},
    )
    for i, c in enumerate(chunks):
        meta = c.get("metadata") or {}
        text_preview = (c.get("text") or "")[:300].replace("\n", " ")
        log.info(
            f"  chunk[{i}]",
            extra={
                "extra": {
                    "index": i,
                    "distance": round(c.get("distance", 0), 4),
                    "priority": meta.get("priority"),
                    "type": meta.get("type"),
                    "title": meta.get("title"),
                    "url": meta.get("url"),
                    "doc_id": meta.get("doc_id"),
                    "text_preview": text_preview,
                }
            },
        )

    # Sort by priority: high → medium → low
    priority_order = {"high": 0, "medium": 1, "low": 2}
    chunks.sort(key=lambda c: priority_order.get(c["metadata"].get("priority", "low"), 2))

    expanded_scraped = _expand_scraped_docs(collection, chunks)

    # Keep any non-scraped hits (e.g., cair_info) and append expanded full documents.
    non_scraped = [c for c in chunks if (c.get("metadata") or {}).get("type") != "scraped_content"]
    combined = non_scraped + expanded_scraped

    log.info(
        "Chroma retrieval complete — final context chunks",
        extra={
            "extra": {
                "returned_chunks": len(chunks),
                "non_scraped": len(non_scraped),
                "expanded_docs": len(expanded_scraped),
                "combined_total": len(combined),
                "top_distance": round(chunks[0]["distance"], 4) if chunks else None,
            }
        },
    )
    for i, c in enumerate(combined):
        meta = c.get("metadata") or {}
        text_preview = (c.get("text") or "")[:300].replace("\n", " ")
        log.info(
            f"  context[{i}]",
            extra={
                "extra": {
                    "index": i,
                    "priority": meta.get("priority"),
                    "type": meta.get("type"),
                    "title": meta.get("title"),
                    "url": meta.get("url"),
                    "doc_id": meta.get("doc_id"),
                    "retrieval_mode": meta.get("retrieval_mode", "direct"),
                    "text_preview": text_preview,
                }
            },
        )
    return combined


def format_documents(docs: list[Document]) -> str:
    """
    Serialize a list of LangChain Documents into a ranked context string
    that the system prompt can reference directly.
    """
    if not docs:
        return ""

    priority_order = {"high": 0, "medium": 1, "low": 2}
    sorted_docs = sorted(
        docs,
        key=lambda d: priority_order.get(d.metadata.get("priority", "low"), 2),
    )

    parts = []
    for doc in sorted_docs:
        priority = doc.metadata.get("priority", "low").upper()
        source   = doc.metadata.get("type", "unknown")
        title    = doc.metadata.get("title", "")
        header   = f"[{priority} | {source}]"
        if title:
            header += f" {title}"
        parts.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(parts)


class CerebraRetriever(BaseRetriever):
    """LangChain-compatible retriever that wraps ChromaDB Cloud."""

    k: int = Field(default=8)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        chunks = retrieve(query, n_results=self.k)
        return [
            Document(page_content=c["text"], metadata=c["metadata"])
            for c in chunks
        ]

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)


def get_retriever(k: int = 8) -> CerebraRetriever:
    """Return a LangChain-compatible retriever instance."""
    return CerebraRetriever(k=k)