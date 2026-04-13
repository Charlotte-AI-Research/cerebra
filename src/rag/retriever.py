"""
retriever.py — Cerebra retrieval layer

Queries ChromaDB Cloud and returns the most relevant chunks,
with high-priority documents (cair_overview) surfaced first.

Embeddings → Local Qwen3-Embedding-4B via vLLM (http://localhost:8001)
"""

from __future__ import annotations

from typing import List

import chromadb
from langchain.schema import BaseRetriever, Document
from pydantic import Field

from config import (
    CHROMA_API_KEY,
    CHROMA_TENANT,
    CHROMA_DATABASE,
    COLLECTION_NAME,
)

# Re-use the embedding function from ingest to keep a single source of truth
from ingest import VLLMEmbeddingFunction

_collection = None


def get_collection():
    global _collection
    if _collection is not None:
        return _collection

    client = chromadb.CloudClient(
        api_key=CHROMA_API_KEY,
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE,
    )

    vllm_ef = VLLMEmbeddingFunction()

    _collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=vllm_ef,
    )

    return _collection


def retrieve(query: str, n_results: int = 10) -> list[dict]:
    """
    Query ChromaDB and return a list of result dicts sorted by priority.
    Each dict has 'text' and 'metadata' keys.
    """
    collection = get_collection()

    total = collection.count()
    n = min(n_results, total)
    if n == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )

    docs      = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Bundle into dicts
    chunks = [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(docs, metadatas, distances)
    ]

    # Sort by priority: high → medium → low
    priority_order = {"high": 0, "medium": 1, "low": 2}
    chunks.sort(key=lambda c: priority_order.get(c["metadata"].get("priority", "low"), 2))

    return chunks


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