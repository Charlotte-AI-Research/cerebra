#Claude Slop

"""
retriever.py — Cerebra retrieval layer

Queries ChromaDB Cloud and returns the most relevant chunks,
with high-priority documents (cair_overview) surfaced first.
"""

import chromadb
from chromadb.utils import embedding_functions

from config import (
    CHROMA_API_KEY,
    CHROMA_TENANT,
    CHROMA_DATABASE,
    COLLECTION_NAME,
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
)

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

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBEDDING_MODEL,
    )

    _collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef,
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