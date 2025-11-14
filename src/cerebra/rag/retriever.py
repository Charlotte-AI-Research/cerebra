import os
from pathlib import Path
import chromadb

BASE_DIR = Path(__file__).resolve().parents[3]
CHROMA_PATH = BASE_DIR / "data" / "chroma_db"

client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = client.get_collection("cerebra_knowledge")


def retrieve(query: str, k: int = 5):
    results = collection.query(
        query_texts=[query],
        n_results=k,
    )
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return list(zip(docs, metas))


if __name__ == "__main__":
    q = "What is the CCI Dean's Ambassador Program?"
    for i, (doc, meta) in enumerate(retrieve(q, k=3), start=1):
        print(f"\n== Result {i} ==")
        print("Title:", meta.get("title"))
        print("URL:", meta.get("url"))
        print("Snippet:", doc[:300].replace("\n", " "))
