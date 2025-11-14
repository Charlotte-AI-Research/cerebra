import os
import glob
import uuid
import yaml
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

# Paths
BASE_DIR = Path(__file__).resolve().parents[3]   # go up to project root
MD_DIR = BASE_DIR / "data" / "processed" / "markdown"
CHROMA_PATH = BASE_DIR / "data" / "chroma_db"

def parse_frontmatter(fm_text: str) -> dict:
    """
    Very simple 'key: value' parser that is tolerant of colons in values.
    Example:
      title: From Mumbai to the C-Suite: Amish Patel’s Rise
    """
    meta = {}
    for line in fm_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, val = line.split(":", 1)  # split on FIRST colon only
        meta[key.strip()] = val.strip()
    return meta


def load_markdown_docs(md_dir: Path = MD_DIR):
    docs = []
    md_paths = glob.glob(str(md_dir / "**" / "*.md"), recursive=True)

    for path in md_paths:
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        
        if text.startswith("---"):
            _, fm_text, body = text.split("---", 2)
            meta = parse_frontmatter(fm_text)
        else:
            meta = {}
            body = text


        docs.append(
            {
                "id": meta.get("id") or str(uuid.uuid4()),
                "title": meta.get("title") or path.stem,
                "url": meta.get("url", ""),
                "section": meta.get("section", ""),
                "source": meta.get("source", ""),
                "depth": meta.get("depth", ""),
                "path": str(path),
                "text": body.strip(),
            }
        )
    return docs


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200):
    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)

    return chunks


def build_chunks(docs):
    ids, documents, metadatas = [], [], []

    for doc in docs:
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            cid = f'{doc["id"]}_chunk_{i}'
            ids.append(cid)
            documents.append(chunk)
            metadatas.append(
                {
                    "page_id": doc["id"],
                    "title": doc["title"],
                    "url": doc["url"],
                    "section": doc["section"],
                    "source": doc["source"],
                    "depth": doc["depth"],
                    "chunk_index": i,
                    "path": doc["path"],
                }
            )

    return ids, documents, metadatas


def run_ingest():
    print(f"Loading markdown from {MD_DIR}...")
    docs = load_markdown_docs()
    print(f"Loaded {len(docs)} docs")

    ids, documents, metadatas = build_chunks(docs)
    print(f"Created {len(documents)} chunks")

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    # 🔁 If you're using Kronos embeddings, swap this out
    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small",
    )

    collection = client.get_or_create_collection(
        name="cerebra_knowledge",
        embedding_function=embedding_fn,
    )

    print("Adding chunks to Chroma...")
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )

    print("✅ Ingestion complete.")


if __name__ == "__main__":
    run_ingest()
