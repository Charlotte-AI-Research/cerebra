from dotenv import load_dotenv
load_dotenv()

import chromadb
import os
from collections import Counter

client = chromadb.CloudClient(
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE"),
    api_key=os.getenv("CHROMA_API_KEY")
)
collection = client.get_collection("cerebra")

# Fetch all documents and their metadata
results = collection.get(include=["metadatas"])

# Count chunks per source_file
source_files = [m.get("source_file", "unknown") for m in results["metadatas"]]
counts = Counter(source_files)

most_common_file, chunk_count = counts.most_common(1)[0]
print(f"Document with most chunks: {most_common_file}")
print(f"Chunk count: {chunk_count}")

# Fetch and print all chunks for that document
chunks = collection.get(
    where={"source_file": most_common_file},
    include=["documents", "metadatas"]
)

full_text = "\n\n".join(chunks["documents"])
print(f"\nFull reconstructed document ({len(full_text)} chars):\n")
print(full_text)