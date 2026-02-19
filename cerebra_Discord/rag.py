import os
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

BASE_DIR = Path(__file__).parent


class RAGPipeline:
    def __init__(self, data_dir=None, processed_dir=None):
        # md files (cair_overview, courses, past_events) live in cerebra_Discord/data/
        self.data_dir = Path(data_dir) if data_dir else BASE_DIR / "data"
        # scraped processed files live in cerebra/data/processed (one level up)
        self.processed_dir = Path(processed_dir) if processed_dir else BASE_DIR.parent / "data" / "processed"

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.chroma_client = chromadb.PersistentClient(path=str(self.data_dir / "chroma_db"))
        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="cair_club_data",
            embedding_function=self.openai_ef
        )
        self.load_data()
        self.load_cair_md_files()

    def parse_frontmatter(self, content):
        """Extract frontmatter metadata and body from markdown files."""
        if not content.startswith("---"):
            return {}, content
        parts = content.split("---", 2)
        if len(parts) < 3:
            return {}, content
        frontmatter_text = parts[1]
        body = parts[2].strip()
        metadata = {}
        for line in frontmatter_text.strip().split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip()
        return metadata, body

    def chunk_text(self, text, chunk_size=100, overlap=20, min_words=10):
        """Split text into overlapping chunks, skipping tiny trailing chunks."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk.split()) >= min_words:
                chunks.append(chunk)
        return chunks

    def load_cair_md_files(self):
        """Always upsert the CAIR markdown files so they stay current without re-embedding everything."""
        md_files = {
            "cair_overview.md": ("cair_overview", "high"),
            "courses_summer2026.md": ("courses", "high"),
            "past_events.md": ("past_events", "high"),
        }
        for filename, (source_name, priority) in md_files.items():
            md_path = self.data_dir / filename
            if not md_path.exists():
                print(f"WARNING: {filename} not found at {md_path.resolve()}")
                continue

            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()

            raw_chunks = [c.strip() for c in content.split("\n\n") if c.strip() and len(c.strip()) > 3]
            chunks = [c for c in raw_chunks if sum(ch.isalpha() for ch in c) >= 5]

            if not chunks:
                print(f"WARNING: No valid chunks found in {filename}")
                continue

            if len(chunks) < len(raw_chunks):
                print(f"  Filtered {len(raw_chunks) - len(chunks)} invalid chunks from {filename}")

            documents = chunks
            metadatas = [{"source": source_name, "priority": priority, "type": "cair_data"} for _ in chunks]
            ids = [f"{source_name}_{i}" for i in range(len(chunks))]

            batch_size = 25
            total = len(documents)
            for i in range(0, total, batch_size):
                self.collection.upsert(
                    documents=documents[i:i + batch_size],
                    metadatas=metadatas[i:i + batch_size],
                    ids=ids[i:i + batch_size]
                )
            print(f"Loaded {total} chunks from {filename} into ChromaDB.")

    def load_data(self):
        """Load processed markdown files into ChromaDB if not already loaded."""
        existing = self.collection.get(where={"type": "scraped_content"})
        if existing["ids"]:
            print(f"Scraped content already loaded ({len(existing['ids'])} chunks). Skipping.")
            return

        print("No existing scraped data found. Embedding all documents (this may take a while)...")

        documents = []
        metadatas = []
        ids = []

        print(f"Looking for scraped data in: {self.processed_dir.resolve()}")

        if self.processed_dir.exists():
            md_files = list(self.processed_dir.glob("**/*.md"))
            print(f"Found {len(md_files)} markdown files to process")

            for md_file in md_files:
                try:
                    content = md_file.read_text(encoding="utf-8")
                    metadata, body = self.parse_frontmatter(content)
                    if not body.strip():
                        continue

                    chunks = self.chunk_text(body)
                    for i, chunk in enumerate(chunks):
                        doc_id = f"md_{md_file.parent.name}_{md_file.stem}_{i}"
                        doc_metadata = {
                            "source": metadata.get("source", "processed"),
                            "priority": "high",
                            "type": "scraped_content",
                            "title": metadata.get("title", ""),
                            "url": metadata.get("url", ""),
                            "section": metadata.get("section", ""),
                            "source_file": metadata.get("source_file", ""),
                            "college": md_file.parent.name
                        }
                        enriched_text = f"Title: {metadata.get('title', 'N/A')}\n"
                        if metadata.get('url'):
                            enriched_text += f"URL: {metadata.get('url')}\n"
                        enriched_text += f"\n{chunk}"

                        documents.append(enriched_text)
                        metadatas.append(doc_metadata)
                        ids.append(doc_id)

                except Exception as e:
                    print(f"Error processing {md_file}: {e}")
        else:
            print(f"WARNING: processed_dir not found at {self.processed_dir.resolve()}")

        if documents:
            batch_size = 10
            total = len(documents)
            for i in range(0, total, batch_size):
                self.collection.upsert(
                    documents=documents[i:i + batch_size],
                    metadatas=metadatas[i:i + batch_size],
                    ids=ids[i:i + batch_size]
                )
                print(f"  Upserted {min(i + batch_size, total)}/{total} documents...", end="\r")

            print(f"\nLoaded {total} documents into ChromaDB.")
            scraped_count = sum(1 for m in metadatas if m['type'] == 'scraped_content')
            print(f"  - Scraped content chunks: {scraped_count}")
        else:
            print("WARNING: No documents were loaded. Check your data paths.")

    def query(self, user_query):
        try:
            results = self.collection.query(
                query_texts=[user_query],
                n_results=min(10, self.collection.count())
            )

            priority_order = {"high": 0, "low": 1}
            pairs = sorted(
                zip(results['documents'][0], results['metadatas'][0]),
                key=lambda x: priority_order.get(x[1].get('priority', 'low'), 1)
            )

            context_parts = []
            for doc, meta in pairs:
                priority = meta.get('priority', 'low')
                source = meta.get('source', 'unknown')
                doc_type = meta.get('type', 'unknown')
                title = meta.get('title', '')
                if title:
                    context_parts.append(f"[{source.upper()} - {priority} priority - {doc_type}]: {doc}")
                else:
                    context_parts.append(f"[{source.upper()} - {priority} priority]: {doc}")

            context = "\n\n".join(context_parts)

            system_prompt = """You are a helpful assistant for the Charlotte AI Research (CAIR) club at UNC Charlotte.
Use the provided context to answer the user's question.

Priority order:
1. cair_overview (high priority) - General information about CAIR
2. courses (high priority) - Summer 2026 course offerings
3. past_events (high priority) - Past CAIR events and activities
4. scraped_content (high priority) - Information from UNC Charlotte and CCI websites

If the answer is found in higher priority sources, prioritize that information.
If you don't know the answer based on the context, say so politely and suggest they contact the club officers.
Do not make up information. Always base your answer on the provided context."""

            user_message = f"Context:\n{context}\n\nQuestion: {user_query}"

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=1024,
                temperature=0.2
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Query error: {e}")
            return (
                "I'm sorry, I ran into an issue processing your question. "
                "Please try again or contact the club officers directly for assistance."
            )