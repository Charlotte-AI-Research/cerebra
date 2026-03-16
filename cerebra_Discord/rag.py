import os
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

BASE_DIR = Path(__file__).parent

SYSTEM_PROMPT = """Your name is Cerebra, you are a friendly and knowledgeable assistant for the Charlotte AI Research (CAIR) club at UNC Charlotte.

Answer questions helpfully using the information available to you. Be warm, conversational, and concise. Never ever lie to the user.

Guidelines:
- Never reference internal system details, document names, or data sources
- Never ask the user to provide documents or more context
- Never fabricate information — only answer based on what you know
- If you don't know something, say: "I'm not sure about that — I'd recommend reaching out to the CAIR officers directly for the latest info!"
- You may freely share names, events, and club details when you have them
- Keep answers focused and helpful"""


class RAGPipeline:
    def __init__(self, data_dir=None, processed_dir=None):
        self.data_dir = Path(data_dir) if data_dir else BASE_DIR / "data"
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
        self.conversation_history = []

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

    def chunk_text(self, text, chunk_size=200, overlap=50, min_words=10):
        """Split text into overlapping chunks, skipping tiny trailing chunks."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk.split()) >= min_words:
                chunks.append(chunk)
        return chunks

    def load_cair_md_files(self):
        """Always upsert the CAIR markdown files so they stay current."""
        md_files = {
            "cair_overview.md": ("cair_overview", "high"),
            "past_events.md":   ("past_events",   "high"),
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

            metadatas = [{"source": source_name, "priority": priority, "type": "cair_data"} for _ in chunks]
            ids = [f"{source_name}_{i}" for i in range(len(chunks))]

            for i in range(0, len(chunks), 25):
                self.collection.upsert(
                    documents=chunks[i:i + 25],
                    metadatas=metadatas[i:i + 25],
                    ids=ids[i:i + 25]
                )
            print(f"Loaded {len(chunks)} chunks from {filename} into ChromaDB.")

    def _get_already_loaded_files(self):
        """Return a set of source_file values already present in ChromaDB."""
        try:
            existing = self.collection.get(where={"type": "scraped_content"})
            loaded = set()
            for meta in existing.get("metadatas", []):
                sf = meta.get("source_file", "")
                if sf:
                    loaded.add(sf)
            return loaded
        except Exception as e:
            print(f"Warning: could not query existing files: {e}")
            return set()

    def load_data(self):
        """Load processed markdown files into ChromaDB, skipping already-embedded files."""
        print(f"Looking for scraped data in: {self.processed_dir.resolve()}")

        if not self.processed_dir.exists():
            print(f"WARNING: processed_dir not found at {self.processed_dir.resolve()}")
            return

        md_files = list(self.processed_dir.glob("**/*.md"))
        if not md_files:
            print("No markdown files found in processed_dir.")
            return

        print(f"Found {len(md_files)} markdown files in processed dir.")

        already_loaded = self._get_already_loaded_files()
        if already_loaded:
            print(f"{len(already_loaded)} file(s) already embedded — checking for new ones...")

        new_files = [
            (f, f"{f.parent.name}/{f.name}")
            for f in md_files
            if f"{f.parent.name}/{f.name}" not in already_loaded
        ]

        if not new_files:
            print("All files already loaded. Nothing new to embed.")
            return

        print(f"Embedding {len(new_files)} new file(s)...")

        documents, metadatas, ids = [], [], []

        for md_file, rel_key in new_files:
            try:
                content = md_file.read_text(encoding="utf-8")
                metadata, body = self.parse_frontmatter(content)
                if not body.strip():
                    continue

                for i, chunk in enumerate(self.chunk_text(body)):
                    enriched = f"Title: {metadata.get('title', 'N/A')}\n"
                    if metadata.get("url"):
                        enriched += f"URL: {metadata['url']}\n"
                    enriched += f"\n{chunk}"

                    documents.append(enriched)
                    metadatas.append({
                        "source":      metadata.get("source", "processed"),
                        "priority":    "high",
                        "type":        "scraped_content",
                        "title":       metadata.get("title", ""),
                        "url":         metadata.get("url", ""),
                        "section":     metadata.get("section", ""),
                        "source_file": rel_key,
                        "college":     md_file.parent.name,
                    })
                    ids.append(f"md_{md_file.parent.name}_{md_file.stem}_{i}")

            except Exception as e:
                print(f"Error processing {md_file}: {e}")

        if not documents:
            print("WARNING: No valid chunks extracted from new files.")
            return

        total = len(documents)
        for i in range(0, total, 10):
            self.collection.upsert(
                documents=documents[i:i + 10],
                metadatas=metadatas[i:i + 10],
                ids=ids[i:i + 10]
            )
            print(f"  Upserted {min(i + 10, total)}/{total} chunks...", end="\r")

        print(f"\nDone! Embedded {total} chunks from {len(new_files)} new file(s).")

    def query(self, user_query: str) -> str:
        try:
            if self.collection.count() == 0:
                return "I don't have any data loaded yet. Please contact the club officers directly."

            results = self.collection.query(
                query_texts=[user_query],
                n_results=min(20, self.collection.count())
            )

            if not results["documents"] or not results["documents"][0]:
                return "I couldn't find any relevant information. Please contact the club officers directly."

            # Filter out low-confidence results, then sort high priority first
            priority_order = {"high": 0, "low": 1}
            filtered = sorted(
                [
                    (doc, meta) for doc, meta, dist in zip(
                        results["documents"][0],
                        results["metadatas"][0],
                        results["distances"][0]
                    )
                    if dist < 1.2
                ],
                key=lambda x: priority_order.get(x[1].get("priority", "low"), 1)
            )

            if not filtered:
                return "I couldn't find any relevant information. Please contact the club officers directly."

            # Clean context — no internal labels exposed to the model
            context = "\n\n".join(doc for doc, _ in filtered if doc and doc.strip())
            if not context.strip():
                return "I couldn't find any relevant information. Please contact the club officers directly."

            # Context into system prompt, not user message
            contextual_system = SYSTEM_PROMPT + f"\n\n---\nRelevant information:\n{context}"

            MAX_EXCHANGES = 10
            trimmed_history = self.conversation_history[-(MAX_EXCHANGES * 2):]

            print(f"DEBUG - History: {len(trimmed_history)} msgs, Context: {len(context)} chars, Chunks: {len(filtered)}")

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=(
                    [{"role": "system", "content": contextual_system}]
                    + trimmed_history
                    + [{"role": "user", "content": user_query}]
                ),
                max_completion_tokens=1024,
            )

            assistant_reply = response.choices[0].message.content

            if not assistant_reply or not assistant_reply.strip():
                return "I wasn't able to generate a response. Please try rephrasing your question."

            # Only update history after a confirmed successful response
            self.conversation_history.append({"role": "user", "content": user_query})
            self.conversation_history.append({"role": "assistant", "content": assistant_reply})
            return assistant_reply

        except Exception as e:
            print(f"Query error: {e}")
            return (
                "I'm sorry, I ran into an issue processing your question. "
                "Please try again or contact the club officers directly for assistance."
            )

    def clear_history(self):
        """Start a fresh conversation."""
        self.conversation_history = []