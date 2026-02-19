import os
import json
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class RAGPipeline:
    def __init__(self, data_dir="../data"):
        self.data_dir = Path(data_dir)
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
        self.load_summary()

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

    def chunk_text(self, text, chunk_size=100, overlap=20):
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks

    def load_summary(self):
        """Always upsert the club summary so it stays current without re-embedding everything."""
        summary_path = Path("data/club-summary.txt")
        if not summary_path.exists():
            print(f"WARNING: club-summary.txt not found at {summary_path.resolve()}")
            return

        with open(summary_path, "r", encoding="utf-8") as f:
            summary_text = f.read()
            chunks = summary_text.split("\n\n")
            documents, metadatas, ids = [], [], []
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    documents.append(chunk.strip())
                    metadatas.append({
                        "source": "club_summary",
                        "priority": "high",
                        "type": "summary"
                    })
                    ids.append(f"summary_{i}")

            self.collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
            print(f"Loaded {len(documents)} summary chunks into ChromaDB.")

    def load_data(self):
        # If collection already has data, skip re-embedding
        if self.collection.count() > 0:
            print(f"ChromaDB already has {self.collection.count()} documents. Skipping load.")
            return

        print("No existing data found. Embedding all documents (this may take a while)...")
        documents = []
        metadatas = []
        ids = []

        # Load Processed Markdown Files (Medium Priority)
        processed_dir = self.data_dir / "processed"
        if processed_dir.exists():
            md_files = list(processed_dir.glob("**/*.md"))
            print(f"Found {len(md_files)} markdown files to process")

            for md_file in md_files:
                try:
                    content = md_file.read_text(encoding="utf-8")
                    metadata, body = self.parse_frontmatter(content)

                    if not body.strip():
                        continue

                    chunks = self.chunk_text(body)

                    for i, chunk in enumerate(chunks):
                        doc_id = f"md_{metadata.get('id', md_file.stem)}_{i}"

                        doc_metadata = {
                            "source": metadata.get("source", "processed"),
                            "priority": "medium",
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
            print(f"WARNING: processed_dir not found at {processed_dir.resolve()}")

        # Load Announcements (Low Priority)
        announcements_path = self.data_dir / "../cerebra_Discord/data/channel-announcements.jsonl"
        if announcements_path.exists():
            with open(announcements_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        ann = json.loads(line)
                        content = ann.get("content", "")
                        if content:
                            full_text = f"Announcement from {ann.get('author')} on {ann.get('timestamp')}: {content}"
                            documents.append(full_text)
                            metadatas.append({
                                "source": "announcements",
                                "priority": "low",
                                "type": "announcement",
                                "timestamp": ann.get("timestamp")
                            })
                            ids.append(f"announcement_{ann.get('message_id')}")
                    except json.JSONDecodeError:
                        print(f"Error decoding line in announcements.jsonl: {line[:50]}...")
        else:
            print(f"WARNING: announcements not found at {announcements_path.resolve()}")

        if documents:
            batch_size = 10
            total = len(documents)
            for i in range(0, total, batch_size):
                batch_docs = documents[i:i + batch_size]
                batch_metas = metadatas[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]
                self.collection.upsert(
                    documents=batch_docs,
                    metadatas=batch_metas,
                    ids=batch_ids
                )
                print(f"  Upserted {min(i + batch_size, total)}/{total} documents...", end="\r")

            print(f"\nLoaded {total} documents into ChromaDB.")
            scraped_count = sum(1 for m in metadatas if m['type'] == 'scraped_content')
            announcement_count = sum(1 for m in metadatas if m['type'] == 'announcement')
            print(f"  - Scraped content chunks: {scraped_count}")
            print(f"  - Announcements: {announcement_count}")
        else:
            print("WARNING: No documents were loaded. Check your data paths.")

    def query(self, user_query):
        results = self.collection.query(
            query_texts=[user_query],
            n_results=min(10, self.collection.count())
        )

        retrieved_docs = results['documents'][0]
        retrieved_metas = results['metadatas'][0]

        context_parts = []
        for doc, meta in zip(retrieved_docs, retrieved_metas):
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
1. club_summary (high priority) - General information about CAIR
2. scraped_content (medium priority) - Information from UNC Charlotte and CCI websites
3. announcements (low priority) - Recent club announcements and events

If the answer is found in higher priority sources, prioritize that information.
If you don't know the answer based on the context, say so politely and suggest they contact the club officers.
Do not make up information. Always base your answer on the provided context."""

        user_message = f"Context:\n{context}\n\nQuestion: {user_query}"

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )

        return response.choices[0].message.content