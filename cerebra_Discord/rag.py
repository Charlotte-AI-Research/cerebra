import os
import json
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class RAGPipeline:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.chroma_client = chromadb.Client()
        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="cair_club_data",
            embedding_function=self.openai_ef
        )
        self.load_data()

    def load_data(self):
        # Clear existing data to avoid duplicates on reload (simple approach)
        if self.collection.count() > 0:
            # In a real persistent app we might check for updates, but for this bot we'll reload on startup
            # Note: chroma_client.delete_collection("cair_club_data") might be better if we want a fresh start
            pass 

        documents = []
        metadatas = []
        ids = []

        # Load Club Summary (High Priority)
        summary_path = os.path.join(self.data_dir, "club-summary.txt")
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                summary_text = f.read()
                # Simple chunking by paragraphs or lines could be done here. 
                # For a summary, we might just treat it as one or a few chunks.
                # Let's split by double newlines for paragraphs.
                chunks = summary_text.split("\n\n")
                for i, chunk in enumerate(chunks):
                    if chunk.strip():
                        documents.append(chunk.strip())
                        metadatas.append({"source": "club_summary", "priority": "high"})
                        ids.append(f"summary_{i}")

        # Load Announcements (Low Priority)
        announcements_path = os.path.join(self.data_dir, "channel-announcements.jsonl")
        if os.path.exists(announcements_path):
            with open(announcements_path, "r") as f:
                for line in f:
                    try:
                        ann = json.loads(line)
                        content = ann.get("content", "")
                        if content:
                            # Enrich content with metadata for better context in retrieval
                            full_text = f"Announcement from {ann.get('author')} on {ann.get('timestamp')}: {content}"
                            documents.append(full_text)
                            metadatas.append({"source": "announcements", "priority": "low", "timestamp": ann.get("timestamp")})
                            ids.append(f"announcement_{ann.get('message_id')}")
                    except json.JSONDecodeError:
                        print(f"Error decoding line in announcements.jsonl: {line[:50]}...")

        if documents:
            self.collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Loaded {len(documents)} documents into ChromaDB.")

    def query(self, user_query):
        # Retrieve relevant documents
        results = self.collection.query(
            query_texts=[user_query],
            n_results=5
        )

        # Flatten results
        retrieved_docs = results['documents'][0]
        retrieved_metas = results['metadatas'][0]

        # Construct context
        context_parts = []
        for doc, meta in zip(retrieved_docs, retrieved_metas):
            priority = meta.get('priority', 'low')
            source = meta.get('source', 'unknown')
            context_parts.append(f"[{source.upper()} - {priority} priority]: {doc}")

        context = "\n\n".join(context_parts)

        # Generate response with OpenAI
        system_prompt = """You are a helpful assistant for the Charlotte AI Research (CAIR) club at UNC Charlotte.
Use the provided context to answer the user's question.
If the answer is found in the 'club_summary' (high priority), prioritize that information.
If the answer is in 'announcements', use it to provide timely or specific details.
If you don't know the answer based on the context, say so politely and suggest they contact the club officers.
Do not make up information."""

        user_message = f"Context:\n{context}\n\nQuestion: {user_query}"

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )

        return response.choices[0].message.content
