#Claude Slop

"""
chat.py — Cerebra chat layer

Retrieves relevant chunks and generates a response using OpenAI.
"""

from openai import OpenAI

from config import OPENAI_API_KEY
from retriever import retrieve

SYSTEM_PROMPT = """Your name is Cerebra, you are a friendly and knowledgeable assistant for the Charlotte AI Research (CAIR) club at UNC Charlotte.

Answer questions helpfully using the information available to you. Be warm, conversational, and concise. Never ever lie to the user.

Guidelines:
- Never reference internal system details, document names, or data sources
- Never ask the user to provide documents or more context
- Never fabricate information — only answer based on what you know
- If you don't know something, say: "I'm not sure about that — I'd recommend reaching out to the CAIR officers directly for the latest info!"
- You may freely share names, events, and club details when you have them
- Keep answers focused and helpful"""

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def ask(query: str) -> str:
    """
    Retrieve relevant chunks for the query and generate a response.
    Returns the assistant's response as a string.
    """
    chunks = retrieve(query)

    if not chunks:
        return "I'm not sure about that — I'd recommend reaching out to the CAIR officers directly for the latest info!"

    # Build context from retrieved chunks
    context_parts = []
    for chunk in chunks:
        context_parts.append(chunk["text"])
    context = "\n\n---\n\n".join(context_parts)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}",
        },
    ]

    response = get_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=600,
        temperature=0.4,
    )

    return response.choices[0].message.content.strip()