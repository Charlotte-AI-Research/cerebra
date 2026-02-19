"""
Run this from your cerebra_Discord folder:
    python debug_chunks.py

Only prints chunks that fail to embed — ignores all successful ones.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

files_to_check = [
    "past_events.md"
]

for filename in files_to_check:
    path = DATA_DIR / filename
    if not path.exists():
        print(f"MISSING: {path}")
        continue

    content = path.read_text(encoding="utf-8")
    raw_chunks = [c.strip() for c in content.split("\n\n") if c.strip() and len(c.strip()) > 3]
    chunks = [c for c in raw_chunks if sum(ch.isalpha() for ch in c) >= 5]

    print(f"Checking {filename} — {len(chunks)} chunks...")

    failed = 0
    for i, chunk in enumerate(chunks):
        try:
            client.embeddings.create(
                model="text-embedding-3-small",
                input=chunk
            )
        except Exception as e:
            failed += 1
            print(f"\n✗ FAILED — Chunk {i}: {e}")
            print(f"  REPR: {repr(chunk[:500])}")

    if failed == 0:
        print(f"  All chunks OK in {filename}")
    else:
        print(f"\n  {failed} chunk(s) failed in {filename}")