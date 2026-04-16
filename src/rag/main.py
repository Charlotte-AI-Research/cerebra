"""
main.py — Cerebra RAG pipeline runner

Runs in order:
  1. Ingest — chunks + embeds data into ChromaDB Cloud
  2. Bot    — starts the Discord bot

Prerequisites:
  - Make sure your local embedding server is running in another terminal!
    Command: vllm serve Qwen/Qwen3-Embedding-0.6B --task embed --port 8001

Usage:
  python -m rag.main               # ingest + start bot
  python -m rag.main --ingest-only # only run ingest
  python -m rag.main --bot-only    # only start the bot
"""

import argparse
import subprocess
import sys
PYTHON        = sys.executable


def run_module(module: str, label: str):
    print(f"\n{'='*50}\n  {label}\n{'='*50}")
    result = subprocess.run([PYTHON, "-m", module])
    if result.returncode != 0:
        print(f"\n[ERROR] {label} failed. Fix the error above and re-run.")
        sys.exit(result.returncode)
    print(f"[OK] {label} done.")


def parse_args():
    parser = argparse.ArgumentParser(description="Cerebra RAG runner")
    parser.add_argument("--ingest-only", action="store_true", help="Only run ingest")
    parser.add_argument("--bot-only", action="store_true", help="Only start the bot")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.ingest_only:
        run_module("rag.ingest", "Ingesting data into ChromaDB")
    elif args.bot_only:
        run_module("rag.bot", "Starting Discord Bot")
    else:
        run_module("rag.ingest", "Step 1/2 — Ingesting data into ChromaDB")
        run_module("rag.bot",    "Step 2/2 — Starting Discord Bot")


if __name__ == "__main__":
    main()