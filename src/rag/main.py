"""
main.py — Cerebra RAG pipeline runner

Runs in order:
  1. Ingest — chunks + embeds data into ChromaDB Cloud
  2. Bot    — starts the Discord bot

Usage:
  python main.py               # ingest + start bot
  python main.py --ingest-only # only run ingest
  python main.py --bot-only    # only start the bot
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT          = Path(__file__).parent
INGEST_SCRIPT = ROOT / "ingest.py"
BOT_SCRIPT    = ROOT / "bot.py"
PYTHON        = sys.executable


def run(script: Path, label: str):
    print(f"\n{'='*50}\n  {label}\n{'='*50}")
    result = subprocess.run([PYTHON, str(script)])
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
        run(INGEST_SCRIPT, "Ingesting data into ChromaDB")
    elif args.bot_only:
        run(BOT_SCRIPT, "Starting Discord Bot")
    else:
        run(INGEST_SCRIPT, "Step 1/2 — Ingesting data into ChromaDB")
        run(BOT_SCRIPT,    "Step 2/2 — Starting Discord Bot")


if __name__ == "__main__":
    main()