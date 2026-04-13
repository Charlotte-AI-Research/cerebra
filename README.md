# 🧠 Cerebra

Cerebra is a group project at UNC Charlotte that builds a **Retrieval-Augmented Generation (RAG)**-style assistant.  
It uses scraped university data, vector search, and large language models to answer questions about campus and CCI resources.

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Charlotte-AI-Research/cerebra.git
cd cerebra
```

### 2️⃣ Activate the environment

```bash
uv sync
source .venv/bin/activate
```

### 3️⃣ Test Environment

```bash
uv run scripts/dev_check.py
```

If correctly setup, terminal should show:

```bash
Imports OK - environment setup correctly!
```

## Commands to run bot
You need one terminal dedicated to run the embedding model.

```bash
vllm serve Qwen/Qwen3-Embedding-0.6B --quantization=fp8  --kv-cache-dtype=fp8 --port 8001 --gpu-memory-utilization 0.7
```

You need another terminal dedicated to running the bot.

```bash
uv run src/rag/main.py
```

You can also tweak the run commands slightly to run only certain parts of the bot.

```bash
uv run src/rag/main.py --ingest-only
```

or

```bash
uv run src/rag/main.py --bot-only
```