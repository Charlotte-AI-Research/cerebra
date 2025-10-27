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
conda env create -f environment.yml
conda activate cerebra
```


### Test Environment
```bash
python scripts/dev_check.py
```
If correctly setup, terminal should show:
```bash
Imports OK - environment setup correctly!
```

