# Unified Memory System v0.8-alpha

_A production-grade long-term memory backend for AI agents, assistants, and generative-retrieval workloads._

---

## Overview

Unified Memory System (UMS) combines classic database technology, a high-performance FAISS index, and modern engineering practices to provide a **fast, secure, and scalable** store for “memories” (text ＋ embeddings).  
It ships with:

* **Asynchronous** storage and retrieval APIs  
* **Semantic search** via FAISS HNSW + cosine similarity  
* **PII filtering**, encryption-at-rest, automated backups  
* A **FastAPI** layer with auth, CORS, Prometheus metrics, rate limiting, and session isolation  
* Flexible configuration through environment variables, `.env`, or Python objects

---

## Key Features

| Category            | Details                                                                                 |
| ------------------- | --------------------------------------------------------------------------------------- |
| **Architecture**    | Fully async; supports high concurrency and streaming workloads                          |
| **Vector search**   | FAISS HNSW index with dynamic `ef_search` tuning for large datasets                     |
| **Privacy**         | PII redaction, Fernet AES-GCM encryption, configurable retention policies               |
| **Observability**   | Prometheus metrics, health/liveness/readiness probes, structured logging                |
| **Automation**      | Scheduled backups & restore, connection-pool recycling, schema migrations              |
| **Ease of use**     | Swagger / Redoc docs, typed Python client, Docker image, `.env`-first configuration    |

---

## Installation

### Requirements

* Python ≥ 3.10  
* Linux / macOS / Windows (Linux recommended)  
* Docker (optional, for containerised deployment)

```bash
# local install
pip install -r requirements.txt                # production
pip install -r requirements_dev.txt            # with dev tools

# or build a container
docker build -t unified-memory-system:0.8-alpha .


---

Quick Start

# 1 ) install deps (see above)

# 2 ) copy env template
cp .env.example .env          # edit DB paths, tokens, etc.

# 3 ) launch
uvicorn app:create_app --host 0.0.0.0 --port 8000 --reload

# 4 ) open docs
open http://localhost:8000/docs    # Swagger UI


---

API Examples

Add a memory

POST /api/v1/memory/add
{
  "user_id": "user123",
  "session_id": "sess1",
  "text": "The user is interested in quantum computing."
}

Semantic search

POST /api/v1/memory/get
{
  "user_id": "user123",
  "session_id": "sess1",
  "query": "quantum algorithms",
  "top_k": 5
}

Backups & metrics

GET /api/v1/backup - create / download snapshot

GET /api/v1/metrics - Prometheus endpoint

GET /api/v1/health  - deep health-check (index, DB, dependencies)



---

Development & Testing

# run unit + integration tests
pytest -q

# build & start containerised dev stack
docker compose up --build


---

Architecture

memory_system/
├── api/            # FastAPI layers, routers, middleware
├── core/           # embedding.py, index.py, vector_store.py, store.py
├── utils/          # metrics.py, security.py, cache.py, exceptions.py
├── db.py           # SQLite connection pool + migrations
└── settings.py     # Pydantic v2 config

Major dependencies: FastAPI, Pydantic, FAISS, PyTorch, Prometheus client, SQLite / aiosqlite, cryptography.


---

License

Apache 2.0


---

Authors

Evgeny Leshchenko — project maintainer
ChatGPT & Claude — co-authors (AI assistance)


---

Contact

Telegram — https://t.me/Evgeny_LV

GitHub  — https://github.com/Ingver1



---

> Unified Memory System – a universal memory backend for your next-generation AI applications.



*Re-flowed lists/headings, corrected paths (`/api/v1/...`), added missing code fences, updated version strings, and ensured consistent Markdown syntax.*

