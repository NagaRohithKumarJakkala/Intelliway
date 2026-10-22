# Intelliway

An OpenAI-compatible AI gateway and routing layer built with FastAPI, LiteLLM, vLLM Semantic Router, Redis, and TimescaleDB.

---

## Architecture Overview

| Component | Port | Description |
|---|---|---|
| **API Gateway** (`gateway`) | `8000` | FastAPI proxy handling request routing, retries, fallbacks, and ledger logging |
| **LiteLLM Proxy** | `4000` | Multi-provider LLM interface (Ollama, Gemini, OpenAI, etc.) |
| **Semantic Router** | `8899` | Multi-signal semantic classifier and Mixture-of-Models dispatcher |
| **TimescaleDB** | `5432` | PostgreSQL persistence for LiteLLM metrics and state |
| **Redis** | `6379` | Response caching and rate-limiting |

---

## Prerequisites

- **Docker** and **Docker Compose** (or Podman equivalent)
- **Python 3.12+** and [**uv**](https://github.com/astral-sh/uv) (recommended)
- *(Optional)* [Ollama](https://ollama.ai/) running on the host if using local models (`ollama pull qwen2.5:1.5b`)
- *(Optional)* `GEMINI_API_KEY` or `OPENAI_API_KEY` for external provider access

---

## Quick Start

### 1. Configure Environment

Copy the example environment file and set your keys and passwords:

```bash
cp .env.example .env
```

Edit `.env` to configure:
- `LITELLM_MASTER_KEY`: Secret master key for LiteLLM.
- `POSTGRES_PASSWORD` and `REDIS_PASSWORD`: Credentials for database and cache services.
- `GEMINI_API_KEY`: Required if using `external-general` (Google Gemini).
- `OLLAMA_BASE_URL`: Base URL if using local Ollama (`http://host.docker.internal:11434` for Docker or `http://host.containers.internal:11434` for Podman).

### 2. Start Infrastructure Services

Start LiteLLM, TimescaleDB, Redis, and Semantic Router:

```bash
docker compose up -d
```

Verify that all containers are healthy:

```bash
docker compose ps
```

### 3. Run the Gateway

Install Python dependencies and start the FastAPI gateway:

```bash
# Using uv (recommended)
uv sync
uv run uvicorn gateway.main:app --host 0.0.0.0 --port 8000 --reload

# Or using standard venv & pip
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn gateway.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Verification & Usage

### Health Check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status":"ok","service":"dat-ai-gateway"}
```

### Chat Completion Request

Send a request through the gateway. You can use explicit models (`local-general`, `external-general`, etc.) or one of the 4 automatic routing profiles:
- `auto` / `auto-balanced`: Balanced trade-off across cost, latency, and capability.
- `auto-cost`: Cost-minimized routing emphasizing free local execution.
- `auto-quality`: Quality-maximized routing prioritizing frontier cloud intelligence.
- `auto-local`: Strictly local routing guaranteeing zero external data egress.

For in-depth details on configuring model pools and rule priorities in vLLM Semantic Router, see [`semantic-router/README.md`](semantic-router/README.md).

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [
      {"role": "user", "content": "Write a python function to compute fibonacci numbers"}
    ],
    "temperature": 0.7
  }'
```

Request logs are recorded to `evidence/requests.jsonl`.

---

## Running Tests

Run the test suite using `pytest`:

```bash
uv run pytest
```
