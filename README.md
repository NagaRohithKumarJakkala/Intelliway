# Intelliway

An OpenAI-compatible AI gateway and routing layer built with FastAPI, LiteLLM, vLLM Semantic Router, Redis, and TimescaleDB.

---

## Architecture Overview

| Component | Port | Technology | Description |
|---|---|---|---|
| **API Gateway** (`gateway`) | `8000` | FastAPI, httpx, Pydantic | Ingress proxy handling request routing, retries, fallbacks, and ledger logging |
| **Semantic Router** | `8899` | vLLM Semantic Router (`vllm-sr`) | Multi-signal semantic classifier and Mixture-of-Models dispatcher |
| **LiteLLM Proxy** | `4000` | LiteLLM | Multi-provider LLM interface (Ollama, Gemini, OpenAI, etc.) |
| **TimescaleDB** | `5432` | PostgreSQL 17 + TimescaleDB | Persistent database for LiteLLM metrics, state, and spend tracking |
| **Redis** | `6379` | Redis 7 (Alpine) | Response caching and rate-limiting |

---

## Prerequisites

Before starting, ensure you have the following installed:

- **Docker** and **Docker Compose** (v2.20+ or Podman equivalent)
- **Python 3.12+**
- [**uv**](https://github.com/astral-sh/uv) (recommended) or standard Python `venv` + `pip`
- *(Optional - for local inference)* [Ollama](https://ollama.ai/) installed and running on the host machine
- *(Optional - for cloud providers)* API keys for external models (e.g. `GEMINI_API_KEY`, `OPENAI_API_KEY`)

---

## Setup Instructions

Follow these step-by-step instructions to set up and run Intelliway locally.

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-org/Intelliway.git
cd Intelliway
```

### Step 2: Configure Environment Variables

Copy the example environment file to `.env`:

```bash
cp .env.example .env
```

Open `.env` and review the configuration parameters:

| Variable | Description | Default / Example |
|---|---|---|
| `LITELLM_MASTER_KEY` | Master authentication key for LiteLLM Proxy | `sk-change-me-to-a-secure-master-key` |
| `LITELLM_SALT_KEY` | Salt key for LiteLLM encryption | `change-me-to-a-secure-random-salt` |
| `POSTGRES_USER` | Username for TimescaleDB | `litellm` |
| `POSTGRES_PASSWORD` | Password for TimescaleDB | `change-me-to-a-secure-db-password` |
| `POSTGRES_DB` | Database name for TimescaleDB | `litellm` |
| `REDIS_PASSWORD` | Access password for Redis cache | `change-me-to-a-secure-redis-password` |
| `OLLAMA_BASE_URL` | Base URL used by Docker to connect to host Ollama | `http://host.docker.internal:11434` *(Docker)* or `http://host.containers.internal:11434` *(Podman)* |
| `GEMINI_API_KEY` | API Key for Google Gemini (required for `external-*` models) | `AIzaSy...` (optional) |
| `OPENAI_API_KEY` | API Key for OpenAI models | `sk-...` (optional) |
| `LITELLM_URL` | Upstream LiteLLM Proxy endpoint for the Gateway | `http://127.0.0.1:4000` |
| `SEMANTIC_ROUTER_URL` | Upstream vLLM Semantic Router endpoint for the Gateway | `http://127.0.0.1:8899` |

> [!TIP]
> On **Linux** with Docker, `host.docker.internal` is supported in Docker Engine 20.10+ by default. If your containers cannot resolve the host, add `--add-host=host.docker.internal:host-gateway` or set `OLLAMA_BASE_URL=http://172.17.0.1:11434`.

---

### Step 3: Start Infrastructure Services

Start LiteLLM Proxy, TimescaleDB, Redis, and vLLM Semantic Router using Docker Compose:

```bash
docker compose up -d
```

Check the health and status of all running services:

```bash
docker compose ps
```

All 4 services should report `running` (and `healthy` for TimescaleDB and Redis):
```
NAME                   IMAGE                                                STATUS
litellm-proxy          ghcr.io/berriai/litellm:latest                       Up (running)
litellm-redis          redis:7-alpine                                       Up (healthy)
litellm-timescaledb    timescale/timescaledb:latest-pg17                    Up (healthy)
vllm-semantic-router   ghcr.io/vllm-project/semantic-router/vllm-sr:latest   Up (running)
```

To tail container logs:
```bash
docker compose logs -f semantic-router
docker compose logs -f litellm
```

---

### Step 4: (Optional) Prepare Local Ollama Models

If you plan to use local models without cloud API keys, ensure Ollama is serving on the host and pull the default model:

```bash
# Verify Ollama is running
ollama list

# Pull the default local model
ollama pull qwen2.5:1.5b
```

> [!NOTE]
> By default in `litellm/config.yaml`, the local models (`local-general`, `local-coder`, `local-reasoner`, `local-fast`) point to `ollama/qwen2.5:1.5b`. You can pull additional specialized models (e.g., `qwen2.5-coder:1.5b`, `deepseek-r1:1.5b`) and configure them in `litellm/config.yaml`.

---

### Step 5: Install Dependencies and Start the Gateway

The API Gateway runs on the host (port `8000`) and dispatches requests to the infrastructure containers.

#### Option A: Using `uv` (Recommended)

Install dependencies and start the gateway with auto-reload:

```bash
# Sync environment dependencies
uv sync

# Run the FastAPI gateway
uv run uvicorn gateway.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Option B: Using Standard Python `venv` & `pip`

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn gateway.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Verification & Usage

### 1. Gateway Health Check

Verify the gateway is operational:

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status":"ok","service":"dat-ai-gateway"}
```

---

### 2. Chat Completion Requests

#### A. Automatic Semantic Routing (`model: "auto"`)

Requests specifying automatic profiles evaluate the multi-signal routing pipeline in vLLM Semantic Router (embeddings, complexity scoring, domain classification, context length) and dispatch to the optimal model target:

- `auto` / `auto-balanced`: Balanced trade-off across cost, latency, and capability.
- `auto-cost`: Cost-minimized routing prioritizing local execution.
- `auto-quality`: Quality-maximized routing prioritizing frontier cloud reasoning.
- `auto-local`: Strictly local routing with guaranteed zero external data egress.

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

#### B. Explicit Model Selection (`model: "<alias>"`)

Clients can bypass semantic routing to query specific target models directly:
- **Local Pool**: `local-general`, `local-coder`, `local-reasoner`, `local-fast`
- **External Pool**: `external-general`, `external-frontier`, `external-coder`, `external-fast`

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-general",
    "messages": [
      {"role": "user", "content": "Hello, describe what an AI gateway does in one sentence."}
    ]
  }'
```

> [!IMPORTANT]
> **Explicit Model Invariant**: Explicit model requests never use fallbacks. If the selected provider is offline or errors, the gateway fails immediately with a `502 Bad Gateway` to preserve strict routing predictability.

---

### 3. Inspecting the Telemetry Ledger

Every invocation records an audit log to `evidence/requests.jsonl` with request UUID, requested model, selected model, attempt count, latency, and token usage:

```bash
tail -n 5 evidence/requests.jsonl
```

---

### 4. Running Automated Tests

Run the complete test suite (unit tests, failure recovery, semantic router integration, and containment checks):

```bash
uv run pytest
```

---

## Customizing Routing Policies

For detailed guides on defining anchor utterances, cosine thresholds, model pools, and priority recipes in vLLM Semantic Router, see [`semantic-router/README.md`](semantic-router/README.md).

---

## Troubleshooting

### Container cannot reach host Ollama (`Connection refused`)
- **Docker on Linux**: Make sure your firewall allows traffic on `172.17.0.1:11434` or set `OLLAMA_HOST=0.0.0.0` when running `ollama serve`.
- **Podman**: Ensure `OLLAMA_BASE_URL=http://host.containers.internal:11434` is set in `.env`.
- **macOS/Windows**: `host.docker.internal` resolves automatically in Docker Desktop.

### Port Conflicts
Ensure ports `8000` (Gateway), `4000` (LiteLLM), `8899` (Semantic Router), `5432` (TimescaleDB), and `6379` (Redis) are available on your host before running `docker compose up -d`.

### Resetting Infrastructure State
To wipe all stored database metrics and Redis cache:
```bash
docker compose down -v
docker compose up -d
```
