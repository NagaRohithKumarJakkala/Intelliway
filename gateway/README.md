# AI Gateway

An OpenAI-compatible AI Gateway providing a unified interface to multiple LLM providers.

## Current Implementation

- OpenAI-compatible `POST /v1/chat/completions`
- Ollama local provider
- Google Gemini external provider
- LiteLLM provider normalization
- Deterministic request routing
- Project-level authentication
- Project isolation
- Request-level observability
- Request latency and token-usage recording
- Basic upstream failure handling

### Provider Mapping

| Logical Model | Provider |
|---|---|
| `local-general` | Ollama |
| `external-general` | Google Gemini |

## Architecture

```text
                    OpenAI-Compatible Client
                              |
                              v
                    +---------------------+
                    |    DAT Gateway      |
                    |      :8000          |
                    +---------------------+
                              |
                +-------------+-------------+
                |             |             |
                v             v             v
             Auth         Routing       Observability
                |             |
                +-------------+
                              |
                              v
                    +---------------------+
                    |      LiteLLM        |
                    |       :4000         |
                    +----------+----------+
                               |
                    +----------+----------+
                    |                     |
                    v                     v
              +-----------+        +-------------+
              |  Ollama   |        |   Gemini    |
              |  Local    |        |  External   |
              +-----------+        +-------------+
```

## Project Structure

```text
gateway/
├── gateway/
│   ├── main.py
│   ├── models.py
│   ├── routing.py
│   ├── isolation.py
│   ├── ledger.py
│   └── upstream.py
├── litellm/
│   └── config.yaml
├── tests/
├── benchmarks/
├── docs/
├── evidence/
├── .env
└── README.md
```

## Requirements

- Python
- Ollama
- LiteLLM
- Gemini API key

## Environment

Create `.env`:

```env
LITELLM_MASTER_KEY=sk-dev-master-key
GEMINI_API_KEY=<your-gemini-api-key>
LITELLM_URL=http://127.0.0.1:4000
```

Do not commit `.env`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate

pip install 'litellm[proxy]'
pip install fastapi uvicorn httpx pydantic-settings
```

Verify Ollama:

```bash
ollama list
```

## Running

### 1. Start LiteLLM

```bash
source .venv/bin/activate

set -a
source .env
set +a

litellm \
  --config litellm/config.yaml \
  --port 4000
```

### 2. Start the DAT Gateway

In another terminal:

```bash
source .venv/bin/activate

set -a
source .env
set +a

uvicorn gateway.main:app \
  --host 127.0.0.1 \
  --port 8000
```

## Health Checks

### Gateway

```bash
curl http://127.0.0.1:8000/health
```

Expected:

```json
{
  "status": "ok",
  "service": "dat-ai-gateway"
}
```

### LiteLLM

```bash
curl http://127.0.0.1:4000/health \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY"
```

Both Ollama and Gemini should be reported as healthy.

## API Tests

### Test 1 — Local/Ollama Routing

Coding requests are routed to `local-general`.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-project-a" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [
      {
        "role": "user",
        "content": "Write a Python function that calculates factorial."
      }
    ],
    "temperature": 0
  }'
```

Expected route:

```text
Gateway
  -> local-general
  -> LiteLLM
  -> Ollama
```

Expected: HTTP `200`.

---

### Test 2 — External/Gemini Routing

General requests are routed to `external-general`.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-project-a" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [
      {
        "role": "user",
        "content": "Explain why the sky appears blue."
      }
    ]
  }'
```

Expected route:

```text
Gateway
  -> external-general
  -> LiteLLM
  -> Gemini
```

Expected: HTTP `200` when Gemini is available.

---

### Test 3 — Invalid Authentication

```bash
curl -i http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer invalid-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [
      {
        "role": "user",
        "content": "Hello"
      }
    ]
  }'
```

Expected:

```text
HTTP 401
```

---

### Test 4 — Missing Authentication

```bash
curl -i http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [
      {
        "role": "user",
        "content": "Hello"
      }
    ]
  }'
```

Expected:

```text
HTTP 401
```

---

### Test 5 — Unsupported Request Field

Unknown fields must not be silently ignored.

```bash
curl -i http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-project-a" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [
      {
        "role": "user",
        "content": "Hello"
      }
    ],
    "unsupported_option": true
  }'
```

Expected:

```text
HTTP 422
```

---

### Test 6 — Project Isolation

Authenticated project identity must not be overridden by a user-supplied project ID.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-project-a" \
  -H "X-Project-ID: project-b" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [
      {
        "role": "user",
        "content": "Hello"
      }
    ]
  }'
```

Expected:

```text
project_id = project-a
```

---

## Observability

Request information is recorded in:

```text
evidence/requests.jsonl
```

View the records with:

```bash
cat evidence/requests.jsonl
```

Each record contains information such as:

```json
{
  "request_id": "...",
  "project_id": "project-a",
  "requested_model": "auto",
  "selected_model": "local-general",
  "status": "success",
  "latency_ms": 11200.74,
  "attempts": 1,
  "failure_class": null,
  "usage": {
    "completion_tokens": 327,
    "prompt_tokens": 40,
    "total_tokens": 367
  }
}
```
