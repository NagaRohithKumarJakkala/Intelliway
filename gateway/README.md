# AI Gateway

This repository contains a Python FastAPI gateway sitting in front of LiteLLM. The gateway exposes an OpenAI-style chat-completions endpoint, validates request payloads, selects a logical model, and logs request outcomes to JSONL evidence.

The current implementation is intentionally small and deterministic. It does not implement cost-quality routing, learned routing, quota enforcement, or a full production auth system.

## What the gateway does today

- Accepts `POST /v1/chat/completions`
- Validates request shape and supported message roles
- Rejects unsupported fields and rejects `stream=true`
- Routes requests by model alias using a deterministic heuristic
- Calls LiteLLM with the selected logical model
- Records outcome metadata in `evidence/requests.jsonl`
- Retries only for eligible transient upstream failures
- Avoids creating a second upstream call when ledger persistence fails after a successful provider response

## Current model-routing logic

The app uses two logical models:

- `local-general` -> local/LiteLLM-backed Ollama route
- `external-general` -> external/LiteLLM-backed provider route

Routing behavior:

- If `model` is explicitly set to `local-general` or `external-general`, that explicit value is used.
- If `model` is `auto`, the gateway inspects the user message text for coding keywords such as `python`, `code`, `debug`, `function`, `rust`, `javascript`, and `compile`.
- Coding-heavy prompts route to `local-general`.
- Other prompts route to `external-general`.
- Unknown model aliases raise a validation error instead of silently falling back to another provider.

## Current request validation

The request model is defined in `gateway/models.py` and enforces:

- `messages` must be present and non-empty
- supported roles are `system`, `user`, and `assistant`
- `temperature` must be between `0` and `2` when provided
- `max_tokens` must be positive when provided
- `stream` must be `False`
- extra fields are rejected

## Fallback and failure handling

The gateway tries the selected model and, for `auto` routing only, may also try the permitted fallback model.

Fallback behavior in the current code is limited to:

- `external-general` -> `local-general`
- only for retryable upstream failures such as timeout, transport error, HTTP `429`, or HTTP `5xx`

Permanent failures and non-retryable errors stop the attempt loop without silently retrying forever.

## Project structure

```text
gateway/
├── gateway/
│   ├── __init__.py
│   ├── isolation.py
│   ├── ledger.py
│   ├── main.py
│   ├── models.py
│   ├── routing.py
│   └── upstream.py
├── litellm/
│   └── config.yaml
├── tests/
│   ├── test_main_failure_behaviour.py
│   └── test_models_routing.py
├── evidence/
│   └── requests.jsonl
├── .env.example
├── requirements.lock.txt
├── README.md
└── .venv/
```

## Requirements

- Python 3.11+ recommended
- a working LiteLLM instance
- a local Ollama model for the local route
- any required external provider credentials in the configured provider stack
- a valid environment for running the app and tests

## Environment setup

Copy `.env.example` if present and set values for your local environment.

Typical variables used by the project include:

```bash
LITELLM_MASTER_KEY=your-dev-key
LITELLM_URL=http://127.0.0.1:4000
GEMINI_API_KEY=your-gemini-key
PROJECT_A_API_KEY=project-a-key
PROJECT_B_API_KEY=project-b-key
```

If the current code uses a different environment convention, follow the actual values defined in `gateway/isolation.py` and `gateway/upstream.py` rather than assuming the examples are universal.

## Running the gateway

From the project root:

```bash
cd /home/rohith/Projects/Intelliway/gateway
. .venv/bin/activate
```

Then start the app:

```bash
uvicorn gateway.main:app --host 127.0.0.1 --port 8000
```

If LiteLLM is running separately, point it at the configured proxy URL and keep the `LITELLM_MASTER_KEY` value in sync with the gateway.

## Health check

```bash
curl http://127.0.0.1:8000/health
```

Expected response shape:

```json
{
  "status": "ok",
  "service": "dat-ai-gateway"
}
```

## Example request

```bash
curl -i http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [
      {"role": "user", "content": "Write a Python function to reverse a string."}
    ],
    "temperature": 0,
    "max_tokens": 256
  }'
```

## Ledger output

Successful and failed requests are appended to the JSONL ledger:

```text
evidence/requests.jsonl
```

Each line contains metadata such as:

- request ID
- project identifier
- requested model
- selected model
- status
- latency
- number of attempts
- failure class when applicable
- usage payload returned by the provider

## Test status

The current project tests are run with:

```bash
cd /home/rohith/Projects/Intelliway/gateway
. .venv/bin/activate
python -m pytest -q
```

Current status in this workspace: the suite is passing.

## Known limits

This implementation is not a production routing or cost-optimization system. It is a small gateway layer that currently focuses on:

- request validation
- deterministic routing
- retryable upstream fallback
- request ledger logging
- testable behavior around model selection and failure cases

The following are intentionally outside the current scope:

- cost-quality routing
- advanced quota and policy enforcement
- production-grade auth and project isolation
- replica scheduling or load-aware routing
- distributed cache and benchmark orchestration

## Notes for future work

If this project grows, the next natural additions are:

1. a proper auth layer with project-scoped API keys
2. per-project policy and quota checks
3. a safe cache layer with invalidation rules
4. quality/cost-aware routing using a documented policy
5. a reproducible benchmark suite for latency, quality, and cost

For now, the gateway is best understood as a controlled, small OpenAI-compatible wrapper around LiteLLM with explicit routing rules and basic observability.
