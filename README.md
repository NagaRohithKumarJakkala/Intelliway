# Intelliway

Intelliway is an OpenAI-compatible AI gateway built from FastAPI, LiteLLM,
vLLM Semantic Router, Ollama, Redis, and TimescaleDB.

This project exposes three API surfaces:

| Surface | URL | Use it for |
|---|---|---|
| LiteLLM Proxy | `http://localhost:4000` | The complete LiteLLM API, model aliases, provider integrations, tools, streaming, usage, spend tracking, caching, and the dashboard |
| Semantic Router | `http://localhost:8899` | Automatic model selection using `auto`, `auto-local`, `auto-cost`, or `auto-quality` |
| FastAPI Gateway | `http://localhost:8000` | The repository's narrow compatibility API, fallback behavior, and JSONL request ledger |

Use LiteLLM directly when you need all LiteLLM features. Use Semantic Router
when the request should be classified and dispatched automatically.

## 1. Prerequisites

Install:

- Docker Engine and Docker Compose v2
- Python 3.12+ and [uv](https://docs.astral.sh/uv/) for the optional FastAPI gateway
- Ollama for local inference, unless you only use cloud providers

Clone the repository and enter it:

```bash
git clone https://github.com/NagaRohithKumarJakkala/Intelliway.git
cd Intelliway
```

## 2. Configure environment variables

Create the local environment file:

```bash
cp .env.example .env
```

At minimum, replace the development values for:

```dotenv
LITELLM_MASTER_KEY=sk-your-secure-master-key
LITELLM_SALT_KEY=your-secure-random-salt
POSTGRES_PASSWORD=your-secure-db-password
REDIS_PASSWORD=your-secure-redis-password
```

For the default local models, Docker LiteLLM connects to Ollama on the host
through:

```dotenv
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

For OpenRouter GLM models, add:

```dotenv
OPENROUTER_API_KEY=your-openrouter-key
```

For other providers, add the corresponding key, for example
`GEMINI_API_KEY` or `OPENAI_API_KEY`. Never commit `.env` or API keys.

## 3. Start the services

If using local inference, start Ollama and pull the configured model:

```bash
ollama serve
ollama pull gemma4
ollama pull ornith
ollama list
```

Start LiteLLM, its database and cache, and Semantic Router:

```bash
docker compose up -d
docker compose ps
```

Expected services:

- `litellm-proxy` on port `4000`
- `vllm-semantic-router` on port `8899`
- `litellm-timescaledb` on port `5432`
- `litellm-redis` on port `6379`

View logs:

```bash
docker compose logs -f litellm
docker compose logs -f semantic-router
```

Stop the Compose-managed services without deleting persistent data:

```bash
docker compose down
```

Delete the database and Redis volumes only when a full reset is intended:

```bash
docker compose down -v
docker compose up -d
```

## 4. Check that the services work

Set the LiteLLM key in the shell:

```bash
KEY=$(grep '^LITELLM_MASTER_KEY=' .env | cut -d= -f2-)
```

Check LiteLLM:

```bash
curl -sS http://localhost:4000/health/liveliness
curl -sS http://localhost:4000/v1/models \
  -H "Authorization: Bearer $KEY"
```

Check Semantic Router:

```bash
curl -sS http://localhost:8899/health
```

Check the optional FastAPI gateway:

```bash
curl -sS http://localhost:8000/health
```

## 5. Access the LiteLLM dashboard

Open [http://localhost:4000/ui/](http://localhost:4000/ui/) in a browser and
sign in with `LITELLM_MASTER_KEY`.

The dashboard can be used to inspect requests, spend, keys, teams, provider
configuration, and model activity. The file-based model aliases used by this
repository are defined in [`litellm/config.yaml`](./litellm/config.yaml).
Changes to that file require recreating LiteLLM:

```bash
docker compose up -d --force-recreate litellm
```

## 6. Use each API surface

### A. LiteLLM directly: all LiteLLM features

Use port `4000` for streaming, tools, structured output, provider-specific
parameters, embeddings, image APIs, model management, and other LiteLLM
features.

List configured aliases:

```bash
curl -sS http://localhost:4000/v1/models \
  -H "Authorization: Bearer $KEY"
```

Send a local request:

```bash
curl -sS http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4",
    "messages": [{"role": "user", "content": "Explain an API gateway in one sentence."}]
  }'
```

Send a GLM request through OpenRouter:

```bash
curl -sS http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-5.3",
    "messages": [{"role": "user", "content": "Say hello briefly."}],
    "max_tokens": 256
  }'
```

Stream a response:

```bash
curl -N http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4",
    "messages": [{"role": "user", "content": "List three short ideas."}],
    "stream": true
  }'
```

### B. Semantic Router: automatic model selection

The router accepts OpenAI-compatible chat requests and selects a LiteLLM alias
from the configured ten-model catalogue: two local Ollama models and eight
OpenRouter models. The `auto-cost`, `auto-balanced`, and `auto-quality`
entrypoints apply their configured model pools and RouterDC selection policy.

```bash
curl -sS http://localhost:8899/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Write a Python function to sort a list."}]
  }'
```

Available automatic entrypoints:

| Model | Behavior |
|---|---|
| `auto` / `auto-balanced` | Default keyword and modality routing |
| `auto-local` | Always selects a local model |
| `auto-cost` | Uses the default cost-oriented recipe |
| `auto-quality` | Uses the external frontier alias |

Example quality-first request:

```bash
curl -sS http://localhost:8899/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto-quality",
    "messages": [{"role": "user", "content": "Explain a distributed migration strategy."}]
  }'
```

The router's selected alias is returned in the response `model` field.
Routing rules and entrypoints are defined in
[`semantic-router/config.yaml`](./semantic-router/config.yaml).

### C. FastAPI Gateway: compatibility and fallback API

Start it outside Docker:

```bash
uv sync
uv run uvicorn gateway.main:app --host 0.0.0.0 --port 8000 --reload
```

Then call it:

```bash
curl -sS http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Write a short Python example."}]
  }'
```

The gateway supports a restricted request schema and does not support
streaming, tools, or arbitrary LiteLLM fields. Its supported explicit aliases
and fallback behavior are implemented in [`gateway/main.py`](./gateway/main.py).
For the complete LiteLLM API, call port `4000` instead.

Requests recorded by the gateway are written to:

```bash
tail -n 5 evidence/requests.jsonl
```

## 7. Run the routing benchmark

The benchmark evaluates the MMLU-Pro subset through the OpenAI-compatible
Semantic Router endpoint. It uses `auto-cost`, `auto-balanced`, and
`auto-quality`, with the same ten-model catalogue configured in LiteLLM and
Semantic Router.

```bash
uv sync --group benchmark
uv run --group benchmark python scripts/benchmark_mmlu_pro.py prepare
uv run --group benchmark python scripts/benchmark_mmlu_pro.py run \
  --base-url http://127.0.0.1:8899/v1
uv run --group benchmark python scripts/benchmark_mmlu_pro.py summarize
```

## 8. Add a model or agent

In this repository, an “agent” is composed of:

1. A LiteLLM model alias in [`litellm/config.yaml`](./litellm/config.yaml).
2. Optionally, a Semantic Router model and decision in
   [`semantic-router/config.yaml`](./semantic-router/config.yaml).
3. Optionally, an explicit alias in the FastAPI gateway if it must be
   accessible through port `8000`.

### Add a LiteLLM model alias

Add a new entry under `model_list`. For example, an OpenRouter model:

```yaml
- model_name: external-research
  litellm_params:
    model: openrouter/z-ai/glm-4.5-air
    api_base: https://openrouter.ai/api/v1
    api_key: os.environ/OPENROUTER_API_KEY
    max_tokens: 4096
```

Or a second Ollama alias:

```yaml
- model_name: local-research
  litellm_params:
    model: ollama/qwen2.5:1.5b
    api_base: os.environ/OLLAMA_BASE_URL
```

Restart LiteLLM and verify the alias:

```bash
docker compose up -d --force-recreate litellm
curl -sS http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"local-research","messages":[{"role":"user","content":"Say hello."}]}'
```

### Make the alias routable by Semantic Router

Add the model under `providers.models`, including a backend pointing to the
internal LiteLLM service:

```yaml
- name: external-research
  provider_model_id: external-research
  api_format: openai
  backend_refs:
    - name: litellm-external-research
      endpoint: litellm:4000
      provider: openai
      auth_header: Authorization
      auth_prefix: ****** LITELLM_MASTER_KEY
      weight: 100
```

Add a model card under `routing.modelCards`:

```yaml
- name: external-research
  context_window_size: 131072
  capabilities: [chat, tools]
```

Add a decision to route a signal to it:

```yaml
- name: research_external
  description: Route research requests to the research alias.
  priority: 80
  rules:
    operator: OR
    conditions:
      - type: keyword
        name: research
  modelRefs:
    - model: external-research
  algorithm:
    type: static
```

Define the matching keyword signal in `routing.signals.keywords`, then recreate
the router:

```bash
docker compose up -d --force-recreate semantic-router
docker compose logs --tail 100 semantic-router
```

The router-to-LiteLLM endpoint must be `litellm:4000`, not
`http://localhost:4000` and not a `/v1` URL.

### Expose the alias through the FastAPI Gateway

The FastAPI gateway intentionally maintains an allowlist. Add the alias to
`EXPLICIT_MODELS` in [`gateway/routing.py`](./gateway/routing.py) and ensure
the request path is appropriate for the gateway's restricted schema. Restart
the gateway after changing Python files.

For tools, streaming, structured output, or provider-specific fields, do not
add the request to the gateway; use LiteLLM directly.

## 8. Configure routing behavior

The default routing signals are in
[`semantic-router/config.yaml`](./semantic-router/config.yaml):

- `coding`: `code`, `python`, `function`, `class`, `debug`, and similar terms
- `complex`: `architecture`, `distributed`, `prove`, `derive`, `optimize`,
  and `migration`

Decision priority is highest first. A coding request therefore selects
`local-coder` before the default local decision. Keep model aliases consistent
between both YAML files.

## 9. Tests and validation

Run the test suite:

```bash
uv run pytest
```

Validate Compose configuration:

```bash
docker compose config
```

Check configuration changes after editing:

```bash
docker compose up -d --force-recreate litellm semantic-router
docker compose ps
docker compose logs --tail 100 litellm semantic-router
```

## 10. Troubleshooting

### OpenRouter returns HTTP 402

Confirm `OPENROUTER_API_KEY` is present in `.env`, use a small
`max_tokens` value, and check the OpenRouter account balance or limits. The
default GLM aliases cap output at `4096` tokens.

### Container cannot connect to Ollama

Verify Ollama is running and listening on a host-reachable address:

```bash
ollama list
curl http://localhost:11434/api/tags
```

Docker uses `host.docker.internal`; Podman generally uses
`host.containers.internal`. Update `OLLAMA_BASE_URL` in `.env` accordingly.

### Router cannot reach LiteLLM

Check both services and use the internal hostname:

```bash
docker compose ps
docker compose logs --tail 100 semantic-router
```

Router backends must use `endpoint: litellm:4000`.

### Port conflict

The default ports are `4000`, `8899`, `8000`, `5432`, and `6379`. Stop the
conflicting process or change the host-side port in `docker-compose.yml`.

### Stale network or orphan-container warning

The gateway and Ollama may be managed outside this Compose project. The warning
is safe if those containers are still needed. Do not use
`--remove-orphans` unless those external containers should be removed.
