# Intelliway

Intelliway is an OpenAI-compatible AI gateway built from FastAPI, LiteLLM,
vLLM Semantic Router, Ollama, Redis, and TimescaleDB.

This project exposes three API surfaces:

| Surface | URL | Use it for |
|---|---|---|
| LiteLLM Proxy | `http://localhost:4000` | The complete LiteLLM API, model aliases, provider integrations, tools, streaming, usage, spend tracking, caching, and the dashboard |
| Semantic Router | `http://localhost:8899` | Automatic model selection using `auto`, `auto-local`, `auto-cost`, or `auto-quality` |
| FastAPI Gateway | `http://localhost:8000` | Chat, embeddings, image, audio, moderation, and reranking APIs with auth, caching, rate limits, and fallback |

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
GATEWAY_API_KEY=optional-gateway-client-key
UPSTREAM_TIMEOUT=30
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

- `intelliway-ollama` on port `11434`
- `litellm-proxy` on port `4000`
- `vllm-semantic-router` on port `8899`
- `intelliway-gateway` on port `8000`
- `intelliway-chat-app` on port `3000`
- `litellm-timescaledb` on port `5432`
- `litellm-redis` on port `6379`
- Prometheus on `http://localhost:9090`
- Grafana on `http://localhost:3001`
- Jaeger traces on `http://localhost:16686`

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

Check the Prometheus scrape and gateway alerts at `http://localhost:9090`;
the Grafana dashboard is available at `http://localhost:3001` using the
credentials in `.env`. Jaeger shows gateway and outbound HTTP traces at
`http://localhost:16686`.

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

LiteLLM persists token usage and calculated spend in the configured
TimescaleDB/PostgreSQL database and shows it in the dashboard's Usage and Logs
pages. The FastAPI JSONL ledger is an additional local request/fallback audit
record, not the spend source of truth. Configured cloud model prices are kept
in sync with the Semantic Router price catalog; local Ollama models are priced
at zero. For newly added non-Ollama models, provide both price flags to
`intelliway-model add` so LiteLLM records spend using a defined rate.

The current gateway authenticates with a shared key and sends requests to
LiteLLM under its master key, so LiteLLM can show aggregate model spend but
cannot attribute it to individual gateway users. Per-user quotas and billing
require managed virtual keys or another trusted user identity layer. LiteLLM
provides usage and spend APIs, but this project does not connect a payment
processor or issue customer invoices.

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

It is included in the root Docker Compose stack:

```bash
docker compose up -d gateway
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

The gateway supports chat completions, explicit and automatic routing,
fallback behavior, tools, structured output, per-request LiteLLM cache
controls, and SSE when the request sets `"stream": true`.
For example, use `curl -N` to display streamed events as they arrive:

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","stream":true,"messages":[{"role":"user","content":"Count from one to five."}]}'
```

Caching is enabled in LiteLLM and backed by Redis. The gateway accepts
LiteLLM's `cache` options on chat requests:

```json
{
  "model": "auto",
  "cache": {"ttl": 300, "namespace": "assistant-v1"},
  "messages": [{"role": "user", "content": "Explain caching briefly."}]
}
```

Use `{"cache":{"no-cache":true}}` to bypass cache reads, or
`{"cache":{"no-store":true}}` to skip storing the response. Supported
options are `ttl`, `s-maxage`, `no-cache`, `no-store`, `use-cache`, and
`namespace`; LiteLLM applies them to its Redis response cache.

The gateway also proxies these LiteLLM endpoints for configured model aliases:
`/v1/embeddings`, `/v1/images/generations`, `/v1/images/edits`,
`/v1/images/variations`, `/v1/audio/transcriptions`,
`/v1/audio/translations`, `/v1/audio/speech`, `/v1/moderations`, and
`/v1/rerank`. Multipart uploads are size-limited by `GATEWAY_MAX_BODY_BYTES`.
For example, generate an embedding:

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Authorization: Bearer $GATEWAY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma4","input":"A short example."}'
```

### Security and operations

Compose publishes service ports on loopback and adds a Caddy HTTPS front
door. For public TLS, set `PUBLIC_DOMAIN` to a registered domain and create A
records pointing the root name plus `api`, `litellm`, and `router` subdomains
to this host. Open inbound TCP ports 80 and 443, then check DNS before starting
Caddy:

```bash
PUBLIC_DOMAIN=example.com PUBLIC_IPV4=203.0.113.10 uv run python scripts/check-public-dns.py
```

Caddy obtains and renews trusted certificates for those names. The default
`localhost` site uses Caddy's local CA, which browser clients may need to
trust. Direct loopback ports remain available for local development and use
plain HTTP.

The FastAPI endpoint limits each client IP to `GATEWAY_RPM_LIMIT` requests per
minute (default 60). Docker Compose uses Redis sorted-set counters and an
atomic Lua script, so limits are shared between gateway replicas. When Redis
is not configured, local development uses a process-local limiter. LiteLLM
separately enforces per-model RPM, TPM, and concurrency caps. The gateway JSONL audit file is restricted to the container
owner; records are HMAC-signed and linked to the previous record. Set a random
`GATEWAY_AUDIT_HMAC_KEY` and retain it to verify records. Keep the mounted
`evidence` directory on protected storage. Verify the ledger with
`GATEWAY_AUDIT_HMAC_KEY=... uv run python scripts/verify-audit-ledger.py`;
entries written before signing was enabled are reported as legacy unsigned
records.

Use `/health` for a lightweight liveness check and `/health/ready` for
readiness. The readiness endpoint checks LiteLLM, Semantic Router, Redis when
configured, and LiteLLM/database readiness. It returns HTTP 503 with dependency
details until the required dependencies are available.

Rotate the shared API credentials with:

```bash
uv run python scripts/rotate-api-secrets.py
docker compose up -d --force-recreate litellm semantic-router gateway
```

The script replaces `LITELLM_MASTER_KEY` and `GATEWAY_API_KEY` without printing
their values. Update clients with the new gateway key. Do not rotate
`LITELLM_SALT_KEY` casually: LiteLLM uses it to decrypt provider credentials
stored in its database. Rotate provider credentials at the provider, update
`.env`, and recreate the service that consumes them.

LiteLLM retries transient failures twice and cools down failing deployments
for 30 seconds after the configured failure threshold. `simple-shuffle`
balances calls across deployments with the same alias; the current catalog
has one deployment per alias, so add a second deployment under an alias to get
provider-level load balancing. Automatic requests also retain the gateway's
local fallback behavior.

The gateway also has per-upstream circuit breakers. After three transient
failures (configurable with `GATEWAY_CIRCUIT_FAILURES`), it opens the circuit
for 30 seconds (`GATEWAY_CIRCUIT_RESET_SECONDS`), then permits one probe
request. `/health/upstreams` reports each upstream's state and rolling health
score. Prometheus exports request, latency, upstream outcome, and circuit
metrics; alert rules cover sustained 5xx responses, open circuits, and
rate-limited traffic.

### Live Docker end-to-end suite

The live suite starts LiteLLM, Redis, Caddy, the gateway, Jaeger, and a local
OpenAI-compatible mock model, then exercises HTTPS, chat/streaming, embeddings,
cache controls, tracing, metrics, and shared Redis rate limits. It runs on
pushes and pull requests in GitHub Actions:

```bash
docker compose -f docker-compose.e2e.yml up -d --build --wait
uv run --group dev pytest tests/e2e -q
docker compose -f docker-compose.e2e.yml down -v
```

The gateway does not support arbitrary LiteLLM fields. Set `GATEWAY_API_KEY`
to require `Authorization: Bearer ...` requests; leave it empty to retain open
local access. For provider-specific fields and LiteLLM APIs outside the
allowlist above, call port `4000` instead.

LiteLLM enforces per-model deployment limits from `litellm_params`: the
current aliases are capped at 60 requests/minute, 100,000 tokens/minute, and
10 simultaneous requests each. New aliases added with `intelliway-model` get
the same defaults; pass `--rpm`, `--tpm`, or `--max-parallel-requests` to
change them. These are shared per-model caps, not individual user quotas.
LiteLLM also supports per-key and per-team limits when clients use virtual
keys with those limits configured.

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

## 8. Add or remove a model

Use `intelliway-model` to keep the LiteLLM alias and Semantic Router catalog
in sync. Run it from the repository root; after `uv sync`, `uv run` makes the
command available.

Add an Ollama model and include it in the balanced routing pool:

```bash
uv run intelliway-model add local-research \
  --provider ollama \
  --model qwen2.5:1.5b \
  --pool balanced \
  --description "Small local model for research and summaries" \
  --context-window 32768
```

Add an OpenRouter model (set `OPENROUTER_API_KEY` in `.env`):

```bash
uv run intelliway-model add external-research \
  --provider openrouter \
  --model openrouter/z-ai/glm-4.5-air \
  --api-key-env OPENROUTER_API_KEY \
  --api-base https://openrouter.ai/api/v1 \
  --pool cost \
  --prompt-price 0.1 \
  --completion-price 0.5
```

Available pools are `none`, `local`, `cost`, `balanced`, and `quality`.
Removing an alias also removes its Semantic Router model card and pool
references. The configured default model is protected from removal.

```bash
uv run intelliway-model remove local-research
```

To edit the YAML files and immediately recreate both containers in one
command, add `--restart` to `add` or `remove`:

```bash
uv run intelliway-model add local-research \
  --provider ollama --model qwen2.5:1.5b --pool balanced --restart

uv run intelliway-model remove local-research --restart
```

To restart both services after editing the configs manually:

```bash
uv run intelliway-model restart
```

Recreate both services to load the updated files:

```bash
docker compose up -d --force-recreate litellm semantic-router
docker compose logs --tail 100 litellm semantic-router
```

The CLI manages LiteLLM and Semantic Router. The FastAPI gateway dynamically
loads explicit aliases from [`litellm/config.yaml`](./litellm/config.yaml) on
each chat and proxy request, so added or removed aliases are available through
port `8000` without changing `EXPLICIT_MODELS`. Recreate LiteLLM and the gateway
when necessary to load provider credentials and configuration.

### Expose the alias through the FastAPI Gateway

The FastAPI gateway reads aliases from
[`litellm/config.yaml`](./litellm/config.yaml) on each request, so aliases
added with `intelliway-model` are accepted automatically once the LiteLLM
configuration is recreated. Ensure the request path and fields fit the
gateway's restricted schema.

For provider-specific fields, use LiteLLM directly. Streaming, tool, and
structured-output fields can pass through the gateway to LiteLLM or Semantic
Router.

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
