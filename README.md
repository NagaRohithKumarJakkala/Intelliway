# Intelliway

A Rust-based AI Gateway that provides a single OpenAI-compatible API for interacting with multiple LLM providers.

## Implemented

* Axum-based HTTP server
* `/health` endpoint
* OpenAI-compatible `/v1/chat/completions` endpoint
* Common provider interface
* Provider registry
* Mock provider
* Ollama provider
* Configuration through `config.toml`
* Logical model aliases
* Model categories/capabilities
* Model and category resolution
* Provider-specific model mapping
* Basic request and routing logs
* Provider error handling
* Routing integration tests

## Architecture

The gateway separates the client-facing model name from the underlying provider and provider-specific model.

```text
Client
  │
  │ model = "chat"
  ▼
Axum Gateway
  │
  ▼
Model / Category Resolution
  │
  ▼
Logical Model
  │
  ▼
Provider Registry
  │
  ├── Mock Provider
  │
  └── Ollama Provider
          │
          ▼
    Provider-specific Model
```

For example:

```text
chat
  ↓
qwen-small
  ↓
ollama
  ↓
qwen2.5:1.5b
```

The client only needs to know the logical model or category.

## Configuration

Providers, logical models, and categories are configured separately in `config.toml`.

```toml
[providers.mock]
type = "mock"

[providers.ollama]
type = "ollama"
base_url = "http://localhost:11434"

[models.qwen-small]
provider = "ollama"
model = "qwen2.5:1.5b"

[models.qwen-coder]
provider = "ollama"
model = "qwen2.5-coder"

[categories.chat]
models = ["qwen-small"]

[categories.coding]
models = ["qwen-coder"]
```

### Configuration layers

| Layer          | Example        | Purpose                                 |
| -------------- | -------------- | --------------------------------------- |
| Provider       | `ollama`       | Upstream provider implementation        |
| Provider model | `qwen2.5:1.5b` | Actual model understood by the provider |
| Logical model  | `qwen-small`   | Gateway-facing model identity           |
| Category       | `chat`         | Groups models by capability/task        |

## API

### Health

```http
GET /health
```

Response:

```text
OK
```

### Chat Completions

```http
POST /v1/chat/completions
```

Example using a logical model:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-small",
    "messages": [
      {
        "role": "user",
        "content": "Hello"
      }
    ]
  }'
```

Example using a category:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chat",
    "messages": [
      {
        "role": "user",
        "content": "Hello"
      }
    ]
  }'
```

## Model Resolution

A request can specify either:

* a logical model, such as `qwen-small`
* a category, such as `chat`

The gateway resolves the request into:

```text
requested name
      ↓
logical model
      ↓
provider
      ↓
provider-specific model
```

For example:

```text
chat
 ↓
qwen-small
 ↓
ollama
 ↓
qwen2.5:1.5b
```

Category selection is currently deterministic: the first model configured in the category is selected.

The resolved provider and concrete model are retained internally for future routing, observability, token accounting, and cost/billing calculations.

## Testing

Tests are maintained separately from the implementation.

```text
tests/
└── routing_tests.rs
```

Run all tests with:

```bash
cargo test
```

The current routing tests verify:

* Direct logical-model resolution
* Category resolution
* Coding category resolution
* Unknown model/category handling
* Empty category handling
* Categories referencing unknown models
* Preservation of the requested model/category name

## Project Structure

```text
.
├── Cargo.toml
├── Cargo.lock
├── config.toml
├── README.md
├── docs/
│   └── ARCHITECTURE.md
├── src/
│   ├── main.rs
│   ├── lib.rs
│   ├── config.rs
│   ├── handlers.rs
│   ├── models.rs
│   ├── provider.rs
│   ├── registry.rs
│   ├── routing.rs
│   ├── state.rs
│   └── providers/
│       ├── mod.rs
│       ├── mock.rs
│       └── ollama.rs
└── tests/
    └── routing_tests.rs
```

## Running

Start Ollama if using the Ollama provider:

```bash
ollama serve
```

Then run the gateway:

```bash
cargo run
```

The gateway listens on:

```text
127.0.0.1:8080
```

Run the test suite with:

```bash
cargo test
```

## Next

* Configuration validation
* Provider normalization
* Deterministic routing policy
* Static-routing baseline
* Rate limiting and quotas
* Bounded retries and fallback
* Token usage tracking
* Cost/billing calculation
* Request-level observability
* Caching
* Additional providers
* Load balancing
* Reproducible benchmark harness
* Docker support

