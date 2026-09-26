# Intelliway

A Rust-based AI Gateway that provides a single API for interacting with multiple LLM providers.

## Implemented

* Axum-based HTTP server
* `/health` endpoint
* OpenAI-compatible `/v1/chat/completions` endpoint
* Common provider interface
* Provider registry
* Mock provider
* Ollama provider
* Basic request logging

## Next

* Configuration
* Provider error handling
* More providers
* Rate limiting
* Retries and fallback
* Intelligent routing
* Caching
* Observability
* Load balancing
* Benchmarking
* Docker support

## Project Structure

```text
src/
├── main.rs
├── handlers.rs
├── models.rs
├── provider.rs
├── registry.rs
├── state.rs
└── providers/
    ├── mod.rs
    ├── mock.rs
    └── ollama.rs
```


