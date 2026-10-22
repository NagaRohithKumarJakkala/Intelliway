# vLLM Semantic Router Configuration & Routing Profiles

Intelliway integrates [vLLM Semantic Router](https://github.com/vllm-project/semantic-router) (`vllm-sr`) as an intelligent, multi-signal request classifier and Mixture-of-Models (MoM) dispatcher.

It listens on port `8899`, intercepts incoming requests specifying virtual model profiles (or explicit model aliases), evaluates a best-in-class multi-signal pipeline, and proxies the request to the upstream LiteLLM Proxy (`litellm:4000`).

---

## 1. Multi-Signal Decision Architecture

Instead of naive string matching, the router leverages vLLM Semantic Router's multi-signal evaluation layers:

1. **Semantic Embeddings (`type: embedding`)**: Dense vector representations and cosine-similarity matching against curated candidate anchor utterances (with calibrated soft-matching thresholds, e.g. `0.70` - `0.74`).
2. **Complexity Scoring (`type: complexity`)**: Differential scoring separating routine/easy tasks from high-difficulty, multi-step synthesis (e.g. `code_task:hard`, `reasoning_task:hard`).
3. **Domain Classification (`type: domain`)**: Intent classification across technical domains (`computer science`, `math`, `engineering`, `general`).
4. **Context Constraints (`type: context`)**: Token length thresholding (`short_context`, `long_context`).
5. **Priority Fallbacks (`priority: 100 -> 1`)**: Strict precedence hierarchy ending in a default baseline catch-all rule.

---

## 1. The 4 Routing Profiles

The system defines 4 canonical routing profiles, with `auto` acting as an alias for `auto-balanced`:

| Profile | Public Names / Entrypoints | Objective | Model Pool | Default Fallback |
|---|---|---|---|---|
| **`auto-balanced`** (and **`auto`**) | `auto`, `auto-balanced`, `vllm-sr/auto`, `vllm-sr/auto-balanced` | Balanced trade-off across capability, latency, and operational cost. Coding runs locally; complex reasoning routes to cloud frontier. | Local + External | `external-general` |
| **`auto-cost`** | `auto-cost`, `vllm-sr/auto-cost` | Cost minimization. Prioritizes free local inference for code, chit-chat, and text; escalates to low-cost cloud tiers only for complex queries. | Local (primary) + Low-cost External | `local-general` |
| **`auto-quality`** | `auto-quality`, `vllm-sr/auto-quality` | Quality maximization. Prioritizes frontier cloud reasoning and specialist models for engineering and proofs. Local model used only for chitchat deflection. | External (primary) + Local (deflection only) | `external-general` |
| **`auto-local`** | `auto-local`, `vllm-sr/auto-local` | Strict data containment & privacy. Routes exclusively among local models with zero external egress. | Strictly Local | `local-general` |

---

## 2. Model Pools Architecture

Intelliway logically divides models into two pools:

### Local Model Pool (Zero marginal cost, on-premise privacy)
- **`local-general`**: General-purpose local model (e.g., `ollama/qwen2.5:1.5b` or `qwen2.5:7b`).
- **`local-coder`**: Specialized coding and technical programming model (e.g., `ollama/qwen2.5-coder:1.5b` or `qwen2.5-coder:7b`).
- **`local-reasoner`**: Math and step-by-step logic specialist (e.g., `ollama/deepseek-r1:1.5b` or `deepseek-r1:7b`).
- **`local-fast`**: Low-latency, ultra-compact local model (e.g., `ollama/qwen2.5:0.5b` or `llama3.2:1b`) for trivial greetings, classifications, and basic QA.

### External Model Pool (Cloud-hosted, paid per token, high capacity)
- **`external-general`**: General-purpose cloud model (e.g., `gemini/gemini-3.8-flash`, `gpt-4o-mini`).
- **`external-frontier`**: High-reasoning cloud frontier model (e.g., `gemini/gemini-3.1-pro`, `gpt-4o`, `claude-3-7-sonnet`).
- **`external-coder`**: Advanced cloud coding model (e.g., Claude 3.7 Sonnet / Gemini Pro for complex software engineering).
- **`external-fast`**: High-throughput, cost-efficient cloud tier (e.g., `gemini/gemini-2.5-flash-lite`).

---

## 3. How to Set Up and Customize Model Pools

Since the exact model pool is operator-dependent, follow these 3 steps whenever adding or modifying models:

### Step 1: Register Upstream Target in LiteLLM (`litellm/config.yaml`)

Every model name referenced in the Semantic Router must resolve to a valid model entry in LiteLLM:

```yaml
model_list:
  # Local Pool
  - model_name: local-general
    litellm_params:
      model: ollama/qwen2.5:1.5b
      api_base: os.environ/OLLAMA_BASE_URL

  - model_name: local-coder
    litellm_params:
      model: ollama/qwen2.5-coder:1.5b
      api_base: os.environ/OLLAMA_BASE_URL

  # External Pool
  - model_name: external-general
    litellm_params:
      model: gemini/gemini-3.8-flash
      api_key: os.environ/GEMINI_API_KEY

  - model_name: external-frontier
    litellm_params:
      model: gemini/gemini-3.1-pro
      api_key: os.environ/GEMINI_API_KEY
```

> **Note:** If a specialized model (e.g. `local-coder`) is not yet deployed, you can point its `model_name` to an existing model (e.g. `ollama/qwen2.5:1.5b`) in `litellm/config.yaml` as an alias until the dedicated instance is running.

### Step 2: Declare the Model in `semantic-router/config.yaml`

Under `providers.models`, define the model card with its pricing and upstream endpoint:

```yaml
providers:
  models:
    - name: "my-custom-model"
      provider_model_id: "my-custom-model"
      pricing:
        prompt_per_1m: 0.10
        completion_per_1m: 0.40
      backend_refs:
        - name: "litellm-proxy"
          endpoint: "litellm:4000"
          protocol: "http"
          weight: 100
```

- **`weight`**: If you have multiple backends for a model, adjust `weight` (e.g., `80` / `20`) for load balancing.

### Step 3: Assign to Recipe Rules & Set Priorities

Open the desired profile under `routing.recipes` in `semantic-router/config.yaml`:

```yaml
- name: "auto-balanced"
  rules:
    - name: "my-custom-rule"
      description: "Route domain-specific queries to my custom model"
      priority: 85                       # Higher numbers evaluate first
      rules:
        operator: "OR"                   # "OR" or "AND"
        conditions:
          - type: "keyword"
            name: "my_keywords"
            keywords:
              - "finance"
              - "accounting"
              - "ledger"
      modelRefs:
        - model: "my-custom-model"       # Primary choice
        - model: "local-general"         # Fallback if primary fails
```

---

## 4. Understanding Rule Priorities & Evaluation

In each recipe:
1. **Priority Ordering**: Rules are evaluated in descending order of `priority` (e.g. `100` before `90`, down to `1`).
2. **First Match Wins**: The first rule whose condition matches determines the candidate model list (`modelRefs`).
3. **Candidate Fallback**: If the primary candidate model is unhealthy or unreachable, the router moves to the next model in `modelRefs`.
4. **Catch-All Baseline (`priority: 1`)**: Every recipe includes a default baseline with `conditions: []` that captures any request that didn't trigger higher-priority rules.

---

## 5. Testing & Verification

### Test via Semantic Router Directly (Port 8899)

```bash
# Test 'auto-local' profile (strictly local)
curl -X POST http://localhost:8899/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto-local",
    "messages": [{"role": "user", "content": "Write a python function to compute factorial"}]
  }'

# Test 'auto-cost' profile (cheapest execution)
curl -X POST http://localhost:8899/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto-cost",
    "messages": [{"role": "user", "content": "Summarize this brief announcement"}]
  }'

# Test 'auto-quality' profile (frontier intelligence)
curl -X POST http://localhost:8899/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto-quality",
    "messages": [{"role": "user", "content": "Prove the Banach fixed-point theorem step-by-step"}]
  }'
```

### Test via API Gateway (Port 8000)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Hello, how does Intelliway routing work?"}]
  }'
```
