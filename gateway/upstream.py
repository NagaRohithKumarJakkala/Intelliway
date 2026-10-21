import os

import httpx

LITELLM_URL = os.getenv(
    "LITELLM_URL",
    "http://127.0.0.1:4000",
)

SEMANTIC_ROUTER_URL = os.getenv(
    "SEMANTIC_ROUTER_URL",
    "http://127.0.0.1:8899",
)


async def send_to_litellm(
    model: str,
    messages: list[dict],
    temperature: float | None = None,
    max_tokens: int | None = None,
):
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }

    if temperature is not None:
        payload["temperature"] = temperature

    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    headers = {
        "Authorization": f"Bearer {os.environ.get('LITELLM_MASTER_KEY', 'default-key')}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{LITELLM_URL}/v1/chat/completions",
            json=payload,
            headers=headers,
        )

    response.raise_for_status()

    return response.json()


async def send_to_semantic_router(
    messages: list[dict],
    temperature: float | None = None,
    max_tokens: int | None = None,
):
    """
    Send an automatic-routing request through the vLLM Semantic Router container.
    The container evaluates semantic rules and forwards to the upstream provider (LiteLLM).
    """
    payload = {
        "model": "vllm-sr/auto",
        "messages": messages,
        "stream": False,
    }

    if temperature is not None:
        payload["temperature"] = temperature

    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    headers = {
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{SEMANTIC_ROUTER_URL}/v1/chat/completions",
            json=payload,
            headers=headers,
        )

    response.raise_for_status()

    return response.json()
