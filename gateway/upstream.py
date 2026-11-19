import os
from dataclasses import dataclass

import httpx


LITELLM_URL = os.getenv(
    "LITELLM_URL",
    "http://127.0.0.1:4000",
)

ALLOWED_RESPONSE_HEADERS = frozenset(
    {
        "x-litellm-response-cost",
        "x-litellm-call-id",
        "x-litellm-model-id",
        "x-litellm-model-group",
        "x-litellm-version",
        "x-litellm-cache-key",
    }
)


@dataclass(frozen=True)
class LiteLLMResponse:
    body: dict
    headers: dict[str, str]


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
        "Authorization": f"Bearer {os.environ['LITELLM_MASTER_KEY']}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{LITELLM_URL}/v1/chat/completions",
            json=payload,
            headers=headers,
        )

    response.raise_for_status()

    return LiteLLMResponse(
        body=response.json(),
        headers={
            key: value
            for key, value in response.headers.items()
            if key.lower() in ALLOWED_RESPONSE_HEADERS
        },
    )
