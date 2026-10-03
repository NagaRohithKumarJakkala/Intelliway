import os

import httpx


LITELLM_URL = os.getenv(
    "LITELLM_URL",
    "http://127.0.0.1:4000",
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

    return response.json()
