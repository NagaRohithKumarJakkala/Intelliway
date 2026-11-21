import os

import httpx


LITELLM_URL = os.getenv("LITELLM_URL", "http://127.0.0.1:4000")
SEMANTIC_ROUTER_URL = os.getenv(
    "SEMANTIC_ROUTER_URL",
    "http://127.0.0.1:8899",
)
UPSTREAM_TIMEOUT = float(os.getenv("UPSTREAM_TIMEOUT", "30"))


def _litellm_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    master_key = os.getenv("LITELLM_MASTER_KEY")
    if master_key:
        headers["Authorization"] = f"Bearer {master_key}"
    return headers


async def send_to_litellm(
    model: str,
    messages: list[dict],
    temperature: float | None = None,
    max_tokens: int | None = None,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
    response_format: dict | None = None,
    stream: bool = False,
    cache: dict | None = None,
):
    payload = {"model": model, "messages": messages, "stream": stream}
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if tools is not None:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    if response_format is not None:
        payload["response_format"] = response_format
    if cache is not None:
        payload["cache"] = cache

    async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
        response = await client.post(
            f"{LITELLM_URL}/v1/chat/completions",
            json=payload,
            headers=_litellm_headers(),
        )
    response.raise_for_status()
    return response.json()


async def send_to_semantic_router(
    messages: list[dict] | None = None,
    model: str = "auto",
    temperature: float | None = None,
    max_tokens: int | None = None,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
    response_format: dict | None = None,
    stream: bool = False,
    cache: dict | None = None,
):
    if messages is None:
        messages = []
    payload = {"model": model, "messages": messages, "stream": stream}
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if tools is not None:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    if response_format is not None:
        payload["response_format"] = response_format
    if cache is not None:
        payload["cache"] = cache

    master_key = os.getenv("LITELLM_MASTER_KEY")
    headers = {"Content-Type": "application/json"}
    if master_key:
        headers["Authorization"] = f"Bearer {master_key}"

    async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
        response = await client.post(
            f"{SEMANTIC_ROUTER_URL}/v1/chat/completions",
            json=payload,
            headers=headers,
        )
    response.raise_for_status()
    return response.json()


async def _open_stream(url: str, payload: dict, headers: dict[str, str]):
    """Open and validate an upstream SSE response before committing client headers."""
    client = httpx.AsyncClient(timeout=httpx.Timeout(UPSTREAM_TIMEOUT, read=None))
    try:
        request = client.build_request("POST", url, json=payload, headers=headers)
        response = await client.send(request, stream=True)
        response.raise_for_status()
        return client, response
    except Exception:
        await client.aclose()
        raise


async def open_stream_to_litellm(
    model: str,
    messages: list[dict],
    temperature: float | None = None,
    max_tokens: int | None = None,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
    response_format: dict | None = None,
    cache: dict | None = None,
):
    payload = {"model": model, "messages": messages, "stream": True}
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if tools is not None:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    if response_format is not None:
        payload["response_format"] = response_format
    if cache is not None:
        payload["cache"] = cache
    return await _open_stream(
        f"{LITELLM_URL}/v1/chat/completions", payload, _litellm_headers()
    )


async def open_stream_to_semantic_router(
    model: str,
    messages: list[dict],
    temperature: float | None = None,
    max_tokens: int | None = None,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
    response_format: dict | None = None,
    cache: dict | None = None,
):
    payload = {"model": model, "messages": messages, "stream": True}
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if tools is not None:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    if response_format is not None:
        payload["response_format"] = response_format
    if cache is not None:
        payload["cache"] = cache
    headers = {"Content-Type": "application/json"}
    master_key = os.getenv("LITELLM_MASTER_KEY")
    if master_key:
        headers["Authorization"] = f"Bearer {master_key}"
    return await _open_stream(
        f"{SEMANTIC_ROUTER_URL}/v1/chat/completions", payload, headers
    )
