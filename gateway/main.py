from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request

from .models import ChatCompletionRequest
from .routing import RoutingError, route_request
from .routing_config import load_routing_config
from .upstream import send_to_litellm


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.routing_config = load_routing_config()
    yield


app = FastAPI(
    title="DAT AI Gateway",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "dat-ai-gateway",
    }


def _is_retryable_upstream_error(exc: Exception) -> bool:
    """Allow fallback only for eligible transient upstream failures."""

    if isinstance(
        exc,
        (httpx.TimeoutException, httpx.TransportError),
    ):
        return True

    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500

    return False


@app.post("/v1/chat/completions")
async def chat_completion(
    request: ChatCompletionRequest,
    http_request: Request,
):
    try:
        decision = await route_request(request, http_request.app.state.routing_config)
    except RoutingError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=str(exc),
        ) from exc

    models_to_try = decision.attempts

    for attempt, selected_model in enumerate(
        models_to_try,
        start=1,
    ):
        try:
            response = await send_to_litellm(
                model=selected_model,
                messages=[message.model_dump() for message in request.messages],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )

        except Exception as exc:
            # Never retry a permanent error or exceed the
            # configured list of eligible models.
            if attempt >= len(models_to_try) or not _is_retryable_upstream_error(exc):
                break

            continue

        return response

    # All permitted attempts failed.
    raise HTTPException(
        status_code=502,
        detail=("Upstream request failed; no further eligible provider is available."),
    )
