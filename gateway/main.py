
import httpx
from fastapi import FastAPI, HTTPException

from .models import ChatCompletionRequest
from .routing import (
    canonical_auto_preset,
    fallback_model,
    route_request,
)
from .upstream import send_to_litellm



app = FastAPI(
    title="DAT AI Gateway",
    version="0.2.0",
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
):
    try:
        initial_model = route_request(request)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    # Explicit model selection is strict.
    # Only automatic routing is allowed to use fallback.
    models_to_try = [initial_model]

    if canonical_auto_preset(request.model) is not None:
        fallback = fallback_model(
            initial_model,
            request.model,
        )

        if fallback is not None:
            models_to_try.append(fallback)

    last_error: Exception | None = None

    for attempt, selected_model in enumerate(
        models_to_try,
        start=1,
    ):
        try:
            response = await send_to_litellm(
                model=selected_model,
                messages=[
                    message.model_dump()
                    for message in request.messages
                ],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )

        except Exception as exc:
            last_error = exc

            # Never retry a permanent error or exceed the
            # configured list of eligible models.
            if (
                attempt >= len(models_to_try)
                or not _is_retryable_upstream_error(exc)
            ):
                break

            continue

        return response

    # All permitted attempts failed.
    raise HTTPException(
        status_code=502,
        detail=(
            "Upstream request failed; "
            "no further eligible provider is available."
        ),
    )
