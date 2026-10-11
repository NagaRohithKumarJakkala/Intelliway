
import time
import uuid

import httpx
from fastapi import FastAPI, HTTPException

from .ledger import record_request
from .models import ChatCompletionRequest
from .routing import fallback_model, route_request
from .upstream import send_to_litellm


from fastapi import FastAPI, Header, HTTPException

from .isolation import authenticate_project



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


def _safe_record_request(**kwargs) -> bool:
    """
    Telemetry failure must not trigger another provider generation.

    Returns True if the record was written, False otherwise.
    """
    try:
        record_request(**kwargs)
        return True
    except Exception:
        # Do not log request contents, credentials, or sensitive data.
        # Durable recovery of failed writes must be added separately.
        return False


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
    authorization: str | None = Header(default=None),
):
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    project_id = authenticate_project(authorization)

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

    if request.model == "auto":
        fallback = fallback_model(initial_model)

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

        # The provider has succeeded. Logging is now independent
        # of generation and cannot cause another provider call.
        latency_ms = (
            time.perf_counter() - start
        ) * 1000

        _safe_record_request(
            request_id=request_id,
            project_id=project_id,
            requested_model=request.model,
            selected_model=selected_model,
            status="success",
            latency_ms=latency_ms,
            attempts=attempt,
            usage=response.get("usage"),
        )

        return response

    # All permitted attempts failed.
    latency_ms = (
        time.perf_counter() - start
    ) * 1000

    _safe_record_request(
        request_id=request_id,
        project_id=project_id,
        requested_model=request.model,
        selected_model=initial_model,
        status="error",
        latency_ms=latency_ms,
        attempts=attempt,
        failure_class=(
            type(last_error).__name__
            if last_error
            else "UnknownError"
        ),
    )

    raise HTTPException(
        status_code=502,
        detail=(
            "Upstream request failed; "
            "no further eligible provider is available."
        ),
    )
