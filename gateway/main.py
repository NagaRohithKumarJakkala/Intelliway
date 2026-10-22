import time
import uuid

import httpx
from fastapi import FastAPI, HTTPException

from .ledger import record_request
from .models import ChatCompletionRequest
from .routing import (
    EXPLICIT_MODELS,
    canonical_auto_preset,
)
from .upstream import send_to_litellm, send_to_semantic_router

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
):
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    requested_model = request.model
    preset = canonical_auto_preset(requested_model)
    is_explicit = requested_model in EXPLICIT_MODELS

    if not is_explicit and preset is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model alias: {requested_model}",
        )

    last_error: Exception | None = None
    messages_payload = [
        message.model_dump()
        for message in request.messages
    ]

    # ── Path 1: Explicit model selection ──────────────────────────────
    # Explicit model choice is strict: dispatches directly to LiteLLM without fallback.
    if is_explicit:
        try:
            response = await send_to_litellm(
                model=requested_model,
                messages=messages_payload,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
            latency_ms = (time.perf_counter() - start) * 1000
            _safe_record_request(
                request_id=request_id,
                requested_model=requested_model,
                selected_model=requested_model,
                status="success",
                latency_ms=latency_ms,
                attempts=1,
                usage=response.get("usage"),
            )
            return response

        except Exception as exc:
            last_error = exc
            latency_ms = (time.perf_counter() - start) * 1000
            _safe_record_request(
                request_id=request_id,
                requested_model=requested_model,
                selected_model=requested_model,
                status="error",
                latency_ms=latency_ms,
                attempts=1,
                failure_class=type(last_error).__name__,
            )
            raise HTTPException(
                status_code=502,
                detail="Upstream request failed; no further eligible provider is available.",
            ) from exc

    # ── Path 2: Automatic preset routing via vLLM Semantic Router ─────
    # Attempt 1: Dispatch to vLLM Semantic Router
    attempt = 1
    try:
        response = await send_to_semantic_router(
            model=preset,
            messages=messages_payload,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        selected_model = response.get("model", preset)
        _safe_record_request(
            request_id=request_id,
            requested_model=requested_model,
            selected_model=selected_model,
            status="success",
            latency_ms=latency_ms,
            attempts=attempt,
            usage=response.get("usage"),
        )
        return response

    except Exception as exc:
        last_error = exc
        if not _is_retryable_upstream_error(exc):
            latency_ms = (time.perf_counter() - start) * 1000
            _safe_record_request(
                request_id=request_id,
                requested_model=requested_model,
                selected_model=preset,
                status="error",
                latency_ms=latency_ms,
                attempts=attempt,
                failure_class=type(last_error).__name__,
            )
            raise HTTPException(
                status_code=502,
                detail="Upstream request failed; no further eligible provider is available.",
            ) from exc

    # Attempt 2: Fallback to LiteLLM directly on transient Semantic Router failure
    # For auto-local: strictly local-general (guarantees zero external data egress).
    # For all other auto presets: local-general acts as the resilient on-premise fallback.
    attempt = 2
    fallback = "local-general"
    try:
        response = await send_to_litellm(
            model=fallback,
            messages=messages_payload,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        _safe_record_request(
            request_id=request_id,
            requested_model=requested_model,
            selected_model=fallback,
            status="success",
            latency_ms=latency_ms,
            attempts=attempt,
            usage=response.get("usage"),
        )
        return response

    except Exception as exc:
        last_error = exc
        latency_ms = (time.perf_counter() - start) * 1000
        _safe_record_request(
            request_id=request_id,
            requested_model=requested_model,
            selected_model=preset,
            status="error",
            latency_ms=latency_ms,
            attempts=attempt,
            failure_class=type(last_error).__name__,
        )
        raise HTTPException(
            status_code=502,
            detail="Upstream request failed; no further eligible provider is available.",
        ) from exc
