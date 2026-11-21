import hmac
import json
import os
import time
import uuid
import hashlib

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from .circuit_breaker import BREAKERS, CircuitOpenError, breaker_for
from .ledger import record_request
from .models import ChatCompletionRequest
from .rate_limit import RateLimiter
from .routing import (
    EXPLICIT_MODELS,
    canonical_auto_preset,
    get_explicit_models,
)
from .upstream import (
    open_stream_to_litellm,
    open_stream_to_semantic_router,
    send_to_litellm,
    send_to_semantic_router,
)
from .telemetry import HTTP_DURATION, HTTP_REQUESTS, configure_tracing, metrics_app
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

app = FastAPI(
    title="DAT AI Gateway",
    version="0.2.0",
)

_rate_limit = int(os.getenv("GATEWAY_RPM_LIMIT", "60"))
_rate_limiter = RateLimiter()


@app.middleware("http")
async def prometheus_request_metrics(request: Request, call_next):
    started = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        route = request.scope.get("route")
        route_label = getattr(route, "path", "unmatched")
        HTTP_REQUESTS.labels(request.method, route_label, str(status)).inc()
        HTTP_DURATION.labels(request.method, route_label).observe(time.perf_counter() - started)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)
app.mount("/metrics", metrics_app)
configure_tracing(app)


@app.get("/metrics", include_in_schema=False)
async def prometheus_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "dat-ai-gateway",
    }


async def _check_http_dependency(url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(url)
        return response.status_code < 500
    except httpx.HTTPError:
        return False


async def _check_litellm() -> bool:
    litellm_url = os.getenv("LITELLM_URL", "http://127.0.0.1:4000")
    if await _check_http_dependency(f"{litellm_url}/health/readiness"):
        return True
    return await _check_http_dependency(f"{litellm_url}/health/liveliness")


async def _check_readiness_dependencies() -> dict[str, str]:
    router_url = os.getenv("SEMANTIC_ROUTER_URL", "http://127.0.0.1:8899")

    litellm_ready = await _check_litellm()
    router_ready = await _check_http_dependency(f"{router_url}/health")

    if _rate_limiter.distributed:
        try:
            await _rate_limiter.ping()
            redis_status = "ok"
        except Exception:
            redis_status = "unavailable"
    else:
        redis_status = "not_configured"

    database_status = "ok" if litellm_ready else "unknown"

    return {
        "litellm": "ok" if litellm_ready else "unavailable",
        "semantic_router": "ok" if router_ready else "unavailable",
        "redis": redis_status,
        "database": database_status,
    }


@app.get("/health/ready")
async def readiness():
    dependencies = await _check_readiness_dependencies()
    redis_ready = dependencies["redis"] in {"ok", "not_configured"}
    required_ready = (
        dependencies["litellm"] == "ok"
        and dependencies["semantic_router"] == "ok"
        and dependencies["database"] == "ok"
        and redis_ready
    )
    status_code = 200 if required_ready else 503
    return Response(
        content=json.dumps({
            "status": "ready" if required_ready else "not_ready",
            "dependencies": dependencies,
        }),
        status_code=status_code,
        media_type="application/json",
    )


@app.get("/health/upstreams")
async def upstream_health():
    return {name: await breaker.snapshot() for name, breaker in BREAKERS.items()}


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
    if isinstance(exc, CircuitOpenError):
        return True
    if isinstance(
        exc,
        (httpx.TimeoutException, httpx.TransportError),
    ):
        return True

    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500

    return False


def _require_gateway_auth(request: Request) -> None:
    expected_key = os.getenv("GATEWAY_API_KEY")
    if not expected_key:
        return
    authorization = request.headers.get("authorization", "")
    supplied_key = authorization.removeprefix("Bearer ").strip()
    if not hmac.compare_digest(supplied_key, expected_key):
        raise HTTPException(status_code=401, detail="Invalid gateway API key")


async def _enforce_rate_limit(request: Request) -> None:
    if _rate_limit <= 0:
        return
    # Caddy overwrites X-Real-IP at the public HTTPS boundary. Direct local
    # clients use the socket peer address instead.
    identity = request.headers.get("x-real-ip") or (
        request.client.host if request.client else "unknown"
    )
    identity_hash = hashlib.sha256(identity.encode()).hexdigest()
    try:
        allowed, retry_after = await _rate_limiter.check(identity_hash, limit=_rate_limit)
    except Exception as exc:
        if _rate_limiter.distributed:
            raise HTTPException(status_code=503, detail="Shared rate limiter unavailable") from exc
        raise
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Gateway request limit exceeded",
            headers={"Retry-After": str(retry_after)},
        )


async def _guarded_call(upstream: str, function, **kwargs):
    breaker = breaker_for(upstream)
    await breaker.before_call()
    try:
        result = await function(**kwargs)
    except Exception as exc:
        if _is_retryable_upstream_error(exc):
            await breaker.failure()
        else:
            # An HTTP 4xx shows the upstream is reachable and responsive.
            await breaker.success()
        raise
    await breaker.success()
    return result


def _sse_response(client, upstream_response, *, request_id, requested_model, selected_model, start, attempts):
    async def chunks():
        completed = False
        try:
            async for chunk in upstream_response.aiter_raw():
                if chunk:
                    yield chunk
            completed = True
        finally:
            await upstream_response.aclose()
            await client.aclose()
            _safe_record_request(
                request_id=request_id,
                requested_model=requested_model,
                selected_model=selected_model,
                status="success" if completed else "error",
                latency_ms=(time.perf_counter() - start) * 1000,
                attempts=attempts,
            )

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(chunks(), media_type="text/event-stream", headers=headers)


@app.post("/v1/chat/completions")
async def chat_completion(
    request: ChatCompletionRequest,
    http_request: Request,
):
    _require_gateway_auth(http_request)
    await _enforce_rate_limit(http_request)
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    requested_model = request.model
    preset = canonical_auto_preset(requested_model)
    explicit_models = get_explicit_models()
    is_explicit = requested_model in explicit_models

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
    cache_options = (
        request.cache.model_dump(by_alias=True, exclude_none=True)
        if request.cache is not None
        else None
    )

    # ── Path 1: Explicit model selection ──────────────────────────────
    # Explicit model choice is strict: dispatches directly to LiteLLM without fallback.
    if is_explicit:
        try:
            if request.stream:
                client, upstream_response = await _guarded_call(
                    f"litellm:{requested_model}", open_stream_to_litellm,
                    model=requested_model,
                    messages=messages_payload,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    tools=request.tools,
                    tool_choice=request.tool_choice,
                    response_format=request.response_format,
                    cache=cache_options,
                )
                return _sse_response(
                    client, upstream_response, request_id=request_id,
                    requested_model=requested_model, selected_model=requested_model,
                    start=start, attempts=1,
                )
            response = await _guarded_call(
                f"litellm:{requested_model}", send_to_litellm,
                model=requested_model,
                messages=messages_payload,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                tools=request.tools,
                tool_choice=request.tool_choice,
                response_format=request.response_format,
                cache=cache_options,
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
        if request.stream:
            client, upstream_response = await _guarded_call(
                "semantic-router", open_stream_to_semantic_router,
                model=preset,
                messages=messages_payload,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                tools=request.tools,
                tool_choice=request.tool_choice,
                response_format=request.response_format,
                cache=cache_options,
            )
            return _sse_response(
                client, upstream_response, request_id=request_id,
                requested_model=requested_model, selected_model=preset,
                start=start, attempts=attempt,
            )
        response = await _guarded_call(
            "semantic-router", send_to_semantic_router,
            model=preset,
            messages=messages_payload,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            tools=request.tools,
            tool_choice=request.tool_choice,
            response_format=request.response_format,
            cache=cache_options,
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
    # For auto-local: strictly gemma4 (zero external data egress).
    # For all other auto presets: gemma4 is the resilient on-premise fallback.
    attempt = 2
    fallback = "gemma4"
    try:
        if request.stream:
            client, upstream_response = await _guarded_call(
                f"litellm:{fallback}", open_stream_to_litellm,
                model=fallback,
                messages=messages_payload,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                tools=request.tools,
                tool_choice=request.tool_choice,
                response_format=request.response_format,
                cache=cache_options,
            )
            return _sse_response(
                client, upstream_response, request_id=request_id,
                requested_model=requested_model, selected_model=fallback,
                start=start, attempts=attempt,
            )
        response = await _guarded_call(
            f"litellm:{fallback}", send_to_litellm,
            model=fallback,
            messages=messages_payload,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            tools=request.tools,
            tool_choice=request.tool_choice,
            response_format=request.response_format,
            cache=cache_options,
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


_LITELLM_ENDPOINTS = {
    "embeddings",
    "images/generations",
    "images/edits",
    "images/variations",
    "audio/transcriptions",
    "audio/translations",
    "audio/speech",
    "moderations",
    "rerank",
}
_MAX_PROXY_BODY_BYTES = int(os.getenv("GATEWAY_MAX_BODY_BYTES", str(25 * 1024 * 1024)))


@app.post("/v1/{path:path}")
async def proxy_litellm_endpoint(path: str, request: Request):
    """Proxy selected non-chat LiteLLM APIs with the same auth and rate controls."""
    _require_gateway_auth(request)
    await _enforce_rate_limit(request)
    if path not in _LITELLM_ENDPOINTS:
        raise HTTPException(status_code=404, detail="Unsupported gateway endpoint")

    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > _MAX_PROXY_BODY_BYTES:
        raise HTTPException(status_code=413, detail="Request body is too large")
    body = await request.body()
    if len(body) > _MAX_PROXY_BODY_BYTES:
        raise HTTPException(status_code=413, detail="Request body is too large")

    content_type = request.headers.get("content-type", "application/json")
    model = None
    if content_type.startswith("application/json"):
        try:
            payload = json.loads(body or b"{}")
        except json.JSONDecodeError as error:
            raise HTTPException(status_code=400, detail="Invalid JSON body") from error
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object")
        model = payload.get("model")
    elif content_type.startswith("multipart/form-data"):
        from starlette.requests import Request as StarletteRequest

        replay = StarletteRequest(request.scope, receive=lambda: _single_body(body))
        form = await replay.form()
        try:
            model = form.get("model")
        finally:
            await form.close()
    else:
        raise HTTPException(status_code=415, detail="Unsupported content type")

    explicit_models = get_explicit_models()
    if model is not None and model not in explicit_models:
        raise HTTPException(status_code=400, detail=f"Unknown model alias: {model}")
    if path != "moderations" and not model:
        raise HTTPException(status_code=400, detail="A configured model alias is required")

    upstream_name = f"litellm:{model or path}"
    try:
        await breaker_for(upstream_name).before_call()
    except CircuitOpenError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
            headers={"Retry-After": str(error.retry_after)},
        ) from error
    headers = {"Content-Type": content_type}
    master_key = os.getenv("LITELLM_MASTER_KEY")
    if master_key:
        headers["Authorization"] = f"Bearer {master_key}"
    try:
        async with httpx.AsyncClient(timeout=float(os.getenv("UPSTREAM_TIMEOUT", "180"))) as client:
            upstream_response = await client.post(
                f"{os.getenv('LITELLM_URL', 'http://127.0.0.1:4000')}/v1/{path}",
                params=request.query_params,
                content=body,
                headers=headers,
            )
    except httpx.HTTPError:
        await breaker_for(upstream_name).failure()
        raise
    if upstream_response.status_code == 429 or upstream_response.status_code >= 500:
        await breaker_for(upstream_name).failure()
    else:
        await breaker_for(upstream_name).success()

    response_headers = {}
    for header_name in ("content-type", "x-litellm-response-cost", "x-litellm-cache-key"):
        if header_name in upstream_response.headers:
            response_headers[header_name] = upstream_response.headers[header_name]
    return Response(
        content=upstream_response.content,
        status_code=upstream_response.status_code,
        headers=response_headers,
    )


async def _single_body(body: bytes):
    return {"type": "http.request", "body": body, "more_body": False}
