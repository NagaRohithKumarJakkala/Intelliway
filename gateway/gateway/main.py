import time
import uuid

from fastapi import FastAPI, Header, HTTPException

from .isolation import authenticate_project
from .ledger import record_request
from .models import ChatCompletionRequest
from .routing import fallback_model, route_request
from .upstream import send_to_litellm


app = FastAPI(
    title="DAT AI Gateway",
    version="0.1.0",
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "dat-ai-gateway",
    }


@app.post("/v1/chat/completions")
async def chat_completion(
    request: ChatCompletionRequest,
    authorization: str | None = Header(default=None),
):
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    project_id = authenticate_project(authorization)

    initial_model = route_request(request)

    models_to_try = [initial_model]

    fallback = fallback_model(initial_model)

    if fallback is not None:
        models_to_try.append(fallback)

    last_error = None

    for attempt, selected_model in enumerate(models_to_try, start=1):
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

            latency_ms = (
                time.perf_counter() - start
            ) * 1000

            record_request(
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

        except Exception as exc:
            last_error = exc

            if attempt == len(models_to_try):
                break

    latency_ms = (
        time.perf_counter() - start
    ) * 1000

    record_request(
        request_id=request_id,
        project_id=project_id,
        requested_model=request.model,
        selected_model=initial_model,
        status="error",
        latency_ms=latency_ms,
        attempts=len(models_to_try),
        failure_class=type(last_error).__name__,
    )

    raise HTTPException(
        status_code=502,
        detail="All eligible upstream providers failed",
    )
