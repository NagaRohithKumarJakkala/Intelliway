
import httpx
from fastapi.testclient import TestClient

from gateway import main


def test_ledger_failure_does_not_generate_twice(monkeypatch):
    provider_calls = []

    async def fake_send_to_litellm(**kwargs):
        provider_calls.append(kwargs["model"])

        return {
            "id": "test-response",
            "choices": [],
            "usage": {"total_tokens": 1},
        }

    def failing_record_request(**kwargs):
        raise OSError("Simulated ledger disk failure")

    monkeypatch.setattr(
        main,
        "send_to_litellm",
        fake_send_to_litellm,
    )
    monkeypatch.setattr(
        main,
        "record_request",
        failing_record_request,
    )

    client = TestClient(main.app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "external-general",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello",
                }
            ],
        },
    )

    assert response.status_code == 200
    assert len(provider_calls) == 1
    assert provider_calls == ["external-general"]


def test_request_without_auth_header_is_allowed(monkeypatch):
    provider_calls = []

    async def fake_send_to_litellm(**kwargs):
        provider_calls.append(kwargs["model"])
        return {
            "id": "test-response-unauthenticated",
            "choices": [],
            "usage": {"total_tokens": 1},
        }

    monkeypatch.setattr(
        main,
        "send_to_litellm",
        fake_send_to_litellm,
    )

    client = TestClient(main.app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "external-general",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello",
                }
            ],
        },
    )

    assert response.status_code == 200
    assert len(provider_calls) == 1
    assert provider_calls == ["external-general"]


def test_streaming_is_rejected_before_provider_call(monkeypatch):
    provider_calls = []

    async def fake_send_to_litellm(**kwargs):
        provider_calls.append(kwargs)
        return {}

    monkeypatch.setattr(
        main,
        "send_to_litellm",
        fake_send_to_litellm,
    )

    client = TestClient(main.app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "stream": True,
            "messages": [
                {
                    "role": "user",
                    "content": "Hello",
                }
            ],
        },
    )

    assert response.status_code == 422
    assert provider_calls == []


def test_auto_local_does_not_fall_back_to_external_model(monkeypatch):
    router_calls = []
    provider_calls = []

    async def failing_send_to_semantic_router(**kwargs):
        router_calls.append(kwargs["model"])
        raise httpx.ReadTimeout("semantic router unavailable")

    async def failing_send_to_litellm(**kwargs):
        provider_calls.append(kwargs["model"])
        raise httpx.ReadTimeout("local provider unavailable")

    monkeypatch.setattr(
        main,
        "send_to_semantic_router",
        failing_send_to_semantic_router,
    )
    monkeypatch.setattr(
        main,
        "send_to_litellm",
        failing_send_to_litellm,
    )

    client = TestClient(main.app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto-local",
            "messages": [
                {
                    "role": "user",
                    "content": "Tell me a joke",
                }
            ],
        },
    )

    assert response.status_code == 502
    assert router_calls == ["auto-local"]
    assert provider_calls == ["local-general"]


def test_auto_preset_dispatches_to_semantic_router(monkeypatch):
    router_calls = []
    litellm_calls = []

    async def fake_send_to_semantic_router(**kwargs):
        router_calls.append(kwargs["model"])
        return {
            "id": "sr-response-1",
            "model": "local-coder",
            "choices": [{"message": {"role": "assistant", "content": "def hello(): pass"}}],
            "usage": {"total_tokens": 12},
        }

    async def fake_send_to_litellm(**kwargs):
        litellm_calls.append(kwargs["model"])
        return {}

    monkeypatch.setattr(main, "send_to_semantic_router", fake_send_to_semantic_router)
    monkeypatch.setattr(main, "send_to_litellm", fake_send_to_litellm)

    client = TestClient(main.app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "Write python function"}],
        },
    )

    assert response.status_code == 200
    assert router_calls == ["auto-balanced"]
    assert litellm_calls == []
    assert response.json()["model"] == "local-coder"


def test_auto_preset_fallback_to_litellm_on_retryable_sr_failure(monkeypatch):
    router_calls = []
    litellm_calls = []

    async def failing_send_to_semantic_router(**kwargs):
        router_calls.append(kwargs["model"])
        raise httpx.ConnectTimeout("Semantic Router is down")

    async def fake_send_to_litellm(**kwargs):
        litellm_calls.append(kwargs["model"])
        return {
            "id": "fallback-response",
            "model": kwargs["model"],
            "choices": [{"message": {"role": "assistant", "content": "Fallback output"}}],
            "usage": {"total_tokens": 8},
        }

    monkeypatch.setattr(main, "send_to_semantic_router", failing_send_to_semantic_router)
    monkeypatch.setattr(main, "send_to_litellm", fake_send_to_litellm)

    client = TestClient(main.app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto-quality",
            "messages": [{"role": "user", "content": "Analyze theorem"}],
        },
    )

    assert response.status_code == 200
    assert router_calls == ["auto-quality"]
    assert litellm_calls == ["local-general"]
    assert response.json()["model"] == "local-general"


def test_explicit_model_does_not_call_semantic_router(monkeypatch):
    router_calls = []
    litellm_calls = []

    async def fake_send_to_semantic_router(**kwargs):
        router_calls.append(kwargs["model"])
        return {}

    async def fake_send_to_litellm(**kwargs):
        litellm_calls.append(kwargs["model"])
        return {
            "id": "direct-litellm-response",
            "model": kwargs["model"],
            "choices": [{"message": {"role": "assistant", "content": "Direct response"}}],
            "usage": {"total_tokens": 10},
        }

    monkeypatch.setattr(main, "send_to_semantic_router", fake_send_to_semantic_router)
    monkeypatch.setattr(main, "send_to_litellm", fake_send_to_litellm)

    client = TestClient(main.app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "local-reasoner",
            "messages": [{"role": "user", "content": "1 + 1"}],
        },
    )

    assert response.status_code == 200
    assert router_calls == []
    assert litellm_calls == ["local-reasoner"]
