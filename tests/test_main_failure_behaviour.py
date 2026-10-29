
import httpx
from fastapi.testclient import TestClient

from gateway import main


def test_successful_request_is_forwarded_once(monkeypatch):
    provider_calls = []

    async def fake_send_to_litellm(**kwargs):
        provider_calls.append(kwargs["model"])

        return {
            "id": "test-response",
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
    provider_calls = []

    async def failing_send_to_litellm(**kwargs):
        provider_calls.append(kwargs["model"])
        raise httpx.ReadTimeout("local provider unavailable")

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
    assert provider_calls == ["local-general"]
