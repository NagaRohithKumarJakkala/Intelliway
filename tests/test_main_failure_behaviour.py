
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
