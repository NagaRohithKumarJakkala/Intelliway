import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from gateway import main
from gateway.routing import AUTO_ROUTING_PRESETS, EXPLICIT_MODELS


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.mark.parametrize("preset", sorted(AUTO_ROUTING_PRESETS | {"auto"}))
def test_all_auto_presets_route_via_semantic_router(client, monkeypatch, preset):
    sr_calls = []
    litellm_calls = []

    expected_canonical = "auto-balanced" if preset == "auto" else preset

    async def fake_sr(**kwargs):
        sr_calls.append(kwargs)
        return {
            "id": f"sr-resp-{preset}",
            "model": "local-coder",
            "choices": [{"message": {"role": "assistant", "content": "routed content"}}],
            "usage": {"total_tokens": 15},
        }

    async def fake_litellm(**kwargs):
        litellm_calls.append(kwargs)
        return {}

    monkeypatch.setattr(main, "send_to_semantic_router", fake_sr)
    monkeypatch.setattr(main, "send_to_litellm", fake_litellm)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": preset,
            "messages": [{"role": "user", "content": "Write a quicksort implementation"}],
            "temperature": 0.5,
            "max_tokens": 100,
        },
    )

    assert response.status_code == 200
    assert len(sr_calls) == 1
    assert sr_calls[0]["model"] == expected_canonical
    assert sr_calls[0]["temperature"] == 0.5
    assert sr_calls[0]["max_tokens"] == 100
    assert litellm_calls == []

    data = response.json()
    assert data["model"] == "local-coder"
    assert data["choices"][0]["message"]["content"] == "routed content"


@pytest.mark.parametrize("model_name", sorted(EXPLICIT_MODELS))
def test_all_explicit_models_route_directly_to_litellm(client, monkeypatch, model_name):
    sr_calls = []
    litellm_calls = []

    async def fake_sr(**kwargs):
        sr_calls.append(kwargs)
        return {}

    async def fake_litellm(**kwargs):
        litellm_calls.append(kwargs)
        return {
            "id": f"litellm-resp-{model_name}",
            "model": model_name,
            "choices": [{"message": {"role": "assistant", "content": f"Hello from {model_name}"}}],
            "usage": {"total_tokens": 5},
        }

    monkeypatch.setattr(main, "send_to_semantic_router", fake_sr)
    monkeypatch.setattr(main, "send_to_litellm", fake_litellm)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code == 200
    assert sr_calls == []
    assert len(litellm_calls) == 1
    assert litellm_calls[0]["model"] == model_name

    data = response.json()
    assert data["model"] == model_name


def test_semantic_router_transient_error_falls_back_to_local_general(client, monkeypatch):
    sr_calls = []
    litellm_calls = []

    async def failing_sr(**kwargs):
        sr_calls.append(kwargs)
        # 503 Service Unavailable is a retryable server error
        req = httpx.Request("POST", "http://127.0.0.1:8899/v1/chat/completions")
        resp = httpx.Response(503, request=req)
        raise httpx.HTTPStatusError("Service Unavailable", request=req, response=resp)

    async def fallback_litellm(**kwargs):
        litellm_calls.append(kwargs)
        return {
            "id": "fallback-resp",
            "model": "local-general",
            "choices": [{"message": {"role": "assistant", "content": "fallback response"}}],
            "usage": {"total_tokens": 10},
        }

    monkeypatch.setattr(main, "send_to_semantic_router", failing_sr)
    monkeypatch.setattr(main, "send_to_litellm", fallback_litellm)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto-balanced",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code == 200
    assert len(sr_calls) == 1
    assert len(litellm_calls) == 1
    assert litellm_calls[0]["model"] == "local-general"
    assert response.json()["model"] == "local-general"


def test_semantic_router_permanent_error_does_not_fall_back(client, monkeypatch):
    sr_calls = []
    litellm_calls = []

    async def failing_sr(**kwargs):
        sr_calls.append(kwargs)
        # 400 Bad Request is permanent; should not trigger fallback
        req = httpx.Request("POST", "http://127.0.0.1:8899/v1/chat/completions")
        resp = httpx.Response(400, request=req)
        raise httpx.HTTPStatusError("Bad Request", request=req, response=resp)

    async def fallback_litellm(**kwargs):
        litellm_calls.append(kwargs)
        return {}

    monkeypatch.setattr(main, "send_to_semantic_router", failing_sr)
    monkeypatch.setattr(main, "send_to_litellm", fallback_litellm)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto-cost",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code == 502
    assert len(sr_calls) == 1
    assert litellm_calls == []  # No fallback attempted


def test_unknown_model_returns_400_before_any_upstream_calls(client, monkeypatch):
    sr_calls = []
    litellm_calls = []

    async def fake_sr(**kwargs):
        sr_calls.append(kwargs)
        return {}

    async def fake_litellm(**kwargs):
        litellm_calls.append(kwargs)
        return {}

    monkeypatch.setattr(main, "send_to_semantic_router", fake_sr)
    monkeypatch.setattr(main, "send_to_litellm", fake_litellm)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-nonexistent-model",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code == 400
    assert "Unknown model alias" in response.json()["detail"]
    assert sr_calls == []
    assert litellm_calls == []
