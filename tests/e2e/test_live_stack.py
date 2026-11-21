from __future__ import annotations

import os
import time
import uuid

import httpx
import pytest

from gateway.rate_limit import RateLimiter


GATEWAY = os.getenv("E2E_GATEWAY_URL", "http://127.0.0.1:28000")
HTTPS_API = os.getenv("E2E_HTTPS_API_URL", "https://api.localhost:28443")
HEADERS = {"Authorization": "Bearer e2e-gateway-key"}


@pytest.fixture(scope="module")
def client():
    with httpx.Client(timeout=30, verify=False) as value:
        yield value


def test_tls_front_door_and_chat_completion(client):
    health = client.get(f"{HTTPS_API}/health")
    assert health.status_code == 200

    response = client.post(
        f"{HTTPS_API}/v1/chat/completions",
        headers=HEADERS,
        json={"model": "e2e-mock", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "e2e response"


def test_streaming_and_embeddings(client):
    with client.stream(
        "POST",
        f"{GATEWAY}/v1/chat/completions",
        headers=HEADERS,
        json={"model": "e2e-mock", "stream": True, "messages": [{"role": "user", "content": "stream"}]},
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())
        assert "data: [DONE]" in body

    embedding = client.post(
        f"{GATEWAY}/v1/embeddings",
        headers=HEADERS,
        json={"model": "e2e-mock", "input": "hello"},
    )
    assert embedding.status_code == 200
    assert embedding.json()["data"][0]["embedding"] == [0.1, 0.2]


def test_request_cache_controls_reach_litellm(client):
    unique = uuid.uuid4().hex
    payload = {
        "model": "e2e-mock",
        "messages": [{"role": "user", "content": unique}],
        "cache": {"ttl": 60, "namespace": "gateway-e2e"},
    }
    first = client.post(f"{GATEWAY}/v1/chat/completions", headers=HEADERS, json=payload)
    second = client.post(f"{GATEWAY}/v1/chat/completions", headers=HEADERS, json=payload)
    assert first.status_code == second.status_code == 200


def test_metrics_and_recent_upstream_health(client):
    metrics = client.get(f"{GATEWAY}/metrics")
    assert metrics.status_code == 200
    assert "intelliway_gateway_http_requests_total" in metrics.text
    health = client.get(f"{GATEWAY}/health/upstreams")
    assert health.status_code == 200
    assert "litellm:e2e-mock" in health.json()


def test_redis_rate_limit_is_shared_between_limiter_instances(monkeypatch):
    monkeypatch.setenv("GATEWAY_REDIS_HOST", "127.0.0.1")
    monkeypatch.setenv("GATEWAY_REDIS_PORT", "26379")
    monkeypatch.setenv("GATEWAY_REDIS_PASSWORD", "e2e-redis-password")
    first = RateLimiter()
    second = RateLimiter()

    async def consume():
        identity = f"e2e-{uuid.uuid4().hex}"
        try:
            return await first.check(identity, limit=1), await second.check(identity, limit=1)
        finally:
            await first.close()
            await second.close()

    import asyncio

    first_result, second_result = asyncio.run(consume())
    assert first_result[0] is True
    assert second_result[0] is False


def test_jaeger_receives_gateway_traces(client):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            response = client.get("http://127.0.0.1:26687/api/services")
            if "intelliway-gateway-e2e" in response.json().get("data", []):
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    pytest.fail("Jaeger did not receive gateway traces")


def test_provider_failure_opens_circuit_and_health_score_drops(client):
    control = "http://127.0.0.1:28081"
    assert client.get(f"{control}/__fail").status_code == 204
    try:
        for _ in range(3):
            response = client.post(
                f"{GATEWAY}/v1/chat/completions",
                headers=HEADERS,
                json={"model": "e2e-mock", "messages": [{"role": "user", "content": "fail"}]},
            )
            assert response.status_code == 502
        status = client.get(f"{GATEWAY}/health/upstreams").json()["litellm:e2e-mock"]
        assert status["state"] == "open"
        assert status["health_score"] < 100
    finally:
        client.get(f"{control}/__recover")
