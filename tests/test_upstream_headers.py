import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from gateway import main, upstream
from gateway.upstream import LiteLLMResponse


def test_upstream_filters_headers_and_preserves_json(monkeypatch):
    async def exercise():
        def handler(request):
            return httpx.Response(
                200,
                json={"ok": True},
                headers={
                    "x-litellm-response-cost": "0.01",
                    "x-litellm-version": "fixture-version",
                    "x-litellm-api-base": "private",
                    "set-cookie": "private=1",
                },
            )

        monkeypatch.setattr(upstream, "LITELLM_URL", "https://litellm.test")
        monkeypatch.setenv("LITELLM_MASTER_KEY", "master-secret")
        # Replace the client transport at the actual upstream HTTP boundary.
        real_client = httpx.AsyncClient

        def factory(*args, **kwargs):
            return real_client(*args, **kwargs, transport=httpx.MockTransport(handler))

        monkeypatch.setattr(upstream.httpx, "AsyncClient", factory)
        result = await upstream.send_to_litellm(
            "alias", [{"role": "user", "content": "hi"}]
        )
        assert result == LiteLLMResponse(
            {"ok": True},
            {"x-litellm-response-cost": "0.01", "x-litellm-version": "fixture-version"},
        )

    asyncio.run(exercise())


def test_gateway_emits_approved_headers_and_unchanged_body(monkeypatch):
    from types import SimpleNamespace

    async def route(*args):
        return SimpleNamespace(attempts=["fixture-model"])

    monkeypatch.setattr(main, "route_request", route)

    async def send(**kwargs):
        return LiteLLMResponse(
            {"id": "fixture", "choices": [], "usage": {}},
            {"x-litellm-response-cost": "0.02"},
        )

    monkeypatch.setattr(main, "send_to_litellm", send)
    with TestClient(main.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "external-general",
                "messages": [{"role": "user", "content": "test"}],
            },
        )
    assert response.json() == {"id": "fixture", "choices": [], "usage": {}}
    assert response.headers["x-litellm-response-cost"] == "0.02"
    assert "x-litellm-api-base" not in response.headers


@pytest.mark.parametrize("fallback", [False, True])
def test_successful_attempt_owns_headers_and_missing_telemetry_is_allowed(
    monkeypatch, fallback
):
    from types import SimpleNamespace

    async def route(*args):
        return SimpleNamespace(attempts=["first", "second"])

    calls = []
    body = {"id": "unchanged", "choices": [], "usage": {}}

    async def send(**kwargs):
        calls.append(kwargs["model"])
        if fallback and kwargs["model"] == "first":
            raise httpx.HTTPStatusError(
                "failed",
                request=httpx.Request("POST", "http://upstream.test"),
                response=httpx.Response(
                    500, headers={"x-litellm-call-id": "failed-attempt"}
                ),
            )
        return LiteLLMResponse(
            body, {"x-litellm-call-id": "successful-attempt"} if fallback else {}
        )

    monkeypatch.setattr(main, "route_request", route)
    monkeypatch.setattr(main, "send_to_litellm", send)
    with TestClient(main.app) as client:
        result = client.post(
            "/v1/chat/completions",
            json={
                "model": "fixture",
                "messages": [{"role": "user", "content": "test"}],
            },
        )
    assert result.status_code == 200 and result.json() == body
    assert calls == (["first", "second"] if fallback else ["first"])
    assert result.headers.get("x-litellm-call-id") == (
        "successful-attempt" if fallback else None
    )


def test_upstream_without_headers_preserves_request_contract(monkeypatch):
    real_client = httpx.AsyncClient

    def handler(request):
        import json

        assert request.extensions["timeout"]["read"] == 30
        assert json.loads(request.content) == {
            "model": "fixture",
            "messages": [{"role": "user", "content": "test"}],
            "stream": False,
            "temperature": 0,
            "max_tokens": 10,
        }
        assert request.headers["authorization"] == "Bearer fixture-master"
        return httpx.Response(
            200,
            json={"id": "same"},
            headers={
                "x-litellm-key-spend": "private",
                "x-litellm-provider": "private",
                "authorization": "private",
            },
        )

    monkeypatch.setenv("LITELLM_MASTER_KEY", "fixture-master")
    monkeypatch.setattr(
        upstream.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(**kwargs, transport=httpx.MockTransport(handler)),
    )
    result = asyncio.run(
        upstream.send_to_litellm(
            "fixture",
            [{"role": "user", "content": "test"}],
            temperature=0,
            max_tokens=10,
        )
    )
    assert result == LiteLLMResponse({"id": "same"}, {})
