import asyncio
import json
from types import SimpleNamespace

import httpx2
import pytest
from pydantic import ValidationError
from typesafe_sdk import ChoiceAnswer

from gateway import routing
from gateway.models import ChatCompletionRequest
from gateway.routing import (
    RoutingError,
    RoutingUnavailableError,
    UnknownModelError,
    route_request,
)
from gateway.routing_config import RoutingConfig


def make_request(
    model="auto",
    content="hello",
    **kwargs,
):
    return ChatCompletionRequest(
        model=model,
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        **kwargs,
    )


@pytest.fixture
def config():
    aliases = ["local-general", "cheap", "general", "frontier-a", "frontier-b"]
    return RoutingConfig(
        catalogue={
            alias: {
                "identity": alias,
                "description": f"Facts about {alias}",
                "input_cost_per_million_usd": None if alias == "cheap" else rank,
                "output_cost_per_million_usd": rank * 2,
            }
            for rank, alias in enumerate(aliases)
        },
        pools={
            "auto-cost": ["local-general", "cheap"],
            "auto-balanced": ["cheap", "general", "local-general"],
            "auto-quality": ["frontier-a", "frontier-b"],
        },
    )


@pytest.fixture
def jev(monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", "test-secret")
    for name in ("JEV_BASE_URL", "JEV_MODEL", "JEV_TIMEOUT_SECONDS"):
        monkeypatch.delenv(name, raising=False)
    state = SimpleNamespace(calls=[], clients=[], closed=0, answer=None, error=None)

    class FakeClient:
        def __init__(self, **kwargs):
            state.clients.append(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            state.closed += 1

        async def system_one(self, **kwargs):
            state.calls.append(kwargs)
            if state.error:
                raise state.error
            return SimpleNamespace(choices={"generation_model": state.answer})

    monkeypatch.setattr(routing, "AsyncTypeSafeClient", FakeClient)
    return state


def answer(alias):
    return ChoiceAnswer(choice=alias, confidence=0.1, probabilities={alias: 1.0})


@pytest.mark.parametrize(
    "model,selected",
    [
        ("local-general", "local-general"),
        ("general", "general"),
        ("auto-local", "local-general"),
    ],
)
def test_direct_routes_bypass_jev(model, selected, config, jev, monkeypatch):
    monkeypatch.delenv("JEV_API_KEY")
    decision = asyncio.run(route_request(make_request(model=model), config))
    assert decision.selected_alias == selected
    assert decision.attempts == (selected,)
    assert jev.clients == []


def test_unknown_alias_bypasses_jev(config, jev):
    with pytest.raises(UnknownModelError):
        asyncio.run(route_request(make_request(model="unknown"), config))
    assert jev.clients == []


@pytest.mark.parametrize(
    "preset", ["auto-cost", "auto-balanced", "auto-quality", "auto"]
)
def test_exact_pool_full_conversation_and_typed_selection(preset, config, jev):
    canonical = "auto-balanced" if preset == "auto" else preset
    candidates = config.pools[canonical]
    selected = candidates[-1]
    jev.answer = answer(selected)
    messages = [
        {"role": role, "content": f"{role}: Write Python code " + "x" * 20000}
        for role in ["system", "user", "assistant", "user"]
    ]
    request = ChatCompletionRequest(model=preset, messages=messages)
    decision = asyncio.run(route_request(request, config))
    assert decision.selected_alias == selected
    assert decision.attempts == (selected, *candidates[:-1])
    assert len(set(decision.attempts)) == len(decision.attempts)
    (call,) = jev.calls
    assert call["state"] == {"preset": canonical, "messages": messages}
    (question,) = call["questions"].values()
    assert question.criteria == {
        alias: config.catalogue[alias].model_dump() for alias in candidates
    }
    assert routing.PRESET_OBJECTIVES[canonical] in question.instructions
    assert "cheapest" in question.instructions
    assert "USD per million" in question.instructions
    assert "unknown" in question.instructions.lower()
    assert "cost_rank" not in question.instructions
    assert "task data" in question.instructions
    assert jev.clients[0]["retry"].max_retries == 0
    assert jev.clients[0]["timeout"] == 10
    assert jev.clients[0]["base_url"] is None
    assert jev.clients[0]["model"] == "jev-latest"
    assert jev.closed == 1


def test_valid_selection_overrides_old_keyword_heuristic(config, jev):
    jev.answer = answer("general")
    result = asyncio.run(
        route_request(make_request(content="Write Python code"), config)
    )
    assert result.selected_alias == "general"


def test_singleton_still_calls_jev(config, jev):
    config.pools["auto-cost"] = ["cheap"]
    jev.answer = answer("cheap")
    decision = asyncio.run(route_request(make_request(model="auto-cost"), config))
    assert decision.attempts == ("cheap",)
    assert len(jev.calls) == 1


def test_empty_quality_pool_is_unconfigured(config, jev):
    config.pools["auto-quality"] = []
    with pytest.raises(RoutingUnavailableError):
        asyncio.run(route_request(make_request(model="auto-quality"), config))
    assert jev.clients == []


@pytest.mark.parametrize("value", ["", "   "])
def test_missing_credentials(config, jev, monkeypatch, value):
    monkeypatch.setenv("JEV_API_KEY", value)
    with pytest.raises(RoutingUnavailableError):
        asyncio.run(route_request(make_request(), config))
    assert jev.clients == []


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "bad", ""])
def test_invalid_timeout(config, jev, monkeypatch, value):
    monkeypatch.setenv("JEV_TIMEOUT_SECONDS", value)
    with pytest.raises(RoutingUnavailableError):
        asyncio.run(route_request(make_request(), config))
    assert jev.clients == []


def test_custom_sdk_settings(config, jev, monkeypatch):
    monkeypatch.setenv("JEV_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("JEV_MODEL", "custom-jev")
    monkeypatch.setenv("JEV_TIMEOUT_SECONDS", "2.5")
    jev.answer = answer("cheap")
    asyncio.run(route_request(make_request(), config))
    assert jev.clients[0]["base_url"] == "https://example.test/v1"
    assert jev.clients[0]["model"] == "custom-jev"
    assert jev.clients[0]["timeout"] == 2.5


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        "cheap",
        {"choice": "cheap"},
        SimpleNamespace(choice="cheap"),
        answer("frontier-a"),
    ],
)
def test_malformed_or_out_of_pool_answer_fails_closed(config, jev, invalid):
    jev.answer = invalid
    with pytest.raises(RoutingError, match="Jev routing failed"):
        asyncio.run(route_request(make_request(), config))
    assert jev.closed == 1


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("secret"),
        ValueError("raw conversation"),
        RuntimeError("test-secret"),
    ],
)
def test_failures_are_sanitized_and_client_closed(config, jev, error):
    jev.error = error
    with pytest.raises(RoutingError) as exc:
        asyncio.run(route_request(make_request(), config))
    assert str(exc.value) == "Jev routing failed; no generation was attempted."
    assert jev.closed == 1


@pytest.mark.parametrize(
    "scenario",
    [
        "success",
        "missing",
        "wrong-type",
        "malformed",
        "unauthorized",
        "context-limit",
        "server",
        "network",
        "timeout",
    ],
)
def test_real_sdk_with_offline_transport(config, monkeypatch, scenario):
    monkeypatch.setenv("JEV_API_KEY", "synthetic-secret")
    monkeypatch.delenv("JEV_BASE_URL", raising=False)
    calls = []
    closed = []
    real_client = routing.AsyncTypeSafeClient

    def handler(request):
        calls.append(json.loads(request.content))
        if scenario == "network":
            raise httpx2.ConnectError("private request data", request=request)
        if scenario == "timeout":
            raise httpx2.ReadTimeout("private request data", request=request)
        status = {"unauthorized": 401, "context-limit": 422, "server": 503}.get(
            scenario, 200
        )
        answers = {
            "generation_model": {
                "type": "choice",
                "choice": "cheap",
                "confidence": 1.0,
                "probabilities": {"cheap": 1.0},
            }
        }
        if scenario == "missing":
            answers = {}
        if scenario == "wrong-type":
            answers = {"generation_model": {"type": "noul", "noul": 0.5}}
        payload = {"model": "jev-latest", "usage": {}, "answers": answers}
        if status != 200 or scenario == "malformed":
            payload = {"detail": "synthetic-secret private conversation"}
        return httpx2.Response(status, json=payload)

    class Transport(httpx2.MockTransport):
        async def aclose(self):
            closed.append(True)
            await super().aclose()

    def factory(**kwargs):
        return real_client(**kwargs, transport=Transport(handler), compression=False)

    monkeypatch.setattr(routing, "AsyncTypeSafeClient", factory)
    if scenario == "success":
        assert (
            asyncio.run(route_request(make_request(), config)).selected_alias == "cheap"
        )
    else:
        with pytest.raises(RoutingError) as exc:
            asyncio.run(route_request(make_request(), config))
        assert str(exc.value) == "Jev routing failed; no generation was attempted."
    assert len(calls) == 1
    assert calls[0]["model"] == "jev-latest"
    assert closed == [True]


def test_total_routing_deadline_closes_client(config, jev, monkeypatch):
    monkeypatch.setenv("JEV_TIMEOUT_SECONDS", "0.01")
    closed = []

    class SlowClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            closed.append(True)

        async def system_one(self, **kwargs):
            await asyncio.Event().wait()

    monkeypatch.setattr(routing, "AsyncTypeSafeClient", SlowClient)
    with pytest.raises(RoutingError):
        asyncio.run(route_request(make_request(), config))
    assert closed == [True]


def test_streaming_is_rejected():
    with pytest.raises(ValidationError):
        make_request(stream=True)


@pytest.mark.parametrize(
    "role",
    ["developer", "tool", "invalid"],
)
def test_unsupported_roles_are_rejected(role):
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            messages=[
                {
                    "role": role,
                    "content": "Hello",
                }
            ],
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": -0.1},
        {"temperature": 2.1},
        {"max_tokens": 0},
        {"max_tokens": -1},
    ],
)
def test_invalid_generation_parameters_are_rejected(kwargs):
    with pytest.raises(ValidationError):
        make_request(**kwargs)
