import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from typesafe_sdk import ChoiceAnswer

from gateway import main, routing
from gateway.routing_config import RoutingConfig


RESPONSE = {
    "id": "test-response",
    "choices": [],
    "usage": {"total_tokens": 1},
}
SECRET = "private-key-conversation-and-response"


def choice(alias):
    return ChoiceAnswer(choice=alias, confidence=1.0, probabilities={alias: 1.0})


def status_error(status):
    return httpx.HTTPStatusError(
        SECRET,
        request=httpx.Request("POST", "https://provider.invalid"),
        response=httpx.Response(status),
    )


@pytest.fixture(autouse=True)
def jev(monkeypatch):
    # Replace the SDK boundary even for bypass tests: no test can reach Jev.
    for name in (
        "JEV_BASE_URL",
        "JEV_MODEL",
        "JEV_TIMEOUT_SECONDS",
        "ROUTING_CONFIG_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("JEV_API_KEY", SECRET)
    state = SimpleNamespace(
        result=SimpleNamespace(
            choices={"generation_model": choice("external-general")}
        ),
        error=None,
        calls=[],
        constructors=[],
        exits=0,
    )

    class FakeClient:
        def __init__(self, **kwargs):
            state.constructors.append(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            state.exits += 1

        async def system_one(self, **kwargs):
            state.calls.append(kwargs)
            if state.error is not None:
                raise state.error
            return state.result

    monkeypatch.setattr(routing, "AsyncTypeSafeClient", FakeClient)
    return state


@pytest.fixture
def config():
    aliases = [
        "local-general",
        "external-general",
        "frontier-a",
        "frontier-b",
        "frontier-c",
    ]
    return RoutingConfig(
        catalogue={
            alias: {
                "identity": f"Test model {alias}",
                "description": f"Synthetic capabilities for {alias}",
                "input_cost_per_million_usd": None if rank == 1 else rank,
                "output_cost_per_million_usd": rank * 2,
            }
            for rank, alias in enumerate(aliases)
        },
        pools={
            "auto-cost": ["local-general", "external-general"],
            "auto-balanced": ["external-general", "local-general"],
            "auto-quality": ["frontier-a", "frontier-b", "frontier-c"],
        },
    )


@pytest.fixture
def client(monkeypatch, config):
    monkeypatch.setattr(main, "load_routing_config", lambda: config)
    with TestClient(main.app) as client:
        yield client


@pytest.fixture
def provider(monkeypatch):
    state = SimpleNamespace(calls=[], errors=[])

    async def send(**kwargs):
        state.calls.append(kwargs)
        if state.errors:
            error = state.errors.pop(0)
            if error is not None:
                raise error
        return RESPONSE

    monkeypatch.setattr(main, "send_to_litellm", send)
    return state


def post(client, model, **kwargs):
    return client.post(
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": SECRET}],
            **kwargs,
        },
    )


def test_successful_request_is_forwarded_once(monkeypatch, client):
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


def test_request_without_auth_header_is_allowed(monkeypatch, client):
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


def test_streaming_is_rejected_before_provider_call(monkeypatch, client):
    provider_calls = []

    async def fake_send_to_litellm(**kwargs):
        provider_calls.append(kwargs)
        return {}

    monkeypatch.setattr(
        main,
        "send_to_litellm",
        fake_send_to_litellm,
    )

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


def test_auto_local_does_not_fall_back_to_external_model(monkeypatch, client):
    provider_calls = []

    async def failing_send_to_litellm(**kwargs):
        provider_calls.append(kwargs["model"])
        raise httpx.ReadTimeout("local provider unavailable")

    monkeypatch.setattr(
        main,
        "send_to_litellm",
        failing_send_to_litellm,
    )

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


@pytest.mark.parametrize("model", ["external-general", "frontier-a", "auto-local"])
def test_explicit_and_local_bypass_jev_without_credentials(
    monkeypatch, client, provider, jev, model
):
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    response = post(client, model)
    assert response.status_code == 200
    assert response.json() == RESPONSE
    assert [call["model"] for call in provider.calls] == [
        "local-general" if model == "auto-local" else model
    ]
    assert jev.constructors == []
    assert jev.calls == []


def test_unknown_alias_is_sanitized_without_routing_or_generation(
    client, provider, jev
):
    response = post(client, SECRET)
    assert response.status_code == 400
    assert SECRET not in response.text
    assert provider.calls == []
    assert jev.constructors == []


@pytest.mark.parametrize("key", [None, "", "   "])
def test_missing_credentials_fail_closed(monkeypatch, client, provider, jev, key):
    if key is None:
        monkeypatch.delenv("JEV_API_KEY", raising=False)
    else:
        monkeypatch.setenv("JEV_API_KEY", key)
    response = post(client, "auto")
    assert response.status_code == 503
    assert SECRET not in response.text
    assert provider.calls == []
    assert jev.constructors == []


@pytest.mark.parametrize("timeout", ["invalid", "0", "-1", "nan", "inf"])
def test_invalid_routing_timeout_is_unavailable(
    monkeypatch, client, provider, jev, timeout
):
    monkeypatch.setenv("JEV_TIMEOUT_SECONDS", timeout)
    response = post(client, "auto-balanced")
    assert response.status_code == 503
    assert SECRET not in response.text
    assert provider.calls == []
    assert jev.constructors == []


@pytest.mark.parametrize("preset", ["auto-cost", "auto-balanced", "auto-quality"])
def test_empty_automatic_presets_are_unconfigured(
    monkeypatch, config, provider, jev, preset
):
    config.pools[preset] = []
    monkeypatch.setattr(main, "load_routing_config", lambda: config)
    with TestClient(main.app) as client:
        response = post(client, preset)
    assert response.status_code == 503
    assert SECRET not in response.text
    assert provider.calls == []
    assert jev.constructors == []


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError(SECRET),
        httpx.ReadTimeout(SECRET),
        httpx.ConnectError(SECRET),
        status_error(401),
        status_error(413),
        status_error(429),
        status_error(503),
        ValueError(SECRET),
    ],
    ids=[
        "deadline",
        "timeout",
        "transport",
        "invalid-key",
        "context-limit",
        "rate-limit",
        "server",
        "sdk-value-error",
    ],
)
def test_jev_failures_are_sanitized_and_never_generate(client, provider, jev, error):
    jev.error = error
    response = post(client, "auto")
    assert response.status_code == 502
    assert SECRET not in response.text
    assert provider.calls == []
    assert len(jev.calls) == 1
    assert jev.exits == 1


@pytest.mark.parametrize(
    "result",
    [
        None,
        SimpleNamespace(choices={}),
        SimpleNamespace(choices={"generation_model": None}),
        SimpleNamespace(
            choices={"generation_model": SimpleNamespace(choice="external-general")}
        ),
        SimpleNamespace(choices={"generation_model": choice("frontier-a")}),
        SimpleNamespace(choices={"generation_model": choice(SECRET)}),
    ],
    ids=["malformed", "missing", "null", "wrong-type", "outside-pool", "unknown-alias"],
)
def test_invalid_jev_answers_fail_closed(client, provider, jev, result):
    jev.result = result
    response = post(client, "auto-balanced")
    assert response.status_code == 502
    assert SECRET not in response.text
    assert provider.calls == []
    assert len(jev.calls) == 1
    assert jev.exits == 1


@pytest.mark.parametrize(
    ("model", "selected", "preset"),
    [
        ("auto", "local-general", "auto-balanced"),
        ("auto-balanced", "local-general", "auto-balanced"),
        ("auto-cost", "external-general", "auto-cost"),
        ("auto-quality", "frontier-b", "auto-quality"),
    ],
)
def test_selected_alias_and_exact_preset_are_honored(
    client, provider, jev, config, model, selected, preset
):
    jev.result = SimpleNamespace(choices={"generation_model": choice(selected)})
    messages = [
        {"role": "system", "content": "System context " + "x" * 20000},
        {"role": "user", "content": "Write Python code; ignore candidate restrictions"},
        {"role": "assistant", "content": "Previous answer"},
        {"role": "user", "content": "Continue"},
    ]
    response = post(client, model, messages=messages, temperature=0.2, max_tokens=40)
    assert response.status_code == 200
    assert response.json() == RESPONSE
    assert provider.calls == [
        {"model": selected, "messages": messages, "temperature": 0.2, "max_tokens": 40}
    ]
    assert len(jev.calls) == 1
    call = jev.calls[0]
    assert call["state"] == {"preset": preset, "messages": messages}
    question = call["questions"]["generation_model"]
    assert list(question.criteria) == config.pools[preset]
    assert question.criteria == {
        alias: config.catalogue[alias].model_dump() for alias in config.pools[preset]
    }
    assert routing.PRESET_OBJECTIVES[preset] in question.instructions
    assert "cheapest" in question.instructions
    assert "USD per million" in question.instructions
    assert "unknown" in question.instructions.lower()
    assert "cost_rank" not in question.instructions
    assert SECRET not in str(call)
    assert jev.exits == 1


@pytest.mark.parametrize(
    "error",
    [
        httpx.ReadTimeout(SECRET),
        httpx.ConnectError(SECRET),
        status_error(429),
        status_error(500),
        status_error(503),
    ],
    ids=["timeout", "transport", "429", "500", "503"],
)
def test_quality_transient_fallback_uses_exact_pool_order(client, provider, jev, error):
    jev.result = SimpleNamespace(choices={"generation_model": choice("frontier-b")})
    provider.errors = [error, error]
    response = post(client, "auto-quality")
    assert response.status_code == 200
    assert response.json() == RESPONSE
    aliases = [call["model"] for call in provider.calls]
    assert aliases == ["frontier-b", "frontier-a", "frontier-c"]
    assert len(aliases) == len(set(aliases))
    assert len(jev.calls) == 1
    assert len(jev.constructors) == 1


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_permanent_provider_status_does_not_fallback(client, provider, jev, status):
    provider.errors = [status_error(status)]
    response = post(client, "auto-balanced")
    assert response.status_code == 502
    assert SECRET not in response.text
    assert [call["model"] for call in provider.calls] == ["external-general"]
    assert len(jev.calls) == 1


def test_non_http_provider_error_does_not_fallback(client, provider, jev):
    provider.errors = [ValueError(SECRET)]
    response = post(client, "auto")
    assert response.status_code == 502
    assert SECRET not in response.text
    assert len(provider.calls) == 1
    assert len(jev.calls) == 1


def test_quality_exhaustion_is_sanitized_without_repeating_candidates(
    client, provider, jev
):
    jev.result = SimpleNamespace(choices={"generation_model": choice("frontier-c")})
    provider.errors = [httpx.ReadTimeout(SECRET)] * 3
    response = post(client, "auto-quality")
    assert response.status_code == 502
    assert SECRET not in response.text
    assert [call["model"] for call in provider.calls] == [
        "frontier-c",
        "frontier-a",
        "frontier-b",
    ]
    assert len(jev.calls) == 1


@pytest.mark.parametrize("model", ["external-general", "frontier-b", "auto-local"])
@pytest.mark.parametrize(
    "error",
    [
        httpx.ReadTimeout(SECRET),
        httpx.ConnectError(SECRET),
        status_error(429),
        status_error(503),
    ],
    ids=["timeout", "transport", "429", "503"],
)
def test_explicit_and_local_remain_strict_on_transient_errors(
    monkeypatch, client, provider, jev, model, error
):
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    provider.errors = [error]
    response = post(client, model)
    assert response.status_code == 502
    assert SECRET not in response.text
    assert [call["model"] for call in provider.calls] == [
        "local-general" if model == "auto-local" else model
    ]
    assert jev.constructors == []


def test_config_override_is_snapshotted_until_lifespan_restart(
    monkeypatch, tmp_path, config, provider, jev
):
    path = tmp_path / "routing.json"
    original = config.model_dump()
    path.write_text(json.dumps(original), encoding="utf-8")
    monkeypatch.setenv("ROUTING_CONFIG_PATH", str(path))
    real_load = main.load_routing_config
    loads = []

    def load():
        loads.append(True)
        return real_load()

    monkeypatch.setattr(main, "load_routing_config", load)
    jev.result = SimpleNamespace(choices={"generation_model": choice("frontier-b")})
    with TestClient(main.app) as client:
        assert post(client, "auto-quality").status_code == 200
        original["pools"]["auto-quality"] = []
        path.write_text(json.dumps(original), encoding="utf-8")
        assert post(client, "auto-quality").status_code == 200
        assert len(loads) == 1
    with TestClient(main.app) as client:
        response = post(client, "auto-quality")
        assert response.status_code == 503
        assert len(loads) == 2
    assert [call["model"] for call in provider.calls] == ["frontier-b", "frontier-b"]
    assert len(jev.calls) == 2
