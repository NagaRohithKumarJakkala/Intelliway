
import pytest
from pydantic import ValidationError
from ruamel.yaml import YAML

from gateway.models import ChatCompletionRequest
from gateway.routing import (
    AUTO_ROUTING_PRESETS,
    fallback_model,
    get_explicit_models,
    route_request,
)



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


def test_explicit_local_model_is_honored():
    request = make_request(model="gemma4")

    assert route_request(request) == "gemma4"


@pytest.mark.parametrize(
    "model_name",
    [
        "gemma4",
        "ornith",
        "gemini-3.5-flash-lite",
        "deepseek-v4.1-flash-1017",
        "glm-5.3-flash",
        "gpt-6-luna",
        "claude-haiku-5.5",
        "gemini-3.8-flash",
        "glm-5.3",
        "gpt-6.1-terra",
    ],
)
def test_all_explicit_models_in_pool_are_honored(model_name):
    request = make_request(model=model_name)
    assert route_request(request) == model_name


def test_explicit_external_model_is_honored_for_code():
    request = make_request(
        model="gemini-3.8-flash",
        content="Write Python code",
    )

    assert route_request(request) == "gemini-3.8-flash"


def test_auto_routes_coding_request_to_local():
    request = make_request(
        model="auto",
        content="Write Python code",
    )

    assert route_request(request) == "gemma4"


def test_auto_routes_general_request_to_external():
    request = make_request(
        model="auto",
        content="Tell me a joke",
    )

    assert route_request(request) == "gemini-3.5-flash-lite"


@pytest.mark.parametrize(
    "preset",
    sorted(AUTO_ROUTING_PRESETS - {"auto-local"}),
)
def test_non_local_presets_reuse_mock_heuristic(preset):
    coding_request = make_request(
        model=preset,
        content="Write Python code",
    )
    general_request = make_request(
        model=preset,
        content="Tell me a joke",
    )

    assert route_request(coding_request) == "gemma4"
    assert route_request(general_request) == "gemini-3.5-flash-lite"


def test_auto_is_an_alias_for_auto_balanced():
    request = make_request(
        model="auto-balanced",
        content="Tell me a joke",
    )

    assert route_request(request) == route_request(
        make_request(model="auto", content="Tell me a joke")
    )


def test_auto_local_always_selects_local_model():
    request = make_request(
        model="auto-local",
        content="Tell me a joke",
    )

    assert route_request(request) == "gemma4"


def test_auto_local_has_no_external_fallback():
    assert fallback_model("gemma4", "auto-local") is None
    assert fallback_model("gemini-3.8-flash", "auto-local") is None


@pytest.mark.parametrize(
    "preset",
    ["auto", "auto-cost", "auto-quality", "auto-balanced"],
)
def test_automatic_presets_allow_external_to_local_fallback(preset):
    assert fallback_model("gemini-3.8-flash", preset) == "gemma4"


def test_unknown_model_alias_is_rejected():
    request = make_request(model="unknown-model")

    with pytest.raises(ValueError):
        route_request(request)


def test_explicit_models_reload_from_litellm_config(monkeypatch, tmp_path):
    import gateway.routing as routing

    config_path = tmp_path / "config.yaml"
    yaml = YAML()
    with config_path.open("w", encoding="utf-8") as stream:
        yaml.dump({
            "model_list": [
                {"model_name": "dynamically-added"},
            ]
        }, stream)

    monkeypatch.setattr(routing, "LITELLM_CONFIG_PATH", config_path)

    assert "dynamically-added" in get_explicit_models()


def test_streaming_is_supported_by_request_model():
    request = make_request(stream=True)
    assert request.stream is True


def test_tool_message_role_is_supported():
    request = ChatCompletionRequest(
        messages=[
            {
                "role": "tool",
                "content": "{\"temperature\":21}",
                "tool_call_id": "call-1",
            }
        ]
    )
    assert request.messages[0].role == "tool"


@pytest.mark.parametrize(
    "role",
    ["developer", "invalid"],
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
