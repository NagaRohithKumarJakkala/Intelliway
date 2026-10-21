
import pytest
from pydantic import ValidationError

from gateway.models import ChatCompletionRequest
from gateway.routing import (
    AUTO_ROUTING_PRESETS,
    fallback_model,
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
    request = make_request(model="local-general")

    assert route_request(request) == "local-general"


def test_explicit_external_model_is_honored_for_code():
    request = make_request(
        model="external-general",
        content="Write Python code",
    )

    assert route_request(request) == "external-general"


def test_auto_routes_coding_request_to_local():
    request = make_request(
        model="auto",
        content="Write Python code",
    )

    assert route_request(request) == "local-general"


def test_auto_routes_general_request_to_external():
    request = make_request(
        model="auto",
        content="Tell me a joke",
    )

    assert route_request(request) == "external-general"


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

    assert route_request(coding_request) == "local-general"
    assert route_request(general_request) == "external-general"


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

    assert route_request(request) == "local-general"


def test_auto_local_has_no_external_fallback():
    assert fallback_model("local-general", "auto-local") is None
    assert fallback_model("external-general", "auto-local") is None


@pytest.mark.parametrize(
    "preset",
    ["auto", "auto-cost", "auto-quality", "auto-balanced"],
)
def test_automatic_presets_allow_external_to_local_fallback(preset):
    assert fallback_model("external-general", preset) == "local-general"


def test_unknown_model_alias_is_rejected():
    request = make_request(model="unknown-model")

    with pytest.raises(ValueError):
        route_request(request)


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
