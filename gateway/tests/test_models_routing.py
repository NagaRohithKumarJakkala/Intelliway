
import pytest
from pydantic import ValidationError

from gateway.models import ChatCompletionRequest
from gateway.routing import route_request



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
