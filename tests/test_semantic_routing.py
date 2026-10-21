import pytest

from gateway.models import ChatCompletionRequest
from gateway.routing import (
    _ROUTE_TO_MODEL,
    _semantic_router,
    route_request,
)


def make_request(content, model="auto"):
    return ChatCompletionRequest(
        model=model,
        messages=[{"role": "user", "content": content}],
    )


class TestSemanticClassification:
    """Verify the semantic router classifies prompts correctly."""

    @pytest.mark.parametrize(
        "prompt",
        [
            "Write a Python function to reverse a string",
            "Debug this segfault in my C code",
            "Explain how merge sort works",
            "Help me refactor this React component",
        ],
    )
    def test_coding_prompts_route_to_local(self, prompt):
        assert route_request(make_request(prompt)) == "local-general"

    @pytest.mark.parametrize(
        "prompt",
        [
            "Tell me a fun fact about dolphins",
            "Write a haiku about winter",
            "What is the population of Tokyo?",
            "Recommend a Netflix show",
        ],
    )
    def test_general_prompts_route_to_external(self, prompt):
        assert route_request(make_request(prompt)) == "external-general"


class TestSemanticRouterObject:
    """Verify the underlying router returns expected route names."""

    def test_coding_route_name(self):
        result = _semantic_router("Implement quicksort in Rust")
        assert result.name == "coding"

    def test_general_route_name(self):
        result = _semantic_router("What's the weather like?")
        assert result.name == "general"
