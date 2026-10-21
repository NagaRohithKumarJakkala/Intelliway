from semantic_router import IndexedRouter, Route
from semantic_router.encoders import FastEmbedEncoder

from .models import ChatCompletionRequest

# These aliases must match the model names configured in LiteLLM.
EXPLICIT_MODELS = {
    "local-general",
    "external-general",
}

AUTO_ROUTING_PRESETS = {
    "auto-cost",
    "auto-quality",
    "auto-balanced",
    "auto-local",
}

AUTO_MODEL_ALIASES = {
    "auto": "auto-balanced",
    **{preset: preset for preset in AUTO_ROUTING_PRESETS},
}

# ── Semantic Router setup ──────────────────────────────────────────

_CODING_ROUTE = Route(
    name="coding",
    utterances=[
        "Write a Python function that sorts a list",
        "Debug this JavaScript code",
        "Explain the algorithm for binary search",
        "Fix the compile error in my C++ program",
        "Refactor this TypeScript class",
        "Write unit tests for this Rust module",
        "How do I use async/await in Python?",
        "Generate a SQL query to join these tables",
        "What is the time complexity of quicksort?",
        "Help me write a bash script to parse logs",
        "Convert this Java code to Kotlin",
        "Implement a linked list in Go",
    ],
)

_GENERAL_ROUTE = Route(
    name="general",
    utterances=[
        "Tell me a joke",
        "What is the capital of France?",
        "Summarize this article for me",
        "Write a poem about the ocean",
        "What are the benefits of meditation?",
        "Explain quantum computing in simple terms",
        "Help me draft an email to my boss",
        "What happened in the news today?",
        "Recommend a good book to read",
        "Translate this sentence to Spanish",
        "What is the meaning of life?",
        "Plan a 7-day trip to Japan",
    ],
)

_encoder = FastEmbedEncoder()
_semantic_router = IndexedRouter(
    encoder=_encoder,
    routes=[_CODING_ROUTE, _GENERAL_ROUTE],
)

# Map semantic route names -> LiteLLM model aliases.
_ROUTE_TO_MODEL = {
    "coding": "local-general",
    "general": "external-general",
}

_DEFAULT_MODEL = "external-general"


def canonical_auto_preset(requested_model: str) -> str | None:
    """Return the canonical automatic preset, or None for explicit models."""
    return AUTO_MODEL_ALIASES.get(requested_model)


def route_request(request: ChatCompletionRequest) -> str:
    """Honor explicit models and apply semantic routing for auto presets."""
    requested_model = request.model

    # An explicit model choice takes precedence over semantic routing.
    if requested_model in EXPLICIT_MODELS:
        return requested_model

    preset = canonical_auto_preset(requested_model)

    # Unknown aliases must not silently route to another model.
    if preset is None:
        raise ValueError(f"Unknown model alias: {requested_model}")

    if preset == "auto-local":
        return "local-general"

    # Concatenate all user messages for semantic classification.
    text = " ".join(
        message.content
        for message in request.messages
        if message.role == "user"
    )

    result = _semantic_router(text)

    if result.name is not None:
        return _ROUTE_TO_MODEL.get(result.name, _DEFAULT_MODEL)

    return _DEFAULT_MODEL


def fallback_model(
    selected_model: str,
    requested_model: str = "auto",
) -> str | None:
    """Return the permitted fallback for the requested automatic preset."""
    if (
        canonical_auto_preset(requested_model) != "auto-local"
        and selected_model == "external-general"
    ):
        return "local-general"

    return None
