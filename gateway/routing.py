from .models import ChatCompletionRequest

# These aliases must match the model names configured in LiteLLM.
EXPLICIT_MODELS = {
    "local-general",
    "local-coder",
    "local-reasoner",
    "local-fast",
    "external-general",
    "external-frontier",
    "external-coder",
    "external-fast",
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

CODING_KEYWORDS = {
    "code",
    "python",
    "rust",
    "c++",
    "javascript",
    "typescript",
    "program",
    "debug",
    "function",
    "class",
    "compile",
    "algorithm",
}


def is_explicit_model(model: str) -> bool:
    """Return True if the model is one of the explicit targets."""
    return model in EXPLICIT_MODELS


def canonical_auto_preset(requested_model: str) -> str | None:
    """Return the canonical automatic preset, or None for explicit models."""
    return AUTO_MODEL_ALIASES.get(requested_model)


def route_request(request: ChatCompletionRequest) -> str:
    """
    Honor explicit models and apply the baseline automatic policies.
    Used for local testing, fallback, and simulation when vLLM Semantic Router
    is not running.
    """
    requested_model = request.model

    # An explicit model choice takes precedence.
    if requested_model in EXPLICIT_MODELS:
        return requested_model

    preset = canonical_auto_preset(requested_model)

    # Unknown aliases must not silently route to another model.
    if preset is None:
        raise ValueError(f"Unknown model alias: {requested_model}")

    if preset == "auto-local":
        return "local-general"

    # Evaluation aligns with the vLLM semantic router rules
    text = " ".join(
        message.content.lower()
        for message in request.messages
        if message.role == "user"
    )

    if any(keyword in text for keyword in CODING_KEYWORDS):
        return "local-general"

    return "external-general"


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
