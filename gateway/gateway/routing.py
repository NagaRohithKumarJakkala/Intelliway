
from .models import ChatCompletionRequest


# These aliases must match the model names configured in LiteLLM.
EXPLICIT_MODELS = {
    "local-general",
    "external-general",
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


def route_request(request: ChatCompletionRequest) -> str:
    """Honor explicit model requests; route automatically only for 'auto'."""

    requested_model = request.model

    # An explicit model choice takes precedence over keyword routing.
    if requested_model in EXPLICIT_MODELS:
        return requested_model

    # Unknown aliases must not silently route to another model.
    if requested_model != "auto":
        raise ValueError(
            f"Unknown model alias: {requested_model}"
        )

    # Automatic routing is used only when model='auto'.
    text = " ".join(
        message.content.lower()
        for message in request.messages
        if message.role == "user"
    )

    if any(keyword in text for keyword in CODING_KEYWORDS):
        return "local-general"

    return "external-general"


def fallback_model(selected_model: str) -> str | None:
    """Return the permitted fallback for automatic routing."""

    if selected_model == "external-general":
        return "local-general"

    return None
