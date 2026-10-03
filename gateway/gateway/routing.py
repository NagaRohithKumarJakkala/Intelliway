from .models import ChatCompletionRequest


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
    text = " ".join(
        message.content.lower()
        for message in request.messages
        if message.role == "user"
    )

    if any(keyword in text for keyword in CODING_KEYWORDS):
        return "local-general"

    return "external-general"

def fallback_model(selected_model: str) -> str | None:
    if selected_model == "external-general":
        return "local-general"

    return None
