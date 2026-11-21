from pathlib import Path

from ruamel.yaml import YAML

from .models import ChatCompletionRequest


LITELLM_CONFIG_PATH = Path(__file__).resolve().parent.parent / "litellm" / "config.yaml"


def _configured_explicit_models() -> set[str]:
    try:
        yaml = YAML(typ="safe")
        with LITELLM_CONFIG_PATH.open(encoding="utf-8") as stream:
            config = yaml.load(stream)
        return {
            item["model_name"]
            for item in (config or {}).get("model_list", [])
            if isinstance(item, dict) and item.get("model_name")
        }
    except (OSError, KeyError, AttributeError, TypeError):
        return set()


EXPLICIT_MODELS = _configured_explicit_models()


def refresh_explicit_models() -> set[str]:
    """Reload aliases after the LiteLLM YAML is edited or managed by the CLI."""
    configured = _configured_explicit_models()
    EXPLICIT_MODELS.clear()
    EXPLICIT_MODELS.update(configured)
    return set(configured)


def get_explicit_models() -> set[str]:
    return refresh_explicit_models()


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
    return model in get_explicit_models()


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
    if requested_model in get_explicit_models():
        return requested_model

    preset = canonical_auto_preset(requested_model)

    # Unknown aliases must not silently route to another model.
    if preset is None:
        raise ValueError(f"Unknown model alias: {requested_model}")

    if preset == "auto-local":
        return "gemma4"

    # Evaluation aligns with the vLLM semantic router rules
    text = " ".join(
        message.content.lower()
        for message in request.messages
        if message.role == "user"
    )

    if any(keyword in text for keyword in CODING_KEYWORDS):
        return "gemma4"

    return "gemini-3.5-flash-lite"


def fallback_model(
    selected_model: str,
    requested_model: str = "auto",
) -> str | None:
    """Return the permitted fallback for the requested automatic preset."""
    if (
        canonical_auto_preset(requested_model) != "auto-local"
        and selected_model not in {"gemma4", "ornith"}
    ):
        return "gemma4"

    return None
