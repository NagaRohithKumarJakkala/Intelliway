"""Manage model aliases shared by LiteLLM and vLLM Semantic Router."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML


ROOT = Path(__file__).resolve().parent.parent
LITELLM_CONFIG = ROOT / "litellm" / "config.yaml"
ROUTER_CONFIG = ROOT / "semantic-router" / "config.yaml"
POOL_PATHS = {
    "cost": [("routing", "decisions", "cost_pool"), ("recipes", "cost", "cost_default")],
    "balanced": [("routing", "decisions", "balanced_pool"), ("recipes", "balanced", "balanced_default")],
    "quality": [("routing", "decisions", "quality_pool"), ("recipes", "quality", "quality_default")],
    "local": [("recipes", "local-only", "local_default")],
}


def _yaml() -> YAML:
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    return yaml


def _load(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return _yaml().load(stream)


def _save(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as stream:
        _yaml().dump(data, stream)


def restart_services() -> None:
    subprocess.run(
        [
            "docker",
            "compose",
            "up",
            "-d",
            "--force-recreate",
            "litellm",
            "semantic-router",
            "gateway",
            "chat-app",
        ],
        cwd=ROOT,
        check=True,
    )


def _decision(router: Any, path: tuple[str, ...]) -> Any:
    target = path[-1]

    def find(node: Any) -> Any | None:
        if isinstance(node, dict):
            if node.get("name") == target and "modelRefs" in node:
                return node
            for value in node.values():
                result = find(value)
                if result is not None:
                    return result
        elif isinstance(node, list):
            for value in node:
                result = find(value)
                if result is not None:
                    return result
        return None

    result = find(router)
    if result is None:
        raise KeyError(f"Router decision not found: {target}")
    return result


def _refs(router: Any) -> list[Any]:
    refs = []

    def collect(node: Any) -> None:
        if isinstance(node, dict):
            if isinstance(node.get("modelRefs"), list):
                refs.append(node["modelRefs"])
            for value in node.values():
                collect(value)
        elif isinstance(node, list):
            for value in node:
                collect(value)

    collect(router)
    return refs


def _validate_name(name: str) -> None:
    if not name or any(char.isspace() for char in name):
        raise ValueError("Model alias must be non-empty and contain no whitespace")


def add_model(args: argparse.Namespace) -> None:
    _validate_name(args.name)
    if args.provider == "ollama":
        model_id = args.model or args.name
        if not model_id.startswith("ollama/"):
            model_id = f"ollama/{model_id}"
        params = {"model": model_id, "api_base": args.api_base or "os.environ/OLLAMA_BASE_URL"}
        model_info = {"input_cost_per_token": 0.0, "output_cost_per_token": 0.0}
    else:
        if not args.model:
            raise ValueError("--model is required for non-Ollama providers")
        params = {"model": args.model}
        if args.api_base:
            params["api_base"] = args.api_base
        if args.api_key_env:
            params["api_key"] = f"os.environ/{args.api_key_env}"
        if (args.prompt_price is None) != (args.completion_price is None):
            raise ValueError("Specify both --prompt-price and --completion-price, or neither")
        if args.prompt_price is None:
            raise ValueError("Set --prompt-price and --completion-price for non-Ollama models")
        model_info = None
        model_info = {
            "input_cost_per_token": args.prompt_price / 1_000_000,
            "output_cost_per_token": args.completion_price / 1_000_000,
        }
    params.update({
        "rpm": args.rpm,
        "tpm": args.tpm,
        "max_parallel_requests": args.max_parallel_requests,
    })

    litellm = _load(LITELLM_CONFIG)
    router = _load(ROUTER_CONFIG)
    if any(item["model_name"] == args.name for item in litellm["model_list"]):
        raise ValueError(f"LiteLLM alias already exists: {args.name}")
    if any(item["name"] == args.name for item in router["providers"]["models"]):
        raise ValueError(f"Semantic Router model already exists: {args.name}")

    litellm_model = {"model_name": args.name, "litellm_params": params}
    if model_info is not None:
        litellm_model["model_info"] = model_info
    litellm["model_list"].append(litellm_model)
    router_model = {
        "name": args.name,
        "provider_model_id": args.name,
        "api_format": "openai",
        "pricing": {"currency": "USD", "prompt_per_1m": args.prompt_price or 0.0, "completion_per_1m": args.completion_price or 0.0},
        "backend_refs": [{
            "name": f"litellm-{args.name}", "endpoint": "litellm:4000", "provider": "openai",
            "auth_header": "Authorization", "auth_prefix": "Bearer ", "weight": 100,
        }],
    }
    router["providers"]["models"].append(router_model)
    router["routing"]["modelCards"].append({
        "name": args.name, "description": args.description,
        "context_window_size": args.context_window, "capabilities": ["chat", "tools"],
    })
    if args.pool != "none":
        for path in POOL_PATHS[args.pool]:
            refs = _decision(router, path)["modelRefs"]
            if not any(ref["model"] == args.name for ref in refs):
                refs.append({"model": args.name})

    _save(LITELLM_CONFIG, litellm)
    _save(ROUTER_CONFIG, router)
    print(f"Added {args.name} to LiteLLM and Semantic Router (pool: {args.pool}).")
    if args.restart:
        restart_services()


def remove_model(args: argparse.Namespace) -> None:
    _validate_name(args.name)
    litellm = _load(LITELLM_CONFIG)
    router = _load(ROUTER_CONFIG)
    litellm_matches = [item for item in litellm["model_list"] if item["model_name"] == args.name]
    router_matches = [item for item in router["providers"]["models"] if item["name"] == args.name]
    if not litellm_matches and not router_matches:
        raise ValueError(f"Model alias not found: {args.name}")
    if router["providers"].get("defaults", {}).get("model") == args.name:
        raise ValueError(f"Cannot remove {args.name}: it is the Semantic Router default model")
    litellm["model_list"] = [item for item in litellm["model_list"] if item["model_name"] != args.name]
    router["providers"]["models"] = [item for item in router["providers"]["models"] if item["name"] != args.name]
    router["routing"]["modelCards"] = [item for item in router["routing"]["modelCards"] if item["name"] != args.name]
    for refs in _refs(router):
        refs[:] = [ref for ref in refs if ref.get("model") != args.name]
    _save(LITELLM_CONFIG, litellm)
    _save(ROUTER_CONFIG, router)
    print(f"Removed {args.name} from LiteLLM and Semantic Router references.")
    if args.restart:
        restart_services()


def restart_config(args: argparse.Namespace) -> None:
    del args
    restart_services()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="intelliway-model", description="Manage model aliases in LiteLLM and Semantic Router configs.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add = subparsers.add_parser("add", help="add a model alias")
    add.add_argument("name", help="public model alias")
    add.add_argument("--provider", choices=("ollama", "openai", "openrouter", "other"), default="ollama")
    add.add_argument("--model", help="provider model ID (for example qwen2.5:1.5b or openrouter/vendor/model)")
    add.add_argument("--api-base", help="provider API base URL; Ollama defaults to OLLAMA_BASE_URL")
    add.add_argument("--api-key-env", help="environment variable containing the provider key")
    add.add_argument("--pool", choices=("none", "local", "cost", "balanced", "quality"), default="balanced")
    add.add_argument("--description", default="General-purpose chat model.")
    add.add_argument("--context-window", type=int, default=32768)
    add.add_argument("--prompt-price", type=float, help="input price per 1M tokens in USD")
    add.add_argument("--completion-price", type=float, help="output price per 1M tokens in USD")
    add.add_argument("--rpm", type=int, default=60, help="requests per minute for this model alias (default: 60)")
    add.add_argument("--tpm", type=int, default=100000, help="tokens per minute for this model alias (default: 100000)")
    add.add_argument("--max-parallel-requests", type=int, default=10, help="simultaneous requests for this model alias (default: 10)")
    add.add_argument(
        "--restart",
        action="store_true",
        help="force-recreate LiteLLM and Semantic Router after changing configs",
    )
    add.set_defaults(func=add_model)
    remove = subparsers.add_parser("remove", help="remove a model alias and its router references")
    remove.add_argument("name", help="public model alias")
    remove.add_argument(
        "--restart",
        action="store_true",
        help="force-recreate LiteLLM and Semantic Router after changing configs",
    )
    remove.set_defaults(func=remove_model)
    restart = subparsers.add_parser("restart", help="restart LiteLLM and Semantic Router")
    restart.set_defaults(func=restart_config)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        args.func(args)
    except (OSError, KeyError, StopIteration, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
