import argparse
from copy import deepcopy

from ruamel.yaml import YAML

from gateway import model_config
from gateway import upstream


def write_yaml(path, data):
    yaml = YAML()
    with path.open("w", encoding="utf-8") as stream:
        yaml.dump(data, stream)


def read_yaml(path):
    yaml = YAML()
    with path.open(encoding="utf-8") as stream:
        return yaml.load(stream)


def litellm_config():
    return {
        "model_list": [
            {
                "model_name": "gemma4",
                "litellm_params": {
                    "model": "ollama/gemma4",
                    "api_base": "os.environ/OLLAMA_BASE_URL",
                },
            }
        ]
    }


def router_config():
    return {
        "providers": {
            "defaults": {"model": "gemma4"},
            "models": [
                {
                    "name": "gemma4",
                    "provider_model_id": "gemma4",
                    "api_format": "openai",
                    "pricing": {
                        "currency": "USD",
                        "prompt_per_1m": 0.0,
                        "completion_per_1m": 0.0,
                    },
                    "backend_refs": [
                        {
                            "name": "litellm-gemma4",
                            "endpoint": "litellm:4000",
                            "provider": "openai",
                            "auth_header": "Authorization",
                            "auth_prefix": "Bearer ",
                            "weight": 100,
                        }
                    ],
                }
            ],
        },
        "routing": {
            "modelCards": [
                {
                    "name": "gemma4",
                    "context_window_size": 32768,
                    "capabilities": ["chat", "tools"],
                }
            ],
            "decisions": [
                {
                    "name": "balanced_pool",
                    "modelRefs": [{"model": "gemma4"}],
                }
            ],
        },
        "recipes": [
            {
                "name": "balanced",
                "routing": {
                    "decisions": [
                        {
                            "name": "balanced_default",
                            "modelRefs": [{"model": "gemma4"}],
                        }
                    ]
                },
            }
        ],
    }


def test_add_model_updates_both_configs(monkeypatch, tmp_path):
    litellm_path = tmp_path / "litellm.yaml"
    router_path = tmp_path / "router.yaml"
    write_yaml(litellm_path, litellm_config())
    write_yaml(router_path, router_config())
    monkeypatch.setattr(model_config, "LITELLM_CONFIG", litellm_path)
    monkeypatch.setattr(model_config, "ROUTER_CONFIG", router_path)

    args = argparse.Namespace(
        name="qwen",
        provider="ollama",
        model="qwen2.5:1.5b",
        api_base=None,
        api_key_env=None,
        pool="balanced",
        description="Local test model",
        context_window=32768,
        prompt_price=0.0,
        completion_price=0.0,
        rpm=60,
        tpm=100000,
        max_parallel_requests=10,
        restart=False,
    )
    model_config.add_model(args)

    litellm = read_yaml(litellm_path)
    assert litellm["model_list"][-1]["model_name"] == "qwen"
    assert litellm["model_list"][-1]["litellm_params"]["model"] == "ollama/qwen2.5:1.5b"

    router = read_yaml(router_path)
    assert any(model["name"] == "qwen" for model in router["providers"]["models"])
    assert any(
        card["name"] == "qwen"
        for card in router["routing"]["modelCards"]
    )
    assert {"model": "qwen"} in router["routing"]["decisions"][0]["modelRefs"]
    assert {"model": "qwen"} in router["recipes"][0]["routing"]["decisions"][0]["modelRefs"]


def test_remove_model_removes_alias_card_and_pool_refs(monkeypatch, tmp_path):
    litellm_path = tmp_path / "litellm.yaml"
    router_path = tmp_path / "router.yaml"
    litellm = litellm_config()
    router = router_config()

    litellm["model_list"].append({
        "model_name": "qwen",
        "litellm_params": {"model": "ollama/qwen2.5:1.5b"},
    })
    router["providers"]["models"].append(deepcopy(router["providers"]["models"][0]) | {"name": "qwen"})
    router["routing"]["modelCards"].append({
        "name": "qwen",
        "context_window_size": 32768,
        "capabilities": ["chat", "tools"],
    })
    router["routing"]["decisions"][0]["modelRefs"].append({"model": "qwen"})
    router["recipes"][0]["routing"]["decisions"][0]["modelRefs"].append({"model": "qwen"})

    write_yaml(litellm_path, litellm)
    write_yaml(router_path, router)
    monkeypatch.setattr(model_config, "LITELLM_CONFIG", litellm_path)
    monkeypatch.setattr(model_config, "ROUTER_CONFIG", router_path)

    model_config.remove_model(argparse.Namespace(name="qwen", restart=False))

    litellm = read_yaml(litellm_path)
    router = read_yaml(router_path)
    assert all(item["model_name"] != "qwen" for item in litellm["model_list"])
    assert all(item["name"] != "qwen" for item in router["providers"]["models"])
    assert all(item["name"] != "qwen" for item in router["routing"]["modelCards"])
    assert all(ref["model"] != "qwen" for ref in router["routing"]["decisions"][0]["modelRefs"])
    assert all(
        ref["model"] != "qwen"
        for ref in router["recipes"][0]["routing"]["decisions"][0]["modelRefs"]
    )


def test_remove_default_model_is_protected(monkeypatch, tmp_path):
    litellm_path = tmp_path / "litellm.yaml"
    router_path = tmp_path / "router.yaml"
    write_yaml(litellm_path, litellm_config())
    write_yaml(router_path, router_config())
    monkeypatch.setattr(model_config, "LITELLM_CONFIG", litellm_path)
    monkeypatch.setattr(model_config, "ROUTER_CONFIG", router_path)

    try:
        model_config.remove_model(argparse.Namespace(name="gemma4", restart=False))
    except ValueError as error:
        assert "Cannot remove gemma4" in str(error)
    else:
        raise AssertionError("Default model removal should fail")


def test_restart_uses_force_recreate(monkeypatch):
    calls = []

    def fake_run(command, cwd, check):
        calls.append((command, cwd, check))

    monkeypatch.setattr(model_config.subprocess, "run", fake_run)

    model_config.restart_services()

    assert calls == [(
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
        model_config.ROOT,
        True,
    )]


def test_upstream_litellm_headers_use_master_key(monkeypatch):
    monkeypatch.setenv("LITELLM_MASTER_KEY", "test-master-key")
    assert upstream._litellm_headers() == {
        "Content-Type": "application/json",
        "Authorization": "Bearer test-master-key",
    }
