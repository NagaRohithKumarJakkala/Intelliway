import json
from importlib import resources

import pytest
from pydantic import ValidationError

from gateway.routing_config import RoutingConfig, load_routing_config


@pytest.fixture
def policy():
    return {
        "catalogue": {
            "local-general": {
                "identity": "local/model",
                "description": "Small local text model.",
                "input_cost_per_million_usd": 0,
                "output_cost_per_million_usd": None,
            },
            "frontier": {
                "identity": "provider/model",
                "description": "Operator-selected quality model.",
                "input_cost_per_million_usd": 1.25,
                "output_cost_per_million_usd": 4.5,
            },
        },
        "pools": {
            "auto-cost": ["local-general"],
            "auto-balanced": ["frontier", "local-general"],
            "auto-quality": [],
        },
    }


def test_valid_policy_preserves_order_and_metadata(policy):
    policy["pools"]["auto-quality"] = ["frontier"]
    config = RoutingConfig.model_validate(policy)
    assert isinstance(config.catalogue, dict)
    assert isinstance(config.pools, dict)
    assert config.pools == policy["pools"]
    for alias, metadata in policy["catalogue"].items():
        assert config.catalogue[alias].model_dump() == metadata


@pytest.mark.parametrize(
    "field", ["input_cost_per_million_usd", "output_cost_per_million_usd"]
)
@pytest.mark.parametrize(
    "value", [-1, -0.01, True, False, float("nan"), float("inf"), float("-inf"), "1"]
)
def test_invalid_price(policy, field, value):
    policy["catalogue"]["local-general"][field] = value
    with pytest.raises(ValidationError):
        RoutingConfig.model_validate(policy)


@pytest.mark.parametrize(
    "field", ["input_cost_per_million_usd", "output_cost_per_million_usd"]
)
@pytest.mark.parametrize("value", [None, 0, 0.0, 0.125, 10])
def test_valid_price_preserved(policy, field, value):
    policy["catalogue"]["local-general"][field] = value
    config = RoutingConfig.model_validate(policy)
    assert config.catalogue["local-general"].model_dump()[field] == value


@pytest.mark.parametrize("field", ["identity", "description"])
@pytest.mark.parametrize("value", ["", " \n\t", 123, None])
def test_invalid_metadata(policy, field, value):
    policy["catalogue"]["local-general"][field] = value
    with pytest.raises(ValidationError):
        RoutingConfig.model_validate(policy)


@pytest.mark.parametrize(
    "field",
    [
        "identity",
        "description",
        "input_cost_per_million_usd",
        "output_cost_per_million_usd",
    ],
)
def test_missing_entry_field(policy, field):
    del policy["catalogue"]["local-general"][field]
    with pytest.raises(ValidationError):
        RoutingConfig.model_validate(policy)


@pytest.mark.parametrize("location", ["root", "entry", "pools"])
def test_extra_fields_forbidden(policy, location):
    target = {
        "root": policy,
        "entry": policy["catalogue"]["local-general"],
        "pools": policy["pools"],
    }[location]
    target["unexpected"] = "not allowed"
    with pytest.raises(ValidationError):
        RoutingConfig.model_validate(policy)


@pytest.mark.parametrize(
    "alias",
    [
        "auto",
        "auto-local",
        "auto-cost",
        "auto-balanced",
        "auto-quality",
        "",
        " local",
        "local ",
    ],
)
def test_invalid_catalogue_alias(policy, alias):
    policy["catalogue"][alias] = policy["catalogue"]["local-general"].copy()
    with pytest.raises(ValidationError):
        RoutingConfig.model_validate(policy)


def test_local_general_required(policy):
    del policy["catalogue"]["local-general"]
    policy["pools"] = {name: ["frontier"] for name in policy["pools"]}
    with pytest.raises(ValidationError, match="local-general"):
        RoutingConfig.model_validate(policy)


@pytest.mark.parametrize("preset", ["auto-cost", "auto-balanced", "auto-quality"])
def test_all_pools_required(policy, preset):
    del policy["pools"][preset]
    with pytest.raises(ValidationError):
        RoutingConfig.model_validate(policy)


@pytest.mark.parametrize("preset", ["auto-cost", "auto-balanced", "auto-quality"])
def test_automatic_pools_may_be_empty(policy, preset):
    policy["pools"][preset] = []
    assert RoutingConfig.model_validate(policy).pools[preset] == []


@pytest.mark.parametrize(
    "candidates",
    [
        ["unknown"],
        ["local-general", "local-general"],
        "local-general",
        ("local-general",),
        [1],
        None,
    ],
)
def test_invalid_pool_candidates(policy, candidates):
    policy["pools"]["auto-quality"] = candidates
    with pytest.raises(ValidationError):
        RoutingConfig.model_validate(policy)


@pytest.mark.parametrize("count", [255, 256])
@pytest.mark.parametrize("preset", ["auto-cost", "auto-balanced", "auto-quality"])
def test_choice_limit_per_pool_not_catalogue(policy, count, preset):
    aliases = [f"model-{index}" for index in range(count)]
    for alias in aliases:
        policy["catalogue"][alias] = policy["catalogue"]["local-general"].copy()
    policy["pools"][preset] = aliases
    if count == 255:
        assert RoutingConfig.model_validate(policy).pools[preset] == aliases
    else:
        with pytest.raises(ValidationError):
            RoutingConfig.model_validate(policy)


def test_override_is_explicit_and_loaded_only_when_called(
    policy, tmp_path, monkeypatch
):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    monkeypatch.setenv("ROUTING_CONFIG_PATH", str(path))
    config = load_routing_config()
    assert config.pools == policy["pools"]
    policy["pools"]["auto-quality"] = ["frontier"]
    path.write_text(json.dumps(policy), encoding="utf-8")
    assert config.pools["auto-quality"] == []
    assert load_routing_config().pools["auto-quality"] == ["frontier"]


@pytest.mark.parametrize(
    "content",
    ["{secret-value", '{"secret-value": 1}', "null", "[]", b"\xffsecret-value"],
)
def test_load_errors_are_sanitized(content, tmp_path, monkeypatch):
    path = tmp_path / "secret-path.json"
    path.write_bytes(content if isinstance(content, bytes) else content.encode())
    monkeypatch.setenv("ROUTING_CONFIG_PATH", str(path))
    with pytest.raises(RuntimeError) as caught:
        load_routing_config()
    message = str(caught.value)
    assert "ROUTING_CONFIG_PATH" in message
    assert "restart" in message
    assert "secret-value" not in message
    assert str(path) not in message
    assert caught.value.__suppress_context__


@pytest.mark.parametrize("override", ["missing.json"])
def test_unreadable_override_does_not_fall_back(override, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ROUTING_CONFIG_PATH", override)
    with pytest.raises(RuntimeError, match="Cannot read.*ROUTING_CONFIG_PATH"):
        load_routing_config()


def test_default_uses_packaged_resource_outside_project(policy, tmp_path, monkeypatch):
    monkeypatch.delenv("ROUTING_CONFIG_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    resource_dir = tmp_path / "resource"
    resource_dir.mkdir()
    (resource_dir / "routing_config.json").write_text(
        json.dumps(policy), encoding="utf-8"
    )
    calls = []

    def files(package):
        calls.append(package)
        return resource_dir

    monkeypatch.setattr(resources, "files", files)
    config = load_routing_config()
    assert calls == ["gateway"]
    assert config.model_dump() == policy


@pytest.mark.parametrize("override", ["", "   "])
def test_blank_override_uses_packaged_default(override, policy, tmp_path, monkeypatch):
    (tmp_path / "routing_config.json").write_text(json.dumps(policy), encoding="utf-8")
    monkeypatch.setattr(resources, "files", lambda package: tmp_path)
    monkeypatch.setenv("ROUTING_CONFIG_PATH", override)
    assert load_routing_config().model_dump() == policy


def test_invalid_packaged_policy_does_not_fall_back(policy, tmp_path, monkeypatch):
    policy["pools"]["auto-quality"] = ["missing-alias"]
    (tmp_path / "routing_config.json").write_text(json.dumps(policy), encoding="utf-8")
    monkeypatch.delenv("ROUTING_CONFIG_PATH", raising=False)
    monkeypatch.setattr(resources, "files", lambda package: tmp_path)
    with pytest.raises(RuntimeError, match="Invalid routing configuration schema"):
        load_routing_config()


def test_missing_packaged_resource_is_actionable(monkeypatch):
    monkeypatch.delenv("ROUTING_CONFIG_PATH", raising=False)

    def missing_files(package):
        raise FileNotFoundError("secret-path")

    monkeypatch.setattr(resources, "files", missing_files)
    with pytest.raises(RuntimeError, match="restore the packaged resource") as caught:
        load_routing_config()
    assert "secret-path" not in str(caught.value)
