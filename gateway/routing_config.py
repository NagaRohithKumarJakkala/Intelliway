"""Validated, operator-maintained routing policy loaded at application startup."""

import json
import os
from importlib import resources
from pathlib import Path
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# https://docs.typesafe.ai/primitives/choice.md
MAX_CHOICE_OPTIONS = 255
POOL_NAMES = frozenset({"auto-cost", "auto-balanced", "auto-quality"})
RESERVED_ALIASES = POOL_NAMES | {"auto", "auto-local"}


class CatalogueEntry(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    identity: str
    description: str
    input_cost_per_million_usd: float | None = Field(ge=0, allow_inf_nan=False)
    output_cost_per_million_usd: float | None = Field(ge=0, allow_inf_nan=False)

    @field_validator("identity", "description")
    @classmethod
    def require_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(
                "identity and description must contain non-whitespace text"
            )
        return value


class RoutingConfig(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    catalogue: dict[str, CatalogueEntry]
    pools: dict[str, Annotated[list[str], Field(max_length=MAX_CHOICE_OPTIONS)]]

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        if "local-general" not in self.catalogue:
            raise ValueError(
                "catalogue must configure local-general for direct local routing"
            )
        if any(not alias or alias.strip() != alias for alias in self.catalogue):
            raise ValueError(
                "catalogue aliases must be nonempty and have no surrounding whitespace"
            )
        if RESERVED_ALIASES.intersection(self.catalogue):
            raise ValueError(
                "catalogue aliases must not collide with reserved automatic aliases"
            )
        if set(self.pools) != POOL_NAMES:
            raise ValueError(
                "pools must contain exactly auto-cost, auto-balanced, and auto-quality"
            )
        for preset, candidates in self.pools.items():
            if len(candidates) != len(set(candidates)):
                raise ValueError("pool candidates must be unique")
            if any(alias not in self.catalogue for alias in candidates):
                raise ValueError(
                    "pool candidates must reference known catalogue aliases"
                )
        return self


def load_routing_config() -> RoutingConfig:
    """Read a fresh policy; the caller retains it for one application lifetime."""
    override = os.environ.get("ROUTING_CONFIG_PATH", "").strip() or None
    source = (
        "ROUTING_CONFIG_PATH"
        if override is not None
        else "packaged gateway/routing_config.json"
    )
    try:
        text = (
            Path(override).read_text(encoding="utf-8")
            if override is not None
            else resources.files("gateway")
            .joinpath("routing_config.json")
            .read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, ValueError):
        raise RuntimeError(
            f"Cannot read routing configuration from {source}. "
            "Ensure it is a readable UTF-8 JSON file; fix ROUTING_CONFIG_PATH "
            "or restore the packaged resource, then restart the gateway."
        ) from None
    try:
        data = json.loads(text)
    except ValueError:
        raise RuntimeError(
            f"Invalid JSON routing configuration in {source}. "
            "Correct the JSON syntax and restart the gateway."
        ) from None
    try:
        return RoutingConfig.model_validate(data)
    except ValueError:
        # Do not expose validation inputs, arbitrary metadata, or filesystem paths.
        raise RuntimeError(
            f"Invalid routing configuration schema in {source}. "
            "Use only catalogue and pools: catalogue entries require nonblank identity "
            "and description plus input_cost_per_million_usd and "
            "output_cost_per_million_usd (nonnegative finite numbers or null if unknown), "
            "with no extra fields. "
            "Include local-general and do not use reserved auto aliases as catalogue keys. "
            "Define exactly auto-cost, auto-balanced, and auto-quality as ordered lists "
            "of unique known aliases (maximum 255 each); pools may be empty until configured. "
            "Correct the configuration and restart the gateway."
        ) from None
