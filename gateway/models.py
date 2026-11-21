
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)
from typing import Any


ALLOWED_ROLES = {"system", "user", "assistant", "tool"}


class CacheOptions(BaseModel):
    ttl: int | None = Field(default=None, gt=0)
    s_maxage: int | None = Field(default=None, alias="s-maxage", gt=0)
    no_cache: bool | None = Field(default=None, alias="no-cache")
    no_store: bool | None = Field(default=None, alias="no-store")
    use_cache: bool | None = Field(default=None, alias="use-cache")
    namespace: str | None = None

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Message(BaseModel):
    role: str
    content: str
    tool_call_id: str | None = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        if value not in ALLOWED_ROLES:
            raise ValueError(
                "Unsupported message role. Supported roles: "
                "system, user, assistant, tool."
            )
        return value


class ChatCompletionRequest(BaseModel):
    model: str = "auto"
    messages: list[Message] = Field(min_length=1)

    temperature: float | None = Field(
        default=None,
        ge=0,
        le=2,
    )
    max_tokens: int | None = Field(
        default=None,
        gt=0,
    )
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    cache: CacheOptions | None = None

    stream: bool = False

    model_config = ConfigDict(extra="forbid")
