
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


ALLOWED_ROLES = {"system", "user", "assistant"}


class Message(BaseModel):
    role: str
    content: str

    model_config = ConfigDict(extra="forbid")

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        if value not in ALLOWED_ROLES:
            raise ValueError(
                "Unsupported message role. Supported roles: "
                "system, user, assistant."
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

    # Streaming is not implemented in this MVP.
    stream: bool = False

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def reject_unsupported_streaming(self):
        if self.stream:
            raise ValueError(
                "Streaming is not supported by this gateway yet. "
                "Set stream=false."
            )
        return self
