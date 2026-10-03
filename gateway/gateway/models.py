from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "auto"
    messages: list[Message] = Field(min_length=1)

    temperature: float | None = None
    max_tokens: int | None = None

    # W2 explicitly does not support streaming.
    stream: bool = False

    model_config = ConfigDict(extra="forbid")
