from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class LLMError(RuntimeError):
    """Base error raised by workspace LLM providers."""


class LLMConfigurationError(LLMError):
    """Raised when a provider is missing required configuration."""


class LLMProviderError(LLMError):
    """Raised when a provider request fails or returns an invalid response."""


@dataclass(frozen=True)
class ToolDefinition:
    """Provider-neutral model tool definition."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation requested by a model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Message:
    """One normalized conversation message."""

    role: str
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    is_error: bool = False


@dataclass(frozen=True)
class Usage:
    """Normalized token usage reported by a provider."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


@dataclass(frozen=True)
class LLMRequest:
    """Provider-neutral synchronous completion request."""

    messages: list[Message]
    system_prompt: str = ""
    tools: list[ToolDefinition] = field(default_factory=list)
    max_output_tokens: int = 4096
    temperature: float | None = None


@dataclass(frozen=True)
class LLMResponse:
    """Provider-neutral completion response."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None
    raw: Any = None


class LLMProvider(ABC):
    """Synchronous provider interface used by workspace agents."""

    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse:
        """Return one assistant response for the supplied conversation."""


def value(source: Any, name: str, default: Any = None) -> Any:
    """Read a field from either a mapping or an SDK response object."""

    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)

