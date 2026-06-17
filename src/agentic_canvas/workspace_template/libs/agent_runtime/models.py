from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from libs.llm_core import Message, ToolCall, Usage


class AgentError(RuntimeError):
    """Base error raised by the workspace agent runtime."""


class ToolDefinitionError(AgentError):
    """Raised when a Python callable cannot be exposed safely as a tool."""


class AgentMaxTurnsError(AgentError):
    """Raised when an agent exhausts its configured model-turn budget."""


@runtime_checkable
class AgentTool(Protocol):
    """Structural interface accepted by Agent for model-callable tools."""

    name: str
    description: str
    input_schema: dict[str, Any]

    def invoke(self, arguments: dict[str, Any]) -> Any:
        ...


@dataclass(frozen=True)
class AgentConfig:
    """Execution limits and generation settings for an Agent."""

    system_prompt: str = ""
    max_turns: int = 8
    max_output_tokens: int = 4096
    temperature: float | None = None

    def __post_init__(self) -> None:
        if self.max_turns < 1:
            raise ValueError("AgentConfig.max_turns must be at least 1.")
        if self.max_output_tokens < 1:
            raise ValueError("AgentConfig.max_output_tokens must be at least 1.")


@dataclass(frozen=True)
class ToolExecution:
    """One local tool execution performed during an agent run."""

    call: ToolCall
    ok: bool
    result: Any = None
    error_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "call": {
                "id": self.call.id,
                "name": self.call.name,
                "arguments": self.call.arguments,
            },
            "ok": self.ok,
            "result": json_value(self.result),
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


@dataclass(frozen=True)
class AgentResult:
    """Structured result of a completed agent run."""

    response: str
    messages: list[Message]
    usage: Usage
    turns: int
    tool_executions: list[ToolExecution]

    def to_dict(self) -> dict[str, Any]:
        return {
            "response": self.response,
            "turns": self.turns,
            "usage": asdict(self.usage),
            "tool_executions": [execution.to_dict() for execution in self.tool_executions],
        }


def json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if value is None or isinstance(value, (str, int, float, bool, list, dict)):
        return value
    return str(value)
