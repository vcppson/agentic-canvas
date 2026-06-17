from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Any]


class LLMProvider(ABC):
    """Provider boundary used by the orchestrator."""

    @abstractmethod
    def chat(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        max_turns: int,
    ) -> str:
        """Return the assistant's final text response."""
