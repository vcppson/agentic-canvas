from __future__ import annotations

import json
from typing import Any, Callable

from libs.agent_runtime.models import (
    AgentConfig,
    AgentMaxTurnsError,
    AgentResult,
    AgentTool,
    ToolDefinitionError,
    ToolExecution,
    json_value,
)
from libs.agent_runtime.tools import FunctionTool, normalize_tool
from libs.llm_core import LLMProvider, LLMRequest, Message, ToolCall, ToolDefinition, Usage


class Agent:
    """Provider-neutral, bounded, synchronous model/tool loop."""

    def __init__(
        self,
        *,
        provider: LLMProvider,
        tools: list[AgentTool | Callable[..., Any]] | None = None,
        config: AgentConfig | None = None,
        system_prompt: str | None = None,
        max_turns: int | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> None:
        self.provider = provider
        base = config or AgentConfig()
        self.config = AgentConfig(
            system_prompt=base.system_prompt if system_prompt is None else system_prompt,
            max_turns=base.max_turns if max_turns is None else max_turns,
            max_output_tokens=(
                base.max_output_tokens if max_output_tokens is None else max_output_tokens
            ),
            temperature=base.temperature if temperature is None else temperature,
        )
        normalized = [normalize_tool(item) for item in tools or []]
        duplicates = sorted(
            {item.name for item in normalized if sum(tool.name == item.name for tool in normalized) > 1}
        )
        if duplicates:
            raise ToolDefinitionError(f"Duplicate agent tool name(s): {duplicates}")
        self.tools = {item.name: item for item in normalized}

    def run(
        self,
        prompt: str,
        *,
        messages: list[Message] | None = None,
    ) -> AgentResult:
        """Run until the provider returns text without tool calls or the turn limit is reached."""

        conversation = list(messages or [])
        conversation.append(Message(role="user", content=prompt))
        usage = Usage()
        executions: list[ToolExecution] = []
        definitions = [
            tool.definition() if isinstance(tool, FunctionTool) else ToolDefinition(
                name=tool.name,
                description=tool.description,
                input_schema=tool.input_schema,
            )
            for tool in self.tools.values()
        ]

        for turn in range(1, self.config.max_turns + 1):
            response = self.provider.complete(
                LLMRequest(
                    system_prompt=self.config.system_prompt,
                    messages=conversation,
                    tools=definitions,
                    max_output_tokens=self.config.max_output_tokens,
                    temperature=self.config.temperature,
                )
            )
            usage = usage + response.usage
            conversation.append(
                Message(
                    role="assistant",
                    content=response.text,
                    tool_calls=response.tool_calls,
                )
            )
            if not response.tool_calls:
                return AgentResult(
                    response=response.text,
                    messages=conversation,
                    usage=usage,
                    turns=turn,
                    tool_executions=executions,
                )

            for call in response.tool_calls:
                execution = self._execute_tool(call)
                executions.append(execution)
                conversation.append(
                    Message(
                        role="tool",
                        name=call.name,
                        tool_call_id=call.id,
                        content=json.dumps(
                            _tool_message(execution),
                            ensure_ascii=False,
                            default=str,
                        ),
                        is_error=not execution.ok,
                    )
                )

        raise AgentMaxTurnsError(
            f"Agent reached max_turns={self.config.max_turns} before completing."
        )

    def _execute_tool(self, call: ToolCall) -> ToolExecution:
        selected = self.tools.get(call.name)
        if selected is None:
            return ToolExecution(
                call=call,
                ok=False,
                error_type="UnknownTool",
                error_message=f"Model requested unknown tool {call.name!r}.",
            )
        try:
            return ToolExecution(call=call, ok=True, result=selected.invoke(call.arguments))
        except Exception as exc:
            return ToolExecution(
                call=call,
                ok=False,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )


def _tool_message(execution: ToolExecution) -> dict[str, Any]:
    if execution.ok:
        return {"ok": True, "result": json_value(execution.result)}
    return {
        "ok": False,
        "error": {
            "type": execution.error_type,
            "message": execution.error_message,
        },
    }
