from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from agentic_canvas.providers.base import LLMProvider, ToolDefinition


class OpenAICompatibleProvider(LLMProvider):
    """OpenAI-compatible chat-completions provider using the official SDK."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: int = 120,
    ) -> None:
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
        )

    def chat(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        max_turns: int,
    ) -> str:
        if len(tools) != 1 or tools[0].name != "call_plugin":
            raise RuntimeError("The orchestrator must expose exactly one tool: call_plugin.")

        conversation = [{"role": "system", "content": system_prompt}, *messages]
        tool_map = {tool.name: tool for tool in tools}
        tool_specs = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in tools
        ]

        for _ in range(max_turns):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=conversation,
                tools=tool_specs,
                tool_choice="auto",
            )
            if not response.choices:
                raise RuntimeError("OpenAI-compatible provider returned no choices.")
            message_obj = response.choices[0].message
            message = message_obj.model_dump(exclude_none=True)
            conversation.append(message)

            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                return str(message.get("content") or "")

            for call in tool_calls:
                function = call.get("function") or {}
                name = function.get("name")
                if name not in tool_map:
                    raise RuntimeError(f"Provider requested unknown tool: {name!r}")
                try:
                    args = json.loads(function.get("arguments") or "{}")
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Invalid tool arguments for {name!r}: {exc}") from exc
                result = tool_map[name].handler(**args)
                conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id"),
                        "name": name,
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    }
                )

        return "Error: reached orchestrator max_turns."
