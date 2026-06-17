from __future__ import annotations

import os
from typing import Any

from libs.llm_core import (
    LLMConfigurationError,
    LLMProvider,
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    ToolCall,
    Usage,
    value,
)


class AnthropicProvider(LLMProvider):
    """Anthropic Messages API adapter."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model or os.getenv("ANTHROPIC_MODEL")
        if not self.model:
            raise LLMConfigurationError(
                "AnthropicProvider requires model or ANTHROPIC_MODEL."
            )
        if client is None:
            import anthropic

            key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise LLMConfigurationError(
                    "AnthropicProvider requires api_key or ANTHROPIC_API_KEY."
                )
            client = anthropic.Anthropic(api_key=key)
        self.client = client

    def complete(self, request: LLMRequest) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_output_tokens,
            "messages": _messages(request),
        }
        if request.system_prompt:
            kwargs["system"] = request.system_prompt
        if request.tools:
            kwargs["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                }
                for tool in request.tools
            ]
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        try:
            response = self.client.messages.create(**kwargs)
        except Exception as exc:
            raise LLMProviderError(f"Anthropic request failed: {exc}") from exc

        calls: list[ToolCall] = []
        text_parts: list[str] = []
        for block in list(value(response, "content", []) or []):
            block_type = value(block, "type", "")
            if block_type == "text":
                text_parts.append(str(value(block, "text", "")))
            elif block_type == "tool_use":
                calls.append(
                    ToolCall(
                        id=str(value(block, "id", "")),
                        name=str(value(block, "name", "")),
                        arguments=dict(value(block, "input", {}) or {}),
                    )
                )
        usage = value(response, "usage", {}) or {}
        input_tokens = int(value(usage, "input_tokens", 0) or 0)
        output_tokens = int(value(usage, "output_tokens", 0) or 0)
        return LLMResponse(
            text=" ".join(text_parts).strip(),
            tool_calls=calls,
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
            finish_reason=str(value(response, "stop_reason", "") or "") or None,
            raw=response,
        )


def _messages(request: LLMRequest) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for message in request.messages:
        if message.role in {"user", "assistant"}:
            blocks: list[dict[str, Any]] = []
            if message.content:
                blocks.append({"type": "text", "text": message.content})
            if message.role == "assistant":
                blocks.extend(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.arguments,
                    }
                    for call in message.tool_calls
                )
            if blocks:
                _append_message(messages, message.role, blocks)
        elif message.role == "tool":
            _append_message(
                messages,
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id,
                        "content": message.content,
                        "is_error": message.is_error,
                    }
                ],
            )
        else:
            raise LLMProviderError(f"Unsupported normalized message role: {message.role!r}")
    return messages


def _append_message(
    messages: list[dict[str, Any]],
    role: str,
    blocks: list[dict[str, Any]],
) -> None:
    if messages and messages[-1]["role"] == role:
        messages[-1]["content"].extend(blocks)
    else:
        messages.append({"role": role, "content": blocks})

