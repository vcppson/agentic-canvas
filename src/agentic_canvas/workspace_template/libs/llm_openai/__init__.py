from __future__ import annotations

import json
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


class OpenAIProvider(LLMProvider):
    """OpenAI Responses API adapter."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model or os.getenv("OPENAI_MODEL")
        if not self.model:
            raise LLMConfigurationError("OpenAIProvider requires model or OPENAI_MODEL.")
        if client is None:
            from openai import OpenAI

            key = api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise LLMConfigurationError(
                    "OpenAIProvider requires api_key or OPENAI_API_KEY."
                )
            kwargs: dict[str, Any] = {"api_key": key}
            endpoint = base_url or os.getenv("OPENAI_BASE_URL")
            if endpoint:
                kwargs["base_url"] = endpoint.rstrip("/")
            client = OpenAI(**kwargs)
        self.client = client

    def complete(self, request: LLMRequest) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": _input_items(request),
            "max_output_tokens": request.max_output_tokens,
        }
        if request.system_prompt:
            kwargs["instructions"] = request.system_prompt
        if request.tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                    "strict": False,
                }
                for tool in request.tools
            ]
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        try:
            response = self.client.responses.create(**kwargs)
        except Exception as exc:
            raise LLMProviderError(f"OpenAI request failed: {exc}") from exc

        output = list(value(response, "output", []) or [])
        calls: list[ToolCall] = []
        text_parts: list[str] = []
        for item in output:
            item_type = value(item, "type", "")
            if item_type == "function_call":
                try:
                    arguments = json.loads(value(item, "arguments", "{}") or "{}")
                except json.JSONDecodeError as exc:
                    raise LLMProviderError(
                        f"OpenAI returned invalid tool arguments: {exc}"
                    ) from exc
                calls.append(
                    ToolCall(
                        id=str(value(item, "call_id", value(item, "id", ""))),
                        name=str(value(item, "name", "")),
                        arguments=arguments,
                    )
                )
            if item_type == "message":
                for block in list(value(item, "content", []) or []):
                    if value(block, "type", "") in {"output_text", "text"}:
                        text_parts.append(str(value(block, "text", "")))
        text = str(value(response, "output_text", "") or "").strip()
        if not text:
            text = " ".join(part for part in text_parts if part).strip()
        usage = value(response, "usage", {}) or {}
        return LLMResponse(
            text=text,
            tool_calls=calls,
            usage=Usage(
                input_tokens=int(value(usage, "input_tokens", 0) or 0),
                output_tokens=int(value(usage, "output_tokens", 0) or 0),
                total_tokens=int(value(usage, "total_tokens", 0) or 0),
            ),
            finish_reason=str(value(response, "status", "") or "") or None,
            raw=response,
        )


def _input_items(request: LLMRequest) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in request.messages:
        if message.role in {"user", "assistant"}:
            if message.content:
                items.append({"role": message.role, "content": message.content})
            if message.role == "assistant":
                for call in message.tool_calls:
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": call.id,
                            "name": call.name,
                            "arguments": json.dumps(call.arguments, ensure_ascii=False),
                        }
                    )
        elif message.role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": message.content,
                }
            )
        else:
            raise LLMProviderError(f"Unsupported normalized message role: {message.role!r}")
    return items

