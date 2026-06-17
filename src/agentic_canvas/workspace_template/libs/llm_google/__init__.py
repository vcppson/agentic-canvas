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


class GoogleProvider(LLMProvider):
    """Google Gemini API adapter using the google-genai SDK."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model or os.getenv("GEMINI_MODEL")
        if not self.model:
            raise LLMConfigurationError("GoogleProvider requires model or GEMINI_MODEL.")
        if client is None:
            from google import genai

            key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not key:
                raise LLMConfigurationError(
                    "GoogleProvider requires api_key, GEMINI_API_KEY, or GOOGLE_API_KEY."
                )
            client = genai.Client(api_key=key)
        self.client = client

    def complete(self, request: LLMRequest) -> LLMResponse:
        config: dict[str, Any] = {"max_output_tokens": request.max_output_tokens}
        if request.system_prompt:
            config["system_instruction"] = request.system_prompt
        if request.tools:
            config["tools"] = [
                {
                    "function_declarations": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.input_schema,
                        }
                        for tool in request.tools
                    ]
                }
            ]
        if request.temperature is not None:
            config["temperature"] = request.temperature
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=_contents(request),
                config=config,
            )
        except Exception as exc:
            raise LLMProviderError(f"Google request failed: {exc}") from exc

        candidates = list(value(response, "candidates", []) or [])
        if not candidates:
            raise LLMProviderError("Google returned no candidates.")
        candidate = candidates[0]
        content = value(candidate, "content")
        parts = list(value(content, "parts", []) or [])
        calls: list[ToolCall] = []
        text_parts: list[str] = []
        for part in parts:
            text = value(part, "text")
            if text:
                text_parts.append(str(text))
            function_call = value(part, "function_call")
            if function_call:
                calls.append(
                    ToolCall(
                        id=str(value(function_call, "id", "") or value(function_call, "name", "")),
                        name=str(value(function_call, "name", "")),
                        arguments=dict(value(function_call, "args", {}) or {}),
                    )
                )
        usage = value(response, "usage_metadata", {}) or {}
        return LLMResponse(
            text=" ".join(text_parts).strip(),
            tool_calls=calls,
            usage=Usage(
                input_tokens=int(value(usage, "prompt_token_count", 0) or 0),
                output_tokens=int(value(usage, "candidates_token_count", 0) or 0),
                total_tokens=int(value(usage, "total_token_count", 0) or 0),
            ),
            finish_reason=str(value(candidate, "finish_reason", "") or "") or None,
            raw=response,
        )


def _contents(request: LLMRequest) -> list[dict[str, Any]]:
    contents: list[dict[str, Any]] = []
    for message in request.messages:
        if message.role in {"user", "assistant"}:
            parts: list[dict[str, Any]] = []
            if message.content:
                parts.append({"text": message.content})
            for call in message.tool_calls:
                parts.append(
                    {
                        "function_call": {
                            "id": call.id,
                            "name": call.name,
                            "args": call.arguments,
                        }
                    }
                )
            if parts:
                contents.append(
                    {"role": "model" if message.role == "assistant" else "user", "parts": parts}
                )
        elif message.role == "tool":
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "function_response": {
                                "id": message.tool_call_id,
                                "name": message.name,
                                "response": {"output": message.content},
                            }
                        }
                    ],
                }
            )
        else:
            raise LLMProviderError(f"Unsupported normalized message role: {message.role!r}")
    return contents

