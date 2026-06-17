from __future__ import annotations

import json
from typing import Any

from google import genai
from google.genai import types

from agentic_canvas.providers.base import LLMProvider, ToolDefinition


class GeminiProvider(LLMProvider):
    """Google Gemini provider using the official ``google-genai`` package."""

    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def chat(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        max_turns: int,
    ) -> str:
        tool_map = {tool.name: tool for tool in tools}
        declarations = [_to_function_declaration(types, tool) for tool in tools]
        gemini_tool = types.Tool(function_declarations=declarations)
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=[gemini_tool],
        )
        contents = _messages_to_contents(types, messages)

        for _ in range(max_turns):
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
            if not response.candidates:
                raise RuntimeError("Gemini returned no candidates.")

            content = response.candidates[0].content
            if content is None or not getattr(content, "parts", None):
                return getattr(response, "text", "") or ""

            contents.append(content)
            parts = list(getattr(content, "parts", []) or [])
            function_calls = [
                part.function_call
                for part in parts
                if getattr(part, "function_call", None) is not None
            ]
            if not function_calls:
                return _extract_text(parts) or getattr(response, "text", "") or ""

            response_parts = []
            for function_call in function_calls:
                name = str(function_call.name)
                if name not in tool_map:
                    raise RuntimeError(f"Gemini requested unknown tool: {name!r}")
                args = dict(getattr(function_call, "args", {}) or {})
                result = tool_map[name].handler(**args)
                response_parts.append(_function_response_part(types, name, result))

            contents.append(types.Content(role="user", parts=response_parts))

        return "Error: reached orchestrator max_turns."


def _to_function_declaration(types: Any, tool: ToolDefinition) -> Any:
    return types.FunctionDeclaration(
        name=tool.name,
        description=tool.description,
        parameters=_gemini_schema(tool.input_schema),
    )


def _gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Keep the JSON schema subset Gemini function declarations accept."""

    allowed = {
        "type",
        "description",
        "properties",
        "required",
        "items",
        "enum",
        "nullable",
    }
    cleaned: dict[str, Any] = {}
    for key, value in schema.items():
        if key not in allowed:
            continue
        if key == "properties" and isinstance(value, dict):
            cleaned[key] = {
                prop_name: _gemini_schema(prop_schema)
                for prop_name, prop_schema in value.items()
                if isinstance(prop_schema, dict)
            }
        elif key == "items" and isinstance(value, dict):
            cleaned[key] = _gemini_schema(value)
        else:
            cleaned[key] = value
    return cleaned or {"type": "object"}


def _messages_to_contents(types: Any, messages: list[dict[str, Any]]) -> list[Any]:
    contents = []
    for message in messages:
        role = "model" if message.get("role") == "assistant" else "user"
        content = message.get("content")
        if content is None:
            continue
        contents.append(types.Content(role=role, parts=[types.Part(text=str(content))]))
    return contents


def _extract_text(parts: list[Any]) -> str:
    return " ".join(str(part.text) for part in parts if getattr(part, "text", None)).strip()


def _function_response_part(types: Any, name: str, result: Any) -> Any:
    payload = result if isinstance(result, dict) else {"result": result}
    if hasattr(types.Part, "from_function_response"):
        return types.Part.from_function_response(name=name, response=payload)
    return types.Part(
        function_response=types.FunctionResponse(
            name=name,
            response=json.loads(json.dumps(payload, ensure_ascii=False, default=str)),
        )
    )
