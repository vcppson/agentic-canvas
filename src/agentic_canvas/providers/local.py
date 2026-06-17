from __future__ import annotations

import json
import shlex
from typing import Any

from agentic_canvas.providers.base import LLMProvider, ToolDefinition


class LocalProvider(LLMProvider):
    """Deterministic local provider for development and tests.

    It keeps the orchestrator provider-backed without requiring network access.
    To exercise the single tool directly, pass input as either:

    - ``call_plugin <name> <json params>``
    - ``{"plugin": "<name>", "params": {...}}``
    """

    def __init__(self, reason: str | None = None) -> None:
        self.reason = reason or "No remote LLM provider is configured."

    def chat(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        max_turns: int,
    ) -> str:
        tool = _single_call_plugin_tool(tools)
        user_text = str(messages[-1].get("content", "")) if messages else ""

        parsed = self._parse_direct_call(user_text)
        if parsed:
            name, params = parsed
            result = tool.handler(name=name, params=params)
            return json.dumps(result, indent=2, ensure_ascii=False)

        catalog = tool.handler(name="plugin_catalog", params={})
        plugin_names = [item.get("name", "?") for item in catalog.get("plugins", [])]
        return (
            f"{self.reason} Configure a provider in workspace.json/.env, or directly "
            f"exercise a plugin with `call_plugin <name> {{...}}`. "
            f"Available plugins: {', '.join(plugin_names) or 'none'}."
        )

    def _parse_direct_call(self, text: str) -> tuple[str, dict[str, Any]] | None:
        stripped = text.strip()
        if not stripped:
            return None

        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = None

        if isinstance(payload, dict) and payload.get("plugin"):
            params = payload.get("params", {})
            return str(payload["plugin"]), params if isinstance(params, dict) else {"value": params}

        if stripped.startswith("call_plugin "):
            parts = shlex.split(stripped)
            if len(parts) >= 2:
                raw_params = stripped.split(parts[1], 1)[1].strip()
                params_text = raw_params or "{}"
                try:
                    params = json.loads(params_text)
                except json.JSONDecodeError:
                    params = {"input": params_text}
                return parts[1], params if isinstance(params, dict) else {"value": params}

        return None


def _single_call_plugin_tool(tools: list[ToolDefinition]) -> ToolDefinition:
    if len(tools) != 1 or tools[0].name != "call_plugin":
        raise RuntimeError("The orchestrator must expose exactly one tool: call_plugin.")
    return tools[0]
