from __future__ import annotations

import json
import sys
import uuid
from typing import Any, Callable

from libs.plugin_runtime import load_plugin_manifest, workspace_root


def call_plugin(
    plugin: str,
    params: dict[str, Any] | None = None,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ask the Kernel to execute another plugin through the runner protocol."""

    request_id = uuid.uuid4().hex
    _write_protocol(
        {
            "type": "plugin_call_request",
            "request_id": request_id,
            "plugin": plugin,
            "params": params or {},
            "metadata": metadata or {},
        }
    )
    response = _read_protocol()
    if response.get("type") != "plugin_call_response" or response.get("request_id") != request_id:
        raise RuntimeError("Invalid plugin call response from plugin runner.")
    value = response.get("value")
    if not isinstance(value, dict):
        raise RuntimeError("Plugin runner returned a non-object plugin call response.")
    return value


def plugin_tool(
    request: dict[str, Any],
    plugin: str,
    *,
    name: str | None = None,
    description: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Callable[..., Any]:
    """Wrap one installed workspace plugin as an explicit agent callable."""

    manifest = load_plugin_manifest(workspace_root(request), plugin)

    def invoke(**kwargs: Any) -> Any:
        result = call_plugin(plugin, kwargs, metadata=metadata)
        if not result.get("ok", False):
            raise RuntimeError(result.get("message") or f"Plugin {plugin!r} failed.")
        if result.get("kind") != "result":
            raise RuntimeError(
                f"Plugin {plugin!r} returned unsupported kind {result.get('kind')!r}."
            )
        return result.get("result") or {}

    invoke.__name__ = name or str(manifest["name"])
    invoke.__doc__ = description or str(manifest.get("summary") or f"Call plugin {plugin}.")
    setattr(
        invoke,
        "__agent_tool__",
        {
            "name": invoke.__name__,
            "description": invoke.__doc__,
            "input_schema": dict(manifest.get("input_schema") or {"type": "object"}),
        },
    )
    return invoke


def _write_protocol(payload: dict[str, Any]) -> None:
    sys.__stdout__.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    sys.__stdout__.flush()


def _read_protocol() -> dict[str, Any]:
    line = sys.__stdin__.readline()
    if not line:
        raise RuntimeError("Plugin runner closed before plugin call response was provided.")
    payload = json.loads(line)
    if not isinstance(payload, dict):
        raise RuntimeError("Plugin runner sent a non-object plugin call response.")
    return payload

