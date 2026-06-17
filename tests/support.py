from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentic_canvas.providers.base import LLMProvider, ToolDefinition


class StaticProvider(LLMProvider):
    def __init__(self, response: str = "ok") -> None:
        self.response = response
        self.seen_messages: list[dict[str, Any]] = []
        self.seen_system_prompt = ""

    def chat(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        max_turns: int,
    ) -> str:
        assert [tool.name for tool in tools] == ["call_plugin"]
        self.seen_system_prompt = system_prompt
        self.seen_messages = messages
        return self.response


class RaisingProvider(LLMProvider):
    def chat(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        max_turns: int,
    ) -> str:
        raise AssertionError("orchestrator should have been skipped")


class CallPluginProvider(LLMProvider):
    def __init__(
        self,
        plugin: str,
        params: dict[str, Any] | None = None,
        *,
        response: str = "plugin returned",
    ) -> None:
        self.plugin = plugin
        self.params = params or {}
        self.response = response

    def chat(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        max_turns: int,
    ) -> str:
        assert len(tools) == 1
        tools[0].handler(name=self.plugin, params=self.params)
        return self.response


def write_plugin(
    workspace_root: Path,
    name: str,
    code: str,
    *,
    libraries: list[str] | None = None,
) -> None:
    plugin_dir = workspace_root / "plugins" / name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.py").write_text(code, encoding="utf-8")
    manifest = {
        "name": name,
        "version": "0.1.0",
        "summary": f"{name} test plugin",
        "documentation": f"Test plugin {name}.",
        "entry_point": "plugin.py:plugin_main",
        "input_schema": {"type": "object"},
        "output_schema": {"type": "object"},
        "requirements": [],
        "libraries": libraries or [],
        "compatibility": {"agentic_canvas": ">=0.1.0"},
    }
    (plugin_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def set_stages(workspace_root: Path, **stages: list[str]) -> None:
    config_path = workspace_root / "workspace.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["stages"] = {
        "pre_orchestrator": stages.get("pre_orchestrator", []),
        "post_orchestrator": stages.get("post_orchestrator", []),
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def read_trace(workspace_root: Path, run_id: str) -> list[dict[str, Any]]:
    trace_path = workspace_root / "runs" / f"{run_id}.trace.jsonl"
    return [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
